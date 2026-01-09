#!/usr/bin/env python3
r"""Mini EA-adapt protocol on small OpenML datasets (fixed version).

Two branches (paired seeds/splits), per dataset:
  1) VANILLA (ID): no training; evaluate with identity metric
  2) EA-ADAPT: enable EA (layer ≥2) in TFicl (+TFrow optional), freeze TFcol, adapt on TRAIN ONLY for 200–300 steps

Key fixes vs your last script:
- No validation leakage: adaptation uses ONLY X_train/y_train inside _train_forward (embed_with_test=False).
- Correct input contract: adaptation batches are built via ensemble_generator_.transform(X_train_num) with feature permutations applied per variant (no class-shift use; we train only on train slice).
- True identity baseline: VANILLA branch forces identity metric and skips EA entirely (and keeps layer-0 identity explicitly).
- Eval geometry consistency: eval toggles ID vs EA with the same TFrow flag you adapted with.

Run:
  python scripts/dev/test_ea_adapt_fixed.py --datasets iris,wine,breast-w,ionosphere,sonar --steps 0,200,300
"""

from __future__ import annotations

import argparse
import os
import sys
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Ensure local src package is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tabicl.sklearn.classifier import TabICLClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_target_if_missing(X: pd.DataFrame, y: Optional[pd.Series]) -> Tuple[pd.DataFrame, pd.Series]:
    if y is not None and not isinstance(y, type(None)):
        return X, y
    cols = list(X.columns)
    preferred = ["class", "Class", "target", "Target", "label", "Label", "y"]
    for c in preferred:
        if c in cols:
            y_series = X[c].copy()
            X_feat = X.drop(columns=[c])
            return X_feat, y_series
    for c in cols:
        nunique = X[c].nunique(dropna=True)
        if 2 <= nunique <= min(50, max(2, X.shape[0] // 2)):
            y_series = X[c].copy()
            X_feat = X.drop(columns=[c])
            return X_feat, y_series
    c = cols[-1]
    y_series = X[c].copy()
    X_feat = X.drop(columns=[c])
    return X_feat, y_series


def fetch_openml_dataset(name_or_id: str | int) -> Tuple[pd.DataFrame, pd.Series, str]:
    import openml
    if isinstance(name_or_id, int) or (str(name_or_id).isdigit()):
        ds = openml.datasets.get_dataset(int(name_or_id))
        X, y, _, _ = ds.get_data(target=getattr(ds, "default_target_attribute", None), dataset_format="dataframe")
        X, y = _extract_target_if_missing(X, y)
        return X, y, ds.name
    name = str(name_or_id).lower()
    df = openml.datasets.list_datasets(output_format="dataframe")
    exact = df[df["name"].str.lower() == name]
    if len(exact) == 0:
        contains = df[df["name"].str.lower().str.contains(name)]
        if len(contains) == 0:
            raise ValueError(f"OpenML dataset not found: {name_or_id}")
        row = contains.sort_values("NumberOfInstances").iloc[0]
    else:
        row = exact.sort_values("NumberOfInstances").iloc[0]
    did = int(row["did"]) if "did" in row else int(row.get("dataset_id", row.get("ID")))
    ds = openml.datasets.get_dataset(did)
    X, y, _, _ = ds.get_data(target=getattr(ds, "default_target_attribute", None), dataset_format="dataframe")
    X, y = _extract_target_if_missing(X, y)
    return X, y, ds.name


def set_attn_mode(model, mode: str = "EA", enable_tfrow: bool = False) -> None:
    """Set attention metric mode.
    mode: 'EA' uses estimator; 'ID' forces identity metric.
    Always force identity on layer-0 (paper requires EA from layer ≥ 2).
    """
    mode = mode.upper()
    # TFicl
    for i, blk in enumerate(model.icl_predictor.tf_icl.blocks):
        blk.elliptical = True  # gate handled inside block by override + prev-V
        blk.elliptical_manual_m = None
        if mode == "ID" or i == 0:
            blk.elliptical_override = "identity"
        else:
            blk.elliptical_override = "none"
    # TFrow (optional)
    if enable_tfrow:
        for i, blk in enumerate(model.row_interactor.tf_row.blocks):
            blk.elliptical = True
            blk.elliptical_manual_m = None
            if mode == "ID" or i == 0:
                blk.elliptical_override = "identity"
            else:
                blk.elliptical_override = "none"


def mcar_mask_df(X_test: pd.DataFrame, p: float = 0.10, rng: np.random.RandomState | None = None) -> pd.DataFrame:
    rng = rng or np.random.RandomState(1)
    Xm = X_test.copy()
    n, d = Xm.shape
    m = int(np.round(p * n * d))
    if m <= 0:
        return Xm
    rows = rng.randint(0, n, size=m)
    cols = rng.randint(0, d, size=m)
    for r, c in zip(rows, cols):
        Xm.iat[r, c] = np.nan
    return Xm


@dataclass
class Scores:
    acc: float
    bacc: float


def eval_with_mode(clf: TabICLClassifier, X_test: pd.DataFrame, y_test: pd.Series, mode: str = "EA", enable_tfrow_eval: bool = False) -> Scores:
    prev_train = clf.model_.training
    set_attn_mode(clf.model_, mode=mode, enable_tfrow=enable_tfrow_eval)
    clf.model_.eval()
    y_pred = clf.predict(X_test)
    if prev_train:
        clf.model_.train()
    return Scores(acc=float(accuracy_score(y_test, y_pred)), bacc=float(balanced_accuracy_score(y_test, y_pred)))


def build_adapt_pack_train_only(clf: TabICLClassifier, X_train: pd.DataFrame, y_train: pd.Series):
    """Build adaptation variants from TRAIN ONLY using TabICL's ensemble generator.
    Returns:
      X_all: (Nvar, T, F)
      Ytrain_all: (Nvar, train_T)
      patterns_all: list of length Nvar, each a list[int] feature permutation
    """
    X_train_num = clf.X_encoder_.transform(X_train)
    data = clf.ensemble_generator_.transform(X_train_num)  # dict[norm_method] -> (Xs, ys)
    X_list, ytrain_list, pattern_list = [], [], []
    for norm_method, (Xs, ys) in data.items():
        X_list.append(Xs)     # (variants, T, F) where T = train_T (since we don't embed test here)
        ytrain_list.append(ys)  # (variants, train_T)
        pattern_list += [list(p) for p in clf.ensemble_generator_.feature_shuffle_patterns_[norm_method]]
    X_all = np.concatenate(X_list, axis=0)            # (Nvar, T, F)  (train tokens only)
    Ytrain_all = np.concatenate(ytrain_list, axis=0)  # (Nvar, train_T)
    return X_all, Ytrain_all, pattern_list


def ea_adapt_train_only(
    clf: TabICLClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    device: torch.device,
    steps: int = 300,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    batch_variants: int = 8,
    grad_clip: float = 1.0,
    adapt_tfrow: bool = False,
) -> None:
    """Adapt TFicl (+ optionally last TFrow layer via requires_grad) using TRAIN ONLY.
    No validation rows enter the graph; embed_with_test=False; loss on train slice only.
    """
    X_all, Ytrain_all, patterns_all = build_adapt_pack_train_only(clf, X_train, y_train)

    # Tensors
    X_all_t = torch.from_numpy(X_all).float().to(device)           # (Nvar, T, F)
    Ytrain_all_t = torch.from_numpy(Ytrain_all).long().to(device)  # (Nvar, train_T)
    Nvar, T, Fdim = X_all_t.shape

    # Enable EA for adaptation and set trainables
    set_attn_mode(clf.model_, mode="EA", enable_tfrow=adapt_tfrow)
    # Freeze COL always
    for p in clf.model_.col_embedder.parameters():
        p.requires_grad = False
    # Optionally allow only the LAST TFrow block to adapt (cheap co-adaptation)
    if adapt_tfrow:
        for i, blk in enumerate(clf.model_.row_interactor.tf_row.blocks):
            requires = (i == len(clf.model_.row_interactor.tf_row.blocks) - 1)
            for p in blk.parameters():
                p.requires_grad = requires
    else:
        for p in clf.model_.row_interactor.parameters():
            p.requires_grad = False
    # Adapt ICL
    for p in clf.model_.icl_predictor.parameters():
        p.requires_grad = True

    params = [p for p in clf.model_.parameters() if p.requires_grad]
    opt = AdamW(params, lr=float(lr), weight_decay=float(weight_decay))

    clf.model_.train()
    for step in range(int(steps)):
        # mini-batch over variants
        start = (step * batch_variants) % Nvar
        end = min(start + batch_variants, Nvar)
        if end <= start:
            start = 0
            end = min(batch_variants, Nvar)
        idx = slice(start, end)

        Xb = X_all_t[idx]         # (B, T, F)
        Ytb = Ytrain_all_t[idx]   # (B, train_T)
        pats = patterns_all[start:end]

        # Apply per-variant feature permutations (match training contract)
        Xb_perm = []
        for i, pat in enumerate(pats):
            pi = torch.tensor(pat, dtype=torch.long, device=device)
            Xb_perm.append(Xb[i][:, pi])
        Xb_perm = torch.stack(Xb_perm, dim=0)  # (B, T, F)

        # Prepare vector of feature dims (per variant)
        d_vec = torch.full((Xb_perm.shape[0],), Xb_perm.shape[-1], dtype=torch.long, device=device)

        # Forward ONLY over train tokens; do NOT embed test; no class shift needed
        out_full = clf.model_._train_forward(
            Xb_perm, y_train=Ytb, d=d_vec, embed_with_test=False
        )  # (B, T, C) where T == train_T

        # CE on train slice (full T)
        Bsz, Tsz, C = out_full.shape
        logits = out_full.reshape(Bsz * Tsz, C)
        targets = Ytb.reshape(Bsz * Tsz)
        loss = F.cross_entropy(logits, targets)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            clip_grad_norm_(params, float(grad_clip))
        opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--checkpoint", default=os.path.join(REPO_ROOT, "checkpoints", "tabicl-classifier-v1.1-0506.ckpt"))
    ap.add_argument("--datasets", default="iris,wine,breast-w,ionosphere,sonar")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_estimators", type=int, default=8)
    ap.add_argument("--steps", type=str, default="0,200,300")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_variants", type=int, default=8)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--adapt_tfrow", action="store_true",
                    help="Also adapt the LAST TFrow block; by default only TFicl adapts.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    set_seed(args.random_state)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    datasets = [s.strip() for s in str(args.datasets).split(",") if s.strip()]
    steps_list = [int(s) for s in str(args.steps).split(",") if s.strip()]
    assert 0 in steps_list, "steps must include 0 for baseline eval"
    max_steps = max(steps_list)

    for ds_name in datasets:
        try:
            X, y, ds_resolved = fetch_openml_dataset(ds_name)
        except Exception as e:
            print(f"[WARN] Failed to load OpenML dataset '{ds_name}': {e}. Skipping.")
            continue

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=float(args.test_size), random_state=args.random_state, stratify=y
        )

        # Build classifier and fit transforms on TRAIN only
        clf_id = TabICLClassifier(
            device=dev,
            model_path=args.checkpoint,
            allow_auto_download=False,
            use_hierarchical=True,
            n_estimators=int(args.n_estimators),
            random_state=args.random_state,
            verbose=False,
        )
        clf_id.fit(X_train, y_train)

        # Clone classifier for EA-adapt by re-loading ckpt into a new instance
        clf_ea = TabICLClassifier(
            device=dev,
            model_path=args.checkpoint,
            allow_auto_download=False,
            use_hierarchical=True,
            n_estimators=int(args.n_estimators),
            random_state=args.random_state,
            verbose=False,
        )
        clf_ea.fit(X_train, y_train)

        # VANILLA branch (identity metric, no training)
        print(f"[{ds_resolved}] VANILLA branch")
        for s in steps_list:
            clean = eval_with_mode(clf_id, X_val, y_val, mode="ID", enable_tfrow_eval=False)
            Xm = mcar_mask_df(X_val, p=0.10, rng=np.random.RandomState(1))
            cor = eval_with_mode(clf_id, Xm, y_val, mode="ID", enable_tfrow_eval=False)
            print(f"  step={s:03d} clean: acc={clean.acc:.3f} bacc={clean.bacc:.3f} | "
                  f"mcar10 drop: acc={clean.acc - cor.acc:.3f} bacc={clean.bacc - cor.bacc:.3f}")

        # EA-ADAPT branch (adapt on TRAIN ONLY)
        print(f"[{ds_resolved}] EA-ADAPT branch (TFicl{' + last-TFrow' if args.adapt_tfrow else ''})")
        clean0 = eval_with_mode(clf_ea, X_val, y_val, mode="EA", enable_tfrow_eval=args.adapt_tfrow)
        Xm = mcar_mask_df(X_val, p=0.10, rng=np.random.RandomState(1))
        cor0 = eval_with_mode(clf_ea, Xm, y_val, mode="EA", enable_tfrow_eval=args.adapt_tfrow)
        print(f"  step=000 clean: acc={clean0.acc:.3f} bacc={clean0.bacc:.3f} | "
              f"mcar10 drop: acc={clean0.acc - cor0.acc:.3f} bacc={clean0.bacc - cor0.bacc:.3f}")

        if max_steps > 0:
            ea_adapt_train_only(
                clf_ea,
                X_train,
                y_train,
                device=dev,
                steps=max_steps,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                batch_variants=int(args.batch_variants),
                grad_clip=float(args.grad_clip),
                adapt_tfrow=bool(args.adapt_tfrow),
            )
            cleanN = eval_with_mode(clf_ea, X_val, y_val, mode="EA", enable_tfrow_eval=args.adapt_tfrow)
            Xm = mcar_mask_df(X_val, p=0.10, rng=np.random.RandomState(1))
            corN = eval_with_mode(clf_ea, Xm, y_val, mode="EA", enable_tfrow_eval=args.adapt_tfrow)
            print(f"  step={max_steps:03d} clean: acc={cleanN.acc:.3f} bacc={cleanN.bacc:.3f} | "
                  f"mcar10 drop: acc={cleanN.acc - corN.acc:.3f} bacc={cleanN.bacc - corN.bacc:.3f}")


if __name__ == "__main__":
    main()
