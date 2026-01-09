#!/usr/bin/env python3
r"""Paired early-curve A/B (sign test) for Elliptical Attention.

Two branches from identical starts per fold/dataset:
  - ID: vanilla attention (M = I)
  - EA: elliptical metric in TFrow+TFicl (layer ≥2)

Setup:
  - Trainable: TFrow last layer + all TFicl
  - Frozen: TFcol + lower TFrow layers
  - Optimizer: AdamW, lr=1e-4, no scheduler, grad_clip=1.0
  - Stratified 3-fold CV, same folds and seeds for ID/EA
  - Budget: 500 steps, log validation every 50 steps
  - Metrics per fold: Best-of-500 validation Balanced Accuracy, Early-Curve AUC (mean over 50..500)
  - Decision rule across all pairs: EA wins if ≥70% pairs have Δbest>0 and median Δbest ≥ 0.5pp; else tie-breaker median ΔAUC>0.

Run: python scripts/dev/test_ea_ab_sign.py
"""

from __future__ import annotations

import argparse
import os
import sys
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

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
    if y is not None and not (isinstance(y, type(None))):
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


def mcar_mask_df(X: pd.DataFrame, p: float = 0.10, rng: np.random.RandomState | None = None) -> pd.DataFrame:
    rng = rng or np.random.RandomState(1)
    Xm = X.copy()
    n, d = Xm.shape
    m = int(np.round(p * n * d))
    if m <= 0:
        return Xm
    rows = rng.randint(0, n, size=m)
    cols = rng.randint(0, d, size=m)
    for r, c in zip(rows, cols):
        Xm.iat[r, c] = np.nan
    return Xm


def set_branch_mode(model, branch: str, enable_tfrow_ea: bool = True) -> None:
    # ID: force identity metric; EA: estimator
    if branch.upper() == "ID":
        for blk in model.icl_predictor.tf_icl.blocks:
            blk.elliptical = True
            blk.elliptical_override = "identity"
            blk.elliptical_manual_m = None
        for blk in model.row_interactor.tf_row.blocks:
            blk.elliptical = True
            blk.elliptical_override = "identity"
            blk.elliptical_manual_m = None
    else:
        for blk in model.icl_predictor.tf_icl.blocks:
            blk.elliptical = True
            blk.elliptical_override = "none"
            blk.elliptical_manual_m = None
        if enable_tfrow_ea:
            for blk in model.row_interactor.tf_row.blocks:
                blk.elliptical = True
                blk.elliptical_override = "none"
                blk.elliptical_manual_m = None
    model.train()  # keep grad flow; dropout is 0 in checkpoints


def freeze_for_protocol(model, adapt_tfrow_last_only: bool = True) -> None:
    # Freeze COL always
    for p in model.col_embedder.parameters():
        p.requires_grad = False

    # ROW: only last block trainable; lower blocks frozen
    n_blocks = len(model.row_interactor.tf_row.blocks)
    for i, blk in enumerate(model.row_interactor.tf_row.blocks):
        req = (i == n_blocks - 1) if adapt_tfrow_last_only else True
        for p in blk.parameters():
            p.requires_grad = req

    # ICL: all trainable
    for p in model.icl_predictor.parameters():
        p.requires_grad = True


def build_adapt_pack(clf: TabICLClassifier, X_val: pd.DataFrame):
    # Ensure numerical transformation consistent with training
    X_val_num = clf.X_encoder_.transform(X_val)
    data = clf.ensemble_generator_.transform(X_val_num)
    X_list, ytrain_list, pattern_list, offset_list = [], [], [], []
    for norm_method, (Xs, ys) in data.items():
        X_list.append(Xs)
        ytrain_list.append(ys)
        pattern_list.append(np.array(clf.ensemble_generator_.feature_shuffle_patterns_[norm_method]))
        offset_list.append(np.array(clf.ensemble_generator_.class_shift_offsets_[norm_method]))
    X_all = np.concatenate(X_list, axis=0)
    Ytrain_all = np.concatenate(ytrain_list, axis=0)
    patterns_all = np.concatenate(pattern_list, axis=0)
    offsets_all = np.concatenate(offset_list, axis=0)
    return X_all, Ytrain_all, patterns_all, offsets_all


@dataclass
class CurveLog:
    steps: List[int]
    bacc_clean: List[float]
    bacc_mcar: List[float]


def evaluate_branch(clf: TabICLClassifier, X_val: pd.DataFrame, y_val: pd.Series, branch: str) -> Tuple[float, float]:
    # Toggle branch mode for prediction
    if branch.upper() == "ID":
        set_branch_mode(clf.model_, "ID", enable_tfrow_ea=True)
    else:
        set_branch_mode(clf.model_, "EA", enable_tfrow_ea=True)
    y_pred = clf.predict(X_val)
    bacc = float(balanced_accuracy_score(y_val, y_pred))
    return bacc, bacc


def eval_clean_and_mcar(clf: TabICLClassifier, X_val: pd.DataFrame, y_val: pd.Series, branch: str) -> Tuple[float, float]:
    # Clean
    if branch.upper() == "ID":
        set_branch_mode(clf.model_, "ID", enable_tfrow_ea=True)
    else:
        set_branch_mode(clf.model_, "EA", enable_tfrow_ea=True)
    y_pred = clf.predict(X_val)
    bacc_clean = float(balanced_accuracy_score(y_val, y_pred))
    # MCAR 10%
    Xm = mcar_mask_df(X_val, p=0.10, rng=np.random.RandomState(1))
    if branch.upper() == "ID":
        set_branch_mode(clf.model_, "ID", enable_tfrow_ea=True)
    else:
        set_branch_mode(clf.model_, "EA", enable_tfrow_ea=True)
    y_pred_m = clf.predict(Xm)
    bacc_mcar = float(balanced_accuracy_score(y_val, y_pred_m))
    return bacc_clean, bacc_mcar


def adapt_curve(
    clf: TabICLClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    device: torch.device,
    branch: str,
    steps: int = 500,
    log_every: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    batch_variants: int = 8,
) -> CurveLog:
    # Prepare adaptation pack for validation set
    X_all, Yt_all, patterns_all, offsets_all = build_adapt_pack(clf, X_val)
    test_size = X_val.shape[0]

    # Encoded y_val once
    y_val_enc = torch.from_numpy(clf.y_encoder_.transform(y_val)).long().to(device)

    # Tensors on device
    X_all_t = torch.from_numpy(X_all).float().to(device)
    Yt_all_t = torch.from_numpy(Yt_all).float().to(device)
    offsets_all_t = torch.from_numpy(offsets_all).long().to(device)
    patterns_all_list = [p.tolist() for p in patterns_all]

    # Mode and freezing
    set_branch_mode(clf.model_, branch, enable_tfrow_ea=True)
    freeze_for_protocol(clf.model_, adapt_tfrow_last_only=True)

    # Optimizer
    params = [p for p in clf.model_.parameters() if p.requires_grad]
    opt = AdamW(params, lr=float(lr), weight_decay=float(weight_decay))

    N = X_all_t.shape[0]
    steps_list = list(range(0, steps + 1, log_every))
    log_steps: List[int] = []
    log_bacc_clean: List[float] = []
    log_bacc_mcar: List[float] = []

    clf.model_.train()

    # Initial eval (step 0)
    bc, bm = eval_clean_and_mcar(clf, X_val, y_val, branch)
    log_steps.append(0)
    log_bacc_clean.append(bc)
    log_bacc_mcar.append(bm)

    for step in range(1, steps + 1):
        # minibatch of ensemble variants
        start = ((step - 1) * batch_variants) % N
        end = min(start + batch_variants, N)
        if end <= start:
            start = 0
            end = min(batch_variants, N)
        sel = slice(start, end)

        Xb = X_all_t[sel]
        Ytb = Yt_all_t[sel]
        patb = patterns_all_list[start:end]
        offb = offsets_all_t[sel]

        # Apply per-variant feature permutations
        Xb_perm_list = []
        for i, pat in enumerate(patb):
            pi = torch.tensor(pat, dtype=torch.long, device=device)
            Xb_perm_list.append(Xb[i][:, pi])
        Xb_perm = torch.stack(Xb_perm_list, dim=0)  # (B, T, F)

        # Train-size and d vector
        train_size = Ytb.shape[1]
        d_vec = torch.full((Xb_perm.shape[0],), Xb_perm.shape[-1], dtype=torch.long, device=device)

        # Forward through training path to get logits for all positions
        out_full = clf.model_._train_forward(Xb_perm, y_train=Ytb.long(), d=d_vec, embed_with_test=False)  # (B, T, C)

        # Reverse class shift per item, then select test positions [train_size:]
        Bsz, Tsz, C = out_full.shape
        corr = []
        for i in range(Bsz):
            offs = int(offb[i].item())
            logits_i = out_full[i]
            if offs == 0:
                logits_corr = logits_i
            else:
                logits_corr = torch.cat([logits_i[:, offs:], logits_i[:, :offs]], dim=-1)
            # If model outputs only test positions (B, T-test, C), skip slice
            if logits_corr.shape[0] == y_val_enc.shape[0]:
                corr.append(logits_corr)
            else:
                corr.append(logits_corr[train_size:])
        out_corr = torch.stack(corr, dim=0)  # (B, test_size, C)

        # CE on validation labels (repeat across B)
        loss = F.cross_entropy(out_corr.reshape(-1, C), y_val_enc.repeat(out_corr.shape[0]))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(params, max_norm=1.0)
        opt.step()

        # Log every log_every steps
        if (step % log_every) == 0:
            bc, bm = eval_clean_and_mcar(clf, X_val, y_val, branch)
            log_steps.append(step)
            log_bacc_clean.append(bc)
            log_bacc_mcar.append(bm)

    return CurveLog(steps=log_steps, bacc_clean=log_bacc_clean, bacc_mcar=log_bacc_mcar)


def median(x: List[float]) -> float:
    return float(np.median(np.array(x, dtype=float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--checkpoint", default=os.path.join(REPO_ROOT, "checkpoints", "tabicl-classifier-v1.1-0506.ckpt"))
    ap.add_argument("--datasets", default="iris,wine,breast-w,ionosphere,sonar")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--n_estimators", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_variants", type=int, default=8)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    set_seed(args.random_state)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    datasets = [s.strip() for s in str(args.datasets).split(",") if s.strip()]

    deltas_best: List[float] = []
    deltas_auc: List[float] = []

    for ds_name in datasets:
        try:
            X, y, ds_resolved = fetch_openml_dataset(ds_name)
        except Exception as e:
            print(f"[WARN] Failed to load OpenML dataset '{ds_name}': {e}. Skipping.")
            continue

        skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=args.random_state)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Build ID and EA classifiers from same checkpoint and transforms
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

            # Adaptation curves
            curve_id = adapt_curve(
                clf_id,
                X_train,
                y_train,
                X_val,
                y_val,
                device=dev,
                branch="ID",
                steps=int(args.steps),
                log_every=int(args.log_every),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                batch_variants=int(args.batch_variants),
            )
            curve_ea = adapt_curve(
                clf_ea,
                X_train,
                y_train,
                X_val,
                y_val,
                device=dev,
                branch="EA",
                steps=int(args.steps),
                log_every=int(args.log_every),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                batch_variants=int(args.batch_variants),
            )

            # Compute per-fold metrics on clean validation bacc
            best_id = max(curve_id.bacc_clean)
            best_ea = max(curve_ea.bacc_clean)
            # AUC as mean over steps 50..500 (exclude step 0)
            auc_id = float(np.mean(curve_id.bacc_clean[1:])) if len(curve_id.bacc_clean) > 1 else best_id
            auc_ea = float(np.mean(curve_ea.bacc_clean[1:])) if len(curve_ea.bacc_clean) > 1 else best_ea

            db = best_ea - best_id
            da = auc_ea - auc_id
            deltas_best.append(db)
            deltas_auc.append(da)

            print(
                f"[{ds_resolved}][fold {fold}] best_bacc: ID={best_id:.3f} EA={best_ea:.3f} (Δ={db*100:.2f}pp) | "
                f"auc(mean 50..{args.steps}): ID={auc_id:.3f} EA={auc_ea:.3f} (Δ={da*100:.2f}pp)"
            )

    # Aggregate decision
    if len(deltas_best) == 0:
        print("No results to aggregate.")
        return

    proportion_pos = float(np.mean(np.array(deltas_best) > 0.0))
    median_delta = median(deltas_best)
    median_auc = median(deltas_auc) if len(deltas_auc) > 0 else 0.0
    print(
        f"Aggregate: Δbest>0 proportion={proportion_pos*100:.1f}% | median Δbest={median_delta*100:.2f}pp | "
        f"median Δauc={median_auc*100:.2f}pp"
    )

    primary_pass = (proportion_pos >= 0.70) and (median_delta >= 0.005)
    if primary_pass:
        print("Decision: PASS (primary) — proceed to full runs.")
    else:
        if median_auc > 0.0:
            print("Decision: PASS (tie-breaker: median ΔAUC > 0).")
        else:
            print("Decision: FAIL — unlikely to benefit.")


if __name__ == "__main__":
    main()
