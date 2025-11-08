#!/usr/bin/env python3
r"""Elliptical Attention robustness test on small OpenML datasets.

Procedure per dataset:
  - Load vanilla TabICL checkpoint via sklearn wrapper; set model.eval()
  - Split into train/test (stratified)
  - Evaluate ON same weights:
      * EA-ON: parameter-free estimator active (M = \hat M), layer ≥ 2
      * EA-OFF: force identity metric (M = I)
    on clean test data
  - Apply small corruption to test set (Gaussian jitter 10% cells and/or MCAR 5–10%)
    Re-evaluate both modes on the same split
  - Compute robustness drop = (clean − corrupted) for Accuracy/Balanced Accuracy
  - Pass if: EA-ON has smaller drop than EA-OFF on ≥1 dataset and never clearly worse

Run: python scripts/test_ea_robustness.py
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Ensure local src package is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
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


def _extract_target_if_missing(X: pd.DataFrame, y: Optional[pd.Series], ds_name: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """If y is None, try to infer a classification target column from X and drop it from features."""
    if y is not None and not (isinstance(y, type(None))):
        return X, y
    cols = list(X.columns)
    # Preferred target column names
    preferred = ["class", "Class", "target", "Target", "label", "Label", "y"]
    for c in preferred:
        if c in cols:
            y_series = X[c].copy()
            X_feat = X.drop(columns=[c])
            return X_feat, y_series
    # Else choose a low-cardinality column (>=2 classes) or last column
    for c in cols:
        nunique = X[c].nunique(dropna=True)
        if 2 <= nunique <= min(50, max(2, X.shape[0] // 2)):
            y_series = X[c].copy()
            X_feat = X.drop(columns=[c])
            return X_feat, y_series
    # Fallback: use last column
    c = cols[-1]
    y_series = X[c].copy()
    X_feat = X.drop(columns=[c])
    return X_feat, y_series


def fetch_openml_dataset(name_or_id: str | int) -> Tuple[pd.DataFrame, pd.Series, str]:
    import openml

    if isinstance(name_or_id, int) or (str(name_or_id).isdigit()):
        ds = openml.datasets.get_dataset(int(name_or_id))
        X, y, _, _ = ds.get_data(target=getattr(ds, "default_target_attribute", None), dataset_format="dataframe")
        X, y = _extract_target_if_missing(X, y, ds_name=ds.name)
        return X, y, ds.name

    # Search by name (case-insensitive, exact match preferred; else contains)
    name = str(name_or_id).lower()
    df = openml.datasets.list_datasets(output_format="dataframe")
    exact = df[df["name"].str.lower() == name]
    if len(exact) == 0:
        contains = df[df["name"].str.lower().str.contains(name)]
        if len(contains) == 0:
            raise ValueError(f"OpenML dataset not found: {name_or_id}")
        # choose the smallest by rows
        row = contains.sort_values("NumberOfInstances").iloc[0]
    else:
        row = exact.sort_values("NumberOfInstances").iloc[0]
    did = int(row["did"]) if "did" in row else int(row["dataset_id"]) if "dataset_id" in row else int(row["ID"]) if "ID" in row else None
    if did is None:
        raise RuntimeError("Could not resolve dataset id from OpenML metadata row")
    ds = openml.datasets.get_dataset(did)
    X, y, _, _ = ds.get_data(target=getattr(ds, "default_target_attribute", None), dataset_format="dataframe")
    X, y = _extract_target_if_missing(X, y, ds_name=ds.name)
    return X, y, ds.name


def set_ea_mode(model, identity: bool) -> None:
    # Enable EA in TFrow and TFicl; force identity or use estimator
    for blk in model.row_interactor.tf_row.blocks:
        blk.elliptical = True
        blk.elliptical_override = "identity" if identity else "none"
        blk.elliptical_manual_m = None
    for blk in model.icl_predictor.tf_icl.blocks:
        blk.elliptical = True
        blk.elliptical_override = "identity" if identity else "none"
        blk.elliptical_manual_m = None
    model.eval()


def gaussian_jitter_df(X_test: pd.DataFrame, X_train: pd.DataFrame, p: float = 0.10, sigma: float = 0.05, rng: np.random.RandomState | None = None) -> pd.DataFrame:
    rng = rng or np.random.RandomState(0)
    Xj = X_test.copy()
    num_cols = Xj.select_dtypes(include=[np.number]).columns
    # Ensure numeric columns are float to accept additive noise cleanly
    for col in num_cols:
        try:
            Xj[col] = Xj[col].astype(float)
        except Exception:
            pass
    if len(num_cols) == 0:
        return Xj
    # Train stats
    mu = X_train[num_cols].mean(axis=0)
    sd = X_train[num_cols].std(axis=0).replace(0, 1.0)
    # Sample mask of cells to perturb
    n = len(Xj)
    for col in num_cols:
        idx = rng.rand(n) < float(p)
        noise = rng.normal(loc=0.0, scale=float(sigma) * float(sd[col]), size=idx.sum())
        vals = Xj.loc[idx, col].astype(float).values
        vals = vals + noise
        Xj.loc[idx, col] = vals
    return Xj


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
        col = Xm.columns[c]
        Xm.iat[r, c] = np.nan
    return Xm


@dataclass
class Scores:
    acc: float
    bacc: float


def eval_modes(clf: TabICLClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Scores, Scores]:
    # EA-OFF
    set_ea_mode(clf.model_, identity=True)
    y_pred_id = clf.predict(X_test)
    s_id = Scores(acc=float(accuracy_score(y_test, y_pred_id)), bacc=float(balanced_accuracy_score(y_test, y_pred_id)))
    # EA-ON
    set_ea_mode(clf.model_, identity=False)
    y_pred_ea = clf.predict(X_test)
    s_ea = Scores(acc=float(accuracy_score(y_test, y_pred_ea)), bacc=float(balanced_accuracy_score(y_test, y_pred_ea)))
    return s_id, s_ea


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--checkpoint", default=os.path.join(REPO_ROOT, "checkpoints", "tabicl-classifier-v1.1-0506.ckpt"))
    ap.add_argument("--datasets", default="iris,wine,breast-w,ionosphere,sonar,mfeat-morphological,credit-g,diabetes,analcatdata_dmft,ilpd,haberman,spectf,heart-statlog,australian,banknote-authentication,pima-indians-diabetes,seeds,log1p,yeast")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_estimators", type=int, default=8)
    ap.add_argument("--corrupt", choices=["jitter", "mcar", "both"], default="mcar")
    ap.add_argument("--jitter_p", type=float, default=0.10)
    ap.add_argument("--jitter_sigma", type=float, default=0.05)
    ap.add_argument("--mcar_p", type=float, default=0.10)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--strict", action="store_true", help="Exit with non-zero code if pass criteria not met")
    args = ap.parse_args()

    set_seed(args.random_state)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [s.strip() for s in str(args.datasets).split(",") if s.strip()]

    wins = 0
    losses = 0
    tol = 1e-3
    results = []

    for ds_name in datasets:
        try:
            X, y, ds_resolved = fetch_openml_dataset(ds_name)
        except Exception as e:
            print(f"[WARN] Failed to load OpenML dataset '{ds_name}': {e}. Skipping.")
            continue

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(args.test_size), random_state=args.random_state, stratify=y
        )

        # Build classifier with checkpoint and fit transforms
        clf = TabICLClassifier(
            device=dev,
            model_path=args.checkpoint,
            allow_auto_download=False,
            use_hierarchical=True,
            n_estimators=int(args.n_estimators),
            random_state=args.random_state,
            verbose=False,
        )
        clf.fit(X_train, y_train)

        # Clean eval
        s_id_clean, s_ea_clean = eval_modes(clf, X_test, y_test)

        # Corruptions
        drops = {}
        if args.corrupt in ("jitter", "both"):
            Xj = gaussian_jitter_df(X_test, X_train, p=args.jitter_p, sigma=args.jitter_sigma, rng=np.random.RandomState(0))
            s_id_cor, s_ea_cor = eval_modes(clf, Xj, y_test)
            drop_id_acc = s_id_clean.acc - s_id_cor.acc
            drop_ea_acc = s_ea_clean.acc - s_ea_cor.acc
            drop_id_ba = s_id_clean.bacc - s_id_cor.bacc
            drop_ea_ba = s_ea_clean.bacc - s_ea_cor.bacc
            drops["jitter"] = dict(acc_id=drop_id_acc, acc_ea=drop_ea_acc, bacc_id=drop_id_ba, bacc_ea=drop_ea_ba)
        if args.corrupt in ("mcar", "both"):
            Xm = mcar_mask_df(X_test, p=args.mcar_p, rng=np.random.RandomState(1))
            s_id_cor, s_ea_cor = eval_modes(clf, Xm, y_test)
            drop_id_acc = s_id_clean.acc - s_id_cor.acc
            drop_ea_acc = s_ea_clean.acc - s_ea_cor.acc
            drop_id_ba = s_id_clean.bacc - s_id_cor.bacc
            drop_ea_ba = s_ea_clean.bacc - s_ea_cor.bacc
            drops["mcar"] = dict(acc_id=drop_id_acc, acc_ea=drop_ea_acc, bacc_id=drop_id_ba, bacc_ea=drop_ea_ba)

        # Decide win/lose for this dataset: if any corruption shows smaller drop for EA (strictly by tol)
        ds_win = False
        ds_lose = False
        for k, v in drops.items():
            win_acc = (v["acc_ea"] < v["acc_id"] - tol)
            win_ba = (v["bacc_ea"] < v["bacc_id"] - tol)
            lose_acc = (v["acc_ea"] > v["acc_id"] + tol)
            lose_ba = (v["bacc_ea"] > v["bacc_id"] + tol)
            ds_win = ds_win or win_acc or win_ba
            ds_lose = ds_lose or lose_acc or lose_ba
        wins += 1 if ds_win else 0
        losses += 1 if ds_lose else 0

        # Log
        entry = {
            "dataset": ds_resolved,
            "acc_clean_id": s_id_clean.acc,
            "acc_clean_ea": s_ea_clean.acc,
            "bacc_clean_id": s_id_clean.bacc,
            "bacc_clean_ea": s_ea_clean.bacc,
            **{f"drop_{k}_acc_id": v["acc_id"] for k, v in drops.items()},
            **{f"drop_{k}_acc_ea": v["acc_ea"] for k, v in drops.items()},
            **{f"drop_{k}_bacc_id": v["bacc_id"] for k, v in drops.items()},
            **{f"drop_{k}_bacc_ea": v["bacc_ea"] for k, v in drops.items()},
            "win": ds_win,
            "lose": ds_lose,
        }
        results.append(entry)

        if args.verbose:
            print(f"[{ds_resolved}] clean acc: ID={s_id_clean.acc:.3f} EA={s_ea_clean.acc:.3f}; bacc: ID={s_id_clean.bacc:.3f} EA={s_ea_clean.bacc:.3f}")
            for k, v in drops.items():
                print(
                    f"  [{k}] drop acc: ID={v['acc_id']:.3f} EA={v['acc_ea']:.3f}; drop bacc: ID={v['bacc_id']:.3f} EA={v['bacc_ea']:.3f}"
                )
            print(f"  outcome: win={ds_win} lose={ds_lose}")

    # Summary and pass criteria
    print("Dataset-level results:")
    for r in results:
        msg = f"- {r['dataset']}: clean acc ID={r['acc_clean_id']:.3f}, EA={r['acc_clean_ea']:.3f}; clean bacc ID={r['bacc_clean_id']:.3f}, EA={r['bacc_clean_ea']:.3f}"
        drop_keys = [k for k in r.keys() if k.startswith("drop_")]
        for dk in sorted(drop_keys):
            msg += f" | {dk}={r[dk]:.3f}"
        msg += f" | win={r['win']} lose={r['lose']}"
        print(msg)

    passed = (wins >= 1) and (losses == 0)
    print(f"Robustness pass: {passed} (wins={wins}, losses={losses})")
    if (not passed) and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
