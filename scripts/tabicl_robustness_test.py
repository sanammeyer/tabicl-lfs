#!/usr/bin/env python3
"""
Literature-aligned robustness diagnostics for a single TabICL checkpoint.

Focuses on perturbations that match existing tabular foundation model robustness evaluations
(e.g., TabPFN) and common ICL-demonstration corruption tests.

This script:
  1) Uses TabPFN-style *uninformative feature injection*
     (append shuffled copies of existing features; preserves marginals + missingness).
  2) Uses TabPFN-style *cell-wise outliers* for numeric columns
     (with probability p per cell, multiply by U(0, outlier_factor)).
  3) Keeps label-flip poisoning (demonstration/ICL corruption).
  4) Optionally keeps context-outlier-row injection (ICL-specific).

It evaluates a single TabICL checkpoint under these corruptions and logs metrics
to ./results/compare_mini_tabicl_robustness_lit*.csv (optionally with a postfix).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

# Make local src importable
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tabicl.sklearn.classifier import TabICLClassifier


# Default OpenML IDs: Panel B (robustness characterization set)
DEFAULT_OPENML_IDS: List[int] = [
    14969,   # GesturePhaseSegmentationProcessed
    14952,   # PhishingWebsites
    7592,    # adult
    14965,   # bank-marketing
    219,     # electricity
    9977,    # nomao
    167141,  # churn
    43,      # spambase
    9976,    # madelon
    9964,    # semeion
    146824,  # mfeat-pixel
    28,      # optdigits
    32,      # pendigits
    2074,    # satimage
    146822,  # segment
    9960,    # wall-robot-navigation
    9952,    # phoneme
    146817,  # steel-plates-fault
    3021,    # sick
    146820,  # wilt
    9978,    # ozone-level-8hr
    3918,    # pc1
    3,       # kr-vs-kp
    49,      # tic-tac-toe
    146821,  # car
    31,      # credit-g
    29,      # credit-approval
    14954,   # cylinder-bands
]


# ---------------------------------------------------------------------------
# Generic utilities (dataset loading, metrics, CSV appending)
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set basic RNG seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _append_rows_csv(
    rows: List[Dict[str, Any]],
    out_path: Path,
    columns: Sequence[str],
) -> None:
    """Append rows to a CSV, keeping schema aligned with current implementation.

    If an existing file has a different header, it is moved to <name>.bak and a new
    file is started with the current columns.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = True
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                first = f.readline().strip()
            existing_cols = first.split(",") if first else []
            if existing_cols == list(columns):
                header = False
            else:
                backup = out_path.with_suffix(out_path.suffix + ".bak")
                out_path.rename(backup)
                print(
                    f"[info] Existing CSV schema for '{out_path.name}' differs from current metrics; "
                    f"moved old file to '{backup.name}' and started a new one."
                )
        except Exception:
            header = True

    df = pd.DataFrame(rows, columns=list(columns))
    df.to_csv(out_path, mode="a", index=False, header=header)


def _extract_target_if_missing(
    X: pd.DataFrame,
    y: pd.Series | None,
    ds_name: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """If y is None, try to infer a classification target column from X and drop it."""
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

    # Fallback: use last column
    c = cols[-1]
    y_series = X[c].copy()
    X_feat = X.drop(columns=[c])
    return X_feat, y_series


def fetch_openml_dataset(name_or_id: str | int) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Load an OpenML dataset by ID or (partial) name, inferring target if needed."""
    import openml

    # By integer ID: try dataset id first, then task id
    if isinstance(name_or_id, int) or (str(name_or_id).isdigit()):
        num_id = int(name_or_id)
        ds = None
        # Try as dataset id
        try:
            ds = openml.datasets.get_dataset(num_id)
        except Exception:
            ds = None
        # If that fails, try as task id and resolve to dataset id
        if ds is None:
            try:
                task = openml.tasks.get_task(num_id)
                ds_id = int(getattr(task, "dataset_id"))
                ds = openml.datasets.get_dataset(ds_id)
            except Exception as e:
                raise ValueError(f"Could not load OpenML dataset or task with id={num_id}: {e}")
        X, y, _, _ = ds.get_data(
            target=getattr(ds, "default_target_attribute", None),
            dataset_format="dataframe",
        )
        X, y = _extract_target_if_missing(X, y, ds_name=ds.name)
        return X, y, ds.name

    # By name: prefer exact match; else substring on name
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
    X, y, _, _ = ds.get_data(
        target=getattr(ds, "default_target_attribute", None),
        dataset_format="dataframe",
    )
    X, y = _extract_target_if_missing(X, y, ds_name=ds.name)
    return X, y, ds.name


def compute_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) over max-probability bins."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if proba.ndim != 2:
        raise ValueError("proba must be 2D (n_samples, n_classes)")

    confidences = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    correct = (preds == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i > 0:
            mask = (confidences > lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        conf_bin = confidences[mask].mean()
        acc_bin = correct[mask].mean()
        ece += (mask.mean()) * abs(acc_bin - conf_bin)
    return float(ece)


@dataclass
class BehaviourMetrics:
    accuracy: float
    f1_macro: float
    log_loss: float
    ece: float


def _eval_metrics_from_fitted_clf(
    clf: TabICLClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> BehaviourMetrics:
    """Evaluate BehaviourMetrics using an already-fitted classifier."""
    proba = clf.predict_proba(X_test)
    y_pred = clf.y_encoder_.inverse_transform(np.argmax(proba, axis=1))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    ll = log_loss(y_test, proba, labels=clf.classes_)
    y_true_int = clf.y_encoder_.transform(y_test)
    ece = compute_ece(y_true_int, proba)
    return BehaviourMetrics(
        accuracy=float(acc),
        f1_macro=float(f1),
        log_loss=float(ll),
        ece=float(ece),
    )


def _flip_labels(
    y: pd.Series,
    frac: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Randomly flip a fraction of labels to other classes."""
    if frac <= 0.0:
        return y

    y_poison = y.reset_index(drop=True).copy()
    n = len(y_poison)
    n_flip = int(round(frac * n))
    if n_flip == 0:
        return y

    classes = np.asarray(sorted(pd.unique(y_poison)))
    if classes.size < 2:
        return y

    idx = rng.choice(n, size=n_flip, replace=False)
    for i in idx:
        current = y_poison.iloc[i]
        choices = classes[classes != current]
        if choices.size == 0:
            continue
        y_poison.iloc[i] = rng.choice(choices)

    return y_poison


def _poison_context(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    poison_frac: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Add synthetic outlier rows to the training context only, keep test clean."""
    if poison_frac <= 0.0:
        return X_train, y_train, X_test, y_test

    n_out = int(round(poison_frac * len(X_train)))
    if n_out == 0:
        return X_train, y_train, X_test, y_test

    num_cols = list(X_train.select_dtypes(include="number").columns)
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    num_stats: Dict[str, Tuple[float, float]] = {}
    for col in num_cols:
        m = float(X_train[col].mean())
        s = float(X_train[col].std() or 1.0)
        num_stats[col] = (m, s)

    cat_values = {c: X_train[c].dropna().unique() for c in cat_cols}

    rows: List[Dict[str, Any]] = []
    for _ in range(n_out):
        row: Dict[str, Any] = {}
        for col in num_cols:
            m, s = num_stats[col]
            row[col] = rng.normal(loc=m, scale=3.0 * s)  # heavy-tailed numeric outlier
        for col in cat_cols:
            vals = cat_values.get(col)
            if vals is not None and len(vals) > 0:
                row[col] = rng.choice(vals)
            else:
                row[col] = np.nan
        rows.append(row)

    X_out = pd.DataFrame(rows, columns=X_train.columns)
    y_out = rng.choice(y_train.values, size=n_out)

    X_train_poisoned = pd.concat(
        [X_train.reset_index(drop=True), X_out.reset_index(drop=True)],
        ignore_index=True,
    )
    y_train_poisoned = pd.concat(
        [
            y_train.reset_index(drop=True),
            pd.Series(y_out, index=range(len(y_train), len(y_train) + n_out)),
        ],
        ignore_index=True,
    )

    return X_train_poisoned, y_train_poisoned, X_test, y_test


# ---------------------------------------------------------------------------
# Literature-aligned corruptions (TabPFN-style)
# ---------------------------------------------------------------------------


def add_uninformative_features_shuffled(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_add: int,
    rng: np.random.Generator,
    *,
    with_replacement: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    TabPFN-style uninformative features:
      - pick existing columns
      - create copies whose values are randomly permuted across rows
        (preserves marginal distribution + missingness, breaks label association)

    We permute train and test independently to preserve each split's marginals.
    Works for numeric + categorical/object columns, because we permute raw values.
    """
    if n_add <= 0:
        return X_train, X_test

    cols = list(X_train.columns)
    if len(cols) == 0:
        return X_train, X_test

    if with_replacement:
        picked = rng.choice(cols, size=n_add, replace=True)
    else:
        n_add = min(n_add, len(cols))
        picked = rng.choice(cols, size=n_add, replace=False)

    X_tr = X_train.reset_index(drop=True).copy()
    X_te = X_test.reset_index(drop=True).copy()

    for i, c in enumerate(picked):
        new_c = f"uninformative_shuffle_{c}_{i}"
        tr_vals = X_tr[c].to_numpy(copy=True)
        te_vals = X_te[c].to_numpy(copy=True)

        tr_perm = rng.permutation(tr_vals)
        te_perm = rng.permutation(te_vals)

        X_tr[new_c] = tr_perm
        X_te[new_c] = te_perm

    return X_tr, X_te


def apply_tabpfn_cell_outliers(
    X: pd.DataFrame,
    rng: np.random.Generator,
    *,
    p_cell: float = 0.02,
    outlier_factor: float = 50.0,
    numeric_only: bool = True,
) -> pd.DataFrame:
    """
    TabPFN-style cell-wise outliers (numeric features):
      - For each numeric cell independently with probability p_cell,
        multiply the value by u ~ Uniform(0, outlier_factor).
    """
    if outlier_factor <= 0 or p_cell <= 0:
        return X

    X_out = X.copy()
    num_cols = (
        list(X_out.select_dtypes(include="number").columns)
        if numeric_only
        else list(X_out.columns)
    )
    if not num_cols:
        return X_out

    for c in num_cols:
        col = X_out[c].to_numpy(copy=True)
        finite = np.isfinite(col)
        mask = (rng.random(size=col.shape[0]) < p_cell) & finite
        if not np.any(mask):
            continue
        mult = rng.uniform(0.0, float(outlier_factor), size=int(mask.sum()))
        col[mask] = col[mask] * mult
        X_out[c] = col

    return X_out


# ---------------------------------------------------------------------------
# Main robustness panel for a single checkpoint
# ---------------------------------------------------------------------------


def run_literature_robustness_panel(
    ds_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    checkpoint: Path,
    device: str,
    n_estimators: int,
    seed: int,
    *,
    uninformative_ns: Sequence[int],
    outlier_factors: Sequence[float],
    outlier_p_cell: float,
    outliers_apply_to: str,  # "train"|"test"|"both"
    label_poison_fracs: Sequence[float],
    context_outlier_fracs: Sequence[float],
    use_hierarchical: bool,
    results_postfix: str = "",
) -> List[Dict[str, Any]]:
    """
    Run robustness diagnostics for a single TabICL checkpoint on one dataset.

    Returns a list of rows with metrics under different corruption conditions.
    """
    print(f"\n=== Literature-aligned robustness (single model): {ds_name} ===")
    rng = np.random.default_rng(seed)

    # Train/test split (similar to other robustness scripts, modest test size)
    stratify = y if y.nunique() > 1 else None
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=None,
        )

    robustness_rows: List[Dict[str, Any]] = []

    def _fit_clf(X_train: pd.DataFrame, y_train: pd.Series) -> TabICLClassifier:
        clf = TabICLClassifier(
            device=device,
            model_path=str(checkpoint),
            allow_auto_download=False,
            use_hierarchical=use_hierarchical,
            n_estimators=n_estimators,
            random_state=seed,
            verbose=False,
        )
        clf.fit(X_train, y_train)
        return clf

    # ---- Clean baseline
    print("  Clean baseline:")
    clf_clean = _fit_clf(X_tr, y_tr)
    m_clean = _eval_metrics_from_fitted_clf(clf_clean, X_te, y_te)

    robustness_rows.append(
        {
            "dataset": ds_name,
            "condition": "clean",
            "param_type": "none",
            "param_value": 0.0,
            "n_test": len(X_te),
            "acc": m_clean.accuracy,
            "f1": m_clean.f1_macro,
            "nll": m_clean.log_loss,
            "ece": m_clean.ece,
            "seed": seed,
            "n_estimators": n_estimators,
            "use_hierarchical": bool(use_hierarchical),
            "checkpoint": str(checkpoint),
        }
    )

    # ---- (1) Uninformative feature injection (TabPFN-style)
    for n_add in [n for n in uninformative_ns if n > 0]:
        print(f"  Uninformative features (shuffled copies): n_add={n_add}")
        X_tr_u, X_te_u = add_uninformative_features_shuffled(X_tr, X_te, n_add, rng)

        clf_u = _fit_clf(X_tr_u, y_tr)
        m_u = _eval_metrics_from_fitted_clf(clf_u, X_te_u, y_te)

        robustness_rows.append(
            {
                "dataset": ds_name,
                "condition": "uninformative_features_shuffled",
                "param_type": "n_add",
                "param_value": int(n_add),
                "n_test": len(X_te_u),
                "acc": m_u.accuracy,
                "f1": m_u.f1_macro,
                "nll": m_u.log_loss,
                "ece": m_u.ece,
                "seed": seed,
                "n_estimators": n_estimators,
                "use_hierarchical": bool(use_hierarchical),
                "checkpoint": str(checkpoint),
            }
        )

    # ---- (2) Cell-wise outliers (TabPFN-style)
    positive_factors = [f for f in outlier_factors if f and f > 0.0]
    if positive_factors:
        for fac in positive_factors:
            print(
                f"  Cell-wise numeric outliers: factor={fac}, "
                f"p_cell={outlier_p_cell}, apply_to={outliers_apply_to}"
            )
            X_tr_o, X_te_o = X_tr, X_te
            if outliers_apply_to in ("train", "both"):
                X_tr_o = apply_tabpfn_cell_outliers(
                    X_tr_o,
                    rng,
                    p_cell=outlier_p_cell,
                    outlier_factor=fac,
                )
            if outliers_apply_to in ("test", "both"):
                X_te_o = apply_tabpfn_cell_outliers(
                    X_te_o,
                    rng,
                    p_cell=outlier_p_cell,
                    outlier_factor=fac,
                )

            clf_o = _fit_clf(X_tr_o, y_tr)
            m_o = _eval_metrics_from_fitted_clf(clf_o, X_te_o, y_te)

            robustness_rows.append(
                {
                    "dataset": ds_name,
                    "condition": "cell_outliers_tabpfn",
                    "param_type": "outlier_factor",
                    "param_value": float(fac),
                    "n_test": len(X_te_o),
                    "acc": m_o.accuracy,
                    "f1": m_o.f1_macro,
                    "nll": m_o.log_loss,
                    "ece": m_o.ece,
                    "seed": seed,
                    "n_estimators": n_estimators,
                    "use_hierarchical": bool(use_hierarchical),
                    "checkpoint": str(checkpoint),
                }
            )

    # ---- (3) Demonstration corruption: label flips on context labels
    for frac in [f for f in label_poison_fracs if f > 0.0]:
        print(f"  Label flip poisoning (context labels): frac={frac}")
        y_tr_poison = _flip_labels(y_tr, frac, rng)

        clf_lp = _fit_clf(X_tr, y_tr_poison)
        m_lp = _eval_metrics_from_fitted_clf(clf_lp, X_te, y_te)

        robustness_rows.append(
            {
                "dataset": ds_name,
                "condition": "label_flip_poison",
                "param_type": "frac",
                "param_value": float(frac),
                "n_test": len(X_te),
                "acc": m_lp.accuracy,
                "f1": m_lp.f1_macro,
                "nll": m_lp.log_loss,
                "ece": m_lp.ece,
                "seed": seed,
                "n_estimators": n_estimators,
                "use_hierarchical": bool(use_hierarchical),
                "checkpoint": str(checkpoint),
            }
        )

    # ---- (4) Optional: ICL-specific context poisoning via synthetic outlier rows
    for frac in [f for f in context_outlier_fracs if f > 0.0]:
        print(f"  Context outlier-row injection (train only): frac={frac}")
        X_tr_p, y_tr_p, X_te_c, y_te_c = _poison_context(
            X_tr,
            y_tr,
            X_te,
            y_te,
            frac,
            rng,
        )

        clf_cp = _fit_clf(X_tr_p, y_tr_p)
        m_cp = _eval_metrics_from_fitted_clf(clf_cp, X_te_c, y_te_c)

        robustness_rows.append(
            {
                "dataset": ds_name,
                "condition": "context_outlier_rows",
                "param_type": "frac",
                "param_value": float(frac),
                "n_test": len(X_te_c),
                "acc": m_cp.accuracy,
                "f1": m_cp.f1_macro,
                "nll": m_cp.log_loss,
                "ece": m_cp.ece,
                "seed": seed,
                "n_estimators": n_estimators,
                "use_hierarchical": bool(use_hierarchical),
                "checkpoint": str(checkpoint),
            }
        )

    # Persist
    if robustness_rows:
        rob_filename = (
            f"compare_mini_tabicl_robustness_lit{results_postfix}.csv"
            if results_postfix
            else "compare_mini_tabicl_robustness_lit.csv"
        )
        out_path = REPO_ROOT / "results" / rob_filename

        cols = [
            "dataset",
            "condition",
            "param_type",
            "param_value",
            "n_test",
            "acc",
            "f1",
            "nll",
            "ece",
            "seed",
            "n_estimators",
            "use_hierarchical",
            "checkpoint",
        ]
        _append_rows_csv(robustness_rows, out_path, cols)

    return robustness_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Literature-aligned robustness diagnostics for a single mini-TabICL checkpoint."
    )

    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the TabICL checkpoint to evaluate.",
    )

    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu).",
    )
    ap.add_argument("--n_estimators", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        help="Optional list of seeds for a small sweep. When provided, overrides --seed.",
    )
    ap.add_argument(
        "--use_hierarchical",
        action="store_true",
        help="Enable hierarchical classification in TabICLClassifier.",
    )

    # Dataset selection: OpenML IDs
    ap.add_argument(
        "--openml_ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of OpenML dataset IDs. If omitted, uses a default CC18-like panel.",
    )

    # Literature-aligned robustness knobs
    ap.add_argument(
        "--uninformative_ns",
        type=int,
        nargs="*",
        default=[0, 10, 50],
        help="How many shuffled-copy uninformative features to append (TabPFN-style).",
    )
    ap.add_argument(
        "--outlier_factors",
        type=float,
        nargs="*",
        default=[0.0, 10.0, 50.0],
        help="Outlier factor(s) for cell-wise numeric outliers (TabPFN-style). 0 disables.",
    )
    ap.add_argument(
        "--outlier_p_cell",
        type=float,
        default=0.02,
        help="Per-cell probability for cell-wise outliers (TabPFN-style).",
    )
    ap.add_argument(
        "--outliers_apply_to",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Apply cell-wise outliers to train, test, or both.",
    )
    ap.add_argument(
        "--label_poison_fracs",
        type=float,
        nargs="*",
        default=[0.0, 0.1],
        help="Fractions of training labels to flip (ICL demonstration corruption).",
    )
    ap.add_argument(
        "--context_outlier_fracs",
        type=float,
        nargs="*",
        default=[0.0],
        help="Fractions of synthetic outlier rows to inject into training context (ICL-specific).",
    )

    ap.add_argument(
        "--results_postfix",
        type=str,
        default="",
        help="Optional string appended before '.csv' in result filenames (e.g. '_run2').",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Choose dataset list
    if args.openml_ids is not None and len(args.openml_ids) > 0:
        openml_ids = list(args.openml_ids)
    else:
        openml_ids = list(DEFAULT_OPENML_IDS)

    # Determine seeds to run
    if args.seeds:
        seed_values = list(dict.fromkeys(args.seeds))
    else:
        seed_values = [int(args.seed)]

    all_rows: List[Dict[str, Any]] = []
    for seed in seed_values:
        set_seed(seed)
        if len(seed_values) > 1:
            print(f"\n##### Seed {seed} #####")

        for ds_id in openml_ids:
            try:
                X, y, ds_name = fetch_openml_dataset(int(ds_id))
            except Exception as e:
                print(f"[WARN] Could not load OpenML dataset '{ds_id}': {e}. Skipping.")
                continue

            rows = run_literature_robustness_panel(
                ds_name=ds_name,
                X=X,
                y=y,
                checkpoint=ckpt,
                device=device,
                n_estimators=args.n_estimators,
                seed=seed,
                uninformative_ns=args.uninformative_ns,
                outlier_factors=args.outlier_factors,
                outlier_p_cell=args.outlier_p_cell,
                outliers_apply_to=args.outliers_apply_to,
                label_poison_fracs=args.label_poison_fracs,
                context_outlier_fracs=args.context_outlier_fracs,
                use_hierarchical=args.use_hierarchical,
                results_postfix=args.results_postfix,
            )
            all_rows.extend(rows)

    # Simple summary: mean ΔNLL vs clean per condition/param
    if all_rows:
        df = pd.DataFrame(all_rows)
        clean = df[df["condition"] == "clean"][["dataset", "nll"]].rename(
            columns={"nll": "nll_clean"}
        )
        non_clean = df[df["condition"] != "clean"]
        if not non_clean.empty:
            merged = non_clean.merge(clean, on="dataset", how="left")
            merged["delta_nll_vs_clean"] = merged["nll"] - merged["nll_clean"]
            print("\n=== Summary: mean ΔNLL (corrupted - clean) ===")
            summary = (
                merged.groupby(["condition", "param_type", "param_value"])[
                    "delta_nll_vs_clean"
                ]
                .mean()
                .sort_values()
            )
            print(summary)


if __name__ == "__main__":
    main()
