#!/usr/bin/env python
from __future__ import annotations

"""
Group CC18 per-dataset ΔNLL by simple OpenML metadata.

This script:
  1. Loads two per-dataset CC18 CSVs (e.g. self-attention vs EA ICL-only),
  2. Pulls OpenML metadata for each dataset_id,
  3. Extracts:
       - number of numeric features
       - number of categorical features
       - sparsity (fraction of zeros over numeric features)
  4. Labels datasets as "numeric-dominated" if > threshold of features are numeric,
  5. Computes mean ΔNLL per group.

ΔNLL is defined as:
    ΔNLL = log_loss_mean_B - log_loss_mean_A
so positive values mean model B has higher (worse) NLL.

Usage example (from repo root):
  python scripts/analysis/cc18_nll_by_meta_groups.py \
      --csv_a results/cc18_tabicl_step-1000_self-attention_per_dataset.csv \
      --csv_b results/cc18_tabicl_step-1000_ea_icl_only_per_dataset.csv \
      --label_a SA \
      --label_b EA
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openml  # type: ignore
import pandas as pd


DEFAULT_METRIC_COL = "log_loss_mean"
DEFAULT_THRESHOLD = 0.70


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Group CC18 per-dataset ΔNLL by simple OpenML metadata "
            "(numeric vs categorical feature counts and sparsity)."
        )
    )
    p.add_argument(
        "--csv_a",
        type=str,
        required=True,
        help="Per-dataset CSV for model A (e.g. self-attention).",
    )
    p.add_argument(
        "--csv_b",
        type=str,
        required=True,
        help="Per-dataset CSV for model B (e.g. EA ICL-only).",
    )
    p.add_argument(
        "--label_a",
        type=str,
        default="A",
        help="Label for model A used in printed output.",
    )
    p.add_argument(
        "--label_b",
        type=str,
        default="B",
        help="Label for model B used in printed output.",
    )
    p.add_argument(
        "--metric_col",
        type=str,
        default=DEFAULT_METRIC_COL,
        help=f"Metric column to treat as NLL (default: {DEFAULT_METRIC_COL}).",
    )
    p.add_argument(
        "--numeric_threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Threshold on fraction of numeric features to label a dataset as "
            f"'numeric-dominated' (default: {DEFAULT_THRESHOLD:.2f})."
        ),
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save per-dataset table with metadata and ΔNLL.",
    )
    return p.parse_args()


def _load_per_dataset_metric(csv_path: Path, metric_col: str) -> pd.DataFrame:
    """Load per-dataset CSV and extract (dataset_id, dataset_name, metric)."""
    df = pd.read_csv(csv_path)
    for col in ("dataset_id", "dataset_name"):
        if col not in df.columns:
            raise RuntimeError(f"Expected column '{col}' in {csv_path}")
    if metric_col not in df.columns:
        raise RuntimeError(f"Metric column '{metric_col}' not found in {csv_path}")

    out = df[["dataset_id", "dataset_name", metric_col]].copy()
    out["dataset_id"] = out["dataset_id"].astype(int)
    return out


def _compute_sparsity(X: pd.DataFrame) -> float:
    """
    Compute sparsity as the fraction of zeros over all numeric features.

    Non-numeric columns are ignored. Returns NaN if there are no numeric features.
    """
    num_df = X.select_dtypes(include=[np.number])
    if num_df.empty:
        return float("nan")
    total = num_df.size
    if total == 0:
        return float("nan")
    zeros = (num_df == 0).sum().sum()
    return float(zeros) / float(total)


def _fetch_openml_metadata(dataset_id: int) -> Dict[str, float]:
    """
    Fetch simple metadata from OpenML for a given dataset_id.

    Uses OpenML's dataset to infer:
      - n_numeric_features (from categorical_indicator)
      - n_categorical_features
      - sparsity over numeric features (fraction of zeros)
    """
    ds = openml.datasets.get_dataset(dataset_id)
    X, y, cat_ind, _ = ds.get_data(
        target=ds.default_target_attribute, dataset_format="dataframe"
    )

    cat_ind_arr = np.asarray(cat_ind, dtype=bool)
    n_features = int(len(cat_ind_arr))
    n_categorical = int(cat_ind_arr.sum())
    n_numeric = n_features - n_categorical

    sparsity = _compute_sparsity(X)

    return {
        "n_numeric_features": float(n_numeric),
        "n_categorical_features": float(n_categorical),
        "sparsity": float(sparsity),
    }


def _bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Bootstrap CI for mean(values)."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        v = float(values[0])
        return v, v

    rng = np.random.default_rng(0)
    n = values.size
    means = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[b] = float(values[idx].mean())

    alpha = 1.0 - ci
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lower, upper


def main() -> None:
    args = parse_args()

    csv_a = Path(args.csv_a)
    csv_b = Path(args.csv_b)
    if not csv_a.is_file():
        raise SystemExit(f"CSV A not found: {csv_a}")
    if not csv_b.is_file():
        raise SystemExit(f"CSV B not found: {csv_b}")

    df_a = _load_per_dataset_metric(csv_a, args.metric_col)
    df_b = _load_per_dataset_metric(csv_b, args.metric_col)

    merged = df_a.merge(
        df_b,
        on=["dataset_id", "dataset_name"],
        suffixes=(f"_{args.label_a}", f"_{args.label_b}"),
    )
    if merged.empty:
        raise SystemExit("No overlapping datasets between the two CSV files.")

    # Compute ΔNLL as model B minus model A
    col_a = f"{args.metric_col}_{args.label_a}"
    col_b = f"{args.metric_col}_{args.label_b}"
    merged["delta_nll"] = merged[col_b] - merged[col_a]

    # Fetch OpenML metadata per dataset_id
    meta_records: List[Dict[str, float]] = []
    for did in merged["dataset_id"].unique():
        try:
            meta = _fetch_openml_metadata(int(did))
        except Exception as e:
            # Keep going but record NaNs if metadata fails
            meta = {
                "n_numeric_features": float("nan"),
                "n_categorical_features": float("nan"),
                "sparsity": float("nan"),
            }
            print(f"[WARN] Failed to fetch OpenML metadata for dataset {did}: {e}")
        meta["dataset_id"] = int(did)
        meta_records.append(meta)

    meta_df = pd.DataFrame.from_records(meta_records)
    merged = merged.merge(meta_df, on="dataset_id", how="left")

    # Compute feature fractions and a simple group label
    total_feats = merged["n_numeric_features"] + merged["n_categorical_features"]
    merged["frac_numeric_features"] = merged["n_numeric_features"] / total_feats.replace(
        {0.0: np.nan}
    )

    thr = float(args.numeric_threshold)
    def _group(row: pd.Series) -> str:
        f = row["frac_numeric_features"]
        if not np.isfinite(f):
            return "unknown"
        return "numeric-dominated" if f >= thr else "other"

    merged["group"] = merged.apply(_group, axis=1)

    # Save per-dataset table if requested
    if args.output_csv is not None:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        print(f"Wrote per-dataset table with metadata and ΔNLL to: {out_path}")

    # Aggregate ΔNLL per group
    print(
        f"\nΔNLL = {args.metric_col}_{args.label_b} - {args.metric_col}_{args.label_a} "
        "(positive = model B worse NLL)"
    )
    print(f"Numeric-dominated threshold: frac_numeric_features >= {thr:.2f}\n")

    rows: List[Dict[str, float]] = []
    for grp, sub in merged.groupby("group"):
        vals = sub["delta_nll"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            mean = std = float("nan")
            ci_lo = ci_hi = float("nan")
        else:
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
            ci_lo, ci_hi = _bootstrap_ci(vals)

        rows.append(
            {
                "group": grp,
                "n_datasets": int(vals.size),
                "delta_nll_mean": mean,
                "delta_nll_std": std,
                "delta_nll_ci_lower": ci_lo,
                "delta_nll_ci_upper": ci_hi,
            }
        )

    summary_df = pd.DataFrame.from_records(rows)
    print("Mean ΔNLL per group:")
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}" if np.isfinite(x) else "nan",
        )
    )


if __name__ == "__main__":
    main()
