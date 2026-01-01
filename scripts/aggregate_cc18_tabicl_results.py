#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_RESULTS_PATH = Path("results/cc18_tabicl_step-1000_ea_row.csv")
DEFAULT_METRICS: List[str] = [
    "accuracy",
    "f1_macro",
    "f1_micro",
    "precision_macro",
    "recall_macro",
    "roc_auc",
    "log_loss",
    "fit_seconds",
    "predict_seconds",
]


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int | None = None,
) -> Tuple[float, float]:
    """Bootstrap CI for the mean of `values`."""
    if n_bootstrap <= 0:
        return float("nan"), float("nan")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        v = float(values[0])
        return v, v

    rng = np.random.default_rng(random_state)
    n = values.size
    means = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[b] = float(values[idx].mean())

    alpha = 1.0 - ci
    lower_q, upper_q = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    lower = float(lower_q)
    upper = float(upper_q)
    return lower, upper


def summarize_per_dataset(
    csv_path: Path,
    metrics: List[str],
    n_bootstrap: int = 0,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Summarize per-fold OpenML-CC18 results into per-dataset metrics.

    For each dataset (dataset_id, dataset_name), computes:
      - mean and std over folds for each metric in `metrics`
      - optional bootstrap CI over folds for the mean if n_bootstrap > 0
        (stored as {metric}_foldboot_ci_lower / {metric}_foldboot_ci_upper)
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"No rows found in CSV file {csv_path}")

    if "error" in df.columns:
        ok_mask = df["error"].isna() | (df["error"] == "")
        df = df[ok_mask]

    if df.empty:
        raise RuntimeError(f"No successful rows (error column empty) in {csv_path}")

    if "fold" in df.columns:
        df = df[df["fold"].notna()]

    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        raise RuntimeError(f"None of the requested metrics {metrics} are present in {csv_path}")

    group_cols = ["dataset_id", "dataset_name"]
    for col in group_cols:
        if col not in df.columns:
            raise RuntimeError(f"Expected column '{col}' not found in {csv_path}")

    grouped = df.groupby(group_cols, dropna=False)

    rows: List[Dict[str, float]] = []
    for (dataset_id, dataset_name), g in grouped:
        row: Dict[str, float] = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "n_folds": int(len(g)),
        }

        for m in available_metrics:
            vals = g[m].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{m}_n"] = int(vals.size)
            if vals.size == 0:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
                if n_bootstrap > 0:
                    row[f"{m}_foldboot_ci_lower"] = float("nan")
                    row[f"{m}_foldboot_ci_upper"] = float("nan")
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std

            if n_bootstrap > 0:
                rs = seed + int(dataset_id) if pd.notna(dataset_id) else seed
                lower, upper = bootstrap_ci(vals, n_bootstrap=n_bootstrap, ci=ci, random_state=rs)
                row[f"{m}_foldboot_ci_lower"] = lower
                row[f"{m}_foldboot_ci_upper"] = upper

        rows.append(row)

    return pd.DataFrame(rows)

def load_per_dataset_means(
    csv_path: Path,
    metrics: List[str],
) -> pd.DataFrame:
    """
    Load per-fold CSV and return per-dataset mean per metric.
    Index is (dataset_id, dataset_name).
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"No rows found in CSV file {csv_path}")

    # Keep only successful fold rows
    if "error" in df.columns:
        ok = df["error"].isna() | (df["error"] == "")
        df = df[ok]
    if "fold" in df.columns:
        df = df[df["fold"].notna()]

    if df.empty:
        raise RuntimeError(f"No successful fold rows in {csv_path}")

    group_cols = ["dataset_id", "dataset_name"]
    for c in group_cols:
        if c not in df.columns:
            raise RuntimeError(f"Expected column '{c}' not found in {csv_path}")

    available = [m for m in metrics if m in df.columns]
    if not available:
        raise RuntimeError(f"None of requested metrics {metrics} present in {csv_path}")

    g = df.groupby(group_cols, dropna=False)

    # Mean over folds per dataset for each metric
    out = g[available].mean()

    # Drop non-finite (important for roc_auc/log_loss edge cases)
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.where(np.isfinite(out), np.nan)      

    return out  # columns = metrics, index = (dataset_id, dataset_name)


def paired_bootstrap_delta_ci(
    deltas: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    random_state: int | None = None,
) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_lower, ci_upper) for mean(delta)."""
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return float("nan"), float("nan"), float("nan")
    if deltas.size == 1:
        v = float(deltas[0])
        return v, v, v

    rng = np.random.default_rng(random_state)
    n = deltas.size
    means = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[b] = float(deltas[idx].mean())

    alpha = 1.0 - ci
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    point = float(deltas.mean())
    return point, lo, hi



def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-fold OpenML-CC18 TabICL results into per-dataset metrics "
            "with mean/std and optional bootstrap confidence intervals."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_RESULTS_PATH),
        help=f"Input per-fold results CSV (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV path for aggregated per-dataset results "
             "(default: '<input>_per_dataset.csv')",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metric columns to aggregate (default: common classification metrics)",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="If set, compute bootstrap confidence intervals for each metric mean",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples (only used when --bootstrap is set)",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap intervals (default: 0.95)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap resampling",
    )
    parser.add_argument(
    "--csv_a",
    type=str,
    default=None,
    help="Model A per-fold results CSV (baseline). If set with --csv_b, runs paired delta bootstrap.",
    )
    parser.add_argument(
        "--csv_b",
        type=str,
        default=None,
        help="Model B per-fold results CSV (comparison). If set with --csv_a, runs paired delta bootstrap.",
    )
    parser.add_argument(
        "--label_a",
        type=str,
        default="A",
        help="Label for model A in printed output (default: A)",
    )
    parser.add_argument(
        "--label_b",
        type=str,
        default="B",
        help="Label for model B in printed output (default: B)",
    )
    parser.add_argument(
        "--lower_is_better",
        type=str,
        nargs="*",
        default=["log_loss", "fit_seconds", "predict_seconds"],
        help="Metrics where lower values are better; deltas will be sign-flipped for reporting.",
    )


    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    
    # ----- Compare mode (paired delta bootstrap) -----
    if args.csv_a is not None or args.csv_b is not None:
        if args.csv_a is None or args.csv_b is None:
            raise SystemExit("Use both --csv_a and --csv_b for compare mode.")

        csv_a = Path(args.csv_a)
        csv_b = Path(args.csv_b)
        if not csv_a.is_file():
            raise SystemExit(f"CSV A not found: {csv_a}")
        if not csv_b.is_file():
            raise SystemExit(f"CSV B not found: {csv_b}")

        n_bootstrap = args.n_bootstrap if args.bootstrap else 0

        A = load_per_dataset_means(csv_a, args.metrics)
        B = load_per_dataset_means(csv_b, args.metrics)

        # Inner join on datasets present in both
        common_idx = A.index.intersection(B.index)
        A = A.loc[common_idx]
        B = B.loc[common_idx]

        print(f"Compare mode: {args.label_b} - {args.label_a}")
        print(f"Common datasets: {len(common_idx)}")

        for metric in [m for m in args.metrics if m in A.columns and m in B.columns]:
            a_vals = A[metric].to_numpy(dtype=float)
            b_vals = B[metric].to_numpy(dtype=float)

            mask = np.isfinite(a_vals) & np.isfinite(b_vals)
            a_vals = a_vals[mask]
            b_vals = b_vals[mask]
            if a_vals.size == 0:
                continue

            deltas = b_vals - a_vals
            LOWER_IS_BETTER = {"log_loss", "fit_seconds", "predict_seconds"}

            deltas = b_vals - a_vals
            if metric in LOWER_IS_BETTER:
                deltas = -deltas   # flip so: positive = improvement for ALL metrics

            point = float(deltas.mean())
            line = (
                f"{metric:16s} mean_delta={point:.6f} "
                f"(n_datasets={deltas.size}) "
                f"win/tie/loss="
                f"{np.mean(deltas>0):.3f}/{np.mean(deltas==0):.3f}/{np.mean(deltas<0):.3f}"
            )

            if args.bootstrap and deltas.size > 1:
                point, lo, hi = paired_bootstrap_delta_ci(
                    deltas,
                    n_bootstrap=args.n_bootstrap,
                    ci=args.ci,
                    random_state=args.seed,
                )
                line += f" ci[{args.ci*100:.1f}%]=({lo:.6f}, {hi:.6f})"

            print(line)

        return

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    n_bootstrap = args.n_bootstrap if args.bootstrap else 0
    summary_df = summarize_per_dataset(
        csv_path=csv_path,
        metrics=args.metrics,
        n_bootstrap=n_bootstrap,
        ci=args.ci,
        seed=args.seed,
    )

    if args.output is not None:
        out_path = Path(args.output)
    else:
        out_path = csv_path.with_name(csv_path.stem + "_per_dataset.csv")

    summary_df.to_csv(out_path, index=False)

    print(f"Read per-fold results from: {csv_path}")
    print(f"Wrote per-dataset summary to: {out_path}")

    if "n_folds" in summary_df.columns:
        dist = summary_df["n_folds"].value_counts().sort_index().to_dict()
        print(f"n_folds distribution per dataset: {dist}")

    # For overall summary, prefer datasets with full 10 folds when available
    summary_df_eval = summary_df
    if "n_folds" in summary_df.columns:
        full_mask = summary_df["n_folds"] == 10
        if full_mask.any():
            summary_df_eval = summary_df[full_mask].copy()
            print("Overall summary uses datasets with n_folds == 10.")
        else:
            print("No datasets with n_folds == 10; using all datasets for overall summary.")

    print("\nOverall summary across datasets:")

    for metric in args.metrics:
        col = f"{metric}_mean"
        if col not in summary_df_eval.columns:
            continue

        vals = summary_df_eval[col].to_numpy(dtype=float)
        if vals.size == 0:
            continue

        overall_mean = float(vals.mean())
        overall_std = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")

        line = (
            f"{metric:16s} mean={overall_mean:.6f} "
            f"std={overall_std:.6f} (n_datasets={vals.size})"
        )

        if args.bootstrap and vals.size > 1:
            lower, upper = bootstrap_ci(
                vals,
                n_bootstrap=n_bootstrap,
                ci=args.ci,
                random_state=args.seed + 1,
            )
            line += (
                f" dataset-bootstrap_ci[{args.ci * 100:.1f}%]"
                f"=({lower:.6f}, {upper:.6f})"
            )

        print(line)


if __name__ == "__main__":
    main()
