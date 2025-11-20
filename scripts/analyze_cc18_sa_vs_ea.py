#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_and_join(sa_csv: Path, ea_csv: Path) -> pd.DataFrame:
    """Load SA and EA benchmark CSVs and return joined per-dataset frame."""
    df_sa = pd.read_csv(sa_csv)
    df_ea = pd.read_csv(ea_csv)

    def _filter_ok(df: pd.DataFrame) -> pd.DataFrame:
        err = df.get("error")
        if err is None:
            return df
        return df[(err.isna()) | (err == "") | (err.str.lower().isin(["nan", "none"]))]

    df_sa = _filter_ok(df_sa)
    df_ea = _filter_ok(df_ea)

    join_keys = ["task_id", "dataset_id", "dataset_name"]
    df = df_sa.merge(df_ea, on=join_keys, suffixes=("_sa", "_ea"))

    # Basic sanity: task_type should match
    if "task_type_sa" in df.columns and "task_type_ea" in df.columns:
        mismatch = (df["task_type_sa"] != df["task_type_ea"]).sum()
        if mismatch:
            print(f"Warning: {mismatch} rows have mismatched task_type between SA and EA.")

    # Per-dataset deltas
    df["delta_accuracy"] = df["accuracy_mean_ea"] - df["accuracy_mean_sa"]
    df["delta_f1"] = df["f1_macro_mean_ea"] - df["f1_macro_mean_sa"]
    df["delta_logloss"] = df["log_loss_mean_ea"] - df["log_loss_mean_sa"]
    return df


def plot_scatter_f1(df: pd.DataFrame, output: Path) -> None:
    """Scatter: F1_EA vs F1_SA, colored by task_type, size by n_rows."""
    f1_sa = df["f1_macro_mean_sa"].to_numpy()
    f1_ea = df["f1_macro_mean_ea"].to_numpy()
    task_type = df.get("task_type_sa", df.get("task_type", pd.Series(["Unknown"] * len(df))))
    n_rows = df.get("n_rows_sa", df.get("n_rows", pd.Series([1] * len(df)))).to_numpy()

    size = 20 + 10 * np.log10(np.clip(n_rows, 10, None))
    size = np.clip(size, 20, 80)

    colors = {"Binary": "#4C72B0", "Multiclass": "#55A868"}
    default_color = "#C44E52"

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
        }
    )

    fig, ax = plt.subplots(figsize=(4.0, 3.5))

    for t in sorted(task_type.unique()):
        mask = task_type == t
        if not mask.any():
            continue
        ax.scatter(
            f1_sa[mask],
            f1_ea[mask],
            s=size[mask],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.4,
            label=str(t),
            color=colors.get(str(t), default_color),
        )

    min_val = min(f1_sa.min(), f1_ea.min())
    max_val = max(f1_sa.max(), f1_ea.max())
    pad = 0.02
    lo, hi = max(0.0, min_val - pad), min(1.0, max_val + pad)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=0.8)

    ax.set_xlabel("F1 macro (SA)")
    ax.set_ylabel("F1 macro (EA)")
    ax.set_title("Per-dataset F1: EA vs SA")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False, title="task_type")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_histograms(df: pd.DataFrame, output: Path) -> None:
    """Histograms of per-dataset deltas for F1 and log-loss."""
    delta_f1 = df["delta_f1"].to_numpy()
    delta_ll = df["delta_logloss"].to_numpy()

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # ΔF1
    ax = axes[0]
    ax.hist(delta_f1, bins=15, color="#4C72B0", edgecolor="black", alpha=0.8)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Δ F1 macro (EA − SA)")
    ax.set_ylabel("#datasets")
    ax.set_title("Distribution of F1 deltas")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Δ log-loss
    ax = axes[1]
    ax.hist(delta_ll, bins=15, color="#C44E52", edgecolor="black", alpha=0.8)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Δ log loss (EA − SA)")
    ax.set_ylabel("#datasets")
    ax.set_title("Distribution of log-loss deltas")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for ax in axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def stratified_summary(df: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    """Return stratified summary table of mean deltas by type/size/imbalance."""
    task_type = df.get("task_type_sa", df.get("task_type"))
    n_rows = df.get("n_rows_sa", df.get("n_rows"))
    imbalance = df.get("imbalance_ratio_sa", df.get("imbalance_ratio"))

    df = df.copy()
    df["task_type_group"] = task_type
    df["size_bin"] = pd.cut(
        n_rows,
        bins=[0, 1000, 5000, 20000, np.inf],
        labels=["<1k", "1k–5k", "5k–20k", ">20k"],
    )
    df["imbalance_bin"] = pd.cut(
        imbalance,
        bins=[0, 1.5, 5.0, np.inf],
        labels=["≈balanced", "moderate", "high"],
    )

    group_cols = ["task_type_group", "size_bin", "imbalance_bin"]
    agg = (
        df.groupby(group_cols)
        .agg(
            n_datasets=("dataset_id", "size"),
            mean_delta_f1=("delta_f1", "mean"),
            mean_delta_logloss=("delta_logloss", "mean"),
        )
        .reset_index()
    )

    agg = agg[agg["n_datasets"] >= min_n]
    agg = agg.sort_values(["task_type_group", "size_bin", "imbalance_bin"])
    return agg


def print_top_bottom(df: pd.DataFrame, k: int = 5) -> None:
    """Print top/bottom-k datasets by ΔF1."""
    df_sorted = df.sort_values("delta_f1")

    cols: List[str] = [
        "dataset_name",
        "n_rows_sa",
        "n_features_sa",
        "n_classes_sa",
        "imbalance_ratio_sa",
        "accuracy_mean_sa",
        "accuracy_mean_ea",
        "f1_macro_mean_sa",
        "f1_macro_mean_ea",
        "delta_f1",
        "log_loss_mean_sa",
        "log_loss_mean_ea",
        "delta_logloss",
    ]
    cols = [c for c in cols if c in df_sorted.columns]

    bottom = df_sorted.head(k)[cols]
    top = df_sorted.tail(k).iloc[::-1][cols]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    print("\nWorst datasets for EA (ΔF1 < 0):")
    print(bottom.to_string(index=False))

    print("\nBest datasets for EA (ΔF1 > 0):")
    print(top.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-dataset analysis of SA vs EA on OpenML CC18.",
    )
    parser.add_argument(
        "--sa_csv",
        type=Path,
        default=Path("results/cc18_tabicl_step-1000.csv"),
        help="CSV with SA benchmark results.",
    )
    parser.add_argument(
        "--ea_csv",
        type=Path,
        default=Path("results/cc18_tabicl_step-1000_ea.csv"),
        help="CSV with EA benchmark results.",
    )
    parser.add_argument(
        "--fig_prefix",
        type=Path,
        default=Path("figures/cc18_sa_vs_ea"),
        help="Prefix for output figures.",
    )
    parser.add_argument(
        "--min_group_size",
        type=int,
        default=3,
        help="Minimum #datasets per group in stratified summary.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top/bottom datasets to show.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.sa_csv.is_file():
        raise SystemExit(f"SA results file not found: {args.sa_csv}")
    if not args.ea_csv.is_file():
        raise SystemExit(f"EA results file not found: {args.ea_csv}")

    df = load_and_join(args.sa_csv, args.ea_csv)
    print(f"Joined {len(df)} datasets with valid results for both SA and EA.")

    fig_prefix: Path = args.fig_prefix

    # 1) Scatter: EA vs SA F1
    plot_scatter_f1(df, fig_prefix.with_name(fig_prefix.name + "_f1_scatter.png"))

    # 2) Histograms of per-dataset deltas
    plot_delta_histograms(df, fig_prefix.with_name(fig_prefix.name + "_delta_hist.png"))

    # 3) Stratified summaries
    summary = stratified_summary(df, min_n=args.min_group_size)
    print("\nStratified summary (mean deltas by task_type, size, imbalance):")
    print(summary.to_string(index=False))

    summary_out = fig_prefix.with_name(fig_prefix.name + "_stratified_deltas.csv")
    summary.to_csv(summary_out, index=False)
    print(f"\nSaved stratified summary to {summary_out}")

    # 4) Top/bottom-k table
    print_top_bottom(df, k=args.topk)


if __name__ == "__main__":
    main()

