#!/usr/bin/env python
"""Generate EA vs SA plots suggested in the results discussion.

Plots produced (saved to figures/ea_sa_plots by default):
1) behaviour_tradeoff.png       : ΔF1 vs ΔNLL per dataset (behaviour file)
2) delta_ece_box.png            : distribution of ΔECE per dataset
3) geometry_pairs.png           : paired SA vs EA values for mean cosine and purity@k
4) robustness_deltas.png        : mean Δacc and ΔNLL by corruption type
5) cc18_delta_logloss_hist.png  : histogram of per-dataset Δlog-loss on CC18
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "ea_sa_plots"

sns.set_style("whitegrid")


def _load_csv(filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def load_behaviour() -> pd.DataFrame:
    beh = _load_csv("compare_mini_tabicl_behaviour.csv")
    agg = beh.groupby("dataset").agg(
        acc_sa=("acc_sa", "mean"),
        acc_ea=("acc_ea", "mean"),
        f1_sa=("f1_sa", "mean"),
        f1_ea=("f1_ea", "mean"),
        nll_sa=("nll_sa", "mean"),
        nll_ea=("nll_ea", "mean"),
        brier_sa=("brier_sa", "mean"),
        brier_ea=("brier_ea", "mean"),
        ece_sa=("ece_sa", "mean"),
        ece_ea=("ece_ea", "mean"),
    )
    agg["delta_f1"] = agg["f1_ea"] - agg["f1_sa"]
    agg["delta_nll"] = agg["nll_ea"] - agg["nll_sa"]
    agg["delta_ece"] = agg["ece_ea"] - agg["ece_sa"]
    return agg.reset_index()


def load_geometry() -> pd.DataFrame:
    geom = _load_csv("compare_mini_tabicl_geometry.csv").copy()
    return geom


def load_robustness() -> pd.DataFrame:
    rob = _load_csv("compare_mini_tabicl_robustness.csv")
    group_cols = ["condition", "param_value"]
    agg = rob.groupby(group_cols).agg(
        acc_sa=("acc_sa", "mean"),
        acc_ea=("acc_ea", "mean"),
        f1_sa=("f1_sa", "mean"),
        f1_ea=("f1_ea", "mean"),
        nll_sa=("nll_sa", "mean"),
        nll_ea=("nll_ea", "mean"),
    )
    agg["delta_acc"] = agg["acc_ea"] - agg["acc_sa"]
    agg["delta_f1"] = agg["f1_ea"] - agg["f1_sa"]
    agg["delta_nll"] = agg["nll_ea"] - agg["nll_sa"]
    return agg.reset_index()


def load_cc18() -> pd.DataFrame:
    sa = _load_csv("cc18_tabicl_step-1000.csv")
    ea = _load_csv("cc18_tabicl_step-1000_ea.csv")
    merged = sa.merge(ea, on=["task_id", "dataset_id", "dataset_name"], suffixes=("_sa", "_ea"))
    valid = merged[merged["error_sa"].isna() & merged["error_ea"].isna()].copy()
    valid["delta_log_loss"] = valid["log_loss_mean_ea"] - valid["log_loss_mean_sa"]
    return valid


def behaviour_tradeoff_plot(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    ax = sns.scatterplot(data=df, x="delta_f1", y="delta_nll")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("ΔF1 (EA - SA)")
    plt.ylabel("ΔNLL (EA - SA)")
    plt.title("Behaviour trade-off per dataset")

    # Annotate the two most extreme NLL changes for quick reference
    worst = df.sort_values("delta_nll", ascending=False).head(1)
    best = df.sort_values("delta_nll", ascending=True).head(1)
    for _, row in pd.concat([worst, best]).iterrows():
        ax.text(row["delta_f1"], row["delta_nll"], row["dataset"], fontsize=8, ha="right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def delta_ece_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(4, 5))
    sns.boxplot(y=df["delta_ece"], color="#4C72B0")
    sns.stripplot(y=df["delta_ece"], color="#DD8452", alpha=0.6)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("ΔECE (EA - SA)")
    plt.title("Calibration change across datasets")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def geometry_pairs_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    def _paired_lines(ax, sa_col: str, ea_col: str, ylabel: str):
        for _, row in df.iterrows():
            ax.plot(["SA", "EA"], [row[sa_col], row[ea_col]], marker="o", alpha=0.6)
        ax.set_ylabel(ylabel)

    _paired_lines(axes[0], "mean_cos_sa", "mean_cos_ea", "Mean cosine similarity")
    axes[0].set_title("Representation tightness")

    _paired_lines(axes[1], "purity_sa_topk", "purity_ea_topk", "Purity@k")
    axes[1].set_title("Label purity of neighborhoods")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def robustness_deltas_plot(df: pd.DataFrame, out_path: Path) -> None:
    # Average over parameter values for readability
    cond_mean = df.groupby("condition").agg({"delta_acc": "mean", "delta_nll": "mean"}).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=cond_mean, x="condition", y="delta_acc", ax=axes[0], color="#4C72B0")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("ΔAcc (EA - SA)")
    axes[0].set_title("Accuracy under corruptions")

    sns.barplot(data=cond_mean, x="condition", y="delta_nll", ax=axes[1], color="#C44E52")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("ΔNLL (EA - SA)")
    axes[1].set_title("Log-loss under corruptions")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def cc18_delta_logloss_hist(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(df["delta_log_loss"], bins=20, kde=False, color="#4C72B0")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Δlog-loss (EA - SA)")
    plt.ylabel("# datasets")
    plt.title("CC18 per-dataset Δlog-loss")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EA vs SA plots")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help="Directory to write figures (default: figures/ea_sa_plots)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir: Path = args.out_dir

    beh = load_behaviour()
    geom = load_geometry()
    rob = load_robustness()
    cc18 = load_cc18()

    behaviour_tradeoff_plot(beh, out_dir / "behaviour_tradeoff.png")
    delta_ece_boxplot(beh, out_dir / "delta_ece_box.png")
    geometry_pairs_plot(geom, out_dir / "geometry_pairs.png")
    robustness_deltas_plot(rob, out_dir / "robustness_deltas.png")
    cc18_delta_logloss_hist(cc18, out_dir / "cc18_delta_logloss_hist.png")
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
