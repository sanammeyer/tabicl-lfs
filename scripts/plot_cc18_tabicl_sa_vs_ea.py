#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays of (accuracy, f1_macro, logloss) over successful datasets."""
    accuracies: List[float] = []
    f1_macros: List[float] = []
    log_losses: List[float] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            error = row.get("error")
            # Keep only successful runs (no error message)
            if error not in (None, "", "nan", "NaN"):
                continue

            try:
                acc = float(row["accuracy_mean"])
                f1 = float(row["f1_macro_mean"])
                logloss = float(row["log_loss_mean"])
            except (KeyError, ValueError):
                # Skip rows that do not have the expected metrics
                continue

            accuracies.append(acc)
            f1_macros.append(f1)
            log_losses.append(logloss)

    if not accuracies:
        raise RuntimeError(f"No valid metric rows found in {csv_path}")

    return np.asarray(accuracies), np.asarray(f1_macros), np.asarray(log_losses)


def make_plot(
    ea_means: Tuple[float, float, float],
    ea_stds: Tuple[float, float, float],
    sa_means: Tuple[float, float, float],
    sa_stds: Tuple[float, float, float],
    n_ea: int,
    n_sa: int,
    title: str,
    output_path: Path,
) -> None:
    metrics = ["Accuracy", "Macro-F1", "Log loss"]
    ea_vals = list(ea_means)
    sa_vals = list(sa_means)
    ea_err = list(ea_stds)
    sa_err = list(sa_stds)

    x = np.arange(len(metrics), dtype=float)
    width = 0.32

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.bar(
        x - width / 2,
        sa_vals,
        width,
        yerr=sa_err,
        label="SA",
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )
    ax.bar(
        x + width / 2,
        ea_vals,
        width,
        yerr=ea_err,
        label="EA",
        color="#55A868",
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title(title + f" (EA n={n_ea}, SA n={n_sa})")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare cc18 TabICL SA vs EA averaged over datasets "
            "for accuracy, macro-F1 and log loss."
        )
    )
    parser.add_argument(
        "--ea_csv",
        type=Path,
        default=Path("results/cc18_tabicl_step-1000_ea.csv"),
        help="CSV with EA benchmark results.",
    )
    parser.add_argument(
        "--sa_csv",
        type=Path,
        default=Path("results/cc18_tabicl_step-1000.csv"),
        help="CSV with SA benchmark results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cc18_tabicl_sa_vs_ea_step-1000.png"),
        help="Output path for the comparison plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="cc18 TabICL SA vs EA (step-1000)",
        help="Title for the plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.ea_csv.is_file():
        raise SystemExit(f"EA results file not found: {args.ea_csv}")
    if not args.sa_csv.is_file():
        raise SystemExit(f"SA results file not found: {args.sa_csv}")

    ea_acc_arr, ea_f1_arr, ea_ll_arr = load_metrics(args.ea_csv)
    sa_acc_arr, sa_f1_arr, sa_ll_arr = load_metrics(args.sa_csv)

    n_ea = ea_acc_arr.size
    n_sa = sa_acc_arr.size

    ea_means = (ea_acc_arr.mean(), ea_f1_arr.mean(), ea_ll_arr.mean())
    ea_stds = (ea_acc_arr.std(ddof=1), ea_f1_arr.std(ddof=1), ea_ll_arr.std(ddof=1))
    sa_means = (sa_acc_arr.mean(), sa_f1_arr.mean(), sa_ll_arr.mean())
    sa_stds = (sa_acc_arr.std(ddof=1), sa_f1_arr.std(ddof=1), sa_ll_arr.std(ddof=1))

    make_plot(
        ea_means=ea_means,
        ea_stds=ea_stds,
        sa_means=sa_means,
        sa_stds=sa_stds,
        n_ea=n_ea,
        n_sa=n_sa,
        title=args.title,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
