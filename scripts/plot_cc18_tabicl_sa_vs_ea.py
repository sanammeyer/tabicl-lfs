#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

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
    metric_name: str,
    means: List[float],
    stds: List[float],
    labels: List[str],
    ns: List[int],
    title: str,
    output_path: Path,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    x = np.arange(len(labels), dtype=float)

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
        }
    )

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.bar(
        x,
        means,
        yerr=stds,
        color=colors[: len(labels)],
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric_name)

    if ylim is not None:
        ax.set_ylim(*ylim)

    n_info = ", ".join(f"{lab}: n={n}" for lab, n in zip(labels, ns))
    ax.set_title(f"{title} – {metric_name}\n({n_info})")

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
        "--standard_csv",
        type=Path,
        default=Path("results/cc18_tabicl_tabicl-classifier-v1.1-0506.csv"),
        help="CSV with standard TabICL classifier benchmark results.",
    )
    parser.add_argument(
        "--sa_label",
        type=str,
        default="mini-TabICL(SA)",
        help="Label for the SA checkpoint.",
    )
    parser.add_argument(
        "--ea_label",
        type=str,
        default="mini-TabICL(EA)",
        help="Label for the EA checkpoint.",
    )
    parser.add_argument(
        "--standard_label",
        type=str,
        default="TabICL",
        help="Label for the standard checkpoint.",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("figures/cc18_tabicl_three_ckpts_step-1000"),
        help=(
            "Prefix for the output figures; "
            "files '<prefix>_accuracy.png', '<prefix>_macro_f1.png', "
            "and '<prefix>_logloss.png' will be created."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="cc18 TabICL SA vs EA (step-1000)",
        help="Title for the plot.",
    )
    parser.add_argument(
        "--ylim_accuracy",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Optional y-axis limits for the accuracy plot.",
    )
    parser.add_argument(
        "--ylim_macro_f1",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Optional y-axis limits for the macro-F1 plot.",
    )
    parser.add_argument(
        "--ylim_logloss",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Optional y-axis limits for the log-loss plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.ea_csv.is_file():
        raise SystemExit(f"EA results file not found: {args.ea_csv}")
    if not args.sa_csv.is_file():
        raise SystemExit(f"SA results file not found: {args.sa_csv}")
    if not args.standard_csv.is_file():
        raise SystemExit(f"Standard results file not found: {args.standard_csv}")

    ea_acc_arr, ea_f1_arr, ea_ll_arr = load_metrics(args.ea_csv)
    sa_acc_arr, sa_f1_arr, sa_ll_arr = load_metrics(args.sa_csv)
    std_acc_arr, std_f1_arr, std_ll_arr = load_metrics(args.standard_csv)

    def mean_std(arr: np.ndarray) -> Tuple[float, float]:
        if arr.size == 0:
            return float("nan"), float("nan")
        if arr.size == 1:
            return float(arr.mean()), 0.0
        return float(arr.mean()), float(arr.std(ddof=1))

    ea_means = (
        mean_std(ea_acc_arr)[0],
        mean_std(ea_f1_arr)[0],
        mean_std(ea_ll_arr)[0],
    )
    ea_stds = (
        mean_std(ea_acc_arr)[1],
        mean_std(ea_f1_arr)[1],
        mean_std(ea_ll_arr)[1],
    )

    sa_means = (
        mean_std(sa_acc_arr)[0],
        mean_std(sa_f1_arr)[0],
        mean_std(sa_ll_arr)[0],
    )
    sa_stds = (
        mean_std(sa_acc_arr)[1],
        mean_std(sa_f1_arr)[1],
        mean_std(sa_ll_arr)[1],
    )

    std_means = (
        mean_std(std_acc_arr)[0],
        mean_std(std_f1_arr)[0],
        mean_std(std_ll_arr)[0],
    )
    std_stds = (
        mean_std(std_acc_arr)[1],
        mean_std(std_f1_arr)[1],
        mean_std(std_ll_arr)[1],
    )

    labels = [args.sa_label, args.ea_label, args.standard_label]
    ns = [sa_acc_arr.size, ea_acc_arr.size, std_acc_arr.size]

    # Print summary statistics to stdout
    print("Summary statistics (mean ± std over datasets):")
    metrics = ["Accuracy", "Macro-F1", "Log loss"]
    all_means = [sa_means, ea_means, std_means]
    all_stds = [sa_stds, ea_stds, std_stds]
    for metric_idx, metric_name in enumerate(metrics):
        print(f"\n{metric_name}:")
        for label, n, means, stds in zip(labels, ns, all_means, all_stds):
            m = means[metric_idx]
            s = stds[metric_idx]
            print(f"  {label:>16}: {m:.4f} ± {s:.4f} (n={n})")

    output_prefix: Path = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Accuracy
    make_plot(
        metric_name="Accuracy",
        means=[sa_means[0], ea_means[0], std_means[0]],
        stds=[sa_stds[0], ea_stds[0], std_stds[0]],
        labels=labels,
        ns=ns,
        title=args.title,
        output_path=output_prefix.with_name(output_prefix.name + "_accuracy.png"),
        ylim=tuple(args.ylim_accuracy) if args.ylim_accuracy is not None else None,
    )

    # Macro-F1
    make_plot(
        metric_name="Macro-F1",
        means=[sa_means[1], ea_means[1], std_means[1]],
        stds=[sa_stds[1], ea_stds[1], std_stds[1]],
        labels=labels,
        ns=ns,
        title=args.title,
        output_path=output_prefix.with_name(output_prefix.name + "_macro_f1.png"),
        ylim=tuple(args.ylim_macro_f1) if args.ylim_macro_f1 is not None else None,
    )

    # Log loss
    make_plot(
        metric_name="Log loss",
        means=[sa_means[2], ea_means[2], std_means[2]],
        stds=[sa_stds[2], ea_stds[2], std_stds[2]],
        labels=labels,
        ns=ns,
        title=args.title,
        output_path=output_prefix.with_name(output_prefix.name + "_logloss.png"),
        ylim=tuple(args.ylim_logloss) if args.ylim_logloss is not None else None,
    )


if __name__ == "__main__":
    main()
