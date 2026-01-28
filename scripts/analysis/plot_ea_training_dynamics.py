#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    return (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=max(1, window // 5))
        .mean()
        .to_numpy()
    )


def _load_csv(csv_path: Path, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, usecols=["step", metric])
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["step", metric]).sort_values("step").reset_index(drop=True)
    return df["step"].to_numpy(dtype=float), df[metric].to_numpy(dtype=float)


def _pretty_label(metric: str) -> str:
    # Keep it simple + LaTeX-friendly.
    if metric.endswith("_sparsity"):
        return r"Anisotropy ($\mathrm{mean}_{h}\,\mathrm{std}_{d}(m_{h,d})$)"
    if metric.endswith("_scale_mean"):
        return r"Mean ellipse scale ($\mathbb{E}[m]$)"
    if metric.endswith("_feat_velocity"):
        return r"Feature velocity ($\mathbb{E}|v-v_{\mathrm{prev}}|$)"
    if metric.endswith("_q_norm"):
        return r"Mean query norm ($\mathbb{E}\|q\|$)"
    if metric.endswith("_logit_mean"):
        return r"Approx. logit scale ($\mathbb{E}[\|q\|\|k\|/\sqrt{d}]$)"
    if metric.endswith("_context_div"):
        return r"Context diversity ($\mathrm{std}_{\mathrm{samples}}(m)$)"
    return metric


def _plot_two_stage(
    stage1_csv: Path,
    stage2_csv: Path,
    metric: str,
    out_path: Path,
    rolling_stage1: int,
    rolling_stage2: int,
    color: str = "#1f77b4",
) -> None:
    s1_step, s1_y = _load_csv(stage1_csv, metric)
    s2_step, s2_y = _load_csv(stage2_csv, metric)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)

    ylab = _pretty_label(metric)

    ax1.plot(s1_step, _rolling_mean(s1_y, rolling_stage1), color=color, linewidth=1.5)
    ax1.set_title("Stage 1: Adaptation")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel(ylab)
    ax1.grid(True, alpha=0.3)

    ax2.plot(s2_step, _rolling_mean(s2_y, rolling_stage2), color=color, linewidth=1.5)
    ax2.set_title("Stage 2: Stability (Long Context)")
    ax2.set_xlabel("Steps")
    ax2.grid(True, alpha=0.3)

    # Nice x-limits
    if s1_step.size:
        ax1.set_xlim(float(s1_step.min()), float(s1_step.max()))
    if s2_step.size:
        ax2.set_xlim(float(s2_step.min()), float(s2_step.max()))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _iter_metrics(default_metric: str, extra: Iterable[str], all_default: bool) -> Iterable[str]:
    if all_default:
        return (
            "icl_L2_sparsity",
            "icl_L2_scale_mean",
            "icl_L2_feat_velocity",
            "icl_L2_logit_mean",
            "icl_L12_sparsity",
            "icl_L12_scale_mean",
            "icl_L12_feat_velocity",
            "icl_L12_logit_mean",
        )
    metrics = [default_metric]
    for m in extra:
        if m not in metrics:
            metrics.append(m)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot EA training dynamics (Stage 1 vs Stage 2) into PDF(s).")
    parser.add_argument(
        "--stage1_csv",
        default="training_metrics/mini_tabicl_stage1_ea_icl_only.csv",
        help="Stage-1 EA metrics CSV.",
    )
    parser.add_argument(
        "--stage2_csv",
        default="training_metrics/mini_tabicl_stage2_ea_icl_only.csv",
        help="Stage-2 EA metrics CSV.",
    )
    parser.add_argument(
        "--metric",
        default="icl_L2_sparsity",
        help="Metric column name to plot (default: icl_L2_sparsity).",
    )
    parser.add_argument(
        "--also",
        nargs="*",
        default=[],
        help="Additional metric columns to also plot (each saved as its own PDF).",
    )
    parser.add_argument(
        "--all_default_metrics",
        action="store_true",
        help="Plot a default set of EA metrics (overrides --metric/--also).",
    )
    parser.add_argument(
        "--out_dir",
        default="results/training_metrics_analysis",
        help="Output directory.",
    )
    parser.add_argument(
        "--rolling_stage1",
        type=int,
        default=401,
        help="Smoothing window for stage1.",
    )
    parser.add_argument(
        "--rolling_stage2",
        type=int,
        default=51,
        help="Smoothing window for stage2.",
    )
    args = parser.parse_args()

    stage1_csv = Path(args.stage1_csv)
    stage2_csv = Path(args.stage2_csv)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    metrics = list(_iter_metrics(args.metric, args.also, args.all_default_metrics))
    for metric in metrics:
        out_path = out_dir / f"ea_training_dynamics_{metric}.pdf"
        _plot_two_stage(
            stage1_csv=stage1_csv,
            stage2_csv=stage2_csv,
            metric=metric,
            out_path=out_path,
            rolling_stage1=int(args.rolling_stage1),
            rolling_stage2=int(args.rolling_stage2),
        )
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

