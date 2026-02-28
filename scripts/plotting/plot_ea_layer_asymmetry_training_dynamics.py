#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

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


def _select_segment(df: pd.DataFrame, which: str) -> pd.DataFrame:
    which = which.strip().lower()
    if which not in {"last", "first", "longest"}:
        raise ValueError(f"Invalid segment selector: {which!r} (expected: last|first|longest)")
    if df.empty:
        return df

    step = df["step"].to_numpy(dtype=float)
    seg = np.zeros(len(df), dtype=int)
    cur = 0
    for i in range(1, len(df)):
        if step[i] < step[i - 1]:
            cur += 1
        seg[i] = cur
    df = df.assign(_seg=seg)
    seg_ids = sorted(df["_seg"].unique().tolist())
    if len(seg_ids) <= 1:
        return df.drop(columns=["_seg"])

    if which == "last":
        chosen = seg_ids[-1]
    elif which == "first":
        chosen = seg_ids[0]
    else:
        sizes = df.groupby("_seg", as_index=True).size()
        chosen = int(sizes.idxmax())
    return df[df["_seg"] == chosen].drop(columns=["_seg"]).reset_index(drop=True)


def _load_metrics(csv_path: Path, metrics: Tuple[str, ...], segment: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    usecols = ["step", *metrics]
    df = pd.read_csv(csv_path, usecols=usecols)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    df = df.dropna(subset=["step"]).reset_index(drop=True)
    df = _select_segment(df, segment)
    df = df.sort_values("step").reset_index(drop=True)

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    x = df["step"].to_numpy(dtype=float)
    for m in metrics:
        y = df[m].to_numpy(dtype=float)
        out[m] = (x, y)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot layer-asymmetry EA training dynamics: Layer-2 vs Layer-12 curves for sparsity and scale "
            "in TF_icl, shown for Stage 1 and Stage 2."
        )
    )
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
        "--stage1_segment",
        default="last",
        choices=["last", "first", "longest"],
        help="Which segment to use for stage1 if runs are appended (default: last).",
    )
    parser.add_argument(
        "--stage2_segment",
        default="last",
        choices=["last", "first", "longest"],
        help="Which segment to use for stage2 if runs are appended (default: last).",
    )
    parser.add_argument("--rolling_stage1", type=int, default=401, help="Smoothing window for stage1.")
    parser.add_argument("--rolling_stage2", type=int, default=51, help="Smoothing window for stage2.")
    parser.add_argument(
        "--label",
        default=r"EA-ICL ($\mathrm{TF}_{\mathrm{icl}}$)",
        help="Optional figure title.",
    )
    parser.add_argument(
        "--out",
        default="figures/ea_training_dynamics_layer_asymmetry.pdf",
        help="Output PDF path.",
    )
    args = parser.parse_args()

    s1_path = Path(args.stage1_csv)
    s2_path = Path(args.stage2_csv)
    out_path = Path(args.out)
    _ensure_dir(out_path.parent)

    metrics = ("icl_L2_sparsity", "icl_L12_sparsity", "icl_L2_scale_mean", "icl_L12_scale_mean")
    s1 = _load_metrics(s1_path, metrics, segment=str(args.stage1_segment))
    s2 = _load_metrics(s2_path, metrics, segment=str(args.stage2_segment))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.5), constrained_layout=True)
    if args.label:
        fig.suptitle(str(args.label), fontsize=11)

    # Stage 1 (left column)
    x1, y1 = s1["icl_L2_sparsity"]
    _, y2 = s1["icl_L12_sparsity"]
    ax = axes[0, 0]
    ax.plot(x1, _rolling_mean(y1, int(args.rolling_stage1)), label="Layer 2", linewidth=1.5)
    ax.plot(x1, _rolling_mean(y2, int(args.rolling_stage1)), label="Layer 12", linewidth=1.5)
    ax.set_title("Stage 1: sparsity")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Metric sparsity")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    x1, y1 = s1["icl_L2_scale_mean"]
    _, y2 = s1["icl_L12_scale_mean"]
    ax = axes[1, 0]
    ax.plot(x1, _rolling_mean(y1, int(args.rolling_stage1)), label="Layer 2", linewidth=1.5)
    ax.plot(x1, _rolling_mean(y2, int(args.rolling_stage1)), label="Layer 12", linewidth=1.5)
    ax.set_title("Stage 1: scale_mean")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Mean metric scale")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    # Stage 2 (right column)
    x2s, y1 = s2["icl_L2_sparsity"]
    _, y2 = s2["icl_L12_sparsity"]
    ax = axes[0, 1]
    ax.plot(x2s, _rolling_mean(y1, int(args.rolling_stage2)), label="Layer 2", linewidth=1.5)
    ax.plot(x2s, _rolling_mean(y2, int(args.rolling_stage2)), label="Layer 12", linewidth=1.5)
    ax.set_title("Stage 2: sparsity")
    ax.set_xlabel("Steps")
    ax.grid(True, alpha=0.3)

    x2s, y1 = s2["icl_L2_scale_mean"]
    _, y2 = s2["icl_L12_scale_mean"]
    ax = axes[1, 1]
    ax.plot(x2s, _rolling_mean(y1, int(args.rolling_stage2)), label="Layer 2", linewidth=1.5)
    ax.plot(x2s, _rolling_mean(y2, int(args.rolling_stage2)), label="Layer 12", linewidth=1.5)
    ax.set_title("Stage 2: scale_mean")
    ax.set_xlabel("Steps")
    ax.grid(True, alpha=0.3)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
