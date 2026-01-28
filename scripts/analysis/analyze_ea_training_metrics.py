#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_METRICS: Tuple[str, ...] = (
    "accuracy",
    "ce",
    "lr",
    "prior_time",
    "train_time",
    # optional probes (present only when probe steps occur)
    "probe/attn_kl_mean",
    "probe/attn_entropy_mean",
    "probe/delta_ce",
    "probe/ce",
    "probe/chance_entropy",
)


EA_METRICS: Tuple[str, ...] = (
    # TFrow
    "row_L2_context_div",
    "row_L2_sparsity",
    "row_L2_scale_mean",
    "row_L2_feat_velocity",
    "row_L2_q_norm",
    "row_L2_logit_mean",
    "row_L12_context_div",
    "row_L12_sparsity",
    "row_L12_scale_mean",
    "row_L12_feat_velocity",
    "row_L12_q_norm",
    "row_L12_logit_mean",
    # TFicl
    "icl_L2_context_div",
    "icl_L2_sparsity",
    "icl_L2_scale_mean",
    "icl_L2_feat_velocity",
    "icl_L2_q_norm",
    "icl_L2_logit_mean",
    "icl_L12_context_div",
    "icl_L12_sparsity",
    "icl_L12_scale_mean",
    "icl_L12_feat_velocity",
    "icl_L12_q_norm",
    "icl_L12_logit_mean",
)


@dataclass(frozen=True)
class MetricSummary:
    n: int
    frac_valid: float
    first: Optional[float]
    last: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    min_step: Optional[int]
    max: Optional[float]
    max_step: Optional[int]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == "step":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["step"] = pd.to_numeric(out["step"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["step"]).copy()
    out["step"] = out["step"].astype(int)
    out = out.sort_values("step").reset_index(drop=True)
    return out


def _split_by_step_resets(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Split a raw metrics dataframe into contiguous segments when `step` resets.

    This handles the common case where multiple training runs were appended to the
    same CSV (step goes 1..T, then 1..T again).
    """
    if "step" not in df.columns:
        return [df]

    step = pd.to_numeric(df["step"], errors="coerce")
    # Keep original row order; split whenever step decreases.
    cut_points: List[int] = []
    prev: Optional[float] = None
    for i, v in enumerate(step.tolist()):
        if prev is not None and pd.notna(v) and pd.notna(prev) and float(v) < float(prev):
            cut_points.append(i)
        prev = v

    if not cut_points:
        return [df]

    segments: List[pd.DataFrame] = []
    start = 0
    for cut in cut_points:
        segments.append(df.iloc[start:cut].reset_index(drop=True))
        start = cut
    segments.append(df.iloc[start:].reset_index(drop=True))
    return segments


def _series_summary(step: pd.Series, s: pd.Series) -> MetricSummary:
    s_num = pd.to_numeric(s, errors="coerce")
    valid = s_num.dropna()
    n = int(s_num.shape[0])
    if n == 0:
        return MetricSummary(
            n=0,
            frac_valid=0.0,
            first=None,
            last=None,
            mean=None,
            std=None,
            min=None,
            min_step=None,
            max=None,
            max_step=None,
        )

    frac_valid = float(valid.shape[0]) / float(n)
    if valid.shape[0] == 0:
        return MetricSummary(
            n=n,
            frac_valid=0.0,
            first=None,
            last=None,
            mean=None,
            std=None,
            min=None,
            min_step=None,
            max=None,
            max_step=None,
        )

    # first/last valid
    first_idx = int(valid.index[0])
    last_idx = int(valid.index[-1])
    first = float(s_num.iloc[first_idx])
    last = float(s_num.iloc[last_idx])

    mean = float(valid.mean())
    std = float(valid.std(ddof=0))

    min_val = float(valid.min())
    max_val = float(valid.max())
    min_step = int(step.loc[valid.idxmin()])
    max_step = int(step.loc[valid.idxmax()])

    return MetricSummary(
        n=n,
        frac_valid=frac_valid,
        first=first,
        last=last,
        mean=mean,
        std=std,
        min=min_val,
        min_step=min_step,
        max=max_val,
        max_step=max_step,
    )


def summarize_run(df: pd.DataFrame, metrics: Iterable[str]) -> Dict[str, object]:
    df = _coerce_numeric(df)
    out: Dict[str, object] = {
        "n_rows": int(df.shape[0]),
        "step_min": int(df["step"].min()) if df.shape[0] else None,
        "step_max": int(df["step"].max()) if df.shape[0] else None,
        "columns": list(df.columns),
        "metric_summaries": {},
    }
    for m in metrics:
        if m not in df.columns:
            continue
        summ = _series_summary(df["step"], df[m])
        out["metric_summaries"][m] = summ.__dict__
    return out


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    s = pd.Series(y)
    return s.rolling(window=window, min_periods=max(1, window // 5), center=True).mean().to_numpy()


def _plot_series(
    ax: plt.Axes,
    step: np.ndarray,
    df: pd.DataFrame,
    keys: Iterable[str],
    rolling_window: int,
    ylabel: str,
    title: Optional[str] = None,
) -> None:
    any_plotted = False
    for k in keys:
        if k not in df.columns:
            continue
        y = df[k].to_numpy(dtype=float)
        if np.all(~np.isfinite(y)):
            continue
        ax.plot(step, _rolling_mean(y, rolling_window), label=k)
        any_plotted = True
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.2)
    if any_plotted:
        ax.legend(loc="best", ncol=2, fontsize=8)


def plot_overview(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    rolling_window: int,
) -> None:
    df = _coerce_numeric(df)
    step = df["step"].to_numpy()
    if step.size == 0:
        return

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(title)

    _plot_series(
        axes[0],
        step,
        df,
        keys=["accuracy"],
        rolling_window=rolling_window,
        ylabel="accuracy",
    )

    _plot_series(
        axes[1],
        step,
        df,
        keys=["ce"],
        rolling_window=rolling_window,
        ylabel="CE",
    )

    _plot_series(
        axes[2],
        step,
        df,
        keys=["icl_L2_scale_mean", "icl_L12_scale_mean", "row_L2_scale_mean", "row_L12_scale_mean"],
        rolling_window=rolling_window,
        ylabel="m mean",
        title="EA: ellipse mean scale (m)",
    )

    _plot_series(
        axes[3],
        step,
        df,
        keys=["icl_L2_sparsity", "icl_L12_sparsity", "row_L2_sparsity", "row_L12_sparsity"],
        rolling_window=rolling_window,
        ylabel="std(m) over dims",
        title="EA: anisotropy / sparsity (std over head-dim)",
    )

    axes[-1].set_xlabel("step")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_diagnostics(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    rolling_window: int,
) -> None:
    df = _coerce_numeric(df)
    step = df["step"].to_numpy()
    if step.size == 0:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(title)

    _plot_series(
        axes[0, 0],
        step,
        df,
        keys=["icl_L2_feat_velocity", "icl_L12_feat_velocity", "row_L2_feat_velocity", "row_L12_feat_velocity"],
        rolling_window=rolling_window,
        ylabel="|v - v_prev|",
        title="EA: feature velocity (stability proxy)",
    )
    _plot_series(
        axes[0, 1],
        step,
        df,
        keys=["icl_L2_q_norm", "icl_L12_q_norm", "row_L2_q_norm", "row_L12_q_norm"],
        rolling_window=rolling_window,
        ylabel="||q||",
        title="EA: mean query norm",
    )
    _plot_series(
        axes[1, 0],
        step,
        df,
        keys=["icl_L2_logit_mean", "icl_L12_logit_mean", "row_L2_logit_mean", "row_L12_logit_mean"],
        rolling_window=rolling_window,
        ylabel="E[||q||·||k||/sqrt(d)]",
        title="EA: approximate attention-logit scale",
    )
    _plot_series(
        axes[1, 1],
        step,
        df,
        keys=["icl_L2_context_div", "icl_L12_context_div", "row_L2_context_div", "row_L12_context_div"],
        rolling_window=rolling_window,
        ylabel="std(m) across samples",
        title="EA: context diversity (may be NaN for batch=1)",
    )
    _plot_series(
        axes[2, 0],
        step,
        df,
        keys=["lr"],
        rolling_window=rolling_window,
        ylabel="lr",
        title="learning rate",
    )
    _plot_series(
        axes[2, 1],
        step,
        df,
        keys=["train_time", "prior_time"],
        rolling_window=rolling_window,
        ylabel="sec",
        title="timings",
    )

    for ax in axes[-1, :]:
        ax.set_xlabel("step")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def format_key_findings(label: str, summ: Dict[str, object]) -> str:
    ms: Dict[str, Dict[str, object]] = summ["metric_summaries"]  # type: ignore[assignment]

    def g(metric: str, key: str) -> Optional[float]:
        if metric not in ms:
            return None
        v = ms[metric].get(key)
        return None if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else float(v)

    def gi(metric: str, key: str) -> Optional[int]:
        if metric not in ms:
            return None
        v = ms[metric].get(key)
        return None if v is None else int(v)

    def fmt_triplet(metric: str) -> Optional[str]:
        first, last, mean = g(metric, "first"), g(metric, "last"), g(metric, "mean")
        if first is None:
            return None
        return f"{metric}: first={first:.4f} last={last:.4f} mean={mean:.4f}"

    part = summ.get("part")
    header = label if part is None else f"{label} (part={int(part)})"
    lines = [f"== {header} =="]
    lines.append(f"rows={summ['n_rows']} steps=[{summ['step_min']}, {summ['step_max']}]")

    acc_first, acc_last, acc_max, acc_max_step = g("accuracy", "first"), g("accuracy", "last"), g("accuracy", "max"), gi("accuracy", "max_step")
    if acc_first is not None:
        lines.append(f"accuracy: first={acc_first:.4f} last={acc_last:.4f} best={acc_max:.4f} @step={acc_max_step}")

    ce_first, ce_last, ce_min, ce_min_step = g("ce", "first"), g("ce", "last"), g("ce", "min"), gi("ce", "min_step")
    if ce_first is not None:
        lines.append(f"CE: first={ce_first:.4f} last={ce_last:.4f} min={ce_min:.4f} @step={ce_min_step}")

    # EA highlights: scale_mean + sparsity + logit scale (L2 and L12, icl/row)
    for prefix in ("icl", "row"):
        for layer in ("L2", "L12"):
            for metric in ("scale_mean", "sparsity", "feat_velocity", "logit_mean"):
                key = f"{prefix}_{layer}_{metric}"
                msg = fmt_triplet(key)
                if msg is not None:
                    lines.append(msg)

    lr_last = g("lr", "last")
    if lr_last is not None:
        lines.append(f"lr last={lr_last:.6g}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Elliptical Attention (EA) training-metric CSVs.\n\n"
            "EA metrics are logged by Trainer._collect_ea_metrics from:\n"
            "- src/tabicl/model/attention.py (stores last_m, last_val_diff, last_q_norm, last_logit_mean)\n"
            "- src/tabicl/train/run.py (derives context_div/sparsity/scale_mean per layer)\n"
        )
    )
    parser.add_argument(
        "--csvs",
        nargs="*",
        default=[
            "training_metrics/mini_tabicl_stage1_ea_icl_only.csv",
            "training_metrics/mini_tabicl_stage1_ea_row_icl.csv",
            "training_metrics/mini_tabicl_stage2_ea_icl_only.csv",
            "training_metrics/mini_tabicl_stage2_ea_row_icl.csv",
        ],
        help="CSV files to analyze.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/training_metrics_analysis",
        help="Output directory for summaries/plots.",
    )
    parser.add_argument(
        "--segment",
        choices=["last", "all", "first", "index"],
        default="last",
        help=(
            "If a CSV has multiple appended runs (step resets), which segment(s) to analyze. "
            "'last' is typically what you want."
        ),
    )
    parser.add_argument(
        "--segment_index",
        type=int,
        default=0,
        help="Used only when --segment=index (0-based).",
    )
    parser.add_argument(
        "--rolling_window",
        type=int,
        default=101,
        help="Rolling window for smoothing plots (per file; will be clipped for short runs).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    all_summaries: List[Dict[str, object]] = []
    metrics = list(BASE_METRICS) + list(EA_METRICS)

    for csv_path in args.csvs:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        df_full = pd.read_csv(p)
        parts = _split_by_step_resets(df_full)
        if len(parts) > 1:
            print(f"[note] Detected {len(parts)} appended runs in {p} (step reset).")

        selected_parts = parts
        selected_part_indices = list(range(len(parts)))
        if len(parts) > 1 and args.segment != "all":
            if args.segment == "last":
                keep_idx = len(parts) - 1
            elif args.segment == "first":
                keep_idx = 0
            else:
                keep_idx = int(args.segment_index)
            if keep_idx < 0 or keep_idx >= len(parts):
                raise ValueError(f"--segment_index {keep_idx} out of range for {p} (num_parts={len(parts)})")
            selected_parts = [parts[keep_idx]]
            selected_part_indices = [keep_idx]

        for local_i, df in enumerate(selected_parts):
            orig_part_idx = selected_part_indices[local_i]
            run_name = p.stem if len(selected_parts) == 1 else f"{p.stem}_part{orig_part_idx}"
            summ = summarize_run(df, metrics)
            summ["csv"] = str(p)
            if len(parts) > 1:
                summ["part"] = int(orig_part_idx)
                summ["num_parts"] = int(len(parts))
            all_summaries.append(summ)

            with (out_dir / f"{run_name}_summary.json").open("w") as f:
                json.dump(summ, f, indent=2, sort_keys=True)

            # Plots
            rw = int(args.rolling_window)
            n = int(summ["n_rows"])
            if n < rw:
                rw = max(3, n // 10)
            plot_overview(
                df=df,
                title=run_name,
                out_path=out_dir / f"{run_name}_ea_overview.png",
                rolling_window=rw,
            )
            plot_diagnostics(
                df=df,
                title=run_name,
                out_path=out_dir / f"{run_name}_ea_diagnostics.png",
                rolling_window=rw,
            )

    out_summary = out_dir / "summary_all_ea.json"
    with out_summary.open("w") as f:
        json.dump({"runs": all_summaries}, f, indent=2, sort_keys=True)

    print(f"Wrote EA analysis to: {out_dir}")
    print(f"Wrote EA summary: {out_summary}")
    for summ in all_summaries:
        label = Path(str(summ["csv"])).name  # type: ignore[index]
        print()
        print(format_key_findings(label, summ))


if __name__ == "__main__":
    main()
