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


CORE_METRICS: Tuple[str, ...] = (
    "accuracy",
    "ce",
    "bce",
    "pdlc_tau",
    "pdlc_pos_frac",
    "pdlc_pos_weight",
    "pdlc_max_prob",
    "pdlc_prob_gap",
    "pdlc_prior_entropy",
    "pdlc_pair_logit_mean",
    "pdlc_pair_logit_std",
    "pdlc_pair_logit_pos_mean",
    "pdlc_pair_logit_neg_mean",
    "pdlc_logit_gap",
    "pdlc_gamma_pos_mean",
    "pdlc_gamma_neg_mean",
    "pdlc_gamma_gap",
    "pdlc_gamma_saturated_low",
    "pdlc_gamma_saturated_high",
    "pdlc_neff_mean",
    "pdlc_neff_median",
    "prior_time",
    "train_time",
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

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(title)

    # 1) accuracy
    if "accuracy" in df.columns:
        y = df["accuracy"].to_numpy(dtype=float)
        axes[0].plot(step, _rolling_mean(y, rolling_window), label="accuracy")
        axes[0].set_ylabel("accuracy")
        axes[0].grid(True, alpha=0.2)
        axes[0].legend(loc="best")

    # 2) losses
    for col, lab in [("ce", "CE"), ("bce", "BCE")]:
        if col in df.columns:
            y = df[col].to_numpy(dtype=float)
            axes[1].plot(step, _rolling_mean(y, rolling_window), label=lab)
    axes[1].set_ylabel("loss")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(loc="best")

    # 3) pair logits (pos/neg) + gap
    if "pdlc_pair_logit_pos_mean" in df.columns and "pdlc_pair_logit_neg_mean" in df.columns:
        y_pos = df["pdlc_pair_logit_pos_mean"].to_numpy(dtype=float)
        y_neg = df["pdlc_pair_logit_neg_mean"].to_numpy(dtype=float)
        axes[2].plot(step, _rolling_mean(y_pos, rolling_window), label="logit pos mean")
        axes[2].plot(step, _rolling_mean(y_neg, rolling_window), label="logit neg mean")
    if "pdlc_logit_gap" in df.columns:
        y_gap = df["pdlc_logit_gap"].to_numpy(dtype=float)
        axes[2].plot(step, _rolling_mean(y_gap, rolling_window), label="logit gap", linestyle="--")
    axes[2].set_ylabel("pair logits")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(loc="best")

    # 4) gamma (pos/neg) + saturation
    if "pdlc_gamma_pos_mean" in df.columns and "pdlc_gamma_neg_mean" in df.columns:
        y_pos = df["pdlc_gamma_pos_mean"].to_numpy(dtype=float)
        y_neg = df["pdlc_gamma_neg_mean"].to_numpy(dtype=float)
        axes[3].plot(step, _rolling_mean(y_pos, rolling_window), label="gamma pos mean")
        axes[3].plot(step, _rolling_mean(y_neg, rolling_window), label="gamma neg mean")
    if "pdlc_gamma_gap" in df.columns:
        y_gap = df["pdlc_gamma_gap"].to_numpy(dtype=float)
        axes[3].plot(step, _rolling_mean(y_gap, rolling_window), label="gamma gap", linestyle="--")
    if "pdlc_gamma_saturated_low" in df.columns:
        y = df["pdlc_gamma_saturated_low"].to_numpy(dtype=float)
        axes[3].plot(step, _rolling_mean(y, rolling_window), label="sat_low", alpha=0.7)
    if "pdlc_gamma_saturated_high" in df.columns:
        y = df["pdlc_gamma_saturated_high"].to_numpy(dtype=float)
        axes[3].plot(step, _rolling_mean(y, rolling_window), label="sat_high", alpha=0.7)
    axes[3].set_ylabel("gamma / saturation")
    axes[3].set_xlabel("step")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].grid(True, alpha=0.2)
    axes[3].legend(loc="best", ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
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

    lines = [f"== {label} =="]
    lines.append(f"rows={summ['n_rows']} steps=[{summ['step_min']}, {summ['step_max']}]")

    acc_first, acc_last, acc_max, acc_max_step = g("accuracy", "first"), g("accuracy", "last"), g("accuracy", "max"), gi("accuracy", "max_step")
    if acc_first is not None:
        lines.append(f"accuracy: first={acc_first:.4f} last={acc_last:.4f} best={acc_max:.4f} @step={acc_max_step}")

    ce_first, ce_last, ce_min, ce_min_step = g("ce", "first"), g("ce", "last"), g("ce", "min"), gi("ce", "min_step")
    if ce_first is not None:
        lines.append(f"CE: first={ce_first:.4f} last={ce_last:.4f} min={ce_min:.4f} @step={ce_min_step}")

    bce_first, bce_last, bce_min, bce_min_step = g("bce", "first"), g("bce", "last"), g("bce", "min"), gi("bce", "min_step")
    if bce_first is not None:
        lines.append(f"BCE: first={bce_first:.4f} last={bce_last:.4f} min={bce_min:.4f} @step={bce_min_step}")

    maxp_first, maxp_last = g("pdlc_max_prob", "first"), g("pdlc_max_prob", "last")
    gap_first, gap_last = g("pdlc_prob_gap", "first"), g("pdlc_prob_gap", "last")
    if maxp_first is not None and gap_first is not None:
        lines.append(f"posterior sharpness: max_prob first={maxp_first:.3f} last={maxp_last:.3f} | prob_gap first={gap_first:.3f} last={gap_last:.3f}")

    tau_first, tau_last = g("pdlc_tau", "first"), g("pdlc_tau", "last")
    if tau_first is not None:
        lines.append(f"tau: first={tau_first:.4f} last={tau_last:.4f}")

    gap_first, gap_last, gap_max, gap_max_step = g("pdlc_logit_gap", "first"), g("pdlc_logit_gap", "last"), g("pdlc_logit_gap", "max"), gi("pdlc_logit_gap", "max_step")
    if gap_first is not None:
        lines.append(f"logit gap (pos-neg): first={gap_first:.4f} last={gap_last:.4f} best={gap_max:.4f} @step={gap_max_step}")

    ggap_first, ggap_last, ggap_max, ggap_max_step = g("pdlc_gamma_gap", "first"), g("pdlc_gamma_gap", "last"), g("pdlc_gamma_gap", "max"), gi("pdlc_gamma_gap", "max_step")
    if ggap_first is not None:
        lines.append(f"gamma gap (pos-neg): first={ggap_first:.4f} last={ggap_last:.4f} best={ggap_max:.4f} @step={ggap_max_step}")

    sat_low_first, sat_low_last = g("pdlc_gamma_saturated_low", "first"), g("pdlc_gamma_saturated_low", "last")
    sat_high_first, sat_high_last = g("pdlc_gamma_saturated_high", "first"), g("pdlc_gamma_saturated_high", "last")
    if sat_low_first is not None and sat_high_first is not None:
        lines.append(f"gamma saturation: sat_low first={sat_low_first:.3f} last={sat_low_last:.3f} | sat_high first={sat_high_first:.3f} last={sat_high_last:.3f}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PDL training metric CSVs.")
    parser.add_argument(
        "--csvs",
        nargs="*",
        default=[
            "training_metrics/mini_tabicl_stage1_pdl_posterior_avg.csv",
            "training_metrics/mini_tabicl_stage2_pdl_posterior_avg.csv",
        ],
        help="CSV files to analyze (default: the 2 training_metrics/*.csv files).",
    )
    parser.add_argument(
        "--out_dir",
        default="results/training_metrics_analysis",
        help="Output directory for summaries/plots.",
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
    for csv_path in args.csvs:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        df = pd.read_csv(p)
        summ = summarize_run(df, CORE_METRICS)
        summ["csv"] = str(p)
        all_summaries.append(summ)

        run_name = p.stem
        with (out_dir / f"{run_name}_summary.json").open("w") as f:
            json.dump(summ, f, indent=2, sort_keys=True)

        # Plot overview
        rw = int(args.rolling_window)
        n = int(summ["n_rows"])
        if n < rw:
            rw = max(3, n // 10)
        plot_overview(
            df=df,
            title=run_name,
            out_path=out_dir / f"{run_name}_overview.png",
            rolling_window=rw,
        )

    with (out_dir / "summary_all.json").open("w") as f:
        json.dump({"runs": all_summaries}, f, indent=2, sort_keys=True)

    # Console report
    print(f"Wrote analysis to: {out_dir}")
    for summ in all_summaries:
        label = Path(str(summ['csv'])).name  # type: ignore[index]
        print()
        print(format_key_findings(label, summ))


if __name__ == "__main__":
    main()
