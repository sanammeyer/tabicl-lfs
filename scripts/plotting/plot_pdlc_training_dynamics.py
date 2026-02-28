#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == "step":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out = out.dropna(subset=["step"]).copy()
    out["step"] = out["step"].astype(int)
    return out.sort_values("step").reset_index(drop=True)


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    return (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=max(1, window // 5))
        .mean()
        .to_numpy()
    )


def _first_step_where(step: np.ndarray, y: np.ndarray, pred) -> Optional[int]:
    idx = np.where(pred(y))[0]
    if idx.size == 0:
        return None
    return int(step[int(idx[0])])


def _percentile(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _find_peak(step: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
    y = np.asarray(y, dtype=float)
    if y.size == 0 or not np.isfinite(y).any():
        return None, None
    i = int(np.nanargmax(y))
    return int(step[i]), float(y[i])


def _tau_convergence_step(
    step: np.ndarray,
    tau: np.ndarray,
    *,
    rolling_window: int,
    abs_tol: float = 0.02,
    sustain_steps: int = 500,
) -> Optional[int]:
    """Heuristic: first step where tau stays within abs_tol of its final plateau."""
    step = np.asarray(step, dtype=int)
    tau = np.asarray(tau, dtype=float)
    if step.size == 0 or tau.size != step.size:
        return None

    tau_s = _rolling_mean(tau, rolling_window)
    ok = np.isfinite(tau_s)
    if ok.sum() < 50:
        return None

    # "Final" tau = robust central tendency over the tail of training.
    tail_start = int(0.95 * step.size)
    tail = tau_s[tail_start:]
    tail = tail[np.isfinite(tail)]
    if tail.size == 0:
        return None
    tau_final = float(np.median(tail))

    within = np.isfinite(tau_s) & (np.abs(tau_s - tau_final) <= abs_tol)
    if sustain_steps <= 1:
        idx = np.where(within)[0]
        return int(step[int(idx[0])]) if idx.size else None

    for i in range(0, within.size - sustain_steps + 1):
        if within[i] and within[i : i + sustain_steps].all():
            return int(step[i])
    return None


def _load_metrics_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return _coerce_numeric(pd.read_csv(path))


def _infer_run_label(stage1_csv: Path, stage2_csv: Optional[Path]) -> str:
    if stage2_csv is None:
        return stage1_csv.stem

    def canonical(stem: str) -> str:
        # Prefer a stable label that removes the stage marker.
        stem = stem.replace("_stage1_", "_").replace("_stage2_", "_")
        stem = re.sub(r"(^|_)stage[12](_|$)", "_", stem)
        stem = re.sub(r"__+", "_", stem).strip("_")
        return stem

    a, b = canonical(stage1_csv.stem), canonical(stage2_csv.stem)
    if a == b:
        return a

    # Fallback: longest common prefix, trimmed to a sensible token boundary.
    i = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        i += 1
    prefix = a[:i].rstrip("_-")
    return prefix if len(prefix) >= 8 else stage1_csv.stem


def _concat_stages(stage1: pd.DataFrame, stage2: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    s1_max = int(stage1["step"].max()) if len(stage1) else 0
    out1 = stage1.copy().rename(columns={"step": "step_stage"})
    out1["stage"] = 1
    out1["step_full"] = out1["step_stage"]

    out2 = stage2.copy().rename(columns={"step": "step_stage"})
    out2["stage"] = 2
    out2["step_full"] = out2["step_stage"] + s1_max

    merged = pd.concat([out1, out2], axis=0, ignore_index=True, sort=False)
    merged = merged.sort_values(["step_full", "stage"]).reset_index(drop=True)
    return merged, s1_max


def plot_timeseries_two_stage(
    stage1: pd.DataFrame,
    stage2: pd.DataFrame,
    out_path: Path,
    rolling_stage1: int,
    rolling_stage2: int,
    title: str,
) -> None:
    # Thesis-first layout: Stage 1 and Stage 2 side-by-side for readability.
    s1_step = stage1["step"].to_numpy(dtype=int)
    s2_step = stage2["step"].to_numpy(dtype=int)
    tau_conv_step = None
    if "pdlc_tau" in stage1.columns:
        tau_conv_step = _tau_convergence_step(
            s1_step,
            stage1["pdlc_tau"].to_numpy(dtype=float),
            rolling_window=rolling_stage1,
            abs_tol=0.02,
            sustain_steps=500,
        )

    # Full-run figure: focus on metric learning + temperature/saturation (omit Neff panel).
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 5.6), sharey="row", constrained_layout=True)
    fig.suptitle(title)

    # Column titles
    axes[0, 0].set_title("Stage 1: Adaptation")
    axes[0, 1].set_title("Stage 2: Stability (Long Context)")

    def plot_on(ax: plt.Axes, step: np.ndarray, y: np.ndarray, window: int, **kwargs) -> None:
        ax.plot(step, _rolling_mean(y, window), **kwargs)

    # Row 1: pair logits + gap
    for ax, step, df, rw in [
        (axes[0, 0], s1_step, stage1, rolling_stage1),
        (axes[0, 1], s2_step, stage2, rolling_stage2),
    ]:
        if "pdlc_pair_logit_pos_mean" in df.columns:
            plot_on(
                ax,
                step,
                df["pdlc_pair_logit_pos_mean"].to_numpy(dtype=float),
                rw,
                label=r"$s_{\mathrm{pos}}$",
                color="#1f77b4",
                linewidth=1.6,
            )
        if "pdlc_pair_logit_neg_mean" in df.columns:
            plot_on(
                ax,
                step,
                df["pdlc_pair_logit_neg_mean"].to_numpy(dtype=float),
                rw,
                label=r"$s_{\mathrm{neg}}$",
                color="#d62728",
                linewidth=1.6,
            )
        if "pdlc_logit_gap" in df.columns:
            plot_on(
                ax,
                step,
                df["pdlc_logit_gap"].to_numpy(dtype=float),
                rw,
                label=r"$s_{\mathrm{pos}}-s_{\mathrm{neg}}$",
                color="black",
                linestyle="--",
                linewidth=1.8,
            )
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("Pair logits")
    axes[0, 0].legend(loc="best", ncol=2, frameon=True)
    axes[0, 1].legend(loc="best", ncol=2, frameon=True)

    # Row 2: tau + total saturation
    for ax, step, df, rw in [
        (axes[1, 0], s1_step, stage1, rolling_stage1),
        (axes[1, 1], s2_step, stage2, rolling_stage2),
    ]:
        ax2 = ax.twinx()
        if "pdlc_tau" in df.columns:
            tau_arr = df["pdlc_tau"].to_numpy(dtype=float)
            plot_on(
                ax,
                step,
                tau_arr,
                rw,
                label=r"Temperature $\tau$",
                color="#2ca02c",
                linewidth=1.8,
            )
        if df is stage1 and tau_conv_step is not None:
            ax.axvline(tau_conv_step, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
            tau_s = _rolling_mean(tau_arr, rw) if "pdlc_tau" in df.columns else None
            if tau_s is not None and np.isfinite(tau_s).any():
                idx = int(np.argmin(np.abs(step - tau_conv_step)))
                tau_at = float(tau_s[idx]) if np.isfinite(tau_s[idx]) else float(np.nanmedian(tau_s[np.isfinite(tau_s)]))
            else:
                tau_at = float(np.nan)
            ax.annotate(
                r"$\tau$ converges",
                xy=(tau_conv_step, tau_at),
                xytext=(10, 12),
                textcoords="offset points",
                fontsize=9,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.8),
                arrowprops=dict(arrowstyle="-", lw=0.9, color="black", alpha=0.5, shrinkA=0, shrinkB=0),
            )
        if {"pdlc_gamma_saturated_low", "pdlc_gamma_saturated_high"}.issubset(set(df.columns)):
            tot = (
                df["pdlc_gamma_saturated_low"].to_numpy(dtype=float)
                + df["pdlc_gamma_saturated_high"].to_numpy(dtype=float)
            ) * 100.0
            plot_on(ax2, step, tot, rw, label="Total saturation", color="#7f7f7f", linewidth=1.4)
            ax2.axhline(5.0, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
            ax2.set_ylim(-1.0, 100.0)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(r"$\tau$")
        ax2.set_ylabel("Saturated pairs (%)")
        ax.legend(loc="upper left", frameon=True)
        ax2.legend(loc="upper right", frameon=True)

    axes[1, 0].set_xlabel("Training step")
    axes[1, 1].set_xlabel("Training step")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(df: pd.DataFrame, out_path: Path, rolling: int, title: str) -> None:
    step = df["step"].to_numpy(dtype=int)

    # Stage-1 figure: keep it focused on metric learning + temperature/saturation.
    # (Neff can be shown elsewhere if needed.)
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True, constrained_layout=True)
    fig.suptitle(title)

    # Panel 1: Pair logits and margin
    ax = axes[0]
    if "pdlc_pair_logit_pos_mean" in df.columns:
        y = df["pdlc_pair_logit_pos_mean"].to_numpy(dtype=float)
        ax.plot(step, _rolling_mean(y, rolling), label=r"$s_{\mathrm{pos}}$", color="#1f77b4", linewidth=1.6)
    if "pdlc_pair_logit_neg_mean" in df.columns:
        y = df["pdlc_pair_logit_neg_mean"].to_numpy(dtype=float)
        ax.plot(step, _rolling_mean(y, rolling), label=r"$s_{\mathrm{neg}}$", color="#d62728", linewidth=1.6)
    if "pdlc_logit_gap" in df.columns:
        y = df["pdlc_logit_gap"].to_numpy(dtype=float)
        ax.plot(step, _rolling_mean(y, rolling), label=r"$s_{\mathrm{pos}}-s_{\mathrm{neg}}$", color="black", linestyle="--", linewidth=1.8)
    ax.set_ylabel("Pair logits")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2, frameon=True)

    # Panel 2: Temperature and saturation
    ax = axes[1]
    ax2 = ax.twinx()
    if "pdlc_tau" in df.columns:
        y = df["pdlc_tau"].to_numpy(dtype=float)
        y_s = _rolling_mean(y, rolling)
        ax.plot(step, y_s, label=r"Temperature $\tau$", color="#2ca02c", linewidth=1.8)
        tau_conv_step = _tau_convergence_step(step, y, rolling_window=rolling, abs_tol=0.02, sustain_steps=500)
        if tau_conv_step is not None:
            ax.axvline(tau_conv_step, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
            idx = int(np.argmin(np.abs(step - tau_conv_step)))
            tau_at = float(y_s[idx]) if np.isfinite(y_s[idx]) else float(np.nanmedian(y_s[np.isfinite(y_s)]))
            ax.annotate(
                r"$\tau$ converges",
                xy=(tau_conv_step, tau_at),
                xytext=(10, 12),
                textcoords="offset points",
                fontsize=9,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.8),
                arrowprops=dict(arrowstyle="-", lw=0.9, color="black", alpha=0.5, shrinkA=0, shrinkB=0),
            )
    sat_plotted = False
    if "pdlc_gamma_saturated_low" in df.columns:
        y = df["pdlc_gamma_saturated_low"].to_numpy(dtype=float) * 100.0
        ax2.plot(step, _rolling_mean(y, rolling), label=r"$\gamma<0.01$", color="#9467bd", linewidth=1.4, alpha=0.9)
        sat_plotted = True
    if "pdlc_gamma_saturated_high" in df.columns:
        y = df["pdlc_gamma_saturated_high"].to_numpy(dtype=float) * 100.0
        ax2.plot(step, _rolling_mean(y, rolling), label=r"$\gamma>0.99$", color="#ff7f0e", linewidth=1.4, alpha=0.9)
        sat_plotted = True
    if "pdlc_gamma_saturated_low" in df.columns and "pdlc_gamma_saturated_high" in df.columns:
        y = (df["pdlc_gamma_saturated_low"].to_numpy(dtype=float) + df["pdlc_gamma_saturated_high"].to_numpy(dtype=float)) * 100.0
        ax2.plot(step, _rolling_mean(y, rolling), label=r"Total saturation", color="#7f7f7f", linewidth=1.4, alpha=0.9)
        sat_plotted = True
    if sat_plotted:
        ax2.axhline(5.0, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.set_ylabel(r"$\tau$")
    ax2.set_ylabel("Saturated pairs (%)")
    ax.grid(True, alpha=0.25)
    if "pdlc_tau" in df.columns:
        ax.legend(loc="upper left", frameon=True)
    if sat_plotted:
        ax2.legend(loc="upper right", frameon=True)

    axes[-1].set_xlabel("Training step")

    # Mark the peak margin step (often a useful anchor in the thesis narrative).
    if "pdlc_logit_gap" in df.columns:
        peak_step, _ = _find_peak(step, df["pdlc_logit_gap"].to_numpy(dtype=float))
        if peak_step is not None:
            for a in axes:
                a.axvline(peak_step, color="black", linewidth=1.0, alpha=0.25)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_phase_tau_vs_gap(df: pd.DataFrame, out_path: Path, title: str, sample_every: int) -> None:
    required = {"step", "pdlc_tau", "pdlc_logit_gap"}
    if not required.issubset(set(df.columns)):
        return

    step = df["step"].to_numpy(dtype=int)
    tau = df["pdlc_tau"].to_numpy(dtype=float)
    gap = df["pdlc_logit_gap"].to_numpy(dtype=float)

    ok = np.isfinite(step) & np.isfinite(tau) & np.isfinite(gap)
    step, tau, gap = step[ok], tau[ok], gap[ok]
    if step.size == 0:
        return

    if sample_every > 1:
        step, tau, gap = step[::sample_every], tau[::sample_every], gap[::sample_every]

    fig, ax = plt.subplots(1, 1, figsize=(7.1, 4.6), constrained_layout=True)
    ax.set_title(title)

    sc = ax.scatter(gap, tau, c=step, cmap="viridis", s=10, alpha=0.85, linewidths=0.0)
    ax.plot(gap, tau, color="black", alpha=0.15, linewidth=0.8)

    # Start / end markers
    ax.scatter([gap[0]], [tau[0]], s=55, facecolors="none", edgecolors="black", linewidths=1.2, zorder=3, label="start")
    ax.scatter([gap[-1]], [tau[-1]], s=55, color="black", zorder=3, label="end")

    peak_step, peak_gap = _find_peak(step, gap)
    if peak_step is not None and peak_gap is not None:
        tau_at_peak = float(tau[int(np.argmax(gap))])
        ax.scatter([peak_gap], [tau_at_peak], s=70, color="#d62728", zorder=4, label="peak margin")

    r = float(np.corrcoef(gap, tau)[0, 1]) if gap.size > 2 else float("nan")

    ax.set_xlabel(r"Margin $s_{\mathrm{pos}}-s_{\mathrm{neg}}$")
    ax.set_ylabel(r"Temperature $\tau$")
    ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Training step")

    # Legend in the top-right; put the stats box just below it to avoid covering early points.
    ax.legend(loc="upper right", frameon=True)
    ax.text(
        0.98,
        0.70,
        "L-shaped (non-linear)\n"
        + f"Pearson r = {r:+.3f}\n(n={gap.size})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_phase_tau_vs_gap_two_stage(
    stage1: pd.DataFrame,
    stage2: pd.DataFrame,
    out_path: Path,
    title: str,
    sample_every: int,
) -> None:
    if not {"step", "pdlc_tau", "pdlc_logit_gap"}.issubset(set(stage1.columns)) or not {
        "step",
        "pdlc_tau",
        "pdlc_logit_gap",
    }.issubset(set(stage2.columns)):
        return
    merged, boundary = _concat_stages(stage1, stage2)
    # Use the same scatter as the single-stage plot, but with a visual stage cue.
    step = merged["step_full"].to_numpy(dtype=int)
    tau = merged["pdlc_tau"].to_numpy(dtype=float)
    gap = merged["pdlc_logit_gap"].to_numpy(dtype=float)
    stage = merged["stage"].to_numpy(dtype=int)

    ok = np.isfinite(step) & np.isfinite(tau) & np.isfinite(gap)
    step, tau, gap, stage = step[ok], tau[ok], gap[ok], stage[ok]
    if step.size == 0:
        return

    if sample_every > 1:
        step, tau, gap, stage = step[::sample_every], tau[::sample_every], gap[::sample_every], stage[::sample_every]

    fig, ax = plt.subplots(1, 1, figsize=(7.1, 4.6), constrained_layout=True)
    ax.set_title(title)

    # Color by step, marker by stage
    is_s1 = stage == 1
    is_s2 = stage == 2
    sc1 = ax.scatter(gap[is_s1], tau[is_s1], c=step[is_s1], cmap="viridis", s=10, alpha=0.85, linewidths=0.0, marker="o")
    ax.scatter(gap[is_s2], tau[is_s2], c=step[is_s2], cmap="viridis", s=12, alpha=0.90, linewidths=0.0, marker="^")
    ax.plot(gap, tau, color="black", alpha=0.12, linewidth=0.8)

    # Start / stage2 start / end
    ax.scatter([gap[0]], [tau[0]], s=55, facecolors="none", edgecolors="black", linewidths=1.2, zorder=3, label="start (Stage 1)")
    # first stage2 point (if any)
    idx2 = np.where(is_s2)[0]
    if idx2.size:
        j = int(idx2[0])
        ax.scatter([gap[j]], [tau[j]], s=55, facecolors="none", edgecolors="#d62728", linewidths=1.2, zorder=3, label="start (Stage 2)")
    ax.scatter([gap[-1]], [tau[-1]], s=55, color="black", zorder=3, label="end")

    peak_step, peak_gap = _find_peak(step, gap)
    if peak_step is not None and peak_gap is not None:
        tau_at_peak = float(tau[int(np.argmax(gap))])
        ax.scatter([peak_gap], [tau_at_peak], s=70, color="#d62728", zorder=4, label="peak margin")

    r = float(np.corrcoef(gap, tau)[0, 1]) if gap.size > 2 else float("nan")

    ax.set_xlabel(r"Margin $s_{\mathrm{pos}}-s_{\mathrm{neg}}$")
    ax.set_ylabel(r"Temperature $\tau$")
    ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(sc1, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Training step (Stage 2 offset)")

    # Legend in the top-right; put the stats box just below it to avoid covering early points.
    ax.legend(loc="upper right", frameon=True)
    ax.text(
        0.98,
        0.70,
        "L-shaped (non-linear)\n"
        + f"Pearson r = {r:+.3f}\n(n={gap.size})\nStage boundary @ step {boundary}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def print_key_findings(df: pd.DataFrame, label: str, cutoff: int) -> None:
    step = df["step"].to_numpy(dtype=int)
    print(f"== {label} ==")
    print(f"rows={len(df)} steps=[{int(step.min())}, {int(step.max())}]")

    if "pdlc_logit_gap" in df.columns:
        gap = df["pdlc_logit_gap"].to_numpy(dtype=float)
        peak_step, peak_gap = _find_peak(step, gap)
        print(f"logit gap: first={gap[0]:.4f} last={gap[-1]:.4f} peak={peak_gap:.4f} @step={peak_step}")
        s1 = _first_step_where(step, gap, lambda x: x >= 1.0)
        s2 = _first_step_where(step, gap, lambda x: x >= 2.0)
        if s1 is not None and s2 is not None:
            print(f"logit gap milestones: >=1.0 @step={s1}, >=2.0 @step={s2}")

    if "pdlc_tau" in df.columns:
        tau = df["pdlc_tau"].to_numpy(dtype=float)
        s_tau = _first_step_where(step, tau, lambda x: x <= 0.30)
        print(f"tau: first={tau[0]:.4f} last={tau[-1]:.4f} first<=0.30 @step={s_tau}")

    if "pdlc_pair_logit_mean" in df.columns:
        m = df["pdlc_pair_logit_mean"].to_numpy(dtype=float)
        sd = df["pdlc_pair_logit_std"].to_numpy(dtype=float) if "pdlc_pair_logit_std" in df.columns else None
        if sd is None:
            print(f"pair logits (mean): first={m[0]:.3f} last={m[-1]:.3f}")
        else:
            print(f"pair logits: mean first={m[0]:.3f} last={m[-1]:.3f} | std first={sd[0]:.3f} last={sd[-1]:.3f}")

    if "pdlc_neff_mean" in df.columns:
        neff = df["pdlc_neff_mean"].to_numpy(dtype=float)
        print(f"Neff mean: first={neff[0]:.1f} last={neff[-1]:.1f}")

    sat_cols = {"pdlc_gamma_saturated_low", "pdlc_gamma_saturated_high"}
    if sat_cols.issubset(set(df.columns)):
        sub = df[df["step"] >= cutoff]
        low = sub["pdlc_gamma_saturated_low"].to_numpy(dtype=float)
        high = sub["pdlc_gamma_saturated_high"].to_numpy(dtype=float)
        total = low + high
        print(
            "gamma saturation (after step>=%d): mean=%.2f%% p95=%.2f%% max=%.2f%%"
            % (cutoff, 100.0 * float(np.mean(total)), 100.0 * _percentile(total, 0.95), 100.0 * float(np.max(total)))
        )

    if "pdlc_entropy" in df.columns and df["pdlc_entropy"].notna().sum() == 0:
        print("[note] pdlc_entropy is NaN throughout (likely due to padded classes producing 0 * -inf in entropy).")


def print_key_findings_two_stage(stage1: pd.DataFrame, stage2: pd.DataFrame, label: str, cutoff: int) -> None:
    merged, boundary = _concat_stages(stage1, stage2)
    print(f"== {label} (Stage 1 + Stage 2) ==")
    print(f"stage1_steps=[{int(stage1['step'].min())}, {int(stage1['step'].max())}]  stage2_steps=[{int(stage2['step'].min())}, {int(stage2['step'].max())}]")
    print(f"stage2_offset={boundary}  full_steps=[{int(merged['step_full'].min())}, {int(merged['step_full'].max())}]")
    merged_for_single = merged.copy()
    merged_for_single["step"] = merged_for_single["step_full"]
    print_key_findings(merged_for_single, label="full", cutoff=cutoff)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create thesis-ready plots for PDLC training metrics (PDF).")
    parser.add_argument(
        "--csv",
        default=None,
        help="Single training-metrics CSV to plot (Stage 1 only). Mutually exclusive with --stage1_csv/--stage2_csv.",
    )
    parser.add_argument(
        "--stage1_csv",
        default="training_metrics/mini_tabicl_stage1_pdl_posterior_avg.csv",
        help="Stage-1 training-metrics CSV.",
    )
    parser.add_argument(
        "--stage2_csv",
        default="training_metrics/mini_tabicl_stage2_pdl_posterior_avg.csv",
        help="Stage-2 training-metrics CSV (steps will be offset by Stage-1 max).",
    )
    parser.add_argument(
        "--out_dir",
        default="figures/pdlc_training",
        help="Output directory for PDF figures.",
    )
    parser.add_argument(
        "--rolling",
        type=int,
        default=201,
        help="Smoothing window for time-series plots (single-stage mode).",
    )
    parser.add_argument(
        "--rolling_stage1",
        type=int,
        default=201,
        help="Smoothing window for Stage 1 (two-stage mode).",
    )
    parser.add_argument(
        "--rolling_stage2",
        type=int,
        default=51,
        help="Smoothing window for Stage 2 (two-stage mode).",
    )
    parser.add_argument(
        "--phase_sample_every",
        type=int,
        default=10,
        help="Downsample factor for phase plot (every N steps).",
    )
    parser.add_argument(
        "--print_cutoff",
        type=int,
        default=1000,
        help="Cutoff step used when summarizing saturation statistics.",
    )
    args = parser.parse_args()

    _setup_style()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    if args.csv is not None:
        csv_path = Path(args.csv)
        df = _load_metrics_csv(csv_path)
        label = csv_path.stem

        # Console summary (handy for thesis text).
        print_key_findings(df, label=label, cutoff=int(args.print_cutoff))

        out_ts = out_dir / f"pdlc_training_dynamics_{label}_timeseries.pdf"
        plot_timeseries(df, out_path=out_ts, rolling=int(args.rolling), title="PDLC training dynamics (single stage)")
        print(f"Wrote: {out_ts}")

        out_phase = out_dir / f"pdlc_training_dynamics_{label}_tau_vs_gap.pdf"
        plot_phase_tau_vs_gap(
            df,
            out_path=out_phase,
            title=r"Automatic temperature scaling vs. learned margin",
            sample_every=max(1, int(args.phase_sample_every)),
        )
        print(f"Wrote: {out_phase}")
        return

    stage1_csv = Path(args.stage1_csv)
    stage2_csv = Path(args.stage2_csv)
    s1 = _load_metrics_csv(stage1_csv)
    s2 = _load_metrics_csv(stage2_csv)
    label = _infer_run_label(stage1_csv, stage2_csv)

    print_key_findings_two_stage(s1, s2, label=label, cutoff=int(args.print_cutoff))

    out_ts = out_dir / f"pdlc_training_dynamics_full_{label}_timeseries.pdf"
    plot_timeseries_two_stage(
        s1,
        s2,
        out_path=out_ts,
        rolling_stage1=int(args.rolling_stage1),
        rolling_stage2=int(args.rolling_stage2),
        title="TabICL w. PDLC head training dynamics (Stage 1 + Stage 2)",
    )
    print(f"Wrote: {out_ts}")

    out_phase = out_dir / f"pdlc_training_dynamics_full_{label}_tau_vs_gap.pdf"
    plot_phase_tau_vs_gap_two_stage(
        s1,
        s2,
        out_path=out_phase,
        title=r"Automatic temperature scaling vs. learned margin (full run)",
        sample_every=max(1, int(args.phase_sample_every)),
    )
    print(f"Wrote: {out_phase}")


if __name__ == "__main__":
    main()
