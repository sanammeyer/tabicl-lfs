#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_deltas(
    metrics_csv: Path,
    *,
    case: str,
    sa_name: str,
    ea_name: str,
    dataset_col: str,
    neff_col: str,
    purity_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    required = {dataset_col, "seed", "checkpoint_name", "case", neff_col, purity_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{metrics_csv} missing columns: {sorted(missing)}")

    df = df[df["case"].astype(str) == str(case)].copy()
    df = df[df["checkpoint_name"].isin([sa_name, ea_name])].copy()

    for col in (neff_col, purity_col):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[dataset_col, "seed", "checkpoint_name", neff_col, purity_col])

    g = (
        df.groupby([dataset_col, "checkpoint_name"], as_index=False)[[neff_col, purity_col]]
        .mean(numeric_only=True)
        .pivot(index=dataset_col, columns="checkpoint_name")
    )

    out = pd.DataFrame(index=g.index)
    out["dataset"] = out.index.astype(str)
    out["d_neff"] = g[(neff_col, ea_name)] - g[(neff_col, sa_name)]
    out["d_purity_top5"] = g[(purity_col, ea_name)] - g[(purity_col, sa_name)]
    return out.reset_index(drop=True).sort_values("dataset").reset_index(drop=True)


def _axis_limits(values_full: pd.Series, values_icl: pd.Series) -> Tuple[float, float]:
    vmin = float(min(values_full.min(), values_icl.min()))
    vmax = float(max(values_full.max(), values_icl.max()))
    if vmin == vmax:
        pad = 1.0
    else:
        pad = 0.05 * (vmax - vmin)
    return vmin - pad, vmax + pad


def _scatter_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    ax.scatter(df["d_neff"], df["d_purity_top5"], s=46, alpha=0.9, linewidths=0.0)
    ax.set_title(title)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    median_x = float(df["d_neff"].median())
    ax.axvline(median_x, color="black", linestyle="--", linewidth=1.0, alpha=0.5)

    r = float(df["d_neff"].corr(df["d_purity_top5"]))
    ax.text(
        0.98,
        0.98,
        f"median ΔNeff = {median_x:.0f}\nPearson r = {r:+.3f}\n(n={df['dataset'].nunique()})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
    )
    ax.grid(True, alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ΔNeff vs Δpurity_top5 on the clean case as two panels (EA-Full and EA-ICL)."
    )
    parser.add_argument(
        "--metrics_csv",
        default="results/robustness/diag10_w_collapse_measure/metrics.csv",
        help="Long-format robustness metrics.csv.",
    )
    parser.add_argument("--case", default="clean", help="Which case to use (default: clean).")
    parser.add_argument("--dataset_col", default="dataset_name", help="Dataset column name in metrics.csv.")
    parser.add_argument("--neff_col", default="neff_mean", help="Neff column name in metrics.csv.")
    parser.add_argument("--purity_col", default="purity_top5", help="Purity(top5) column name in metrics.csv.")
    parser.add_argument(
        "--title_full",
        default=r"EA-Full ($\mathrm{TF}_{\mathrm{row}},\,\mathrm{TF}_{\mathrm{icl}}$)",
        help="Panel title for EA-Full.",
    )
    parser.add_argument(
        "--title_icl",
        default=r"EA-ICL ($\mathrm{TF}_{\mathrm{icl}}$)",
        help="Panel title for EA-ICL.",
    )
    parser.add_argument(
        "--out",
        default="figures/ea_neff_vs_purity_top5_scatter.pdf",
        help="Output PDF path.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv)
    full = _load_deltas(
        metrics_path,
        case=str(args.case),
        sa_name="SA",
        ea_name="EA-FULL",
        dataset_col=str(args.dataset_col),
        neff_col=str(args.neff_col),
        purity_col=str(args.purity_col),
    )
    icl = _load_deltas(
        metrics_path,
        case=str(args.case),
        sa_name="SA",
        ea_name="EA-ICL",
        dataset_col=str(args.dataset_col),
        neff_col=str(args.neff_col),
        purity_col=str(args.purity_col),
    )

    out_path = Path(args.out)
    _ensure_dir(out_path.parent)

    xlim = _axis_limits(full["d_neff"], icl["d_neff"])
    ylim = _axis_limits(full["d_purity_top5"], icl["d_purity_top5"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True, constrained_layout=True)
    _scatter_panel(ax1, full, str(args.title_full))
    _scatter_panel(ax2, icl, str(args.title_icl))

    ax1.set_xlim(*xlim)
    ax2.set_xlim(*xlim)
    ax1.set_ylim(*ylim)

    ax1.set_xlabel(r"$\Delta N_{\mathrm{eff}}$ (EA $-$ SA)")
    ax2.set_xlabel(r"$\Delta N_{\mathrm{eff}}$ (EA $-$ SA)")
    ax1.set_ylabel(r"$\Delta\,\mathrm{purity}_{\mathrm{top5}}$ (EA $-$ SA)")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_path}")
    for label, d in [("EA-FULL", full), ("EA-ICL", icl)]:
        both = int(((d["d_neff"] > 0) & (d["d_purity_top5"] > 0)).sum())
        n = int(d["dataset"].nunique())
        print(
            f"[{label}] ΔNeff>0: {(d['d_neff']>0).sum()}/{n} | Δpurity_top5>0: {(d['d_purity_top5']>0).sum()}/{n} | both: {both}/{n}"
        )


if __name__ == "__main__":
    main()
