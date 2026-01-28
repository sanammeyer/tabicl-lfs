#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_geo(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"dataset", "seed", "neff_sa_mean", "neff_ea_mean", "cka_sa_ea"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    for col in ("neff_sa_mean", "neff_ea_mean", "cka_sa_ea"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["dataset", "seed", "neff_sa_mean", "neff_ea_mean", "cka_sa_ea"])

    # Per-dataset mean across seeds (this is what the thesis text uses).
    out = (
        df.groupby("dataset", as_index=True)[["neff_sa_mean", "neff_ea_mean", "cka_sa_ea"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    out["d_neff"] = out["neff_ea_mean"] - out["neff_sa_mean"]
    return out


def _load_behaviour_clean_nll(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"dataset", "seed", "nll_sa", "nll_ea"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    for col in ("nll_sa", "nll_ea"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["dataset", "seed", "nll_sa", "nll_ea"])

    out = df.groupby("dataset", as_index=True)[["nll_sa", "nll_ea"]].mean(numeric_only=True).reset_index()
    out["d_nll_clean"] = out["nll_ea"] - out["nll_sa"]
    return out


def _merge_geo_nll(geo: pd.DataFrame, nll: pd.DataFrame) -> pd.DataFrame:
    merged = geo.merge(nll[["dataset", "d_nll_clean"]], on="dataset", how="inner")
    if merged["dataset"].nunique() < geo["dataset"].nunique():
        missing = set(geo["dataset"]).difference(set(merged["dataset"]))
        if missing:
            print(f"[warn] Missing NLL rows for datasets: {sorted(missing)}")
    return merged.sort_values("dataset").reset_index(drop=True)


def _axis_limits(values_full: pd.Series, values_icl: pd.Series) -> Tuple[float, float]:
    vmin = float(min(values_full.min(), values_icl.min()))
    vmax = float(max(values_full.max(), values_icl.max()))
    if vmin == vmax:
        pad = 1.0
    else:
        pad = 0.05 * (vmax - vmin)
    return vmin - pad, vmax + pad


def _annotate_madelon(ax: plt.Axes, df: pd.DataFrame) -> None:
    row = df[df["dataset"] == "madelon"]
    if row.empty:
        return
    x = float(row["d_neff"].iloc[0])
    y = float(row["d_nll_clean"].iloc[0])
    ax.scatter([x], [y], s=70, facecolors="none", edgecolors="black", linewidths=1.2, zorder=5)
    ax.annotate(
        "madelon",
        xy=(x, y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="-", lw=0.8, color="black", shrinkA=0, shrinkB=0),
    )


def _scatter_panel(ax: plt.Axes, df: pd.DataFrame, title: str, vmin: float, vmax: float, cmap: str) -> None:
    sc = ax.scatter(
        df["d_neff"],
        df["d_nll_clean"],
        c=df["cka_sa_ea"],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=42,
        alpha=0.9,
        linewidths=0.0,
    )
    ax.set_title(title)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    median_x = float(df["d_neff"].median())
    ax.axvline(median_x, color="black", linestyle="--", linewidth=1.0, alpha=0.5)

    r = float(df["d_neff"].corr(df["d_nll_clean"]))
    ax.text(
        0.02,
        0.98,
        f"median ΔNeff = {median_x:.0f}\nPearson r = {r:+.3f}\n(n={df['dataset'].nunique()})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
    )

    _annotate_madelon(ax, df)

    ax.grid(True, alpha=0.25)
    return sc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot EA mechanistic diagnostics: ΔNeff vs clean ΔNLL, colored by CKA(SA,EA), "
            "as two panels (EA-Full and EA-ICL)."
        )
    )
    parser.add_argument(
        "--geo_full_csv",
        default="results/compare_mini_tabicl_geometry_ea_row.csv",
        help="Geometry CSV for EA-Full (EA row+icl).",
    )
    parser.add_argument(
        "--geo_icl_csv",
        default="results/compare_mini_tabicl_geometry_ea_icl.csv",
        help="Geometry CSV for EA-ICL (EA icl-only).",
    )
    parser.add_argument(
        "--behaviour_full_csv",
        default="results/compare_mini_tabicl_behaviour_ea_row.csv",
        help="Behaviour clean CSV for EA-Full (contains nll_sa/nll_ea).",
    )
    parser.add_argument(
        "--behaviour_icl_csv",
        default="results/compare_mini_tabicl_behaviour_ea_icl.csv",
        help="Behaviour clean CSV for EA-ICL (contains nll_sa/nll_ea).",
    )
    parser.add_argument(
        "--out",
        default="results/ea_mechanism_analysis/ea_mechanism_scatter.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap for CKA.",
    )
    args = parser.parse_args()

    geo_full = _load_geo(Path(args.geo_full_csv))
    geo_icl = _load_geo(Path(args.geo_icl_csv))

    nll_full = _load_behaviour_clean_nll(Path(args.behaviour_full_csv))
    nll_icl = _load_behaviour_clean_nll(Path(args.behaviour_icl_csv))

    full = _merge_geo_nll(geo_full, nll_full)
    icl = _merge_geo_nll(geo_icl, nll_icl)

    # Shared color scale for interpretability.
    cka_min = float(min(full["cka_sa_ea"].min(), icl["cka_sa_ea"].min()))
    cka_max = float(max(full["cka_sa_ea"].max(), icl["cka_sa_ea"].max()))

    # Shared axis ranges.
    xlim = _axis_limits(full["d_neff"], icl["d_neff"])
    ylim = _axis_limits(full["d_nll_clean"], icl["d_nll_clean"])

    out_path = Path(args.out)
    _ensure_dir(out_path.parent)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True, constrained_layout=True)
    _scatter_panel(ax1, full, "EA-Full (row + ICL)", vmin=cka_min, vmax=cka_max, cmap=str(args.cmap))
    _scatter_panel(ax2, icl, "EA-ICL (ICL only)", vmin=cka_min, vmax=cka_max, cmap=str(args.cmap))

    ax1.set_xlim(*xlim)
    ax2.set_xlim(*xlim)
    ax1.set_ylim(*ylim)

    ax1.set_xlabel(r"$\Delta N_{\mathrm{eff}}$ (EA $-$ SA)")
    ax2.set_xlabel(r"$\Delta N_{\mathrm{eff}}$ (EA $-$ SA)")
    ax1.set_ylabel(r"$\Delta\,\mathrm{NLL}_{\mathrm{clean}}$ (EA $-$ SA)")

    # One shared colorbar.
    sm = plt.cm.ScalarMappable(cmap=str(args.cmap), norm=plt.Normalize(vmin=cka_min, vmax=cka_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    cbar.set_label("CKA(SA, EA)")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # Print key summary stats (for copy/paste into thesis text).
    print(f"Wrote: {out_path}")
    print(f"[EA-Full] median ΔNeff={full['d_neff'].median():.3f}  mean CKA={full['cka_sa_ea'].mean():.6f}  r={full['d_neff'].corr(full['d_nll_clean']):+.6f}")
    print(f"[EA-ICL ] median ΔNeff={icl['d_neff'].median():.3f}  mean CKA={icl['cka_sa_ea'].mean():.6f}  r={icl['d_neff'].corr(icl['d_nll_clean']):+.6f}")


if __name__ == "__main__":
    main()
