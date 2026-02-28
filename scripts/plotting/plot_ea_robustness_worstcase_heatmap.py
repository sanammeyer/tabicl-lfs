#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import AxesImage


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _mean_by_dataset(
    df: pd.DataFrame,
    *,
    case: str,
    checkpoints: List[str],
    dataset_col: str,
    metric: str,
) -> pd.DataFrame:
    sub = df[(df["case"].astype(str) == str(case)) & (df["checkpoint_name"].isin(checkpoints))].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=[dataset_col, "checkpoint_name", "seed", metric])
    g = (
        sub.groupby([dataset_col, "checkpoint_name"], as_index=False)[[metric]]
        .mean(numeric_only=True)
        .pivot(index=dataset_col, columns="checkpoint_name", values=metric)
    )
    return g


def _heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    cmap: str,
    center_zero: bool,
    fmt: str,
    title_font_size: float,
    tick_font_size: float,
    cell_font_size: float,
) -> AxesImage:
    if center_zero:
        vmax = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 1.0
        vmin = -vmax
    else:
        vmin = float(np.nanmin(data)) if np.isfinite(data).any() else 0.0
        vmax = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0

    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=title_font_size)
    ax.set_xticks(range(len(col_labels)), labels=col_labels, rotation=0)
    ax.set_yticks(range(len(row_labels)), labels=row_labels)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=cell_font_size, color="black")

    ax.tick_params(axis="both", which="both", length=0, labelsize=tick_font_size)
    ax.grid(False)
    return im


def _pretty_case(case: str) -> str:
    case = str(case)
    mapping = {
        "clean": "Clean",
        "uninformative_both_n50": r"Uninformative feature injection ($k=50$)",
        "uninformative_both_n10": r"Uninformative feature injection ($k=10$)",
        "label_noise_train_frac0.1": r"Noisy demonstration labels ($\\rho=0.1$)",
        "outliers_both_fac10": r"Cell-wise numeric outliers ($\\alpha=10$)",
        "outliers_both_fac50": r"Cell-wise numeric outliers ($\\alpha=50$)",
    }
    return mapping.get(case, case)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot worst-case robustness summary as a heatmap: per-dataset change from clean to a stress case "
            "(default: uninformative_both_n50) for SA/EA-ICL/EA-FULL."
        )
    )
    parser.add_argument(
        "--metrics_csv",
        default="results/robustness/diag10_w_collapse_measure/metrics.csv",
        help="Long-format robustness metrics.csv.",
    )
    parser.add_argument("--dataset_col", default="dataset_name", help="Dataset column name in metrics.csv.")
    parser.add_argument("--case_clean", default="clean", help="Clean case name.")
    parser.add_argument("--case_stress", default="uninformative_both_n50", help="Stress case name.")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=["SA", "EA-ICL", "EA-FULL"],
        help="Checkpoint names to include (default: SA EA-ICL EA-FULL).",
    )
    parser.add_argument(
        "--out",
        default="figures/ea_robustness_worstcase_heatmap.pdf",
        help="Output PDF path.",
    )
    parser.add_argument("--font_size", type=float, default=13.0, help="Base font size for the figure (pt).")
    parser.add_argument("--title_font_size", type=float, default=14.0, help="Heatmap title font size (pt).")
    parser.add_argument("--tick_font_size", type=float, default=12.0, help="Axis tick label font size (pt).")
    parser.add_argument("--cell_font_size", type=float, default=10.0, help="Cell annotation font size (pt).")
    parser.add_argument("--cbar_label_size", type=float, default=12.0, help="Colorbar label font size (pt).")
    parser.add_argument("--cbar_tick_size", type=float, default=11.0, help="Colorbar tick label font size (pt).")
    args = parser.parse_args()

    plt.rcParams.update({"font.size": float(args.font_size)})

    metrics_path = Path(args.metrics_csv)
    df = pd.read_csv(metrics_path)

    needed = {"checkpoint_name", "case", "seed", str(args.dataset_col), "acc", "nll"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"{metrics_path} missing columns required for this plot: {sorted(missing)}")

    ckpts = [str(x) for x in args.checkpoints]
    dataset_col = str(args.dataset_col)

    acc_clean = _mean_by_dataset(df, case=str(args.case_clean), checkpoints=ckpts, dataset_col=dataset_col, metric="acc")
    acc_stress = _mean_by_dataset(
        df, case=str(args.case_stress), checkpoints=ckpts, dataset_col=dataset_col, metric="acc"
    )
    nll_clean = _mean_by_dataset(df, case=str(args.case_clean), checkpoints=ckpts, dataset_col=dataset_col, metric="nll")
    nll_stress = _mean_by_dataset(
        df, case=str(args.case_stress), checkpoints=ckpts, dataset_col=dataset_col, metric="nll"
    )

    # Align datasets across all pivots.
    datasets = sorted(set(acc_clean.index) & set(acc_stress.index) & set(nll_clean.index) & set(nll_stress.index))
    if not datasets:
        raise ValueError("No overlapping datasets found between clean and stress cases.")

    acc_delta = (acc_stress.loc[datasets, ckpts] - acc_clean.loc[datasets, ckpts]).to_numpy(dtype=float)
    nll_delta = (nll_stress.loc[datasets, ckpts] - nll_clean.loc[datasets, ckpts]).to_numpy(dtype=float)

    out_path = Path(args.out)
    _ensure_dir(out_path.parent)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.0, 6.5), constrained_layout=True, sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    col_labels = [
        r"SA",
        r"EA-ICL ($\mathrm{TF}_{\mathrm{icl}}$)",
        r"EA-Full ($\mathrm{TF}_{\mathrm{row}},\,\mathrm{TF}_{\mathrm{icl}}$)",
    ]
    # If user overrides checkpoints, fall back to raw names.
    if ckpts != ["SA", "EA-ICL", "EA-FULL"]:
        col_labels = ckpts

    im1 = _heatmap(
        ax1,
        acc_delta,
        row_labels=datasets,
        col_labels=col_labels,
        title=f"Accuracy change: {_pretty_case(args.case_stress)} − {_pretty_case(args.case_clean)}",
        cmap="coolwarm",
        center_zero=True,
        fmt="+.3f",
        title_font_size=float(args.title_font_size),
        tick_font_size=float(args.tick_font_size),
        cell_font_size=float(args.cell_font_size),
    )
    im2 = _heatmap(
        ax2,
        nll_delta,
        row_labels=datasets,
        col_labels=col_labels,
        title=f"NLL change: {_pretty_case(args.case_stress)} − {_pretty_case(args.case_clean)}",
        cmap="coolwarm",
        center_zero=True,
        fmt="+.3f",
        title_font_size=float(args.title_font_size),
        tick_font_size=float(args.tick_font_size),
        cell_font_size=float(args.cell_font_size),
    )

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.set_label("Δacc", fontsize=float(args.cbar_label_size))
    cbar1.ax.tick_params(labelsize=float(args.cbar_tick_size))

    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.set_label("ΔNLL", fontsize=float(args.cbar_label_size))
    cbar2.ax.tick_params(labelsize=float(args.cbar_tick_size))

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_path}")

    # Quick textual sanity.
    def _summ(arr: np.ndarray) -> Tuple[float, float]:
        return float(np.nanmin(arr)), float(np.nanmax(arr))

    acc_min, acc_max = _summ(acc_delta)
    nll_min, nll_max = _summ(nll_delta)
    print(f"[acc] min={acc_min:+.4f} max={acc_max:+.4f}")
    print(f"[nll] min={nll_min:+.4f} max={nll_max:+.4f}")


if __name__ == "__main__":
    main()
