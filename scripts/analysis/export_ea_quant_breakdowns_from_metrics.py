#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

Variant = Literal["EA-FULL", "EA-ICL"]
Metric = Literal["nll", "acc", "f1_macro", "ece", "brier"]


@dataclass(frozen=True)
class RowSpec:
    key: str
    label: str
    cases: Tuple[str, ...]


ABS_ROWS: Tuple[RowSpec, ...] = (
    RowSpec("clean", "Clean", ("clean",)),
    RowSpec("out_a10", r"Cell-wise outliers ($\alpha=10$)", ("outliers_both_fac10",)),
    RowSpec("out_a50", r"Cell-wise outliers ($\alpha=50$)", ("outliers_both_fac50",)),
    RowSpec("lab_r01", r"Label poisoning ($\rho=0.1$)", ("label_noise_train_frac0.1",)),
    RowSpec("uni_k10", r"Uninformative features ($k=10$)", ("uninformative_both_n10",)),
    RowSpec("uni_k50", r"Uninformative features ($k=50$)", ("uninformative_both_n50",)),
    RowSpec(
        "rot_avg",
        r"Feature rotation ($k=\min(8,d_{\text{num}})$)",
        ("rotation_both_k2", "rotation_both_k6", "rotation_both_k7", "rotation_both_k8"),
    ),
)

ROB_ROWS: Tuple[RowSpec, ...] = tuple(r for r in ABS_ROWS if r.key != "clean")


def _load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"checkpoint_name", "case", "dataset_name", "seed", "nll", "acc", "f1_macro", "ece", "brier"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df.copy()
    for m in ["nll", "acc", "f1_macro", "ece", "brier"]:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return df.dropna(
        subset=["checkpoint_name", "case", "dataset_name", "seed", "nll", "acc", "f1_macro", "ece", "brier"]
    )


def _metric_prefers_lower(metric: str) -> bool:
    return metric in {"nll", "ece", "brier"}


def _wilcoxon_p(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float("nan")
    if np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values).pvalue)


def _per_dataset_means(df: pd.DataFrame, cases: Tuple[str, ...], ckpts: Tuple[str, ...]) -> pd.DataFrame:
    sub = df[df["case"].isin(list(cases)) & df["checkpoint_name"].isin(list(ckpts))].copy()
    # dataset mean over seeds for each case, then average across cases/settings inside each dataset
    g = (
        sub.groupby(["case", "dataset_name", "checkpoint_name"], as_index=False)[["nll", "acc", "f1_macro", "ece", "brier"]]
        .mean(numeric_only=True)
        .groupby(["dataset_name", "checkpoint_name"], as_index=False)[["nll", "acc", "f1_macro", "ece", "brier"]]
        .mean(numeric_only=True)
    )
    piv = g.pivot(index="dataset_name", columns="checkpoint_name", values=["nll", "acc", "f1_macro", "ece", "brier"])
    return piv


def _abs_deltas(df: pd.DataFrame, row: RowSpec, variant: Variant) -> pd.DataFrame:
    piv = _per_dataset_means(df, row.cases, ckpts=("SA", variant))
    out = piv.xs(variant, level=1, axis=1) - piv.xs("SA", level=1, axis=1)
    out = out.dropna()
    out.index.name = "dataset_name"
    return out.reset_index()


def _rob_gains(df: pd.DataFrame, row: RowSpec, variant: Variant) -> pd.DataFrame:
    abs_case = _abs_deltas(df, row, variant).set_index("dataset_name")
    abs_clean = _abs_deltas(df, next(r for r in ABS_ROWS if r.key == "clean"), variant).set_index("dataset_name")
    merged = abs_case.join(abs_clean, how="inner", lsuffix="_case", rsuffix="_clean")
    out = pd.DataFrame(index=merged.index)
    for m in ["nll", "acc", "f1_macro", "ece", "brier"]:
        out[m] = merged[f"{m}_case"] - merged[f"{m}_clean"]
    out.index.name = "dataset_name"
    return out.reset_index()


def _fmt(x: float, ndigits: int = 4, signed: bool = True) -> str:
    if pd.isna(x):
        return "nan"
    return f"{float(x):+.{ndigits}f}" if signed else f"{float(x):.{ndigits}f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export per-severity, per-metric EA absolute deltas and robustness gains from metrics.csv."
    )
    ap.add_argument(
        "--metrics_csv",
        type=Path,
        default=Path("results/robustness/diag10_w_collapse_measure/metrics.csv"),
        help="Path to metrics.csv.",
    )
    ap.add_argument("--ndigits", type=int, default=4, help="Decimal places.")
    ap.add_argument("--out_json", type=Path, default=None, help="Optional output JSON.")
    ap.add_argument("--out_tex", type=Path, default=None, help="Optional output LaTeX snippet (comment lines).")
    args = ap.parse_args()

    df = _load_metrics(Path(args.metrics_csv))
    variants: Tuple[Variant, ...] = ("EA-FULL", "EA-ICL")
    metrics: Tuple[Metric, ...] = ("nll", "acc", "f1_macro", "ece", "brier")

    abs_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    rob_summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    # Absolute deltas (EA - SA)
    for row in ABS_ROWS:
        abs_summary[row.key] = {}
        for v in variants:
            d = _abs_deltas(df, row, v).set_index("dataset_name")
            abs_summary[row.key][v] = {
                "n_datasets": int(d.shape[0]),
                **{m: float(d[m].mean()) for m in metrics},
            }

    # Robustness gains (Δabs_case - Δabs_clean)
    for row in ROB_ROWS:
        rob_summary[row.key] = {}
        for v in variants:
            d = _rob_gains(df, row, v).set_index("dataset_name")
            rob_summary[row.key][v] = {
                "n_datasets": int(d.shape[0]),
                **{m: float(d[m].mean()) for m in metrics},
                **{
                    f"improved_{m}": int(((d[m] < 0) if _metric_prefers_lower(m) else (d[m] > 0)).sum())
                    for m in metrics
                },
                **{f"p_wilcoxon_{m}": _wilcoxon_p(d[m].to_numpy(dtype=float)) for m in metrics},
            }

    payload = {
        "metrics_csv": str(Path(args.metrics_csv)),
        "abs_delta_vs_sa": abs_summary,
        "robustness_gain_vs_clean": rob_summary,
        "notes": [
            "Dataset is the unit: metrics are averaged over seeds within each dataset.",
            "For multi-setting rows (e.g., rotation), values are averaged over available settings within each dataset.",
            "Absolute deltas: (EA - SA). Robustness gains: (Δabs_case - Δabs_clean).",
            "Wilcoxon p-values are two-sided tests of median=0 across datasets.",
        ],
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote: {args.out_json}")

    if args.out_tex:
        lines: List[str] = []
        lines.append("% Auto-generated: per-severity, per-metric EA breakdowns from metrics.csv")
        lines.append("% Absolute deltas (EA - SA) [mean across datasets]")
        for row in ABS_ROWS:
            lines.append(f"% {row.label}")
            for v in variants:
                d = abs_summary[row.key][v]
                lines.append(
                    f"%   {v}: ΔNLL={_fmt(d['nll'], args.ndigits)}; Δacc={_fmt(d['acc'], args.ndigits)}; "
                    f"ΔF1={_fmt(d['f1_macro'], args.ndigits)}; ΔECE={_fmt(d['ece'], args.ndigits)} (n={d['n_datasets']})"
                )
        lines.append("% Robustness gains (relative to clean) [mean across datasets] + Wilcoxon p-values")
        for row in ROB_ROWS:
            lines.append(f"% {row.label}")
            for v in variants:
                d = rob_summary[row.key][v]
                lines.append(
                    f"%   {v}: Δrob(NLL)={_fmt(d['nll'], args.ndigits)} (p={d['p_wilcoxon_nll']:.4g}); "
                    f"Δrob(ECE)={_fmt(d['ece'], args.ndigits)} (p={d['p_wilcoxon_ece']:.4g}); "
                    f"Δrob(acc)={_fmt(d['acc'], args.ndigits)} (p={d['p_wilcoxon_acc']:.4g}); "
                    f"Δrob(F1)={_fmt(d['f1_macro'], args.ndigits)} (p={d['p_wilcoxon_f1_macro']:.4g}); "
                    f"n={d['n_datasets']}"
                )

        args.out_tex.parent.mkdir(parents=True, exist_ok=True)
        args.out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote: {args.out_tex}")

    # Console summary focused on requested metrics
    print("\nAbsolute deltas vs SA (mean across datasets):")
    for row in ABS_ROWS:
        ef = abs_summary[row.key]["EA-FULL"]
        ei = abs_summary[row.key]["EA-ICL"]
        print(
            f"- {row.label}: "
            f"EA-Full ΔNLL={_fmt(ef['nll'], args.ndigits)} Δacc={_fmt(ef['acc'], args.ndigits)} "
            f"ΔF1={_fmt(ef['f1_macro'], args.ndigits)} ΔECE={_fmt(ef['ece'], args.ndigits)} | "
            f"EA-ICL ΔNLL={_fmt(ei['nll'], args.ndigits)} Δacc={_fmt(ei['acc'], args.ndigits)} "
            f"ΔF1={_fmt(ei['f1_macro'], args.ndigits)} ΔECE={_fmt(ei['ece'], args.ndigits)}"
        )

    print("\nRobustness gains vs clean (mean across datasets), with Wilcoxon p (two-sided):")
    for row in ROB_ROWS:
        ef = rob_summary[row.key]["EA-FULL"]
        ei = rob_summary[row.key]["EA-ICL"]
        print(
            f"- {row.label}: "
            f"EA-Full Δrob(NLL)={_fmt(ef['nll'], args.ndigits)} (p={ef['p_wilcoxon_nll']:.4g}) "
            f"Δrob(ECE)={_fmt(ef['ece'], args.ndigits)} (p={ef['p_wilcoxon_ece']:.4g}) | "
            f"EA-ICL Δrob(NLL)={_fmt(ei['nll'], args.ndigits)} (p={ei['p_wilcoxon_nll']:.4g}) "
            f"Δrob(ECE)={_fmt(ei['ece'], args.ndigits)} (p={ei['p_wilcoxon_ece']:.4g})"
        )


if __name__ == "__main__":
    main()

