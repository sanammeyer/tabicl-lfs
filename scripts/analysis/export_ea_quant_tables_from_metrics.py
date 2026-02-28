#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import pandas as pd


Variant = Literal["EA-FULL", "EA-ICL"]


@dataclass(frozen=True)
class GroupSpec:
    key: str
    label: str
    cases: Tuple[str, ...]


GROUPS: Tuple[GroupSpec, ...] = (
    GroupSpec("clean", "Clean", ("clean",)),
    GroupSpec("outliers", r"Cell-wise outliers ($\alpha\in\{10,50\}$)", ("outliers_both_fac10", "outliers_both_fac50")),
    GroupSpec("label_noise", r"Label poisoning ($\rho=0.1$)", ("label_noise_train_frac0.1",)),
    GroupSpec(
        "uninformative",
        r"Uninformative features ($k\in\{10,50\}$)",
        ("uninformative_both_n10", "uninformative_both_n50"),
    ),
    GroupSpec(
        "rotation",
        r"Feature rotation ($k=\min(8,d_{\text{num}})$)",
        ("rotation_both_k2", "rotation_both_k6", "rotation_both_k7", "rotation_both_k8"),
    ),
)


def _load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"checkpoint_name", "case", "dataset_name", "seed", "nll"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df.copy()
    df["nll"] = pd.to_numeric(df["nll"], errors="coerce")
    return df.dropna(subset=["checkpoint_name", "case", "dataset_name", "seed", "nll"])


def _per_dataset_case_delta_nll(df: pd.DataFrame, case: str, variant: Variant) -> pd.Series:
    """Return Δabs NLL (EA - SA) per dataset for a single case, averaged over seeds."""
    sub = df[(df["case"] == case) & (df["checkpoint_name"].isin(["SA", variant]))]
    piv = (
        sub.groupby(["dataset_name", "checkpoint_name"], as_index=True)["nll"]
        .mean()
        .unstack("checkpoint_name")
        .dropna(subset=["SA", variant])
    )
    return piv[variant] - piv["SA"]


def _collect_group_abs_deltas(df: pd.DataFrame, group: GroupSpec, variant: Variant) -> pd.Series:
    """Concatenate per-dataset deltas across all cases in the group (dataset×setting combos)."""
    parts = []
    for c in group.cases:
        d = _per_dataset_case_delta_nll(df, c, variant)
        d.index = pd.MultiIndex.from_product([[c], d.index], names=["case", "dataset_name"])
        parts.append(d)
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).rename("d_abs_nll")


def _collect_group_rob_gains(df: pd.DataFrame, group: GroupSpec, variant: Variant) -> pd.Series:
    """Return Δrob per dataset×case in group: (Δabs_case - Δabs_clean)."""
    if group.key == "clean":
        raise ValueError("Robustness gains are not defined for the clean group.")
    clean = _per_dataset_case_delta_nll(df, "clean", variant).rename("d_abs_clean")

    parts = []
    for c in group.cases:
        d_case = _per_dataset_case_delta_nll(df, c, variant).rename("d_abs_case")
        merged = pd.concat([d_case, clean], axis=1, join="inner").dropna()
        d_rob = (merged["d_abs_case"] - merged["d_abs_clean"]).rename("d_rob")
        d_rob.index = pd.MultiIndex.from_product([[c], d_rob.index], names=["case", "dataset_name"])
        parts.append(d_rob)
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts)


def _mean(x: pd.Series) -> float:
    return float(x.mean()) if len(x) else float("nan")


def _count_improved(x: pd.Series) -> int:
    return int((x < 0.0).sum())


def _latex_float(x: float, ndigits: int = 4, signed: bool = True) -> str:
    if pd.isna(x):
        return "nan"
    fmt = f"{{:{'+' if signed else ''}.{ndigits}f}}"
    return fmt.format(float(x))


def main() -> None:
    ap = argparse.ArgumentParser(description="Export EA quantitative tables from a robustness run metrics.csv.")
    ap.add_argument(
        "--metrics_csv",
        type=Path,
        default=Path("results/robustness/diag10_w_collapse_measure/metrics.csv"),
        help="Path to metrics.csv.",
    )
    ap.add_argument(
        "--out_json",
        type=Path,
        default=None,
        help="Optional output JSON path with all computed numbers.",
    )
    ap.add_argument(
        "--out_tex",
        type=Path,
        default=None,
        help="Optional output .tex snippet path with the two tables' numbers (no table environment).",
    )
    ap.add_argument("--ndigits", type=int, default=4, help="Decimal places for LaTeX floats (default: 4).")
    args = ap.parse_args()

    df = _load_metrics(Path(args.metrics_csv))

    # Absolute deltas (averaged across dataset×setting combos)
    abs_table: Dict[str, Dict[Variant, Dict[str, float]]] = {}
    for g in GROUPS:
        abs_table[g.key] = {}
        for v in ("EA-FULL", "EA-ICL"):
            d = _collect_group_abs_deltas(df, g, v)
            abs_table[g.key][v] = {"mean": _mean(d), "n": int(len(d))}

    # Robustness gains (relative to clean)
    rob_table: Dict[str, Dict[Variant, Dict[str, float]]] = {}
    for g in GROUPS:
        if g.key == "clean":
            continue
        rob_table[g.key] = {}
        for v in ("EA-FULL", "EA-ICL"):
            d = _collect_group_rob_gains(df, g, v)
            rob_table[g.key][v] = {"mean": _mean(d), "n": int(len(d)), "improved": _count_improved(d)}

    payload = {
        "metrics_csv": str(Path(args.metrics_csv)),
        "abs_delta_nll": abs_table,
        "robustness_gains_nll": rob_table,
        "notes": [
            "All deltas computed per dataset (mean over seeds) and then averaged across dataset×setting combos.",
            "Improved counts are the number of dataset×setting combos with Δrob < 0.",
        ],
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote: {args.out_json}")

    if args.out_tex:
        lines: List[str] = []
        lines.append("% Auto-generated from metrics.csv via export_ea_quant_tables_from_metrics.py")
        lines.append("% Absolute ΔNLL (EA - SA)")
        for g in GROUPS:
            ef = abs_table[g.key]["EA-FULL"]["mean"]
            ei = abs_table[g.key]["EA-ICL"]["mean"]
            lines.append(
                f"% {g.label}: EA-Full { _latex_float(ef, ndigits=int(args.ndigits), signed=True) }, "
                f"EA-ICL { _latex_float(ei, ndigits=int(args.ndigits), signed=True) }"
            )
        lines.append("% Robustness gains Δrob (relative to clean)")
        for g in GROUPS:
            if g.key == "clean":
                continue
            ef = rob_table[g.key]["EA-FULL"]["mean"]
            ei = rob_table[g.key]["EA-ICL"]["mean"]
            ef_imp = rob_table[g.key]["EA-FULL"]["improved"]
            ei_imp = rob_table[g.key]["EA-ICL"]["improved"]
            n = rob_table[g.key]["EA-FULL"]["n"]
            lines.append(
                f"% {g.label}: EA-Full { _latex_float(ef, ndigits=int(args.ndigits), signed=True) } "
                f"({ef_imp}/{n}), EA-ICL { _latex_float(ei, ndigits=int(args.ndigits), signed=True) } ({ei_imp}/{n})"
            )

        args.out_tex.parent.mkdir(parents=True, exist_ok=True)
        args.out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote: {args.out_tex}")

    # Always print a compact console summary for quick copy/paste
    print("\nAbsolute ΔNLL (EA - SA):")
    for g in GROUPS:
        ef = abs_table[g.key]["EA-FULL"]["mean"]
        ei = abs_table[g.key]["EA-ICL"]["mean"]
        print(f"- {g.label}: EA-Full {_latex_float(ef, ndigits=int(args.ndigits))} | EA-ICL {_latex_float(ei, ndigits=int(args.ndigits))}")

    print("\nΔrob (relative to clean):")
    for g in GROUPS:
        if g.key == "clean":
            continue
        ef = rob_table[g.key]["EA-FULL"]["mean"]
        ei = rob_table[g.key]["EA-ICL"]["mean"]
        ef_imp = rob_table[g.key]["EA-FULL"]["improved"]
        ei_imp = rob_table[g.key]["EA-ICL"]["improved"]
        n = rob_table[g.key]["EA-FULL"]["n"]
        print(
            f"- {g.label}: EA-Full {_latex_float(ef, ndigits=int(args.ndigits))} ({ef_imp}/{n}) | "
            f"EA-ICL {_latex_float(ei, ndigits=int(args.ndigits))} ({ei_imp}/{n})"
        )


if __name__ == "__main__":
    main()

