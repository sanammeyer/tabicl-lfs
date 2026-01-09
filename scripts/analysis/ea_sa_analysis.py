#!/usr/bin/env python
"""Compute EA vs SA deltas and summary stats used in the results discussion.

This script reproduces the descriptive numbers mentioned in the write-up:
- CC18 performance deltas (accuracy, macro-F1, log-loss)
- Geometry changes (cosine similarity, collapse, purity, effective contexts)
- Behaviour/calibration trade-offs (acc, F1, NLL, Brier, ECE)
- Robustness under corruptions
- Optional correlations between geometry shifts and behavioural deltas
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


@dataclass
class MetricDelta:
    metric: str
    mean_sa: float
    mean_ea: float
    delta_mean: float
    delta_std: float
    frac_ea_better: float


def _format_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _load_csv(filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def summarize_cc18() -> Dict[str, pd.DataFrame]:
    sa = _load_csv("cc18_tabicl_step-1000.csv")
    ea = _load_csv("cc18_tabicl_step-1000_ea.csv")
    merged = sa.merge(ea, on=["task_id", "dataset_id", "dataset_name"], suffixes=("_sa", "_ea"))
    valid = merged[merged["error_sa"].isna() & merged["error_ea"].isna()]

    rows: List[MetricDelta] = []
    best_if_lower = {"log_loss"}
    for metric in ["accuracy", "f1_macro", "log_loss"]:
        m_sa = valid[f"{metric}_mean_sa"]
        m_ea = valid[f"{metric}_mean_ea"]
        delta = m_ea - m_sa
        better = delta < 0 if metric in best_if_lower else delta > 0
        rows.append(
            MetricDelta(
                metric=metric,
                mean_sa=m_sa.mean(),
                mean_ea=m_ea.mean(),
                delta_mean=delta.mean(),
                delta_std=delta.std(),
                frac_ea_better=better.mean(),
            )
        )

    summary_df = pd.DataFrame(rows)

    valid = valid.copy()
    valid["delta_log_loss"] = valid["log_loss_mean_ea"] - valid["log_loss_mean_sa"]
    worst = valid.sort_values("delta_log_loss", ascending=False)[
        ["dataset_name", "log_loss_mean_sa", "log_loss_mean_ea", "delta_log_loss"]
    ].head()
    best = valid.sort_values("delta_log_loss", ascending=True)[
        ["dataset_name", "log_loss_mean_sa", "log_loss_mean_ea", "delta_log_loss"]
    ].head()

    return {"summary": summary_df, "worst_logloss": worst, "best_logloss": best}


def summarize_geometry() -> pd.DataFrame:
    geom = _load_csv("compare_mini_tabicl_geometry.csv").copy()
    geom["delta_collapse_top1"] = geom["collapse_ea_top1"] - geom["collapse_sa_top1"]
    geom["delta_collapse_top5"] = geom["collapse_ea_top5"] - geom["collapse_sa_top5"]
    geom["delta_mean_cos"] = geom["mean_cos_ea"] - geom["mean_cos_sa"]
    geom["delta_neff"] = geom["neff_ea_mean"] - geom["neff_sa_mean"]
    geom["delta_purity_top1"] = geom["purity_ea_top1"] - geom["purity_sa_top1"]
    geom["delta_purity_topk"] = geom["purity_ea_topk"] - geom["purity_sa_topk"]

    return geom


def summarize_behaviour() -> pd.DataFrame:
    beh = _load_csv("compare_mini_tabicl_behaviour.csv")
    agg = beh.groupby("dataset").agg(
        acc_sa=("acc_sa", "mean"),
        acc_ea=("acc_ea", "mean"),
        f1_sa=("f1_sa", "mean"),
        f1_ea=("f1_ea", "mean"),
        nll_sa=("nll_sa", "mean"),
        nll_ea=("nll_ea", "mean"),
        brier_sa=("brier_sa", "mean"),
        brier_ea=("brier_ea", "mean"),
        ece_sa=("ece_sa", "mean"),
        ece_ea=("ece_ea", "mean"),
    )
    agg["delta_acc"] = agg["acc_ea"] - agg["acc_sa"]
    agg["delta_f1"] = agg["f1_ea"] - agg["f1_sa"]
    agg["delta_nll"] = agg["nll_ea"] - agg["nll_sa"]
    agg["delta_brier"] = agg["brier_ea"] - agg["brier_sa"]
    agg["delta_ece"] = agg["ece_ea"] - agg["ece_sa"]
    return agg


def summarize_robustness() -> pd.DataFrame:
    rob = _load_csv("compare_mini_tabicl_robustness.csv")
    group_cols = ["dataset", "condition", "param_type", "param_value"]
    agg = rob.groupby(group_cols).agg(
        acc_sa=("acc_sa", "mean"),
        acc_ea=("acc_ea", "mean"),
        f1_sa=("f1_sa", "mean"),
        f1_ea=("f1_ea", "mean"),
        nll_sa=("nll_sa", "mean"),
        nll_ea=("nll_ea", "mean"),
    )
    agg["delta_acc"] = agg["acc_ea"] - agg["acc_sa"]
    agg["delta_f1"] = agg["f1_ea"] - agg["f1_sa"]
    agg["delta_nll"] = agg["nll_ea"] - agg["nll_sa"]
    return agg.reset_index()


def _fraction_better(series: pd.Series, higher_is_better: bool = True) -> float:
    return (series > 0).mean() if higher_is_better else (series < 0).mean()


def print_cc18_summary():
    cc18 = summarize_cc18()
    print("\n=== CC18 summary (EA vs SA, valid datasets only) ===")
    print(cc18["summary"].to_string(index=False, formatters={
        "mean_sa": "{:.4f}".format,
        "mean_ea": "{:.4f}".format,
        "delta_mean": "{:.4f}".format,
        "delta_std": "{:.4f}".format,
        "frac_ea_better": _format_pct,
    }))

    print("\nTop 5 datasets with highest log-loss penalty (EA - SA):")
    print(cc18["worst_logloss"].to_string(index=False, formatters={
        "log_loss_mean_sa": "{:.4f}".format,
        "log_loss_mean_ea": "{:.4f}".format,
        "delta_log_loss": "{:.4f}".format,
    }))

    print("\nTop 5 datasets with best log-loss improvement (EA - SA):")
    print(cc18["best_logloss"].to_string(index=False, formatters={
        "log_loss_mean_sa": "{:.4f}".format,
        "log_loss_mean_ea": "{:.4f}".format,
        "delta_log_loss": "{:.4f}".format,
    }))


def print_geometry_summary():
    geom = summarize_geometry()
    print("\n=== Geometry deltas (EA - SA) ===")
    cols = [
        "delta_collapse_top1",
        "delta_collapse_top5",
        "delta_mean_cos",
        "delta_neff",
        "delta_purity_top1",
        "delta_purity_topk",
    ]
    mean_row = geom[cols].mean()
    print("Mean deltas:")
    print(mean_row.to_string())

    print("\nFraction of datasets where EA > SA:")
    comparisons = [
        ("collapse", "collapse_sa_top1", "collapse_ea_top1"),
        ("collapse_top5", "collapse_sa_top5", "collapse_ea_top5"),
        ("mean_cos", "mean_cos_sa", "mean_cos_ea"),
        ("neff", "neff_sa_mean", "neff_ea_mean"),
        ("purity_top1", "purity_sa_top1", "purity_ea_top1"),
        ("purity_topk", "purity_sa_topk", "purity_ea_topk"),
    ]
    for name, sa_col, ea_col in comparisons:
        frac = (geom[ea_col] > geom[sa_col]).mean()
        print(f"  {name:14s}: {_format_pct(frac)} of datasets")

    print("\nTop |Δneff| datasets (EA - SA):")
    abs_neff = geom.assign(abs_delta_neff=lambda d: d["delta_neff"].abs())
    cols_show = ["dataset", "neff_sa_mean", "neff_ea_mean", "delta_neff"]
    print(abs_neff.sort_values("delta_neff").head()[cols_show].to_string(index=False))
    print("---")
    print(abs_neff.sort_values("delta_neff", ascending=False).head()[cols_show].to_string(index=False))


def print_behaviour_summary():
    beh = summarize_behaviour()
    print("\n=== Behaviour / calibration deltas (EA - SA, averaged over context lengths) ===")
    delta_cols = ["delta_acc", "delta_f1", "delta_nll", "delta_brier", "delta_ece"]
    print("Mean deltas across datasets:")
    print(beh[delta_cols].mean().to_string())

    print("\nFraction of datasets where EA is better:")
    print(f"  acc   : {_format_pct(_fraction_better(beh['delta_acc'], higher_is_better=True))}")
    print(f"  f1    : {_format_pct(_fraction_better(beh['delta_f1'], higher_is_better=True))}")
    print(f"  nll   : {_format_pct(_fraction_better(beh['delta_nll'], higher_is_better=False))}")
    print(f"  brier : {_format_pct(_fraction_better(beh['delta_brier'], higher_is_better=False))}")
    print(f"  ece   : {_format_pct(_fraction_better(beh['delta_ece'], higher_is_better=False))}")

    print("\nWorst EA log-loss deltas:")
    cols = ["acc_sa", "acc_ea", "f1_sa", "f1_ea", "nll_sa", "nll_ea", "delta_nll", "delta_ece"]
    print(beh.sort_values("delta_nll", ascending=False).head()[cols].to_string())

    print("\nBest EA calibration deltas (ECE):")
    print(beh.sort_values("delta_ece").head()[cols].to_string())


def print_robustness_summary():
    rob = summarize_robustness()
    print("\n=== Robustness deltas by corruption type (EA - SA) ===")
    for cond, sub in rob.groupby("condition"):
        mean_delta = sub[["delta_acc", "delta_f1", "delta_nll"]].mean()
        print(f"{cond:8s} -> Δacc={mean_delta['delta_acc']:+.4f}, Δf1={mean_delta['delta_f1']:+.4f}, Δnll={mean_delta['delta_nll']:+.4f}")

    print("\nWorst and best settings by ΔNLL:")
    cols = [
        "dataset",
        "condition",
        "param_value",
        "delta_acc",
        "delta_f1",
        "delta_nll",
    ]
    print("Worst 5 (highest ΔNLL):")
    print(rob.sort_values("delta_nll", ascending=False).head()[cols].to_string(index=False))
    print("\nBest 5 (lowest ΔNLL):")
    print(rob.sort_values("delta_nll", ascending=True).head()[cols].to_string(index=False))


def print_geometry_behaviour_correlations():
    geom = summarize_geometry()
    beh = summarize_behaviour().reset_index()
    merged = geom.merge(beh, on="dataset")
    merged["delta_mean_cos"] = merged["mean_cos_ea"] - merged["mean_cos_sa"]
    merged["delta_purity_top1"] = merged["purity_ea_top1"] - merged["purity_sa_top1"]
    merged["delta_neff"] = merged["neff_ea_mean"] - merged["neff_sa_mean"]
    corr = merged[["delta_nll", "delta_f1", "delta_mean_cos", "delta_purity_top1", "delta_neff"]].corr()
    print("\n=== Correlation between geometry shifts and behaviour deltas ===")
    print(corr.to_string())


def main():
    print_cc18_summary()
    print_geometry_summary()
    print_behaviour_summary()
    print_robustness_summary()
    print_geometry_behaviour_correlations()


if __name__ == "__main__":
    main()
