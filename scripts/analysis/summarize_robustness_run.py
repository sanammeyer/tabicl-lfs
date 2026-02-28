#!/usr/bin/env python3
"""Summarize robustness-run results produced by scripts/benchmarks/tabicl_robustness_test.py.

By default, summarizes all metric columns present in metrics.csv and reports per metric:
  - Absolute metric per condition/checkpoint
  - Degradation vs clean (Δ; definition depends on metric direction)
  - Robustness gains vs SA (Δ_model - Δ_SA; negative is better)
  - Dataset-level "improved" counts vs SA (paired, averaged over seeds)

By default, this script prints a report to stdout and does not write any files.

Usage:
  python scripts/analysis/summarize_robustness_run.py --run_dir results/robustness/diag10_complete

To also write files:
  python scripts/analysis/summarize_robustness_run.py --run_dir results/robustness/diag10_complete --write_md --write_json

Outputs (when enabled):
  - run_dir/summary.md
  - run_dir/summary.json
"""

from __future__ import annotations

import argparse
import json
import numbers
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import pandas as pd


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_table(df: pd.DataFrame, *, floatfmt: str = "{:.4f}") -> str:
    df_out = df.copy()
    for c in df_out.columns:
        if pd.api.types.is_float_dtype(df_out[c]) or pd.api.types.is_integer_dtype(df_out[c]):
            df_out[c] = df_out[c].map(lambda x: floatfmt.format(x) if pd.notna(x) else "")
    return df_out.to_string(index=True)


def _assert_columns(df: pd.DataFrame, required: List[str], *, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


_NON_METRIC_COLUMNS: Tuple[str, ...] = (
    # checkpoint
    "checkpoint_name",
    "checkpoint_path",
    "checkpoint_sha256",
    # dataset
    "dataset_key",
    "dataset_name",
    "resolved_dataset_id",
    "resolved_task_id",
    "dataset_version",
    "input_spec",
    # split
    "seed",
    "test_size",
    "stratified",
    "n_train",
    "n_test",
    # condition/case
    "case_id",
    "condition",
    "param_type",
    "param_value",
    "applies_to",
    "p_cell",
    "rotation_seed",
    "rotation_cols_hash",
    # settings
    "n_estimators",
    "batch_size",
    "use_hierarchical",
    "device",
)


def _infer_metric_direction(metric: str) -> Literal["higher", "lower"]:
    """Return whether higher metric values are better, for delta sign conventions."""
    higher = {
        "acc",
        "accuracy",
        #"f1",
        "f1_macro",
        #"f1_micro",
        #"f1_weighted",
        #"precision",
        #"recall",
        #"auc",
        #"auroc",
        #"ap",
        #"r2",
    }
    lower = {
        "nll",
        #"log_loss",
        #"loss",
        "ece",
        #"brier",
        #"rmse",
        #"mae",
        #"mse",
    }
    if metric in higher:
        return "higher"
    if metric in lower:
        return "lower"
    m = metric.lower()
    if any(k in m for k in ("acc", "auc", "auroc", "precision", "recall", "f1", "r2")):
        return "higher"
    return "lower"


def _detect_metric_columns(df: pd.DataFrame) -> List[str]:
    metrics: List[str] = []
    for c in df.columns:
        if c in _NON_METRIC_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].isna().all():
            continue
        metrics.append(c)
    return metrics


def compute_delta_vs_clean(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    key_cols = ["dataset_key", "seed", "checkpoint_name"]
    direction = _infer_metric_direction(metric)
    clean = (
        df[df["condition"] == "clean"][key_cols + [metric]]
        .rename(columns={metric: f"{metric}_clean"})
        .drop_duplicates(subset=key_cols)
    )
    merged = df.merge(clean, on=key_cols, how="left")
    # Define Δ as "degradation vs clean" so that:
    #   - Δ > 0 means worse under corruption (for both higher-better and lower-better metrics).
    #   - lower is better for Δ-based summaries and robustness gains.
    if direction == "lower":
        merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_clean"]
    else:
        merged[f"delta_{metric}"] = merged[f"{metric}_clean"] - merged[metric]
    return merged


def _load_run_tables(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[pd.DataFrame]]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv: {metrics_path}")

    df = pd.read_csv(metrics_path)
    run_config = _read_json(run_dir / "run_config.json") or {}
    datasets_df: Optional[pd.DataFrame] = None
    datasets_path = run_dir / "datasets.csv"
    if datasets_path.exists():
        datasets_df = pd.read_csv(datasets_path)
    return df, run_config, datasets_df


def summarize_metric(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    _assert_columns(df, ["checkpoint_name", "dataset_key", "seed", "condition", metric], name="metrics.csv")

    merged = compute_delta_vs_clean(df, metric=metric)
    corrupt = merged[merged["condition"] != "clean"].copy()

    # Absolute metric by condition
    abs_mean = (
        df.groupby(["condition", "checkpoint_name"])[metric]
        .mean()
        .unstack("checkpoint_name")
        .sort_index()
    )

    # Mean Δ metric vs clean by condition
    delta_mean = (
        corrupt.groupby(["condition", "checkpoint_name"])[f"delta_{metric}"]
        .mean()
        .unstack("checkpoint_name")
        .sort_index()
    )

    # Δ by param
    delta_by_param = (
        corrupt.groupby(["condition", "param_type", "param_value", "checkpoint_name"])[f"delta_{metric}"]
        .mean()
        .reset_index()
        .pivot_table(
            index=["condition", "param_type", "param_value"],
            columns="checkpoint_name",
            values=f"delta_{metric}",
        )
        .sort_index()
    )

    # Overall robustness score (row-weighted): mean Δ across all corruption rows
    overall_delta = corrupt.groupby("checkpoint_name")[f"delta_{metric}"].mean().sort_values()

    # Equal-weight across corruption types (condition), using the condition-level mean deltas
    equal_weight_delta = delta_mean.mean(axis=0).sort_values()

    # Robustness gains vs SA: (Δ_model - Δ_SA) per dataset×seed×case, then mean
    sa = corrupt[corrupt["checkpoint_name"] == "SA"][
        ["dataset_key", "seed", "condition", "param_type", "param_value", f"delta_{metric}"]
    ].rename(columns={f"delta_{metric}": f"delta_{metric}_SA"})
    comp = corrupt.merge(sa, on=["dataset_key", "seed", "condition", "param_type", "param_value"], how="left")
    comp["rob_gain_vs_SA"] = comp[f"delta_{metric}"] - comp[f"delta_{metric}_SA"]
    rob_gain_cond = comp.groupby(["condition", "checkpoint_name"])["rob_gain_vs_SA"].mean().unstack("checkpoint_name").sort_index()
    rob_gain_param = (
        comp.groupby(["condition", "param_type", "param_value", "checkpoint_name"])["rob_gain_vs_SA"]
        .mean()
        .reset_index()
        .pivot_table(index=["condition", "param_type", "param_value"], columns="checkpoint_name", values="rob_gain_vs_SA")
        .sort_index()
    )

    # Dataset-level "improved" counts vs SA (avg over seeds first)
    by_ds = (
        corrupt.groupby(["dataset_key", "checkpoint_name", "condition", "param_type", "param_value"])[f"delta_{metric}"]
        .mean()
        .reset_index()
    )
    sa_ds = by_ds[by_ds["checkpoint_name"] == "SA"].rename(columns={f"delta_{metric}": "delta_sa"})[
        ["dataset_key", "condition", "param_type", "param_value", "delta_sa"]
    ]

    improved_tables: Dict[str, pd.DataFrame] = {}
    for ckpt in sorted(set(by_ds["checkpoint_name"]) - {"SA"}):
        comp_ds = (
            by_ds[by_ds["checkpoint_name"] == ckpt]
            .merge(sa_ds, on=["dataset_key", "condition", "param_type", "param_value"], how="left")
        )
        comp_ds["improved_vs_SA"] = comp_ds[f"delta_{metric}"] < comp_ds["delta_sa"]
        improved = (
            comp_ds.groupby(["condition", "param_type", "param_value"])["improved_vs_SA"]
            .agg(sum="sum", count="count")
            .sort_index()
        )
        improved_tables[ckpt] = improved

    # Coverage (missing cases) per dataset/checkpoint
    expected_cases = (
        df.groupby(["dataset_key", "seed", "checkpoint_name"])["case_id"].nunique().reset_index(name="n_cases")
    )
    coverage = expected_cases.groupby("dataset_key")["n_cases"].agg(["min", "max", "mean", "count"]).sort_values("mean")

    summary: Dict[str, Any] = {
        "metric": metric,
        "absolute_mean": abs_mean,
        "delta_mean_vs_clean": delta_mean,
        "delta_by_param": delta_by_param,
        "overall_mean_delta_rows": overall_delta,
        "equal_weight_mean_delta_conditions": equal_weight_delta,
        "robustness_gain_vs_SA_by_condition": rob_gain_cond,
        "robustness_gain_vs_SA_by_param": rob_gain_param,
        "improved_counts_vs_SA": improved_tables,
        "coverage_cases_per_dataset": coverage,
    }

    return summary


def summarize_run(run_dir: Path, metrics: Sequence[str]) -> Dict[str, Any]:
    df, run_config, datasets_df = _load_run_tables(run_dir)
    _assert_columns(df, ["checkpoint_name", "dataset_key", "seed", "condition"], name="metrics.csv")

    metrics = list(metrics)
    if not metrics:
        raise ValueError("No metrics selected.")

    per_metric: Dict[str, Any] = {}
    for m in metrics:
        per_metric[m] = summarize_metric(df, m)

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "n_rows": int(df.shape[0]),
        "n_datasets": int(df["dataset_key"].nunique()),
        "n_seeds": int(df["seed"].nunique()),
        "checkpoints": sorted(df["checkpoint_name"].unique().tolist()),
        "conditions": df["condition"].value_counts().to_dict(),
        "run_config": run_config,
        "metrics": metrics,
        "per_metric": per_metric,
    }

    if datasets_df is not None and "dataset_key" in datasets_df.columns and "dataset_name" in datasets_df.columns:
        summary["dataset_key_to_name"] = dict(zip(datasets_df["dataset_key"], datasets_df["dataset_name"]))

    return summary


def _jsonify_value(v: Any) -> Any:
    if isinstance(v, pd.DataFrame):
        return v.reset_index().to_dict(orient="records")
    if isinstance(v, pd.Series):
        return v.to_dict()
    if isinstance(v, dict) and v and all(isinstance(x, pd.DataFrame) for x in v.values()):
        return {kk: vv.reset_index().to_dict(orient="records") for kk, vv in v.items()}
    return v


def _to_py_scalar(x: Any) -> Any:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, (str, bool)):
        return x
    if isinstance(x, numbers.Integral):
        return int(x)
    if isinstance(x, numbers.Real):
        return float(x)
    if hasattr(x, "item"):
        try:
            return _to_py_scalar(x.item())
        except Exception:
            pass
    return x


def _col_key(c: Any) -> str:
    if isinstance(c, tuple):
        return "|".join(str(x) for x in c)
    return str(c)


def _df_to_nested(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for idx, row in df.iterrows():
        idx_vals: Tuple[Any, ...]
        if isinstance(idx, tuple):
            idx_vals = idx
        else:
            idx_vals = (idx,)

        cursor: Dict[str, Any] = out
        for level in idx_vals[:-1]:
            cursor = cursor.setdefault(str(level), {})
        last = str(idx_vals[-1])
        cursor[last] = {_col_key(c): _to_py_scalar(row[c]) for c in df.columns}
    return out


def _series_to_dict(s: pd.Series) -> Dict[str, Any]:
    return {str(k): _to_py_scalar(v) for k, v in s.items()}


def _llmify_metric_summary(metric: str, ms: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"metric": metric, "direction": _infer_metric_direction(metric)}
    out["absolute_mean"] = _df_to_nested(ms["absolute_mean"])
    out["delta_mean_vs_clean"] = _df_to_nested(ms["delta_mean_vs_clean"])
    out["delta_by_param"] = _df_to_nested(ms["delta_by_param"])
    out["overall_mean_delta_rows"] = _series_to_dict(ms["overall_mean_delta_rows"])
    out["equal_weight_mean_delta_conditions"] = _series_to_dict(ms["equal_weight_mean_delta_conditions"])
    out["robustness_gain_vs_SA_by_condition"] = _df_to_nested(ms["robustness_gain_vs_SA_by_condition"])
    out["robustness_gain_vs_SA_by_param"] = _df_to_nested(ms["robustness_gain_vs_SA_by_param"])
    out["improved_counts_vs_SA"] = {m: _df_to_nested(t) for m, t in ms["improved_counts_vs_SA"].items()}
    out["coverage_cases_per_dataset"] = _df_to_nested(ms["coverage_cases_per_dataset"])
    return out


def write_outputs(
    run_dir: Path,
    summary: Dict[str, Any],
    *,
    write_md: bool,
    write_json: bool,
    json_format: Literal["records", "llm"] = "llm",
) -> None:
    if write_json:
        json_path = run_dir / "summary.json"
        if json_format == "llm":
            payload: Dict[str, Any] = {
                "schema_version": 1,
                "json_format": "llm",
                "notes": {
                    "delta_definition": "Δ is 'degradation vs clean' so Δ>0 means worse and lower is better for Δ-based summaries.",
                    "robustness_gain_vs_SA": "rob_gain_vs_SA = Δ_model - Δ_SA; negative is better.",
                },
            }
            for k in (
                "run_dir",
                "n_rows",
                "n_datasets",
                "n_seeds",
                "checkpoints",
                "conditions",
                "run_config",
                "metrics",
                "dataset_key_to_name",
            ):
                if k in summary:
                    payload[k] = summary[k]
            payload["per_metric"] = {
                metric: _llmify_metric_summary(metric, summary["per_metric"][metric]) for metric in summary["metrics"]
            }
        else:
            payload = {}
            for k, v in summary.items():
                if k == "per_metric":
                    payload[k] = {m: {kk: _jsonify_value(vv) for kk, vv in ms.items()} for m, ms in v.items()}
                else:
                    payload[k] = _jsonify_value(v)
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")

    if write_md:
        md_path = run_dir / "summary.md"
        lines: List[str] = []
        lines.append("# Robustness run summary")
        lines.append("")
        lines.append(f"- run_dir: `{summary['run_dir']}`")
        lines.append(f"- rows: `{summary['n_rows']}`")
        lines.append(f"- datasets: `{summary['n_datasets']}`")
        lines.append(f"- seeds: `{summary['n_seeds']}`")
        lines.append(f"- checkpoints: `{', '.join(summary['checkpoints'])}`")
        lines.append(f"- metrics: `{', '.join(summary['metrics'])}`")
        lines.append("")

        if summary.get("run_config"):
            cfg = summary["run_config"]
            keep = ["device", "n_estimators", "batch_size", "seed", "n_seeds", "openml_ids", "outliers_apply_to", "outlier_p_cell", "outlier_factors", "uninformative_ns", "enable_label_noise", "label_poison_fracs", "enable_rotation", "rotation_k"]
            lines.append("## Run config (selected)")
            lines.append("```json")
            lines.append(json.dumps({k: cfg.get(k) for k in keep if k in cfg}, indent=2))
            lines.append("```")
            lines.append("")

        lines.append("## Condition counts")
        lines.append("```")
        for k, v in summary["conditions"].items():
            lines.append(f"{k}: {v}")
        lines.append("```")
        lines.append("")

        for metric in summary["metrics"]:
            ms = summary["per_metric"][metric]
            direction = _infer_metric_direction(metric)

            lines.append(f"## Metric: `{metric}`")
            lines.append(f"- direction: `{'higher is better' if direction == 'higher' else 'lower is better'}`")
            lines.append("")

            lines.append("### Absolute mean")
            lines.append("```")
            lines.append(_fmt_table(ms["absolute_mean"]))
            lines.append("```")
            lines.append("")

            lines.append("### Mean degradation vs clean (Δ; lower is better)")
            lines.append("```")
            lines.append(_fmt_table(ms["delta_mean_vs_clean"]))
            lines.append("```")
            lines.append("")

            lines.append("### Overall mean Δ across all corruption rows (lower is better)")
            lines.append("```")
            for ckpt, v in ms["overall_mean_delta_rows"].items():
                lines.append(f"{ckpt}: {float(v):.5f}")
            lines.append("```")
            lines.append("")

            lines.append("### Equal-weight mean Δ across corruption types (condition means; lower is better)")
            lines.append("```")
            for ckpt, v in ms["equal_weight_mean_delta_conditions"].items():
                lines.append(f"{ckpt}: {float(v):.5f}")
            lines.append("```")
            lines.append("")

            lines.append("### Robustness gain vs SA (mean of Δ_model - Δ_SA; negative is better)")
            lines.append("```")
            lines.append(_fmt_table(ms["robustness_gain_vs_SA_by_condition"]))
            lines.append("```")
            lines.append("")

            lines.append("### Δ by condition/param (mean; lower is better)")
            lines.append("```")
            lines.append(_fmt_table(ms["delta_by_param"]))
            lines.append("```")
            lines.append("")

            lines.append("### Improved datasets vs SA (count over datasets; avg over seeds first)")
            for ckpt, t in ms["improved_counts_vs_SA"].items():
                lines.append("")
                lines.append(f"#### {ckpt}")
                lines.append("```")
                lines.append(t.to_string())
                lines.append("```")

            lines.append("")
            lines.append("### Case coverage per dataset (n_cases per dataset×seed×checkpoint)")
            lines.append("```")
            lines.append(ms["coverage_cases_per_dataset"].to_string())
            lines.append("```")
            lines.append("")

        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_report(summary: Dict[str, Any]) -> None:
    print("=== Robustness run summary ===")
    print(f"run_dir: {summary['run_dir']}")
    print(f"rows: {summary['n_rows']} | datasets: {summary['n_datasets']} | seeds: {summary['n_seeds']}")
    print(f"checkpoints: {', '.join(summary['checkpoints'])}")
    print(f"metrics: {', '.join(summary['metrics'])}")

    if summary.get("run_config"):
        cfg = summary["run_config"]
        keep = [
            "device",
            "n_estimators",
            "batch_size",
            "seed",
            "n_seeds",
            "openml_ids",
            "outliers_apply_to",
            "outlier_p_cell",
            "outlier_factors",
            "uninformative_ns",
            "enable_label_noise",
            "label_poison_fracs",
            "enable_rotation",
            "rotation_k",
        ]
        print("\n--- Run config (selected) ---")
        print(json.dumps({k: cfg.get(k) for k in keep if k in cfg}, indent=2))

    print("\n--- Condition counts ---")
    for k, v in summary["conditions"].items():
        print(f"{k}: {v}")

    for metric in summary["metrics"]:
        ms = summary["per_metric"][metric]
        direction = _infer_metric_direction(metric)
        print(f"\n=== Metric: {metric} ({'higher is better' if direction == 'higher' else 'lower is better'}) ===")

        print("\n--- Absolute mean ---")
        print(_fmt_table(ms["absolute_mean"]))

        print("\n--- Mean degradation vs clean (Δ; lower is better) ---")
        print(_fmt_table(ms["delta_mean_vs_clean"]))

        print("\n--- Overall mean Δ across all corruption rows (lower is better) ---")
        for ckpt, v in ms["overall_mean_delta_rows"].items():
            print(f"{ckpt}: {float(v):.6f}")

        print("\n--- Equal-weight mean Δ across corruption types (condition means; lower is better) ---")
        for ckpt, v in ms["equal_weight_mean_delta_conditions"].items():
            print(f"{ckpt}: {float(v):.6f}")

        print("\n--- Robustness gain vs SA (mean of Δ_model - Δ_SA; negative is better) ---")
        print(_fmt_table(ms["robustness_gain_vs_SA_by_condition"]))

        print("\n--- Δ by condition/param (mean; lower is better) ---")
        print(_fmt_table(ms["delta_by_param"]))

        print("\n--- Improved datasets vs SA (count over datasets; avg over seeds first) ---")
        for ckpt, t in ms["improved_counts_vs_SA"].items():
            print(f"\n[{ckpt}]")
            print(t.to_string())

        print("\n--- Case coverage per dataset (n_cases per dataset×seed×checkpoint) ---")
        print(ms["coverage_cases_per_dataset"].to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a tabicl_robustness_test.py run directory.")
    parser.add_argument("--run_dir", type=Path, required=True, help="Run directory containing metrics.csv.")
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Single metric column to summarize (deprecated; use --metrics). If omitted, summarizes all metrics.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="One or more metric columns to summarize. If omitted, summarizes all metrics.",
    )
    parser.add_argument(
        "--write_md",
        action="store_true",
        help="Write a Markdown report to run_dir/summary.md.",
    )
    parser.add_argument(
        "--write_json",
        action="store_true",
        help="Write a JSON summary to run_dir/summary.json.",
    )
    parser.add_argument(
        "--json_format",
        choices=["llm", "records"],
        default="llm",
        help="JSON format for summary.json (default: llm; use records for legacy list-of-records).",
    )
    args = parser.parse_args()

    if args.metric is not None and args.metrics is not None:
        raise SystemExit("Use only one of --metric or --metrics.")

    df, _, _ = _load_run_tables(args.run_dir)
    if args.metric is not None:
        metrics = [args.metric]
    elif args.metrics is not None:
        metrics = list(args.metrics)
    else:
        metrics = _detect_metric_columns(df)
        if not metrics:
            raise SystemExit("Could not detect metric columns in metrics.csv; pass --metrics explicitly.")

    summary = summarize_run(args.run_dir, metrics=metrics)
    print_report(summary)

    if args.write_md or args.write_json:
        write_outputs(
            args.run_dir,
            summary,
            write_md=bool(args.write_md),
            write_json=bool(args.write_json),
            json_format=args.json_format,
        )
        if args.write_md:
            print(f"\n[wrote] {args.run_dir / 'summary.md'}")
        if args.write_json:
            print(f"[wrote] {args.run_dir / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        raise SystemExit(0)
