#!/usr/bin/env python3
"""
Export a robustness+diagnostics run directory into a single LLM-friendly JSON file.

This script is intended to sit on top of the auditable run directories produced by:
  - scripts/benchmarks/tabicl_robustness_test.py

It creates one JSON file (default: run_dir/_llm_summary.json) that:
  - embeds run metadata (config/env/checkpoints/datasets)
  - groups results by dataset -> seed -> case -> models
  - adds per-metric "degradation vs clean" deltas with consistent sign
    (delta > 0 means worse under corruption for both higher- and lower-better metrics)
  - optionally adds robustness gains vs SA (delta_model - delta_SA)

For token-efficient LLM analysis, use `--mode compact_agg`, which:
  - drops most metadata
  - keeps only a small metric set
  - aggregates across datasets/seeds per (case, severity, scope, checkpoint_name)
  - minifies JSON and rounds floats

For analytics-first processing (pandas-ready, no nesting), use `--mode records`,
which writes:
  - one metadata block with indices
  - one flat `records` array (one per (dataset, seed, case, model))
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


SCHEMA_VERSION = 2


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return None
        return xf
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    return str(x)


def _round_floats(obj: Any, *, decimals: int) -> Any:
    if decimals < 0:
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, decimals)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, decimals)
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals=decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, decimals=decimals) for v in obj]
    return obj


def _sha256_json(obj: Any) -> str:
    txt = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(txt.encode("utf-8")).hexdigest()


def _infer_metric_kind(metric: str) -> str:
    m = metric.lower()
    if m in {"acc", "accuracy", "f1", "f1_macro", "f1_micro", "f1_weighted"}:
        return "perf"
    if m in {"nll", "log_loss", "loss"}:
        return "perf"
    if m in {"ece", "brier"}:
        return "calibration"
    if m.startswith("collapse") or m in {"d_eff"}:
        return "collapse"
    if m.startswith("cos_") or m.startswith("neff") or m.startswith("purity"):
        return "geometry"
    return "other"


def _infer_case_family(case: str, param_type: Optional[str] = None) -> str:
    c = (case or "").lower()
    if c == "clean":
        return "clean"
    for prefix in (
        "outliers_",
        "label_noise_",
        "uninformative_",
        "uninformative_features_",
        "missing_",
        "rotation_",
        "permute_",
        "shuffle_",
        "scale_",
        "shift_",
        "flip_",
    ):
        if c.startswith(prefix):
            return prefix.rstrip("_")
    pt = (param_type or "").lower()
    if pt:
        return pt
    return "other"


def _metric_direction(metric: str) -> Literal["higher", "lower"]:
    higher = {
        "acc",
        "accuracy",
        "f1",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "d_eff",
        "neff_mean",
        "purity_top1",
        "purity_top5",
    }
    lower = {
        "nll",
        "log_loss",
        "loss",
        "ece",
        "brier",
        "collapse_top1",
        "cos_mean",
        "cos_p95",
        "neff_std",
    }
    if metric in higher:
        return "higher"
    if metric in lower:
        return "lower"
    m = metric.lower()
    if any(k in m for k in ("acc", "auc", "auroc", "precision", "recall", "f1", "purity", "neff", "d_eff")):
        return "higher"
    return "lower"


def _delta_degradation(value: float, clean: float, *, direction: Literal["higher", "lower"]) -> float:
    # delta > 0 means worse under corruption
    if direction == "lower":
        return float(value - clean)
    return float(clean - value)


def _detect_metric_columns(df: pd.DataFrame) -> List[str]:
    non_metrics = {
        "checkpoint_name",
        "checkpoint_path",
        "checkpoint_sha256",
        "variant",
        "checkpoint",
        "dataset",
        "dataset_key",
        "dataset_name",
        "resolved_dataset_id",
        "resolved_task_id",
        "dataset_version",
        "input_spec",
        "seed",
        "test_size",
        "stratified",
        "n_train",
        "n_test",
        "case_id",
        "case",
        "severity",
        "scope",
        "condition",
        "param_type",
        "param_value",
        "applies_to",
        "p_cell",
        "rotation_seed",
        "rotation_cols_hash",
        "n_estimators",
        "batch_size",
        "use_hierarchical",
        "device",
        "error",
    }
    metrics: List[str] = []
    for c in df.columns:
        if c in non_metrics:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].isna().all():
            continue
        metrics.append(c)
    return metrics


def _load_tables(
    run_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv: {metrics_path}")
    df = pd.read_csv(metrics_path)

    run_config = _read_json(run_dir / "run_config.json") or {}
    env = _read_json(run_dir / "env.json") or {}
    checkpoints = _read_json(run_dir / "checkpoints.json") or []
    splits_meta = _read_json(run_dir / "splits_meta.json") or []

    datasets_df = None
    datasets_path = run_dir / "datasets.csv"
    if datasets_path.exists():
        datasets_df = pd.read_csv(datasets_path)
    datasets_rows = datasets_df.to_dict(orient="records") if datasets_df is not None else []

    return df, run_config, env, checkpoints, splits_meta, datasets_rows


def _split_meta_index(splits_meta: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in splits_meta:
        ds_key = str(r.get("dataset_key", ""))
        seed = int(r.get("seed", -1))
        if ds_key and seed >= 0:
            out[(ds_key, seed)] = r
    return out


def build_llm_summary(run_dir: Path, *, include_gains_vs_sa: bool = True) -> Dict[str, Any]:
    df, run_config, env, checkpoints, splits_meta, datasets_rows = _load_tables(run_dir)
    for col in ["dataset_key", "seed", "checkpoint_name", "case"]:
        if col not in df.columns:
            raise ValueError(f"metrics.csv missing required column: {col}")

    metric_cols = _detect_metric_columns(df)
    # Common expected metrics (keep stable ordering; add whatever exists)
    preferred = [
        "acc",
        "f1_macro",
        "nll",
        "ece",
        "brier",
        "collapse_top1",
        "d_eff",
        "cos_mean",
        "cos_p95",
        "neff_mean",
        "neff_std",
        "purity_top1",
        "purity_top5",
    ]
    metrics = [m for m in preferred if m in df.columns] + [m for m in metric_cols if m not in preferred]
    directions = {m: _metric_direction(m) for m in metrics}

    # Clean baselines for deltas
    clean = df[df["case"] == "clean"].copy()
    clean_key = ["dataset_key", "seed", "checkpoint_name"]
    clean_vals = clean[clean_key + metrics].drop_duplicates(subset=clean_key).set_index(clean_key)

    def _row_clean_lookup(row: pd.Series) -> Optional[pd.Series]:
        k = (row["dataset_key"], int(row["seed"]), row["checkpoint_name"])
        try:
            return clean_vals.loc[k]
        except Exception:
            return None

    # Split meta index
    smi = _split_meta_index(splits_meta if isinstance(splits_meta, list) else [])

    # Dataset info index
    ds_info = {str(r.get("dataset_key")): r for r in datasets_rows if "dataset_key" in r}

    # Build nested structure
    checkpoints_list = sorted(df["checkpoint_name"].unique().tolist())
    cases_list = sorted(df["case"].unique().tolist())
    dataset_keys = sorted(df["dataset_key"].unique().tolist())
    seeds = sorted({int(s) for s in df["seed"].unique().tolist()})

    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "files": {
            "metrics_csv": str((run_dir / "metrics.csv").resolve()),
            "run_config_json": str((run_dir / "run_config.json").resolve()),
            "env_json": str((run_dir / "env.json").resolve()),
            "checkpoints_json": str((run_dir / "checkpoints.json").resolve()),
            "datasets_csv": str((run_dir / "datasets.csv").resolve()),
            "splits_meta_json": str((run_dir / "splits_meta.json").resolve()),
        },
        "notes": {
            "delta_definition": "degradation_vs_clean is defined so delta>0 means worse under corruption for both higher- and lower-better metrics.",
            "gains_vs_sa_definition": "robustness_gain_vs_SA = degradation_vs_clean(model) - degradation_vs_clean(SA); negative is better.",
            "missing_values": "NaN/inf are exported as null.",
        },
        "run_config": run_config,
        "env": env,
        "checkpoints": checkpoints,
        "index": {
            "checkpoints": checkpoints_list,
            "cases": cases_list,
            "datasets": dataset_keys,
            "seeds": seeds,
            "metrics": [{"name": m, "direction": directions[m]} for m in metrics],
        },
        "datasets": [],
    }

    # Precompute SA deltas for gains
    sa_name = "SA" if "SA" in checkpoints_list else None
    sa_delta_index: Dict[Tuple[str, int, str, float, str], Dict[str, float]] = {}
    if include_gains_vs_sa and sa_name is not None:
        sa_rows = df[df["checkpoint_name"] == sa_name]
        for _, r in sa_rows.iterrows():
            clean_r = _row_clean_lookup(r)
            if clean_r is None:
                continue
            case = str(r["case"])
            sev = float(r.get("severity", r.get("param_value", 0.0)))
            key = (str(r["dataset_key"]), int(r["seed"]), case, sev, str(r.get("case_id", "")))
            d: Dict[str, float] = {}
            for m in metrics:
                v = float(r[m]) if pd.notna(r[m]) else float("nan")
                c = float(clean_r[m]) if pd.notna(clean_r[m]) else float("nan")
                if not (np.isfinite(v) and np.isfinite(c)):
                    d[m] = float("nan")
                else:
                    d[m] = _delta_degradation(v, c, direction=directions[m])
            sa_delta_index[key] = d

    # Dataset -> seed -> case groups
    group_key = ["dataset_key", "seed", "case", "severity", "case_id"]
    for ds_key in dataset_keys:
        ds_block: Dict[str, Any] = {
            "dataset_key": ds_key,
            "dataset_name": str(df[df["dataset_key"] == ds_key]["dataset_name"].iloc[0]) if "dataset_name" in df.columns else None,
            "meta": _to_jsonable(ds_info.get(ds_key, {})),
            "seeds": [],
        }

        df_ds = df[df["dataset_key"] == ds_key]
        for seed in sorted({int(s) for s in df_ds["seed"].unique().tolist()}):
            df_dss = df_ds[df_ds["seed"] == seed]
            seed_block: Dict[str, Any] = {
                "seed": int(seed),
                "split_meta": _to_jsonable(smi.get((ds_key, int(seed)), {})),
                "cases": [],
            }

            # Ensure severity exists even if older runs don't have it
            if "severity" not in df_dss.columns:
                df_dss = df_dss.copy()
                df_dss["severity"] = df_dss.get("param_value", 0.0)
            if "case_id" not in df_dss.columns:
                df_dss = df_dss.copy()
                df_dss["case_id"] = ""

            for (case, sev, case_id), g in df_dss.groupby(["case", "severity", "case_id"], dropna=False):
                # Use first row for shared metadata about the case
                r0 = g.iloc[0]
                case_rec: Dict[str, Any] = {
                    "case": str(case),
                    "severity": _to_jsonable(float(sev) if pd.notna(sev) else None),
                    "case_id": str(case_id),
                    "scope": str(r0.get("scope", "")) if "scope" in g.columns else None,
                    "condition": str(r0.get("condition", "")) if "condition" in g.columns else None,
                    "param_type": str(r0.get("param_type", "")) if "param_type" in g.columns else None,
                    "param_value": _to_jsonable(float(r0.get("param_value")) if pd.notna(r0.get("param_value")) else None),
                    "applies_to": str(r0.get("applies_to", "")) if "applies_to" in g.columns else None,
                    "p_cell": _to_jsonable(float(r0.get("p_cell")) if pd.notna(r0.get("p_cell")) else None),
                    "rotation_seed": _to_jsonable(int(r0.get("rotation_seed")) if pd.notna(r0.get("rotation_seed")) else None),
                    "rotation_cols_hash": str(r0.get("rotation_cols_hash")) if pd.notna(r0.get("rotation_cols_hash")) else None,
                    "models": {},
                }

                for _, r in g.iterrows():
                    ck = str(r["checkpoint_name"])
                    clean_r = _row_clean_lookup(r)
                    metrics_val = {m: _to_jsonable(r.get(m)) for m in metrics}

                    deg: Dict[str, Any] = {}
                    if clean_r is not None:
                        for m in metrics:
                            v = float(r[m]) if pd.notna(r[m]) else float("nan")
                            c = float(clean_r[m]) if pd.notna(clean_r[m]) else float("nan")
                            if np.isfinite(v) and np.isfinite(c):
                                deg[m] = _to_jsonable(_delta_degradation(v, c, direction=directions[m]))
                            else:
                                deg[m] = None
                    else:
                        deg = {m: None for m in metrics}

                    gain_vs_sa: Dict[str, Any] = {}
                    if include_gains_vs_sa and sa_name is not None and ck != sa_name:
                        sa_key = (ds_key, int(seed), str(case), float(sev) if pd.notna(sev) else 0.0, str(case_id))
                        sa_deg = sa_delta_index.get(sa_key)
                        if sa_deg is not None:
                            for m in metrics:
                                dm = deg.get(m)
                                dsav = sa_deg.get(m, float("nan"))
                                if dm is None or not np.isfinite(dsav):
                                    gain_vs_sa[m] = None
                                else:
                                    gain_vs_sa[m] = _to_jsonable(float(dm) - float(dsav))
                        else:
                            gain_vs_sa = {m: None for m in metrics}

                    case_rec["models"][ck] = {
                        "checkpoint_name": ck,
                        "variant": str(r.get("variant", "")) if "variant" in g.columns else None,
                        "checkpoint_path": str(r.get("checkpoint_path", "")) if "checkpoint_path" in g.columns else None,
                        "checkpoint_sha256": str(r.get("checkpoint_sha256", "")) if "checkpoint_sha256" in g.columns else None,
                        "metrics": metrics_val,
                        "degradation_vs_clean": deg,
                        **({"robustness_gain_vs_SA": gain_vs_sa} if gain_vs_sa else {}),
                        "error": str(r.get("error")) if "error" in g.columns and pd.notna(r.get("error")) else None,
                    }

                seed_block["cases"].append(case_rec)

            ds_block["seeds"].append(seed_block)

        out["datasets"].append(ds_block)

    return out


def build_compact_llm_summary(
    run_dir: Path,
    *,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    df, _, _, _, _, _ = _load_tables(run_dir)
    required = ["dataset_key", "seed", "checkpoint_name", "case"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"metrics.csv missing required column: {col}")

    # Default: small but useful set; keep only those present and numeric.
    default_metrics = [
        "acc",
        "f1_macro",
        "nll",
        "ece",
        "brier",
        "collapse_top1",
        "d_eff",
        "cos_p95",
        "neff_mean",
        "purity_top5",
    ]
    use_metrics = metrics or default_metrics
    use_metrics = [
        m
        for m in use_metrics
        if m in df.columns and pd.api.types.is_numeric_dtype(df[m]) and not df[m].isna().all()
    ]
    directions = {m: _metric_direction(m) for m in use_metrics}

    # Ensure severity/scope exist (older runs may not have them).
    df2 = df.copy()
    if "severity" not in df2.columns:
        df2["severity"] = df2.get("param_value", 0.0)
    if "scope" not in df2.columns:
        df2["scope"] = ""

    clean = df2[df2["case"] == "clean"][["dataset_key", "seed", "checkpoint_name"] + use_metrics].drop_duplicates(
        subset=["dataset_key", "seed", "checkpoint_name"]
    )
    clean_idx = clean.set_index(["dataset_key", "seed", "checkpoint_name"])

    # Compute degradation deltas vs clean row-wise.
    delta_rows: List[Dict[str, Any]] = []
    for _, r in df2.iterrows():
        key = (str(r["dataset_key"]), int(r["seed"]), str(r["checkpoint_name"]))
        clean_r = None
        if key in clean_idx.index:
            clean_r = clean_idx.loc[key]

        d: Dict[str, Any] = {}
        for m in use_metrics:
            v = r.get(m, np.nan)
            if clean_r is None:
                d[m] = None
                continue
            c = clean_r.get(m, np.nan)
            if pd.isna(v) or pd.isna(c):
                d[m] = None
            else:
                d[m] = _delta_degradation(float(v), float(c), direction=directions[m])

        delta_rows.append(
            {
                "case": str(r["case"]),
                "severity": None if pd.isna(r.get("severity")) else float(r.get("severity")),
                "scope": str(r.get("scope", "")),
                "checkpoint_name": str(r["checkpoint_name"]),
                **{m: _to_jsonable(r.get(m)) for m in use_metrics},
                **{f"delta_{m}": _to_jsonable(d[m]) for m in use_metrics},
            }
        )

    dd = pd.DataFrame(delta_rows)

    # Aggregate per case/severity/scope/checkpoint.
    group_cols = ["case", "severity", "scope", "checkpoint_name"]
    cases_out: List[Dict[str, Any]] = []
    for (case, sev, scope, ckpt), g in dd.groupby(group_cols, dropna=False):
        rec: Dict[str, Any] = {
            "case": str(case),
            "severity": None if pd.isna(sev) else float(sev),
            "scope": str(scope),
            "checkpoint_name": str(ckpt),
            "n": int(g.shape[0]),
        }
        for m in use_metrics:
            sv = pd.to_numeric(g[m], errors="coerce")
            sd = pd.to_numeric(g[f"delta_{m}"], errors="coerce")
            if not sv.isna().all():
                rec[f"{m}_mean"] = float(sv.mean(skipna=True))
            if not sd.isna().all():
                rec[f"delta_{m}_mean"] = float(sd.mean(skipna=True))
                rec[f"delta_{m}_std"] = float(sd.std(skipna=True, ddof=0))
        cases_out.append(rec)

    # Clean aggregates per checkpoint.
    clean_out: List[Dict[str, Any]] = []
    dd_clean = dd[dd["case"] == "clean"]
    if dd_clean.shape[0]:
        for ckpt, g in dd_clean.groupby(["checkpoint_name"], dropna=False):
            rec: Dict[str, Any] = {"checkpoint_name": str(ckpt), "n": int(g.shape[0])}
            for m in use_metrics:
                s = pd.to_numeric(g[m], errors="coerce")
                if not s.isna().all():
                    rec[f"{m}_mean"] = float(s.mean(skipna=True))
                    rec[f"{m}_std"] = float(s.std(skipna=True, ddof=0))
            clean_out.append(rec)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "mode": "compact_agg",
        "metrics": [{"name": m, "direction": directions[m], "kind": _infer_metric_kind(m)} for m in use_metrics],
        "clean": clean_out,
        "cases": cases_out,
    }


def build_records_llm_summary(
    run_dir: Path,
    *,
    include_gains_vs_sa: bool = True,
    clean_case_key: str = "clean",
    metric_names: Optional[List[str]] = None,
    metrics_format: Literal["array", "dict"] = "array",
) -> Dict[str, Any]:
    df, run_config, env, checkpoints, splits_meta, datasets_rows = _load_tables(run_dir)
    for col in ["dataset_key", "seed", "checkpoint_name", "case"]:
        if col not in df.columns:
            raise ValueError(f"metrics.csv missing required column: {col}")

    # Ensure case parameter columns exist for uniform export
    case_cols = [
        "severity",
        "scope",
        "condition",
        "param_type",
        "param_value",
        "applies_to",
        "p_cell",
        "rotation_seed",
        "rotation_cols_hash",
        "case_id",
    ]
    for c in case_cols:
        if c not in df.columns:
            df[c] = None

    metric_cols = _detect_metric_columns(df)
    # Keep a stable, useful ordering if available.
    preferred = [
        "acc",
        "f1_macro",
        "nll",
        "ece",
        "brier",
        "collapse_top1",
        "d_eff",
        "cos_mean",
        "cos_p95",
        "neff_mean",
        "neff_std",
        "purity_top1",
        "purity_top5",
    ]
    base = [m for m in preferred if m in metric_cols] + [m for m in metric_cols if m not in preferred]
    if metric_names:
        metrics = [m for m in base if m in set(metric_names)]
    else:
        metrics = base
    directions = {m: _metric_direction(m) for m in metrics}

    # Indices
    metrics_index = [{"name": m, "direction": directions[m], "kind": _infer_metric_kind(m)} for m in metrics]

    smi = _split_meta_index(splits_meta if isinstance(splits_meta, list) else [])
    splits_index = [
        {
            "dataset_key": ds_key,
            "seed": int(seed),
            "n_train": _to_jsonable(r.get("n_train")),
            "n_test": _to_jsonable(r.get("n_test")),
            "test_size": _to_jsonable(r.get("test_size")),
            "stratified": _to_jsonable(r.get("stratified")),
        }
        for (ds_key, seed), r in sorted(smi.items(), key=lambda x: (x[0][0], x[0][1]))
    ]

    ds_info = {str(r.get("dataset_key")): r for r in datasets_rows if "dataset_key" in r}
    datasets_index = []
    for ds_key in sorted(df["dataset_key"].unique().tolist()):
        r = ds_info.get(str(ds_key), {})
        datasets_index.append(
            {
                "dataset_key": str(ds_key),
                "dataset_name": _to_jsonable(r.get("dataset_name"))
                if r
                else (
                    _to_jsonable(df[df["dataset_key"] == ds_key]["dataset_name"].iloc[0])
                    if "dataset_name" in df.columns
                    else None
                ),
                "resolved_dataset_id": _to_jsonable(
                    r.get("resolved_dataset_id")
                    or r.get("openml_id")
                    or (
                        df[df["dataset_key"] == ds_key]["resolved_dataset_id"].iloc[0]
                        if "resolved_dataset_id" in df.columns
                        else None
                    )
                ),
                "resolved_task_id": _to_jsonable(
                    r.get("resolved_task_id")
                    or (
                        df[df["dataset_key"] == ds_key]["resolved_task_id"].iloc[0]
                        if "resolved_task_id" in df.columns
                        else None
                    )
                ),
                "dataset_version": _to_jsonable(
                    r.get("dataset_version")
                    or (
                        df[df["dataset_key"] == ds_key]["dataset_version"].iloc[0]
                        if "dataset_version" in df.columns
                        else None
                    )
                ),
            }
        )

    # Cases index
    cases_index: List[Dict[str, Any]] = []
    seen = set()
    for _, r in df[["case"] + case_cols].drop_duplicates().iterrows():
        key = tuple(r.get(c) for c in (["case"] + case_cols))
        if key in seen:
            continue
        seen.add(key)
        param_type = r.get("param_type")
        cases_index.append(
            {
                "case": str(r.get("case")),
                "family": _infer_case_family(str(r.get("case")), str(param_type) if pd.notna(param_type) else None),
                "scope": _to_jsonable(r.get("scope")),
                "severity": _to_jsonable(float(r.get("severity")) if pd.notna(r.get("severity")) else None),
                "param_type": _to_jsonable(param_type),
                "param_value": _to_jsonable(float(r.get("param_value")) if pd.notna(r.get("param_value")) else None),
                "applies_to": _to_jsonable(r.get("applies_to")),
                "p_cell": _to_jsonable(float(r.get("p_cell")) if pd.notna(r.get("p_cell")) else None),
                "rotation_seed": _to_jsonable(int(r.get("rotation_seed")) if pd.notna(r.get("rotation_seed")) else None),
                "rotation_cols_hash": _to_jsonable(r.get("rotation_cols_hash")),
                "condition": _to_jsonable(r.get("condition")),
                "case_id": _to_jsonable(r.get("case_id")),
            }
        )

    # Clean baselines (per dataset_key, seed, checkpoint_name)
    clean = df[df["case"] == clean_case_key].copy()
    clean_key = ["dataset_key", "seed", "checkpoint_name"]
    clean_vals = clean[clean_key + metrics].drop_duplicates(subset=clean_key).set_index(clean_key)

    def _row_clean_lookup(row: pd.Series) -> Optional[pd.Series]:
        k = (row["dataset_key"], int(row["seed"]), row["checkpoint_name"])
        try:
            return clean_vals.loc[k]
        except Exception:
            return None

    # Precompute SA deltas for gains
    checkpoints_list = sorted(df["checkpoint_name"].unique().tolist())
    sa_name = "SA" if "SA" in checkpoints_list else None
    sa_delta_index: Dict[Tuple[str, int, str, float, str], Dict[str, float]] = {}
    if include_gains_vs_sa and sa_name is not None:
        sa_rows = df[df["checkpoint_name"] == sa_name]
        for _, r in sa_rows.iterrows():
            clean_r = _row_clean_lookup(r)
            if clean_r is None:
                continue
            case = str(r["case"])
            sev = float(r.get("severity", r.get("param_value", 0.0)) or 0.0)
            case_id = str(r.get("case_id", "") or "")
            key = (str(r["dataset_key"]), int(r["seed"]), case, sev, case_id)
            d: Dict[str, float] = {}
            for m in metrics:
                v = float(r[m]) if pd.notna(r[m]) else float("nan")
                c = float(clean_r[m]) if pd.notna(clean_r[m]) else float("nan")
                if not (np.isfinite(v) and np.isfinite(c)):
                    continue
                d[m] = _delta_degradation(v, c, direction=directions[m])
            sa_delta_index[key] = d

    # Checkpoints index + run_id
    ckpt_id: List[Dict[str, Any]] = []
    if isinstance(checkpoints, list) and checkpoints:
        for c in checkpoints:
            ckpt_id.append(
                {
                    "name": c.get("checkpoint_name") or c.get("name"),
                    "path": c.get("checkpoint_path") or c.get("path"),
                    "sha256": c.get("checkpoint_sha256") or c.get("sha256"),
                }
            )
    else:
        ckpt_id = (
            df[["checkpoint_name", "checkpoint_path", "checkpoint_sha256"]]
            .drop_duplicates()
            .fillna("")
            .rename(columns={"checkpoint_name": "name", "checkpoint_path": "path", "checkpoint_sha256": "sha256"})
            .to_dict(orient="records")
        )
    run_id = _sha256_json({"run_config": run_config, "checkpoints": ckpt_id})

    # Records (one per row). To reduce tokens, metrics are arrays aligned to `metrics_index` by default.
    records: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        ds_key = str(r.get("dataset_key"))
        seed = int(r.get("seed"))
        case = str(r.get("case"))
        case_id = str(r.get("case_id", "") or "")
        sev_raw = r.get("severity", r.get("param_value", None))
        severity = None if pd.isna(sev_raw) else float(sev_raw)
        scope = str(r.get("scope", "")) if pd.notna(r.get("scope", "")) else ""
        param_type = r.get("param_type", None)
        family = _infer_case_family(case, str(param_type) if pd.notna(param_type) else None)

        ck = str(r.get("checkpoint_name"))
        clean_r = _row_clean_lookup(r)

        if metrics_format == "dict":
            metrics_val: Any = {}
            degradation: Any = {}
            for m in metrics:
                v = r.get(m)
                if pd.isna(v):
                    continue
                metrics_val[m] = _to_jsonable(v)
                if clean_r is not None:
                    cv = clean_r.get(m, np.nan)
                    if pd.notna(cv):
                        degradation[m] = _to_jsonable(
                            _delta_degradation(float(v), float(cv), direction=directions[m])
                        )
        else:
            metrics_val = []
            degradation = []
            for m in metrics:
                v = r.get(m)
                metrics_val.append(None if pd.isna(v) else _to_jsonable(v))
                if clean_r is None:
                    degradation.append(None)
                else:
                    cv = clean_r.get(m, np.nan)
                    if pd.isna(v) or pd.isna(cv):
                        degradation.append(None)
                    else:
                        degradation.append(
                            _to_jsonable(_delta_degradation(float(v), float(cv), direction=directions[m]))
                        )

        gain_vs_sa: Any = {} if metrics_format == "dict" else []
        if include_gains_vs_sa and sa_name is not None and ck != sa_name:
            sa_key = (ds_key, seed, case, float(severity or 0.0), case_id)
            sa_deg = sa_delta_index.get(sa_key)
            if sa_deg is not None:
                if metrics_format == "dict":
                    if isinstance(degradation, dict):
                        for m, dm in degradation.items():
                            dsav = sa_deg.get(m)
                            if dsav is None:
                                continue
                            gain_vs_sa[m] = _to_jsonable(float(dm) - float(dsav))
                else:
                    # array aligned to metrics
                    for i, m in enumerate(metrics):
                        dm = degradation[i] if i < len(degradation) else None
                        dsav = sa_deg.get(m)
                        if dm is None or dsav is None:
                            gain_vs_sa.append(None)
                        else:
                            gain_vs_sa.append(_to_jsonable(float(dm) - float(dsav)))

        err = r.get("error") if "error" in df.columns else None
        status = "error" if (pd.notna(err) and str(err).strip()) else "ok"
        error_obj = None
        if status == "error":
            msg = str(err)
            error_obj = {"message": msg, "traceback_short": msg.splitlines()[-1] if msg.splitlines() else msg}

        rec: Dict[str, Any] = {
            "dataset_key": ds_key,
            "seed": seed,
            "split_id": f"{ds_key}__seed{seed}",
            "case": case,
            "case_id": case_id or None,
            "corruption_family": family,
            "scope": scope or None,
            "severity": severity,
            "param_type": _to_jsonable(param_type),
            "param_value": _to_jsonable(float(r.get("param_value")) if pd.notna(r.get("param_value")) else None),
            "applies_to": _to_jsonable(r.get("applies_to")),
            "p_cell": _to_jsonable(float(r.get("p_cell")) if pd.notna(r.get("p_cell")) else None),
            "rotation_seed": _to_jsonable(int(r.get("rotation_seed")) if pd.notna(r.get("rotation_seed")) else None),
            "rotation_cols_hash": _to_jsonable(r.get("rotation_cols_hash")),
            "condition": _to_jsonable(r.get("condition")),
            "model": ck,
            "checkpoint_sha256": _to_jsonable(r.get("checkpoint_sha256")),
            "metrics": metrics_val,
            "degradation_vs_clean": degradation,
            **({"robustness_gain_vs_SA": gain_vs_sa} if (metrics_format == "dict" and gain_vs_sa) else {}),
            **({"robustness_gain_vs_SA": gain_vs_sa} if (metrics_format == "array" and include_gains_vs_sa and sa_name is not None and ck != sa_name) else {}),
            "status": status,
            "error": error_obj,
        }
        records.append(rec)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_id": run_id,
        "clean_case_key": clean_case_key,
        "run_dir": str(run_dir),
        "mode": "records",
        "records_format": "objects",
        "metrics_format": metrics_format,
        "run_config": run_config,
        "env": env,
        "checkpoints": ckpt_id,
        "metrics_index": metrics_index,
        "cases_index": cases_index,
        "datasets_index": datasets_index,
        "splits_index": splits_index,
        "notes": {
            "delta_definition": "degradation_vs_clean is defined so delta>0 means worse under corruption for both higher- and lower-better metrics.",
            "gains_vs_sa_definition": "robustness_gain_vs_SA = degradation_vs_clean(model) - degradation_vs_clean(SA); negative is better.",
            "missing_values": "NaN/inf are omitted from metric dicts; missing keys imply null.",
        },
        "records": records,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a robustness run directory into an LLM-friendly JSON.")
    ap.add_argument("--run_dir", type=Path, required=True, help="Run directory containing metrics.csv.")
    ap.add_argument(
        "--out_path",
        type=Path,
        default=None,
        help="Output path (default: <run_dir>/_llm_summary.json).",
    )
    ap.add_argument(
        "--mode",
        choices=("records", "compact_agg", "nested"),
        default="records",
        help="Output format (default: records).",
    )
    ap.add_argument(
        "--no_gains_vs_sa",
        action="store_true",
        help="Disable robustness_gain_vs_SA computation.",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Comma-separated metric list to include (records/compact_agg only).",
    )
    ap.add_argument(
        "--metrics_format",
        choices=("array", "dict"),
        default="array",
        help="How to store per-record metrics (default: array aligned to metrics_index).",
    )
    ap.add_argument(
        "--round",
        type=int,
        default=4,
        help="Round floats to this many decimals (-1 disables rounding).",
    )
    ap.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (default is minified to reduce tokens).",
    )
    ap.add_argument(
        "--compact",
        action="store_true",
        help="Deprecated: minify output JSON (nested mode only).",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    out_path = (run_dir / "_llm_summary.json") if args.out_path is None else args.out_path.expanduser().resolve()

    metric_list = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    mode = str(args.mode)
    if mode == "nested":
        summary = build_llm_summary(run_dir, include_gains_vs_sa=not bool(args.no_gains_vs_sa))
        # Preserve old --compact behavior for nested mode.
        indent = 2 if (bool(args.pretty) and not bool(args.compact)) else None
        separators = (",", ":") if indent is None else None
    elif mode == "compact_agg":
        summary = build_compact_llm_summary(run_dir, metrics=metric_list or None)
        indent = 2 if bool(args.pretty) else None
        separators = (",", ":") if indent is None else None
    else:
        summary = build_records_llm_summary(
            run_dir,
            include_gains_vs_sa=not bool(args.no_gains_vs_sa),
            metric_names=metric_list or None,
            metrics_format=str(args.metrics_format),
        )
        indent = 2 if bool(args.pretty) else None
        separators = (",", ":") if indent is None else None

    payload = _to_jsonable(_round_floats(summary, decimals=int(args.round)))
    txt = json.dumps(payload, indent=indent, separators=separators, sort_keys=False)
    out_path.write_text(txt + ("\n" if not txt.endswith("\n") else ""), encoding="utf-8")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
