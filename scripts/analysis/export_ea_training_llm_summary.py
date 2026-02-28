#!/usr/bin/env python3
"""
Export EA training log CSVs into a compact, LLM-friendly JSON summary (token-efficient).

Intended for logs written by `src/tabicl/train/run.py` (Trainer CSV logging), e.g.:
  - training_metrics/mini_tabicl_stage{1,2}_ea_row_icl.csv   (EA-FULL: row+icl ellipse)
  - training_metrics/mini_tabicl_stage{1,2}_ea_icl_only.csv  (EA-ICL: icl-only ellipse)

Important:
  - Some CSVs may contain multiple runs appended back-to-back (step resets 1..T, then 1..T).
  - For stage2 EA-FULL specifically, we select **run 2** by default when a step reset is detected.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1


BASE_METRICS: Tuple[str, ...] = (
    "ce",
    "accuracy",
    "lr",
    "prior_time",
    "train_time",
)

EA_METRICS: Tuple[str, ...] = (
    # TFrow
    "row_L2_context_div",
    "row_L2_sparsity",
    "row_L2_scale_mean",
    "row_L2_feat_velocity",
    "row_L2_q_norm",
    "row_L2_logit_mean",
    "row_L12_context_div",
    "row_L12_sparsity",
    "row_L12_scale_mean",
    "row_L12_feat_velocity",
    "row_L12_q_norm",
    "row_L12_logit_mean",
    # TFicl
    "icl_L2_context_div",
    "icl_L2_sparsity",
    "icl_L2_scale_mean",
    "icl_L2_feat_velocity",
    "icl_L2_q_norm",
    "icl_L2_logit_mean",
    "icl_L12_context_div",
    "icl_L12_sparsity",
    "icl_L12_scale_mean",
    "icl_L12_feat_velocity",
    "icl_L12_q_norm",
    "icl_L12_logit_mean",
)

DEFAULT_COMPACT_METRICS: Tuple[str, ...] = (
    # Base
    "ce",
    "accuracy",
    "lr",
    # TFicl (main driver of ICL behavior)
    "icl_L2_scale_mean",
    "icl_L2_sparsity",
    "icl_L2_q_norm",
    "icl_L2_logit_mean",
    "icl_L12_scale_mean",
    "icl_L12_sparsity",
    "icl_L12_q_norm",
    "icl_L12_logit_mean",
    # TFrow (only present for EA-FULL row+icl)
    "row_L2_scale_mean",
    "row_L2_sparsity",
    "row_L2_q_norm",
    "row_L2_logit_mean",
)


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


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == "step":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["step"] = pd.to_numeric(out["step"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["step"]).copy()
    out["step"] = out["step"].astype(int)
    return out


def _split_by_step_resets_keep_order(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Split when `step` decreases (multiple runs appended to the same CSV)."""
    if "step" not in df.columns or df.shape[0] == 0:
        return [df]

    step = pd.to_numeric(df["step"], errors="coerce")
    cut_points: List[int] = []
    prev: Optional[float] = None
    for i, v in enumerate(step.tolist()):
        if prev is not None and pd.notna(v) and pd.notna(prev) and float(v) < float(prev):
            cut_points.append(i)
        prev = v

    if not cut_points:
        return [df]

    segments: List[pd.DataFrame] = []
    start = 0
    for cut in cut_points:
        segments.append(df.iloc[start:cut].reset_index(drop=True))
        start = cut
    segments.append(df.iloc[start:].reset_index(drop=True))
    return segments


@dataclass(frozen=True)
class RunKey:
    stage: int
    variant: str  # EA-FULL or EA-ICL
    component: str  # row+icl or icl-only


def _infer_run_key(path: Path) -> RunKey:
    name = path.name.lower()
    stage = 1 if "stage1" in name else 2 if "stage2" in name else 0
    if stage == 0:
        raise ValueError(f"Could not infer stage from filename: {path.name}")

    if "ea_row_icl" in name or "row_icl" in name:
        return RunKey(stage=stage, variant="EA-FULL", component="row+icl")
    if "ea_icl_only" in name or "icl_only" in name:
        return RunKey(stage=stage, variant="EA-ICL", component="icl-only")

    raise ValueError(f"Could not infer EA variant from filename: {path.name}")


def _window_by_steps(df: pd.DataFrame, *, n_steps: int, tail: bool) -> pd.DataFrame:
    if df.shape[0] == 0:
        return df
    df = df.sort_values("step").reset_index(drop=True)
    steps = df["step"].to_numpy()
    unique_steps = np.unique(steps)
    if unique_steps.size == 0:
        return df.iloc[0:0]
    n = int(min(n_steps, unique_steps.size))
    if n <= 0:
        return df.iloc[0:0]
    selected = unique_steps[-n:] if tail else unique_steps[:n]
    selected_set = set(int(x) for x in selected.tolist())
    return df[df["step"].isin(selected_set)].copy()


def _metric_stats(
    df: pd.DataFrame,
    metric: str,
    *,
    tail_steps: int,
    min_valid_frac: float,
) -> Optional[Dict[str, Any]]:
    if metric not in df.columns:
        return None

    s = pd.to_numeric(df[metric], errors="coerce")
    valid_frac = float(s.notna().mean()) if s.shape[0] else 0.0
    if valid_frac < min_valid_frac:
        return None

    tail_df = _window_by_steps(df, n_steps=tail_steps, tail=True)
    early_df = _window_by_steps(df, n_steps=tail_steps, tail=False)

    tail = pd.to_numeric(tail_df[metric], errors="coerce")
    early = pd.to_numeric(early_df[metric], errors="coerce")

    def _finite(a: pd.Series) -> np.ndarray:
        x = a.to_numpy(dtype=float)
        return x[np.isfinite(x)]

    tail_x = _finite(tail)
    early_x = _finite(early)
    all_x = _finite(s)

    if all_x.size == 0:
        return None

    out: Dict[str, Any] = {
        "valid_frac": valid_frac,
        "mean": float(np.nanmean(all_x)) if all_x.size else None,
        "std": float(np.nanstd(all_x)) if all_x.size else None,
        "tail": {},
        "early": {},
    }

    if tail_x.size:
        out["tail"] = {
            "n": int(tail_x.size),
            "mean": float(np.nanmean(tail_x)),
            "std": float(np.nanstd(tail_x)),
            "p05": float(np.nanpercentile(tail_x, 5)),
            "p95": float(np.nanpercentile(tail_x, 95)),
        }
        if tail_x.size >= 3:
            out["tail"]["volatility_mean_abs_delta"] = float(np.nanmean(np.abs(np.diff(tail_x))))

    if early_x.size:
        out["early"] = {
            "n": int(early_x.size),
            "mean": float(np.nanmean(early_x)),
            "std": float(np.nanstd(early_x)),
            "p05": float(np.nanpercentile(early_x, 5)),
            "p95": float(np.nanpercentile(early_x, 95)),
        }

    if tail_x.size and early_x.size:
        out["tail_minus_early_mean"] = float(out["tail"]["mean"] - out["early"]["mean"])

    return out


def _load_selected_segment(
    path: Path,
    *,
    prefer_run2_if_multiple: bool,
    force_run2_for_stage2_ea_full: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw = pd.read_csv(path)
    raw = _coerce_numeric(raw)
    segments = _split_by_step_resets_keep_order(raw)

    key = _infer_run_key(path)
    meta: Dict[str, Any] = {
        "file": path.name,
        "segments_detected": int(len(segments)),
        "selected_segment": 1,
        "selection_reason": "single_segment",
    }

    if len(segments) == 1:
        df = segments[0]
        return df.sort_values("step").reset_index(drop=True), meta

    # Multiple segments: select run2 if requested (or forced for stage2 EA-FULL)
    want_run2 = prefer_run2_if_multiple
    if force_run2_for_stage2_ea_full and key.stage == 2 and key.variant == "EA-FULL":
        want_run2 = True

    sel_idx = 1 if want_run2 else 0
    sel_idx = min(sel_idx, len(segments) - 1)
    meta.update(
        {
            "selected_segment": int(sel_idx + 1),
            "selection_reason": "step_reset_detected",
        }
    )
    df = segments[sel_idx]
    return df.sort_values("step").reset_index(drop=True), meta


def _compact_run_payload(
    df: pd.DataFrame,
    *,
    tail_steps: int,
    metrics: List[str],
    min_valid_frac: float,
) -> Dict[str, Any]:
    df = df.sort_values("step").reset_index(drop=True)
    tail_df = _window_by_steps(df, n_steps=tail_steps, tail=True)
    early_df = _window_by_steps(df, n_steps=tail_steps, tail=False)

    def _mean(d: pd.DataFrame, m: str) -> Optional[float]:
        if m not in d.columns:
            return None
        s = pd.to_numeric(d[m], errors="coerce")
        if s.isna().mean() > (1.0 - min_valid_frac):
            return None
        if s.isna().all():
            return None
        return float(np.nanmean(s.to_numpy(dtype=float)))

    def _std(d: pd.DataFrame, m: str) -> Optional[float]:
        if m not in d.columns:
            return None
        s = pd.to_numeric(d[m], errors="coerce")
        if s.isna().mean() > (1.0 - min_valid_frac):
            return None
        if s.isna().all():
            return None
        return float(np.nanstd(s.to_numpy(dtype=float)))

    def _vol(d: pd.DataFrame, m: str) -> Optional[float]:
        if m not in d.columns:
            return None
        s = pd.to_numeric(d[m], errors="coerce")
        if s.isna().mean() > (1.0 - min_valid_frac):
            return None
        x = s.to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 3:
            return None
        return float(np.nanmean(np.abs(np.diff(x))))

    tail_mean: Dict[str, float] = {}
    tail_std: Dict[str, float] = {}
    tail_vol: Dict[str, float] = {}
    tail_minus_early: Dict[str, float] = {}

    for m in metrics:
        tm = _mean(tail_df, m)
        if tm is None:
            continue
        tail_mean[m] = tm
        ts = _std(tail_df, m)
        if ts is not None:
            tail_std[m] = ts
        tv = _vol(tail_df, m)
        if tv is not None:
            tail_vol[m] = tv
        em = _mean(early_df, m)
        if em is not None:
            tail_minus_early[m] = float(tm - em)

    return {
        "tail_mean": tail_mean,
        "tail_std": tail_std,
        "tail_volatility": tail_vol,
        "tail_minus_early_mean": tail_minus_early,
    }


def export_llm_training_summary(
    input_dir: Path,
    *,
    out_path: Path,
    tail_stage1: int,
    tail_stage2: int,
    min_valid_frac: float,
    prefer_run2_if_multiple: bool,
    force_run2_for_stage2_ea_full: bool,
    mode: str,
    compact_metrics: List[str],
    round_decimals: int,
    pretty: bool,
) -> Dict[str, Any]:
    input_dir = input_dir.resolve()

    files = sorted(input_dir.glob("*.csv"))
    ea_files: List[Path] = []
    for p in files:
        name = p.name.lower()
        if "ea_" not in name:
            continue
        if "stage1" not in name and "stage2" not in name:
            continue
        if "row_icl" not in name and "icl_only" not in name:
            continue
        ea_files.append(p)

    if not ea_files:
        raise FileNotFoundError(f"No EA training CSVs found under: {input_dir}")

    notes: List[str] = [
        "PDLC/probe columns may be all-NaN depending on training configuration (probe logging, PDLC enabled).",
        "Some CSVs may include multiple runs appended (detected via `step` reset).",
    ]

    runs_out: List[Dict[str, Any]] = []
    # For comparisons
    stage2_tail_means: Dict[str, Dict[str, float]] = {}

    for path in ea_files:
        key = _infer_run_key(path)
        df, seg_meta = _load_selected_segment(
            path,
            prefer_run2_if_multiple=prefer_run2_if_multiple,
            force_run2_for_stage2_ea_full=force_run2_for_stage2_ea_full,
        )

        tail_steps = tail_stage1 if key.stage == 1 else tail_stage2

        run_rec: Dict[str, Any] = {
            "id": f"{path.stem}/seg{seg_meta['selected_segment']}",
            "stage": key.stage,
            "variant": key.variant,
            "component": key.component,
            "source": seg_meta,
            "n_rows": int(df.shape[0]),
            "step_min": int(df["step"].min()) if df.shape[0] else None,
            "step_max": int(df["step"].max()) if df.shape[0] else None,
            "tail_window_steps": int(tail_steps),
        }

        if mode == "full":
            run_rec["metrics"] = {}
            metrics = list(BASE_METRICS) + list(EA_METRICS)
            metrics_present = [m for m in metrics if m in df.columns and not df[m].isna().all()]
            for m in metrics_present:
                stats = _metric_stats(df, m, tail_steps=tail_steps, min_valid_frac=min_valid_frac)
                if stats is not None:
                    run_rec["metrics"][m] = stats
        else:
            # Compact: keep only tail means/trends for a small metric set.
            metrics = compact_metrics
            if not metrics:
                metrics = list(DEFAULT_COMPACT_METRICS)
            run_rec.update(
                _compact_run_payload(df, tail_steps=tail_steps, metrics=metrics, min_valid_frac=min_valid_frac)
            )

        # Tail means for stage2 comparisons
        if key.stage == 2:
            means: Dict[str, float] = {}
            if mode == "full":
                tail_df = _window_by_steps(df, n_steps=tail_steps, tail=True)
                for m in DEFAULT_COMPACT_METRICS:
                    if m not in tail_df.columns:
                        continue
                    s = pd.to_numeric(tail_df[m], errors="coerce")
                    if s.isna().all():
                        continue
                    means[m] = float(np.nanmean(s.to_numpy(dtype=float)))
            else:
                means = dict(run_rec.get("tail_mean", {}))
            stage2_tail_means[f"{key.variant}"] = means

        runs_out.append(run_rec)

    # Compact stage2 comparison (EA-FULL selected segment vs EA-ICL)
    comparisons: Dict[str, Any] = {}
    if "EA-FULL" in stage2_tail_means and "EA-ICL" in stage2_tail_means:
        full = stage2_tail_means["EA-FULL"]
        icl = stage2_tail_means["EA-ICL"]
        shared = sorted(set(full.keys()) & set(icl.keys()))
        comparisons["stage2_tail_mean_diff"] = {
            "EA-ICL_minus_EA-FULL": {m: float(icl[m] - full[m]) for m in shared}
        }

    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "out_path": str(out_path),
        "mode": mode,
        "notes": notes,
        "runs": runs_out,
        "comparisons": comparisons,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        payload = _to_jsonable(_round_floats(out, decimals=round_decimals))
        if pretty:
            json.dump(payload, f, indent=2, sort_keys=True)
        else:
            json.dump(payload, f, separators=(",", ":"), sort_keys=False)

    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input_dir",
        type=Path,
        default=Path("training_metrics"),
        help="Directory containing EA training CSVs (default: training_metrics).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("training_metrics/_llm_ea_training_summary.json"),
        help="Output JSON path (default: training_metrics/_llm_ea_training_summary.json).",
    )
    p.add_argument(
        "--mode",
        choices=("compact", "full"),
        default="compact",
        help="Output detail level (default: compact).",
    )
    p.add_argument("--tail_stage1", type=int, default=2000, help="Tail window (steps) for stage1 logs.")
    p.add_argument("--tail_stage2", type=int, default=200, help="Tail window (steps) for stage2 logs.")
    p.add_argument(
        "--min_valid_frac",
        type=float,
        default=0.1,
        help="Only include metrics with at least this fraction of non-NaN values.",
    )
    p.add_argument(
        "--compact_metrics",
        type=str,
        default=",".join(DEFAULT_COMPACT_METRICS),
        help="Comma-separated metrics to include in compact mode.",
    )
    p.add_argument(
        "--round",
        type=int,
        default=4,
        help="Round floats to this many decimals (-1 disables rounding).",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (default is minified to reduce tokens).",
    )
    p.add_argument(
        "--prefer_run2_if_multiple",
        action="store_true",
        help="If a CSV has multiple segments (step resets), select run2 (segment 2) instead of run1.",
    )
    p.add_argument(
        "--no_force_run2_for_stage2_ea_full",
        action="store_false",
        dest="force_run2_for_stage2_ea_full",
        help="Disable forcing run2 selection for stage2 EA-FULL (row+icl) when multiple segments are detected.",
    )

    args = p.parse_args()
    compact_metrics = [m.strip() for m in str(args.compact_metrics).split(",") if m.strip()]
    export_llm_training_summary(
        args.input_dir,
        out_path=args.out,
        tail_stage1=args.tail_stage1,
        tail_stage2=args.tail_stage2,
        min_valid_frac=args.min_valid_frac,
        prefer_run2_if_multiple=args.prefer_run2_if_multiple,
        force_run2_for_stage2_ea_full=args.force_run2_for_stage2_ea_full,
        mode=str(args.mode),
        compact_metrics=compact_metrics,
        round_decimals=int(args.round),
        pretty=bool(args.pretty),
    )


if __name__ == "__main__":
    main()
