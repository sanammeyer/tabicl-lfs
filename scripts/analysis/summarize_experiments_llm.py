#!/usr/bin/env python3
from __future__ import annotations

"""
LLM-friendly summarizer for experiment outputs under `results/**/summary_all.json`.

Why this exists
---------------
Many scripts in this repo write a `summary_all.json` that contains:
  - the full CLI/config (`args`)
  - a list of per-dataset outputs (`datasets`)

These JSONs are great for reproducibility but often contain very large fields
(`variant.support_mask`, `variant.test_index`, confusion matrices, etc.) that
are not convenient to feed into an LLM.

This script produces a compact summary that preserves:
  - provenance (which datasets, checkpoints, key hyperparameters)
  - key metrics (macro-F1/acc, gamma AUC/AP, uncertainty AUROCs, head-swap scores, …)

Outputs
-------
Two output shapes are supported:
  1) JSON (default): one file with a list of runs and compacted per-dataset entries.
  2) JSONL: one record per (run, dataset) to make filtering/streaming easier for LLMs.

Examples
--------
  # PDLC diagnosis suite only (recommended)
  python scripts/analysis/summarize_experiments_llm.py \
    --results_root results \
    --include rerun_correct_pdl \
    --suite_depth 1 \
    --out results/rerun_correct_pdl/_llm_summary.json

  # Same, but JSONL (one record per dataset)
  python scripts/analysis/summarize_experiments_llm.py \
    --results_root results \
    --include rerun_correct_pdl \
    --suite_depth 1 \
    --jsonl \
    --out results/rerun_correct_pdl/_llm_summary.jsonl
"""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _as_posix(path: Path) -> str:
    try:
        return path.as_posix()
    except Exception:
        return str(path)


def _maybe_rel(path: Path) -> str:
    try:
        return _as_posix(path.relative_to(REPO_ROOT))
    except Exception:
        return _as_posix(path)


def _is_truthy_flag(v: Any) -> bool:
    return bool(v) is True


def _select_keys(d: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    return {k: d.get(k) for k in keys if k in d}


def _compact_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _maybe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _thin_variant(variant: Any) -> Any:
    """
    Remove giant fields from the episode/variant payload while keeping identifiers.
    """
    if not isinstance(variant, dict):
        return variant

    out: Dict[str, Any] = {}
    for k in (
        "variant_index",
        "X_shape",
        "train_size",
        "class_shift_offset",
        "norm_method",
    ):
        if k in variant:
            out[k] = variant.get(k)

    if "feature_permutation" in variant:
        fp = variant.get("feature_permutation")
        out["feature_permutation_len"] = len(fp) if isinstance(fp, list) else None
        out["feature_permutation_sha256"] = _sha256_json(fp) if fp is not None else None

    if "label_map" in variant:
        lm = variant.get("label_map")
        out["label_map_n"] = len(lm) if isinstance(lm, dict) else None
        out["label_map_sha256"] = _sha256_json(lm) if lm is not None else None

    # These are commonly huge lists; keep only sizes/hashes when present.
    for huge_key in ("support_mask", "test_index", "train_index"):
        if huge_key in variant:
            v = variant.get(huge_key)
            if isinstance(v, list):
                out[f"{huge_key}_len"] = len(v)
                # Hashing can be expensive; avoid hashing giant arrays.
            else:
                out[f"{huge_key}_len"] = None

    return _compact_none(out)


def _thin_models(models: Any, include_confusion: bool) -> Any:
    if not isinstance(models, dict):
        return models

    keep = (
        "acc",
        "macro_f1",
        "weighted_f1",
        "binary_f1_pos",
        "n_classes",
        "n_test",
        "n_predicted_classes",
        "pred_entropy_norm",
        "pred_entropy",
        "pred_counts",
        "margin_median",
        "margin_mean",
        "margin_p10",
        "margin_p90",
        "pred_top1_frac",
        "pred_top2_frac",
    )

    out: Dict[str, Any] = {}
    for model_name, md in models.items():
        if not isinstance(md, dict):
            out[model_name] = md
            continue

        thin = _select_keys(md, keep)
        if include_confusion and "confusion" in md:
            thin["confusion"] = md.get("confusion")
        out[model_name] = _compact_none(thin)
    return out


def _thin_head_swap(head_swap: Any) -> Any:
    if head_swap in (None, False):
        return head_swap
    if not isinstance(head_swap, dict):
        return head_swap

    out: Dict[str, Any] = {}
    for k in ("pdl_on_sa", "sa_on_pdl"):
        v = head_swap.get(k)
        if isinstance(v, dict):
            out[k] = _compact_none(
                _select_keys(
                    v,
                    (
                        "acc",
                        "macro_f1",
                        "weighted_f1",
                        "n_predicted_classes",
                        "pred_entropy_norm",
                        "margin_median",
                    ),
                )
            )
    return out


def _thin_pdl_gamma(pdl_gamma: Any) -> Any:
    if not isinstance(pdl_gamma, dict):
        return pdl_gamma
    return _compact_none(
        _select_keys(
            pdl_gamma,
            (
                "auc",
                "ap",
                "gamma_same_mean",
                "gamma_diff_mean",
                "gamma_same_p10",
                "gamma_same_p90",
                "gamma_diff_p10",
                "gamma_diff_p90",
                "n_pairs_used",
                "n_support",
            ),
        )
    )


def _thin_uncertainty_eval(ueval: Any) -> Any:
    if not isinstance(ueval, dict):
        return ueval
    return _compact_none(
        _select_keys(
            ueval,
            (
                "n",
                "err_rate",
                "tu_mean_correct",
                "tu_mean_wrong",
                "au_mean_correct",
                "au_mean_wrong",
                "eu_mean_correct",
                "eu_mean_wrong",
                "tu_err_auc",
                "eu_err_auc",
                "tu_err_ap",
                "eu_err_ap",
                "au_err_auc",
                "au_err_ap",
                "eu_min",
                "frac_eu_negative",
            ),
        )
    )


def _thin_dataset_entry(ds: Dict[str, Any], include_confusion: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for k in ("dataset_id", "dataset_name", "n_rows", "n_features"):
        if k in ds:
            out[k] = ds.get(k)

    if "split" in ds:
        split = ds.get("split")
        if isinstance(split, dict):
            out["split"] = _compact_none(
                _select_keys(split, ("kind", "seed", "test_size", "fold", "task_id", "dataset_id"))
            )
        else:
            out["split"] = split

    out["models"] = _thin_models(ds.get("models"), include_confusion=include_confusion)

    if "head_swap" in ds:
        out["head_swap"] = _thin_head_swap(ds.get("head_swap"))

    if "pdl_gamma" in ds:
        out["pdl_gamma"] = _thin_pdl_gamma(ds.get("pdl_gamma"))

    if "pdl_anchor_uncertainty_eval" in ds:
        out["pdl_anchor_uncertainty_eval"] = _thin_uncertainty_eval(ds.get("pdl_anchor_uncertainty_eval"))

    if "errors" in ds:
        errs = ds.get("errors")
        if isinstance(errs, list) and errs:
            out["errors"] = errs[:5]

    if "pdlc_topk_tune" in ds and ds.get("pdlc_topk_tune") not in (None, False):
        out["pdlc_topk_tune"] = ds.get("pdlc_topk_tune")

    if "variant" in ds:
        out["variant"] = _thin_variant(ds.get("variant"))

    if "variant_runs" in ds:
        vr = ds.get("variant_runs")
        out["variant_runs_n"] = len(vr) if isinstance(vr, list) else None

    # Convenience deltas if both SA and PDL are present
    try:
        models = ds.get("models") or {}
        if isinstance(models, dict) and ("sa" in models) and ("pdl" in models):
            sa = models.get("sa") or {}
            pdl = models.get("pdl") or {}
            if isinstance(sa, dict) and isinstance(pdl, dict):
                sa_f1 = _maybe_float(sa.get("macro_f1"))
                pdl_f1 = _maybe_float(pdl.get("macro_f1"))
                if sa_f1 is not None and pdl_f1 is not None:
                    out["delta_macro_f1"] = pdl_f1 - sa_f1
    except Exception:
        pass

    return _compact_none(out)


def _guess_suite(
    out_dir: str | None,
    summary_all_path: Path,
    results_root: Path,
    suite_depth: int,
) -> Dict[str, Any]:
    """
    Return a small hierarchical tag object:
      - `suite`: first path component under `results_root`
      - `subrun`: second component (often dsXXXX for OpenML-based sweeps)
    """
    p: Optional[Path] = None
    if out_dir:
        p = (REPO_ROOT / out_dir).resolve() if not Path(out_dir).is_absolute() else Path(out_dir).resolve()
    else:
        p = summary_all_path.parent.resolve()

    try:
        rel = p.relative_to(results_root.resolve())
        parts = rel.parts
        suite = parts[suite_depth] if 0 <= suite_depth < len(parts) else None
        subrun = parts[suite_depth + 1] if 0 <= (suite_depth + 1) < len(parts) else None
        return {
            "suite": suite,
            "subrun": subrun,
            "out_dir_rel": rel.as_posix(),
            "out_dir_parts": list(parts),
        }
    except Exception:
        return {"suite": None, "subrun": None, "out_dir_rel": _maybe_rel(p), "out_dir_parts": None}


def _build_reproduce_cmd(args: Dict[str, Any]) -> Optional[str]:
    """
    Best-effort reconstruction of the CLI for `diagnose_tabpdl_head.py`.
    """
    if not isinstance(args, dict):
        return None
    if "pdl_checkpoint" not in args and "sa_checkpoint" not in args:
        return None

    cmd: List[str] = ["python", "scripts/diagnostics/diagnose_tabpdl_head.py"]

    # Dataset selection (mutually exclusive in the script; keep both if present for transparency)
    if args.get("datasets"):
        cmd += ["--datasets", str(args["datasets"])]
    if args.get("openml_tasks"):
        cmd += ["--openml_tasks", str(args["openml_tasks"])]
    if args.get("fold") is not None:
        cmd += ["--fold", str(args["fold"])]

    # Checkpoints
    if args.get("sa_checkpoint"):
        cmd += ["--sa_checkpoint", str(args["sa_checkpoint"])]
    if args.get("pdl_checkpoint"):
        cmd += ["--pdl_checkpoint", str(args["pdl_checkpoint"])]

    # Core inference knobs
    for k, flag in (
        ("pdlc_agg", "--pdlc_agg"),
        ("pdlc_topk", "--pdlc_topk"),
        ("pdlc_inference_temperature", "--pdlc_inference_temperature"),
        ("pdlc_bilinear", "--pdlc_bilinear"),
        ("pdlc_tau_override", "--pdlc_tau_override"),
        ("n_estimators", "--n_estimators"),
        ("batch_size", "--batch_size"),
        ("seed", "--seed"),
        ("test_size", "--test_size"),
        ("softmax_temperature", "--softmax_temperature"),
        ("device", "--device"),
        ("variant", "--variant"),
        ("max_pairs", "--max_pairs"),
    ):
        v = args.get(k)
        if v is None:
            continue
        cmd += [flag, str(v)]

    # Feature flags
    for k, flag in (
        ("head_swap", "--head_swap"),
        ("pdl_uncertainty", "--pdl_uncertainty"),
        ("pdl_learn_anchor_weights", "--pdl_learn_anchor_weights"),
        ("pdl_anchor_weight_use_topk", "--pdl_anchor_weight_use_topk"),
        ("pdlc_topk_tune", "--pdlc_topk_tune"),
        ("use_amp", "--use_amp"),
        ("verbose", "--verbose"),
        ("no_save", "--no_save"),
    ):
        if _is_truthy_flag(args.get(k)):
            cmd.append(flag)

    # Tuning/ablation strings (only include if set and non-empty)
    for k, flag in (
        ("pdlc_topk_tune_candidates", "--pdlc_topk_tune_candidates"),
        ("pdlc_topk_tune_metric", "--pdlc_topk_tune_metric"),
        ("pdlc_topk_tune_val_frac", "--pdlc_topk_tune_val_frac"),
        ("pdl_agg_sweep", "--pdl_agg_sweep"),
        ("pdl_topk_sweep", "--pdl_topk_sweep"),
        ("pdl_embed_norm_sweep", "--pdl_embed_norm_sweep"),
        ("pdl_infer_temp_sweep", "--pdl_infer_temp_sweep"),
        ("pdl_anchor_val_frac", "--pdl_anchor_val_frac"),
        ("pdl_anchor_weight_steps", "--pdl_anchor_weight_steps"),
        ("pdl_anchor_weight_lr", "--pdl_anchor_weight_lr"),
        ("pdl_anchor_weight_prune_lambda", "--pdl_anchor_weight_prune_lambda"),
        ("pdl_anchor_weight_entropy_lambda", "--pdl_anchor_weight_entropy_lambda"),
        ("pdl_anchor_weight_prune_topm", "--pdl_anchor_weight_prune_topm"),
    ):
        v = args.get(k)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        cmd += [flag, str(v)]

    if args.get("out_dir"):
        cmd += ["--out_dir", str(args["out_dir"])]

    return " ".join(cmd)


def _find_summary_all_files(results_root: Path) -> List[Path]:
    return sorted(results_root.rglob("summary_all.json"))


def _matches_filters(s: str, includes: Sequence[str], excludes: Sequence[str]) -> bool:
    if includes and not any(tok in s for tok in includes):
        return False
    if excludes and any(tok in s for tok in excludes):
        return False
    return True


def _collect_checkpoint_status(args: Dict[str, Any]) -> Dict[str, Any]:
    ckpts: Dict[str, Any] = {}
    for key in ("sa_checkpoint", "pdl_checkpoint", "checkpoint"):
        v = args.get(key)
        if not v:
            continue
        p = Path(str(v))
        if not p.is_absolute():
            p = REPO_ROOT / p
        ckpts[key] = {"path": _maybe_rel(p), "exists": p.exists()}
    return ckpts


def _thin_args(args: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(args, dict):
        return {}

    # Keep all scalar-ish args; drop very long strings (rare) and nested objects.
    out: Dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, str):
            out[k] = v if len(v) <= 500 else (v[:500] + "…")
        else:
            # e.g., dict/list: keep as-is only if small
            try:
                blob = json.dumps(v, ensure_ascii=False)
                out[k] = v if len(blob) <= 500 else {"_omitted": True, "type": type(v).__name__}
            except Exception:
                out[k] = {"_omitted": True, "type": type(v).__name__}
    return out


def summarize_run(
    summary_all_path: Path,
    results_root: Path,
    suite_depth: int,
    include_confusion: bool,
) -> Dict[str, Any]:
    obj = _read_json(summary_all_path)
    args = obj.get("args") or {}
    datasets = obj.get("datasets") or []

    out_dir = None
    if isinstance(args, dict):
        out_dir = args.get("out_dir")

    suite = _guess_suite(
        out_dir=out_dir,
        summary_all_path=summary_all_path,
        results_root=results_root,
        suite_depth=suite_depth,
    )
    reproduce_cmd = _build_reproduce_cmd(args) if isinstance(args, dict) else None

    run: Dict[str, Any] = {
        "schema_version": 1,
        "source_summary_all": _maybe_rel(summary_all_path),
        "out_dir": out_dir,
        "suite": _compact_none(suite),
        "args": _thin_args(args) if isinstance(args, dict) else args,
        "checkpoints": _collect_checkpoint_status(args) if isinstance(args, dict) else {},
        "reproduce_cmd": reproduce_cmd,
        "n_datasets": len(datasets) if isinstance(datasets, list) else None,
        "datasets": [],
    }

    if isinstance(datasets, list):
        run["datasets"] = [
            _thin_dataset_entry(ds, include_confusion=include_confusion)
            for ds in datasets
            if isinstance(ds, dict)
        ]

    return _compact_none(run)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a compact, LLM-friendly experiment summary from results/**/summary_all.json.")
    p.add_argument("--results_root", type=str, default="results", help="Root folder to scan for summary_all.json (default: results).")
    p.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=[],
        help="Only include runs whose path contains any of these substrings (e.g. rerun_correct_pdl).",
    )
    p.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="Exclude runs whose path contains any of these substrings.",
    )
    p.add_argument(
        "--include_confusion",
        action="store_true",
        help="Include confusion matrices if present (can be large).",
    )
    p.add_argument(
        "--suite_depth",
        type=int,
        default=0,
        help=(
            "Which path component under --results_root to treat as the 'suite' label "
            "(default: 0). For example with --include rerun_correct_pdl, use --suite_depth 1 "
            "to get suite=head_swap|pdl_uncertainty_topk32|... instead of suite=rerun_correct_pdl."
        ),
    )
    p.add_argument(
        "--jsonl",
        action="store_true",
        help="Write JSONL instead of JSON (one record per dataset per run).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path. If omitted, prints JSON to stdout.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    results_root = (REPO_ROOT / args.results_root).resolve() if not Path(args.results_root).is_absolute() else Path(args.results_root).resolve()

    files = _find_summary_all_files(results_root)
    # Filter on both path and out_dir string (where available) by simply matching the path string.
    filtered: List[Path] = []
    for pth in files:
        s = _as_posix(pth)
        if _matches_filters(s, includes=args.include, excludes=args.exclude):
            filtered.append(pth)

    runs = [
        summarize_run(
            p,
            results_root=results_root,
            suite_depth=int(args.suite_depth),
            include_confusion=args.include_confusion,
        )
        for p in filtered
    ]

    envelope = {
        "generated_at_utc": _utc_now_iso(),
        "repo_root": _maybe_rel(REPO_ROOT),
        "results_root": _maybe_rel(results_root),
        "include": list(args.include),
        "exclude": list(args.exclude),
        "suite_depth": int(args.suite_depth),
        "n_runs": len(runs),
        "runs": runs,
    }

    if args.jsonl:
        records: List[Dict[str, Any]] = []
        for run in runs:
            run_meta = {k: v for k, v in run.items() if k not in {"datasets"}}
            for ds in run.get("datasets", []) or []:
                records.append({"run": run_meta, "dataset": ds})

        if args.out:
            write_jsonl(Path(args.out), records)
        else:
            for rec in records:
                print(json.dumps(rec, ensure_ascii=False))
        return

    if args.out:
        write_json(Path(args.out), envelope)
    else:
        print(json.dumps(envelope, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
