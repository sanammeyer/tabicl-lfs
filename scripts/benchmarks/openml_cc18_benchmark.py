"""
OpenML-CC18 10-fold CV Benchmark

This script benchmarks models on the OpenML-CC18 suite using official 10-fold
cross-validation splits for each task.

Supported models (select via --model):
- tabicl: Uses TabICLClassifier; pass --checkpoint to select checkpoint file.
- tabpfn: Uses TabPFNClassifier with default settings.

Results are saved per task *fold* (one row per fold) instead of aggregating across folds.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import openml
import pandas as pd
import torch
import time
import json
import gc

try:
    import wandb

    HAVE_WANDB = True
except Exception:
    wandb = None  # type: ignore[assignment]
    HAVE_WANDB = False

# Repo root (this file lives at scripts/benchmarks/openml_cc18_benchmark.py)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Ensure repo root + local sources are importable when invoking via file path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Reuse helpers
from benchmark_utils import load_dataset as _unused_load_dataset  # kept for reference
from benchmark_utils import compute_metrics, compute_dataset_metadata


def _str2bool(v: str | bool) -> bool:
    """Minimal boolean parser for CLI flags (mirrors training str2bool behavior)."""
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in {"yes", "true", "t", "y", "1"}:
        return True
    if val in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark OpenML-CC18 with 10-fold CV")
    p.add_argument("--model", choices=["tabicl", "tabpfn"], default="tabicl", help="Model to benchmark")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "step-1300.ckpt"),
        help="TabICL checkpoint path (only used for --model tabicl)",
    )
    p.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|cuda")
    p.add_argument("--n_estimators", type=int, default=32, help="TabICL ensemble size")
    p.add_argument("--batch_size", type=int, default=8, help="TabICL inference batch size (smaller reduces OOM risk)")
    p.add_argument("--elliptical_scale_boost", type=float, default=1.0, help="Extra multiplicative factor for elliptical scale (ICL)")
    p.add_argument(
        "--pdlc_inference_temperature",
        type=float,
        default=None,
        help=(
            "Optional override for PDLC inference-time temperature when using "
            "a checkpoint trained with the TabPDL head. Ignored for standard TabICL."
        ),
    )
    p.add_argument(
        "--pdlc_agg",
        type=str,
        default="posterior_avg",
        help=(
            "Optional override for PDLC aggregation mode at inference when using "
            "a checkpoint trained with the TabPDL head. Ignored for standard TabICL."
        ),
    )
    p.add_argument(
        "--pdlc_topk",
        type=int,
        default=None,
        help=(
            "Optional override for PDLC top-k gating at inference when using "
            "a checkpoint trained with the TabPDL head. Set <=0 to disable gating."
        ),
    )
    p.add_argument(
        "--pdlc_topk_tune",
        action="store_true",
        help=(
            "Tune PDLC top-k per dataset on a held-out subset of the fold-0 train split, "
            "then reuse the tuned k for all folds. Uses candidates derived from dataset size."
        ),
    )
    p.add_argument(
        "--pdlc_topk_tune_per_fold",
        action="store_true",
        help=(
            "Rigorous (nested) tuning: tune PDLC top-k separately inside each CV fold using only that fold's "
            "train split (inner support/val split), then evaluate on the fold's test split."
        ),
    )
    p.add_argument(
        "--pdlc_topk_tune_n_estimators",
        type=int,
        default=None,
        help=(
            "Ensemble size used during top-k tuning. If omitted, uses n_estimators for --pdlc_topk_tune_per_fold "
            "and uses 1 for dataset-level tuning."
        ),
    )
    p.add_argument(
        "--pdlc_topk_tune_metric",
        type=str,
        default="f1_macro",
        choices=["f1_macro", "accuracy", "roc_auc"],
        help="Metric optimized during top-k tuning.",
    )
    p.add_argument(
        "--pdlc_topk_tune_val_frac",
        type=float,
        default=0.2,
        help="Fraction of fold-0 train split used as tuning validation queries.",
    )
    p.add_argument(
        "--pdlc_topk_tune_abs",
        type=str,
        default="8,16,32,64,128",
        help="Comma-separated absolute k candidates (clipped to support size).",
    )
    p.add_argument(
        "--pdlc_topk_tune_fracs",
        type=str,
        default="0,0.001,0.002,0.005,0.01,0.02,0.05",
        help=(
            "Comma-separated fractional k candidates (k=round(frac*N_support)); "
            "use 0 to include the 'no top-k gating' candidate."
        ),
    )
    p.add_argument(
        "--pdlc_topk_tune_cache",
        type=str,
        default=None,
        help="Optional JSON cache path to store tuned best_frac per dataset_id for resumable runs.",
    )
    # Deprecated: PDLC symmetrization is no longer supported and this flag is ignored.
    # p.add_argument(
    #     "--pdlc_symmetrize",
    #     type=_str2bool,
    #     default=None,
    #     help=(
    #         "Optional override for PDLC symmetrization at inference when using "
    #         "a checkpoint trained with the TabPDL head. When omitted, the checkpoint "
    #         "setting is used; when set, it forces symmetrization on or off."
    #     ),
    # )
    p.add_argument("--limit", type=int, default=None, help="Limit number of tasks for quick runs")
    p.add_argument("--folds", type=int, default=10, help="Number of CV folds to run (default: 10)")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: results/cc18_<model>[_<ckptstem>].csv)",
    )
    p.add_argument(
        "--csv_postfix",
        type=str,
        default="",
        help="String appended to default CSV filename after checkpoint stem (e.g. '_run1')",
    )
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    p.add_argument("--wandb_project", type=str, default="tabicl_cc18", help="Weights & Biases project name")
    p.add_argument("--wandb_run_name", type=str, default=None, help="Optional Weights & Biases run name")
    p.add_argument("--n_rows", type=int, default=50000, help="Skip datasets with more than this many rows")
    p.add_argument("--max_features", type=int, default=500, help="Skip datasets with more than this many features")
    p.add_argument("--max_classes", type=int, default=10, help="Skip datasets with more than this many classes")
    p.add_argument("--seed", type=int, default=42, help="Global random seed for reproducibility")
    return p.parse_args()


def resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def encode_non_numeric_for_tabpfn(X_df: pd.DataFrame) -> pd.DataFrame:
    X = X_df.copy()
    for col in X.select_dtypes(include=["category", "object", "bool"]).columns:
        X[col] = X[col].astype("category").cat.codes
    return X


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _is_oom_error(e: BaseException) -> bool:
    s = str(e).lower()
    return isinstance(e, torch.OutOfMemoryError) or ("out of memory" in s) or ("cuda out of memory" in s) or ("hip out of memory" in s)


def _cleanup_after_oom() -> None:
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _parse_int_list(csv: str) -> List[int]:
    out: List[int] = []
    for s in str(csv).split(","):
        s = s.strip()
        if not s:
            continue
        out.append(int(s))
    return out


def _parse_float_list(csv: str) -> List[float]:
    out: List[float] = []
    for s in str(csv).split(","):
        s = s.strip()
        if not s:
            continue
        out.append(float(s))
    return out


def _dedup_preserve_order(seq: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for x in seq:
        key = ("__none__" if x is None else x)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _tune_pdlc_topk_for_task_fold0(
    *,
    X_train0: pd.DataFrame,
    y_train0: pd.Series,
    device: str,
    checkpoint: str,
    batch_size: int,
    elliptical_scale_boost: float,
    pdlc_agg: Optional[str],
    pdlc_inference_temperature: Optional[float],
    metric: str,
    val_frac: float,
    abs_candidates: List[int],
    frac_candidates: List[float],
    seed: int,
    n_estimators_tune: int,
) -> Dict[str, Any]:
    """Tune PDLC top-k on a fold train split (inner support/val split).

    Returns `best_frac` intended to be applied as:
      k_fold = round(best_frac * N_train_fold)
    so that the tuned setting naturally scales with the fold train size.
    """
    from tabicl import TabICLClassifier
    from sklearn.model_selection import train_test_split

    if not (0.0 < float(val_frac) < 1.0):
        raise ValueError(f"pdlc_topk_tune_val_frac must be in (0,1), got {val_frac}")

    # Split fold-0 train into anchors (support) and validation queries.
    try:
        Xa, Xv, ya, yv = train_test_split(
            X_train0,
            y_train0,
            test_size=float(val_frac),
            random_state=int(seed),
            stratify=y_train0,
        )
    except Exception:
        Xa, Xv, ya, yv = train_test_split(
            X_train0,
            y_train0,
            test_size=float(val_frac),
            random_state=int(seed),
            shuffle=True,
        )

    N_train0 = int(len(X_train0))
    N_support = int(len(Xa))

    # Candidates are defined relative to the *fold train size* (N_train0), then
    # clipped to the actual support size (N_support) during inner-tuning.
    #
    # This avoids an implicit (1 - val_frac) scaling that would otherwise change
    # the meaning of "k as a fraction of available supports".
    candidates_total: List[Optional[int]] = []
    # Fractions: 0 -> None (no gating)
    for frac in frac_candidates:
        if float(frac) <= 0:
            candidates_total.append(None)
            continue
        k_total = int(round(float(frac) * float(N_train0)))
        k_total = max(1, min(N_train0, k_total))
        candidates_total.append(k_total)
    # Absolutes (also in terms of fold train size)
    for k in abs_candidates:
        if int(k) <= 0:
            candidates_total.append(None)
        else:
            candidates_total.append(min(N_train0, int(k)))
    candidates_total = _dedup_preserve_order(candidates_total)

    clf = TabICLClassifier(
        device=device,
        model_path=checkpoint,
        allow_auto_download=False,
        use_hierarchical=True,
        n_estimators=int(n_estimators_tune),
        batch_size=int(batch_size),
        elliptical_scale_boost=float(elliptical_scale_boost),
        random_state=int(seed),
        pdlc_agg=pdlc_agg,
        pdlc_inference_temperature=pdlc_inference_temperature,
        pdlc_topk=None,
    )
    clf.fit(Xa, ya)

    # If this isn't a TabPDL checkpoint, nothing to tune.
    if getattr(clf, "icl_head", "tabicl") != "tabpdl":
        return {"best_frac": None, "best_k_support": None, "note": "Checkpoint is not TabPDL (icl_head!=tabpdl)."}

    pdlc_head = getattr(getattr(getattr(clf, "model_", None), "icl_predictor", None), "pdlc_head", None)
    if pdlc_head is None:
        return {"best_frac": None, "best_k_support": None, "note": "Model has no pdlc_head; cannot tune."}

    metric_key = str(metric)
    rows: List[Dict[str, Any]] = []
    best_score = float("-inf")
    best_acc = float("-inf")
    best_k_total: Optional[int] = None

    old_topk = getattr(pdlc_head.cfg, "topk", None)
    try:
        for k_total in candidates_total:
            k_support = None if k_total is None else min(int(k_total), int(N_support))
            pdlc_head.cfg.topk = k_support
            y_pred = clf.predict(Xv)
            y_prob = clf.predict_proba(Xv)
            m = compute_metrics(yv, y_pred, y_prob, classes=clf.classes_)
            score = float(m.get(metric_key, float("nan")))
            acc = float(m.get("accuracy", float("nan")))
            rows.append(
                {
                    "topk_train": k_total,
                    "topk_support_applied": k_support,
                    "train_n": N_train0,
                    "support_n": N_support,
                    "val_n": int(len(Xv)),
                    **m,
                }
            )

            k_rank = int(k_total) if k_total is not None else -1
            best_rank = int(best_k_total) if best_k_total is not None else -1
            if (score > best_score) or (score == best_score and acc > best_acc) or (
                score == best_score and acc == best_acc and k_rank < best_rank
            ):
                best_score = float(score)
                best_acc = float(acc)
                best_k_total = k_total
    finally:
        pdlc_head.cfg.topk = old_topk

    best_frac = None if best_k_total is None else float(best_k_total) / float(N_train0)
    return {
        "best_frac": best_frac,
        "best_k_train": best_k_total,
        "train_n": N_train0,
        "support_n": N_support,
        "val_n": int(len(Xv)),
        "metric": metric_key,
        "rows": rows,
        "note": (
            "best_frac is defined over the fold train size (N_train0) and applied as "
            "k=round(best_frac*N_train_fold) (clipped to [1,N_train_fold]). During tuning we apply "
            "min(k, N_support) because the inner support set is smaller due to the held-out val split."
        ),
    }


def evaluate_task_tabicl(
    task_id: int,
    device: str,
    checkpoint: str,
    n_estimators: int,
    batch_size: int,
    elliptical_scale_boost: float,
    pdlc_agg: Optional[str],
    pdlc_inference_temperature: Optional[float],
    pdlc_topk: Optional[int],
    pdlc_topk_tune: bool,
    pdlc_topk_tune_metric: str,
    pdlc_topk_tune_val_frac: float,
    pdlc_topk_tune_abs: str,
    pdlc_topk_tune_fracs: str,
    pdlc_topk_tune_cache: Optional[str],
    pdlc_topk_tune_per_fold: bool,
    pdlc_topk_tune_n_estimators: Optional[int],
    n_rows: int,
    max_features: int,
    max_classes: int,
    seed: int,
    folds: int,
) -> List[Dict[str, Any]]:
    from tabicl import TabICLClassifier

    task = openml.tasks.get_task(task_id)
    dataset = openml.datasets.get_dataset(task.dataset_id)

    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")

    # Always fit a LabelEncoder for consistent metadata
    le = __import__("sklearn.preprocessing", fromlist=["LabelEncoder"]).LabelEncoder()
    y_enc = le.fit_transform(y)

    metadata = compute_dataset_metadata(X.copy(), y_enc, le, dataset)
    meta_cols = {
        "n_rows": int(metadata.get("n_samples", X.shape[0])),
        "n_features": int(metadata.get("n_features", X.shape[1])),
        "n_classes": int(metadata.get("n_classes", len(np.unique(y_enc)))),
        "imbalance_ratio": float(metadata.get("imbalance_ratio", np.nan)),
    }

    # Dataset-level filters
    if n_rows is not None and len(X) > n_rows:
        print(f"[SKIP] Task {task_id}: rows {len(X)} > n_rows {n_rows}")
        return [
            {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "fold": None,
                "n_valid_folds": 0,
                "error": f"Dataset has {len(X)} rows, exceeds limit of {n_rows}",
                **meta_cols,
                "seed": int(seed),
            }
        ]
    if max_features is not None and X.shape[1] > max_features:
        print(f"[SKIP] Task {task_id}: features {X.shape[1]} > max_features {max_features}")
        return [
            {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "fold": None,
                "n_valid_folds": 0,
                "error": f"Dataset has {X.shape[1]} features, exceeds limit of {max_features}",
                **meta_cols,
                "seed": int(seed),
            }
        ]
    n_cls = len(np.unique(y))
    if max_classes is not None and n_cls > max_classes:
        print(f"[SKIP] Task {task_id}: classes {n_cls} > max_classes {max_classes}")
        return [
            {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "fold": None,
                "n_valid_folds": 0,
                "error": f"Dataset has {n_cls} classes, exceeds limit of {max_classes}",
                **meta_cols,
                "seed": int(seed),
            }
        ]

    fold_rows: List[Dict[str, Any]] = []
    valid_folds = 0

    tuned_best_frac_dataset: Optional[float] = None
    # Dataset-level tuning (fast, but not fully rigorous): tune on fold-0 train, reuse for all folds.
    if bool(pdlc_topk_tune) and not bool(pdlc_topk_tune_per_fold):
        cache_path = None if pdlc_topk_tune_cache is None else Path(str(pdlc_topk_tune_cache))
        cache: Dict[str, Any] = {} if cache_path is None else _load_json(cache_path)
        cache_key = str(int(dataset.dataset_id))
        if cache_key in cache:
            tuned_best_frac_dataset = cache[cache_key].get("best_frac")
            print(f"[TOPK-TUNE] Dataset {dataset.dataset_id} cached best_frac={tuned_best_frac_dataset}")
        else:
            try:
                train_idx0, _ = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
                n_est_tune = int(pdlc_topk_tune_n_estimators) if pdlc_topk_tune_n_estimators is not None else 1
                tuned = _tune_pdlc_topk_for_task_fold0(
                    X_train0=X.iloc[train_idx0],
                    y_train0=y.iloc[train_idx0],
                    device=device,
                    checkpoint=checkpoint,
                    batch_size=int(batch_size),
                    elliptical_scale_boost=float(elliptical_scale_boost),
                    pdlc_agg=pdlc_agg,
                    pdlc_inference_temperature=pdlc_inference_temperature,
                    metric=str(pdlc_topk_tune_metric),
                    val_frac=float(pdlc_topk_tune_val_frac),
                    abs_candidates=_parse_int_list(pdlc_topk_tune_abs),
                    frac_candidates=_parse_float_list(pdlc_topk_tune_fracs),
                    seed=int(seed),
                    n_estimators_tune=n_est_tune,
                )
                tuned_best_frac_dataset = tuned.get("best_frac")
                print(
                    f"[TOPK-TUNE] Dataset {dataset.dataset_id} best_frac={tuned_best_frac_dataset} "
                    f"(best_k_support={tuned.get('best_k_support')}, support_n={tuned.get('support_n')})"
                )
                if cache_path is not None:
                    cache[cache_key] = tuned
                    _save_json(cache_path, cache)
            except Exception as e:
                print(f"[WARN] Top-k tuning failed for dataset {dataset.dataset_id}: {e}")
    n_folds = int(folds)
    if not (1 <= n_folds <= 10):
        raise ValueError(f"--folds must be in [1,10], got {n_folds}")
    for fold in range(n_folds):
        try:
            train_idx, test_idx = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        except Exception as e:
            print(f"[SKIP] Task {task_id} fold {fold+1}: split error: {e}")
            continue

        print(f"  Running Task {task_id} Fold {fold+1}/{n_folds}...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if len(np.unique(y_train)) < len(np.unique(y)):
            print(f"[SKIP] Task {task_id} fold {fold+1}: missing classes in train.")
            continue

        # Determine effective top-k for this fold.
        pdlc_topk_eff: Optional[int] = pdlc_topk
        tuned_best_frac_fold: Optional[float] = None

        if bool(pdlc_topk_tune):
            if bool(pdlc_topk_tune_per_fold):
                # Rigorous (nested) per-fold tuning: tune on this fold's train split only.
                cache_path = None if pdlc_topk_tune_cache is None else Path(str(pdlc_topk_tune_cache))
                cache: Dict[str, Any] = {} if cache_path is None else _load_json(cache_path)
                cache_key = f"{int(dataset.dataset_id)}:{int(fold)}"
                if cache_key in cache:
                    tuned_best_frac_fold = cache[cache_key].get("best_frac")
                    print(f"[TOPK-TUNE] Dataset {dataset.dataset_id} fold {fold}: cached best_frac={tuned_best_frac_fold}")
                else:
                    try:
                        n_est_tune = int(pdlc_topk_tune_n_estimators) if pdlc_topk_tune_n_estimators is not None else int(n_estimators)
                        tuned = _tune_pdlc_topk_for_task_fold0(
                            X_train0=X_train,
                            y_train0=y_train,
                            device=device,
                            checkpoint=checkpoint,
                            batch_size=int(batch_size),
                            elliptical_scale_boost=float(elliptical_scale_boost),
                            pdlc_agg=pdlc_agg,
                            pdlc_inference_temperature=pdlc_inference_temperature,
                            metric=str(pdlc_topk_tune_metric),
                            val_frac=float(pdlc_topk_tune_val_frac),
                            abs_candidates=_parse_int_list(pdlc_topk_tune_abs),
                            frac_candidates=_parse_float_list(pdlc_topk_tune_fracs),
                            seed=int(seed) + int(fold),
                            n_estimators_tune=n_est_tune,
                        )
                        tuned_best_frac_fold = tuned.get("best_frac")
                        print(
                            f"[TOPK-TUNE] Dataset {dataset.dataset_id} fold {fold}: best_frac={tuned_best_frac_fold} "
                            f"(best_k_support={tuned.get('best_k_support')}, support_n={tuned.get('support_n')})"
                        )
                        if cache_path is not None:
                            cache[cache_key] = tuned
                            _save_json(cache_path, cache)
                    except Exception as e:
                        print(f"[WARN] Top-k tuning failed for dataset {dataset.dataset_id} fold {fold}: {e}")
                        tuned_best_frac_fold = None
            else:
                tuned_best_frac_fold = tuned_best_frac_dataset

        if tuned_best_frac_fold is not None:
            N_fold = int(len(train_idx))
            k = int(round(float(tuned_best_frac_fold) * float(N_fold)))
            k = max(1, min(N_fold, k))
            pdlc_topk_eff = int(k)

        clf = TabICLClassifier(
            device=device,
            model_path=checkpoint,
            allow_auto_download=False,
            use_hierarchical=True,
            n_estimators=n_estimators,
            batch_size=int(batch_size),
            elliptical_scale_boost=elliptical_scale_boost,
            random_state=seed,
            pdlc_agg=pdlc_agg,
            pdlc_inference_temperature=pdlc_inference_temperature,
            pdlc_topk=pdlc_topk_eff,
        )
        t0 = time.perf_counter()
        try:
            clf.fit(X_train, y_train)
            t1 = time.perf_counter()
            y_pred = clf.predict(X_test)
            t2 = time.perf_counter()
            y_prob = clf.predict_proba(X_test)
            t3 = time.perf_counter()
        except Exception as e:
            if _is_oom_error(e):
                _cleanup_after_oom()
                print(f"[OOM] Task {task_id} fold {fold+1}: {e}")
                fold_rows.append(
                    {
                        "task_id": task_id,
                        "dataset_id": int(dataset.dataset_id),
                        "dataset_name": dataset.name,
                        "fold": int(fold),
                        "task_type": "Binary" if len(np.unique(y)) == 2 else "Multiclass",
                        **meta_cols,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "error": f"OOM: {e}",
                        "seed": int(seed),
                        "pdlc_topk_effective": (None if pdlc_topk_eff is None else int(pdlc_topk_eff)),
                        "pdlc_topk_tuned_best_frac": tuned_best_frac_fold,
                    }
                )
                continue
            raise

        metrics = compute_metrics(y_test, y_pred, y_prob, classes=clf.classes_)
        metrics.update(
            {
                "fit_seconds": t1 - t0,
                "predict_seconds": (t3 - t2),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
        row: Dict[str, Any] = {
            "task_id": task_id,
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": dataset.name,
            "fold": int(fold),
            "task_type": "Binary" if len(np.unique(y)) == 2 else "Multiclass",
            **meta_cols,
            **metrics,
            "seed": int(seed),
            "pdlc_topk_effective": (None if pdlc_topk_eff is None else int(pdlc_topk_eff)),
            "pdlc_topk_tuned_best_frac": tuned_best_frac_fold,
        }
        fold_rows.append(row)
        valid_folds += 1

    if valid_folds == 0 or not fold_rows:
        return [
            {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "fold": None,
                "n_valid_folds": 0,
                "error": "No valid folds",
                **meta_cols,
                "seed": int(seed),
            }
        ]

    for row in fold_rows:
        row["n_valid_folds"] = int(valid_folds)
    return fold_rows


def evaluate_task_tabpfn(
    task_id: int,
    device: str,
    n_rows: int,
    max_features: int,
    max_classes: int,
    seed: int,
    folds: int,
) -> List[Dict[str, Any]]:
    try:
        from tabpfn import TabPFNClassifier  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "tabpfn package is not installed. Install with `pip install tabpfn` to run TabPFN benchmark."
        ) from e

    task = openml.tasks.get_task(task_id)
    dataset = openml.datasets.get_dataset(task.dataset_id)

    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")

    # TabPFN expects numerical X and typically no NaNs (default benchmark)
    if X.isna().any().any():
        return [
            {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "fold": None,
                "n_valid_folds": 0,
                "error": "Dataset contains missing values (TabPFN default benchmark does not impute)",
                "seed": int(seed),
            }
        ]

    # Dataset-level filters
    if n_rows is not None and len(X) > n_rows:
        print(f"[SKIP] Task {task_id}: rows {len(X)} > n_rows {n_rows}")
        return [
            {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "fold": None,
                "n_valid_folds": 0,
                "error": f"Dataset has {len(X)} rows, exceeds limit of {n_rows}",
                "seed": int(seed),
            }
        ]
    if max_features is not None and X.shape[1] > max_features:
        print(f"[SKIP] Task {task_id}: features {X.shape[1]} > max_features {max_features}")
        return [
            {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "fold": None,
                "n_valid_folds": 0,
                "error": f"Dataset has {X.shape[1]} features, exceeds limit of {max_features}",
                "seed": int(seed),
            }
        ]

    le = __import__("sklearn.preprocessing", fromlist=["LabelEncoder"]).LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = np.unique(y_enc)

    if max_classes is not None and len(classes) > max_classes:
        print(f"[SKIP] Task {task_id}: classes {len(classes)} > max_classes {max_classes}")
        return [
            {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "fold": None,
                "n_valid_folds": 0,
                "error": f"Dataset has {len(classes)} classes, exceeds limit of {max_classes}",
                "seed": int(seed),
            }
        ]

    metadata = compute_dataset_metadata(X.copy(), y_enc, le, dataset)
    meta_cols = {
        "n_rows": int(metadata.get("n_samples", X.shape[0])),
        "n_features": int(metadata.get("n_features", X.shape[1])),
        "n_classes": int(metadata.get("n_classes", len(classes))),
        "imbalance_ratio": float(metadata.get("imbalance_ratio", np.nan)),
    }

    fold_rows: List[Dict[str, Any]] = []
    valid_folds = 0
    n_folds = int(folds)
    if not (1 <= n_folds <= 10):
        raise ValueError(f"--folds must be in [1,10], got {n_folds}")
    for fold in range(n_folds):
        try:
            train_idx, test_idx = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        except Exception as e:
            print(f"[SKIP] Task {task_id} fold {fold+1}: split error: {e}")
            continue

        # Fit OrdinalEncoder on train fold only to avoid leakage
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OrdinalEncoder

        X_train_df, X_test_df = X.iloc[train_idx], X.iloc[test_idx]
        cat_cols = list(X_train_df.select_dtypes(include=["category", "object", "bool"]).columns)
        cat_pos = [X_train_df.columns.get_loc(c) for c in cat_cols]

        ct = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OrdinalEncoder(
                        dtype=np.int64,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                    cat_pos,
                )
            ],
            remainder="passthrough",
        )
        X_train_enc = ct.fit_transform(X_train_df)
        X_test_enc = ct.transform(X_test_df)

        X_train, X_test = X_train_enc.astype(np.float32), X_test_enc.astype(np.float32)
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        if len(np.unique(y_train)) < len(classes):
            print(f"[SKIP] Task {task_id} fold {fold+1}: missing classes in train.")
            continue

        clf = TabPFNClassifier(device=device, random_state=seed)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        t1 = time.perf_counter()
        y_pred = clf.predict(X_test)
        t2 = time.perf_counter()
        y_prob = clf.predict_proba(X_test)
        t3 = time.perf_counter()

        metrics = compute_metrics(y_test, y_pred, y_prob, classes=classes)
        metrics.update(
            {
                "fit_seconds": t1 - t0,
                "predict_seconds": (t3 - t2),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
        row: Dict[str, Any] = {
            "task_id": task_id,
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": dataset.name,
            "fold": int(fold),
            "task_type": "Binary" if len(classes) == 2 else "Multiclass",
            **meta_cols,
            **metrics,
            "seed": int(seed),
        }
        fold_rows.append(row)
        valid_folds += 1

    if valid_folds == 0 or not fold_rows:
        return [
            {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "fold": None,
                "n_valid_folds": 0,
                "error": "No valid folds",
                **meta_cols,
                "seed": int(seed),
            }
        ]

    for row in fold_rows:
        row["n_valid_folds"] = int(valid_folds)
    return fold_rows


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_global_seed(args.seed)
    print(f"Using device: {device}")

    wandb_run = None
    if args.wandb:
        if not HAVE_WANDB:
            raise RuntimeError("wandb is not installed. Install with `pip install wandb` or omit --wandb.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    suite = openml.study.get_suite(99)  # OpenML-CC18
    task_ids = list(suite.tasks)
    if args.limit is not None:
        task_ids = task_ids[: args.limit]
    print(f"OpenML-CC18: evaluating {len(task_ids)} tasks with {int(args.folds)}-fold CV")

    results = []
    for i, tid in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] Task {tid}")
        try:
            if args.model == "tabicl":
                res_rows = evaluate_task_tabicl(
                    tid,
                    device=device,
                    checkpoint=args.checkpoint,
                    n_estimators=args.n_estimators,
                    batch_size=args.batch_size,
                    elliptical_scale_boost=args.elliptical_scale_boost,
                    pdlc_agg=args.pdlc_agg,
                    pdlc_inference_temperature=args.pdlc_inference_temperature,
                    pdlc_topk=args.pdlc_topk,
                    pdlc_topk_tune=bool(args.pdlc_topk_tune),
                    pdlc_topk_tune_metric=str(args.pdlc_topk_tune_metric),
                    pdlc_topk_tune_val_frac=float(args.pdlc_topk_tune_val_frac),
                    pdlc_topk_tune_abs=str(args.pdlc_topk_tune_abs),
                    pdlc_topk_tune_fracs=str(args.pdlc_topk_tune_fracs),
                    pdlc_topk_tune_cache=args.pdlc_topk_tune_cache,
                    pdlc_topk_tune_per_fold=bool(args.pdlc_topk_tune_per_fold),
                    pdlc_topk_tune_n_estimators=args.pdlc_topk_tune_n_estimators,
                    n_rows=args.n_rows,
                    max_features=args.max_features,
                    max_classes=args.max_classes,
                    seed=args.seed,
                    folds=int(args.folds),
                )
            else:
                res_rows = evaluate_task_tabpfn(
                    tid,
                    device=device,
                    n_rows=args.n_rows,
                    max_features=args.max_features,
                    max_classes=args.max_classes,
                    seed=args.seed,
                    folds=int(args.folds),
                )
        except Exception as e:
            res_rows = [
                {
                    "task_id": tid,
                    "dataset_id": None,
                    "dataset_name": "N/A",
                    "fold": None,
                    "n_valid_folds": 0,
                    "error": str(e),
                    "seed": int(args.seed),
                }
            ]
        if args.wandb and HAVE_WANDB:
            for row in res_rows:
                wandb.log(row)
        results.extend(res_rows)

    df = pd.DataFrame(results)
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output is not None:
        out_path = Path(args.output)
    else:
        if args.model == "tabicl":
            ckptstem = Path(args.checkpoint).stem
            postfix = args.csv_postfix or ""
            out_path = out_dir / f"cc18_tabicl_{ckptstem}{postfix}.csv"
        else:
            out_path = out_dir / "cc18_tabpfn.csv"

    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
