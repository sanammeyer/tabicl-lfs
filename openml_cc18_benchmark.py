"""
OpenML-CC18 10-fold CV Benchmark

This script benchmarks models on the OpenML-CC18 suite using official 10-fold
cross-validation splits for each task.

Supported models (select via --model):
- tabicl: Uses TabICLClassifier; pass --checkpoint to select checkpoint file.
- tabpfn: Uses TabPFNClassifier with default settings.

Results are aggregated per task (mean/std across folds) and saved to CSV.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import openml
import pandas as pd
import torch
import time

# Prefer local sources for tabicl (src/tabicl) over venv dir named 'tabicl'
ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

# Reuse helpers
from benchmark_utils import load_dataset as _unused_load_dataset  # kept for reference
from benchmark_utils import compute_metrics, compute_dataset_metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark OpenML-CC18 with 10-fold CV")
    p.add_argument("--model", choices=["tabicl", "tabpfn"], default="tabicl", help="Model to benchmark")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "checkpoints" / "step-1300.ckpt"),
        help="TabICL checkpoint path (only used for --model tabicl)",
    )
    p.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|cuda")
    p.add_argument("--n_estimators", type=int, default=32, help="TabICL ensemble size")
    p.add_argument("--elliptical_scale_boost", type=float, default=1.0, help="Extra multiplicative factor for elliptical scale (ICL)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of tasks for quick runs")
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
    p.add_argument("--n_rows", type=int, default=10000, help="Skip datasets with more than this many rows")
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


def evaluate_task_tabicl(
    task_id: int,
    device: str,
    checkpoint: str,
    n_estimators: int,
    elliptical_scale_boost: float,
    n_rows: int,
    max_features: int,
    max_classes: int,
    seed: int,
) -> Dict[str, Any]:
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
        return {
            "task_id": task_id,
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": dataset.name,
            "n_valid_folds": 0,
            "error": f"Dataset has {len(X)} rows, exceeds limit of {n_rows}",
            **meta_cols,
        }
    if max_features is not None and X.shape[1] > max_features:
        print(f"[SKIP] Task {task_id}: features {X.shape[1]} > max_features {max_features}")
        return {
            "task_id": task_id,
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": dataset.name,
            "n_valid_folds": 0,
            "error": f"Dataset has {X.shape[1]} features, exceeds limit of {max_features}",
            **meta_cols,
        }
    n_cls = len(np.unique(y))
    if max_classes is not None and n_cls > max_classes:
        print(f"[SKIP] Task {task_id}: classes {n_cls} > max_classes {max_classes}")
        return {
            "task_id": task_id,
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": dataset.name,
            "n_valid_folds": 0,
            "error": f"Dataset has {n_cls} classes, exceeds limit of {max_classes}",
            **meta_cols,
        }
    else:
        fold_results = []
        valid_folds = 0
        for fold in range(10):
            try:
                train_idx, test_idx = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
            except Exception as e:
                print(f"[SKIP] Task {task_id} fold {fold+1}: split error: {e}")
                continue
            
            print(f"  Running Task {task_id} Fold {fold+1}/10...")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if len(np.unique(y_train)) < len(np.unique(y)):
                print(f"[SKIP] Task {task_id} fold {fold+1}: missing classes in train.")
                continue

            clf = TabICLClassifier(
                device=device,
                model_path=checkpoint,
                allow_auto_download=False,
                use_hierarchical=True,
                n_estimators=n_estimators,
                elliptical_scale_boost=elliptical_scale_boost,
                random_state=seed,
            )
            t0 = time.perf_counter()
            clf.fit(X_train, y_train)
            t1 = time.perf_counter()
            y_pred = clf.predict(X_test)
            t2 = time.perf_counter()
            y_prob = clf.predict_proba(X_test)
            t3 = time.perf_counter()

            metrics = compute_metrics(y_test, y_pred, y_prob, classes=clf.classes_)
            metrics.update(
                {
                    "fit_seconds": t1 - t0,
                    "predict_seconds": (t3 - t2),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                }
            )
            fold_results.append(metrics)
            valid_folds += 1

    if valid_folds == 0:
        return {
            "task_id": task_id,
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": dataset.name,
            "n_valid_folds": 0,
            "error": "No valid folds",
            **meta_cols,
        }

    df = pd.DataFrame(fold_results)
    out: Dict[str, Any] = {
        "task_id": task_id,
        "dataset_id": int(dataset.dataset_id),
        "dataset_name": dataset.name,
        "n_valid_folds": int(valid_folds),
        "task_type": "Binary" if len(np.unique(y)) == 2 else "Multiclass",
        **meta_cols,
    }
    for c in df.columns:
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std(ddof=1))
    out["seed"] = int(seed)
    return out


def evaluate_task_tabpfn(task_id: int, device: str, n_rows: int, max_features: int, max_classes: int, seed: int) -> Dict[str, Any]:
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
        return {
            "task_id": task_id,
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": dataset.name,
            "n_valid_folds": 0,
            "error": "Dataset contains missing values (TabPFN default benchmark does not impute)",
        }

    # Dataset-level filters
    if n_rows is not None and len(X) > n_rows:
        print(f"[SKIP] Task {task_id}: rows {len(X)} > n_rows {n_rows}")
        return {
            "task_id": task_id,
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": dataset.name,
            "n_valid_folds": 0,
            "error": f"Dataset has {len(X)} rows, exceeds limit of {n_rows}",
        }
    if max_features is not None and X.shape[1] > max_features:
        print(f"[SKIP] Task {task_id}: features {X.shape[1]} > max_features {max_features}")
        return {
            "task_id": task_id,
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": dataset.name,
            "n_valid_folds": 0,
            "error": f"Dataset has {X.shape[1]} features, exceeds limit of {max_features}",
        }

    le = __import__("sklearn.preprocessing", fromlist=["LabelEncoder"]).LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = np.unique(y_enc)

    if max_classes is not None and len(classes) > max_classes:
        print(f"[SKIP] Task {task_id}: classes {len(classes)} > max_classes {max_classes}")
        return {
            "task_id": task_id,
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": dataset.name,
            "n_valid_folds": 0,
            "error": f"Dataset has {len(classes)} classes, exceeds limit of {max_classes}",
        }

    metadata = compute_dataset_metadata(X.copy(), y_enc, le, dataset)
    meta_cols = {
        "n_rows": int(metadata.get("n_samples", X.shape[0])),
        "n_features": int(metadata.get("n_features", X.shape[1])),
        "n_classes": int(metadata.get("n_classes", len(classes))),
        "imbalance_ratio": float(metadata.get("imbalance_ratio", np.nan)),
    }

    fold_results = []
    valid_folds = 0
    for fold in range(10):
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
        fold_results.append(metrics)
        valid_folds += 1

    if valid_folds == 0:
        return {
            "task_id": task_id,
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": dataset.name,
            "n_valid_folds": 0,
            "error": "No valid folds",
            **meta_cols,
        }

    df = pd.DataFrame(fold_results)
    out: Dict[str, Any] = {
        "task_id": task_id,
        "dataset_id": int(dataset.dataset_id),
        "dataset_name": dataset.name,
        "n_valid_folds": int(valid_folds),
        "task_type": "Binary" if len(classes) == 2 else "Multiclass",
        **meta_cols,
    }
    for c in df.columns:
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std(ddof=1))
    out["seed"] = int(seed)
    return out


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_global_seed(args.seed)
    print(f"Using device: {device}")

    suite = openml.study.get_suite(99)  # OpenML-CC18
    task_ids = list(suite.tasks)
    if args.limit is not None:
        task_ids = task_ids[: args.limit]
    print(f"OpenML-CC18: evaluating {len(task_ids)} tasks with 10-fold CV")

    results = []
    for i, tid in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] Task {tid}")
        try:
            if args.model == "tabicl":
                res = evaluate_task_tabicl(
                    tid,
                    device=device,
                    checkpoint=args.checkpoint,
                    n_estimators=args.n_estimators,
                    elliptical_scale_boost=args.elliptical_scale_boost,
                    n_rows=args.n_rows,
                    max_features=args.max_features,
                    max_classes=args.max_classes,
                    seed=args.seed,
                )
            else:
                res = evaluate_task_tabpfn(
                    tid,
                    device=device,
                    n_rows=args.n_rows,
                    max_features=args.max_features,
                    max_classes=args.max_classes,
                    seed=args.seed,
                )
        except Exception as e:
            res = {"task_id": tid, "dataset_id": None, "dataset_name": "N/A", "n_valid_folds": 0, "error": str(e)}
        results.append(res)

    df = pd.DataFrame(results)
    out_dir = ROOT / "results"
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


if __name__ == "__main__":
    main()
