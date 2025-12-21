"""
OpenML-CC18 10-fold CV Benchmark for ExtraTrees vs. PDL-ExtraTrees

This script benchmarks two models on the OpenML-CC18 suite using the official
10-fold cross-validation splits for each task:

- ExtraTrees: scikit-learn ExtraTreesClassifier with class_weight="balanced".
- PDL-ExtraTrees: PairwiseDifferenceClassifier wrapping ExtraTreesClassifier.

Results are aggregated per task (mean/std across folds) and saved to CSV in a
similar format and location as openml_cc18_benchmark.py.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import openml
import pandas as pd
from pdll import PairwiseDifferenceClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

from benchmark_utils import compute_metrics, compute_dataset_metadata

# Suppress common warnings from openml and sklearn for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark ExtraTrees vs PDL-ExtraTrees on OpenML-CC18 with 10-fold CV"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of OpenML-CC18 tasks (for quick runs)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: results/cc18_pdl-extratrees.csv)",
    )
    p.add_argument(
        "--csv_postfix",
        type=str,
        default="",
        help="String appended to default CSV filename (e.g. '_run1')",
    )
    p.add_argument(
        "--n_rows",
        type=int,
        default=10000,
        help="Skip datasets with more than this many rows",
    )
    p.add_argument(
        "--max_features",
        type=int,
        default=500,
        help="Skip datasets with more than this many features",
    )
    p.add_argument(
        "--max_classes",
        type=int,
        default=10,
        help="Skip datasets with more than this many classes",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility",
    )
    return p.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def evaluate_task(
    model: Any,
    model_name: str,
    task_id: int,
    n_rows: int,
    max_features: int,
    max_classes: int,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate a single model on one OpenML-CC18 task with 10-fold CV."""
    print(f"-- Task {task_id} | Model: {model_name}")

    try:
        task = openml.tasks.get_task(task_id)
        dataset = openml.datasets.get_dataset(task.dataset_id)

        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )

        # ExtraTrees/PDL do not handle NaNs; skip such datasets.
        if X.isna().any().any():
            return {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "model": model_name,
                "n_valid_folds": 0,
                "error": "Dataset contains missing values (no imputation for ExtraTrees/PDL).",
            }

        # Dataset-level filters (aligned with openml_cc18_benchmark.py)
        if n_rows is not None and len(X) > n_rows:
            print(f"[SKIP] Task {task_id}: rows {len(X)} > n_rows {n_rows}")
            return {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "model": model_name,
                "n_valid_folds": 0,
                "error": f"Dataset has {len(X)} rows, exceeds limit of {n_rows}",
            }

        if max_features is not None and X.shape[1] > max_features:
            print(
                f"[SKIP] Task {task_id}: features {X.shape[1]} > max_features {max_features}"
            )
            return {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "model": model_name,
                "n_valid_folds": 0,
                "error": f"Dataset has {X.shape[1]} features, exceeds limit of {max_features}",
            }

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        classes = np.unique(y_enc)

        if max_classes is not None and len(classes) > max_classes:
            print(
                f"[SKIP] Task {task_id}: classes {len(classes)} > max_classes {max_classes}"
            )
            return {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "model": model_name,
                "n_valid_folds": 0,
                "error": f"Dataset has {len(classes)} classes, exceeds limit of {max_classes}",
            }

        # Metadata (no missingness fields; aligned with benchmark_utils usage elsewhere)
        metadata = compute_dataset_metadata(X.copy(), y_enc, le, dataset)
        meta_cols = {
            "n_rows": int(metadata.get("n_samples", X.shape[0])),
            "n_features": int(metadata.get("n_features", X.shape[1])),
            "n_classes": int(metadata.get("n_classes", len(classes))),
            "imbalance_ratio": float(metadata.get("imbalance_ratio", np.nan)),
        }

        # Encode non-numeric columns for modeling
        X_model = X.copy()
        for col in X_model.select_dtypes(["category", "object", "bool"]).columns:
            X_model[col] = X_model[col].astype("category").cat.codes

        if X_model.isna().any().any():
            return {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "model": model_name,
                "n_valid_folds": 0,
                "error": "NaNs present after encoding categorical features.",
                **meta_cols,
            }

        X_np = np.asarray(X_model, dtype=np.float32)

        fold_results = []
        valid_folds = 0
        for fold in range(10):
            try:
                train_idx, test_idx = task.get_train_test_split_indices(
                    repeat=0, fold=fold, sample=0
                )
            except Exception as e:
                print(f"[SKIP] Task {task_id} fold {fold+1}: split error: {e}")
                continue

            X_train, X_test = X_np[train_idx], X_np[test_idx]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]

            if len(np.unique(y_train)) < len(classes):
                print(
                    f"[SKIP] Task {task_id} fold {fold+1}: missing classes in training data."
                )
                continue

            # Ensure per-fold reproducibility if model uses random_state
            if hasattr(model, "random_state"):
                model.random_state = seed

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            fold_metrics = compute_metrics(y_test, y_pred, y_prob, classes)
            fold_results.append(fold_metrics)
            valid_folds += 1

        if valid_folds == 0:
            return {
                "task_id": task_id,
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": dataset.name,
                "model": model_name,
                "n_valid_folds": 0,
                "error": "No valid folds",
                **meta_cols,
            }

        results_df = pd.DataFrame(fold_results)
        task_type = "Binary" if len(classes) == 2 else "Multiclass"

        out: Dict[str, Any] = {
            "task_id": task_id,
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": dataset.name,
            "model": model_name,
            "n_valid_folds": int(valid_folds),
            "task_type": task_type,
            **meta_cols,
        }

        for metric in results_df.columns:
            out[f"{metric}_mean"] = float(results_df[metric].mean())
            out[f"{metric}_std"] = float(results_df[metric].std(ddof=1))

        out["seed"] = int(seed)
        return out

    except Exception as e:
        return {
            "task_id": task_id,
            "dataset_id": None,
            "dataset_name": "N/A",
            "model": model_name,
            "n_valid_folds": 0,
            "error": str(e),
        }


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    print("Using CPU-based models (ExtraTrees / PDL-ExtraTrees)")

    suite = openml.study.get_suite(99)  # OpenML-CC18
    task_ids = list(suite.tasks)
    if args.limit is not None:
        task_ids = task_ids[: args.limit]
    print(f"OpenML-CC18: evaluating {len(task_ids)} tasks with 10-fold CV")

    results = []
    for i, tid in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] Task {tid}")

        # 1. ExtraTrees
        extratrees_model = ExtraTreesClassifier(
            random_state=args.seed, n_jobs=-1, class_weight="balanced"
        )
        res_et = evaluate_task(
            extratrees_model,
            "ExtraTrees",
            tid,
            n_rows=args.n_rows,
            max_features=args.max_features,
            max_classes=args.max_classes,
            seed=args.seed,
        )
        results.append(res_et)

        # 2. PDL-ExtraTrees
        pdl_base_estimator = ExtraTreesClassifier(
            random_state=args.seed, n_jobs=-1, class_weight="balanced"
        )
        pdl_model = PairwiseDifferenceClassifier(estimator=pdl_base_estimator)
        res_pdl = evaluate_task(
            pdl_model,
            "PDL-ExtraTrees",
            tid,
            n_rows=args.n_rows,
            max_features=args.max_features,
            max_classes=args.max_classes,
            seed=args.seed,
        )
        results.append(res_pdl)

    if not results:
        print("No results were generated. Exiting.")
        return

    df = pd.DataFrame(results)
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output is not None:
        out_path = Path(args.output)
    else:
        postfix = args.csv_postfix or ""
        out_path = out_dir / f"cc18_pdl-extratrees{postfix}.csv"

    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
