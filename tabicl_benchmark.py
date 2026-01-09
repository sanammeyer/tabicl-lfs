"""
TabICL Benchmark Script with Official OpenML Splits (Strict)

This script evaluates TabICL (with your elliptical-attention checkpoint) on
OpenML datasets strictly using official 10-fold cross-validation tasks. If a
dataset does not have a usable official 10-fold, 1-repeat task, it is skipped.

Key features:
- Filters for and evaluates ONLY datasets with official 10-fold, 1-repeat OpenML tasks.
- Skips any dataset that cannot be evaluated using its official splits.
- Uses TabICLClassifier with your checkpoint at `checkpoints/step-1300.ckpt`.
- Calculates a comprehensive set of classification metrics.
- Aggregates results (mean and std) across all 10 folds for each dataset.
- Saves detailed results to a CSV file.
"""
from __future__ import annotations

import sys
from pathlib import Path
import warnings
from typing import Any, Dict, Optional

# Ensure local source package "src/tabicl" takes precedence over the venv folder "tabicl/"
ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import openml
import pandas as pd
import torch
from openml import tasks
from sklearn.preprocessing import LabelEncoder

from tabicl import TabICLClassifier
from benchmark_utils import load_dataset, compute_metrics, compute_dataset_metadata

# Suppress warnings from the OpenML library
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Helpers are imported from benchmark_utils.py

#Combined and de-duplicated list of dataset IDs.
# ALL_DATASET_IDS = sorted(list(set([
#     43, 48, 59, 61, 164, 333, 377, 444, 464, 475, 714, 717, 721, 733, 736,
#     744, 750, 756, 766, 767, 768, 773, 779, 782, 784, 788, 792, 793, 811,
#     812, 814, 824, 850, 853, 860, 863, 870, 873, 877, 879, 880, 889, 895,
#     896, 902, 906, 909, 911, 915, 918, 925, 932, 933, 935, 936, 937, 969,
#     973, 974, 1005, 1011, 1012, 1054, 1063, 1065, 1073, 1100, 1115, 1413,
#     1467, 1480, 1488, 1490, 1499, 1510, 1511, 1523, 1554, 1556, 1600, 4329,
#     40663, 40681, 41568, 41977, 41978, 42011, 42021, 42026, 42051, 42066,
#     42071, 42186, 42700, 43859, 44149, 44151, 44344, 45711,
#     1049, 1067, 12, 1464, 1475, 1487, 1489, 1494,
#     181, 188, 23, 31, 3, 40498, 40670, 40701,
#     40900, 40975, 40981, 40982, 40983, 40984,
#     41143, 41144, 41145, 41146, 41156, 4538, 54
# ])))


ALL_DATASET_IDS = sorted(list(set([
    475, 906, 909, 1100, 915, 1554, 23, 48, 714, 1475, 40663, 750, 41978, 45711
])))
#909, 1100, 915, 1554, 23, 48, 714, 1475, 40663, 750, 41978, 45711
def evaluate_dataset(did: int, task_id: int, device: str) -> Dict[str, Any]:
    """Evaluate TabICL on a single dataset using its official 10-fold CV task.
    Returns aggregated metrics + dataset metadata."""
    print(f"Processing DID: {did}, Task ID: {task_id}")

    try:
        dataset = openml.datasets.get_dataset(did)
        task = openml.tasks.get_task(task_id)

        # Keep DataFrame for clean dtype handling + metadata
        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format="dataframe"
        )

        # Encode target only for metadata; TabICLClassifier handles label encoding internally
        le = LabelEncoder().fit(y)
        y_enc = le.transform(y)

        # Preserve original X for metadata BEFORE any numeric encoding
        X_df_for_meta = X.copy()
        metadata = compute_dataset_metadata(X_df_for_meta, y_enc, le, dataset)
        meta_cols = {
            "n_rows": int(metadata.get("n_samples", X.shape[0])),
            "n_features": int(metadata.get("n_features", X.shape[1])),
            "n_classes": int(metadata.get("n_classes", len(np.unique(y_enc)))),
            "imbalance_ratio": float(metadata.get("imbalance_ratio", np.nan)),
        }

        # ---- 10-fold CV via official OpenML splits ----
        fold_results = []
        valid_folds = 0
        for fold in range(10):
            print(f"  Running Fold {fold+1}/10...")
            try:
                train_idx, test_idx = task.get_train_test_split_indices(fold=fold)
            except Exception as e:
                print(f"[SKIP] Could not get official split for fold {fold+1}. Error: {e}")
                continue

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Ensure all classes appear in train (required by TabICL's in-context learning)
            if len(np.unique(y_train)) < len(np.unique(y)):
                print(f"[SKIP] Fold {fold+1} is missing classes in the training split.")
                continue

            clf = TabICLClassifier(
                device=device,
                model_path=str(ROOT / "checkpoints" / "mini_tabicl_stage1_sa" / "step-25000.ckpt"),
                allow_auto_download=False,
                use_hierarchical=True,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)

            fold_metrics = compute_metrics(y_test, y_pred, y_prob, classes=clf.classes_)
            fold_results.append(fold_metrics)
            valid_folds += 1

        if valid_folds < 10:
            return {
                "did": did,
                "task_id": task_id,
                "dataset_name": dataset.name,
                "n_valid_folds": valid_folds,
                "error": f"Evaluation incomplete. Only {valid_folds}/10 folds were successful.",
                **meta_cols,
            }

        task_type = "Binary" if len(np.unique(y)) == 2 else "Multiclass"

        results_df = pd.DataFrame(fold_results)
        aggregated_results: Dict[str, Any] = {
            "did": did,
            "task_id": task_id,
            "dataset_name": dataset.name,
            "n_valid_folds": int(valid_folds),
            "task_type": task_type,
            **meta_cols,
        }
        for metric in results_df.columns:
            aggregated_results[f"{metric}_mean"] = float(results_df[metric].mean())
            aggregated_results[f"{metric}_std"]  = float(results_df[metric].std(ddof=1))
        return aggregated_results

    except Exception as e:
        return {"did": did, "task_id": task_id, "dataset_name": "N/A", "error": str(e)}

def main():
    """Main function to identify usable datasets and run the benchmark."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    print("--- Identifying usable datasets with 10-fold, 1-repeat CV tasks ---")
    usable_tasks = []
    skipped_tasks = []
    for did in ALL_DATASET_IDS:
        info = load_dataset(did)
        if info['status'] == 'success' and info['repeats'] == 1 and info['folds'] == 10:
            usable_tasks.append({'did': info['did'], 'task_id': info['task_id']})
        else:
            skipped_tasks.append({'did': did, 'reason': info.get('message', 'Does not have 10 folds / 1 repeat')})

    print(f"Found {len(usable_tasks)} datasets to evaluate.")
    if skipped_tasks:
        print(f"Skipped {len(skipped_tasks)} datasets that did not meet the criteria.\n")
    
    print("--- Step 2: Running benchmark ---")
    all_results = []
    for task_info in usable_tasks:
        dataset_result = evaluate_dataset(task_info['did'], task_info['task_id'], device)
        all_results.append(dataset_result)

    results_df = pd.DataFrame(all_results)
    output_path = "results/tabicl_benchmark_results_test_25000.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\nBenchmark finished. Results saved to {output_path}")
    print("\nResults summary (first 10 datasets):")

    # **FIX:** Check if the 'error' column exists before filtering.
    # This handles the case where all runs are successful.
    if 'error' in results_df.columns:
        successful_runs = results_df[results_df['error'].isna()]
    else:
        # If the column doesn't exist, it means there were no errors.
        successful_runs = results_df
        
    #Print average accuracy and f1_score across successful runs
    if not successful_runs.empty:
        avg_accuracy = successful_runs['accuracy_mean'].mean()
        avg_f1 = successful_runs['f1_macro_mean'].mean()
        print(f"Average Accuracy across successful runs: {avg_accuracy:.4f}")
        print(f"Average F1 Score across successful runs: {avg_f1:.4f}")
    else:
        print("No successful runs to report average metrics.")

if __name__ == "__main__":
    main()
