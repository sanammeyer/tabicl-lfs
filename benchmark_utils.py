import openml
from typing import Any, Dict
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, log_loss, precision_score, recall_score
from sklearn.preprocessing import label_binarize
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder


def load_dataset(did:int) -> Dict[str, Any]:
    """
    Checks if a dataset is usable for the benchmark based on predefined criteria,
    using the specified method for checking task splits.
    """
    try:
        # 1. Apply dataset quality filtering
        dataset = openml.datasets.get_dataset(did, download_data=False)
        qualities = dataset.qualities
        if int(qualities['NumberOfMissingValues']) > 0:
            return {"status": "skip", "reason": "Has missing values."}
        if str(dataset.format).lower() == 'sparse_arff':
            return {"status": "skip", "reason": "Sparse ARFF format."}
        if int(qualities.get('MinorityClassSize', 0)) < 20:
             return {"status": "skip", "reason": f"Minority class size is too small ({qualities.get('MinorityClassSize', 0)})."}
        if float(qualities.get('MinorityClassPercentage', 0)) < 0.03:
            return {"status": "skip", "reason": f"Minority class ratio is too low ({qualities.get('MinorityClassPercentage', 0):.2f})."}

        # 2. Find and verify a suitable classification task using the specified logic
        all_tasks_df = openml.tasks.list_tasks(output_format="dataframe", data_id=did) #
        if all_tasks_df.empty: #
            return {"status": "skip", "reason": "No tasks found."} #
        
        class_tasks_df = all_tasks_df[all_tasks_df['task_type'] == 'Supervised Classification'] #
        if class_tasks_df.empty: #
            return {"status": "skip", "reason": "No classification task found."} #
            
        task_id = int(class_tasks_df.iloc[0]["tid"]) #
        task = openml.tasks.get_task(task_id, download_splits=False) #
        num_repeats, num_folds, _ = task.get_split_dimensions() #

        # 3. Final check to ensure it's a 10-fold, 1-repeat task
        if not (num_repeats == 1 and num_folds == 10):
            return {"status": "skip", "reason": f"First task found has {num_repeats} repeats and {num_folds} folds, not 1 and 10."}
        
        return {
            "status": "success",
            "did": did,
            "task_id": task_id,
            "dataset_name": dataset.name,
            "repeats": num_repeats,
            "folds": num_folds,
        }
        
    except Exception as e:
        return {"status": "error", "did": did, "reason": str(e)}
    
def compute_metrics(y_test: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    """Calculates a comprehensive set of classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
    }
    try:
        if len(classes) == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob[:, 1])
        else:
            y_test_binarized = label_binarize(y_test, classes=classes)
            metrics['roc_auc'] = roc_auc_score(y_test_binarized, y_prob, average='macro', multi_class='ovr')
        metrics['log_loss'] = log_loss(y_test, y_prob, labels=classes)
    except ValueError as e:
        print(f"Metric calculation error: {e}")
        metrics['roc_auc'] = np.nan
        metrics['log_loss'] = np.nan
    return metrics

def compute_dataset_metadata(X_df: pd.DataFrame, y_encoded: np.ndarray, le: LabelEncoder, dataset) -> Dict[str, Any]:
    """Compute dataset-level metadata without any missingness fields."""
    n_samples, n_features = X_df.shape
    num_cols = list(X_df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X_df.select_dtypes(include=["object", "category", "bool"]).columns)

    # Constant features (quick signal)
    nunique_per_col = X_df.nunique(dropna=False)
    n_constant = int((nunique_per_col <= 1).sum())

    # Class distribution & imbalance
    counts = Counter(y_encoded)
    class_labels = list(le.classes_)
    class_counts_named = {str(class_labels[k]): int(v) for k, v in counts.items()}
    majority = max(counts.values()) if counts else 0
    minority = min(counts.values()) if counts else 0
    imbalance_ratio = float(majority / minority) if minority > 0 and len(counts) > 1 else np.nan
    majority_fraction = float(majority / n_samples) if n_samples > 0 else np.nan

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_numeric_features": len(num_cols),
        "n_categorical_features": len(cat_cols),
        "numeric_feature_names": num_cols,
        "categorical_feature_names": cat_cols,
        "n_constant_features": n_constant,
        "n_classes": len(class_labels),
        "class_labels": [str(c) for c in class_labels],
        "class_counts": class_counts_named,
        "majority_class_fraction": majority_fraction,
        "imbalance_ratio": imbalance_ratio
    }
