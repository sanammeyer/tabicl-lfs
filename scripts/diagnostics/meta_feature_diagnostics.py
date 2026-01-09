#!/usr/bin/env python3
"""
Meta-feature diagnostics for TabICL pretrain priors vs OpenML-CC18 evaluation tasks.

This script computes light-weight task descriptors phi(D) for:
  - Synthetic pretrain tasks sampled from TabICL prior files on disk (Stage-1/Stage-2)
  - Real OpenML-CC18 datasets used in evaluation

It then:
  1) Saves Phi_pretrain and Phi_cc18 matrices to CSV
  2) Computes per-feature coverage / percentile reports
  3) Runs a repeated bootstrap two-sample classifier test (C2ST) with logistic regression

Usage (example, from repo root):
  cd tabicl-lfs
  python scripts/diagnostics/meta_feature_diagnostics.py \\
    --stage1_prior_dir /dss/lxclscratch/0E/ra63pux2/ra63pux2/tabicl_s1_priors \\
    --stage2_prior_dir /dss/lxclscratch/0E/ra63pux2/ra63pux2/mini_tabicl_s2_priors \\
    --stage1_max_batches 2500 \\
    --stage2_max_batches 1000 \\
    --max_pretrain_tasks 200000
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


# Make local src importable
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"

# Ensure both repo root (for benchmark_utils) and src/ (for tabicl) are importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark_utils import compute_dataset_metadata  # type: ignore
from tabicl.prior.genload import LoadPriorDataset  # type: ignore
from tabicl.sklearn.classifier import TabICLClassifier  # type: ignore


META_FEATURE_COLUMNS = (
    "n_rows",
    "n_features",
    "n_classes",
    "class_entropy",
    "class_entropy_norm",
    "max_class_fraction",
    "overall_missing_rate",
    "frac_cols_any_missing",
    "unique_ratio_mean",
    "unique_ratio_min",
    "unique_ratio_max",
    "numeric_abs_skew_mean",
    "numeric_kurtosis_mean",
    "numeric_heavy_tail_frac",
    "mean_abs_corr",
    "effective_rank",
    "n_numeric_features",
    "n_categorical_features",
    "frac_numeric_features",
    "frac_categorical_features",
)


class MetaFeatureConfig(object):
    """Configuration for meta-feature computation (sampling sizes, thresholds)."""

    def __init__(
        self,
        max_unique_cols=20,
        max_moment_cols=20,
        max_corr_cols=20,
        max_corr_rows=512,
        max_pca_rows=512,
        heavy_tail_excess_threshold=1.0,
    ):
        # Column sampling for expensive stats
        self.max_unique_cols = int(max_unique_cols)
        self.max_moment_cols = int(max_moment_cols)
        self.max_corr_cols = int(max_corr_cols)
        # Row sampling for dependence / PCA stats
        self.max_corr_rows = int(max_corr_rows)
        self.max_pca_rows = int(max_pca_rows)
        # Heavy-tail criterion on excess kurtosis
        self.heavy_tail_excess_threshold = float(heavy_tail_excess_threshold)


def _safe_entropy_from_counts(counts: np.ndarray) -> Tuple[float, float]:
    """Return (entropy, max_class_fraction) from integer class counts."""
    total = counts.sum()
    if total <= 0:
        return math.nan, math.nan
    probs = counts.astype(float) / float(total)
    probs = probs[probs > 0]
    if probs.size == 0:
        return math.nan, math.nan
    entropy = float(-np.sum(probs * np.log(probs)))
    max_frac = float(np.max(probs))
    return entropy, max_frac


def _sample_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.empty((0,), dtype=int)
    k = min(k, n)
    if k == n:
        return np.arange(n, dtype=int)
    return rng.choice(n, size=k, replace=False)


def _impute_nan_with_mean(X: np.ndarray) -> np.ndarray:
    """Impute NaNs with column means (0.0 if all-NaN)."""
    if X.size == 0:
        return X
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X_filled = np.where(np.isnan(X), col_means, X)
    return X_filled


def _numeric_distribution_features(
    X: np.ndarray,
    rng: np.random.Generator,
    cfg: MetaFeatureConfig,
) -> Tuple[float, float, float]:
    """Compute mean |skew|, mean excess kurtosis, and heavy-tail fraction."""
    n_rows, n_features = X.shape
    if n_rows <= 1 or n_features == 0:
        return math.nan, math.nan, math.nan

    col_idx = _sample_indices(n_features, cfg.max_moment_cols, rng)
    abs_skews: List[float] = []
    kurt_excess: List[float] = []

    for j in col_idx:
        col = X[:, j]
        col = col[~np.isnan(col)]
        if col.size <= 1:
            continue
        mean = float(col.mean())
        std = float(col.std(ddof=0))
        if std <= 1e-8:
            continue
        centered = col - mean
        m2 = float(np.mean(centered**2))
        m3 = float(np.mean(centered**3))
        m4 = float(np.mean(centered**4))
        if m2 <= 1e-12:
            continue
        skew = m3 / (m2 ** 1.5 + 1e-12)
        kurt = m4 / (m2**2 + 1e-12) - 3.0  # excess
        abs_skews.append(float(abs(skew)))
        kurt_excess.append(float(kurt))

    if not abs_skews:
        return math.nan, math.nan, math.nan

    abs_skew_mean = float(np.mean(abs_skews))
    kurt_mean = float(np.mean(kurt_excess))
    heavy_tail_frac = float(np.mean(np.array(kurt_excess) > cfg.heavy_tail_excess_threshold))
    return abs_skew_mean, kurt_mean, heavy_tail_frac


def _dependence_features(
    X: np.ndarray,
    rng: np.random.Generator,
    cfg: MetaFeatureConfig,
) -> Tuple[float, float]:
    """Compute mean absolute correlation and effective rank."""
    n_rows, n_features = X.shape
    if n_rows <= 1 or n_features <= 1:
        return math.nan, math.nan

    # Subsample rows and columns for correlation / PCA
    row_idx = _sample_indices(n_rows, cfg.max_corr_rows, rng)
    col_idx = _sample_indices(n_features, cfg.max_corr_cols, rng)
    X_sub = X[np.ix_(row_idx, col_idx)]
    # Impute NaNs
    X_sub = _impute_nan_with_mean(X_sub)

    # Drop constant columns
    stds = X_sub.std(axis=0, ddof=0)
    valid = stds > 1e-8
    if not np.any(valid):
        return math.nan, math.nan
    X_sub = X_sub[:, valid]

    # Correlation matrix
    if X_sub.shape[1] <= 1:
        mean_abs_corr = math.nan
    else:
        C = np.corrcoef(X_sub, rowvar=False)
        if np.any(np.isnan(C)):
            mean_abs_corr = math.nan
        else:
            off_diag = C[~np.eye(C.shape[0], dtype=bool)]
            mean_abs_corr = float(np.mean(np.abs(off_diag)))

    # Effective rank via SVD on centered X_sub
    X_centered = X_sub - X_sub.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(X_centered, full_matrices=False, compute_uv=False)
        eigvals = (s**2) / max(X_centered.shape[0] - 1, 1)
        eigvals = np.maximum(eigvals, 0.0)
        total = eigvals.sum()
        if total <= 0:
            eff_rank = math.nan
        else:
            p = eigvals / total
            p = p[p > 0]
            ent = -np.sum(p * np.log(p))
            eff_rank = float(np.exp(ent))
    except Exception:
        eff_rank = math.nan

    return mean_abs_corr, eff_rank


def compute_meta_features_from_array(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    cfg: MetaFeatureConfig,
    n_numeric_features: Optional[int] = None,
    n_categorical_features: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute phi(D) for a numeric dataset represented as arrays.

    Parameters
    ----------
    X : array-like, shape (n_rows, n_features)
    y : array-like, shape (n_rows,)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    n_rows, n_features = X.shape if X.ndim == 2 else (0, 0)
    # Class distribution (ignore NaNs and negative "ignore" labels)
    y_flat = y.reshape(-1)
    if np.issubdtype(y_flat.dtype, np.floating):
        mask = (~np.isnan(y_flat)) & (y_flat >= 0)
        y_valid = y_flat[mask]
    else:
        y_valid = y_flat[y_flat >= 0]

    if y_valid.size == 0:
        n_classes = 0
        class_entropy = math.nan
        max_class_fraction = math.nan
        class_entropy_norm = math.nan
    else:
        # Map to contiguous integer labels
        _, inv = np.unique(y_valid, return_inverse=True)
        counts = np.bincount(inv)
        n_classes = int(counts.size)
        class_entropy, max_class_fraction = _safe_entropy_from_counts(counts)
        if n_classes > 1 and not math.isnan(class_entropy):
            class_entropy_norm = float(class_entropy / math.log(n_classes))
        else:
            class_entropy_norm = math.nan

    # Missingness
    if n_rows > 0 and n_features > 0:
        missing_mask = np.isnan(X)
        overall_missing_rate = float(missing_mask.mean())
        frac_cols_any_missing = float(missing_mask.any(axis=0).mean())
    else:
        overall_missing_rate = math.nan
        frac_cols_any_missing = math.nan

    # Unique-ratio proxies (sample columns)
    if n_rows > 0 and n_features > 0:
        col_idx = _sample_indices(n_features, cfg.max_unique_cols, rng)
        unique_ratios: List[float] = []
        for j in col_idx:
            col = X[:, j]
            col = col[~np.isnan(col)]
            n_valid = col.size
            if n_valid == 0:
                continue
            nunique = np.unique(col).size
            unique_ratios.append(float(nunique) / float(n_valid))
        if unique_ratios:
            unique_ratio_mean = float(np.mean(unique_ratios))
            unique_ratio_min = float(np.min(unique_ratios))
            unique_ratio_max = float(np.max(unique_ratios))
        else:
            unique_ratio_mean = unique_ratio_min = unique_ratio_max = math.nan
    else:
        unique_ratio_mean = unique_ratio_min = unique_ratio_max = math.nan

    # Numeric distribution features
    abs_skew_mean, kurt_mean, heavy_tail_frac = _numeric_distribution_features(X, rng, cfg)

    # Dependence features
    mean_abs_corr, eff_rank = _dependence_features(X, rng, cfg)

    # Type counts (for priors everything is numeric)
    if n_numeric_features is None:
        n_numeric_features = n_features
    if n_categorical_features is None:
        n_categorical_features = 0
    total_features = max(n_numeric_features + n_categorical_features, 1)

    frac_numeric = float(n_numeric_features) / float(total_features)
    frac_categorical = float(n_categorical_features) / float(total_features)

    return {
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "n_classes": int(n_classes),
        "class_entropy": float(class_entropy),
        "class_entropy_norm": float(class_entropy_norm),
        "max_class_fraction": float(max_class_fraction),
        "overall_missing_rate": float(overall_missing_rate),
        "frac_cols_any_missing": float(frac_cols_any_missing),
        "unique_ratio_mean": float(unique_ratio_mean),
        "unique_ratio_min": float(unique_ratio_min),
        "unique_ratio_max": float(unique_ratio_max),
        "numeric_abs_skew_mean": float(abs_skew_mean),
        "numeric_kurtosis_mean": float(kurt_mean),
        "numeric_heavy_tail_frac": float(heavy_tail_frac),
        "mean_abs_corr": float(mean_abs_corr),
        "effective_rank": float(eff_rank),
        "n_numeric_features": int(n_numeric_features),
        "n_categorical_features": int(n_categorical_features),
        "frac_numeric_features": float(frac_numeric),
        "frac_categorical_features": float(frac_categorical),
    }


def compute_meta_features_from_dataframe(
    X_df: pd.DataFrame,
    y: np.ndarray,
    rng: np.random.Generator,
    cfg: MetaFeatureConfig,
) -> Dict[str, Any]:
    """Compute phi(D) for a pandas DataFrame dataset."""
    X_df = X_df.copy()
    y = np.asarray(y)
    n_rows, n_features = X_df.shape

    # Identify numeric vs non-numeric columns
    num_cols = list(X_df.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    n_numeric = len(num_cols)
    n_categorical = len(cat_cols)

    # Encode y with LabelEncoder to ensure contiguous labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    counts = np.bincount(y_enc)
    n_classes = int(counts.size)
    class_entropy, max_class_fraction = _safe_entropy_from_counts(counts)
    if n_classes > 1 and not math.isnan(class_entropy):
        class_entropy_norm = float(class_entropy / math.log(n_classes))
    else:
        class_entropy_norm = math.nan

    # Missingness (all columns)
    if n_rows > 0 and n_features > 0:
        missing_mask = X_df.isna().to_numpy()
        overall_missing_rate = float(missing_mask.mean())
        frac_cols_any_missing = float(missing_mask.any(axis=0).mean())
    else:
        overall_missing_rate = math.nan
        frac_cols_any_missing = math.nan

    # Unique-ratio proxies (all columns but subsampled if many)
    if n_rows > 0 and n_features > 0:
        all_cols = np.array(X_df.columns)
        col_idx = _sample_indices(n_features, cfg.max_unique_cols, rng)
        chosen_cols = all_cols[col_idx]
        unique_ratios: List[float] = []
        for col in chosen_cols:
            nunique = X_df[col].nunique(dropna=True)
            unique_ratios.append(float(nunique) / float(n_rows))
        if unique_ratios:
            unique_ratio_mean = float(np.mean(unique_ratios))
            unique_ratio_min = float(np.min(unique_ratios))
            unique_ratio_max = float(np.max(unique_ratios))
        else:
            unique_ratio_mean = unique_ratio_min = unique_ratio_max = math.nan
    else:
        unique_ratio_mean = unique_ratio_min = unique_ratio_max = math.nan

    # Numeric distribution + dependence only on numeric columns
    if n_numeric > 0:
        X_num = X_df[num_cols].to_numpy(dtype=float)
        abs_skew_mean, kurt_mean, heavy_tail_frac = _numeric_distribution_features(X_num, rng, cfg)
        mean_abs_corr, eff_rank = _dependence_features(X_num, rng, cfg)
    else:
        abs_skew_mean = kurt_mean = heavy_tail_frac = math.nan
        mean_abs_corr = eff_rank = math.nan

    total_features = max(n_numeric + n_categorical, 1)
    frac_numeric = float(n_numeric) / float(total_features)
    frac_categorical = float(n_categorical) / float(total_features)

    return {
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "n_classes": int(n_classes),
        "class_entropy": float(class_entropy),
        "class_entropy_norm": float(class_entropy_norm),
        "max_class_fraction": float(max_class_fraction),
        "overall_missing_rate": float(overall_missing_rate),
        "frac_cols_any_missing": float(frac_cols_any_missing),
        "unique_ratio_mean": float(unique_ratio_mean),
        "unique_ratio_min": float(unique_ratio_min),
        "unique_ratio_max": float(unique_ratio_max),
        "numeric_abs_skew_mean": float(abs_skew_mean),
        "numeric_kurtosis_mean": float(kurt_mean),
        "numeric_heavy_tail_frac": float(heavy_tail_frac),
        "mean_abs_corr": float(mean_abs_corr),
        "effective_rank": float(eff_rank),
        "n_numeric_features": int(n_numeric),
        "n_categorical_features": int(n_categorical),
        "frac_numeric_features": float(frac_numeric),
        "frac_categorical_features": float(frac_categorical),
    }


def stream_pretrain_priors(
    stage1_prior_dir: Optional[str],
    stage2_prior_dir: Optional[str],
    stage1_max_batches: int,
    stage2_max_batches: int,
    max_pretrain_tasks: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """Stream over Stage-1 and Stage-2 prior datasets and compute phi(D)."""
    cfg = MetaFeatureConfig()
    rng = np.random.default_rng(seed)

    records: List[Dict[str, Any]] = []
    total_tasks = 0

    # (label, directory, max_batches)
    stages: List[Tuple[str, Optional[str], int]] = [
        ("stage1", stage1_prior_dir, stage1_max_batches),
        ("stage2", stage2_prior_dir, stage2_max_batches),
    ]

    for source, prior_dir, max_batches in stages:
        if prior_dir is None:
            continue
        data_dir = Path(prior_dir)
        if not data_dir.is_dir():
            print(f"[WARN] Prior dir for {source} does not exist: {data_dir}")
            continue

        # Try to infer the original generation batch size from metadata.json
        batch_size_gen = 512
        meta_path = data_dir / "metadata.json"
        if meta_path.exists():
            try:
                with meta_path.open("r") as f:
                    meta = json.load(f)
                if isinstance(meta, dict) and "batch_size" in meta:
                    batch_size_gen = int(meta["batch_size"])
            except Exception as e:
                print(f"[WARN] Could not read metadata.json in {data_dir}: {e}")

        # Note: LoadPriorDataset.get_batch() returns the full stored batch regardless
        # of self.batch_size; we set batch_size for informational purposes only.
        ds = LoadPriorDataset(
            data_dir=str(data_dir),
            batch_size=batch_size_gen,
            ddp_world_size=1,
            ddp_rank=0,
            start_from=0,
            max_batches=None,
            delete_after_load=False,
            device="cpu",
        )

        # Discover available batch indices and randomly sample up to max_batches
        batch_indices: List[int] = []
        for p in data_dir.glob("batch_*.pt"):
            stem = p.stem  # e.g., "batch_000123"
            try:
                idx_str = stem.split("batch_")[1]
                batch_indices.append(int(idx_str))
            except Exception:
                continue

        if not batch_indices:
            print(f"[WARN] No batch_*.pt files found in {data_dir} for {source}")
            continue

        batch_indices = sorted(batch_indices)
        k = min(max_batches, len(batch_indices))
        # Randomly sample batch indices but iterate in sorted order for locality
        chosen = np.sort(rng.choice(batch_indices, size=k, replace=False))

        print(
            f"[INFO] Streaming priors from {data_dir} ({source}), "
            f"using {k} / {len(batch_indices)} available batch files"
        )

        for batch_idx in chosen:
            if max_pretrain_tasks is not None and total_tasks >= max_pretrain_tasks:
                break
            try:
                X_b, y_b, d_b, seq_lens_b, train_sizes_b = ds.get_batch(idx=batch_idx, batch_size=None)
            except FileNotFoundError:
                print(f"[WARN] Missing batch file {batch_idx:06d} in {data_dir}, stopping for {source}.")
                break

            X_np = X_b.numpy()
            y_np = y_b.numpy()
            d_np = d_b.numpy()
            seq_np = seq_lens_b.numpy()

            B = X_np.shape[0]
            for i in range(B):
                if max_pretrain_tasks is not None and total_tasks >= max_pretrain_tasks:
                    break
                # Respect per-dataset feature and sequence lengths
                T_i = int(seq_np[i])
                H_i = int(d_np[i])
                X_i = X_np[i, :T_i, :H_i]
                y_i = y_np[i, :T_i]

                phi = compute_meta_features_from_array(X_i, y_i, rng=rng, cfg=cfg)
                phi.update(
                    {
                        "source": source,
                        "prior_dir": str(data_dir),
                        "prior_batch_idx": int(batch_idx),
                        "prior_dataset_idx": int(i),
                        "global_task_idx": int(total_tasks),
                    }
                )
                records.append(phi)
                total_tasks += 1

        print(f"[INFO] Collected {total_tasks} tasks so far (after {source}).")

    if not records:
        print("[WARN] No pretrain prior tasks processed.")
        return pd.DataFrame(columns=list(META_FEATURE_COLUMNS) + ["source"])

    df = pd.DataFrame.from_records(records)
    return df


def compute_cc18_meta_features(
    max_features: int,
    max_classes: int,
    limit_tasks: Optional[int],
    seed: int,
    tabicl_checkpoint: Optional[str] = None,
    tabicl_device: str = "cpu",
    tabicl_n_estimators: int = 1,
) -> pd.DataFrame:
    """Compute phi(D) for OpenML-CC18 tasks using official train/test split (fold 0)."""
    try:
        import openml  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "The 'openml' package is required to compute CC18 meta-features. "
            "Install it in your environment (e.g., the .tabicl venv) or run "
            "this script in that environment."
        ) from e

    cfg = MetaFeatureConfig()
    rng = np.random.default_rng(seed)

    suite = openml.study.get_suite(99)  # OpenML-CC18
    task_ids: List[int] = list(suite.tasks)
    if limit_tasks is not None:
        task_ids = task_ids[:limit_tasks]

    print(f"[INFO] OpenML-CC18: considering {len(task_ids)} tasks")

    # Use a fixed OpenML split for consistency with evaluation:
    # repeat=0, fold=0, sample=0 (10-fold CV, first fold)
    split_repeat = 0
    split_fold = 0
    split_sample = 0

    records: List[Dict[str, Any]] = []

    for idx, task_id in enumerate(task_ids, 1):
        print(f"[CC18] Task {idx}/{len(task_ids)}: {task_id}")
        try:
            task = openml.tasks.get_task(task_id)
            dataset = openml.datasets.get_dataset(task.dataset_id)

            # Use task-defined target and feature set to stay consistent with evaluation
            X, y = task.get_X_and_y(dataset_format="dataframe")

            # Basic metadata / filters consistent with openml_cc18_benchmark.evaluate_task_tabicl
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            metadata = compute_dataset_metadata(X.copy(), y_enc, le, dataset)
            meta_cols = {
                "n_rows_raw": int(metadata.get("n_samples", X.shape[0])),
                "n_features_raw": int(metadata.get("n_features", X.shape[1])),
                "n_classes_raw": int(metadata.get("n_classes", len(np.unique(y_enc)))),
            }

            if max_features is not None and X.shape[1] > max_features:
                print(f"  [SKIP] features {X.shape[1]} > max_features {max_features}")
                continue

            n_cls = len(np.unique(y))
            if max_classes is not None and n_cls > max_classes:
                print(f"  [SKIP] classes {n_cls} > max_classes {max_classes}")
                continue

            # Use official train/test split (fixed fold) for encoder fitting, but compute phi(D)
            # on the full dataset to keep label statistics consistent with priors.
            try:
                train_idx, test_idx = task.get_train_test_split_indices(
                    repeat=split_repeat,
                    fold=split_fold,
                    sample=split_sample,
                )
            except Exception as e:
                print(f"  [WARN] Error getting split for task {task_id}: {e}")
                continue

            # Optionally process CC18 via TabICL X_encoder before computing meta-features
            if tabicl_checkpoint:
                try:
                    X_train = X.iloc[train_idx]
                    y_train = y.iloc[train_idx]

                    clf = TabICLClassifier(
                        device=tabicl_device,
                        model_path=tabicl_checkpoint,
                        allow_auto_download=False,
                        use_hierarchical=True,
                        n_estimators=tabicl_n_estimators,
                        random_state=seed,
                        verbose=False,
                    )
                    # Fit only to obtain a dataset-specific X_encoder_
                    clf.fit(X_train, y_train)
                    # Transform the full dataset into the model's numeric feature space
                    X_enc_full = clf.X_encoder_.transform(X)
                    phi = compute_meta_features_from_array(
                        X_enc_full,
                        y_enc,
                        rng=rng,
                        cfg=cfg,
                        n_numeric_features=X_enc_full.shape[1],
                        n_categorical_features=0,
                    )
                except Exception as e:
                    print(f"  [WARN] TabICL preprocessing failed for task {task_id}: {e}")
                    phi = compute_meta_features_from_dataframe(X, y_enc, rng=rng, cfg=cfg)
            else:
                # Raw DataFrame meta-features on the full dataset
                phi = compute_meta_features_from_dataframe(X, y_enc, rng=rng, cfg=cfg)

            rec: Dict[str, Any] = {
                "task_id": int(task_id),
                "dataset_id": int(dataset.dataset_id),
                "dataset_name": str(dataset.name),
                **meta_cols,
                **phi,
            }
            records.append(rec)
        except Exception as e:
            print(f"  [WARN] Error processing task {task_id}: {e}")
            continue

    if not records:
        print("[WARN] No CC18 tasks passed filters.")
        cols = ["task_id", "dataset_id", "dataset_name", "n_rows_raw", "n_features_raw", "n_classes_raw", *META_FEATURE_COLUMNS]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame.from_records(records)
    return df


def compute_coverage_and_percentiles(
    df_pretrain: pd.DataFrame,
    df_cc18: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-feature coverage stats and CC18 percentiles wrt pretrain distribution."""
    coverage_records: List[Dict[str, Any]] = []
    percentiles_records: List[Dict[str, Any]] = []

    for feat in feature_cols:
        pre_vals = df_pretrain[feat].to_numpy(dtype=float)
        pre_vals = pre_vals[~np.isnan(pre_vals)]
        if pre_vals.size == 0:
            continue

        # Precompute sorted values for fast percentile queries
        pre_sorted = np.sort(pre_vals)
        q1, q5, q50, q95, q99 = np.percentile(pre_sorted, [1, 5, 50, 95, 99])

        cc_vals = df_cc18[feat].to_numpy(dtype=float)
        outs = 0
        perc_list: List[float] = []
        for v in cc_vals:
            if math.isnan(v):
                perc = math.nan
            else:
                # Mid-rank percentile using searchsorted
                lo = float(np.searchsorted(pre_sorted, v, side="left"))
                hi = float(np.searchsorted(pre_sorted, v, side="right"))
                mid_rank = (lo + hi) * 0.5
                perc = mid_rank / float(pre_sorted.size) * 100.0
                if v < q1 or v > q99:
                    outs += 1
            perc_list.append(perc)

        out_frac = float(outs) / float(len(cc_vals)) if len(cc_vals) > 0 else math.nan
        coverage_records.append(
            {
                "feature": feat,
                "q01": float(q1),
                "q05": float(q5),
                "q50": float(q50),
                "q95": float(q95),
                "q99": float(q99),
                "cc18_outside_1_99_frac": float(out_frac),
            }
        )

        for idx, v in enumerate(cc_vals):
            percentiles_records.append(
                {
                    "dataset_index": int(idx),
                    "dataset_name": df_cc18.get("dataset_name", pd.Series(["unknown"] * len(cc_vals))).iloc[idx],
                    "feature": feat,
                    "percentile": perc_list[idx],
                    "value": float(v),
                }
            )

    coverage_df = pd.DataFrame.from_records(coverage_records)
    percentiles_df = pd.DataFrame.from_records(percentiles_records)
    return coverage_df, percentiles_df


def run_c2st_bootstrap(
    df_pretrain: pd.DataFrame,
    df_cc18: pd.DataFrame,
    feature_cols: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Repeated bootstrap C2ST with logistic regression."""
    rng = np.random.default_rng(seed)
    N_pre = len(df_pretrain)
    N_cc = len(df_cc18)
    if N_pre == 0 or N_cc == 0:
        print("[WARN] Skipping C2ST: empty pretrain or CC18 matrix.")
        return pd.DataFrame(columns=["bootstrap_id", "auc_mean", "auc_std"])

    if N_cc < 2:
        print("[WARN] Skipping C2ST: need at least 2 CC18 datasets.")
        return pd.DataFrame(columns=["bootstrap_id", "auc_mean", "auc_std"])

    records: List[Dict[str, Any]] = []

    for b in range(n_bootstrap):
        # Sample 60 pretrain tasks (or as many as we have)
            # actual sample size: min(N_cc, N_pre)
        m = min(N_cc, N_pre)
        idx_pre = rng.choice(N_pre, size=m, replace=False)

        X_pre = df_pretrain.iloc[idx_pre][feature_cols].to_numpy(dtype=float)
        X_cc = df_cc18[feature_cols].to_numpy(dtype=float)
        # If CC18 has more than m tasks, subsample to m for balance
        if X_cc.shape[0] > m:
            idx_cc = rng.choice(X_cc.shape[0], size=m, replace=False)
            X_cc = X_cc[idx_cc]
        else:
            idx_cc = np.arange(X_cc.shape[0], dtype=int)

        X_all = np.vstack([X_pre, X_cc])
        y_all = np.concatenate(
            [np.zeros(X_pre.shape[0], dtype=int), np.ones(X_cc.shape[0], dtype=int)],
            axis=0,
        )

        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=0.1,
                        solver="lbfgs",
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        # Use a fixed, small number of CV folds; datasets are balanced by construction
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + b)
        aucs: List[float] = []
        for train_idx, test_idx in cv.split(X_all, y_all):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]
            pipe.fit(X_train, y_train)
            probs = pipe.predict_proba(X_test)[:, 1]
            try:
                auc = roc_auc_score(y_test, probs)
            except ValueError:
                continue
            aucs.append(float(auc))

        if not aucs:
            mean_auc = math.nan
            std_auc = math.nan
        else:
            mean_auc = float(np.mean(aucs))
            std_auc = float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0)

        records.append(
            {
                "bootstrap_id": int(b),
                "auc_mean": float(mean_auc),
                "auc_std": float(std_auc),
                "n_train_pretrain": int(X_pre.shape[0]),
                "n_train_cc18": int(X_cc.shape[0]),
            }
        )

    return pd.DataFrame.from_records(records)


def _load_cc18_benchmark(path: str, metric_col: str) -> pd.DataFrame:
    """Load a CC18 benchmark CSV and extract a clean per-dataset metric column."""
    df = pd.read_csv(path)
    if "dataset_id" not in df.columns:
        raise ValueError(f"Benchmark CSV at {path} has no 'dataset_id' column.")

    # Filter out rows without valid folds or with explicit error messages
    if "n_valid_folds" in df.columns:
        df = df[df["n_valid_folds"].fillna(0) > 0]
    if "error" in df.columns:
        err = df["error"].astype(str)
        df = df[(err == "") | err.isna()]

    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in {path}.")

    out = df[["dataset_id", "dataset_name", metric_col]].copy()
    # Normalize types
    out["dataset_id"] = out["dataset_id"].astype(int)
    out = out.rename(columns={metric_col: "perf"})
    # If multiple rows per dataset remain, aggregate by mean
    out = out.groupby(["dataset_id", "dataset_name"], as_index=False)["perf"].mean()
    return out


def analyze_meta_vs_performance(
    df_cc: pd.DataFrame,
    out_dir: Path,
    benchmark_csv: Optional[str],
    baseline_csv: Optional[str],
    metric_col: str,
) -> Optional[pd.DataFrame]:
    """Relate CC18 meta-features to benchmark performance (and optionally performance gaps)."""
    if benchmark_csv is None:
        return None

    bench_main = _load_cc18_benchmark(benchmark_csv, metric_col)
    df = df_cc.copy()
    if "dataset_id" not in df.columns:
        print("[WARN] meta_phi_cc18 has no 'dataset_id' column; skipping perf vs meta analysis.")
        return None

    df["dataset_id"] = df["dataset_id"].astype(int)
    merged = df.merge(bench_main.rename(columns={"perf": "perf_main"}), on="dataset_id", how="inner")

    if merged.empty:
        print(
            "[WARN] No overlap between meta_phi_cc18 datasets and benchmark CSV; "
            "skipping perf vs meta analysis."
        )
        return None

    perf_gap_available = False
    if baseline_csv is not None:
        try:
            bench_base = _load_cc18_benchmark(baseline_csv, metric_col)
            merged = merged.merge(
                bench_base.rename(columns={"perf": "perf_baseline"}),
                on="dataset_id",
                how="inner",
                suffixes=("", "_baseline"),
            )
            merged["perf_gap"] = merged["perf_main"] - merged["perf_baseline"]
            perf_gap_available = True
        except Exception as e:
            print(f"[WARN] Failed to load baseline benchmark CSV '{baseline_csv}': {e}")

    feature_cols = [c for c in META_FEATURE_COLUMNS if c in merged.columns]
    if not feature_cols:
        print("[WARN] No overlapping meta-feature columns for perf vs meta analysis.")
        return None

    # Prepare design matrix
    X = merged[feature_cols].to_numpy(dtype=float)
    if X.shape[0] == 0:
        print("[WARN] Empty design matrix for perf vs meta analysis; skipping.")
        return None
    # Impute NaNs with column means
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size > 0:
        X[inds] = np.take(col_means, inds[1])

    results: List[Dict[str, Any]] = []

    # Correlations and linear regression for perf_main
    y_main = merged["perf_main"].to_numpy(dtype=float)
    corr_main: Dict[str, float] = {}
    for j, feat in enumerate(feature_cols):
        x = X[:, j]
        mask = np.isfinite(x) & np.isfinite(y_main)
        if mask.sum() < 3:
            corr = math.nan
        else:
            corr = float(np.corrcoef(x[mask], y_main[mask])[0, 1])
        corr_main[feat] = corr

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lin_main = LinearRegression()
    lin_main.fit(X_scaled, y_main)
    coef_main = lin_main.coef_

    # Optional: correlations / regression for perf_gap
    if perf_gap_available:
        y_gap = merged["perf_gap"].to_numpy(dtype=float)
        corr_gap: Dict[str, float] = {}
        for j, feat in enumerate(feature_cols):
            x = X[:, j]
            mask = np.isfinite(x) & np.isfinite(y_gap)
            if mask.sum() < 3:
                corr = math.nan
            else:
                corr = float(np.corrcoef(x[mask], y_gap[mask])[0, 1])
            corr_gap[feat] = corr

        lin_gap = LinearRegression()
        lin_gap.fit(X_scaled, y_gap)
        coef_gap = lin_gap.coef_
    else:
        corr_gap = {}
        coef_gap = np.full_like(coef_main, np.nan, dtype=float)

    for j, feat in enumerate(feature_cols):
        results.append(
            {
                "feature": feat,
                "corr_perf_main": corr_main.get(feat, math.nan),
                "coef_perf_main": float(coef_main[j]),
                "corr_perf_gap": corr_gap.get(feat, math.nan),
                "coef_perf_gap": float(coef_gap[j]),
            }
        )

    stats_df = pd.DataFrame.from_records(results)
    out_path = out_dir / "meta_vs_perf_stats.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved meta-feature vs performance stats to {out_path}")

    # Print a brief summary
    try:
        r2_main = float(lin_main.score(X_scaled, y_main))
        print(f"[INFO] Linear R^2 for perf_main ~ phi(D): {r2_main:.3f}")
    except Exception:
        pass

    return stats_df


def compute_stage_specific_stats(
    df_pre: pd.DataFrame,
    df_cc: pd.DataFrame,
    feature_cols: Sequence[str],
    out_dir: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    """Compute source-specific coverage and C2ST (stage1 vs cc18, stage2 vs cc18)."""
    if "source" not in df_pre.columns:
        return

    sources = sorted(df_pre["source"].dropna().unique())
    for i, src in enumerate(sources):
        df_src = df_pre[df_pre["source"] == src]
        if df_src.empty:
            continue

        cov_df, _ = compute_coverage_and_percentiles(df_src, df_cc, feature_cols)
        cov_path = out_dir / f"meta_feature_coverage_{src}.csv"
        cov_df.to_csv(cov_path, index=False)
        print(f"[INFO] Saved coverage report for {src} to {cov_path}")

        c2st_df = run_c2st_bootstrap(
            df_pretrain=df_src,
            df_cc18=df_cc,
            feature_cols=feature_cols,
            n_bootstrap=n_bootstrap,
            seed=seed + i,
        )
        c2st_path = out_dir / f"meta_feature_c2st_bootstrap_{src}.csv"
        c2st_df.to_csv(c2st_path, index=False)
        print(f"[INFO] Saved C2ST bootstrap AUCs for {src} to {c2st_path}")


def compute_pca_embedding(
    df_pre: pd.DataFrame,
    df_cc: pd.DataFrame,
    feature_cols: Sequence[str],
    out_dir: Path,
    seed: int,
) -> None:
    """Compute a 2D PCA embedding for stage1, stage2, and CC18 meta-features."""
    records: List[Dict[str, Any]] = []

    # Priors (stage1 / stage2)
    for _, row in df_pre.iterrows():
        rec: Dict[str, Any] = {
            "group": row.get("source", "pretrain"),
            "kind": "pretrain",
            "prior_dir": row.get("prior_dir", ""),
            "prior_batch_idx": row.get("prior_batch_idx", np.nan),
            "prior_dataset_idx": row.get("prior_dataset_idx", np.nan),
        }
        for feat in feature_cols:
            rec[feat] = row.get(feat, np.nan)
        records.append(rec)

    # CC18
    for _, row in df_cc.iterrows():
        rec = {
            "group": "cc18",
            "kind": "cc18",
            "dataset_id": row.get("dataset_id", np.nan),
            "dataset_name": row.get("dataset_name", ""),
        }
        for feat in feature_cols:
            rec[feat] = row.get(feat, np.nan)
        records.append(rec)

    if not records:
        print("[WARN] No data for PCA embedding.")
        return

    df_all = pd.DataFrame.from_records(records)
    if not feature_cols:
        print("[WARN] No feature columns for PCA embedding.")
        return

    X = df_all[feature_cols].to_numpy(dtype=float)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size > 0:
        X[inds] = np.take(col_means, inds[1])

    pca = PCA(n_components=2, random_state=seed)
    pcs = pca.fit_transform(X)
    df_all["pc1"] = pcs[:, 0]
    df_all["pc2"] = pcs[:, 1]

    out_path = out_dir / "meta_feature_pca_embedding.csv"
    df_all.to_csv(out_path, index=False)
    print(f"[INFO] Saved PCA embedding of meta-features to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meta-feature diagnostics: TabICL priors vs OpenML-CC18.")
    # Prior dirs and ranges
    p.add_argument("--stage1_prior_dir", type=str, default=None, help="Stage-1 prior directory (batch_*.pt files).")
    p.add_argument("--stage2_prior_dir", type=str, default=None, help="Stage-2 prior directory (batch_*.pt files).")
    p.add_argument(
        "--stage1_max_batches",
        type=int,
        default=2500,
        help="Maximum number of Stage-1 prior batch files to scan (starting at 0).",
    )
    p.add_argument(
        "--stage2_max_batches",
        type=int,
        default=1000,
        help="Maximum number of Stage-2 prior batch files to scan (starting at 0).",
    )
    p.add_argument(
        "--max_pretrain_tasks",
        type=int,
        default=200000,
        help="Upper bound on total number of pretrain tasks to process (across both stages).",
    )

    # CC18 filters (mirroring openml_cc18_benchmark defaults)
    p.add_argument("--cc18_max_features", type=int, default=500, help="Skip CC18 datasets with more features.")
    p.add_argument("--cc18_max_classes", type=int, default=10, help="Skip CC18 datasets with more classes.")
    p.add_argument(
        "--cc18_limit",
        type=int,
        default=None,
        help="Optional limit on number of CC18 tasks (for smoke tests).",
    )
    # Optional TabICL preprocessing for CC18 (X_encoder)
    p.add_argument(
        "--tabicl_checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "mini_tabicl_stage2_sa" / "step-1000.ckpt"),
        help=(
            "Optional TabICL checkpoint; if set, CC18 meta-features are computed on "
            "TabICL X_encoder-transformed data (using the official train split)."
        ),
    )
    p.add_argument(
        "--tabicl_device",
        type=str,
        default="cpu",
        help="Device to use for TabICL preprocessing (cpu|cuda).",
    )
    p.add_argument(
        "--tabicl_n_estimators",
        type=int,
        default=1,
        help="Ensemble size for TabICLClassifier used to build X_encoder (affects speed, not phi(D) directly).",
    )
    # C2ST options
    p.add_argument(
        "--c2st_bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap repetitions for C2ST.",
    )

    # Performance vs meta-feature analysis (CC18)
    p.add_argument(
        "--cc18_benchmark_csv",
        type=str,
        default=None,
        help="Path to CC18 benchmark CSV for the main model (e.g., results/cc18_tabicl_step-1000.csv).",
    )
    p.add_argument(
        "--cc18_baseline_csv",
        type=str,
        default=None,
        help="Optional path to a baseline CC18 benchmark CSV to compute performance gaps.",
    )
    p.add_argument(
        "--perf_metric",
        type=str,
        default="accuracy_mean",
        help="Metric column name in CC18 benchmark CSV to relate to meta-features (default: accuracy_mean).",
    )

    # Output paths
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO_ROOT / "results"),
        help="Directory to store diagnostic CSV files.",
    )
    p.add_argument(
        "--pretrain_out",
        type=str,
        default=None,
        help="Optional explicit CSV path for Phi_pretrain (default: out_dir/meta_phi_pretrain.csv).",
    )
    p.add_argument(
        "--cc18_out",
        type=str,
        default=None,
        help="Optional explicit CSV path for Phi_cc18 (default: out_dir/meta_phi_cc18.csv).",
    )
    p.add_argument(
        "--coverage_out",
        type=str,
        default=None,
        help="Optional explicit CSV path for coverage stats (default: out_dir/meta_feature_coverage.csv).",
    )
    p.add_argument(
        "--percentiles_out",
        type=str,
        default=None,
        help="Optional explicit CSV path for CC18 percentiles (default: out_dir/meta_feature_percentiles_cc18.csv).",
    )
    p.add_argument(
        "--c2st_out",
        type=str,
        default=None,
        help="Optional explicit CSV path for C2ST bootstrap AUCs (default: out_dir/meta_feature_c2st_bootstrap.csv).",
    )

    p.add_argument("--seed", type=int, default=42, help="Random seed for subsampling and bootstrap.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrain_out = Path(args.pretrain_out) if args.pretrain_out is not None else out_dir / "meta_phi_pretrain.csv"
    cc18_out = Path(args.cc18_out) if args.cc18_out is not None else out_dir / "meta_phi_cc18.csv"
    coverage_out = Path(args.coverage_out) if args.coverage_out is not None else out_dir / "meta_feature_coverage.csv"
    percentiles_out = (
        Path(args.percentiles_out)
        if args.percentiles_out is not None
        else out_dir / "meta_feature_percentiles_cc18.csv"
    )
    c2st_out = Path(args.c2st_out) if args.c2st_out is not None else out_dir / "meta_feature_c2st_bootstrap.csv"

    # 1) Stream priors and compute phi(D)
    df_pre = stream_pretrain_priors(
        stage1_prior_dir=args.stage1_prior_dir,
        stage2_prior_dir=args.stage2_prior_dir,
        stage1_max_batches=args.stage1_max_batches,
        stage2_max_batches=args.stage2_max_batches,
        max_pretrain_tasks=args.max_pretrain_tasks,
        seed=args.seed,
    )

    print(f"[INFO] Phi_pretrain shape: {df_pre.shape}")
    df_pre.to_csv(pretrain_out, index=False)
    print(f"[INFO] Saved Phi_pretrain to {pretrain_out}")

    # 2) CC18 phi(D)
    df_cc = compute_cc18_meta_features(
        max_features=args.cc18_max_features,
        max_classes=args.cc18_max_classes,
        limit_tasks=args.cc18_limit,
        seed=args.seed + 1,
        tabicl_checkpoint=args.tabicl_checkpoint,
        tabicl_device=args.tabicl_device,
        tabicl_n_estimators=args.tabicl_n_estimators,
    )
    print(f"[INFO] Phi_cc18 shape: {df_cc.shape}")
    df_cc.to_csv(cc18_out, index=False)
    print(f"[INFO] Saved Phi_cc18 to {cc18_out}")

    # 3) Coverage / percentiles
    feature_cols = [c for c in META_FEATURE_COLUMNS if c in df_pre.columns and c in df_cc.columns]
    if feature_cols:
        coverage_df, percentiles_df = compute_coverage_and_percentiles(df_pre, df_cc, feature_cols)
        coverage_df.to_csv(coverage_out, index=False)
        percentiles_df.to_csv(percentiles_out, index=False)
        print(f"[INFO] Saved coverage report to {coverage_out}")
        print(f"[INFO] Saved CC18 percentiles to {percentiles_out}")
    else:
        print("[WARN] No overlapping meta-feature columns between pretrain and CC18; skipping coverage.")

    # 4) C2ST logistic regression
    if feature_cols and len(df_pre) > 0 and len(df_cc) > 1:
        c2st_df = run_c2st_bootstrap(
            df_pretrain=df_pre,
            df_cc18=df_cc,
            feature_cols=feature_cols,
            n_bootstrap=args.c2st_bootstrap,
            seed=args.seed + 2,
        )
        c2st_df.to_csv(c2st_out, index=False)
        print(f"[INFO] Saved C2ST bootstrap AUCs to {c2st_out}")
        if len(c2st_df) > 0:
            print(
                f"[INFO] C2ST AUC summary: mean={c2st_df['auc_mean'].mean():.3f}, "
                f"std={c2st_df['auc_mean'].std(ddof=1) if len(c2st_df)>1 else 0.0:.3f}"
            )
    else:
        print("[WARN] Skipping C2ST (insufficient features or data).")

    # 5) Source-specific coverage and C2ST (stage1 vs stage2 vs CC18)
    if feature_cols:
        compute_stage_specific_stats(
            df_pre=df_pre,
            df_cc=df_cc,
            feature_cols=feature_cols,
            out_dir=out_dir,
            n_bootstrap=args.c2st_bootstrap,
            seed=args.seed + 3,
        )

    # 6) Meta-features vs performance on CC18 (requires benchmark CSV)
    analyze_meta_vs_performance(
        df_cc=df_cc,
        out_dir=out_dir,
        benchmark_csv=args.cc18_benchmark_csv,
        baseline_csv=args.cc18_baseline_csv,
        metric_col=args.perf_metric,
    )

    # 7) PCA embedding of Stage-1, Stage-2 priors and CC18 datasets
    if feature_cols:
        compute_pca_embedding(
            df_pre=df_pre,
            df_cc=df_cc,
            feature_cols=feature_cols,
            out_dir=out_dir,
            seed=args.seed + 4,
        )


if __name__ == "__main__":
    main()
