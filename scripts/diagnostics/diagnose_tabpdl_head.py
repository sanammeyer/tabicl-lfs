#!/usr/bin/env python3
"""
TabPDL head diagnosis script (macro-collapse, similarity quality, margins, head swaps, ablations).

This script is intended to debug failure cases where a TabPDL (PDLC-style) head
collapses to predicting 1–2 classes.

Implements:
  A) Predicted-class histogram + per-class recall + confusion matrix
  B) Pairwise AUC/AP of PDLC similarity gamma on support-support pairs
  C) Margin analysis (top-1 minus top-2 gap) per query
  D) Head swap test (SA head <-> PDL head across backbones)
  E) Ablations: agg / topk / embed_norm / inference temperature

Examples
--------
  # Diagnose a few CC18 datasets by name (random stratified split)
  python scripts/diagnostics/diagnose_tabpdl_head.py \
    --datasets car,steel-plates-fault,wall-robot-navigation \
    --sa_checkpoint checkpoints/mini_tabicl_stage2_sa/step-1000.ckpt \
    --pdl_checkpoint checkpoints/mini_tabicl_stage2_pdl/step-1000.ckpt \
    --out_dir results/pdl_head_diag

  # Use official OpenML task fold
  python scripts/diagnostics/diagnose_tabpdl_head.py \
    --openml_tasks 1461 --fold 0 \
    --sa_checkpoint checkpoints/mini_tabicl_stage2_sa/step-1000.ckpt \
    --pdl_checkpoint checkpoints/mini_tabicl_stage2_pdl/step-1000.ckpt

  # Quick ablation sweep on aggregation only (single-variant episode)
  python scripts/diagnostics/diagnose_tabpdl_head.py \
    --datasets car \
    --pdl_checkpoint checkpoints/mini_tabicl_stage2_pdl/step-1000.ckpt \
    --pdl_agg_sweep posterior_avg,class_pool,sum

Notes
-----
- `posterior_avg` is the faithful PDLC posterior aggregation used for TabPDL runs in this repo.
  The other aggregations are provided as controlled counterfactuals to diagnose whether pooling
  is responsible for macro-collapse.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Import setup
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _csv_list(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagnose TabPDL head failures.")
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated OpenML dataset IDs or (partial) names (random stratified split).",
    )
    data.add_argument(
        "--openml_tasks",
        type=str,
        default=None,
        help="Comma-separated OpenML task IDs (uses official fold splits).",
    )
    p.add_argument("--fold", type=int, default=0, help="Fold index for OpenML tasks (default: 0).")
    p.add_argument("--test_size", type=float, default=0.5, help="Test size for random split (default: 0.5).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    p.add_argument("--sa_checkpoint", type=str, default=None, help="Checkpoint trained with standard head (SA).")
    p.add_argument("--pdl_checkpoint", type=str, default=None, help="Checkpoint trained with TabPDL head.")

    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--use_amp", action="store_true", help="Enable AMP (mostly relevant on CUDA).")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for sklearn wrapper inference.")
    p.add_argument("--n_estimators", type=int, default=32, help="Ensemble size for sklearn wrapper inference.")
    p.add_argument("--softmax_temperature", type=float, default=0.9, help="Softmax temperature (standard head).")

    # PDLC inference overrides (wrapper-level)
    p.add_argument(
        "--pdlc_agg",
        type=str,
        default="posterior_avg",
        help="PDLC aggregation override for TabPDL inference (wrapper). 'posterior_avg' is the faithful default.",
    )
    p.add_argument(
        "--pdlc_inference_temperature",
        type=float,
        default=None,
        help="PDLC inference temperature override (wrapper).",
    )
    p.add_argument(
        "--pdlc_topk",
        type=int,
        default=None,
        help="PDLC top-k override (wrapper). Use <=0 to disable gating.",
    )
    p.add_argument(
        "--pdlc_bilinear",
        type=str,
        default=None,
        help="PDLC bilinear override (wrapper): 'learned' (default) or 'identity' (W_Q=W_K=I).",
    )
    p.add_argument(
        "--pdlc_tau_override",
        type=float,
        default=None,
        help="Optional override for PDLC tau at inference (wrapper). Must be > 0.",
    )

    # Diagnostics knobs
    p.add_argument(
        "--max_pairs",
        type=int,
        default=250000,
        help="Max support-support pairs used for gamma AUC/AP (sampling if needed).",
    )
    p.add_argument("--variant", choices=["identity", "random"], default="identity", help="Which ensemble variant to probe.")

    # Optional extra tests
    p.add_argument("--head_swap", action="store_true", help="Run head swap test (requires both checkpoints).")

    # Ablation sweeps (single-variant episode; requires --pdl_checkpoint)
    p.add_argument("--pdl_agg_sweep", type=str, default=None, help="Comma-separated agg modes to sweep.")
    p.add_argument("--pdl_topk_sweep", type=str, default=None, help="Comma-separated topk values (use 'none').")
    p.add_argument("--pdl_embed_norm_sweep", type=str, default=None, help="Comma-separated embed_norm values (none,l2).")
    p.add_argument(
        "--pdl_infer_temp_sweep",
        type=str,
        default=None,
        help="Comma-separated inference temperatures (e.g. 0.5,1.0,2.0).",
    )
    p.add_argument(
        "--pdl_uncertainty",
        action="store_true",
        help="Compute PDLC-style anchor uncertainty breakdown (TU/AU/EU) for the single-variant episode.",
    )

    # Weighted PDLC (learn anchor weights on a held-out validation set)
    p.add_argument(
        "--pdl_learn_anchor_weights",
        action="store_true",
        help="Learn nonnegative anchor weights on a validation split of the support set (no checkpoint retraining).",
    )
    p.add_argument(
        "--pdl_anchor_val_frac",
        type=float,
        default=0.2,
        help="Fraction of the train split to hold out for learning anchor weights (default: 0.2).",
    )
    p.add_argument(
        "--pdl_anchor_weight_steps",
        type=int,
        default=300,
        help="Optimization steps for learning anchor weights (default: 300).",
    )
    p.add_argument(
        "--pdl_anchor_weight_lr",
        type=float,
        default=0.5,
        help="Learning rate for anchor-weight optimization (default: 0.5).",
    )
    p.add_argument(
        "--pdl_anchor_weight_prune_lambda",
        type=float,
        default=0.0,
        help="Weight-pruning strength (maximize w_max on the simplex): loss = NLL - lambda*w_max (default: 0).",
    )
    p.add_argument(
        "--pdl_anchor_weight_entropy_lambda",
        type=float,
        default=0.0,
        help="Entropy penalty (encourage sparsity): loss += lambda*H(w) where H is entropy (default: 0).",
    )
    p.add_argument(
        "--pdl_anchor_weight_prune_topm",
        type=int,
        default=None,
        help="After learning, keep only the top-m anchors by weight (others weight=0) and renormalize.",
    )
    p.add_argument(
        "--pdl_anchor_weight_use_topk",
        action="store_true",
        help="When learning and applying weights, also apply the current PDLC top-k gating (query-dependent) if set.",
    )

    # Dataset-tuned top-k (select k on a support-validation split, then evaluate on test).
    p.add_argument(
        "--pdlc_topk_tune",
        action="store_true",
        help="Tune PDLC top-k per dataset on a held-out split of the training supports (single-variant tuning).",
    )
    p.add_argument(
        "--pdlc_topk_tune_candidates",
        type=str,
        default="8,16,32,64,128",
        help="Comma-separated topk candidates for tuning (use 'none' to disable gating).",
    )
    p.add_argument(
        "--pdlc_topk_tune_metric",
        choices=["macro_f1", "acc", "weighted_f1"],
        default="macro_f1",
        help="Metric to maximize on the tuning split.",
    )
    p.add_argument(
        "--pdlc_topk_tune_val_frac",
        type=float,
        default=0.2,
        help="Fraction of training supports used as tuning queries (default: 0.2).",
    )

    p.add_argument("--out_dir", type=str, default="results/pdl_head_diag", help="Output directory.")
    p.add_argument("--no_save", action="store_true", help="Do not write outputs to disk; print only.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# OpenML loading
# ---------------------------------------------------------------------------


def fetch_openml_dataset(name_or_id: str | int) -> Tuple[pd.DataFrame, pd.Series, str, int]:
    import openml

    if isinstance(name_or_id, int) or (str(name_or_id).isdigit()):
        did = int(name_or_id)
        ds = openml.datasets.get_dataset(did)
        X, y, _, _ = ds.get_data(target=getattr(ds, "default_target_attribute", None), dataset_format="dataframe")
        if y is None:
            raise ValueError(f"Dataset {ds.name} (id={did}) has no target.")
        return X, y, ds.name, int(ds.dataset_id)

    name = str(name_or_id).lower()
    df = openml.datasets.list_datasets(output_format="dataframe")

    def _pick_best(cand: pd.DataFrame) -> pd.Series:
        # Prefer richer-classification datasets when multiple versions share a name.
        # Tie-break by more instances, then smaller did for determinism.
        cand = cand.copy()
        if "NumberOfClasses" in cand.columns:
            cand["_noc"] = cand["NumberOfClasses"].fillna(-1.0)
        else:
            cand["_noc"] = -1.0
        if "NumberOfInstances" in cand.columns:
            cand["_noi"] = cand["NumberOfInstances"].fillna(-1.0)
        else:
            cand["_noi"] = -1.0
        cand = cand.sort_values(["_noc", "_noi", "did"], ascending=[False, False, True])
        return cand.iloc[0]

    exact = df[df["name"].str.lower() == name]
    if len(exact) == 0:
        contains = df[df["name"].str.lower().str.contains(name)]
        if len(contains) == 0:
            raise ValueError(f"OpenML dataset not found: {name_or_id}")
        row = _pick_best(contains)
    else:
        row = _pick_best(exact)
    did = int(row["did"]) if "did" in row else int(row.get("dataset_id", row.get("ID")))
    ds = openml.datasets.get_dataset(did)
    X, y, _, _ = ds.get_data(target=getattr(ds, "default_target_attribute", None), dataset_format="dataframe")
    if y is None:
        raise ValueError(f"Dataset {ds.name} (id={did}) has no target.")
    return X, y, ds.name, int(did)


def fetch_openml_task(task_id: int, fold: int) -> Tuple[pd.DataFrame, pd.Series, str, int, np.ndarray, np.ndarray]:
    import openml

    task = openml.tasks.get_task(int(task_id))
    dataset = openml.datasets.get_dataset(int(task.dataset_id))
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    if y is None:
        raise ValueError(f"Task {task_id}: dataset has no target.")
    train_idx, test_idx = task.get_train_test_split_indices(fold=fold)
    return X, y, dataset.name, int(dataset.dataset_id), np.asarray(train_idx), np.asarray(test_idx)


# ---------------------------------------------------------------------------
# Diagnostics: A/C
# ---------------------------------------------------------------------------


@dataclass
class BasicDiagnostics:
    n_test: int
    n_classes: int
    acc: float
    macro_f1: float
    weighted_f1: float
    binary_f1_pos: float
    pred_entropy: float
    pred_entropy_norm: float
    pred_top1_frac: float
    pred_top2_frac: float
    n_predicted_classes: int
    per_class_recall: List[float]
    confusion: List[List[int]]
    pred_counts: List[int]
    margin_mean: float
    margin_median: float
    margin_p10: float
    margin_p90: float


def _safe_entropy(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64)
    tot = counts.sum()
    if tot <= 0:
        return float("nan")
    p = counts / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def compute_basic_diagnostics(y_true: np.ndarray, proba: np.ndarray) -> Tuple[BasicDiagnostics, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float64)
    assert y_true.ndim == 1, f"Expected y_true to be 1D, got shape={y_true.shape}"
    assert proba.ndim == 2, f"Expected proba to be 2D (n_samples, n_classes), got shape={proba.shape}"
    n_test, n_classes = proba.shape
    assert n_test == int(y_true.shape[0]), f"proba.shape[0]={n_test} != len(y_true)={int(y_true.shape[0])}"
    assert n_classes >= 1, f"Expected n_classes>=1, got {n_classes}"
    assert np.isfinite(proba).all(), "Non-finite values found in proba (nan/inf)."
    assert np.all((y_true >= 0) & (y_true < n_classes)), (
        f"y_true contains labels outside [0, {n_classes-1}]. "
        f"min={int(y_true.min()) if y_true.size else None} max={int(y_true.max()) if y_true.size else None}"
    )
    y_pred = proba.argmax(axis=1).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    with np.errstate(divide="ignore", invalid="ignore"):
        support = cm.sum(axis=1)
        per_class_recall = np.where(support > 0, np.diag(cm) / support, np.nan)

    pred_counts = np.bincount(y_pred, minlength=n_classes)
    order = np.argsort(-pred_counts)
    top1_frac = float(pred_counts[order[0]] / max(1, n_test)) if n_classes >= 1 else float("nan")
    top2_frac = float(pred_counts[order[1]] / max(1, n_test)) if n_classes >= 2 else float("nan")
    pred_entropy = _safe_entropy(pred_counts)
    pred_entropy_norm = float(pred_entropy / math.log(n_classes)) if n_classes > 1 and np.isfinite(pred_entropy) else float("nan")

    # F1 scores (macro is the main imbalanced-friendly diagnostic).
    # Use explicit labels to keep shape/indexing invariants tight.
    labels_all = list(range(n_classes))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels_all, average="weighted", zero_division=0))
    if n_classes == 2:
        binary_f1_pos = float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
    else:
        binary_f1_pos = float("nan")

    # Margins in probability space (argmax-relevant for both heads under the sklearn API)
    if n_classes >= 2:
        top2 = np.partition(proba, kth=n_classes - 2, axis=1)[:, -2:]
        top2_sorted = np.sort(top2, axis=1)
        margin = top2_sorted[:, 1] - top2_sorted[:, 0]
    else:
        margin = np.full((n_test,), np.nan, dtype=np.float64)

    diag = BasicDiagnostics(
        n_test=int(n_test),
        n_classes=int(n_classes),
        acc=float(accuracy_score(y_true, y_pred)),
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        binary_f1_pos=binary_f1_pos,
        pred_entropy=float(pred_entropy),
        pred_entropy_norm=float(pred_entropy_norm),
        pred_top1_frac=top1_frac,
        pred_top2_frac=top2_frac,
        n_predicted_classes=int((pred_counts > 0).sum()),
        per_class_recall=[float(x) if np.isfinite(x) else float("nan") for x in per_class_recall.tolist()],
        confusion=cm.astype(int).tolist(),
        pred_counts=pred_counts.astype(int).tolist(),
        margin_mean=float(np.nanmean(margin)),
        margin_median=float(np.nanmedian(margin)),
        margin_p10=float(np.nanpercentile(margin, 10)),
        margin_p90=float(np.nanpercentile(margin, 90)),
    )
    return diag, y_pred, margin


def print_basic(name: str, labels: Sequence[str], diag: BasicDiagnostics) -> None:
    counts = np.asarray(diag.pred_counts, dtype=int)
    order = np.argsort(-counts)
    top = [(labels[i], int(counts[i])) for i in order[: min(10, len(order))]]
    f1_extra = f" f1_pos={diag.binary_f1_pos:.4f}" if diag.n_classes == 2 and np.isfinite(diag.binary_f1_pos) else ""
    print(f"\n[{name}] acc={diag.acc:.4f} macro_f1={diag.macro_f1:.4f} weighted_f1={diag.weighted_f1:.4f}{f1_extra}  "
          f"n_pred_classes={diag.n_predicted_classes}/{diag.n_classes}  "
          f"top1={diag.pred_top1_frac:.3f} top2={diag.pred_top2_frac:.3f}  "
          f"entropy={diag.pred_entropy:.3f} (norm={diag.pred_entropy_norm:.3f})  margin_med={diag.margin_median:.3f}")
    print(f"[{name}] predicted counts (top): {top}")


def print_recall_and_confusion(name: str, labels: Sequence[str], diag: BasicDiagnostics, *, max_classes: int = 12) -> None:
    labels = list(map(str, labels))
    rec = list(zip(labels, diag.per_class_recall))
    print(f"[{name}] per-class recall: {rec}")
    if len(labels) <= max_classes:
        try:
            df = pd.DataFrame(diag.confusion, index=labels, columns=labels)
            print(f"[{name}] confusion matrix (rows=true, cols=pred):\n{df}")
        except Exception:
            pass


def encode_y_true_for_index(clf, y: pd.Series, index: Sequence[Any]) -> np.ndarray:
    """Encode y for an explicit index selection, asserting ordering integrity."""
    y_sub = y.loc[list(index)]
    assert len(y_sub) == len(index), "Indexing y changed the number of rows; check duplicates/missing indices."
    assert [str(i) for i in list(y_sub.index)] == [str(i) for i in list(index)], "y_sub index order mismatch."
    y_enc = clf.y_encoder_.transform(y_sub)
    return np.asarray(y_enc, dtype=np.int64)


# ---------------------------------------------------------------------------
# Single-variant episode materialization
# ---------------------------------------------------------------------------


@dataclass
class EpisodeVariant:
    norm_method: str
    variant_index: int
    feature_permutation: List[int]
    class_shift_offset: int
    train_size: int
    test_size: int
    train_index: List[str]
    test_index: List[str]
    support_mask: List[bool]
    label_map: Dict[int, str]
    X_variant: np.ndarray  # (T, H)
    y_train_shifted: np.ndarray  # (train_size,)


def materialize_episode_variant(
    clf,
    X_test_df: pd.DataFrame,
    variant: str,
    rng: np.random.Generator,
    *,
    train_index: Sequence[Any],
    test_index: Sequence[Any],
) -> EpisodeVariant:
    # Transform test data through fitted encoder
    X_test_num = clf.X_encoder_.transform(X_test_df)
    data = clf.ensemble_generator_.transform(X_test_num)

    norm_method = list(data.keys())[0]
    Xs, ys_shifted = data[norm_method]
    shuffle_patterns = clf.ensemble_generator_.feature_shuffle_patterns_[norm_method]
    shift_offsets = clf.ensemble_generator_.class_shift_offsets_[norm_method]

    if variant == "random":
        variant_index = int(rng.integers(0, len(shuffle_patterns)))
    else:
        variant_index = None
        for i, pattern in enumerate(shuffle_patterns):
            if list(pattern) == sorted(pattern):
                variant_index = i
                break
        if variant_index is None:
            variant_index = 0

    X_variant = np.asarray(Xs[variant_index], dtype=np.float32)
    y_train_shifted = np.asarray(ys_shifted[variant_index], dtype=np.int64)
    shift_offset = int(shift_offsets[variant_index])
    train_size = int(y_train_shifted.shape[0])
    test_size = int(X_variant.shape[0] - train_size)
    assert test_size == int(len(X_test_df)), (
        f"Episode test_size={test_size} does not match len(X_test_df)={int(len(X_test_df))}. "
        "This indicates the episode generator changed ordering/subsetting."
    )
    label_map = {int(i): str(lbl) for i, lbl in enumerate(getattr(clf, "classes_", []))}
    return EpisodeVariant(
        norm_method=str(norm_method),
        variant_index=int(variant_index),
        feature_permutation=list(shuffle_patterns[variant_index]),
        class_shift_offset=shift_offset,
        train_size=train_size,
        test_size=test_size,
        train_index=[str(x) for x in list(train_index)],
        test_index=[str(x) for x in list(test_index)],
        support_mask=[True] * train_size,
        label_map=label_map,
        X_variant=X_variant,
        y_train_shifted=y_train_shifted,
    )


def unshift_class_axis(x: np.ndarray, offset: int) -> np.ndarray:
    if offset == 0:
        return x
    return np.concatenate([x[..., offset:], x[..., :offset]], axis=-1)


# ---------------------------------------------------------------------------
# PDL internals: B/E
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_icl_embeddings(
    model,
    episode: EpisodeVariant,
    inference_config,
    device: torch.device,
) -> torch.Tensor:
    train_size = episode.train_size
    X = torch.from_numpy(episode.X_variant).float().unsqueeze(0).to(device)
    y_train = torch.from_numpy(episode.y_train_shifted).float().unsqueeze(0).to(device)
    feature_shuffles = [episode.feature_permutation]

    col_out = model.col_embedder(
        X,
        train_size=train_size,
        feature_shuffles=feature_shuffles,
        mgr_config=inference_config.COL_CONFIG,
    )
    R = model.row_interactor(col_out, mgr_config=inference_config.ROW_CONFIG)  # (1, T, D)

    # Label-condition supports (match ICLearning._icl_predictions)
    R = R.clone()
    R[:, :train_size] = R[:, :train_size] + model.icl_predictor.y_encoder(y_train.float())

    src = model.icl_predictor.tf_icl(R, attn_mask=train_size)
    if getattr(model.icl_predictor, "norm_first", False):
        src = model.icl_predictor.ln(src)
    return src.squeeze(0)  # (T, D)


@dataclass
class GammaSeparation:
    n_support: int
    n_pairs_used: int
    auc: float
    ap: float
    gamma_same_mean: float
    gamma_diff_mean: float
    gamma_same_p10: float
    gamma_same_p90: float
    gamma_diff_p10: float
    gamma_diff_p90: float


@torch.no_grad()
def compute_support_support_gamma_auc(
    model,
    episode: EpisodeVariant,
    inference_config,
    device: torch.device,
    max_pairs: int,
    seed: int,
    *,
    src: Optional[torch.Tensor] = None,
) -> GammaSeparation:
    head = getattr(model.icl_predictor, "pdlc_head", None)
    if head is None:
        raise RuntimeError("Model has no pdlc_head; did you pass a TabPDL checkpoint?")

    # Similarity-quality check: disable top-k gating (gating is an inference heuristic,
    # not part of the underlying learned similarity).
    old_topk = getattr(head.cfg, "topk", None)
    head.cfg.topk = None
    try:
        if src is None:
            src = compute_icl_embeddings(model, episode, inference_config, device)
        train_size = episode.train_size
        Hs = src[:train_size].unsqueeze(0)  # (1, N, D)
        y = torch.from_numpy(episode.y_train_shifted).to(device).long()
        support_mask = torch.ones((1, train_size), dtype=torch.bool, device=device)

        # Pairwise logits/gamma for support-support
        logits_ss = head._pair_logits(H_query=Hs, H_support=Hs, support_mask=support_mask).squeeze(0)  # (N, N)
        gamma_ss = torch.sigmoid(logits_ss)
    finally:
        head.cfg.topk = old_topk

    N = train_size
    if N < 2:
        return GammaSeparation(
            n_support=int(N),
            n_pairs_used=0,
            auc=float("nan"),
            ap=float("nan"),
            gamma_same_mean=float("nan"),
            gamma_diff_mean=float("nan"),
            gamma_same_p10=float("nan"),
            gamma_same_p90=float("nan"),
            gamma_diff_p10=float("nan"),
            gamma_diff_p90=float("nan"),
        )

    # Use only i<j pairs; optionally subsample
    iu, ju = torch.triu_indices(N, N, offset=1, device=device)
    total_pairs = int(iu.numel())
    if max_pairs is not None and total_pairs > int(max_pairs):
        rng = np.random.default_rng(seed)
        sel = rng.choice(total_pairs, size=int(max_pairs), replace=False)
        sel_t = torch.from_numpy(sel).to(device)
        iu = iu[sel_t]
        ju = ju[sel_t]

    g = gamma_ss[iu, ju].detach().cpu().numpy().astype(np.float64)
    same = (y[iu] == y[ju]).detach().cpu().numpy().astype(np.int64)

    # Guard: if only one class in sampled labels, AUC undefined.
    if int(np.unique(same).size) < 2:
        auc = float("nan")
        ap = float("nan")
    else:
        auc = float(roc_auc_score(same, g))
        ap = float(average_precision_score(same, g))

    same_g = g[same == 1]
    diff_g = g[same == 0]

    def _q(a: np.ndarray, q: float) -> float:
        return float(np.nanpercentile(a, q)) if a.size else float("nan")

    return GammaSeparation(
        n_support=int(N),
        n_pairs_used=int(g.shape[0]),
        auc=auc,
        ap=ap,
        gamma_same_mean=float(np.nanmean(same_g)) if same_g.size else float("nan"),
        gamma_diff_mean=float(np.nanmean(diff_g)) if diff_g.size else float("nan"),
        gamma_same_p10=_q(same_g, 10),
        gamma_same_p90=_q(same_g, 90),
        gamma_diff_p10=_q(diff_g, 10),
        gamma_diff_p90=_q(diff_g, 90),
    )


@dataclass
class UncertaintyBreakdown:
    n_queries: int
    n_support: int
    n_classes: int
    topk: Optional[int]
    kept_min: int
    kept_mean: float
    kept_max: int
    tu_mean: float
    tu_median: float
    tu_p10: float
    tu_p90: float
    au_mean: float
    au_median: float
    au_p10: float
    au_p90: float
    eu_mean: float
    eu_median: float
    eu_p10: float
    eu_p90: float
    tu_norm_mean: float
    au_norm_mean: float
    eu_norm_mean: float


@torch.no_grad()
def compute_pdl_anchor_uncertainty_breakdown(
    model,
    episode: EpisodeVariant,
    inference_config,
    device: torch.device,
    *,
    src: Optional[torch.Tensor] = None,
) -> Tuple[UncertaintyBreakdown, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute PDLC-style uncertainty decomposition on a single episode.

    TU(q) = H(p_post(q))
    AU(q) = mean_i H(p_post,i(q))
    EU(q) = TU(q) - AU(q)

    If top-k gating is enabled for posterior_avg, we interpret it as w_i(q)=0 for dropped anchors
    and uniform weights over kept anchors.
    """
    head = getattr(model.icl_predictor, "pdlc_head", None)
    if head is None:
        raise RuntimeError("Model has no pdlc_head; did you pass a TabPDL checkpoint?")
    if str(getattr(head.cfg, "agg", "")) != "posterior_avg":
        raise ValueError("Uncertainty breakdown is only defined for agg='posterior_avg'.")

    if src is None:
        src = compute_icl_embeddings(model, episode, inference_config, device)
    src = src.to(device)

    train_size = int(episode.train_size)
    test_size = int(episode.test_size)
    H_support = src[:train_size].unsqueeze(0)  # (1, N, D)
    H_query_all = src[train_size:]  # (M, D)
    y_support = torch.from_numpy(episode.y_train_shifted).to(device).long()  # (N,)
    support_mask = torch.ones((1, train_size), dtype=torch.bool, device=device)

    C = int(y_support.max().item()) + 1 if y_support.numel() else 0
    if C <= 0:
        raise RuntimeError("Empty support set.")

    # Degenerate: only one class in supports => deterministic prediction, no uncertainty.
    uniq = torch.unique(y_support)
    if int(uniq.numel()) <= 1:
        tu = np.zeros((test_size,), dtype=np.float64)
        au = np.zeros((test_size,), dtype=np.float64)
        eu = np.zeros((test_size,), dtype=np.float64)
        kept = np.full((test_size,), float(train_size), dtype=np.float64)
        out = UncertaintyBreakdown(
            n_queries=int(test_size),
            n_support=int(train_size),
            n_classes=int(C),
            topk=getattr(head.cfg, "topk", None),
            kept_min=int(train_size),
            kept_mean=float(train_size),
            kept_max=int(train_size),
            tu_mean=0.0,
            tu_median=0.0,
            tu_p10=0.0,
            tu_p90=0.0,
            au_mean=0.0,
            au_median=0.0,
            au_p10=0.0,
            au_p90=0.0,
            eu_mean=0.0,
            eu_median=0.0,
            eu_p10=0.0,
            eu_p90=0.0,
            tu_norm_mean=0.0,
            au_norm_mean=0.0,
            eu_norm_mean=0.0,
        )
        return out, tu, au, eu, kept

    counts = torch.bincount(y_support, minlength=C).float()
    prior = counts / counts.sum().clamp_min(1.0)

    # Precompute rho for the "negative mass" redistribution term.
    prior_y = prior.gather(0, y_support)  # (N,)
    denom_y = (1.0 - prior_y).clamp_min(1e-6)  # (N,)
    rho = prior.unsqueeze(0) / denom_y.unsqueeze(1)  # (N, C)
    rho.scatter_(1, y_support.view(-1, 1), 0.0)

    # Precompute terms for per-anchor entropy H(p_post,i).
    prior_log = torch.where(prior > 0, prior * torch.log(prior), torch.zeros_like(prior))
    prior_log_sum = prior_log.sum()  # scalar
    prior_y_log = torch.where(prior_y > 0, prior_y * torch.log(prior_y), torch.zeros_like(prior_y))
    sum_priorlog_excl = prior_log_sum - prior_y_log  # (N,)

    # Precompute supports' K projection once (match head._forward_chunked).
    Hs_norm = head._normalize_embeddings(H_support)  # (1, N, D)
    K_support = head.W_K(Hs_norm)  # (1, N, D)

    k = getattr(head.cfg, "topk", None)
    if k is not None:
        k = int(k)
        if k <= 0:
            k = None

    # Chunk along queries to control memory.
    chunk_size = int(getattr(head, "_CHUNK_SIZE", 1024))
    tu_list: List[np.ndarray] = []
    au_list: List[np.ndarray] = []
    eu_list: List[np.ndarray] = []
    kept_list: List[np.ndarray] = []

    eps = 1e-12
    logC = float(math.log(C)) if C > 1 else float("nan")

    for start in range(0, test_size, chunk_size):
        end = min(test_size, start + chunk_size)
        Hq_chunk = H_query_all[start:end].unsqueeze(0)  # (1, m, D)

        Hq_norm = head._normalize_embeddings(Hq_chunk)  # (1, m, D)
        Q_chunk = head.W_Q(Hq_norm)  # (1, m, D)

        logits_qs = torch.matmul(Q_chunk, K_support.transpose(-1, -2))  # (1, m, N)
        logits_qs = head.tau * logits_qs + head.bias
        logits_qs = head._apply_support_mask_and_topk(logits_qs, support_mask)  # topk ignored for posterior_avg
        gamma = torch.sigmoid(logits_qs).squeeze(0).float()  # (m, N)

        m = int(gamma.shape[0])
        N = int(gamma.shape[1])
        assert N == train_size, f"Expected N={train_size}, got {N}"

        if k is not None and k > 0 and k < train_size:
            topk_idx = gamma.topk(k, dim=1).indices  # (m, k)
            keep = torch.zeros_like(gamma, dtype=torch.bool)
            keep.scatter_(1, topk_idx, True)
            keep_f = keep.to(dtype=gamma.dtype)
        else:
            keep_f = torch.ones_like(gamma, dtype=gamma.dtype)

        denom_q = keep_f.sum(dim=1).clamp_min(1.0)  # (m,)
        kept_list.append(denom_q.detach().cpu().numpy().astype(np.float64))

        g_used = gamma * keep_f
        one_minus_g_used = (1.0 - gamma) * keep_f

        # p_post(q) = average_i p_post,i(q)
        idx = y_support.unsqueeze(0).expand(m, train_size)
        pos = gamma.new_zeros((m, C))
        pos.scatter_add_(1, idx, g_used)
        neg = one_minus_g_used @ rho  # (m, C)

        p = (pos + neg) / denom_q.unsqueeze(1)
        p = p.clamp_min(0.0)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)

        tu = -(p * torch.log(p.clamp_min(eps))).sum(dim=1)  # (m,)

        # AU(q) = mean_i H(p_post,i(q)) over kept anchors
        g_clamped = gamma.clamp(min=eps, max=1.0 - eps)
        s = (1.0 - gamma) / denom_y.unsqueeze(0)  # (m, N)
        s_clamped = s.clamp_min(eps)
        ent_i = -(
            g_clamped * torch.log(g_clamped)
            + s * sum_priorlog_excl.unsqueeze(0)
            + (1.0 - gamma) * torch.log(s_clamped)
        )  # (m, N)
        au = (ent_i * keep_f).sum(dim=1) / denom_q

        eu = tu - au

        tu_list.append(tu.detach().cpu().numpy().astype(np.float64))
        au_list.append(au.detach().cpu().numpy().astype(np.float64))
        eu_list.append(eu.detach().cpu().numpy().astype(np.float64))

    tu_all = np.concatenate(tu_list, axis=0)
    au_all = np.concatenate(au_list, axis=0)
    eu_all = np.concatenate(eu_list, axis=0)
    kept_all = np.concatenate(kept_list, axis=0)

    assert tu_all.shape == (test_size,)
    assert au_all.shape == (test_size,)
    assert eu_all.shape == (test_size,)
    assert kept_all.shape == (test_size,)
    assert np.isfinite(tu_all).all(), "Non-finite TU values."
    assert np.isfinite(au_all).all(), "Non-finite AU values."
    assert np.isfinite(eu_all).all(), "Non-finite EU values."

    def _qs(x: np.ndarray) -> Tuple[float, float, float, float]:
        return (
            float(np.nanmean(x)),
            float(np.nanmedian(x)),
            float(np.nanpercentile(x, 10)),
            float(np.nanpercentile(x, 90)),
        )

    tu_mean, tu_med, tu_p10, tu_p90 = _qs(tu_all)
    au_mean, au_med, au_p10, au_p90 = _qs(au_all)
    eu_mean, eu_med, eu_p10, eu_p90 = _qs(eu_all)

    if C > 1 and np.isfinite(logC) and logC > 0:
        tu_norm_mean = float(tu_mean / logC)
        au_norm_mean = float(au_mean / logC)
        eu_norm_mean = float(eu_mean / logC)
    else:
        tu_norm_mean = float("nan")
        au_norm_mean = float("nan")
        eu_norm_mean = float("nan")

    out = UncertaintyBreakdown(
        n_queries=int(test_size),
        n_support=int(train_size),
        n_classes=int(C),
        topk=k,
        kept_min=int(np.min(kept_all)) if kept_all.size else 0,
        kept_mean=float(np.mean(kept_all)) if kept_all.size else float("nan"),
        kept_max=int(np.max(kept_all)) if kept_all.size else 0,
        tu_mean=tu_mean,
        tu_median=tu_med,
        tu_p10=tu_p10,
        tu_p90=tu_p90,
        au_mean=au_mean,
        au_median=au_med,
        au_p10=au_p10,
        au_p90=au_p90,
        eu_mean=eu_mean,
        eu_median=eu_med,
        eu_p10=eu_p10,
        eu_p90=eu_p90,
        tu_norm_mean=tu_norm_mean,
        au_norm_mean=au_norm_mean,
        eu_norm_mean=eu_norm_mean,
    )
    return out, tu_all, au_all, eu_all, kept_all


@torch.no_grad()
def compute_weighted_posterior_avg_proba(
    head,
    H_support: torch.Tensor,  # (1, N, D)
    H_query: torch.Tensor,  # (1, M, D)
    y_support: torch.Tensor,  # (1, N)
    support_mask: torch.Tensor,  # (1, N)
    *,
    weights: torch.Tensor,  # (N,)
    topk: Optional[int],
) -> np.ndarray:
    """Compute PDLC posterior_avg with anchor weights (w_i >= 0, sum w_i = 1).

    If topk is set, apply per-query top-k gating based on gamma(q,i). Dropped anchors
    are excluded (weight 0) and we renormalize by sum(w_i for kept anchors).
    """
    assert weights.ndim == 1
    B, N, D = H_support.shape
    _, M, _ = H_query.shape
    assert B == 1, "Weighted diagnostics currently support B=1 episodes."
    assert int(weights.shape[0]) == int(N)

    # Compute gamma(q,i) for all query-support pairs.
    logits_qs = head._pair_logits(H_query=H_query, H_support=H_support, support_mask=support_mask)  # (1, M, N)
    gamma = torch.sigmoid(logits_qs).squeeze(0).float()  # (M, N)

    y = y_support.squeeze(0).long()  # (N,)
    mask = support_mask.squeeze(0).bool()
    if not mask.any():
        C = int(y.max().item()) + 1 if y.numel() else 1
        out = np.full((M, C), 1.0 / float(C), dtype=np.float64)
        return out

    gamma = gamma[:, mask]
    y = y[mask]
    w = weights[mask].float()
    N_eff = int(y.numel())
    assert N_eff == int(w.numel())

    C = int(y.max().item()) + 1
    uniq = torch.unique(y)
    if int(uniq.numel()) == 1:
        out = torch.zeros((M, C), dtype=torch.float32, device=gamma.device)
        out[:, int(uniq.item())] = 1.0
        return out.detach().cpu().numpy().astype(np.float64)

    counts = torch.bincount(y, minlength=C).float()
    prior = counts / counts.sum().clamp_min(1.0)

    denom = (1.0 - prior.gather(0, y)).clamp_min(1e-6)  # (N_eff,)
    rho = prior.unsqueeze(0) / denom.unsqueeze(1)  # (N_eff, C)
    rho.scatter_(1, y.view(-1, 1), 0.0)

    if topk is not None:
        k = int(topk)
        if k <= 0 or k >= N_eff:
            keep_f = torch.ones_like(gamma, dtype=gamma.dtype)
        else:
            topk_idx = gamma.topk(k, dim=1).indices  # (M, k)
            keep = torch.zeros_like(gamma, dtype=torch.bool)
            keep.scatter_(1, topk_idx, True)
            keep_f = keep.to(dtype=gamma.dtype)
    else:
        keep_f = torch.ones_like(gamma, dtype=gamma.dtype)

    # Effective weights per query: w_i * keep(q,i)
    w_eff = w.unsqueeze(0) * keep_f  # (M, N_eff)
    denom_q = w_eff.sum(dim=1).clamp_min(1e-12)  # (M,)

    # positive: sum_{i:y_i=c} w_i * gamma(q,i)
    w_g = w_eff * gamma  # (M, N_eff)
    idx = y.unsqueeze(0).expand(M, N_eff)
    pos = gamma.new_zeros((M, C))
    pos.scatter_add_(1, idx, w_g)

    # negative: sum_i w_i * (1-gamma(q,i)) * rho_i,c
    w_one_minus = w_eff * (1.0 - gamma)  # (M, N_eff)
    neg = w_one_minus @ rho  # (M, C)

    p = (pos + neg) / denom_q.unsqueeze(1)
    p = p.clamp_min(0.0)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return p.detach().cpu().numpy().astype(np.float64)


def _split_train_for_anchor_weights(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    val_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split X_train/y_train into (anchors, val) for learning anchor weights."""
    val_frac = float(val_frac)
    if not (0.0 < val_frac < 0.9):
        raise ValueError("--pdl_anchor_val_frac must be in (0, 0.9).")
    try:
        Xa, Xv, ya, yv = train_test_split(
            X_train,
            y_train,
            test_size=val_frac,
            random_state=int(seed),
            stratify=y_train,
        )
    except Exception:
        # Fallback if stratify fails (e.g., very rare classes).
        Xa, Xv, ya, yv = train_test_split(
            X_train,
            y_train,
            test_size=val_frac,
            random_state=int(seed),
            shuffle=True,
        )
    return Xa, ya, Xv, yv


@torch.no_grad()
def _make_pdl_anchor_clf(
    *,
    checkpoint_path: str,
    device: torch.device,
    n_estimators: int,
    batch_size: int,
    seed: int,
    pdlc_agg: str,
    pdlc_inference_temperature: Optional[float],
    pdlc_topk: Optional[int],
    pdlc_bilinear: Optional[str],
    pdlc_tau_override: Optional[float],
) -> "TabICLClassifier":
    # Local import to keep script startup fast when only printing help.
    from tabicl.sklearn.classifier import TabICLClassifier

    return TabICLClassifier(
        model_path=str(checkpoint_path),
        device=str(device),
        n_estimators=int(n_estimators),
        batch_size=int(batch_size),
        random_state=int(seed),
        pdlc_agg=pdlc_agg,
        pdlc_inference_temperature=pdlc_inference_temperature,
        pdlc_topk=pdlc_topk,
        pdlc_bilinear=pdlc_bilinear,
        pdlc_tau_override=pdlc_tau_override,
        allow_auto_download=False,
    )


def learn_anchor_weights_on_validation(
    *,
    gamma_val: torch.Tensor,  # (M_val, N)
    y_val: torch.Tensor,  # (M_val,)
    y_support: torch.Tensor,  # (N,)
    topk: Optional[int],
    steps: int,
    lr: float,
    prune_lambda: float,
    entropy_lambda: float,
    prune_topm: Optional[int],
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Learn anchor weights on a simplex (nonnegative, sum=1) by minimizing NLL on validation queries."""
    assert gamma_val.ndim == 2
    assert y_val.ndim == 1
    assert y_support.ndim == 1
    M_val, N = gamma_val.shape
    assert int(y_support.shape[0]) == int(N)
    assert int(y_val.shape[0]) == int(M_val)

    y_support = y_support.long()
    y_val = y_val.long()
    C = int(y_support.max().item()) + 1

    uniq = torch.unique(y_support)
    if int(uniq.numel()) <= 1:
        w = torch.ones((N,), dtype=torch.float32, device=device) / float(N)
        return w

    counts = torch.bincount(y_support, minlength=C).float()
    prior = counts / counts.sum().clamp_min(1.0)
    denom_y = (1.0 - prior.gather(0, y_support)).clamp_min(1e-6)
    rho = prior.unsqueeze(0) / denom_y.unsqueeze(1)  # (N, C)
    rho.scatter_(1, y_support.view(-1, 1), 0.0)

    gamma_val = gamma_val.float()

    if topk is not None:
        k = int(topk)
        if k <= 0 or k >= N:
            keep_f = torch.ones_like(gamma_val, dtype=gamma_val.dtype)
        else:
            topk_idx = gamma_val.topk(k, dim=1).indices  # (M, k)
            keep = torch.zeros_like(gamma_val, dtype=torch.bool)
            keep.scatter_(1, topk_idx, True)
            keep_f = keep.to(dtype=gamma_val.dtype)
    else:
        keep_f = torch.ones_like(gamma_val, dtype=gamma_val.dtype)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    v = torch.zeros((N,), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([v], lr=float(lr))

    eps = 1e-12
    for _ in range(int(steps)):
        w = torch.softmax(v, dim=0)  # (N,)
        w_eff = w.unsqueeze(0) * keep_f  # (M, N)
        denom_q = w_eff.sum(dim=1).clamp_min(eps)  # (M,)

        # pos
        w_g = w_eff * gamma_val
        idx = y_support.unsqueeze(0).expand(M_val, N)
        pos = gamma_val.new_zeros((M_val, C))
        pos.scatter_add_(1, idx, w_g)

        # neg
        w_one_minus = w_eff * (1.0 - gamma_val)
        neg = w_one_minus @ rho

        p = (pos + neg) / denom_q.unsqueeze(1)
        p = p.clamp_min(eps)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)

        nll = -torch.log(p.gather(1, y_val.view(-1, 1)).squeeze(1)).mean()
        loss = nll

        if float(prune_lambda) != 0.0:
            loss = loss - float(prune_lambda) * w.max()
        if float(entropy_lambda) != 0.0:
            ent = -(w * torch.log(w.clamp_min(eps))).sum()
            loss = loss + float(entropy_lambda) * ent

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    w = torch.softmax(v.detach(), dim=0)
    if prune_topm is not None:
        m = int(prune_topm)
        if m > 0 and m < N:
            idx = torch.topk(w, k=m).indices
            w_new = torch.zeros_like(w)
            w_new[idx] = w[idx]
            w = w_new / w_new.sum().clamp_min(eps)
    return w


def _parse_topk_candidates(v: str) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for s in _csv_list(v):
        if s.lower() in {"none", "null"}:
            out.append(None)
        else:
            k = int(s)
            out.append(None if k <= 0 else k)
    # de-dup, preserve order
    seen = set()
    uniq: List[Optional[int]] = []
    for k in out:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq


def _diag_metric(diag: BasicDiagnostics, metric: str) -> float:
    if metric == "macro_f1":
        return float(diag.macro_f1)
    if metric == "weighted_f1":
        return float(diag.weighted_f1)
    if metric == "acc":
        return float(diag.acc)
    raise ValueError(f"Unknown metric: {metric!r}")


def tune_pdlc_topk_on_support_val(
    *,
    checkpoint_path: str,
    device: torch.device,
    n_estimators: int,
    batch_size: int,
    seed: int,
    pdlc_agg: str,
    pdlc_inference_temperature: Optional[float],
    pdlc_bilinear: Optional[str],
    pdlc_tau_override: Optional[float],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    val_frac: float,
    candidates: List[Optional[int]],
    metric: str,
    variant: str,
) -> Tuple[Optional[int], List[Dict[str, Any]]]:
    """Tune top-k gating on a held-out subset of the training set.

    Uses the single-variant embedding path for speed. Returns (best_k, rows).
    """
    Xa, ya, Xv, yv = _split_train_for_anchor_weights(X_train, y_train, val_frac=val_frac, seed=seed)
    # Disable gating during tuning-data generation; we sweep it manually at the head level.
    pdl_tune_clf = _make_pdl_anchor_clf(
        checkpoint_path=str(checkpoint_path),
        device=device,
        n_estimators=int(n_estimators),
        batch_size=int(batch_size),
        seed=int(seed),
        pdlc_agg=str(pdlc_agg),
        pdlc_inference_temperature=pdlc_inference_temperature,
        pdlc_topk=None,
        pdlc_bilinear=pdlc_bilinear,
        pdlc_tau_override=pdlc_tau_override,
    )
    pdl_tune_clf.fit(Xa, ya)

    episode = materialize_episode_variant(
        pdl_tune_clf,
        Xv,
        variant=str(variant),
        rng=np.random.default_rng(seed),
        train_index=Xa.index,
        test_index=Xv.index,
    )
    infer_cfg = pdl_tune_clf.inference_config_
    model = pdl_tune_clf.model_
    model.eval()
    src = compute_icl_embeddings(model, episode, infer_cfg, device=device)

    # Encode validation labels in the tuning-clf label space
    y_true = encode_y_true_for_index(pdl_tune_clf, y_train, Xv.index)

    head = model.icl_predictor.pdlc_head
    base_norm = str(head.cfg.embed_norm)
    base_temp = float(getattr(head.cfg, "Inference_temperature", 1.0))

    rows: List[Dict[str, Any]] = []
    best_k: Optional[int] = None
    best_score: float = float("-inf")
    best_acc: float = float("-inf")

    for k in candidates:
        logits = pdl_variant_logits_from_embeddings(
            model,
            src=src,
            episode=episode,
            agg=str(pdlc_agg),
            topk=k,
            embed_norm=base_norm,
            infer_temp=base_temp if pdlc_inference_temperature is None else float(pdlc_inference_temperature),
            device=device,
        )
        proba = logits_to_proba(logits)
        diag, _, _ = compute_basic_diagnostics(y_true, proba)
        score = _diag_metric(diag, metric)
        row = {
            "topk": k,
            "score": float(score),
            "acc": float(diag.acc),
            "macro_f1": float(diag.macro_f1),
            "weighted_f1": float(diag.weighted_f1),
            "n_predicted_classes": int(diag.n_predicted_classes),
            "pred_entropy_norm": float(diag.pred_entropy_norm),
            "margin_median": float(diag.margin_median),
        }
        rows.append(row)
        # Tie-breaks: higher score, then higher acc, then smaller k (for speed).
        k_rank = int(k) if k is not None else -1
        best_rank = int(best_k) if best_k is not None else -1
        if (score > best_score) or (score == best_score and diag.acc > best_acc) or (
            score == best_score and diag.acc == best_acc and k_rank < best_rank
        ):
            best_score = float(score)
            best_acc = float(diag.acc)
            best_k = k

    return best_k, rows


@dataclass
class VariantRun:
    name: str
    acc: float
    pred_counts: List[int]
    n_predicted_classes: int
    top1_frac: float
    top2_frac: float
    margin_median: float


@torch.no_grad()
def pdl_variant_logits_from_embeddings(
    model,
    src: torch.Tensor,  # (T, D)
    episode: EpisodeVariant,
    *,
    agg: str,
    topk: Optional[int],
    embed_norm: str,
    infer_temp: Optional[float],
    device: torch.device,
) -> np.ndarray:
    head = getattr(model.icl_predictor, "pdlc_head", None)
    if head is None:
        raise RuntimeError("Model has no pdlc_head; did you pass a TabPDL checkpoint?")

    old_cfg = copy.deepcopy(head.cfg)
    try:
        head.cfg.agg = str(agg)
        head.cfg.topk = topk
        head.cfg.embed_norm = str(embed_norm)
        if infer_temp is not None:
            head.cfg.Inference_temperature = float(infer_temp)

        train_size = episode.train_size
        H_support = src[:train_size].unsqueeze(0).to(device)
        H_query = src[train_size:].unsqueeze(0).to(device)
        y_support = torch.from_numpy(episode.y_train_shifted).long().unsqueeze(0).to(device)
        support_mask = torch.ones((1, train_size), dtype=torch.bool, device=device)

        logits_q, _aux = head(H_support=H_support, H_query=H_query, y_support=y_support, support_mask=support_mask)
        logits_q = logits_q.squeeze(0).detach().cpu().numpy()
        logits_q = unshift_class_axis(logits_q, episode.class_shift_offset)
        return logits_q
    finally:
        head.cfg = old_cfg


def logits_to_proba(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x_max = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=-1, keepdims=True)


def summarize_variant_run(
    name: str,
    y_true: np.ndarray,
    logits: np.ndarray,
) -> VariantRun:
    proba = logits_to_proba(logits)
    diag, y_pred, margin = compute_basic_diagnostics(y_true, proba)
    return VariantRun(
        name=name,
        acc=diag.acc,
        pred_counts=diag.pred_counts,
        n_predicted_classes=diag.n_predicted_classes,
        top1_frac=diag.pred_top1_frac,
        top2_frac=diag.pred_top2_frac,
        margin_median=diag.margin_median,
    )


# ---------------------------------------------------------------------------
# Head swap (D)
# ---------------------------------------------------------------------------


def load_training_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt_path = Path(path).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "config" not in ckpt or "state_dict" not in ckpt:
        raise RuntimeError(f"Checkpoint at {ckpt_path} does not contain 'config' and 'state_dict'.")
    return ckpt


def _config_core(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "max_classes",
        "embed_dim",
        "col_num_blocks",
        "col_nhead",
        "col_num_inds",
        "row_num_blocks",
        "row_nhead",
        "row_num_cls",
        "icl_num_blocks",
        "icl_nhead",
        "ff_factor",
        "dropout",
        "norm_first",
        "row_elliptical",
        "icl_elliptical",
        "elliptical_delta",
        "elliptical_scale_mode",
    ]
    return {k: cfg.get(k) for k in keys if k in cfg}


def merge_state_dicts(
    base: Dict[str, torch.Tensor],
    override: Dict[str, torch.Tensor],
    prefixes: Sequence[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    out = dict(base)
    notes: Dict[str, str] = {}
    for k, v in override.items():
        if not any(k.startswith(p) for p in prefixes):
            continue
        if k not in out:
            notes[k] = "skip_missing_in_base"
            continue
        if tuple(out[k].shape) != tuple(v.shape):
            notes[k] = f"skip_shape_mismatch base={tuple(out[k].shape)} override={tuple(v.shape)}"
            continue
        out[k] = v
        notes[k] = "overrode"
    return out, notes


@torch.no_grad()
def eval_model_on_episode_logits(
    model,
    episode: EpisodeVariant,
    inference_config,
    device: torch.device,
    *,
    return_proba: bool = True,
    softmax_temperature: float = 0.9,
) -> np.ndarray:
    train_size = episode.train_size
    X = torch.from_numpy(episode.X_variant).float().unsqueeze(0).to(device)
    y_train = torch.from_numpy(episode.y_train_shifted).float().unsqueeze(0).to(device)
    feature_shuffles = [episode.feature_permutation]
    model.eval()
    logits = model(
        X,
        y_train,
        feature_shuffles=feature_shuffles,
        return_logits=True,
        softmax_temperature=softmax_temperature,
        inference_config=inference_config,
    ).squeeze(0)
    logits = logits.detach().cpu().numpy()
    logits = unshift_class_axis(logits, episode.class_shift_offset)
    if return_proba:
        return logits_to_proba(logits)
    return logits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _maybe_write_json(path: Path, obj: Any, no_save: bool) -> None:
    if no_save:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _maybe_write_csv(path: Path, df: pd.DataFrame, no_save: bool) -> None:
    if no_save:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = make_parser().parse_args()
    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Lazily import to keep this script usable even when only some checkpoints are provided.
    from tabicl.sklearn.classifier import TabICLClassifier  # type: ignore
    from tabicl import InferenceConfig  # type: ignore
    from tabicl.model.tabicl import TabICL  # type: ignore

    dataset_specs: List[Dict[str, Any]] = []
    if args.datasets is not None:
        for d in _csv_list(args.datasets):
            dataset_specs.append({"kind": "dataset", "id_or_name": d})
    if args.openml_tasks is not None:
        for t in _csv_list(args.openml_tasks):
            dataset_specs.append({"kind": "task", "task_id": int(t), "fold": int(args.fold)})

    if args.sa_checkpoint is None and args.pdl_checkpoint is None:
        raise SystemExit("Provide at least one of --sa_checkpoint or --pdl_checkpoint.")

    overall: Dict[str, Any] = {
        "args": vars(args),
        "datasets": [],
    }

    for spec in dataset_specs:
        if spec["kind"] == "task":
            X, y, ds_name, ds_id, tr_idx, te_idx = fetch_openml_task(int(spec["task_id"]), int(spec["fold"]))
            X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
            X_test, y_test = X.iloc[te_idx], y.iloc[te_idx]
            split_info = {"kind": "openml_task", "task_id": int(spec["task_id"]), "fold": int(spec["fold"])}
        else:
            X, y, ds_name, ds_id = fetch_openml_dataset(spec["id_or_name"])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(args.test_size), random_state=int(args.seed), stratify=y
            )
            split_info = {"kind": "random_stratified", "test_size": float(args.test_size), "seed": int(args.seed)}

        ds_out = {
            "dataset_name": ds_name,
            "dataset_id": int(ds_id),
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "split": split_info,
            "models": {},
            "pdl_gamma": None,
            "variant": None,
            "variant_runs": [],
            "head_swap": None,
            "pdlc_topk_tune": None,
        }
        print(f"\n=== Dataset: {ds_name} (id={ds_id}) | rows={X.shape[0]} feats={X.shape[1]} ===")

        # Allow per-dataset top-k tuning (select k on a held-out split of the train supports).
        pdlc_topk_effective: Optional[int] = args.pdlc_topk
        if bool(args.pdlc_topk_tune):
            if args.pdl_checkpoint is None:
                raise SystemExit("--pdlc_topk_tune requires --pdl_checkpoint.")
            try:
                candidates = _parse_topk_candidates(str(args.pdlc_topk_tune_candidates))
                best_k, rows = tune_pdlc_topk_on_support_val(
                    checkpoint_path=str(args.pdl_checkpoint),
                    device=device,
                    n_estimators=int(args.n_estimators),
                    batch_size=int(args.batch_size),
                    seed=int(args.seed),
                    pdlc_agg=str(args.pdlc_agg),
                    pdlc_inference_temperature=args.pdlc_inference_temperature,
                    pdlc_bilinear=args.pdlc_bilinear,
                    pdlc_tau_override=args.pdlc_tau_override,
                    X_train=X_train,
                    y_train=y_train,
                    val_frac=float(args.pdlc_topk_tune_val_frac),
                    candidates=candidates,
                    metric=str(args.pdlc_topk_tune_metric),
                    variant=str(args.variant),
                )
                pdlc_topk_effective = best_k
                ds_out["pdlc_topk_tune"] = {
                    "val_frac": float(args.pdlc_topk_tune_val_frac),
                    "metric": str(args.pdlc_topk_tune_metric),
                    "candidates": candidates,
                    "best_topk": best_k,
                    "rows": rows,
                    "note": "Tuning uses a single-variant episode on a held-out subset of the train split; the chosen topk is then used for full-ensemble evaluation on the full train split.",
                }
                k_str = "None" if best_k is None else str(int(best_k))
                print(f"[PDL topk-tune] metric={args.pdlc_topk_tune_metric} best_topk={k_str} candidates={candidates}")
                if not args.no_save:
                    base = out_dir / ds_name / "pdlc_topk_tune"
                    _maybe_write_csv(base / "tune.csv", pd.DataFrame(rows), args.no_save)
            except Exception as e:
                print(f"[warn] Failed to tune PDLC top-k for dataset {ds_name}: {e}")

        # -------------------------------
        # A/C on sklearn wrapper outputs
        # -------------------------------
        if args.sa_checkpoint is not None:
            sa = TabICLClassifier(
                n_estimators=int(args.n_estimators),
                batch_size=int(args.batch_size),
                use_amp=bool(args.use_amp),
                verbose=bool(args.verbose),
                model_path=args.sa_checkpoint,
                allow_auto_download=False,
                device=str(device),
                softmax_temperature=float(args.softmax_temperature),
                average_logits=True,
            )
            sa.fit(X_train, y_train)
            y_true_sa = encode_y_true_for_index(sa, y_test, X_test.index)
            proba_sa = sa.predict_proba(X_test)
            diag_sa, _, margin_sa = compute_basic_diagnostics(y_true_sa, proba_sa)
            print_basic("SA", list(map(str, sa.classes_)), diag_sa)
            print_recall_and_confusion("SA", list(map(str, sa.classes_)), diag_sa)
            ds_out["models"]["sa"] = asdict(diag_sa)

            if not args.no_save:
                base = out_dir / ds_name / "sa"
                _maybe_write_csv(base / "pred_counts.csv", pd.DataFrame({"class": sa.classes_, "count": diag_sa.pred_counts}), args.no_save)
                _maybe_write_csv(base / "per_class_recall.csv", pd.DataFrame({"class": sa.classes_, "recall": diag_sa.per_class_recall}), args.no_save)
                _maybe_write_csv(base / "confusion.csv", pd.DataFrame(diag_sa.confusion, columns=list(sa.classes_)), args.no_save)
                np.savez(base / "margins.npz", margin=margin_sa.astype(np.float32))

        pdl_clf = None
        if args.pdl_checkpoint is not None:
            pdl_clf = TabICLClassifier(
                n_estimators=int(args.n_estimators),
                batch_size=int(args.batch_size),
                use_amp=bool(args.use_amp),
                verbose=bool(args.verbose),
                model_path=args.pdl_checkpoint,
                allow_auto_download=False,
                device=str(device),
                softmax_temperature=float(args.softmax_temperature),
                average_logits=True,
                pdlc_agg=str(args.pdlc_agg) if args.pdlc_agg is not None else None,
                pdlc_inference_temperature=args.pdlc_inference_temperature,
                pdlc_topk=pdlc_topk_effective,
                pdlc_bilinear=args.pdlc_bilinear,
                pdlc_tau_override=args.pdlc_tau_override,
            )
            pdl_clf.fit(X_train, y_train)
            # Log the effective PDLC head config after applying wrapper overrides.
            try:
                head = pdl_clf.model_.icl_predictor.pdlc_head
                cfg = getattr(head, "cfg", None)
                if cfg is not None:
                    print(
                        f"[PDL cfg] agg={getattr(cfg, 'agg', None)} topk={getattr(cfg, 'topk', None)} "
                        f"embed_norm={getattr(cfg, 'embed_norm', None)} "
                        f"Inference_temperature={getattr(cfg, 'Inference_temperature', None)}"
                    )
            except Exception:
                pass
            y_true_pdl = encode_y_true_for_index(pdl_clf, y_test, X_test.index)
            proba_pdl = pdl_clf.predict_proba(X_test)
            diag_pdl, _, margin_pdl = compute_basic_diagnostics(y_true_pdl, proba_pdl)
            print_basic("PDL", list(map(str, pdl_clf.classes_)), diag_pdl)
            print_recall_and_confusion("PDL", list(map(str, pdl_clf.classes_)), diag_pdl)
            ds_out["models"]["pdl"] = asdict(diag_pdl)

            if not args.no_save:
                base = out_dir / ds_name / "pdl"
                _maybe_write_csv(base / "pred_counts.csv", pd.DataFrame({"class": pdl_clf.classes_, "count": diag_pdl.pred_counts}), args.no_save)
                _maybe_write_csv(base / "per_class_recall.csv", pd.DataFrame({"class": pdl_clf.classes_, "recall": diag_pdl.per_class_recall}), args.no_save)
                _maybe_write_csv(base / "confusion.csv", pd.DataFrame(diag_pdl.confusion, columns=list(pdl_clf.classes_)), args.no_save)
                np.savez(base / "margins.npz", margin=margin_pdl.astype(np.float32))

        # -------------------------------
        # B/E on a single variant episode
        # -------------------------------
        if pdl_clf is not None:
            episode = materialize_episode_variant(
                pdl_clf,
                X_test,
                variant=str(args.variant),
                rng=rng,
                train_index=X_train.index,
                test_index=X_test.index,
            )
            ds_out["variant"] = asdict(episode) | {"X_variant": None, "y_train_shifted": None}
            # Only store shapes for JSON
            ds_out["variant"]["X_shape"] = list(map(int, episode.X_variant.shape))
            ds_out["variant"]["y_train_shifted_shape"] = list(map(int, episode.y_train_shifted.shape))
            ds_out["variant"].pop("X_variant", None)
            ds_out["variant"].pop("y_train_shifted", None)

            # Precompute ICL embeddings once (used for all single-variant diagnostics).
            infer_cfg = pdl_clf.inference_config_
            pdl_model = pdl_clf.model_
            pdl_model.eval()
            src = compute_icl_embeddings(pdl_model, episode, infer_cfg, device=device)

            # Compute gamma separation on supports
            gamma_sep = compute_support_support_gamma_auc(
                pdl_model,
                episode,
                infer_cfg,
                device=device,
                max_pairs=int(args.max_pairs),
                seed=int(args.seed),
                src=src,
            )
            ds_out["pdl_gamma"] = asdict(gamma_sep)
            print(
                f"\n[PDL gamma (topk=None)] support pairs={gamma_sep.n_pairs_used} (N={gamma_sep.n_support}) "
                f"AUC={gamma_sep.auc:.4f} AP={gamma_sep.ap:.4f} | "
                f"mean(same)={gamma_sep.gamma_same_mean:.3f} mean(diff)={gamma_sep.gamma_diff_mean:.3f}"
            )

            # PDLC anchor-based uncertainty decomposition (TU/AU/EU).
            if bool(args.pdl_uncertainty):
                try:
                    u, tu, au, eu, kept = compute_pdl_anchor_uncertainty_breakdown(
                        pdl_model, episode, infer_cfg, device=device, src=src
                    )
                    ds_out["pdl_anchor_uncertainty"] = asdict(u)
                    print(
                        f"[PDL uncertainty] topk={u.topk} kept={u.kept_min}/{u.kept_mean:.1f}/{u.kept_max} "
                        f"TU={u.tu_mean:.3f} AU={u.au_mean:.3f} EU={u.eu_mean:.3f} "
                        f"(norm TU/AU/EU={u.tu_norm_mean:.3f}/{u.au_norm_mean:.3f}/{u.eu_norm_mean:.3f})"
                    )
                    if not args.no_save:
                        base = out_dir / ds_name / "pdl_anchor_uncertainty"
                        base.mkdir(parents=True, exist_ok=True)
                        np.savez(
                            base / "uncertainty.npz",
                            tu=tu.astype(np.float32),
                            au=au.astype(np.float32),
                            eu=eu.astype(np.float32),
                            kept=kept.astype(np.float32),
                        )
                except Exception as e:
                    print(f"[warn] Failed to compute PDL uncertainty breakdown: {e}")

            # Weighted PDLC: learn anchor weights on a held-out validation split of the support set.
            if bool(args.pdl_learn_anchor_weights):
                try:
                    if args.pdl_checkpoint is None:
                        raise ValueError("--pdl_learn_anchor_weights requires --pdl_checkpoint.")

                    Xa, ya, Xv, yv = _split_train_for_anchor_weights(
                        X_train, y_train, val_frac=float(args.pdl_anchor_val_frac), seed=int(args.seed)
                    )
                    # Learn weights on Xv, evaluate on X_test. Both use the same anchor set Xa.
                    Xq = pd.concat([Xv, X_test], axis=0)
                    yq = pd.concat([yv, y_test.loc[X_test.index]], axis=0)
                    assert [str(i) for i in list(Xq.index)] == [str(i) for i in list(yq.index)]

                    pdl_anchor_clf = _make_pdl_anchor_clf(
                        checkpoint_path=str(args.pdl_checkpoint),
                        device=device,
                        n_estimators=int(args.n_estimators),
                        batch_size=int(args.batch_size),
                        seed=int(args.seed),
                        pdlc_agg=str(args.pdlc_agg),
                        pdlc_inference_temperature=args.pdlc_inference_temperature,
                        pdlc_topk=pdlc_topk_effective,
                        pdlc_bilinear=args.pdlc_bilinear,
                        pdlc_tau_override=args.pdlc_tau_override,
                    )
                    pdl_anchor_clf.fit(Xa, ya)

                    episode_w = materialize_episode_variant(
                        pdl_anchor_clf,
                        Xq,
                        variant=str(args.variant),
                        rng=rng,
                        train_index=Xa.index,
                        test_index=Xq.index,
                    )

                    infer_cfg_w = pdl_anchor_clf.inference_config_
                    model_w = pdl_anchor_clf.model_
                    model_w.eval()
                    src_w = compute_icl_embeddings(model_w, episode_w, infer_cfg_w, device=device)

                    train_size_w = int(episode_w.train_size)
                    val_size = int(len(Xv))
                    test_size = int(len(X_test))
                    assert int(episode_w.test_size) == val_size + test_size

                    head_w = model_w.icl_predictor.pdlc_head
                    H_support_w = src_w[:train_size_w].unsqueeze(0).to(device)
                    H_query_all_w = src_w[train_size_w:].unsqueeze(0).to(device)
                    H_query_val_w = H_query_all_w[:, :val_size, :]
                    H_query_test_w = H_query_all_w[:, val_size:, :]
                    support_mask_w = torch.ones((1, train_size_w), dtype=torch.bool, device=device)
                    y_support_w = torch.from_numpy(episode_w.y_train_shifted).long().unsqueeze(0).to(device)

                    # Encode labels in the anchor-clf label space
                    y_val_enc = encode_y_true_for_index(pdl_anchor_clf, yq, Xv.index)
                    y_val_t = torch.from_numpy(y_val_enc).to(device).long()
                    y_test_enc = encode_y_true_for_index(pdl_anchor_clf, yq, X_test.index)

                    # Gamma for val queries vs anchors
                    logits_val = head_w._pair_logits(
                        H_query=H_query_val_w, H_support=H_support_w, support_mask=support_mask_w
                    ).squeeze(0)
                    gamma_val = torch.sigmoid(logits_val).detach()

                    use_topk = None
                    if bool(args.pdl_anchor_weight_use_topk):
                        use_topk = getattr(head_w.cfg, "topk", None)

                    w = learn_anchor_weights_on_validation(
                        gamma_val=gamma_val,
                        y_val=y_val_t,
                        y_support=y_support_w.squeeze(0),
                        topk=use_topk,
                        steps=int(args.pdl_anchor_weight_steps),
                        lr=float(args.pdl_anchor_weight_lr),
                        prune_lambda=float(args.pdl_anchor_weight_prune_lambda),
                        entropy_lambda=float(args.pdl_anchor_weight_entropy_lambda),
                        prune_topm=args.pdl_anchor_weight_prune_topm,
                        seed=int(args.seed),
                        device=device,
                    )

                    # Baseline: uniform weights on the same anchor set
                    w_uniform = torch.ones_like(w) / w.numel()

                    proba_uniform = compute_weighted_posterior_avg_proba(
                        head_w,
                        H_support=H_support_w,
                        H_query=H_query_test_w,
                        y_support=y_support_w,
                        support_mask=support_mask_w,
                        weights=w_uniform,
                        topk=use_topk,
                    )
                    proba_w = compute_weighted_posterior_avg_proba(
                        head_w,
                        H_support=H_support_w,
                        H_query=H_query_test_w,
                        y_support=y_support_w,
                        support_mask=support_mask_w,
                        weights=w.detach(),
                        topk=use_topk,
                    )
                    # Undo any class-shift for metric computation (match the wrapper's behavior).
                    proba_uniform = unshift_class_axis(proba_uniform, episode_w.class_shift_offset)
                    proba_w = unshift_class_axis(proba_w, episode_w.class_shift_offset)
                    diag_w, _, _ = compute_basic_diagnostics(y_test_enc, proba_w)
                    diag_u, _, _ = compute_basic_diagnostics(y_test_enc, proba_uniform)
                    print_basic("PDL uniform@anchors", list(map(str, pdl_anchor_clf.classes_)), diag_u)
                    print_basic("PDL weighted", list(map(str, pdl_anchor_clf.classes_)), diag_w)

                    ds_out["pdl_weighted"] = {
                        "anchor_val_frac": float(args.pdl_anchor_val_frac),
                        "n_anchors": int(train_size_w),
                        "n_val": int(val_size),
                        "n_test": int(test_size),
                        "use_topk": use_topk,
                        "prune_lambda": float(args.pdl_anchor_weight_prune_lambda),
                        "entropy_lambda": float(args.pdl_anchor_weight_entropy_lambda),
                        "prune_topm": int(args.pdl_anchor_weight_prune_topm) if args.pdl_anchor_weight_prune_topm else None,
                        "diag_uniform": asdict(diag_u),
                        "diag_weighted": asdict(diag_w),
                    }

                    if not args.no_save:
                        base = out_dir / ds_name / "pdl_weighted"
                        base.mkdir(parents=True, exist_ok=True)
                        weights_np = w.detach().cpu().numpy().astype(np.float32)
                        np.save(base / "weights.npy", weights_np)
                        _maybe_write_csv(
                            base / "weights.csv",
                            pd.DataFrame(
                                {
                                    "anchor_index": list(map(str, Xa.index)),
                                    "anchor_label": list(map(str, ya.values)),
                                    "weight": weights_np,
                                }
                            ),
                            args.no_save,
                        )
                        np.savez(
                            base / "proba_test.npz",
                            proba_uniform=proba_uniform.astype(np.float32),
                            proba_weighted=proba_w.astype(np.float32),
                            y_true=y_test_enc.astype(np.int64),
                        )
                except Exception as e:
                    print(f"[warn] Failed to learn/apply anchor weights: {e}")

            # Ablations / single-variant runs
            assert [str(i) for i in list(X_test.index)] == episode.test_index, "Episode test_index mismatch vs X_test."
            y_true_variant = encode_y_true_for_index(pdl_clf, y_test, X_test.index)

            # Precompute ICL embeddings once (used for all sweeps that only affect the head)
            # (already computed above)

            def _parse_topk_list(v: Optional[str]) -> List[Optional[int]]:
                out: List[Optional[int]] = []
                for s in _csv_list(v):
                    if s.lower() in {"none", "null"}:
                        out.append(None)
                    else:
                        out.append(int(s))
                return out

            agg_sweep = _csv_list(args.pdl_agg_sweep) or []
            topk_sweep = _parse_topk_list(args.pdl_topk_sweep) if args.pdl_topk_sweep else []
            norm_sweep = _csv_list(args.pdl_embed_norm_sweep) or []
            temp_sweep = [float(x) for x in _csv_list(args.pdl_infer_temp_sweep)] if args.pdl_infer_temp_sweep else []

            if agg_sweep or topk_sweep or norm_sweep or temp_sweep:
                # Build a manageable cartesian product; default to current values when not swept.
                head = pdl_model.icl_predictor.pdlc_head
                base_agg = str(head.cfg.agg)
                base_topk = head.cfg.topk
                base_norm = str(head.cfg.embed_norm)
                base_temp = float(getattr(head.cfg, "Inference_temperature", 1.0))

                aggs = agg_sweep or [base_agg]
                if agg_sweep and base_agg not in aggs:
                    # Ensure the faithful/baseline config is included for direct comparison.
                    aggs = [base_agg] + aggs
                topks = topk_sweep or [base_topk]
                norms = norm_sweep or [base_norm]
                temps = temp_sweep or [base_temp]

                # Hard cap to avoid accidental combinatorial explosions
                cap = 48
                combos_all = [(a, k, n, t) for a in aggs for k in topks for n in norms for t in temps]
                baseline = (base_agg, base_topk, base_norm, base_temp)
                if len(combos_all) > cap:
                    # Randomly sample configs to avoid systematic ordering bias.
                    # Keep the baseline config if present.
                    idx_all = np.arange(len(combos_all))
                    if baseline in combos_all and cap >= 2:
                        baseline_idx = combos_all.index(baseline)
                        remaining = np.delete(idx_all, baseline_idx)
                        pick = rng.choice(remaining, size=cap - 1, replace=False)
                        idx = np.concatenate([[baseline_idx], pick])
                    else:
                        idx = rng.choice(idx_all, size=cap, replace=False)
                    combos = [combos_all[int(i)] for i in idx]
                    print(f"[warn] Ablation combinations randomly capped to {cap} configs (total={len(combos_all)}).")
                else:
                    combos = combos_all

                for agg, topk, norm, temp in combos:
                    tag = f"agg={agg}|topk={topk}|norm={norm}|temp={temp:g}"
                    logits = pdl_variant_logits_from_embeddings(
                        pdl_model,
                        src=src,
                        episode=episode,
                        agg=agg,
                        topk=topk,
                        embed_norm=norm,
                        infer_temp=temp,
                        device=device,
                    )
                    run = summarize_variant_run(f"pdl_ablate:{tag}", y_true_variant, logits)
                    ds_out["variant_runs"].append(asdict(run))
                    if args.verbose:
                        print(
                            f"[ablate] {tag} acc={run.acc:.4f} n_pred={run.n_predicted_classes}/{len(run.pred_counts)} "
                            f"top1={run.top1_frac:.3f} margin_med={run.margin_median:.3f}"
                        )

        # -------------------------------
        # D: head swap test (single-variant)
        # -------------------------------
        if bool(args.head_swap):
            if args.sa_checkpoint is None or args.pdl_checkpoint is None:
                print("[warn] --head_swap requires both --sa_checkpoint and --pdl_checkpoint; skipping.")
            else:
                # Load checkpoints and verify core architecture compatibility
                ckpt_sa = load_training_checkpoint(args.sa_checkpoint, device=torch.device("cpu"))
                ckpt_pdl = load_training_checkpoint(args.pdl_checkpoint, device=torch.device("cpu"))
                cfg_sa = ckpt_sa["config"]
                cfg_pdl = ckpt_pdl["config"]
                core_sa = _config_core(cfg_sa)
                core_pdl = _config_core(cfg_pdl)
                if core_sa != core_pdl:
                    print("[warn] SA and PDL configs differ; head swap may be invalid. Differences:")
                    for k in sorted(set(core_sa) | set(core_pdl)):
                        if core_sa.get(k) != core_pdl.get(k):
                            print(f"  {k}: sa={core_sa.get(k)} pdl={core_pdl.get(k)}")

                # Episode: use PDL classifier (already fitted) if available, else fit a small helper
                if pdl_clf is None:
                    helper = TabICLClassifier(
                        n_estimators=8,
                        batch_size=int(args.batch_size),
                        use_amp=bool(args.use_amp),
                        verbose=bool(args.verbose),
                        model_path=args.pdl_checkpoint,
                        allow_auto_download=False,
                        device=str(device),
                        softmax_temperature=float(args.softmax_temperature),
                        average_logits=True,
                    )
                    helper.fit(X_train, y_train)
                    pdl_clf_for_episode = helper
                else:
                    pdl_clf_for_episode = pdl_clf

                episode = materialize_episode_variant(
                    pdl_clf_for_episode,
                    X_test,
                    variant=str(args.variant),
                    rng=rng,
                    train_index=X_train.index,
                    test_index=X_test.index,
                )
                assert [str(i) for i in list(X_test.index)] == episode.test_index, "Episode test_index mismatch vs X_test."
                y_true_variant = encode_y_true_for_index(pdl_clf_for_episode, y_test, X_test.index)

                backbone_prefixes = [
                    "col_embedder.",
                    "row_interactor.",
                    "icl_predictor.tf_icl.",
                    "icl_predictor.y_encoder.",
                    "icl_predictor.ln.",
                ]

                # PDL head on SA backbone: base = PDL checkpoint, override backbone from SA
                model_pdl_on_sa = TabICL(**cfg_pdl).to(device)
                merged_a, notes_a = merge_state_dicts(
                    base=ckpt_pdl["state_dict"], override=ckpt_sa["state_dict"], prefixes=backbone_prefixes
                )
                missing, unexpected = model_pdl_on_sa.load_state_dict(merged_a, strict=False)
                if args.verbose:
                    print(f"[swap] PDL-on-SA load: missing={len(missing)} unexpected={len(unexpected)} overrode={sum(v=='overrode' for v in notes_a.values())}")

                # SA head on PDL backbone: base = SA checkpoint, override backbone from PDL
                model_sa_on_pdl = TabICL(**cfg_sa).to(device)
                merged_b, notes_b = merge_state_dicts(
                    base=ckpt_sa["state_dict"], override=ckpt_pdl["state_dict"], prefixes=backbone_prefixes
                )
                missing, unexpected = model_sa_on_pdl.load_state_dict(merged_b, strict=False)
                if args.verbose:
                    print(f"[swap] SA-on-PDL load: missing={len(missing)} unexpected={len(unexpected)} overrode={sum(v=='overrode' for v in notes_b.values())}")

                infer_cfg = InferenceConfig()
                init = {
                    "COL_CONFIG": {"device": device, "use_amp": bool(args.use_amp), "verbose": bool(args.verbose)},
                    "ROW_CONFIG": {"device": device, "use_amp": bool(args.use_amp), "verbose": bool(args.verbose)},
                    "ICL_CONFIG": {"device": device, "use_amp": bool(args.use_amp), "verbose": bool(args.verbose)},
                }
                infer_cfg.update_from_dict(init)

                proba_pdl_on_sa = eval_model_on_episode_logits(
                    model_pdl_on_sa, episode, infer_cfg, device=device, return_proba=True, softmax_temperature=float(args.softmax_temperature)
                )
                diag_a, _, _ = compute_basic_diagnostics(y_true_variant, proba_pdl_on_sa)

                proba_sa_on_pdl = eval_model_on_episode_logits(
                    model_sa_on_pdl, episode, infer_cfg, device=device, return_proba=True, softmax_temperature=float(args.softmax_temperature)
                )
                diag_b, _, _ = compute_basic_diagnostics(y_true_variant, proba_sa_on_pdl)

                episode_meta = asdict(episode) | {"X_variant": None, "y_train_shifted": None}
                try:
                    episode_meta["X_shape"] = list(map(int, episode.X_variant.shape))
                    episode_meta["y_train_shifted_shape"] = list(map(int, episode.y_train_shifted.shape))
                except Exception:
                    pass
                episode_meta.pop("X_variant", None)
                episode_meta.pop("y_train_shifted", None)

                ds_out["head_swap"] = {
                    "pdl_on_sa": asdict(diag_a),
                    "sa_on_pdl": asdict(diag_b),
                    "episode": episode_meta,
                }

                print_basic("swap:PDL-on-SA", list(map(str, pdl_clf_for_episode.classes_)), diag_a)
                print_basic("swap:SA-on-PDL", list(map(str, pdl_clf_for_episode.classes_)), diag_b)

        # Persist
        overall["datasets"].append(ds_out)
        _maybe_write_json(out_dir / ds_name / "summary.json", ds_out, args.no_save)

    _maybe_write_json(out_dir / "summary_all.json", overall, args.no_save)


if __name__ == "__main__":
    main()
