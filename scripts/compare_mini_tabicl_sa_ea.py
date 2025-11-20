#!/usr/bin/env python3
"""
Compare mini-TabICL Stage-1 SA vs EA (behaviour + geometry).

This script compares two TabICL checkpoints:
  - M_SA: standard attention  (stage-1 mini TabICL)
  - M_EA: elliptical attention (stage-1 mini TabICL)

It runs both models on the same evaluation panel and reports:

1) Behavioural metrics on a fixed panel (no fine-tuning):
   - Accuracy, macro-F1
   - Log-loss (NLL), Brier score
   - Expected Calibration Error (ECE)
   - Optionally as a function of context length (train_size / ICL length)

2) Representation geometry on TFrow / TFicl:
   - Covariance spectrum & collapse scores (variance in top-k PCs)
   - Mean pairwise cosine similarity between row embeddings
   - Linear CKA similarity between SA and EA embeddings
   - Effective number of attended neighbours N_eff from TFicl attention

Defaults:
  - Uses OpenML IDs or names passed via --datasets
  - Uses mini-TabICL Stage-1 checkpoints at:
       checkpoints_mini_tabicl_stage1_sa/step-25000.ckpt
       checkpoints_mini_tabicl_stage1_ea/step-25000.ckpt

Usage examples
--------------
  # Quick behavioural + geometry comparison on a few OpenML datasets
  python scripts/compare_mini_tabicl_sa_ea.py \\
      --datasets 23,48,475,714 \\
      --context_lengths 128 256 512

  # Use custom checkpoints and only behavioural metrics
  python scripts/compare_mini_tabicl_sa_ea.py \\
      --sa_checkpoint checkpoints_mini_tabicl_stage1_sa/step-25000.ckpt \\
      --ea_checkpoint checkpoints_mini_tabicl_stage1_ea/step-25000.ckpt \\
      --datasets 23,48 \\
      --context_lengths 256 \\
      --skip_geometry
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

# Make local src importable
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tabicl.sklearn.classifier import TabICLClassifier
from tabicl.pdlc.embed import extract_tf_row_embeddings
from tabicl.model.attention import compute_elliptical_diag


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed_all(seed)


def fetch_openml_dataset(name_or_id: str | int) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Load an OpenML dataset by ID or (partial) name.

    Returns
    -------
    X : DataFrame
    y : Series
    name : str
    """
    import openml

    # By integer ID
    if isinstance(name_or_id, int) or (str(name_or_id).isdigit()):
        ds = openml.datasets.get_dataset(int(name_or_id))
        X, y, _, _ = ds.get_data(
            target=getattr(ds, "default_target_attribute", None),
            dataset_format="dataframe",
        )
        if y is None:
            raise ValueError(f"Dataset {ds.name} (id={ds.dataset_id}) has no target.")
        return X, y, ds.name

    # By name: prefer exact match; else substring on name
    name = str(name_or_id).lower()
    df = openml.datasets.list_datasets(output_format="dataframe")
    exact = df[df["name"].str.lower() == name]
    if len(exact) == 0:
        contains = df[df["name"].str.lower().str.contains(name)]
        if len(contains) == 0:
            raise ValueError(f"OpenML dataset not found: {name_or_id}")
        row = contains.sort_values("NumberOfInstances").iloc[0]
    else:
        row = exact.sort_values("NumberOfInstances").iloc[0]
    did = int(row["did"]) if "did" in row else int(row.get("dataset_id", row.get("ID")))
    ds = openml.datasets.get_dataset(did)
    X, y, _, _ = ds.get_data(
        target=getattr(ds, "default_target_attribute", None),
        dataset_format="dataframe",
    )
    if y is None:
        raise ValueError(f"Dataset {ds.name} (id={did}) has no target.")
    return X, y, ds.name


def compute_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) over max-prob bins."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if proba.ndim != 2:
        raise ValueError("proba must be 2D (n_samples, n_classes)")
    confidences = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    correct = (preds == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        conf_bin = confidences[mask].mean()
        acc_bin = correct[mask].mean()
        ece += (mask.mean()) * abs(acc_bin - conf_bin)
    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Multi-class Brier score (mean squared error on one-hot targets)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n_samples, n_classes = proba.shape
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def covariance_spectrum(Z: np.ndarray) -> np.ndarray:
    """Return eigenvalues (sorted descending) of covariance of Z (rows = samples)."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] < 2:
        return np.asarray([])
    Zc = Z - Z.mean(axis=0, keepdims=True)
    C = np.cov(Zc, rowvar=False)
    evals = np.linalg.eigvalsh(C)
    evals = np.sort(np.clip(evals, a_min=0.0, a_max=None))[::-1]
    return evals


def collapse_score(evals: np.ndarray, k: int = 1) -> float:
    """Fraction of variance in top-k eigenvalues."""
    if evals.size == 0:
        return float("nan")
    k = max(1, min(k, evals.size))
    num = evals[:k].sum()
    den = evals.sum()
    return float(num / den) if den > 0 else float("nan")


def mean_pairwise_cosine(Z: np.ndarray, max_samples: int = 256) -> float:
    """Average cosine similarity over a subset of pairs."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] < 2:
        return float("nan")
    n = Z.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_samples, replace=False)
        Z = Z[idx]
        n = Z.shape[0]
    # Normalize
    norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    Z_norm = Z / norms
    S = Z_norm @ Z_norm.T
    mask = ~np.eye(n, dtype=bool)
    return float(S[mask].mean())


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA similarity between two embedding sets."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape != Y.shape or X.ndim != 2:
        raise ValueError("X and Y must have shape (n_samples, d) and match")
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    XtY = Xc.T @ Yc
    hsic = np.linalg.norm(XtY, ord="fro") ** 2
    XX = Xc.T @ Xc
    YY = Yc.T @ Yc
    denom = np.linalg.norm(XX, ord="fro") * np.linalg.norm(YY, ord="fro")
    return float(hsic / denom) if denom > 0 else float("nan")


@torch.no_grad()
def _compute_rank_matrices(
    model,
    R_cond: torch.Tensor,
    train_size: int,
    avg_last2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return batched ranking matrices for all test rows (adapted from extract_row_embed_icl_topk.py).

    Returns
    -------
    weights_avg : Tensor (T, T)
        Mean-head attention of the selected layer (last or avg of last two).

    scores_axv_all : Tensor (T_test, J)
        Mean-head attention × ||W_out v|| from last layer, over (test, train).
        (Kept for completeness; not used directly here.)
    """
    from tabicl.model.learning import ICLearning  # local import to avoid cycles

    device = R_cond.device
    enc: ICLearning = model.icl_predictor
    tf_icl = enc.tf_icl
    blocks = list(tf_icl.blocks)

    x = R_cond
    v_prev = None
    for i, blk in enumerate(blocks[:-1]):
        x = blk(x, key_padding_mask=None, attn_mask=train_size, rope=tf_icl.rope, v_prev=v_prev, block_index=i)
        v_prev = getattr(blk, "_last_v", None)

    last = blocks[-1]
    if last.norm_first:
        q_in = last.norm1(x)
    else:
        q_in = x
    B, T, E = q_in.shape
    nh = last.attn.num_heads
    hs = E // nh
    q, k, v = F._in_projection_packed(q_in, q_in, q_in, last.attn.in_proj_weight, last.attn.in_proj_bias)
    q = q.view(B, T, nh, hs).transpose(-3, -2)
    k = k.view(B, T, nh, hs).transpose(-3, -2)
    v = v.view(B, T, nh, hs).transpose(-3, -2)
    if last.elliptical and (v_prev is not None) and (len(blocks) - 1 >= 1):
        keep = torch.zeros(T, device=device, dtype=torch.float32)
        keep[:train_size] = 1.0
        m = compute_elliptical_diag(
            v,
            v_prev,
            delta=last.elliptical_delta,
            scale_mode=last.elliptical_scale_mode,
            mask_keep=keep,
        )
        m_bc = m.view(1, 1, nh, 1, hs)
        q = q * m_bc
        k = k * m_bc
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hs)
    allowed = torch.zeros(T, T, dtype=torch.bool, device=device)
    if train_size > 0:
        allowed[:train_size, :train_size] = True
        allowed[train_size:, :train_size] = True
    # Broadcast mask over any leading batch/head dims
    scores = scores.masked_fill(~allowed.view(*([1] * (scores.dim() - 2)), T, T), float("-inf"))
    weights_last = torch.softmax(scores, dim=-1)  # (..., T, T)

    # Optional average with N-2
    if avg_last2 and len(blocks) >= 2:
        x2 = R_cond
        v_prev2 = None
        for i, blk in enumerate(blocks[:-2]):
            x2 = blk(x2, key_padding_mask=None, attn_mask=train_size, rope=tf_icl.rope, v_prev=v_prev2, block_index=i)
            v_prev2 = getattr(blk, "_last_v", None)
        prev_blk = blocks[-2]
        if prev_blk.norm_first:
            q_in2 = prev_blk.norm1(x2)
        else:
            q_in2 = x2
        B2, T2, E2 = q_in2.shape
        nh2 = prev_blk.attn.num_heads
        hs2 = E2 // nh2
        q2, k2, v2 = F._in_projection_packed(
            q_in2, q_in2, q_in2, prev_blk.attn.in_proj_weight, prev_blk.attn.in_proj_bias
        )
        q2 = q2.view(B2, T2, nh2, hs2).transpose(-3, -2)
        k2 = k2.view(B2, T2, nh2, hs2).transpose(-3, -2)
        if prev_blk.elliptical and (v_prev2 is not None) and (len(blocks) - 2 >= 1):
            keep2 = torch.zeros(T2, device=device, dtype=torch.float32)
            keep2[:train_size] = 1.0
            m2 = compute_elliptical_diag(
                v2,
                v_prev2,
                delta=prev_blk.elliptical_delta,
                scale_mode=prev_blk.elliptical_scale_mode,
                mask_keep=keep2,
            )
            m2_bc = m2.view(1, 1, nh2, 1, hs2)
            q2 = q2 * m2_bc
            k2 = k2 * m2_bc
        scores2 = torch.matmul(q2, k2.transpose(-1, -2)) / math.sqrt(hs2)
        scores2 = scores2.masked_fill(~allowed.view(*([1] * (scores2.dim() - 2)), T2, T2), float("-inf"))
        weights_prev = torch.softmax(scores2, dim=-1)  # (..., T, T)
        weights_sel = 0.5 * (weights_last + weights_prev)
    else:
        weights_sel = weights_last

    # Average over all leading dims (batch + heads), keep (T, T)
    if weights_sel.dim() > 2:
        lead_dims = tuple(range(weights_sel.dim() - 2))
        weights_avg = weights_sel.mean(dim=lead_dims)
    else:
        weights_avg = weights_sel

    # For this script, downstream code only uses `weights_avg` (for N_eff).
    # We return an empty placeholder for scores_axv_all to keep the signature
    # compatible with callers but avoid potential shape mismatches here.
    scores_axv_all = torch.empty(0, train_size, device=device)

    return weights_avg, scores_axv_all


def effective_num_neighbors(weights_avg: torch.Tensor, train_size: int) -> Tuple[float, float]:
    """Return mean and std of N_eff over test rows given attention weights.

    This implementation is robust to non-finite attention weights:
    - restricts to the (test_rows, train_rows) slice
    - zeroes out non-finite entries and renormalizes rows to sum to 1
    - computes N_eff = 1 / sum_i alpha_i^2 per test row
    """
    if train_size <= 0 or weights_avg.ndim != 2:
        return float("nan"), float("nan")

    T = weights_avg.shape[0]
    if T <= train_size:
        return float("nan"), float("nan")

    # Slice to test->train block: (T_test, J)
    W = weights_avg[train_size:, :train_size].clone()
    if W.numel() == 0:
        return float("nan"), float("nan")

    # Zero-out non-finite values and renormalize each row to sum 1
    finite = torch.isfinite(W)
    any_finite = finite.any(dim=1)
    if not any_finite.any():
        return float("nan"), float("nan")

    W = W[any_finite]
    finite = finite[any_finite]

    W_clean = torch.where(finite, W, torch.zeros_like(W))
    row_sums = W_clean.sum(dim=1, keepdim=True).clamp_min(1e-12)
    alpha = W_clean / row_sums  # (T_eff, J), rows now sum to 1

    s2 = (alpha**2).sum(dim=1)  # (T_eff,)
    # Guard against any numerical issues
    mask = (s2 > 0) & torch.isfinite(s2)
    if not mask.any():
        return float("nan"), float("nan")

    neff = 1.0 / s2[mask]
    neff_np = neff.detach().cpu().numpy().astype(np.float64)
    return float(neff_np.mean()), float(neff_np.std())


def neighbor_label_purity(
    weights_avg: torch.Tensor,
    y_train: pd.Series,
    y_test: pd.Series,
    train_size: int,
    topk: int = 5,
) -> Tuple[float, float]:
    """Compute neighbour label purity for test rows.

    Returns
    -------
    top1_hit : float
        Fraction of test rows whose top-1 attended neighbour has the same label.

    topk_purity : float
        Average fraction of top-k neighbours sharing the test row's label.
    """
    if train_size <= 0 or weights_avg.ndim != 2:
        return float("nan"), float("nan")

    T = weights_avg.shape[0]
    n_test = T - train_size
    if n_test <= 0:
        return float("nan"), float("nan")

    W = weights_avg.detach().cpu().numpy()
    y_tr = np.asarray(y_train)
    y_te = np.asarray(y_test)

    if y_tr.shape[0] != train_size or y_te.shape[0] != n_test:
        return float("nan"), float("nan")

    top1_hits: List[float] = []
    purities: List[float] = []
    k = max(1, min(topk, train_size))

    for t_idx in range(n_test):
        row_idx = train_size + t_idx
        alpha = W[row_idx, :train_size]
        if not np.isfinite(alpha).any():
            continue
        # Top-k neighbour indices by attention
        top_idx = np.argsort(-alpha)[:k]
        neigh_labels = y_tr[top_idx]
        qlab = y_te[t_idx]

        top1_hits.append(float(neigh_labels[0] == qlab))
        purities.append(float((neigh_labels == qlab).mean()))

    if not purities:
        return float("nan"), float("nan")

    return float(np.mean(top1_hits)), float(np.mean(purities))


# ---------------------------------------------------------------------------
# Behavioural evaluation
# ---------------------------------------------------------------------------


@dataclass
class BehaviourMetrics:
    accuracy: float
    f1_macro: float
    log_loss: float
    brier: float
    ece: float


def _fit_and_eval(
    model_path: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    device: str,
    n_estimators: int,
    random_state: int,
    use_hierarchical: bool,
) -> BehaviourMetrics:
    clf = TabICLClassifier(
        device=device,
        model_path=str(model_path),
        allow_auto_download=False,
        use_hierarchical=use_hierarchical,
        n_estimators=n_estimators,
        random_state=random_state,
        verbose=False,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    y_pred = clf.y_encoder_.inverse_transform(np.argmax(proba, axis=1))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    ll = log_loss(y_test, proba, labels=clf.classes_)
    # Convert true labels to encoded ints for Brier/ECE
    y_true_int = clf.y_encoder_.transform(y_test)
    brier = multiclass_brier_score(y_true_int, proba)
    ece = compute_ece(y_true_int, proba)
    return BehaviourMetrics(
        accuracy=float(acc),
        f1_macro=float(f1),
        log_loss=float(ll),
        brier=float(brier),
        ece=float(ece),
    )


def _eval_metrics_from_fitted_clf(
    clf: TabICLClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> BehaviourMetrics:
    """Evaluate BehaviourMetrics using an already-fitted classifier."""
    proba = clf.predict_proba(X_test)
    y_pred = clf.y_encoder_.inverse_transform(np.argmax(proba, axis=1))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    ll = log_loss(y_test, proba, labels=clf.classes_)
    y_true_int = clf.y_encoder_.transform(y_test)
    brier = multiclass_brier_score(y_true_int, proba)
    ece = compute_ece(y_true_int, proba)
    return BehaviourMetrics(
        accuracy=float(acc),
        f1_macro=float(f1),
        log_loss=float(ll),
        brier=float(brier),
        ece=float(ece),
    )


def run_behavioural_panel(
    ds_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    model_sa: Path,
    model_ea: Path,
    device: str,
    context_lengths: Sequence[int],
    n_estimators: int,
    seed: int,
    use_hierarchical: bool,
) -> None:
    N = len(X)
    print(f"\n=== Behavioural: {ds_name} (N={N}, classes={y.nunique()}) ===")

    context_lengths = sorted(set(int(c) for c in context_lengths if c > 0))
    if not context_lengths:
        context_lengths = [min(256, max(2, N // 2))]

    rng = np.random.RandomState(seed)

    rows = []
    for L in context_lengths:
        if L >= N:
            print(f"  [skip] context_length={L} >= N={N}")
            continue

        # Stratified split with fixed train_size = L (or N-1 if needed)
        train_size = min(L, N - 1)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            train_size=train_size,
            random_state=int(rng.randint(0, 1_000_000)),
            stratify=y,
        )

        print(f"  Context length (train_size) = {train_size}  | test_size = {len(X_te)}")

        m_sa = _fit_and_eval(model_sa, X_tr, y_tr, X_te, y_te, device, n_estimators, seed, use_hierarchical)
        m_ea = _fit_and_eval(model_ea, X_tr, y_tr, X_te, y_te, device, n_estimators, seed, use_hierarchical)

        def _fmt(m: BehaviourMetrics) -> str:
            return (
                f"acc={m.accuracy:.4f}, f1={m.f1_macro:.4f}, "
                f"NLL={m.log_loss:.4f}, Brier={m.brier:.4f}, ECE={m.ece:.4f}"
            )

        print(f"    SA: {_fmt(m_sa)}")
        print(f"    EA: {_fmt(m_ea)}")

        rows.append(
            dict(
                dataset=ds_name,
                context_length=train_size,
                acc_sa=m_sa.accuracy,
                acc_ea=m_ea.accuracy,
                f1_sa=m_sa.f1_macro,
                f1_ea=m_ea.f1_macro,
                nll_sa=m_sa.log_loss,
                nll_ea=m_ea.log_loss,
                brier_sa=m_sa.brier,
                brier_ea=m_ea.brier,
                ece_sa=m_sa.ece,
                ece_ea=m_ea.ece,
            )
        )

    if not rows:
        print("  No valid context lengths for this dataset.")
        return

    df = pd.DataFrame(rows)
    print("\n  Summary (per context length, SA vs EA):")
    for L, sub in df.groupby("context_length"):
        row = sub.iloc[0]
        print(
            f"    L={L}: "
            f"acc(SA)={row['acc_sa']:.4f}, acc(EA)={row['acc_ea']:.4f}; "
            f"f1(SA)={row['f1_sa']:.4f}, f1(EA)={row['f1_ea']:.4f}; "
            f"NLL(SA)={row['nll_sa']:.4f}, NLL(EA)={row['nll_ea']:.4f}; "
            f"ECE(SA)={row['ece_sa']:.4f}, ECE(EA)={row['ece_ea']:.4f}"
        )


# ---------------------------------------------------------------------------
# Representation geometry evaluation
# ---------------------------------------------------------------------------


@dataclass
class GeometryMetrics:
    collapse_top1: float
    collapse_top5: float
    mean_cosine: float


def _choose_identity_variant(patterns: List[List[int]]) -> int:
    for i, p in enumerate(patterns):
        if list(p) == sorted(p):
            return i
    return 0


def run_geometry_panel(
    ds_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    model_sa: Path,
    model_ea: Path,
    device: str,
    n_estimators: int,
    seed: int,
    use_hierarchical: bool,
) -> None:
    print(f"\n=== Geometry: {ds_name} ===")

    # Single split for geometry: use moderate test size
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y,
    )

    def _fit_clf(model_path: Path) -> TabICLClassifier:
        clf = TabICLClassifier(
            device=device,
            model_path=str(model_path),
            allow_auto_download=False,
            use_hierarchical=use_hierarchical,
            n_estimators=n_estimators,
            random_state=seed,
            verbose=False,
        )
        clf.fit(X_tr, y_tr)
        return clf

    clf_sa = _fit_clf(model_sa)
    clf_ea = _fit_clf(model_ea)

    # TFrow embeddings via helper (pre-ICL)
    res_sa = extract_tf_row_embeddings(clf_sa, X_te, choose_random_variant=False)
    res_ea = extract_tf_row_embeddings(clf_ea, X_te, choose_random_variant=False)

    emb_test_sa = res_sa["embeddings_test"]
    emb_test_ea = res_ea["embeddings_test"]

    # Collapse / cosine per model
    evals_sa = covariance_spectrum(emb_test_sa)
    evals_ea = covariance_spectrum(emb_test_ea)
    geom_sa = GeometryMetrics(
        collapse_top1=collapse_score(evals_sa, k=1),
        collapse_top5=collapse_score(evals_sa, k=5),
        mean_cosine=mean_pairwise_cosine(emb_test_sa),
    )
    geom_ea = GeometryMetrics(
        collapse_top1=collapse_score(evals_ea, k=1),
        collapse_top5=collapse_score(evals_ea, k=5),
        mean_cosine=mean_pairwise_cosine(emb_test_ea),
    )

    print(
        f"  TFrow collapse (top-1 var): SA={geom_sa.collapse_top1:.4f}, EA={geom_ea.collapse_top1:.4f}"
    )
    print(
        f"  TFrow collapse (top-5 var): SA={geom_sa.collapse_top5:.4f}, EA={geom_ea.collapse_top5:.4f}"
    )
    print(
        f"  TFrow mean pairwise cos:    SA={geom_sa.mean_cosine:.4f}, EA={geom_ea.mean_cosine:.4f}"
    )

    # CKA similarity between SA and EA embeddings (same test rows)
    if emb_test_sa.shape == emb_test_ea.shape:
        cka = linear_cka(emb_test_sa, emb_test_ea)
        print(f"  TFrow linear CKA(SA,EA) on test embeddings: {cka:.4f}")
    else:
        print("  [warn] Embedding shapes differ between SA and EA; skipping CKA.")

    # Effective number of attended neighbours (ICL attention)
    # Build one in-context episode using ensemble generator (same variant for both models)
    def _build_episode(clf: TabICLClassifier) -> Tuple[torch.Tensor, int]:
        X_te_num = clf.X_encoder_.transform(X_te)
        data = clf.ensemble_generator_.transform(X_te_num)
        # Use first norm method and identity/first variant
        methods = list(data.keys())
        norm_method = methods[0]
        Xs, ys_shifted = data[norm_method]
        shuffle_patterns = clf.ensemble_generator_.feature_shuffle_patterns_[norm_method]
        shift_offsets = clf.ensemble_generator_.class_shift_offsets_[norm_method]

        vidx = _choose_identity_variant([list(p) for p in shuffle_patterns])
        X_variant = Xs[vidx]
        y_train_shifted = ys_shifted[vidx]
        train_size = y_train_shifted.shape[0]
        n_classes = clf.n_classes_

        device_t = clf.device_
        model = clf.model_.to(device_t)
        model.eval()

        X_tensor = torch.from_numpy(X_variant).float().unsqueeze(0).to(device_t)
        with torch.no_grad():
            col_out = model.col_embedder(
                X_tensor,
                train_size=train_size,
                mgr_config=clf.inference_config_.COL_CONFIG,
            )
            row_reps = model.row_interactor(col_out, mgr_config=clf.inference_config_.ROW_CONFIG)
            yt = torch.as_tensor(y_train_shifted, device=device_t, dtype=torch.float32).unsqueeze(0)
            R_cond = row_reps.clone()
            # Label conditioning (class-shifted)
            R_cond[:, :train_size] = R_cond[:, :train_size] + model.icl_predictor.y_encoder(yt)

        return R_cond, train_size

    R_sa, train_size = _build_episode(clf_sa)
    R_ea, _ = _build_episode(clf_ea)

    w_sa, _ = _compute_rank_matrices(clf_sa.model_, R_sa, train_size, avg_last2=False)
    w_ea, _ = _compute_rank_matrices(clf_ea.model_, R_ea, train_size, avg_last2=False)
    neff_sa_mean, neff_sa_std = effective_num_neighbors(w_sa, train_size)
    neff_ea_mean, neff_ea_std = effective_num_neighbors(w_ea, train_size)

    print(
        f"  N_eff (ICL attention over train rows): "
        f"SA mean={neff_sa_mean:.2f}±{neff_sa_std:.2f}, "
        f"EA mean={neff_ea_mean:.2f}±{neff_ea_std:.2f}"
    )

    # Neighbour label purity (soft k-NN behaviour under attention)
    top1_sa, purity_sa = neighbor_label_purity(w_sa, y_tr, y_te, train_size, topk=5)
    top1_ea, purity_ea = neighbor_label_purity(w_ea, y_tr, y_te, train_size, topk=5)
    print(
        f"  Neighbour label purity (top-1 / top-5): "
        f"SA={top1_sa:.3f}/{purity_sa:.3f}, "
        f"EA={top1_ea:.3f}/{purity_ea:.3f}"
    )


# ---------------------------------------------------------------------------
# Robustness / inductive bias evaluation
# ---------------------------------------------------------------------------


def _add_gaussian_noise(
    X: pd.DataFrame,
    numeric_cols: List[str],
    col_stds: pd.Series,
    sigma: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    X_noisy = X.copy()
    if sigma <= 0.0 or not numeric_cols:
        return X_noisy
    for col in numeric_cols:
        std = float(col_stds.get(col, 1.0) or 1.0)
        noise = rng.normal(loc=0.0, scale=sigma * std, size=len(X_noisy))
        X_noisy[col] = X_noisy[col].astype(float) + noise
    return X_noisy


def _add_outlier_rows(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    frac: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series]:
    if frac <= 0.0:
        return X_test, y_test
    n_out = int(round(frac * len(X_test)))
    if n_out == 0:
        return X_test, y_test

    # Basic statistics from train for numeric/categorical synthesis
    num_cols = list(X_train.select_dtypes(include="number").columns)
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    num_stats = {}
    for col in num_cols:
        m = float(X_train[col].mean())
        s = float(X_train[col].std() or 1.0)
        num_stats[col] = (m, s)

    cat_values = {c: X_train[c].dropna().unique() for c in cat_cols}

    rows = []
    for _ in range(n_out):
        row = {}
        for col in num_cols:
            m, s = num_stats[col]
            row[col] = rng.normal(loc=m, scale=3.0 * s)  # heavy-tailed numeric outlier
        for col in cat_cols:
            vals = cat_values.get(col)
            if vals is not None and len(vals) > 0:
                row[col] = rng.choice(vals)
            else:
                row[col] = np.nan
        rows.append(row)
    # Build outlier DataFrame directly with the same columns as X_test
    X_out = pd.DataFrame(rows, columns=X_test.columns)

    # Random labels from train distribution
    y_out = rng.choice(y_train.values, size=n_out)

    X_aug = pd.concat([X_test.reset_index(drop=True), X_out.reset_index(drop=True)], ignore_index=True)
    y_aug = pd.concat(
        [y_test.reset_index(drop=True), pd.Series(y_out, index=range(len(y_test), len(y_test) + n_out))],
        ignore_index=True,
    )
    return X_aug, y_aug


def _rescale_and_rotate(
    X: pd.DataFrame,
    numeric_cols: List[str],
    factor: float,
    angle_rad: float,
) -> pd.DataFrame:
    X_tr = X.copy()
    if not numeric_cols:
        return X_tr
    # Rescale all numeric columns
    X_tr[numeric_cols] = X_tr[numeric_cols].astype(float) * factor
    # Rotate first 2 numeric features if available
    if len(numeric_cols) >= 2:
        c1, c2 = numeric_cols[0], numeric_cols[1]
        v = X_tr[[c1, c2]].values
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]], dtype=float)
        v_rot = v @ R.T
        X_tr[c1] = v_rot[:, 0]
        X_tr[c2] = v_rot[:, 1]
    return X_tr


def run_robustness_panel(
    ds_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    model_sa: Path,
    model_ea: Path,
    device: str,
    n_estimators: int,
    seed: int,
    noise_levels: Sequence[float],
    outlier_fracs: Sequence[float],
    rescale_factors: Sequence[float],
    use_hierarchical: bool,
) -> None:
    print(f"\n=== Robustness / inductive bias: {ds_name} ===")

    # Fixed split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y,
    )

    def _fit_clf(model_path: Path) -> TabICLClassifier:
        clf = TabICLClassifier(
            device=device,
            model_path=str(model_path),
            allow_auto_download=False,
            use_hierarchical=use_hierarchical,
            n_estimators=n_estimators,
            random_state=seed,
            verbose=False,
        )
        clf.fit(X_tr, y_tr)
        return clf

    clf_sa = _fit_clf(model_sa)
    clf_ea = _fit_clf(model_ea)

    rng = np.random.default_rng(seed)

    # Collect rows for CSV logging
    robustness_rows: List[Dict[str, Any]] = []

    # Shared numeric feature info
    num_cols = list(X_tr.select_dtypes(include="number").columns)
    col_stds = X_tr[num_cols].std().replace(0, 1.0) if num_cols else pd.Series(dtype=float)

    # Baseline on clean test data
    base_sa = _eval_metrics_from_fitted_clf(clf_sa, X_te, y_te)
    base_ea = _eval_metrics_from_fitted_clf(clf_ea, X_te, y_te)
    print(
        f"  Clean test: "
        f"acc(SA)={base_sa.accuracy:.4f}, acc(EA)={base_ea.accuracy:.4f}; "
        f"NLL(SA)={base_sa.log_loss:.4f}, NLL(EA)={base_ea.log_loss:.4f}"
    )

    robustness_rows.append(
        {
            "dataset": ds_name,
            "condition": "clean",
            "param_type": "none",
            "param_value": 0.0,
            "n_test": len(X_te),
            "acc_sa": base_sa.accuracy,
            "acc_ea": base_ea.accuracy,
            "f1_sa": base_sa.f1_macro,
            "f1_ea": base_ea.f1_macro,
            "nll_sa": base_sa.log_loss,
            "nll_ea": base_ea.log_loss,
            "brier_sa": base_sa.brier,
            "brier_ea": base_ea.brier,
            "ece_sa": base_sa.ece,
            "ece_ea": base_ea.ece,
            "seed": seed,
            "n_estimators": n_estimators,
            "use_hierarchical": bool(use_hierarchical),
            "sa_checkpoint": str(model_sa),
            "ea_checkpoint": str(model_ea),
        }
    )

    # (a) Noise / outlier robustness
    if noise_levels:
        print("  [a] Gaussian noise on numeric features:")
        for sigma in noise_levels:
            X_noisy = _add_gaussian_noise(X_te, num_cols, col_stds, sigma, rng)
            m_sa = _eval_metrics_from_fitted_clf(clf_sa, X_noisy, y_te)
            m_ea = _eval_metrics_from_fitted_clf(clf_ea, X_noisy, y_te)
            print(
                f"    sigma={sigma:.3f}: "
                f"acc(SA)={m_sa.accuracy:.4f}, acc(EA)={m_ea.accuracy:.4f}; "
                f"NLL(SA)={m_sa.log_loss:.4f}, NLL(EA)={m_ea.log_loss:.4f}"
            )
            robustness_rows.append(
                {
                    "dataset": ds_name,
                    "condition": "noise",
                    "param_type": "sigma",
                    "param_value": float(sigma),
                    "n_test": len(X_noisy),
                    "acc_sa": m_sa.accuracy,
                    "acc_ea": m_ea.accuracy,
                    "f1_sa": m_sa.f1_macro,
                    "f1_ea": m_ea.f1_macro,
                    "nll_sa": m_sa.log_loss,
                    "nll_ea": m_ea.log_loss,
                    "brier_sa": m_sa.brier,
                    "brier_ea": m_ea.brier,
                    "ece_sa": m_sa.ece,
                    "ece_ea": m_ea.ece,
                    "seed": seed,
                    "n_estimators": n_estimators,
                    "use_hierarchical": bool(use_hierarchical),
                    "sa_checkpoint": str(model_sa),
                    "ea_checkpoint": str(model_ea),
                }
            )

    if outlier_fracs:
        print("  [a] Outlier rows (random features + labels):")
        for frac in outlier_fracs:
            X_aug, y_aug = _add_outlier_rows(X_te, y_te, X_tr, y_tr, frac, rng)
            m_sa = _eval_metrics_from_fitted_clf(clf_sa, X_aug, y_aug)
            m_ea = _eval_metrics_from_fitted_clf(clf_ea, X_aug, y_aug)
            print(
                f"    outlier_frac={frac:.3f}: "
                f"acc(SA)={m_sa.accuracy:.4f}, acc(EA)={m_ea.accuracy:.4f}; "
                f"NLL(SA)={m_sa.log_loss:.4f}, NLL(EA)={m_ea.log_loss:.4f}"
            )
            robustness_rows.append(
                {
                    "dataset": ds_name,
                    "condition": "outliers",
                    "param_type": "frac",
                    "param_value": float(frac),
                    "n_test": len(X_aug),
                    "acc_sa": m_sa.accuracy,
                    "acc_ea": m_ea.accuracy,
                    "f1_sa": m_sa.f1_macro,
                    "f1_ea": m_ea.f1_macro,
                    "nll_sa": m_sa.log_loss,
                    "nll_ea": m_ea.log_loss,
                    "brier_sa": m_sa.brier,
                    "brier_ea": m_ea.brier,
                    "ece_sa": m_sa.ece,
                    "ece_ea": m_ea.ece,
                    "seed": seed,
                    "n_estimators": n_estimators,
                    "use_hierarchical": bool(use_hierarchical),
                    "sa_checkpoint": str(model_sa),
                    "ea_checkpoint": str(model_ea),
                }
            )

    # (b) Feature rescaling / correlation (test-time transforms)
    if rescale_factors and num_cols:
        print("  [b] Feature rescaling + 2D rotation on numeric features:")
        for f in rescale_factors:
            X_trf = _rescale_and_rotate(X_te, num_cols, factor=f, angle_rad=math.pi / 4.0)
            m_sa = _eval_metrics_from_fitted_clf(clf_sa, X_trf, y_te)
            m_ea = _eval_metrics_from_fitted_clf(clf_ea, X_trf, y_te)
            print(
                f"    factor={f:.2f}, angle=45deg: "
                f"acc(SA)={m_sa.accuracy:.4f}, acc(EA)={m_ea.accuracy:.4f}; "
                f"NLL(SA)={m_sa.log_loss:.4f}, NLL(EA)={m_ea.log_loss:.4f}"
            )
            robustness_rows.append(
                {
                    "dataset": ds_name,
                    "condition": "rescale",
                    "param_type": "factor",
                    "param_value": float(f),
                    "n_test": len(X_trf),
                    "acc_sa": m_sa.accuracy,
                    "acc_ea": m_ea.accuracy,
                    "f1_sa": m_sa.f1_macro,
                    "f1_ea": m_ea.f1_macro,
                    "nll_sa": m_sa.log_loss,
                    "nll_ea": m_ea.log_loss,
                    "brier_sa": m_sa.brier,
                    "brier_ea": m_ea.brier,
                    "ece_sa": m_sa.ece,
                    "ece_ea": m_ea.ece,
                    "seed": seed,
                    "n_estimators": n_estimators,
                    "use_hierarchical": bool(use_hierarchical),
                    "sa_checkpoint": str(model_sa),
                    "ea_checkpoint": str(model_ea),
                }
            )

    # Persist robustness metrics to CSV
    if robustness_rows:
        out_path = REPO_ROOT / "results" / "compare_mini_tabicl_robustness.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame(robustness_rows)
        header = not out_path.exists()
        df_out.to_csv(out_path, mode="a", index=False, header=header)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare mini-TabICL Stage-1 standard vs elliptical attention."
    )
    ap.add_argument(
        "--sa_checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints_mini_tabicl_stage2_sa" / "step-1000.ckpt"),
        help="Stage-2 mini-TabICL checkpoint with standard attention.",
    )
    ap.add_argument(
        "--ea_checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints_mini_tabicl_stage2_ea" / "step-1000.ckpt"),
        help="Stage-2 mini-TabICL checkpoint with elliptical attention.",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default="23,48,475,714",
        help="Comma-separated list of OpenML IDs or names (e.g. '23,48,cmc').",
    )
    ap.add_argument(
        "--context_lengths",
        type=int,
        nargs="*",
        default=[256, 512, 1024],
        help="Context lengths (train_size / ICL length) to evaluate.",
    )
    ap.add_argument(
        "--n_estimators",
        type=int,
        default=8,
        help="Number of ensemble estimators for TabICLClassifier.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits and ensemble configs.",
    )
    ap.add_argument(
        "--skip_behaviour",
        action="store_true",
        help="Skip behavioural metrics (accuracy, F1, NLL, calibration).",
    )
    ap.add_argument(
        "--skip_geometry",
        action="store_true",
        help="Skip representation geometry metrics (collapse, CKA, N_eff).",
    )
    ap.add_argument(
        "--skip_robustness",
        action="store_true",
        help="Skip robustness / inductive-bias tests (noise, outliers, rescaling).",
    )
    ap.add_argument(
        "--noise_levels",
        type=float,
        nargs="*",
        default=[0.0, 0.25, 0.5],
        help="Gaussian noise levels (as multiples of feature std) for robustness test (a).",
    )
    ap.add_argument(
        "--outlier_fracs",
        type=float,
        nargs="*",
        default=[0.0, 0.05, 0.1],
        help="Fractions of synthetic outlier rows to add to the test set for robustness test (a).",
    )
    ap.add_argument(
        "--rescale_factors",
        type=float,
        nargs="*",
        default=[0.5, 2.0],
        help="Feature rescaling factors for robustness test (b).",
    )
    ap.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional max rows per dataset (random subsample) for quick runs.",
    )
    ap.add_argument(
        "--use_hierarchical",
        default=True,
        action="store_true",
        help="Use hierarchical attention in TabICLClassifier.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sa_ckpt = Path(args.sa_checkpoint)
    ea_ckpt = Path(args.ea_checkpoint)
    if not sa_ckpt.is_file():
        raise SystemExit(f"SA checkpoint not found: {sa_ckpt}")
    if not ea_ckpt.is_file():
        raise SystemExit(f"EA checkpoint not found: {ea_ckpt}")

    dataset_specs = [s.strip() for s in str(args.datasets).split(",") if s.strip()]
    if not dataset_specs:
        raise SystemExit("No datasets specified; use --datasets id1,id2,...")

    for ds in dataset_specs:
        try:
            X, y, name = fetch_openml_dataset(ds)
        except Exception as e:
            print(f"[WARN] Could not load dataset '{ds}': {e}. Skipping.")
            continue

        if args.max_rows is not None and len(X) > args.max_rows:
            # Subsample rows for quick runs (stratified)
            frac = args.max_rows / float(len(X))
            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=args.max_rows,
                random_state=args.seed,
                stratify=y,
            )
            print(f"[info] Subsampled to {len(X)} rows for dataset {name}")

        if not args.skip_behaviour:
            run_behavioural_panel(
                ds_name=name,
                X=X,
                y=y,
                model_sa=sa_ckpt,
                model_ea=ea_ckpt,
                device=device,
                context_lengths=args.context_lengths,
                n_estimators=args.n_estimators,
                seed=args.seed,
                use_hierarchical=args.use_hierarchical,
            )

        if not args.skip_geometry:
            run_geometry_panel(
                ds_name=name,
                X=X,
                y=y,
                model_sa=sa_ckpt,
                model_ea=ea_ckpt,
                device=device,
                n_estimators=args.n_estimators,
                seed=args.seed,
                use_hierarchical=args.use_hierarchical,
            )

        if not args.skip_robustness:
            run_robustness_panel(
                ds_name=name,
                X=X,
                y=y,
                model_sa=sa_ckpt,
                model_ea=ea_ckpt,
                device=device,
                n_estimators=args.n_estimators,
                seed=args.seed,
                noise_levels=args.noise_levels,
                outlier_fracs=args.outlier_fracs,
                rescale_factors=args.rescale_factors,
                use_hierarchical=args.use_hierarchical,
            )


if __name__ == "__main__":
    main()
