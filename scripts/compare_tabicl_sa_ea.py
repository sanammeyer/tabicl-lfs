#!/usr/bin/env python3
"""
Compare mini-TabICL SA vs EA (behaviour + geometry + robustness).

This script compares two TabICL checkpoints:
  - M_SA: standard attention  (mini TabICL)
  - M_EA: elliptical attention (mini TabICL)

It runs both models on the same evaluation panel and reports:

1) Behavioural metrics on a fixed panel (no fine-tuning):
   - Accuracy, macro-F1
   - Log-loss (NLL)
   - Expected Calibration Error (ECE)

2) Representation geometry on TFrow / TFicl:
   - Covariance spectrum & collapse scores (variance in top-k PCs)
   - Mean pairwise cosine similarity between row embeddings
   - Linear CKA similarity between SA and EA embeddings
   - Effective number of attended neighbours N_eff from TFicl attention

Defaults:
  - Uses OpenML IDs or names passed via --datasets
  - Uses mini-TabICL Stage-2 checkpoints at:
       checkpoints_mini_tabicl_stage2_sa/step-1000.ckpt
       checkpoints_mini_tabicl_stage2_ea_icl_only/step-1000.ckpt

Usage examples
--------------
  # Quick behavioural + geometry comparison on a few OpenML datasets
  python scripts/compare_mini_tabicl_sa_ea.py \\
      --datasets 3,11,15,23,29,31,37,44,50,54

  # Use custom checkpoints and only behavioural metrics
  python scripts/compare_mini_tabicl_sa_ea.py \\
      --sa_checkpoint checkpoints_mini_tabicl_stage2_sa/step-1000.ckpt \\
      --ea_checkpoint checkpoints_mini_tabicl_stage2_ea_icl_only/step-1000.ckpt \\
      --datasets 3,11,15,23 \\
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
        torch.cuda.manual_seed_all(seed)


def _handle_oom(stage: str, ds_name: str, err: Exception) -> None:
    """Best-effort handler for GPU OOM: log and free cache."""
    print(f"[OOM] Skipping {stage} for dataset '{ds_name}' due to GPU memory error: {err}")
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _append_rows_csv(
    rows: List[Dict[str, Any]],
    out_path: Path,
    columns: Sequence[str],
) -> None:
    """Append rows to a CSV, keeping schema aligned with current implementation.

    If an existing file has a different header, it is moved to <name>.bak and a new
    file is started with the current columns.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = True
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                first = f.readline().strip()
            existing_cols = first.split(",") if first else []
            if existing_cols == list(columns):
                header = False
            else:
                backup = out_path.with_suffix(out_path.suffix + ".bak")
                out_path.rename(backup)
                print(
                    f"[info] Existing CSV schema for '{out_path.name}' differs from current metrics; "
                    f"moved old file to '{backup.name}' and started a new one."
                )
        except Exception:
            # On any error, fall back to writing a fresh file with header.
            header = True

    df = pd.DataFrame(rows, columns=list(columns))
    df.to_csv(out_path, mode="a", index=False, header=header)


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
@torch.no_grad()
def _compute_test_to_train_weights(
    model,
    R_cond: torch.Tensor,
    train_size: int,
    ea_strength: float = 1.0,
    chunk_test: int = 4096,
    use_fp16: bool = True,
) -> Tuple[torch.Tensor, float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute mean-head test->train attention weights with low memory footprint.

    Returns a (T_test, T_train) matrix (averaged over batch and heads) with softmax
    taken over keys (train rows). Avoids constructing full (T,T) attention.
    """
    from tabicl.model.learning import ICLearning  # local import to avoid cycles

    device = R_cond.device
    enc: ICLearning = model.icl_predictor
    tf_icl = enc.tf_icl
    blocks = list(tf_icl.blocks)

    x = R_cond
    v_prev = None
    for i, blk in enumerate(blocks[:-1]):
        x = blk(x, key_padding_mask=None, attn_mask=train_size, rope=None, v_prev=v_prev, block_index=i)
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
    q = q.view(B, T, nh, hs).transpose(-3, -2)  # (B, nh, T, hs)
    k = k.view(B, T, nh, hs).transpose(-3, -2)
    v = v.view(B, T, nh, hs).transpose(-3, -2)
    m_mean = float("nan")
    m_cv = float("nan")
    # Reconstruct EA for the last ICL block in a way that mirrors the
    # current implementation in tabicl.model.attention:
    # - m is computed per-batch, per-head, per-dim from (v, v_prev)
    # - EA scales q and k by sqrt(m), so the effective metric is diag(m).
    if last.elliptical and (v_prev is not None) and (len(blocks) - 1 >= 1):
        keep = torch.zeros(T, device=device, dtype=torch.float32)
        keep[:train_size] = 1.0
        m = compute_elliptical_diag(
            v,
            v_prev,
            delta=float(last.elliptical_delta),
            scale_mode=last.elliptical_scale_mode,
            mask_keep=keep,
        )  # shape: (B, nh, hs)

        # Optional strength knob: interpolate between identity (1) and full metric (m)
        if ea_strength != 1.0:
            m = 1.0 + ea_strength * (m - 1.0)

        m_flat = m.to(dtype=torch.float32).reshape(-1)
        m_mean = float(m_flat.mean().item())
        m_std = float(m_flat.std(unbiased=False).item())
        denom = m_mean if m_mean != 0.0 else 1e-12
        m_cv = float(m_std / denom)
        # Apply sqrt(m) per head/dim, broadcast over time
        sqrt_m = torch.sqrt(m.clamp_min(1e-12)).to(dtype=q.dtype, device=q.device)  # (B, nh, hs)
        m_bc = sqrt_m.unsqueeze(-2)  # (B, nh, 1, hs)
        q = q * m_bc
        k = k * m_bc

    # Restrict to test queries and train keys
    q_test = q[..., train_size:, :]   # (B, nh, T_test, hs)
    k_train = k[..., :train_size, :]  # (B, nh, T_train, hs)
    T_test = q_test.shape[-2]
    T_train = k_train.shape[-2]

    # Merge batch and heads -> (B*nh, T, hs) for efficient bmm
    Bnh = B * nh
    q_test_b = q_test.reshape(Bnh, T_test, hs)
    k_train_b = k_train.reshape(Bnh, T_train, hs)
    if use_fp16:
        q_test_b = q_test_b.to(torch.float16)
        k_train_b = k_train_b.to(torch.float16)

    out = torch.zeros(T_test, T_train, device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(hs)

    # Per-head N_eff accumulators (mean/std over test rows and batch)
    neff_head_sum = torch.zeros(nh, dtype=torch.float64)
    neff_head_sq_sum = torch.zeros(nh, dtype=torch.float64)
    neff_head_count = torch.zeros(nh, dtype=torch.int64)

    for start in range(0, T_test, chunk_test):
        end = min(T_test, start + chunk_test)
        q_chunk = q_test_b[:, start:end, :]  # (Bnh, t, hs)
        scores = torch.bmm(q_chunk, k_train_b.transpose(1, 2)) * scale  # (Bnh, t, T_train)
        w = torch.softmax(scores.to(torch.float32), dim=-1)  # (Bnh, t, T_train)

        # Average over batch*heads for the global weights matrix
        w_mean = w.mean(dim=0)  # (t, T_train)
        out[start:end, :] += w_mean

        # Per-head N_eff: work in float32 on CPU for stability
        s2 = (w ** 2).sum(dim=-1)  # (Bnh, t)
        s2 = s2.view(B, nh, -1)    # (B, nh, t)
        valid = (s2 > 0) & torch.isfinite(s2)
        s2 = s2.clamp_min(1e-12)
        neff_chunk = torch.where(valid, 1.0 / s2, torch.zeros_like(s2))  # (B, nh, t)

        neff_chunk_cpu = neff_chunk.double().cpu()
        valid_cpu = valid.cpu()
        for h in range(nh):
            mask_h = valid_cpu[:, h, :]
            if mask_h.any():
                vals = neff_chunk_cpu[:, h, :][mask_h]
                neff_head_sum[h] += vals.sum()
                neff_head_sq_sum[h] += (vals * vals).sum()
                neff_head_count[h] += vals.numel()

    # Convert per-head N_eff stats to numpy arrays (mean / std per head)
    neff_head_mean: Optional[np.ndarray]
    neff_head_std: Optional[np.ndarray]
    if neff_head_count.sum().item() == 0:
        neff_head_mean = None
        neff_head_std = None
    else:
        neff_head_mean = np.full(nh, np.nan, dtype=np.float64)
        neff_head_std = np.full(nh, np.nan, dtype=np.float64)
        for h in range(nh):
            cnt = int(neff_head_count[h].item())
            if cnt == 0:
                continue
            s = float(neff_head_sum[h].item())
            ss = float(neff_head_sq_sum[h].item())
            mean = s / cnt
            var = max(0.0, ss / cnt - mean * mean)
            neff_head_mean[h] = mean
            neff_head_std[h] = math.sqrt(var)

    return out, m_mean, m_cv, neff_head_mean, neff_head_std  # (T_test, T_train)


def effective_num_neighbors(weights_tt: torch.Tensor, train_size: int) -> Tuple[float, float]:
    """Return mean/std N_eff over test rows given (T_test, T_train) weights."""
    if train_size <= 0 or weights_tt.ndim != 2:
        return float("nan"), float("nan")
    if weights_tt.numel() == 0:
        return float("nan"), float("nan")

    W = weights_tt
    finite = torch.isfinite(W)
    any_finite = finite.any(dim=1)
    if not any_finite.any():
        return float("nan"), float("nan")

    W = W[any_finite]
    finite = finite[any_finite]

    W_clean = torch.where(finite, W, torch.zeros_like(W))
    row_sums = W_clean.sum(dim=1, keepdim=True).clamp_min(1e-12)
    alpha = W_clean / row_sums  # (T_eff, T_train), rows sum to 1

    s2 = (alpha**2).sum(dim=1)
    mask = (s2 > 0) & torch.isfinite(s2)
    if not mask.any():
        return float("nan"), float("nan")

    neff = 1.0 / s2[mask]
    neff_np = neff.detach().cpu().numpy().astype(np.float64)
    return float(neff_np.mean()), float(neff_np.std())


def neighbor_label_purity(
    weights_tt: torch.Tensor,
    y_train: pd.Series,
    y_test: pd.Series,
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
    if weights_tt.ndim != 2 or weights_tt.numel() == 0:
        return float("nan"), float("nan")

    T_test, T_train = weights_tt.shape
    W = weights_tt.detach().cpu().numpy()
    y_tr = np.asarray(y_train)
    y_te = np.asarray(y_test)

    if y_tr.shape[0] != T_train or y_te.shape[0] != T_test:
        return float("nan"), float("nan")

    top1_hits: List[float] = []
    purities: List[float] = []
    k = max(1, min(topk, T_train))

    for t_idx in range(T_test):
        alpha = W[t_idx]
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
    # Convert true labels to encoded ints for ECE
    y_true_int = clf.y_encoder_.transform(y_test)
    ece = compute_ece(y_true_int, proba)
    return BehaviourMetrics(
        accuracy=float(acc),
        f1_macro=float(f1),
        log_loss=float(ll),
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
    ece = compute_ece(y_true_int, proba)
    return BehaviourMetrics(
        accuracy=float(acc),
        f1_macro=float(f1),
        log_loss=float(ll),
        ece=float(ece),
    )


def run_behavioural_panel(
    ds_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    model_sa: Path,
    model_ea: Path,
    device: str,
    context_sizes: Optional[Sequence[int]],
    n_estimators: int,
    seed: int,
    use_hierarchical: bool,
) -> List[Dict[str, Any]]:
    N = len(X)
    print(f"\n=== Behavioural: {ds_name} (N={N}, classes={y.nunique()}) ===")

    # Determine context sizes (train_size per dataset)
    if context_sizes:
        raw_sizes = sorted({int(s) for s in context_sizes if int(s) > 0})
        train_sizes = [max(2, min(s, N - 1)) for s in raw_sizes]
    else:
        # Default: 80% of rows as context
        frac = 0.8
        train_sizes = [max(2, min(int(frac * N), N - 1))]

    rows: List[Dict[str, Any]] = []
    rng = np.random.RandomState(seed)

    for train_size in train_sizes:
        if train_size >= N:
            continue

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X,
                y,
                train_size=train_size,
                random_state=int(rng.randint(0, 1_000_000)),
                stratify=y,
            )
        except ValueError as e:
            # Can happen for very small train_size with stratify on many classes
            print(
                f"  [warn] Skipping context_length={train_size} due to stratified split error: {e}"
            )
            continue

        print(f"  Context length (train_size) = {train_size}  | test_size = {len(X_te)}")

        m_sa = _fit_and_eval(model_sa, X_tr, y_tr, X_te, y_te, device, n_estimators, seed, use_hierarchical)
        m_ea = _fit_and_eval(model_ea, X_tr, y_tr, X_te, y_te, device, n_estimators, seed, use_hierarchical)

        def _fmt(m: BehaviourMetrics) -> str:
            return (
                f"acc={m.accuracy:.4f}, f1={m.f1_macro:.4f}, "
                f"NLL={m.log_loss:.4f}, ECE={m.ece:.4f}"
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
                ece_sa=m_sa.ece,
                ece_ea=m_ea.ece,
                seed=int(seed),
            )
        )

    if not rows:
        print("  No valid context sizes for this dataset.")
        return []

    # Persist behavioural metrics
    out_path = REPO_ROOT / "results" / "compare_mini_tabicl_behaviour.csv"
    behaviour_cols = [
        "dataset",
        "context_length",
        "acc_sa",
        "acc_ea",
        "f1_sa",
        "f1_ea",
        "nll_sa",
        "nll_ea",
        "ece_sa",
        "ece_ea",
        "seed",
    ]
    _append_rows_csv(rows, out_path, behaviour_cols)
    return rows


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
) -> List[Dict[str, Any]]:
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
    def _build_episode(clf: TabICLClassifier) -> Tuple[torch.Tensor, int, str, int]:
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
            # Compute row representations for the full (train+test) episode
            col_out = model.col_embedder(
                X_tensor,
                train_size=train_size,
                mgr_config=clf.inference_config_.COL_CONFIG,
            )
            row_reps = model.row_interactor(col_out, mgr_config=clf.inference_config_.ROW_CONFIG)

            # Choose labels for conditioning in a way that mirrors the model's
            # own hierarchical vs non-hierarchical inference behavior.
            enc = model.icl_predictor
            y_train_tensor = torch.as_tensor(y_train_shifted, device=device_t, dtype=torch.long)

            # If the classifier is using hierarchical classification (more classes
            # than the ICL head supports natively), build the same root grouping
            # the model would use and condition on group indices instead of raw
            # class indices. This ensures we never pass out-of-range indices to
            # the OneHotAndLinear encoder.
            use_hier = bool(getattr(clf, "use_hierarchical", False))
            n_classes = int(getattr(clf, "n_classes_", y_train_tensor.max().item() + 1))
            max_classes = int(getattr(model, "max_classes", n_classes))

            if use_hier and n_classes > max_classes:
                # Build hierarchical tree on the training rows only
                enc._fit_hierarchical(
                    row_reps[0, :train_size, :].detach(),
                    y_train_tensor.detach(),
                )
                root = enc.root
                labels_for_encoding = root.group_indices.to(device_t)  # per-train-row group id
            else:
                # Standard (non-hierarchical) conditioning on class indices
                labels_for_encoding = y_train_tensor

            yt = labels_for_encoding.to(dtype=torch.float32, device=device_t).unsqueeze(0)
            R_cond = row_reps.clone()
            # Label conditioning
            R_cond[:, :train_size] = R_cond[:, :train_size] + enc.y_encoder(yt)

        return R_cond, train_size, norm_method, vidx

    R_sa, train_size, norm_method, vidx = _build_episode(clf_sa)
    R_ea, _, _, _ = _build_episode(clf_ea)

    w_sa_tt, m_mean_sa, m_cv_sa, neff_head_mean_sa, neff_head_std_sa = _compute_test_to_train_weights(
        clf_sa.model_, R_sa, train_size, ea_strength=1.0, chunk_test=2048
    )
    w_ea_tt, m_mean_ea, m_cv_ea, neff_head_mean_ea, neff_head_std_ea = _compute_test_to_train_weights(
        clf_ea.model_, R_ea, train_size, ea_strength=1.0, chunk_test=2048
    )
    print(
        f"  ICL last-block EA metric mean(m): "
        f"SA={m_mean_sa:.4f}, EA={m_mean_ea:.4f}"
    )
    print(
        f"  ICL last-block EA metric CV(m): "
        f"SA={m_cv_sa:.4f}, EA={m_cv_ea:.4f}"
    )
    neff_sa_mean, neff_sa_std = effective_num_neighbors(w_sa_tt, train_size)
    neff_ea_mean, neff_ea_std = effective_num_neighbors(w_ea_tt, train_size)

    print(
        f"  N_eff (ICL attention over train rows): "
        f"SA mean={neff_sa_mean:.2f}±{neff_sa_std:.2f}, "
        f"EA mean={neff_ea_mean:.2f}±{neff_ea_std:.2f}"
    )

    # Optional: per-head N_eff diagnostics
    if neff_head_mean_sa is not None and neff_head_mean_ea is not None:
        # Short, rounded per-head summaries for readability
        sa_heads_str = ", ".join(f"{v:.1f}" for v in neff_head_mean_sa)
        ea_heads_str = ", ".join(f"{v:.1f}" for v in neff_head_mean_ea)
        print(f"  N_eff per head (SA): [{sa_heads_str}]")
        print(f"  N_eff per head (EA): [{ea_heads_str}]")

    # Convert to encoded ints (works for string labels etc.)
    y_tr_int = clf_sa.y_encoder_.transform(y_tr)
    y_te_int = clf_sa.y_encoder_.transform(y_te)

    # Use the SAME norm_method + vidx you used for the episode
    offset = int(clf_sa.ensemble_generator_.class_shift_offsets_[norm_method][vidx])
    n_classes = int(clf_sa.n_classes_)

    # Apply the class shift used by the generator
    y_tr_shift = (np.asarray(y_tr_int) + offset) % n_classes
    y_te_shift = (np.asarray(y_te_int) + offset) % n_classes

    top1_sa, purity_sa = neighbor_label_purity(w_sa_tt, y_tr_shift, y_te_shift, topk=5)
    top1_ea, purity_ea = neighbor_label_purity(w_ea_tt, y_tr_shift, y_te_shift, topk=5)

    # Neighbour label purity (soft k-NN behaviour under attention)
    # top1_sa, purity_sa = neighbor_label_purity(w_sa_tt, y_tr, y_te, topk=5)
    # top1_ea, purity_ea = neighbor_label_purity(w_ea_tt, y_tr, y_te, topk=5)
    print(
        f"  Neighbour label purity (top-1 / top-5): "
        f"SA={top1_sa:.3f}/{purity_sa:.3f}, "
        f"EA={top1_ea:.3f}/{purity_ea:.3f}"
    )

    # Persist geometry metrics
    geom_row = {
        "dataset": ds_name,
        "collapse_sa_top1": geom_sa.collapse_top1,
        "collapse_ea_top1": geom_ea.collapse_top1,
        "collapse_sa_top5": geom_sa.collapse_top5,
        "collapse_ea_top5": geom_ea.collapse_top5,
        "mean_cos_sa": geom_sa.mean_cosine,
        "mean_cos_ea": geom_ea.mean_cosine,
        "cka_sa_ea": cka if emb_test_sa.shape == emb_test_ea.shape else float("nan"),
        "neff_sa_mean": neff_sa_mean,
        "neff_sa_std": neff_sa_std,
        "neff_ea_mean": neff_ea_mean,
        "neff_ea_std": neff_ea_std,
        "purity_sa_top1": top1_sa,
        "purity_sa_topk": purity_sa,
        "purity_ea_top1": top1_ea,
        "purity_ea_topk": purity_ea,
        "m_mean_sa": m_mean_sa,
        "m_mean_ea": m_mean_ea,
        "m_cv_sa": m_cv_sa,
        "m_cv_ea": m_cv_ea,
        "seed": int(seed),
    }
    out_path = REPO_ROOT / "results" / "compare_mini_tabicl_geometry.csv"
    geom_cols = [
        "dataset",
        "collapse_sa_top1",
        "collapse_ea_top1",
        "collapse_sa_top5",
        "collapse_ea_top5",
        "mean_cos_sa",
        "mean_cos_ea",
        "cka_sa_ea",
        "neff_sa_mean",
        "neff_sa_std",
        "neff_ea_mean",
        "neff_ea_std",
        "purity_sa_top1",
        "purity_sa_topk",
        "purity_ea_top1",
        "purity_ea_topk",
        "m_mean_sa",
        "m_mean_ea",
        "m_cv_sa",
        "m_cv_ea",
        "seed",
    ]
    _append_rows_csv([geom_row], out_path, geom_cols)
    return [geom_row]


# ---------------------------------------------------------------------------
# Robustness / inductive bias evaluation
# ---------------------------------------------------------------------------


def _add_irrelevant_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_noise_cols: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate pure noise columns to test feature selection capability."""
    if n_noise_cols <= 0:
        return X_train, X_test

    noise_tr = rng.normal(0.0, 1.0, size=(len(X_train), n_noise_cols))
    noise_te = rng.normal(0.0, 1.0, size=(len(X_test), n_noise_cols))

    new_cols = [f"noise_{i}" for i in range(n_noise_cols)]

    X_tr_aug = pd.concat(
        [X_train.reset_index(drop=True), pd.DataFrame(noise_tr, columns=new_cols)],
        axis=1,
    )
    X_te_aug = pd.concat(
        [X_test.reset_index(drop=True), pd.DataFrame(noise_te, columns=new_cols)],
        axis=1,
    )

    return X_tr_aug, X_te_aug


def _flip_labels(
    y: pd.Series,
    frac: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Randomly flip a fraction of labels to other classes."""
    if frac <= 0.0:
        return y

    y_poison = y.reset_index(drop=True).copy()
    n = len(y_poison)
    n_flip = int(round(frac * n))
    if n_flip == 0:
        return y

    classes = np.asarray(sorted(pd.unique(y_poison)))
    if classes.size < 2:
        return y

    idx = rng.choice(n, size=n_flip, replace=False)
    for i in idx:
        current = y_poison.iloc[i]
        choices = classes[classes != current]
        if choices.size == 0:
            continue
        y_poison.iloc[i] = rng.choice(choices)

    return y_poison


def _poison_context(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    poison_frac: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Add synthetic outlier rows to the training context only, keep test clean."""
    if poison_frac <= 0.0:
        return X_train, y_train, X_test, y_test

    n_out = int(round(poison_frac * len(X_train)))
    if n_out == 0:
        return X_train, y_train, X_test, y_test

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
    # Build outlier DataFrame directly with the same columns as X_train
    X_out = pd.DataFrame(rows, columns=X_train.columns)

    # Random labels from train distribution
    y_out = rng.choice(y_train.values, size=n_out)

    X_train_poisoned = pd.concat(
        [X_train.reset_index(drop=True), X_out.reset_index(drop=True)],
        ignore_index=True,
    )
    y_train_poisoned = pd.concat(
        [y_train.reset_index(drop=True), pd.Series(y_out, index=range(len(y_train), len(y_train) + n_out))],
        ignore_index=True,
    )

    # Test set remains clean
    return X_train_poisoned, y_train_poisoned, X_test, y_test


def _rotate_numeric_pairs(
    X: pd.DataFrame,
    numeric_cols: List[str],
    angle_rad: float,
    max_pairs: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Rotate up to `max_pairs` disjoint numeric column pairs by a fixed angle."""
    X_rot = X.copy()
    if not numeric_cols or angle_rad == 0.0 or max_pairs <= 0:
        return X_rot
    n_cols = len(numeric_cols)
    if n_cols < 2:
        return X_rot

    idx = np.arange(n_cols)
    rng.shuffle(idx)
    n_pairs = min(max_pairs, n_cols // 2)
    if n_pairs <= 0:
        return X_rot

    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]], dtype=float)

    for i in range(n_pairs):
        i1, i2 = idx[2 * i], idx[2 * i + 1]
        c1, c2 = numeric_cols[i1], numeric_cols[i2]
        v = X_rot[[c1, c2]].to_numpy(dtype=float)
        v_rot = v @ R.T
        X_rot.loc[:, c1] = v_rot[:, 0]
        X_rot.loc[:, c2] = v_rot[:, 1]

    return X_rot


def _sample_random_orthogonal_matrix(k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a random k x k orthogonal matrix via QR decomposition."""
    if k <= 0:
        raise ValueError("k must be positive for orthogonal matrix sampling.")
    A = rng.normal(size=(k, k))
    Q, _ = np.linalg.qr(A)
    # Ensure a proper rotation (determinant +1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _apply_kdim_rotation_matrix(
    X: pd.DataFrame,
    cols: List[str],
    R: np.ndarray,
) -> pd.DataFrame:
    """Apply a fixed orthogonal matrix R to the numeric subspace given by `cols`."""
    if not cols:
        return X
    X_rot = X.copy()
    Z = X_rot[cols].to_numpy(dtype=float)
    if R.shape != (Z.shape[1], Z.shape[1]):
        raise ValueError("Rotation matrix shape does not match number of columns.")
    Z_rot = Z @ R.T
    X_rot.loc[:, cols] = Z_rot
    return X_rot


def run_robustness_panel(
    ds_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    model_sa: Path,
    model_ea: Path,
    device: str,
    n_estimators: int,
    seed: int,
    outlier_fracs: Sequence[float],
    n_noise_features: int,
    label_poison_fracs: Sequence[float],
    use_hierarchical: bool,
) -> List[Dict[str, Any]]:
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
            "ece_sa": base_sa.ece,
            "ece_ea": base_ea.ece,
            "seed": seed,
            "n_estimators": n_estimators,
            "use_hierarchical": bool(use_hierarchical),
            "sa_checkpoint": str(model_sa),
            "ea_checkpoint": str(model_ea),
        }
    )

    # Feature-rotation robustness: stronger tests for diagonal metric assumptions
    num_cols = list(X_tr.select_dtypes(include="number").columns)
    if num_cols:
        # Standardize numeric columns using train statistics before rotation.
        # Note: rotation metrics should be interpreted as within-condition EA-vs-SA
        # comparisons on this standardized space, not as absolute difficulty shifts.
        mu = X_tr[num_cols].mean()
        sigma = X_tr[num_cols].std().replace(0.0, 1.0)
        X_tr_std = X_tr.copy()
        X_te_std = X_te.copy()
        X_tr_std[num_cols] = (X_tr_std[num_cols] - mu) / sigma
        X_te_std[num_cols] = (X_te_std[num_cols] - mu) / sigma

        # (1) Rotate multiple disjoint numeric pairs by a fixed 45° angle (train+test)
        angle_rad = math.pi / 4.0  # 45 degrees
        max_pairs = 5
        print(
            f"  Feature rotation on numeric features (45deg, up to {max_pairs} disjoint pairs, train+test)."
        )
        # Use a fixed seed so that train and test receive the same pair rotations
        pairs_seed = int(rng.integers(0, 1_000_000))
        rng_pairs_tr = np.random.default_rng(pairs_seed)
        rng_pairs_te = np.random.default_rng(pairs_seed)
        X_tr_pairs = _rotate_numeric_pairs(X_tr_std, num_cols, angle_rad, max_pairs, rng_pairs_tr)
        X_te_pairs = _rotate_numeric_pairs(X_te_std, num_cols, angle_rad, max_pairs, rng_pairs_te)

        clf_sa_pairs = TabICLClassifier(
            device=device,
            model_path=str(model_sa),
            allow_auto_download=False,
            use_hierarchical=use_hierarchical,
            n_estimators=n_estimators,
            random_state=seed,
            verbose=False,
        )
        clf_sa_pairs.fit(X_tr_pairs, y_tr)

        clf_ea_pairs = TabICLClassifier(
            device=device,
            model_path=str(model_ea),
            allow_auto_download=False,
            use_hierarchical=use_hierarchical,
            n_estimators=n_estimators,
            random_state=seed,
            verbose=False,
        )
        clf_ea_pairs.fit(X_tr_pairs, y_tr)

        m_sa_pairs = _eval_metrics_from_fitted_clf(clf_sa_pairs, X_te_pairs, y_te)
        m_ea_pairs = _eval_metrics_from_fitted_clf(clf_ea_pairs, X_te_pairs, y_te)
        print(
            f"  Feature rotation (pairs, 45deg, train+test): "
            f"acc(SA)={m_sa_pairs.accuracy:.4f}, acc(EA)={m_ea_pairs.accuracy:.4f}; "
            f"NLL(SA)={m_sa_pairs.log_loss:.4f}, NLL(EA)={m_ea_pairs.log_loss:.4f}"
        )
        robustness_rows.append(
            {
                "dataset": ds_name,
                "condition": "feature_rotation_pairs_both",
                "param_type": "angle_deg",
                "param_value": 45.0,
                "rotation_seed": float(pairs_seed),
                "rotation_cols_hash": float("nan"),
                "n_test": len(X_te_pairs),
                "acc_sa": m_sa_pairs.accuracy,
                "acc_ea": m_ea_pairs.accuracy,
                "f1_sa": m_sa_pairs.f1_macro,
                "f1_ea": m_ea_pairs.f1_macro,
                "nll_sa": m_sa_pairs.log_loss,
                "nll_ea": m_ea_pairs.log_loss,
                "ece_sa": m_sa_pairs.ece,
                "ece_ea": m_ea_pairs.ece,
                "seed": seed,
                "n_estimators": n_estimators,
                "use_hierarchical": bool(use_hierarchical),
                "sa_checkpoint": str(model_sa),
                "ea_checkpoint": str(model_ea),
            }
        )

        # (2) k-dim random orthogonal rotations in numeric subspace (strongest stress test)
        k_subspace = min(8, len(num_cols))
        if k_subspace >= 2:
            n_k_rot = 3
            print(
                f"  k-dim random orthogonal rotations (k={k_subspace}, {n_k_rot} draws; "
                f"train+test and context-only variants)."
            )
            for ridx in range(n_k_rot):
                rot_seed = int(rng.integers(0, 1_000_000))
                rng_rot = np.random.default_rng(rot_seed)
                col_idx = rng_rot.choice(len(num_cols), size=k_subspace, replace=False)
                cols_k = [num_cols[i] for i in col_idx]
                R = _sample_random_orthogonal_matrix(k_subspace, rng_rot)
                # Stable hash of selected columns for reproducibility
                cols_k_str = "|".join(cols_k)
                cols_k_hash = float(abs(hash(cols_k_str)) % 1_000_000)

                # Train on rotated context; evaluate on rotated test (consistent reparametrization)
                X_tr_k = _apply_kdim_rotation_matrix(X_tr_std, cols_k, R)
                X_te_k = _apply_kdim_rotation_matrix(X_te_std, cols_k, R)

                clf_sa_k = TabICLClassifier(
                    device=device,
                    model_path=str(model_sa),
                    allow_auto_download=False,
                    use_hierarchical=use_hierarchical,
                    n_estimators=n_estimators,
                    random_state=seed,
                    verbose=False,
                )
                clf_sa_k.fit(X_tr_k, y_tr)

                clf_ea_k = TabICLClassifier(
                    device=device,
                    model_path=str(model_ea),
                    allow_auto_download=False,
                    use_hierarchical=use_hierarchical,
                    n_estimators=n_estimators,
                    random_state=seed,
                    verbose=False,
                )
                clf_ea_k.fit(X_tr_k, y_tr)

                m_sa_k_both = _eval_metrics_from_fitted_clf(clf_sa_k, X_te_k, y_te)
                m_ea_k_both = _eval_metrics_from_fitted_clf(clf_ea_k, X_te_k, y_te)
                print(
                    f"  k-dim rotation (train+test, k={k_subspace}, rot_seed={rot_seed}): "
                    f"acc(SA)={m_sa_k_both.accuracy:.4f}, acc(EA)={m_ea_k_both.accuracy:.4f}; "
                    f"NLL(SA)={m_sa_k_both.log_loss:.4f}, NLL(EA)={m_ea_k_both.log_loss:.4f}"
                )
                robustness_rows.append(
                    {
                        "dataset": ds_name,
                        "condition": "feature_rotation_kdim_both",
                        "param_type": "k_subspace",
                        "param_value": float(k_subspace),
                        "rotation_seed": float(rot_seed),
                        "rotation_cols_hash": cols_k_hash,
                        "n_test": len(X_te_k),
                        "acc_sa": m_sa_k_both.accuracy,
                        "acc_ea": m_ea_k_both.accuracy,
                        "f1_sa": m_sa_k_both.f1_macro,
                        "f1_ea": m_ea_k_both.f1_macro,
                        "nll_sa": m_sa_k_both.log_loss,
                        "nll_ea": m_ea_k_both.log_loss,
                        "ece_sa": m_sa_k_both.ece,
                        "ece_ea": m_ea_k_both.ece,
                        "seed": seed,
                        "n_estimators": n_estimators,
                        "use_hierarchical": bool(use_hierarchical),
                        "sa_checkpoint": str(model_sa),
                        "ea_checkpoint": str(model_ea),
                    }
                )

                # Context-only rotation: rotated train, unrotated test (representation mismatch)
                m_sa_k_ctx = _eval_metrics_from_fitted_clf(clf_sa_k, X_te_std, y_te)
                m_ea_k_ctx = _eval_metrics_from_fitted_clf(clf_ea_k, X_te_std, y_te)
                print(
                    f"  k-dim rotation (context-only, k={k_subspace}, rot_seed={rot_seed}): "
                    f"acc(SA)={m_sa_k_ctx.accuracy:.4f}, acc(EA)={m_ea_k_ctx.accuracy:.4f}; "
                    f"NLL(SA)={m_sa_k_ctx.log_loss:.4f}, NLL(EA)={m_ea_k_ctx.log_loss:.4f}"
                )
                robustness_rows.append(
                    {
                        "dataset": ds_name,
                        "condition": "feature_rotation_kdim_context",
                        "param_type": "k_subspace",
                        "param_value": float(k_subspace),
                        "rotation_seed": float(rot_seed),
                        "rotation_cols_hash": cols_k_hash,
                        "n_test": len(X_te_std),
                        "acc_sa": m_sa_k_ctx.accuracy,
                        "acc_ea": m_ea_k_ctx.accuracy,
                        "f1_sa": m_sa_k_ctx.f1_macro,
                        "f1_ea": m_ea_k_ctx.f1_macro,
                        "nll_sa": m_sa_k_ctx.log_loss,
                        "nll_ea": m_ea_k_ctx.log_loss,
                        "ece_sa": m_sa_k_ctx.ece,
                        "ece_ea": m_ea_k_ctx.ece,
                        "seed": seed,
                        "n_estimators": n_estimators,
                        "use_hierarchical": bool(use_hierarchical),
                        "sa_checkpoint": str(model_sa),
                        "ea_checkpoint": str(model_ea),
                    }
                )

    # (a) Irrelevant feature robustness: add pure noise columns
    if n_noise_features > 0:
        print("  Irrelevant features (Gaussian noise columns):")
        X_tr_noise, X_te_noise = _add_irrelevant_features(X_tr, X_te, n_noise_features, rng)

        clf_sa_noise = TabICLClassifier(
            device=device,
            model_path=str(model_sa),
            allow_auto_download=False,
            use_hierarchical=use_hierarchical,
            n_estimators=n_estimators,
            random_state=seed,
            verbose=False,
        )
        clf_sa_noise.fit(X_tr_noise, y_tr)

        clf_ea_noise = TabICLClassifier(
            device=device,
            model_path=str(model_ea),
            allow_auto_download=False,
            use_hierarchical=use_hierarchical,
            n_estimators=n_estimators,
            random_state=seed,
            verbose=False,
        )
        clf_ea_noise.fit(X_tr_noise, y_tr)

        m_sa_noise = _eval_metrics_from_fitted_clf(clf_sa_noise, X_te_noise, y_te)
        m_ea_noise = _eval_metrics_from_fitted_clf(clf_ea_noise, X_te_noise, y_te)
        print(
            f"  Irrelevant features (n_noise={n_noise_features}): "
            f"acc(SA)={m_sa_noise.accuracy:.4f}, acc(EA)={m_ea_noise.accuracy:.4f}; "
            f"NLL(SA)={m_sa_noise.log_loss:.4f}, NLL(EA)={m_ea_noise.log_loss:.4f}"
        )
        robustness_rows.append(
            {
                "dataset": ds_name,
                "condition": "irrelevant_features",
                "param_type": "n_noise",
                "param_value": int(n_noise_features),
                "n_test": len(X_te_noise),
                "acc_sa": m_sa_noise.accuracy,
                "acc_ea": m_ea_noise.accuracy,
                "f1_sa": m_sa_noise.f1_macro,
                "f1_ea": m_ea_noise.f1_macro,
                "nll_sa": m_sa_noise.log_loss,
                "nll_ea": m_ea_noise.log_loss,
                "ece_sa": m_sa_noise.ece,
                "ece_ea": m_ea_noise.ece,
                "seed": seed,
                "n_estimators": n_estimators,
                "use_hierarchical": bool(use_hierarchical),
                "sa_checkpoint": str(model_sa),
                "ea_checkpoint": str(model_ea),
            }
        )

    # (b) Label poisoning on context (train) labels
    # Sweep over small contamination fractions for more diagnostic behaviour
    # Note: frac=0.0 is effectively redundant with the clean condition above, so we skip it.
    positive_label_fracs = [f for f in label_poison_fracs if f > 0.0]
    if positive_label_fracs:
        print("  Label poisoning on training labels:")
        for frac in positive_label_fracs:
            y_tr_poison = _flip_labels(y_tr, frac, rng)

            clf_sa_poison = TabICLClassifier(
                device=device,
                model_path=str(model_sa),
                allow_auto_download=False,
                use_hierarchical=use_hierarchical,
                n_estimators=n_estimators,
                random_state=seed,
                verbose=False,
            )
            clf_sa_poison.fit(X_tr, y_tr_poison)

            clf_ea_poison = TabICLClassifier(
                device=device,
                model_path=str(model_ea),
                allow_auto_download=False,
                use_hierarchical=use_hierarchical,
                n_estimators=n_estimators,
                random_state=seed,
                verbose=False,
            )
            clf_ea_poison.fit(X_tr, y_tr_poison)

            m_sa_poison = _eval_metrics_from_fitted_clf(clf_sa_poison, X_te, y_te)
            m_ea_poison = _eval_metrics_from_fitted_clf(clf_ea_poison, X_te, y_te)
            print(
                f"  Label poisoning (frac={frac:.3f}): "
                f"acc(SA)={m_sa_poison.accuracy:.4f}, acc(EA)={m_ea_poison.accuracy:.4f}; "
                f"NLL(SA)={m_sa_poison.log_loss:.4f}, NLL(EA)={m_ea_poison.log_loss:.4f}"
            )
            robustness_rows.append(
                {
                    "dataset": ds_name,
                    "condition": "label_poison",
                    "param_type": "frac",
                    "param_value": float(frac),
                    "n_test": len(X_te),
                    "acc_sa": m_sa_poison.accuracy,
                    "acc_ea": m_ea_poison.accuracy,
                    "f1_sa": m_sa_poison.f1_macro,
                    "f1_ea": m_ea_poison.f1_macro,
                    "nll_sa": m_sa_poison.log_loss,
                    "nll_ea": m_ea_poison.log_loss,
                    "ece_sa": m_sa_poison.ece,
                    "ece_ea": m_ea_poison.ece,
                    "seed": seed,
                    "n_estimators": n_estimators,
                    "use_hierarchical": bool(use_hierarchical),
                    "sa_checkpoint": str(model_sa),
                    "ea_checkpoint": str(model_ea),
                }
            )

    # (c) Context poisoning: inject synthetic outliers into training context only
    positive_fracs = [f for f in outlier_fracs if f > 0.0]
    if positive_fracs:
        print("  Context poisoning with synthetic outlier rows (train only):")
        for frac in positive_fracs:
            X_tr_poison, y_tr_poison, X_te_clean, y_te_clean = _poison_context(
                X_tr, y_tr, X_te, y_te, frac, rng
            )

            clf_sa_ctx = TabICLClassifier(
                device=device,
                model_path=str(model_sa),
                allow_auto_download=False,
                use_hierarchical=use_hierarchical,
                n_estimators=n_estimators,
                random_state=seed,
                verbose=False,
            )
            clf_sa_ctx.fit(X_tr_poison, y_tr_poison)

            clf_ea_ctx = TabICLClassifier(
                device=device,
                model_path=str(model_ea),
                allow_auto_download=False,
                use_hierarchical=use_hierarchical,
                n_estimators=n_estimators,
                random_state=seed,
                verbose=False,
            )
            clf_ea_ctx.fit(X_tr_poison, y_tr_poison)

            m_sa_ctx = _eval_metrics_from_fitted_clf(clf_sa_ctx, X_te_clean, y_te_clean)
            m_ea_ctx = _eval_metrics_from_fitted_clf(clf_ea_ctx, X_te_clean, y_te_clean)
            print(
                f"    poison_frac={frac:.3f}: "
                f"acc(SA)={m_sa_ctx.accuracy:.4f}, acc(EA)={m_ea_ctx.accuracy:.4f}; "
                f"NLL(SA)={m_sa_ctx.log_loss:.4f}, NLL(EA)={m_ea_ctx.log_loss:.4f}"
            )
            robustness_rows.append(
                {
                    "dataset": ds_name,
                    "condition": "context_poison",
                    "param_type": "frac",
                    "param_value": float(frac),
                    "n_test": len(X_te_clean),
                    "acc_sa": m_sa_ctx.accuracy,
                    "acc_ea": m_ea_ctx.accuracy,
                    "f1_sa": m_sa_ctx.f1_macro,
                    "f1_ea": m_ea_ctx.f1_macro,
                    "nll_sa": m_sa_ctx.log_loss,
                    "nll_ea": m_ea_ctx.log_loss,
                    "ece_sa": m_sa_ctx.ece,
                    "ece_ea": m_ea_ctx.ece,
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
        robust_cols = [
            "dataset",
            "condition",
            "param_type",
            "param_value",
            "rotation_seed",
            "rotation_cols_hash",
            "n_test",
            "acc_sa",
            "acc_ea",
            "f1_sa",
            "f1_ea",
            "nll_sa",
            "nll_ea",
            "ece_sa",
            "ece_ea",
            "seed",
            "n_estimators",
            "use_hierarchical",
            "sa_checkpoint",
            "ea_checkpoint",
        ]
        _append_rows_csv(robustness_rows, out_path, robust_cols)
    return robustness_rows


def _print_aggregated_summary(
    all_beh_rows: List[Dict[str, Any]],
    all_geom_rows: List[Dict[str, Any]],
    all_rob_rows: List[Dict[str, Any]],
) -> None:
    """Print an aggregated summary across all datasets / seeds."""
    print("\n===== Aggregated summary across datasets/seeds =====")

    # Behavioural summary
    if all_beh_rows:
        beh_df = pd.DataFrame(all_beh_rows)
        beh_df["d_acc"] = beh_df["acc_ea"] - beh_df["acc_sa"]
        beh_df["d_f1"] = beh_df["f1_ea"] - beh_df["f1_sa"]
        beh_df["d_nll"] = beh_df["nll_ea"] - beh_df["nll_sa"]
        beh_df["d_ece"] = beh_df["ece_ea"] - beh_df["ece_sa"]

        print("\n-- Behavioural metrics (all, with EA - SA deltas) --")
        beh_grp = (
            beh_df.groupby(["dataset", "context_length"])
            .agg(
                n_runs=("seed", "count"),
                acc_sa_mean=("acc_sa", "mean"),
                acc_ea_mean=("acc_ea", "mean"),
                d_acc_mean=("d_acc", "mean"),
                f1_sa_mean=("f1_sa", "mean"),
                f1_ea_mean=("f1_ea", "mean"),
                d_f1_mean=("d_f1", "mean"),
                nll_sa_mean=("nll_sa", "mean"),
                nll_ea_mean=("nll_ea", "mean"),
                d_nll_mean=("d_nll", "mean"),
                ece_sa_mean=("ece_sa", "mean"),
                ece_ea_mean=("ece_ea", "mean"),
                d_ece_mean=("d_ece", "mean"),
            )
            .reset_index()
            .sort_values(["dataset", "context_length"])
        )
        print(beh_grp.round(4).to_string())

        overall_beh = beh_df[["d_acc", "d_f1", "d_nll", "d_ece"]].mean()
        print("\nOverall mean deltas (EA - SA) across all behavioural runs:")
        print(overall_beh.round(4).to_string())
    else:
        print("\n[info] No behavioural results to aggregate.")

    # Geometry summary
    if all_geom_rows:
        geom_df = pd.DataFrame(all_geom_rows)
        geom_df["d_collapse_top1"] = geom_df["collapse_ea_top1"] - geom_df["collapse_sa_top1"]
        geom_df["d_collapse_top5"] = geom_df["collapse_ea_top5"] - geom_df["collapse_sa_top5"]
        geom_df["d_mean_cos"] = geom_df["mean_cos_ea"] - geom_df["mean_cos_sa"]
        geom_df["d_neff_mean"] = geom_df["neff_ea_mean"] - geom_df["neff_sa_mean"]
        geom_df["d_purity_top1"] = geom_df["purity_ea_top1"] - geom_df["purity_sa_top1"]
        geom_df["d_purity_topk"] = geom_df["purity_ea_topk"] - geom_df["purity_sa_topk"]
        geom_df["d_m_mean"] = geom_df["m_mean_ea"] - geom_df["m_mean_sa"]
        geom_df["d_m_cv"] = geom_df["m_cv_ea"] - geom_df["m_cv_sa"]

        print("\n-- Geometry metrics (all, with EA - SA deltas where applicable) --")
        geom_grp = (
            geom_df.groupby("dataset")
            .agg(
                collapse_sa_top1_mean=("collapse_sa_top1", "mean"),
                collapse_ea_top1_mean=("collapse_ea_top1", "mean"),
                d_collapse_top1_mean=("d_collapse_top1", "mean"),
                collapse_sa_top5_mean=("collapse_sa_top5", "mean"),
                collapse_ea_top5_mean=("collapse_ea_top5", "mean"),
                d_collapse_top5_mean=("d_collapse_top5", "mean"),
                mean_cos_sa_mean=("mean_cos_sa", "mean"),
                mean_cos_ea_mean=("mean_cos_ea", "mean"),
                d_mean_cos_mean=("d_mean_cos", "mean"),
                neff_sa_mean=("neff_sa_mean", "mean"),
                neff_ea_mean=("neff_ea_mean", "mean"),
                d_neff_mean=("d_neff_mean", "mean"),
                purity_sa_top1_mean=("purity_sa_top1", "mean"),
                purity_ea_top1_mean=("purity_ea_top1", "mean"),
                d_purity_top1_mean=("d_purity_top1", "mean"),
                purity_sa_topk_mean=("purity_sa_topk", "mean"),
                purity_ea_topk_mean=("purity_ea_topk", "mean"),
                d_purity_topk_mean=("d_purity_topk", "mean"),
                cka_sa_ea_mean=("cka_sa_ea", "mean"),
                m_mean_sa_mean=("m_mean_sa", "mean"),
                m_mean_ea_mean=("m_mean_ea", "mean"),
                d_m_mean_mean=("d_m_mean", "mean"),
                m_cv_sa_mean=("m_cv_sa", "mean"),
                m_cv_ea_mean=("m_cv_ea", "mean"),
                d_m_cv_mean=("d_m_cv", "mean"),
            )
            .sort_index()
        )
        print(geom_grp.round(4).to_string())
    else:
        print("\n[info] No geometry results to aggregate.")

    # Robustness summary
    if all_rob_rows:
        rob_df = pd.DataFrame(all_rob_rows)
        rob_df["d_acc"] = rob_df["acc_ea"] - rob_df["acc_sa"]
        rob_df["d_nll"] = rob_df["nll_ea"] - rob_df["nll_sa"]
        rob_df["d_f1"] = rob_df["f1_ea"] - rob_df["f1_sa"]
        rob_df["d_ece"] = rob_df["ece_ea"] - rob_df["ece_sa"]

        print("\n-- Robustness metrics (all, with EA - SA deltas) --")
        rob_grp = (
            rob_df.groupby(["dataset", "condition", "param_type", "param_value"])
            .agg(
                n_runs=("seed", "count"),
                n_test_mean=("n_test", "mean"),
                acc_sa_mean=("acc_sa", "mean"),
                acc_ea_mean=("acc_ea", "mean"),
                d_acc_mean=("d_acc", "mean"),
                nll_sa_mean=("nll_sa", "mean"),
                nll_ea_mean=("nll_ea", "mean"),
                d_nll_mean=("d_nll", "mean"),
                f1_sa_mean=("f1_sa", "mean"),
                f1_ea_mean=("f1_ea", "mean"),
                d_f1_mean=("d_f1", "mean"),
                ece_sa_mean=("ece_sa", "mean"),
                ece_ea_mean=("ece_ea", "mean"),
                d_ece_mean=("d_ece", "mean"),
            )
            .reset_index()
            .sort_values(["dataset", "condition", "param_type", "param_value"])
        )
        print(rob_grp.round(4).to_string(index=False))
    else:
        print("\n[info] No robustness results to aggregate.")


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
        default=str(REPO_ROOT / "checkpoints_mini_tabicl_stage2_ea_icl_only" / "step-1000.ckpt"),
        help="Stage-2 mini-TabICL checkpoint with elliptical attention.",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default="3,6,11,15,23,28,29,31,32,37,44,46,50,54,151,182",
        help="Comma-separated list of OpenML IDs or names (e.g. '3,11,cmc').",
    )
    ap.add_argument(
        "--context_sizes",
        type=int,
        nargs="*",
        default=None,
        help="Context sizes (train_size values) to evaluate in the behavioural panel.",
    )
    ap.add_argument(
        "--n_estimators",
        type=int,
        default=32,
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
        help="Base random seed for splits and ensemble configs (used to derive a small default seed sweep).",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of seeds for a small seed sweep. When provided, overrides the default sweep derived from --seed.",
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
        help="Skip robustness / inductive-bias tests (irrelevant features, label poisoning, outliers).",
    )
    ap.add_argument(
        "--n_noise_features",
        type=int,
        default=50,
        help="Number of pure Gaussian noise feature columns for the irrelevant-feature robustness test.",
    )
    ap.add_argument(
        "--label_poison_fracs",
        type=float,
        nargs="*",
        default=[0.1],
        help="Fractions of training labels to randomly flip for the label-poisoning robustness test.",
    )
    ap.add_argument(
        "--outlier_fracs",
        type=float,
        nargs="*",
        default=[0.0, 0.05, 0.1],
        help="Fractions of synthetic outlier rows to inject into the training context for the context-poisoning robustness test.",
    )
    ap.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional max rows per dataset (random subsample) for quick runs.",
    )
    # Hierarchical classification toggle (default: enabled)
    ap.add_argument(
        "--use_hierarchical",
        dest="use_hierarchical",
        action="store_true",
        help="Enable hierarchical classification in TabICLClassifier (default).",
    )
    ap.add_argument(
        "--no_hierarchical",
        dest="use_hierarchical",
        action="store_false",
        help="Disable hierarchical classification in TabICLClassifier.",
    )
    ap.set_defaults(use_hierarchical=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
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

    # Determine seeds for sweep (at least one).
    # By default, run a small sweep over three seeds derived from --seed.
    if args.seeds:
        seed_values = list(dict.fromkeys(args.seeds))
    else:
        base = int(args.seed)
        seed_values = [base + i for i in range(3)]

    # Collect results across all datasets / seeds for aggregated summary
    all_beh_rows: List[Dict[str, Any]] = []
    all_geom_rows: List[Dict[str, Any]] = []
    all_rob_rows: List[Dict[str, Any]] = []

    for seed in seed_values:
        set_seed(seed)
        if len(seed_values) > 1:
            print(f"\n##### Seed {seed} #####")

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
                    random_state=seed,
                    stratify=y,
                )
                print(f"[info] Subsampled to {len(X)} rows for dataset {name} (seed={seed})")

            if not args.skip_behaviour:
                try:
                    beh_rows = run_behavioural_panel(
                        ds_name=name,
                        X=X,
                        y=y,
                        model_sa=sa_ckpt,
                        model_ea=ea_ckpt,
                        device=device,
                        context_sizes=args.context_sizes,
                        n_estimators=args.n_estimators,
                        seed=seed,
                        use_hierarchical=args.use_hierarchical,
                    )
                    if beh_rows:
                        all_beh_rows.extend(beh_rows)
                except torch.OutOfMemoryError as e:
                    _handle_oom("behavioural panel", name, e)
                    continue

            if not args.skip_geometry:
                try:
                    geom_rows = run_geometry_panel(
                        ds_name=name,
                        X=X,
                        y=y,
                        model_sa=sa_ckpt,
                        model_ea=ea_ckpt,
                        device=device,
                        n_estimators=args.n_estimators,
                        seed=seed,
                        use_hierarchical=args.use_hierarchical,
                    )
                    if geom_rows:
                        all_geom_rows.extend(geom_rows)
                except torch.OutOfMemoryError as e:
                    _handle_oom("geometry panel", name, e)
                    continue

            if not args.skip_robustness:
                try:
                    rob_rows = run_robustness_panel(
                        ds_name=name,
                        X=X,
                        y=y,
                        model_sa=sa_ckpt,
                        model_ea=ea_ckpt,
                        device=device,
                        n_estimators=args.n_estimators,
                        seed=seed,
                        outlier_fracs=args.outlier_fracs,
                        n_noise_features=args.n_noise_features,
                        label_poison_fracs=args.label_poison_fracs,
                        use_hierarchical=args.use_hierarchical,
                    )
                    if rob_rows:
                        all_rob_rows.extend(rob_rows)
                except torch.OutOfMemoryError as e:
                    _handle_oom("robustness panel", name, e)
                    continue

    # Print aggregated summary at the end (if any results exist)
    _print_aggregated_summary(all_beh_rows, all_geom_rows, all_rob_rows)


if __name__ == "__main__":
    main()
