#!/usr/bin/env python3
"""
Thesis-rigorous robustness runner for TabICL checkpoints (auditable runs).

Focuses on perturbations that match existing tabular foundation model robustness evaluations
(e.g., TabPFN) and common ICL-demonstration corruption tests.

This script:
  - Creates an auditable run directory (env/config/checkpoints/datasets/splits/corruptions).
  - Evaluates multiple checkpoints under identical splits and corruption realizations.
  - Logs long-format metrics to run_dir/metrics.csv.

Core corruptions (literature-aligned):
  1) TabPFN-style *uninformative feature injection*
     (append shuffled copies of existing features; preserves marginals + missingness).
  2) TabPFN-style *cell-wise outliers* for numeric columns
     (with probability p per numeric cell, multiply by U(0, outlier_factor)).
  3) ICL-style *noisy demonstration labels*
     (flip a fraction of training/context labels to a different class).

Optional extension:
  - Rotation invariance stress test on a numeric subspace (train+test rotated together).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

# Make local src importable
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tabicl.sklearn.classifier import TabICLClassifier
from tabicl.pdlc.embed import extract_tf_row_embeddings
from tabicl.model.attention import compute_elliptical_diag


# Default OpenML IDs: Panel B (robustness characterization set)
DEFAULT_OPENML_IDS: List[int] = [
    14969,   # GesturePhaseSegmentationProcessed
    14952,   # PhishingWebsites
    7592,    # adult
    14965,   # bank-marketing
    219,     # electricity
    9977,    # nomao
    167141,  # churn
    43,      # spambase
    9976,    # madelon
    9964,    # semeion
    146824,  # mfeat-pixel
    28,      # optdigits
    32,      # pendigits
    2074,    # satimage
    146822,  # segment
    9960,    # wall-robot-navigation
    9952,    # phoneme
    146817,  # steel-plates-fault
    3021,    # sick
    146820,  # wilt
    9978,    # ozone-level-8hr
    3918,    # pc1
    3,       # kr-vs-kp
    49,      # tic-tac-toe
    146821,  # car
    31,      # credit-g
    29,      # credit-approval
    14954,   # cylinder-bands
]


# ---------------------------------------------------------------------------
# Generic utilities (dataset loading, metrics, CSV appending)
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set basic RNG seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_env_info(device: str) -> Dict[str, Any]:
    import sklearn

    try:
        import openml  # noqa: F401

        openml_version = getattr(__import__("openml"), "__version__", None)
    except Exception:
        openml_version = None

    info: Dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "openml": openml_version,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_cudnn_version": getattr(torch.backends.cudnn, "version", lambda: None)(),
    }
    if torch.cuda.is_available():
        try:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return info


def maybe_free_torch_memory() -> None:
    """Best-effort GPU memory cleanup (CUDA/ROCm share torch.cuda API)."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # On some setups this can help with fragmentation; ignore if unsupported.
        try:
            torch.cuda.ipc_collect()
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
            header = True

    df = pd.DataFrame(rows, columns=list(columns))
    df.to_csv(out_path, mode="a", index=False, header=header)


def _extract_target_if_missing(
    X: pd.DataFrame,
    y: pd.Series | None,
    ds_name: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """If y is None, try to infer a classification target column from X and drop it."""
    if y is not None and not isinstance(y, type(None)):
        return X, y

    cols = list(X.columns)
    preferred = ["class", "Class", "target", "Target", "label", "Label", "y"]
    for c in preferred:
        if c in cols:
            y_series = X[c].copy()
            X_feat = X.drop(columns=[c])
            return X_feat, y_series

    for c in cols:
        nunique = X[c].nunique(dropna=True)
        if 2 <= nunique <= min(50, max(2, X.shape[0] // 2)):
            y_series = X[c].copy()
            X_feat = X.drop(columns=[c])
            return X_feat, y_series

    # Fallback: use last column
    c = cols[-1]
    y_series = X[c].copy()
    X_feat = X.drop(columns=[c])
    return X_feat, y_series


@dataclass(frozen=True)
class DatasetInfo:
    input_spec: str
    dataset_name: str
    resolved_dataset_id: int | None
    resolved_task_id: int | None
    dataset_version: int | None


def fetch_openml_dataset_with_info(name_or_id: str | int) -> Tuple[pd.DataFrame, pd.Series, DatasetInfo]:
    """Load an OpenML dataset and return richer identifiers for auditing."""
    import openml

    spec = str(name_or_id)
    resolved_task_id: int | None = None
    ds = None

    if isinstance(name_or_id, int) or spec.isdigit():
        num_id = int(name_or_id)
        try:
            ds = openml.datasets.get_dataset(num_id)
        except Exception:
            ds = None
        if ds is None:
            task = openml.tasks.get_task(num_id)
            resolved_task_id = num_id
            ds_id = int(getattr(task, "dataset_id"))
            ds = openml.datasets.get_dataset(ds_id)
    else:
        name = spec.lower()
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
    X, y = _extract_target_if_missing(X, y, ds_name=ds.name)

    info = DatasetInfo(
        input_spec=spec,
        dataset_name=str(ds.name),
        resolved_dataset_id=int(getattr(ds, "dataset_id", None)) if getattr(ds, "dataset_id", None) else None,
        resolved_task_id=resolved_task_id,
        dataset_version=int(getattr(ds, "version", None)) if getattr(ds, "version", None) else None,
    )

    if isinstance(name_or_id, int) or spec.isdigit():
        if resolved_task_id is None:
            print(
                f"[openml] Interpreted '{spec}' as dataset id -> {info.resolved_dataset_id} "
                f"(name={info.dataset_name}, version={info.dataset_version})"
            )
        else:
            print(
                f"[openml] Interpreted '{spec}' as task id -> dataset id {info.resolved_dataset_id} "
                f"(name={info.dataset_name}, version={info.dataset_version})"
            )

    return X, y, info


def compute_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) over max-probability bins."""
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
        if i > 0:
            mask = (confidences > lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        conf_bin = confidences[mask].mean()
        acc_bin = correct[mask].mean()
        ece += (mask.mean()) * abs(acc_bin - conf_bin)
    return float(ece)


def compute_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Multi-class Brier score: mean_i sum_k (p_ik - 1[y_i=k])^2 (lower is better)."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError("proba must be 2D (n_samples, n_classes)")
    n, k = proba.shape
    if y_true.shape[0] != n:
        raise ValueError("y_true length must match proba rows")
    y_onehot = np.zeros((n, k), dtype=float)
    valid = (y_true >= 0) & (y_true < k)
    y_onehot[np.arange(n)[valid], y_true[valid]] = 1.0
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def _stable_subsample_idx(n: int, k: int, *, seed: int) -> np.ndarray:
    if n <= k:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, size=int(k), replace=False).astype(int))


def _subsample_seed_for_dataset(ds_key: str, seed: int) -> int:
    """Deterministic seed used for row-subsampling, tied to dataset_key and split seed."""
    h = hashlib.sha256(f"{ds_key}__{int(seed)}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


def _pairwise_cosine_stats(Z: np.ndarray, *, max_rows: int, subsample_seed: int) -> Tuple[float, float]:
    """Return (mean_cos, p95_cos) over off-diagonal pairwise cosines (on a stable subset)."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] < 2:
        return float("nan"), float("nan")
    n = int(Z.shape[0])
    idx = _stable_subsample_idx(n, int(max_rows), seed=int(subsample_seed))
    Zs = Z[idx]
    norms = np.linalg.norm(Zs, axis=1, keepdims=True) + 1e-12
    Z_norm = Zs / norms
    S = Z_norm @ Z_norm.T
    mask = ~np.eye(S.shape[0], dtype=bool)
    vals = S[mask]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(np.quantile(vals, 0.95))


def _collapse_from_embeddings(Z: np.ndarray, *, max_rows: int, subsample_seed: int) -> Tuple[float, float]:
    """Return (collapse_top1, d_eff) from covariance spectrum on a stable subset."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] < 2:
        return float("nan"), float("nan")
    n = int(Z.shape[0])
    idx = _stable_subsample_idx(n, int(max_rows), seed=int(subsample_seed))
    Zs = Z[idx]
    Zc = Zs - Zs.mean(axis=0, keepdims=True)
    # Eigenvalues of covariance via SVD for stability; cov eigenvalues = s^2/(n-1)
    try:
        s = np.linalg.svd(Zc, full_matrices=False, compute_uv=False)
    except Exception:
        C = np.cov(Zc, rowvar=False)
        evals = np.linalg.eigvalsh(C)
        evals = np.sort(np.clip(evals, a_min=0.0, a_max=None))[::-1]
        if evals.size == 0:
            return float("nan"), float("nan")
        s1 = float(evals.sum())
        collapse_top1 = float(evals[0] / s1) if s1 > 0 else float("nan")
        s2 = float((evals**2).sum())
        d_eff = float((s1 * s1) / s2) if s2 > 0 else float("nan")
        return collapse_top1, d_eff

    evals = (s**2) / max(1, (Zc.shape[0] - 1))
    if evals.size == 0:
        return float("nan"), float("nan")
    s1 = float(evals.sum())
    collapse_top1 = float(evals[0] / s1) if s1 > 0 else float("nan")
    s2 = float((evals**2).sum())
    d_eff = float((s1 * s1) / s2) if s2 > 0 else float("nan")
    return collapse_top1, d_eff


def compute_tfrow_collapse_metrics(
    clf: TabICLClassifier,
    X_test: pd.DataFrame,
    *,
    max_rows_spectrum: int,
    max_rows_cosine: int,
    subsample_seed: int,
) -> Dict[str, float]:
    """Collapse metrics on TFrow output for test rows (embeddings_test)."""
    res = extract_tf_row_embeddings(clf, X_test, choose_random_variant=False)
    Z = np.asarray(res["embeddings_test"])
    collapse_top1, d_eff = _collapse_from_embeddings(Z, max_rows=int(max_rows_spectrum), subsample_seed=int(subsample_seed))
    cos_mean, cos_p95 = _pairwise_cosine_stats(Z, max_rows=int(max_rows_cosine), subsample_seed=int(subsample_seed))
    return {
        "collapse_top1": float(collapse_top1),
        "d_eff": float(d_eff),
        "cos_mean": float(cos_mean),
        "cos_p95": float(cos_p95),
    }


@torch.no_grad()
def _compute_test_to_train_weights(
    model,
    R_cond: torch.Tensor,
    train_size: int,
    *,
    chunk_test: int,
    use_fp16: bool,
) -> torch.Tensor:
    """Mean-head test→train attention weights (T_test, T_train) with low memory."""
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
    q_in = last.norm1(x) if last.norm_first else x
    B, T, E = q_in.shape
    nh = last.attn.num_heads
    hs = E // nh
    q, k, v = F._in_projection_packed(q_in, q_in, q_in, last.attn.in_proj_weight, last.attn.in_proj_bias)
    q = q.view(B, T, nh, hs).transpose(-3, -2)  # (B, nh, T, hs)
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

    q_test = q[..., train_size:, :]  # (B, nh, T_test, hs)
    k_train = k[..., :train_size, :]  # (B, nh, T_train, hs)
    T_test = int(q_test.shape[-2])
    T_train = int(k_train.shape[-2])

    Bnh = int(B * nh)
    q_test_b = q_test.reshape(Bnh, T_test, hs)
    k_train_b = k_train.reshape(Bnh, T_train, hs)
    if use_fp16:
        q_test_b = q_test_b.to(torch.float16)
        k_train_b = k_train_b.to(torch.float16)

    out = torch.zeros(T_test, T_train, device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(hs)

    for start in range(0, T_test, int(chunk_test)):
        end = min(T_test, start + int(chunk_test))
        q_chunk = q_test_b[:, start:end, :]  # (Bnh, t, hs)
        scores = torch.bmm(q_chunk, k_train_b.transpose(1, 2)) * scale  # (Bnh, t, T_train)
        w = torch.softmax(scores.to(torch.float32), dim=-1)
        w_mean = w.mean(dim=0)  # (t, T_train)
        out[start:end, :] += w_mean

    return out


def _build_episode(clf: TabICLClassifier, X_te: pd.DataFrame) -> Tuple[torch.Tensor, int]:
    """Construct a single deterministic ICL episode from X_te.

    Returns
    -------
    (R_cond, train_size, y_train_shifted, shift_offset)
      - R_cond: conditioned row representations for the combined (train+test) sequence
      - train_size: number of support rows
      - y_train_shifted: shifted (encoded) train labels used for conditioning
      - shift_offset: cyclic class shift offset used for this variant
    """
    X_te_num = clf.X_encoder_.transform(X_te)
    data = clf.ensemble_generator_.transform(X_te_num)
    methods = list(data.keys())
    norm_method = methods[0]
    Xs, ys_shifted = data[norm_method]
    shuffle_patterns = clf.ensemble_generator_.feature_shuffle_patterns_[norm_method]
    shift_offsets = clf.ensemble_generator_.class_shift_offsets_[norm_method]

    vidx = 0
    for i, p in enumerate(shuffle_patterns):
        if list(p) == sorted(p):
            vidx = i
            break

    X_variant = Xs[vidx]
    y_train_shifted = ys_shifted[vidx]
    shift_offset = int(shift_offsets[vidx]) if len(shift_offsets) > vidx else 0
    train_size = int(y_train_shifted.shape[0])

    model = clf.model_.to(clf.device_)
    model.eval()

    X_tensor = torch.from_numpy(X_variant).float().unsqueeze(0).to(clf.device_)
    with torch.no_grad():
        col_out = model.col_embedder(X_tensor, train_size=train_size, mgr_config=clf.inference_config_.COL_CONFIG)
        row_reps = model.row_interactor(col_out, mgr_config=clf.inference_config_.ROW_CONFIG)
        yt = torch.as_tensor(y_train_shifted, device=clf.device_, dtype=torch.float32).unsqueeze(0)
        R_cond = row_reps.clone()
        R_cond[:, :train_size] = R_cond[:, :train_size] + model.icl_predictor.y_encoder(yt)
    return R_cond, train_size, np.asarray(y_train_shifted, dtype=int), int(shift_offset)


def attention_neff_summary(weights_tt: torch.Tensor) -> Tuple[float, float]:
    """Mean/std N_eff over test rows given (T_test, T_train) weights."""
    if weights_tt.ndim != 2 or weights_tt.numel() == 0:
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
    alpha = W_clean / row_sums

    s2 = (alpha**2).sum(dim=1)
    mask = (s2 > 0) & torch.isfinite(s2)
    if not mask.any():
        return float("nan"), float("nan")
    neff = 1.0 / s2[mask]
    neff_np = neff.detach().cpu().numpy().astype(np.float64)
    return float(neff_np.mean()), float(neff_np.std())


def neighbor_label_purity(
    weights_tt: torch.Tensor,
    y_train_shifted: np.ndarray,
    y_test_shifted: np.ndarray,
    *,
    topk: int = 5,
) -> Tuple[float, float]:
    """Top-1 hit rate and top-k label purity using attention as neighbourhood weights."""
    if weights_tt.ndim != 2 or weights_tt.numel() == 0:
        return float("nan"), float("nan")

    T_test, T_train = weights_tt.shape
    W = weights_tt.detach().cpu().numpy()
    y_tr = np.asarray(y_train_shifted)
    y_te = np.asarray(y_test_shifted)
    if y_tr.shape[0] != T_train or y_te.shape[0] != T_test:
        return float("nan"), float("nan")

    k = max(1, min(int(topk), int(T_train)))
    top1_hits: List[float] = []
    purities: List[float] = []
    for t_idx in range(T_test):
        alpha = W[t_idx]
        if not np.isfinite(alpha).any():
            continue
        top_idx = np.argsort(-alpha)[:k]
        neigh_labels = y_tr[top_idx]
        qlab = y_te[t_idx]
        top1_hits.append(float(neigh_labels[0] == qlab))
        purities.append(float((neigh_labels == qlab).mean()))

    if not purities:
        return float("nan"), float("nan")
    return float(np.mean(top1_hits)), float(np.mean(purities))


def compute_tficl_attention_metrics(
    clf: TabICLClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    max_test_rows: int,
    chunk_test: int,
    use_fp16: bool,
    subsample_seed: int,
) -> Dict[str, float]:
    if int(max_test_rows) > 0 and int(X_test.shape[0]) > int(max_test_rows):
        idx = _stable_subsample_idx(int(X_test.shape[0]), int(max_test_rows), seed=int(subsample_seed))
        X_use = X_test.iloc[idx].reset_index(drop=True)
        y_use = y_test.iloc[idx].reset_index(drop=True)
    else:
        X_use = X_test
        y_use = y_test

    R_cond, train_size, y_train_shifted, shift_offset = _build_episode(clf, X_use)
    w_tt = _compute_test_to_train_weights(
        clf.model_,
        R_cond,
        train_size,
        chunk_test=int(chunk_test),
        use_fp16=bool(use_fp16),
    )
    neff_mean, neff_std = attention_neff_summary(w_tt)

    y_te_int = clf.y_encoder_.transform(y_use)
    y_te_shifted = (np.asarray(y_te_int, dtype=int) + int(shift_offset)) % int(clf.n_classes_)
    purity_top1, purity_top5 = neighbor_label_purity(w_tt, y_train_shifted, y_te_shifted, topk=5)
    return {
        "neff_mean": float(neff_mean),
        "neff_std": float(neff_std),
        "purity_top1": float(purity_top1),
        "purity_top5": float(purity_top5),
    }


@dataclass
class BehaviourMetrics:
    accuracy: float
    f1_macro: float
    log_loss: float
    ece: float
    brier: float


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
    brier = compute_brier(y_true_int, proba)
    return BehaviourMetrics(
        accuracy=float(acc),
        f1_macro=float(f1),
        log_loss=float(ll),
        ece=float(ece),
        brier=float(brier),
    )


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: str
    sha256: str


def _dedupe_names(names: List[str]) -> List[str]:
    out: List[str] = []
    seen: Dict[str, int] = {}
    for n in names:
        base = n
        k = seen.get(base, 0)
        if k == 0:
            out.append(base)
        else:
            out.append(f"{base}_{k}")
        seen[base] = k + 1
    return out


def parse_checkpoints(checkpoint_paths: Sequence[str], checkpoint_names: Sequence[str] | None) -> List[CheckpointSpec]:
    paths = [Path(p).expanduser().resolve() for p in checkpoint_paths]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"Checkpoint not found: {p}")

    if checkpoint_names is not None and len(checkpoint_names) > 0:
        if len(checkpoint_names) != len(paths):
            raise SystemExit(
                f"--checkpoint_names must match --checkpoints length "
                f"({len(checkpoint_names)} vs {len(paths)})"
            )
        names = list(checkpoint_names)
    else:
        names = [p.stem for p in paths]
    names = _dedupe_names(names)

    out: List[CheckpointSpec] = []
    for name, p in zip(names, paths):
        out.append(CheckpointSpec(name=name, path=str(p), sha256=sha256_file(p)))
    return out


def infer_variant(checkpoint_name: str) -> str:
    u = str(checkpoint_name).strip().upper()
    if u.startswith("SA"):
        return "SA"
    if u.startswith("EA"):
        return "EA"
    return ""


def infer_case_label(case: Case) -> str:
    def _fmt(v: float) -> str:
        try:
            fv = float(v)
        except Exception:
            return str(v)
        if np.isfinite(fv) and abs(fv - round(fv)) < 1e-12:
            return str(int(round(fv)))
        return f"{fv:g}"

    if case.condition == "clean":
        return "clean"
    kind = case.artifacts.get("kind")
    if kind == "cell_outliers_tabpfn":
        return f"outliers_{case.applies_to}_fac{_fmt(case.param_value)}"
    if kind == "icl_noisy_demo_labels":
        return f"label_noise_train_frac{_fmt(case.param_value)}"
    if kind == "uninformative_features_shuffled":
        return f"uninformative_{case.applies_to}_n{_fmt(case.param_value)}"
    if kind == "feature_rotation_kdim_both":
        return f"rotation_both_k{_fmt(case.param_value)}"
    return str(case.condition)


def infer_scope(case: Case, *, refit: bool) -> str:
    """Scope describes *what was corrupted* (not whether a refit happened)."""
    if case.condition == "clean":
        return "none"
    if not refit:
        return "test_only"
    if case.applies_to == "train":
        return "train_only"
    if case.applies_to == "test":
        return "test_only"
    if case.applies_to == "both":
        return "train_and_test"
    return "none"


def _slug(s: str, max_len: int = 80) -> str:
    s = str(s)
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    slug = "_".join([p for p in slug.split("_") if p])
    return slug[:max_len] if len(slug) > max_len else slug


def dataset_key(info: DatasetInfo) -> str:
    did = info.resolved_dataset_id
    ver = info.dataset_version
    base = f"openml_{did}" if did is not None else f"openml_{_slug(info.dataset_name)}"
    if ver is not None:
        base += f"_v{ver}"
    return f"{base}__{_slug(info.dataset_name)}"


@dataclass(frozen=True)
class SplitInfo:
    idx_train: np.ndarray
    idx_test: np.ndarray
    stratified: bool


def make_split_indices(
    y: pd.Series,
    seed: int,
    *,
    test_size: float,
    stratify: bool = True,
) -> SplitInfo:
    idx = np.arange(len(y))
    strat = y if (stratify and y.nunique() > 1) else None
    try:
        idx_tr, idx_te = train_test_split(
            idx,
            test_size=test_size,
            random_state=seed,
            stratify=strat,
        )
        return SplitInfo(idx_train=np.asarray(idx_tr, dtype=int), idx_test=np.asarray(idx_te, dtype=int), stratified=strat is not None)
    except ValueError as e:
        # Fall back to unstratified split (important for small class counts).
        print(f"[warn] Stratified split failed (seed={seed}): {e}. Falling back to unstratified split.")
        idx_tr, idx_te = train_test_split(
            idx,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
        return SplitInfo(idx_train=np.asarray(idx_tr, dtype=int), idx_test=np.asarray(idx_te, dtype=int), stratified=False)


@dataclass(frozen=True)
class Case:
    condition: str
    param_type: str
    param_value: float
    applies_to: Literal["none", "train", "test", "both"]
    artifacts: Dict[str, Any]


def case_id(case: Case) -> str:
    return _slug(f"{case.condition}__{case.param_type}__{case.param_value}__{case.applies_to}", max_len=160)


# ---------------------------------------------------------------------------
# Corruption artifacts (paired across checkpoints)
# ---------------------------------------------------------------------------


def _stable_case_rng(dataset_key: str, seed: int) -> np.random.Generator:
    h = hashlib.sha256(f"{dataset_key}__{seed}".encode("utf-8")).digest()
    s = int.from_bytes(h[:4], byteorder="little", signed=False)
    return np.random.default_rng(s)


def make_uninformative_artifacts(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_add: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    cols = list(X_train.columns)
    if n_add <= 0 or not cols:
        return {"kind": "uninformative_features_shuffled", "n_add": 0, "picked_cols": [], "train_perm": None, "test_perm": None}

    picked = rng.choice(cols, size=int(n_add), replace=True)
    train_perm = np.stack([rng.permutation(len(X_train)) for _ in range(int(n_add))], axis=0).astype(np.int32)
    test_perm = np.stack([rng.permutation(len(X_test)) for _ in range(int(n_add))], axis=0).astype(np.int32)
    return {
        "kind": "uninformative_features_shuffled",
        "n_add": int(n_add),
        "picked_cols": picked.tolist(),
        "train_perm": train_perm,
        "test_perm": test_perm,
    }


def apply_uninformative_artifacts(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    art: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_add = int(art.get("n_add", 0) or 0)
    if n_add <= 0:
        return X_train, X_test

    picked_cols: List[str] = list(art["picked_cols"])
    train_perm: np.ndarray = np.asarray(art["train_perm"])
    test_perm: np.ndarray = np.asarray(art["test_perm"])

    X_tr = X_train.reset_index(drop=True).copy()
    X_te = X_test.reset_index(drop=True).copy()

    for i, src in enumerate(picked_cols):
        new_c = f"uninformative_shuffle_{src}_{i}"
        tr_vals = X_tr[src].to_numpy(copy=True)
        te_vals = X_te[src].to_numpy(copy=True)
        X_tr[new_c] = tr_vals[train_perm[i]]
        X_te[new_c] = te_vals[test_perm[i]]

    return X_tr, X_te


def make_outlier_artifacts(
    X: pd.DataFrame,
    rng: np.random.Generator,
    *,
    p_cell: float,
    outlier_factor: float,
) -> Dict[str, Any]:
    num_cols = list(X.select_dtypes(include="number").columns)
    idx_map: Dict[str, np.ndarray] = {}
    mult_map: Dict[str, np.ndarray] = {}
    for c in num_cols:
        col = X[c].to_numpy(copy=False)
        finite = np.isfinite(col)
        mask = (rng.random(size=col.shape[0]) < float(p_cell)) & finite
        idx = np.where(mask)[0].astype(np.int32)
        mult = rng.uniform(0.0, float(outlier_factor), size=int(idx.size)).astype(np.float32)
        idx_map[c] = idx
        mult_map[c] = mult
    return {
        "kind": "cell_outliers_tabpfn",
        "p_cell": float(p_cell),
        "outlier_factor": float(outlier_factor),
        "numeric_cols": num_cols,
        "idx_map": idx_map,
        "mult_map": mult_map,
    }


def apply_outlier_artifacts(X: pd.DataFrame, art: Dict[str, Any]) -> pd.DataFrame:
    X_out = X.copy()
    num_cols: List[str] = list(art.get("numeric_cols", []))
    idx_map: Dict[str, np.ndarray] = art.get("idx_map", {})
    mult_map: Dict[str, np.ndarray] = art.get("mult_map", {})
    for c in num_cols:
        idx = np.asarray(idx_map.get(c, np.asarray([], dtype=np.int32)))
        if idx.size == 0:
            continue
        mult = np.asarray(mult_map.get(c, np.asarray([], dtype=np.float32)))
        col = X_out[c].to_numpy(copy=True)
        col[idx] = col[idx] * mult
        X_out[c] = col
    return X_out


def make_label_flip_artifacts(y_train: pd.Series, frac: float, rng: np.random.Generator) -> Dict[str, Any]:
    y = y_train.reset_index(drop=True)
    n = len(y)
    n_flip = int(round(float(frac) * n))
    if n_flip <= 0:
        return {"kind": "icl_noisy_demo_labels", "frac": float(frac), "idx": np.asarray([], dtype=np.int32), "new_labels": np.asarray([], dtype=object)}

    classes = np.asarray(pd.unique(y))
    if classes.size < 2:
        return {"kind": "icl_noisy_demo_labels", "frac": float(frac), "idx": np.asarray([], dtype=np.int32), "new_labels": np.asarray([], dtype=object)}

    idx = rng.choice(n, size=n_flip, replace=False).astype(np.int32)
    new_labels: List[Any] = []
    for i in idx:
        current = y.iloc[int(i)]
        choices = classes[classes != current]
        new_labels.append(rng.choice(choices))

    return {"kind": "icl_noisy_demo_labels", "frac": float(frac), "idx": idx, "new_labels": np.asarray(new_labels, dtype=object)}


def apply_label_flip_artifacts(y_train: pd.Series, art: Dict[str, Any]) -> pd.Series:
    y_poison = y_train.reset_index(drop=True).copy()
    idx = np.asarray(art.get("idx", np.asarray([], dtype=np.int32)))
    new_labels = np.asarray(art.get("new_labels", np.asarray([], dtype=object)), dtype=object)
    for i, lab in zip(idx, new_labels):
        y_poison.iloc[int(i)] = lab
    return y_poison


def _sample_random_orthogonal_matrix(k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive for orthogonal matrix sampling.")
    A = rng.normal(size=(k, k))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _apply_kdim_rotation_matrix(
    X: pd.DataFrame,
    cols: List[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    R: np.ndarray,
) -> pd.DataFrame:
    if not cols:
        return X
    X_rot = X.copy()
    # Ensure float dtype for rotated columns to avoid pandas dtype warnings/errors
    # when the original columns are integer/boolean types.
    X_rot[cols] = X_rot[cols].astype(float)
    Z = X_rot[cols].to_numpy(dtype=float)
    Z = (Z - mu) / sigma
    Z_rot = Z @ R.T
    X_rot.loc[:, cols] = Z_rot
    return X_rot


def build_cases(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    rng: np.random.Generator,
    *,
    uninformative_ns: Sequence[int],
    outlier_factors: Sequence[float],
    outlier_p_cell: float,
    outliers_apply_to: Literal["train", "test", "both"],
    label_poison_fracs: Sequence[float],
    enable_rotation: bool,
    rotation_k: int,
) -> List[Case]:
    cases: List[Case] = [
        Case(condition="clean", param_type="none", param_value=0.0, applies_to="none", artifacts={})
    ]

    for n_add in [n for n in uninformative_ns if int(n) > 0]:
        art = make_uninformative_artifacts(X_train, X_test, int(n_add), rng)
        cases.append(
            Case(
                condition="uninformative_features_shuffled",
                param_type="n_add",
                param_value=float(int(n_add)),
                applies_to="both",
                artifacts=art,
            )
        )

    for fac in [f for f in outlier_factors if float(f) > 0.0]:
        train_art = make_outlier_artifacts(X_train, rng, p_cell=outlier_p_cell, outlier_factor=float(fac)) if outliers_apply_to in ("train", "both") else None
        test_art = make_outlier_artifacts(X_test, rng, p_cell=outlier_p_cell, outlier_factor=float(fac)) if outliers_apply_to in ("test", "both") else None
        cases.append(
            Case(
                condition="cell_outliers_tabpfn",
                param_type="outlier_factor",
                param_value=float(fac),
                applies_to=outliers_apply_to,
                artifacts={
                    "kind": "cell_outliers_tabpfn",
                    "p_cell": float(outlier_p_cell),
                    "outlier_factor": float(fac),
                    "train": train_art,
                    "test": test_art,
                },
            )
        )

    for frac in [f for f in label_poison_fracs if float(f) > 0.0]:
        art = make_label_flip_artifacts(y_train, float(frac), rng)
        cases.append(
            Case(
                condition="icl_noisy_demo_labels",
                param_type="frac",
                param_value=float(frac),
                applies_to="train",
                artifacts=art,
            )
        )

    if enable_rotation:
        num_cols = list(X_train.select_dtypes(include="number").columns)
        k = int(min(max(2, int(rotation_k)), len(num_cols))) if num_cols else 0
        if k >= 2:
            rot_seed = int(rng.integers(0, 1_000_000))
            rng_rot = np.random.default_rng(rot_seed)
            col_idx = rng_rot.choice(len(num_cols), size=k, replace=False)
            cols_k = [num_cols[i] for i in col_idx]
            mu = X_train[cols_k].mean().fillna(0.0).to_numpy(dtype=float)
            sigma = X_train[cols_k].std().replace(0.0, 1.0).fillna(1.0).to_numpy(dtype=float)
            R = _sample_random_orthogonal_matrix(k, rng_rot)
            cols_hash = hashlib.sha256("|".join(cols_k).encode("utf-8")).hexdigest()[:12]
            cases.append(
                Case(
                    condition="feature_rotation_kdim_both",
                    param_type="k_subspace",
                    param_value=float(k),
                    applies_to="both",
                    artifacts={
                        "kind": "feature_rotation_kdim_both",
                        "rotation_seed": rot_seed,
                        "rotation_cols": cols_k,
                        "rotation_cols_hash": cols_hash,
                        "mu": mu,
                        "sigma": sigma,
                        "R": R,
                    },
                )
            )
        else:
            print("[info] Rotation enabled but dataset has <2 numeric columns; skipping rotation case.")

    return cases


def save_corruptions(
    run_dir: Path,
    ds_key: str,
    seed: int,
    cases: Sequence[Case],
) -> None:
    out_dir = run_dir / "corruptions" / ds_key
    ensure_dir(out_dir)

    meta_rows: List[Dict[str, Any]] = []
    arrays: Dict[str, Any] = {}

    for case in cases:
        cid = case_id(case)
        row = {
            "case_id": cid,
            "condition": case.condition,
            "param_type": case.param_type,
            "param_value": case.param_value,
            "applies_to": case.applies_to,
            "artifact_kind": case.artifacts.get("kind"),
        }

        kind = case.artifacts.get("kind")
        if kind == "uninformative_features_shuffled":
            arrays[f"{cid}__picked_cols"] = np.asarray(case.artifacts["picked_cols"], dtype=object)
            arrays[f"{cid}__train_perm"] = np.asarray(case.artifacts["train_perm"], dtype=np.int32)
            arrays[f"{cid}__test_perm"] = np.asarray(case.artifacts["test_perm"], dtype=np.int32)
        elif kind == "icl_noisy_demo_labels":
            arrays[f"{cid}__idx"] = np.asarray(case.artifacts["idx"], dtype=np.int32)
            arrays[f"{cid}__new_labels"] = np.asarray(case.artifacts["new_labels"], dtype=object)
        elif kind == "cell_outliers_tabpfn":
            row["p_cell"] = case.artifacts.get("p_cell")
            row["outlier_factor"] = case.artifacts.get("outlier_factor")
            for which in ("train", "test"):
                art = case.artifacts.get(which)
                if art is None:
                    continue
                num_cols = list(art.get("numeric_cols", []))
                arrays[f"{cid}__{which}__numeric_cols"] = np.asarray(num_cols, dtype=object)
                idx_map = art.get("idx_map", {})
                mult_map = art.get("mult_map", {})
                for c in num_cols:
                    cslug = _slug(c, max_len=80)
                    arrays[f"{cid}__{which}__{cslug}__idx"] = np.asarray(idx_map.get(c, np.asarray([], dtype=np.int32)), dtype=np.int32)
                    arrays[f"{cid}__{which}__{cslug}__mult"] = np.asarray(mult_map.get(c, np.asarray([], dtype=np.float32)), dtype=np.float32)
        elif kind == "feature_rotation_kdim_both":
            row["rotation_seed"] = int(case.artifacts["rotation_seed"])
            row["rotation_cols_hash"] = str(case.artifacts["rotation_cols_hash"])
            arrays[f"{cid}__rotation_cols"] = np.asarray(case.artifacts["rotation_cols"], dtype=object)
            arrays[f"{cid}__mu"] = np.asarray(case.artifacts["mu"], dtype=np.float64)
            arrays[f"{cid}__sigma"] = np.asarray(case.artifacts["sigma"], dtype=np.float64)
            arrays[f"{cid}__R"] = np.asarray(case.artifacts["R"], dtype=np.float64)

        meta_rows.append(row)

    save_json(out_dir / f"seed_{seed}.json", {"dataset_key": ds_key, "seed": int(seed), "cases": meta_rows})
    if arrays:
        np.savez_compressed(out_dir / f"seed_{seed}.npz", **arrays)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Thesis-rigorous robustness runner for TabICL checkpoints.")

    ap.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory (default: results/robustness/<run_name_or_timestamp>/).",
    )
    ap.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name used when --run_dir is not provided.",
    )

    ap.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Checkpoint paths to evaluate (e.g. SA EA-ICL EA-FULL).",
    )
    ap.add_argument(
        "--checkpoint_names",
        type=str,
        nargs="*",
        default=None,
        help="Optional names for checkpoints (must match --checkpoints length).",
    )

    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu).",
    )
    ap.add_argument("--n_estimators", type=int, default=32)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size over ensemble members inside TabICLClassifier (lower = less VRAM).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of seeds. When provided, overrides the default derived sweep.",
    )
    ap.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of seeds to run when --seeds is not provided (seeds = seed..seed+n_seeds-1).",
    )
    ap.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split fraction.",
    )
    ap.add_argument(
        "--no_stratify",
        action="store_true",
        help="Disable stratified splitting (default: stratify when possible).",
    )

    ap.add_argument("--use_hierarchical", dest="use_hierarchical", action="store_true")
    ap.add_argument("--no_hierarchical", dest="use_hierarchical", action="store_false")
    ap.set_defaults(use_hierarchical=True)

    # Dataset selection: OpenML IDs
    ap.add_argument(
        "--openml_ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of OpenML dataset IDs. If omitted, uses a default CC18-like panel.",
    )
    ap.add_argument(
        "--openml_cache_dir",
        type=str,
        default=None,
        help="Override OpenML cache directory (also sets env OPENML_CACHE_DIR). Useful on read-only $HOME.",
    )

    # Literature-aligned robustness knobs
    ap.add_argument(
        "--uninformative_ns",
        type=int,
        nargs="*",
        default=[0, 10, 50],
        help="How many shuffled-copy uninformative features to append (TabPFN-style).",
    )
    ap.add_argument(
        "--outlier_factors",
        type=float,
        nargs="*",
        default=[0.0, 10.0, 50.0],
        help="Outlier factor(s) for cell-wise numeric outliers (TabPFN-style). 0 disables.",
    )
    ap.add_argument(
        "--outlier_p_cell",
        type=float,
        default=0.02,
        help="Per-cell probability for cell-wise outliers (TabPFN-style).",
    )
    ap.add_argument(
        "--outliers_apply_to",
        type=str,
        default="test",
        choices=["train", "test", "both"],
        help="Apply cell-wise outliers to train, test, or both (default: test).",
    )
    ap.add_argument(
        "--label_poison_fracs",
        type=float,
        nargs="*",
        default=[0.1],
        help="Fractions of training labels to flip (ICL noisy demonstration labels).",
    )
    lg = ap.add_mutually_exclusive_group()
    lg.add_argument(
        "--enable_label_noise",
        dest="enable_label_noise",
        action="store_true",
        help="Enable label-noise / noisy-demonstration-label cases (default).",
    )
    lg.add_argument(
        "--disable_label_noise",
        dest="enable_label_noise",
        action="store_false",
        help="Disable label-noise / noisy-demonstration-label cases.",
    )
    ap.set_defaults(enable_label_noise=True)

    ap.add_argument(
        "--enable_rotation",
        action="store_true",
        help="Enable rotation invariance stress test on a numeric subspace (train+test rotated together).",
    )
    ap.add_argument(
        "--rotation_k",
        type=int,
        default=8,
        help="Rotation subspace dimension (k) when --enable_rotation is set.",
    )

    # Geometry + attention hooks (for unified behaviour/geometry tables)
    ap.add_argument(
        "--skip_geometry",
        action="store_true",
        help="Skip TFrow collapse metrics (faster).",
    )
    ap.add_argument(
        "--geom_max_rows_spectrum",
        type=int,
        default=2048,
        help="Max #test rows used for covariance-spectrum collapse metrics (stable subsample).",
    )
    ap.add_argument(
        "--geom_max_rows_cosine",
        type=int,
        default=512,
        help="Max #test rows used for pairwise-cosine metrics (stable subsample).",
    )

    ap.add_argument(
        "--skip_attention",
        action="store_true",
        help="Skip TFicl attention behaviour metrics (N_eff; much faster).",
    )
    ap.add_argument(
        "--attn_max_test_rows",
        type=int,
        default=1024,
        help="Max #test rows used for attention behaviour metrics (stable subsample).",
    )
    ap.add_argument(
        "--attn_chunk_test",
        type=int,
        default=1024,
        help="Chunk size over test queries for attention-weight computation (lower = less VRAM).",
    )
    ap.add_argument(
        "--no_attn_fp16",
        dest="attn_use_fp16",
        action="store_false",
        help="Disable fp16 matmuls inside attention-weight computation.",
    )
    ap.set_defaults(attn_use_fp16=True)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.openml_cache_dir:
        cache_dir = str(Path(args.openml_cache_dir).expanduser().resolve())
        os.environ["OPENML_CACHE_DIR"] = cache_dir
        try:
            import openml  # lazy import only when needed

            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            openml.config.set_root_cache_directory(cache_dir)
            print(f"[info] OpenML cache dir: {cache_dir}")
        except Exception as e:
            print(f"[warn] Failed to set OpenML cache dir to '{cache_dir}': {e}")

    # --- Run directory + run state
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
        if not run_dir.is_absolute():
            run_dir = (REPO_ROOT / run_dir).resolve()
    else:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = args.run_name or ts
        run_dir = (REPO_ROOT / "results" / "robustness" / name).resolve()
    ensure_dir(run_dir)

    save_json(run_dir / "env.json", get_env_info(device))
    save_json(run_dir / "run_config.json", vars(args))

    ckpts = parse_checkpoints(args.checkpoints, args.checkpoint_names)
    save_json(run_dir / "checkpoints.json", [c.__dict__ for c in ckpts])

    # Choose dataset list
    if args.openml_ids is not None and len(args.openml_ids) > 0:
        openml_ids = list(args.openml_ids)
    else:
        openml_ids = list(DEFAULT_OPENML_IDS)

    # Determine seeds to run
    if args.seeds:
        seed_values = list(dict.fromkeys(int(s) for s in args.seeds))
    else:
        base = int(args.seed)
        seed_values = [base + i for i in range(int(args.n_seeds))]

    metrics_path = run_dir / "metrics.csv"
    metrics_cols = [
        # checkpoint
        "checkpoint_name",
        "checkpoint_path",
        "checkpoint_sha256",
        "variant",
        "checkpoint",
        # dataset
        "dataset",
        "dataset_key",
        "dataset_name",
        "resolved_dataset_id",
        "resolved_task_id",
        "dataset_version",
        "input_spec",
        # split
        "seed",
        "test_size",
        "stratified",
        "n_train",
        "n_test",
        # condition/case
        "case_id",
        "case",
        "severity",
        "scope",
        "condition",
        "param_type",
        "param_value",
        "applies_to",
        "p_cell",
        "rotation_seed",
        "rotation_cols_hash",
        # metrics
        "acc",
        "f1_macro",
        "nll",
        "ece",
        "brier",
        "collapse_top1",
        "d_eff",
        "cos_mean",
        "cos_p95",
        "neff_mean",
        "neff_std",
        "purity_top1",
        "purity_top5",
        # settings
        "n_estimators",
        "batch_size",
        "use_hierarchical",
        "device",
    ]

    dataset_rows: List[Dict[str, Any]] = []
    splits_arrays: Dict[str, Any] = {}
    splits_meta: List[Dict[str, Any]] = []

    for ds_id in openml_ids:
        try:
            X, y, info = fetch_openml_dataset_with_info(int(ds_id))
        except Exception as e:
            print(f"[WARN] Could not load OpenML dataset '{ds_id}': {e}. Skipping.")
            continue

        ds_key = dataset_key(info)
        dataset_rows.append(
            {
                "dataset_key": ds_key,
                "dataset_name": info.dataset_name,
                "input_spec": info.input_spec,
                "resolved_dataset_id": info.resolved_dataset_id,
                "resolved_task_id": info.resolved_task_id,
                "dataset_version": info.dataset_version,
                "n_rows": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_classes": int(pd.Series(y).nunique(dropna=True)),
            }
        )

        for seed in seed_values:
            set_seed(seed)
            print(f"\n=== Dataset: {info.dataset_name} | seed={seed} ===")

            split = make_split_indices(y, seed, test_size=float(args.test_size), stratify=not bool(args.no_stratify))
            idx_tr, idx_te = split.idx_train, split.idx_test

            splits_arrays[f"{ds_key}__seed_{seed}__train_idx"] = idx_tr.astype(np.int32)
            splits_arrays[f"{ds_key}__seed_{seed}__test_idx"] = idx_te.astype(np.int32)
            splits_meta.append(
                {
                    "dataset_key": ds_key,
                    "seed": int(seed),
                    "test_size": float(args.test_size),
                    "stratified": bool(split.stratified),
                    "n_train": int(idx_tr.size),
                    "n_test": int(idx_te.size),
                }
            )

            X_tr = X.iloc[idx_tr].reset_index(drop=True)
            y_tr = y.iloc[idx_tr].reset_index(drop=True)
            X_te = X.iloc[idx_te].reset_index(drop=True)
            y_te = y.iloc[idx_te].reset_index(drop=True)

            case_rng = _stable_case_rng(ds_key, seed)
            cases = build_cases(
                X_train=X_tr,
                X_test=X_te,
                y_train=y_tr,
                rng=case_rng,
                uninformative_ns=args.uninformative_ns,
                outlier_factors=args.outlier_factors,
                outlier_p_cell=float(args.outlier_p_cell),
                outliers_apply_to=args.outliers_apply_to,
                label_poison_fracs=(args.label_poison_fracs if bool(args.enable_label_noise) else []),
                enable_rotation=bool(args.enable_rotation),
                rotation_k=int(args.rotation_k),
            )
            save_corruptions(run_dir, ds_key, seed, cases)
            expected_rows_per_ds_seed = len(cases) * len(ckpts)

            rows_out: List[Dict[str, Any]] = []
            subsample_seed = _subsample_seed_for_dataset(ds_key, seed)
            for ck in ckpts:
                print(f"  [checkpoint] {ck.name}")

                def _fit_clf(X_train: pd.DataFrame, y_train: pd.Series) -> TabICLClassifier:
                    clf = TabICLClassifier(
                        device=device,
                        model_path=str(ck.path),
                        allow_auto_download=False,
                        use_hierarchical=bool(args.use_hierarchical),
                        n_estimators=int(args.n_estimators),
                        batch_size=int(args.batch_size),
                        random_state=int(seed),
                        verbose=False,
                    )
                    clf.fit(X_train, y_train)
                    return clf

                # Important for VRAM: never keep 2 models on GPU at once.
                # We evaluate "no-refit" cases (clean + test-only shifts) using a single clean-fit model,
                # then delete it before running refit-requiring cases.
                cases_no_refit: List[Case] = []
                cases_refit: List[Case] = []
                for case in cases:
                    kind = case.artifacts.get("kind")
                    if case.condition == "clean":
                        cases_no_refit.append(case)
                    elif kind == "cell_outliers_tabpfn" and case.artifacts.get("train") is None:
                        # test-only outliers: reuse clean-context model
                        cases_no_refit.append(case)
                    else:
                        cases_refit.append(case)

                clf_clean: TabICLClassifier | None = None
                try:
                    maybe_free_torch_memory()
                    clf_clean = _fit_clf(X_tr, y_tr)

                    for case in cases_no_refit:
                        cid = case_id(case)
                        case_label = infer_case_label(case)
                        scope = infer_scope(case, refit=False)
                        X_te_c = X_te
                        p_cell: float | None = None
                        rotation_seed: int | None = None
                        rotation_cols_hash: str | None = None

                        kind = case.artifacts.get("kind")
                        if case.condition == "clean":
                            pass
                        elif kind == "cell_outliers_tabpfn":
                            p_cell = float(case.artifacts.get("p_cell", float("nan")))
                            test_art = case.artifacts.get("test")
                            if test_art is not None:
                                X_te_c = apply_outlier_artifacts(X_te_c, test_art)
                        else:
                            raise RuntimeError(f"Unexpected no-refit case kind: {kind} ({case.condition})")

                        try:
                            m = _eval_metrics_from_fitted_clf(clf_clean, X_te_c, y_te)
                        except Exception as e:
                            print(
                                f"[WARN] Failed case {case.condition} ({cid}) for {ck.name} on {info.dataset_name}: {e}"
                            )
                            maybe_free_torch_memory()
                            continue

                        geom: Dict[str, float] = {"collapse_top1": float("nan"), "d_eff": float("nan"), "cos_mean": float("nan"), "cos_p95": float("nan")}
                        attn: Dict[str, float] = {
                            "neff_mean": float("nan"),
                            "neff_std": float("nan"),
                            "purity_top1": float("nan"),
                            "purity_top5": float("nan"),
                        }
                        if not bool(args.skip_geometry):
                            try:
                                geom = compute_tfrow_collapse_metrics(
                                    clf_clean,
                                    X_te_c,
                                    max_rows_spectrum=int(args.geom_max_rows_spectrum),
                                    max_rows_cosine=int(args.geom_max_rows_cosine),
                                    subsample_seed=int(subsample_seed),
                                )
                            except Exception as e:
                                print(f"[WARN] Geometry metrics failed for {ck.name} case={case_label} ({cid}): {e}")
                                maybe_free_torch_memory()
                        if not bool(args.skip_attention):
                            try:
                                attn = compute_tficl_attention_metrics(
                                    clf_clean,
                                    X_te_c,
                                    y_te,
                                    max_test_rows=int(args.attn_max_test_rows),
                                    chunk_test=int(args.attn_chunk_test),
                                    use_fp16=bool(args.attn_use_fp16),
                                    subsample_seed=int(subsample_seed),
                                )
                            except Exception as e:
                                print(f"[WARN] Attention metrics failed for {ck.name} case={case_label} ({cid}): {e}")
                                maybe_free_torch_memory()

                        rows_out.append(
                            {
                                "checkpoint_name": ck.name,
                                "checkpoint_path": ck.path,
                                "checkpoint_sha256": ck.sha256,
                                "variant": infer_variant(ck.name),
                                "checkpoint": ck.name,
                                "dataset": ds_key,
                                "dataset_key": ds_key,
                                "dataset_name": info.dataset_name,
                                "resolved_dataset_id": info.resolved_dataset_id,
                                "resolved_task_id": info.resolved_task_id,
                                "dataset_version": info.dataset_version,
                                "input_spec": info.input_spec,
                                "seed": int(seed),
                                "test_size": float(args.test_size),
                                "stratified": bool(split.stratified),
                                "n_train": int(len(X_tr)),
                                "n_test": int(len(X_te_c)),
                                "case_id": cid,
                                "case": case_label,
                                "severity": float(case.param_value),
                                "scope": scope,
                                "condition": case.condition,
                                "param_type": case.param_type,
                                "param_value": float(case.param_value),
                                "applies_to": case.applies_to,
                                "p_cell": p_cell,
                                "rotation_seed": rotation_seed,
                                "rotation_cols_hash": rotation_cols_hash,
                                "acc": m.accuracy,
                                "f1_macro": m.f1_macro,
                                "nll": m.log_loss,
                                "ece": m.ece,
                                "brier": m.brier,
                                "collapse_top1": geom["collapse_top1"],
                                "d_eff": geom["d_eff"],
                                "cos_mean": geom["cos_mean"],
                                "cos_p95": geom["cos_p95"],
                                "neff_mean": attn["neff_mean"],
                                "neff_std": attn["neff_std"],
                                "purity_top1": attn["purity_top1"],
                                "purity_top5": attn["purity_top5"],
                                "n_estimators": int(args.n_estimators),
                                "batch_size": int(args.batch_size),
                                "use_hierarchical": bool(args.use_hierarchical),
                                "device": device,
                            }
                        )
                finally:
                    # Free clean model before any refit cases to avoid peak VRAM doubling.
                    if clf_clean is not None:
                        try:
                            del clf_clean
                        except Exception:
                            pass
                    maybe_free_torch_memory()

                for case in cases_refit:
                    cid = case_id(case)
                    case_label = infer_case_label(case)
                    scope = infer_scope(case, refit=True)
                    X_tr_c, y_tr_c = X_tr, y_tr
                    X_te_c = X_te
                    p_cell: float | None = None
                    rotation_seed: int | None = None
                    rotation_cols_hash: str | None = None

                    kind = case.artifacts.get("kind")
                    if kind == "uninformative_features_shuffled":
                        X_tr_c, X_te_c = apply_uninformative_artifacts(X_tr, X_te, case.artifacts)
                    elif kind == "cell_outliers_tabpfn":
                        p_cell = float(case.artifacts.get("p_cell", float("nan")))
                        train_art = case.artifacts.get("train")
                        test_art = case.artifacts.get("test")
                        if train_art is not None:
                            X_tr_c = apply_outlier_artifacts(X_tr_c, train_art)
                        if test_art is not None:
                            X_te_c = apply_outlier_artifacts(X_te_c, test_art)
                    elif kind == "icl_noisy_demo_labels":
                        y_tr_c = apply_label_flip_artifacts(y_tr, case.artifacts)
                    elif kind == "feature_rotation_kdim_both":
                        rotation_seed = int(case.artifacts["rotation_seed"])
                        rotation_cols_hash = str(case.artifacts["rotation_cols_hash"])
                        cols = list(case.artifacts["rotation_cols"])
                        mu = np.asarray(case.artifacts["mu"], dtype=float)
                        sigma = np.asarray(case.artifacts["sigma"], dtype=float)
                        R = np.asarray(case.artifacts["R"], dtype=float)
                        X_tr_c = _apply_kdim_rotation_matrix(X_tr_c, cols, mu, sigma, R)
                        X_te_c = _apply_kdim_rotation_matrix(X_te_c, cols, mu, sigma, R)
                    else:
                        raise RuntimeError(f"Unknown case kind: {kind} (condition={case.condition})")

                    clf_case: TabICLClassifier | None = None
                    geom = {"collapse_top1": float("nan"), "d_eff": float("nan"), "cos_mean": float("nan"), "cos_p95": float("nan")}
                    attn = {
                        "neff_mean": float("nan"),
                        "neff_std": float("nan"),
                        "purity_top1": float("nan"),
                        "purity_top5": float("nan"),
                    }
                    try:
                        maybe_free_torch_memory()
                        clf_case = _fit_clf(X_tr_c, y_tr_c)
                        m = _eval_metrics_from_fitted_clf(clf_case, X_te_c, y_te)

                        if not bool(args.skip_geometry):
                            try:
                                geom = compute_tfrow_collapse_metrics(
                                    clf_case,
                                    X_te_c,
                                    max_rows_spectrum=int(args.geom_max_rows_spectrum),
                                    max_rows_cosine=int(args.geom_max_rows_cosine),
                                    subsample_seed=int(subsample_seed),
                                )
                            except Exception as e:
                                print(f"[WARN] Geometry metrics failed for {ck.name} case={case_label} ({cid}): {e}")
                                maybe_free_torch_memory()

                        if not bool(args.skip_attention):
                            try:
                                attn = compute_tficl_attention_metrics(
                                    clf_case,
                                    X_te_c,
                                    y_te,
                                    max_test_rows=int(args.attn_max_test_rows),
                                    chunk_test=int(args.attn_chunk_test),
                                    use_fp16=bool(args.attn_use_fp16),
                                    subsample_seed=int(subsample_seed),
                                )
                            except Exception as e:
                                print(f"[WARN] Attention metrics failed for {ck.name} case={case_label} ({cid}): {e}")
                                maybe_free_torch_memory()
                    except Exception as e:
                        print(
                            f"[WARN] Failed case {case.condition} ({cid}) for {ck.name} on {info.dataset_name}: {e}"
                        )
                        maybe_free_torch_memory()
                        continue
                    finally:
                        if clf_case is not None:
                            try:
                                del clf_case
                            except Exception:
                                pass
                        maybe_free_torch_memory()

                    rows_out.append(
                        {
                            "checkpoint_name": ck.name,
                            "checkpoint_path": ck.path,
                            "checkpoint_sha256": ck.sha256,
                            "variant": infer_variant(ck.name),
                            "checkpoint": ck.name,
                            "dataset": ds_key,
                            "dataset_key": ds_key,
                            "dataset_name": info.dataset_name,
                            "resolved_dataset_id": info.resolved_dataset_id,
                            "resolved_task_id": info.resolved_task_id,
                            "dataset_version": info.dataset_version,
                            "input_spec": info.input_spec,
                            "seed": int(seed),
                            "test_size": float(args.test_size),
                            "stratified": bool(split.stratified),
                            "n_train": int(len(X_tr_c)),
                            "n_test": int(len(X_te_c)),
                            "case_id": cid,
                            "case": case_label,
                            "severity": float(case.param_value),
                            "scope": scope,
                            "condition": case.condition,
                            "param_type": case.param_type,
                            "param_value": float(case.param_value),
                            "applies_to": case.applies_to,
                            "p_cell": p_cell,
                            "rotation_seed": rotation_seed,
                            "rotation_cols_hash": rotation_cols_hash,
                            "acc": m.accuracy,
                            "f1_macro": m.f1_macro,
                            "nll": m.log_loss,
                            "ece": m.ece,
                            "brier": m.brier,
                            "collapse_top1": geom["collapse_top1"],
                            "d_eff": geom["d_eff"],
                            "cos_mean": geom["cos_mean"],
                            "cos_p95": geom["cos_p95"],
                            "neff_mean": attn["neff_mean"],
                            "neff_std": attn["neff_std"],
                            "purity_top1": attn["purity_top1"],
                            "purity_top5": attn["purity_top5"],
                            "n_estimators": int(args.n_estimators),
                            "batch_size": int(args.batch_size),
                            "use_hierarchical": bool(args.use_hierarchical),
                            "device": device,
                        }
                    )

            _append_rows_csv(rows_out, metrics_path, metrics_cols)
            print(f"  Wrote {len(rows_out)} rows to {metrics_path}")
            if len(rows_out) != expected_rows_per_ds_seed:
                print(
                    f"[warn] Row count mismatch for {info.dataset_name} seed={seed}: "
                    f"expected {expected_rows_per_ds_seed}, got {len(rows_out)}"
                )

    # Persist dataset list and splits
    if dataset_rows:
        (run_dir / "datasets.csv").parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dataset_rows).to_csv(run_dir / "datasets.csv", index=False)
    if splits_arrays:
        np.savez_compressed(run_dir / "splits.npz", **splits_arrays)
        save_json(run_dir / "splits_meta.json", splits_meta)


if __name__ == "__main__":
    main()
