#!/usr/bin/env python3
"""
Model-space diagnostics: C2ST on TabICL representations / attention.

Goal: check whether CC18 tasks look different from synthetic prior tasks in the
representation and attention space that the trained TabICL model actually uses.

For each dataset (pretrain or CC18), this script:
  1) Builds a single in-context "episode" using TabICLClassifier's ensemble machinery.
  2) Extracts row representations after TFrow and conditional ICL embedding.
  3) Computes a compact signature vector with:
       - row-norm stats,
       - mean pairwise cosine between rows (approximate),
       - attention sharpness: N_eff, entropy, top-1 mass,
       - neighbour purity: top-1 and top-5 label purity.
  4) Stores signatures to CSV for priors and CC18.
  5) Runs a logistic-regression C2ST on the signatures (pretrain vs CC18).

Usage (example, from repo root):
  cd tabicl-lfs
  python scripts/model_space_diagnostics.py \\
    --checkpoint checkpoints_mini_tabicl_stage2_pdl_posterior_avg/step-1000.ckpt \\
    --device cuda \\
    --stage2_prior_dir /dss/lxclscratch/.../mini_tabicl_s2_priors \\
    --stage2_max_batches 50 \\
    --max_pretrain_datasets 1000
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import openml
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Make local src importable
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tabicl.sklearn.classifier import TabICLClassifier  # type: ignore
from tabicl.prior.genload import LoadPriorDataset  # type: ignore
from tabicl.model.learning import ICLearning  # type: ignore
from tabicl.model.attention import compute_elliptical_diag  # type: ignore


@torch.no_grad()
def _compute_test_to_train_weights(
    model,
    R_cond: torch.Tensor,
    train_size: int,
    chunk_test: int = 4096,
    use_fp16: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Mean-head attention weights for TFicl last block, restricted to test→train.

    Returns
    -------
    tuple
        (W_tt, M_diag) where:
        - W_tt: Weights of shape (T_test, T_train), averaged over batch and heads.
        - M_diag: Elliptical diagonal metric of shape (nh, Dh) if EA is active, else None.
    """

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

    m_diag: Optional[torch.Tensor] = None
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
        m_diag = m  # (nh, Dh)
        m_bc = m.view(1, 1, nh, 1, hs)
        q = q * m_bc
        k = k * m_bc

    q_test = q[..., train_size:, :]
    k_train = k[..., :train_size, :]
    T_test = q_test.shape[-2]
    T_train = k_train.shape[-2]

    out = torch.zeros(T_test, T_train, device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(hs)

    Bnh = B * nh
    q_test_b = q_test.reshape(Bnh, T_test, hs)
    k_train_b = k_train.reshape(Bnh, T_train, hs)
    if use_fp16:
        q_test_b = q_test_b.to(torch.float16)
        k_train_b = k_train_b.to(torch.float16)

    for start in range(0, T_test, chunk_test):
        end = min(T_test, start + chunk_test)
        t = end - start
        q_chunk = q_test_b[:, start:end, :]
        scores = torch.bmm(q_chunk, k_train_b.transpose(1, 2)) * scale
        w = torch.softmax(scores.to(torch.float32), dim=-1)
        w_mean = w.mean(dim=0)
        out[start:end, :] += w_mean

    return out, m_diag


def per_row_neff(W_tt: torch.Tensor) -> np.ndarray:
    """Per-test-row N_eff from attention weights (T_test, T_train)."""
    if W_tt.ndim != 2 or W_tt.numel() == 0:
        return np.full(0, np.nan)
    finite = torch.isfinite(W_tt)
    any_finite = finite.any(dim=1)
    if not any_finite.any():
        return np.full(0, np.nan)
    W = W_tt[any_finite]
    finite = finite[any_finite]
    W_clean = torch.where(finite, W, torch.zeros_like(W))
    row_sums = W_clean.sum(dim=1, keepdim=True).clamp_min(1e-12)
    alpha = W_clean / row_sums
    s2 = (alpha**2).sum(dim=1)
    mask = (s2 > 0) & torch.isfinite(s2)
    neff = torch.empty_like(s2)
    neff[mask] = 1.0 / s2[mask]
    neff[~mask] = float("nan")
    return neff.cpu().numpy()


def per_row_entropy_and_top1(W_tt: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-test-row attention entropy and top-1 mass."""
    if W_tt.ndim != 2 or W_tt.numel() == 0:
        return np.full(0, np.nan), np.full(0, np.nan)
    W = W_tt.detach().cpu().numpy()
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    alpha = W / row_sums
    alpha = np.clip(alpha, 1e-12, 1.0)
    ent = -np.sum(alpha * np.log(alpha), axis=1)
    top1 = alpha.max(axis=1)
    return ent, top1


def per_row_purity(
    W_tt: torch.Tensor,
    y_train: np.ndarray,
    y_test: np.ndarray,
    topk: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-test-row top-1 hit and top-k purity from (T_test, T_train) weights."""
    if W_tt.ndim != 2 or W_tt.numel() == 0:
        return np.full(0, np.nan), np.full(0, np.nan)

    W = W_tt.detach().cpu().numpy()
    T_test, T_train = W.shape
    assert y_train.shape[0] == T_train
    assert y_test.shape[0] == T_test

    k = max(1, min(topk, T_train))
    top1 = np.full(T_test, np.nan, dtype=float)
    purity = np.full(T_test, np.nan, dtype=float)

    for t_idx in range(T_test):
        alpha = W[t_idx]
        if not np.isfinite(alpha).any():
            continue
        top_idx = np.argsort(-alpha)[:k]
        neigh_labels = y_train[top_idx]
        qlab = y_test[t_idx]
        top1[t_idx] = float(neigh_labels[0] == qlab)
        purity[t_idx] = float((neigh_labels == qlab).mean())

    return top1, purity


def _row_geometry_stats(R: torch.Tensor, rng: np.random.Generator, max_rows: int = 512) -> Dict[str, float]:
    """Compute mean/std row norms and mean pairwise cosine (approx)."""
    if R.ndim != 3:
        return {"row_norm_mean": float("nan"), "row_norm_std": float("nan"), "row_pairwise_cos_mean": float("nan")}
    R2 = R[0]  # (T, E)
    T, E = R2.shape
    if T == 0:
        return {"row_norm_mean": float("nan"), "row_norm_std": float("nan"), "row_pairwise_cos_mean": float("nan")}

    norms = torch.linalg.norm(R2, dim=-1).cpu().numpy()
    norm_mean = float(np.mean(norms))
    norm_std = float(np.std(norms, ddof=1)) if T > 1 else float("nan")

    m = min(T, max_rows)
    idx = np.arange(T) if T <= m else rng.choice(T, size=m, replace=False)
    R_sub = R2[idx].cpu().numpy()
    nrm = np.linalg.norm(R_sub, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    R_norm = R_sub / nrm
    C = R_norm @ R_norm.T
    mask = ~np.eye(C.shape[0], dtype=bool)
    if mask.sum() == 0:
        cos_mean = float("nan")
    else:
        cos_mean = float(C[mask].mean())

    return {
        "row_norm_mean": norm_mean,
        "row_norm_std": norm_std,
        "row_pairwise_cos_mean": cos_mean,
    }


def _episode_signature(
    clf: TabICLClassifier,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Compute model-space signature for a single dataset (one episode)."""

    # Build ICL episode via EnsembleGenerator (train part is stored in clf)
    X_te_num = clf.X_encoder_.transform(X_te)
    data = clf.ensemble_generator_.transform(X_te_num)
    methods = list(data.keys())
    norm_method = methods[0]
    Xs, ys_shifted = data[norm_method]
    shuffle_patterns = clf.ensemble_generator_.feature_shuffle_patterns_[norm_method]

    vidx = 0
    for i, p in enumerate(shuffle_patterns):
        if list(p) == sorted(p):
            vidx = i
            break

    X_variant = Xs[vidx]
    y_train_shifted = ys_shifted[vidx]
    train_size = y_train_shifted.shape[0]

    model = clf.model_.to(clf.device_)
    model.eval()
    X_tensor = torch.from_numpy(X_variant).float().unsqueeze(0).to(clf.device_)
    with torch.no_grad():
        col_out = model.col_embedder(
            X_tensor,
            train_size=train_size,
            mgr_config=clf.inference_config_.COL_CONFIG,
        )
        row_reps = model.row_interactor(col_out, mgr_config=clf.inference_config_.ROW_CONFIG)
        yt = torch.as_tensor(y_train_shifted, device=clf.device_, dtype=torch.float32).unsqueeze(0)
        R_cond = row_reps.clone()
        R_cond[:, :train_size] = R_cond[:, :train_size] + model.icl_predictor.y_encoder(yt)

    # Row geometry
    sig = _row_geometry_stats(R_cond, rng=rng)
    sig["train_size"] = float(train_size)
    sig["n_rows_total"] = float(R_cond.shape[1])

    # Attention weights and sharpness (+ EA metric if active)
    W_tt, m_diag = _compute_test_to_train_weights(model, R_cond, train_size)
    neff = per_row_neff(W_tt)
    ent, top1_mass = per_row_entropy_and_top1(W_tt)

    y_tr_np = np.asarray(y_tr)
    y_te_np = np.asarray(y_te)
    top1_purity, topk_purity = per_row_purity(W_tt, y_tr_np, y_te_np, topk=5)

    def _safe_mean_std(arr: np.ndarray) -> Tuple[float, float]:
        if arr.size == 0:
            return float("nan"), float("nan")
        m = float(np.nanmean(arr))
        s = float(np.nanstd(arr, ddof=1)) if np.sum(np.isfinite(arr)) > 1 else float("nan")
        return m, s

    sig["attn_neff_mean"], sig["attn_neff_std"] = _safe_mean_std(neff)
    sig["attn_entropy_mean"], sig["attn_entropy_std"] = _safe_mean_std(ent)
    sig["attn_top1_mass_mean"], sig["attn_top1_mass_std"] = _safe_mean_std(top1_mass)
    sig["neigh_top1_purity_mean"], sig["neigh_top1_purity_std"] = _safe_mean_std(top1_purity)
    sig["neigh_top5_purity_mean"], sig["neigh_top5_purity_std"] = _safe_mean_std(topk_purity)

    # EA metric M statistics (if EA is active)
    if m_diag is not None:
        m_flat = m_diag.view(-1)
        m_flat = torch.clamp(m_flat, min=1e-12)
        log_m = torch.log(m_flat)
        m_log_mean = float(log_m.mean())
        m_log_std = float(log_m.std(unbiased=True)) if m_flat.numel() > 1 else float("nan")
        m_min = float(m_flat.min())
        m_max = float(m_flat.max())
    else:
        m_log_mean = float("nan")
        m_log_std = float("nan")
        m_min = float("nan")
        m_max = float("nan")

    sig["M_log_mean"] = m_log_mean
    sig["M_log_std"] = m_log_std
    sig["M_min"] = m_min
    sig["M_max"] = m_max

    return sig


def sample_cc18_signatures(
    checkpoint: str,
    device: str,
    n_estimators: int,
    max_features: int,
    max_classes: int,
    limit_tasks: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """Compute model signatures for CC18 datasets."""
    rng = np.random.default_rng(seed)

    suite = openml.study.get_suite(99)
    task_ids = list(suite.tasks)
    if limit_tasks is not None:
        task_ids = task_ids[:limit_tasks]

    rows: List[Dict[str, Any]] = []

    for i, tid in enumerate(task_ids, 1):
        print(f"[CC18] ({i}/{len(task_ids)}) Task {tid}")
        try:
            task = openml.tasks.get_task(tid)
            ds = openml.datasets.get_dataset(task.dataset_id)
            X, y, _, _ = ds.get_data(
                target=ds.default_target_attribute,
                dataset_format="dataframe",
            )
            if max_features is not None and X.shape[1] > max_features:
                print(f"  [SKIP] features {X.shape[1]} > max_features {max_features}")
                continue
            y_arr = np.asarray(y)
            n_cls = len(np.unique(y_arr))
            if max_classes is not None and n_cls > max_classes:
                print(f"  [SKIP] classes {n_cls} > max_classes {max_classes}")
                continue

            X_tr, X_te, y_tr, y_te = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=seed,
                stratify=y,
            )

            clf = TabICLClassifier(
                device=device,
                model_path=checkpoint,
                allow_auto_download=False,
                use_hierarchical=True,
                n_estimators=n_estimators,
                random_state=seed,
                verbose=False,
            )
            clf.fit(X_tr, y_tr)
            sig = _episode_signature(clf, X_tr, y_tr, X_te, y_te, rng=rng)
            sig.update(
                {
                    "source": "cc18",
                    "task_id": int(tid),
                    "dataset_id": int(ds.dataset_id),
                    "dataset_name": str(ds.name),
                    "n_rows": float(X.shape[0]),
                    "n_features": float(X.shape[1]),
                    "n_classes": float(n_cls),
                }
            )
            rows.append(sig)
        except Exception as e:
            print(f"  [WARN] Error on task {tid}: {e}")
            continue

    return pd.DataFrame.from_records(rows)


def sample_pretrain_signatures(
    checkpoint: str,
    device: str,
    n_estimators: int,
    prior_dir: str,
    max_batches: int,
    max_datasets: int,
    seed: int,
    label: str,
) -> pd.DataFrame:
    """Compute model signatures for synthetic prior tasks via TabICLClassifier."""
    rng = np.random.default_rng(seed)
    data_dir = Path(prior_dir)
    if not data_dir.is_dir():
        print(f"[WARN] Prior dir does not exist: {data_dir}")
        return pd.DataFrame()

    ds = LoadPriorDataset(
        data_dir=str(data_dir),
        batch_size=512,
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
        stem = p.stem
        try:
            idx_str = stem.split("batch_")[1]
            batch_indices.append(int(idx_str))
        except Exception:
            continue

    if not batch_indices:
        print(f"[WARN] No batch_*.pt files found in {data_dir}")
        return pd.DataFrame()

    batch_indices = sorted(batch_indices)
    k = min(max_batches, len(batch_indices))
    chosen = np.sort(rng.choice(batch_indices, size=k, replace=False))
    print(f"[INFO] Sampling pretrain signatures from {data_dir} ({label}), batches={k}")

    rows: List[Dict[str, Any]] = []
    total = 0

    for batch_idx in chosen:
        if total >= max_datasets:
            break
        try:
            X_b, y_b, d_b, seq_lens_b, train_sizes_b = ds.get_batch(idx=batch_idx, batch_size=None)
        except FileNotFoundError:
            print(f"[WARN] Missing batch file {batch_idx:06d} in {data_dir}")
            continue

        X_np = X_b.numpy()
        y_np = y_b.numpy()
        d_np = d_b.numpy()
        seq_np = seq_lens_b.numpy()
        train_np = train_sizes_b.numpy()

        B = X_np.shape[0]
        for i in range(B):
            if total >= max_datasets:
                break
            T_i = int(seq_np[i])
            H_i = int(d_np[i])
            train_size = int(train_np[i])
            if train_size <= 0 or train_size >= T_i:
                continue

            X_i = X_np[i, :T_i, :H_i]
            y_i = y_np[i, :T_i]

            # Build DataFrame view
            cols = [f"f{j}" for j in range(H_i)]
            X_df = pd.DataFrame(X_i, columns=cols)
            y_ser = pd.Series(y_i)

            X_tr = X_df.iloc[:train_size].reset_index(drop=True)
            X_te = X_df.iloc[train_size:].reset_index(drop=True)
            y_tr = y_ser.iloc[:train_size].reset_index(drop=True)
            y_te = y_ser.iloc[train_size:].reset_index(drop=True)

            if len(np.unique(y_tr)) < 2 or len(X_te) == 0:
                continue

            try:
                clf = TabICLClassifier(
                    device=device,
                    model_path=checkpoint,
                    allow_auto_download=False,
                    use_hierarchical=True,
                    n_estimators=n_estimators,
                    random_state=seed,
                    verbose=False,
                )
                clf.fit(X_tr, y_tr)
                sig = _episode_signature(clf, X_tr, y_tr, X_te, y_te, rng=rng)
                sig.update(
                    {
                        "source": label,
                        "prior_dir": str(data_dir),
                        "prior_batch_idx": int(batch_idx),
                        "prior_dataset_idx": int(i),
                        "n_rows": float(T_i),
                        "n_features": float(H_i),
                    }
                )
                rows.append(sig)
                total += 1
            except Exception as e:
                print(f"[WARN] Error fitting / signing prior dataset (batch={batch_idx}, idx={i}): {e}")
                continue

    return pd.DataFrame.from_records(rows)


def run_c2st_signatures(
    df_pre: pd.DataFrame,
    df_cc: pd.DataFrame,
    feature_cols: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """C2ST on model-space signatures (pretrain vs CC18)."""
    N_pre = len(df_pre)
    N_cc = len(df_cc)
    if N_pre == 0 or N_cc == 0:
        print("[WARN] Skipping model-space C2ST: empty pretrain or CC18 signatures.")
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []

    X_pre_full = df_pre[feature_cols].to_numpy(dtype=float)
    X_cc_full = df_cc[feature_cols].to_numpy(dtype=float)

    for b in range(n_bootstrap):
        m = min(N_cc, N_pre)
        idx_pre = rng.choice(N_pre, size=m, replace=False)
        idx_cc = rng.choice(N_cc, size=m, replace=False)

        X_pre = X_pre_full[idx_pre]
        X_cc = X_cc_full[idx_cc]
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

        cv = StratifiedKFold(n_splits=min(5, max(2, y_all.sum())), shuffle=True, random_state=seed + b)
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

        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        records.append(
            {
                "bootstrap_id": int(b),
                "auc_mean": mean_auc,
                "auc_std": std_auc,
                "n_train_pretrain": int(X_pre.shape[0]),
                "n_train_cc18": int(X_cc.shape[0]),
            }
        )

    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Model-space diagnostics: TabICL priors vs CC18.")
    ap.add_argument("--checkpoint", type=str, required=True, help="TabICL checkpoint path.")
    ap.add_argument("--device", type=str, default="cuda", help="Device: cuda|cpu.")
    ap.add_argument("--n_estimators", type=int, default=4, help="TabICL ensemble size for signatures.")

    # Priors
    ap.add_argument("--stage1_prior_dir", type=str, default=None)
    ap.add_argument("--stage2_prior_dir", type=str, default=None)
    ap.add_argument("--stage1_max_batches", type=int, default=0)
    ap.add_argument("--stage2_max_batches", type=int, default=50)
    ap.add_argument("--max_pretrain_datasets", type=int, default=1000)

    # CC18
    ap.add_argument("--cc18_max_features", type=int, default=500)
    ap.add_argument("--cc18_max_classes", type=int, default=10)
    ap.add_argument("--cc18_limit", type=int, default=None)

    # C2ST
    ap.add_argument("--c2st_bootstrap", type=int, default=50)

    # Output
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO_ROOT / "results"),
        help="Directory to store signature and C2ST CSV files.",
    )
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # 1) Pretrain signatures
    df_pre_list: List[pd.DataFrame] = []
    if args.stage1_prior_dir:
        df_s1 = sample_pretrain_signatures(
            checkpoint=args.checkpoint,
            device=args.device,
            n_estimators=args.n_estimators,
            prior_dir=args.stage1_prior_dir,
            max_batches=args.stage1_max_batches,
            max_datasets=args.max_pretrain_datasets // 2,
            seed=args.seed,
            label="stage1",
        )
        df_pre_list.append(df_s1)
    if args.stage2_prior_dir:
        df_s2 = sample_pretrain_signatures(
            checkpoint=args.checkpoint,
            device=args.device,
            n_estimators=args.n_estimators,
            prior_dir=args.stage2_prior_dir,
            max_batches=args.stage2_max_batches,
            max_datasets=args.max_pretrain_datasets // 2 if args.stage1_prior_dir else args.max_pretrain_datasets,
            seed=args.seed + 1,
            label="stage2",
        )
        df_pre_list.append(df_s2)

    if df_pre_list:
        df_pre = pd.concat(df_pre_list, ignore_index=True)
    else:
        df_pre = pd.DataFrame()

    pre_path = out_dir / "model_sig_pretrain.csv"
    df_pre.to_csv(pre_path, index=False)
    print(f"[INFO] Saved pretrain model signatures to {pre_path} (n={len(df_pre)})")

    # 2) CC18 signatures
    df_cc = sample_cc18_signatures(
        checkpoint=args.checkpoint,
        device=args.device,
        n_estimators=args.n_estimators,
        max_features=args.cc18_max_features,
        max_classes=args.cc18_max_classes,
        limit_tasks=args.cc18_limit,
        seed=args.seed + 2,
    )
    cc_path = out_dir / "model_sig_cc18.csv"
    df_cc.to_csv(cc_path, index=False)
    print(f"[INFO] Saved CC18 model signatures to {cc_path} (n={len(df_cc)})")

    # 3) C2ST in model space
    feature_cols = [
        c
        for c in df_pre.columns
        if c
        not in {
            "source",
            "prior_dir",
            "prior_batch_idx",
            "prior_dataset_idx",
            "n_rows",
            "n_features",
        }
    ]
    feature_cols = [c for c in feature_cols if c in df_cc.columns]
    if not feature_cols:
        print("[WARN] No overlapping signature features for model-space C2ST.")
        return

    df_c2st = run_c2st_signatures(
        df_pre=df_pre,
        df_cc=df_cc,
        feature_cols=feature_cols,
        n_bootstrap=args.c2st_bootstrap,
        seed=args.seed + 3,
    )
    c2st_path = out_dir / "model_sig_c2st.csv"
    df_c2st.to_csv(c2st_path, index=False)
    print(f"[INFO] Saved model-space C2ST results to {c2st_path}")
    if len(df_c2st):
        print(
            f"[INFO] Model-space C2ST AUC: mean={df_c2st['auc_mean'].mean():.3f}, "
            f"std={df_c2st['auc_mean'].std(ddof=1) if len(df_c2st)>1 else 0.0:.3f}"
        )


if __name__ == "__main__":
    main()
