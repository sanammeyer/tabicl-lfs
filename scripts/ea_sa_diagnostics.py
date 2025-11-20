#!/usr/bin/env python3
"""
EA vs SA diagnostics on a single dataset.

This script is meant to answer: *why* does EA behave differently from SA,
by comparing per-example predictions, attention geometry, and neighbour behaviour.

Given:
  - an SA checkpoint (standard attention)
  - an EA checkpoint (elliptical attention)

It:
  1. Fits both models on the same train/test split.
  2. For each test example, records:
       - y_true, p_SA(true), p_EA(true), predicted labels.
       - Group membership: SA-only correct, EA-only correct, both-correct, both-wrong.
  3. Reconstructs last TFicl attention for each model on a single ensemble variant and computes:
       - per-test-row N_eff (effective number of attended neighbours),
       - per-test-row neighbour label purity (top-1 / top-5).
  4. Summarises these quantities per group.

Usage example:
  python scripts/ea_sa_diagnostics.py --dataset 23 \
      --sa_checkpoint checkpoints_mini_tabicl_stage2_sa/step-1000.ckpt \
      --ea_checkpoint checkpoints_mini_tabicl_stage2_ea/step-1000.ckpt
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openml
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

# Make local src importable
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_DIR))

from tabicl.sklearn.classifier import TabICLClassifier  # type: ignore
from tabicl.model.attention import compute_elliptical_diag  # type: ignore
from tabicl.model.learning import ICLearning  # type: ignore


def set_ea_mode(model, row_mode: str = "ea", icl_mode: str = "ea") -> None:
    """Toggle EA usage separately in TFrow and TFicl blocks."""
    row_identity = row_mode.lower() == "id"
    icl_identity = icl_mode.lower() == "id"

    for blk in model.row_interactor.tf_row.blocks:
        blk.elliptical = not row_identity

    for blk in model.icl_predictor.tf_icl.blocks:
        blk.elliptical = not icl_identity


def fetch_openml_dataset(name_or_id: str | int) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Load an OpenML dataset by ID or (partial) name."""
    if isinstance(name_or_id, int) or (str(name_or_id).isdigit()):
        ds = openml.datasets.get_dataset(int(name_or_id))
        X, y, _, _ = ds.get_data(
            target=getattr(ds, "default_target_attribute", None),
            dataset_format="dataframe",
        )
        if y is None:
            raise ValueError(f"Dataset {ds.name} (id={ds.dataset_id}) has no target.")
        return X, y, ds.name

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


@torch.no_grad()
def _compute_rank_matrices(
    model,
    R_cond: torch.Tensor,
    train_size: int,
    ea_strength: float = 1.0,
) -> torch.Tensor:
    """Return mean-head attention weights for TFicl last block (T,T)."""

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
        # Strength knob: interpolate between identity (1) and full metric (m)
        if ea_strength != 1.0:
            m = 1.0 + ea_strength * (m - 1.0)
        m_bc = m.view(1, 1, nh, 1, hs)
        q = q * m_bc
        k = k * m_bc

    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hs)  # (B, nh, T, T)
    allowed = torch.zeros(T, T, dtype=torch.bool, device=device)
    if train_size > 0:
        allowed[:train_size, :train_size] = True
        allowed[train_size:, :train_size] = True
    scores = scores.masked_fill(~allowed.view(1, 1, T, T), float("-inf"))
    weights = torch.softmax(scores, dim=-1)  # (B, nh, T, T)

    # Average over batch and heads to get (T,T)
    while weights.dim() > 2:
        weights = weights.mean(dim=0)
    return weights  # (T, T)


def per_row_neff(weights_avg: torch.Tensor, train_size: int) -> np.ndarray:
    """Per-test-row N_eff from attention weights."""
    if train_size <= 0 or weights_avg.ndim != 2:
        return np.full(0, np.nan)
    T = weights_avg.shape[0]
    if T <= train_size:
        return np.full(0, np.nan)
    W = weights_avg[train_size:, :train_size].clone()  # (T_test, J)
    if W.numel() == 0:
        return np.full(0, np.nan)
    finite = torch.isfinite(W)
    any_finite = finite.any(dim=1)
    if not any_finite.any():
        return np.full(0, np.nan)
    W = W[any_finite]
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


def per_row_purity(
    weights_avg: torch.Tensor,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_size: int,
    topk: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-test-row top-1 hit and top-k purity."""
    if train_size <= 0 or weights_avg.ndim != 2:
        return np.full(0, np.nan), np.full(0, np.nan)
    T = weights_avg.shape[0]
    n_test = T - train_size
    if n_test <= 0:
        return np.full(0, np.nan), np.full(0, np.nan)

    W = weights_avg.detach().cpu().numpy()
    assert y_train.shape[0] == train_size
    assert y_test.shape[0] == n_test

    k = max(1, min(topk, train_size))
    top1 = np.full(n_test, np.nan, dtype=float)
    purity = np.full(n_test, np.nan, dtype=float)

    for t_idx in range(n_test):
        row_idx = train_size + t_idx
        alpha = W[row_idx, :train_size]
        if not np.isfinite(alpha).any():
            continue
        top_idx = np.argsort(-alpha)[:k]
        neigh_labels = y_train[top_idx]
        qlab = y_test[t_idx]
        top1[t_idx] = float(neigh_labels[0] == qlab)
        purity[t_idx] = float((neigh_labels == qlab).mean())

    return top1, purity


@dataclass
class GroupStats:
    n: int
    acc: float
    mean_ptrue_sa: float
    mean_ptrue_ea: float
    mean_neff_sa: float
    mean_neff_ea: float
    mean_purity_sa: float
    mean_purity_ea: float


def analyze_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
    sa_ckpt: Path,
    ea_ckpt: Path,
    device: str,
    n_estimators: int,
    seed: int,
    ea_strength: float,
) -> None:
    print(f"\n=== Diagnostics: {name} ===")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y,
    )

    def _fit(path: Path) -> TabICLClassifier:
        clf = TabICLClassifier(
            device=device,
            model_path=str(path),
            allow_auto_download=False,
            use_hierarchical=True,
            n_estimators=n_estimators,
            random_state=seed,
            verbose=False,
        )
        clf.fit(X_tr, y_tr)
        return clf

    clf_sa = _fit(sa_ckpt)
    clf_ea = _fit(ea_ckpt)

    # Per-example predictions
    proba_sa = clf_sa.predict_proba(X_te)
    proba_ea = clf_ea.predict_proba(X_te)

    y_true = np.asarray(y_te)
    y_sa = clf_sa.y_encoder_.inverse_transform(np.argmax(proba_sa, axis=1))
    y_ea = clf_ea.y_encoder_.inverse_transform(np.argmax(proba_ea, axis=1))

    correct_sa = (y_sa == y_true)
    correct_ea = (y_ea == y_true)

    ptrue_sa = np.array([proba_sa[i, np.where(clf_sa.classes_ == y_true[i])[0][0]] for i in range(len(y_true))])
    ptrue_ea = np.array([proba_ea[i, np.where(clf_ea.classes_ == y_true[i])[0][0]] for i in range(len(y_true))])

    # Build one ICL episode per model and derive per-test-row N_eff and neighbour purity
    def _episode_metrics(clf: TabICLClassifier, row_mode: str, icl_mode: str, ea_strength: float):
        X_te_num = clf.X_encoder_.transform(X_te)
        data = clf.ensemble_generator_.transform(X_te_num)
        methods = list(data.keys())
        norm_method = methods[0]
        Xs, ys_shifted = data[norm_method]
        shuffle_patterns = clf.ensemble_generator_.feature_shuffle_patterns_[norm_method]

        # Choose identity permutation if present, else first
        vidx = 0
        for i, p in enumerate(shuffle_patterns):
            if list(p) == sorted(p):
                vidx = i
                break

        X_variant = Xs[vidx]
        y_train_shifted = ys_shifted[vidx]
        train_size = y_train_shifted.shape[0]

        model = clf.model_.to(clf.device_)
        set_ea_mode(model, row_mode=row_mode, icl_mode=icl_mode)
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

        weights = _compute_rank_matrices(model, R_cond, train_size, ea_strength=ea_strength)  # (T,T)
        neff = per_row_neff(weights, train_size)

        # Mapping: first train_size rows -> X_tr, remaining -> X_te in order
        y_tr_np = np.asarray(y_tr)
        y_te_np = np.asarray(y_te)
        top1, pur = per_row_purity(weights, y_tr_np, y_te_np, train_size, topk=5)
        return neff, top1, pur

    # SA baseline: identity metric in both TFrow and TFicl (strength unused)
    neff_sa, top1_sa, pur_sa = _episode_metrics(clf_sa, row_mode="id", icl_mode="id", ea_strength=0.0)
    # EA full: EA in both TFrow and TFicl
    neff_ea_full, top1_ea_full, pur_ea_full = _episode_metrics(clf_ea, row_mode="ea", icl_mode="ea", ea_strength=ea_strength)
    # EA row-only: EA in TFrow, identity in TFicl
    neff_ea_row, top1_ea_row, pur_ea_row = _episode_metrics(clf_ea, row_mode="ea", icl_mode="id", ea_strength=ea_strength)
    # EA icl-only: identity in TFrow, EA in TFicl
    neff_ea_icl, top1_ea_icl, pur_ea_icl = _episode_metrics(clf_ea, row_mode="id", icl_mode="ea", ea_strength=ea_strength)

    # Groups: SA-only, EA-only, both-correct, both-wrong
    groups: Dict[str, np.ndarray] = {
        "SA_only": np.where(correct_sa & ~correct_ea)[0],
        "EA_only": np.where(correct_ea & ~correct_sa)[0],
        "both_correct": np.where(correct_sa & correct_ea)[0],
        "both_wrong": np.where(~correct_sa & ~correct_ea)[0],
    }

    def _group_stats(idx: np.ndarray) -> GroupStats:
        if idx.size == 0:
            return GroupStats(
                n=0,
                acc=float("nan"),
                mean_ptrue_sa=float("nan"),
                mean_ptrue_ea=float("nan"),
                mean_neff_sa=float("nan"),
                mean_neff_ea=float("nan"),
                mean_purity_sa=float("nan"),
                mean_purity_ea=float("nan"),
            )
        acc = float((correct_sa[idx] & correct_ea[idx]).mean())
        m_psa = float(ptrue_sa[idx].mean())
        m_pea = float(ptrue_ea[idx].mean())

        # N_eff / purity arrays are indexed over test rows;
        # idx already refers to test indices (0..n_test-1)
        m_neff_sa = float(np.nanmean(neff_sa[idx])) if neff_sa.size else float("nan")
        m_neff_ea = float(np.nanmean(neff_ea_full[idx])) if neff_ea_full.size else float("nan")
        m_pur_sa = float(np.nanmean(pur_sa[idx])) if pur_sa.size else float("nan")
        m_pur_ea = float(np.nanmean(pur_ea_full[idx])) if pur_ea_full.size else float("nan")

        return GroupStats(
            n=int(idx.size),
            acc=acc,
            mean_ptrue_sa=m_psa,
            mean_ptrue_ea=m_pea,
            mean_neff_sa=m_neff_sa,
            mean_neff_ea=m_neff_ea,
            mean_purity_sa=m_pur_sa,
            mean_purity_ea=m_pur_ea,
        )

    # Global metrics
    acc_sa = accuracy_score(y_true, y_sa)
    acc_ea = accuracy_score(y_true, y_ea)
    nll_sa = log_loss(y_true, proba_sa, labels=clf_sa.classes_)
    nll_ea = log_loss(y_true, proba_ea, labels=clf_ea.classes_)

    print(
        f"  Global: acc(SA)={acc_sa:.4f}, acc(EA)={acc_ea:.4f}, "
        f"NLL(SA)={nll_sa:.4f}, NLL(EA)={nll_ea:.4f}"
    )

    for name_g, idx in groups.items():
        g = _group_stats(idx)
        print(
            f"  [{name_g}] n={g.n:4d} | "
            f"mean p_true SA={g.mean_ptrue_sa:.4f}, EA={g.mean_ptrue_ea:.4f} | "
            f"mean N_eff SA={g.mean_neff_sa:.1f}, EA={g.mean_neff_ea:.1f} | "
            f"mean purity SA={g.mean_purity_sa:.3f}, EA={g.mean_purity_ea:.3f}"
        )

    # Summarise global N_eff / purity for each EA mode
    def _safe_mean(arr: np.ndarray) -> float:
        return float(np.nanmean(arr)) if arr.size else float("nan")

    print(
        f"  N_eff summary (mean over test rows): "
        f"SA={_safe_mean(neff_sa):.1f}, "
        f"EA-full={_safe_mean(neff_ea_full):.1f}, "
        f"EA-row={_safe_mean(neff_ea_row):.1f}, "
        f"EA-icl={_safe_mean(neff_ea_icl):.1f}"
    )
    print(
        f"  Purity summary (top-1 / top-5, mean over test rows): "
        f"SA={_safe_mean(top1_sa):.3f}/{_safe_mean(pur_sa):.3f}, "
        f"EA-full={_safe_mean(top1_ea_full):.3f}/{_safe_mean(pur_ea_full):.3f}, "
        f"EA-row={_safe_mean(top1_ea_row):.3f}/{_safe_mean(pur_ea_row):.3f}, "
        f"EA-icl={_safe_mean(top1_ea_icl):.3f}/{_safe_mean(pur_ea_icl):.3f}"
    )

    # Save diagnostics to CSV for later analysis
    out_path = ROOT / "results" / "ea_sa_diagnostics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_eff_sa_global = _safe_mean(neff_sa)
    n_eff_full_global = _safe_mean(neff_ea_full)
    n_eff_row_global = _safe_mean(neff_ea_row)
    n_eff_icl_global = _safe_mean(neff_ea_icl)
    pur_sa_global = (_safe_mean(top1_sa), _safe_mean(pur_sa))
    pur_full_global = (_safe_mean(top1_ea_full), _safe_mean(pur_ea_full))
    pur_row_global = (_safe_mean(top1_ea_row), _safe_mean(pur_ea_row))
    pur_icl_global = (_safe_mean(top1_ea_icl), _safe_mean(pur_ea_icl))

    for name_g, idx in groups.items():
        g = _group_stats(idx)
        rows.append(
            {
                "dataset": name,
                "group": name_g,
                "n": g.n,
                "acc_sa_global": acc_sa,
                "acc_ea_global": acc_ea,
                "nll_sa_global": nll_sa,
                "nll_ea_global": nll_ea,
                "mean_ptrue_sa": g.mean_ptrue_sa,
                "mean_ptrue_ea": g.mean_ptrue_ea,
                "mean_neff_sa": g.mean_neff_sa,
                "mean_neff_ea": g.mean_neff_ea,
                "mean_purity_sa": g.mean_purity_sa,
                "mean_purity_ea": g.mean_purity_ea,
                "n_eff_sa_global": n_eff_sa_global,
                "n_eff_ea_full_global": n_eff_full_global,
                "n_eff_ea_row_global": n_eff_row_global,
                "n_eff_ea_icl_global": n_eff_icl_global,
                "pur_sa_top1_global": pur_sa_global[0],
                "pur_sa_topk_global": pur_sa_global[1],
                "pur_ea_full_top1_global": pur_full_global[0],
                "pur_ea_full_topk_global": pur_full_global[1],
                "pur_ea_row_top1_global": pur_row_global[0],
                "pur_ea_row_topk_global": pur_row_global[1],
                "pur_ea_icl_top1_global": pur_icl_global[0],
                "pur_ea_icl_topk_global": pur_icl_global[1],
                "ea_strength": ea_strength,
                "n_estimators": n_estimators,
                "seed": seed,
                "sa_checkpoint": str(sa_ckpt),
                "ea_checkpoint": str(ea_ckpt),
            }
        )

    df_out = pd.DataFrame(rows)
    header = not out_path.exists()
    df_out.to_csv(out_path, mode="a", index=False, header=header)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EA vs SA diagnostics on one or more OpenML datasets.")
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single OpenML dataset id or name (e.g. 23 or 'cmc').",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of OpenML dataset ids/names (e.g. '23,48,750').",
    )
    ap.add_argument(
        "--sa_checkpoint",
        type=str,
        default=str(ROOT / "checkpoints_mini_tabicl_stage2_sa" / "step-1000.ckpt"),
        help="Stage-2 SA checkpoint path.",
    )
    ap.add_argument(
        "--ea_checkpoint",
        type=str,
        default=str(ROOT / "checkpoints_mini_tabicl_stage2_ea" / "step-1000.ckpt"),
        help="Stage-2 EA checkpoint path.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu).",
    )
    ap.add_argument(
        "--n_estimators",
        type=int,
        default=4,
        help="Number of ensemble estimators for TabICLClassifier.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    ap.add_argument(
        "--ea_strength",
        type=float,
        default=1.0,
        help="EA strength knob for diagnostics (1.0 = full EA, 0.0 = identity).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build list of dataset specs to process
    specs = []
    if args.datasets:
        specs = [s.strip() for s in args.datasets.split(",") if s.strip()]
    elif args.dataset:
        specs = [args.dataset.strip()]
    else:
        raise SystemExit("Please provide --dataset ID/NAME or --datasets id1,id2,...")

    for spec in specs:
        X, y, name = fetch_openml_dataset(spec)
        analyze_dataset(
            X=X,
            y=y,
            name=name,
            sa_ckpt=Path(args.sa_checkpoint),
            ea_ckpt=Path(args.ea_checkpoint),
            device=device,
            n_estimators=args.n_estimators,
            seed=args.seed,
            ea_strength=args.ea_strength,
        )


if __name__ == "__main__":
    main()
