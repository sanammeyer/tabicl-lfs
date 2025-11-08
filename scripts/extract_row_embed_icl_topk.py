#!/usr/bin/env python3
"""Extract TFrow feature-token embeddings and TFicl last-layer top-k attentions.

This script shows how to:
  1) Get row embeddings without using CLS tokens (i.e., TFrow outputs for feature tokens)
  2) For each test (query) row, find the top-k training rows it attends to in the
     last TFicl block, based on actual attention scores reconstructed from Q/K.

It uses a small dataset (Iris by default) via the sklearn interface to set up
preprocessing and a pretrained TabICL checkpoint.

Note on attention directions:
  With the in-context split-attention mask, test rows attend only to training rows,
  while training rows attend within the training slice. Therefore, the meaningful
  top-k neighbors (by attention) for a test row are among training rows.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Tuple

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Make local src importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tabicl.sklearn.classifier import TabICLClassifier
from tabicl.model.attention import compute_elliptical_diag


def _choose_identity_variant(patterns: List[List[int]]) -> int:
    for i, p in enumerate(patterns):
        if list(p) == sorted(p):
            return i
    return 0


@torch.no_grad()
def tfrow_feature_tokens(model, X_tensor: torch.Tensor, train_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run TFcol+TFrow and return per-row feature-token embeddings and mean-pooled vector.

    Returns
    -------
    feat_tokens : (T, H, E)
        TFrow final outputs at feature-token positions, excluding CLS tokens.

    feat_mean : (T, E)
        Mean over feature tokens per row (simple pooling to a single vector per row).
    """
    device = X_tensor.device
    model.eval()

    # Stage 1: TFcol
    emb = model.col_embedder(X_tensor, train_size=train_size)  # (1, T, H+C, E)

    # Stage 2: TFrow (manually, to retain token outputs)
    ri = model.row_interactor
    tfrow = ri.tf_row
    num_cls = ri.num_cls
    B, T = emb.shape[:2]
    cls = ri.cls_tokens.expand(B, T, num_cls, ri.embed_dim).to(device)
    x = emb.clone()
    x[:, :, :num_cls, :] = cls

    v_prev = None
    for idx, blk in enumerate(tfrow.blocks):
        x = blk(x, key_padding_mask=None, attn_mask=None, rope=tfrow.rope, v_prev=v_prev, block_index=idx)
        v_prev = getattr(blk, "_last_v", None)

    # Exclude CLS tokens
    feat_tokens = x[:, :, num_cls:, :].squeeze(0)  # (T, H, E)
    feat_mean = feat_tokens.mean(dim=1)  # (T, E)
    return feat_tokens, feat_mean


@torch.no_grad()
def last_icl_topk_train_neighbors(
    model,
    R_cond: torch.Tensor,
    train_size: int,
    topk: int = 5,
) -> Dict[int, List[Tuple[int, float]]]:
    """Compute, for each test row, the top-k training rows attended by the last TFicl block.

    Parameters
    ----------
    model : TabICL
        The loaded TabICL backbone.

    R_cond : Tensor (1, T, D)
        Row representations with label-conditioning already added to the first `train_size` rows.

    train_size : int
        Number of training rows at the beginning of the episode.

    topk : int, default=5
        Number of neighbors to return per test row.

    Returns
    -------
    dict
        Mapping test_row_idx -> list of (train_row_idx, attention_weight) sorted by weight desc.
    """
    device = R_cond.device
    enc = model.icl_predictor.tf_icl
    blocks = list(enc.blocks)

    # Run all but the last block to get v_prev for the final block
    x = R_cond
    v_prev = None
    for i, blk in enumerate(blocks[:-1]):
        x = blk(x, key_padding_mask=None, attn_mask=train_size, rope=enc.rope, v_prev=v_prev, block_index=i)
        v_prev = getattr(blk, "_last_v", None)

    # Prepare inputs for the last block exactly as in pre-norm/post-norm
    last = blocks[-1]
    if last.norm_first:
        q_in = last.norm1(x)
    else:
        q_in = x

    # Self-attention: k=v=q
    k_in = q_in
    v_in = q_in

    # Project to Q/K/V heads (packed projection identical to core path)
    q, k, v = F._in_projection_packed(q_in, k_in, v_in, last.attn.in_proj_weight, last.attn.in_proj_bias)

    B, T, E = q.shape
    nh = last.attn.num_heads
    hs = E // nh
    q = q.view(B, T, nh, hs).transpose(-3, -2)  # (B, nh, T, hs)
    k = k.view(B, T, nh, hs).transpose(-3, -2)  # (B, nh, T, hs)
    v = v.view(B, T, nh, hs).transpose(-3, -2)  # (B, nh, T, hs)

    # Elliptical scaling (if enabled and v_prev available); restrict estimator to train keys when split mask
    if last.elliptical and (v_prev is not None) and (len(blocks) - 1 >= 1):
        keep = torch.zeros(T, device=device, dtype=torch.float32)
        keep[:train_size] = 1.0
        m = compute_elliptical_diag(v, v_prev, delta=last.elliptical_delta, scale_mode=last.elliptical_scale_mode, mask_keep=keep)
        m_bc = m.view(1, 1, nh, 1, hs)
        q = q * m_bc
        k = k * m_bc

    # Raw attention scores: (B, nh, T, T)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hs)

    # Build split-mask: train<->train allowed; test->train allowed; others masked
    allowed = torch.zeros(T, T, dtype=torch.bool, device=device)
    if train_size > 0:
        allowed[:train_size, :train_size] = True
        allowed[train_size:, :train_size] = True
    scores = scores.masked_fill(~allowed.view(1, 1, T, T), float("-inf"))

    # Softmax over keys
    weights = torch.softmax(scores, dim=-1)  # (B, nh, T, T)
    weights_h = weights.squeeze(0)          # (nh, T, T)
    weights_avg = weights_h.mean(dim=0)     # (T, T)

    # Compute per-head value contribution magnitude ||W_out,h v_{j,h}|| for each training row j
    W = last.attn.out_proj.weight           # (E, E)
    nh = last.attn.num_heads
    hs = E // nh
    W_blocks = torch.stack([W[:, h * hs : (h + 1) * hs] for h in range(nh)], dim=0)  # (nh, E, hs)
    v_h_t = v.squeeze(0)                    # (nh, T, hs)
    v_h_j = v_h_t[:, :train_size, :]        # (nh, J, hs)
    contrib = torch.einsum('h e s, h j s -> h j e', W_blocks, v_h_j)  # (nh, J, E)
    vnorm = torch.linalg.vector_norm(contrib, dim=-1)  # (nh, J)

    # Vectorized scores for attn * vnorm across all test rows: (T_test, J)
    if train_size < T:
        alpha_all = weights_h[:, train_size:, :train_size]   # (nh, T_test, J)
        scores_axv_all = (alpha_all * vnorm[:, None, :]).mean(dim=0)  # (T_test, J)
    else:
        scores_axv_all = torch.empty(0, train_size, device=device)

    # Build outputs
    results_attn: Dict[int, List[Tuple[int, float]]] = {}
    results_axv: Dict[int, List[Tuple[int, float]]] = {}

    for i in range(train_size, T):
        # Attention-only
        row_attn = weights_avg[i, :train_size]
        if topk is not None and topk > 0:
            k_eff = min(topk, train_size)
            if k_eff == 0:
                results_attn[i] = []
            else:
                vals, idx = torch.topk(row_attn, k=k_eff, dim=-1)
                results_attn[i] = [(int(idx[j].item()), float(vals[j].item())) for j in range(k_eff)]
        else:
            idx = torch.argsort(row_attn, descending=True)
            results_attn[i] = [(int(j.item()), float(row_attn[j].item())) for j in idx]

        # Attention × value magnitude
        if train_size < T:
            row_axv = scores_axv_all[i - train_size]  # (J,)
            if topk is not None and topk > 0:
                k_eff = min(topk, train_size)
                if k_eff == 0:
                    results_axv[i] = []
                else:
                    vals, idx = torch.topk(row_axv, k=k_eff, dim=-1)
                    results_axv[i] = [(int(idx[j].item()), float(vals[j].item())) for j in range(k_eff)]
            else:
                idx = torch.argsort(row_axv, descending=True)
                results_axv[i] = [(int(j.item()), float(row_axv[j].item())) for j in idx]
        else:
            results_axv[i] = []

    return results_attn, results_axv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="iris", choices=["iris"], help="Small dataset to try")
    ap.add_argument("--model_path", type=str, default="checkpoints/tabicl-classifier-v1.1-0506.ckpt", help="Pretrained TabICL checkpoint")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--topk", type=int, default=5, help="Return top-k train rows per test row; 0=all sorted")
    args = ap.parse_args()

    # Load small dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Simple split for classifier fit/test
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Set up classifier for preprocessing + backbone loading
    clf = TabICLClassifier(
        n_estimators=1,
        norm_methods="none",
        feat_shuffle_method="none",
        class_shift=True,
        use_amp=True,
        batch_size=1,
        model_path=args.model_path,
        device=args.device,
        verbose=False,
    )
    clf.fit(X_tr, y_tr)

    # Build a single in-context episode using the fitted ensemble generator
    X_te_num = clf.X_encoder_.transform(X_te)
    data = clf.ensemble_generator_.transform(X_te_num)
    # Pick the only method/variant (identity)
    method = list(data.keys())[0]
    Xs, ys_shifted = data[method]
    patterns = clf.ensemble_generator_.feature_shuffle_patterns_[method]
    vidx = _choose_identity_variant(patterns)

    X_variant = Xs[vidx]  # (T, H)
    y_train_shifted = ys_shifted[vidx]  # (train_size,)
    train_size = y_train_shifted.shape[0]
    T, H = X_variant.shape

    device = torch.device(args.device)
    model = clf.model_.to(device)
    model.eval()

    X_tensor = torch.from_numpy(X_variant).float().unsqueeze(0).to(device)  # (1, T, H)

    # 1) TFrow feature-token embeddings (exclude CLS)
    feat_tokens, feat_mean = tfrow_feature_tokens(model, X_tensor, train_size=train_size)

    # 2) Row reps + label conditioning, then last TFicl attention top-k
    with torch.no_grad():
        # Pre-ICL row reps
        col_out = model.col_embedder(X_tensor, train_size=train_size, mgr_config=clf.inference_config_.COL_CONFIG)
        row_reps = model.row_interactor(col_out, mgr_config=clf.inference_config_.ROW_CONFIG)  # (1, T, C*E)
        # Condition on training labels (shifted ints)
        yt = torch.as_tensor(y_train_shifted, device=device, dtype=torch.float32).unsqueeze(0)  # (1, train_size)
        R_cond = row_reps.clone()
        R_cond[:, :train_size] = R_cond[:, :train_size] + model.icl_predictor.y_encoder(yt)

    topk_attn, topk_axv = last_icl_topk_train_neighbors(model, R_cond, train_size=train_size, topk=args.topk)

    # Report
    print("TFrow feature-token embeddings shape (T,H,E):", tuple(feat_tokens.shape))
    print("TFrow mean-pooled feature embeddings shape (T,E):", tuple(feat_mean.shape))
    print(f"Episode sizes: train={train_size}, test={T - train_size}, H={H}")
    hdr = f"Top-{args.topk} training neighbors" if args.topk and args.topk > 0 else "All training rows (sorted)"
    print(f"{hdr} by TFicl last-layer attention (per test row):")
    for i in range(train_size, T):
        pairs = topk_attn.get(i, [])
        pretty = ", ".join([f"(train_idx={j}, w={w:.4f})" for j, w in pairs])
        print(f"  test_idx={i} -> [ {pretty} ]")

    print()
    print(f"{hdr} by TFicl last-layer attention × ||W_out v|| (per test row):")
    for i in range(train_size, T):
        pairs = topk_axv.get(i, [])
        pretty = ", ".join([f"(train_idx={j}, s={w:.4f})" for j, w in pairs])
        print(f"  test_idx={i} -> [ {pretty} ]")


if __name__ == "__main__":
    main()
