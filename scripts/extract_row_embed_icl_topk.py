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
    avg_last2: bool = False,
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

    # Softmax over keys (last layer)
    weights = torch.softmax(scores, dim=-1)  # (B, nh, T, T)
    weights_h_last = weights.squeeze(0)      # (nh, T, T)
    weights_h = weights_h_last

    # Optionally average with penultimate layer attention
    if avg_last2 and len(blocks) >= 2:
        x2 = R_cond
        v_prev2 = None
        for i, blk in enumerate(blocks[:-2]):
            x2 = blk(x2, key_padding_mask=None, attn_mask=train_size, rope=enc.rope, v_prev=v_prev2, block_index=i)
            v_prev2 = getattr(blk, "_last_v", None)
        prev_blk = blocks[-2]
        q_in2 = prev_blk.norm1(x2) if prev_blk.norm_first else x2
        B2, T2, E2 = q_in2.shape
        nh2 = prev_blk.attn.num_heads
        hs2 = E2 // nh2
        q2, k2, v2 = F._in_projection_packed(q_in2, q_in2, q_in2, prev_blk.attn.in_proj_weight, prev_blk.attn.in_proj_bias)
        q2 = q2.view(B2, T2, nh2, hs2).transpose(-3, -2)
        k2 = k2.view(B2, T2, nh2, hs2).transpose(-3, -2)
        v2 = v2.view(B2, T2, nh2, hs2).transpose(-3, -2)
        if prev_blk.elliptical and (v_prev2 is not None) and (len(blocks) - 2 >= 1):
            keep2 = torch.zeros(T2, device=device, dtype=torch.float32)
            keep2[:train_size] = 1.0
            m2 = compute_elliptical_diag(v2, v_prev2, delta=prev_blk.elliptical_delta, scale_mode=prev_blk.elliptical_scale_mode, mask_keep=keep2)
            m2_bc = m2.view(1, 1, nh2, 1, hs2)
            q2 = q2 * m2_bc
            k2 = k2 * m2_bc
        scores2 = torch.matmul(q2, k2.transpose(-1, -2)) / math.sqrt(hs2)
        scores2 = scores2.masked_fill(~allowed.view(1, 1, T2, T2), float("-inf"))
        weights_prev = torch.softmax(scores2, dim=-1)
        weights_h_prev = weights_prev.squeeze(0)
        weights_h = 0.5 * (weights_h + weights_h_prev)

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
        alpha_all = weights_h_last[:, train_size:, :train_size]   # (nh, T_test, J)
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


@torch.no_grad()
def _compute_rank_matrices(
    model,
    R_cond: torch.Tensor,
    train_size: int,
    avg_last2: bool = False,
):
    """Return batched ranking matrices for all test rows.

    Returns
    -------
    weights_avg : Tensor (T, T)
        Mean-head attention of the selected layer (last or avg of last two).

    scores_axv_all : Tensor (T_test, J)
        Mean-head attention × ||W_out v|| from last layer, over (test, train).
    """
    device = R_cond.device
    enc = model.icl_predictor.tf_icl
    blocks = list(enc.blocks)

    # Run through N-1 blocks
    x = R_cond
    v_prev = None
    for i, blk in enumerate(blocks[:-1]):
        x = blk(x, key_padding_mask=None, attn_mask=train_size, rope=enc.rope, v_prev=v_prev, block_index=i)
        v_prev = getattr(blk, "_last_v", None)

    # Last block projections
    last = blocks[-1]
    # Use explicit branch to avoid Python truthiness on Tensors
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
        m = compute_elliptical_diag(v, v_prev, delta=last.elliptical_delta, scale_mode=last.elliptical_scale_mode, mask_keep=keep)
        m_bc = m.view(1, 1, nh, 1, hs)
        q = q * m_bc
        k = k * m_bc
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hs)
    allowed = torch.zeros(T, T, dtype=torch.bool, device=device)
    if train_size > 0:
        allowed[:train_size, :train_size] = True
        allowed[train_size:, :train_size] = True
    scores = scores.masked_fill(~allowed.view(1, 1, T, T), float("-inf"))
    weights_last = torch.softmax(scores, dim=-1).squeeze(0)  # (nh, T, T)

    # Optional average with N-2
    if avg_last2 and len(blocks) >= 2:
        x2 = R_cond
        v_prev2 = None
        for i, blk in enumerate(blocks[:-2]):
            x2 = blk(x2, key_padding_mask=None, attn_mask=train_size, rope=enc.rope, v_prev=v_prev2, block_index=i)
            v_prev2 = getattr(blk, "_last_v", None)
        prev_blk = blocks[-2]
        if prev_blk.norm_first:
            q_in2 = prev_blk.norm1(x2)
        else:
            q_in2 = x2
        B2, T2, E2 = q_in2.shape
        nh2 = prev_blk.attn.num_heads
        hs2 = E2 // nh2
        q2, k2, v2 = F._in_projection_packed(q_in2, q_in2, q_in2, prev_blk.attn.in_proj_weight, prev_blk.attn.in_proj_bias)
        q2 = q2.view(B2, T2, nh2, hs2).transpose(-3, -2)
        k2 = k2.view(B2, T2, nh2, hs2).transpose(-3, -2)
        if prev_blk.elliptical and (v_prev2 is not None) and (len(blocks) - 2 >= 1):
            keep2 = torch.zeros(T2, device=device, dtype=torch.float32)
            keep2[:train_size] = 1.0
            m2 = compute_elliptical_diag(v2, v_prev2, delta=prev_blk.elliptical_delta, scale_mode=prev_blk.elliptical_scale_mode, mask_keep=keep2)
            m2_bc = m2.view(1, 1, nh2, 1, hs2)
            q2 = q2 * m2_bc
            k2 = k2 * m2_bc
        scores2 = torch.matmul(q2, k2.transpose(-1, -2)) / math.sqrt(hs2)
        scores2 = scores2.masked_fill(~allowed.view(1, 1, T2, T2), float("-inf"))
        weights_prev = torch.softmax(scores2, dim=-1).squeeze(0)  # (nh, T, T)
        weights_sel = 0.5 * (weights_last + weights_prev)
    else:
        weights_sel = weights_last

    weights_avg = weights_sel.mean(dim=0)  # (T, T)

    # Build attention×value magnitude (from last layer)
    W = last.attn.out_proj.weight  # (E, E)
    W_blocks = torch.stack([W[:, h * hs : (h + 1) * hs] for h in range(nh)], dim=0)  # (nh, E, hs)
    v_h_t = v.squeeze(0)  # (nh, T, hs)
    v_h_j = v_h_t[:, :train_size, :]
    contrib = torch.einsum('h e s, h j s -> h j e', W_blocks, v_h_j)
    vnorm = torch.linalg.vector_norm(contrib, dim=-1)  # (nh, J)
    scores_axv_all = (weights_last[:, train_size:, :train_size] * vnorm[:, None, :]).mean(dim=0)  # (T_test, J)

    return weights_avg, scores_axv_all


@torch.no_grad()
def build_pairs_batched(
    feat_mean: torch.Tensor,
    train_size: int,
    attn_or_axv: str,
    model,
    R_cond: torch.Tensor,
    topk: int,
    avg_last2: bool,
    train_labels_int: np.ndarray,
    test_labels_int: np.ndarray,
):
    """Vectorized pair building for all test rows in one pass.

    Returns
    -------
    X_pairs : (T_test*K, 2E)
    y_pairs : (T_test*K,)
    idx_topk : (T_test, K)
    """
    device = feat_mean.device
    weights_avg, scores_axv_all = _compute_rank_matrices(model, R_cond, train_size, avg_last2=avg_last2)
    T = weights_avg.shape[0]
    J = train_size
    T_test = T - J
    if J <= 0 or T_test <= 0:
        return torch.empty(0, 2 * feat_mean.shape[-1], device=device), torch.empty(0, device=device), torch.empty(0, 0, dtype=torch.long)

    if attn_or_axv == "attn":
        mat = weights_avg[J:, :J]  # (T_test, J)
    else:
        mat = scores_axv_all  # (T_test, J)

    K = min(int(topk), J) if topk and topk > 0 else J
    vals, idx = torch.topk(mat, k=K, dim=-1)

    # Gather features
    test_feat = feat_mean[J:, :].unsqueeze(1).expand(T_test, K, -1)  # (T_test, K, E)
    anchor_feat = feat_mean[idx]  # (T_test, K, E)
    pair_feats = torch.cat([(test_feat - anchor_feat).abs(), test_feat * anchor_feat], dim=-1)  # (T_test, K, 2E)
    X_pairs = pair_feats.reshape(T_test * K, -1)

    # Labels (vectorized equality on integer labels)
    train_lab_t = torch.as_tensor(train_labels_int, device=device)
    test_lab_t = torch.as_tensor(test_labels_int, device=device)
    anchor_y = train_lab_t[idx]  # (T_test, K)
    test_y = test_lab_t.unsqueeze(1).expand(T_test, K)
    y_pairs = (anchor_y == test_y).to(torch.float32).reshape(-1)

    return X_pairs, y_pairs, idx


@torch.no_grad()
def pdlc_posterior_from_gamma(
    gamma: torch.Tensor,              # (T_test, K) in [0,1]
    idx_topk: torch.Tensor,           # (T_test, K) long indices into [0..J-1]
    train_labels_int: np.ndarray,     # (J,)
    num_classes: int,
    priors: torch.Tensor,             # (C,) class prior pi_c
    agg: str = "uniform",            # 'uniform' or 'attn'
    attn_weights: torch.Tensor | None = None,  # (T_test, K) optional
) -> torch.Tensor:
    """Convert pairwise gamma to PDLC posterior over classes per test row.

    For each anchor i with class c_i:
      p_i(c_i) = gamma(q,i);
      p_i(c≠c_i) = (1 - gamma(q,i)) * pi_c / (1 - pi_{c_i}).
    Then average across anchors (uniform or attention-weighted) to get P(c|q).
    """
    device = gamma.device
    T_test, K = gamma.shape
    # Priors
    pi = priors.to(device=device, dtype=torch.float32)  # (C,)
    # Anchor classes (T_test, K)
    train_lab_t = torch.as_tensor(train_labels_int, device=device, dtype=torch.long)
    ci = train_lab_t[idx_topk]  # (T_test, K)
    C = int(num_classes)
    oh = torch.nn.functional.one_hot(ci, num_classes=C).to(dtype=torch.float32)  # (T_test,K,C)
    den = 1.0 - pi[ci]  # (T_test,K)
    den = den.clamp_min(1e-6)
    # Background distribution with anchor class zeroed
    base = pi.view(1, 1, C).expand(T_test, K, C)
    base = base * (1.0 - oh)  # zero at anchor class
    p_other = (1.0 - gamma)[..., None] * (base / den[..., None])
    p_anchor = gamma[..., None] * oh
    p_i = p_anchor + p_other  # (T_test, K, C)

    if agg == "attn":
        assert attn_weights is not None, "attn_weights required when agg='attn'"
        w = attn_weights.to(device=device, dtype=torch.float32)
        w = w / (w.sum(dim=1, keepdim=True).clamp_min(1e-12))
        P = (p_i * w[..., None]).sum(dim=1)
    else:
        P = p_i.mean(dim=1)
    # Normalize (safety): each row sums to 1 by construction, but guard against tiny drift
    P = P / (P.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return P


class ResidualMLP(torch.nn.Module):
    """Two-block residual MLP for pairwise same-class scoring.

    Input: pair features of dim 2*d0 = concat(|x_t - x_a|, x_t ⊙ x_a)
    Width: w = min(4*d0, 512); if d0 >= 256, cap=768
    Blocks: pre-norm residual (LN → Linear → GELU → Dropout → Linear) + skip
    Output: single logit; temperature buffer for post-hoc calibration
    """

    def __init__(self, base_dim: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.base_dim = int(base_dim)
        self.pair_dim = 2 * self.base_dim
        cap = 768 if self.base_dim >= 256 else 512
        self.width = min(4 * self.base_dim, cap)

        Act = torch.nn.GELU if activation.lower() == "gelu" else torch.nn.SiLU

        self.in_ln = torch.nn.LayerNorm(self.pair_dim)

        # Block 1
        self.blk1_ln = torch.nn.LayerNorm(self.pair_dim)
        self.blk1_fc1 = torch.nn.Linear(self.pair_dim, self.width)
        self.blk1_act = Act()
        self.blk1_drop = torch.nn.Dropout(dropout)
        self.blk1_fc2 = torch.nn.Linear(self.width, self.pair_dim)

        # Block 2
        self.blk2_ln = torch.nn.LayerNorm(self.pair_dim)
        self.blk2_fc1 = torch.nn.Linear(self.pair_dim, self.width)
        self.blk2_act = Act()
        self.blk2_drop = torch.nn.Dropout(dropout)
        self.blk2_fc2 = torch.nn.Linear(self.width, self.pair_dim)

        # Output head
        self.out = torch.nn.Linear(self.pair_dim, 1)
        self.register_buffer("temperature", torch.tensor(1.0))  # not used in loss by default

    def _resblk(self, x, ln, fc1, act, drop, fc2):
        y = ln(x)
        y = fc1(y)
        y = act(y)
        y = drop(y)
        y = fc2(y)
        return x + y

    def forward(self, pair_feats: torch.Tensor, apply_temperature: bool = False) -> torch.Tensor:
        x = self.in_ln(pair_feats)
        x = self._resblk(x, self.blk1_ln, self.blk1_fc1, self.blk1_act, self.blk1_drop, self.blk1_fc2)
        x = self._resblk(x, self.blk2_ln, self.blk2_fc1, self.blk2_act, self.blk2_drop, self.blk2_fc2)
        logits = self.out(x).squeeze(-1)
        if apply_temperature:
            return logits / self.temperature.clamp_min(1e-6)
        return logits


def build_pairs(
    feat_mean: torch.Tensor,
    train_size: int,
    topk_map: Dict[int, List[Tuple[int, float]]],
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build pair features and binary labels.

    Pair features: [|x_t - x_a|, x_t ⊙ x_a]
    Label: 1 if same class else 0
    """
    device = feat_mean.device
    E = feat_mean.shape[-1]
    X_list: List[torch.Tensor] = []
    y_list: List[float] = []
    for i, nbrs in topk_map.items():
        xt = feat_mean[i]
        yti = test_labels[i - train_size]
        for j, _ in nbrs:
            xa = feat_mean[j]
            yj = train_labels[j]
            pair = torch.cat([(xt - xa).abs(), xt * xa], dim=-1)
            X_list.append(pair)
            y_list.append(1.0 if yti == yj else 0.0)

    if not X_list:
        return torch.empty(0, 2 * E, device=device), torch.empty(0, device=device)
    X = torch.stack(X_list, dim=0).to(device)
    y = torch.tensor(y_list, dtype=torch.float32, device=device)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="iris", choices=["iris"], help="Small dataset to try")
    ap.add_argument("--model_path", type=str, default="checkpoints/tabicl-classifier-v1.1-0506.ckpt", help="Pretrained TabICL checkpoint")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--topk", type=int, default=5, help="Return top-k train rows per test row; 0=all sorted")
    ap.add_argument("--avg_last2", action="store_true", help="Average attention from last two TFicl layers for attn-only ranking")
    ap.add_argument("--rank_metric", type=str, default="attn", choices=["attn", "axv"], help="Ranking to build pairs from")
    ap.add_argument("--n_estimators", type=int, default=1, help="Number of ensemble variants to aggregate (TabICLClassifier)")
    args = ap.parse_args()

    # Load small dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Simple split for classifier fit/test
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Set up classifier for preprocessing + backbone loading
    clf = TabICLClassifier(
        n_estimators=max(1, int(args.n_estimators)),
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

    # Build in-context episodes using the fitted ensemble generator
    X_te_num = clf.X_encoder_.transform(X_te)
    data = clf.ensemble_generator_.transform(X_te_num)
    n_classes = clf.n_classes_
    device = torch.device(args.device)
    model = clf.model_.to(device)
    model.eval()
    # Lazy MLP init
    mlp = None
    # Accumulators
    P_uniform_acc = None
    P_attn_acc = None
    n_variants = 0
    # Ground-truth test labels for diagnostics
    test_labels = clf.y_encoder_.inverse_transform(y_te.astype(int))

    for method, (Xs, ys_shifted) in data.items():
        shift_offsets = clf.ensemble_generator_.class_shift_offsets_[method]
        V = Xs.shape[0]
        for vidx in range(V):
            X_variant = Xs[vidx]
            y_train_shifted = ys_shifted[vidx]
            shift_offset = int(shift_offsets[vidx]) if isinstance(shift_offsets[vidx], (int, np.integer)) else int(shift_offsets[vidx])
            y_train_orig_int = (y_train_shifted - shift_offset) % n_classes
            train_size = y_train_shifted.shape[0]
            T = X_variant.shape[0]

            X_tensor = torch.from_numpy(X_variant).float().unsqueeze(0).to(device)
            # TFrow features (exclude CLS)
            _, feat_mean = tfrow_feature_tokens(model, X_tensor, train_size=train_size)
            with torch.no_grad():
                col_out = model.col_embedder(X_tensor, train_size=train_size, mgr_config=clf.inference_config_.COL_CONFIG)
                row_reps = model.row_interactor(col_out, mgr_config=clf.inference_config_.ROW_CONFIG)
                yt = torch.as_tensor(y_train_shifted, device=device, dtype=torch.float32).unsqueeze(0)
                R_cond = row_reps.clone()
                R_cond[:, :train_size] = R_cond[:, :train_size] + model.icl_predictor.y_encoder(yt)

            X_pairs, y_pairs, idx_topk = build_pairs_batched(
                feat_mean=feat_mean,
                train_size=train_size,
                attn_or_axv=args.rank_metric,
                model=model,
                R_cond=R_cond,
                topk=args.topk,
                avg_last2=args.avg_last2,
                train_labels_int=y_train_orig_int,
                test_labels_int=y_te,
            )
            if X_pairs.shape[0] == 0:
                continue

            if mlp is None:
                mlp = ResidualMLP(base_dim=feat_mean.shape[-1]).to(device)
            with torch.no_grad():
                logits_all = mlp(X_pairs)
                gamma_all = torch.sigmoid(logits_all).view(-1)
            T_test = T - train_size
            K = idx_topk.shape[1] if idx_topk.numel() > 0 else 0
            gamma_mat = gamma_all.view(T_test, K)
            # Priors from anchors (per variant)
            C = int(n_classes)
            counts = np.bincount(y_train_orig_int.astype(int), minlength=C).astype(np.float32)
            pi = torch.from_numpy(counts / max(1, counts.sum()))
            # Variant posteriors
            P_uni_v = pdlc_posterior_from_gamma(
                gamma=gamma_mat,
                idx_topk=idx_topk,
                train_labels_int=y_train_orig_int,
                num_classes=C,
                priors=pi,
                agg="uniform",
            )
            weights_avg, _ = _compute_rank_matrices(model, R_cond, train_size, avg_last2=args.avg_last2)
            attn_full = weights_avg[train_size:, :train_size]
            attn_topk = torch.gather(attn_full, dim=1, index=idx_topk.to(attn_full.device))
            P_attn_v = pdlc_posterior_from_gamma(
                gamma=gamma_mat,
                idx_topk=idx_topk,
                train_labels_int=y_train_orig_int,
                num_classes=C,
                priors=pi,
                agg="attn",
                attn_weights=attn_topk,
            )
            P_uniform_acc = P_uni_v if P_uniform_acc is None else (P_uniform_acc + P_uni_v)
            P_attn_acc = P_attn_v if P_attn_acc is None else (P_attn_acc + P_attn_v)
            n_variants += 1

    if n_variants > 0:
        P_uniform = P_uniform_acc / float(n_variants)
        P_attn = P_attn_acc / float(n_variants)
        y_te_t = torch.as_tensor(y_te, dtype=torch.long, device=P_uniform.device)
        nll_uniform = torch.nn.functional.nll_loss(torch.log(P_uniform.clamp_min(1e-12)), y_te_t)
        nll_attn = torch.nn.functional.nll_loss(torch.log(P_attn.clamp_min(1e-12)), y_te_t)
        print(f"Aggregated over {n_variants} variants: PDLC NLL uniform={nll_uniform.item():.4f}  attn-weighted={nll_attn.item():.4f}")
    else:
        print("Warning: no pairs built across variants; nothing to aggregate.")


if __name__ == "__main__":
    main()
