from __future__ import annotations
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from .rope import RotaryEmbedding


def compute_elliptical_diag(
    v: Tensor,
    v_prev: Tensor,
    delta: float = 1.0,
    scale_mode: str = "max",
    eps: float = 1e-12,
    mask_keep: Optional[Tensor] = None,
) -> Tensor:
    """Compute per-head diagonal scaling from consecutive layers' values.

    Parameters
    ----------
    v : Tensor
        Current block values of shape (..., nh, L, Dh)

    v_prev : Tensor
        Previous block values with the same shape as `v`

    delta : float, default=1.0
        Finite-difference step size

    scale_mode : str, default="max"
        How to normalize per-head scales: "max" or "mean"

    eps : float, default=1e-12
        Numerical stability term

    Returns
    -------
    Tensor
        Per-head, per-dim scaling of shape (nh, Dh)
    """
    with torch.no_grad():
        value_diff = (v - v_prev) / float(delta)
        nd = value_diff.dim()
        len_idx = -2
        reduce_dims = (len_idx,)
        if mask_keep is not None:
            # mask_keep expected to indicate which key positions to include
            # Accept 1D (L,) mask; broadcast to (..., 1, L, 1)
            if mask_keep.dim() == 1:
                mk = mask_keep.view(*([1] * (nd - 2)), mask_keep.shape[0], 1).to(device=v.device, dtype=value_diff.dtype)
            else:
                # Best-effort: try to align to (..., L) then unsqueeze (-2 for head, -1 for dim)
                mk = mask_keep.to(device=v.device, dtype=value_diff.dtype)
                for _ in range(max(0, nd - mk.dim() - 1)):
                    mk = mk.unsqueeze(0)
                mk = mk.unsqueeze(-1)
            abs_w = value_diff.abs() * mk
            denom = mk.sum(dim=reduce_dims).clamp_min(eps)
            m = abs_w.sum(dim=reduce_dims) / denom
        else:
            m = value_diff.abs().mean(dim=reduce_dims)  # (nh, Dh)

        if scale_mode == "mean":
            denom = m.mean(dim=-1, keepdim=True)
        else:
            denom = m.amax(dim=-1, keepdim=True)
        denom = denom.clamp_min(eps)
        m = m / denom

        # Numerical guard: sanitize any accidental NaN/Inf that may arise from extreme inputs
        m = torch.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0)

    return m


def sdpa_with_flattened_batch(
    q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0
) -> Tensor:
    """Applies scaled dot-product attention with flattened batch dimensions.

    This function handles arbitrary batch dimensions by flattening them before
    applying PyTorch's scaled_dot_product_attention and then reshaping the output
    back to the original shape. This flattening is necessary to properly trigger
    Flash Attention.

    Parameters
    ----------
    q : Tensor
        Query tensor of shape (..., nh, tgt_len, hs) where:
        - ... represents arbitrary batch dimensions
        - nh is the number of attention heads
        - tgt_len is the target sequence length
        - hs is the head size (embedding dimension per head)

    k : Tensor
        Key tensor of shape (..., nh, src_len, hs) with matching batch dimensions

    v : Tensor
        Value tensor of shape (..., nh, src_len, hs) with matching batch dimensions

    attn_mask : Optional[Tensor], default=None
        Attention mask of shape (..., nh, tgt_len, src_len)

    dropout_p : float, default=0.0
        Dropout probability applied to attention weights

    Returns
    -------
    Tensor
        Attention output tensor of shape (..., nh, tgt_len, hs) preserving the
        original batch dimensions of the input
    """

    q_shape = q.shape
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])
    if attn_mask is not None:
        attn_mask = attn_mask.reshape(-1, *attn_mask.shape[-3:])
    out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    # Numerical guard: replace NaNs that can occur in pathological all-masked rows
    out = torch.nan_to_num(out, nan=0.0)

    return out.view(q_shape)


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor | int] = None,
    rope: Optional[RotaryEmbedding] = None,
    elliptical_scale: Optional[Tensor] = None,
    # Parameter-free elliptical estimator controls
    elliptical: bool = False,
    v_prev: Optional[Tensor] = None,
    elliptical_delta: float = 1.0,
    elliptical_scale_mode: str = "max",
    # Testing/override controls
    elliptical_force_identity: bool = False,
    # TFrow-specific controls: exclude CLS tokens from EA
    row_num_cls: Optional[int] = None,
    exclude_cls_from_ea: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Multi-head attention with support for rotary position embeddings
    as well as specialized processing when attn_mask is an integer.

    Parameters
    ----------
    query : Tensor
        Query tensor of shape (..., tgt_len, embed_dim)

    key : Tensor
        Key tensor of shape (..., src_len, embed_dim)

    value : Tensor
        Value tensor of shape (..., src_len, embed_dim)

    num_heads : int
        Number of attention heads

    in_proj_weight : Tensor
        Combined weight matrix for Q, K, V input projections

    in_proj_bias : Tensor
        Combined bias vector for input projections

    dropout_p : float
        Dropout probability applied to attention weights

    out_proj_weight : Tensor
        Output projection weight matrix

    out_proj_bias : Tensor
        Output projection bias vector

    training : bool, default=True
        Whether the model is in training mode (affects dropout)

    key_padding_mask : Optional[Tensor], default=None
        Mask of shape (..., src_len) that identifies padding elements
        in the key sequence to be ignored:
            - For binary masks: True values indicate positions to ignore
            - For float masks: Values are directly added to attention scores

    attn_mask : Optional[Tensor | int], default=None
        Controls attention pattern in two possible ways:
        1. When provided as Tensor: Traditional mask preventing attention to certain positions
            - Shape: (tgt_len, src_len) or (..., num_heads, tgt_len, src_len)
        2. When provided as integer: Creates a split attention pattern where:
            - The first `attn_mask` tokens perform self-attention only (attend to themselves)
            - The remaining tokens attend only to the first `attn_mask` tokens

    rope : Optional[RotaryEmbedding]
        Rotary positional encoding

    Returns
    -------
    Tuple[Tensor, Tensor]
        - Attention output tensor of shape (..., tgt_len, embed_dim)
        - Current value projections reshaped to (..., nh, src_len, head_dim)
    """

    if isinstance(attn_mask, int):
        assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
        assert rope is None, "Rotary position embedding is not supported with attn_mask as int"

    # Extract shape information, supporting arbitrary batch dimensions
    *batch_shape, tgt_len, embed_dim = query.shape
    src_len = key.shape[-2]

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    # Joint projection of query, key, value
    q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    # Reshape for multi-head attention
    q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)  # (batch_shape, nh, tgt_len, hs)
    k = k.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)  # (batch_shape, nh, src_len, hs)
    v = v.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)  # (batch_shape, nh, src_len, hs)
    
        
    if elliptical and v_prev is not None:
        if v_prev.shape != v.shape:
            raise ValueError(f"v_prev shape {v_prev.shape} must match v shape {v.shape} for elliptical estimator")
        if exclude_cls_from_ea and (row_num_cls is not None) and (row_num_cls > 0):
            # Exclude CLS tokens from estimator and application
            if v.shape[-2] > row_num_cls and v_prev.shape[-2] > row_num_cls:
                v_feat = v[..., :, row_num_cls:, :]
                v_prev_feat = v_prev[..., :, row_num_cls:, :]
                m = compute_elliptical_diag(v_feat, v_prev_feat, delta=elliptical_delta, scale_mode=elliptical_scale_mode)
            else:
                m = compute_elliptical_diag(v, v_prev, delta=elliptical_delta, scale_mode=elliptical_scale_mode)
            # Build per-time scale
            sqrt_m = torch.sqrt(m.clamp_min(1e-12)).to(dtype=q.dtype, device=q.device)
            m_bc = sqrt_m.unsqueeze(-2)
            scale = torch.ones_like(q)
            if q.shape[-2] > row_num_cls:
                tgt = scale[..., :, row_num_cls:, :]
                scale[..., :, row_num_cls:, :] = m_bc.expand(tgt.shape)
            q = q * scale
            k = k * scale
        else:
            # For ICL int mask: restrict estimator to allowed key set (train slice only)
            if isinstance(attn_mask, int):
                cut = int(attn_mask)
                keep = torch.zeros(v.shape[-2], device=v.device, dtype=torch.float32)
                keep[:cut] = 1.0
                m = compute_elliptical_diag(v, v_prev, delta=elliptical_delta, scale_mode=elliptical_scale_mode, mask_keep=keep)
            else:
                m = compute_elliptical_diag(v, v_prev, delta=elliptical_delta, scale_mode=elliptical_scale_mode)
            # Broadcast to (..., nh, 1, hs)
            sqrt_m = torch.sqrt(m.clamp_min(1e-12)).to(dtype=q.dtype, device=q.device)
            m_bc = sqrt_m.unsqueeze(-2)
            q = q * m_bc
            k = k * m_bc
    elif elliptical_force_identity:
        # Identity metric fallback: no scaling
        pass
    # elif elliptical_scale is not None:
    #     # Expected to be broadcastable over q: (..., nh, tgt_len, head_dim)
    #     # Typical shapes: (1, nh, 1, head_dim) or (B, nh, 1, head_dim)
    #     if elliptical_scale.dim() < 4:
    #         raise ValueError(
    #             f"elliptical_scale must have at least 4 dims (..., nh, 1|T, hs), got {elliptical_scale.shape}"
    #         )
    #     if elliptical_scale.shape[-1] != head_dim:
    #         raise ValueError(
    #             f"elliptical_scale head_dim mismatch: expected {head_dim}, got {elliptical_scale.shape[-1]}"
    #         )
    #     if elliptical_scale.shape[-3] != num_heads:
    #         raise ValueError(
    #             f"elliptical_scale num_heads mismatch: expected {num_heads} at dim -3, got {elliptical_scale.shape[-3]}"
    #         )
    #     if elliptical_scale.shape[-2] not in (1, tgt_len):
    #         raise ValueError(
    #             f"elliptical_scale time dim must be 1 or tgt_len ({tgt_len}), got {elliptical_scale.shape[-2]}"
    #         )

    #     # Cast to q/k dtype/device
    #     elliptical_scale = elliptical_scale.to(dtype=q.dtype, device=q.device)
    #     # No normalization here: any regularization handled during training
    #     q = q * elliptical_scale
    #     k = k * elliptical_scale

    
    # Apply rotary position embeddings if provided
    if rope is not None:
        q = rope.rotate_queries_or_keys(q)
        k = rope.rotate_queries_or_keys(k)

    # Disable dropout during evaluation
    if not training:
        dropout_p = 0.0

    if isinstance(attn_mask, int):
        cut_pos = attn_mask  # For better readability

        # Pre-allocate output tensor to avoid concatenation
        attn_output = torch.empty(*batch_shape, tgt_len, embed_dim, device=query.device, dtype=query.dtype)

        # Process left segment (self-attention within first cut_pos tokens)
        q_left = q[..., :cut_pos, :]  # (batch_shape, nh, cut_pos, hs)
        k_left = k[..., :cut_pos, :]
        v_left = v[..., :cut_pos, :]

        attn_left = sdpa_with_flattened_batch(q_left, k_left, v_left, dropout_p=dropout_p)
        attn_left = attn_left.transpose(-3, -2).contiguous().view(*batch_shape, cut_pos, embed_dim)
        attn_output[..., :cut_pos, :] = F.linear(attn_left, out_proj_weight, out_proj_bias)

        # Process right segment (tokens after cut_pos attending to tokens before cut_pos)
        if cut_pos < tgt_len:
            q_right = q[..., cut_pos:, :]  # (batch_shape, nh, tgt_len - cut_pos, hs)
            attn_right = sdpa_with_flattened_batch(q_right, k_left, v_left, dropout_p=dropout_p)
            attn_right = attn_right.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len - cut_pos, embed_dim)
            attn_output[..., cut_pos:, :] = F.linear(attn_right, out_proj_weight, out_proj_bias)
    else:
        # Process attention mask
        correct_2d_shape = (tgt_len, src_len)
        correct_nd_shape = (*batch_shape, num_heads, tgt_len, src_len)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                if attn_mask.shape != correct_2d_shape:
                    raise ValueError(f"2D attn_mask should have shape {correct_2d_shape}, but got {attn_mask.shape}")
                attn_mask = attn_mask.expand(*batch_shape, num_heads, tgt_len, src_len)
            elif attn_mask.dim() == len(correct_nd_shape):
                if attn_mask.shape != correct_nd_shape:
                    raise ValueError(
                        f"{len(correct_nd_shape)}D attn_mask should have shape {correct_nd_shape}, "
                        f"but got {attn_mask.shape}"
                    )
            else:
                raise ValueError(f"attn_mask must be 2D or {len(correct_nd_shape)}D, got {attn_mask.dim()}D")

        # Process key padding mask
        if key_padding_mask is not None:
            if key_padding_mask.shape != (*batch_shape, src_len):
                raise ValueError(
                    f"key_padding_mask should have shape {(*batch_shape, src_len)}, but got {key_padding_mask.shape}"
                )
            key_padding_mask = key_padding_mask.view(*batch_shape, 1, 1, src_len).expand(
                *batch_shape, num_heads, tgt_len, src_len
            )

            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        # Ensure mask dtype/semantics: convert masks to additive form (0 keep, -inf ignore)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # bool -> additive: True => -inf, False => 0
                attn_mask = attn_mask.to(dtype=q.dtype, device=q.device)
                attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), float("-inf"))
            elif attn_mask.dtype.is_floating_point:
                # already additive; cast to q's dtype/device
                attn_mask = attn_mask.to(dtype=q.dtype, device=q.device)
            else:
                # integer/binary mask: treat non-zeros as ignore
                mask_bool = attn_mask != 0
                attn_mask = mask_bool.to(dtype=q.dtype, device=q.device)
                attn_mask = attn_mask.masked_fill(mask_bool, float("-inf"))

        attn_output = sdpa_with_flattened_batch(q, k, v, attn_mask, dropout_p)  # (..., nh, tgt_len, hs)

        # Reshape and project output
        attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)  # (batch_shape, tgt_len, E)

    return attn_output, v
