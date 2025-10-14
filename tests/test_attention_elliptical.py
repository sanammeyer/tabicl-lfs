import math
import pytest
import torch

from tabicl.model.attention import multi_head_attention_forward


def make_identity_proj(embed_dim: int):
    """Return in_proj/out_proj weights that make q=query, k=key, v=value and out proj identity."""
    I = torch.eye(embed_dim)
    in_proj_weight = torch.cat([I, I, I], dim=0)  # (3E, E)
    in_proj_bias = torch.zeros(3 * embed_dim)
    out_proj_weight = torch.eye(embed_dim)
    out_proj_bias = torch.zeros(embed_dim)
    return in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias


@pytest.mark.parametrize("B,Tq,Tk,H,hd", [(2, 5, 7, 4, 8)])
@pytest.mark.parametrize("use_masks", [False, True])
def test_equivalence_elliptical_ones(B, Tq, Tk, H, hd, use_masks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    E = H * hd

    query = torch.randn(B, Tq, E, device=device, dtype=dtype)
    key = torch.randn(B, Tk, E, device=device, dtype=dtype)
    value = torch.randn(B, Tk, E, device=device, dtype=dtype)

    in_w, in_b, out_w, out_b = make_identity_proj(E)
    in_w, in_b, out_w, out_b = (
        in_w.to(device=device, dtype=dtype),
        in_b.to(device=device, dtype=dtype),
        out_w.to(device=device, dtype=dtype),
        out_b.to(device=device, dtype=dtype),
    )

    attn_mask = None
    key_padding_mask = None
    if use_masks:
        # Boolean masks
        attn_mask = torch.rand(Tq, Tk, device=device) < 0.2
        key_padding_mask = torch.rand(B, Tk, device=device) < 0.2

    # Baseline
    out_dot = multi_head_attention_forward(
        query,
        key,
        value,
        num_heads=H,
        in_proj_weight=in_w,
        in_proj_bias=in_b,
        dropout_p=0.0,
        out_proj_weight=out_w,
        out_proj_bias=out_b,
        training=False,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        rope=None,
        elliptical_scale=None,
    )

    # Elliptical with all-ones scale
    ones = torch.ones(1, H, 1, hd, device=device, dtype=dtype)
    out_ell = multi_head_attention_forward(
        query,
        key,
        value,
        num_heads=H,
        in_proj_weight=in_w,
        in_proj_bias=in_b,
        dropout_p=0.0,
        out_proj_weight=out_w,
        out_proj_bias=out_b,
        training=False,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        rope=None,
        elliptical_scale=ones,
    )

    max_abs_diff = (out_dot - out_ell).abs().max().item()
    assert max_abs_diff < 1e-6


def test_mask_combination_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    B, Tq, Tk, H, hd = 2, 5, 7, 4, 8
    E = H * hd

    query = torch.randn(B, Tq, E, device=device, dtype=dtype)
    key = torch.randn(B, Tk, E, device=device, dtype=dtype)
    value = torch.randn(B, Tk, E, device=device, dtype=dtype)

    in_w, in_b, out_w, out_b = make_identity_proj(E)
    in_w, in_b, out_w, out_b = (
        in_w.to(device=device, dtype=dtype),
        in_b.to(device=device, dtype=dtype),
        out_w.to(device=device, dtype=dtype),
        out_b.to(device=device, dtype=dtype),
    )

    # Boolean masks
    attn_mask_bool = torch.rand(Tq, Tk, device=device) < 0.2
    key_padding_mask_bool = torch.rand(B, Tk, device=device) < 0.2

    # Direct call with both masks (boolean)
    out_bools = multi_head_attention_forward(
        query,
        key,
        value,
        num_heads=H,
        in_proj_weight=in_w,
        in_proj_bias=in_b,
        dropout_p=0.0,
        out_proj_weight=out_w,
        out_proj_bias=out_b,
        training=False,
        key_padding_mask=key_padding_mask_bool,
        attn_mask=attn_mask_bool,
        rope=None,
        elliptical_scale=None,
    )

    # Manually combine into additive mask and pass as single mask
    # attn_mask_bool: (Tq, Tk) -> (B, H, Tq, Tk)
    attn_add = attn_mask_bool.to(dtype)
    attn_add = attn_add.masked_fill(attn_mask_bool, float("-inf"))
    attn_add = attn_add.expand(B, H, Tq, Tk)

    # key_padding_mask_bool: (B, Tk) -> (B, H, Tq, Tk)
    kpm_add = key_padding_mask_bool.view(B, 1, 1, Tk).expand(B, H, Tq, Tk)
    kpm_add = kpm_add.to(dtype).masked_fill(kpm_add.to(torch.bool), float("-inf"))

    combined = attn_add + kpm_add

    out_add = multi_head_attention_forward(
        query,
        key,
        value,
        num_heads=H,
        in_proj_weight=in_w,
        in_proj_bias=in_b,
        dropout_p=0.0,
        out_proj_weight=out_w,
        out_proj_bias=out_b,
        training=False,
        key_padding_mask=None,
        attn_mask=combined,
        rope=None,
        elliptical_scale=None,
    )

    max_abs_diff = (out_bools - out_add).abs().max().item()
    assert max_abs_diff < 1e-6


@pytest.mark.parametrize("shape_case", ["ones_1H1d", "ones_BH1d"])
def test_elliptical_broadcast_shapes(shape_case):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    B, Tq, Tk, H, hd = 2, 5, 7, 4, 8
    E = H * hd

    query = torch.randn(B, Tq, E, device=device, dtype=dtype)
    key = torch.randn(B, Tk, E, device=device, dtype=dtype)
    value = torch.randn(B, Tk, E, device=device, dtype=dtype)

    in_w, in_b, out_w, out_b = make_identity_proj(E)
    in_w, in_b, out_w, out_b = (
        in_w.to(device=device, dtype=dtype),
        in_b.to(device=device, dtype=dtype),
        out_w.to(device=device, dtype=dtype),
        out_b.to(device=device, dtype=dtype),
    )

    base = multi_head_attention_forward(
        query,
        key,
        value,
        num_heads=H,
        in_proj_weight=in_w,
        in_proj_bias=in_b,
        dropout_p=0.0,
        out_proj_weight=out_w,
        out_proj_bias=out_b,
        training=False,
        key_padding_mask=None,
        attn_mask=None,
        rope=None,
        elliptical_scale=None,
    )

    if shape_case == "ones_1H1d":
        scale = torch.ones(1, H, 1, hd, device=device, dtype=dtype)
    else:
        scale = torch.ones(B, H, 1, hd, device=device, dtype=dtype)

    out = multi_head_attention_forward(
        query,
        key,
        value,
        num_heads=H,
        in_proj_weight=in_w,
        in_proj_bias=in_b,
        dropout_p=0.0,
        out_proj_weight=out_w,
        out_proj_bias=out_b,
        training=False,
        key_padding_mask=None,
        attn_mask=None,
        rope=None,
        elliptical_scale=scale,
    )

    max_abs_diff = (base - out).abs().max().item()
    assert max_abs_diff < 1e-6


def test_amp_bf16_no_nan():
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    B, Tq, Tk, H, hd = 2, 5, 7, 4, 8
    E = H * hd

    # Autocast may not be supported for SDPA on some CPU builds; skip on failure
    try:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            query = torch.randn(B, Tq, E, device=device)
            key = torch.randn(B, Tk, E, device=device)
            value = torch.randn(B, Tk, E, device=device)

            in_w, in_b, out_w, out_b = make_identity_proj(E)
            in_w, in_b, out_w, out_b = (
                in_w.to(device=device),
                in_b.to(device=device),
                out_w.to(device=device),
                out_b.to(device=device),
            )

            attn_mask = torch.rand(Tq, Tk, device=device) < 0.2
            key_padding_mask = torch.rand(B, Tk, device=device) < 0.2

            scale = torch.ones(1, H, 1, hd, device=device)

            out = multi_head_attention_forward(
                query,
                key,
                value,
                num_heads=H,
                in_proj_weight=in_w,
                in_proj_bias=in_b,
                dropout_p=0.0,
                out_proj_weight=out_w,
                out_proj_bias=out_b,
                training=False,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope=None,
                elliptical_scale=scale,
            )

            assert torch.isfinite(out).all()
    except (RuntimeError, TypeError) as e:
        pytest.skip(f"Autocast bf16 not supported in this environment: {e}")

