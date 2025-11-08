#!/usr/bin/env python3
"""Quick smoke tests for parameter-free Elliptical Attention.

S0.1 Identity-metric check: with M=I, outputs match baseline.
S0.2 Monotonicity check: manual m emphasizing dim-0 shifts attention mass accordingly.

Run: python scripts/test_elliptical_smoke.py
"""

import math
import os
import sys
import torch

# Ensure 'src' is on the path for local package imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tabicl.model.encoders import Encoder


@torch.inference_mode()
def s01_identity_check(device: str = "cpu"):
    torch.manual_seed(0)
    B, L, E = 2, 7, 32
    nhead, ff = 4, 64

    x = torch.randn(B, L, E, device=device)

    # Build a 2-layer encoder (no RoPE) with elliptical gate enabled
    enc = Encoder(
        num_blocks=2,
        d_model=E,
        nhead=nhead,
        dim_feedforward=ff,
        dropout=0.0,
        norm_first=True,
        use_rope=False,
        elliptical=True,
        elliptical_delta=1.0,
        elliptical_scale_mode="max",
    ).to(device)
    enc.eval()  # disable dropout for exact comparison

    # Baseline: disable elliptical scaling entirely
    for blk in enc.blocks:
        blk.elliptical = False
        blk.elliptical_override = "none"
        blk.elliptical_manual_m = None
    out_base = enc(x)

    # Identity-metric path: enable elliptical but force identity scaling
    for blk in enc.blocks:
        blk.elliptical = True
        blk.elliptical_override = "identity"  # keep EA path engaged, but M = I
        blk.elliptical_manual_m = None
    out_id = enc(x)

    max_abs_diff = (out_base - out_id).abs().max().item()
    print(f"[S0.1] Identity-metric max |Δ| = {max_abs_diff:.3e} (should be ~0)")
    assert torch.allclose(out_base, out_id, atol=1e-6, rtol=1e-6)


def _random_attn_mask(L: int, p: float = 0.2, device: str = "cpu"):
    """Additive attention mask: 0 (keep) or -inf (mask). Shape (L, L)."""
    m = torch.zeros(L, L, device=device)
    if p > 0:
        keep = torch.rand(L, L, device=device) > p
        m[~keep] = float("-inf")
    return m


@torch.inference_mode()
def s01_identity_check_masked_multihead(device: str = "cpu"):
    torch.manual_seed(1)
    B, L, E = 2, 9, 48
    nhead, ff = 3, 96

    x = torch.randn(B, L, E, device=device)
    mask = _random_attn_mask(L, p=0.15, device=device)

    enc = Encoder(
        num_blocks=2,
        d_model=E,
        nhead=nhead,
        dim_feedforward=ff,
        dropout=0.0,
        norm_first=True,
        use_rope=False,
        elliptical=True,
        elliptical_delta=1.0,
        elliptical_scale_mode="max",
    ).to(device)
    enc.eval()

    # Baseline
    for blk in enc.blocks:
        blk.elliptical = False
        blk.elliptical_override = "none"
        blk.elliptical_manual_m = None
    out_base = enc(x, attn_mask=mask)

    # Identity-metric override (EA path engaged)
    for blk in enc.blocks:
        blk.elliptical = True
        blk.elliptical_override = "identity"
        blk.elliptical_manual_m = None
    out_id = enc(x, attn_mask=mask)

    diff = (out_base - out_id).abs().max().item()
    print(f"[S0.1/masked] Identity-metric max |Δ| = {diff:.3e} (should be ~0)")
    assert torch.allclose(out_base, out_id, atol=1e-6, rtol=1e-6)


@torch.inference_mode()
def s01_const_vector_invariance(device: str = "cpu"):
    torch.manual_seed(2)
    B, L, E = 2, 6, 32
    nhead, ff = 4, 64

    x = torch.randn(B, L, E, device=device)
    enc = Encoder(
        num_blocks=2,
        d_model=E,
        nhead=nhead,
        dim_feedforward=ff,
        dropout=0.0,
        norm_first=True,
        use_rope=False,
        elliptical=True,
        elliptical_delta=1.0,
        elliptical_scale_mode="max",
    ).to(device)
    enc.eval()

    # Baseline
    for blk in enc.blocks:
        blk.elliptical = False
        blk.elliptical_override = "none"
        blk.elliptical_manual_m = None
    out_base = enc(x)

    # Manual m of all ones per head/dim
    head_dim = E // nhead
    m_ones = torch.ones(nhead, head_dim, device=device)
    for blk in enc.blocks:
        blk.elliptical = True
        blk.elliptical_override = "manual"
        blk.elliptical_manual_m = m_ones
    out_manual = enc(x)

    diff = (out_base - out_manual).abs().max().item()
    print(f"[S0.1/const] All-ones m max |Δ| = {diff:.3e} (should be ~0)")
    assert torch.allclose(out_base, out_manual, atol=1e-6, rtol=1e-6)


@torch.inference_mode()
def s01_layer1_guard(device: str = "cpu"):
    torch.manual_seed(3)
    B, L, E = 2, 8, 32
    nhead, ff = 4, 64
    x = torch.randn(B, L, E, device=device)

    # One-block encoder, EA should be a no-op due to missing V_prev
    enc = Encoder(
        num_blocks=1,
        d_model=E,
        nhead=nhead,
        dim_feedforward=ff,
        dropout=0.0,
        norm_first=True,
        use_rope=False,
        elliptical=True,
        elliptical_delta=1.0,
        elliptical_scale_mode="max",
    ).to(device)
    enc.eval()

    # Baseline
    for blk in enc.blocks:
        blk.elliptical = False
        blk.elliptical_override = "none"
        blk.elliptical_manual_m = None
    out_base = enc(x)

    # EA on (but layer-1 fallback to identity expected)
    for blk in enc.blocks:
        blk.elliptical = True
        blk.elliptical_override = "none"
        blk.elliptical_manual_m = None
    out_ea = enc(x)

    diff = (out_base - out_ea).abs().max().item()
    print(f"[S0.1/layer1] Layer-1 guard max |Δ| = {diff:.3e} (should be ~0)")
    assert torch.allclose(out_base, out_ea, atol=1e-6, rtol=1e-6)


def s02_monotonicity_check(device: str = "cpu"):
    torch.manual_seed(0)
    B, H, Dh = 1, 1, 2
    tgt_len, src_len = 1, 5

    # Construct a toy query and keys
    # q emphasizes dim-0, with a smaller component on dim-1
    q = torch.zeros(B, H, tgt_len, Dh, device=device)
    q[..., 0] = 1.0
    q[..., 1] = 0.5

    k = torch.zeros(B, H, src_len, Dh, device=device)
    # Keys varying along dim-0 (similarity to q on dim-0 increases with index)
    k_vals0 = torch.tensor([0.0, 0.1, 0.2, 0.0, 0.0], device=device)
    k[..., 0] = k_vals0
    # Keys varying along dim-1 for some distractors
    k_vals1 = torch.tensor([0.0, 0.0, 0.0, 0.5, -0.5], device=device)
    k[..., 1] = k_vals1

    # Baseline attention weights
    scale = 1.0 / math.sqrt(Dh)
    scores_base = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, H, 1, src_len)
    w_base = scores_base.softmax(dim=-1).squeeze().cpu().numpy()

    # Elliptical manual scaling: emphasize dim-0 strongly
    m = torch.tensor([[10.0, 1.0]], device=device)  # (H, Dh)
    q_e = q * m.view(1, H, 1, Dh)
    k_e = k * m.view(1, H, 1, Dh)
    scores_e = torch.matmul(q_e, k_e.transpose(-1, -2)) * scale
    w_e = scores_e.softmax(dim=-1).squeeze().cpu().numpy()

    print("[S0.2] Baseline weights:", w_base)
    print("[S0.2] Elliptical weights (m0≫others):", w_e)
    # Expect increased mass on keys with higher alignment on dim-0 (idx=2 has highest dim-0)
    print(f"[S0.2] w[idx=2] baseline={w_base[2]:.3f} -> elliptical={w_e[2]:.3f} (should increase)")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")
    s01_identity_check(dev)
    s01_identity_check_masked_multihead(dev)
    s01_const_vector_invariance(dev)
    s01_layer1_guard(dev)
    s02_monotonicity_check(dev)
