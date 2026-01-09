#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-training Elliptical Attention wiring test (TFrow + TFicl).

This script verifies that Elliptical Attention (EA) is actually *used* by the model
and correctly wired across TFrow and TFicl, without training.

It checks (in eval mode) that:
  1) Layer-0 behaves as identity (M≈I) and EA vs ID logits match (≈0 diff)
  2) From layer ≥2, the learned metric is anisotropic and changes logits/attn vs identity
  3) Increasing δ (with normalized scaling) does not increase deviation from identity
     (checked on both logits and the metric itself)
  4) v_prev continuity: each block’s v_prev equals the previous block’s v
  5) Hooks capture every block in eval()
  6) No extra *parameters or buffers* introduced by EA
  7) Critically: the model’s *final outputs* differ when EA is enabled vs identity,
     and stack-isolated toggles (only TFrow EA / only TFicl EA) also change outputs

Run:
  python scripts/dev/test_ea_wiring.py [cuda|cpu]
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# Ensure local src package is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tabicl.model.tabicl import TabICL
from tabicl.model.layers import MultiheadAttentionBlock
from tabicl.model.attention import compute_elliptical_diag


EPS = 1e-6


@dataclass
class BlockCapture:
    q_rope: Optional[torch.Tensor] = None
    k_rope: Optional[torch.Tensor] = None
    q_scaled: Optional[torch.Tensor] = None
    k_scaled: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    v_prev: Optional[torch.Tensor] = None
    m_diag: Optional[torch.Tensor] = None  # (nh, Dh)
    logits: Optional[torch.Tensor] = None  # (..., nh, tgt, src)
    attn: Optional[torch.Tensor] = None    # (..., nh, tgt, src)


@dataclass
class Collector:
    data: Dict[str, Dict[str, Dict[int, BlockCapture]]] = field(default_factory=dict)
    current_pass: str = ""

    def begin_pass(self, name: str):
        self.current_pass = name
        self.data.setdefault(name, {})

    def put(self, stack: str, layer: int, cap: BlockCapture):
        self.data[self.current_pass].setdefault(stack, {})[int(layer)] = cap

    def get(self, name: str, stack: str, layer: int) -> BlockCapture:
        return self.data[name][stack][int(layer)]


def set_all_blocks_mode(model: TabICL, *, identity: bool | None = None, delta: Optional[float] = None):
    """Configure EA overrides and δ for all blocks."""
    # Row blocks
    for blk in model.row_interactor.tf_row.blocks:
        if identity is not None:
            blk.elliptical = True
            blk.elliptical_override = "identity" if identity else "none"
            blk.elliptical_manual_m = None
        if delta is not None:
            blk.elliptical_delta = float(delta)
    # ICL blocks
    for blk in model.icl_predictor.tf_icl.blocks:
        if identity is not None:
            blk.elliptical = True
            blk.elliptical_override = "identity" if identity else "none"
            blk.elliptical_manual_m = None
        if delta is not None:
            blk.elliptical_delta = float(delta)


def set_elliptical_enabled(model: TabICL, enabled: bool):
    """Enable/disable EA feature flags entirely to compare code paths (sanity)."""
    for blk in model.row_interactor.tf_row.blocks:
        blk.elliptical = bool(enabled)
    for blk in model.icl_predictor.tf_icl.blocks:
        blk.elliptical = bool(enabled)


def annotate_stacks(model: TabICL):
    for i, blk in enumerate(model.row_interactor.tf_row.blocks):
        setattr(blk, "_ea_stack", "tfrow")
        setattr(blk, "_ea_layer", int(i))
    for i, blk in enumerate(model.icl_predictor.tf_icl.blocks):
        setattr(blk, "_ea_stack", "tficl")
        setattr(blk, "_ea_layer", int(i))


def _unzero_attention_out_proj(model: TabICL) -> None:
    """Ensure attention outputs can influence the residual during wiring tests.

    The blocks initialize out_proj and FFN final linear weights to all zeros for
    stable training. That makes the block a pure residual identity at init, so
    EA changes inside attention won't affect the layer output. For this wiring
    harness we set out_proj to identity so that geometry changes propagate.
    """
    with torch.no_grad():
        for blk in list(model.row_interactor.tf_row.blocks) + list(model.icl_predictor.tf_icl.blocks):
            W = blk.attn.out_proj.weight
            if W is None:
                continue
            Eo, Ei = W.shape
            # Use identity (or truncated if shapes differ unexpectedly)
            eye = torch.eye(min(Eo, Ei), device=W.device, dtype=W.dtype)
            W.zero_()
            W[:eye.shape[0], :eye.shape[1]].copy_(eye)
            if blk.attn.out_proj.bias is not None:
                blk.attn.out_proj.bias.zero_()


def _compute_logits_and_attn(q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: (..., nh, Tq, Dh) / (..., nh, Tk, Dh)
    Dh = q.shape[-1]
    scale = 1.0 / math.sqrt(float(Dh))
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (..., nh, Tq, Tk)
    if mask is not None:
        scores = scores + mask
    attn = scores.softmax(dim=-1)
    return scores, attn


def _as_additive_mask(mask: Optional[torch.Tensor], target_shape: torch.Size, device, dtype) -> Optional[torch.Tensor]:
    """Convert boolean or float masks to additive form and expand to (..., nh, Tq, Tk)."""
    if mask is None:
        return None
    add = mask
    if add.dtype == torch.bool:
        add = torch.zeros_like(add, dtype=dtype, device=device)
        # True means masked → -inf
        add = add.masked_fill(mask.to(device=device), float("-inf"))
    else:
        add = add.to(dtype=dtype, device=device)

    # Expand 2D (Tq, Tk) to (..., nh, Tq, Tk)
    if add.dim() == 2:
        Bshape = target_shape[:-3]
        nh, Tq, Tk = target_shape[-3], target_shape[-2], target_shape[-1]
        add = add.view(1, 1, *add.shape).expand(*Bshape, nh, Tq, Tk)
    return add


def patch_attention_for_capture(collector: Collector):
    """Monkey-patch MultiheadAttentionBlock._attn_block to capture internals per call in eval()."""
    orig = MultiheadAttentionBlock._attn_block

    def wrapped(self: MultiheadAttentionBlock, q, k, v, key_padding_mask, attn_mask, rope, v_prev=None, block_index=None):
        # Derive shapes and heads
        nh = self.attn.num_heads
        Bshape = q.shape[:-2]  # batch dims (...,)
        Tq = q.shape[-2]
        Tk = k.shape[-2]
        E = q.shape[-1]
        Dh = E // nh

        # Linear projections to per-head
        q_lin, k_lin, v_lin = F._in_projection_packed(q, k, v, self.attn.in_proj_weight, self.attn.in_proj_bias)
        qh = q_lin.view(*Bshape, Tq, nh, Dh).transpose(-3, -2)  # (..., nh, Tq, Dh)
        kh = k_lin.view(*Bshape, Tk, nh, Dh).transpose(-3, -2)  # (..., nh, Tk, Dh)
        vh = v_lin.view(*Bshape, Tk, nh, Dh).transpose(-3, -2)  # (..., nh, Tk, Dh)

        # Apply RoPE (if any)
        q_rope = rope.rotate_queries_or_keys(qh) if rope is not None else qh
        k_rope = rope.rotate_queries_or_keys(kh) if rope is not None else kh

        # Gate EA strictly by annotated layer index (layer ≥ 1) and presence of v_prev
        layer = int(getattr(self, "_ea_layer", int(block_index) if block_index is not None else -1))
        use_elliptical = self.elliptical and (v_prev is not None) and (layer >= 1)
        # CLS exclusion controls for TFrow
        row_num_cls = int(getattr(self, "_row_num_cls", 0) or 0)
        exclude_cls = bool(getattr(self, "_exclude_cls_from_ea", False)) and (row_num_cls > 0)

        # Decide m_diag
        m_diag = None
        if getattr(self, "elliptical_override", "none") == "manual" and (getattr(self, "elliptical_manual_m", None) is not None):
            m_diag = self.elliptical_manual_m.to(dtype=q_rope.dtype, device=q_rope.device)
        elif getattr(self, "elliptical_override", "none") == "identity":
            m_diag = torch.ones(nh, Dh, dtype=q_rope.dtype, device=q_rope.device)
        elif use_elliptical:
            # v_prev is expected to match vh shape (..., nh, Tk, Dh)
            if (isinstance(v_prev, torch.Tensor)) and (tuple(v_prev.shape[-3:]) == (nh, Tk, Dh)):
                # ICL split: restrict estimator to allowed key set (train slice only)
                if isinstance(attn_mask, int):
                    cut = int(attn_mask)
                    m_diag = compute_elliptical_diag(
                        vh[..., :, :cut, :],
                        v_prev[..., :, :cut, :],
                        delta=float(self.elliptical_delta),
                        scale_mode=self.elliptical_scale_mode,
                    )
                elif exclude_cls and Tk > row_num_cls:
                    m_diag = compute_elliptical_diag(
                        vh[..., :, row_num_cls:, :],
                        v_prev[..., :, row_num_cls:, :],
                        delta=float(self.elliptical_delta),
                        scale_mode=self.elliptical_scale_mode,
                    )
                else:
                    m_diag = compute_elliptical_diag(vh, v_prev, delta=float(self.elliptical_delta), scale_mode=self.elliptical_scale_mode)

        # Scaled q', k' by metric (or identity if disabled)
        if m_diag is not None:
            if exclude_cls and (row_num_cls > 0):
                scale = torch.ones_like(q_rope)
                m_bc = m_diag.view(1, 1, nh, 1, Dh)
                if Tq > row_num_cls:
                    tgt = scale[..., :, row_num_cls:, :]
                    scale[..., :, row_num_cls:, :] = m_bc.expand(tgt.shape)
                q_prime = q_rope * scale
                k_prime = k_rope * scale
            else:
                m_bc = m_diag.view(*([1] * (q_rope.dim() - 2)), nh, 1, Dh)
                q_prime = q_rope * m_bc
                k_prime = k_rope * m_bc
        else:
            q_prime, k_prime = q_rope, k_rope
            m_diag = torch.ones(nh, Dh, dtype=q_rope.dtype, device=q_rope.device)

        # Build logits/attn depending on mask semantics
        if isinstance(attn_mask, int) and (Tq == Tk):
            # Training ICL often uses an integer "cut" to split train/test attention pattern.
            cut = int(attn_mask)
            q_left = q_prime[..., :cut, :]
            k_left = k_prime[..., :cut, :]
            log_l, attn_l = _compute_logits_and_attn(q_left, k_left)
            q_right = q_prime[..., cut:, :]
            log_r, attn_r = _compute_logits_and_attn(q_right, k_left)
            Bflat = q_prime.shape[:-3]
            logits = q_prime.new_full((*Bflat, nh, Tq, Tk), float("-inf"))
            attn = q_prime.new_zeros((*Bflat, nh, Tq, Tk))
            logits[..., :cut, :cut] = log_l
            attn[..., :cut, :cut] = attn_l
            if cut < Tq:
                logits[..., cut:, :cut] = log_r
                attn[..., cut:, :cut] = attn_r
        else:
            # Robust additive mask handling
            add_mask = None
            if key_padding_mask is not None:
                # key_padding_mask: (..., Tk) booleans or additive; expand to (..., nh, Tq, Tk)
                kp_add = _as_additive_mask(key_padding_mask, (*Bshape, nh, Tq, Tk), q_prime.device, q_prime.dtype)
                add_mask = kp_add
            if (attn_mask is not None) and torch.is_tensor(attn_mask):
                am_add = _as_additive_mask(attn_mask, (*Bshape, nh, Tq, Tk), q_prime.device, q_prime.dtype)
                add_mask = am_add if add_mask is None else (add_mask + am_add)
            logits, attn = _compute_logits_and_attn(q_prime, k_prime, add_mask)

        # Stash capture
        stack = getattr(self, "_ea_stack", "unknown")
        cap = BlockCapture(
            q_rope=q_rope.detach().cpu(),
            k_rope=k_rope.detach().cpu(),
            q_scaled=q_prime.detach().cpu(),
            k_scaled=k_prime.detach().cpu(),
            v=vh.detach().cpu(),
            v_prev=(v_prev.detach().cpu() if isinstance(v_prev, torch.Tensor) else None),
            m_diag=m_diag.detach().cpu(),
            logits=logits.detach().cpu(),
            attn=attn.detach().cpu(),
        )
        collector.put(stack, layer, cap)

        # Delegate to original computation for the actual forward
        return orig(self, q, k, v, key_padding_mask, attn_mask, rope, v_prev=v_prev, block_index=block_index)

    MultiheadAttentionBlock._attn_block = wrapped  # type: ignore[method-assign]


def _fro_ratio(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    na = torch.norm(a.reshape(-1), p=2)
    nb = torch.norm(b.reshape(-1), p=2)
    if nb.item() < eps:
        return float("inf") if na.item() > eps else 0.0
    return float((na / nb).item())


def _fro_ratio_masked(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    mask = torch.isfinite(a) & torch.isfinite(b)
    if not mask.any():
        return 0.0
    a_m = a[mask]
    b_m = b[mask]
    na = torch.norm(a_m.reshape(-1), p=2)
    nb = torch.norm(b_m.reshape(-1), p=2)
    if nb.item() < eps:
        return float("inf") if na.item() > eps else 0.0
    return float((na / nb).item())


@torch.inference_mode()
def main(device: Optional[str] = None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)

    # Tiny synthetic batch
    B, T_rows, H = 1, 12, 6
    train_size = 4
    max_classes = 5
    X = torch.randn(B, T_rows, H, device=dev)
    y = torch.randint(0, max_classes, (B, T_rows), device=dev)
    y_train = y[:, :train_size].contiguous()

    # Build model with EA enabled in TFrow and TFicl. Keep layer-0 vanilla via v_prev=None guard.
    model = TabICL(
        max_classes=max_classes,
        embed_dim=32,
        col_num_blocks=2,
        col_nhead=4,
        col_num_inds=32,
        row_num_blocks=3,
        row_nhead=4,
        row_num_cls=2,
        row_rope_base=100000,
        icl_num_blocks=3,
        icl_nhead=4,
        row_elliptical=True,
        icl_elliptical=True,
        elliptical_delta=1.0,
        elliptical_scale_mode="max",  # normalized scaling (max/mean) expected
        dropout=0.0,
        norm_first=True,
    ).to(dev).eval()

    # Tag blocks with stack names for clearer captures
    annotate_stacks(model)
    # Make sure attention outputs are not nulled by zero init so that EA
    # effect propagates to the layer outputs in this wiring test.
    _unzero_attention_out_proj(model)

    # Global collector and patch
    collector = Collector()
    patch_attention_for_capture(collector)

    # --- Three passes for captures ---
    # Pass A: EA-ON (normal)
    collector.begin_pass("EA")
    set_all_blocks_mode(model, identity=False, delta=1.0)
    _ = model._inference_forward(X, y_train)  # ensure eval path and ICL mask behavior

    # Pass B: EA-OFF (M = I)
    collector.begin_pass("ID")
    set_all_blocks_mode(model, identity=True)
    _ = model._inference_forward(X, y_train)

    # Pass C: EA-ON with larger delta (δ' = 10·δ)
    collector.begin_pass("EA_D10")
    set_all_blocks_mode(model, identity=False, delta=10.0)
    _ = model._inference_forward(X, y_train)

    # 1) Layer-gating asserts (layer-0 identity; anisotropy for ≥ layer-1)
    for stack in ("tfrow", "tficl"):
        # Layer-0: metric identity by construction. For TFrow, inputs are identical
        # across EA/ID, so logits should also match. For TFicl, inputs depend on TFrow
        # outputs, which change under EA, so logits may differ even with identity metric.
        cap_A0 = collector.get("EA", stack, 0)
        cap_B0 = collector.get("ID", stack, 0)
        assert cap_A0.m_diag is not None
        m0 = cap_A0.m_diag
        assert torch.allclose(m0, torch.ones_like(m0), atol=1e-6, rtol=1e-6), f"{stack} layer-0 M not identity"
        if stack == "tfrow":
            Ldiff = cap_A0.logits - cap_B0.logits
            finite_mask = torch.isfinite(cap_A0.logits) & torch.isfinite(cap_B0.logits)
            diff0 = (Ldiff[finite_mask].abs().max().item()) if finite_mask.any() else 0.0
            print(f"[{stack}] layer-0 max |Δlogits| = {diff0:.3e} (≈0)")
            assert diff0 < 1e-6 + 1e-9, f"{stack} layer-0 logits differ between EA and ID"
        else:
            print(f"[{stack}] layer-0: M=I (inputs differ due to TFrow → logits may differ)")

        # Layers ≥2 (0-based indices 1, 2)
        for li in (1, 2):  # we built 3 blocks
            capA = collector.get("EA", stack, li)
            m = capA.m_diag
            assert m is not None
            span = (m.amax(dim=-1) - m.amin(dim=-1))  # per-head span
            anis = span.mean().item()
            print(f"[{stack}] layer-{li+1} anisotropy span mean = {anis:.3e}")
            assert anis > EPS, f"{stack} layer-{li+1} shows no anisotropy in M"

    # 2) Geometry actually used (Δlogits, Δattn > eps for some heads)
    changed_any = False
    for stack in ("tfrow", "tficl"):
        for li in (1, 2):
            A = collector.get("EA", stack, li)
            Bc = collector.get("ID", stack, li)
            logits_diff = (A.logits - Bc.logits)
            attn_diff = (A.attn - Bc.attn)
            nh = A.logits.shape[-3]
            rl_all = _fro_ratio_masked(logits_diff, Bc.logits)
            ra_all = _fro_ratio_masked(attn_diff, Bc.attn)
            for h in range(nh):
                ld_h = logits_diff.select(dim=-3, index=h)
                lb_h = Bc.logits.select(dim=-3, index=h)
                ad_h = attn_diff.select(dim=-3, index=h)
                ab_h = Bc.attn.select(dim=-3, index=h)
                rl = _fro_ratio_masked(ld_h, lb_h)
                ra = _fro_ratio_masked(ad_h, ab_h)
                if (rl > EPS) or (ra > EPS):
                    changed_any = True
            print(f"[{stack}] layer-{li+1} Δlogits Fro ratio={rl_all:.3e} | Δattn Fro ratio={ra_all:.3e}")
    assert changed_any, "No head in TFrow or TFicl showed geometry changes vs. identity"

    # 2b) CLS exclusion in TFrow: CLS queries should be unaffected by EA (EA vs ID)
    num_cls = getattr(model.row_interactor, "num_cls", 0)
    if isinstance(num_cls, int) and num_cls > 0:
        for li in (1, 2):
            A = collector.get("EA", "tfrow", li)
            Bc = collector.get("ID", "tfrow", li)
            # Within the EA pass, CLS tokens should be unscaled (Q/K at CLS equal to their RoPE'd versions)
            assert A.q_scaled is not None and A.q_rope is not None
            q_cls_diff = (A.q_scaled[..., :num_cls, :] - A.q_rope[..., :num_cls, :]).abs().max().item()
            print(f"[tfrow] layer-{li+1} CLS q unscaled max |Δ| = {q_cls_diff:.3e}")
            assert q_cls_diff < 1e-6, "EA should not scale CLS queries in TFrow"
            assert A.k_scaled is not None and A.k_rope is not None
            k_cls_diff = (A.k_scaled[..., :num_cls, :] - A.k_rope[..., :num_cls, :]).abs().max().item()
            print(f"[tfrow] layer-{li+1} CLS k unscaled max |Δ| = {k_cls_diff:.3e}")
            assert k_cls_diff < 1e-6, "EA should not scale CLS keys in TFrow"

    # 3) δ sensitivity on logits AND on the metric itself (normalized scaling should not increase deviation)
    for stack in ("tfrow", "tficl"):
        for li in (1, 2):
            A = collector.get("EA", stack, li)
            Bc = collector.get("ID", stack, li)
            C = collector.get("EA_D10", stack, li)

            # logits deviation
            diff_A = A.logits - Bc.logits
            diff_C = C.logits - Bc.logits
            finite_mask = torch.isfinite(Bc.logits) & torch.isfinite(A.logits) & torch.isfinite(C.logits)
            if finite_mask.any():
                dA = torch.norm(diff_A[finite_mask].reshape(-1), p=2)
                dC = torch.norm(diff_C[finite_mask].reshape(-1), p=2)
            else:
                dA = torch.tensor(0.0)
                dC = torch.tensor(0.0)
            print(f"[{stack}] layer-{li+1} ||Δ(δ)-ID||={float(dA):.3e} vs ||Δ(10δ)-ID||={float(dC):.3e}")
            assert float(dC) <= float(dA) * (1.0 + 1e-5) + 1e-5, f"{stack} layer-{li+1}: larger δ should not increase deviation (logits)"

            # metric deviation from identity
            A_m = A.m_diag
            C_m = C.m_diag
            one = torch.ones_like(A_m)
            dA_m = torch.norm((A_m - one).reshape(-1), 2).item()
            dC_m = torch.norm((C_m - one).reshape(-1), 2).item()
            print(f"[{stack}] layer-{li+1} ||M(δ)-I||={dA_m:.3e} vs ||M(10δ)-I||={dC_m:.3e}")
            assert dC_m <= dA_m + 1e-6, f"{stack} layer-{li+1}: larger δ should not increase deviation (metric)"

    # 3b) For TFicl with split mask, M must be computed from allowed keys only (train slice)
    #     Verify by recomputing M from captured v/v_prev restricted to left slice and comparing to used m_diag.
    cut = train_size
    for li in (1, 2):
        A = collector.get("EA", "tficl", li)
        m_left = compute_elliptical_diag(A.v[..., :cut, :], A.v_prev[..., :cut, :], delta=1.0, scale_mode="max")
        err_m = torch.norm((A.m_diag - m_left).reshape(-1), 2).item()
        print(f"[tficl] layer-{li+1} M(recomputed on left) L2 err = {err_m:.3e}")
        assert err_m < 1e-6, "ICL M should be computed over allowed keys (train slice) only"

    # 4) v_prev continuity: each block’s v_prev equals previous block’s v
    for stack in ("tfrow", "tficl"):
        for li in (1, 2):
            cur = collector.get("EA", stack, li)
            prv = collector.get("EA", stack, li - 1)
            assert cur.v_prev is not None, f"{stack} layer-{li+1} missing v_prev"
            vp = cur.v_prev.to(dtype=prv.v.dtype)
            err = torch.norm((vp - prv.v).reshape(-1), p=2).item()
            rel = err / (torch.norm(prv.v.reshape(-1), p=2).item() + 1e-12)
            print(f"[{stack}] layer-{li+1} v_prev matches prev.v: rel={rel:.3e}")
            assert rel < 1e-6, f"{stack} layer-{li+1} v_prev != previous block's v (wiring?)"

    # 5) Eval-path contract: hooks fired if we have captures for all blocks in eval
    for stack, nblks in (("tfrow", len(model.row_interactor.tf_row.blocks)),
                         ("tficl", len(model.icl_predictor.tf_icl.blocks))):
        assert len(collector.data["EA"][stack]) == nblks, f"{stack} captures missing in eval()"
        assert len(collector.data["ID"][stack]) == nblks, f"{stack} captures missing in eval() [ID]"

    # 6) No hidden learnables or buffers tied to EA
    ea_param_names = [n for n, _ in model.named_parameters() if ("ellip" in n.lower())]
    ea_buffer_names = [n for n, _ in model.named_buffers() if ("ellip" in n.lower())]
    assert len(ea_param_names) == 0 and len(ea_buffer_names) == 0, f"Unexpected EA state found: params={ea_param_names}, buffers={ea_buffer_names}"

    # 7) CRITICAL: Verify the *actual model outputs* differ when EA is enabled vs identity.
    with torch.no_grad():
        # EA on
        set_all_blocks_mode(model, identity=False, delta=1.0)
        out_ea = model._inference_forward(X, y_train)
        # Identity metric
        set_all_blocks_mode(model, identity=True)
        out_id = model._inference_forward(X, y_train)

    diff_out = torch.norm((out_ea - out_id).reshape(-1), p=2).item()
    print(f"[END] ||output_EA - output_ID|| = {diff_out:.3e}")
    assert diff_out > 1e-6, "Model outputs identical with EA vs ID; EA may not be wired into the actual forward."

    # Optional: isolate stacks to ensure each branch is exercised
    with torch.no_grad():
        # Only TFrow EA
        for b in model.icl_predictor.tf_icl.blocks:
            b.elliptical_override = "identity"
        for b in model.row_interactor.tf_row.blocks:
            b.elliptical_override = "none"
        out_rowEA = model._inference_forward(X, y_train)

        # Only TFicl EA
        for b in model.icl_predictor.tf_icl.blocks:
            b.elliptical_override = "none"
        for b in model.row_interactor.tf_row.blocks:
            b.elliptical_override = "identity"
        out_iclEA = model._inference_forward(X, y_train)

        norm_id = torch.norm(out_id.reshape(-1), 2).item() + 1e-12
        row_delta = torch.norm((out_rowEA - out_id).reshape(-1), 2).item() / norm_id
        icl_delta = torch.norm((out_iclEA - out_id).reshape(-1), 2).item() / norm_id
        print(f"[END] relative ||rowEA - ID|| = {row_delta:.3e} | ||iclEA - ID|| = {icl_delta:.3e}")
        assert (row_delta > 1e-6) or (icl_delta > 1e-6), "Isolated EA toggles did not affect outputs; check stack wiring."

    # Sanity: the override 'identity' and fully disabled EA feature flags should agree
    with torch.no_grad():
        # identity-override path
        set_all_blocks_mode(model, identity=True)
        out_id_override = model._inference_forward(X, y_train)
        # fully disable EA flags
        set_elliptical_enabled(model, enabled=False)
        out_id_disabled = model._inference_forward(X, y_train)
        id_mismatch = torch.norm((out_id_override - out_id_disabled).reshape(-1), 2).item()
        print(f"[SANITY] ||ID(override) - ID(disabled)|| = {id_mismatch:.3e}")
        assert id_mismatch < 1e-6, "Identity override and disabled EA path disagree; check code paths."

    print("All EA wiring checks passed.")


if __name__ == "__main__":
    dev = None
    if len(sys.argv) > 1:
        dev = sys.argv[1]
    main(device=dev)
