#!/usr/bin/env python3
"""Quick TFicl EA sanity metrics (no training).

Computes, per test row i:
  1) Attention entropy
  2) Jaccard@k overlap vs. baseline (neighbors on train rows)
  3) Label-agreement@k
  4) Mask-TopK Δ (proxy): drop in correct-label attention mass after masking top-k
  5) Distance-view sanity: top-k by s_ij = q^T M k aligns with attention top-k

Run: python scripts/test_icl_ea_metrics.py
"""

import os
import sys
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Ensure local src package is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tabicl.model.tabicl import TabICL
from tabicl.sklearn.classifier import TabICLClassifier


def make_synth(B=1, T=64, H=8, num_classes=3, device="cpu", seed=0):
    torch.manual_seed(seed)
    X = torch.randn(B, T, H, device=device)
    # Let label depend on feature 0 quantiles for structure
    x0 = X[..., 0]
    q = torch.quantile(x0, torch.tensor([0.33, 0.66], device=device))
    y = torch.zeros(B, T, device=device)
    y = torch.where(x0 < q[0], torch.zeros_like(y), torch.where(x0 < q[1], torch.ones_like(y), 2 * torch.ones_like(y)))
    y = y.long()
    # Variable number of features per dataset (for masking test behavior); here fixed
    d = torch.full((B,), H, device=device, dtype=torch.long)
    return X, y, d, num_classes


@torch.inference_mode()
def build_row_reprs(model: TabICL, X: torch.Tensor, y: torch.Tensor, d: torch.Tensor, train_size: int):
    # Freeze stages 1-2 by evaluating once
    emb = model.col_embedder._train_forward(X, d=d, train_size=train_size)
    R = model.row_interactor._train_forward(emb, d=d)
    return R


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a_set, b_set = set(a.tolist()), set(b.tolist())
    if not a_set and not b_set:
        return 1.0
    return float(len(a_set & b_set)) / float(len(a_set | b_set))


def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


@torch.inference_mode()
def compute_lastblock_attn(
    R: torch.Tensor,
    y_train: torch.Tensor,
    blocks,
    train_size: int,
    use_ea: bool,
    temp: float = 1.0,
    manual_m: torch.Tensor | None = None,
) -> tuple:
    """Iterate TFicl blocks, collect last block test->train attention and M (per-head diag) for distance sanity.

    Returns (attn: (B, heads, test_len, train_len), m_diag: (heads, head_dim), top-level x after last block)
    """
    B, T, E = R.shape
    x = R.clone()
    cut = int(train_size)

    # R is expected to already include label conditioning on the training slice

    v_prev = None
    attn_last = None
    m_last = None
    for idx, blk in enumerate(blocks):
        # Pre-norm projections
        q_in = blk.norm1(x)
        k_in = blk.norm1(x)
        v_in = blk.norm1(x)
        nh = blk.attn.num_heads
        head_dim = E // nh

        q_lin, k_lin, v_lin = F._in_projection_packed(q_in, k_in, v_in, blk.attn.in_proj_weight, blk.attn.in_proj_bias)
        q = q_lin.view(B, T, nh, head_dim).transpose(1, 2)
        k = k_lin.view(B, T, nh, head_dim).transpose(1, 2)
        v = v_lin.view(B, T, nh, head_dim).transpose(1, 2)

        # EA scaling (from layer 2 onward)
        m = None
        if manual_m is not None:
            m = manual_m  # (nh, Dh)
            q = q * m.view(1, nh, 1, head_dim)
            k = k * m.view(1, nh, 1, head_dim)
        elif use_ea and (v_prev is not None):
            # compute per-head diag via finite differences
            vd = (v - v_prev).abs().mean(dim=(0, 2))  # (nh, Dh)
            denom = vd.amax(dim=-1, keepdim=True).clamp_min(1e-12)
            m = vd / denom
            q = q * m.view(1, nh, 1, head_dim)
            k = k * m.view(1, nh, 1, head_dim)
        if idx == 0:
            assert m is None, "EA must be inactive on first ICL block (no previous V)."

        scale = float(head_dim) ** -0.5

        # Collect last block test->train attention
        if idx == len(blocks) - 1:
            q_right = q[..., cut:, :]
            k_left = k[..., :cut, :]
            scores = torch.matmul(q_right, k_left.transpose(-1, -2)) * scale
            attn = (scores / float(temp)).softmax(dim=-1)
            attn_last = attn  # (B, nh, test_len, train_len)
            m_last = m

        # Advance x through the actual block (uses internal EA path), to retain consistency
        x = blk(x, attn_mask=train_size)
        v_prev = v

    return attn_last, m_last, x


def run_eval(
    device: str = "cpu",
    k: int = 5,
    temp: float = 1.0,
    label_gain: float = 2.0,
    manual_m: bool = False,
    manual_m_scale: float = 5.0,
    manual_m_ratio: float = 0.25,
    headwise: bool = False,
    k_list: list[int] = [1, 3, 5, 8],
    use_pretrained: bool = False,
    ckpt_path: str | None = None,
    micro_fit: bool = False,
    micro_steps: int = 200,
    micro_lr: float = 1e-3,
    freeze_first_icl: bool = True,
    seed_data: int = 42,
    seed_model: int = 42,
    seed_fit: int = 42,
):
    B, T, H = 1, 64, 8
    train_size = 16

    X, y, d, num_classes = make_synth(B=B, T=T, H=H, device=device, seed=seed_data)
    y_train = y[:, :train_size]
    y_test = y[:, train_size:]

    # Build model; enable EA only in ICL
    if use_pretrained or ckpt_path:
        # Load pretrained via sklearn wrapper (handles HF download + state dict mapping)
        clf = TabICLClassifier(model_path=ckpt_path, allow_auto_download=True, verbose=False)
        clf._load_model()
        model = clf.model_.to(device).eval()
    else:
        # Seed model params for reproducibility
        torch.manual_seed(seed_model)
        model = TabICL(
            embed_dim=32,
            col_num_blocks=2,
            col_nhead=4,
            col_num_inds=64,
            row_num_blocks=2,
            row_nhead=4,
            row_num_cls=2,
            icl_num_blocks=2,
            icl_nhead=4,
            row_elliptical=False,
            icl_elliptical=True,
            elliptical_delta=1.0,
            elliptical_scale_mode="max",
        ).to(device).eval()

    # Stage 1-2 representations (frozen for both baselines)
    R = build_row_reprs(model, X, y, d, train_size)
    # Label conditioning on train slice: use model's y_encoder when pretrained, else tile one-hot
    D = int(R.shape[-1])
    if use_pretrained or ckpt_path:
        y_enc = model.icl_predictor.y_encoder(y_train.float())
        if y_enc.shape[-1] != D:
            pad = max(0, D - y_enc.shape[-1])
            y_enc = F.pad(y_enc, (0, pad))[:, :, :D]
        R_lab = R.clone()
        R_lab[:, :train_size, :] = R[:, :train_size, :] + y_enc
    else:
        # Parameter-free: repeat one-hot to R dim and scale by label_gain to be visible
        num_cls_eff = int(num_classes)
        y_1h = F.one_hot(y_train.long(), num_classes=num_cls_eff).to(dtype=R.dtype)
        rep = (D + num_cls_eff - 1) // num_cls_eff
        y_enc = y_1h.repeat_interleave(rep, dim=-1)[..., :D]  # (B, train_size, D)
        R_lab = R.clone()
        label_gain = float(label_gain)
        train_slice = R[:, :train_size, :]
        g = (train_slice.norm() / (y_enc.norm() + 1e-12)).item()
        R_lab[:, :train_size, :] = R_lab[:, :train_size, :] + (label_gain * g) * y_enc
    

    # Optional tiny identity-geometry fit: freeze everything except y_encoder + last ICL block
    if micro_fit:
        # Freeze col/row
        if seed_fit is not None:
            torch.manual_seed(seed_fit)
        for p in model.col_embedder.parameters():
            p.requires_grad = False
        for p in model.row_interactor.parameters():
            p.requires_grad = False
        # Optionally freeze first ICL block
        if freeze_first_icl and len(model.icl_predictor.tf_icl.blocks) > 1:
            for p in model.icl_predictor.tf_icl.blocks[0].parameters():
                p.requires_grad = False
        # Trainables
        last_blk = model.icl_predictor.tf_icl.blocks[-1]
        params = list(model.icl_predictor.y_encoder.parameters()) + list(last_blk.parameters())
        opt = torch.optim.Adam(params, lr=float(micro_lr))
        # Identity geometry during micro-fit
        for blk in model.icl_predictor.tf_icl.blocks:
            blk.elliptical = True
            blk.elliptical_override = "identity"
            blk.elliptical_manual_m = None

        model.train()
        Bsz = X.shape[0]
        for step in range(int(micro_steps)):
            # Rebuild R without grad
            with torch.no_grad():
                R = build_row_reprs(model, X, y, d, train_size)
            # Use the same label injection rule as above
            if use_pretrained or ckpt_path:
                y_enc = model.icl_predictor.y_encoder(y_train.float())
                if y_enc.shape[-1] != D:
                    pad = max(0, D - y_enc.shape[-1])
                    y_enc = F.pad(y_enc, (0, pad))[:, :, :D]
                R_lab = R.clone()
                R_lab[:, :train_size, :] = R[:, :train_size, :] + y_enc
                noise_std = 1e-3
                R_lab[:, :train_size, :] += noise_std * torch.randn_like(R_lab[:, :train_size, :])
            else:
                num_cls_eff = int(num_classes)
                y_1h = F.one_hot(y_train.long(), num_classes=num_cls_eff).to(dtype=R.dtype)
                rep = (D + num_cls_eff - 1) // num_cls_eff
                y_enc = y_1h.repeat_interleave(rep, dim=-1)[..., :D]
                R_lab = R.clone()
                train_slice = R[:, :train_size, :]
                g = (train_slice.norm() / (y_enc.norm() + 1e-12)).item()
                noise_std = 1e-3
                R_lab[:, :train_size, :] += noise_std * torch.randn_like(R_lab[:, :train_size, :])
                R_lab[:, :train_size, :] = R_lab[:, :train_size, :] + (label_gain * g) * y_enc

            # Forward up to last ICL block
            x_fit = R_lab
            for blk in model.icl_predictor.tf_icl.blocks[:-1]:
                x_fit = blk(x_fit, attn_mask=train_size)
            # Last block attention (identity geometry)
            q_in = last_blk.norm1(x_fit)
            k_in = q_in
            v_in = q_in
            q_lin, k_lin, v_lin = F._in_projection_packed(
                q_in, k_in, v_in, last_blk.attn.in_proj_weight, last_blk.attn.in_proj_bias
            )
            nh = last_blk.attn.num_heads
            Dh = D // nh
            q = q_lin.view(Bsz, T, nh, Dh).transpose(1, 2)
            keys_fit = k_lin.view(Bsz, T, nh, Dh).transpose(1, 2)
            q_r = q[..., train_size:, :]
            k_l = keys_fit[..., :train_size, :]
            scores = torch.matmul(q_r, k_l.transpose(-1, -2)) * (float(Dh) ** -0.5)
            A = scores.softmax(dim=-1).mean(dim=1)[0]  # (test_len, train_len), B==1
            # Label-mass CE loss
            C = int(y.max().item() + 1)
            y_train_oh = F.one_hot(y_train[0], num_classes=C).float()
            P = A @ y_train_oh  # (test_len, C)
            loss = F.nll_loss((P + 1e-8).log(), y[:, train_size:][0])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        # Recompute R_lab after micro-fit for fair eval
        with torch.no_grad():
            R = build_row_reprs(model, X, y, d, train_size)
            if use_pretrained or ckpt_path:
                y_enc = model.icl_predictor.y_encoder(y_train.float())
                if y_enc.shape[-1] != D:
                    pad = max(0, D - y_enc.shape[-1])
                    y_enc = F.pad(y_enc, (0, pad))[:, :, :D]
                R_lab = R.clone()
                R_lab[:, :train_size, :] = R[:, :train_size, :] + y_enc
            else:
                num_cls_eff = int(num_classes)
                y_1h = F.one_hot(y_train.long(), num_classes=num_cls_eff).to(dtype=R.dtype)
                rep = (D + num_cls_eff - 1) // num_cls_eff
                y_enc = y_1h.repeat_interleave(rep, dim=-1)[..., :D]
                R_lab = R.clone()
                train_slice = R[:, :train_size, :]
                g = (train_slice.norm() / (y_enc.norm() + 1e-12)).item()
                R_lab[:, :train_size, :] = R_lab[:, :train_size, :] + (label_gain * g) * y_enc

    # Convenience
    blocks = model.icl_predictor.tf_icl.blocks

    # Baseline (EA identity): force identity scaling in all blocks
    for blk in blocks:
        blk.elliptical = True
        blk.elliptical_override = "identity"
        blk.elliptical_manual_m = None
    attn_base, m_base, _ = compute_lastblock_attn(R_lab, y_train, blocks, train_size, use_ea=False, temp=temp)

    # EA enabled (parameter-free)
    for blk in blocks:
        blk.elliptical = True
        blk.elliptical_override = "none"
        blk.elliptical_manual_m = None

    manual_m_tensor = None
    if manual_m:
        nh = blocks[0].attn.num_heads
        Dh = R_lab.shape[-1] // nh
        manual_m_tensor = torch.ones(nh, Dh, device=R_lab.device, dtype=R_lab.dtype)
        d_cut = max(1, int(Dh * float(manual_m_ratio)))
        manual_m_tensor[:, :d_cut] = float(manual_m_scale)
        # Engage manual override inside blocks for the actual forward path
        for blk in blocks:
            blk.elliptical = True
            blk.elliptical_override = "manual"
            blk.elliptical_manual_m = manual_m_tensor

    attn_ea, m_ea, _ = compute_lastblock_attn(
        R_lab, y_train, blocks, train_size, use_ea=(not manual_m), temp=temp, manual_m=manual_m_tensor
    )

    # Aggregate over heads (mean), normalize across train tokens
    # Per-head and head-mean views
    A_base_h = attn_base[0]            # (heads, test_len, train_len)
    A_ea_h = attn_ea[0]
    A_base = A_base_h.mean(dim=0)      # (test_len, train_len)
    A_ea = A_ea_h.mean(dim=0)
    A_base = A_base / (A_base.sum(dim=-1, keepdim=True) + 1e-12)
    A_ea = A_ea / (A_ea.sum(dim=-1, keepdim=True) + 1e-12)

    test_len = T - train_size
    print(f"B={B}, T={T}, train_size={train_size}, test_len={test_len}, k={k}")
    if m_ea is not None:
        cv = (m_ea.std(dim=-1) / (m_ea.mean(dim=-1) + 1e-12)).mean().item()
        print(f"[diag-M anisotropy] head-avg CV = {cv:.3f}")
        print(f"[diag-M stats] mean={m_ea.mean().item():.3f} std={m_ea.std().item():.3f} max={m_ea.max().item():.3f}")

    # Distance-view s_ij sanity (eval-only; disable autograd to avoid inference-tensor issues)
    # Use last block q,k and m to compute s_ij; we reuse compute_lastblock_attn’s inputs via recomputation
    with torch.no_grad():
        last_blk = blocks[-1]
        x = R_lab.clone()
        for blk in blocks[:-1]:
            x = blk(x, attn_mask=train_size)
        q_in = last_blk.norm1(x)
        k_in = last_blk.norm1(x)
        v_in = last_blk.norm1(x)
        nh = last_blk.attn.num_heads
        head_dim = x.shape[-1] // nh
        q_lin, k_lin, _ = F._in_projection_packed(
            q_in, k_in, v_in, last_blk.attn.in_proj_weight, last_blk.attn.in_proj_bias
        )
        q = q_lin.view(B, T, nh, head_dim).transpose(1, 2)
        keys = k_lin.view(B, T, nh, head_dim).transpose(1, 2)
        q_r = q[..., train_size:, :]
        k_l = keys[..., :train_size, :]

        def topk_from_scores(scores: torch.Tensor) -> np.ndarray:
            # scores: (B, nh, test_len, train_len)
            s = scores.mean(dim=1)[0]  # (test_len, train_len)
            _, idx = torch.topk(s, k=k, dim=-1)
            return idx.cpu().numpy()

        # s_ij baseline and EA
        scale = float(head_dim) ** -0.5
        s_base = torch.matmul(q_r, k_l.transpose(-1, -2)) * scale
        if m_ea is not None:
            m = m_ea.detach().clone().to(dtype=q_r.dtype, device=q_r.device).view(1, nh, 1, head_dim)
            s_e = torch.matmul(q_r * m, (k_l * m).transpose(-1, -2)) * scale
        else:
            s_e = s_base.clone()

    idx_attn_base = torch.topk(A_base, k=int(k), dim=-1).indices.cpu().numpy()
    idx_attn_ea = torch.topk(A_ea, k=int(k), dim=-1).indices.cpu().numpy()
    idx_s_base = topk_from_scores(s_base)
    idx_s_e = topk_from_scores(s_e)

    idx_b_k = idx_attn_base
    idx_e_k = idx_attn_ea
    la_k_b = np.mean([
        (y_train[0, idx_b_k[i]] == y_test[0, i]).float().mean().item()
        for i in range(test_len)
    ])
    la_k_e = np.mean([
        (y_train[0, idx_e_k[i]] == y_test[0, i]).float().mean().item()
        for i in range(test_len)
    ])

    # Precompute one-hot of train labels for readout metrics
    C = int(y.max().item() + 1)
    y_train_oh_full = F.one_hot(y_train[0].long(), num_classes=C).float()  # (train_len, C)

    # Per-test-row logs (head-mean)
    for i in range(test_len):
        p_b = A_base[i]
        p_e = A_ea[i]
        H_b = float(entropy(p_b).item())
        H_e = float(entropy(p_e).item())
        pmax_b = float(p_b.max().item())
        pmax_e = float(p_e.max().item())

        nb_b = idx_attn_base[i]
        nb_e = idx_attn_ea[i]
        jac = jaccard(nb_b, nb_e)

        # Label agreement@k
        yt = int(y_test[0, i].item())
        ya_b = float((y_train[0, nb_b] == yt).float().mean().item())
        ya_e = float((y_train[0, nb_e] == yt).float().mean().item())

        # Top-k correct mass: sum_{j in top-k} p(j) * 1[y_j == y_t]
        correct_mask = (y_train[0] == yt).float()
        topk_b = torch.topk(p_b, k=int(k), dim=-1).indices
        topk_e = torch.topk(p_e, k=int(k), dim=-1).indices
        tk_mass_b = float((p_b[topk_b] * correct_mask[topk_b]).sum().item())
        tk_mass_e = float((p_e[topk_e] * correct_mask[topk_e]).sum().item())

        # Distance-view sanity
        s_nb_b = idx_s_base[i]
        s_nb_e = idx_s_e[i]
        cov_b = jaccard(nb_b, s_nb_b)
        cov_e = jaccard(nb_e, s_nb_e)

        linf = float((p_e - p_b).abs().max().item())
        print(
            f"row={i:02d} H: base={H_b:.3f} ea={H_e:.3f} | J@{k}={jac:.2f} | LA@{k}: base={ya_b:.2f} ea={ya_e:.2f} | "
            f"TK-correct-mass: base={tk_mass_b:.3f} ea={tk_mass_e:.3f} | s-topk cov: base={cov_b:.2f} ea={cov_e:.2f} | "
            f"pmax: base={pmax_b:.3f} ea={pmax_e:.3f} | ||ΔA||_inf={linf:.3e}"
        )

    # Small k-sweep summary (averaged over heads and test rows)
    k_list = list(k_list)
    for kk in k_list:
        idx_b = torch.topk(A_base, k=kk, dim=-1).indices.cpu().numpy()
        idx_e = torch.topk(A_ea, k=kk, dim=-1).indices.cpu().numpy()
        j_scores = [jaccard(idx_b[i], idx_e[i]) for i in range(test_len)]
        # Label agreement@k
        ya_b_list, ya_e_list = [], []
        for i in range(test_len):
            yt = int(y_test[0, i].item())
            ya_b_list.append(float((y_train[0, idx_b[i]] == yt).float().mean().item()))
            ya_e_list.append(float((y_train[0, idx_e[i]] == yt).float().mean().item()))
        print(
            f"[k={kk}] J@k mean={np.mean(j_scores):.2f} | LA@k mean: base={np.mean(ya_b_list):.2f} ea={np.mean(ya_e_list):.2f}"
        )

    # Optional: per-head diagnostics before averaging
    if headwise:
        nh = A_base_h.shape[0]
        print("[per-head summary] heads=", nh)
        for h in range(nh):
            Ab = A_base_h[h] / (A_base_h[h].sum(dim=-1, keepdim=True) + 1e-12)
            Ae = A_ea_h[h] / (A_ea_h[h].sum(dim=-1, keepdim=True) + 1e-12)
            diffs = (Ae - Ab).abs().amax(dim=-1).cpu().numpy()
            print(f"  head {h}: mean ||ΔA||_inf={np.mean(diffs):.3e}, max={np.max(diffs):.3e}")

    # Attention-readout classification accuracy (argmax over A @ onehot(y_train))
    P_base = A_base @ y_train_oh_full  # (test_len, C)
    P_ea = A_ea @ y_train_oh_full
    pred_base = P_base.argmax(dim=-1)
    pred_ea = P_ea.argmax(dim=-1)
    acc_base = float((pred_base == y_test[0]).float().mean().item())
    acc_ea = float((pred_ea == y_test[0]).float().mean().item())
    print(f"[attn-readout acc] base={acc_base:.3f} ea={acc_ea:.3f}")
    return {
        "acc_base": acc_base,
        "acc_ea": acc_ea,
        "label_acc_k_base": la_k_b,
        "label_acc_k_ea": la_k_e,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--k_list", type=str, default="1,3,5,8")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--label_gain", type=float, default=2.0)
    parser.add_argument("--manual_m", action="store_true")
    parser.add_argument("--manual_m_scale", type=float, default=5.0)
    parser.add_argument("--manual_m_ratio", type=float, default=0.25)
    parser.add_argument("--headwise", action="store_true")
    parser.add_argument("--use_pretrained", action="store_true", help="Load default pretrained checkpoint via HF")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional explicit checkpoint path (.ckpt)")
    parser.add_argument("--micro_fit", action="store_true", help="Run tiny identity-geometry fit before eval")
    parser.add_argument("--micro_steps", type=int, default=200)
    parser.add_argument("--micro_lr", type=float, default=1e-3)
    parser.add_argument("--no_freeze_first_icl", action="store_true")
    parser.add_argument("--seed_data", type=int, default=42)
    parser.add_argument("--seed_model", type=int, default=42)
    parser.add_argument("--seed_fit", type=int, default=42)
    parser.add_argument("--fit_seed_sweep", action="store_true", help="Run eval over multiple fit seeds")
    args = parser.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    if args.fit_seed_sweep:
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        acc_b, acc_e, la_k_b, la_k_e = [], [], [], []
        for seed in seeds:
            print(f"\n=== Running eval with micro-fit seed={seed} ===")
            metrics =run_eval(
                dev,
                k=int(args.k),
                temp=float(args.temp),
                label_gain=float(args.label_gain),
                manual_m=bool(args.manual_m),
                manual_m_scale=float(args.manual_m_scale),
                manual_m_ratio=float(args.manual_m_ratio),
                headwise=bool(args.headwise),
                k_list=k_list,
                use_pretrained=bool(args.use_pretrained),
                ckpt_path=args.ckpt,
                micro_fit=bool(args.micro_fit),
                micro_steps=int(args.micro_steps),
                micro_lr=float(args.micro_lr),
                freeze_first_icl=not bool(args.no_freeze_first_icl),
                seed_data=int(args.seed_data),
                seed_model=int(args.seed_model),
                seed_fit=seed,
            )
            acc_b.append(metrics["acc_base"])
            acc_e.append(metrics["acc_ea"])
            la_k_b.append(metrics["label_acc_k_base"])
            la_k_e.append(metrics["label_acc_k_ea"])
        print("\n=== Fit seed sweep summary ===")
        print(f"Attn-readout acc base: mean={np.mean(acc_b):.3f} std={np.std(acc_b):.3f}")
        print(f"Attn-readout acc ea:   mean={np.mean(acc_e):.3f} std={np.std(acc_e):.3f}")
        print(f"Label-acc@k base:     mean={np.mean(la_k_b):.3f} std={np.std(la_k_b):.3f}")
        print(f"Label-acc@k ea:       mean={np.mean(la_k_e):.3f} std={np.std(la_k_e):.3f}")
    else:
        run_eval(
            dev,
            k=int(args.k),
            temp=float(args.temp),
            label_gain=float(args.label_gain),
            manual_m=bool(args.manual_m),
            manual_m_scale=float(args.manual_m_scale),
            manual_m_ratio=float(args.manual_m_ratio),
            headwise=bool(args.headwise),
            k_list=k_list,
            use_pretrained=bool(args.use_pretrained),
            ckpt_path=args.ckpt,
            micro_fit=bool(args.micro_fit),
            micro_steps=int(args.micro_steps),
            micro_lr=float(args.micro_lr),
            freeze_first_icl=not bool(args.no_freeze_first_icl),
            seed_data=int(args.seed_data),
            seed_model=int(args.seed_model),
            seed_fit=int(args.seed_fit),
        )
