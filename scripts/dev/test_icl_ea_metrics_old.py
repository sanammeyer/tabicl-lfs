#!/usr/bin/env python3
"""Quick TFicl EA sanity metrics (no training).

Computes, per test row i:
  1) Attention entropy
  2) Jaccard@k overlap vs. baseline (neighbors on train rows)
  3) Label-agreement@k
  4) Mask-TopK Δ (proxy): drop in correct-label attention mass after masking top-k
  5) Distance-view sanity: top-k by s_ij = q^T M k aligns with attention top-k

Run: python scripts/dev/test_icl_ea_metrics.py
"""

import os
import sys
import argparse
import random
import inspect
import numpy as np
import torch
import torch.nn.functional as F

# Ensure local src package is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tabicl.model.tabicl import TabICL
from tabicl.sklearn.classifier import TabICLClassifier


def set_seed(seed: int = 7, deterministic: bool = True):
    """Set seeds for reproducibility across Python, NumPy, and Torch.

    If deterministic is True, also toggle PyTorch deterministic algorithms and cudnn flags.
    Note: For full CUDA matmul determinism, consider launching with
    CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) in the environment before Python starts.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Best-effort set; may be too late if torch already initialized certain backends
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def make_synth(B=1, T=64, H=8, num_classes=3, device="cpu", seed=0):
    torch.manual_seed(int(seed))
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
    seed: int = 7,
    deterministic: bool = True,
):
    set_seed(int(seed), deterministic=deterministic)
    B, T, H = 1, 64, 8
    train_size = 16

    X, y, d, num_classes = make_synth(B=B, T=T, H=H, device=device, seed=seed)
    y_train = y[:, :train_size]
    y_test = y[:, train_size:]

    # Build model; enable EA only in ICL
    if use_pretrained or ckpt_path:
        # Load pretrained via sklearn wrapper (handles HF download + state dict mapping)
        clf = TabICLClassifier(model_path=ckpt_path, allow_auto_download=True, verbose=False)
        clf._load_model()
        model = clf.model_.to(device).eval()
    else:
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
        # Deterministic optimizer configuration where possible
        adam_kwargs = {}
        try:
            if 'foreach' in inspect.signature(torch.optim.Adam).parameters:
                adam_kwargs['foreach'] = False
            if 'fused' in inspect.signature(torch.optim.Adam).parameters:
                adam_kwargs['fused'] = False
        except Exception:
            pass
        opt = torch.optim.Adam(params, lr=float(micro_lr), **adam_kwargs)
        # Identity geometry during micro-fit
        for blk in model.icl_predictor.tf_icl.blocks:
            blk.elliptical = True
            blk.elliptical_override = "identity"
            blk.elliptical_manual_m = None

        # Re-enable gradients inside micro-fit despite the outer @torch.inference_mode on run_eval
        with torch.inference_mode(False):
            with torch.enable_grad():
                model.train()
                Bsz = X.shape[0]
                for step in range(int(micro_steps)):
                    # Rebuild R without grad (frozen col/row)
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
                        # Convert to a normal tensor (not an inference tensor) for autograd
                        R_lab = R_lab.clone()
                    else:
                        num_cls_eff = int(num_classes)
                        y_1h = F.one_hot(y_train.long(), num_classes=num_cls_eff).to(dtype=R.dtype)
                        rep = (D + num_cls_eff - 1) // num_cls_eff
                        y_enc = y_1h.repeat_interleave(rep, dim=-1)[..., :D]
                        R_lab = R.clone()
                        train_slice = R[:, :train_size, :]
                        g = (train_slice.norm() / (y_enc.norm() + 1e-12)).item()
                        R_lab[:, :train_size, :] = R_lab[:, :train_size, :] + (label_gain * g) * y_enc
                        # Convert to a normal tensor (not an inference tensor) for autograd
                        R_lab = R_lab.clone()

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

    # Optional: attention-readout accuracy (argmax of attention-weighted label distribution)
    C = int(y.max().item() + 1)
    y_train_oh = F.one_hot(y_train[0].long(), num_classes=C).float()
    P_base = A_base @ y_train_oh
    P_ea = A_ea @ y_train_oh
    acc_base = float((P_base.argmax(-1) == y_test[0]).float().mean().item())
    acc_ea = float((P_ea.argmax(-1) == y_test[0]).float().mean().item())
    print(f"[attn-readout acc] base={acc_base:.3f} ea={acc_ea:.3f}")

    test_len = T - train_size
    # Ensure k never exceeds number of train tokens
    k_eff = int(min(int(k), train_size))
    print(f"B={B}, T={T}, train_size={train_size}, test_len={test_len}, k={k_eff}")
    if m_ea is not None:
        cv = (m_ea.std(dim=-1) / (m_ea.mean(dim=-1) + 1e-12)).mean().item()
        print(f"[diag-M anisotropy] head-avg CV = {cv:.3f}")
        print(f"[diag-M stats] mean={m_ea.mean().item():.3f} std={m_ea.std().item():.3f} max={m_ea.max().item():.3f}")

    # Distance-view s_ij sanity
    # Use last block q,k and m to compute s_ij; we reuse compute_lastblock_attn’s inputs via recomputation
    # (only for EA; baseline uses m=None)
    with torch.no_grad():
        # Recompute q,k for last block
        last_blk = blocks[-1]
        x = R_lab.clone()
        for blk in blocks[:-1]:
            x = blk(x, attn_mask=train_size)
        q_in = last_blk.norm1(x)
        k_in = last_blk.norm1(x)
        v_in = last_blk.norm1(x)
        nh = last_blk.attn.num_heads
        head_dim = x.shape[-1] // nh
        q_lin, k_lin, _ = F._in_projection_packed(q_in, k_in, v_in, last_blk.attn.in_proj_weight, last_blk.attn.in_proj_bias)
        q = q_lin.view(B, T, nh, head_dim).transpose(1, 2)
        keys = k_lin.view(B, T, nh, head_dim).transpose(1, 2)
        q_r = q[..., train_size:, :]
        k_l = keys[..., :train_size, :]

    def topk_from_scores(scores: torch.Tensor, kk: int) -> np.ndarray:
        # scores: (B, nh, test_len, train_len)
        s = scores.mean(dim=1)[0]  # (test_len, train_len)
        _, idx = torch.topk(s, k=kk, dim=-1)
        return idx.cpu().numpy()

    # s_ij baseline and EA
    scale = float(head_dim) ** -0.5
    s_base = torch.matmul(q_r, k_l.transpose(-1, -2)) * scale
    if m_ea is not None:
        m = m_ea.view(1, nh, 1, head_dim)
        s_e = torch.matmul(q_r * m, (k_l * m).transpose(-1, -2)) * scale
    else:
        s_e = s_base.clone()

    # Keep indices as torch tensors for safe tensor indexing
    idx_attn_base_t = torch.topk(A_base, k=k_eff, dim=-1).indices  # (test_len, k_eff)
    idx_attn_ea_t = torch.topk(A_ea, k=k_eff, dim=-1).indices
    idx_s_base = topk_from_scores(s_base, k_eff)
    idx_s_e = topk_from_scores(s_e, k_eff)

    # Per-test-row logs (head-mean)
    for i in range(test_len):
        p_b = A_base[i]
        p_e = A_ea[i]
        H_b = float(entropy(p_b).item())
        H_e = float(entropy(p_e).item())
        pmax_b = float(p_b.max().item())
        pmax_e = float(p_e.max().item())

        nb_b_t = idx_attn_base_t[i]
        nb_e_t = idx_attn_ea_t[i]
        jac = jaccard(nb_b_t.cpu().numpy(), nb_e_t.cpu().numpy())

        # Label agreement@k
        yt = int(y_test[0, i].item())
        ya_b = float((y_train[0, nb_b_t] == yt).float().mean().item())
        ya_e = float((y_train[0, nb_e_t] == yt).float().mean().item())

        # Mask-TopK Δ (proxy): drop in correct-label attention mass after masking & renorm
        correct_mask = (y_train[0] == yt).float()
        mass_b = float((p_b * correct_mask).sum().item())
        mass_e = float((p_e * correct_mask).sum().item())
        def drop_mass(p, nb):
            p2 = p.clone()
            p2[nb] = 0.0
            p2 = p2 / (p2.sum() + 1e-12)
            return float((p2 * correct_mask).sum().item())
        mass_b_drop = drop_mass(p_b, nb_b_t)
        mass_e_drop = drop_mass(p_e, nb_e_t)
        dlt_b = mass_b - mass_b_drop
        dlt_e = mass_e - mass_e_drop

        # Distance-view sanity
        s_nb_b = idx_s_base[i]
        s_nb_e = idx_s_e[i]
        cov_b = jaccard(nb_b_t.cpu().numpy(), s_nb_b)
        cov_e = jaccard(nb_e_t.cpu().numpy(), s_nb_e)

        linf = float((p_e - p_b).abs().max().item())
        print(
            f"row={i:02d} H: base={H_b:.3f} ea={H_e:.3f} | J@{k_eff}={jac:.2f} | LA@{k_eff}: base={ya_b:.2f} ea={ya_e:.2f} | "
            f"Δmask: base={dlt_b:.3f} ea={dlt_e:.3f} | s-topk cov: base={cov_b:.2f} ea={cov_e:.2f} | "
            f"pmax: base={pmax_b:.3f} ea={pmax_e:.3f} | ||ΔA||_inf={linf:.3e}"
        )

    # Small k-sweep summary (averaged over heads and test rows) + requested LA@k print
    k_list_eff = [int(min(int(kk), train_size)) for kk in list(k_list)]
    metrics_out = {"la_base": {}, "la_ea": {}, "jaccard_mean": {}}
    for kk in k_list_eff:
        idx_b = torch.topk(A_base, k=kk, dim=-1).indices
        idx_e = torch.topk(A_ea, k=kk, dim=-1).indices
        # Jaccard needs numpy for set operations
        j_scores = [jaccard(idx_b[i].cpu().numpy(), idx_e[i].cpu().numpy()) for i in range(test_len)]

        def la_from_idx(idx: torch.Tensor) -> float:
            idx_t = idx.to(y_train.device)
            y_neighbors = y_train[0, idx_t]
            y_targets = y_test[0].unsqueeze(-1).expand_as(y_neighbors)
            return (y_neighbors == y_targets).float().mean().item()

        la_k_base = la_from_idx(idx_b)
        la_k_ea = la_from_idx(idx_e)
        print(f"[k={kk}] J@k mean={np.mean(j_scores):.2f} | LA@k mean: base={la_k_base:.2f} ea={la_k_ea:.2f}")
        # Requested explicit LA line
        print(f"[LA@{kk}] base={la_k_base:.3f} ea={la_k_ea:.3f}")
        metrics_out["la_base"][kk] = la_k_base
        metrics_out["la_ea"][kk] = la_k_ea
        metrics_out["jaccard_mean"][kk] = float(np.mean(j_scores))

    # Optional: per-head diagnostics before averaging
    if headwise:
        nh = A_base_h.shape[0]
        print("[per-head summary] heads=", nh)
        for h in range(nh):
            Ab = A_base_h[h] / (A_base_h[h].sum(dim=-1, keepdim=True) + 1e-12)
            Ae = A_ea_h[h] / (A_ea_h[h].sum(dim=-1, keepdim=True) + 1e-12)
            diffs = (Ae - Ab).abs().amax(dim=-1).cpu().numpy()
            print(f"  head {h}: mean ||ΔA||_inf={np.mean(diffs):.3e}, max={np.max(diffs):.3e}")

    return metrics_out


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
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of seeds to sweep")
    parser.add_argument(
        "--no_deterministic",
        action="store_true",
        help="Disable deterministic algorithms/flags (enabled by default)",
    )
    args = parser.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    deterministic = not bool(args.no_deterministic)
    if args.seeds:
        seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
        agg_la_base = {}
        agg_la_ea = {}
        count = 0
        for s in seed_list:
            metrics = run_eval(
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
                seed=int(s),
                deterministic=deterministic,
            )
            for kk, v in metrics["la_base"].items():
                agg_la_base[kk] = agg_la_base.get(kk, 0.0) + float(v)
            for kk, v in metrics["la_ea"].items():
                agg_la_ea[kk] = agg_la_ea.get(kk, 0.0) + float(v)
            count += 1
        if count > 0:
            print("\n[Seed sweep averages]")
            for kk in sorted(agg_la_base.keys()):
                base_avg = agg_la_base[kk] / count
                ea_avg = agg_la_ea[kk] / count
                print(f"[LA@{kk}] base={base_avg:.3f} ea={ea_avg:.3f}")
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
            seed=int(args.seed),
            deterministic=deterministic,
        )
