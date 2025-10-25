#!/usr/bin/env python
from __future__ import annotations

"""
Plot per-head mean and std of the elliptical scale m for ICL blocks.

Usage:
  python scripts/plot_elliptical_stats.py --ckpt checkpoints/step-2000.ckpt --out results/elliptical_stats_step-2000.png

Notes:
  - m is computed as softplus(elliptical_m_raw) per head-dimension (no normalization).
  - The figure contains two histograms: per-head mean(m) and per-head std(m).
  - Prints summary statistics to stdout.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_matrices(ckpt_path: Path) -> np.ndarray:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    sd = ckpt["state_dict"]
    # Collect m across all ICL blocks
    mats = []
    # Try to locate number of blocks from keys; fallback to 48
    keys = [k for k in sd.keys() if k.startswith("icl_predictor.tf_icl.blocks.") and k.endswith("elliptical_m_raw")]
    if not keys:
        return np.zeros((0,))
    for k in sorted(keys, key=lambda s: int(s.split(".")[3])):
        m_raw = sd[k]
        m = torch.nn.functional.softplus(m_raw).detach().cpu().numpy()  # (nhead, head_dim)
        mats.append(m)
    arr = np.stack(mats)  # (num_blocks, nhead, head_dim)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .ckpt file")
    ap.add_argument("--out", type=str, default=None, help="Output image path (.png)")
    ap.add_argument("--attn_diag_len", type=int, default=0, help="If >0, compute attention diagnostics on random Q/K of this length")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    arr = load_matrices(ckpt_path)  # (B, H, D)
    if arr.size == 0:
        print("No elliptical parameters found in checkpoint (is icl_elliptical enabled?)")
        return

    # Per-head stats over head_dim
    per_head_mean = arr.mean(axis=-1).reshape(-1)
    per_head_std = arr.std(axis=-1).reshape(-1)
    # Anisotropy proxy: std of mean-one normalized m
    arr_hat = arr / (arr.mean(axis=-1, keepdims=True) + 1e-12)
    per_head_std_hat = arr_hat.std(axis=-1).reshape(-1)

    print(f"Blocks: {arr.shape[0]}  Heads: {arr.shape[1]}  Head dim: {arr.shape[2]}")
    print(f"mean(m): mean={per_head_mean.mean():.6f} min={per_head_mean.min():.6f} max={per_head_mean.max():.6f}")
    print(f"std(m):  mean={per_head_std.mean():.6f} min={per_head_std.min():.6f} max={per_head_std.max():.6f}")
    print(f"std(m_hat): mean={per_head_std_hat.mean():.6f} min={per_head_std_hat.min():.6f} max={per_head_std_hat.max():.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(per_head_mean, bins=50, color="#4C78A8")
    axes[0].set_title("Per-head mean(m)")
    axes[0].set_xlabel("mean(m)")
    axes[0].set_ylabel("count")

    axes[1].hist(per_head_std, bins=50, color="#F58518")
    axes[1].set_title("Per-head std(m)")
    axes[1].set_xlabel("std(m)")
    axes[1].set_ylabel("count")
    axes[2].hist(per_head_std_hat, bins=50, color="#54A24B")
    axes[2].set_title("Per-head std(m_hat)")
    axes[2].set_xlabel("std(m_hat)")
    axes[2].set_ylabel("count")
    fig.tight_layout()

    out_path = Path(args.out) if args.out else (ckpt_path.parent / f"elliptical_stats_{ckpt_path.stem}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")

    # Optional: random attention diagnostics to estimate entropy and KL vs. dot-product baseline
    if args.attn_diag_len and args.attn_diag_len > 0:
        T = int(args.attn_diag_len)
        B, Hh, D = arr.shape
        rng = np.random.default_rng(42)
        ent_base, ent_ell, kl_ell_base, l1 = [], [], [], []
        for b in range(B):
            for h in range(Hh):
                m = arr[b, h]
                m_hat = m / (m.mean() + 1e-12)
                s = np.sqrt(m_hat + 1e-12)
                # Random queries/keys per head
                Q = rng.standard_normal((T, D)).astype(np.float32)
                K = rng.standard_normal((T, D)).astype(np.float32)
                # Baseline logits
                logits0 = (Q @ K.T) / np.sqrt(D)
                P0 = np.exp(logits0 - logits0.max(axis=-1, keepdims=True))
                P0 /= P0.sum(axis=-1, keepdims=True)
                # Elliptical logits (mean-one + sqrt scaling)
                Qe = Q * s
                Ke = K * s
                logits1 = (Qe @ Ke.T) / np.sqrt(D)
                P1 = np.exp(logits1 - logits1.max(axis=-1, keepdims=True))
                P1 /= P1.sum(axis=-1, keepdims=True)
                # Row-wise entropy and KL
                eps = 1e-12
                H0 = -(P0 * (np.log(P0 + eps))).sum(axis=-1)
                H1 = -(P1 * (np.log(P1 + eps))).sum(axis=-1)
                KL = (P1 * (np.log((P1 + eps) / (P0 + eps)))).sum(axis=-1)
                L1 = np.abs(P1 - P0).sum(axis=-1)
                ent_base.append(H0.mean()); ent_ell.append(H1.mean()); kl_ell_base.append(KL.mean()); l1.append(L1.mean())
        print(f"Attn diag (T={T}): H_base={np.mean(ent_base):.4f} H_ell={np.mean(ent_ell):.4f} KL(ell||base)={np.mean(kl_ell_base):.5f} L1={np.mean(l1):.5f}")


if __name__ == "__main__":
    main()
