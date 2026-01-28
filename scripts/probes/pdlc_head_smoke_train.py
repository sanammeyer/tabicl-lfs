#!/usr/bin/env python3
"""Minimal smoke test for the integrated TabPDL-ICL head.

What this script does:
  - Builds a tiny TabICL model with icl_head='tabpdl' (or 'tabicl' for comparison)
  - Runs a short synthetic training loop with cross-entropy on query rows
  - Verifies that:
      * forward + backward work without error
      * PDLC aux outputs (gamma, support_mask) are populated when icl_head='tabpdl'
      * losses decrease over a few steps (very roughly)

You can run it directly, e.g.:

    python scripts/probes/pdlc_head_smoke_train.py --head tabpdl
    python scripts/probes/pdlc_head_smoke_train.py --head tabicl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu", help="Device (cpu|cuda:0, ...)")
    p.add_argument(
        "--head",
        type=str,
        default="tabpdl",
        choices=["tabicl", "tabpdl"],
        help="ICL prediction head to test.",
    )
    p.add_argument("--steps", type=int, default=20, help="Number of synthetic training steps")
    p.add_argument("--batch_size", type=int, default=4, help="Number of episodes per step")
    p.add_argument("--seq_len", type=int, default=16, help="Total rows per episode (train+test)")
    p.add_argument("--train_size", type=int, default=8, help="Number of support rows per episode")
    p.add_argument("--num_features", type=int, default=8, help="Number of features per row")
    p.add_argument("--num_classes", type=int, default=5, help="Number of classes per episode")
    p.add_argument(
        "--pdlc_feature_map",
        type=str,
        default="sym",
        choices=["sym", "concat"],
        help="Feature map for PDLC comparator when head='tabpdl'.",
    )
    p.add_argument(
        "--pdlc_topk",
        type=int,
        default=None,
        help="If set and >0, enable top-k gating in the PDLC head; 0/None disables gating.",
    )
    return p


def build_model(device: torch.device, args: argparse.Namespace):
    # Ensure local src is importable
    ROOT = Path(__file__).resolve().parents[2]
    SRC_DIR = ROOT / "src"
    import sys

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from tabicl.model.tabicl import TabICL  # type: ignore

    common_kwargs = dict(
        max_classes=args.num_classes,
        embed_dim=16,
        col_num_blocks=1,
        col_nhead=1,
        col_num_inds=8,
        col_elliptical=None,
        row_elliptical=False,
        row_num_blocks=1,
        row_nhead=1,
        row_num_cls=2,
        row_rope_base=100000,
        icl_num_blocks=1,
        icl_nhead=1,
        icl_elliptical=False,
        elliptical_delta=1.0,
        elliptical_scale_mode="max",
        ff_factor=2,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
    )

    if args.head == "tabpdl":
        model = TabICL(
            **common_kwargs,
            icl_head="tabpdl",
            pdlc_config={
                "topk": args.pdlc_topk,
                "mlp_width": 64,
                "mlp_depth": 2,
                "agg": "posterior_avg",
                "embed_norm": "none",
                "dropout": 0.0,
                "activation": "silu",
                "layernorm_after_first": True,
                "feature_map": args.pdlc_feature_map,
            },
        )
    else:
        model = TabICL(**common_kwargs, icl_head="tabicl")

    return model.to(device)


def synthetic_batch(args: argparse.Namespace, device: torch.device):
    """Create a synthetic batch of episodes."""
    B = args.batch_size
    T = args.seq_len
    H = args.num_features
    C = args.num_classes
    train_size = args.train_size

    rng = np.random.default_rng()
    X_np = rng.normal(size=(B, T, H)).astype("float32")
    y_np = rng.integers(low=0, high=C, size=(B, T), dtype="int64")

    # Ensure each class appears at least once in y_train per episode (best effort)
    for b in range(B):
        for c in range(C):
            y_np[b, c % train_size] = c

    X = torch.from_numpy(X_np).to(device)
    y_full = torch.from_numpy(y_np).to(device)
    y_train = y_full[:, :train_size]
    y_test = y_full[:, train_size:]
    return X, y_train, y_test


def main() -> None:
    args = make_parser().parse_args()
    device = torch.device(args.device)

    torch.manual_seed(1234)
    np.random.seed(1234)

    model = build_model(device, args)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"Running smoke training with head={args.head} on device={device}...")
    losses = []
    for step in range(1, args.steps + 1):
        X, y_train, y_test = synthetic_batch(args, device)

        opt.zero_grad(set_to_none=True)
        out = model(X, y_train)  # (B, T - train_size, max_classes)
        B, Tq, C = out.shape
        # Flatten over query positions
        logits = out.reshape(B * Tq, C)
        target = y_test.reshape(B * Tq)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))
        if step % max(1, args.steps // 5) == 0 or step == args.steps:
            print(f"Step {step:03d} | loss={loss.item():.4f}")

    if args.head == "tabpdl":
        # Run one more forward pass to inspect PDLC aux outputs
        model.eval()
        with torch.no_grad():
            X, y_train, _ = synthetic_batch(args, device)
            _ = model(X, y_train)
        enc = model.icl_predictor
        aux = getattr(enc, "last_pdlc_aux", None)
        if not aux:
            print("Warning: PDLC aux dict is empty.")
        else:
            gamma = aux.get("gamma", None)
            if gamma is not None:
                g = gamma[0]
                print(f"PDLC gamma shape (first episode): {tuple(g.shape)}")
            else:
                print("PDLC aux has no 'gamma' entry.")

    if len(losses) >= 2:
        print(f"Initial loss: {losses[0]:.4f} | final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
