#!/usr/bin/env python3
"""Reproduce CE=NaN with Elliptical Attention enabled (TFrow + TFicl).

This script runs a small training loop (using the standard Trainer) with EA
turned on in both TFrow and TFicl, and watches for NaNs/Infs in loss, logits,
parameters and grads. When a NaN is detected, it prints diagnostics and exits.

Options allow toggling AMP, dtype, delta/scale_mode, gradient clipping, etc.

Run examples:
  - Default (CPU/GPU auto), 500 steps, EA on for both stacks:
      python scripts/repro_ea_nan.py --steps 500

  - Try bfloat16 on GPU with AMP off (more conservative numerics):
      python scripts/repro_ea_nan.py --device cuda --amp false --dtype float32

  - Switch scale_mode to mean and larger delta:
      python scripts/repro_ea_nan.py --elliptical_scale_mode mean --elliptical_delta 2.0
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from typing import Optional

import torch

# Ensure local src importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tabicl.train.train_config import build_parser
from tabicl.train.run import Trainer
from tabicl.model.layers import MultiheadAttentionBlock
from tabicl.model.attention import compute_elliptical_diag


def annotate_blocks(model):
    for i, blk in enumerate(model.row_interactor.tf_row.blocks):
        setattr(blk, "_ea_stack", "tfrow")
        setattr(blk, "_ea_layer", i)
    for i, blk in enumerate(model.icl_predictor.tf_icl.blocks):
        setattr(blk, "_ea_stack", "tficl")
        setattr(blk, "_ea_layer", i)


def patch_ea_hook():
    """Patch _attn_block to record per-call EA stats (min/max/mean of M_diag) and NaN flags."""
    orig = MultiheadAttentionBlock._attn_block

    def wrapped(self, q, k, v, key_padding_mask, attn_mask, rope, v_prev=None, block_index=None):
        # Attempt to reconstruct m_diag as used; store simple stats for debugging
        try:
            nh = self.attn.num_heads
            E = q.shape[-1]
            Dh = E // nh
            # linear projections
            import torch.nn.functional as F  # local import
            q_lin, k_lin, v_lin = F._in_projection_packed(q, k, v, self.attn.in_proj_weight, self.attn.in_proj_bias)
            # heads
            Tq = q.shape[-2]
            Tk = k.shape[-2]
            qh = q_lin.view(*q.shape[:-2], Tq, nh, Dh).transpose(-3, -2)
            vh = v_lin.view(*v.shape[:-2], Tk, nh, Dh).transpose(-3, -2)
            use_ell = self.elliptical and (v_prev is not None) and (block_index is None or block_index >= 1)
            m = None
            if self.elliptical_override == "manual" and (self.elliptical_manual_m is not None):
                m = self.elliptical_manual_m
            elif self.elliptical_override == "identity":
                m = torch.ones(nh, Dh, device=qh.device, dtype=qh.dtype)
            elif use_ell and isinstance(v_prev, torch.Tensor) and v_prev.shape[-1] == Dh:
                m = compute_elliptical_diag(vh, v_prev, delta=float(self.elliptical_delta), scale_mode=self.elliptical_scale_mode)
            if m is not None:
                setattr(self, "_dbg_last_m_min", float(m.min().detach().cpu()))
                setattr(self, "_dbg_last_m_max", float(m.max().detach().cpu()))
                setattr(self, "_dbg_last_m_mean", float(m.mean().detach().cpu()))
                setattr(self, "_dbg_last_m_has_nan", bool(torch.isnan(m).any().item()))
                setattr(self, "_dbg_last_m_has_inf", bool(torch.isinf(m).any().item()))
        except Exception:
            pass
        return orig(self, q, k, v, key_padding_mask, attn_mask, rope, v_prev=v_prev, block_index=block_index)

    MultiheadAttentionBlock._attn_block = wrapped  # type: ignore


def scan_model_anomalies(model) -> dict:
    """Scan parameters and grads for NaN/Inf and report basic norms."""
    anomalies = {
        "param_nan": 0,
        "param_inf": 0,
        "grad_nan": 0,
        "grad_inf": 0,
        "param_max": 0.0,
        "grad_max": 0.0,
    }
    with torch.no_grad():
        for p in model.parameters():
            if p is None:
                continue
            if torch.isnan(p).any():
                anomalies["param_nan"] += 1
            if torch.isinf(p).any():
                anomalies["param_inf"] += 1
            anomalies["param_max"] = max(anomalies["param_max"], float(p.abs().max().item()))
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    anomalies["grad_nan"] += 1
                if torch.isinf(p.grad).any():
                    anomalies["grad_inf"] += 1
                anomalies["grad_max"] = max(anomalies["grad_max"], float(p.grad.abs().max().item()))
    return anomalies


class DebugTrainer(Trainer):
    def train(self):  # override to intercept results and stop on NaN
        if self.master_process:
            step_iter = range(self.curr_step, self.config.max_steps)
            print("DebugTrainer: watching CE for NaNs/Infs...")
        else:
            step_iter = range(self.curr_step, self.config.max_steps)
        dataloader = iter(self.dataloader)
        for step in step_iter:
            self._update_qk_warmup_freeze(step)
            batch = next(dataloader)
            results = self.run_batch(batch)
            ce = float(results.get("ce", float("nan")))
            if not math.isfinite(ce):
                print(f"[NaN/Inf detected] step={step+1} ce={ce}")
                # quick model scan
                anomalies = scan_model_anomalies(self.raw_model)
                print(
                    f"param_nan={anomalies['param_nan']} param_inf={anomalies['param_inf']} "
                    f"grad_nan={anomalies['grad_nan']} grad_inf={anomalies['grad_inf']} "
                    f"param_max={anomalies['param_max']:.3e} grad_max={anomalies['grad_max']:.3e}"
                )
                # EA per-block recent stats
                try:
                    print("Last EA m stats per block (where available):")
                    for stack_name, blocks in (
                        ("tfrow", self.raw_model.row_interactor.tf_row.blocks),
                        ("tficl", self.raw_model.icl_predictor.tf_icl.blocks),
                    ):
                        for i, blk in enumerate(blocks):
                            mmin = getattr(blk, "_dbg_last_m_min", None)
                            mmax = getattr(blk, "_dbg_last_m_max", None)
                            mmean = getattr(blk, "_dbg_last_m_mean", None)
                            has_nan = getattr(blk, "_dbg_last_m_has_nan", False)
                            has_inf = getattr(blk, "_dbg_last_m_has_inf", False)
                            if mmin is not None:
                                print(
                                    f"  [{stack_name} L{i}] m[min={mmin:.3e}, max={mmax:.3e}, mean={mmean:.3e}, nan={has_nan}, inf={has_inf}] "
                                    f"override={blk.elliptical_override} delta={blk.elliptical_delta} mode={blk.elliptical_scale_mode}"
                                )
                except Exception:
                    pass
                # Exit after first detection
                break


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--amp", type=str, default="True")
    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--elliptical_delta", type=float, default=1.0)
    ap.add_argument("--elliptical_scale_mode", type=str, default="max", choices=["max", "mean"])
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--micro_batch_size", type=int, default=8)
    ap.add_argument("--gradient_clipping", type=float, default=1.0)
    ap.add_argument("--freeze_qk_warmup_steps", type=int, default=0)
    ap.add_argument("--log_csv", type=str, default=os.path.join(REPO_ROOT, "runs", "repro_ea_nan", "metrics.csv"))
    return ap.parse_args()


def main():
    args = build_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    # Patch hooks to capture EA diag stats
    patch_ea_hook()

    p = build_parser()
    cli = [
        "--device",
        dev,
        "--dtype",
        args.dtype,
        "--amp",
        args.amp,
        "--max_steps",
        str(int(args.steps)),
        "--batch_size",
        str(int(args.batch_size)),
        "--micro_batch_size",
        str(int(args.micro_batch_size)),
        "--lr",
        str(float(args.lr)),
        "--warmup_proportion",
        "0.0",
        # Prior smallish
        "--batch_size_per_gp",
        "2",
        "--min_features",
        "4",
        "--max_features",
        "24",
        "--max_classes",
        "8",
        "--min_seq_len",
        "16",
        "--max_seq_len",
        "48",
        "--min_train_size",
        "0.3",
        "--max_train_size",
        "0.5",
        "--prior_type",
        "mix_scm",
        "--prior_device",
        "cpu",
        # Model smallish
        "--embed_dim",
        "64",
        "--col_num_blocks",
        "2",
        "--col_nhead",
        "2",
        "--col_num_inds",
        "32",
        "--row_num_blocks",
        "3",
        "--row_nhead",
        "4",
        "--row_num_cls",
        "4",
        "--icl_num_blocks",
        "4",
        "--icl_nhead",
        "2",
        # EA flags
        "--row_elliptical",
        "True",
        "--icl_elliptical",
        "True",
        "--elliptical_delta",
        str(float(args.elliptical_delta)),
        "--elliptical_scale_mode",
        args.elliptical_scale_mode,
        # Misc/logging
        "--gradient_clipping",
        str(float(args.gradient_clipping)),
        "--freeze_qk_warmup_steps",
        str(int(args.freeze_qk_warmup_steps)),
        "--wandb_log",
        "False",
        "--metrics_csv",
        args.log_csv,
    ]

    # Ensure log dir exists
    try:
        os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
    except Exception:
        pass

    config = p.parse_args(cli)

    trainer = DebugTrainer(config)
    # Annotate blocks for nicer printing
    try:
        annotate_blocks(trainer.raw_model)
    except Exception:
        pass
    trainer.train()


if __name__ == "__main__":
    main()

