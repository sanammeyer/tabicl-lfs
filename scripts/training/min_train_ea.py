#!/usr/bin/env python3
"""Minimal TabICL training run with Elliptical Attention (EA).

This script launches a tiny training job on synthetic prior data to sanity‑check
EA end‑to‑end. It configures a very small model and short sequences, and runs a
handful of steps (default: 10) so it finishes quickly on CPU or GPU.

Usage examples:
  - Default 10 steps on auto device:
      python scripts/training/min_train_ea.py

  - 25 steps on GPU if available; EA in TFrow only:
      python scripts/training/min_train_ea.py --steps 25 --device auto --icl_ea false

  - 10 steps and save a temporary checkpoint dir:
      python scripts/training/min_train_ea.py --out runs/min-ea --save
"""

from __future__ import annotations

import argparse
import os
import sys

# Make local src importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tabicl.train.train_config import build_parser
from tabicl.train.run import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10, help="Number of training steps")
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device: auto|cpu|cuda|cuda:0",
    )
    ap.add_argument("--icl_ea", type=str, default="false", help="Enable EA in TFicl too (true/false)")
    ap.add_argument("--out", type=str, default=os.path.join(REPO_ROOT, "runs", "min-ea"))
    ap.add_argument("--save", action="store_true", help="Save checkpoints under --out")
    # Optional wandb controls (forwarded to main trainer CLI)
    ap.add_argument("--wandb_log", type=str, default="false", help="Enable wandb logging (true/false)")
    ap.add_argument(
        "--wandb_mode",
        type=str,
        default="offline",
        help="Wandb mode: online, offline, or disabled",
    )
    ap.add_argument(
        "--wandb_project",
        type=str,
        default="TabICL-min-ea",
        help="Wandb project name for this minimal EA run.",
    )
    ap.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Optional wandb run name for this minimal EA run.",
    )
    args = ap.parse_args()

    # Resolve device
    if args.device == "auto":
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device

    # Build the full trainer config using the standard parser, overriding key args
    p = build_parser()

    # Minimal, fast configuration:
    # - Use the default "full" TabICL architecture (embed_dim, num blocks, heads, etc.)
    # - But keep prior datasets and batch sizes small so runs are cheap.
    cli = [
        # Core training
        "--device",
        dev,
        "--dtype",
        "float32",
        "--max_steps",
        str(int(args.steps)),
        "--batch_size",
        "8",
        "--micro_batch_size",
        "8",
        "--lr",
        "5e-4",
        "--warmup_proportion",
        "0.0",
        # Prior (small n: few rows/features/classes)
        "--batch_size_per_gp",
        "2",
        "--min_features",
        "4",
        "--max_features",
        "12",
        "--max_classes",
        "6",
        "--min_seq_len",
        "16",
        "--max_seq_len",
        "32",
        "--min_train_size",
        "0.3",
        "--max_train_size",
        "0.5",
        "--prior_type",
        "mix_scm",
        "--prior_device",
        "cpu",
        # Model: default TabICL size (from stage scripts), but with small data above
        "--embed_dim",
        "128",
        "--col_num_blocks",
        "3",
        "--col_nhead",
        "4",
        "--col_num_inds",
        "128",
        "--row_num_blocks",
        "3",
        "--row_nhead",
        "8",
        "--row_num_cls",
        "4",
        "--icl_num_blocks",
        "12",
        "--icl_nhead",
        "4",
        # Elliptical Attention (EA): enable TFrow by default; TFicl optional
        "--row_elliptical",
        "True",
        "--icl_elliptical",
        "True" if args.icl_ea.lower() == "true" else "False",
        "--elliptical_delta",
        "1.0",
        "--elliptical_scale_mode",
        "max",
        # Misc / logging
        "--amp",
        "True",
    ]

    # Wandb configuration (optional)
    cli += [
        "--wandb_log",
        "True" if args.wandb_log.lower() == "true" else "False",
        "--wandb_mode",
        args.wandb_mode,
        "--wandb_project",
        args.wandb_project,
    ]
    if args.wandb_name is not None:
        cli += ["--wandb_name", args.wandb_name]

    # Always log a small CSV locally for inspection
    cli += [
        "--metrics_csv",
        os.path.join(args.out, "min_ea_metrics.csv"),
    ]

    if args.save:
        cli += [
            "--checkpoint_dir",
            os.path.join(args.out, "ckpts"),
            "--save_temp_every",
            "5",
            "--max_checkpoints",
            "2",
        ]
    else:
        # Avoid checkpoint writes
        cli += ["--save_temp_every", str(10 ** 9)]

    config = p.parse_args(cli)

    # Ensure output directory exists for CSV
    try:
        os.makedirs(args.out, exist_ok=True)
    except Exception:
        pass

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
