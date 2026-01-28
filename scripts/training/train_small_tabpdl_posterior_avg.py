#!/usr/bin/env python3
"""
Small TabICL training script with TabPDL head (posterior_avg) and CSV logging.

This wraps the main Trainer from src/tabicl/train/run.py but:
  - forces icl_head='tabpdl' (PDL-style head)
  - sets pdlc_agg='posterior_avg'
  - uses a small model and few training steps by default
  - logs per-step metrics to a CSV file

Run from the repo root, for example:

    python scripts/training/train_small_tabpdl_posterior_avg.py \
        --device cpu \
        --steps 200 \
        --batch_size 64

You can override the CSV path / checkpoint dir via flags below.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def make_wrapper_args() -> argparse.Namespace:
    """CLI for the small training wrapper itself."""
    root = Path(__file__).resolve().parents[2]

    p = argparse.ArgumentParser(description="Small TabICL+TabPDL (posterior_avg) training with CSV metrics.")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for training (e.g. 'cpu', 'cuda', 'cuda:0').",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of training steps (max_steps for the Trainer).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for the prior datasets.",
    )
    p.add_argument(
        "--micro_batch_size",
        type=int,
        default=8,
        help="Micro-batch size used for gradient accumulation.",
    )
    p.add_argument(
        "--metrics_csv",
        type=str,
        default=str(root / "training_metrics" / "small_tabpdl_posterior_avg.csv"),
        help="Path to CSV file for per-step metrics.",
    )
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(root / "checkpoints_small_tabpdl"),
        help="Directory where checkpoints will be saved.",
    )
    p.add_argument(
        "--prior_dir",
        type=str,
        default=None,
        help=(
            "Optional path to a directory with pre-generated prior batches. "
            "If omitted, PriorDataset will generate episodes on the fly."
        ),
    )
    return p.parse_args()


def main() -> None:
    # Ensure local src/ is importable as 'tabicl'
    root = Path(__file__).resolve().parents[2]
    src_dir = root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    from tabicl.train.train_config import build_parser
    from tabicl.train.run import Trainer

    args = make_wrapper_args()

    # Build the underlying training parser and synthesize a minimal argv
    # that sets TabPDL head + posterior_avg and small model/steps.
    base_parser = build_parser()

    train_argv = [
        # Core runtime / logging
        "--device",
        args.device,
        "--wandb_log",
        "False",
        "--max_steps",
        str(args.steps),
        "--batch_size",
        str(args.batch_size),
        "--micro_batch_size",
        str(args.micro_batch_size),
        "--metrics_csv",
        args.metrics_csv,
        "--checkpoint_dir",
        args.checkpoint_dir,
        # Use synthetic prior by default; prior_dir override below if provided.
        "--prior_type",
        "mix_scm",
        "--prior_device",
        "cpu",
        # Small-ish model to keep runtime modest.
        "--embed_dim",
        "64",
        "--col_num_blocks",
        "1",
        "--col_nhead",
        "2",
        "--col_num_inds",
        "32",
        "--row_num_blocks",
        "1",
        "--row_nhead",
        "2",
        "--row_num_cls",
        "2",
        "--icl_num_blocks",
        "1",
        "--icl_nhead",
        "2",
        "--ff_factor",
        "2",
        "--norm_first",
        "True",
        # TabPDL head configuration: use posterior_avg aggregation.
        "--icl_head",
        "tabpdl",
        "--pdlc_agg",
        "posterior_avg",
        "--pdlc_ce_weight",
        "0.2",
        # Light checkpointing for quick runs.
        "--save_temp_every",
        str(max(1, args.steps // 2)),
        "--save_perm_every",
        str(args.steps),
        "--max_checkpoints",
        "1",
    ]

    if args.prior_dir:
        train_argv.extend(["--prior_dir", args.prior_dir])

    config = base_parser.parse_args(train_argv)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
