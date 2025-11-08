#!/usr/bin/env python3
"""Benchmark: pre-generate prior vs on-the-fly generation.

This script measures training throughput in two scenarios for a small config:
  1) On-the-fly: generate prior inside the training loop
  2) Pre-generated: generate N batches to disk first, then train by loading

It reports wall-clock times and aggregated per-step timings (prior_time and
train_time) collected from the trainer's metrics CSV.

Examples
  - Quick 50-step benchmark on auto device
    python scripts/benchmark_prior_io.py --steps 50

  - CPU-only and keep generated data
    python scripts/benchmark_prior_io.py --steps 50 --device cpu --keep
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Make local src importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tabicl.train.train_config import build_parser as build_train_parser
from tabicl.train.run import Trainer
from tabicl.prior.genload import SavePriorDataset


def parse_csv_times(csv_path: Path) -> Tuple[float, float]:
    prior_sum = 0.0
    train_sum = 0.0
    if not csv_path.exists():
        return prior_sum, train_sum
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                prior_sum += float(row.get("prior_time", 0.0) or 0.0)
                train_sum += float(row.get("train_time", 0.0) or 0.0)
            except Exception:
                continue
    return prior_sum, train_sum


def run_train(steps: int, device: str, metrics_csv: Path, prior_dir: str | None) -> Dict[str, float]:
    p = build_train_parser()
    # Small, fast config (matches SavePriorDataset defaults below)
    cli = [
        "--device", device,
        "--dtype", "float32",
        "--max_steps", str(int(steps)),
        "--batch_size", "64",
        "--micro_batch_size", "64",
        "--lr", "5e-4",
        "--warmup_proportion", "0.0",
        # Prior cfg (must match generator for apples-to-apples)
        "--batch_size_per_gp", "2",
        "--min_features", "4",
        "--max_features", "12",
        "--max_classes", "6",
        "--min_seq_len", "16",
        "--max_seq_len", "32",
        "--min_train_size", "0.3",
        "--max_train_size", "0.5",
        "--prior_type", "mix_scm",
        "--prior_device", "cpu",
        # Model cfg (small)
        "--embed_dim", "64",
        "--col_num_blocks", "2",
        "--col_nhead", "2",
        "--col_num_inds", "32",
        "--row_num_blocks", "3",
        "--row_nhead", "4",
        "--row_num_cls", "4",
        "--icl_num_blocks", "4",
        "--icl_nhead", "2",
        # No checkpoint writes during benchmark
        "--save_temp_every", str(10 ** 9),
        # Logging
        "--wandb_log", "False",
        "--metrics_csv", str(metrics_csv),
    ]
    if prior_dir is not None:
        cli += ["--prior_dir", prior_dir]
    config = p.parse_args(cli)

    t0 = time.time()
    trainer = Trainer(config)
    trainer.train()
    wall = time.time() - t0

    prior_sum, train_sum = parse_csv_times(metrics_csv)
    return dict(wall=wall, prior_time=prior_sum, train_time=train_sum)


def run_generate(num_batches: int, save_dir: Path) -> float:
    # Build a namespace matching SavePriorDataset CLI
    class Args:
        pass

    args = Args()
    args.save_dir = str(save_dir)
    args.np_seed = 42
    args.torch_seed = 42
    args.num_batches = int(num_batches)
    args.resume_from = 0
    args.batch_size = 64
    args.batch_size_per_gp = 2
    args.min_features = 4
    args.max_features = 12
    args.max_classes = 6
    args.min_seq_len = 16
    args.max_seq_len = 32
    args.log_seq_len = False
    args.seq_len_per_gp = False
    args.min_train_size = 0.3
    args.max_train_size = 0.5
    args.replay_small = False
    args.prior_type = "mix_scm"
    args.n_jobs = -1
    args.num_threads_per_generate = 1
    args.device = "cpu"

    t0 = time.time()
    SavePriorDataset(args).run()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default=os.path.join(REPO_ROOT, "runs", "prior_bench"))
    ap.add_argument("--keep", action="store_true", help="Keep generated prior data directory")
    args = ap.parse_args()

    import torch

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) On-the-fly
    csv_onfly = out_dir / "onfly_metrics.csv"
    res_onfly = run_train(args.steps, dev, csv_onfly, prior_dir=None)

    # 2) Pre-generate + load
    prior_dir = out_dir / "pre_gen_data"
    gen_time = run_generate(args.steps, prior_dir)
    csv_pre = out_dir / "pre_metrics.csv"
    res_pre = run_train(args.steps, dev, csv_pre, prior_dir=str(prior_dir))

    # Summary
    def fmt(d: Dict[str, float]) -> str:
        return f"wall={d['wall']:.2f}s prior_time(sum)={d['prior_time']:.2f}s train_time(sum)={d['train_time']:.2f}s"

    print("Benchmark summary (steps={}, device={}):".format(args.steps, dev))
    print("- On-the-fly:", fmt(res_onfly))
    print("- Pre-generate: gen_time={:.2f}s,".format(gen_time), fmt(res_pre))
    print("- End-to-end: onfly={:.2f}s vs pregen={:.2f}s".format(
        res_onfly["wall"], gen_time + res_pre["wall"]))

    if not args.keep:
        # Cleanup
        try:
            for p in prior_dir.glob("batch_*.pt"):
                p.unlink()
            (prior_dir / "metadata.json").unlink(missing_ok=True)
            prior_dir.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    main()

