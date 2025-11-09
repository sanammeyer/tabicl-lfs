#!/usr/bin/env python3
"""Smoke-test for Trainer.materialize_probe_batch.

This script confirms that a fixed probe batch is materialized correctly for both
dataset modes:
  1) Pre-generated prior (uses --prior_dir)
  2) On-the-fly generation (no --prior_dir)

It does NOT run the training loop; it only instantiates the Trainer and triggers
probe materialization, then validates and prints the probe shapes.

Examples
  - Prior-dir mode (default):
      python scripts/test_probe_materialize.py

  - On-the-fly mode:
      python scripts/test_probe_materialize.py --mode onfly
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Make local src importable when running as a script
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tabicl.train.train_config import build_parser
from tabicl.train.run import Trainer


def gen_tiny_prior(save_dir: Path, num_batches: int = 2) -> None:
    """Generate a small pre-generated prior directory for testing."""
    from tabicl.prior.genload import SavePriorDataset
    from types import SimpleNamespace

    save_dir.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        save_dir=str(save_dir), np_seed=0, torch_seed=0, num_batches=int(num_batches), resume_from=0,
        batch_size=8, batch_size_per_gp=2,
        min_features=4, max_features=6, max_classes=4,
        # Use min_seq_len=None so the generator uses max_seq_len directly (avoids randint low>=high)
        min_seq_len=None, max_seq_len=8, log_seq_len=False,
        seq_len_per_gp=False, min_train_size=0.3, max_train_size=0.5, replay_small=False,
        prior_type='mix_scm', n_jobs=1, num_threads_per_generate=1, device='cpu'
    )
    print(f"[setup] Generating tiny prior to {save_dir} ...")
    SavePriorDataset(args).run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prior", "onfly"], default="prior")
    ap.add_argument("--prior_dir", default=str(REPO_ROOT / "runs" / "prior_probe_test"))
    ap.add_argument("--probe_batch_size", type=int, default=2)
    ap.add_argument("--probe_every", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--gen_batches", type=int, default=None, help="Number of prior batches to pre-generate (defaults to --steps)")
    ap.add_argument("--metrics_csv", default=str(REPO_ROOT / "runs" / "probe_test_metrics.csv"))
    ap.add_argument("--print_probe_rows", action="store_true", help="Print probe rows from CSV after run")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build trainer args
    parser = build_parser()
    cli = [
        "--device", device,
        "--dtype", "float32",
        "--max_steps", str(int(args.steps)),
        "--batch_size", "8",
        "--micro_batch_size", "8",
        "--lr", "1e-4",
        "--warmup_proportion", "0.0",
        # Prior sampling config (small, fast)
        "--batch_size_per_gp", "2",
        "--min_features", "4",
        "--max_features", "6",
        "--max_classes", "4",
        "--min_seq_len", "8",
        "--max_seq_len", "8",
        "--prior_type", "mix_scm",
        # Probe config
        "--probe_every", str(int(args.probe_every)),
        "--probe_batch_size", str(int(args.probe_batch_size)),
        "--probe_seed", "1337",
        # Logging off
        "--wandb_log", "False",
    ]

    if args.mode == "prior":
        prior_dir = Path(args.prior_dir)
        gen_count = int(args.gen_batches) if args.gen_batches is not None else int(args.steps)
        if gen_count < 2:
            gen_count = 2
        gen_tiny_prior(prior_dir, num_batches=gen_count)
        # Use CPU to load shards; avoids CUDA requirement for loading pre-gen files
        cli += ["--prior_dir", str(prior_dir), "--prior_device", "cpu"]
    else:
        # On-the-fly generation on CPU; we won't iterate the loader
        cli += ["--prior_device", "cpu"]

    config = parser.parse_args(cli)

    print("[test] Instantiating Trainer ...")
    tr = Trainer(config)

    # Either __init__ already materialized a probe (when probe_every>0), or do it now
    if getattr(tr, "_probe_data", None) is None:
        print("[test] Forcing materialize_probe_batch() ...")
        tr.materialize_probe_batch()

    pd = getattr(tr, "_probe_data", None)
    assert pd is not None, "Probe data was not materialized."
    X = pd["X"]; y = pd["y"]; d = pd["d"]; ts = pd["train_size"]
    print(f"[ok] Probe shapes: X={tuple(X.shape)}, y={tuple(y.shape)}, d={tuple(d.shape)}, train_size[0]={int(ts[0].item())}")
    print("[ok] materialize_probe_batch smoke-test passed.")

    # Run a short training to trigger probes and write CSV
    print(f"[test] Running training for {args.steps} steps to emit probe metrics ...")
    # Rebuild config with metrics CSV (in case not set)
    if "--metrics_csv" not in cli:
        # Add metrics_csv and rebuild the trainer to ensure CSV logging is enabled
        cli_with_csv = cli + ["--metrics_csv", args.metrics_csv, "--save_temp_every", str(10 ** 9)]
        config = parser.parse_args(cli_with_csv)
        tr = Trainer(config)
    # Ensure metrics_csv set even if trainer was constructed earlier with it
    tr.metrics_csv_path = args.metrics_csv
    tr.train()

    # Print probe rows from CSV
    try:
        import csv
        rows = []
        with open(args.metrics_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if any(k.startswith("probe/") for k in row.keys()):
                    rows.append(row)
        if rows:
            print(f"[ok] Found {len(rows)} probe rows in {args.metrics_csv} (showing last 3):")
            for r in rows[-3:]:
                step = r.get("step", "?")
                delta_ce = r.get("probe/delta_ce", "")
                ce = r.get("probe/ce", "")
                chance = r.get("probe/chance_entropy", "")
                attn_kl = r.get("probe/attn_kl_mean", "")
                attn_ent = r.get("probe/attn_entropy_mean", "")
                print(f"  step={step} delta_ce={delta_ce} ce={ce} chance={chance} attn_kl={attn_kl} attn_ent={attn_ent}")
        else:
            print(f"[warn] No probe rows found in {args.metrics_csv}. Check --probe_every and steps.")
    except Exception as e:
        print(f"[warn] Failed to read metrics CSV: {e}")


if __name__ == "__main__":
    main()
