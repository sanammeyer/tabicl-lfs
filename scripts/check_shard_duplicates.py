#!/usr/bin/env python3
"""
Check for duplicate prior shards (batch_*.pt files) by computing a content hash
over tensors saved in each file. Reports exact duplicates and a percentage.

Usage:
  python -m tabicl.scripts.check_shard_duplicates \
      --dir /path/to/priors \
      --workers 16

Notes:
  - This checks shard-level duplicates (entire files). It does NOT check for
    per-dataset duplicates within or across shards.
  - For Stage-1 (seq_len_per_gp=False), saved X is non-nested (sparse-packed).
    We hash the tensors as saved to avoid heavy dense reconstruction.
  - For nested X (if present), we convert to a padded dense tensor with 0.0 pad
    before hashing to establish a canonical representation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from glob import glob
from multiprocessing import Pool, cpu_count

import torch
from tqdm import tqdm


def hash_tensor(h: "hashlib._Hash", t: torch.Tensor) -> None:
    if t is None:
        return
    # Ensure CPU, contiguous, and numeric dtype
    t = t.detach().cpu().contiguous()
    # Stream in reasonable chunks to avoid huge intermediate buffers
    # Flatten to 1D bytes
    mv = memoryview(t.numpy())
    h.update(mv.tobytes())


def canonicalize_and_hash(path: str) -> tuple[str, str]:
    """Return (path, hex_digest) for a shard file."""
    try:
        batch = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch without weights_only
        batch = torch.load(path, map_location="cpu")

    # Expected keys: X, y, d, seq_lens, train_sizes, batch_size
    X = batch.get("X")
    y = batch.get("y")
    d = batch.get("d")
    seq_lens = batch.get("seq_lens")
    train_sizes = batch.get("train_sizes")
    bsz = batch.get("batch_size")

    h = hashlib.sha256()
    # Include a header with shapes/meta to be safe
    header = f"bsz={int(bsz) if bsz is not None else -1}|keys={sorted(batch.keys())}".encode()
    h.update(header)

    # X can be nested; if so, convert to padded dense for canonical hashing
    try:
        is_nested = getattr(X, "is_nested", False)
    except Exception:
        is_nested = False
    if is_nested:
        try:
            Xc = X.to_padded_tensor(0.0)
        except Exception:
            # Fallback: best-effort conversion
            Xc = torch.nested.to_padded_tensor(X, 0.0)
        hash_tensor(h, Xc)
    else:
        hash_tensor(h, X)

    hash_tensor(h, y)
    hash_tensor(h, d)
    hash_tensor(h, seq_lens)
    hash_tensor(h, train_sizes)

    return path, h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing batch_*.pt files")
    ap.add_argument("--pattern", default="batch_*.pt", help="Glob to match shard files")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() // 2), help="Parallel workers")
    args = ap.parse_args()

    files = sorted(glob(os.path.join(args.dir, args.pattern)))
    if not files:
        raise SystemExit(f"No files matching {args.pattern} in {args.dir}")

    print(f"[info] scanning {len(files)} shards in {args.dir} ...")
    with Pool(processes=int(args.workers)) as pool:
        results = list(tqdm(pool.imap_unordered(canonicalize_and_hash, files), total=len(files)))

    # Aggregate
    by_hash = {}
    for path, digest in results:
        by_hash.setdefault(digest, []).append(path)

    unique = sum(1 for v in by_hash.values() if len(v) == 1)
    dups = [(k, v) for k, v in by_hash.items() if len(v) > 1]
    num_dupe_files = sum(len(v) for _, v in dups)
    pct = 100.0 * num_dupe_files / len(files)

    print(f"[result] total_shards={len(files)} unique={unique} duplicates_files={num_dupe_files} dup_pct={pct:.6f}%")
    if dups:
        print("[detail] duplicate groups (showing up to 5 groups):")
        for digest, paths in dups[:5]:
            print(f"  hash={digest} count={len(paths)} any={paths[0]}")


if __name__ == "__main__":
    main()

