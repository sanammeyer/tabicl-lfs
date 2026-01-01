#!/usr/bin/env python3
"""
Compare learned weights between a TabICL SA checkpoint and an EA checkpoint.

By default this:
  - loads both checkpoints,
  - aligns their state_dict keys,
  - ignores any parameters whose names contain substrings like "elliptical",
  - computes per-parameter relative L2 difference and cosine similarity,
  - prints global summary statistics and top-k most different parameters.

Example
-------
  python scripts/compare_sa_ea_weights.py \
      --sa_checkpoint checkpoints_mini_tabicl_stage2_sa/step-1000.ckpt \
      --ea_checkpoint checkpoints_mini_tabicl_stage2_ea/step-1000_ea.ckpt \
      --topk 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


TensorDict = Dict[str, torch.Tensor]


def load_state_dict(path: Path) -> TensorDict:
    """Load a checkpoint and return its state_dict."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    if not isinstance(sd, dict):
        raise ValueError(f"Unexpected checkpoint format at {path}: expected dict or a dict with 'state_dict'.")

    # Filter to tensor-like entries only
    out: TensorDict = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v
    return out


def _filter_keys(keys: Iterable[str], ignore_substrings: Sequence[str]) -> List[str]:
    """Return keys that do not contain any of the ignore substrings."""
    if not ignore_substrings:
        return list(keys)
    ignores = tuple(ignore_substrings)
    return [k for k in keys if not any(s in k for s in ignores)]


def _quantiles(values: Sequence[float]) -> Tuple[float, float, float, float]:
    """Return (min, median, mean, max) for a non-empty sequence."""
    vals = sorted(values)
    n = len(vals)
    mid = vals[n // 2]
    return vals[0], mid, float(mean(vals)), vals[-1]


def compare_state_dicts(
    sa_sd: TensorDict,
    ea_sd: TensorDict,
    ignore_substrings: Sequence[str],
) -> Tuple[List[str], List[str], List[Tuple[str, float, float, float, float]]]:
    """Compare two state dicts and return SA-only, EA-only, and per-parameter metrics.

    Returns
    -------
    sa_only : list[str]
        Parameter names present only in the SA checkpoint (after filtering).

    ea_only : list[str]
        Parameter names present only in the EA checkpoint (after filtering).

    rows : list[tuple]
        Per-parameter metrics for shared parameters:
          (name, rel_l2_diff, cosine_similarity, norm_sa, l2_diff)
    """
    sa_keys = set(_filter_keys(sa_sd.keys(), ignore_substrings))
    ea_keys = set(_filter_keys(ea_sd.keys(), ignore_substrings))

    common = sorted(sa_keys & ea_keys)
    sa_only = sorted(sa_keys - ea_keys)
    ea_only = sorted(ea_keys - sa_keys)

    rows: List[Tuple[str, float, float, float, float]] = []
    for name in common:
        w_sa = sa_sd[name].detach().float().view(-1)
        w_ea = ea_sd[name].detach().float().view(-1)
        if w_sa.shape != w_ea.shape:
            # Shape mismatch: skip but warn via a synthetic entry with NaNs
            continue

        diff = w_ea - w_sa
        norm_sa = w_sa.norm().item()
        norm_diff = diff.norm().item()
        rel = norm_diff / (norm_sa + 1e-8)

        denom = (w_sa.norm().item() * w_ea.norm().item()) + 1e-8
        cos = float(torch.dot(w_sa, w_ea).item() / denom)

        rows.append((name, float(rel), cos, float(norm_sa), float(norm_diff)))

    return sa_only, ea_only, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare TabICL SA vs EA checkpoints at the weight level.")
    ap.add_argument(
        "--sa_checkpoint",
        type=str,
        required=True,
        help="Path to the standard-attention (SA) checkpoint.",
    )
    ap.add_argument(
        "--ea_checkpoint",
        type=str,
        required=True,
        help="Path to the elliptical-attention (EA) checkpoint.",
    )
    ap.add_argument(
        "--ignore_substrings",
        type=str,
        nargs="*",
        default=["elliptical"],
        help="Parameter-name substrings to ignore (e.g. elliptical-specific params).",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of most-different parameters to list.",
    )
    args = ap.parse_args()

    sa_path = Path(args.sa_checkpoint)
    ea_path = Path(args.ea_checkpoint)

    print(f"Loading SA checkpoint from: {sa_path}")
    sa_sd = load_state_dict(sa_path)
    print(f"  Loaded {len(sa_sd)} tensor parameters.")

    print(f"\nLoading EA checkpoint from: {ea_path}")
    ea_sd = load_state_dict(ea_path)
    print(f"  Loaded {len(ea_sd)} tensor parameters.")

    print("\nComparing state_dicts...")
    sa_only, ea_only, rows = compare_state_dicts(sa_sd, ea_sd, ignore_substrings=args.ignore_substrings)

    print(f"\nAfter filtering (ignore_substrings={args.ignore_substrings}):")
    print(f"  SA-only params: {len(sa_only)}")
    print(f"  EA-only params: {len(ea_only)}")
    print(f"  Shared params:  {len(rows)}")

    if sa_only:
        print("  Example SA-only:", sa_only[:5])
    if ea_only:
        print("  Example EA-only:", ea_only[:5])

    if not rows:
        print("\nNo shared parameters to compare after filtering; nothing to do.")
        return

    rel_vals = [r[1] for r in rows]
    cos_vals = [r[2] for r in rows]

    rel_min, rel_med, rel_mean, rel_max = _quantiles(rel_vals)
    cos_min, cos_med, cos_mean, cos_max = _quantiles(cos_vals)

    print("\nGlobal stats over shared parameters:")
    print(
        "  Relative L2 diff: "
        f"min={rel_min:.4g}, median={rel_med:.4g}, mean={rel_mean:.4g}, max={rel_max:.4g}"
    )
    print(
        "  Cosine similarity: "
        f"min={cos_min:.4g}, median={cos_med:.4g}, mean={cos_mean:.4g}, max={cos_max:.4g}"
    )

    topk = max(1, min(args.topk, len(rows)))

    # Top-k by relative difference
    rows_by_rel = sorted(rows, key=lambda x: x[1], reverse=True)
    print(f"\nTop {topk} parameters by relative L2 difference (EA vs SA):")
    for name, rel, cos, norm_sa, norm_diff in rows_by_rel[:topk]:
        print(f"  {name:60s}  rel={rel:.3f}  cos={cos:.3f}")

    # Top-k by lowest cosine similarity
    rows_by_cos = sorted(rows, key=lambda x: x[2])
    print(f"\nTop {topk} parameters by lowest cosine similarity:")
    for name, rel, cos, norm_sa, norm_diff in rows_by_cos[:topk]:
        print(f"  {name:60s}  rel={rel:.3f}  cos={cos:.3f}")


if __name__ == "__main__":
    main()

