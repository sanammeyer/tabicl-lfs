#!/usr/bin/env python
from __future__ import annotations

"""
Inspect learned parameters of a TabPDL (PDLC-style) head in a TabICL checkpoint.

Usage
-----
    python scripts/probes/inspect_pdlc_head.py --checkpoint checkpoints/mini_tabicl_stage2_pdl/step-1000.ckpt

Optional:
    --save_npz path/to/file.npz   # Save TabPDL head parameters to a NumPy .npz for further analysis
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect learned parameters of the TabPDL head.")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a TabICL training checkpoint (.ckpt) trained with icl_head='tabpdl'.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading the model (default: cpu).",
    )
    p.add_argument(
        "--save_npz",
        type=str,
        default=None,
        help="Optional path to save TabPDL head parameters as a NumPy .npz archive.",
    )
    p.add_argument(
        "--print_full",
        action="store_true",
        help="If set, print full tensors for small parameters (use with care).",
    )
    return p


def _ensure_src_on_path():
    """Ensure local src/ is importable (prioritized over venv-installed tabicl)."""
    import sys

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def load_model_and_head(ckpt_path: Path, device: torch.device):
    """Load TabICL model and return its TabPDL head."""
    _ensure_src_on_path()
    from tabicl.model.tabicl import TabICL  # type: ignore

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint at {ckpt_path} does not look like a TabICL training checkpoint "
            "(missing 'config' or 'state_dict')."
        )

    cfg = checkpoint["config"]
    model = TabICL(**cfg).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(
        f"Loaded checkpoint from {ckpt_path}\n"
        f"  Missing keys:   {len(missing)}\n"
        f"  Unexpected keys:{len(unexpected)}"
    )

    icl = getattr(model, "icl_predictor", None)
    if icl is None:
        raise RuntimeError("Loaded model has no 'icl_predictor' module.")

    head = getattr(icl, "pdlc_head", None)
    if head is None:
        raise RuntimeError(
            "No TabPDL head found on model (icl_predictor.pdlc_head is None). "
            "Was this checkpoint trained with icl_head='tabpdl'?"
        )

    head.eval()
    return model, head


def summarize_head(head: torch.nn.Module, print_full: bool = False) -> Dict[str, np.ndarray]:
    """Print a summary of TabPDL head parameters and return them as numpy arrays."""
    print("\n=== TabPDL head parameter summary ===")
    np_params: Dict[str, np.ndarray] = {}
    total = 0

    for name, param in head.named_parameters():
        tensor = param.detach().cpu()
        arr = tensor.numpy()
        np_params[name] = arr
        total += arr.size

        stats = {
            "shape": tuple(arr.shape),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        print(f"{name:20} shape={stats['shape']}, "
              f"mean={stats['mean']:+.4f}, std={stats['std']:.4f}, "
              f"min={stats['min']:+.4f}, max={stats['max']:+.4f}")

        if print_full and arr.size <= 64:
            print(f"  values: {arr}")

    print(f"\nTotal number of TabPDL head parameters: {total}")

    # Also expose the temperature tau (derived from _tau_param)
    tau = getattr(head, "tau", None)
    if callable(tau):
        tau_val = tau()
    else:
        tau_val = tau

    if tau_val is not None:
        tau_arr = tau_val.detach().cpu().numpy()
        np_params["tau"] = tau_arr
        print(f"\nDerived temperature tau (after softplus): {float(tau_arr):.6f}")

    # Optionally print config if present
    cfg = getattr(head, "cfg", None)
    if cfg is not None:
        print("\nTabPDL head config (cfg):")
        print(cfg)

    return np_params


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint file not found: {ckpt_path}")

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    _, head = load_model_and_head(ckpt_path, device)
    np_params = summarize_head(head, print_full=args.print_full)

    if args.save_npz is not None:
        out_path = Path(args.save_npz).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **{k.replace(".", "_"): v for k, v in np_params.items()})
        print(f"\nSaved TabPDL head parameters to {out_path}")


if __name__ == "__main__":
    main()
