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
import math
from pathlib import Path
from typing import Dict, Tuple

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
    p.add_argument(
        "--analyze_bilinear",
        action="store_true",
        help=(
            "If set, compute diagnostics for the learned bilinear similarity "
            "(W_Q, W_K, tau, and the effective metric M=W_Q^T W_K)."
        ),
    )
    p.add_argument(
        "--svd_topk",
        type=int,
        default=8,
        help="How many top singular values to print for each matrix (default: 8).",
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


def _svdvals(x: np.ndarray) -> np.ndarray:
    # Use numpy for broad compatibility across torch builds.
    x = np.asarray(x, dtype=np.float64)
    try:
        s = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        # Fallback: eigvals of X^T X (can be less stable)
        xtx = x.T @ x
        evals = np.linalg.eigvalsh(xtx)
        s = np.sqrt(np.clip(evals, a_min=0.0, a_max=None))[::-1]
    return np.asarray(s, dtype=np.float64)


def _effective_rank(s: np.ndarray, eps: float = 1e-12) -> float:
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0:
        return float("nan")
    z = s / max(float(s.sum()), eps)
    z = z[z > 0]
    h = float(-(z * np.log(z)).sum())
    return float(np.exp(h))


def _matrix_summary(name: str, w: np.ndarray, *, svd_topk: int = 8) -> Dict[str, float]:
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"{name} expected 2D weight matrix, got shape={w.shape}")
    s = _svdvals(w)
    smax = float(s[0]) if s.size else float("nan")
    smin = float(s[-1]) if s.size else float("nan")
    cond = float(smax / max(smin, 1e-12)) if np.isfinite(smax) and np.isfinite(smin) else float("nan")
    fro = float(np.linalg.norm(w, ord="fro"))
    erank = _effective_rank(s)

    print(f"\n[{name}] shape={w.shape}")
    print(f"[{name}] ||W||_F={fro:.4f}  sigma_max={smax:.4f}  sigma_min={smin:.4f}  cond~={cond:.2e}  eff_rank~={erank:.2f}")
    if s.size:
        k = max(1, min(int(svd_topk), int(s.size)))
        top = ", ".join(f"{float(x):.4f}" for x in s[:k])
        print(f"[{name}] top-{k} singular values: [{top}]")

    if w.shape[0] == w.shape[1]:
        d = w.shape[0]
        eye = np.eye(d, dtype=np.float64)
        rel_eye = float(np.linalg.norm(w - eye, ord="fro") / max(np.linalg.norm(eye, ord="fro"), 1e-12))
        rel_zero = float(np.linalg.norm(w, ord="fro") / max(np.linalg.norm(eye, ord="fro"), 1e-12))
        print(f"[{name}] ||W-I||_F/||I||_F={rel_eye:.4f}  ||W||_F/||I||_F={rel_zero:.4f}")
    return {
        "fro": fro,
        "sigma_max": smax,
        "sigma_min": smin,
        "cond_approx": cond,
        "effective_rank": erank,
    }


def analyze_bilinear_similarity(head: torch.nn.Module, *, svd_topk: int = 8) -> Dict[str, Dict[str, float]]:
    """Diagnose whether W_Q/W_K define a non-trivial bilinear similarity."""
    out: Dict[str, Dict[str, float]] = {}

    W_Q = getattr(head, "W_Q", None)
    W_K = getattr(head, "W_K", None)
    if W_Q is None or W_K is None:
        raise RuntimeError("Head has no W_Q/W_K; cannot analyze bilinear similarity.")

    Wq = W_Q.weight.detach().cpu().numpy()
    Wk = W_K.weight.detach().cpu().numpy()
    out["W_Q"] = _matrix_summary("W_Q", Wq, svd_topk=svd_topk)
    out["W_K"] = _matrix_summary("W_K", Wk, svd_topk=svd_topk)

    # Effective bilinear form in the original embedding space:
    #   (W_Q h_q) · (W_K h_s) = h_q^T (W_Q^T W_K) h_s
    M = Wq.T @ Wk
    out["M"] = _matrix_summary("M = W_Q^T W_K", M, svd_topk=svd_topk)

    # Symmetry / skew: M need not be symmetric, but if it's close to I or symmetric,
    # then the similarity is close to a standard dot-product metric.
    sym = 0.5 * (M + M.T)
    skew = 0.5 * (M - M.T)
    sym_f = float(np.linalg.norm(sym, ord="fro"))
    skew_f = float(np.linalg.norm(skew, ord="fro"))
    rel_skew = float(skew_f / max(sym_f, 1e-12))
    print(f"\n[M] symmetry check: ||skew||_F / ||sym||_F = {rel_skew:.4f}")

    # Tau and bias scale the logits; if tau is tiny, the learned bilinear form is effectively muted.
    tau = getattr(head, "tau", None)
    tau_val = tau() if callable(tau) else tau
    if tau_val is not None:
        tau_f = float(tau_val.detach().cpu().item())
        print(f"[PDL] tau={tau_f:.6f}")
        out["tau"] = {"value": tau_f}
    bias = getattr(head, "bias", None)
    if bias is not None:
        b = bias.detach().cpu().numpy()
        print(f"[PDL] bias: shape={b.shape} mean={float(b.mean()):+.6f} std={float(b.std()):.6f}")
        out["bias"] = {"mean": float(b.mean()), "std": float(b.std())}

    return out


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

    if args.analyze_bilinear:
        print("\n=== Bilinear similarity diagnostics ===")
        _ = analyze_bilinear_similarity(head, svd_topk=int(args.svd_topk))

    if args.save_npz is not None:
        out_path = Path(args.save_npz).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **{k.replace(".", "_"): v for k, v in np_params.items()})
        print(f"\nSaved TabPDL head parameters to {out_path}")


if __name__ == "__main__":
    main()
