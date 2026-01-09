#!/usr/bin/env python3
"""
Collapse-focused comparison of mini-TabICL SA vs EA.

This script quantifies *representation collapse* and *attention collapse*
for standard attention (SA) vs elliptical attention (EA), without doing
any extra training beyond fitting the TabICLClassifier.

For each dataset and model (SA / EA) it reports:

  1) Global TFrow representation collapse on test embeddings:
       - collapse_top1, collapse_top5  (fraction of variance in top-1 / top-5 PCs)
       - d_eff_global                  (effective dimension from spectrum)
       - mean_cosine_global            (mean pairwise cosine over rows)

  2) Class-wise structure on test embeddings:
       - within-class collapse_top1/5 (averaged over classes)
       - within-class d_eff_within
       - between-class d_eff_between
       - mean cosine between class means

  3) Attention collapse (TFicl, last block):
       - mean/std N_eff over test rows (effective number of attended neighbours)

Results are appended to:
    results/ea_sa_collapse.csv

Usage example
-------------
  python scripts/analysis/ea_sa_collapse.py \\
      --datasets 23,48,750 \\
      --sa_checkpoint checkpoints/mini_tabicl_stage2_sa/step-1000.ckpt \\
      --ea_checkpoint checkpoints/mini_tabicl_stage2_ea_row_icl/step-1000.ckpt
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

# Make local src importable
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tabicl.sklearn.classifier import TabICLClassifier  # type: ignore
from tabicl.pdlc.embed import extract_tf_row_embeddings  # type: ignore
from tabicl.model.attention import compute_elliptical_diag  # type: ignore


# We re-use a few helpers from compare_mini_tabicl_sa_ea to keep behaviour aligned.
try:
    import scripts.analysis.compare_tabicl_sa_ea as cmp  # type: ignore
except ImportError:  # pragma: no cover - should not happen when run from repo root
    cmp = None  # type: ignore


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fetch_openml_dataset(name_or_id: str | int) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Load an OpenML dataset by ID or (partial) name."""
    import openml  # lazy import

    if isinstance(name_or_id, int) or (str(name_or_id).isdigit()):
        ds = openml.datasets.get_dataset(int(name_or_id))
        X, y, _, _ = ds.get_data(
            target=getattr(ds, "default_target_attribute", None),
            dataset_format="dataframe",
        )
        if y is None:
            raise ValueError(f"Dataset {ds.name} (id={ds.dataset_id}) has no target.")
        return X, y, ds.name

    name = str(name_or_id).lower()
    df = openml.datasets.list_datasets(output_format="dataframe")
    exact = df[df["name"].str.lower() == name]
    if len(exact) == 0:
        contains = df[df["name"].str.lower().str.contains(name)]
        if len(contains) == 0:
            raise ValueError(f"OpenML dataset not found: {name_or_id}")
        row = contains.sort_values("NumberOfInstances").iloc[0]
    else:
        row = exact.sort_values("NumberOfInstances").iloc[0]
    did = int(row["did"]) if "did" in row else int(row.get("dataset_id", row.get("ID")))
    ds = openml.datasets.get_dataset(did)
    X, y, _, _ = ds.get_data(
        target=getattr(ds, "default_target_attribute", None),
        dataset_format="dataframe",
    )
    if y is None:
        raise ValueError(f"Dataset {ds.name} (id={did}) has no target.")
    return X, y, ds.name


# ---------------------------------------------------------------------------
# Representation collapse metrics
# ---------------------------------------------------------------------------


def covariance_spectrum(Z: np.ndarray) -> np.ndarray:
    """Eigenvalues (descending) of covariance of Z (rows = samples)."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] < 2:
        return np.asarray([])
    Zc = Z - Z.mean(axis=0, keepdims=True)
    C = np.cov(Zc, rowvar=False)
    evals = np.linalg.eigvalsh(C)
    evals = np.sort(np.clip(evals, a_min=0.0, a_max=None))[::-1]
    return evals


def collapse_score(evals: np.ndarray, k: int = 1) -> float:
    """Fraction of variance in top-k eigenvalues."""
    if evals.size == 0:
        return float("nan")
    k = max(1, min(k, evals.size))
    num = evals[:k].sum()
    den = evals.sum()
    return float(num / den) if den > 0 else float("nan")


def effective_dimension(evals: np.ndarray) -> float:
    """Participation-ratio effective dimension from eigenvalues."""
    if evals.size == 0:
        return float("nan")
    s1 = float(evals.sum())
    s2 = float((evals**2).sum())
    if s2 <= 0.0:
        return float("nan")
    return (s1 * s1) / s2


def mean_pairwise_cosine(Z: np.ndarray, max_samples: int = 512) -> float:
    """Average cosine similarity over a subset of pairs."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] < 2:
        return float("nan")
    n = Z.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_samples, replace=False)
        Z = Z[idx]
        n = Z.shape[0]
    norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    Z_norm = Z / norms
    S = Z_norm @ Z_norm.T
    mask = ~np.eye(n, dtype=bool)
    return float(S[mask].mean())


@dataclass
class CollapseMetrics:
    collapse_top1: float
    collapse_top5: float
    d_eff: float
    mean_cos: float
    collapse_within_top1: float
    collapse_within_top5: float
    d_eff_within: float
    d_eff_between: float
    mean_cos_between_means: float


def _classwise_collapse(Z: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Within-class collapse (top1/top5, d_eff) averaged over classes."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    vals_top1: List[float] = []
    vals_top5: List[float] = []
    vals_deff: List[float] = []
    for c in classes:
        mask = y == c
        if mask.sum() < 2:
            continue
        evals_c = covariance_spectrum(Z[mask])
        if evals_c.size == 0:
            continue
        vals_top1.append(collapse_score(evals_c, k=1))
        vals_top5.append(collapse_score(evals_c, k=5))
        vals_deff.append(effective_dimension(evals_c))
    if not vals_top1:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals_top1)), float(np.mean(vals_top5)), float(np.mean(vals_deff))


def _between_class_metrics(Z: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Effective dimension and mean cosine between class means."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    if classes.size < 2:
        return float("nan"), float("nan")
    means: List[np.ndarray] = []
    for c in classes:
        mask = y == c
        if not np.any(mask):
            continue
        means.append(Z[mask].mean(axis=0))
    if not means:
        return float("nan"), float("nan")
    M = np.stack(means, axis=0)
    evals_means = covariance_spectrum(M)
    d_eff_between = effective_dimension(evals_means)
    cos_between = mean_pairwise_cosine(M, max_samples=1024)
    return d_eff_between, cos_between


def compute_collapse_metrics(Z: np.ndarray, y: np.ndarray) -> CollapseMetrics:
    """Full set of collapse metrics for a set of row embeddings."""
    evals = covariance_spectrum(Z)
    collapse_top1 = collapse_score(evals, k=1)
    collapse_top5 = collapse_score(evals, k=5)
    d_eff = effective_dimension(evals)
    mean_cos = mean_pairwise_cosine(Z)

    cw1, cw5, deff_w = _classwise_collapse(Z, y)
    deff_b, cos_b = _between_class_metrics(Z, y)

    return CollapseMetrics(
        collapse_top1=collapse_top1,
        collapse_top5=collapse_top5,
        d_eff=d_eff,
        mean_cos=mean_cos,
        collapse_within_top1=cw1,
        collapse_within_top5=cw5,
        d_eff_within=deff_w,
        d_eff_between=deff_b,
        mean_cos_between_means=cos_b,
    )


# ---------------------------------------------------------------------------
# Attention collapse metrics (TFicl last block)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _compute_test_to_train_weights(
    model,
    R_cond: torch.Tensor,
    train_size: int,
    chunk_test: int = 1024,
    use_fp16: bool = True,
) -> torch.Tensor:
    """Mean-head test→train attention weights (T_test, T_train) with low memory."""
    from tabicl.model.learning import ICLearning  # local import to avoid cycles

    device = R_cond.device
    enc: ICLearning = model.icl_predictor
    tf_icl = enc.tf_icl
    blocks = list(tf_icl.blocks)

    x = R_cond
    v_prev = None
    for i, blk in enumerate(blocks[:-1]):
        x = blk(x, key_padding_mask=None, attn_mask=train_size, rope=tf_icl.rope, v_prev=v_prev, block_index=i)
        v_prev = getattr(blk, "_last_v", None)

    last = blocks[-1]
    if last.norm_first:
        q_in = last.norm1(x)
    else:
        q_in = x
    B, T, E = q_in.shape
    nh = last.attn.num_heads
    hs = E // nh
    q, k, v = F._in_projection_packed(q_in, q_in, q_in, last.attn.in_proj_weight, last.attn.in_proj_bias)
    q = q.view(B, T, nh, hs).transpose(-3, -2)  # (B, nh, T, hs)
    k = k.view(B, T, nh, hs).transpose(-3, -2)
    v = v.view(B, T, nh, hs).transpose(-3, -2)

    if last.elliptical and (v_prev is not None) and (len(blocks) - 1 >= 1):
        keep = torch.zeros(T, device=device, dtype=torch.float32)
        keep[:train_size] = 1.0
        m = compute_elliptical_diag(
            v,
            v_prev,
            delta=last.elliptical_delta,
            scale_mode=last.elliptical_scale_mode,
            mask_keep=keep,
        )
        m_bc = m.view(1, 1, nh, 1, hs)
        q = q * m_bc
        k = k * m_bc

    # Restrict to test queries and train keys
    q_test = q[..., train_size:, :]   # (B, nh, T_test, hs)
    k_train = k[..., :train_size, :]  # (B, nh, T_train, hs)
    T_test = q_test.shape[-2]
    T_train = k_train.shape[-2]

    Bnh = B * nh
    q_test_b = q_test.reshape(Bnh, T_test, hs)
    k_train_b = k_train.reshape(Bnh, T_train, hs)
    if use_fp16:
        q_test_b = q_test_b.to(torch.float16)
        k_train_b = k_train_b.to(torch.float16)

    out = torch.zeros(T_test, T_train, device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(hs)

    for start in range(0, T_test, chunk_test):
        end = min(T_test, start + chunk_test)
        q_chunk = q_test_b[:, start:end, :]  # (Bnh, t, hs)
        scores = torch.bmm(q_chunk, k_train_b.transpose(1, 2)) * scale  # (Bnh, t, T_train)
        w = torch.softmax(scores.to(torch.float32), dim=-1)
        w_mean = w.mean(dim=0)  # (t, T_train)
        out[start:end, :] += w_mean

    return out  # (T_test, T_train)


def attention_neff_summary(weights_tt: torch.Tensor) -> Tuple[float, float]:
    """Mean/std N_eff over test rows given (T_test, T_train) weights."""
    if weights_tt.ndim != 2 or weights_tt.numel() == 0:
        return float("nan"), float("nan")

    W = weights_tt
    finite = torch.isfinite(W)
    any_finite = finite.any(dim=1)
    if not any_finite.any():
        return float("nan"), float("nan")

    W = W[any_finite]
    finite = finite[any_finite]

    W_clean = torch.where(finite, W, torch.zeros_like(W))
    row_sums = W_clean.sum(dim=1, keepdim=True).clamp_min(1e-12)
    alpha = W_clean / row_sums  # (T_eff, T_train)

    s2 = (alpha**2).sum(dim=1)
    mask = (s2 > 0) & torch.isfinite(s2)
    if not mask.any():
        return float("nan"), float("nan")

    neff = 1.0 / s2[mask]
    neff_np = neff.detach().cpu().numpy().astype(np.float64)
    return float(neff_np.mean()), float(neff_np.std())


def _build_episode(clf: TabICLClassifier, X_te: pd.DataFrame) -> Tuple[torch.Tensor, int]:
    """Construct a single ICL episode (R_cond, train_size) from X_te."""
    X_te_num = clf.X_encoder_.transform(X_te)
    data = clf.ensemble_generator_.transform(X_te_num)
    methods = list(data.keys())
    norm_method = methods[0]
    Xs, ys_shifted = data[norm_method]
    shuffle_patterns = clf.ensemble_generator_.feature_shuffle_patterns_[norm_method]

    # Prefer identity permutation if present
    vidx = 0
    for i, p in enumerate(shuffle_patterns):
        if list(p) == sorted(p):
            vidx = i
            break

    X_variant = Xs[vidx]
    y_train_shifted = ys_shifted[vidx]
    train_size = y_train_shifted.shape[0]

    model = clf.model_.to(clf.device_)
    model.eval()

    X_tensor = torch.from_numpy(X_variant).float().unsqueeze(0).to(clf.device_)
    with torch.no_grad():
        col_out = model.col_embedder(
            X_tensor,
            train_size=train_size,
            mgr_config=clf.inference_config_.COL_CONFIG,
        )
        row_reps = model.row_interactor(col_out, mgr_config=clf.inference_config_.ROW_CONFIG)
        yt = torch.as_tensor(y_train_shifted, device=clf.device_, dtype=torch.float32).unsqueeze(0)
        R_cond = row_reps.clone()
        # Label conditioning (class-shifted)
        R_cond[:, :train_size] = R_cond[:, :train_size] + model.icl_predictor.y_encoder(yt)

    return R_cond, train_size


# ---------------------------------------------------------------------------
# Dataset-level analysis
# ---------------------------------------------------------------------------


def _fit_classifier(
    checkpoint: Path,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    device: str,
    n_estimators: int,
    seed: int,
    use_hierarchical: bool = True,
) -> TabICLClassifier:
    clf = TabICLClassifier(
        device=device,
        model_path=str(checkpoint),
        allow_auto_download=False,
        use_hierarchical=use_hierarchical,
        n_estimators=n_estimators,
        random_state=seed,
        verbose=False,
    )
    clf.fit(X_tr, y_tr)
    return clf


def analyze_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
    sa_ckpt: Path,
    ea_ckpt: Path,
    device: str,
    n_estimators: int,
    seed: int,
    test_size: float,
    use_hierarchical: bool,
) -> List[Dict[str, object]]:
    """Compute collapse metrics for SA and EA on a single dataset."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=seed,
        stratify=y,
    )

    clf_sa = _fit_classifier(sa_ckpt, X_tr, y_tr, device, n_estimators, seed, use_hierarchical)
    clf_ea = _fit_classifier(ea_ckpt, X_tr, y_tr, device, n_estimators, seed, use_hierarchical)

    # Behavioural metrics on the held-out test split (for context)
    def _behaviour(clf: TabICLClassifier) -> Tuple[float, float, float]:
        proba = clf.predict_proba(X_te)
        y_pred = clf.y_encoder_.inverse_transform(np.argmax(proba, axis=1))
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="macro")
        ll = log_loss(y_te, proba, labels=clf.classes_)
        return float(acc), float(f1), float(ll)

    acc_sa, f1_sa, nll_sa = _behaviour(clf_sa)
    acc_ea, f1_ea, nll_ea = _behaviour(clf_ea)

    print(f"\n=== Collapse: {name} ===")
    print(
        f"  Behaviour (test): "
        f"acc(SA)={acc_sa:.4f}, acc(EA)={acc_ea:.4f}; "
        f"F1(SA)={f1_sa:.4f}, F1(EA)={f1_ea:.4f}; "
        f"NLL(SA)={nll_sa:.4f}, NLL(EA)={nll_ea:.4f}"
    )

    # TFrow embeddings for test rows (pre-ICL)
    res_sa = extract_tf_row_embeddings(clf_sa, X_te, choose_random_variant=False)
    res_ea = extract_tf_row_embeddings(clf_ea, X_te, choose_random_variant=False)
    Z_sa = res_sa["embeddings_test"]
    Z_ea = res_ea["embeddings_test"]

    cm_sa = compute_collapse_metrics(Z_sa, y_te.values)
    cm_ea = compute_collapse_metrics(Z_ea, y_te.values)

    print(
        f"  Global collapse_top1: SA={cm_sa.collapse_top1:.4f}, EA={cm_ea.collapse_top1:.4f} | "
        f"d_eff_global: SA={cm_sa.d_eff:.1f}, EA={cm_ea.d_eff:.1f}"
    )
    print(
        f"  Within-class collapse_top1: SA={cm_sa.collapse_within_top1:.4f}, "
        f"EA={cm_ea.collapse_within_top1:.4f}"
    )
    print(
        f"  Between-class d_eff: SA={cm_sa.d_eff_between:.2f}, EA={cm_ea.d_eff_between:.2f}; "
        f"mean cos(μ_c): SA={cm_sa.mean_cos_between_means:.4f}, EA={cm_ea.mean_cos_between_means:.4f}"
    )

    # Attention collapse via N_eff from TFicl attention over train rows
    R_sa, train_size = _build_episode(clf_sa, X_te)
    R_ea, _ = _build_episode(clf_ea, X_te)
    w_sa_tt = _compute_test_to_train_weights(clf_sa.model_, R_sa, train_size, chunk_test=1024)
    w_ea_tt = _compute_test_to_train_weights(clf_ea.model_, R_ea, train_size, chunk_test=1024)
    neff_sa_mean, neff_sa_std = attention_neff_summary(w_sa_tt)
    neff_ea_mean, neff_ea_std = attention_neff_summary(w_ea_tt)

    print(
        f"  Attention N_eff (test→train): "
        f"SA={neff_sa_mean:.1f}±{neff_sa_std:.1f}, "
        f"EA={neff_ea_mean:.1f}±{neff_ea_std:.1f}"
    )

    rows: List[Dict[str, object]] = []
    for model_name, cm, acc, f1, nll, neff_mean, neff_std in [
        ("SA", cm_sa, acc_sa, f1_sa, nll_sa, neff_sa_mean, neff_sa_std),
        ("EA", cm_ea, acc_ea, f1_ea, nll_ea, neff_ea_mean, neff_ea_std),
    ]:
        rows.append(
            {
                "dataset": name,
                "model": model_name,
                "n_train": int(len(X_tr)),
                "n_test": int(len(X_te)),
                "n_classes": int(y.nunique()),
                "acc": acc,
                "f1_macro": f1,
                "nll": nll,
                "collapse_top1_global": cm.collapse_top1,
                "collapse_top5_global": cm.collapse_top5,
                "d_eff_global": cm.d_eff,
                "mean_cos_global": cm.mean_cos,
                "collapse_top1_within": cm.collapse_within_top1,
                "collapse_top5_within": cm.collapse_within_top5,
                "d_eff_within": cm.d_eff_within,
                "d_eff_between": cm.d_eff_between,
                "mean_cos_between_means": cm.mean_cos_between_means,
                "neff_mean": neff_mean,
                "neff_std": neff_std,
                "seed": seed,
                "n_estimators": n_estimators,
                "use_hierarchical": bool(use_hierarchical),
                "sa_checkpoint": str(sa_ckpt),
                "ea_checkpoint": str(ea_ckpt),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Measure representation and attention collapse for SA vs EA.")
    ap.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated list of OpenML IDs or names (e.g. '23,48,750').",
    )
    ap.add_argument(
        "--sa_checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "mini_tabicl_stage2_sa" / "step-1000.ckpt"),
        help="Stage-2 SA checkpoint path.",
    )
    ap.add_argument(
        "--ea_checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "mini_tabicl_stage2_ea_row_icl" / "step-1000.ckpt"),
        help="Stage-2 EA checkpoint path.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu).",
    )
    ap.add_argument(
        "--n_estimators",
        type=int,
        default=32,
        help="Number of ensemble estimators for TabICLClassifier.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    ap.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Test fraction for the collapse analysis split.",
    )
    ap.add_argument(
        "--use_hierarchical",
        default=True,
        action="store_true",
        help="Use hierarchical TabICLClassifier (default: True).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sa_ckpt = Path(args.sa_checkpoint)
    ea_ckpt = Path(args.ea_checkpoint)
    if not sa_ckpt.is_file():
        raise SystemExit(f"SA checkpoint not found: {sa_ckpt}")
    if not ea_ckpt.is_file():
        raise SystemExit(f"EA checkpoint not found: {ea_ckpt}")

    specs = [s.strip() for s in str(args.datasets).split(",") if s.strip()]
    if not specs:
        raise SystemExit("No datasets specified; use --datasets id1,id2,...")

    all_rows: List[Dict[str, object]] = []
    for spec in specs:
        try:
            X, y, name = fetch_openml_dataset(spec)
        except Exception as e:  # pragma: no cover - runtime guard
            print(f"[WARN] Could not load dataset '{spec}': {e}. Skipping.")
            continue
        rows = analyze_dataset(
            X=X,
            y=y,
            name=name,
            sa_ckpt=sa_ckpt,
            ea_ckpt=ea_ckpt,
            device=device,
            n_estimators=args.n_estimators,
            seed=args.seed,
            test_size=args.test_size,
            use_hierarchical=bool(args.use_hierarchical),
        )
        all_rows.extend(rows)

    if not all_rows:
        print("No results to save.")
        return

    out_path = REPO_ROOT / "results" / "ea_sa_collapse.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    header = not out_path.exists()
    df.to_csv(out_path, mode="a", index=False, header=header)
    print(f"\nSaved collapse metrics to {out_path}")


if __name__ == "__main__":
    main()
