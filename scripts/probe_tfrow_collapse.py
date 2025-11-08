#!/usr/bin/env python3
"""Frozen collapse probe on TFrow (EA vs SA).

Runs inference only (no training) on a few validation batches twice—once with
standard attention (SA) and once with elliptical attention (EA)—and measures
representation collapse inside TFrow.

For each run (SA, EA), the script captures per-row embeddings before TFrow
(the row’s token sequence after TFcol) and after TFrow (the 4×[CLS] outputs
concatenated). It then computes:
  - mean pairwise cosine similarity among row embeddings after TFrow
  - optional Δ-similarity = (mean cosine after TFrow) − (mean cosine before TFrow)
  - optional attention entropy per head in TFrow’s last block (higher is less peaky)

Go / no-go rule:
If EA gives lower collapse (i.e., lower mean pairwise cosine or lower Δ-similarity)
and equal or better zero-shot accuracy than SA on the same batches, that’s a strong
“EA will help” signal for TabICL.

Example:
  python scripts/probe_tfrow_collapse.py \
    --datasets iris,wine \
    --n_estimators 4 --max_variants 4 --log_attn_entropy
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split


# Make local src importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tabicl.model.attention import compute_elliptical_diag
from tabicl.model.layers import MultiheadAttentionBlock
from tabicl.sklearn.classifier import TabICLClassifier


# ---------------------- utils ----------------------


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fetch_openml_dataset(name_or_id: str | int) -> Tuple[pd.DataFrame, pd.Series, str]:
    import openml  # lazy import

    if isinstance(name_or_id, int) or (str(name_or_id).isdigit()):
        ds = openml.datasets.get_dataset(int(name_or_id))
        X, y, _, _ = ds.get_data(
            target=getattr(ds, "default_target_attribute", None), dataset_format="dataframe"
        )
        X, y = _extract_target_if_missing(X, y, ds_name=ds.name)
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
    X, y, _, _ = ds.get_data(target=getattr(ds, "default_target_attribute", None), dataset_format="dataframe")
    X, y = _extract_target_if_missing(X, y, ds_name=ds.name)
    return X, y, ds.name


def _extract_target_if_missing(
    X: pd.DataFrame, y: Optional[pd.Series], ds_name: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    if y is not None and not (isinstance(y, type(None))):
        return X, y
    cols = list(X.columns)
    preferred = ["class", "Class", "target", "Target", "label", "Label", "y"]
    for c in preferred:
        if c in cols:
            return X.drop(columns=[c]), X[c].copy()
    for c in cols:
        nunique = X[c].nunique(dropna=True)
        if 2 <= nunique <= min(50, max(2, X.shape[0] // 2)):
            return X.drop(columns=[c]), X[c].copy()
    c = cols[-1]
    return X.drop(columns=[c]), X[c].copy()


def set_attn_mode(model, mode: str = "EA") -> None:
    """Toggle attention mode for TFrow and TFicl blocks.

    mode: 'EA' for parameter-free elliptical; 'SA' for identity metric.
    """
    identity = (mode.upper() == "SA")
    for blk in model.row_interactor.tf_row.blocks:
        blk.elliptical = True
        blk.elliptical_override = "identity" if identity else "none"
        blk.elliptical_manual_m = None
    for blk in model.icl_predictor.tf_icl.blocks:
        blk.elliptical = True
        blk.elliptical_override = "identity" if identity else "none"
        blk.elliptical_manual_m = None
    model.eval()


def mean_pairwise_cos(X: torch.Tensor) -> float:
    """Mean pairwise cosine similarity for rows of X (N,D)."""
    if X.ndim != 2:
        X = X.view(X.shape[0], -1)
    N = X.shape[0]
    if N <= 1:
        return float("nan")
    Xn = F.normalize(X, dim=-1)
    S = Xn @ Xn.t()
    # exclude diagonal
    mask = torch.ones_like(S, dtype=torch.bool)
    mask.fill_diagonal_(False)
    return float(S[mask].mean().item())


@dataclass
class BatchProbeResult:
    mean_cos_after: float
    mean_cos_before: Optional[float]
    delta_sim: Optional[float]
    attn_entropy_per_head: Optional[np.ndarray]


def _compute_last_block_attn_entropy(
    blocks: List[MultiheadAttentionBlock],
    rope,
    x_input: torch.Tensor,
    use_ea: bool,
    num_cls: int,
) -> np.ndarray:
    """Attention entropy per head at the last TFrow block.

    We run all but the last block with their native forward to get the correct
    input to the last block and the previous V. Then we reconstruct Q/K/V at the
    last block input and estimate softmax attention to compute entropy.
    """
    with torch.no_grad():
        x = x_input
        v_prev = None
        # Run through all but last block to get x at last block input and v_prev
        for idx, blk in enumerate(blocks[:-1]):
            x = blk(x, key_padding_mask=None, attn_mask=None, rope=rope, v_prev=v_prev, block_index=idx)
            v_prev = getattr(blk, "_last_v", None)

        # Last block: reconstruct q/k/v and compute attention entropy
        last = blocks[-1]
        q = x
        qn = last.norm1(q) if last.norm_first else q
        # Match projection weight dtype to avoid AMP dtype mismatch
        qn = qn.to(dtype=last.attn.in_proj_weight.dtype)
        E = qn.shape[-1]
        nh = last.attn.num_heads
        hd = E // nh
        B, T, L, _ = qn.shape

        q_proj, k_proj, v_proj = F._in_projection_packed(
            qn, qn, qn, last.attn.in_proj_weight, last.attn.in_proj_bias
        )
        qh = q_proj.view(B, T, L, nh, hd).transpose(-3, -2)
        kh = k_proj.view(B, T, L, nh, hd).transpose(-3, -2)
        vh = v_proj.view(B, T, L, nh, hd).transpose(-3, -2)
        if rope is not None:
            qh = rope.rotate_queries_or_keys(qh)
            kh = rope.rotate_queries_or_keys(kh)

        if use_ea and (v_prev is not None):
            m = compute_elliptical_diag(vh, v_prev, delta=last.elliptical_delta, scale_mode=last.elliptical_scale_mode)
            m_bc = m.view(1, 1, nh, 1, hd).to(device=qn.device, dtype=qn.dtype)
            qh = qh * m_bc
            kh = kh * m_bc

        scale = 1.0 / math.sqrt(hd)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale  # (B,T,nh,L,L)
        probs = scores.softmax(dim=-1)
        probs_cls = probs[..., :num_cls, :]  # (B,T,nh,C,L)
        ent = -(probs_cls * (probs_cls.clamp_min(1e-12).log())).sum(dim=-1)  # (B,T,nh,C)
        head_ent = ent.mean(dim=(0, 1, -1)).detach().cpu().numpy()  # (nh,)

    return head_ent


def probe_one_variant(
    model,
    X_variant: np.ndarray,
    y_train_variant: np.ndarray,
    mode: str,
    log_attn_entropy: bool,
) -> BatchProbeResult:
    """Probe collapse metrics for a single variant (one ensemble view)."""
    set_attn_mode(model, mode=mode)

    dev = next(model.parameters()).device
    num_cls = model.row_interactor.num_cls

    # Prepare tensors
    X_tensor = torch.as_tensor(X_variant, dtype=torch.float32, device=dev)[None, ...]  # (1, T, H)
    y_train = torch.as_tensor(y_train_variant, dtype=torch.long, device=dev)[None, ...]  # (1, train_size)

    T = X_tensor.shape[1]
    train_size = y_train.shape[1]
    test_slice = slice(train_size, T)

    # Before TFrow: token embeddings after TFcol
    with torch.no_grad():
        embeddings = model.col_embedder(X_tensor, train_size=train_size)  # (1, T, H+C, E)
        # Mean-pool over feature tokens (exclude reserved CLS slots)
        feats = embeddings[:, :, num_cls:, :]  # (1,T,H,E)
        pre_row_vecs = feats.mean(dim=-2).squeeze(0)[test_slice].float()  # (T_test, E)

        # After TFrow: manual run through TFrow to get CLS outputs and (optionally) attn entropy
        blocks = list(model.row_interactor.tf_row.blocks)
        rope = model.row_interactor.tf_row.rope

        # Compute last-block attention entropy (approx.) if requested
        attn_entropy = None
        if log_attn_entropy:
            attn_entropy = _compute_last_block_attn_entropy(
                blocks=blocks,
                rope=rope,
                x_input=embeddings,
                use_ea=(mode.upper() == "EA"),
                num_cls=num_cls,
            )

        # Full TFrow forward for final CLS embeddings (use the actual module for fidelity)
        row_repr = model.row_interactor(embeddings)  # (1, T, C*E)
        post_row_vecs = row_repr.squeeze(0)[test_slice].float()  # (T_test, C*E)

    mean_cos_after = mean_pairwise_cos(post_row_vecs)
    mean_cos_before = mean_pairwise_cos(pre_row_vecs)
    delta_sim = None if math.isnan(mean_cos_before) else (mean_cos_after - mean_cos_before)

    return BatchProbeResult(
        mean_cos_after=mean_cos_after,
        mean_cos_before=mean_cos_before,
        delta_sim=delta_sim,
        attn_entropy_per_head=attn_entropy,
    )


def evaluate_accuracy(clf: TabICLClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float]:
    """Return (accuracy, balanced_accuracy)."""
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    bacc = float(balanced_accuracy_score(y_test, y_pred))
    return acc, bacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--checkpoint",
        default=os.path.join(REPO_ROOT, "checkpoints", "tabicl-classifier-v1.1-0506.ckpt"),
    )
    ap.add_argument("--datasets", default="iris,wine,breast-w")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_estimators", type=int, default=4)
    ap.add_argument("--max_variants", type=int, default=4, help="Number of ensemble variants to probe per dataset")
    ap.add_argument("--log_attn_entropy", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    set_seed(args.random_state)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [s.strip() for s in str(args.datasets).split(",") if s.strip()]

    results = []

    for ds in datasets:
        try:
            X, y, ds_name = fetch_openml_dataset(ds)
        except Exception as e:
            print(f"[WARN] Could not load dataset '{ds}': {e}. Skipping.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(args.test_size), random_state=args.random_state, stratify=y
        )

        # Build classifier + transforms
        clf = TabICLClassifier(
            device=dev,
            model_path=args.checkpoint,
            allow_auto_download=False,
            use_hierarchical=True,
            n_estimators=int(args.n_estimators),
            random_state=args.random_state,
            verbose=False,
        )
        clf.fit(X_train, y_train)
        model = clf.model_
        model.eval()

        # Prepare ensemble batches for probing
        X_val_num = clf.X_encoder_.transform(X_test)
        data = clf.ensemble_generator_.transform(X_val_num)  # Ordered by normalization method

        # Helper to iterate a few variants only
        def iter_variants(max_k: int):
            for norm_method, (Xs, ys) in data.items():
                n = Xs.shape[0]
                for i in range(min(n, max_k)):
                    yield (norm_method, i, Xs[i], ys[i])

        metrics = {"SA": [], "EA": []}

        for mode in ("SA", "EA"):
            set_attn_mode(model, mode=mode)
            for norm_method, i, Xv, yv in iter_variants(args.max_variants):
                try:
                    res = probe_one_variant(model, Xv, yv, mode=mode, log_attn_entropy=args.log_attn_entropy)
                except RuntimeError as e:
                    print(f"[WARN] Probe failed on {ds_name}/{norm_method}[{i}] in {mode}: {e}")
                    continue
                metrics[mode].append(res)

        # Aggregate collapse metrics
        def agg(vals: List[BatchProbeResult]) -> Dict[str, float]:
            if not vals:
                return {"mean_cos_after": float("nan"), "delta_sim": float("nan")}
            m_after = float(np.nanmean([v.mean_cos_after for v in vals]))
            d_sim = float(np.nanmean([v.delta_sim for v in vals if v.delta_sim is not None]))
            return {"mean_cos_after": m_after, "delta_sim": d_sim}

        agg_SA = agg(metrics["SA"]) ; agg_EA = agg(metrics["EA"]) 

        # Accuracy per mode
        set_attn_mode(model, mode="SA")
        acc_SA, bacc_SA = evaluate_accuracy(clf, X_test, y_test)
        set_attn_mode(model, mode="EA")
        acc_EA, bacc_EA = evaluate_accuracy(clf, X_test, y_test)

        # Go/no-go decision for this dataset
        lower_collapse = (
            (agg_EA["mean_cos_after"] < agg_SA["mean_cos_after"]) or (
                (not math.isnan(agg_EA["delta_sim"])) and (agg_EA["delta_sim"] < agg_SA["delta_sim"]))
        )
        no_worse_acc = (acc_EA >= acc_SA - 1e-6) and (bacc_EA >= bacc_SA - 1e-6)
        go = lower_collapse and no_worse_acc

        results.append(
            dict(
                dataset=ds_name,
                mean_cos_after_SA=agg_SA["mean_cos_after"],
                mean_cos_after_EA=agg_EA["mean_cos_after"],
                delta_sim_SA=agg_SA["delta_sim"],
                delta_sim_EA=agg_EA["delta_sim"],
                acc_SA=acc_SA,
                acc_EA=acc_EA,
                bacc_SA=bacc_SA,
                bacc_EA=bacc_EA,
                go=go,
            )
        )

        # Verbose per-dataset log
        print(
            f"[{ds_name}] collapse-SA={agg_SA['mean_cos_after']:.4f} collapse-EA={agg_EA['mean_cos_after']:.4f} "
            f"Δsim-SA={agg_SA['delta_sim']:.4f} Δsim-EA={agg_EA['delta_sim']:.4f} "
            f"acc-SA={acc_SA:.4f} acc-EA={acc_EA:.4f} bacc-SA={bacc_SA:.4f} bacc-EA={bacc_EA:.4f} go={go}"
        )

        if args.log_attn_entropy and metrics["SA"] and metrics["EA"]:
            # Report average head entropy (last block)
            def avg_head_ent(res_list: List[BatchProbeResult]) -> Optional[np.ndarray]:
                ents = [r.attn_entropy_per_head for r in res_list if r.attn_entropy_per_head is not None]
                if not ents:
                    return None
                lens = [e.shape[0] for e in ents]
                nh = min(lens)
                ents = np.stack([e[:nh] for e in ents], axis=0)
                return ents.mean(axis=0)

            ent_SA = avg_head_ent(metrics["SA"]) ; ent_EA = avg_head_ent(metrics["EA"]) 
            if ent_SA is not None and ent_EA is not None:
                print(f"  head-entropy-SA: {np.round(ent_SA, 4)}")
                print(f"  head-entropy-EA: {np.round(ent_EA, 4)}")

    # Summary
    if not results:
        print("No results to summarize.")
        return
    wins = sum(1 for r in results if r["go"]) ; losses = len(results) - wins
    print("Summary:")
    for r in results:
        print(
            f"- {r['dataset']}: collapse-SA={r['mean_cos_after_SA']:.4f}, collapse-EA={r['mean_cos_after_EA']:.4f}, "
            f"Δsim-SA={r['delta_sim_SA']:.4f}, Δsim-EA={r['delta_sim_EA']:.4f}, "
            f"acc-SA={r['acc_SA']:.4f}, acc-EA={r['acc_EA']:.4f}, "
            f"bacc-SA={r['bacc_SA']:.4f}, bacc-EA={r['bacc_EA']:.4f}, go={r['go']}"
        )
    print(f"Go on {wins}/{len(results)} datasets (losses={losses}).")


if __name__ == "__main__":
    main()
