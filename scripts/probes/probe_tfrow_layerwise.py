#!/usr/bin/env python3
"""Layer-wise TFrow collapse probe (SA vs EA) with class stratification.

Runs inference only (no training) on the same validation batches twice:
  - SA: standard attention (identity metric)
  - EA: parameter-free elliptical attention, active only from TFrow layer 2 onward

After each TFrow layer, collects the row embeddings that would be passed to TFicl
(concatenated CLS outputs, applying RowInteraction's out_ln), then computes:
  - mean pairwise cosine similarity among test rows
  - mean pairwise cosine within-class and between-class (using val labels)

Decision rule (simple):
  If EA has smaller similarity growth past L1 for between-class pairs and does not
  worsen zero-shot accuracy vs SA on the same batches, it is a green light.

Example:
  python scripts/probes/probe_tfrow_layerwise.py \
    --datasets iris,wine --n_estimators 4 --max_variants 4 --plot
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
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tabicl.sklearn.classifier import TabICLClassifier


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def set_attn_mode(model, mode: str = "EA") -> None:
    """Toggle attention mode for TFrow/TFicl blocks.

    mode: 'EA' uses parameter-free elliptical (from layer 2 onward),
          'SA' forces identity metric (all layers).
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


def set_tfrow_mode(model, identity: bool) -> None:
    """Set TFrow attention mode only; leave TFicl unchanged."""
    for blk in model.row_interactor.tf_row.blocks:
        blk.elliptical = True
        blk.elliptical_override = "identity" if identity else "none"
        blk.elliptical_manual_m = None
    model.eval()


def force_icl_identity(model) -> None:
    """Force TFicl to identity metric (SA) for accuracy gating."""
    for blk in model.icl_predictor.tf_icl.blocks:
        blk.elliptical = True
        blk.elliptical_override = "identity"
        blk.elliptical_manual_m = None
    model.eval()


def _layerwise_row_reprs(model, X_variant: np.ndarray, train_size: int) -> List[torch.Tensor]:
    """Return row embeddings after each TFrow layer for test rows only.

    Each element is a (N_test, C*E) float32 tensor.
    """
    with torch.no_grad():
        dev = next(model.parameters()).device
        ri = model.row_interactor
        num_cls = ri.num_cls
        tfrow = ri.tf_row
        rope = tfrow.rope
        blocks = list(tfrow.blocks)

        X_tensor = torch.as_tensor(X_variant, dtype=torch.float32, device=dev)[None, ...]  # (1,T,H)
        T_total = X_tensor.shape[1]
        test_slice = slice(train_size, T_total)

        # TFcol embeddings
        embeddings = model.col_embedder(X_tensor, train_size=train_size)  # (1,T,H+C,E)
        # Insert CLS tokens
        B, T = embeddings.shape[:2]
        cls_tokens = ri.cls_tokens.expand(B, T, num_cls, ri.embed_dim)
        embeddings[:, :, :num_cls] = cls_tokens.to(embeddings.device)

        # Iterate blocks and collect per-layer row representations
        x = embeddings
        v_prev = None
        outs: List[torch.Tensor] = []
        for idx, blk in enumerate(blocks):
            x = blk(x, key_padding_mask=None, attn_mask=None, rope=rope, v_prev=v_prev, block_index=idx)
            v_prev = getattr(blk, "_last_v", None)
            # CLS-only, apply out_ln like RowInteraction, then flatten
            cls_out = x[..., :num_cls, :]
            cls_out = ri.out_ln(cls_out)
            row_repr = cls_out.flatten(-2)  # (1,T,C*E)
            outs.append(row_repr.squeeze(0)[test_slice].float().cpu())  # (N_test, C*E)

    return outs


def _mean_pairwise_cos_stratified(X: torch.Tensor, y: np.ndarray) -> Tuple[float, float, float]:
    """Return (mean_all, mean_within, mean_between) pairwise cosine for rows X and labels y.

    Excludes diagonal for within-class computation.
    """
    if X.ndim != 2:
        X = X.view(X.shape[0], -1)
    n = X.shape[0]
    if n <= 1:
        return float("nan"), float("nan"), float("nan")
    Xn = F.normalize(X, dim=-1)
    S = (Xn @ Xn.t()).cpu().numpy()  # (n,n)
    np.fill_diagonal(S, np.nan)
    # Masks
    y = np.asarray(y)
    same = (y[:, None] == y[None, :])
    diff = ~same
    mean_all = np.nanmean(S)
    within = np.nanmean(S[same]) if np.any(same) else float("nan")
    between = np.nanmean(S[diff]) if np.any(diff) else float("nan")
    return float(mean_all), float(within), float(between)


@dataclass
class LayerwiseStats:
    layer_idx: int
    mean_all: float
    mean_within: float
    mean_between: float


def evaluate_accuracy(clf: TabICLClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float]:
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
    ap.add_argument("--n_estimators", type=int, default=32)
    ap.add_argument("--max_variants", type=int, default=4, help="Max ensemble variants to probe per dataset")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "runs"))
    args = ap.parse_args()

    set_seed(args.random_state)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [s.strip() for s in str(args.datasets).split(",") if s.strip()]

    for ds in datasets:
        try:
            X, y, ds_name = fetch_openml_dataset(ds)
        except Exception as e:
            print(f"[WARN] Could not load dataset '{ds}': {e}. Skipping.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(args.test_size), random_state=args.random_state, stratify=y
        )

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

        # Prepare ensemble variants (consistent across SA/EA)
        X_val_num = clf.X_encoder_.transform(X_test)
        data = clf.ensemble_generator_.transform(X_val_num)  # OrderedDict[norm_method] -> (Xs, ys_train)

        # Helper over variants
        def iter_variants(max_k: int):
            for norm_method, (Xs, ys_tr) in data.items():
                n = Xs.shape[0]
                for i in range(min(n, max_k)):
                    # also yield the feature permutation for alignment checks
                    perm = clf.ensemble_generator_.feature_shuffle_patterns_[norm_method][i]
                    yield norm_method, i, perm, Xs[i], ys_tr[i]

        # Aggregate per-layer stats per mode
        per_mode_stats: Dict[str, List[List[float]]] = {"SA": [], "EA": []}
        layers_count = None

        for mode in ("SA", "EA"):
            set_attn_mode(model, mode=mode)
            layer_sums = None  # shape (L, 3)
            layer_cnts = None  # shape (L, 3)
            for norm_method, v_idx, perm, Xv, yv_train in iter_variants(args.max_variants):
                train_size = int(yv_train.shape[0])
                # Alignment check: test rows should match preprocessor.transform(X_test)
                try:
                    pre = clf.ensemble_generator_.preprocessors_[norm_method]
                    X_test_tx = pre.transform(X_val_num)  # (N_test, F)
                    # Inverse the feature permutation used to construct Xv
                    inv = np.empty_like(np.array(perm))
                    inv[np.array(perm)] = np.arange(len(perm))
                    Xv_test_unperm = Xv[train_size:, :][:, inv]
                    if not np.allclose(Xv_test_unperm, X_test_tx, atol=1e-6):
                        print(
                            f"[WARN] Variant alignment failed for {ds_name}/{norm_method}[{v_idx}] — skipping this variant."
                        )
                        continue
                except Exception as e:
                    print(
                        f"[WARN] Alignment check error for {ds_name}/{norm_method}[{v_idx}] in {mode}: {e} — skipping."
                    )
                    continue

                layer_reprs = _layerwise_row_reprs(model, Xv, train_size)
                if layers_count is None:
                    layers_count = len(layer_reprs)
                # Compute stratified means per layer
                for li, Z in enumerate(layer_reprs):
                    mean_all, within, between = _mean_pairwise_cos_stratified(Z, y_test.values)
                    if layer_sums is None:
                        layer_sums = [[0.0, 0.0, 0.0] for _ in range(len(layer_reprs))]
                        layer_cnts = [[0, 0, 0] for _ in range(len(layer_reprs))]
                    # Accumulate only when finite; keep separate counts per metric
                    if not math.isnan(mean_all):
                        layer_sums[li][0] += mean_all
                        layer_cnts[li][0] += 1
                    if not math.isnan(within):
                        layer_sums[li][1] += within
                        layer_cnts[li][1] += 1
                    if not math.isnan(between):
                        layer_sums[li][2] += between
                        layer_cnts[li][2] += 1
            # Average across variants
            if layer_sums is None:
                print(f"[WARN] No valid variants for {ds_name} in mode {mode}")
                continue
            per_mode_stats[mode] = []
            for li in range(len(layer_sums)):
                means = []
                for j in range(3):
                    c = layer_cnts[li][j]
                    means.append(layer_sums[li][j] / c if c > 0 else float("nan"))
                per_mode_stats[mode].append(means)

        # Accuracy comparison (isolate TFrow: keep TFicl on SA for both)
        force_icl_identity(model)
        set_tfrow_mode(model, identity=True)   # TFrow-SA
        acc_SA, bacc_SA = evaluate_accuracy(clf, X_test, y_test)
        set_tfrow_mode(model, identity=False)  # TFrow-EA (L2+)
        acc_EA, bacc_EA = evaluate_accuracy(clf, X_test, y_test)

        # Decision rule: smaller growth past L1 on between-class similarity
        tol = 1e-6
        growth_SA = None
        growth_EA = None
        if per_mode_stats["SA"] and per_mode_stats["EA"]:
            bc_SA = [v[2] for v in per_mode_stats["SA"]]
            bc_EA = [v[2] for v in per_mode_stats["EA"]]
            if len(bc_SA) >= 2 and len(bc_EA) >= 2:
                growth_SA = bc_SA[-1] - bc_SA[0]
                growth_EA = bc_EA[-1] - bc_EA[0]
        no_worse_acc = (acc_EA >= acc_SA - tol) and (bacc_EA >= bacc_SA - tol)
        green = (growth_SA is not None and growth_EA is not None and (growth_EA < growth_SA)) and no_worse_acc

        # Report
        print(f"Dataset: {ds_name}")
        print(f"  Accuracy: SA={acc_SA:.4f} EA={acc_EA:.4f} | BAcc: SA={bacc_SA:.4f} EA={bacc_EA:.4f}")
        if per_mode_stats["SA"] and per_mode_stats["EA"]:
            print("  Layer-wise mean cosine (ALL / WITHIN / BETWEEN):")
            for li in range(layers_count or 0):
                sa = per_mode_stats["SA"][li]
                ea = per_mode_stats["EA"][li]
                print(
                    f"    L{li+1}: SA=({sa[0]:.4f}/{sa[1]:.4f}/{sa[2]:.4f}) "
                    f"EA=({ea[0]:.4f}/{ea[1]:.4f}/{ea[2]:.4f})"
                )
        print(
            f"  Decision: growth_SA={growth_SA if growth_SA is not None else 'NA'} "
            f"growth_EA={growth_EA if growth_EA is not None else 'NA'} | no_worse_acc={no_worse_acc} => GREEN={green}"
        )

        # Plot if requested
        if args.plot and per_mode_stats["SA"] and per_mode_stats["EA"]:
            try:
                import matplotlib.pyplot as plt

                layers = np.arange(1, (layers_count or 0) + 1)
                sa = np.array(per_mode_stats["SA"])  # (L,3)
                ea = np.array(per_mode_stats["EA"])  # (L,3)

                fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
                # Within-class
                ax[0].plot(layers, sa[:, 1], "-o", label="SA")
                ax[0].plot(layers, ea[:, 1], "-o", label="EA")
                ax[0].set_title("Within-class cosine")
                ax[0].set_xlabel("TFrow layer")
                ax[0].set_ylabel("mean cosine")
                ax[0].legend()
                # Between-class
                ax[1].plot(layers, sa[:, 2], "-o", label="SA")
                ax[1].plot(layers, ea[:, 2], "-o", label="EA")
                ax[1].set_title("Between-class cosine")
                ax[1].set_xlabel("TFrow layer")
                ax[1].legend()
                plt.tight_layout()
                os.makedirs(args.outdir, exist_ok=True)
                out_path = os.path.join(args.outdir, f"tfrow_layerwise_{ds_name.replace(' ', '_')}.png")
                plt.savefig(out_path, dpi=180)
                print(f"  Saved plot to {out_path}")
            except Exception as e:
                print(f"  [WARN] Plotting failed: {e}")


if __name__ == "__main__":
    main()
