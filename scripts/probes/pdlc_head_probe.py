#!/usr/bin/env python3
"""Lightweight probe for the integrated TabPDL-ICL head.

This script:
  1. Builds a TabICL model with icl_head='tabpdl' (optionally loading a checkpoint).
  2. Runs a single synthetic or loaded episode through the full backbone.
  3. Exposes:
       - pairwise probabilities gamma(q,i) from the PDLC head
       - per-query PDLC posteriors over classes
       - per-query, per-support contributions to each class score (before log)
  4. Saves them to a .npz file for inspection (e.g., heatmaps in a notebook).

Usage example (synthetic episode):
    python scripts/probes/pdlc_head_probe.py --out pdlc_probe.npz

You can then inspect `gamma` (shape [M,N]) and `posteriors` (shape [M,C])
from the saved file to verify behaviour and wiring, and `contrib` (shape [M,N,C])
to see how much each support row contributes to each class for every query.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu", help="Device for the probe model (cpu|cuda:0, ...)")
    p.add_argument("--checkpoint", type=str, default=None, help="TabICL training checkpoint (.ckpt)")
    p.add_argument("--out", type=str, default="pdlc_probe.npz", help="Output .npz file for diagnostics")
    p.add_argument("--num_support", type=int, default=8, help="Number of support rows in the synthetic episode")
    p.add_argument("--num_query", type=int, default=4, help="Number of query rows in the synthetic episode")
    p.add_argument("--num_classes", type=int, default=3, help="Number of classes in the synthetic episode")
    p.add_argument("--num_features", type=int, default=16, help="Number of features per row")
    p.add_argument(
        "--pdlc_feature_map",
        type=str,
        default="sym",
        choices=["sym", "concat"],
        help="Feature map for PDLC comparator: 'sym'=[|hq-hs|,hq*hs], 'concat'=[hq,hs].",
    )
    p.add_argument(
        "--pdlc_topk",
        type=int,
        default=None,
        help="If set and >0, enable top-k gating in the PDLC head; 0/None disables gating.",
    )
    p.add_argument(
        "--pdlc_agg",
        type=str,
        default="class_pool",
        choices=["posterior_avg", "class_pool", "sum"],
        help="Aggregation mode for the PDLC head (default: class_pool).",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a CSV file with a real dataset (optional; if set, overrides synthetic episode).",
    )
    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="Name of target column in CSV (default: last column).",
    )
    p.add_argument(
        "--openml_id",
        type=int,
        default=None,
        help="Optional OpenML dataset ID to use a real dataset instead of a synthetic one.",
    )
    p.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test fraction for real dataset splits.",
    )
    p.add_argument(
        "--max_queries",
        type=int,
        default=5,
        help="Maximum number of query rows to print for real dataset mode.",
    )
    p.add_argument(
        "--top_support",
        type=int,
        default=0,
        help="If >0, print the top-k support rows contributing most to the predicted class for each query.",
    )
    p.add_argument(
        "--n_estimators",
        type=int,
        default=1,
        help="Number of ensemble members when using TabICLClassifier in real dataset mode (default: 1).",
    )
    p.add_argument(
        "--normalize_contrib",
        action="store_true",
        help=(
            "If set, normalize contributions per query and class so that they sum to 1 "
            "over support rows (within the support mask)."
        ),
    )
    p.add_argument(
        "--print_stats",
        action="store_true",
        help=(
            "If set, print per-query pair_logits/gamma statistics and per-class gamma "
            "distributions for a subset of queries."
        ),
    )
    return p


def build_model(device: torch.device, args: argparse.Namespace):
    # Local import after sys.path tweak
    from tabicl.model.tabicl import TabICL  # type: ignore

    # Minimal, small model for probing; dimensions kept small for speed.
    model = TabICL(
        max_classes=args.num_classes,
        embed_dim=32,
        col_num_blocks=1,
        col_nhead=1,
        col_num_inds=16,
        row_elliptical=False,
        row_num_blocks=1,
        row_nhead=1,
        row_num_cls=2,
        row_rope_base=100000,
        icl_num_blocks=1,
        icl_nhead=1,
        icl_elliptical=False,
        ff_factor=2,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
        icl_head="tabpdl",
        pdlc_config={
            "topk": args.pdlc_topk,
            "mlp_width": 128,
            "mlp_depth": 2,
            "agg": args.pdlc_agg,
            "embed_norm": "none",
            "dropout": 0.0,
            "activation": "silu",
            "layernorm_after_first": True,
            "feature_map": args.pdlc_feature_map,
        },
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if "state_dict" in ckpt:
            missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
            print(f"Loaded checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys.")
        else:
            print("Checkpoint has no 'state_dict'; skipping weight load.")

    return model


def compute_support_contributions(
    pair_scores: torch.Tensor,
    y_support: torch.Tensor,
    support_mask: torch.Tensor,
    num_classes: int,
    agg: str,
) -> torch.Tensor:
    """Decompose per-pair scores into per-support, per-class contributions.

    Parameters
    ----------
    pair_scores : Tensor, shape (B, M, N)
        Raw pairwise scores for each (query, support) pair, e.g. pair_logits.
    y_support : Tensor, shape (B, N)
        Integer labels for support rows.
    support_mask : Tensor, shape (B, N)
        Boolean mask of valid supports.
    num_classes : int
        Number of classes C.
    agg : str
        Aggregation mode used by the head: 'posterior_avg', 'class_pool', or 'sum'.

    Returns
    -------
    Tensor, shape (B, M, N, C)
        Contribution of each support row i to each class c for each query q,
        derived directly from pair_scores. For all aggregation modes currently
        used in the probe, we assign:

            contrib[b, q, i, c] = pair_scores[b, q, i] if y_support[b, i] == c else 0

        respecting the support_mask. This makes contributions live in the same
        space as the PDLC logits, which is more informative than saturated
        probabilities when many supports are very similar.
    """
    B, M, N = pair_scores.shape
    C = num_classes
    contrib = pair_scores.new_zeros((B, M, N, C))

    for b in range(B):
        scores = pair_scores[b]  # (M, N)
        y = y_support[b].long()  # (N,)
        mask = support_mask[b]  # (N,)

        if not mask.any():
            continue

        # One-hot labels for supports and mask them
        oh = F.one_hot(y, num_classes=C).to(scores.dtype)  # (N, C)
        oh = oh.unsqueeze(0)  # (1, N, C)

        scores_exp = scores.unsqueeze(-1)  # (M, N, 1)
        mask_exp = mask.view(1, N, 1)  # (1, N, 1)

        # Assign each support's score to its own class, zero elsewhere
        per_anchor = scores_exp * oh * mask_exp  # (M, N, C)

        contrib[b] = per_anchor

    return contrib


def normalize_contrib_per_query_class(contrib: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    """Normalize contributions so they sum to 1 over supports for each (query, class).

    Parameters
    ----------
    contrib : np.ndarray, shape (M, N, C)
        Raw contributions for each query, support, and class.
    support_mask : np.ndarray, shape (N,)
        Boolean mask indicating valid supports.

    Returns
    -------
    np.ndarray, shape (M, N, C)
        Normalized contributions. For each query q and class c:
          sum_i contrib_norm[q, i, c] == 1 over valid supports (when the raw sum > 0).
        Invalid supports remain 0.
    """
    if contrib.ndim != 3:
        raise ValueError(f"Expected contrib with shape (M, N, C), got {contrib.shape}")
    if support_mask.ndim != 1 or support_mask.shape[0] != contrib.shape[1]:
        raise ValueError(
            f"support_mask shape {support_mask.shape} incompatible with contrib shape {contrib.shape}"
        )

    mask = support_mask[None, :, None]  # (1, N, 1)
    contrib_masked = np.where(mask, contrib, 0.0)
    sums = contrib_masked.sum(axis=1, keepdims=True)  # (M, 1, C)
    sums_safe = np.where(sums > 0.0, sums, 1.0)
    contrib_norm = contrib_masked / sums_safe
    contrib_norm = np.where(mask, contrib_norm, 0.0)
    return contrib_norm


def print_pair_gamma_stats(
    pair_scores: torch.Tensor,
    gamma: torch.Tensor,
    y_support: torch.Tensor,
    support_mask: torch.Tensor,
    logits_query: torch.Tensor,
    true_labels: np.ndarray | None,
    classes: np.ndarray | None,
    max_queries: int,
    header: str = "",
) -> None:
    """Print per-query statistics for pair_logits and gamma, plus per-class gamma stats."""
    pair_np = pair_scores[0].detach().cpu().numpy()  # (M, N)
    gamma_np = gamma[0].detach().cpu().numpy()  # (M, N)
    support_mask_np = support_mask[0].detach().cpu().numpy().astype(bool)  # (N,)
    logits_np = logits_query.detach().cpu().numpy()  # (M, C)
    y_support_np = y_support[0].detach().cpu().numpy()  # (N,)

    M, N = pair_np.shape
    C = logits_np.shape[1]
    max_q = min(M, max(1, max_queries))

    if header:
        print(header)

    for q in range(max_q):
        scores_q = pair_np[q, support_mask_np]
        gamma_q = gamma_np[q, support_mask_np]

        pred_idx = int(logits_np[q].argmax())
        pred_label = classes[pred_idx] if classes is not None else pred_idx
        true_label = true_labels[q] if true_labels is not None else None

        print(f"\nQuery {q}: pred={pred_label}" + (f", true={true_label}" if true_label is not None else ""))
        print(
            f"  pair_logits: min={scores_q.min():+.4f}, "
            f"mean={scores_q.mean():+.4f}, max={scores_q.max():+.4f}, std={scores_q.std():.4f}"
        )
        print(
            f"  gamma:       min={gamma_q.min():+.4f}, "
            f"mean={gamma_q.mean():+.4f}, max={gamma_q.max():+.4f}, std={gamma_q.std():.4f}"
        )

        # Per-class gamma distributions over supports (focus on predicted class)
        unique_classes = np.unique(y_support_np[support_mask_np])
        print("  per-class gamma over supports:")
        for c in unique_classes:
            mask_c = (y_support_np == c) & support_mask_np
            if not mask_c.any():
                continue
            g_c = gamma_np[q, mask_c]
            c_label = classes[c] if classes is not None else c
            tag = " (pred)" if c == pred_idx else ""
            print(
                f"    class={c_label}{tag}: n={mask_c.sum()}, "
                f"min={g_c.min():+.4f}, mean={g_c.mean():+.4f}, "
                f"max={g_c.max():+.4f}, std={g_c.std():.4f}"
            )


def run_synthetic_probe(args: argparse.Namespace, device: torch.device) -> None:
    """Run the PDLC contribution probe on a synthetic toy episode."""
    model = build_model(device, args)
    model.eval()

    if args.checkpoint is None:
        print("Warning: --checkpoint not set; using randomly initialised TabICL+TabPDL head.")

    B = 1
    N = args.num_support
    M = args.num_query
    T = N + M
    H = args.num_features
    C = args.num_classes

    # Synthetic episode: features and labels
    rng = np.random.default_rng(123)
    X_np = rng.normal(size=(B, T, H)).astype("float32")
    # Force all classes to appear in supports
    y_support = np.arange(N) % C
    rng.shuffle(y_support)
    y_query = rng.integers(0, C, size=M)
    y_full = np.concatenate([y_support, y_query], axis=0)

    X = torch.from_numpy(X_np).to(device)
    y_train = torch.from_numpy(y_support[None, :].astype("int64")).to(device)

    with torch.no_grad():
        _ = model(X, y_train)  # training mode wiring: produces logits over all positions

    enc = model.icl_predictor
    aux = getattr(enc, "last_pdlc_aux", None)
    if not aux:
        raise RuntimeError("No PDLC aux info found; ensure icl_head='tabpdl' and a forward pass was executed.")

    # Pairwise logits/scores and support mask (torch tensors)
    pair_scores_t = aux.get("pair_logits", None)
    if pair_scores_t is None:
        raise RuntimeError("PDLC aux did not contain 'pair_logits'; cannot compute logit-based contributions.")
    support_mask_t = aux["support_mask"]  # (B, N)

    # Determine aggregation mode from the head config
    head = getattr(enc, "pdlc_head", None)
    agg_mode = getattr(getattr(head, "cfg", None), "agg", "posterior_avg")

    # Compute per-support contributions to class scores (pre-log)
    contrib_t = compute_support_contributions(
        pair_scores=pair_scores_t,
        y_support=y_train,
        support_mask=support_mask_t,
        num_classes=C,
        agg=agg_mode,
    )  # (B, M, N, C)

    # Recompute PDLC posteriors for the same episode in eval mode to get final class probs
    with torch.no_grad():
        logits = model(X, y_train, return_logits=True)  # (B, M, C_eff)
    logits_q = logits[0]  # (M, C_eff)
    probs_q = torch.softmax(logits_q, dim=-1).cpu().numpy()

    gamma = aux["gamma"][0].cpu().numpy()  # (M, N)  # still saved for reference
    support_mask = support_mask_t[0].cpu().numpy().astype(bool)
    contrib = contrib_t[0].cpu().numpy()  # (M, N, C)

    if args.normalize_contrib:
        contrib_for_print = normalize_contrib_per_query_class(contrib, support_mask)
    else:
        contrib_for_print = contrib

    # Optional diagnostics: per-query stats for pair_logits and gamma
    if args.print_stats:
        print_pair_gamma_stats(
            pair_scores=pair_scores_t,
            gamma=aux["gamma"],
            y_support=y_train,
            support_mask=support_mask_t,
            logits_query=logits_q,
            true_labels=y_query,
            classes=np.arange(C),
            max_queries=args.max_queries,
            header="\n=== Synthetic episode: pair_logits / gamma stats ===",
        )

    # Optionally print top-k contributing supports for each query and its predicted class
    if args.top_support and args.top_support > 0:
        top_k = int(args.top_support)
        print(f"\nTop-{top_k} support rows per query for the predicted class:")
        for q in range(M):
            pred_class = int(probs_q[q].argmax())
            contrib_q = contrib_for_print[q, :, pred_class]  # (N,)
            # Mask out invalid supports
            contrib_q_masked = np.where(support_mask, contrib_q, -np.inf)
            k = min(top_k, int(support_mask.sum()))
            if k <= 0:
                continue
            top_idx = np.argpartition(-contrib_q_masked, k - 1)[:k]
            top_idx = top_idx[np.argsort(-contrib_q_masked[top_idx])]
            print(f"  Query {q}: predicted class {pred_class}")
            for rank, idx in enumerate(top_idx, start=1):
                print(
                    f"    #{rank}: support idx={idx}, y_support={int(y_support[idx])}, "
                    f"contrib={contrib_q[idx]:.6f}"
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "gamma": gamma,
        "support_mask": support_mask,
        "posteriors": probs_q,
        "y_support": y_support,
        "y_query": y_query,
        "contrib": contrib,
    }
    if args.normalize_contrib:
        save_dict["contrib_norm"] = contrib_for_print
    np.savez(out_path, **save_dict)
    print(f"Saved PDLC probe to {out_path.resolve()}")


def run_real_dataset_probe(args: argparse.Namespace, device: torch.device) -> None:
    """Run the PDLC contribution probe on a real dataset (CSV or OpenML)."""
    import sys

    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Local import after sys.path tweak
    from tabicl import TabICLClassifier  # type: ignore

    if args.checkpoint is None:
        raise SystemExit("--checkpoint must be provided when using a real dataset.")

    # Load dataset
    if args.openml_id is not None:
        try:
            import openml  # type: ignore
        except Exception as e:  # pragma: no cover - import error path
            raise SystemExit(f"openml package is required for --openml_id but could not be imported: {e}")

        dataset = openml.datasets.get_dataset(args.openml_id)
        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format="dataframe",
        )
        dataset_name = f"OpenML-{args.openml_id} ({dataset.name})"
    elif args.csv is not None:
        csv_path = Path(args.csv)
        if not csv_path.is_file():
            raise SystemExit(f"CSV dataset not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            raise SystemExit(f"CSV file {csv_path} must have at least 2 columns (features + target).")

        if args.target is None:
            target_col = df.columns[-1]
        else:
            target_col = args.target
            if target_col not in df.columns:
                raise SystemExit(f"Target column '{target_col}' not found in CSV.")

        y = df[target_col]
        X = df.drop(columns=[target_col])
        dataset_name = csv_path.name
    else:
        raise SystemExit("Real dataset mode requires either --csv or --openml_id.")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    # Build classifier with PDLC-head checkpoint
    clf = TabICLClassifier(
        n_estimators=max(1, args.n_estimators),
        model_path=args.checkpoint,
        allow_auto_download=False,
        device=device,
        use_hierarchical=True,
        pdlc_agg=args.pdlc_agg,
    )
    clf.fit(X_train, y_train)

    # Encode test labels to match head outputs (for saving)
    y_query_enc = clf.y_encoder_.transform(y_test)

    # Build a single TabICL episode from the first ensemble member
    X_test_enc = clf.X_encoder_.transform(X_test)
    data = clf.ensemble_generator_.transform(X_test_enc)
    if not data:
        raise RuntimeError("EnsembleGenerator returned no data for the provided dataset.")

    norm_method, (Xs, ys) = next(iter(data.items()))
    X_ensemble = Xs[0]  # (T, H')
    y_ensemble = ys[0]  # (train_size,)
    train_size = y_ensemble.shape[0]
    T = X_ensemble.shape[0]
    M = T - train_size
    if M <= 0:
        raise RuntimeError("No query rows found in constructed episode; check test_size > 0.")

    print(f"Using dataset: {dataset_name}")
    print(
        f"  norm_method={norm_method}, train_size={train_size}, "
        f"test_size={M}, num_features={X_ensemble.shape[1]}"
    )

    # Run underlying TabICL model in inference mode to get PDLC aux on this episode
    model = clf.model_
    model.eval()
    X_t = torch.from_numpy(X_ensemble[None, :, :].astype("float32")).to(clf.device_)
    y_train_t = torch.from_numpy(y_ensemble[None, :].astype("int64")).to(clf.device_)

    with torch.no_grad():
        # In inference mode TabICL returns logits for query positions only: shape (1, M, num_classes)
        logits_all = model(
            X_t,
            y_train_t,
            return_logits=True,
            inference_config=clf.inference_config_,
        )

    # Retrieve PDLC aux
    enc = model.icl_predictor
    aux = getattr(enc, "last_pdlc_aux", None)
    if not aux:
        raise RuntimeError(
            "No PDLC aux info found; ensure the checkpoint uses icl_head='tabpdl' "
            "and that the PDLC head is present."
        )

    pair_scores_t = aux.get("pair_logits", None)
    if pair_scores_t is None:
        raise RuntimeError("PDLC aux did not contain 'pair_logits'; cannot compute logit-based contributions.")
    support_mask_t = aux["support_mask"]  # (1, N)

    head = getattr(enc, "pdlc_head", None)
    agg_mode = getattr(getattr(head, "cfg", None), "agg", "posterior_avg")

    B, M_aux, N = pair_scores_t.shape
    if M_aux != M:
        raise RuntimeError("Mismatch between query size inferred from logits and PDLC aux gamma.")

    contrib_t = compute_support_contributions(
        pair_scores=pair_scores_t,
        y_support=y_train_t,
        support_mask=support_mask_t,
        num_classes=logits_all.shape[-1],
        agg=agg_mode,
    )  # (1, M, N, C)

    # Query logits/probs from inference forward (already only queries)
    logits_query = logits_all[0]  # (M, C)
    probs_q = torch.softmax(logits_query, dim=-1).cpu().numpy()

    gamma = aux["gamma"][0].cpu().numpy()  # (M, N)  # still saved for reference
    support_mask = support_mask_t[0].cpu().numpy().astype(bool)
    contrib = contrib_t[0].cpu().numpy()  # (M, N, C)

    if args.normalize_contrib:
        contrib_for_print = normalize_contrib_per_query_class(contrib, support_mask)
    else:
        contrib_for_print = contrib

    # Optional diagnostics: per-query stats for pair_logits and gamma
    if args.print_stats:
        print_pair_gamma_stats(
            pair_scores=pair_scores_t,
            gamma=aux["gamma"],
            y_support=y_train_t,
            support_mask=support_mask_t,
            logits_query=logits_query,
            true_labels=y_test.to_numpy(),
            classes=clf.classes_,
            max_queries=args.max_queries,
            header="\n=== Real dataset episode: pair_logits / gamma stats ===",
        )

    # Optionally print top-k supports per query (for the first few queries)
    if args.top_support and args.top_support > 0:
        top_k = int(args.top_support)
        max_q = min(M, max(1, args.max_queries))
        classes = clf.classes_
        print(f"\nTop-{top_k} support rows for first {max_q} queries (real dataset).")
        for q in range(max_q):
            pred_idx = int(probs_q[q].argmax())
            pred_label = classes[pred_idx]
            true_label = y_test.iloc[q]
            contrib_q = contrib_for_print[q, :, pred_idx]  # (N,)
            contrib_q_masked = np.where(support_mask, contrib_q, -np.inf)
            k = min(top_k, int(support_mask.sum()))
            if k <= 0:
                continue
            top_idx = np.argpartition(-contrib_q_masked, k - 1)[:k]
            top_idx = top_idx[np.argsort(-contrib_q_masked[top_idx])]
            print(
                f"\nQuery {q} (test idx {X_test.index[q]}): "
                f"true={true_label}, pred={pred_label}"
            )
            for rank, idx in enumerate(top_idx, start=1):
                support_y_idx = int(y_ensemble[idx])
                support_label = classes[support_y_idx]
                print(
                    f"  #{rank}: support idx={idx} (train idx {X_train.index[idx]}), "
                    f"label={support_label}, contrib={contrib_q[idx]:.6f}"
                )

    # Save arrays for further inspection (note: y_support/y_query are label-encoded)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "gamma": gamma,
        "support_mask": support_mask,
        "posteriors": probs_q,
        "y_support": y_ensemble,
        "y_query": y_query_enc,
        "contrib": contrib,
        "classes": clf.classes_,
    }
    if args.normalize_contrib:
        save_dict["contrib_norm"] = contrib_for_print
    np.savez(out_path, **save_dict)
    print(f"\nSaved PDLC probe (real dataset) to {out_path.resolve()}")


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    # Ensure local src is importable
    ROOT = Path(__file__).resolve().parents[2]
    SRC_DIR = ROOT / "src"
    import sys

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    device = torch.device(args.device)

    if args.csv is not None or args.openml_id is not None:
        run_real_dataset_probe(args, device)
    else:
        run_synthetic_probe(args, device)


if __name__ == "__main__":
    main()
