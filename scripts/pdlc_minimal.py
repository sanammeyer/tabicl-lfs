"""Minimal PDLC head training on top of TabICL.

Implements the no-code plan:
- Load frozen TabICL backbone through TabICLClassifier (encoders + ensemble only)
- Extract pre-ICL TF-row embeddings for (anchors ∪ queries)
- Train a tiny symmetric MLP on balanced anchor pairs
- Validate with the PDLC update (optional top-K)

Example:
    python scripts/pdlc_minimal.py --episodes 5 --pairs 4096 --topk 256
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tabicl import TabICLClassifier
from tabicl.pdlc.embed import extract_tf_row_embeddings
from tabicl.pdlc.head import (
    PDLCHead,
    TrainConfig,
    build_balanced_pairs,
    l2_normalize,
    pdlc_posteriors,
    nll_accuracy,
)


def stratified_split(X, y, test_size: float, seed: int) -> Tuple:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Ensure each class appears in both splits
    tr_classes = set(np.unique(y_tr))
    te_classes = set(np.unique(y_te))
    if tr_classes != te_classes:
        # If some class missing in test, resample with a new seed offset
        for add in range(1, 1000):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=seed + add, stratify=y
            )
            tr_classes = set(np.unique(y_tr))
            te_classes = set(np.unique(y_te))
            if tr_classes == te_classes:
                break
    return X_tr, X_te, y_tr, y_te


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--pairs", type=int, default=4096, help="pairs per episode (unique, doubled by symmetry)")
    p.add_argument("--topk", type=int, default=256, help="top-K anchors per query for PDLC (0 disables)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--symmetry_weight", type=float, default=0.1)
    p.add_argument("--anchor_frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="models/pdlc_head.pt")
    p.add_argument("--model_path", type=str, default=None, help="Optional TabICL checkpoint to force local load")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1) Data: use Iris as a minimal working example
    iris = load_iris(as_frame=True)
    X_all, y_all = iris.data, iris.target

    # 2) Head (initialized after first episode when we know embedding dim)
    head = None

    # 3) Training episodes
    best_nll = float("inf")
    best_state = None

    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.episodes + 1):
        # Episode split
        X_tr, X_te, y_tr, y_te = stratified_split(X_all, y_all, test_size=1 - args.anchor_frac, seed=args.seed + ep)

        # Freeze/prep TabICL sklearn wrapper (loads checkpoint and fits transforms)
        clf = TabICLClassifier(
            n_estimators=4,
            batch_size=4,
            use_amp=True,
            verbose=False,
            model_path=args.model_path,
            allow_auto_download=True if args.model_path is None else False,
        )
        clf.fit(X_tr, y_tr)

        # Extract TF-row embeddings (pre-ICL), train first then test
        res = extract_tf_row_embeddings(clf, X_te)
        emb_tr = l2_normalize(res["embeddings_train"])  # anchors
        emb_te = l2_normalize(res["embeddings_test"])   # queries
        y_anchor = np.asarray(res["train_labels"])      # original labels (decoded)
        y_query = np.asarray(y_te)

        # Init PDLC head lazily
        if head is None:
            head = PDLCHead(emb_dim=emb_tr.shape[1]).to(device)

        # Build balanced pairs from anchors and train head briefly
        ab, ba, t = build_balanced_pairs(emb_tr, y_anchor, num_pairs=args.pairs, rng=rng)
        cfg = TrainConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            symmetry_weight=args.symmetry_weight,
        )
        # Train for a single pass over the generated pairs
        from tabicl.pdlc.head import train_head_on_pairs as _train
        loss = _train(head, ab, ba, t, cfg, device)

        # PDLC validation on this episode
        topk = None if args.topk <= 0 else args.topk
        posts, classes = pdlc_posteriors(head, emb_tr, y_anchor, emb_te, topk=topk, device=device)
        nll, acc = nll_accuracy(posts, y_query, classes)
        print(f"[Episode {ep:02d}] pair-loss={loss:.4f}  PDLC: NLL={nll:.4f}  acc={acc:.3f}")

        # Track best by NLL
        if np.isfinite(nll) and nll < best_nll:
            best_nll = nll
            best_state = {"model": head.state_dict(), "emb_dim": head.emb_dim, "classes": classes}
            torch.save(best_state, out_path)

    if best_state is not None:
        print(f"Saved best PDLC head to {out_path.resolve()} (best NLL={best_nll:.4f})")
    else:
        print("Training finished but no finite NLL was recorded; nothing saved.")


if __name__ == "__main__":
    main()
