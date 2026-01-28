#!/usr/bin/env python3
"""Head-only viability check via pairwise ROC-AUC.

Freeze TabICL, use it only as a feature extractor, and:

  - Build a simple logistic comparator on features:
        phi(q,i) = [ E_q^T E_i , ||E_q - E_i||_2^2 ]
    where E are L2-normalized row embeddings.
  - Train the logistic head on a balanced set of same-class / different-class pairs.
  - Evaluate ROC-AUC on a held-out set of pairs.

Heuristic:
  - AUC >= 0.85–0.90  → head-only PDLC is likely viable.
  - AUC < 0.8         → you will probably need to unfreeze top blocks.

By default this script uses Iris as a minimal dataset and the pre-trained
TabICL checkpoint used in the sklearn wrapper (auto-downloaded via HF hub).

Example:
    python scripts/probes/pdlc_head_pair_auc.py --pairs_train 4096 --pairs_eval 4096
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def stratified_split(X, y, test_size: float, seed: int) -> Tuple:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Ensure each class appears in both splits
    tr_classes = set(np.unique(y_tr))
    te_classes = set(np.unique(y_te))
    if tr_classes != te_classes:
        for add in range(1, 1000):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=seed + add, stratify=y
            )
            tr_classes = set(np.unique(y_tr))
            te_classes = set(np.unique(y_te))
            if tr_classes == te_classes:
                break
    return X_tr, X_te, y_tr, y_te


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def build_balanced_pairs_indices(
    labels: np.ndarray,
    num_pairs: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices (i,j) and targets t for balanced same/different pairs.

    The output t is 1 for same-class pairs, 0 for different-class pairs.
    """
    n = labels.shape[0]
    classes = np.unique(labels)
    by_class = {c: np.where(labels == c)[0] for c in classes}
    all_idx = np.arange(n)

    pos_needed = num_pairs // 2
    neg_needed = num_pairs - pos_needed

    pos_pairs = []
    # Positives
    eligible = [c for c in classes if by_class[c].size >= 2]
    for _ in range(pos_needed):
        if not eligible:
            break
        c = eligible[rng.integers(0, len(eligible))]
        idx = by_class[c]
        if idx.size >= 2:
            i, j = rng.choice(idx, size=2, replace=False)
            pos_pairs.append((i, j))
    while len(pos_pairs) < pos_needed:
        cs = [c for c in classes if by_class[c].size >= 2]
        if not cs:
            break
        c = cs[rng.integers(0, len(cs))]
        i, j = rng.choice(by_class[c], size=2, replace=False)
        pos_pairs.append((i, j))

    # Negatives
    neg_pairs = []
    for _ in range(neg_needed):
        for _try in range(10):
            i, j = rng.choice(all_idx, size=2, replace=False)
            if labels[i] != labels[j]:
                neg_pairs.append((i, j))
                break
        else:
            if len(classes) >= 2:
                c1, c2 = rng.choice(classes, size=2, replace=False)
                i = rng.choice(by_class[c1])
                j = rng.choice(by_class[c2])
                neg_pairs.append((i, j))

    pairs = pos_pairs + neg_pairs
    t = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs), dtype=np.int64)
    idx_i = np.array([i for (i, _) in pairs], dtype=np.int64)
    idx_j = np.array([j for (_, j) in pairs], dtype=np.int64)
    return np.stack([idx_i, idx_j], axis=1), t


class LogisticComparator(nn.Module):
    """Simple logistic head on [dot, sq_dist] features."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)  # [dot, dist2] -> logit

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (N, 2)
        return self.linear(feats).squeeze(-1)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_train", type=int, default=4096, help="Number of train pairs")
    p.add_argument("--pairs_eval", type=int, default=4096, help="Number of eval pairs")
    p.add_argument("--epochs", type=int, default=5, help="Epochs over train pairs")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size for logistic head training")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for logistic head")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for logistic head")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the logistic head (TabICL runs mostly on CPU via sklearn wrapper).",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional TabICL checkpoint path to force local load (passed to TabICLClassifier).",
    )
    return p


def main() -> None:
    args = make_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Ensure local src is importable
    ROOT = Path(__file__).resolve().parents[2]
    SRC_DIR = ROOT / "src"
    import sys

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from tabicl import TabICLClassifier  # type: ignore
    from tabicl.pdlc.embed import extract_tf_row_embeddings  # type: ignore

    device = torch.device(args.device)

    # 1) Dataset: Iris as a small, clean classification benchmark
    iris = load_iris(as_frame=True)
    X_all, y_all = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = stratified_split(X_all, y_all, test_size=0.5, seed=args.seed)

    # 2) TabICLClassifier with frozen backbone
    clf = TabICLClassifier(
        n_estimators=4,
        batch_size=4,
        use_amp=True,
        verbose=False,
        model_path=args.model_path,
        allow_auto_download=True if args.model_path is None else False,
    )
    clf.fit(X_tr, y_tr)

    # 3) Extract row embeddings for the test side (anchors+queries in one episode)
    res = extract_tf_row_embeddings(clf, X_te)
    emb_all = res["embeddings_all"]  # (T, D)
    labels_all = np.asarray(res["train_labels"].tolist() + y_te.tolist())
    # Row order: [train embeddings, test embeddings] in res; we mirror labels accordingly.

    emb_all = l2_normalize_np(emb_all)

    # 4) Build train and eval pairs
    idx_train, t_train = build_balanced_pairs_indices(labels_all, args.pairs_train, rng)
    idx_eval, t_eval = build_balanced_pairs_indices(labels_all, args.pairs_eval, rng)

    def make_features(idx_pairs: np.ndarray) -> np.ndarray:
        i = idx_pairs[:, 0]
        j = idx_pairs[:, 1]
        a = emb_all[i]
        b = emb_all[j]
        dot = np.sum(a * b, axis=1, keepdims=True)
        dist2 = np.sum((a - b) ** 2, axis=1, keepdims=True)
        return np.concatenate([dot, dist2], axis=1).astype("float32")

    X_train = make_features(idx_train)
    X_eval = make_features(idx_eval)

    # 5) Train logistic head
    head = LogisticComparator().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(t_train.astype("float32")).to(device)

    N = X_train_t.shape[0]
    bs = args.batch_size
    num_batches = max(1, (N + bs - 1) // bs)

    head.train()
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(N, device=device)
        X_train_t = X_train_t[perm]
        y_train_t = y_train_t[perm]
        losses = []
        for b in range(num_batches):
            s = b * bs
            e = min(N, (b + 1) * bs)
            xb = X_train_t[s:e]
            yb = y_train_t[s:e]
            opt.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[epoch {ep:02d}] train BCE={mean_loss:.4f}")

    # 6) Evaluate ROC-AUC on held-out pairs
    head.eval()
    with torch.no_grad():
        X_eval_t = torch.from_numpy(X_eval).to(device)
        logits_eval = head(X_eval_t).cpu().numpy()
        probs_eval = 1.0 / (1.0 + np.exp(-logits_eval))
    auc = roc_auc_score(t_eval, probs_eval)
    print(f"Head-only pair ROC-AUC: {auc:.4f}")
    if auc >= 0.9:
        print("=> AUC >= 0.90: head-only PDLC looks very viable.")
    elif auc >= 0.85:
        print("=> AUC in [0.85,0.90): promising for head-only PDLC.")
    elif auc < 0.8:
        print("=> AUC < 0.80: likely need to unfreeze top blocks for best PDLC.")


if __name__ == "__main__":
    main()
