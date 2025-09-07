from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PDLCHead(nn.Module):
    """Tiny symmetric MLP to score if two rows share the same class.

    Forward takes concatenated embeddings of shape (N, 2*D) and returns logits (N, 1).
    """

    def __init__(self, emb_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        return self.mlp(pairs).squeeze(-1)  # (N,)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_grad_norm: float = 1.0
    symmetry_weight: float = 0.1


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def build_balanced_pairs(
    emb: np.ndarray,
    labels: np.ndarray,
    num_pairs: int,
    rng: np.random.Generator,
    *,
    class_balance: bool = True,
    hard_neg_frac: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create balanced positive/negative pairs from anchor set.

    Returns (pair_ab, pair_ba, targets), where pairs are float32 arrays of shape (N, 2D)
    and targets are int64 of shape (N,), with N = 2 * num_pairs (both orders).
    """
    n = emb.shape[0]
    classes = np.unique(labels)
    D = emb.shape[1]

    # index by class for fast sampling
    by_class = {c: np.where(labels == c)[0] for c in classes}
    all_idx = np.arange(n)

    pos_needed = num_pairs // 2
    neg_needed = num_pairs - pos_needed

    # Positives: uniform over classes when requested; with replacement to ensure balance
    pos_pairs = []
    if class_balance:
        eligible = [c for c in classes if by_class[c].size >= 2]
        for _ in range(pos_needed):
            if not eligible:
                break
            c = eligible[rng.integers(0, len(eligible))]
            idx = by_class[c]
            # with replacement across samples, without replacement within pair
            if idx.size >= 2:
                i, j = rng.choice(idx, size=2, replace=False)
                pos_pairs.append((i, j))
    else:
        for _ in range(pos_needed):
            c = classes[rng.integers(0, len(classes))]
            idx = by_class[c]
            if idx.size >= 2:
                i, j = rng.choice(idx, size=2, replace=False)
                pos_pairs.append((i, j))
    # pad if we couldn't get enough due to tiny classes
    while len(pos_pairs) < pos_needed:
        cs = [c for c in classes if by_class[c].size >= 2]
        if not cs:
            break
        c = cs[rng.integers(0, len(cs))]
        i, j = rng.choice(by_class[c], size=2, replace=False)
        pos_pairs.append((i, j))

    # Negatives: mix of random and hard negatives
    neg_pairs = []
    n_hard = int(round(hard_neg_frac * neg_needed))
    n_rand = max(0, neg_needed - n_hard)

    # Random negatives
    for _ in range(n_rand):
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

    # Hard negatives by nearest different-class in cosine similarity
    if n_hard > 0 and emb.shape[0] >= 2:
        # emb must be L2-normalized for cosine
        S = emb @ emb.T
        # for each i, find best j with different label
        cand = []
        for i in range(emb.shape[0]):
            # mask out same-class and self
            mask = labels != labels[i]
            if mask.any():
                sims = S[i][mask]
                js = np.where(mask)[0]
                j = int(js[np.argmax(sims)])
                cand.append((i, j))
        # sample from candidates
        if cand:
            for _ in range(min(n_hard, len(cand))):
                i, j = cand[rng.integers(0, len(cand))]
                neg_pairs.append((i, j))

    pairs = pos_pairs + neg_pairs
    targets = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs), dtype=np.int64)

    # both orders
    a = emb[[i for i, _ in pairs]]
    b = emb[[j for _, j in pairs]]
    ab = np.concatenate([a, b], axis=1).astype(np.float32)
    ba = np.concatenate([b, a], axis=1).astype(np.float32)

    # duplicate targets for both orders
    t = np.concatenate([targets, targets], axis=0)
    return ab, ba, t


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None


@torch.no_grad()
def pair_auc_head(
    head: PDLCHead,
    emb: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    device: torch.device,
    num_eval_pairs: int = 4096,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute pair AUC for the head and cosine baseline on sampled pairs."""
    # Build eval pairs (balanced, no training signal used here explicitly)
    ab, ba, t = build_balanced_pairs(emb, labels, num_eval_pairs, rng, class_balance=True, hard_neg_frac=0.0)
    N = t.shape[0] // 2
    # head scores (symmetrized)
    head.eval()
    bs = 1024
    scores = []
    for s in range(0, N, bs):
        e = min(N, s + bs)
        logit_ab = head(torch.from_numpy(ab[s:e]).to(device)).float()
        logit_ba = head(torch.from_numpy(ba[s:e]).to(device)).float()
        p = torch.sigmoid(logit_ab).add(torch.sigmoid(logit_ba)).mul_(0.5)
        scores.append(p.cpu().numpy())
    scores = np.concatenate(scores, axis=0)
    auc_head = _roc_auc(t[:N], scores)

    # cosine baseline on L2-normalized embeddings
    d = emb.shape[1]
    a_idx = [i for i, _ in zip(range(N), range(N))]  # dummy for construction below
    # recover actual indices used
    # ab is [a; b], both blocks are D-wide
    a = ab[:N, :d]
    b = ab[:N, d:]
    cos_scores = np.sum(a * b, axis=1)
    auc_cos = _roc_auc(t[:N], cos_scores)
    return auc_head, auc_cos


@torch.no_grad()
def pair_auc_on_pairs(
    head: PDLCHead,
    pair_ab: np.ndarray,
    pair_ba: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute pair AUCs on a fixed set of pairs for before/after comparisons.

    targets is expected to be length 2*N (duplicated for both orders); we use the first N.
    """
    head.eval()
    N = targets.shape[0] // 2
    bs = 1024
    scores = []
    for s in range(0, N, bs):
        e = min(N, s + bs)
        logit_ab = head(torch.from_numpy(pair_ab[s:e]).to(device)).float()
        logit_ba = head(torch.from_numpy(pair_ba[s:e]).to(device)).float()
        p = torch.sigmoid(logit_ab).add(torch.sigmoid(logit_ba)).mul_(0.5)
        scores.append(p.cpu().numpy())
    scores = np.concatenate(scores, axis=0)
    auc_head = _roc_auc(targets[:N], scores)

    # cosine baseline
    d = pair_ab.shape[1] // 2
    a = pair_ab[:N, :d]
    b = pair_ab[:N, d:]
    cos_scores = np.sum(a * b, axis=1)
    auc_cos = _roc_auc(targets[:N], cos_scores)
    return auc_head, auc_cos


def train_head_on_pairs(
    head: PDLCHead,
    pair_ab: np.ndarray,
    pair_ba: np.ndarray,
    targets: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epochs: int = 1,
    mimic_weight: float = 0.0,
) -> float:
    """Train on provided pairs using a persistent optimizer. Returns avg loss over all epochs.

    - optimizer: if None, a fresh AdamW is created; otherwise it's reused.
    - epochs: repeat over the same provided pairs this many times (useful when resampling pairs per call).
    - mimic_weight: if >0, add BCEWithLogits loss towards cosine similarity s in [0,1].
    """
    head.train()
    opt = optimizer or torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    N = targets.shape[0] // 2  # unique pairs count

    # Pos weight balancing
    n_pos = targets.sum()
    n_neg = targets.shape[0] - n_pos
    pos_weight = torch.tensor([max(1.0, n_neg / max(1.0, n_pos))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_mimic = nn.BCEWithLogitsLoss()

    # Precompute cosine targets if requested
    d2 = pair_ab.shape[1] // 2
    if mimic_weight > 0:
        a = pair_ab[:N, :d2]
        b = pair_ab[:N, d2:]
        # embeddings are expected L2-normalized; clip numerical issues
        cos = np.clip((a * b).sum(axis=1), -1.0, 1.0)
        s = (cos + 1.0) * 0.5  # in [0,1]
        s_ab = torch.from_numpy(s).float().to(device)
        s_ba = s_ab  # same pairs reversed

    # mini-batches
    bs = cfg.batch_size
    n_batches = math.ceil(N / bs)
    losses = []
    for _ in range(max(1, epochs)):
        for bidx in range(n_batches):
            s_i = bidx * bs
            e_i = min((bidx + 1) * bs, N)
            ab = torch.from_numpy(pair_ab[s_i:e_i]).to(device)
            ba = torch.from_numpy(pair_ba[s_i:e_i]).to(device)
            t = torch.from_numpy(targets[s_i:e_i]).float().to(device)

            opt.zero_grad(set_to_none=True)
            logit_ab = head(ab)
            logit_ba = head(ba)
            loss_main = criterion(logit_ab, t) + criterion(logit_ba, t)
            loss_sym = F.mse_loss(logit_ab, logit_ba)
            loss = loss_main + cfg.symmetry_weight * loss_sym

            if mimic_weight > 0:
                loss_mimic = criterion_mimic(logit_ab, s_ab[s_i:e_i]) + criterion_mimic(logit_ba, s_ba[s_i:e_i])
                loss = loss + float(mimic_weight) * loss_mimic

            loss.backward()
            if cfg.max_grad_norm:
                nn.utils.clip_grad_norm_(head.parameters(), cfg.max_grad_norm)
            opt.step()
            losses.append(loss.detach().item())

    return float(np.mean(losses))


@torch.no_grad()
def pair_gamma(
    head: PDLCHead,
    q_emb: np.ndarray,
    a_emb: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Compute symmetric pair probabilities γ between queries and anchors.

    Returns array of shape (n_q, n_a) with values in [0, 1].
    """
    head.eval()
    n_q, d = q_emb.shape
    n_a = a_emb.shape[0]
    out = np.empty((n_q, n_a), dtype=np.float32)

    # Process queries in mini-batches to limit memory
    for i in range(0, n_q, 1):
        q = q_emb[i : i + 1]  # (1, D)
        # tile against all anchors
        ab = np.concatenate([np.repeat(q, n_a, axis=0), a_emb], axis=1).astype(np.float32)
        ba = np.concatenate([a_emb, np.repeat(q, n_a, axis=0)], axis=1).astype(np.float32)

        # evaluate in chunks along anchors
        gam = []
        for s in range(0, n_a, batch_size):
            e = min(n_a, s + batch_size)
            ab_t = torch.from_numpy(ab[s:e]).to(device)
            ba_t = torch.from_numpy(ba[s:e]).to(device)
            logit_ab = head(ab_t)
            logit_ba = head(ba_t)
            p = torch.sigmoid(logit_ab).add(torch.sigmoid(logit_ba)).mul_(0.5).float().cpu().numpy()
            gam.append(p)
        out[i] = np.concatenate(gam, axis=0)
    return out


def cosine_topk(q_emb: np.ndarray, a_emb: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k anchors per query by cosine similarity."""
    # both must be L2-normalized
    sims = q_emb @ a_emb.T  # (n_q, n_a)
    k = min(k, a_emb.shape[0])
    return np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]


def pdlc_posteriors(
    head: PDLCHead,
    a_emb: np.ndarray,
    a_labels: np.ndarray,
    q_emb: np.ndarray,
    topk: Optional[int],
    device: torch.device,
    gamma_batch: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PDLC posteriors for queries.

    Returns (posteriors, class_order) where posteriors has shape (n_q, C) and rows sum to 1.
    """
    # Encode labels to 0..C-1 consistently, preserve class order for mapping back
    classes, y_enc = np.unique(a_labels, return_inverse=True)
    C = classes.shape[0]
    n_a = a_emb.shape[0]
    prior = np.bincount(y_enc, minlength=C).astype(np.float32)
    prior = prior / prior.sum()

    # Optional top-K preselection
    if topk is not None and topk > 0 and topk < n_a:
        idx_k = cosine_topk(q_emb, a_emb, k=topk)
    else:
        idx_k = np.tile(np.arange(n_a), (q_emb.shape[0], 1))

    # For each query, compute γ to its selected anchors
    # We'll batch queries by 1 for simplicity; head eval is cheap
    posts = np.zeros((q_emb.shape[0], C), dtype=np.float32)
    eps = 1e-12
    for qi in range(q_emb.shape[0]):
        sel = idx_k[qi]
        a_sel = a_emb[sel]
        y_sel = y_enc[sel]
        gam = pair_gamma(head, q_emb[qi : qi + 1], a_sel, device=device, batch_size=gamma_batch)[0]
        # contributions per class
        accum = np.zeros(C, dtype=np.float32)
        for i, yi in enumerate(y_sel):
            # positive vote for its own class
            accum[yi] += gam[i]
            # negative vote distributed to other classes proportionally to prior
            denom = max(eps, 1.0 - prior[yi])
            contrib = (1.0 - gam[i]) * (prior / denom)
            contrib[yi] = 0.0
            accum += contrib
        # average over anchors and renormalize
        accum /= max(1, len(sel))
        s = accum.sum()
        posts[qi] = accum / max(s, eps)

    return posts, classes


def nll_accuracy(posteriors: np.ndarray, true_labels: np.ndarray, classes: np.ndarray) -> Tuple[float, float]:
    """Compute mean NLL and accuracy for queries whose class exists in anchors."""
    # Map true labels to class indices; mark OOV as -1
    class_to_idx = {c: i for i, c in enumerate(classes.tolist())}
    idx = np.array([class_to_idx.get(t, -1) for t in true_labels])
    mask = idx >= 0
    idx = idx[mask]
    P = posteriors[mask]
    eps = 1e-12
    nll = -np.log(np.take_along_axis(P, idx[:, None], axis=1) + eps).mean() if P.size else float("nan")
    acc = (P.argmax(axis=1) == idx).mean() if P.size else float("nan")
    return float(nll), float(acc)
