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


@dataclass
class TabPDLHeadConfig:
    """Configuration for the integrated TabPDL-ICL head.

    This is intentionally lightweight so it can be passed around or
    serialized in checkpoints without pulling in training-specific knobs.
    """

    topk: Optional[int] = None
    agg: str = "class_pool"  # or "posterior_avg"
    embed_norm: str = "none"  # "none", "l2", "layernorm"


class TabPDLHead(nn.Module):
    """Pairwise-decomposition head operating on ICL embeddings.

    Given:
        - H_support: (B, N, D)
        - H_query:   (B, M, D)
        - y_support: (B, N) with integer class ids
        - support_mask: (B, N) boolean, True for valid anchors

    it produces:
        - logits_query: (B, M, C) where C is the number of unique
          classes in y_support (padded by caller if needed)
        - aux: dict with diagnostics (pair logits/probs, optional indices)
    """

    def __init__(self, d_model: int, cfg: TabPDLHeadConfig, max_classes: int):
        super().__init__()
        self.d_model = d_model
        self.max_classes = max_classes
        self.cfg = cfg

        # Mandatory LayerNorm to operate on semantic direction rather than magnitude
        self.ln_emb = nn.LayerNorm(d_model)

        # Bilinear projections for query and support embeddings
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)

        # Calibration: global temperature tau > 0 and bias b
        # Initialize tau ~ 1.0 via inverse softplus
        init_tau = 1.0
        self._tau_param = nn.Parameter(torch.log(torch.expm1(torch.tensor(init_tau, dtype=torch.float32))))
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    @property
    def tau(self) -> torch.Tensor:
        # Ensure strictly positive temperature
        return torch.nn.functional.softplus(self._tau_param) + 1e-6

    def _normalize_embeddings(self, H: torch.Tensor) -> torch.Tensor:
        # Always apply LayerNorm as the primary normalization step
        H = self.ln_emb(H)
        # Optional additional L2 normalization on top of LayerNorm
        if self.cfg.embed_norm == "l2":
            H = torch.nn.functional.normalize(H, p=2.0, dim=-1, eps=1e-12)
        return H

    def _pair_logits(
        self,
        H_query: torch.Tensor,
        H_support: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise logits ell(q, i) for all query-support pairs.

        Shapes
        ------
        H_query : (B, M, D)
        H_support : (B, N, D)
        support_mask : (B, N) boolean

        Returns
        -------
        Tensor
            Pairwise logits of shape (B, M, N)
        """

        B, M, D = H_query.shape
        _, N, _ = H_support.shape

        # Normalize embeddings first (LayerNorm + optional L2)
        H_query = self._normalize_embeddings(H_query)   # (B, M, D)
        H_support = self._normalize_embeddings(H_support)  # (B, N, D)

        # Project queries and supports
        Q = self.W_Q(H_query)   # (B, M, D)
        K = self.W_K(H_support)  # (B, N, D)

        # Bilinear similarity via batched matrix multiplication
        # s_{ij} = tau * ( (h_q W_Q) · (h_s W_K)^T ) + b
        logits = torch.matmul(Q, K.transpose(-1, -2))  # (B, M, N)
        tau = self.tau
        logits = tau * logits + self.bias

        # Optional top-k gating: keep only top-k supports per query (others get large negative logits)
        if self.cfg.topk is not None and self.cfg.topk > 0 and self.cfg.topk < N:
            B, M, N = logits.shape
            k = self.cfg.topk
            # Respect support mask when selecting top-k
            mask = support_mask.unsqueeze(1)  # (B, 1, N)
            logits_masked = logits.masked_fill(~mask, float("-inf"))
            _, topk_idx = torch.topk(logits_masked, k=min(k, N), dim=-1)
            keep = torch.zeros_like(logits, dtype=torch.bool)
            keep.scatter_(2, topk_idx, True)
            # Positions not in top-k get strongly negative logits
            logits = logits.masked_fill(~keep, float("-inf"))
        else:
            # Apply support mask directly
            mask = support_mask.unsqueeze(1)  # (B, 1, N)
            logits = logits.masked_fill(~mask, float("-inf"))

        return logits

    def _aggregate_posterior_avg(
        self,
        gamma: torch.Tensor,
        y_support: torch.Tensor,
        support_mask: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Posterior-average aggregation (primary PDLC-style aggregator).

        For each anchor i with label c:
            p_c^{(i)}   = gamma(q, i)
            p_{c'}^{(i)} = (1 - gamma(q, i)) / (C-1) for c' != c

        Then average over anchors (respecting masks) and renormalize.
        """
        B, M, N = gamma.shape
        C = num_classes
        eps = 1e-12
        out = gamma.new_zeros((B, M, C))

        for b in range(B):
            g = gamma[b]  # (M, N)
            y = y_support[b].long()  # (N,)
            mask = support_mask[b]  # (N,)

            if not mask.any():
                continue

            g = g[:, mask]  # (M, N_eff)
            y = y[mask]  # (N_eff,)
            N_eff = g.shape[1]

            # Empirical episode prior over classes from supports
            prior = torch.bincount(y, minlength=C).to(g.dtype)  # (C,)
            prior = prior / torch.clamp(prior.sum(), min=eps)   # (C,)

            # Positive vote for own class: sum_i gamma(q,i) * 1_{y_i}
            oh = torch.nn.functional.one_hot(y, num_classes=C).to(g.dtype)  # (N_eff, C)
            oh_exp = oh.unsqueeze(0).expand(M, -1, -1)  # (M, N_eff, C)
            g_exp = g.unsqueeze(-1)  # (M, N_eff, 1)
            pos_contrib = (g_exp * oh_exp).sum(dim=1)  # (M, C)

            # Negative vote: distribute (1 - gamma(q,i)) to other classes ∝ prior
            one_minus_g = 1.0 - g  # (M, N_eff)
            denom = 1.0 - prior[y]  # (N_eff,)
            denom = torch.clamp(denom, min=eps)
            scale = one_minus_g / denom.unsqueeze(0)  # (M, N_eff)
            scale_exp = scale.unsqueeze(-1)  # (M, N_eff, 1)
            prior_exp = prior.view(1, 1, C)  # (1, 1, C)
            neg_contrib = scale_exp * prior_exp  # (M, N_eff, C)
            # Zero out own-class component in the negative vote
            mask_other = (1.0 - oh).unsqueeze(0)  # (1, N_eff, C)
            neg_contrib = neg_contrib * mask_other  # (M, N_eff, C)
            neg_contrib = neg_contrib.sum(dim=1)  # (M, C)

            # Average over anchors and renormalize
            accum = (pos_contrib + neg_contrib) / float(N_eff)  # (M, C)
            s = accum.sum(dim=-1, keepdim=True)  # (M, 1)
            out[b] = accum / torch.clamp(s, min=eps)

        return out

    def _aggregate_class_pool(
        self,
        gamma: torch.Tensor,
        y_support: torch.Tensor,
        support_mask: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Simpler class-pooling aggregation for ablations.

        p(y=c | q) ∝ ∑_{i: y_i=c} gamma(q, i)
        """
        B, M, N = gamma.shape
        C = num_classes
        eps = 1e-12
        out = gamma.new_zeros((B, M, C))

        for b in range(B):
            g = gamma[b]  # (M, N)
            y = y_support[b].long()  # (N,)
            mask = support_mask[b]  # (N,)

            if not mask.any():
                continue

            g = g[:, mask]  # (M, N_eff)
            y = y[mask]  # (N_eff,)

            # Scatter-add over classes
            idx = y.unsqueeze(0).expand(g.shape[0], -1)  # (M, N_eff)
            P = torch.zeros((g.shape[0], C), device=g.device, dtype=g.dtype)
            P.scatter_add_(1, idx, g)

            s = P.sum(dim=-1, keepdim=True)
            out[b] = P / torch.clamp(s, min=eps)

        return out

    def forward(
        self,
        H_support: torch.Tensor,
        H_query: torch.Tensor,
        y_support: torch.Tensor,
        support_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute query logits and auxiliary diagnostics.

        Parameters
        ----------
        H_support : Tensor
            Support embeddings of shape (B, N, D)

        H_query : Tensor
            Query embeddings of shape (B, M, D)

        y_support : Tensor
            Support labels of shape (B, N)

        support_mask : Optional[Tensor], default=None
            Boolean mask of shape (B, N) where True marks valid supports.

        Returns
        -------
        logits_query : Tensor
            Log-probabilities over classes for each query, shape (B, M, C)

        aux : dict
            Auxiliary diagnostics (pair logits, gammas, etc.)
        """
        if support_mask is None:
            support_mask = torch.ones_like(y_support, dtype=torch.bool)

        B, N, D = H_support.shape
        _, M, _ = H_query.shape
        assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"

        # Pairwise logits via bilinear similarity and probabilities via sigmoid
        pair_logits = self._pair_logits(H_query, H_support, support_mask)  # (B, M, N)
        gamma = torch.sigmoid(pair_logits)  # (B, M, N)

        # Determine number of classes present
        # Assumes y_support are already encoded as contiguous ints per episode
        num_classes = int(y_support.max().item()) + 1

        if self.cfg.agg == "posterior_avg":
            P = self._aggregate_posterior_avg(gamma, y_support, support_mask, num_classes)
        elif self.cfg.agg == "class_pool":
            P = self._aggregate_class_pool(gamma, y_support, support_mask, num_classes)
        else:
            raise ValueError(f"Unknown aggregation mode '{self.cfg.agg}' for TabPDLHead.")

        eps = 1e-12
        P = torch.clamp(P, min=eps)
        logP = torch.log(P)

        aux = {
            "pair_logits": pair_logits,
            "gamma": gamma,
            "support_mask": support_mask,
        }
        return logP, aux
