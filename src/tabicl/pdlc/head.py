from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TabPDLHeadConfig:
    """Configuration for the TabPDLHead.

    Attributes
    ----------
    topk:
        If not None, keep only the top-k anchors per query.
        - For `agg in {"class_pool","sum"}`: logits for non-topk anchors are set to -inf
          before the sigmoid (so they contribute gamma=0).
        - For `agg == "posterior_avg"`: non-topk anchors are excluded from BOTH the
          positive and negative terms of the posterior average (so they contribute 0
          mass, rather than contributing a large "negative" mass via (1-gamma)).
        Acts as a sparsifier / gating mechanism.
    agg:
        Aggregation mode. One of:
        - "class_pool": mean gamma per class (density-agnostic voting)
        - "posterior_avg": normalize class scores per query (posterior view)
        - "sum": sum gamma per class (evidence mass / kernel density)
    embed_norm:
        Embedding normalization mode. Currently:
        - "none" / "layernorm": apply LayerNorm only
        - "l2": LayerNorm followed by L2-normalization
    Inference_temperature:
        Additional temperature scaling applied at inference time
        (training is unaffected). Ignored when agg == "sum", where an
        automatic sqrt(N) scaling is used instead.
    symmetrize:
        Deprecated. Kept for config compatibility but ignored.
    """

    topk: Optional[int] = None
    agg: str = "class_pool"
    embed_norm: str = "none"
    Inference_temperature: float = 1.0
    symmetrize: bool = False


class TabPDLHead(nn.Module):
    """Pairwise Distance Learning (PDL) head for TabICL.

    This head operates on the ICL embeddings and produces class-level
    logits by:
      1. Projecting query/support embeddings into a PDL metric space
         via W_Q and W_K.
      2. Computing pairwise logits ell(q, a) = tau * <Q_q, K_a> + bias.
      3. Converting logits to pairwise "same-class" probabilities gamma.
      4. Aggregating gamma over supports grouped by class.
    """

    # Heuristic threshold for when to switch to chunked inference
    _CHUNK_FLOP_THRESHOLD = 1e8
    _CHUNK_SIZE = 1024

    def __init__(self, d_model: int, max_classes: int, cfg: TabPDLHeadConfig) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_classes = max_classes
        self.cfg = cfg

        # Embedding normalization
        self.ln_emb = nn.LayerNorm(d_model)

        # Projections into the PDL space
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)

        # Learnable temperature (tau > 0 via softplus) and bias
        self._tau_param = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------
    @property
    def tau(self) -> torch.Tensor:
        # Softplus to guarantee positivity; add small epsilon to avoid 0
        return F.softplus(self._tau_param) + 1e-6

    def _normalize_embeddings(self, H: torch.Tensor) -> torch.Tensor:
        """Apply embedding normalization according to config.

        We always apply LayerNorm for stability, and optionally L2-normalize.
        """
        Hn = self.ln_emb(H)
        if self.cfg.embed_norm == "l2":
            Hn = F.normalize(Hn, p=2.0, dim=-1)
        # For "none" / "layernorm" we just return the LayerNorm output.
        return Hn

    # ------------------------------------------------------------------
    # Pairwise logits
    # ------------------------------------------------------------------
    def _apply_support_mask_and_topk(
        self,
        logits: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply support mask and optional top-k gating to logits."""
        # Mask invalid anchors along the support dimension (N)
        # logits: (B, M, N), support_mask: (B, N)
        if support_mask is not None:
            if support_mask.dtype is not torch.bool:
                support_mask = support_mask.bool()
            mask = support_mask.unsqueeze(1)  # (B, 1, N)
            logits = logits.masked_fill(~mask, float("-inf"))

        # Optional top-k gating per query.
        # NOTE: For posterior_avg we apply top-k inside the aggregator instead,
        # because setting logits=-inf implies gamma=0 and thus (1-gamma)=1, which
        # would incorrectly add a strong "negative" contribution from dropped anchors.
        if self.cfg.topk is not None and self.cfg.agg != "posterior_avg":
            k = int(self.cfg.topk)
            if k > 0 and k < logits.size(-1):
                # Find per-query threshold
                topk_vals, _ = logits.topk(k, dim=-1)
                thresh = topk_vals[..., -1].unsqueeze(-1)  # (B, M, 1)
                logits = logits.masked_fill(logits < thresh, float("-inf"))

        return logits

    def _pair_logits(
        self,
        H_query: torch.Tensor,
        H_support: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise logits ell(q, a) for all query-support pairs."""
        # Normalize
        Hq = self._normalize_embeddings(H_query)   # (B, M, D)
        Hs = self._normalize_embeddings(H_support)  # (B, N, D)

        # Project
        Q = self.W_Q(Hq)  # (B, M, D)
        K = self.W_K(Hs)  # (B, N, D)

        # Bilinear score
        logits = torch.matmul(Q, K.transpose(-1, -2))  # (B, M, N)
        logits = self.tau * logits + self.bias  # broadcast over (B, M, N)

        # Mask + optional top-k
        logits = self._apply_support_mask_and_topk(logits, support_mask)
        return logits

    # ------------------------------------------------------------------
    # Aggregators
    # ------------------------------------------------------------------
    def _aggregate_posterior_avg(self, gamma, y_support, support_mask, num_classes):
        """
        Stable PDLC posterior averaging:
        p(c|q) = (1/N) sum_i [ 1[y_i=c]*gamma(q,i) + (1-gamma(q,i)) * prior(c)/(1-prior(y_i)) for c!=y_i ]
        """
        B, M, N = gamma.shape
        C = int(num_classes)
        out = gamma.new_zeros((B, M, C))

        # do math in fp32 for AMP stability
        gamma_f = gamma.float()

        for b in range(B):
            mask = support_mask[b].bool()
            if not mask.any():
                out[b].fill_(1.0 / C)
                continue

            g = gamma_f[b][:, mask]          # (M, N_eff)
            y = y_support[b][mask].long()    # (N_eff,)
            N_eff = y.numel()

            # If somehow only one class exists in supports, the "negative distribution" is undefined.
            # In that degenerate case, prediction must be that class with prob 1.
            uniq = torch.unique(y)
            if uniq.numel() == 1:
                out[b].zero_()
                out[b, :, int(uniq.item())] = 1.0
                continue

            counts = torch.bincount(y, minlength=C).float()
            # Faithful PDLC prior (no clamping). If you want smoothing: counts + alpha.
            prior = counts / counts.sum().clamp_min(1.0)

            # positive: sum_{i:y_i=c} gamma(q,i)
            idx = y.unsqueeze(0).expand(M, N_eff)
            # Optional top-k gating: exclude non-topk anchors from both pos and neg.
            if self.cfg.topk is not None:
                k = int(self.cfg.topk)
                if k > 0 and k < N_eff:
                    topk_idx = g.topk(k, dim=1).indices  # (M, k)
                    keep = torch.zeros_like(g, dtype=torch.bool)
                    keep.scatter_(1, topk_idx, True)
                else:
                    keep = torch.ones_like(g, dtype=torch.bool)
                keep_f = keep.to(dtype=g.dtype)
                g_used = g * keep_f
                one_minus_g_used = (1.0 - g) * keep_f
                denom_q = keep_f.sum(dim=1).clamp_min(1.0)  # (M,)
            else:
                g_used = g
                one_minus_g_used = 1.0 - g
                denom_q = g.new_full((M,), float(N_eff))

            pos = g.new_zeros((M, C))
            pos.scatter_add_(1, idx, g_used)

            # build rho (N_eff, C): rho[i,c] = prior[c] / (1 - prior[y_i]) for c != y_i; 0 for c==y_i
            denom = (1.0 - prior.gather(0, y)).clamp_min(1e-6)         # (N_eff,)
            rho = prior.unsqueeze(0) / denom.unsqueeze(1)              # (N_eff, C)
            rho.scatter_(1, y.view(-1, 1), 0.0)

            # negative: sum_i (1-gamma(q,i)) * rho_i,c
            neg = one_minus_g_used @ rho                                 # (M, C)

            p = (pos + neg) / denom_q.unsqueeze(1)                       # (M, C)
            # p should already be normalized; minor safety:
            p = p.clamp_min(0.0)
            p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)

            out[b] = p.to(dtype=gamma.dtype)

        return out


    def _aggregate_class_pool(
        self,
        gamma: torch.Tensor,          # (B, M, N)
        y_support: torch.Tensor,      # (B, N)
        support_mask: torch.Tensor,   # (B, N)
        num_classes: int,
    ) -> torch.Tensor:
        """Class-pool aggregation: mean gamma per class.

        For each query q:
            score_c(q) = mean_{i: y_i = c} gamma(q, i)
        """
        B, M, N = gamma.shape
        C = num_classes
        out = gamma.new_zeros((B, M, C))

        for b in range(B):
            g = gamma[b]                  # (M, N)
            y = y_support[b].long()       # (N,)
            mask = support_mask[b]        # (N,)

            if not mask.any():
                continue

            g = g[:, mask]                # (M, N_eff)
            y = y[mask]                   # (N_eff,)

            idx = y.unsqueeze(0).expand(g.shape[0], -1)  # (M, N_eff)
            P_sum = torch.zeros((g.shape[0], C), device=g.device, dtype=g.dtype)
            P_sum.scatter_add_(1, idx, g)               # (M, C)

            # Normalize by support count per class to obtain mean gamma
            counts = torch.bincount(y, minlength=C).to(g.dtype)  # (C,)
            counts = counts.clamp_min(1.0)                       # avoid division by zero
            P_mean = P_sum / counts.unsqueeze(0)                 # (M, C)

            out[b] = P_mean

        return out

    def _aggregate_sum(
        self,
        gamma: torch.Tensor,          # (B, M, N)
        y_support: torch.Tensor,      # (B, N)
        support_mask: torch.Tensor,   # (B, N)
        num_classes: int,
    ) -> torch.Tensor:
        """Sum aggregation (Kernel Density / Evidence Mass).

        Score_c(q) = sum_{i: y_i = c} gamma(q, i)

        Unlike class_pool, this does NOT divide by the class count.
        This allows the density of the support set to act as a prior:
        dense clusters have more voting mass than sparse outliers.
        """
        B, M, N = gamma.shape
        C = num_classes
        out = gamma.new_zeros((B, M, C))

        for b in range(B):
            g = gamma[b]                  # (M, N)
            y = y_support[b].long()       # (N,)
            mask = support_mask[b]        # (N,)

            if not mask.any():
                continue

            g = g[:, mask]                # (M, N_eff)
            y = y[mask]                   # (N_eff,)

            idx = y.unsqueeze(0).expand(g.shape[0], -1)  # (M, N_eff)
            P_sum = torch.zeros((g.shape[0], C), device=g.device, dtype=g.dtype)
            P_sum.scatter_add_(1, idx, g)               # (M, C)

            # No division by counts; return total mass.
            out[b] = P_sum

        return out

    def _aggregate(
        self,
        gamma: torch.Tensor,
        y_support: torch.Tensor,
        support_mask: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Dispatch to the configured aggregation method."""
        mode = self.cfg.agg
        if mode == "posterior_avg":
            return self._aggregate_posterior_avg(gamma, y_support, support_mask, num_classes)
        if mode == "class_pool":
            return self._aggregate_class_pool(gamma, y_support, support_mask, num_classes)
        if mode == "sum":
            return self._aggregate_sum(gamma, y_support, support_mask, num_classes)
        raise ValueError(f"Unknown PDLC aggregation mode: {mode}")

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------
    def forward(
        self,
        H_support: torch.Tensor,        # (B, N, D)
        H_query: torch.Tensor,          # (B, M, D)
        y_support: torch.Tensor,        # (B, N)
        support_mask: Optional[torch.Tensor] = None,  # (B, N)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute class logits for queries given support embeddings."""
        if support_mask is None:
            support_mask = torch.ones_like(y_support, dtype=torch.bool)

        B, N, _ = H_support.shape
        _, M, _ = H_query.shape

        num_classes = int(y_support[support_mask].max().item()) + 1
        num_classes = min(num_classes, self.max_classes)

        should_chunk = (not self.training) and (B * M * N > self._CHUNK_FLOP_THRESHOLD)

        if should_chunk:
            return self._forward_chunked(H_support, H_query, y_support, support_mask, num_classes)
        else:
            return self._forward_full(H_support, H_query, y_support, support_mask, num_classes)

    def _apply_inference_temperature(
        self,
        logits_query: torch.Tensor,    # (B, M, C)
        support_mask: torch.Tensor,    # (B, N)
    ) -> torch.Tensor:
        """Apply inference-time temperature scaling."""
        if self.training:
            return logits_query

        # Special handling for "sum" aggregation: scale by sqrt(N_eff)
        if self.cfg.agg == "sum":
            N_eff = support_mask.sum(dim=1).clamp_min(1).to(logits_query.dtype)  # (B,)
            scale = torch.sqrt(N_eff).view(-1, 1, 1)                             # (B,1,1)
            return logits_query / scale

        # Generic fixed temperature scaling
        temp = getattr(self.cfg, "Inference_temperature", 0.0)
        if temp is not None and temp > 0:
            return logits_query / float(temp)

        return logits_query

    def _forward_full(
        self,
        H_support: torch.Tensor,
        H_query: torch.Tensor,
        y_support: torch.Tensor,
        support_mask: torch.Tensor,
        num_classes: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Non-chunked forward pass (no symmetrization)."""
        # Forward logits ell(q, a)
        pair_logits_qs = self._pair_logits(H_query, H_support, support_mask)  # (B, M, N)

        # Pairwise "same-class" probabilities gamma(q, a)
        gamma = torch.sigmoid(pair_logits_qs)  # (B, M, N)

        # Aggregate to class-level scores
        scores = self._aggregate(gamma, y_support, support_mask, num_classes)  # (B, M, C)

        # Convert to log-space
        eps = 1e-12
        logits_query = torch.log(torch.clamp(scores, min=eps))  # (B, M, C)

        # Inference-time temperature scaling
        logits_query = self._apply_inference_temperature(logits_query, support_mask)

        aux: Dict[str, torch.Tensor] = {
            "pair_logits": pair_logits_qs,
            "gamma": gamma,
            "support_mask": support_mask,
            "tau": self.tau,
        }

        return logits_query, aux

    def _forward_chunked(
        self,
        H_support: torch.Tensor,
        H_query: torch.Tensor,
        y_support: torch.Tensor,
        support_mask: torch.Tensor,
        num_classes: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Chunked inference for very large (B, M, N).

        We chunk along the query dimension M to keep memory usage bounded
        while still supporting symmetrization.
        """
        B, N, _ = H_support.shape
        _, M, _ = H_query.shape

        # Pre-normalize supports and project once
        Hs_norm = self._normalize_embeddings(H_support)  # (B, N, D)
        K_support = self.W_K(Hs_norm)                   # (B, N, D)

        scores = H_support.new_zeros((B, M, num_classes))

        for start in range(0, M, self._CHUNK_SIZE):
            end = min(M, start + self._CHUNK_SIZE)
            Hq_chunk = H_query[:, start:end, :]          # (B, m, D)

            # Normalize and project queries
            Hq_norm = self._normalize_embeddings(Hq_chunk)  # (B, m, D)
            Q_chunk = self.W_Q(Hq_norm)                     # (B, m, D)

            # Forward logits ell(q, a): (B, m, N)
            logits_qs = torch.matmul(Q_chunk, K_support.transpose(-1, -2))
            logits_qs = self.tau * logits_qs + self.bias
            logits_qs = self._apply_support_mask_and_topk(logits_qs, support_mask)
            gamma_qs = torch.sigmoid(logits_qs)             # (B, m, N)

            gamma_chunk = gamma_qs

            # Aggregate for this chunk
            scores_chunk = self._aggregate(gamma_chunk, y_support, support_mask, num_classes)  # (B, m, C)
            scores[:, start:end, :] = scores_chunk

        eps = 1e-12
        logits_query = torch.log(torch.clamp(scores, min=eps))  # (B, M, C)
        logits_query = self._apply_inference_temperature(logits_query, support_mask)

        aux: Dict[str, torch.Tensor] = {
            # In chunked mode we do not keep full pair_logits / gamma tensors
            # to save memory; training does not use chunked mode.
            "support_mask": support_mask,
            "tau": self.tau,
        }

        return logits_query, aux


__all__ = ["TabPDLHeadConfig", "TabPDLHead"]
