"""
Activation probe architectures (Appendix C.3).

Three probe types as described in the paper:
  1. Logistic Regression  — Eq. 3
  2. MLP (1 hidden layer)  — Eq. 4
  3. Attention-based        — Eqs. 5-7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LogisticProbe(nn.Module):
    """
    Linear probe: p(h) = σ(w^T h + b)   (Eq. 3)

    Operates per-token: input (batch, seq_len, d) → output (batch, seq_len).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, seq_len, d) or (batch, d) hidden states
        Returns:
            scores: (batch, seq_len) or (batch,) probe scores in [0, 1]
        """
        return torch.sigmoid(self.linear(h)).squeeze(-1)

    @property
    def direction(self) -> torch.Tensor:
        """Return the probe's weight vector (for mechanistic analysis)."""
        return self.linear.weight.detach().squeeze(0)


class MLPProbe(nn.Module):
    """
    MLP probe with one hidden layer and ReLU (Eq. 4):
        p(h) = σ(w2^T ReLU(W1^T h + b1) + b2)

    Default hidden_dim = 64 (Appendix C.3).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(h)).squeeze(-1)


class AttentionProbe(nn.Module):
    """
    Sequence-level attention probe (Eqs. 5-7).

    Uses K attention heads to pool across the sequence, then classifies:
        α_k  = softmax(H q_k)           (Eq. 5)
        c_k  = α_k^T H                  (Eq. 6)
        p(H) = σ(Σ_k c_k^T w_k + b)    (Eq. 7)

    Default K = 4 attention heads (Appendix C.3).
    """

    def __init__(self, input_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        # Each head has a query vector q_k and output weight w_k
        self.queries = nn.Parameter(torch.randn(n_heads, input_dim) * 0.01)
        self.output_weights = nn.Parameter(torch.randn(n_heads, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        H: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            H: (batch, seq_len, d) hidden states
            attention_mask: (batch, seq_len) binary mask (1 = keep, 0 = ignore)
        Returns:
            scores: (batch,) probe scores in [0, 1]
        """
        batch, seq_len, d = H.shape

        # Compute attention weights for each head: (batch, n_heads, seq_len)
        # H @ q_k: (batch, seq_len, d) @ (d,) → (batch, seq_len)
        logits = torch.einsum("bsd,kd->bks", H, self.queries)  # (batch, n_heads, seq_len)

        if attention_mask is not None:
            # Mask out padding positions
            mask = attention_mask.unsqueeze(1).expand_as(logits)  # (batch, n_heads, seq_len)
            logits = logits.masked_fill(mask == 0, float("-inf"))

        alpha = F.softmax(logits, dim=-1)  # (batch, n_heads, seq_len)

        # Context vectors: c_k = α_k^T H → (batch, n_heads, d)
        context = torch.einsum("bks,bsd->bkd", alpha, H)

        # Output: σ(Σ_k c_k^T w_k + b)
        scores = torch.einsum("bkd,kd->b", context, self.output_weights) + self.bias

        return torch.sigmoid(scores)


# ── Factory ──────────────────────────────────────────────────────────────────

def build_probe(probe_type: str, input_dim: int, **kwargs) -> nn.Module:
    """Build a probe by name."""
    if probe_type == "logistic":
        return LogisticProbe(input_dim)
    elif probe_type == "mlp":
        return MLPProbe(input_dim, hidden_dim=kwargs.get("hidden_dim", 64))
    elif probe_type == "attention":
        return AttentionProbe(input_dim, n_heads=kwargs.get("n_heads", 4))
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")


# ── Ensemble wrapper ─────────────────────────────────────────────────────────

class ProbeEnsemble(nn.Module):
    """
    Ensemble of probes across multiple layers (Section 4.1, Figure 3).
    Aggregation via mean or max (Eq. 9).
    """

    def __init__(self, probes: dict[int, nn.Module], agg: str = "mean"):
        """
        Args:
            probes: mapping from layer index → probe module
            agg: "mean" or "max"
        """
        super().__init__()
        self.probes = nn.ModuleDict({str(k): v for k, v in probes.items()})
        self.agg = agg

    def forward(self, activations: dict[int, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Args:
            activations: mapping from layer index → (batch, seq_len, d) tensors
        Returns:
            scores: (batch,) aggregated scores
        """
        layer_scores = []
        for layer_str, probe in self.probes.items():
            layer = int(layer_str)
            h = activations[layer]
            if isinstance(probe, AttentionProbe):
                score = probe(h, **kwargs)  # (batch,)
            else:
                # Token-level probe → aggregate via mean (Eq. 8)
                token_scores = probe(h)     # (batch, seq_len)
                mask = kwargs.get("attention_mask", None)
                if mask is not None:
                    token_scores = token_scores * mask
                    score = token_scores.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
                else:
                    score = token_scores.mean(dim=-1)
            layer_scores.append(score)

        stacked = torch.stack(layer_scores, dim=-1)  # (batch, n_layers)
        if self.agg == "mean":
            return stacked.mean(dim=-1)
        elif self.agg == "max":
            return stacked.max(dim=-1).values
        else:
            raise ValueError(f"Unknown aggregation: {self.agg}")
