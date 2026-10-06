"""Projection from latent space into the frozen LLM's input-embedding space."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def mean_embedding_norm(embedding_weight: Tensor) -> float:
    """Average L2 norm of the LLM's token embeddings: the scale soft prompts should match."""
    return float(embedding_weight.float().norm(dim=-1).mean())


class Projector(nn.Module):
    """MLP into the LLM embedding space, rescaled to the norm of real token embeddings.

    A soft prompt whose vectors are much larger or smaller than real embeddings is out of
    distribution for the frozen LLM and is a common cause of unstable training. Each output vector
    is normalised and multiplied by a learned gain that starts at the mean embedding norm.
    """

    def __init__(self, d: int, d_llm: int, target_norm: float, pe_dim: int = 0):
        super().__init__()
        self.pe = nn.Linear(pe_dim, d) if pe_dim else None
        self.mlp = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 2 * d_llm), nn.GELU(), nn.Linear(2 * d_llm, d_llm)
        )
        self.log_gain = nn.Parameter(torch.tensor(float(target_norm)).log())

    def forward(self, z: Tensor, pe: Tensor | None = None) -> Tensor:
        if self.pe is not None and pe is not None:
            z = z + self.pe(pe.to(z.dtype))  # graph positional bias
        out = self.mlp(z)
        out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return out * self.log_gain.exp()
