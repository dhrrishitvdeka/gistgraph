"""Relational message passing over the induced latent graph."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RelationalMP(nn.Module):
    """A few residual layers of relation-specific message passing.

    Layer update for node ``i``::

        m_i = sum_p sum_j A[p, i, j] * W_p z_j
        z_i <- z_i + O m_i
        z_i <- z_i + FFN(z_i)

    with pre-LayerNorm. After ``L`` layers a node's state depends on its ``L``-hop neighbourhood,
    which is how multi-hop relations get packed into the node vectors. Nothing here uses a node's
    index, so the layer is permutation-equivariant: permuting the nodes (and the adjacency to
    match) permutes the outputs the same way.
    """

    def __init__(self, d: int, relations: int, layers: int = 2):
        super().__init__()
        self.rel_weights = nn.ParameterList(
            nn.Parameter(torch.randn(relations, d, d) * d**-0.5) for _ in range(layers)
        )
        self.out = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.ln_msg = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.ln_ffn = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            for _ in range(layers)
        )
        for o in self.out:  # start close to the identity so training begins from the routed model
            nn.init.zeros_(o.weight)
            nn.init.zeros_(o.bias)

    def forward(self, z: Tensor, adj: Tensor, valid: Tensor) -> Tensor:
        """``z``: ``[B, K, d]``; ``adj``: ``[B, R, K, K]``; ``valid``: ``[B, K]`` bool."""
        mask = valid.unsqueeze(-1).to(z.dtype)
        for w, out, ln_m, ln_f, ffn in zip(
            self.rel_weights, self.out, self.ln_msg, self.ln_ffn, self.ffn, strict=True
        ):
            if adj is not None:
                h = ln_m(z)
                per_rel = torch.einsum("bkd,rde->brke", h, w.to(h.dtype))  # W_p z_j, all p, j
                msg = torch.einsum("brij,brje->bie", adj.to(h.dtype), per_rel)
                z = (z + out(msg)) * mask
            z = (z + ffn(ln_f(z))) * mask
        return z
