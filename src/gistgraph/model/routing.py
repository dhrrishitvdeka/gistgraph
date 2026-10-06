"""Routed writing: segments choose which latent nodes to write to (MoE-style top-k routing)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from gistgraph.model.writer import slot_mask_from_counts


class RoutedWriter(nn.Module):
    """Each segment is written to its ``top_k`` highest-scoring nodes.

    A segment's routing weights are a softmax over nodes, truncated to the top-k and renormalised,
    so every segment touches exactly ``top_k`` nodes (sparse writing). A node's state is the
    weighted mean of the values written to it, plus a learned node key, followed by per-node
    feed-forward blocks. Nodes do not attend to each other here: any interaction between nodes is
    added explicitly (edges and message passing in the graph model), so the flat Perceiver baseline
    and this model differ in a controlled way.

    A Switch-style load-balancing loss discourages the router from sending everything to a few
    nodes.
    """

    def __init__(self, d: int, k_max: int, top_k: int = 2, layers: int = 2, noise: float = 0.1):
        super().__init__()
        self.k_max, self.top_k, self.noise, self.d = k_max, top_k, noise, d
        self.keys = nn.Parameter(torch.randn(k_max, d) * d**-0.5)
        self.ratio_embed = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.route = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d)
        self.key_proj = nn.Linear(d, d)
        self.ln_in = nn.LayerNorm(d)
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            for _ in range(layers)
        )
        self.out_norm = nn.LayerNorm(d)

    def node_keys(self, k: int, ratio: Tensor) -> Tensor:
        """Keys of the first ``k`` nodes, shifted by a learned embedding of the target ratio."""
        return self.keys[None, :k] + self.ratio_embed(torch.log(ratio.float()).view(-1, 1, 1))

    def route_write(
        self, u: Tensor, u_valid: Tensor, node_keys: Tensor, z_valid: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Route segments to nodes and write. Returns ``(z_raw, write, mass, probs)``.

        ``z_raw`` is the weighted mean of the values written to each node (before the
        normalisation and feed-forward blocks of :meth:`finalize`), ``write`` the sparse routing
        weights ``[B, M, K]``, ``mass`` the total weight each node received, and ``probs`` the
        dense router probabilities used by the load-balance loss.
        """
        k = node_keys.shape[1]
        logits = torch.einsum("bmd,bkd->bmk", self.route(u), node_keys) * self.d**-0.5
        if self.training and self.noise > 0:
            logits = logits + torch.randn_like(logits) * self.noise
        logits = logits.masked_fill(~z_valid[:, None, :], float("-inf"))
        probs = logits.softmax(-1)  # [B, M, K]

        top_k = min(self.top_k, k)
        vals, idx = probs.topk(top_k, dim=-1)
        weights = vals / vals.sum(-1, keepdim=True).clamp(min=1e-9)
        write = torch.zeros_like(probs).scatter(-1, idx, weights) * u_valid[:, :, None]
        mass = write.sum(1)  # [B, K]
        z_raw = torch.einsum("bmk,bmd->bkd", write, self.value(u))
        return z_raw / mass.unsqueeze(-1).clamp(min=1e-6), write, mass, probs

    def finalize(self, z_raw: Tensor, node_keys: Tensor, z_valid: Tensor) -> Tensor:
        """Add the node key, then LayerNorm and per-node feed-forward blocks."""
        z = self.ln_in(z_raw + self.key_proj(node_keys))
        for block in self.ffn:
            z = z + block(z)
        return self.out_norm(z) * z_valid.unsqueeze(-1)

    @staticmethod
    def load_balance(probs: Tensor, write: Tensor, u_valid: Tensor, counts: Tensor) -> Tensor:
        """K * sum_k (fraction of segments whose top choice is k) * (mean router prob of k)."""
        n_units = u_valid.sum(1).clamp(min=1).float()
        top1 = (write == write.max(-1, keepdim=True).values) & (write > 0)
        frac = (top1.float() * u_valid[:, :, None]).sum(1) / n_units[:, None]
        mean_p = (probs * u_valid[:, :, None]).sum(1) / n_units[:, None]
        return (counts.float() * (frac * mean_p).sum(1)).mean()

    def forward(
        self, u: Tensor, u_valid: Tensor, counts: Tensor, ratio: Tensor
    ) -> tuple[Tensor, Tensor, dict]:
        b, m, _ = u.shape
        k = int(counts.max())
        z_valid = slot_mask_from_counts(counts, k)
        keys = self.node_keys(k, ratio)
        z_raw, write, mass, probs = self.route_write(u, u_valid, keys, z_valid)
        z = self.finalize(z_raw, keys, z_valid)

        n_units = u_valid.sum(1).clamp(min=1).float()
        pos = torch.arange(m, device=u.device)[None, :] / n_units[:, None]  # position in [0, 1)
        centroid = torch.einsum("bmk,bm->bk", write, pos) / mass.clamp(min=1e-6)
        centroid = torch.where(mass > 1e-6, centroid, torch.full_like(centroid, 2.0))
        lb = self.load_balance(probs, write, u_valid, counts)
        return z, z_valid, {"lb": lb, "centroid": centroid, "write": write, "mass": mass}
