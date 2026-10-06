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

    def forward(
        self, u: Tensor, u_valid: Tensor, counts: Tensor, ratio: Tensor
    ) -> tuple[Tensor, Tensor, dict]:
        b, m, _ = u.shape
        k = int(counts.max())
        z_valid = slot_mask_from_counts(counts, k)
        node_keys = self.keys[None, :k] + self.ratio_embed(torch.log(ratio.float()).view(-1, 1, 1))

        logits = torch.einsum("bmd,bkd->bmk", self.route(u), node_keys) * self.d**-0.5
        if self.training and self.noise > 0:
            logits = logits + torch.randn_like(logits) * self.noise
        logits = logits.masked_fill(~z_valid[:, None, :], float("-inf"))
        probs = logits.softmax(-1)  # [B, M, K]

        top_k = min(self.top_k, k)
        vals, idx = probs.topk(top_k, dim=-1)
        weights = vals / vals.sum(-1, keepdim=True).clamp(min=1e-9)
        write = torch.zeros_like(probs).scatter(-1, idx, weights) * u_valid[:, :, None]  # [B,M,K]

        mass = write.sum(1)  # [B, K]
        z = torch.einsum("bmk,bmd->bkd", write, self.value(u)) / mass.unsqueeze(-1).clamp(min=1e-6)
        z = self.ln_in(z + self.key_proj(node_keys))
        for block in self.ffn:
            z = z + block(z)
        z = self.out_norm(z) * z_valid.unsqueeze(-1)

        # load balance: K * sum_k (fraction routed to k) * (mean router prob of k), per document
        n_units = u_valid.sum(1).clamp(min=1).float()
        top1 = torch.zeros_like(probs).scatter(-1, idx[..., :1], 1.0) * u_valid[:, :, None]
        frac = top1.sum(1) / n_units[:, None]
        mean_p = (probs * u_valid[:, :, None]).sum(1) / n_units[:, None]
        lb = (counts.float() * (frac * mean_p).sum(1)).mean()

        pos = torch.arange(m, device=u.device)[None, :] / n_units[:, None]  # position in [0, 1)
        centroid = torch.einsum("bmk,bm->bk", write, pos) / mass.clamp(min=1e-6)
        centroid = torch.where(mass > 1e-6, centroid, torch.full_like(centroid, 2.0))
        return z, z_valid, {"lb": lb, "centroid": centroid, "write": write, "mass": mass}
