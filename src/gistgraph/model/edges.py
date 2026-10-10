"""Latent edge induction: sparse typed edges between nodes, learned without any parser."""

from __future__ import annotations

import torch
from torch import Tensor, nn

NEG = -1e4  # stands in for -inf: finite, so sorting and cumulative sums stay well defined


def entmax15(x: Tensor) -> Tensor:
    """Entmax with alpha = 1.5 along the last dimension (Peters et al., 2019).

    Like softmax it returns a probability vector, but low scores get *exactly* zero weight, which
    gives genuinely sparse edge sets. This is the sort-based exact solution, written with plain
    differentiable tensor ops, so autograd gives the correct gradient.
    """
    x = (x - x.max(-1, keepdim=True).values) / 2
    xs, _ = torch.sort(x, dim=-1, descending=True)
    rho = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
    mean = xs.cumsum(-1) / rho
    mean_sq = (xs**2).cumsum(-1) / rho
    delta = (1 - rho * (mean_sq - mean**2)) / rho
    tau = mean - delta.clamp(min=1e-12).sqrt()
    support = (tau <= xs).sum(-1, keepdim=True)
    tau_star = tau.gather(-1, support - 1)
    return (x - tau_star).clamp(min=0) ** 2


def cap_degree(weights: Tensor, max_deg: int) -> Tensor:
    """Keep each row's ``max_deg`` largest weights and rescale to the row's original total.

    Chosen by index (not by a value threshold) so ties can never push a row above ``max_deg``.
    """
    if max_deg >= weights.shape[-1]:
        return weights
    top = weights.topk(max_deg, dim=-1).indices
    kept = weights * torch.zeros_like(weights).scatter(-1, top, 1.0)
    scale = weights.sum(-1, keepdim=True) / kept.sum(-1, keepdim=True).clamp(min=1e-9)
    return kept * scale


class EdgeInducer(nn.Module):
    """Typed, sparse, directed edges from a bilinear score with learned relation embeddings.

    For relation ``p`` the score from node ``i`` to node ``j`` is
    ``(U z_i)^T diag(w_p) (V z_j) / sqrt(r)``. Every node ``i`` chooses its incoming neighbours
    over *all* (relation, node) pairs jointly, plus a learned "no edge" option, using entmax (or a
    Gumbel top-k ablation). Rows of the returned adjacency therefore sum to at most one, and most
    entries are exactly zero.
    """

    def __init__(
        self,
        d: int,
        relations: int = 4,
        rank: int = 64,
        sparsifier: str = "entmax15",
        max_deg: int = 8,
    ):
        super().__init__()
        if sparsifier not in ("entmax15", "gumbel_topk"):
            raise ValueError(f"unknown sparsifier {sparsifier!r}")
        self.relations, self.rank, self.sparsifier, self.max_deg = (
            relations,
            rank,
            sparsifier,
            max_deg,
        )
        self.src = nn.Linear(d, rank, bias=False)
        self.dst = nn.Linear(d, rank, bias=False)
        self.rel = nn.Parameter(torch.randn(relations, rank))
        self.null = nn.Linear(d, 1)
        nn.init.constant_(self.null.bias, 0.0)

    def forward(self, z: Tensor, valid: Tensor) -> tuple[Tensor, dict]:
        """``z``: ``[B, K, d]``; ``valid``: ``[B, K]`` bool. Returns ``A [B, R, K, K]`` and aux.

        ``A[b, p, i, j]`` is the weight with which node ``i`` reads from node ``j`` over relation
        ``p``. Self-loops are excluded. Runs in float32 even under mixed precision: low-precision
        scores produce many ties and the entmax sort/cumsum needs the extra range.
        """
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self._induce(z, valid)

    def _induce(self, z: Tensor, valid: Tensor) -> tuple[Tensor, dict]:
        b, k, _ = z.shape
        r = self.relations
        z = z.float()
        q, key = self.src(z), self.dst(z)
        scores = torch.einsum("bir,pr,bjr->bpij", q, self.rel, key) * self.rank**-0.5
        allowed = valid[:, None, :, None] & valid[:, None, None, :]
        allowed = allowed & ~torch.eye(k, dtype=torch.bool, device=z.device)
        scores = scores.masked_fill(~allowed, NEG)
        flat = scores.permute(0, 2, 1, 3).reshape(b, k, r * k)  # per source node i: (p, j)
        null = self.null(z)  # [B, K, 1]
        logits = torch.cat([flat, null], dim=-1)

        probs = entmax15(logits) if self.sparsifier == "entmax15" else self._gumbel_topk(logits)
        edge_probs = probs[..., :-1]
        edge_probs = (
            cap_degree(edge_probs, self.max_deg) if self.sparsifier == "entmax15" else edge_probs
        )
        edge_probs = edge_probs * valid[:, :, None]
        adj = edge_probs.reshape(b, k, r, k).permute(0, 2, 1, 3)  # [B, R, K, K]

        n_valid = valid.sum(1).clamp(min=1).float()
        nnz = (edge_probs > 0).sum(-1).float() * valid  # edges per node
        row = edge_probs.sum(-1).clamp(min=1e-9)
        ent = -(edge_probs / row[..., None] * torch.log(edge_probs / row[..., None] + 1e-9)).sum(-1)
        # the Gumbel top-k weights are piecewise constant in the null logit, so the edge penalty
        # reads the dense softmax "no edge" probability instead to keep a gradient
        null_prob = probs[..., -1] if self.sparsifier == "entmax15" else logits.softmax(-1)[..., -1]
        aux = {
            "exp_edges": ((1 - null_prob) * valid).sum(1),
            "mean_degree": nnz.sum(1) / n_valid,
            "edge_entropy": (ent * valid).sum(1) / n_valid,
            "edge_mass": (edge_probs.sum(-1) * valid).sum(1) / n_valid,
        }
        return adj, aux

    def _gumbel_topk(self, logits: Tensor) -> Tensor:
        """Pick ``max_deg`` entries per node (Gumbel-perturbed while training) and weight them by a
        softmax of their clean logits. Gradients reach the logits through the weights."""
        noisy = logits
        if self.training:
            g = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-9, 1 - 1e-9)))
            noisy = logits + g
        top = noisy.topk(min(self.max_deg, logits.shape[-1]), dim=-1).indices
        chosen = logits.gather(-1, top)
        weights = chosen.softmax(-1)
        return torch.zeros_like(logits).scatter(-1, top, weights)


def random_adjacency(
    valid: Tensor, relations: int, max_deg: int, generator: torch.Generator | None = None
) -> Tensor:
    """Control: each valid node reads from ``max_deg`` random valid nodes with random relations.

    Same degree and relation count as the learned graph, but no information about the content.
    Weights are uniform and rows sum to one. Pass a seeded ``generator`` (on ``valid``'s device)
    for a reproducible graph. Returns ``[B, R, K, K]``.
    """
    b, k = valid.shape
    r = relations
    scores = torch.rand(b, k, r * k, device=valid.device, generator=generator)
    ok = valid[:, None, :] & valid[:, :, None]  # [B, i, j]
    ok = ok & ~torch.eye(k, dtype=torch.bool, device=valid.device)
    ok = ok.unsqueeze(2).expand(b, k, r, k).reshape(b, k, r * k)
    scores = scores.masked_fill(~ok, -1.0)
    deg = min(max_deg, r * k)
    top = scores.topk(deg, dim=-1)
    w = (top.values >= 0).float()
    w = w / w.sum(-1, keepdim=True).clamp(min=1)
    flat = torch.zeros_like(scores).scatter(-1, top.indices, w)
    return flat.reshape(b, k, r, k).permute(0, 2, 1, 3) * valid[:, None, :, None]


def rw_positional_encoding(adj: Tensor, steps: int) -> Tensor:
    """Random-walk return probabilities ``[B, K, steps]`` from the relation-summed adjacency.

    Feature ``t`` of node ``i`` is the probability that a ``t``-step random walk starting at ``i``
    returns to ``i``. It describes the node's place in the graph without referring to its index,
    so it is permutation-equivariant. Computed without gradient: it is a structural feature.
    """
    with torch.no_grad():
        p = adj.sum(1)  # [B, K, K]
        cur = p
        feats = []
        for _ in range(steps):
            feats.append(torch.diagonal(cur, dim1=-2, dim2=-1))
            cur = cur @ p
    return torch.stack(feats, dim=-1)
