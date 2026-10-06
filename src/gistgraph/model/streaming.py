"""Incremental (streaming) memory: new segments merge into or overwrite existing nodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as nnf
from torch import Tensor, nn

from gistgraph.model.routing import RoutedWriter
from gistgraph.model.writer import slot_mask_from_counts


@dataclass
class StreamState:
    """Node memory after some chunks: raw (pre-finalise) states plus bookkeeping."""

    z: Tensor  # [B, K, d] weighted-mean write states
    mass: Tensor  # [B, K] total write weight each node has received
    pos_sum: Tensor  # [B, K] sum of (write weight * source position), for node ordering


class MergeGate(nn.Module):
    """How much a chunk's write overwrites an existing node (1 = overwrite, 0 = keep).

    The gate starts as the *mass-weighted running average*, ``new_mass / (old_mass + new_mass)``,
    i.e. each node stays the average of everything written to it. A learned correction (zero at
    initialisation) is added to its logit, so a model can learn to overwrite stale content or
    protect important nodes once it is trained with streaming.
    """

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d + 1, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_old: Tensor, z_new: Tensor, m_old: Tensor, m_new: Tensor) -> Tensor:
        prior = (m_new / (m_old + m_new).clamp(min=1e-6)).clamp(1e-4, 1 - 1e-4)
        correction = self.net(torch.cat([z_old, z_new, torch.log1p(m_new).unsqueeze(-1)], -1))
        return torch.sigmoid(torch.logit(prior).unsqueeze(-1) + correction)


def _grow(x: Tensor, k: int) -> Tensor:
    """Zero-pad dimension 1 up to ``k`` nodes."""
    pad = k - x.shape[1]
    return x if pad <= 0 else nnf.pad(x, (0, 0) * (x.dim() - 2) + (0, pad))


def stream_update(
    writer: RoutedWriter,
    merge: MergeGate | None,
    state: StreamState | None,
    u: Tensor,
    u_valid: Tensor,
    pos: Tensor,
    counts: Tensor,
    ratio: Tensor,
) -> tuple[StreamState, Tensor, Tensor]:
    """Fold one chunk of segments into the node memory.

    ``counts`` is the node budget *after* this chunk (``floor(tokens_seen / ratio)`` per
    document), which can only grow, so the memory always holds ``N_seen / ratio`` nodes. The chunk
    is routed over all nodes. Existing nodes that receive weight are merged with the new write;
    fresh nodes take it as is; nodes that receive nothing are left alone. Returns the new state,
    the chunk's load-balance loss and the node validity mask.
    """
    k = int(counts.max())
    z_valid = slot_mask_from_counts(counts, k)
    keys = writer.node_keys(k, ratio)
    z_c, write, mass_c, probs = writer.route_write(u, u_valid, keys, z_valid)
    lb = writer.load_balance(probs, write, u_valid, counts)
    chunk_pos = torch.einsum("bmk,bm->bk", write, pos)
    if state is None:
        return StreamState(z_c, mass_c, chunk_pos), lb, z_valid

    z_old, m_old, ps_old = _grow(state.z, k), _grow(state.mass, k), _grow(state.pos_sum, k)
    if merge is None:
        alpha = (mass_c / (m_old + mass_c).clamp(min=1e-6)).unsqueeze(-1)
    else:
        alpha = merge(z_old, z_c, m_old, mass_c)
    has_old, has_new = (m_old > 0).unsqueeze(-1), (mass_c > 0).unsqueeze(-1)
    merged = torch.where(has_old, (1 - alpha) * z_old + alpha * z_c, z_c)
    z = torch.where(has_new, merged, z_old)
    return StreamState(z, m_old + mass_c, ps_old + chunk_pos), lb, z_valid
