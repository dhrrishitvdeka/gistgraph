"""Compressor assembly. ``build_compressor`` picks the variant named in the config."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from gistgraph.config import Config
from gistgraph.model.encoder import SegmentEncoder
from gistgraph.model.gates import HardConcreteGate
from gistgraph.model.projector import Projector
from gistgraph.model.routing import RoutedWriter
from gistgraph.model.writer import FlatSlotWriter, num_slots


@dataclass
class Memory:
    """What the LLM reads: ``embeds[b, :counts[b]]`` are the soft-prompt vectors of document b."""

    embeds: Tensor  # [B, K, d_llm]
    mask: Tensor  # [B, K] bool
    aux: dict = field(default_factory=dict)

    @property
    def counts(self) -> Tensor:
        return self.mask.sum(1)

    def prefix(self, i: int) -> Tensor:
        return self.embeds[i, self.mask[i]]


class FlatCompressor(nn.Module):
    """Segment encoder -> flat Perceiver-style slots -> projector. The M2 baseline."""

    def __init__(self, d_llm: int, embed_norm: float, k_max: int, cfg: Config):
        super().__init__()
        c = cfg.compressor
        self.k_max = k_max
        self.encoder = SegmentEncoder(
            d_llm, c.encoder.d, c.encoder.layers, seg_len=c.encoder.seg_len
        )
        self.writer = FlatSlotWriter(c.encoder.d, k_max, layers=c.encoder.layers)
        self.projector = Projector(c.encoder.d, d_llm, embed_norm)

    def forward(self, h: Tensor, mask: Tensor, ratio: float | Tensor) -> Memory:
        b = h.shape[0]
        ratio_t = torch.as_tensor(ratio, dtype=torch.float32, device=h.device).expand(b)
        counts = num_slots(mask.sum(1), ratio_t, self.k_max)
        u, u_valid = self.encoder(h, mask)
        z, z_valid, aux = self.writer(u, u_valid, counts, ratio_t)
        return Memory(self.projector(z), z_valid, aux)


class RoutedCompressor(nn.Module):
    """Segment encoder -> routed sparse writing -> (optional) adaptive gates -> projector.

    With ``writer.adaptive`` false every document gets exactly ``N / ratio`` nodes (fixed rate).
    With it true the candidate pool is ``capacity_factor`` times larger and hard-concrete gates
    decide how many nodes stay open, so information-dense documents can keep more of them. The
    rate penalty (see the trainer) holds the *average* open fraction at ``1 / ratio``.
    """

    def __init__(self, d_llm: int, embed_norm: float, k_max: int, cfg: Config):
        super().__init__()
        c = cfg.compressor
        self.k_max, self.adaptive, self.cf = k_max, c.writer.adaptive, c.writer.capacity_factor
        d = c.encoder.d
        self.encoder = SegmentEncoder(d_llm, d, c.encoder.layers, seg_len=c.encoder.seg_len)
        self.writer = RoutedWriter(
            d, k_max, c.writer.top_k, c.encoder.layers, c.writer.router_noise
        )
        self.gate = HardConcreteGate(d, c.writer.gate_init) if self.adaptive else None
        self.projector = Projector(d, d_llm, embed_norm)

    def _nodes(self, h: Tensor, mask: Tensor, ratio: float | Tensor):
        """Encode and write. Returns node states, node validity, aux, ratio tensor, token counts."""
        b = h.shape[0]
        ratio_t = torch.as_tensor(ratio, dtype=torch.float32, device=h.device).expand(b)
        n_tokens = mask.sum(1)
        pool_ratio = ratio_t / self.cf if self.adaptive else ratio_t
        counts = num_slots(n_tokens, pool_ratio, self.k_max)
        u, u_valid = self.encoder(h, mask)
        z, z_valid, aux = self.writer(u, u_valid, counts, ratio_t)
        aux["n_tokens"] = n_tokens.float()
        return z, z_valid, aux, ratio_t

    def _open(self, z: Tensor, z_valid: Tensor, aux: dict) -> Tensor:
        """Gate values ``[B, K]``; fixed-rate models keep every valid node fully open."""
        if not self.adaptive:
            return z_valid.float()
        gate, expected_open = self.gate(z, z_valid)
        aux["exp_active"] = expected_open
        # never let a document end up with an empty memory: reopen its best node
        empty = gate.sum(1) == 0
        if empty.any():
            best = self.gate.log_alpha(z).masked_fill(~z_valid, float("-inf")).argmax(1)
            rescue = torch.zeros_like(gate).scatter(1, best[:, None], 1.0)
            gate = torch.where(empty[:, None], rescue.detach(), gate)
        return gate

    def forward(self, h: Tensor, mask: Tensor, ratio: float | Tensor) -> Memory:
        z, z_valid, aux, _ = self._nodes(h, mask, ratio)
        gate = self._open(z, z_valid, aux)
        return self._finish(z, gate, aux)

    def _finish(self, z: Tensor, gate: Tensor, aux: dict, order_key: Tensor | None = None):
        embeds = self.projector(z) * gate.unsqueeze(-1)
        active = gate > 0
        key = aux["centroid"] if order_key is None else order_key
        order = torch.where(active, key, torch.full_like(key, 3.0)).argsort(1)
        embeds = embeds.gather(1, order[..., None].expand(-1, -1, embeds.shape[-1]))
        active = active.gather(1, order)
        aux["active"] = active.sum(1)
        return Memory(embeds, active, aux)


def max_slots(cfg: Config) -> int:
    """Largest slot count any training or eval ratio can ask for."""
    lowest = min([*cfg.train.ratios, cfg.compressor.ratio])
    pool = cfg.compressor.writer.capacity_factor if cfg.compressor.writer.adaptive else 1.0
    return math.ceil(pool * cfg.data.max_ctx_tokens / lowest)


def build_compressor(cfg: Config, d_llm: int, embed_norm: float) -> nn.Module:
    kind = cfg.compressor.type
    if kind == "flat":
        return FlatCompressor(d_llm, embed_norm, max_slots(cfg), cfg)
    if kind == "routed":
        return RoutedCompressor(d_llm, embed_norm, max_slots(cfg), cfg)
    raise KeyError(f"unknown compressor type {kind!r}")
