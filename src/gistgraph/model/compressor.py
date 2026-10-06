"""Compressor assembly. ``build_compressor`` picks the variant named in the config."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from gistgraph.config import Config
from gistgraph.model.encoder import SegmentEncoder
from gistgraph.model.projector import Projector
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


def max_slots(cfg: Config) -> int:
    """Largest slot count any training or eval ratio can ask for."""
    lowest = min([*cfg.train.ratios, cfg.compressor.ratio])
    return math.ceil(cfg.data.max_ctx_tokens / lowest)


def build_compressor(cfg: Config, d_llm: int, embed_norm: float) -> nn.Module:
    kind = cfg.compressor.type
    if kind == "flat":
        return FlatCompressor(d_llm, embed_norm, max_slots(cfg), cfg)
    raise KeyError(f"unknown compressor type {kind!r}")
