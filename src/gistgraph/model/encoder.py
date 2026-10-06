"""Segment encoder: LLM token features -> a short sequence of segment representations."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sinusoidal_positions(n: int, d: int, device=None) -> Tensor:
    """Fixed sin/cos position codes ``[n, d]`` (``d`` must be even)."""
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class SegmentEncoder(nn.Module):
    """Pool tokens into segments of ``seg_len`` and contextualise them with a small transformer.

    Input is the frozen LLM's hidden states for the context (``[B, N, d_in]``). Using the LLM's own
    features means no second language model has to be loaded. Output is ``[B, M, d]`` with
    ``M = ceil(N / seg_len)`` and a boolean mask of valid segments.
    """

    def __init__(self, d_in: int, d: int = 512, layers: int = 2, heads: int = 8, seg_len: int = 1):
        super().__init__()
        self.seg_len = seg_len
        self.d = d
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=4 * d, dropout=0.0, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, h: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """``h``: ``[B, N, d_in]``; ``mask``: ``[B, N]`` bool/0-1 (1 = real token)."""
        mask = mask.bool()
        x = self.inp(h.float())
        if self.seg_len > 1:
            b, n, _ = x.shape
            pad = (-n) % self.seg_len
            if pad:
                x = torch.cat([x, x.new_zeros(b, pad, self.d)], dim=1)
                mask = torch.cat([mask, mask.new_zeros(b, pad)], dim=1)
            m = x.shape[1] // self.seg_len
            xs = x.view(b, m, self.seg_len, self.d)
            ms = mask.view(b, m, self.seg_len)
            weights = ms.unsqueeze(-1).to(x.dtype)
            x = (xs * weights).sum(2) / weights.sum(2).clamp(min=1)
            mask = ms.any(-1)
        x = x + sinusoidal_positions(x.shape[1], self.d, x.device)
        x = self.blocks(x, src_key_padding_mask=~mask)
        return self.out_norm(x), mask
