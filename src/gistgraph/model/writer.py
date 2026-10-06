"""Writers: turn segment representations into latent slots."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def num_slots(n_tokens: Tensor, ratio: float | Tensor, k_max: int) -> Tensor:
    """Slots allotted to each document: ``floor(n_tokens / ratio)``, at least 1, at most k_max."""
    k = torch.floor(n_tokens.float() / torch.as_tensor(ratio, device=n_tokens.device).float())
    return k.clamp(min=1, max=k_max).long()


def slot_mask_from_counts(counts: Tensor, k_max: int) -> Tensor:
    """``[B, k_max]`` bool mask that is True for the first ``counts[b]`` slots of each row."""
    return torch.arange(k_max, device=counts.device)[None, :] < counts[:, None]


class _Block(nn.Module):
    """Pre-norm block: slots cross-attend to segments, self-attend, then a feed-forward layer."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln_q, self.ln_kv, self.ln_s, self.ln_f = (nn.LayerNorm(d) for _ in range(4))
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, z: Tensor, u: Tensor, u_valid: Tensor, z_valid: Tensor) -> Tensor:
        q, kv = self.ln_q(z), self.ln_kv(u)
        z = z + self.cross(q, kv, kv, key_padding_mask=~u_valid, need_weights=False)[0]
        s = self.ln_s(z)
        z = z + self.self_attn(s, s, s, key_padding_mask=~z_valid, need_weights=False)[0]
        return z + self.ffn(self.ln_f(z))


class FlatSlotWriter(nn.Module):
    """Perceiver-style flat writer: ``k_max`` learned queries, the first ``K_b`` used per document.

    This is the "flat latent slots" baseline (ICAE/Q-Former style) the graph model is compared
    against. A learned embedding of ``log(ratio)`` is added to the queries so one network can serve
    several compression ratios.
    """

    def __init__(self, d: int, k_max: int, layers: int = 2, heads: int = 8):
        super().__init__()
        self.k_max = k_max
        self.queries = nn.Parameter(torch.randn(k_max, d) * d**-0.5)
        self.ratio_embed = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(_Block(d, heads) for _ in range(layers))
        self.out_norm = nn.LayerNorm(d)

    def forward(
        self, u: Tensor, u_valid: Tensor, counts: Tensor, ratio: Tensor
    ) -> tuple[Tensor, Tensor, dict]:
        """Return slots ``[B, K, d]`` and a ``[B, K]`` validity mask, with ``K = counts.max()``."""
        k = int(counts.max())
        z_valid = slot_mask_from_counts(counts, k)
        r = torch.log(ratio.float()).view(-1, 1, 1)
        z = self.queries[None, :k] + self.ratio_embed(r)
        for block in self.blocks:
            z = block(z, u, u_valid, z_valid)
        return self.out_norm(z) * z_valid.unsqueeze(-1), z_valid, {}
