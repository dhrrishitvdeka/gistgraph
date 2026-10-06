"""Hard-concrete gates (Louizos et al., 2018) for differentiable L0 regularisation."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

BETA, GAMMA, ZETA = 2.0 / 3.0, -0.1, 1.1


class HardConcreteGate(nn.Module):
    """Per-node gate in ``[0, 1]`` that is *exactly* 0 or 1 with non-zero probability.

    ``log_alpha`` (a learned function of the node state) sets how likely the gate is open. During
    training a stochastic sample is used. At evaluation the gate is deterministic. The expected
    number of open gates, ``expected_open``, is differentiable and is what the rate penalty acts
    on.
    """

    def __init__(self, d: int, init_log_alpha: float = 2.0):
        super().__init__()
        self.proj = nn.Linear(d, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, init_log_alpha)

    def log_alpha(self, z: Tensor) -> Tensor:
        return self.proj(z).squeeze(-1)

    def forward(self, z: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``gate [B, K]`` (zero on invalid nodes) and ``expected_open [B]``."""
        la = self.log_alpha(z).float()
        if self.training:
            u = torch.rand_like(la).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + la) / BETA)
        else:
            s = torch.sigmoid(la)
        gate = (s * (ZETA - GAMMA) + GAMMA).clamp(0.0, 1.0) * valid
        p_open = torch.sigmoid(la - BETA * math.log(-GAMMA / ZETA)) * valid
        return gate, p_open.sum(1)
