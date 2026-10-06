"""Augmented-Lagrangian rate controller: holds the average open-node fraction at ``1 / ratio``."""

from __future__ import annotations


class RateController:
    """One Lagrange multiplier per target ratio.

    With ``g = open_fraction - 1 / ratio`` the penalty added to the loss is
    ``lam * g + rho / 2 * g**2``. After each batch the multiplier ``lam`` moves in the direction of
    the violation: it grows while the model keeps too many nodes open and shrinks (even below
    zero) while it keeps too few. Compared with a fixed penalty weight, this lands the average
    compression on the requested ratio without tuning a weight per ratio.

    The quadratic term matters. Plain dual ascent winds up and oscillates when the model responds
    quickly: on a toy problem the open fraction collapsed to ~0 or swung between ~0 and ~1. The
    augmented term damps the loop so it settles on the target.
    """

    def __init__(self, lr: float = 0.05, rho: float = 10.0, limit: float = 20.0):
        self.lr, self.rho, self.limit = lr, rho, limit
        self.lam: dict[float, float] = {}

    def multiplier(self, ratio: float) -> float:
        return self.lam.get(float(ratio), 0.0)

    def penalty(self, ratio: float, open_fraction):
        """Differentiable penalty term (``open_fraction`` is a tensor with gradient)."""
        g = open_fraction - 1.0 / ratio
        return self.multiplier(ratio) * g + 0.5 * self.rho * g * g

    def update(self, ratio: float, open_fraction: float) -> None:
        lam = self.multiplier(ratio) + self.lr * (open_fraction - 1.0 / ratio)
        self.lam[float(ratio)] = max(-self.limit, min(self.limit, lam))

    def state_dict(self) -> dict:
        return {"lam": dict(self.lam)}

    def load_state_dict(self, state: dict) -> None:
        self.lam = {float(k): float(v) for k, v in state["lam"].items()}
