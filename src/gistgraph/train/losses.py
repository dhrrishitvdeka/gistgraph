"""Training losses and schedules."""

from __future__ import annotations

import math

import torch.nn.functional as nnf
from torch import Tensor


def kd_loss(
    student_logits: Tensor,
    topk_idx: Tensor,
    topk_logits: Tensor,
    valid: Tensor,
    temp: float = 1.0,
) -> Tensor:
    """KL(teacher || student) over the teacher's top-k support, averaged over valid positions.

    The teacher distribution is its top-k logits renormalised (the tail mass is dropped). The
    student's log-probabilities come from a softmax over the *full* vocabulary and are read at the
    same k indices, so the student is still penalised for putting mass outside the teacher's
    support. Scaled by ``temp**2`` as in standard distillation.
    """
    log_p_t = nnf.log_softmax(topk_logits.float() / temp, dim=-1)
    log_p_s = nnf.log_softmax(student_logits.float() / temp, dim=-1).gather(-1, topk_idx)
    kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(-1)  # [B, T]
    return (kl * valid).sum() / valid.sum().clamp(min=1) * temp**2


def ce_loss(logits: Tensor, targets: Tensor, valid: Tensor) -> Tensor:
    """Mean token cross-entropy over valid positions. ``targets``: ``[B, T]`` token ids."""
    nll = nnf.cross_entropy(logits.float().transpose(1, 2), targets, reduction="none")
    return (nll * valid).sum() / valid.sum().clamp(min=1)


def lr_at(step: int, total: int, base_lr: float, warmup: int, floor: float = 0.1) -> float:
    """Linear warm-up then cosine decay to ``floor * base_lr``."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def recon_weight(step: int, total: int, weight: float, anneal_frac: float) -> float:
    """Reconstruction-loss weight: full at the start, linearly annealed to zero."""
    end = max(1.0, anneal_frac * total)
    return weight * max(0.0, 1.0 - step / end)
