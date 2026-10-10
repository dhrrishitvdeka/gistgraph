"""Global seeding so that runs are reproducible from a config."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    ``deterministic`` additionally asks PyTorch for deterministic kernels. That can be slower,
    so experiment configs may turn it off for long training runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
