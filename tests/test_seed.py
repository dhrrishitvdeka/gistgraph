import random

import numpy as np

from gistgraph.utils.seed import seed_everything


def _draw():
    return random.random(), float(np.random.rand())


def test_same_seed_gives_same_draws():
    seed_everything(123)
    first = _draw()
    seed_everything(123)
    assert _draw() == first


def test_different_seeds_differ():
    seed_everything(1)
    a = _draw()
    seed_everything(2)
    assert _draw() != a
