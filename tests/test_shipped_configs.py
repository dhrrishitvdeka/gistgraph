from pathlib import Path

import pytest

from gistgraph.config import load_config

CONFIGS = sorted((Path(__file__).parent.parent / "configs").rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    load_config(path)
