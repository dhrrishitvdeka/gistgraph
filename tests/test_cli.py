from pathlib import Path

import pytest

from gistgraph.cli import REPRODUCE, main
from gistgraph.config import load_config

ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("milestone", sorted(REPRODUCE))
def test_every_reproduce_config_exists_and_loads(milestone):
    command, configs = REPRODUCE[milestone]
    assert command in ("baselines", "train") and configs
    for rel in configs:
        assert (ROOT / rel).exists(), rel
        load_config(ROOT / rel)


def test_each_trained_config_has_a_distinct_name_and_output_dir():
    cfgs = [
        load_config(ROOT / rel)
        for cmd, configs in REPRODUCE.values()
        if cmd == "train"
        for rel in configs
    ]
    assert len({c.name for c in cfgs}) == len(cfgs)
    assert len({c.out_dir for c in cfgs}) == len(cfgs)


def test_unknown_milestone_is_rejected():
    with pytest.raises(SystemExit):
        main(["reproduce", "m9"])


def test_report_command_runs_on_empty_directory(tmp_path):
    main(["report", str(tmp_path / "nothing"), "--out", str(tmp_path / "out")])
    assert "No results found" in (tmp_path / "out" / "results.md").read_text(encoding="utf-8")
