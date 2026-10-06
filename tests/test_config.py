import pytest
import yaml

from gistgraph.config import Config, apply_overrides, load_config, save_config


def test_defaults_without_file():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.compressor.ratio == 4.0
    assert cfg.llm.name.startswith("Qwen")


def test_yaml_values_override_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 7\ncompressor:\n  ratio: 3\n  edges: {enabled: true}\n")
    cfg = load_config(p)
    assert cfg.seed == 7
    assert cfg.compressor.ratio == 3.0 and isinstance(cfg.compressor.ratio, float)
    assert cfg.compressor.edges.enabled is True
    assert cfg.compressor.edges.relations == 4  # untouched default


def test_base_inheritance_later_wins(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 1\ntrain: {lr: 0.1, steps: 10}\n")
    (tmp_path / "child.yaml").write_text("base: base.yaml\ntrain: {steps: 99}\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.seed == 1
    assert cfg.train.lr == 0.1
    assert cfg.train.steps == 99


def test_cli_overrides_parse_types():
    cfg = load_config(overrides=["train.lr=1e-4", "compressor.ratio=5", "llm.grad_ckpt=false"])
    assert cfg.train.lr == pytest.approx(1e-4)
    assert cfg.compressor.ratio == 5.0
    assert cfg.llm.grad_ckpt is False


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("compressor: {ratoi: 3}\n")
    with pytest.raises(KeyError, match="ratoi"):
        load_config(p)


def test_wrong_type_raises():
    with pytest.raises(TypeError):
        load_config(overrides=["train.steps=abc"])


def test_override_needs_equals():
    with pytest.raises(ValueError):
        apply_overrides({}, ["train.lr"])


def test_save_roundtrip(tmp_path):
    cfg = load_config(overrides=["seed=5", "compressor.type=graph"])
    out = tmp_path / "run" / "resolved.yaml"
    save_config(cfg, out)
    again = load_config(out)
    assert again == cfg
    assert yaml.safe_load(out.read_text())["seed"] == 5
