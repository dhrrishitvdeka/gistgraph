from pathlib import Path

import pytest

from gistgraph.cli import main
from gistgraph.config import load_config

ROOT = Path(__file__).parent.parent
SHIPPED = sorted((ROOT / "configs").rglob("*.yaml"))


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_every_shipped_config_loads_and_validates(path):
    load_config(path)


@pytest.mark.parametrize(
    "override",
    [
        "compressor.edges.mode=randon",
        "compressor.edges.sparsifier=softmax",
        "llm.dtype=int8",
        "compressor.ratio=0",
        "train.ratios=[2,0]",
        "train.grad_accum=0",
        "compressor.writer.top_k=0",
        "compressor.encoder.d=100",
        "compressor.streaming.chunks=0",
        "compressor.encoder.seg_len=0",
        "eval.ratios=[1.5,4]",
    ],
)
def test_bad_values_raise(override):
    with pytest.raises(ValueError):
        load_config(None, [override])


def test_unknown_compressor_type_raises():
    with pytest.raises(KeyError):
        load_config(None, ["compressor.type=grpah"])


def test_seg_len_above_smallest_ratio_warns():
    with pytest.warns(UserWarning, match="seg_len"):
        load_config(None, ["compressor.encoder.seg_len=3"])


@pytest.mark.parametrize("override", ["train.steps=true", "train.lr=false", "llm.grad_ckpt=1"])
def test_bool_int_confusion_is_rejected(override):
    with pytest.raises(TypeError):
        load_config(None, [override])


def test_bad_float_names_the_key():
    with pytest.raises(ValueError, match="train.lr"):
        load_config(None, ["train.lr=fast"])


def test_list_item_types_are_checked():
    with pytest.raises(ValueError, match=r"train.ratios\[1\]"):
        load_config(None, ["train.ratios=[2, abc]"])
    with pytest.raises(TypeError, match=r"train.ratios\[0\]"):
        load_config(None, ["train.ratios=[true]"])
    with pytest.raises(TypeError, match=r"data.eval\[0\]"):
        load_config(None, ["data.eval=[3]"])
    assert load_config(None, ["train.ratios=[2, 4]"]).train.ratios == [2.0, 4.0]


def test_circular_base_is_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("base: b.yaml\nseed: 1\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("base: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_diamond_base_is_not_circular(tmp_path):
    (tmp_path / "root.yaml").write_text("seed: 3\n", encoding="utf-8")
    (tmp_path / "l.yaml").write_text("base: root.yaml\n", encoding="utf-8")
    (tmp_path / "r.yaml").write_text("base: root.yaml\n", encoding="utf-8")
    (tmp_path / "top.yaml").write_text("base: [l.yaml, r.yaml]\n", encoding="utf-8")
    assert load_config(tmp_path / "top.yaml").seed == 3


def test_removed_fields_are_rejected():
    with pytest.raises(KeyError):
        load_config(None, ["data.n_train=10"])
    with pytest.raises(KeyError):
        load_config(None, ["compressor.writer.kind=routed"])


def test_cli_config_error_exits_cleanly(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("compressor: {type: nope}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["train", "--config", str(bad)])
    assert exc.value.code == 2
    assert capsys.readouterr().err.startswith("error: ")


@pytest.mark.parametrize("args", [["--n", "0"], ["--ratio", "-1"], ["--n", "x"]])
def test_cli_probe_rejects_non_positive(args):
    with pytest.raises(SystemExit):
        main(["probe", "--config", "x.yaml", *args])


def test_cli_train_passes_force(monkeypatch):
    import gistgraph.train.run as run

    seen = {}
    monkeypatch.setattr(run, "run_training", lambda cfg, device, **kw: seen.update(kw))
    main(["train", "--config", str(ROOT / "configs/experiments/m2_flat.yaml"), "--force"])
    assert seen == {"force": True}
