import subprocess
import sys

import pytest
import torch

from gistgraph.api import Compressed, GistGraph
from gistgraph.cli import main
from gistgraph.config import load_config, save_config
from gistgraph.llm.frozen import FrozenLM
from gistgraph.model.compressor import build_compressor
from gistgraph.model.projector import mean_embedding_norm

CONTEXT = "The Eiffel Tower was built in 1889. It is located in Paris, France. " * 2


def _cfg():
    return load_config(
        overrides=[
            "compressor.encoder.d=32",
            "compressor.encoder.layers=1",
            "data.max_ctx_tokens=256",
            "train.ratios=[2,4]",
            "eval.ratios=[4]",
            "train.amp=false",
            "llm.feature_layer=1",
        ]
    )


@pytest.fixture
def gg(lm):
    cfg = _cfg()
    torch.manual_seed(0)
    norm = mean_embedding_norm(lm.model.get_input_embeddings().weight)
    return GistGraph(cfg, lm, build_compressor(cfg, lm.d_model, norm))


def test_save_load_roundtrip(gg, lm, tmp_path):
    gg.save_pretrained(tmp_path / "m")
    assert {"config.json", "compressor.safetensors", "README.md"} <= {
        p.name for p in (tmp_path / "m").iterdir()
    }
    assert "library_name: gistgraph" in (tmp_path / "m" / "README.md").read_text("utf-8")
    other = GistGraph.from_pretrained(tmp_path / "m", lm=lm)
    a, b = gg.compressor.state_dict(), other.compressor.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    m1, m2 = gg.compress(CONTEXT), other.compress(CONTEXT)
    assert torch.equal(m1.embeds, m2.embeds)


def test_compress_ratio(gg):
    mem = gg.compress(CONTEXT, ratio=4)
    assert isinstance(mem, Compressed)
    assert mem.embeds.shape == (mem.n_nodes, gg.lm.d_model)
    assert mem.n_tokens == len(CONTEXT)  # char tokenizer
    assert 2 <= mem.ratio <= 8


def test_compress_truncates_long_context(gg):
    with pytest.warns(UserWarning, match="truncated"):
        mem = gg.compress("x" * 400)
    assert mem.n_tokens == 256


def test_answer_with_memory_and_context(gg):
    mem = gg.compress(CONTEXT)
    assert isinstance(gg.answer("When?", memory=mem, max_new_tokens=4), str)
    assert isinstance(gg.answer("When?", context=CONTEXT, max_new_tokens=4), str)
    out = gg.answer_batch(["When?", "Where?"], [CONTEXT, CONTEXT[:40]], max_new_tokens=3)
    assert len(out) == 2 and all(isinstance(s, str) for s in out)


def test_answer_needs_exactly_one_source(gg):
    with pytest.raises(ValueError):
        gg.answer("q")
    with pytest.raises(ValueError):
        gg.answer("q", memory=gg.compress(CONTEXT), context=CONTEXT)


def test_ratio_below_training_min_raises(gg):
    with pytest.raises(ValueError, match="smallest trained ratio"):
        gg.compress(CONTEXT, ratio=1.5)


def test_cli_ask_and_export(gg, lm, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(FrozenLM, "from_pretrained", classmethod(lambda cls, *a, **k: lm))
    run = tmp_path / "run"
    run.mkdir()
    save_config(gg.cfg, run / "config.resolved.yaml")
    torch.save(
        {"state_dict": gg.compressor.state_dict(), "config_hash": "x"}, run / "compressor.pt"
    )

    main(["export", str(run), str(tmp_path / "out")])
    assert (tmp_path / "out" / "compressor.safetensors").exists()
    capsys.readouterr()

    ctx = tmp_path / "ctx.txt"
    ctx.write_text(CONTEXT, encoding="utf-8")
    for model in (run, tmp_path / "out"):
        main(["ask", str(model), "When?", "--context-file", str(ctx), "--max-new-tokens", "3"])
        main(["ask", str(model), "When?", "--context", CONTEXT, "--verbose"])
    assert "ratio=" in capsys.readouterr().out


def test_import_is_light():
    code = "import sys, gistgraph; assert 'torch' not in sys.modules; print(gistgraph.GistGraph)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "GistGraph" in out.stdout
