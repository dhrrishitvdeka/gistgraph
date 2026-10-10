import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from gistgraph.config import load_config
from gistgraph.data.teacher_cache import build_teacher_cache, load_teacher_cache, make_train_items
from gistgraph.eval.harness import read_rows
from gistgraph.model.compressor import build_compressor
from gistgraph.model.projector import mean_embedding_norm
from gistgraph.train.trainer import Trainer, param_groups
from gistgraph.utils.fingerprint import config_hash, load_compressor_state, train_hash
from helpers import toy_examples


def _cfg(tmp_path, steps=4, **extra):
    over = [
        f"out_dir={tmp_path}",
        "seed=0",
        "llm.feature_layer=1",
        "compressor.encoder.d=32",
        "compressor.encoder.layers=1",
        "data.max_ctx_tokens=64",
        "train.ratios=[2]",
        "train.batch=2",
        "train.grad_accum=1",
        f"train.steps={steps}",
        "train.warmup=1",
        "train.ckpt_every=2",
        "train.eval_every=1000",
        "train.recon_tokens=8",
    ] + [f"{k}={v}" for k, v in extra.items()]
    return load_config(overrides=over)


@pytest.fixture
def items(lm, tmp_path):
    exs = toy_examples()
    path = tmp_path / "teacher.npz"
    build_teacher_cache(lm, exs, path, topk=8, max_new_tokens=4, batch_size=2)
    return make_train_items(lm.tokenizer, exs, load_teacher_cache(path), 64, eos_id=0)


def _trainer(lm, cfg, items, **kw):
    comp = build_compressor(
        cfg, lm.d_model, mean_embedding_norm(lm.model.get_input_embeddings().weight)
    )
    return Trainer(cfg, lm, comp, items, **kw), comp


def test_config_hash_ignores_out_dir_and_tracks_changes(tmp_path):
    a, b = _cfg(tmp_path / "a"), _cfg(tmp_path / "b")
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(_cfg(tmp_path, **{"train.lr": 1e-2}))
    assert train_hash(a) == train_hash(_cfg(tmp_path, steps=9))  # longer runs may resume


def test_resume_refuses_changed_config_unless_forced(lm, items, tmp_path):
    _trainer(lm, _cfg(tmp_path, steps=2), items)[0].train()
    changed = _cfg(tmp_path, steps=4, **{"train.lr": 1e-2})
    with pytest.raises(RuntimeError, match="config hash"):
        _trainer(lm, changed, items)[0].train()
    t, _ = _trainer(lm, changed, items, force=True)
    t.train()
    assert t.step == 4


def test_compressor_file_carries_hash_and_loads(lm, items, tmp_path):
    cfg = _cfg(tmp_path, steps=2)
    t, comp = _trainer(lm, cfg, items)
    final = t.train()
    payload = torch.load(final, weights_only=True)
    assert payload["config_hash"] == train_hash(cfg) and payload["step"] == 2
    comp.load_state_dict(load_compressor_state(final, "cpu"))
    raw = tmp_path / "raw.pt"
    torch.save(comp.state_dict(), raw)
    assert load_compressor_state(raw).keys() == comp.state_dict().keys()
    assert not list(Path(tmp_path).glob("*.tmp"))  # atomic writes leave no temp files
    assert "rng" in torch.load(tmp_path / "ckpt.pt", weights_only=True)


def test_nan_step_is_skipped_and_weights_stay_finite(lm, items, tmp_path):
    t, comp = _trainer(lm, _cfg(tmp_path, steps=3), items)
    real, calls = t._losses, {"n": 0}

    def flaky(*a, **kw):
        out = real(*a, **kw)
        calls["n"] += 1
        if calls["n"] == 2:
            out["total"] = out["total"] * float("nan")
        return out

    t._losses = flaky
    t.train()
    assert t.step == 3 and calls["n"] == 3  # the bad batch is skipped, not retried
    assert all(torch.isfinite(p).all() for p in comp.parameters())
    log = [json.loads(x) for x in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert any(r.get("skipped") for r in log)


def test_persistent_nan_aborts(lm, items, tmp_path):
    t, _ = _trainer(lm, _cfg(tmp_path, steps=50), items)
    real = t._losses

    def broken(*a, **kw):
        out = real(*a, **kw)
        out["total"] = out["total"] * math.inf
        return out

    t._losses = broken
    with pytest.raises(RuntimeError, match="consecutive"):
        t.train()


def test_save_refuses_non_finite_weights(lm, items, tmp_path):
    t, comp = _trainer(lm, _cfg(tmp_path), items)
    with torch.no_grad():
        next(comp.parameters()).fill_(float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        t._save()


def test_empty_items_and_param_groups(lm, tmp_path):
    with pytest.raises(ValueError):
        _trainer(lm, _cfg(tmp_path), [])
    lin = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.LayerNorm(3))
    decay, no_decay = param_groups(lin, 0.01)
    assert len(decay["params"]) == 1 and len(no_decay["params"]) == 3


def test_teacher_cache_is_atomic_and_roundtrips(lm, tmp_path, capsys):
    exs = toy_examples(3)
    path = tmp_path / "t.npz"
    build_teacher_cache(lm, exs, path, topk=4, max_new_tokens=3, batch_size=2)
    assert [p.name for p in tmp_path.iterdir()] == ["t.npz"]
    cache = load_teacher_cache(path)
    assert set(cache) == {"e0", "e1", "e2"}
    items = make_train_items(lm.tokenizer, toy_examples(4), cache, 64, eos_id=0)
    assert len(items) == 3 and "1 examples have no teacher output" in capsys.readouterr().out


def test_read_rows_skips_torn_last_line(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": ', encoding="utf-8")
    with pytest.warns(UserWarning):
        assert read_rows(p) == [{"a": 1}, {"a": 2}]
    p.write_text('{"a": \n{"a": 2}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_rows(p)


def _run_experiments():
    path = Path(__file__).parents[1] / "scripts" / "run_experiments.py"
    spec = importlib.util.spec_from_file_location("run_experiments", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_experiments_collects_failures_and_exits_nonzero(tmp_path, monkeypatch):
    mod = _run_experiments()
    done, todo = tmp_path / "done", tmp_path / "todo"
    done.mkdir()
    (done / "compressor.pt").write_bytes(b"")
    cfgs = []
    for d in (done, todo):
        c = tmp_path / f"{d.name}.yaml"
        c.write_text(f"out_dir: {d.as_posix()}\ncompressor:\n  type: routed\n", encoding="utf-8")
        cfgs.append(c)
    calls = []

    def fake(cmd):
        calls.append(cmd)
        return 1 if "train" in cmd else 0

    failures = mod.run_all(cfgs, "cpu", 4, [], call=fake)
    assert [f[1] for f in failures] == ["train"]
    assert sum("eval" in c for c in calls) == 1  # the trained run is still stream-evaluated
    monkeypatch.setattr(mod, "run_all", lambda *a, **k: failures)
    monkeypatch.setattr("sys.argv", ["run_experiments.py", str(cfgs[0])])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert e.value.code == 1
    assert mod.pid_alive(__import__("os").getpid())
