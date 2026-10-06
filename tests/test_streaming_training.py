import json

import pytest
import torch

from gistgraph.config import load_config
from gistgraph.data.teacher_cache import build_teacher_cache, load_teacher_cache, make_train_items
from gistgraph.eval.harness import run_eval
from gistgraph.eval.run_learned import make_prepare, method_label
from gistgraph.model.compressor import build_compressor
from gistgraph.model.projector import mean_embedding_norm
from gistgraph.train.trainer import Trainer
from helpers import toy_examples


def _cfg(tmp_path, chunks=3, **extra):
    over = [
        f"out_dir={tmp_path}",
        "name=stream",
        "compressor.type=routed",
        "compressor.writer.adaptive=true",
        "compressor.writer.capacity_factor=2.0",
        "compressor.encoder.d=32",
        "compressor.encoder.layers=1",
        "compressor.streaming.enabled=true",
        f"compressor.streaming.chunks={chunks}",
        "llm.feature_layer=1",
        "data.max_ctx_tokens=64",
        "train.ratios=[2,4]",
        "train.batch=2",
        "train.grad_accum=1",
        "train.steps=12",
        "train.warmup=2",
        "train.lr=3e-3",
        "train.ckpt_every=100",
        "train.eval_every=1000",
        "train.recon_tokens=8",
        "loss.qa_ce_weight=0.0",
        "loss.rate_start_frac=0.1",
    ] + [f"{k}={v}" for k, v in extra.items()]
    return load_config(overrides=over)


@pytest.fixture
def items(lm, tmp_path):
    exs = toy_examples()
    path = tmp_path / "teacher.npz"
    build_teacher_cache(lm, exs, path, topk=8, max_new_tokens=4, batch_size=2)
    return make_train_items(lm.tokenizer, exs, load_teacher_cache(path), 64, eos_id=0)


def test_streaming_training_updates_the_merge_gate(lm, items, tmp_path):
    torch.manual_seed(0)
    cfg = _cfg(tmp_path)
    norm = mean_embedding_norm(lm.model.get_input_embeddings().weight)
    comp = build_compressor(cfg, lm.d_model, norm)
    before = comp.merge.net[-1].weight.clone()
    trainer = Trainer(cfg, lm, comp, items)
    assert trainer._chunks() == 3
    trainer.train()
    log = [json.loads(x) for x in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert "kd" in log[-1] and "lb" in log[-1]
    assert not torch.equal(before, comp.merge.net[-1].weight)  # the gate received gradient


def test_method_label_marks_streamed_evaluation(tmp_path):
    assert method_label(_cfg(tmp_path)) == "stream"
    assert method_label(_cfg(tmp_path, **{"eval.stream_chunks": 4})) == "stream+stream4"


def test_one_shot_and_streamed_eval_share_a_trained_model(lm, tmp_path):
    cfg_one = _cfg(tmp_path)
    cfg_str = _cfg(tmp_path, **{"eval.stream_chunks": 3})
    norm = mean_embedding_norm(lm.model.get_input_embeddings().weight)
    comp = build_compressor(cfg_one, lm.d_model, norm).eval()
    exs = toy_examples(3)
    kw = {"batch_size": 2, "max_new_tokens": 2}
    a = run_eval(lm, exs, make_prepare(lm, comp, cfg_one, 4.0), "x", 4.0, **kw)
    b = run_eval(lm, exs, make_prepare(lm, comp, cfg_str, 4.0), "y", 4.0, **kw)
    assert [r["n_comp"] for r in a] and len(a) == len(b) == 3
    assert all(r["achieved_ratio"] > 1 for r in b)
