import json

import numpy as np
import pytest
import torch

from helpers import toy_examples

from gistgraph.config import load_config
from gistgraph.data.teacher_cache import build_teacher_cache, load_teacher_cache, make_train_items
from gistgraph.model.compressor import build_compressor
from gistgraph.model.projector import mean_embedding_norm
from gistgraph.train.trainer import Trainer, pad_ids, pad_targets


def _cfg(tmp_path, steps=40, **extra):
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
        "train.warmup=2",
        "train.lr=3e-3",
        "train.ckpt_every=5",
        "train.eval_every=1000",
        "train.recon_tokens=8",
        "loss.recon_anneal_frac=0.3",
        "loss.qa_ce_weight=0.1",
    ] + [f"{k}={v}" for k, v in extra.items()]
    return load_config(overrides=over)


@pytest.fixture
def items(lm, tmp_path):
    exs = toy_examples()
    path = tmp_path / "teacher.npz"
    build_teacher_cache(lm, exs, path, topk=8, max_new_tokens=4, batch_size=2)
    teacher = load_teacher_cache(path)
    return make_train_items(lm.tokenizer, exs, teacher, 64, eos_id=0)


def test_teacher_cache_roundtrip(lm, tmp_path):
    exs = toy_examples(3)
    path = tmp_path / "t.npz"
    build_teacher_cache(lm, exs, path, topk=6, max_new_tokens=3, batch_size=2)
    cache = load_teacher_cache(path)
    assert set(cache) == {"e0", "e1", "e2"}
    ids, idx, logits = cache["e1"]
    assert idx.shape == (len(ids), 6) and logits.shape == idx.shape and 1 <= len(ids) <= 3
    assert np.all(idx[:, 0] == ids)  # top-1 logit is the greedy token
    assert np.all(np.diff(logits.astype(np.float32), axis=1) <= 1e-3)  # sorted descending


def test_make_train_items_adds_gold_with_end_token(lm, items):
    assert len(items) == 4
    assert items[0].gold_ids[-1] == 0 and len(items[0].gold_ids) == 2
    assert items[0].ctx_ids.dtype == np.int64


def test_pad_helpers():
    ids, mask = pad_ids([np.array([5, 6, 7]), np.array([8])], 0, "cpu")
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0]] and mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    tgt, lens = pad_targets([np.array([1, 2]), np.array([3])], "cpu")
    assert tgt.tolist() == [[1, 2], [3, 0]] and lens.tolist() == [2, 1]


def _trainer(lm, cfg, items):
    comp = build_compressor(
        cfg, lm.d_model, mean_embedding_norm(lm.model.get_input_embeddings().weight)
    )
    return Trainer(cfg, lm, comp, items), comp


def test_training_reduces_distillation_loss(lm, items, tmp_path):
    torch.manual_seed(0)
    cfg = _cfg(tmp_path)
    trainer, comp = _trainer(lm, cfg, items)
    before = {k: v.clone() for k, v in comp.state_dict().items()}
    final = trainer.train()
    log = [json.loads(line) for line in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    first, last = log[0]["kd"], log[-1]["kd"]
    assert last < first
    assert final.exists() and trainer.step == 40
    assert any(not torch.equal(before[k], v) for k, v in comp.state_dict().items())
    # the frozen LM is untouched
    assert all(p.grad is None for p in lm.model.parameters())


def test_recon_loss_is_dropped_after_annealing(lm, items, tmp_path):
    cfg = _cfg(tmp_path, steps=100)
    trainer, _ = _trainer(lm, cfg, items)
    trainer.train()
    log = [json.loads(line) for line in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert "recon" in log[0]  # step 10 is inside the first 30% of 100 steps
    assert "recon" not in log[-1]


def test_resume_continues_from_checkpoint(lm, items, tmp_path):
    cfg = _cfg(tmp_path, steps=10)
    t1, _ = _trainer(lm, cfg, items)
    t1.train()
    cfg2 = _cfg(tmp_path, steps=14)
    t2, comp2 = _trainer(lm, cfg2, items)
    t2.train()
    assert t2.step == 14
    steps = [
        json.loads(line)["step"] for line in (tmp_path / "train_log.jsonl").read_text().splitlines()
    ]
    assert steps == sorted(steps) and steps[-1] == 14
