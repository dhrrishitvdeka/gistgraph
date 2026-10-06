import json

import pytest
import torch

from gistgraph.config import load_config
from gistgraph.data.teacher_cache import build_teacher_cache, load_teacher_cache, make_train_items
from gistgraph.model.compressor import RoutedCompressor, build_compressor, max_slots
from gistgraph.model.projector import mean_embedding_norm
from gistgraph.train.trainer import Trainer
from helpers import toy_examples

D_IN = 16


def _cfg(adaptive: bool, tmp_path=None, **extra):
    over = [
        "compressor.type=routed",
        f"compressor.writer.adaptive={str(adaptive).lower()}",
        "compressor.writer.top_k=2",
        "compressor.writer.capacity_factor=2.0",
        "compressor.encoder.d=32",
        "compressor.encoder.layers=1",
        "data.max_ctx_tokens=64",
        "train.ratios=[2,4]",
        "llm.feature_layer=1",
    ]
    if tmp_path is not None:
        over += [
            f"out_dir={tmp_path}",
            "train.batch=2",
            "train.grad_accum=1",
            "train.steps=60",
            "train.warmup=2",
            "train.lr=3e-3",
            "train.ckpt_every=20",
            "train.eval_every=1000",
            "train.recon_tokens=8",
            "loss.rate_start_frac=0.1",
            "loss.qa_ce_weight=0.0",
        ]
    over += [f"{k}={v}" for k, v in extra.items()]
    return load_config(overrides=over)


def _batch(b=2, n=24, lens=(24, 17)):
    torch.manual_seed(0)
    h = torch.randn(b, n, D_IN)
    mask = torch.zeros(b, n, dtype=torch.bool)
    for i, ln in enumerate(lens):
        mask[i, :ln] = True
    return h, mask


def test_max_slots_includes_adaptive_pool():
    assert max_slots(_cfg(False)) == 32
    assert max_slots(_cfg(True)) == 64  # capacity factor 2


def test_fixed_routed_gives_exactly_n_over_ratio_nodes():
    comp = build_compressor(_cfg(False), D_IN, 0.5).eval()
    assert isinstance(comp, RoutedCompressor) and comp.gate is None
    h, mask = _batch()
    mem = comp(h, mask, 4.0)
    assert mem.counts.tolist() == [6, 4]
    assert "exp_active" not in mem.aux  # no gates, so no rate statistics to penalise


def test_adaptive_starts_with_pool_open_and_exposes_rate_stats():
    comp = build_compressor(_cfg(True), D_IN, 0.5).eval()
    h, mask = _batch()
    mem = comp(h, mask, 4.0)
    assert mem.counts.tolist() == [12, 8]  # pool = 2 * N / ratio, all gates open at init
    assert mem.aux["exp_active"].shape == (2,) and mem.aux["n_tokens"].tolist() == [24.0, 17.0]


def test_closed_gates_shrink_the_memory_but_never_to_zero():
    comp = build_compressor(_cfg(True), D_IN, 0.5).eval()
    h, mask = _batch()
    with torch.no_grad():
        comp.gate.proj.bias.fill_(-6.0)  # every gate closed
    mem = comp(h, mask, 4.0)
    assert mem.counts.tolist() == [1, 1]  # the best node is reopened


def test_nodes_are_ordered_by_source_position():
    comp = build_compressor(_cfg(False), D_IN, 0.5).eval()
    h, mask = _batch()
    mem = comp(h, mask, 2.0)
    c = mem.aux["centroid"]
    z, zv, aux, _ = comp._nodes(h, mask, 2.0)
    order = torch.where(zv, aux["centroid"], torch.full_like(c, 3.0)).argsort(1)
    assert order.shape == mem.mask.shape  # ordering is a permutation of the nodes
    assert mem.mask[0].tolist() == zv[0].gather(0, order[0]).tolist()


def test_inactive_nodes_contribute_zero_vectors_and_prefix_skips_them():
    comp = build_compressor(_cfg(True), D_IN, 0.5).eval()
    h, mask = _batch()
    mem = comp(h, mask, 4.0)
    assert torch.all(mem.embeds[~mem.mask] == 0)
    assert mem.prefix(1).shape[0] == int(mem.mask[1].sum())


def test_training_mode_gates_are_stochastic_and_differentiable():
    comp = build_compressor(_cfg(True), D_IN, 0.5).train()
    h, mask = _batch()
    mem = comp(h, mask, 4.0)
    frac = (mem.aux["exp_active"] / mem.aux["n_tokens"]).mean()
    (mem.embeds.sum() + frac + mem.aux["lb"]).backward()
    assert comp.gate.proj.bias.grad.abs().sum() > 0
    assert comp.writer.route.weight.grad.abs().sum() > 0


def test_ratio_conditioning_changes_adaptive_pool():
    comp = build_compressor(_cfg(True), D_IN, 0.5).eval()
    h, mask = _batch()
    assert comp(h, mask, 2.0).counts.tolist() == [
        24,
        17,
    ]  # pool = floor(2 * N / 2), capped by k_max
    assert comp(h, mask, 4.0).counts.tolist() == [12, 8]


@pytest.fixture
def items(lm, tmp_path):
    exs = toy_examples()
    path = tmp_path / "teacher.npz"
    build_teacher_cache(lm, exs, path, topk=8, max_new_tokens=4, batch_size=2)
    return make_train_items(lm.tokenizer, exs, load_teacher_cache(path), 64, eos_id=0)


def _trainer(lm, cfg, items):
    norm = mean_embedding_norm(lm.model.get_input_embeddings().weight)
    comp = build_compressor(cfg, lm.d_model, norm)
    return Trainer(cfg, lm, comp, items), comp


def test_adaptive_training_logs_rate_terms_and_updates_multipliers(lm, items, tmp_path):
    torch.manual_seed(0)
    trainer, comp = _trainer(lm, _cfg(True, tmp_path), items)
    trainer.train()
    log = [json.loads(line) for line in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert all(k in log[-1] for k in ("kd", "lb", "frac", "lambda", "rate"))
    assert any(abs(v) > 0 for v in trainer.rate.lam.values())
    # the multipliers survive a checkpoint round trip
    other, _ = _trainer(lm, _cfg(True, tmp_path), items)
    other._resume()
    assert other.rate.lam == trainer.rate.lam


def test_fixed_routed_training_has_no_rate_term(lm, items, tmp_path):
    trainer, _ = _trainer(lm, _cfg(False, tmp_path, **{"train.steps": 20}), items)
    trainer.train()
    log = [json.loads(line) for line in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert "lb" in log[-1] and "rate" not in log[-1]
