import pytest
import torch

from gistgraph.config import load_config
from gistgraph.model.compressor import build_compressor
from gistgraph.model.encoder import SegmentEncoder
from gistgraph.model.routing import RoutedWriter
from gistgraph.model.streaming import MergeGate, StreamState, stream_update

D, D_IN = 16, 16


def _chunk(b=1, m=10, lens=None, seed=0):
    torch.manual_seed(seed)
    u = torch.randn(b, m, D)
    valid = torch.ones(b, m, dtype=torch.bool)
    if lens:
        valid = torch.arange(m)[None, :] < torch.tensor(lens)[:, None]
    pos = torch.arange(m)[None, :].float() / m
    return u, valid, pos.expand(b, m)


def _writer():
    torch.manual_seed(1)
    return RoutedWriter(D, k_max=20, top_k=2, layers=1, noise=0.0).eval()


def test_merge_gate_starts_as_mass_weighted_average():
    gate = MergeGate(D)
    z_old, z_new = torch.randn(1, 5, D), torch.randn(1, 5, D)
    m_old, m_new = (
        torch.tensor([[3.0, 1.0, 2.0, 5.0, 0.5]]),
        torch.tensor([[1.0, 1.0, 6.0, 5.0, 0.5]]),
    )
    alpha = gate(z_old, z_new, m_old, m_new).squeeze(-1)
    assert torch.allclose(alpha, m_new / (m_old + m_new), atol=1e-4)


def test_merge_gate_correction_is_trainable():
    gate = MergeGate(D)
    z = torch.randn(1, 4, D)
    out = gate(z, z, torch.ones(1, 4), torch.ones(1, 4))
    out.sum().backward()
    assert gate.net[-1].weight.grad.abs().sum() > 0


def test_first_chunk_becomes_the_state():
    w = _writer()
    u, valid, pos = _chunk()
    counts = torch.tensor([4])
    state, lb, z_valid = stream_update(w, None, None, u, valid, pos, counts, torch.tensor([3.0]))
    assert state.z.shape == (1, 4, D) and z_valid.sum().item() == 4 and torch.isfinite(lb)
    assert state.mass.sum().item() == pytest.approx(10.0, abs=1e-4)  # each segment writes weight 1


def test_second_chunk_merges_with_mass_weighted_average_and_adds_fresh_nodes():
    w = _writer()
    r = torch.tensor([3.0])
    u1, v1, p1 = _chunk(seed=0)
    s1, _, _ = stream_update(w, None, None, u1, v1, p1, torch.tensor([3]), r)
    u2, v2, p2 = _chunk(seed=1)
    s2, _, zv = stream_update(w, None, s1, u2, v2, p2, torch.tensor([5]), r)
    assert s2.z.shape[1] == 5 and zv.sum().item() == 5
    keys = w.node_keys(5, r)
    z_c, _, m_c, _ = w.route_write(u2, v2, keys, zv)
    m_old = torch.cat([s1.mass, torch.zeros(1, 2)], 1)
    z_old = torch.cat([s1.z, torch.zeros(1, 2, D)], 1)
    for k in range(5):
        if m_c[0, k] > 0 and m_old[0, k] > 0:
            expect = (m_old[0, k] * z_old[0, k] + m_c[0, k] * z_c[0, k]) / (m_old[0, k] + m_c[0, k])
            assert torch.allclose(s2.z[0, k], expect, atol=1e-4)
        elif m_c[0, k] > 0:
            assert torch.allclose(s2.z[0, k], z_c[0, k], atol=1e-5)  # fresh node
        else:
            assert torch.allclose(s2.z[0, k], z_old[0, k], atol=1e-6)  # untouched
    assert s2.mass.sum().item() == pytest.approx(20.0, abs=1e-3)  # mass is conserved


def test_empty_chunk_leaves_state_unchanged():
    w = _writer()
    r = torch.tensor([3.0])
    u1, v1, p1 = _chunk()
    s1, _, _ = stream_update(w, None, None, u1, v1, p1, torch.tensor([3]), r)
    u2, _, p2 = _chunk(seed=3)
    empty = torch.zeros(1, 10, dtype=torch.bool)
    s2, _, _ = stream_update(w, None, s1, u2, empty, p2, torch.tensor([3]), r)
    assert torch.allclose(s2.z, s1.z) and torch.allclose(s2.mass, s1.mass)


def test_learned_merge_gate_is_used_when_given():
    w = _writer()
    r = torch.tensor([3.0])
    u1, v1, p1 = _chunk(seed=0)
    s1, _, _ = stream_update(w, None, None, u1, v1, p1, torch.tensor([3]), r)
    u2, v2, p2 = _chunk(seed=1)
    mean_state, _, _ = stream_update(w, None, s1, u2, v2, p2, torch.tensor([3]), r)
    gate = MergeGate(D)
    with torch.no_grad():
        gate.net[-1].bias.fill_(8.0)  # push towards overwriting
    over, _, _ = stream_update(w, gate, s1, u2, v2, p2, torch.tensor([3]), r)
    assert isinstance(over, StreamState) and not torch.allclose(over.z, mean_state.z)


def test_encoder_position_offset_changes_codes_and_zero_is_default():
    enc = SegmentEncoder(D_IN, 32, layers=1, heads=4).eval()
    h = torch.randn(1, 6, D_IN)
    mask = torch.ones(1, 6, dtype=torch.bool)
    a, _ = enc(h, mask)
    b, _ = enc(h, mask, pos_offset=0)
    c, _ = enc(h, mask, pos_offset=7)
    assert torch.allclose(a, b) and not torch.allclose(a, c)


# ---------------- compressor level ----------------


def _cfg(adaptive=False, streaming=True, **extra):
    over = [
        "compressor.type=routed",
        f"compressor.writer.adaptive={str(adaptive).lower()}",
        "compressor.writer.capacity_factor=2.0",
        "compressor.encoder.d=32",
        "compressor.encoder.layers=1",
        f"compressor.streaming.enabled={str(streaming).lower()}",
        "data.max_ctx_tokens=64",
        "train.ratios=[2,4]",
    ] + [f"{k}={v}" for k, v in extra.items()]
    return load_config(overrides=over)


def _batch():
    torch.manual_seed(0)
    h = torch.randn(2, 24, D_IN)
    mask = torch.zeros(2, 24, dtype=torch.bool)
    mask[0, :24] = True
    mask[1, :13] = True  # the second document ends inside the second chunk of three
    return h, mask


@pytest.mark.parametrize("chunks", [2, 3, 4])
def test_streaming_ends_with_the_same_node_budget_as_one_shot(chunks):
    comp = build_compressor(_cfg(), D_IN, 0.5).eval()
    h, mask = _batch()
    one_shot = comp(h, mask, 4.0)
    streamed = comp(h, mask, 4.0, chunks=chunks)
    assert streamed.counts.tolist() == one_shot.counts.tolist() == [6, 3]
    assert torch.isfinite(streamed.embeds).all()


def test_streaming_conserves_write_mass():
    comp = build_compressor(_cfg(), D_IN, 0.5).eval()
    h, mask = _batch()
    mem = comp(h, mask, 4.0, chunks=3)
    assert torch.allclose(mem.aux["mass"].sum(1), mask.sum(1).float(), atol=1e-3)


def test_chunks_one_is_the_default_path():
    comp = build_compressor(_cfg(), D_IN, 0.5).eval()
    h, mask = _batch()
    assert torch.equal(comp(h, mask, 4.0).embeds, comp(h, mask, 4.0, chunks=1).embeds)


def test_streaming_works_without_a_learned_merge_gate():
    comp = build_compressor(_cfg(streaming=False), D_IN, 0.5).eval()
    assert comp.merge is None  # zero-shot streaming falls back to the running average
    mem = comp(*_batch(), 4.0, chunks=2)
    assert mem.counts.tolist() == [6, 3]


def test_streaming_with_adaptive_gates_and_gradients():
    comp = build_compressor(_cfg(adaptive=True), D_IN, 0.5).train()
    mem = comp(*_batch(), 4.0, chunks=2)
    assert "exp_active" in mem.aux and mem.aux["n_tokens"].tolist() == [24.0, 13.0]
    (mem.embeds.sum() + mem.aux["lb"] + mem.aux["exp_active"].sum()).backward()
    assert comp.merge.net[0].weight.grad is not None
    assert comp.encoder.inp[1].weight.grad.abs().sum() > 0
    assert comp.writer.route.weight.grad.abs().sum() > 0


def test_graph_model_supports_streaming():
    cfg = _cfg(
        adaptive=True,
        **{
            "compressor.type": "graph",
            "compressor.edges.enabled": "true",
            "compressor.edges.rank": 8,
            "compressor.edges.relations": 2,
            "compressor.gnn.enabled": "true",
        },
    )
    comp = build_compressor(cfg, D_IN, 0.5).eval()
    mem = comp(*_batch(), 4.0, chunks=2)
    assert mem.aux["adj"].shape[1] == 2 and torch.isfinite(mem.embeds).all()


def test_flat_compressor_rejects_streaming():
    cfg = load_config(overrides=["compressor.type=flat", "data.max_ctx_tokens=64"])
    comp = build_compressor(cfg, D_IN, 0.5)
    with pytest.raises(ValueError):
        comp(*_batch(), 4.0, chunks=2)
