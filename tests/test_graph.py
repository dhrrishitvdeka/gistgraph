import pytest
import torch

from gistgraph.config import load_config
from gistgraph.model.compressor import GraphCompressor, build_compressor
from gistgraph.model.edges import (
    EdgeInducer,
    cap_degree,
    entmax15,
    random_adjacency,
    rw_positional_encoding,
)
from gistgraph.model.gnn import RelationalMP

D, R = 16, 3


def _nodes(b=2, k=10, lens=(10, 7), seed=0):
    torch.manual_seed(seed)
    z = torch.randn(b, k, D)
    valid = torch.zeros(b, k, dtype=torch.bool)
    for i, n in enumerate(lens):
        valid[i, :n] = True
    return z, valid


# ---------------- entmax ----------------


def _entmax_bisect(x, iters=100):
    """Reference solution of entmax-1.5: p_i = max(x_i / 2 - tau, 0)^2 with sum(p) = 1."""
    x = x / 2
    lo, hi = x.max() - 1, x.max()
    for _ in range(iters):
        tau = (lo + hi) / 2
        total = (x - tau).clamp(min=0).pow(2).sum()
        lo, hi = (tau, hi) if total > 1 else (lo, tau)
    return (x - (lo + hi) / 2).clamp(min=0) ** 2


def test_entmax_matches_reference_and_sums_to_one():
    torch.manual_seed(0)
    for _ in range(5):
        x = torch.randn(9) * 2
        p = entmax15(x)
        assert p.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert torch.allclose(p, _entmax_bisect(x.double()).float(), atol=1e-4)


def test_entmax_gives_exact_zeros_unlike_softmax():
    x = torch.tensor([3.0, 2.5, 0.0, -1.0, -4.0])
    p = entmax15(x)
    assert (p == 0).sum() >= 2 and (x.softmax(0) > 0).all()


def test_entmax_shift_invariant_and_batched():
    x = torch.randn(3, 4, 7)
    assert torch.allclose(entmax15(x), entmax15(x + 100.0), atol=1e-5)
    assert torch.allclose(entmax15(x).sum(-1), torch.ones(3, 4), atol=1e-5)


def test_entmax_gradient_is_finite_and_nonzero():
    x = torch.randn(4, 8, requires_grad=True)
    (entmax15(x) * torch.randn(4, 8)).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_entmax_ignores_masked_large_negatives():
    x = torch.tensor([1.0, 0.5, -1e4, -1e4])
    p = entmax15(x)
    assert p[2:].sum() == 0 and p.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_cap_degree_never_exceeds_max_with_ties():
    w = torch.full((2, 40), 0.025)  # all tied
    out = cap_degree(w, 5)
    assert ((out > 0).sum(-1) == 5).all() and torch.allclose(out.sum(-1), w.sum(-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA autocast")
def test_edge_induction_stays_float32_under_autocast():
    z, valid = _nodes()
    ind = EdgeInducer(D, R, rank=8).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        adj, _ = ind(z.cuda(), valid.cuda())
    assert adj.dtype == torch.float32


def test_cap_degree_keeps_top_and_total_mass():
    w = torch.tensor([[0.4, 0.3, 0.2, 0.1, 0.0]])
    out = cap_degree(w, 2)
    assert (out > 0).sum().item() == 2 and out.sum().item() == pytest.approx(w.sum().item())


# ---------------- edge induction ----------------


@pytest.mark.parametrize("sparsifier", ["entmax15", "gumbel_topk"])
def test_edges_structure(sparsifier):
    z, valid = _nodes()
    ind = EdgeInducer(D, R, rank=8, sparsifier=sparsifier, max_deg=3).eval()
    adj, aux = ind(z, valid)
    assert adj.shape == (2, R, 10, 10) and torch.isfinite(adj).all()
    assert torch.all(torch.diagonal(adj, dim1=-2, dim2=-1) == 0)  # no self loops
    assert torch.all(adj[1, :, 7:, :] == 0) and torch.all(adj[1, :, :, 7:] == 0)  # padding
    assert torch.all(adj[0, :, :, :].sum((0, 2)) <= 1 + 1e-5)  # rows sum to at most one
    nnz = (adj > 0).sum((1, 3))  # edges read by each node
    assert nnz.max() <= 3
    assert {"exp_edges", "mean_degree", "edge_entropy", "edge_mass"} <= set(aux)


def test_entmax_edges_are_sparse_with_exact_zeros():
    z, valid = _nodes(k=24, lens=(24, 24))
    ind = EdgeInducer(D, 4, rank=8, max_deg=8).eval()
    adj, aux = ind(z * 3, valid)
    frac_nonzero = (adj > 0).float().mean().item()
    assert frac_nonzero < 0.05  # out of 4 * 24 * 24 possible entries
    assert aux["mean_degree"].max() <= 8


@pytest.mark.parametrize("sparsifier", ["entmax15", "gumbel_topk"])
def test_gradients_flow_through_the_edge_sparsifier(sparsifier):
    z, valid = _nodes()
    ind = EdgeInducer(D, R, rank=8, sparsifier=sparsifier, max_deg=3).train()
    adj, aux = ind(z, valid)
    values = torch.randn_like(adj)
    ((adj * values).sum() + aux["exp_edges"].sum()).backward()
    for name, p in ind.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert ind.src.weight.grad.abs().sum() > 0 and ind.rel.grad.abs().sum() > 0


def test_edge_induction_is_permutation_equivariant():
    z, valid = _nodes(b=1, k=9, lens=(9,), seed=3)
    ind = EdgeInducer(D, R, rank=8, max_deg=4).eval()
    perm = torch.randperm(9, generator=torch.Generator().manual_seed(1))
    adj, _ = ind(z, valid)
    adj_p, _ = ind(z[:, perm], valid[:, perm])
    assert torch.allclose(adj[:, :, perm][:, :, :, perm], adj_p, atol=1e-5)


def test_edges_depend_on_content():
    z, valid = _nodes(b=1, k=12, lens=(12,))
    ind = EdgeInducer(D, R, rank=8).eval()
    a1, _ = ind(z, valid)
    a2, _ = ind(torch.randn_like(z), valid)
    assert not torch.allclose(a1, a2)


# ---------------- message passing ----------------


def _random_adj(b, k, valid):
    torch.manual_seed(5)
    adj = torch.rand(b, R, k, k)
    adj = adj * (torch.rand_like(adj) < 0.3) * valid[:, None, :, None] * valid[:, None, None, :]
    return adj / adj.sum((1, 3), keepdim=True).clamp(min=1)


def _mp(layers=2):
    mp = RelationalMP(D, R, layers)
    for o in mp.out:  # undo the identity-friendly zero init so messages matter
        torch.nn.init.normal_(o.weight, std=0.3)
    return mp


def test_message_passing_is_permutation_equivariant():
    z, valid = _nodes(b=1, k=9, lens=(9,))
    adj = _random_adj(1, 9, valid)
    mp = _mp().eval()
    perm = torch.randperm(9, generator=torch.Generator().manual_seed(2))
    out = mp(z, adj, valid)
    out_p = mp(z[:, perm], adj[:, :, perm][:, :, :, perm], valid[:, perm])
    assert torch.allclose(out[:, perm], out_p, atol=1e-5)


def test_message_passing_zero_on_invalid_and_ignores_padding():
    z, valid = _nodes()
    adj = _random_adj(2, 10, valid)
    mp = _mp().eval()
    out = mp(z, adj, valid)
    assert torch.all(out[~valid] == 0)
    z2 = z.clone()
    z2[~valid] = 50.0
    assert torch.allclose(mp(z2, adj, valid)[valid], out[valid], atol=1e-5)


def test_two_layers_reach_two_hops_but_one_layer_does_not():
    # chain: node 0 reads from node 1, node 1 reads from node 2
    k = 4
    valid = torch.ones(1, k, dtype=torch.bool)
    adj = torch.zeros(1, R, k, k)
    adj[0, 0, 0, 1] = 1.0
    adj[0, 0, 1, 2] = 1.0
    z, _ = _nodes(b=1, k=k, lens=(k,))
    z_far = z.clone()
    # change the node two hops from node 0 (a constant shift would be removed by LayerNorm)
    z_far[0, 2] = torch.randn(D, generator=torch.Generator().manual_seed(9)) * 3

    def effect(layers):
        mp = _mp(layers).eval()
        return (mp(z, adj, valid)[0, 0] - mp(z_far, adj, valid)[0, 0]).abs().max().item()

    assert effect(1) < 1e-6 and effect(2) > 1e-3


def test_message_passing_without_edges_is_finite():
    z, valid = _nodes()
    out = _mp().eval()(z, None, valid)
    assert torch.isfinite(out).all()


# ---------------- positional encoding and random control ----------------


def test_rw_positional_encoding_two_cycle():
    adj = torch.zeros(1, 1, 3, 3)
    adj[0, 0, 0, 1] = adj[0, 0, 1, 0] = 1.0  # nodes 0 and 1 point at each other
    pe = rw_positional_encoding(adj, 4)
    assert pe.shape == (1, 3, 4)
    assert pe[0, 0].tolist() == [0.0, 1.0, 0.0, 1.0] and torch.all(pe[0, 2] == 0)


def test_rw_positional_encoding_is_equivariant():
    z, valid = _nodes(b=1, k=8, lens=(8,))
    adj = _random_adj(1, 8, valid)
    perm = torch.randperm(8, generator=torch.Generator().manual_seed(4))
    pe = rw_positional_encoding(adj, 5)
    pe_p = rw_positional_encoding(adj[:, :, perm][:, :, :, perm], 5)
    assert torch.allclose(pe[:, perm], pe_p, atol=1e-6)


def test_random_adjacency_matches_degree_and_respects_masks():
    valid = torch.tensor([[1] * 10, [1] * 6 + [0] * 4], dtype=torch.bool)
    adj = random_adjacency(valid, relations=R, max_deg=4)
    assert adj.shape == (2, R, 10, 10)
    deg = (adj > 0).sum((1, 3))
    assert torch.all(deg[0] == 4) and torch.all(deg[1, :6] == 4) and torch.all(deg[1, 6:] == 0)
    assert torch.allclose(adj.sum((1, 3))[valid], torch.ones(int(valid.sum())))
    assert torch.all(torch.diagonal(adj, dim1=-2, dim2=-1) == 0)
    assert torch.all(adj[1, :, :, 6:] == 0)


# ---------------- graph compressor ----------------

D_IN = 16


def _cfg(**extra):
    over = [
        "compressor.type=graph",
        "compressor.writer.adaptive=true",
        "compressor.writer.capacity_factor=2.0",
        "compressor.encoder.d=32",
        "compressor.encoder.layers=1",
        "compressor.edges.enabled=true",
        "compressor.edges.relations=3",
        "compressor.edges.rank=8",
        "compressor.edges.max_deg=4",
        "compressor.gnn.enabled=true",
        "compressor.gnn.layers=2",
        "data.max_ctx_tokens=64",
        "train.ratios=[2,4]",
    ] + [f"{k}={v}" for k, v in extra.items()]
    return load_config(overrides=over)


def _batch():
    torch.manual_seed(0)
    h = torch.randn(2, 24, D_IN)
    mask = torch.zeros(2, 24, dtype=torch.bool)
    mask[0, :24] = True
    mask[1, :17] = True
    return h, mask


def test_graph_compressor_end_to_end_and_edge_gradients():
    comp = build_compressor(_cfg(), D_IN, 0.5).train()
    assert isinstance(comp, GraphCompressor)
    for o in comp.gnn.out:
        torch.nn.init.normal_(o.weight, std=0.3)
    mem = comp(*_batch(), 4.0)
    assert mem.embeds.shape[-1] == D_IN and mem.embeds.shape[0] == 2
    assert mem.aux["adj"].shape[1] == 3 and "mean_degree" in mem.aux and "exp_edges" in mem.aux
    (mem.embeds.sum() + mem.aux["exp_edges"].sum()).backward()
    assert comp.edges.src.weight.grad.abs().sum() > 0
    assert comp.gnn.rel_weights[0].grad.abs().sum() > 0
    assert comp.projector.pe.weight.grad is not None


def test_edges_off_keeps_gnn_parameters_but_has_no_edge_module():
    comp = build_compressor(_cfg(**{"compressor.edges.enabled": "false"}), D_IN, 0.5).eval()
    assert comp.edges is None and comp.gnn is not None
    mem = comp(*_batch(), 4.0)
    assert "adj" not in mem.aux and "exp_edges" not in mem.aux and torch.isfinite(mem.embeds).all()


def test_random_mode_uses_random_graph_without_edge_parameters():
    comp = build_compressor(_cfg(**{"compressor.edges.mode": "random"}), D_IN, 0.5).eval()
    assert comp.edges is None
    mem = comp(*_batch(), 4.0)
    assert (mem.aux["adj"] > 0).sum((1, 3)).max() <= 4 and "exp_edges" not in mem.aux


def test_graph_without_structure_matches_routed_node_counts():
    cfg = _cfg(**{"compressor.edges.enabled": "false", "compressor.gnn.enabled": "false"})
    comp = build_compressor(cfg, D_IN, 0.5).eval()
    assert comp.edges is None and comp.gnn is None
    assert comp(*_batch(), 4.0).counts.tolist() == [12, 8]


def test_inactive_nodes_are_excluded_from_the_graph():
    comp = build_compressor(_cfg(), D_IN, 0.5).eval()
    with torch.no_grad():
        comp.gate.proj.bias.fill_(-6.0)
    mem = comp(*_batch(), 4.0)
    assert mem.counts.tolist() == [1, 1]  # only the rescued node remains, so there are no edges
    assert mem.aux["adj"].sum() == 0


def test_graph_positional_bias_changes_embeddings():
    # Return probabilities are all zero when the sampled graph has no cycle, so test the
    # projector with a known non-zero encoding instead of relying on the random graph.
    comp = build_compressor(_cfg(), D_IN, 0.5).eval()
    assert comp.pe_steps == 8 and comp.projector.pe is not None
    z = torch.randn(1, 3, 32)
    pe = torch.rand(1, 3, comp.pe_steps) + 0.5
    assert not torch.allclose(comp.projector(z, pe), comp.projector(z, None))
