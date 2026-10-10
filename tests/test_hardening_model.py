import numpy as np
import pytest
import torch

from gistgraph.config import load_config
from gistgraph.eval.probes import segment_paragraphs
from gistgraph.llm.frozen import pick_dtype
from gistgraph.model.compressor import GraphCompressor
from gistgraph.model.edges import EdgeInducer, random_adjacency


def test_gumbel_exp_edges_has_gradient():
    torch.manual_seed(0)
    edges = EdgeInducer(16, relations=2, rank=8, sparsifier="gumbel_topk", max_deg=3)
    z = torch.randn(2, 6, 16)
    _, aux = edges(z, torch.ones(2, 6, dtype=torch.bool))
    aux["exp_edges"].sum().backward()
    assert edges.null.weight.grad is not None
    assert edges.null.weight.grad.abs().sum() > 0


def test_random_adjacency_generator_is_reproducible():
    valid = torch.ones(2, 7, dtype=torch.bool)
    a = random_adjacency(valid, 3, 4, torch.Generator().manual_seed(5))
    b = random_adjacency(valid, 3, 4, torch.Generator().manual_seed(5))
    assert torch.equal(a, b)


def _graph(mode: str):
    cfg = load_config(
        None,
        [
            "compressor.type=graph",
            "compressor.encoder.d=32",
            "compressor.encoder.layers=1",
            "compressor.edges.enabled=true",
            f"compressor.edges.mode={mode}",
            "compressor.gnn.enabled=true",
            "data.max_ctx_tokens=24",
        ],
    )
    return GraphCompressor(16, 1.0, 12, cfg)


def test_random_graph_control_is_deterministic_across_calls():
    torch.manual_seed(0)
    comp = _graph("random").eval()
    h = torch.randn(2, 20, 16)
    mask = torch.ones(2, 20, dtype=torch.bool)
    with torch.no_grad():
        a1 = comp(h, mask, 4.0).aux["adj"]
        torch.rand(10)  # disturb the global RNG
        a2 = comp(h, mask, 4.0).aux["adj"]
    assert torch.equal(a1, a2)


def test_unknown_edge_mode_raises_in_model():
    cfg = load_config(None, ["compressor.type=graph", "compressor.encoder.d=32"])
    cfg.compressor.edges.mode = "bogus"
    with pytest.raises(ValueError):
        GraphCompressor(16, 1.0, 12, cfg)


def test_frozen_dtype_allowlist():
    assert pick_dtype("float16") is torch.float16
    assert pick_dtype("bfloat16") is torch.bfloat16
    for bad in ("Tensor", "manual_seed", "int8"):
        with pytest.raises(ValueError):
            pick_dtype(bad)


def test_segment_paragraphs_majority():
    token_para = np.array([0, 0, 1, 1, 1, 2, 2])
    assert segment_paragraphs(token_para, 1, 3).tolist() == [0, 0, 1]
    assert segment_paragraphs(token_para, 3, 10).tolist() == [0, 1, 2]


def test_edges_cannot_collapse_when_no_edge_dominates():
    torch.manual_seed(0)
    ind = EdgeInducer(32, 2, rank=8).train()
    torch.nn.init.constant_(ind.null.bias, 1e3)  # push every node towards "no edge"
    z = torch.randn(2, 10, 32) * 20
    adj, _ = ind(z, torch.ones(2, 10, dtype=torch.bool))
    assert adj.sum((1, 3)).min() > 0.04  # each node keeps a share of its best edge
    adj.sum().backward()
    assert ind.src.weight.grad.abs().sum() > 0 and ind.dst.weight.grad.abs().sum() > 0
