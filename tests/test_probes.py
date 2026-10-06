import numpy as np
import pytest
import torch

from gistgraph.config import load_config
from gistgraph.data.schema import Example
from gistgraph.eval.probes import (
    edge_enrichment,
    node_home_paragraph,
    paragraph_token_map,
    probe_model,
    summarise,
)
from gistgraph.model.compressor import build_compressor


class OffsetEnc(dict):
    """Minimal stand-in for a Hugging Face encoding: item and attribute access."""

    def __getattr__(self, name):
        return self[name]


class WordTok:
    """One token per character with exact offsets, so paragraph boundaries are checkable."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        enc = OffsetEnc(input_ids=[[ord(c) % 90 + 5 for c in text]])
        if return_offsets_mapping:
            enc["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return enc


def _example():
    paras = ["aaaa", "bbbb", "cccc"]
    return Example(
        id="e",
        context="\n\n".join(paras),
        question="q",
        answers=["a"],
        dataset="hotpotqa",
        paragraphs=paras,
        supporting=[0, 1],
    )


def test_paragraph_token_map_assigns_tokens_to_paragraphs():
    ex = _example()
    para = paragraph_token_map(WordTok(), ex)
    assert para.tolist() == [0] * 4 + [0, 0] + [1] * 4 + [1, 1] + [2] * 4  # separators join left
    ex.paragraphs = ["only one"]
    assert paragraph_token_map(WordTok(), ex) is None
    cut = _example()
    cut.context = cut.context[:5]  # a cut context no longer matches its paragraphs
    assert paragraph_token_map(WordTok(), cut) is None


def test_node_home_paragraph_follows_write_mass():
    token_para = np.array([0, 0, 1, 1, 2])
    write = np.zeros((5, 3))
    write[0, 0] = write[1, 0] = 1.0  # node 0 reads paragraph 0
    write[2, 1] = write[3, 1] = write[4, 1] = 1.0  # node 1 mostly paragraph 1
    home, has = node_home_paragraph(write, token_para, 3)
    assert home[:2].tolist() == [0, 1] and has.tolist() == [True, True, False]


def _graph(edges, k=6, r=2):
    adj = np.zeros((r, k, k))
    for i, j in edges:
        adj[0, i, j] = 1.0
    return adj


HOME = np.array([0, 0, 1, 1, 2, 2])  # nodes 0,1 -> paragraph 0; 2,3 -> 1; 4,5 -> 2 (distractor)
USABLE = np.ones(6, dtype=bool)


def test_edges_between_supporting_paragraphs_are_enriched():
    out = edge_enrichment(_graph([(0, 2), (3, 1)]), HOME, USABLE, [0, 1])
    assert out["bridge_fraction"] == 1.0
    assert out["expected_bridge"] == pytest.approx(8 / 30)
    assert out["enrichment"] == pytest.approx(30 / 8)


def test_edges_to_distractors_are_not_enriched():
    out = edge_enrichment(_graph([(0, 4), (2, 5)]), HOME, USABLE, [0, 1])
    assert out["bridge_fraction"] == 0.0 and out["enrichment"] == 0.0


def test_within_paragraph_enrichment():
    out = edge_enrichment(_graph([(0, 1), (2, 3)]), HOME, USABLE, [0, 1])
    assert out["within_enrichment"] == pytest.approx(1.0 / (6 / 30))  # 6 same-paragraph pairs


def test_inactive_nodes_are_excluded():
    usable = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    out = edge_enrichment(_graph([(0, 4)]), HOME, usable, [0, 1])
    assert np.isnan(out["enrichment"])  # the only edge touches an inactive node


def test_random_edges_score_about_one():
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(400):
        pairs = {tuple(rng.choice(6, 2, replace=False)) for _ in range(5)}
        vals.append(edge_enrichment(_graph(pairs), HOME, USABLE, [0, 1])["enrichment"])
    assert np.nanmean(vals) == pytest.approx(1.0, abs=0.15)


def test_summarise_handles_nan_and_reports_fraction_above_random():
    rows = [
        {"enrichment": 2.0, "within_enrichment": 1.0},
        {"enrichment": 0.5, "within_enrichment": 1.0},
        {"enrichment": float("nan"), "within_enrichment": float("nan")},
    ]
    s = summarise(rows)
    assert s["n_examples"] == 3 and s["enrichment"]["n"] == 2
    assert s["enrichment"]["mean"] == pytest.approx(1.25) and s["fraction_above_random"] == 0.5


def test_probe_model_runs_on_a_graph_compressor(lm, monkeypatch):
    cfg = load_config(
        overrides=[
            "compressor.type=graph",
            "compressor.writer.adaptive=true",
            "compressor.encoder.d=32",
            "compressor.encoder.layers=1",
            "compressor.edges.enabled=true",
            "compressor.edges.rank=8",
            "compressor.edges.relations=2",
            "compressor.gnn.enabled=true",
            "data.max_ctx_tokens=64",
            "train.ratios=[2]",
            "llm.feature_layer=1",
        ]
    )
    comp = build_compressor(cfg, lm.d_model, 0.5)
    tok = WordTok()
    tok.pad_token_id = tok.eos_token_id = 0
    monkeypatch.setattr(lm, "tokenizer", tok)
    monkeypatch.setattr(
        lm, "encode_text", lambda text: torch.tensor([ord(c) % 90 + 5 for c in text])
    )
    summary = probe_model(lm, comp, cfg, [_example()], 2.0)
    assert summary["n_examples"] == 1 and "mean" in summary["enrichment"]
