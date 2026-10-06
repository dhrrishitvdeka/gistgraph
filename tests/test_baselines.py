import numpy as np
import pytest

from gistgraph.baselines.base import token_budget
from gistgraph.baselines.extractive import (
    Extractive,
    bm25_scores,
    split_sentences,
    textrank_scores,
)
from gistgraph.baselines.truncation import Truncation


def n_tokens(tok, text):
    return len(tok.encode(text))


def test_token_budget():
    assert token_budget(100, 4) == 25
    assert token_budget(3, 5) == 1  # never below one token
    with pytest.raises(ValueError):
        token_budget(10, 0.5)


@pytest.mark.parametrize("mode", ["head", "headtail"])
@pytest.mark.parametrize("ratio", [2, 3, 5])
def test_truncation_respects_budget(example, tok, mode, ratio):
    out = Truncation(mode).compress(example, ratio, tok)
    total = n_tokens(tok, example.context)
    assert n_tokens(tok, out) == token_budget(total, ratio)


def test_truncation_head_is_prefix_and_headtail_keeps_end(example, tok):
    head = Truncation("head").compress(example, 2, tok)
    assert example.context.startswith(head)
    ht = Truncation("headtail").compress(example, 2, tok)
    assert ht.split()[-1] == example.context.split()[-1]


def test_truncation_ratio_one_is_identity(example, tok):
    assert Truncation().compress(example, 1, tok) == example.context


def test_split_sentences():
    assert split_sentences("A b. C d!\nE f?") == ["A b.", "C d!", "E f?"]


def test_textrank_scores_are_a_distribution_like_vector():
    sents = ["cats chase mice", "mice fear cats", "stocks fell sharply today"]
    s = textrank_scores(sents)
    assert s.shape == (3,) and np.all(s > 0)
    assert s[0] > s[2] and s[1] > s[2]  # related sentences are more central


def test_bm25_prefers_matching_sentence(example):
    sents = split_sentences(example.context)
    s = bm25_scores(sents, example.question)
    assert "Eiffel" in sents[int(np.argmax(s))]


@pytest.mark.parametrize("scorer", ["lead", "textrank", "bm25"])
@pytest.mark.parametrize("ratio", [2, 3, 4])
def test_extractive_within_budget_and_original_order(example, tok, scorer, ratio):
    out = Extractive(scorer).compress(example, ratio, tok)
    total = sum(n_tokens(tok, s) for s in split_sentences(example.context))
    assert 0 < n_tokens(tok, out) <= token_budget(total, ratio)
    # every kept sentence appears in order in the original
    pos = [example.context.index(s) for s in split_sentences(out)]
    assert pos == sorted(pos)


def test_extractive_bm25_keeps_answer_sentence(example, tok):
    out = Extractive("bm25").compress(example, 4, tok)
    assert "1889" in out


def test_query_awareness_flags():
    assert Extractive("bm25").query_aware is True
    assert Extractive("textrank").query_aware is False
    assert Truncation().query_aware is False
