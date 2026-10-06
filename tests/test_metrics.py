import math

import pytest

from gistgraph.eval.metrics import (
    best_over_golds,
    exact_match,
    f1_score,
    normalize_answer,
    retention,
)


def test_normalize_strips_articles_punct_case():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"


def test_exact_match():
    assert exact_match("the Paris.", "Paris") == 1.0
    assert exact_match("Paris, France", "Paris") == 0.0


def test_f1_partial_overlap():
    # pred tokens: [paris, france]; gold: [paris] -> P=0.5 R=1 -> F1=2/3
    assert f1_score("Paris, France", "Paris") == pytest.approx(2 / 3)


def test_f1_no_overlap_and_empty():
    assert f1_score("london", "paris") == 0.0
    assert f1_score("", "") == 1.0
    assert f1_score("", "paris") == 0.0


def test_f1_counts_repeated_tokens_once_per_gold_occurrence():
    assert f1_score("x x b", "b c") == pytest.approx(0.4)  # overlap 1, P=1/3, R=1/2


def test_best_over_golds():
    assert best_over_golds(f1_score, "paris", ["london", "paris"]) == 1.0
    assert best_over_golds(f1_score, "paris", []) == 0.0


def test_retention():
    assert retention(0.45, 0.5) == pytest.approx(0.9)
    assert math.isnan(retention(0.1, 0.0))
