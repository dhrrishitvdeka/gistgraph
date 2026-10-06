import pytest
import torch

from gistgraph.baselines.truncation import Truncation
from gistgraph.data.schema import Example
from gistgraph.eval.harness import (
    aggregate,
    read_rows,
    retention_table,
    run_eval,
    run_text_method,
    score_row,
    write_rows,
)
from gistgraph.eval.report import full_context_table, retention_markdown


def _examples(n=5):
    return [
        Example(
            id=f"e{i}",
            context="word " * (20 + 7 * i),
            question="what?",
            answers=["word"],
            dataset="toy",
            hops=1,
        )
        for i in range(n)
    ]


def test_score_row_uses_best_gold():
    ex = Example("1", "c", "q", ["Paris", "paris france"], "toy")
    row = score_row(ex, "paris")
    assert row["em"] == 1.0 and row["f1"] == 1.0


def test_full_context_run_produces_scored_rows_at_ratio_one(lm):
    rows = run_text_method(lm, None, _examples(), 1.0, batch_size=2, max_new_tokens=2)
    assert len(rows) == 5 and {r["method"] for r in rows} == {"full"}
    assert all(r["achieved_ratio"] == pytest.approx(1.0) for r in rows)
    assert all(0.0 <= r["f1"] <= 1.0 for r in rows)


def test_truncation_run_hits_requested_ratio(lm):
    from gistgraph.baselines.base import token_budget

    rows = run_text_method(lm, Truncation("head"), _examples(), 4.0, batch_size=3, max_new_tokens=2)
    for r in rows:
        assert r["n_comp"] == token_budget(r["n_orig"], 4.0)
        assert r["achieved_ratio"] == pytest.approx(r["n_orig"] / r["n_comp"])


def test_soft_prefix_path_counts_prefix_tokens(lm):
    prefix = torch.zeros(6, lm.d_model)
    rows = run_eval(
        lm, _examples(3), lambda ex: (None, prefix, 6), "soft", 4.0, batch_size=2, max_new_tokens=2
    )
    assert all(r["n_comp"] == 6 for r in rows)


def test_batching_does_not_change_predictions(lm):
    exs = _examples(4)
    a = run_text_method(lm, None, exs, 1.0, batch_size=1, max_new_tokens=3)
    b = run_text_method(lm, None, exs, 1.0, batch_size=4, max_new_tokens=3)
    assert {r["id"]: r["pred"] for r in a} == {r["id"]: r["pred"] for r in b}


def _row(method, ratio, ds, f1, em=0.0, achieved=None):
    return {
        "method": method,
        "target_ratio": ratio,
        "dataset": ds,
        "f1": f1,
        "em": em,
        "achieved_ratio": achieved or ratio,
    }


def test_aggregate_and_retention():
    rows = [
        _row("full", 1.0, "d", 0.8),
        _row("full", 1.0, "d", 0.4),
        _row("trunc", 4.0, "d", 0.3),
        _row("trunc", 4.0, "d", 0.3),
    ]
    stats = aggregate(rows)
    assert stats[("full", 1.0, "d")]["f1"] == pytest.approx(0.6)
    (t,) = retention_table(rows)
    assert t["f1_retention"] == pytest.approx(0.5) and t["method"] == "trunc"


def test_rows_roundtrip_and_markdown(tmp_path):
    rows = [
        _row("full", 1.0, "d", 0.6),
        _row("trunc", 2.0, "d", 0.3),
        _row("trunc", 4.0, "d", 0.15),
    ]
    p = tmp_path / "r.jsonl"
    write_rows(rows, p)
    assert read_rows(p) == rows
    md = retention_markdown(rows)
    assert "2x" in md and "4x" in md and "50%" in md and "25%" in md
    assert "60.0" in full_context_table(rows)
