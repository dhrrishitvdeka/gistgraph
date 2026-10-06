import numpy as np
import pytest

from gistgraph.eval.analysis import (
    achieved_ratio,
    budget_markdown,
    compare,
    comparison_markdown,
    hop_table,
    load_runs,
    mean_ci,
    node_budget_stats,
    paired_difference,
    retention_of,
)
from gistgraph.eval.harness import write_rows
from gistgraph.eval.results import build_results


def _row(method, ratio, ds, i, f1, n_orig=400, n_comp=100, hops=None):
    return {
        "id": f"{ds}-{i}",
        "dataset": ds,
        "hops": hops,
        "method": method,
        "target_ratio": ratio,
        "f1": f1,
        "em": float(f1 == 1.0),
        "n_orig": n_orig,
        "n_comp": n_comp,
        "achieved_ratio": n_orig / n_comp,
        "pred": "",
        "gold": [],
    }


def _synthetic(n=60, seed=0, gain=0.15):
    """full ~0.6; flat ~0.3; graph = flat + gain (paired by example)."""
    rng = np.random.default_rng(seed)
    rows = []
    for ds, hops in (("hotpotqa", 2), ("squad", 1)):
        for i in range(n):
            base = float(np.clip(rng.normal(0.3, 0.1), 0, 1))
            rows.append(_row("full", 1.0, ds, i, float(np.clip(base + 0.3, 0, 1)), hops=hops))
            rows.append(_row("flat", 4.0, ds, i, base, hops=hops))
            g = gain if ds == "hotpotqa" else 0.0
            rows.append(_row("graph", 4.0, ds, i, float(np.clip(base + g, 0, 1)), hops=hops))
    return rows


def test_mean_ci_contains_mean_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small = mean_ci(rng.normal(0, 1, 20))
    large = mean_ci(rng.normal(0, 1, 2000))
    assert small[1] < small[0] < small[2]
    assert (large[2] - large[1]) < (small[2] - small[1])
    assert np.isnan(mean_ci([])[0])


def test_paired_difference_detects_a_real_gain_and_ignores_noise():
    a = {i: 0.5 + 0.1 for i in range(50)}
    b = {i: 0.5 for i in range(50)}
    d = paired_difference(a, b)
    assert d["diff"] == pytest.approx(10.0) and d["significant"] and d["n"] == 50
    rng = np.random.default_rng(1)
    noise_a = {i: float(v) for i, v in enumerate(rng.normal(0.5, 0.1, 80))}
    noise_b = {i: float(v) for i, v in enumerate(rng.normal(0.5, 0.1, 80))}
    assert not paired_difference(noise_a, noise_b)["significant"]


def test_paired_difference_uses_only_shared_examples():
    d = paired_difference({1: 1.0, 2: 1.0, 3: 0.0}, {2: 0.0, 3: 0.0, 4: 1.0})
    assert d["n"] == 2 and d["diff"] == pytest.approx(50.0)


def test_retention_and_achieved_ratio():
    rows = [_row("full", 1.0, "d", 0, 0.8), _row("m", 4.0, "d", 0, 0.4, n_orig=300, n_comp=100)]
    assert retention_of(rows, "m", 4.0, "d") == pytest.approx(0.5)
    assert achieved_ratio(rows, "m", 4.0, "d") == pytest.approx(3.0)
    assert np.isnan(retention_of(rows, "m", 2.0, "d"))


def test_compare_finds_gain_only_where_it_exists():
    recs = {r["dataset"]: r for r in compare(_synthetic(), "graph", "flat")}
    assert recs["hotpotqa"]["significant"] and recs["hotpotqa"]["diff"] > 10
    assert recs["squad"]["diff"] == pytest.approx(0.0) and not recs["squad"]["significant"]


def test_comparison_markdown_marks_significance():
    md = comparison_markdown(_synthetic(), "graph", "flat")
    hot = next(line for line in md.splitlines() if line.startswith("| hotpotqa"))
    sq = next(line for line in md.splitlines() if line.startswith("| squad"))
    assert hot.count("*") == 1 and "*" not in sq
    assert "_No overlapping" in comparison_markdown(_synthetic(), "graph", "missing")


def test_hop_table_separates_single_and_multi_hop():
    md = hop_table(_synthetic(), "graph", "flat", 4.0)
    rows = {line.split("|")[1].strip(): line for line in md.splitlines()[2:]}
    assert "*" in rows["multi-hop"] and "*" not in rows["single-hop"]
    assert "2-hop (hotpotqa)" in rows


def test_node_budget_stats_and_markdown():
    rows = [
        _row("adapt", 4.0, "d", i, 0.1, n_orig=400, n_comp=c) for i, c in enumerate([60, 100, 140])
    ]
    s = node_budget_stats(rows, "adapt", 4.0, "d")
    assert s["mean_fraction"] == pytest.approx(0.25) and s["std_fraction"] > 0
    assert node_budget_stats(rows, "none", 4.0, "d") == {}
    assert "adapt" in budget_markdown(rows, ["adapt"], 4.0)


def test_load_runs_dedupes_shared_reference_rows(tmp_path):
    full = [_row("full", 1.0, "d", 0, 0.7)]
    write_rows(full + [_row("a", 4.0, "d", 0, 0.3)], tmp_path / "a" / "results.jsonl")
    write_rows(full + [_row("b", 4.0, "d", 0, 0.4)], tmp_path / "b" / "results.jsonl")
    rows = load_runs([tmp_path / "a", tmp_path / "b", tmp_path / "missing"])
    assert sorted(r["method"] for r in rows) == ["a", "b", "full"]


def test_build_results_writes_page_and_figures(tmp_path):
    write_rows(_synthetic(), tmp_path / "run" / "results.jsonl")
    page = build_results([tmp_path / "run"], tmp_path / "out")
    text = page.read_text(encoding="utf-8")
    assert "## H1" in text and "## H4" in text and "How to read these results" in text
    assert (tmp_path / "out" / "figures" / "retention.png").stat().st_size > 1000


def test_build_results_with_no_runs_is_graceful(tmp_path):
    page = build_results([tmp_path / "nothing"], tmp_path / "out")
    assert "No results found" in page.read_text(encoding="utf-8")
