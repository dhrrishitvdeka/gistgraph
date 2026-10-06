"""Turn raw result rows into the tables that test the four hypotheses.

All comparisons are paired by example id and carry a bootstrap confidence interval over
examples, so a difference is only called a difference if the interval excludes zero. Scores are in
F1 points (0-100). Retention is ``mean F1 of the method / mean F1 of the full context``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from gistgraph.eval.harness import read_rows
from gistgraph.eval.report import markdown_table

MULTI_HOP = ("hotpotqa", "2wiki", "longbench_hotpotqa", "longbench_2wiki", "longbench_musique")
SINGLE_HOP = ("squad",)


def load_runs(dirs: list[str | Path]) -> list[dict]:
    """Concatenate ``results.jsonl`` from several run directories, dropping duplicate rows.

    Every learned run re-evaluates the full-context reference on the same examples, so rows are
    de-duplicated by (method, ratio, dataset, example id), keeping the first.
    """
    seen, out = set(), []
    for d in dirs:
        path = Path(d) / "results.jsonl"
        if not path.exists():
            continue
        for r in read_rows(path):
            key = (r["method"], r["target_ratio"], r["dataset"], r["id"])
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out


def scores(rows: list[dict], method: str, ratio: float, dataset: str, key: str = "f1") -> dict:
    """``{example_id: score}`` for one (method, ratio, dataset) cell."""
    return {
        r["id"]: r[key]
        for r in rows
        if r["method"] == method and r["target_ratio"] == ratio and r["dataset"] == dataset
    }


def mean_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Mean and bootstrap percentile interval of ``values``."""
    v = np.asarray(list(values), dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(1)
    return (
        float(v.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def paired_difference(a: dict, b: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of ``a - b`` over shared examples (in points) with a bootstrap interval."""
    shared = sorted(set(a) & set(b))
    diffs = [100.0 * (a[i] - b[i]) for i in shared]
    mean, lo, hi = mean_ci(diffs, n_boot, seed)
    return {"n": len(shared), "diff": mean, "lo": lo, "hi": hi, "significant": lo > 0 or hi < 0}


def retention_of(rows, method: str, ratio: float, dataset: str, full: str = "full") -> float:
    num = list(scores(rows, method, ratio, dataset).values())
    den = list(scores(rows, full, 1.0, dataset).values())
    if not num or not den or np.mean(den) == 0:
        return float("nan")
    return float(np.mean(num) / np.mean(den))


def achieved_ratio(rows, method: str, ratio: float, dataset: str) -> float:
    vals = [
        r["achieved_ratio"]
        for r in rows
        if r["method"] == method and r["target_ratio"] == ratio and r["dataset"] == dataset
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _cells(rows) -> dict:
    out = defaultdict(set)
    for r in rows:
        out[r["method"]].add((r["target_ratio"], r["dataset"]))
    return out


def compare(rows, a: str, b: str, key: str = "f1", seed: int = 0) -> list[dict]:
    """Paired comparison of methods ``a`` and ``b`` in every (dataset, ratio) cell both have."""
    cells = _cells(rows)
    out = []
    for ratio, dataset in sorted(cells.get(a, set()) & cells.get(b, set()), key=str):
        d = paired_difference(
            scores(rows, a, ratio, dataset, key), scores(rows, b, ratio, dataset, key), seed=seed
        )
        out.append(
            {
                "dataset": dataset,
                "ratio": ratio,
                **d,
                "ret_a": retention_of(rows, a, ratio, dataset),
                "ret_b": retention_of(rows, b, ratio, dataset),
                "ach_a": achieved_ratio(rows, a, ratio, dataset),
                "ach_b": achieved_ratio(rows, b, ratio, dataset),
            }
        )
    return out


def _fmt_diff(r: dict) -> str:
    mark = "*" if r["significant"] else ""
    return f"{r['diff']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}]{mark}"


def comparison_markdown(rows, a: str, b: str) -> str:
    """Table of ``a - b`` F1 points with 95% intervals; ``*`` marks intervals excluding zero."""
    recs = compare(rows, a, b)
    if not recs:
        return f"_No overlapping results for `{a}` and `{b}` yet._"
    body = [
        [
            r["dataset"],
            f"{r['ratio']:g}x",
            str(r["n"]),
            _fmt_diff(r),
            f"{100 * r['ret_a']:.0f}% ({r['ach_a']:.1f}x)",
            f"{100 * r['ret_b']:.0f}% ({r['ach_b']:.1f}x)",
        ]
        for r in sorted(recs, key=lambda r: (r["dataset"], r["ratio"]))
    ]
    headers = ["dataset", "ratio", "n", f"F1 diff ({a} - {b})", f"{a} retention", f"{b} retention"]
    return markdown_table(headers, body)


def hop_table(rows, a: str, b: str, ratio: float) -> str:
    """H3: advantage of ``a`` over ``b`` split by single-hop, multi-hop and number of hops."""
    by_group: dict[str, list[tuple[float, float]]] = defaultdict(list)
    hops = {(r["dataset"], r["id"]): r.get("hops") for r in rows}
    for ds in {r["dataset"] for r in rows}:
        sa, sb = scores(rows, a, ratio, ds), scores(rows, b, ratio, ds)
        for i in set(sa) & set(sb):
            kind = "single-hop" if ds in SINGLE_HOP else "multi-hop"
            h = hops.get((ds, i))
            groups = [kind, f"{ds}"] + ([f"{h}-hop ({ds})"] if h and ds not in SINGLE_HOP else [])
            for g in groups:
                by_group[g].append((sa[i], sb[i]))
    body = []
    for g, pairs in sorted(by_group.items()):
        d = paired_difference(
            {i: p[0] for i, p in enumerate(pairs)}, {i: p[1] for i, p in enumerate(pairs)}
        )
        body.append([g, str(d["n"]), _fmt_diff(d)])
    if not body:
        return f"_No overlapping results for `{a}` and `{b}` at {ratio:g}x yet._"
    return markdown_table(["group", "n", f"F1 diff ({a} - {b}) at {ratio:g}x"], body)


def node_budget_stats(rows, method: str, ratio: float, dataset: str) -> dict:
    """Spread of the per-document compression, the signature of adaptive allocation (H2)."""
    sel = [
        r
        for r in rows
        if r["method"] == method and r["target_ratio"] == ratio and r["dataset"] == dataset
    ]
    if not sel:
        return {}
    frac = np.array([r["n_comp"] / max(r["n_orig"], 1) for r in sel])
    return {
        "n": len(sel),
        "mean_fraction": float(frac.mean()),
        "std_fraction": float(frac.std()),
        "min_fraction": float(frac.min()),
        "max_fraction": float(frac.max()),
    }


def budget_markdown(rows, methods: list[str], ratio: float) -> str:
    body = []
    for m in methods:
        for ds in sorted({r["dataset"] for r in rows}):
            s = node_budget_stats(rows, m, ratio, ds)
            if s:
                body.append(
                    [
                        m,
                        ds,
                        str(s["n"]),
                        f"{s['mean_fraction']:.3f}",
                        f"{s['std_fraction']:.3f}",
                        f"{s['min_fraction']:.3f}-{s['max_fraction']:.3f}",
                    ]
                )
    if not body:
        return "_No results yet._"
    return markdown_table(["method", "dataset", "n", "mean n_comp/n_orig", "std", "range"], body)
