"""Turn evaluation rows into Markdown tables."""

from __future__ import annotations

from gistgraph.eval.harness import aggregate, retention_table


def markdown_table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def full_context_table(rows: list[dict]) -> str:
    stats = aggregate(rows)
    body = [
        [d, str(v["n"]), f"{100 * v['f1']:.1f}", f"{100 * v['em']:.1f}"]
        for (m, _, d), v in sorted(stats.items(), key=lambda kv: str(kv[0]))
        if m == "full"
    ]
    return markdown_table(["dataset", "n", "F1", "EM"], body)


def retention_markdown(rows: list[dict]) -> str:
    """One row per (dataset, method) with F1 retention at each target ratio."""
    table = retention_table(rows)
    ratios = sorted({t["target_ratio"] for t in table})
    cells: dict[tuple, dict] = {}
    for t in table:
        cells.setdefault((t["dataset"], t["method"]), {})[t["target_ratio"]] = t
    body = []
    for (dataset, method), per in sorted(cells.items()):
        row = [dataset, method]
        for r in ratios:
            t = per.get(r)
            row.append(
                "-" if t is None else f"{100 * t['f1_retention']:.0f}% ({t['achieved_ratio']:.1f}x)"
            )
        body.append(row)
    headers = ["dataset", "method"] + [f"{r:g}x" for r in ratios]
    return markdown_table(headers, body)
