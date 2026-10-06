"""Assemble ``docs/results/results.md`` and figures from finished runs."""

from __future__ import annotations

from pathlib import Path

from gistgraph.eval.analysis import (
    budget_markdown,
    comparison_markdown,
    hop_table,
    load_runs,
)
from gistgraph.eval.plots import budget_figure, retention_figure
from gistgraph.eval.report import full_context_table, retention_markdown

# method names used by the shipped experiment configs
ROLES = {
    "flat": "flat",
    "fixed": "routed_fixed",
    "adaptive": "routed_adaptive",
    "graph": "graph",
    "noedges": "graph_noedges",
    "random": "graph_random",
    "gumbel": "graph_gumbel",
    "stream": "graph_stream",
}
HEADLINE_RATIO = 4.0

CAVEATS = """\
## How to read these results

- Every number comes from `results.jsonl` files produced by the commands in the README. F1
  differences are paired by example and shown with a 95% bootstrap interval over examples; `*`
  marks an interval that excludes zero. Intervals reflect example sampling only: with one training
  seed per model they do **not** include training-run variance.
- Methods are compared at the compression ratio they *achieved* (shown in brackets), not the one
  requested. Query-aware baselines (`extractive_bm25`) see the question; the learned compressors
  do not.
- The frozen LLM is a 0.5B model, so absolute F1 is low and retention is relative to its own
  full-context behaviour.
"""


def build_results(run_dirs: list[str | Path], out_dir: str | Path = "docs/results") -> Path:
    """Write the results page and figures. Missing runs simply produce empty sections."""
    rows = load_runs(run_dirs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r = ROLES
    parts = ["# Results\n", CAVEATS]
    if not rows:
        parts.append("\n_No results found in the given run directories._\n")
        page = out / "results.md"
        page.write_text("\n".join(parts), encoding="utf-8")
        return page

    fig = retention_figure(rows, out / "figures" / "retention.png")
    parts += [
        "\n## Retention at every ratio\n",
        f"![F1 retention versus achieved compression]({fig.relative_to(out).as_posix()})\n",
        "### Full-context reference\n",
        full_context_table(rows),
        "\n### F1 retention (achieved ratio in brackets)\n",
        retention_markdown(rows),
    ]

    parts += [
        "\n## H1: does structure beat flat slots?\n",
        "Graph model against each control. The graph model has edges and message passing; "
        "`graph_noedges` has the same parameters but empty messages; `graph_random` has a random "
        "graph of equal degree; `routed_adaptive` has neither.\n",
    ]
    for name in ("flat", "noedges", "random", "adaptive"):
        parts += [
            f"\n**{r['graph']} vs {r[name]}**\n",
            comparison_markdown(rows, r["graph"], r[name]),
        ]

    parts += [
        "\n## H2: adaptive routing versus a fixed rate\n",
        comparison_markdown(rows, r["adaptive"], r["fixed"]),
        f"\nPer-document memory length at {HEADLINE_RATIO:g}x (a fixed-rate model has none):\n",
        budget_markdown(rows, [r["fixed"], r["adaptive"], r["graph"]], HEADLINE_RATIO),
    ]
    budget_figure(
        rows,
        [r["fixed"], r["adaptive"], r["graph"]],
        HEADLINE_RATIO,
        out / "figures" / "budget.png",
    )

    parts += [
        f"\n## H3: where does structure help? ({HEADLINE_RATIO:g}x)\n",
        hop_table(rows, r["graph"], r["flat"], HEADLINE_RATIO),
    ]

    parts += ["\n## H4: incremental versus one-shot\n"]
    for base in (r["graph"], r["adaptive"], r["stream"]):
        for suffix in ("+stream4", "+stream2"):
            parts += [
                f"\n**{base}{suffix} vs {base}**\n",
                comparison_markdown(rows, base + suffix, base),
            ]
    page = out / "results.md"
    page.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return page
