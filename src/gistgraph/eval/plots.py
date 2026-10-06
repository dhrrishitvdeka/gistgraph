"""Figures for the results write-up."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gistgraph.eval.analysis import achieved_ratio, retention_of  # noqa: E402

# Okabe-Ito colour-blind-safe palette; marker shape and line style carry the same information
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
TEXT_METHODS = ("truncation", "extractive", "llmlingua")


def retention_figure(rows: list[dict], path: str | Path, methods: list[str] | None = None) -> Path:
    """Retention versus achieved compression ratio, one panel per dataset."""
    datasets = sorted({r["dataset"] for r in rows})
    methods = methods or sorted({r["method"] for r in rows if r["method"] != "full"})
    fig, axes = plt.subplots(1, max(1, len(datasets)), figsize=(4.6 * max(1, len(datasets)), 3.8))
    axes = [axes] if len(datasets) <= 1 else list(axes)
    for ax, ds in zip(axes, datasets, strict=False):
        for i, m in enumerate(methods):
            ratios = sorted(
                {r["target_ratio"] for r in rows if r["method"] == m and r["dataset"] == ds}
            )
            xs = [achieved_ratio(rows, m, t, ds) for t in ratios]
            ys = [100 * retention_of(rows, m, t, ds) for t in ratios]
            if not xs:
                continue
            dashed = m.startswith(TEXT_METHODS)
            ax.plot(
                xs,
                ys,
                marker=MARKERS[i % len(MARKERS)],
                color=PALETTE[i % len(PALETTE)],
                linestyle="--" if dashed else "-",
                linewidth=1.6,
                markersize=5,
                label=m,
            )
        ax.axhline(90, color="#888888", linewidth=0.8, linestyle=":")
        ax.set_title(ds)
        ax.set_xlabel("achieved compression ratio (x)")
        ax.set_ylabel("F1 retention (%)")
        ax.grid(alpha=0.25)
    axes[-1].legend(fontsize=7, loc="best", frameon=False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def budget_figure(rows: list[dict], methods: list[str], ratio: float, path: str | Path) -> Path:
    """Distribution of per-document compression for each method at one target ratio (H2)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    plotted = False
    for i, m in enumerate(methods):
        frac = [
            r["n_comp"] / max(r["n_orig"], 1)
            for r in rows
            if r["method"] == m and r["target_ratio"] == ratio
        ]
        if frac:
            ax.hist(frac, bins=30, alpha=0.55, color=PALETTE[i % len(PALETTE)], label=m)
            plotted = True
    ax.axvline(1 / ratio, color="#888888", linestyle=":", linewidth=1)
    ax.set_xlabel("memory length / context length")
    ax.set_ylabel("documents")
    if plotted:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
