"""Evaluation harness: run a context-compression method, generate answers, score them.

Every method produces one row per example with the prediction, EM, F1 and the *achieved*
compression ratio measured in the frozen LLM's tokens, so methods are compared at the ratio they
really reached, not the one they were asked for.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from gistgraph.baselines.base import TextCompressor
from gistgraph.data.schema import Example
from gistgraph.eval.metrics import best_over_golds, exact_match, f1_score, retention

Row = dict


def score_row(ex: Example, prediction: str) -> dict:
    return {
        "id": ex.id,
        "dataset": ex.dataset,
        "hops": ex.hops,
        "pred": prediction,
        "gold": ex.answers,
        "em": best_over_golds(exact_match, prediction, ex.answers),
        "f1": best_over_golds(f1_score, prediction, ex.answers),
    }


def _batches(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_eval(
    lm,
    examples: list[Example],
    prepare: Callable[[Example], tuple[str | None, object, int]],
    method: str,
    target_ratio: float,
    batch_size: int = 8,
    max_new_tokens: int = 32,
) -> list[Row]:
    """Generate and score answers.

    ``prepare(ex)`` returns ``(context_text, soft_prefix, n_compressed_tokens)``. Exactly one of
    the first two is used: a text context for text methods, or a ``[K, d]`` embedding prefix for
    learned compressors (the text is then ``None``).
    """
    rows: list[Row] = []
    prepared = [(ex, *prepare(ex)) for ex in examples]
    prepared.sort(key=lambda t: len(t[0].context))  # similar lengths batch together
    for batch in _batches(prepared, batch_size):
        embeds, mask = lm.build_inputs(
            [ex.question for ex, *_ in batch],
            [prefix for _, _, prefix, _ in batch],
            contexts=[text for _, text, _, _ in batch],
        )
        preds = lm.generate(embeds, mask, max_new_tokens=max_new_tokens)
        for (ex, _, _, n_comp), pred in zip(batch, preds, strict=True):
            n_orig = len(lm.tokenizer.encode(ex.context, add_special_tokens=False))
            row = score_row(ex, pred.split("\n")[0].strip())
            row.update(
                method=method,
                target_ratio=target_ratio,
                n_orig=n_orig,
                n_comp=n_comp,
                achieved_ratio=n_orig / max(n_comp, 1),
            )
            rows.append(row)
    return rows


def run_text_method(
    lm,
    method: TextCompressor | None,
    examples: list[Example],
    ratio: float,
    **kwargs,
) -> list[Row]:
    """Evaluate a text-in/text-out baseline. ``method=None`` is the full-context reference."""
    tok = lm.tokenizer

    def prepare(ex: Example):
        text = ex.context if method is None else method.compress(ex, ratio, tok)
        return text, None, len(tok.encode(text, add_special_tokens=False))

    name = "full" if method is None else method.name
    return run_eval(lm, examples, prepare, name, 1.0 if method is None else ratio, **kwargs)


def write_rows(rows: list[Row], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_rows(path: str | Path) -> list[Row]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def aggregate(rows: list[Row], by: tuple[str, ...] = ("method", "target_ratio", "dataset")):
    """Mean EM/F1/achieved ratio per group. Returns ``{group_key: stats}``."""
    groups: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in by)].append(r)
    out = {}
    for key, rs in groups.items():
        out[key] = {
            "n": len(rs),
            "em": float(np.mean([r["em"] for r in rs])),
            "f1": float(np.mean([r["f1"] for r in rs])),
            "achieved_ratio": float(np.mean([r["achieved_ratio"] for r in rs])),
        }
    return out


def retention_table(rows: list[Row]) -> list[dict]:
    """F1 retention of each (method, ratio, dataset) relative to the full-context F1."""
    stats = aggregate(rows)
    full = {k[2]: v["f1"] for k, v in stats.items() if k[0] == "full"}
    table = []
    for (method, ratio, dataset), v in sorted(stats.items(), key=lambda kv: str(kv[0])):
        if method == "full" or dataset not in full:
            continue
        table.append(
            {
                "method": method,
                "target_ratio": ratio,
                "dataset": dataset,
                "n": v["n"],
                "f1": v["f1"],
                "em": v["em"],
                "achieved_ratio": v["achieved_ratio"],
                "f1_retention": retention(v["f1"], full[dataset]),
            }
        )
    return table
