"""Dataset loaders. Parsing is separated from downloading so parsers can be unit tested offline."""

from __future__ import annotations

import json
import re
import zipfile

from gistgraph.data.schema import Example

HF_IDS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor"),
    "2wiki": ("framolfese/2WikiMultihopQA", None),
    "squad": ("rajpurkar/squad", None),
}
LONGBENCH_FILES = {
    "longbench_hotpotqa": "data/hotpotqa.jsonl",
    "longbench_2wiki": "data/2wikimqa.jsonl",
    "longbench_musique": "data/musique.jsonl",
}
CONTEXT_SEP = "\n\n"


def _wiki_paragraphs(ctx: dict) -> list[str]:
    """HotpotQA / 2Wiki context ``{'title': [...], 'sentences': [[...], ...]}`` to paragraphs."""
    return [
        f"{title}\n{' '.join(s.strip() for s in sents)}"
        for title, sents in zip(ctx["title"], ctx["sentences"], strict=True)
    ]


def _supporting(ctx: dict, supporting_facts: dict) -> list[int]:
    wanted = set(supporting_facts["title"])
    return [i for i, t in enumerate(ctx["title"]) if t in wanted]


def parse_hotpotqa(row: dict) -> Example:
    paras = _wiki_paragraphs(row["context"])
    return Example(
        id=row["id"],
        context=CONTEXT_SEP.join(paras),
        question=row["question"],
        answers=[row["answer"]],
        dataset="hotpotqa",
        paragraphs=paras,
        supporting=_supporting(row["context"], row["supporting_facts"]),
        hops=2,
    )


def parse_2wiki(row: dict) -> Example:
    paras = _wiki_paragraphs(row["context"])
    return Example(
        id=row["id"],
        context=CONTEXT_SEP.join(paras),
        question=row["question"],
        answers=[row["answer"]],
        dataset="2wiki",
        paragraphs=paras,
        supporting=_supporting(row["context"], row["supporting_facts"]),
        hops=max(2, len(row.get("evidences") or [])),
    )


def parse_squad(row: dict) -> Example:
    answers = list(dict.fromkeys(row["answers"]["text"]))  # unique, order kept
    return Example(
        id=row["id"],
        context=row["context"],
        question=row["question"],
        answers=answers,
        dataset="squad",
        paragraphs=[row["context"]],
        supporting=[0],
        hops=1,
    )


_PASSAGE = re.compile(r"(?m)^Passage \d+:\n")


def parse_longbench(row: dict, name: str) -> Example:
    paras = [p.strip() for p in _PASSAGE.split(row["context"]) if p.strip()]
    return Example(
        id=row["_id"],
        context=CONTEXT_SEP.join(paras),
        question=row["input"],
        answers=list(row["answers"]),
        dataset=name,
        paragraphs=paras,
        hops=2,
    )


_PARSERS = {"hotpotqa": parse_hotpotqa, "2wiki": parse_2wiki, "squad": parse_squad}


def load_examples(name: str, split: str = "validation", n: int | None = None, seed: int = 0):
    """Load ``n`` examples (a seeded random subset) of dataset ``name``.

    LongBench has only a test split, so ``split`` is ignored for it.
    """
    if name in LONGBENCH_FILES:
        return _load_longbench(name, n, seed)
    if name not in HF_IDS:
        raise KeyError(
            f"unknown dataset {name!r}; choose from {sorted(HF_IDS) + sorted(LONGBENCH_FILES)}"
        )
    from datasets import load_dataset

    repo, cfg = HF_IDS[name]
    ds = load_dataset(repo, cfg, split=split)
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    parse = _PARSERS[name]
    return [parse(row) for row in ds]


def _load_longbench(name: str, n: int | None, seed: int) -> list[Example]:
    import random

    from huggingface_hub import hf_hub_download

    path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    with zipfile.ZipFile(path) as z, z.open(LONGBENCH_FILES[name]) as f:
        rows = [json.loads(line) for line in f]
    random.Random(seed).shuffle(rows)
    if n is not None:
        rows = rows[:n]
    return [parse_longbench(r, name) for r in rows]


def fit_context(ex: Example, tokenizer, max_tokens: int) -> Example:
    """Cut the context to ``max_tokens`` (keeping the head) so every method sees the same input.

    Paragraph and supporting-fact bookkeeping is dropped for cut examples, because the cut
    can fall inside a paragraph.
    """
    ids = tokenizer.encode(ex.context, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return ex
    cut = tokenizer.decode(ids[:max_tokens])
    return Example(
        id=ex.id,
        context=cut,
        question=ex.question,
        answers=ex.answers,
        dataset=ex.dataset,
        paragraphs=[cut],
        supporting=[],
        hops=ex.hops,
    )
