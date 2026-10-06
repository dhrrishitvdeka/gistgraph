"""Truncation baselines: keep a prefix, or a prefix plus a suffix."""

from __future__ import annotations

from gistgraph.baselines.base import Tokenizer, token_budget
from gistgraph.data.schema import Example


class Truncation:
    """Keep ``budget`` tokens from the head (``mode='head'``) or half from each end.

    The head+tail variant follows the common LongBench practice of keeping both ends of a long
    input, since important content is often at the start or the end.
    """

    query_aware = False

    def __init__(self, mode: str = "head"):
        if mode not in ("head", "headtail"):
            raise ValueError(f"unknown truncation mode {mode!r}")
        self.mode = mode
        self.name = f"truncation_{mode}"

    def compress(self, ex: Example, ratio: float, tokenizer: Tokenizer) -> str:
        ids = tokenizer.encode(ex.context)
        budget = token_budget(len(ids), ratio)
        if budget >= len(ids):
            return ex.context
        if self.mode == "head":
            return tokenizer.decode(ids[:budget])
        head = budget - budget // 2
        tail = budget // 2
        kept = ids[:head] + (ids[-tail:] if tail else [])
        return tokenizer.decode(kept)
