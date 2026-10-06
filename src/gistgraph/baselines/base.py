"""Interface shared by all non-learned (text-in, text-out) compressors."""

from __future__ import annotations

import math
from typing import Protocol

from gistgraph.data.schema import Example


class Tokenizer(Protocol):
    """The two methods we need. Hugging Face tokenizers satisfy this."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class TextCompressor(Protocol):
    """Turns an example's context into a shorter text.

    ``query_aware`` says whether the question is used. Learned compressors in this project are
    query-agnostic, so query-aware baselines are reported in a separate group.
    """

    name: str
    query_aware: bool

    def compress(self, ex: Example, ratio: float, tokenizer: Tokenizer) -> str: ...


def token_budget(n_tokens: int, ratio: float) -> int:
    """Tokens to keep for a target compression ``ratio`` (original / compressed)."""
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1, got {ratio}")
    return max(1, math.floor(n_tokens / ratio))
