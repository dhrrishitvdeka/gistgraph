"""Common example format shared by all datasets."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Example:
    """One QA example: a context to compress, a question, and gold answers.

    ``paragraphs`` keeps the context split into its natural units (e.g. HotpotQA passages) so that
    baselines and probes can work on them. ``context`` is always the concatenation the LLM reads.
    ``supporting`` holds the indices of paragraphs that contain supporting facts, when known.
    """

    id: str
    context: str
    question: str
    answers: list[str]
    dataset: str
    paragraphs: list[str] = field(default_factory=list)
    supporting: list[int] = field(default_factory=list)
    hops: int | None = None  # 1 for single-hop, 2+ for multi-hop, None when unknown
