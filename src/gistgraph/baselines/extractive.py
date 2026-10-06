"""Extractive baselines: score sentences, keep the best ones that fit the token budget."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from gistgraph.baselines.base import Tokenizer, token_budget
from gistgraph.data.schema import Example

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"\w+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def textrank_scores(sentences: list[str], iters: int = 30, damping: float = 0.85) -> np.ndarray:
    """Centrality of each sentence in a TF-IDF cosine-similarity graph (query-agnostic)."""
    n = len(sentences)
    if n == 0:
        return np.zeros(0)
    docs = [Counter(_words(s)) for s in sentences]
    df = Counter(w for d in docs for w in d)
    vocab = {w: i for i, w in enumerate(df)}
    mat = np.zeros((n, len(vocab)))
    for i, d in enumerate(docs):
        for w, c in d.items():
            mat[i, vocab[w]] = c * math.log((1 + n) / (1 + df[w]) + 1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms == 0, 1, norms)
    sim = mat @ mat.T
    np.fill_diagonal(sim, 0.0)
    row = sim.sum(axis=1, keepdims=True)
    trans = np.divide(sim, row, out=np.full_like(sim, 1.0 / n), where=row > 0)
    rank = np.full(n, 1.0 / n)
    for _ in range(iters):
        rank = (1 - damping) / n + damping * (trans.T @ rank)
    return rank


def bm25_scores(sentences: list[str], query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Okapi BM25 of every sentence against the question (query-aware)."""
    docs = [_words(s) for s in sentences]
    n = len(docs)
    if n == 0:
        return np.zeros(0)
    avgdl = max(sum(len(d) for d in docs) / n, 1e-9)
    df = Counter(w for d in docs for w in set(d))
    scores = np.zeros(n)
    for i, d in enumerate(docs):
        tf = Counter(d)
        for q in set(_words(query)):
            if q not in tf:
                continue
            idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
            scores[i] += idf * tf[q] * (k1 + 1) / (tf[q] + k1 * (1 - b + b * len(d) / avgdl))
    return scores


class Extractive:
    """Keep the highest-scoring sentences within the budget, in original order.

    ``scorer``: ``lead`` (earliest first), ``textrank`` (query-agnostic) or ``bm25`` (query-aware,
    scores against the question).
    """

    def __init__(self, scorer: str = "textrank"):
        if scorer not in ("lead", "textrank", "bm25"):
            raise ValueError(f"unknown scorer {scorer!r}")
        self.scorer = scorer
        self.name = f"extractive_{scorer}"
        self.query_aware = scorer == "bm25"

    def _scores(self, sentences: list[str], question: str) -> np.ndarray:
        if self.scorer == "lead":
            return -np.arange(len(sentences), dtype=float)
        if self.scorer == "textrank":
            return textrank_scores(sentences)
        return bm25_scores(sentences, question)

    def compress(self, ex: Example, ratio: float, tokenizer: Tokenizer) -> str:
        sentences = split_sentences(ex.context)
        lengths = [len(tokenizer.encode(s)) for s in sentences]
        budget = token_budget(sum(lengths), ratio)
        order = np.argsort(-self._scores(sentences, ex.question), kind="stable")
        pieces: dict[int, str] = {}
        used = 0
        for i in order:
            i = int(i)
            if used + lengths[i] <= budget:
                pieces[i] = sentences[i]
                used += lengths[i]
        # Fill the leftover budget with a cut of the best sentence that did not fit, so the
        # achieved ratio lands on the target instead of overshooting it.
        left = budget - used
        if left > 0:
            for i in order:
                i = int(i)
                if i not in pieces:
                    pieces[i] = tokenizer.decode(tokenizer.encode(sentences[i])[:left])
                    break
        return " ".join(pieces[i] for i in sorted(pieces))
