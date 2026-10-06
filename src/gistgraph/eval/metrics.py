"""SQuAD-style answer normalisation, exact match and token-level F1."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace (official SQuAD recipe)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_over_golds(metric, prediction: str, golds: list[str]) -> float:
    """Score against every acceptable answer and keep the best, as SQuAD does."""
    return max(metric(prediction, g) for g in golds) if golds else 0.0


def retention(compressed_score: float, full_score: float) -> float:
    """Fraction of the uncompressed-context score that is kept."""
    return compressed_score / full_score if full_score > 0 else float("nan")
