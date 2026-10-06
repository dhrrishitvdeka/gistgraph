import pytest

from gistgraph.data.schema import Example


class WhitespaceTokenizer:
    """Deterministic stand-in for a real tokenizer: one token per whitespace-separated word."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
                self.inv[self.vocab[w]] = w
            ids.append(self.vocab[w])
        return ids

    def decode(self, ids):
        return " ".join(self.inv[i] for i in ids)


@pytest.fixture
def tok():
    return WhitespaceTokenizer()


@pytest.fixture
def example():
    paragraphs = [
        "Paris is the capital of France. It lies on the Seine river.",
        "The Eiffel Tower was built in 1889. It is located in Paris.",
        "Bananas are yellow fruit. They grow in tropical climates.",
        "Mount Everest is the tallest mountain. It is in the Himalayas.",
    ]
    return Example(
        id="ex1",
        context=" ".join(paragraphs),
        question="When was the Eiffel Tower built?",
        answers=["1889"],
        dataset="toy",
        paragraphs=paragraphs,
        supporting=[1],
    )
