import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from gistgraph.data.schema import Example
from gistgraph.llm.frozen import FrozenLM


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


class CharTokenizer:
    """Character-level tokenizer with the few attributes FrozenLM needs."""

    eos_token = "<eos>"
    pad_token = "<eos>"
    pad_token_id = 0

    def _ids(self, text):
        return [(ord(c) % 90) + 5 for c in text]

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def decode(self, ids):
        # Not an exact inverse (the id mapping is lossy); good enough to keep lengths intact.
        return "".join(chr(int(i) + 27) for i in ids)

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = torch.tensor([self._ids(text)])
        return type("Enc", (), {"input_ids": ids})()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<u>{messages[0]['content']}</u><a>"

    def batch_decode(self, out, skip_special_tokens=True):
        return ["".join(chr(int(i) + 30) for i in row) for row in out]


@pytest.fixture(scope="session")
def lm():
    """A tiny randomly initialised Qwen2 behind FrozenLM, so no download is needed."""
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    return FrozenLM(Qwen2ForCausalLM(cfg), CharTokenizer())


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
