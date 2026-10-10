import re
from pathlib import Path

import pytest

from gistgraph.data.schema import Example

try:
    import torch
    import transformers  # noqa: F401
except ImportError:  # torch-free environment: only the pure-Python tests can run
    torch = None

# Test modules that need torch, directly or through a gistgraph module that imports it.
_NEEDS_TORCH = re.compile(
    r"^\s*(import (torch|transformers)|from (torch|transformers)[\s.]"
    r"|from gistgraph\.(?!config|data\.schema)\S+ import|import gistgraph\.)",
    re.M,
)

collect_ignore_glob = []
if torch is None:
    collect_ignore_glob = [
        p.name
        for p in Path(__file__).parent.glob("test_*.py")
        if _NEEDS_TORCH.search(p.read_text(encoding="utf-8"))
    ]


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``gpu`` when no CUDA device is available."""
    if torch is not None and torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


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
    eos_token_id = 0

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
    pytest.importorskip("torch")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from gistgraph.llm.frozen import FrozenLM

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
