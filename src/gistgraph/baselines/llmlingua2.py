"""LLMLingua-2 baseline (task-agnostic token-classification prompt compression)."""

from __future__ import annotations

from gistgraph.baselines.base import Tokenizer
from gistgraph.data.schema import Example

DEFAULT_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


class LLMLingua2:
    """Wraps the ``llmlingua`` package.

    The package asks for a *rate* in its own tokenizer's units, while we report ratios in the
    frozen LLM's tokens. The two differ a little, so the evaluation harness always measures and
    reports the achieved ratio instead of trusting the requested one.
    """

    query_aware = False
    name = "llmlingua2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._compressor = None

    def _load(self):
        if self._compressor is None:
            try:
                from llmlingua import PromptCompressor
            except ImportError as err:  # pragma: no cover - depends on optional install
                raise ImportError(
                    "LLMLingua-2 needs the optional 'llmlingua' package: pip install llmlingua"
                ) from err
            self._compressor = PromptCompressor(
                model_name=self.model_name, use_llmlingua2=True, device_map=self.device
            )
        return self._compressor

    def compress(self, ex: Example, ratio: float, tokenizer: Tokenizer) -> str:
        if ratio <= 1:
            return ex.context
        out = self._load().compress_prompt_llmlingua2(
            ex.context,
            rate=1.0 / ratio,
            force_tokens=["\n", "?", ".", "!"],
            chunk_end_tokens=[".", "\n"],
            drop_consecutive=True,
        )
        return out["compressed_prompt"]
