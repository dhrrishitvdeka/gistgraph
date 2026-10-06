"""Non-learned baselines, built by name."""

from gistgraph.baselines.base import TextCompressor


def build_baseline(name: str, **kwargs) -> TextCompressor:
    """Construct a text baseline from its config name."""
    from gistgraph.baselines.extractive import Extractive
    from gistgraph.baselines.llmlingua2 import LLMLingua2
    from gistgraph.baselines.truncation import Truncation

    if name == "truncation_head":
        return Truncation("head")
    if name == "truncation_headtail":
        return Truncation("headtail")
    if name.startswith("extractive_"):
        return Extractive(name.removeprefix("extractive_"))
    if name == "llmlingua2":
        return LLMLingua2(**kwargs)
    raise KeyError(f"unknown baseline {name!r}")
