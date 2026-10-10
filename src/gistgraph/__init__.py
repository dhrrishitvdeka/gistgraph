__version__ = "0.2.0"

__all__ = ["GistGraph", "__version__"]


def __getattr__(name: str):
    # imported lazily so ``import gistgraph`` stays light and works without torch
    if name == "GistGraph":
        from gistgraph.api import GistGraph

        return GistGraph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
