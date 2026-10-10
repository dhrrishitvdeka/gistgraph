"""Inference API: load a trained compressor, compress contexts and answer questions.

```python
from gistgraph import GistGraph

gg = GistGraph.from_pretrained("dhrrishitvdeka/gistgraph-qwen2.5-0.5b")
mem = gg.compress(long_text, ratio=4)
gg.answer("Who founded the company?", memory=mem)
```
"""

from __future__ import annotations

import json
import tempfile
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from gistgraph import __version__
from gistgraph.config import Config, _build, config_to_dict, validate
from gistgraph.eval.run_learned import amp_dtype
from gistgraph.llm.frozen import FrozenLM
from gistgraph.model.compressor import build_compressor
from gistgraph.model.projector import mean_embedding_norm

FORMAT_VERSION = 1
CONFIG_FILE = "config.json"
WEIGHTS_FILE = "compressor.safetensors"
REPO_URL = "https://github.com/dhrrishitvdeka/gistgraph"


@dataclass
class Compressed:
    """A compressed context: ``embeds [K, d]`` soft-prompt vectors the frozen LLM reads."""

    embeds: Tensor
    mask: Tensor  # [K] bool, all True; kept for symmetry with ``Memory``
    n_nodes: int
    n_tokens: int

    @property
    def ratio(self) -> float:
        """Achieved compression: context tokens per latent node."""
        return self.n_tokens / max(self.n_nodes, 1)


def _config_from_dict(data: dict) -> Config:
    data = {k: v for k, v in data.items() if k not in ("format_version", "gistgraph_version")}
    return validate(_build(Config, data))


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class GistGraph:
    """A frozen LLM plus a trained compressor, ready for inference."""

    def __init__(self, cfg: Config, lm: FrozenLM, compressor: torch.nn.Module):
        self.cfg = cfg
        self.lm = lm
        self.compressor = compressor.to(lm.device).eval()

    @property
    def device(self) -> torch.device:
        return self.lm.device

    # ---- loading and saving ------------------------------------------------------------------

    @classmethod
    def _assemble(cls, cfg: Config, state: dict, device, dtype, lm: FrozenLM | None):
        if dtype is not None:
            cfg = replace(cfg, llm=replace(cfg.llm, dtype=dtype))
        if lm is None:
            lm = FrozenLM.from_pretrained(
                cfg.llm.name, cfg.llm.dtype, False, device or _default_device()
            )
        elif device is not None:
            lm = lm.to(device)
        norm = mean_embedding_norm(lm.model.get_input_embeddings().weight)
        comp = build_compressor(cfg, lm.d_model, norm)
        comp.load_state_dict(state)
        return cls(cfg, lm, comp)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path,
        device: str | None = None,
        dtype: str | None = None,
        revision: str | None = None,
        lm: FrozenLM | None = None,
    ) -> GistGraph:
        """Load from a directory written by ``save_pretrained`` or a Hugging Face Hub repo id.

        ``lm`` skips loading the base LLM named in the config and uses this one instead.
        """
        from safetensors.torch import load_file

        path = Path(path_or_repo)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(path_or_repo), revision=revision))
        data = json.loads((path / CONFIG_FILE).read_text(encoding="utf-8"))
        fmt = data.get("format_version", FORMAT_VERSION)
        if fmt > FORMAT_VERSION:
            raise ValueError(f"{path}: format_version {fmt} is newer than this gistgraph supports")
        state = load_file(str(path / WEIGHTS_FILE))
        return cls._assemble(_config_from_dict(data), state, device, dtype, lm)

    @classmethod
    def from_run(
        cls,
        run_dir: str | Path,
        device: str | None = None,
        dtype: str | None = None,
        lm: FrozenLM | None = None,
    ) -> GistGraph:
        """Load from a training run directory (``config.resolved.yaml`` + ``compressor.pt``)."""
        from gistgraph.config import load_config
        from gistgraph.utils.fingerprint import load_compressor_state

        run_dir = Path(run_dir)
        cfg = load_config(run_dir / "config.resolved.yaml")
        state = load_compressor_state(run_dir / "compressor.pt", map_location="cpu")
        return cls._assemble(cfg, state, device, dtype, lm)

    def save_pretrained(self, directory: str | Path) -> Path:
        """Write ``config.json``, ``compressor.safetensors`` and a model card stub."""
        from safetensors.torch import save_file

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        meta = {**config_to_dict(self.cfg), "format_version": FORMAT_VERSION}
        meta["gistgraph_version"] = __version__
        (out / CONFIG_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # clone so tied/shared tensors are stored separately (safetensors refuses shared storage)
        state = {
            k: v.detach().to("cpu").clone().contiguous()
            for k, v in self.compressor.state_dict().items()
        }
        save_file(state, str(out / WEIGHTS_FILE), metadata={"format": "pt"})
        (out / "README.md").write_text(self._model_card(), encoding="utf-8")
        return out

    def _model_card(self) -> str:
        ratios = ", ".join(f"{r:g}x" for r in self.cfg.train.ratios)
        return (
            "---\nlibrary_name: gistgraph\nlicense: apache-2.0\n"
            f"base_model: {self.cfg.llm.name}\n---\n\n"
            f"# {self.cfg.name}\n\n"
            f"A Gist Graph compressor ({self.cfg.compressor.type}) for the frozen LLM "
            f"`{self.cfg.llm.name}`. It turns a long context into a few latent nodes that the LLM "
            f"reads as a soft prompt. Trained compression ratios: {ratios}.\n\n"
            "```python\nfrom gistgraph import GistGraph\n\n"
            'gg = GistGraph.from_pretrained("<this repo>")\n'
            'gg.answer("question", context="long text", ratio=4)\n```\n\n'
            f"Code: {REPO_URL}\n"
        )

    def push_to_hub(self, repo_id: str, private: bool = False, token: str | None = None) -> str:
        """Save to a temporary directory and upload it as a Hub model repo. Returns the URL."""
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id, private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.save_pretrained(tmp)
            api.upload_folder(repo_id=repo_id, folder_path=tmp)
        return f"https://huggingface.co/{repo_id}"

    # ---- inference ---------------------------------------------------------------------------

    def _autocast(self):
        use_amp = self.device.type == "cuda" and self.cfg.train.amp
        return torch.autocast(self.device.type, dtype=amp_dtype()) if use_amp else nullcontext()

    def compress(self, context: str, ratio: float = 4.0) -> Compressed:
        """Compress ``context`` to roughly ``n_tokens / ratio`` latent nodes."""
        lowest = min(self.cfg.train.ratios)
        if ratio < lowest:
            raise ValueError(f"ratio {ratio:g} is below the smallest trained ratio {lowest:g}")
        ids = self.lm.encode_text(context)
        limit = self.cfg.data.max_ctx_tokens
        if ids.shape[0] > limit:
            warnings.warn(f"context has {ids.shape[0]} tokens; truncated to {limit}", stacklevel=2)
            ids = ids[:limit]
        ids = ids[None]
        mask = torch.ones_like(ids)
        dtype = self.lm.model.get_input_embeddings().weight.dtype
        with torch.inference_mode():
            h = self.lm.token_features(ids, mask, self.cfg.llm.feature_layer)
            with self._autocast():
                mem = self.compressor(h, mask.bool(), ratio, chunks=self.cfg.eval.stream_chunks)
            prefix = mem.prefix(0).to(dtype)
        n = prefix.shape[0]
        return Compressed(prefix, torch.ones(n, dtype=torch.bool), n, ids.shape[1])

    def _generate(self, questions: list[str], prefixes: list[Tensor], max_new_tokens: int):
        # same prompt assembly and greedy decoding as the evaluation harness
        out = []
        bs = self.cfg.eval.batch_size
        with torch.inference_mode():
            for i in range(0, len(questions), bs):
                embeds, mask = self.lm.build_inputs(questions[i : i + bs], prefixes[i : i + bs])
                preds = self.lm.generate(embeds, mask, max_new_tokens=max_new_tokens)
                out += [p.split("\n")[0].strip() for p in preds]
        return out

    def answer(
        self,
        question: str,
        memory: Compressed | None = None,
        context: str | None = None,
        ratio: float = 4.0,
        max_new_tokens: int = 32,
    ) -> str:
        """Answer ``question`` from a compressed ``memory`` or a raw ``context`` (exactly one)."""
        if (memory is None) == (context is None):
            raise ValueError("pass exactly one of memory= or context=")
        if memory is None:
            memory = self.compress(context, ratio)
        return self._generate([question], [memory.embeds], max_new_tokens)[0]

    def answer_batch(
        self,
        questions: list[str],
        contexts: list[str],
        ratio: float = 4.0,
        max_new_tokens: int = 32,
    ) -> list[str]:
        """Answer each question from its own context, batching generation."""
        if len(questions) != len(contexts):
            raise ValueError("questions and contexts must have the same length")
        prefixes = [self.compress(c, ratio).embeds for c in contexts]
        return self._generate(list(questions), prefixes, max_new_tokens)
