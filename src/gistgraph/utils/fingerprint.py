"""Config fingerprints and atomic writes, tying outputs to the config that made them."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from gistgraph.config import Config, config_to_dict

# keys that may change between a run and its resumption without invalidating the weights
TRAIN_IGNORE = ("eval", "train.steps", "train.ckpt_every", "train.eval_every")
# streamed and one-shot evaluations share a results file (and the full-context rows)
EVAL_IGNORE = ("eval.stream_chunks",)


def _drop(d: dict, dotted: str) -> None:
    *parents, last = dotted.split(".")
    for p in parents:
        d = d.get(p, {})
    if isinstance(d, dict):
        d.pop(last, None)


def config_hash(cfg: Config, ignore: tuple[str, ...] = ()) -> str:
    """Short sha256 of the resolved config, ignoring ``out_dir`` and any dotted ``ignore`` keys."""
    d = config_to_dict(cfg)
    for key in ("out_dir", *ignore):
        _drop(d, key)
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def train_hash(cfg: Config) -> str:
    return config_hash(cfg, TRAIN_IGNORE)


def eval_hash(cfg: Config) -> str:
    return config_hash(cfg, EVAL_IGNORE)


def atomic_torch_save(obj, path: str | Path) -> None:
    """``torch.save`` to a temp file then rename, so a crash never leaves a torn file."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_compressor_state(path: str | Path, map_location=None) -> dict:
    """Compressor weights from ``compressor.pt``: either a raw state dict or the newer
    ``{"state_dict", "config_hash", "step"}`` wrapper."""
    obj = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj and "config_hash" in obj:
        return obj["state_dict"]
    return obj
