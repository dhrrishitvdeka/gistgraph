"""Config schema and loader.

Experiments are described by YAML files that are parsed into nested dataclasses. Three features
keep runs reproducible and easy to sweep:

* ``base:`` lets a file inherit from other YAML files (merged recursively, later files win).
* ``overrides`` such as ``["train.lr=1e-4", "compressor.ratio=3"]`` patch single values.
* Unknown keys raise, so a typo in a config can never be silently ignored.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "auto"
    feature_layer: int = 12
    grad_ckpt: bool = True


@dataclass
class DataConfig:
    train: list[str] = field(default_factory=lambda: ["hotpotqa", "2wiki", "squad"])
    eval: list[str] = field(default_factory=lambda: ["hotpotqa", "2wiki", "squad"])
    max_ctx_tokens: int = 1536
    n_eval: int = 500
    n_train: int = 30000


@dataclass
class EncoderConfig:
    d: int = 512
    layers: int = 2
    seg_len: int = 1


@dataclass
class WriterConfig:
    kind: str = "flat"  # flat | routed
    top_k: int = 2
    capacity_factor: float = 1.5  # adaptive: candidate nodes = capacity_factor * N / ratio
    adaptive: bool = False
    router_noise: float = 0.1
    gate_init: float = 2.0  # initial hard-concrete log-alpha (gates start mostly open)


@dataclass
class EdgesConfig:
    enabled: bool = False
    relations: int = 4
    mode: str = "learned"  # learned | random (random graph of equal degree, a control)
    rank: int = 64
    sparsifier: str = "entmax15"  # entmax15 | gumbel_topk
    max_deg: int = 8


@dataclass
class GNNConfig:
    enabled: bool = False
    layers: int = 2


@dataclass
class ProjectorConfig:
    pe: str = "rw"  # rw | none
    pe_steps: int = 8


@dataclass
class StreamingConfig:
    enabled: bool = False
    chunks: int = 4


@dataclass
class CompressorConfig:
    type: str = "flat"  # flat | routed | graph | text:<baseline>
    ratio: float = 4.0
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    writer: WriterConfig = field(default_factory=WriterConfig)
    edges: EdgesConfig = field(default_factory=EdgesConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)


@dataclass
class LossConfig:
    kd_temp: float = 1.0
    kd_topk: int = 64
    rate_dual_lr: float = 0.05
    rate_quad: float = 10.0  # augmented-Lagrangian weight that damps the rate controller
    edge_l0: float = 0.01  # weight on the (normalised) expected-edge penalty
    recon_weight: float = 1.0
    recon_anneal_frac: float = 0.3
    lb_weight: float = 0.01
    qa_ce_weight: float = 0.2
    rate_start_frac: float = 0.2  # fraction of training before the rate penalty switches on


@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch: int = 4
    grad_accum: int = 8
    steps: int = 6000
    warmup: int = 300
    ckpt_every: int = 500
    amp: bool = True
    ratios: list[float] = field(default_factory=lambda: [2.0, 3.0, 4.0, 5.0])  # sampled per batch
    n_teacher: int = 4000  # training examples with cached teacher outputs
    recon_tokens: int = 48  # context tokens the LLM must rebuild for the reconstruction loss
    answer_tokens: int = 24  # teacher answer length kept for distillation
    eval_every: int = 500
    n_dev: int = 100


@dataclass
class EvalConfig:
    methods: list[str] = field(default_factory=lambda: ["truncation_head"])
    ratios: list[float] = field(default_factory=lambda: [2.0, 3.0, 4.0, 5.0])
    split: str = "validation"
    batch_size: int = 8
    max_new_tokens: int = 32


@dataclass
class Config:
    name: str = "run"  # label for this method in result tables
    seed: int = 0
    out_dir: str = "runs/default"
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    compressor: CompressorConfig = field(default_factory=CompressorConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _deep_merge(base: dict, new: dict) -> dict:
    out = dict(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict:
    """Read a YAML file, resolving its ``base:`` parents (relative to the file) first."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parents = raw.pop("base", [])
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for parent in parents:
        merged = _deep_merge(merged, _read_yaml((path.parent / parent).resolve()))
    return _deep_merge(merged, raw)


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply ``dotted.key=value`` overrides. Values are parsed as YAML (so ``1e-4`` is a float)."""
    data = _deep_merge({}, data)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like 'a.b=value', got {item!r}")
        *parents, leaf = key.strip().split(".")
        node = data
        for name in parents:
            node = node.setdefault(name, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot descend into non-mapping at {name!r} in {item!r}")
        node[leaf] = yaml.safe_load(value)
    return data


def _is_dataclass_type(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _build(cls: type, data: dict, path: str = ""):
    """Recursively turn a plain dict into dataclass ``cls``, rejecting unknown keys."""
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"unknown config key(s) under {path or '<root>'}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        tp = hints[name]
        where = f"{path}.{name}" if path else name
        if _is_dataclass_type(tp):
            if not isinstance(value, dict):
                raise TypeError(f"{where} must be a mapping")
            kwargs[name] = _build(tp, value, where)
        else:
            kwargs[name] = _coerce(tp, value, where)
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if origin in (list, typing.Union, types.UnionType) or tp in (int, float, bool, str):
        if tp is float and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        if tp is float and isinstance(value, str):  # e.g. "1e-4" parsed as str by YAML 1.1
            return float(value)
        if tp in (int, float, bool, str) and not isinstance(value, tp):
            raise TypeError(f"{where} expects {tp.__name__}, got {type(value).__name__}")
        if origin is list and not isinstance(value, list):
            raise TypeError(f"{where} expects a list, got {type(value).__name__}")
    return value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a config from YAML (optional) and apply overrides. Defaults fill any gaps."""
    data = _read_yaml(Path(path)) if path is not None else {}
    data = apply_overrides(data, overrides or [])
    return _build(Config, data)


def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def save_config(cfg: Config, path: str | Path) -> None:
    """Write the fully resolved config next to a run's outputs."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")
