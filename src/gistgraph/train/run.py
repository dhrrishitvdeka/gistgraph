"""End-to-end run: build the teacher cache, train a compressor, evaluate it."""

from __future__ import annotations

from pathlib import Path

from gistgraph.config import Config, save_config
from gistgraph.data.datasets import fit_context, load_examples
from gistgraph.data.teacher_cache import build_teacher_cache, load_teacher_cache, make_train_items
from gistgraph.eval.run_learned import evaluate_learned, load_eval_examples
from gistgraph.llm.frozen import FrozenLM
from gistgraph.model.compressor import build_compressor
from gistgraph.model.projector import mean_embedding_norm
from gistgraph.train.trainer import Trainer
from gistgraph.utils.seed import seed_everything


def cache_path(cfg: Config) -> Path:
    key = "_".join(
        [
            cfg.llm.name.replace("/", "-"),
            "+".join(cfg.data.train),
            str(cfg.train.n_teacher),
            str(cfg.data.max_ctx_tokens),
            str(cfg.train.answer_tokens),
            str(cfg.seed),
        ]
    )
    return Path("cache") / "teacher" / f"{key}.npz"


def train_examples(cfg: Config, lm: FrozenLM):
    per = max(1, cfg.train.n_teacher // len(cfg.data.train))
    out = []
    for name in cfg.data.train:
        for ex in load_examples(name, "train", per, cfg.seed):
            if ex.answers and ex.answers[0].strip():
                out.append(fit_context(ex, lm.tokenizer, cfg.data.max_ctx_tokens))
    return out


def run_training(cfg: Config, device: str = "cuda", evaluate: bool = True) -> Path:
    seed_everything(cfg.seed, deterministic=False)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.resolved.yaml")

    lm = FrozenLM.from_pretrained(cfg.llm.name, cfg.llm.dtype, cfg.llm.grad_ckpt, device)
    examples = train_examples(cfg, lm)
    path = cache_path(cfg)
    if not path.exists():
        print(f"building teacher cache for {len(examples)} examples -> {path}", flush=True)
        build_teacher_cache(
            lm,
            examples,
            path,
            max_new_tokens=cfg.train.answer_tokens,
            batch_size=cfg.eval.batch_size,
        )
    eos_id = lm.tokenizer.eos_token_id
    items = make_train_items(
        lm.tokenizer, examples, load_teacher_cache(path), cfg.data.max_ctx_tokens, eos_id
    )
    print(f"{len(items)} training items", flush=True)

    embed_norm = mean_embedding_norm(lm.model.get_input_embeddings().weight)
    compressor = build_compressor(cfg, lm.d_model, embed_norm).to(lm.device)
    n_params = sum(p.numel() for p in compressor.parameters())
    print(f"compressor '{cfg.compressor.type}': {n_params / 1e6:.1f}M trainable parameters")

    dev = load_eval_examples(cfg, lm, cfg.data.eval[0], cfg.train.n_dev) if cfg.train.n_dev else []
    weights = Trainer(cfg, lm, compressor, items, dev).train()
    if evaluate:
        print(f"report written to {evaluate_learned(cfg, lm, compressor)}")
    return weights
