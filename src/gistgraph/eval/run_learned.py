"""Evaluate learned (soft-prompt) compressors with the same harness as the text baselines."""

from __future__ import annotations

from pathlib import Path

import torch

from gistgraph.config import Config
from gistgraph.data.datasets import fit_context, load_examples
from gistgraph.data.schema import Example
from gistgraph.eval.harness import read_rows, run_eval, run_text_method, write_rows
from gistgraph.eval.report import full_context_table, retention_markdown
from gistgraph.llm.frozen import FrozenLM


def amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def make_prepare(lm: FrozenLM, compressor, cfg: Config, ratio: float):
    """Build the ``prepare`` callback for ``run_eval``: compress one example to a soft prefix."""
    use_amp = lm.device.type == "cuda" and cfg.train.amp
    dtype = lm.model.get_input_embeddings().weight.dtype

    @torch.no_grad()
    def prepare(ex: Example):
        ids = lm.encode_text(ex.context)[None]
        mask = torch.ones_like(ids)
        h = lm.token_features(ids, mask, cfg.llm.feature_layer)
        with torch.autocast(lm.device.type, dtype=amp_dtype(), enabled=use_amp):
            mem = compressor(h, mask.bool(), ratio)
        prefix = mem.prefix(0).detach().to(dtype)
        return None, prefix, prefix.shape[0]

    return prepare


def load_eval_examples(cfg: Config, lm: FrozenLM, dataset: str, n: int | None = None):
    n = cfg.data.n_eval if n is None else n
    return [
        fit_context(ex, lm.tokenizer, cfg.data.max_ctx_tokens)
        for ex in load_examples(dataset, cfg.eval.split, n, cfg.seed)
    ]


def dev_f1(lm: FrozenLM, compressor, cfg: Config, examples: list[Example], ratio: float) -> float:
    """Mean F1 on a small dev set at one ratio: the quick signal logged during training."""
    compressor.eval()
    try:
        rows = run_eval(
            lm,
            examples,
            make_prepare(lm, compressor, cfg, ratio),
            cfg.name,
            ratio,
            batch_size=cfg.eval.batch_size,
            max_new_tokens=cfg.eval.max_new_tokens,
        )
    finally:
        compressor.train()
    return sum(r["f1"] for r in rows) / max(len(rows), 1)


def evaluate_learned(cfg: Config, lm: FrozenLM, compressor) -> Path:
    """Evaluate ``compressor`` at every configured ratio and dataset. Resumable."""
    out = Path(cfg.out_dir)
    results = out / "results.jsonl"
    finished = {
        (r["method"], r["target_ratio"], r["dataset"])
        for r in (read_rows(results) if results.exists() else [])
    }
    compressor.eval()
    kw = {"batch_size": cfg.eval.batch_size, "max_new_tokens": cfg.eval.max_new_tokens}
    for dataset in cfg.data.eval:
        examples = load_eval_examples(cfg, lm, dataset)
        if ("full", 1.0, dataset) not in finished:
            write_rows(run_text_method(lm, None, examples, 1.0, **kw), results)
        for ratio in cfg.eval.ratios:
            if (cfg.name, ratio, dataset) in finished:
                continue
            rows = run_eval(
                lm, examples, make_prepare(lm, compressor, cfg, ratio), cfg.name, ratio, **kw
            )
            write_rows(rows, results)
            print(f"[{dataset}] {cfg.name} @ {ratio:g}x: {len(rows)} examples", flush=True)
    rows = read_rows(results)
    report = out / "report.md"
    report.write_text(
        "## Full-context reference\n\n"
        + full_context_table(rows)
        + "\n\n## F1 retention (achieved ratio in brackets)\n\n"
        + retention_markdown(rows)
        + "\n",
        encoding="utf-8",
    )
    return report
