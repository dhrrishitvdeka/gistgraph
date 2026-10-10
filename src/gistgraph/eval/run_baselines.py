"""Run the non-learned baselines and write results plus a Markdown report."""

from __future__ import annotations

from pathlib import Path

from gistgraph.baselines import build_baseline
from gistgraph.config import Config, save_config
from gistgraph.data.datasets import fit_context, load_examples
from gistgraph.eval.harness import read_rows, run_text_method, write_rows
from gistgraph.eval.report import full_context_table, retention_markdown
from gistgraph.llm.frozen import FrozenLM
from gistgraph.utils.seed import seed_everything


def _done(rows: list[dict]) -> set[tuple]:
    return {(r["method"], r["target_ratio"], r["dataset"]) for r in rows}


def run_baselines(cfg: Config, device: str = "cuda") -> Path:
    """Evaluate full context and every configured baseline at every ratio. Resumable."""
    seed_everything(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.resolved.yaml")
    results = out / "results.jsonl"
    finished = _done(read_rows(results)) if results.exists() else set()

    lm = FrozenLM.from_pretrained(cfg.llm.name, cfg.llm.dtype, grad_ckpt=False, device=device)
    kw = {"batch_size": cfg.eval.batch_size, "max_new_tokens": cfg.eval.max_new_tokens}
    # one instance per method, reused across datasets and ratios (llmlingua2 caches its model)
    methods = {
        name: build_baseline(name, device=device) if name == "llmlingua2" else build_baseline(name)
        for name in dict.fromkeys(cfg.eval.methods)
    }
    for dataset in cfg.data.eval:
        examples = [
            fit_context(ex, lm.tokenizer, cfg.data.max_ctx_tokens)
            for ex in load_examples(dataset, cfg.eval.split, cfg.data.n_eval, cfg.seed)
        ]
        jobs = [("full", 1.0, None)] + [
            (name, ratio, methods[name]) for name in cfg.eval.methods for ratio in cfg.eval.ratios
        ]
        for name, ratio, method in jobs:
            if (name, ratio, dataset) in finished:
                continue
            rows = run_text_method(lm, None if name == "full" else method, examples, ratio, **kw)
            write_rows(rows, results)
            print(f"[{dataset}] {name} @ {ratio:g}x: {len(rows)} examples", flush=True)

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
