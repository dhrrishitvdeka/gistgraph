"""Command line entry point: ``python -m gistgraph <command> --config FILE [a.b=value ...]``."""

from __future__ import annotations

import argparse
from pathlib import Path

from gistgraph.config import load_config

EXP = "configs/experiments"
# milestone -> (command, configs). Each is the headline experiment of that milestone.
REPRODUCE = {
    "m1": ("baselines", [f"{EXP}/m1_baselines.yaml"]),
    "m2": ("train", [f"{EXP}/m2_flat.yaml"]),
    "m3": ("train", [f"{EXP}/m3_routed_fixed.yaml", f"{EXP}/m3_routed_adaptive.yaml"]),
    "m4": ("train", [f"{EXP}/m4_graph.yaml"]),
    "m5": (
        "train",
        [
            f"{EXP}/m5_ablations/graph_noedges.yaml",
            f"{EXP}/m5_ablations/graph_random.yaml",
            f"{EXP}/m5_ablations/graph_gumbel.yaml",
            f"{EXP}/m5_ablations/graph_stream.yaml",
        ],
    ),
}


def _cmd_baselines(args) -> None:
    from gistgraph.eval.run_baselines import run_baselines

    cfg = load_config(args.config, args.overrides)
    print(f"report written to {run_baselines(cfg, device=args.device)}")


def _cmd_train(args) -> None:
    from gistgraph.train.run import run_training

    cfg = load_config(args.config, args.overrides)
    run_training(cfg, device=args.device)


def _cmd_eval(args) -> None:
    import torch

    from gistgraph.eval.run_learned import evaluate_learned
    from gistgraph.llm.frozen import FrozenLM
    from gistgraph.model.compressor import build_compressor
    from gistgraph.model.projector import mean_embedding_norm

    cfg = load_config(args.config, args.overrides)
    lm = FrozenLM.from_pretrained(cfg.llm.name, cfg.llm.dtype, False, args.device)
    norm = mean_embedding_norm(lm.model.get_input_embeddings().weight)
    comp = build_compressor(cfg, lm.d_model, norm).to(lm.device)
    comp.load_state_dict(torch.load(Path(cfg.out_dir) / "compressor.pt", map_location=lm.device))
    print(f"report written to {evaluate_learned(cfg, lm, comp)}")


def _cmd_probe(args) -> None:
    import json

    import torch

    from gistgraph.data.datasets import load_examples
    from gistgraph.eval.probes import probe_model
    from gistgraph.llm.frozen import FrozenLM
    from gistgraph.model.compressor import build_compressor
    from gistgraph.model.projector import mean_embedding_norm

    cfg = load_config(args.config, args.overrides)
    lm = FrozenLM.from_pretrained(cfg.llm.name, cfg.llm.dtype, False, args.device)
    norm = mean_embedding_norm(lm.model.get_input_embeddings().weight)
    comp = build_compressor(cfg, lm.d_model, norm).to(lm.device)
    comp.load_state_dict(torch.load(Path(cfg.out_dir) / "compressor.pt", map_location=lm.device))
    examples = load_examples("hotpotqa", cfg.eval.split, args.n, cfg.seed)
    summary = probe_model(lm, comp, cfg, examples, args.ratio)
    out = Path(cfg.out_dir) / "edge_probe.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _cmd_report(args) -> None:
    from gistgraph.eval.results import build_results

    print(f"results written to {build_results(args.runs, args.out)}")


def _cmd_reproduce(args) -> None:
    command, configs = REPRODUCE[args.milestone]
    for config in configs:
        args.config = Path(config)
        {"baselines": _cmd_baselines, "train": _cmd_train}[command](args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gistgraph")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("baselines", help="evaluate non-learned baselines")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*", help="dotted overrides such as eval.ratios=[2,4]")
    p.set_defaults(func=_cmd_baselines)

    p = sub.add_parser("train", help="train a compressor, then evaluate it")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("eval", help="evaluate a trained compressor from its out_dir")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("probe", help="edge-alignment probe on HotpotQA supporting paragraphs")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ratio", type=float, default=4.0)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("overrides", nargs="*")
    p.set_defaults(func=_cmd_probe)

    p = sub.add_parser("report", help="build docs/results from finished run directories")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("docs/results"))
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("reproduce", help="rerun a milestone's headline experiment")
    p.add_argument("milestone", choices=sorted(REPRODUCE))
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    p.set_defaults(func=_cmd_reproduce)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
