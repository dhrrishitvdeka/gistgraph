"""Command line entry point: ``python -m gistgraph <command> --config FILE [a.b=value ...]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gistgraph.config import load_config


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _positive(cast):
    """argparse type that parses with ``cast`` and requires a value > 0."""

    def parse(text: str):
        try:
            value = cast(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid {cast.__name__} value: {text!r}") from None
        if value <= 0:
            raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
        return value

    parse.__name__ = f"positive_{cast.__name__}"
    return parse


def _load(args):
    """Load the config named on the command line; config mistakes exit cleanly with status 2."""
    try:
        return load_config(args.config, args.overrides)
    except (KeyError, TypeError, ValueError, FileNotFoundError) as err:
        print(
            f"error: {err.args[0] if isinstance(err, KeyError) and err.args else err}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


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

    cfg = _load(args)
    print(f"report written to {run_baselines(cfg, device=args.device)}")


def _cmd_train(args) -> None:
    from gistgraph.train.run import run_training

    cfg = _load(args)
    run_training(cfg, device=args.device, force=args.force)


def _cmd_eval(args) -> None:
    from gistgraph.eval.run_learned import evaluate_learned
    from gistgraph.llm.frozen import FrozenLM
    from gistgraph.model.compressor import build_compressor
    from gistgraph.model.projector import mean_embedding_norm
    from gistgraph.utils.fingerprint import load_compressor_state

    cfg = _load(args)
    lm = FrozenLM.from_pretrained(cfg.llm.name, cfg.llm.dtype, False, args.device)
    norm = mean_embedding_norm(lm.model.get_input_embeddings().weight)
    comp = build_compressor(cfg, lm.d_model, norm).to(lm.device)
    comp.load_state_dict(
        load_compressor_state(Path(cfg.out_dir) / "compressor.pt", map_location=lm.device)
    )
    print(f"report written to {evaluate_learned(cfg, lm, comp)}")


def _cmd_probe(args) -> None:
    import json

    from gistgraph.data.datasets import load_examples
    from gistgraph.eval.probes import probe_model
    from gistgraph.llm.frozen import FrozenLM
    from gistgraph.model.compressor import build_compressor
    from gistgraph.model.projector import mean_embedding_norm
    from gistgraph.utils.fingerprint import load_compressor_state

    cfg = _load(args)
    lm = FrozenLM.from_pretrained(cfg.llm.name, cfg.llm.dtype, False, args.device)
    norm = mean_embedding_norm(lm.model.get_input_embeddings().weight)
    comp = build_compressor(cfg, lm.d_model, norm).to(lm.device)
    comp.load_state_dict(
        load_compressor_state(Path(cfg.out_dir) / "compressor.pt", map_location=lm.device)
    )
    examples = load_examples(args.dataset, cfg.eval.split, args.n, cfg.seed)
    summary = probe_model(lm, comp, cfg, examples, args.ratio)
    out = Path(cfg.out_dir) / "edge_probe.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _cmd_report(args) -> None:
    from gistgraph.eval.results import build_results

    print(f"results written to {build_results(args.runs, args.out)}")


def _open_model(model: str, device: str | None = None):
    """A ``GistGraph`` from a saved directory, a training run directory or a Hub repo id."""
    from gistgraph.api import GistGraph

    if (Path(model) / "config.resolved.yaml").exists():
        return GistGraph.from_run(model, device=device)
    return GistGraph.from_pretrained(model, device=device)


def _cmd_ask(args) -> None:
    if args.context_file is not None:
        context = Path(args.context_file).read_text(encoding="utf-8")
    else:
        context = args.context
    gg = _open_model(args.model, args.device)
    try:
        mem = gg.compress(context, args.ratio)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(2) from None
    print(gg.answer(args.question, memory=mem, max_new_tokens=args.max_new_tokens))
    if args.verbose:
        print(f"nodes={mem.n_nodes} tokens={mem.n_tokens} ratio={mem.ratio:.2f}")


def _cmd_export(args) -> None:
    from gistgraph.api import GistGraph

    out = GistGraph.from_run(args.run_dir, device=args.device).save_pretrained(args.out_dir)
    print(f"model written to {out}")


def _cmd_reproduce(args) -> None:
    command, configs = REPRODUCE[args.milestone]
    for config in configs:
        args.config = Path(config)
        args.force = getattr(args, "force", False)
        {"baselines": _cmd_baselines, "train": _cmd_train}[command](args)


def main(argv: list[str] | None = None) -> None:
    device = _default_device()
    parser = argparse.ArgumentParser(prog="gistgraph")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("baselines", help="evaluate non-learned baselines")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default=device)
    p.add_argument("overrides", nargs="*", help="dotted overrides such as eval.ratios=[2,4]")
    p.set_defaults(func=_cmd_baselines)

    p = sub.add_parser("train", help="train a compressor, then evaluate it")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default=device)
    p.add_argument("overrides", nargs="*")
    p.add_argument("--force", action="store_true", help="overwrite an existing run directory")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("eval", help="evaluate a trained compressor from its out_dir")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default=device)
    p.add_argument("overrides", nargs="*")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("probe", help="edge-alignment probe on HotpotQA supporting paragraphs")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default=device)
    p.add_argument("--ratio", type=_positive(float), default=4.0)
    p.add_argument("--n", type=_positive(int), default=200)
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("overrides", nargs="*")
    p.set_defaults(func=_cmd_probe)

    p = sub.add_parser("report", help="build docs/results from finished run directories")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("docs/results"))
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("reproduce", help="rerun a milestone's headline experiment")
    p.add_argument("milestone", choices=sorted(REPRODUCE))
    p.add_argument("--device", default=device)
    p.add_argument("overrides", nargs="*")
    p.set_defaults(func=_cmd_reproduce)

    p = sub.add_parser("ask", help="answer a question from a compressed context")
    p.add_argument("model", help="saved model directory, training run directory or Hub repo id")
    p.add_argument("question")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--context")
    src.add_argument("--context-file", type=Path)
    p.add_argument("--ratio", type=_positive(float), default=4.0)
    p.add_argument("--max-new-tokens", type=_positive(int), default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true", help="also print node count and ratio")
    p.set_defaults(func=_cmd_ask)

    p = sub.add_parser("export", help="convert a training run into a shareable model directory")
    p.add_argument("run_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=_cmd_export)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
