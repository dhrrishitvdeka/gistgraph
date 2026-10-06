"""Command line entry point: ``python -m gistgraph <command> --config FILE [a.b=value ...]``."""

from __future__ import annotations

import argparse
from pathlib import Path

from gistgraph.config import load_config

REPRODUCE = {"m1": "configs/experiments/m1_baselines.yaml"}


def _cmd_baselines(args) -> None:
    from gistgraph.eval.run_baselines import run_baselines

    cfg = load_config(args.config, args.overrides)
    print(f"report written to {run_baselines(cfg, device=args.device)}")


def _cmd_reproduce(args) -> None:
    if args.milestone not in REPRODUCE:
        raise SystemExit(f"nothing to reproduce for {args.milestone!r}; known: {sorted(REPRODUCE)}")
    args.config = REPRODUCE[args.milestone]
    _cmd_baselines(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gistgraph")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("baselines", help="evaluate non-learned baselines")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*", help="dotted overrides such as eval.ratios=[2,4]")
    p.set_defaults(func=_cmd_baselines)

    p = sub.add_parser("reproduce", help="rerun a milestone's headline experiment")
    p.add_argument("milestone", choices=sorted(REPRODUCE))
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    p.set_defaults(func=_cmd_reproduce)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
