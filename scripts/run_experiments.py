"""Run several experiment configs one after another.

Each config is trained with ``python -m gistgraph train``. A run whose ``compressor.pt`` already
exists is skipped, so the script can be restarted after an interruption (training itself resumes
from its own checkpoint). Use ``--wait-pid`` to hold off until another job releases the GPU.

    python scripts/run_experiments.py configs/experiments/m2_flat.yaml configs/experiments/m3_*.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from gistgraph.config import load_config


def wait_for(pid: int) -> None:
    try:
        import psutil
    except ImportError:
        print("psutil not installed; not waiting", flush=True)
        return
    while psutil.pid_exists(pid):
        time.sleep(30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--wait-pid", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stream-eval", type=int, default=0, help="also evaluate in streamed mode")
    ap.add_argument("--set", nargs="*", default=[], help="overrides applied to every run")
    args = ap.parse_args()

    if args.wait_pid:
        print(f"waiting for process {args.wait_pid} to finish", flush=True)
        wait_for(args.wait_pid)

    for path in args.configs:
        cfg = load_config(path, args.set)
        if (Path(cfg.out_dir) / "compressor.pt").exists():
            print(f"skip {path} (already trained)", flush=True)
            continue
        print(f"=== {path} ===", flush=True)
        cmd = [sys.executable, "-m", "gistgraph", "train", "--config", str(path)]
        cmd += ["--device", args.device, *args.set]
        code = subprocess.call(cmd)
        if code != 0:
            print(f"{path} failed with exit code {code}; continuing", flush=True)
        elif args.stream_eval and cfg.compressor.type != "flat":
            stream = [sys.executable, "-m", "gistgraph", "eval", "--config", str(path)]
            stream += ["--device", args.device, *args.set, f"eval.stream_chunks={args.stream_eval}"]
            subprocess.call(stream)


if __name__ == "__main__":
    main()
