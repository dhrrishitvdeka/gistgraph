"""Run several experiment configs one after another.

Each config is trained with ``python -m gistgraph train``. A run whose ``compressor.pt`` already
exists is not retrained, so the script can be restarted after an interruption (training itself
resumes from its own checkpoint). Use ``--wait-pid`` to hold off until another job releases the
GPU. The script exits non-zero if any step failed.

    python scripts/run_experiments.py configs/experiments/m2_flat.yaml configs/experiments/m3_*.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from gistgraph.config import load_config


def pid_alive(pid: int) -> bool:
    """Whether process ``pid`` is still running (stdlib only)."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes

    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
    k32.CloseHandle(handle)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def wait_for(pid: int) -> None:
    while pid_alive(pid):
        time.sleep(30)


def run_all(configs: list[Path], device: str, stream_eval: int, overrides: list[str], call=None):
    """Train (and optionally stream-evaluate) each config; return ``[(config, step, code)]``
    for every failed step."""
    call = call or subprocess.call
    failures = []
    for path in configs:
        cfg = load_config(path, overrides)
        if (Path(cfg.out_dir) / "compressor.pt").exists():
            print(f"skip training {path} (already trained)", flush=True)
        else:
            print(f"=== {path} ===", flush=True)
            cmd = [sys.executable, "-m", "gistgraph", "train", "--config", str(path)]
            code = call(cmd + ["--device", device, *overrides])
            if code != 0:
                print(f"{path} failed with exit code {code}; continuing", flush=True)
                failures.append((path, "train", code))
                continue
        if stream_eval and cfg.compressor.type != "flat":
            cmd = [sys.executable, "-m", "gistgraph", "eval", "--config", str(path)]
            code = call(cmd + ["--device", device, *overrides, f"eval.stream_chunks={stream_eval}"])
            if code != 0:
                print(f"{path} stream eval failed with exit code {code}", flush=True)
                failures.append((path, "stream-eval", code))
    return failures


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

    failures = run_all(args.configs, args.device, args.stream_eval, args.set)
    for path, step, code in failures:
        print(f"FAILED {path} ({step}, exit code {code})", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
