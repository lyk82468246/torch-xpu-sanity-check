#!/usr/bin/env python
"""Measure whether XPU kernel compilation is cached across processes.

Each measurement runs in a *fresh* interpreter (that is the whole point: the JIT
cache only helps if a new process skips compilation), and reports the time of
the first call to each operator.

Typical session on an Intel Iris Xe iGPU:

    $env:NEO_CACHE_DIR = "$env:LOCALAPPDATA\\torch_xpu_cache"
    python diagnostics/cache_probe.py --fresh          # cold: fills the cache
    python diagnostics/cache_probe.py                  # warm: seconds instead of minutes

Without NEO_CACHE_DIR every run is cold, which is what the unconfigured driver
does.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

OPS = ("exp", "sum", "sort", "softmax", "matmul", "conv2d", "layernorm")


def run_op(name):
    """Execute one operator on XPU and return the wall time of the first call."""
    import warnings

    warnings.filterwarnings("ignore")
    import torch

    dev = "xpu"
    a = torch.randn(2048, 2048, device=dev)
    b = torch.randn(2048, 2048, device=dev)
    torch.xpu.synchronize()

    fn = {
        "exp": lambda: a.exp(),
        "sin": lambda: a.sin(),
        "sum": lambda: a.sum(),
        "mean": lambda: a.mean(),
        "sort": lambda: torch.sort(a),
        "softmax": lambda: torch.nn.functional.softmax(a, -1),
        "matmul": lambda: a @ b,
        "conv2d": lambda: torch.nn.functional.conv2d(
            torch.randn(4, 3, 128, 128, device=dev), torch.randn(32, 3, 7, 7, device=dev),
            stride=2, padding=3),
        "layernorm": lambda: torch.nn.functional.layer_norm(
            torch.randn(8, 128, 256, device=dev), (256,)),
    }[name]

    torch.xpu.synchronize()
    start = time.perf_counter()
    fn()
    torch.xpu.synchronize()
    return time.perf_counter() - start


def measure(op):
    """Run one op in a child process; return seconds, or None if the child failed."""
    handle, path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    try:
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", op, path],
                              stderr=subprocess.DEVNULL)
        if proc.returncode != 0:
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)["seconds"]
    finally:
        os.unlink(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ops", default=",".join(OPS),
                        help=f"comma separated operator list (default: {','.join(OPS)})")
    parser.add_argument("--runs", type=int, default=2, help="fresh processes per operator (default: 2)")
    parser.add_argument("--fresh", action="store_true",
                        help="delete the contents of NEO_CACHE_DIR first, so the first run is genuinely cold")
    parser.add_argument("--child", nargs=2, metavar=("OP", "OUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        op, out_path = args.child
        seconds = run_op(op)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"op": op, "seconds": seconds}, fh)
        return

    cache_dir = os.environ.get("NEO_CACHE_DIR")
    if args.fresh:
        if not cache_dir:
            raise SystemExit("--fresh needs NEO_CACHE_DIR to be set (see scripts/enable_jit_cache.ps1)")
        for entry in os.listdir(cache_dir):
            target = os.path.join(cache_dir, entry)
            shutil.rmtree(target) if os.path.isdir(target) else os.unlink(target)
        print(f"cleared cache: {cache_dir}\n")

    print(f"NEO_CACHE_DIR = {cache_dir or '(unset -- every run will be cold)'}")
    ops = [op.strip() for op in args.ops.split(",") if op.strip()]

    print(f"\n{'op':>10} " + " ".join(f"{'run' + str(i + 1):>10}" for i in range(args.runs)))
    for op in ops:
        times = []
        for _ in range(args.runs):
            seconds = measure(op)
            times.append(seconds)
        cells = " ".join(f"{t:>9.2f}s" if t is not None else f"{'FAILED':>10}" for t in times)
        print(f"{op:>10} {cells}")
        if len(times) >= 2 and times[0] and times[1]:
            print(f"{'':>10} first->second: {times[0] / times[1]:.0f}x faster")


if __name__ == "__main__":
    main()
