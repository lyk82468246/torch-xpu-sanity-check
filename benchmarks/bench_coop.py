#!/usr/bin/env python
"""Measure how well the CPU and the XPU work together.

Three patterns, each timed steady state:

  pipeline      the XPU queue runs while the CPU does its own independent work
  split GEMM    one GEMM split by rows across both engines, concurrently
  copy overlap  double-buffered host->device copies hidden behind compute

On an integrated GPU (shared DRAM, no VRAM of its own) only the first one pays
off; the other two lose to bandwidth and core contention. The script prints the
measured deltas rather than assuming either way.

Usage: python benchmarks/bench_coop.py [--out coop.json]
"""

import argparse
import json
import time

import torch

torch.set_num_threads(4)
DEV = torch.device("xpu")


def sync():
    torch.xpu.synchronize()


def best(fn, warmup=2, iters=5, device="xpu"):
    for _ in range(warmup):
        fn()
    if device == "xpu":
        sync()
    result = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device == "xpu":
            sync()
        result = min(result, time.perf_counter() - t0)
    return result * 1000


def pipeline_demo(xpu_iters=6, cpu_iters=2, n=2048):
    a_x, b_x = torch.randn(n, n, device=DEV), torch.randn(n, n, device=DEV)
    a_c, b_c = torch.randn(n, n), torch.randn(n, n)

    def xpu_chunk():
        for _ in range(xpu_iters):
            _ = a_x @ b_x

    def cpu_chunk():
        for _ in range(cpu_iters):
            _ = a_c @ b_c

    xpu_chunk()
    cpu_chunk()
    sync()

    t0 = time.perf_counter()
    xpu_chunk()
    sync()
    cpu_chunk()
    sequential = time.perf_counter() - t0

    t0 = time.perf_counter()
    xpu_chunk()   # queued, not awaited
    cpu_chunk()   # runs while the XPU queue drains
    sync()
    overlapped = time.perf_counter() - t0

    return {"xpu_iters": xpu_iters, "cpu_iters": cpu_iters,
            "sequential_ms": sequential * 1000, "overlapped_ms": overlapped * 1000,
            "saving_pct": (1 - overlapped / sequential) * 100}


def split_gemm(n=2048, xpu_share=0.85):
    a_c, b_c = torch.randn(n, n), torch.randn(n, n)
    a_x, b_x = a_c.to(DEV), b_c.to(DEV)
    _ = a_c @ b_c
    _ = a_x @ b_x
    sync()

    split = int(n * xpu_share)
    xpu_rows, cpu_rows = a_x[:split], a_c[split:]

    xpu_full = best(lambda: a_x @ b_x)
    cpu_full = best(lambda: a_c @ b_c, warmup=1, iters=3, device="cpu")

    def split_run():
        out_x = xpu_rows @ b_x
        out_c = cpu_rows @ b_c
        sync()
        return out_x, out_c

    split_ms = best(split_run)
    return {"n": n, "xpu_share": xpu_share, "xpu_full_ms": xpu_full, "cpu_full_ms": cpu_full,
            "split_ms": split_ms, "gain_vs_xpu_pct": (1 - split_ms / xpu_full) * 100}


def copy_overlap(chunks=8, mib=32):
    elem = mib * 1024 * 1024 // 4
    host_a = torch.empty(elem, dtype=torch.float32, pin_memory=True)
    host_b = torch.empty(elem, dtype=torch.float32, pin_memory=True)
    dev_a = torch.empty(elem, dtype=torch.float32, device=DEV)
    dev_b = torch.empty(elem, dtype=torch.float32, device=DEV)

    def compute(buf):
        return (buf * 2 + 1).sum()

    for _ in range(2):
        dev_a.copy_(host_a, non_blocking=True)
        compute(dev_a)
    sync()

    def sequential():
        for _ in range(chunks):
            dev_a.copy_(host_a, non_blocking=True)
            compute(dev_a)
        sync()

    def overlapped():
        for i in range(chunks):
            src, dst = (host_a, dev_b) if i % 2 else (host_b, dev_a)
            cur = dev_b if i % 2 else dev_a
            dst.copy_(src, non_blocking=True)
            compute(cur)
        sync()

    seq_ms = best(sequential, warmup=1, iters=3)
    ovl_ms = best(overlapped, warmup=1, iters=3)
    return {"chunks": chunks, "mib_per_chunk": mib, "sequential_ms": seq_ms,
            "overlapped_ms": ovl_ms, "saving_pct": (1 - ovl_ms / seq_ms) * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="coop.json")
    args = ap.parse_args()

    out = {"device": torch.xpu.get_device_name(0), "torch": torch.__version__,
           "cpu_capability": torch.backends.cpu.get_cpu_capability()}
    for name, fn in (("pipeline", pipeline_demo),
                     ("split_gemm_2048", lambda: split_gemm(2048)),
                     ("split_gemm_4096", lambda: split_gemm(4096)),
                     ("copy_overlap", copy_overlap)):
        try:
            out[name] = fn()
            print(f"{name}: {json.dumps(out[name])}", flush=True)
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{name}: FAILED {out[name]['error']}", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
