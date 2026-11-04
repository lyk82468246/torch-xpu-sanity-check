#!/usr/bin/env python
"""Dispatch overhead, thread scaling and CPU-only workloads.

Explains the crossover between CPU and XPU on small work:

* the XPU costs ~0.5 ms per operator when you synchronise after each one, and
  ~0.02 ms when you just queue them
* a tiny model therefore runs 4-6x faster on the CPU
* fp64, integer matmul and dtype casts cannot run on the XPU at all

Usage: python benchmarks/bench_dispatch.py [--out dispatch.json]
"""

import argparse
import json
import time

import torch

DEV = torch.device("xpu")
torch.set_num_threads(4)


def sync():
    torch.xpu.synchronize()


def bench(fn, warmup=2, iters=10, device="cpu"):
    for _ in range(warmup):
        fn()
    if device == "xpu":
        sync()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device == "xpu":
            sync()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def thread_scaling():
    print("--- CPU thread scaling, fp32 matmul (ms) ---")
    rows = {}
    for n in (512, 1024, 2048, 4096):
        a, b = torch.randn(n, n), torch.randn(n, n)
        per_threads = {}
        for threads in (4, 8):
            torch.set_num_threads(threads)
            per_threads[threads] = round(bench(lambda: a @ b, warmup=1, iters=5 if n < 4096 else 3), 2)
        torch.set_num_threads(4)
        rows[n] = per_threads
        print(f"  {n:>5}^3   4 threads {per_threads[4]:>8.2f}   8 threads {per_threads[8]:>8.2f}   "
              f"x{per_threads[4] / per_threads[8]:.2f}")
    return rows


def cpu_only_ops():
    print("--- workloads the XPU cannot run ---")
    rows = {}
    a64 = torch.randn(1024, 1024, dtype=torch.float64)
    rows["fp64_matmul_1024_ms"] = round(bench(lambda: a64 @ a64, warmup=1, iters=5), 2)
    print(f"  fp64 matmul 1024^3     {rows['fp64_matmul_1024_ms']:>8.2f} ms   (XPU: no fp64 support)")

    i32 = torch.randint(0, 100, (512, 512), dtype=torch.int32)
    rows["int32_matmul_512_ms"] = round(bench(lambda: i32 @ i32, warmup=1, iters=5), 2)
    print(f"  int32 matmul 512^3     {rows['int32_matmul_512_ms']:>8.2f} ms   (XPU: oneDNN cannot build it)")

    w768 = torch.randn(768, 768)
    rows["cast_768_ms"] = round(bench(lambda: w768.half(), warmup=2, iters=10), 3)
    print(f"  fp32->fp16 768^2       {rows['cast_768_ms']:>8.3f} ms   (XPU: cast raises error 45)")

    big = torch.randn(4096, 4096)
    rows["cast_4096_ms"] = round(bench(lambda: big.half(), warmup=1, iters=5), 2)
    print(f"  fp32->fp16 4096^2      {rows['cast_4096_ms']:>8.2f} ms")
    return rows


def dispatch_cost():
    print("--- per-operator cost, tiny op (64x64 add) ---")
    x = torch.randn(64, 64, device=DEV)
    for _ in range(10):
        x + 1
    sync()
    reps = 200

    t0 = time.perf_counter()
    for _ in range(reps):
        x + 1
        sync()
    sync_per_op = (time.perf_counter() - t0) / reps * 1000

    t0 = time.perf_counter()
    for _ in range(reps):
        x + 1
    sync()
    queued_per_op = (time.perf_counter() - t0) / reps * 1000

    print(f"  synchronise each op : {sync_per_op:.3f} ms/op")
    print(f"  queue and sync once : {queued_per_op:.3f} ms/op")
    return {"sync_ms_per_op": round(sync_per_op, 3), "queued_ms_per_op": round(queued_per_op, 3)}


def tiny_inference():
    print("--- tiny MLP inference (128->128->128->10), ms per forward ---")
    rows = {}
    for batch in (1, 8, 64, 256):
        net_c = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 10))
        net_x = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 10)).to(DEV)
        net_x.load_state_dict(net_c.state_dict())
        x_c = torch.randn(batch, 128)
        x_x = x_c.to(DEV)
        with torch.no_grad():
            for _ in range(5):
                net_c(x_c)
            for _ in range(5):
                net_x(x_x)
            sync()
            iters = 50 if batch <= 8 else 20
            cpu_ms = bench(lambda: net_c(x_c), warmup=2, iters=iters, device="cpu")
            xpu_ms = bench(lambda: net_x(x_x), warmup=2, iters=iters, device="xpu")
        rows[batch] = {"cpu_ms": round(cpu_ms, 3), "xpu_ms": round(xpu_ms, 3)}
        print(f"  batch {batch:>3}: cpu {cpu_ms:>7.3f} ms | xpu {xpu_ms:>7.3f} ms "
              f"-> CPU x{cpu_ms / xpu_ms if xpu_ms < cpu_ms else xpu_ms / cpu_ms:.2f}"
              f"{' faster' if xpu_ms > cpu_ms else ' slower'}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dispatch.json")
    args = ap.parse_args()
    out = {"cpu_capability": torch.backends.cpu.get_cpu_capability(),
           "device": torch.xpu.get_device_name(0)}
    out["thread_scaling"] = thread_scaling()
    out["cpu_only"] = cpu_only_ops()
    out["dispatch"] = dispatch_cost()
    out["tiny_inference"] = tiny_inference()
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
