#!/usr/bin/env python
"""Steady-state throughput benchmark: XPU vs CPU on the same machine.

Measures fp32 matmul (TFLOPS), elementwise memory bandwidth (GiB/s) and
host<->device copy bandwidth. Warm-up iterations are discarded so JIT
compilation cost does not pollute the numbers.

Usage:
    python benchmarks/bench_matmul.py
"""

import time

import torch

DEV = torch.device("xpu")


def bench_xpu(fn, warmup=2, iters=8):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - start) / iters


def bench_cpu(fn, warmup=1, iters=3):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters


def matmul_table(sizes=(512, 1024, 2048, 4096)):
    print(f"torch {torch.__version__} | {torch.xpu.get_device_name(0)}")
    print()
    print("fp32 matmul (TFLOPS = 2*n^3 / time)")
    print(f"{'size':>8} {'XPU ms':>10} {'XPU TFLOPS':>11} {'CPU ms':>10} {'CPU TFLOPS':>11} {'speedup':>8}")
    rows = []
    for n in sizes:
        a_dev = torch.randn(n, n, device=DEV)
        b_dev = torch.randn(n, n, device=DEV)
        a_cpu, b_cpu = a_dev.cpu(), b_dev.cpu()
        xpu_s = bench_xpu(lambda: a_dev @ b_dev, warmup=3, iters=10)
        cpu_s = bench_cpu(lambda: a_cpu @ b_cpu)
        flops = 2 * n ** 3
        rows.append((n, xpu_s, cpu_s))
        print(f"{n:>8} {xpu_s * 1000:>10.2f} {flops / xpu_s / 1e12:>11.3f} "
              f"{cpu_s * 1000:>10.2f} {flops / cpu_s / 1e12:>11.3f} {cpu_s / xpu_s:>7.2f}x")
        del a_dev, b_dev, a_cpu, b_cpu
        torch.xpu.empty_cache()
    return rows


def elementwise_table(sizes=(2048, 4096, 8192)):
    print()
    print("elementwise fp32: (x*2+1).relu().sum()  [4 passes of m*m*4 bytes]")
    print(f"{'size':>8} {'XPU ms':>10} {'XPU GiB/s':>10} {'CPU ms':>10} {'CPU GiB/s':>10} {'speedup':>8}")
    for m in sizes:
        x_dev = torch.randn(m, m, device=DEV)
        x_cpu = x_dev.cpu()
        gib = m * m * 4 * 4 / 2**30
        xpu_s = bench_xpu(lambda: (x_dev * 2 + 1).relu().sum(), warmup=2, iters=10)
        cpu_s = bench_cpu(lambda: (x_cpu * 2 + 1).relu().sum(), warmup=1, iters=5)
        print(f"{m:>8} {xpu_s * 1000:>10.2f} {gib / xpu_s:>10.2f} "
              f"{cpu_s * 1000:>10.2f} {gib / cpu_s:>10.2f} {cpu_s / xpu_s:>7.2f}x")
        del x_dev, x_cpu
        torch.xpu.empty_cache()


def copy_bandwidth(mib=256):
    print()
    print(f"host<->device copy bandwidth ({mib} MiB, pinned host buffer)")
    elements = mib * 1024 * 1024 // 4
    dev_buf = torch.empty(elements, dtype=torch.float32, device=DEV)
    host_buf = torch.empty(elements, dtype=torch.float32, pin_memory=True)

    def h2d():
        dev_buf.copy_(host_buf, non_blocking=True)

    def d2h():
        host_buf.copy_(dev_buf, non_blocking=True)

    h2d()
    torch.xpu.synchronize()
    h2d_s = bench_xpu(h2d, warmup=1, iters=5)
    d2h_s = bench_xpu(d2h, warmup=1, iters=5)
    print(f"  H2D: {h2d_s * 1000:7.1f} ms  ({mib / h2d_s / 1024:6.2f} GiB/s)")
    print(f"  D2H: {d2h_s * 1000:7.1f} ms  ({mib / d2h_s / 1024:6.2f} GiB/s)")


if __name__ == "__main__":
    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu.is_available() is False -- no XPU device found")
    matmul_table()
    elementwise_table()
    copy_bandwidth()
