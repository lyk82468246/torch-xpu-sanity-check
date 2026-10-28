#!/usr/bin/env python
"""First-call JIT compilation cost per operator.

The XPU backend compiles kernels on first use. On unsupported Intel GPUs this
compilation is extremely slow (tens of seconds to minutes per kernel) and is not
reused across processes, so the first run of a short script can be far slower
than plain CPU execution.

Usage:
    python diagnostics/first_call_cost.py
"""

import time

import torch

DEV = torch.device("xpu")


def timed(label, fn):
    torch.xpu.synchronize()
    start = time.perf_counter()
    fn()
    torch.xpu.synchronize()
    return time.perf_counter() - start


def repeat_cost(fn, iters=5):
    times = []
    for _ in range(iters):
        torch.xpu.synchronize()
        start = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        times.append(time.perf_counter() - start)
    return times


def main():
    print(f"torch {torch.__version__} | {torch.xpu.get_device_name(0)}")
    print()
    print("first call vs steady state, 2048x2048 fp32 tensors")
    print(f"{'op':>12} {'first call':>12} {'best repeat':>12}")

    a = torch.randn(2048, 2048, device=DEV)
    b = torch.randn(2048, 2048, device=DEV)
    ops = {
        "add": lambda: a + b,
        "mul": lambda: a * b,
        "sin": lambda: a.sin(),
        "exp": lambda: a.exp(),
        "tanh": lambda: a.tanh(),
        "sum": lambda: a.sum(),
        "mean": lambda: a.mean(),
        "matmul": lambda: a @ b,
        "softmax": lambda: torch.nn.functional.softmax(a, -1),
    }

    for name, fn in ops.items():
        first = timed(name, fn)
        repeats = repeat_cost(fn)
        print(f"{name:>12} {first:>11.2f}s {min(repeats) * 1000:>10.2f}ms")

    print()
    print("Conv / training path (first call only)")
    conv_first = timed("conv2d", lambda: torch.nn.functional.conv2d(
        torch.randn(4, 3, 128, 128, device=DEV), torch.randn(32, 3, 7, 7, device=DEV),
        stride=2, padding=3))
    print(f"  conv2d 4x3x128^2      {conv_first:>8.2f}s")

    lin = torch.nn.Linear(256, 256).to(DEV)
    opt = torch.optim.SGD(lin.parameters(), lr=0.01)
    x, y = torch.randn(32, 256, device=DEV), torch.randn(32, 256, device=DEV)

    def step():
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(lin(x), y)
        loss.backward()
        opt.step()

    first_step = timed("train step", step)
    steps = repeat_cost(step)
    print(f"  linear+backward+step  {first_step:>8.2f}s first, {min(steps) * 1000:.2f}ms after")


if __name__ == "__main__":
    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu.is_available() is False")
    main()
