#!/usr/bin/env python
"""CPU vs XPU throughput on identical workloads (steady state only).

Every XPU kernel is warmed up before it is timed, so JIT compilation never enters
the numbers -- enable the persistent cache (scripts/enable_jit_cache.ps1) and the
warmup is fast.

To compare AVX-512 against AVX2 on the same CPU, run the script once normally and
once with the wide ISA disabled:

    python benchmarks/bench_cpu_vs_xpu.py --out avx512.json
    ONEDNN_MAX_CPU_ISA=AVX2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
        python benchmarks/bench_cpu_vs_xpu.py --out avx2.json

(on PowerShell: `$env:ONEDNN_MAX_CPU_ISA="AVX2"; $env:MKL_ENABLE_INSTRUCTIONS="AVX2"`)
"""

import argparse
import json
import os
import time

import torch

torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "4")))


def sync():
    torch.xpu.synchronize()


def timeit(fn, warmup=3, iters=8, device="cpu"):
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


def case(name, cpu_fn, xpu_fn, iters=8, warmup=3):
    row = {"name": name, "cpu_ms": None, "xpu_ms": None, "speedup": None, "note": ""}
    try:
        row["cpu_ms"] = timeit(cpu_fn, warmup, iters, "cpu")
    except Exception as exc:  # noqa: BLE001
        row["note"] += f"cpu:{type(exc).__name__} "
    try:
        row["xpu_ms"] = timeit(xpu_fn, warmup, iters, "xpu")
    except Exception as exc:  # noqa: BLE001
        row["note"] += f"xpu:{type(exc).__name__}:{str(exc)[:40]} "
    if row["cpu_ms"] and row["xpu_ms"]:
        row["speedup"] = row["cpu_ms"] / row["xpu_ms"]
    print(f"{name:40s} cpu={row['cpu_ms'] if row['cpu_ms'] is None else round(row['cpu_ms'], 3):>10} "
          f"xpu={row['xpu_ms'] if row['xpu_ms'] is None else round(row['xpu_ms'], 3):>10} "
          f"speedup={row['speedup'] if row['speedup'] is None else round(row['speedup'], 2):>6} {row['note']}",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpu_vs_xpu.json")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    dev = torch.device("xpu")
    rows = []
    meta = {
        "tag": args.tag, "torch": torch.__version__,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "threads": torch.get_num_threads(),
        "onednn_max_isa": os.environ.get("ONEDNN_MAX_CPU_ISA"),
        "mkl_instructions": os.environ.get("MKL_ENABLE_INSTRUCTIONS"),
        "device": torch.xpu.get_device_name(0),
    }
    print(json.dumps(meta), flush=True)

    for n in (128, 256, 512, 1024, 2048, 4096):
        a_c, b_c = torch.randn(n, n), torch.randn(n, n)
        a_x, b_x = a_c.to(dev), b_c.to(dev)
        rows.append(case(f"matmul {n}^3",
                         lambda a_c=a_c, b_c=b_c: a_c @ b_c,
                         lambda a_x=a_x, b_x=b_x: a_x @ b_x,
                         iters=5 if n >= 2048 else 10))

    lin_c = torch.nn.Linear(768, 768)
    lin_x = torch.nn.Linear(768, 768).to(dev)
    lin_x.load_state_dict(lin_c.state_dict())
    x_c = torch.randn(128, 768)
    x_x = x_c.to(dev)
    rows.append(case("linear 768->768, batch 128", lambda: lin_c(x_c), lambda: lin_x(x_x)))

    mlp_c = torch.nn.Sequential(torch.nn.Linear(768, 768), torch.nn.GELU(), torch.nn.Linear(768, 768))
    mlp_x = torch.nn.Sequential(torch.nn.Linear(768, 768), torch.nn.GELU(), torch.nn.Linear(768, 768)).to(dev)
    mlp_x.load_state_dict(mlp_c.state_dict())
    opt_c = torch.optim.Adam(mlp_c.parameters(), lr=1e-4)
    opt_x = torch.optim.Adam(mlp_x.parameters(), lr=1e-4)

    def step_cpu():
        opt_c.zero_grad()
        mlp_c(x_c).sum().backward()
        opt_c.step()

    def step_xpu():
        opt_x.zero_grad()
        mlp_x(x_x).sum().backward()
        opt_x.step()

    rows.append(case("MLP 768x2 fwd+bwd+Adam, b=128", step_cpu, step_xpu, iters=6))

    for label, shape, kshape, kw in (
        ("conv2d 3x224x224 -> 64x7x7 s2", (1, 3, 224, 224), (64, 3, 7, 7), {"stride": 2, "padding": 3}),
        ("conv2d 8x64x56x56 -> 64x3x3", (8, 64, 56, 56), (64, 64, 3, 3), {"padding": 1}),
        ("conv2d 32x32x32x32 -> 32x3x3", (32, 32, 32, 32), (32, 32, 3, 3), {"padding": 1}),
    ):
        img_c, w_c = torch.randn(*shape), torch.randn(*kshape)
        img_x, w_x = img_c.to(dev), w_c.to(dev)
        rows.append(case(label,
                         lambda i=img_c, w=w_c, kw=kw: torch.nn.functional.conv2d(i, w, **kw),
                         lambda i=img_x, w=w_x, kw=kw: torch.nn.functional.conv2d(i, w, **kw), iters=6))

    for m in (1024, 4096):
        e_c = torch.randn(m, m)
        e_x = e_c.to(dev)
        rows.append(case(f"elementwise {m}^2 (x*2+1).relu().sum()",
                         lambda e_c=e_c: (e_c * 2 + 1).relu().sum(),
                         lambda e_x=e_x: (e_x * 2 + 1).relu().sum()))

    red_c = torch.randn(1 << 24)
    red_x = red_c.to(dev)
    rows.append(case("sum over 16M floats", lambda: red_c.sum(), lambda: red_x.sum()))

    t_c = torch.randn(32, 128, 768)
    t_x = t_c.to(dev)
    rows.append(case("softmax (32,128,768)", lambda: torch.softmax(t_c, -1), lambda: torch.softmax(t_x, -1)))
    rows.append(case("layernorm (32,128,768)",
                     lambda: torch.nn.functional.layer_norm(t_c, (768,)),
                     lambda: torch.nn.functional.layer_norm(t_x, (768,))))

    q_c, k_c, v_c = (torch.randn(8, 8, 256, 64) for _ in range(3))
    q_x, k_x, v_x = q_c.to(dev), k_c.to(dev), v_c.to(dev)
    rows.append(case("sdpa (8,8,256,64)",
                     lambda: torch.nn.functional.scaled_dot_product_attention(q_c, k_c, v_c),
                     lambda: torch.nn.functional.scaled_dot_product_attention(q_x, k_x, v_x), iters=6))

    tiny_c = torch.randn(16, 16)
    tiny_x = tiny_c.to(dev)
    rows.append(case("add 16x16 (dispatch overhead)",
                     lambda: tiny_c + 1, lambda: tiny_x + 1, iters=50, warmup=5))

    transfer = {}
    host = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.float32, pin_memory=True)
    dev_buf = torch.empty_like(host, device=dev)
    for _ in range(2):
        dev_buf.copy_(host, non_blocking=True)
    sync()
    t0 = time.perf_counter()
    for _ in range(5):
        dev_buf.copy_(host, non_blocking=True)
    sync()
    transfer["h2d_ms_64MiB"] = (time.perf_counter() - t0) / 5 * 1000
    for _ in range(2):
        host.copy_(dev_buf, non_blocking=True)
    sync()
    t0 = time.perf_counter()
    for _ in range(5):
        host.copy_(dev_buf, non_blocking=True)
    sync()
    transfer["d2h_ms_64MiB"] = (time.perf_counter() - t0) / 5 * 1000
    print(f"H2D 64MiB {transfer['h2d_ms_64MiB']:.2f} ms | D2H 64MiB {transfer['d2h_ms_64MiB']:.2f} ms",
          flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "rows": rows, "transfer": transfer}, fh, indent=1)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
