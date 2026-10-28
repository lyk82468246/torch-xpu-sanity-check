#!/usr/bin/env python
"""Staged functional sanity checks for PyTorch XPU.

Usage:
    python run_checks.py            # run every stage
    python run_checks.py info       # single stage: info | elem | matmul | nn | mem

Notes
-----
* Every check is timed and a failure never aborts the run, so a single broken
  kernel does not hide the rest of the results.
* On unsupported Intel GPUs (e.g. Iris Xe) the **first** call to an operator can
  take tens of seconds because of JIT compilation. That time is reported per
  check -- it is not a steady-state measurement. Use benchmarks/bench_matmul.py
  for performance numbers.
* Integer casts and float16/bfloat16 casts are known to fail on some iGPUs
  (UR_RESULT_ERROR_INVALID_ARGUMENT). See diagnostics/cast_workaround.py.
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore")
import torch

DEV = torch.device("xpu")
RESULTS = []


def check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        RESULTS.append((True, name, detail))
        print(f"[ PASS ] {name}  ({time.time() - t0:.1f}s)" + (f"  -> {detail}" if detail else ""), flush=True)
    except Exception as exc:  # noqa: BLE001 - report anything the backend throws
        RESULTS.append((False, name, str(exc)))
        print(f"[ FAIL ] {name}  ({time.time() - t0:.1f}s)  -> {type(exc).__name__}: {exc}", flush=True)


def stage_info():
    props = torch.xpu.get_device_properties(0)
    print("=" * 72, flush=True)
    print("STAGE info", flush=True)
    print(f"torch {torch.__version__} | xpu build {torch.version.xpu} | "
          f"available={torch.xpu.is_available()} | devices={torch.xpu.device_count()}", flush=True)
    print(f"device : {props.name}", flush=True)
    print(f"driver : {props.driver_version} | platform {props.platform_name}", flush=True)
    print(f"memory : {props.total_memory / 2**30:.1f} GiB | EUs {props.gpu_eu_count} | "
          f"subslices {props.gpu_subslice_count} | local mem {props.local_mem_size // 1024} KiB", flush=True)
    print(f"flags  : fp16={props.has_fp16} fp64={props.has_fp64} atomic64={props.has_atomic64} "
          f"dpas={props.has_subgroup_matrix_multiply_accumulate}", flush=True)

    check("create tensor on xpu", lambda: f"sum={torch.ones(4, 4, device=DEV).sum().item()}")
    check("cpu -> xpu -> cpu round trip", lambda: float(torch.arange(8, dtype=torch.float32).to(DEV).sum().cpu()))
    check("explicit .xpu() / .to('xpu:0')",
          lambda: f"{torch.zeros(2, 2).xpu().device}, {torch.zeros(2, 2).to('xpu:0').device}")

    def pinned():
        host = torch.zeros(16, pin_memory=True)
        dev = torch.empty(16, device=DEV).copy_(host, non_blocking=True)
        back = dev.to("cpu", non_blocking=True)
        torch.xpu.synchronize()
        return f"pinned async copy ok, sum={back.sum().item()}"

    check("pinned-memory async copy", pinned)
    check("dtype creation coverage", lambda: ", ".join(
        f"{str(d).replace('torch.', '')}={torch.zeros(4, dtype=d, device=DEV).to('cpu').dtype}"
        for d in (torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.int64, torch.bool)))


def stage_elem():
    print("=" * 72, flush=True)
    print("STAGE elem", flush=True)

    def elementwise():
        a = torch.randn(2048, 2048, device=DEV)
        b = torch.randn(2048, 2048, device=DEV)
        c = (a.sin() * b.exp().clamp(max=10) + a.tanh()).mean()
        torch.xpu.synchronize()
        return f"mean={c.item():.4f} finite={bool(torch.isfinite(c))}"

    check("elementwise math 2048^2", elementwise)

    def reductions():
        x = torch.randn(1 << 21, device=DEV)
        return (f"sum={x.sum().item():.2f} mean={x.mean().item():.4f} max={x.max().item():.3f} "
                f"std={x.std().item():.4f} argmax={int(x.argmax())}")

    check("reductions 2M elements", reductions)
    check("sort / topk", lambda: f"max={torch.sort(torch.randn(1000, device=DEV)).values[-1].item():.4f}")
    # Indexing kernels (index / index_select / gather) are excluded on purpose:
    # on some iGPUs they take 400+ seconds to compile and can lose the device.
    # Use diagnostics/indexing.py to test that family in isolation.
    check("view / reshape / transpose / permute", lambda: str(
        torch.randn(4, 6, device=DEV).view(2, 2, 6).transpose(1, 2).permute(2, 0, 1).shape))
    check("cat / stack / split", lambda: f"{torch.cat([torch.zeros(2, 2, device=DEV)] * 3).shape}")


def stage_matmul():
    print("=" * 72, flush=True)
    print("STAGE matmul", flush=True)

    def mm(dtype, n):
        a = torch.randn(n, n, device=DEV, dtype=dtype)
        b = torch.randn(n, n, device=DEV, dtype=dtype)
        out = a @ b
        torch.xpu.synchronize()
        assert torch.isfinite(out.float()).all(), "non-finite result"
        return f"{n}^3 ok, finite"

    for n in (256, 512, 1024, 2048):
        check(f"matmul fp32 {n}^3", lambda n=n: mm(torch.float32, n))

    def correctness():
        torch.manual_seed(0)
        a, b = torch.randn(256, 256), torch.randn(256, 256)
        ref = a @ b
        got = (a.to(DEV) @ b.to(DEV)).to("cpu")
        err = (ref - got).abs().max().item()
        return f"fp32 max|err| vs CPU = {err:.2e}"

    check("numerical correctness fp32 vs CPU", correctness)


def stage_nn():
    print("=" * 72, flush=True)
    print("STAGE nn", flush=True)

    def conv():
        x = torch.randn(4, 3, 128, 128, device=DEV)
        w = torch.randn(32, 3, 7, 7, device=DEV)
        y = torch.nn.functional.conv2d(x, w, stride=2, padding=3)
        torch.xpu.synchronize()
        return f"out={tuple(y.shape)} finite={bool(torch.isfinite(y).all())}"

    check("conv2d (7x7 stride 2)", conv)

    def bn_pool():
        x = torch.randn(8, 16, 56, 56, device=DEV)
        net = torch.nn.Sequential(torch.nn.BatchNorm2d(16), torch.nn.ReLU(), torch.nn.MaxPool2d(2)).to(DEV)
        return f"batchnorm+relu+maxpool -> {tuple(net(x).shape)}"

    check("batchnorm / relu / maxpool", bn_pool)

    def attention_ops():
        baddbmm = torch.baddbmm(torch.randn(8, 128, 64, device=DEV),
                                torch.randn(8, 128, 64, device=DEV),
                                torch.randn(8, 64, 64, device=DEV))
        sdpa = torch.nn.functional.scaled_dot_product_attention(
            torch.randn(2, 4, 128, 64, device=DEV),
            torch.randn(2, 4, 128, 64, device=DEV),
            torch.randn(2, 4, 128, 64, device=DEV))
        ln = torch.nn.functional.layer_norm(torch.randn(8, 128, 256, device=DEV), (256,))
        torch.xpu.synchronize()
        return f"baddbmm {tuple(baddbmm.shape)}, sdpa {tuple(sdpa.shape)}, layernorm {tuple(ln.shape)}"

    check("baddbmm / SDPA / LayerNorm", attention_ops)

    def autograd_step():
        lin = torch.nn.Linear(256, 256).to(DEV)
        opt = torch.optim.SGD(lin.parameters(), lr=0.01)
        x, y = torch.randn(32, 256, device=DEV), torch.randn(32, 256, device=DEV)
        before = lin.weight.detach().clone()
        for _ in range(3):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(lin(x), y)
            loss.backward()
            opt.step()
        torch.xpu.synchronize()
        delta = (lin.weight.detach() - before).abs().max().item()
        assert delta > 0, "weights did not update"
        return f"loss={loss.item():.4f} |dW|max={delta:.2e}"

    check("autograd + SGD step", autograd_step)

    def cnn():
        torch.manual_seed(1)
        net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(16, 10),
        ).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        x = torch.randn(16, 1, 28, 28, device=DEV)
        y = torch.randint(0, 10, (16,), device=DEV)
        history = []
        for _ in range(5):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(x), y)
            loss.backward()
            opt.step()
            history.append(loss.item())
        torch.xpu.synchronize()
        return f"loss {history[0]:.3f} -> {history[-1]:.3f} over 5 steps"

    check("mini CNN training loop", cnn)


def stage_mem():
    print("=" * 72, flush=True)
    print("STAGE mem", flush=True)

    def big_alloc():
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats()
        buf = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)
        buf.fill_(1.0)
        torch.xpu.synchronize()
        allocated = torch.xpu.memory_allocated() / 2**20
        peak = torch.xpu.max_memory_allocated() / 2**20
        first = float(buf[0].item())
        del buf
        torch.xpu.empty_cache()
        return (f"256 MiB fill ok (value={first}), allocated={allocated:.0f} MiB, "
                f"peak={peak:.0f} MiB, after free={torch.xpu.memory_allocated() / 2**20:.0f} MiB")

    check("256 MiB allocation + fill + free", big_alloc)

    def streams():
        stream = torch.xpu.Stream()
        a = torch.randn(512, 512, device=DEV)
        with torch.xpu.stream(stream):
            b = a * 2
        stream.synchronize()
        assert torch.allclose(b, a * 2)
        return "Stream create / use / synchronize ok"

    check("XPU streams", streams)

    def rng():
        torch.xpu.manual_seed(1234)
        a = torch.randn(8, device=DEV)
        torch.xpu.manual_seed(1234)
        b = torch.randn(8, device=DEV)
        assert torch.equal(a, b)
        return "same seed -> identical values"

    check("RNG reproducibility", rng)
    check("synchronize / empty_cache", lambda: (torch.xpu.synchronize(), torch.xpu.empty_cache(), "ok")[2])


STAGES = {"info": stage_info, "elem": stage_elem, "matmul": stage_matmul, "nn": stage_nn, "mem": stage_mem}


def run_isolated():
    """Run every stage in its own process.

    One failing kernel can leave the XPU context unusable (later calls return
    UR_RESULT_ERROR_* until the interpreter restarts), so stage isolation is the
    only way to get a complete picture in a single invocation.
    """
    import os
    import subprocess
    import sys

    failures = []
    for name in STAGES:
        print(flush=True)
        print("#" * 72, flush=True)
        print(f"# stage: {name}  (isolated process)", flush=True)
        print("#" * 72, flush=True)
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), name])
        if proc.returncode != 0:
            failures.append(name)

    print(flush=True)
    print("=" * 72, flush=True)
    ok = len(STAGES) - len(failures)
    print(f"STAGE SUMMARY: {ok}/{len(STAGES)} stages fully passed", flush=True)
    for name in failures:
        print(f"  STAGE FAILED: {name}", flush=True)
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description="PyTorch XPU functional sanity checks")
    parser.add_argument("stage", nargs="?", default="all", choices=["all", *STAGES],
                        help="which stage to run (default: all, each in its own process)")
    parser.add_argument("--no-isolate", action="store_true",
                        help="run every stage in this process (a single bad kernel poisons the rest)")
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu.is_available() is False -- nothing to test on this machine")

    if args.stage == "all" and not args.no_isolate:
        return run_isolated()

    selected = STAGES.values() if args.stage == "all" else [STAGES[args.stage]]
    t_start = time.time()
    for fn in selected:
        fn()

    passed = sum(1 for ok, _, _ in RESULTS if ok)
    print("=" * 72, flush=True)
    print(f"RESULT: {passed}/{len(RESULTS)} checks passed in {time.time() - t_start:.0f}s", flush=True)
    for ok, name, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} -> {detail}", flush=True)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
