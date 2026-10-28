#!/usr/bin/env python
"""Workaround probe: on-device dtype casts fail, so cast on the CPU instead.

On Intel Iris Xe (and possibly other unsupported iGPUs) every on-device
conversion kernel -- ``.half()``, ``.to(torch.float16)``, ``copy_``,
``aten._to_copy``, ``aten.copy`` -- fails with::

    RuntimeError: level_zero backend failed with error: 45
    (UR_RESULT_ERROR_INVALID_ARGUMENT)

That makes ``model.half()``, ``loss.float()`` and ``torch.autocast("xpu", ...)``
unusable. Casting on the host and uploading the result works and is cheap.

Usage:
    python diagnostics/cast_workaround.py
"""

import copy
import time

import torch

DEV = torch.device("xpu")


def attempt(label, fn):
    try:
        torch.xpu.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.xpu.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        detail = f"{out.dtype}" if isinstance(out, torch.Tensor) else str(out)
        print(f"  OK   {label:46s} {elapsed:7.2f} ms  {detail}")
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {label:46s} {str(exc)[:90]}")
        return None


def on_device_casts():
    print("--- on-device f32 -> f16 conversions (expected to fail on affected GPUs) ---")
    x = torch.randn(1024, 1024, device=DEV)
    attempt(".half()", lambda: x.half())
    attempt(".to(torch.float16)", lambda: x.to(torch.float16))
    attempt("empty_like(f16).copy_(x)", lambda: torch.empty_like(x, dtype=torch.float16).copy_(x))
    attempt("aten._to_copy(x, dtype=f16)", lambda: torch.ops.aten._to_copy(x, dtype=torch.float16))
    attempt("aten.copy(f16, x, non_blocking=True)", lambda: torch.ops.aten.copy.default(
        torch.empty(1024, 1024, dtype=torch.float16, device=DEV), x, True))


def cpu_side_cast():
    print()
    print("--- CPU-side cast then upload (the workaround) ---")
    x = torch.randn(1024, 1024, device=DEV)
    attempt("cpu().half().to('xpu')", lambda: x.cpu().half().to(DEV))
    start = time.perf_counter()
    for _ in range(5):
        x.cpu().half().to(DEV)
        torch.xpu.synchronize()
    print(f"  round-trip cost for 1024^2: {(time.perf_counter() - start) / 5 * 1000:.2f} ms")

    print()
    print("--- fp16 inference through the workaround ---")
    torch.manual_seed(0)
    lin32 = torch.nn.Linear(256, 256).to(DEV)
    inputs = torch.randn(32, 256, device=DEV)
    ref = lin32(inputs).cpu()

    # same weights, converted on the host: .half() on the device raises
    lin16 = copy.deepcopy(lin32).cpu().half().to(DEV)
    x16 = inputs.cpu().half().to(DEV)
    out16 = lin16(x16).cpu().float()
    torch.xpu.synchronize()
    rel = ((out16 - ref).norm() / ref.norm()).item()
    print(f"  fp16 forward OK, relative error vs fp32 = {rel:.3e}")

    print()
    print("--- autocast depends on on-device casts (expected to fail) ---")
    attempt('autocast("xpu", bfloat16)', lambda: _autocast_probe(lin32, inputs))


def _autocast_probe(layer, inputs):
    with torch.autocast("xpu", dtype=torch.bfloat16):
        return layer(inputs)


if __name__ == "__main__":
    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu.is_available() is False")
    print(f"torch {torch.__version__} | {torch.xpu.get_device_name(0)}")
    print()
    on_device_casts()
    cpu_side_cast()
