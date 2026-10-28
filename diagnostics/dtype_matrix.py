#!/usr/bin/env python
"""Dtype / operator support matrix for a PyTorch XPU device.

Important: a failed kernel can poison the XPU context on some drivers, making
every later op fail with UR_RESULT_ERROR_INVALID_ARGUMENT (error 45). Run each
section in a fresh process (see --case) if you need clean results.

Usage:
    python diagnostics/dtype_matrix.py            # everything, one process
    python diagnostics/dtype_matrix.py cast       # one isolated section
"""

import argparse

import torch

DEV = torch.device("xpu")


def report(label, fn):
    try:
        out = fn()
        torch.xpu.synchronize()
        detail = f"-> {tuple(out.shape)} {out.dtype}" if isinstance(out, torch.Tensor) else ""
        print(f"  OK   {label:42s} {detail}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {label:42s} {type(exc).__name__}: {str(exc)[:100]}")
        return False


def section_int():
    print("--- integer / bool arithmetic ---")
    for dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
        a = torch.ones(64, 64, device=DEV, dtype=dtype)
        b = torch.ones(64, 64, device=DEV, dtype=dtype)
        report(f"{str(dtype).replace('torch.', '')} add", lambda a=a, b=b: a + b)
    for dtype in (torch.int8, torch.int32, torch.int64):
        a = torch.ones(64, 64, device=DEV, dtype=dtype)
        report(f"{str(dtype).replace('torch.', '')} matmul", lambda a=a: a.matmul(a))


def section_f16():
    print("--- float16, tensors created directly on device ---")
    x = torch.randn(256, 256, device=DEV, dtype=torch.float16)
    y = torch.randn(256, 256, device=DEV, dtype=torch.float16)
    report("mul", lambda: x * y)
    report("add scalar", lambda: x + 1)
    report("relu", lambda: x.relu())
    report("exp", lambda: x.exp())
    report("sum", lambda: x.sum())
    report("matmul 256^3", lambda: x @ y)
    report("conv2d", lambda: torch.nn.functional.conv2d(
        torch.randn(2, 3, 32, 32, device=DEV, dtype=torch.float16),
        torch.randn(8, 3, 3, 3, device=DEV, dtype=torch.float16), padding=1))


def section_bf16():
    print("--- bfloat16, tensors created directly on device ---")
    x = torch.randn(256, 256, device=DEV, dtype=torch.bfloat16)
    y = torch.randn(256, 256, device=DEV, dtype=torch.bfloat16)
    report("mul", lambda: x * y)
    report("add scalar", lambda: x + 1)
    report("exp", lambda: x.exp())
    report("sum", lambda: x.sum())
    report("matmul 256^3", lambda: x @ y)


def section_cast():
    print("--- cross-dtype conversions (known to break on some iGPUs) ---")
    f32 = torch.randn(256, 256, device=DEV)
    f16 = torch.randn(256, 256, device=DEV, dtype=torch.float16)
    bf16 = torch.randn(256, 256, device=DEV, dtype=torch.bfloat16)
    report("f32 -> f16 (.half)", lambda: f32.half())
    report("f16 -> f32 (.float)", lambda: f16.float())
    report("f32 -> bf16 (.bfloat16)", lambda: f32.bfloat16())
    report("bf16 -> f32 (.float)", lambda: bf16.float())
    report("f32 -> f32 (no-op .to)", lambda: f32.to(torch.float32))
    report("int32 -> int64 (.to)", lambda: torch.ones(8, 8, device=DEV, dtype=torch.int32).to(torch.int64))


def section_f64():
    print("--- float64 ---")
    report("f64 create + matmul", lambda: torch.randn(64, 64, device=DEV, dtype=torch.float64).matmul(
        torch.randn(64, 64, device=DEV, dtype=torch.float64)))


def section_matmul_sizes():
    print("--- fp32 matmul sizes ---")
    for n in (256, 512, 1024, 2048, 4096):
        a = torch.randn(n, n, device=DEV)
        b = torch.randn(n, n, device=DEV)
        report(f"matmul {n}^3", lambda a=a, b=b: a @ b)


SECTIONS = {
    "int": section_int,
    "f16": section_f16,
    "bf16": section_bf16,
    "cast": section_cast,
    "f64": section_f64,
    "mm": section_matmul_sizes,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default="all", choices=["all", *SECTIONS])
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu.is_available() is False")
    print(f"torch {torch.__version__} | {torch.xpu.get_device_name(0)}")
    print()

    for name, fn in SECTIONS.items():
        if args.case in ("all", name):
            fn()
            print()


if __name__ == "__main__":
    main()
