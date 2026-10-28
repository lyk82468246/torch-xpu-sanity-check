#!/usr/bin/env python
"""Numerical accuracy of XPU kernels against a float64 CPU reference.

Tensors are created or converted on the CPU and then uploaded, because on-device
dtype casts are broken on some Intel iGPUs (see diagnostics/cast_workaround.py).

Usage:
    python diagnostics/numerics.py
"""

import torch

DEV = torch.device("xpu")


def relative_error(got, ref):
    return ((got - ref).norm() / ref.norm()).item()


def matmul_accuracy():
    print("matmul relative error vs float64 CPU reference")
    print(f"{'dtype':>10} {'size':>6} {'rel_err':>11} {'max_abs_err':>13} {'ref_max':>9}")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for n in (256, 512, 1024):
            a = torch.randn(n, n)
            b = torch.randn(n, n)
            ref = a.double() @ b.double()
            got = ((a.to(dtype).to(DEV)) @ (b.to(dtype).to(DEV))).cpu().double()
            print(f"{str(dtype).replace('torch.', ''):>10} {n:>6} {relative_error(got, ref):>11.3e} "
                  f"{(got - ref).abs().max().item():>13.4f} {ref.abs().max().item():>9.2f}")


def conv_accuracy():
    print()
    print("conv2d relative error vs float64 CPU reference")
    x = torch.randn(4, 3, 64, 64)
    w = torch.randn(16, 3, 3, 3)
    ref = torch.nn.functional.conv2d(x, w, padding=1).double()
    for dtype in (torch.float32, torch.float16):
        got = torch.nn.functional.conv2d(x.to(dtype).to(DEV), w.to(dtype).to(DEV), padding=1).cpu().double()
        print(f"  {str(dtype).replace('torch.', ''):>9}: rel_err={relative_error(got, ref):.3e} "
              f"max_abs={(got - ref).abs().max().item():.4f}")


def layer_accuracy():
    print()
    print("nn.Linear relative error vs float64 CPU reference")
    torch.manual_seed(0)
    layer = torch.nn.Linear(512, 512)
    x = torch.randn(64, 512)
    ref = x.double() @ layer.weight.double().T + layer.bias.double()
    got = layer.to(DEV)(x.to(DEV)).cpu().double()
    print(f"  fp32: rel_err={relative_error(got, ref):.3e}")


if __name__ == "__main__":
    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu.is_available() is False")
    torch.manual_seed(0)
    print(f"torch {torch.__version__} | {torch.xpu.get_device_name(0)}")
    print()
    matmul_accuracy()
    conv_accuracy()
    layer_accuracy()
