#!/usr/bin/env python
"""Isolate indexing operators on XPU.

Motivation: during a full sequential run of ``run_checks.py`` the advanced
indexing check (``x[idx]``) blocked for ~450 s and then returned
``UR_RESULT_ERROR_UNKNOWN``; every later call in that process failed and the
device eventually reported ``UR_RESULT_ERROR_DEVICE_LOST``.

This script tests the indexing family one by one so the culprit is unambiguous.
The potentially hanging call is deliberately last -- if it wedges the device,
everything printed before it is still valid.

Usage:
    python diagnostics/indexing.py
"""

import time

import torch

DEV = torch.device("xpu")


def attempt(label, fn, timeout_note=""):
    torch.xpu.synchronize()
    start = time.perf_counter()
    try:
        out = fn()
        torch.xpu.synchronize()
        elapsed = time.perf_counter() - start
        detail = f"{tuple(out.shape)} {out.dtype}" if isinstance(out, torch.Tensor) else str(out)
        print(f"  OK   {label:34s} {elapsed:7.2f}s  {detail}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        print(f"  FAIL {label:34s} {elapsed:7.2f}s  {type(exc).__name__}: {str(exc)[:90]}", flush=True)
        return False


def main():
    print(f"torch {torch.__version__} | {torch.xpu.get_device_name(0)}")
    print()
    print("--- metadata / shape ops ---")
    x = torch.randn(4, 6, device=DEV)
    attempt("view", lambda: x.view(2, 2, 6))
    attempt("reshape (copy)", lambda: x.view(2, 12).reshape(12, 2))
    attempt("transpose", lambda: x.transpose(0, 1))
    attempt("permute", lambda: x.permute(1, 0))
    attempt("contiguous", lambda: x.transpose(0, 1).contiguous())
    attempt("cat", lambda: torch.cat([x, x], 0))
    attempt("stack", lambda: torch.stack([x, x], 0))
    attempt("split", lambda: x.split(2, 0)[0])
    attempt("narrow / slice", lambda: x[:, 1:3])

    print()
    print("--- gather family ---")
    src = torch.arange(24, device=DEV, dtype=torch.float32).reshape(4, 6)
    idx_2d = torch.tensor([[0, 1], [2, 3], [4, 5], [0, 1]], device=DEV)
    attempt("index_select(dim=1)", lambda: src.index_select(1, torch.tensor([1, 3], device=DEV)))
    attempt("gather(dim=1)", lambda: src.gather(1, idx_2d))
    attempt("take", lambda: src.take(torch.tensor([0, 5, 10], device=DEV)))
    attempt("masked_select", lambda: src.masked_select(src > 12))
    flat = torch.arange(12, device=DEV, dtype=torch.float32)
    attempt("index_put_", lambda: flat.clone().index_put_(
        (torch.tensor([0, 5], device=DEV),), torch.tensor([1.0, 2.0], device=DEV)))

    print()
    print("--- advanced indexing (this is the one that hung) ---")
    small = torch.arange(10, device=DEV)
    attempt("x[idx] on 1-D, int64 idx", lambda: small[torch.tensor([1, 5, 9], device=DEV)])
    attempt("x[idx] on 2-D, int64 idx", lambda: torch.randn(8, 8, device=DEV)[
        torch.tensor([0, 2], device=DEV)])

    print()
    print("--- scatter / where ---")
    attempt("where", lambda: torch.where(torch.randn(8, device=DEV) > 0,
                                         torch.ones(8, device=DEV), torch.zeros(8, device=DEV)))
    attempt("scatter", lambda: torch.zeros(4, 4, device=DEV).scatter(
        1, torch.tensor([[0], [1], [2], [3]], device=DEV), 1.0))


if __name__ == "__main__":
    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu.is_available() is False")
    main()
