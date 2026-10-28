# torch-xpu-sanity-check

Functional, numerical and performance sanity checks for **PyTorch XPU** (Intel GPUs), plus a worked example of hardware that is *technically* supported by the build but **not** by the vendor's list: an Intel Iris Xe integrated GPU.

The point of this repo is not to benchmark hardware — it is to answer "does `torch.xpu` actually work on this box, and where does it break?" with reproducible scripts and the raw output they produced.

- [What it checks](#what-it-checks)
- [Quickstart](#quickstart)
- [Findings from the reference machine](#findings-from-the-reference-machine)
- [Known limitations on unsupported iGPUs](#known-limitations-on-unsupported-igpus)
- [Recorded results](#recorded-results)
- [中文说明](docs/report.md)

## What it checks

| Path | Purpose |
|---|---|
| `run_checks.py` | Staged functional suite: device info, elementwise + reductions + shape ops, matmul, NN (conv/BN/attention/autograd/training), memory & streams. Every check is timed, and failures never abort the run. **Each stage runs in its own process by default**, because one wedged kernel can leave the XPU context unusable. |
| `benchmarks/bench_matmul.py` | Steady-state throughput: fp32 matmul TFLOPS, elementwise bandwidth, H2D/D2H copy bandwidth, XPU vs CPU on the same machine. Warm-up iterations are discarded so JIT compilation never pollutes the numbers. |
| `diagnostics/dtype_matrix.py` | Which dtypes and operators actually work (fp32, fp16, bf16, int8/16/32/64, bool, cross-dtype casts, fp64). Supports running one section per process. |
| `diagnostics/numerics.py` | Accuracy of matmul / conv2d / Linear against a float64 CPU reference. |
| `diagnostics/cast_workaround.py` | Reproduces the on-device cast failure and demonstrates the CPU-side-cast workaround for fp16/bf16 inference. |
| `diagnostics/first_call_cost.py` | Per-operator JIT compilation cost on first use vs steady state. |
| `diagnostics/indexing.py` | Isolates the indexing family (`view`/`cat`/`index_select`/`gather`/advanced indexing), which is the most fragile area on the reference machine. |

## Quickstart

```bash
python -m pip install --index-url https://download.pytorch.org/whl/xpu torch torchvision torchaudio
python -m pip install -r requirements.txt

python run_checks.py                     # all stages, one process per stage
python run_checks.py nn                  # a single stage: info | elem | matmul | nn | mem
python run_checks.py --no-isolate        # everything in one process (faster, but a bad kernel poisons the rest)
python benchmarks/bench_matmul.py
python diagnostics/dtype_matrix.py
python diagnostics/numerics.py
python diagnostics/cast_workaround.py
python diagnostics/first_call_cost.py
python diagnostics/indexing.py
```

Use the interpreter that actually has the XPU build: `python -m pip show torch` should print a version ending in `+xpu`. The first run on an unsupported GPU is slow — see [limitation 2](#known-limitations-on-unsupported-igpus).

## Findings from the reference machine

Reference machine: Windows 11 (10.0.26100), Python 3.12.10, `torch 2.12.1+xpu` (XPU build 20250302), Intel Iris Xe Graphics (device id `0x9A49`, 80 EU, 6.8 GiB shared memory, driver `32.0.101.7088`, Level Zero `1.6.34728`). PyTorch prints *"The detected GPU … is not officially supported by PyTorch XPU"* on every initialization.

**Verdict: the fp32 path is functionally complete and numerically correct, but this device is not reproducibly usable — heavy kernels take minutes to compile and can reset the GPU mid-run.**

⚠️ **Run-to-run stability.** The same check can pass on one run and take the device down on the next:

| Run | Outcome |
|---|---|
| `run_checks.py` (isolated stages, log in `results/`) | info 5/5, elem 5/5, matmul 5/5, **nn 0/5** (`conv2d` → `UR_RESULT_ERROR_DEVICE_LOST` after 207 s), **mem 1/4** (`Stream` → device lost after 215 s) |
| the same checks run stage-by-stage on other attempts | `conv2d` passed after 153 s, `Stream` passed after 204 s, a small CNN trained 5 steps and converged |
| a single-process run of an earlier revision | the `index` kernel blocked 451 s → `UR_RESULT_ERROR_UNKNOWN` → device lost; 8/25 checks |

A lost device poisons the rest of the process (every later call fails), which is why stages are isolated in separate processes — and why the recorded run stops part-way through the suite.

✅ When a kernel does compile, everything checked behaves correctly: tensor creation, host↔device copies (incl. pinned async), elementwise math, reductions, sort, shape ops, matmul up to 4096³, conv2d, BatchNorm, autograd + optimizer steps, a CNN training loop that converges, `baddbmm` / `scaled_dot_product_attention` / `LayerNorm`, 256 MiB allocations, `Stream`, RNG reproducibility.

Accuracy is healthy:

| dtype | matmul rel. error vs fp64 |
|---|---|
| float32 | 2.9e-7 … 5.7e-7 |
| float16 | 3.6e-4 |
| bfloat16 | 2.9e-3 |
| fp16 conv2d | 8.3e-4 |

Steady-state throughput (same machine, CPU = 4-core Tiger Lake):

| workload | XPU | CPU | speedup |
|---|---|---|---|
| fp32 matmul 512³ | 0.86 TFLOPS | 0.17 TFLOPS | 5.0× |
| fp32 matmul 1024³ | 1.27 TFLOPS | 0.17 TFLOPS | 7.7× |
| fp32 matmul 2048³ | 1.07 TFLOPS | 0.14 TFLOPS | 7.7× |
| fp32 matmul 4096³ | 0.69 TFLOPS | 0.15 TFLOPS | 4.5× |
| elementwise `(x*2+1).relu().sum()`, 4096² | 14.0 GiB/s | 8.8 GiB/s | 1.6× |
| H2D / D2H copy | 9.7 / 10.2 GiB/s | — | — |

## Known limitations on unsupported iGPUs

These reproduce on the reference machine. Whether they also affect Arc / officially supported GPUs is unknown — if you can test one, an issue with your `run_checks.py` output would be welcome.

1. **On-device cross-dtype casts fail.** `.half()`, `.to(torch.float16)`, `copy_`, `aten._to_copy`, `aten.copy` — every path raises `UR_RESULT_ERROR_INVALID_ARGUMENT (45)`. f32↔f16, f32↔bf16 and int32↔int64 are all affected, which kills `model.half()`, `loss.float()` and `torch.autocast("xpu", ...)`.
   *Workaround:* cast on the CPU and upload — `x.cpu().half().to("xpu")` (≈4 ms for a 1024² tensor). fp16/bf16 matmul, conv2d and elementwise then produce correct results.
2. **First-call JIT compilation is extremely slow and is not cached across processes.** `sin` 8 s, `exp` 20 s, `tanh` 3.6 s, `sum` 41 s, `mean` 144 s, `isfinite` 46 s, `conv2d` 153 s, `matmul 1024³` 156 s, creating a `Stream` 204 s. The same expression runs in ~10 ms afterwards, but a fresh process pays the cost again — a short script can be an order of magnitude slower on XPU than on CPU.
3. **Some kernels are borderline and can take the whole device down.** The indexing family is the worst: `index_select` 416 s, `gather` 73 s, `masked_select` 51 s to compile, and in one single-process run the `index` kernel blocked for 451 s, returned `UR_RESULT_ERROR_UNKNOWN`, and left the device in `UR_RESULT_ERROR_DEVICE_LOST`. `conv2d` (207 s) and creating a `Stream` (215 s) ended the same way on the recorded isolated run, although both had succeeded on earlier attempts. `mean` is flaky too — 144 s success once, `UR_RESULT_ERROR_UNKNOWN` after 41 s another time. The practical consequence: a full suite run is not reproducible, and every stage needs its own process.
4. **No float64.** `Required aspect fp64 is not supported on the device` (the device advertises `has_fp64=0`).
5. **Integer matmul is unavailable.** int8 works; int32 fails to build a oneDNN descriptor; int64 → `Long is not supported in oneDNN`.
6. **A failed kernel poisons the context.** After a failing cast or a timed-out kernel, later ops in the same process also fail with an error 45/UNKNOWN until the interpreter restarts. Always look at the *first* failure, not the cascade.

## Recorded results

`results/` holds the raw console output captured on the reference machine, so every number and error message above can be checked without access to that hardware:

| File | Produced by |
|---|---|
| `results/functional-checks.txt` | `run_checks.py` (isolated stages) |
| `results/benchmarks.txt` | `benchmarks/bench_matmul.py` |
| `results/dtype-matrix.txt` | `diagnostics/dtype_matrix.py` |
| `results/numerics.txt` | `diagnostics/numerics.py` |
| `results/cast-workaround.txt` | `diagnostics/cast_workaround.py` |
| `results/first-call-cost.txt` | `diagnostics/first_call_cost.py` |
| `results/indexing.txt` | `diagnostics/indexing.py` (earlier revision, see note in file) |

## Practical advice

- Pick the device defensively (`dev = "xpu" if torch.xpu.is_available() else "cpu"`) and wrap the first kernel call in `try/except` — compilation itself can fail.
- Never convert dtypes on the device; convert on the host and upload. Same for autocast.
- Keep the process alive (a resident service) rather than starting a fresh script per run, and isolate risky operators in separate processes so a wedge does not kill a long job.
- For real training throughput use an Arc A/B-series discrete GPU, a CPU, or cloud GPUs — an 80-EU iGPU with no matrix units tops out around 1 TFLOPS fp32.
- Updating the Intel graphics driver is the most likely fix for the cast failures.

## License

MIT — see [LICENSE](LICENSE).
