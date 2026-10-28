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
| `diagnostics/cache_probe.py` | Measures whether kernel compilation is cached across processes: runs every operator in a fresh interpreter and compares cold vs warm. |
| `scripts/enable_jit_cache.ps1` | Enables (or disables) the persistent Level Zero kernel cache for the current user. |

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

**Verdict: with the persistent kernel cache enabled this iGPU is usable for fp32 work — the whole suite passes 24/24 in 30 s. Without it, the JIT compilation makes short scripts an order of magnitude slower than CPU and can wedge the device.**

⚠️ **Run-to-run stability.** The same check can pass on one run and take the device down on the next — the failure mode is always a multi-minute kernel compilation:

| Run | Outcome |
|---|---|
| `run_checks.py`, **no kernel cache** (log in `results/functional-checks.txt`) | info 5/5, elem 5/5, matmul 5/5, **nn 0/5** (`conv2d` → `UR_RESULT_ERROR_DEVICE_LOST` after 207 s), **mem 1/4** (`Stream` → device lost after 215 s) |
| `run_checks.py`, cache enabled but empty — every kernel compiled once (log `results/persistent-cache-cold.txt`) | **5/5 stages, 24/24 checks in 883 s** |
| `run_checks.py`, cache warm (log `results/persistent-cache-warm.txt`) | **5/5 stages, 24/24 checks in 30 s** |
| the same checks run stage-by-stage on other attempts | `conv2d` passed after 153 s, `Stream` passed after 204 s, a small CNN trained 5 steps and converged |
| a single-process run of an earlier revision, no cache | the `index` kernel blocked 451 s → `UR_RESULT_ERROR_UNKNOWN` → device lost; 8/25 checks |

So the device losses come from the compilation timeouts, not from the operators themselves — remove the compilation and the instability mostly disappears. A lost device still poisons the rest of the process (every later call fails), which is why stages are isolated in separate processes.

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

## Persistent kernel cache: 20 s → 0.1 s

The first-call cost in limitation 2 is **one-time cost that the driver throws away** — unless you tell it where to keep the compiled modules. It is also what causes the device losses above: a kernel that takes 200 s to compile is a kernel that can time out and reset the GPU.

Whole suite (`run_checks.py`, 24 checks, cold cache vs warm cache): **883 s → 30 s**, and the warm run passes every stage including the `conv2d` and `Stream` checks that previously lost the device.

The Intel Level Zero driver binary (`ze_intel_gpu64.dll`) contains exactly three cache knobs: `NEO_CACHE_DIR`, `NEO_CACHE_MAX_SIZE`, `NEO_CACHE_PERSISTENT`. In practice only the directory matters — and it must already exist, because the driver does not create it. Fresh process per measurement, same machine (`results/cache-probe.txt`):

| operator (2048² fp32) | cold process | warm process | speedup |
|---|---|---|---|
| `sort` | 59.9 s | **0.28 s** | 217× |
| `softmax` | 9.5 s | **0.07 s** | 145× |
| `layernorm` | 6.0 s | **0.05 s** | 118× |
| `Linear` + `backward` | 162.7 s | **1.13 s** | 144× |
| whole suite, 24 checks | 883 s | **30 s** | 29× |
| *control*: `sort` with **no** cache dir | 68.6 s | 73.8 s | none — it recompiles every process |

The control row is the important one: two consecutive fresh processes are both slow when no cache dir is set, so the speedup really is the driver's module cache and not the OS page cache. A few kernels (`exp`, `sum`, `matmul`) end up fast even in a cold process once the machine has compiled them at least once, so the visible win is largest on JIT-heavy kernels.

What each environment variable actually does:

| configuration | result |
|---|---|
| nothing set | every process recompiles; `sort` costs 60–74 s forever |
| `NEO_CACHE_PERSISTENT=1` alone | **no effect** — the default directory is never written, no speedup |
| `NEO_CACHE_DIR=<existing, writable dir>` | works — the driver writes `*.l0_cache` files (20 MB for the three ops above, 180 MB for the whole suite) |
| `NEO_CACHE_DIR` pointing at a directory that does not exist | **silently does nothing** — no files, no speedup; create the directory first |
| `NEO_CACHE_DIR` + `NEO_CACHE_MAX_SIZE=<bytes>` | same, with a size cap |
| `SYCL_CACHE_PERSISTENT=1` + `SYCL_CACHE_DIR=…` (already set on this machine) | irrelevant for PyTorch XPU: the SYCL inline cache is bypassed, and the configured directory was admin-owned and not writable anyway |

Enable it for the current user (new processes pick it up):

```powershell
.\scripts\enable_jit_cache.ps1                          # default: %LOCALAPPDATA%\torch_xpu_cache, 4 GiB cap
.\scripts\enable_jit_cache.ps1 -CacheDir D:\xpu-cache -MaxSizeMB 8192
.\scripts\enable_jit_cache.ps1 -Disable
```

or manually:

```powershell
setx NEO_CACHE_DIR "C:\Users\<you>\AppData\Local\torch_xpu_cache"
setx NEO_CACHE_MAX_SIZE 4294967296
```

Caveats: the first run after enabling is still cold (883 s for the full suite, versus 30 s warm) and can still hit a device loss; cache keys include the driver and device, so a driver update starts cold again; `NEO_CACHE_MAX_SIZE` evicts; the cache dir must be writable by your user (`%LOCALAPPDATA%` is; `C:\ProgramData\…` usually is not); and the cache does **not** fix the cast failures from limitation 1 — those are kernel-level errors, not compilation.

## Known limitations on unsupported iGPUs

These reproduce on the reference machine. Whether they also affect Arc / officially supported GPUs is unknown — if you can test one, an issue with your `run_checks.py` output would be welcome.

1. **On-device cross-dtype casts fail.** `.half()`, `.to(torch.float16)`, `copy_`, `aten._to_copy`, `aten.copy` — every path raises `UR_RESULT_ERROR_INVALID_ARGUMENT (45)`. f32↔f16, f32↔bf16 and int32↔int64 are all affected, which kills `model.half()`, `loss.float()` and `torch.autocast("xpu", ...)`.
   *Workaround:* cast on the CPU and upload — `x.cpu().half().to("xpu")` (≈4 ms for a 1024² tensor). fp16/bf16 matmul, conv2d and elementwise then produce correct results.
2. **First-call JIT compilation is extremely slow and, by default, is not cached across processes.** `sin` 8 s, `exp` 20 s, `tanh` 3.6 s, `sum` 41 s, `mean` 144 s, `isfinite` 46 s, `conv2d` 153 s, `matmul 1024³` 156 s, creating a `Stream` 204 s. The same expression runs in ~10 ms afterwards, but a fresh process pays the cost again — a short script can be an order of magnitude slower on XPU than on CPU. **This one is fixable: see [Persistent kernel cache](#persistent-kernel-cache-20-s--01-s) — it turns minutes into milliseconds for every later process.**
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
| `results/persistent-cache-cold.txt` | `run_checks.py` with `NEO_CACHE_DIR` set, cache empty |
| `results/persistent-cache-warm.txt` | `run_checks.py` again, same cache dir |
| `results/cache-probe.txt` | `diagnostics/cache_probe.py`, with and without `NEO_CACHE_DIR` |

## Practical advice

- Pick the device defensively (`dev = "xpu" if torch.xpu.is_available() else "cpu"`) and wrap the first kernel call in `try/except` — compilation itself can fail.
- Never convert dtypes on the device; convert on the host and upload. Same for autocast.
- Keep the process alive (a resident service) rather than starting a fresh script per run, and isolate risky operators in separate processes so a wedge does not kill a long job.
- For real training throughput use an Arc A/B-series discrete GPU, a CPU, or cloud GPUs — an 80-EU iGPU with no matrix units tops out around 1 TFLOPS fp32.
- Updating the Intel graphics driver is the most likely fix for the cast failures.

## License

MIT — see [LICENSE](LICENSE).
