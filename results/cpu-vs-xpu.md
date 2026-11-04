# CPU (AVX-512) vs XPU — measured on an i5-1135G7 + Iris Xe 80EU

Machine: Windows 11, `torch 2.12.1+xpu`, Intel i5-1135G7 (Tiger Lake, 4C/8T, `CPU capability usage: AVX512`), Intel Iris Xe Graphics 80EU, 6.8 GiB shared memory.

Method: XPU kernels are fully compiled before timing (persistent cache enabled, warmup iterations discarded), so no JIT cost is included. `best of N` per row. CPU numbers use MKL/oneDNN with 4 threads.

> This is a 15 W mobile part. Sustained numbers drift 20–30 % with thermal state, which is larger than the AVX-512-vs-AVX2 difference — see the A/B section.

## Per-workload comparison (fp32)

| workload | CPU AVX-512 (ms) | XPU (ms) | winner |
|---|---|---|---|
| matmul 128^3 | 0.028 | 0.498 | CPU 18.10× |
| matmul 256^3 | 0.085 | 0.514 | CPU 6.05× |
| matmul 512^3 | 1.156 | 0.715 | **XPU 1.62×** |
| matmul 1024^3 | 9.931 | 2.034 | **XPU 4.88×** |
| matmul 2048^3 | 67.098 | 17.579 | **XPU 3.82×** |
| matmul 4096^3 | 577.385 | 204.864 | **XPU 2.82×** |
| linear 768->768, batch 128 | 1.105 | 1.056 | **XPU 1.05×** |
| MLP 768x2 fwd+bwd+Adam, b=128 | 8.349 | 6.204 | **XPU 1.35×** |
| conv2d 3x224x224 -> 64x7x7 s2 | 1.698 | 0.737 | **XPU 2.31×** |
| conv2d 8x64x56x56 -> 64x3x3 | 7.200 | 3.359 | **XPU 2.14×** |
| conv2d 32x32x32x32 -> 32x3x3 | 4.180 | 2.152 | **XPU 1.94×** |
| elementwise 1024^2 (x*2+1).relu().sum() | 0.814 | 0.923 | CPU 1.13× |
| elementwise 4096^2 (x*2+1).relu().sum() | 26.051 | 19.968 | **XPU 1.30×** |
| sum over 16M floats | 1.907 | 3.768 | CPU 1.98× |
| softmax (32,128,768) | 1.859 | 1.339 | **XPU 1.39×** |
| layernorm (32,128,768) | 3.164 | 1.400 | **XPU 2.26×** |
| sdpa (8,8,256,64) | 9.570 | 5.948 | **XPU 1.61×** |
| add 16x16 (dispatch overhead) | 0.010 | 0.590 | CPU 58.40× |

## AVX-512 vs AVX2 on the same CPU

Three alternating rounds per configuration (A/B/A/B…), median shown. `ONEDNN_MAX_CPU_ISA=AVX2` + `MKL_ENABLE_INSTRUCTIONS=AVX2` force the narrow ISA.

| workload | AVX-512 (median) | AVX2 (median) | ratio |
|---|---|---|---|
| matmul 1024 | 6.31 | 5.79 | 0.92× |
| matmul 2048 | 83.81 | 88.07 | 1.05× |
| matmul 4096 | 786.88 | 829.99 | 1.05× |
| conv 8x64x56 | 9.24 | 9.38 | 1.02× |
| layernorm | 2.96 | 2.77 | 0.94× |
| softmax | 1.51 | 1.96 | 1.30× |

Reading: AVX-512 is worth roughly 1.0–1.4× here, and most of that disappears into 15 W throttling noise. The wide ISA is not the binding constraint on this machine — power budget and shared memory bandwidth are.

## CPU + XPU cooperation patterns

- **Pipeline** (XPU GEMMs while the CPU does its own GEMMs): 251 ms sequential → 197 ms overlapped (+21.4 %). Works: the XPU queue is asynchronous.
- **Split one GEMM** 2048³ (85% XPU / rest CPU): XPU alone 16.6 ms, split 27.4 ms (-64.7 %). **Counter-productive** — both engines fight for the same DDR bandwidth and the CPU threads steal cores from the driver.
- **Split one GEMM** 4096³ (85% XPU / rest CPU): XPU alone 205.4 ms, split 254.9 ms (-24.1 %). **Counter-productive** — both engines fight for the same DDR bandwidth and the CPU threads steal cores from the driver.
- **Double-buffered copy/compute overlap**: 86.5 ms → 87.2 ms (-0.8 %). No gain: copies and the elementwise compute both saturate the same memory path.

## Dispatch cost (why small work belongs on the CPU)

- one tiny XPU op, synchronised: **0.471 ms/op**
- same op, queued without sync: 0.018 ms/op

| tiny MLP (128→128→128→10), batch | CPU ms | XPU ms (queued) |
|---|---|---|
| 1 | 0.066 | 0.404 |
| 8 | 0.084 | 0.403 |
| 64 | 0.094 | 0.404 |
| 256 | 0.222 | 0.412 |

## Workloads the XPU cannot run at all (CPU only)

- fp64 matmul 1024³: 29.27 ms
- int32 matmul 512³: 67.68 ms
- fp32→fp16 cast, 768×768: 0.055 ms (4096×4096: 4.78 ms)

## Thread scaling (CPU fp32 matmul)

| size | 4 threads | 8 threads | gain |
|---|---|---|---|
| 512³ | 1.17 | 1.04 | 1.12× |
| 1024³ | 6.89 | 5.49 | 1.26× |
| 2048³ | 66.69 | 56.38 | 1.18× |
| 4096³ | 831.69 | 748.83 | 1.11× |

## Choosing, in one table

| situation | use | why |
|---|---|---|
| GEMM ≥ 512³, conv, layernorm/softmax/SDPA on decent batch | **XPU** | 1.4–4.9× faster once kernels are cached |
| GEMM ≤ 256³, batch ≤ 256 inference, tiny models | **CPU (AVX-512)** | XPU dispatch costs ~0.5 ms/op synchronised |
| reductions, light elementwise on small tensors | **CPU** | shared bandwidth + dispatch; CPU wins 1.1–2× |
| fp64, int32/int64 matmul, dtype casts | **CPU** | XPU raises or lacks the kernel |
| preprocessing / augmentation feeding the XPU | **both, pipelined** | +21 % measured; split *different* work, not the same GEMM |
| one GEMM split across both engines | **don't** | −65 % at 2048³, −24 % at 4096³ |
| data must round-trip and compute < ~13 ms | **CPU** | 64 MiB costs 6.4 ms each way |
