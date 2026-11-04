# torch / XPU 本机可用性测试报告

测试时间：2026-10-28 19:18 – 20:30（Asia/Shanghai）
对应脚本：`run_checks.py`、`benchmarks/bench_matmul.py`、`diagnostics/*.py`
原始输出：`results/` 目录

## 结论

**torch + XPU 可用，但该机是"官方不支持"的 Iris Xe 核显。两个硬伤里，编译慢那一半已经解决：设置 `NEO_CACHE_DIR` 即可全局持久化编译缓存，整套检查从 883 s 降到 30 s，并且顺带消除了"设备丢失"——剩下的硬伤只有设备上跨 dtype 转换必崩。**

- fp32 路径功能完整、数值正确，稳态下大尺寸算子比同机 CPU（AVX-512）快 1.4–4.9 倍 → 可用于验证、小规模实验。
- 但**小尺寸反过来**：matmul ≤256³、小 batch 推理、归约这类活儿 CPU 更快（1.1–18 倍），具体分界见「性能」一节的决策表。
- AVX-512 相对 AVX2 只有 1.0–1.4×，且多半被 15 W 功耗墙的抖动掩盖——这台机器的瓶颈不是 SIMD 宽度。
- fp16/bf16 计算本身正常且精度合理，但只要在设备上做 `.half()` / `.bfloat16()` / autocast 就报错 → 必须先在 CPU 侧转 dtype 再上传。
- 无 fp64 硬件支持；int32/int64 矩阵乘走不通。
- 编译缓存默认关闭，每个新进程都要重付几十秒到几分钟；开启 `NEO_CACHE_DIR` 后是毫秒级。
- 因为是核显（共享内存、无 systolic 矩阵单元），不适合正式训练。

## 持久化 JIT 缓存（收益最大的一处优化）

上面那些几十秒到几分钟的编译耗时，其实是"每次进程都重来一遍"造成的浪费。**驱动的编译缓存默认没开**，把目录告诉它之后，第二次启动就是秒级。

线索来自 Level Zero 驱动二进制 `ze_intel_gpu64.dll`，里面只有三个缓存相关的环境变量：`NEO_CACHE_DIR`、`NEO_CACHE_MAX_SIZE`、`NEO_CACHE_PERSISTENT`。实测只有目录是决定性的，而且**目录必须预先存在**（驱动不会自己创建）。每次都是全新进程、同一台机器：

| 算子（2048² fp32） | 冷进程 | 热进程 | 加速 |
|---|---|---|---|
| sort | 59.9 s | **0.28 s** | 217× |
| softmax | 9.5 s | **0.07 s** | 145× |
| layernorm | 6.0 s | **0.05 s** | 118× |
| Linear + backward | 162.7 s | **1.13 s** | 144× |
| 整套套件（24 项检查） | 883 s | **30 s** | 29× |
| *对照*：不设缓存目录跑 sort | 68.6 s | 73.8 s | 无（每次重编译） |

最后那行对照很关键：不设缓存目录时连续两个进程都很慢，所以提速确实来自驱动的模块缓存，而不是操作系统页缓存。少数算子（exp、sum、matmul）在机器上编译过一次之后即使冷进程也很快，所以收益主要体现在 JIT 重的算子上。

各种配置的实际效果：

| 配置 | 结果 |
|---|---|
| 什么都不设 | 每个进程都重编译，sort 永远是 60–74 s |
| 只设 `NEO_CACHE_PERSISTENT=1` | **无效**，默认目录不会被写入，也没有提速 |
| 只设 `NEO_CACHE_DIR=<已存在且可写的目录>` | **有效**，驱动写入 `*.l0_cache`（上面三个算子 20 MB，整套套件 180 MB） |
| `NEO_CACHE_DIR` 指向不存在的目录 | **静默失效**——不写文件也不提速，必须先建目录 |
| `NEO_CACHE_DIR` + `NEO_CACHE_MAX_SIZE=<字节>` | 同上，但带容量上限 |
| 只设 `SYCL_CACHE_PERSISTENT=1` + `SYCL_CACHE_DIR=…`（本机原本就设了） | 对 PyTorch XPU **没用**：SYCL 的 inline cache 这条路被绕开，而且那个目录属管理员所有、当前用户写不进去 |

全局开启（新进程生效）：

```powershell
.\scripts\enable_jit_cache.ps1          # 默认 %LOCALAPPDATA%\torch_xpu_cache，上限 4 GiB
.\scripts\enable_jit_cache.ps1 -Disable
```

或手工执行：

```powershell
setx NEO_CACHE_DIR "C:\Users\<你>\AppData\Local\torch_xpu_cache"
setx NEO_CACHE_MAX_SIZE 4294967296
```

注意：开启后的**第一次运行仍然是冷的**（也可能仍然丢设备），之后才快；缓存键包含驱动和硬件信息，升级驱动后会重新变冷；`NEO_CACHE_MAX_SIZE` 满了会淘汰旧条目；这个缓存**不能**修复 dtype 转换报错（那是内核级错误，不是编译问题）。

## 稳定性：同一个检查，时好时坏

这是本次测试最需要注意的一点。同一份检查在不同次运行里结论不同：

| 运行方式 | 结果 |
|---|---|
| `run_checks.py`，**无内核缓存**（`results/functional-checks.txt`） | info 5/5、elem 5/5、matmul 5/5、**nn 0/5**（`conv2d` 编译 207 s 后 `UR_RESULT_ERROR_DEVICE_LOST`）、**mem 1/4**（建 `Stream` 215 s 后设备丢失） |
| `run_checks.py`，缓存已开启但为空（`results/persistent-cache-cold.txt`） | **5/5 阶段、24/24 检查，883 s** |
| `run_checks.py`，缓存已热（`results/persistent-cache-warm.txt`） | **5/5 阶段、24/24 检查，30 s** |
| 同批检查分 stage 单独跑（其它几次尝试） | `conv2d` 153 s 后通过、`Stream` 204 s 后通过、mini CNN 训练 5 步收敛 |
| 早期版本一次性跑（单进程、无缓存） | `index` 内核卡 451 s → `UR_RESULT_ERROR_UNKNOWN` → 设备丢失，25 项里只过 8 项 |

设备一旦丢失，进程内后续所有调用都会失败，所以套件默认按 stage 分进程执行。

**关键结论：丢设备是"编译超时"造成的，不是算子本身的问题。** 开启持久化编译缓存后，没有一个内核需要几百秒编译，整套检查从不稳定变成 30 秒全绿。也就是说，前面那节提到的持久化缓存不只是加速，它同时是这个环境可用性的前提。

## 环境

| 项目 | 值 |
|---|---|
| OS | Windows 11 (10.0.26100) |
| Python | 3.12.10 — `C:\Program Files\Python312\python.exe` |
| torch | **2.12.1+xpu**（XPU 构建号 20250302） |
| 配套 | torchvision 0.27.1+xpu、torchaudio 2.11.0+xpu、triton-xpu 3.7.1、intel-sycl-rt 2025.3.2、intel-pti 0.16.0 |
| `torch.xpu.is_available()` | **True**，device_count = **1** |
| 设备 | Intel(R) Iris(R) Xe Graphics（device_id `0x9A49`） |
| 计算单元 | 80 EU / 10 subslice，max_work_group_size 512 |
| 显存 | 6.8 GiB（共享系统内存），local_mem 64 KB |
| 驱动 | 32.0.101.7088；Level Zero 1.6.34728；oneAPI Unified Runtime over Level-Zero |
| 能力位 | has_fp16=1、**has_fp64=0**、**DPAS / 矩阵乘单元=False**、bf16 转换=False |
| PyTorch 态度 | 每次初始化都告警：该 GPU **不受官方支持**，建议 Arc(Alchemist) 及以上 |

## 功能测试

通过（19 项）：

- 设备查询、device capability、`torch.xpu.device(0)`
- 张量创建、`.xpu()` / `.to("xpu:0")`、CPU↔XPU 往返、pinned 内存异步拷贝
- dtype 创建覆盖：f32 / f16 / bf16 / int8 / int16 / int32 / int64 / uint8 / bool
- 逐元素运算、归约（sum/mean/max/std/argmax）、sort/topk、index/gather、view/reshape/transpose/permute、cat/stack
- fp32 matmul 256³–4096³
- conv2d、BatchNorm+ReLU+MaxPool
- autograd + SGD 反传更新、mini CNN 训练 5 步收敛（loss 2.342 → 2.293）
- `baddbmm`、`scaled_dot_product_attention`、`LayerNorm`（transformer 关键算子）
- 256 MiB 分配/释放、XPU Stream 创建与同步、RNG 种子可复现、`synchronize` / `empty_cache`
- fp16 / bf16 计算（数据在 CPU 侧转好再上传）：matmul 512³/1024³、conv2d、逐元素

数值精度（对比 CPU float64 参考）：

| dtype | 尺寸 | 相对误差 |
|---|---|---|
| float32 | 256/512/1024 | 2.9e-7 / 4.1e-7 / 5.7e-7 |
| float16 | 256/512/1024 | 3.6e-4（各级一致） |
| bfloat16 | 256/512/1024 | 2.9e-3（符合 bf16 8 位尾数） |
| fp32 conv2d | 64×64 | 8.7e-8 |
| fp16 conv2d | 64×64 | 8.3e-4 |

精度全部在正常范围内，XPU 计算结果可信。

失败 / 受限：

| 问题 | 现象 | 影响 |
|---|---|---|
| **设备上跨 dtype 转换** | `.half()`、`.to(f16)`、`copy_`、`_to_copy`、`aten.copy` 全部报 `UR_RESULT_ERROR_INVALID_ARGUMENT (45)`；f32↔f16、f32↔bf16、int32↔int64 都中招 | **最大硬伤**：`model.half()`、`loss.float()`、`torch.autocast("xpu", bf16)` 全部不可用 |
| fp64 | `Required aspect fp64 is not supported on the device` | 无 float64，符合硬件能力 |
| int matmul | int32 → oneDNN 建不出 descriptor；int64 → `Long is not supported in oneDNN` | int8 matmul 可用，int32/int64 不行 |
| 错误"中毒" | 一次 cast 失败后，同进程后续算子在 bench 里连锁报 45，只能重启进程 | 排查时要看"首个报错"，而不是后面那一串 |
| 索引类算子 | `index_select` 编译 **416 s**、`gather` 73 s、`masked_select` 51 s；一次性完整运行里 `index` 内核 451 s 后直接 `UR_RESULT_ERROR_UNKNOWN`，随后设备 `UR_RESULT_ERROR_DEVICE_LOST` | 该族算子会把整个设备拖垮，因此测试套件默认**按 stage 分进程**运行 |
| 首次 JIT 编译 | 见下表，且**新进程重新编译**（无持久缓存） | 短脚本跑 XPU 可能比 CPU 慢一个数量级 |

首次调用编译耗时实测（同一进程内第二次起恢复正常）：

| 算子 | 首次 | 之后 |
|---|---|---|
| sin (2048²) | 6.3–8.0 s | ~1.5 ms |
| exp | 15–19.6 s | ~1.5 ms |
| tanh | 3.6 s | ~1.5 ms |
| sum | 41.5 s | ~0.9 ms |
| mean（归约） | 144 s（成功）/ 41 s 后报 UNKNOWN（失败） | ~1 ms |
| isfinite | 46 s | — |
| conv2d (4×3×128²) | 153 s | — |
| matmul fp32 1024³ | 156 s | 1.7 ms |
| Stream 创建 | 204 s | — |
| LayerNorm | 5.8 s | — |
| baddbmm | 31.8 s | — |
| index_select / gather / masked_select | 416 / 73 / 51 s | — |
| 整条逐元素表达式 | ~250 s（累计） | 10–13 ms |

## 性能（稳态，排除编译时间）

### CPU（AVX-512）vs XPU，同一 workload

机器：i5-1135G7（Tiger Lake，4C/8T，torch 报 `CPU capability usage: AVX512`）+ Iris Xe 80EU。
方法：XPU 内核全部编译完成后才计时（持久缓存 + 丢弃 warmup），best-of-N；CPU 走 MKL/oneDNN、4 线程。完整表格与原始数据见仓库 `results/cpu-vs-xpu.md`、`results/cpu-vs-xpu-data.json`。

| workload | CPU AVX-512 | XPU | 胜者 |
|---|---|---|---|
| matmul 128³ | 0.028 ms | 0.498 ms | **CPU 18×** |
| matmul 256³ | 0.085 ms | 0.514 ms | **CPU 6.1×** |
| matmul 512³ | 1.16 ms | 0.72 ms | XPU 1.6× |
| matmul 1024³ | 9.9 ms | 2.03 ms | **XPU 4.9×** |
| matmul 2048³ | 67 ms | 17.6 ms | **XPU 3.8×** |
| matmul 4096³ | 577 ms | 205 ms | **XPU 2.8×** |
| conv2d 3×224×224→64×7×7 | 1.70 ms | 0.74 ms | XPU 2.3× |
| conv2d 8×64×56×56 | 7.20 ms | 3.36 ms | XPU 2.1× |
| layernorm (32,128,768) | 3.16 ms | 1.40 ms | XPU 2.3× |
| softmax (32,128,768) | 1.86 ms | 1.34 ms | XPU 1.4× |
| sdpa (8,8,256,64) | 9.57 ms | 5.95 ms | XPU 1.6× |
| MLP 768×2 fwd+bwd+Adam, b=128 | 8.35 ms | 6.20 ms | XPU 1.35× |
| elementwise 4096² | 26.0 ms | 20.0 ms | XPU 1.3× |
| elementwise 1024² | 0.81 ms | 0.92 ms | CPU 1.13× |
| sum over 16M floats | 1.90 ms | 3.77 ms | **CPU 2.0×** |
| 小 MLP（128 宽）batch 1–64 | 0.07–0.09 ms | 0.40 ms | **CPU 4–6×** |
| add 16×16 | 0.010 ms | 0.59 ms | **CPU 58×** |

交叉点很清晰：**矩阵乘在 256³ 与 512³ 之间翻转**（≤256³ CPU 快 6–18 倍，≥512³ XPU 快 1.6–4.9 倍）。

### AVX-512 相对 AVX2 究竟值多少

三轮交替 A/B（`ONEDNN_MAX_CPU_ISA=AVX2` + `MKL_ENABLE_INSTRUCTIONS=AVX2`），取中位数：

| workload | AVX-512 | AVX2 | 比值 |
|---|---|---|---|
| matmul 1024³ | 6.31 ms | 5.79 ms | 0.92× |
| matmul 2048³ | 83.8 ms | 88.1 ms | 1.05× |
| matmul 4096³ | 787 ms | 830 ms | 1.05× |
| conv2d 8×64×56×56 | 9.24 ms | 9.38 ms | 1.01× |
| layernorm | 2.96 ms | 2.77 ms | 0.94× |
| softmax | 1.51 ms | 1.96 ms | 1.30× |

单轮测量里 AVX-512 最好能到 1.3–1.4×（2048³ 67 ms vs 91 ms、4096³ 577 ms vs 851 ms），但交替复测后这些差距大部分被 15 W 功耗墙带来的 20–30% 抖动吃掉。**这台机器真正的瓶颈是功耗和共享内存带宽，不是 SIMD 宽度。**

### 为什么小算子该留给 CPU

XPU 单算子同步延迟 **0.47 ms**（异步排队 0.018 ms/算子），CPU 一个 16×16 加法只要 0.010 ms。小于约 512³ 等效 GEMM 的工作量都会被 dispatch 开销吃掉。

### 协作：哪些能叠加，哪些白费

| 模式 | 实测 |
|---|---|
| 流水线：XPU 算 GEMM，CPU 同时算自己的 GEMM | 251 ms → **197 ms（+21%）**，因为 XPU 队列是异步的 |
| 同一个 GEMM 按行切分（85% XPU / 15% CPU） | 2048³：16.6 → **27.4 ms（−65%）**；4096³：205 → **255 ms（−24%）** |
| 双缓冲拷贝/计算重叠（8×32 MiB + 逐元素） | 86.5 → 87.2 ms（−1%） |

原因：核显与 CPU 共享同一条 DDR 带宽，CPU 线程还会抢占驱动线程；拷贝和带宽型计算也走同一条路，所以重叠不起来。**"不同阶段流水线"有收益，"同一份计算拆开"是负收益。**

### 一句话决策

- **XPU 甜区**：大尺寸、计算密集 —— GEMM ≥512³、conv2d、layernorm/softmax/SDPA、成规模的训练步（1.4–4.9×）。
- **CPU（AVX-512）甜区**：小尺寸、低延迟、以及 XPU 根本没有的算子 —— matmul ≤256³、小模型小 batch 推理、归约、fp64/int32/int64 矩阵乘、dtype 转换。
- **协作甜区**：把 CPU 当作 XPU 的数据准备/兜底阶段并行跑（+21%）；不要试图把同一个矩阵乘劈成两半。

## 建议

1. 设备选择写自适应：`dev = "xpu" if torch.xpu.is_available() else "cpu"`，并用 `try/except` 兜住首个算子的编译失败。
2. **绝不要**在 XPU 上做 dtype 转换：先 CPU 侧 `model.half()` / `x.half()`，再 `.to("xpu")`（1024² 往返仅约 3.6 ms）。同理回避 `torch.autocast("xpu", ...)`。
3. 回避 float64、int32/int64 矩阵乘。
4. 同一进程内复用模型 / 进程常驻（如常驻推理服务），不要频繁启停脚本——每进程都要重付 JIT 编译成本；务必开启 `NEO_CACHE_DIR`。
5. 按尺寸选设备：小张量/小 batch 直接留在 CPU（AVX-512 在这里更快），只有 ≥512³ 等效的 GEMM、conv、norm 才值得搬上 XPU。
6. 需要并行时用"流水线"而不是"切分"：CPU 做预处理/后处理，XPU 做主干计算。
7. CPU 侧把线程数设到 8（比默认 4 线程快 1.1–1.26×）：`torch.set_num_threads(8)`。
8. 该机为 Iris Xe 核显（Tiger Lake），官方不支持；若要跑真实训练，建议换 Arc A/B 系列独显或用 CPU / 云 GPU。
9. 可尝试更新 Intel 显卡驱动（当前 32.0.101.7088）看是否修复 cast 崩溃；这是最可能的改善点。
