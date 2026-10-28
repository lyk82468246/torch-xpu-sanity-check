# torch / XPU 本机可用性测试报告

测试时间：2026-10-28 19:18 – 20:30（Asia/Shanghai）
对应脚本：`run_checks.py`、`benchmarks/bench_matmul.py`、`diagnostics/*.py`
原始输出：`results/` 目录

## 结论

**torch + XPU 可用，但该机是"官方不支持"的 Iris Xe 核显。两个硬伤里，编译慢那一半已经解决：设置 `NEO_CACHE_DIR` 即可全局持久化编译缓存，整套检查从 883 s 降到 30 s，并且顺带消除了"设备丢失"——剩下的硬伤只有设备上跨 dtype 转换必崩。**

- fp32 路径功能完整、数值正确，稳态下比同机 CPU 快约 5 倍 → 可用于验证、小规模实验。
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

fp32 矩阵乘（与同机 CPU 对比）：

| 规模 | XPU | XPU TFLOPS | CPU | CPU TFLOPS | 加速 |
|---|---|---|---|---|---|
| 512³ | 0.33 ms | 0.82 | 1.36 ms | 0.20 | 4.2× |
| 1024³ | 1.75 ms | 1.23 | 9.60 ms | 0.22 | 5.5× |
| 2048³ | 15.8 ms | 1.09 | 75.0 ms | 0.23 | 4.8× |
| 4096³ | 161.9 ms | 0.85 | 871.7 ms | 0.16 | 5.4× |

逐元素 `(x*2+1).relu().sum()`：

| 规模 | XPU | XPU GiB/s | CPU GiB/s | 加速 |
|---|---|---|---|---|
| 2048² | 4.47 ms | 14.0 | 8.2 | 1.7× |
| 4096² | 18.4 ms | 13.6 | 8.9 | 1.5× |
| 8192² | 84.6 ms | 11.8 | 8.5 | 1.4× |

主机↔设备带宽（256 MiB，pinned）：H2D 11.2 GiB/s，D2H 11.4 GiB/s。

一句话：峰值得益于 80 EU 的 SIMD，但受限于双通道共享内存（实测 ~12–14 GiB/s）且无矩阵单元，只有 CPU 的 4–5 倍，远低于独显 Arc 的量级。

## 建议

1. 设备选择写自适应：`dev = "xpu" if torch.xpu.is_available() else "cpu"`，并用 `try/except` 兜住首个算子的编译失败。
2. **绝不要**在 XPU 上做 dtype 转换：先 CPU 侧 `model.half()` / `x.half()`，再 `.to("xpu")`（1024² 往返仅约 3.6 ms）。同理回避 `torch.autocast("xpu", ...)`。
3. 回避 float64、int32/int64 矩阵乘。
4. 同一进程内复用模型 / 进程常驻（如常驻推理服务），不要频繁启停脚本——每进程都要重付 JIT 编译成本。
5. 该机为 Iris Xe 核显（Tiger Lake），官方不支持；若要跑真实训练，建议换 Arc A/B 系列独显或用 CPU / 云 GPU。
6. 可尝试更新 Intel 显卡驱动（当前 32.0.101.7088）看是否修复 cast 崩溃；这是最可能的改善点。
