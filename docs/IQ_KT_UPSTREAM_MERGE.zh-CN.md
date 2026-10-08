# IQ_KT 分支合并上游 0.1.40.3

本文记录 0.1.40.3 合并当时的配置和测试。当前 v2/v4 部署见 [按需启动](IQ_KT_MODEL_SWITCH.zh-CN.md)；
之后新增 128 个提交的 0.1.41 已合入并部署，见 [新版合并记录](IQ_KT_UPSTREAM_0.1.41_MERGE.zh-CN.md)。

2026-10-08，在独立工作树中把上游 `d5ea7133741e67743c0e886bb426c0ce8d69cf6c`
（0.1.40.3）合入本机的 IQ_KT 分支。合并前为 `a0b5b3b`，包含此前的 packed KT decode、
并行 prefill 解码器、CPU AVX2、MTP 侧车和 Vulkan 核显视觉支持。

**本机默认配置的吞吐基本持平，没有测得明确提速。** 新旧交替测试中，decode 均值变化为
−1.2% 至 +0.3%，多千 token 的 prefill 变化为 +0.2% 至 +0.7%，小于这组请求自身的波动。
对应请求的全部输出 token 一致。此前约 40→63 tok/s 的 KT 算子收益已经包含在合并前版本中，
不能再次计入本次上游更新的收益。

## 合并内容与边界

合并提交为 `ca45e20`。上游新增的 265 个提交包含大量 Intel SYCL、AMD、安装器、文档和基准数据，
并非全是本机 CUDA 路径的优化。与当前使用方式较相关的内容包括：

- Windows 的 `--ple-io ram`：通过提高进程工作集下限和 `VirtualLock` 锁定整张 PLE 表。
- 服务在长 prefill 期间响应取消、启动及视觉管道读取超时、重启状态处理、API/工具调用兼容性修复、
  Prometheus `/metrics`，以及视觉 JPEG EXIF 方向处理。
- Prefill 的传输重叠及缓冲检查；另有显式开启的 CPU 分担、CUDA 优化开关。
- 视觉 helper 默认采用 portable 构建。

两处文本冲突均在视觉 helper：保留 Vulkan 选项和 `--device` / `--list-devices`，同时接受上游
portable 默认值和 CPU flash-attention 说明。KT 类型、打包器、CPU/CUDA 算子及 MTP 支持均保留。
没有把 ggml 依赖替换为 IK 后端，也没有重打包或重新量化现有主模型。

本次对照保持 CPU prefill 分担等可选功能关闭。PDL 要求 sm90+，不适用于这两张卡；部分多 token
投影优化在 MTP 关闭、单 token decode 下也没有相应工作量。合并后可用的新功能不等于默认单请求更快。

## 构建及正确性复核

Windows、VS 2022 / MSVC 19.42、CUDA 12.8.61；主程序为 `sm75;sm89`、CUDA runtime Shared、
`STRATA_PORTABLE=OFF`。视觉 helper 为 Vulkan ON、CUDA OFF、`STRATA_PORTABLE=ON`。
两者沿用 `Strata-ggml-kt` 的 `3cf03257f219afbe7334045ff7c6a06ac68c627d`。

测试二进制对应 `15743b6`：它在合并之上仅修正一项 Windows 测试断言，将临时路径的 8.3 别名
与长名称用 `realpath` 归一后比较；运行时代码未因该修正改变。

| 检查 | 结果 |
| --- | --- |
| 主程序、CUDA parity、CPU parity、Vulkan helper | Release 构建通过 |
| 完整 Python 服务套件 | 565 项，6 项跳过，其余通过 |
| 打包 / 词表导出 / GPU 唤醒生命周期 | 24 / 1 / 3 项通过 |
| IQ3_KT / IQ4_KT 与独立 IK DLL 对照 | 37 组行、829 个真实张量跨度；最大相对误差 1.73×10⁻⁷ |
| 两张 GPU 的 KT parity | 各 37 组；解码最大相对误差 1.73×10⁻⁷，GEMV 4.70×10⁻⁷ |
| CPU AVX2 | 37 组 × 1–9 token，与标量结果一致 |
| MTP 数据审计 | 26 个稠密张量，最大相对舍入误差 0.00393；F32 norm 不变，3 个完整专家字节一致 |

上述数值对照覆盖 KT 接入及其已知舍入边界，不构成所有任务质量不变的保证。

## 新旧版本交替基准

硬件为 Core i9-12900HK、64 GiB RAM、RTX 2080 Ti 22 GiB（PCIe 3.0 ×8）和
RTX 4080 SUPER 32 GiB（USB4）。使用相同 IQ3_KT_v2 模型、pack、tokenizer 和输入 token：

| 参数 | 设置 |
| --- | --- |
| 主模型层分配 | 2080 Ti 19 层；4080 SUPER 29 层及输出头 |
| 专家缓存 | 9,728 + 14,848 = 24,576，全部驻留 GPU |
| 显存预留 | 1,536 / 768 MiB |
| 上下文 / KV | 262,144；K8V8；20,480 resident cells |
| Prefill | 2048 |
| MTP / prompt cache / suffix draft / lookup chain / adaptive swaps | 全部关闭 |
| PLE | `direct`，SSD 原表加 RAM 行缓存 |
| 其他 | `--trim-stage-weights`、`--check-logits`；相同 expert profile |
| 计时 | `STRATA_DECODE_TIMING=1`；详细 prefill / verifier profiler 关闭 |

运行顺序为旧 A → 新 A → 旧 B → 新 B。每次进程先做 72 输入 / 32 输出的预热并排除该项，
随后依次测 72、3,570、12,150 输入，每项输出 256 token，再重复这三项。
因此每个版本、每种输入长度各有 **4 次**有效测量。表中为单次请求吞吐的算术均值。

测试期间卸载并停止原 8080 服务，避免其模型、CUDA 唤醒 context 和空闲降频定时器干扰对照。
每个基准进程在 engine READY、专家缓存完成后建立相同的 CUDA 唤醒 probe，并在请求前触发。
未额外将 4080 核心固定到某一值；每 0.5 s 记录核心/显存频率、温度、利用率和可用 RAM。

| 实际输入 token | 旧版 prefill tok/s | 新版 prefill tok/s | 变化 | 旧版 decode tok/s | 新版 decode tok/s | 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 72 | 58.99 | 58.37 | −1.1% | 63.08 | 63.28 | +0.3% |
| 3,570 | 679.97 | 684.88 | +0.7% | 62.83 | 62.10 | −1.2% |
| 12,150 | 1,030.83 | 1,033.21 | +0.2% | 62.34 | 62.12 | −0.4% |

72-token prefill 主要反映准备、恢复等固定成本，不能外推成长输入吞吐。旧版自身的 decode 测量范围
约为 59.38–66.93 tok/s；两个版本的同一进程内，第二轮通常更快。PLE 行缓存会随请求变热，
即便 prefix cache 关闭也存在这种差别。因此，这些 ±1% 左右的变化不足以认定改善或回退。

24 个有效请求的输出按相同输入分组后 token 全部一致。两版日志均确认全专家驻留，decode 的 CPU
专家和 PCIe 专家计数均为零；prefill 借用缓存时读取专家文件不在这个“decode 零流送”的结论内。
采样显示 2080 核心/显存为 1500/6800 MHz，4080 约 2790–2805/11251 MHz，
没有早期 405 MHz 低频问题，也未观察到足以解释明显性能差距的频率变化。

原始本地记录位于 `build-kt/upstream-{before,after}-{a,b}.{json,log,protocol}` 及相应 telemetry。
公开的[汇总数据](../bench/results/2026-10-08-iqkt-upstream-merge/results.json)保留每项计时、
输出 token 哈希及采样摘要；不包含 API key 或私有服务配置。

## Windows PLE 整表驻留 RAM

额外用合并后的同一二进制运行 `--ple-io ram`，其余参数与上述对照相同。日志明确报告
`PLE table locked in RAM (--ple-io ram) in 42.8 s`，不是锁定失败后退回普通 mmap。
约 25.03 GiB 的原始 KT 表没有转换。每种输入仍重复两次，输出 token 与 direct 路径一致。

| 实际输入 token | 新版 direct decode 均值 | 新版 RAM decode 均值 |
| --- | ---: | ---: |
| 72 | 63.28 tok/s | 64.77 tok/s |
| 3,570 | 62.10 tok/s | 63.76 tok/s |
| 12,150 | 62.12 tok/s | 63.67 tok/s |

RAM 的这一次独立运行中，decode 均值高约 2.3–2.7%，多千 token 的 prefill 基本不变。
主要区别在 direct 首轮尚未充分命中行缓存时；direct 第二轮已经接近 RAM 的结果。
RAM 每项只有两次测量，且没有做 direct/RAM 反向交替，因此不将它作为确定的合并提速结论。
反复使用这些合成提示也不能代表大量不同文档下的 PLE 未命中率。

启动加载专家时，可用物理 RAM 曾降至约 0.13 GiB；随后运行时移除已映射专家的进程工作集，
日志报告可用 RAM 从 0.17 GiB 回到 28.08 GiB。实际请求期间最低仍有 21.48 GiB 可用，
并未持续处于启动时的内存压力。整表加载时间和启动峰值内存仍是启用该选项的代价。

日常配置保留 **`--ple-io direct`**。目前这组吞吐收益不足以替换已有设置；如后续针对大量不同输入
优化 PLE 延迟，可以单独比较整表 RAM、行缓存命中率和启动成本。

## 长上下文与 MTP 复测

合并后的 **K8V8 / 2048** 配置成功读入 260,000 token，在输入开头、中间、末尾的检索题中返回
`cedar, cobalt, quartz`。Prefill 为 **270.026 s，962.9 tok/s**；随后 8 个输出 token（含结束 token）
用时 203.2 ms。它确认容量和这项检索仍可用，不能用这么短的输出评价满上下文持续 decode。

此前 [prefill 文档](IQ_KT_PREFILL.md) 的 198.126 s 使用 **4096** 分块。本次为 **2048**，
没有做合并前后 260K、相同分块的交替对照，不能据此把时间差解释为上游合并造成的回退。
本次相同配置的吞吐比较以之前的 72 / 3,570 / 12,150-token 表为准。

IQ4_KT MTP 侧车另用历史 smoke 配置验证：split 20、prefill 512、K8V8、262,144 容量。
数数请求在 24 个输出 token 上限处停止，草稿接受 16/16；简单 Python 函数请求正常结束，
输出 `sum(x for x in lst if x % 2 == 0)`，草稿接受 20/22。它证明原生 KT 侧车仍能加载、执行和接受草稿。
这些短请求和不同分层不适合与上面的无 MTP 吞吐比较；日常仍关闭 MTP。

## 核显视觉与 HTTP 检查

初次 HTTP 检查遇到一个本机设备过滤问题：枚举时 Intel 曾是 Vulkan1，后续启动时变为 Vulkan2，
配置中的数字过滤把它隐藏了。旧 helper 恢复服务时也遇到同一错误。移除
`GGML_VK_VISIBLE_DEVICES` 数字过滤后，由已有的完整名称 `Intel(R) Iris(R) Xe Graphics` 直接选卡；
本机启动脚本也会为子进程清除继承的数字过滤，并在退出后还原 shell 原值。

独立 helper 检查通过：加载并预热 13.46 s，512 图编码 2.945 s，256 × 2560 的 FP32 embedding
全部有限；两张 NVIDIA 的显存读数在加载前、预热后、编码后完全相同。
随后用新版服务、CUDA 引擎和 portable Vulkan helper 重跑五项真实 HTTP 请求，全部通过：

| 请求 | 视觉编码 | 文本 prefill | 整个请求 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 512 图表 | 3.010 s | 2.735 s | 5.997 s | TEST 421；Red、Green、Blue |
| 1024 图表 | 10.617 s | 2.869 s | 13.724 s | 编号和颜色顺序正确 |
| 重复 1024 图表 | embedding cache 命中 | 2.743 s | 2.978 s | 相同回答 |
| 两张图 | 新图 10.645 s；另一张缓存命中 | 3.253 s | 14.045 s | Run 编号 1、2 |
| 图片后继续文本 | — | 0.373 s | 3.742 s | 正常输出 204 token |

最后一项的 engine decode 为 60.9 tok/s。它只是一次功能检查，不是与旧视觉服务的交替性能对照。
INFO 确认版本 0.1.40.3、K8V8、262,144 容量、9,728 + 14,848 个专家槽和 MTP 关闭。
设备修正和具体调用方式已更新到 [视觉文档](IQ_KT_VISION.md)。

五项 HTTP、长上下文、MTP 和数值检查的公开记录见
[validation.json](../bench/results/2026-10-08-iqkt-upstream-merge/validation.json)。
本次没有编译或执行 HIP/SYCL 文本后端。

## 部署与归档

原工作目录的 `feat/iqkt-vulkan-vision` 已通过 fast-forward 接收合并后的代码。
停下空闲服务后替换主程序、GPU/CPU parity 程序及 Vulkan helper，再使用原来的参数启动。
旧二进制及新旧 SHA-256 保存在本机 `build-kt/pre-upstream-0.1.40.3/`，未覆盖此前的模型或 pack。

部署后的实际检查确认：引擎 0.1.40.3、262,144 上下文、全专家驻留双卡、Intel 视觉编码；
API 文本返回 `MERGE_OK`、图片返回 `421`，未认证的 API 请求返回 401，Web 页面可打开，
`10.0.0.3:8080` 的健康检查成功。

日常仍运行 `run-strata-vision-igpu.ps1`，监听 **`0.0.0.0:8080`**，沿用本机 API key。
K8V8、2048 prefill、split 19、MTP 关闭和 PLE direct 均保持原设置。
本机配置和启动脚本只修正了视觉设备过滤；凭据、模型及构建产物不随 Git 发布。

代码检查日志为 `build-kt/merge-*.log`，部署记录为 `build-kt/merge-deployment.json`。
公开数据中保留了部署检查和二进制校验值，便于区分实际测试的程序与之后仅更新文档的提交。
