# IQ_KT 双卡适配与优化总结

记录范围：2026-10-07 至 2026-10-08，Windows、Core i9-12900HK、64 GiB RAM、RTX 2080 Ti 22 GiB、
RTX 4080 SUPER 32 GiB，以及 Intel Iris Xe 核显。以下结果针对本机的 Qwen3.8-Flash-Next Uncensored
IQ3_KT_v2 主模型；后续 v4 的独立分析和基准见 [v4 显存与优化](IQ_KT_V4_ANALYSIS.zh-CN.md)。
IQ4_KT 已接入并验证，但尚未测试完整 IQ4_KT 主模型的吞吐。

本次完成了 KT 格式接入、CPU/CUDA 算子优化、双卡缓存与分层调整、长上下文验证、核显视觉接入和服务部署。
v2 短/中等输入下 decode 实测约 **58–63 tok/s**，MTP 关闭，全部 routed experts 在 decode 时驻留显存。
**262,144-token 容量**保留；实际 260,000-token 输入已完成检索测试。

这两项结果的测试条件不同：260K 测试只生成了 8 个 token；58–63 tok/s 来自较短输入后的持续输出。
目前证据支持“短/中等输入 decode 明显加快，实际长输入可以运行”，尚不足以证明满上下文持续保持该速度，
或任意任务都没有质量变化。

本机现在支持 v2 / v4 按需切换，共用 8080、原 API key、Web 和 Intel 核显视觉。
当前保持 v4 在线：**200K / K8V8 / 21+27 层**、前缀缓存开启，预留 **1,024 / 128 MiB**。
2080 驻留 9,142 个专家；4080 的 **13,824 / 13,824** 个专家全驻留，约 **3.04 GiB** 专家仍在 RAM。
3,305-token 合成输入、两次各 256-token 输出的平均 decode 为 **42.55 tok/s**；
同提示词 22+26 对照为 40.96，回切复测为 41.55，不能据这组小差距外推所有任务。
两次各 1,024-token 持续输出为 **40.1 / 40.8 tok/s**。
204,018-token 输入 prefill 为 **226.903 s**，256-token 输出为 **36.5 tok/s**，三个检索串均正确。
长对话的独立续聊复用 **204,306** token，仅读入 **23** 个、prefill **0.484 s**。
v4 不沿用上面的 v2 全模型专家驻留和约 60 tok/s 性能结论。
使用方法见 [按需启动](IQ_KT_MODEL_SWITCH.zh-CN.md)。

最初的 21＋27 在 4080 预留 768 / 512 MiB 时分别缺 268 / 139 个专家，短输入平均 decode
为 39.70 / 40.36 tok/s。后来按用户要求实际装齐专家，较小预留下完成短、长和持续输出验证，
因此采用当前配置。CUDA 图捕获最低空闲约 75 MiB，NVML 测试最低约 532 MiB。
长输入首次续聊的缓存断言曾被另一客户端插入的请求打断；独立续聊通过，原始记录保留。
完整对照和限制见 v4 分析文档。

本机此前已合入上游 0.1.40.3；同配置的新旧交替基准吞吐基本持平，KT 优化保留。
合并及复测结果见 [上游合并记录](IQ_KT_UPSTREAM_MERGE.zh-CN.md)。下文的算子优化数字保留原始对照口径，
不重复计算为本次上游合并收益。

之后的上游 **0.1.41 / `fb58e0d`** 又新增 128 个提交，本次已完成源码与合并可行性审查，
**尚未合入或部署**。建议为服务修复和后续维护合入验证，不能直接套用其他多卡模式的翻倍数字。
结论见 [0.1.41 更新评估](IQ_KT_UPSTREAM_0.1.41_REVIEW.zh-CN.md)。

## 文档入口

| 文档 | 内容 |
| --- | --- |
| [IQ_KT.md](IQ_KT.md) | 格式布局、实现范围、构建/打包命令、IK 数值对照、MTP 侧车 |
| [IQ_KT_PERFORMANCE.md](IQ_KT_PERFORMANCE.md) | 频率问题、MTP/分层/缓存对照、KV 对照、packed CUDA 算子收益 |
| [IQ_KT_DECODE_PROFILE.md](IQ_KT_DECODE_PROFILE.md) | **优化前版本**的 decode 分段计时及其局限 |
| [IQ_KT_PREFILL.md](IQ_KT_PREFILL.md) | prefill 瓶颈、并行解码器、2048/4096 分块、实际 260K 输入 |
| [IQ_KT_VISION.md](IQ_KT_VISION.md) | Vulkan 核显视觉、启动脚本、Web/API、设备选择和实测 |
| [IQ_KT_UPSTREAM_MERGE.zh-CN.md](IQ_KT_UPSTREAM_MERGE.zh-CN.md) | 上游 0.1.40.3 合并、同配置 A/B、Windows PLE RAM 与功能复测 |
| [IQ_KT_UPSTREAM_0.1.41_REVIEW.zh-CN.md](IQ_KT_UPSTREAM_0.1.41_REVIEW.zh-CN.md) | 新增 128 个提交的适用性、合并检查与待验证项目；未部署 |
| [IQ_KT_MODEL_SWITCH.zh-CN.md](IQ_KT_MODEL_SWITCH.zh-CN.md) | v2/v4 按需启动、停止、模型别名、8080 和视觉验证 |
| [IQ_KT_V4_ANALYSIS.zh-CN.md](IQ_KT_V4_ANALYSIS.zh-CN.md) | v4 权重变化、200K 基准、CPU/流送开销与优化优先级 |

各专题保留了历史配置和对照结果。日常部署以按需启动文档的两套预设为准；
本页后续的算子优化和 K8V8 数字主要是 v2，4096 是 v2 另一个经过长输入测试的配置。

## 当前部署配置

截至 2026-10-08，`run-strata-v2.bat` / `run-strata-v4.bat` 选择模型，
`run-strata-stop.bat` 停止服务。下面是 v2 的详细配置，v4 的并列表见按需启动文档。
原 `run-strata-vision-igpu.ps1` 仍选择 v2，对应
`strata-iq3kt-dual19-int8-pf2048-vision-igpu.json`。

| 项目 | 当前设置 |
| --- | --- |
| 文本模型 | Uncensored IQ3_KT_v2；routed experts 为 IQ3_KT，部分稠密投影使用 IQ4_KT |
| 第一阶段 | 2080 Ti，主模型层 0–18，共 19 层；9,728 个专家 |
| 第二阶段 | 4080 SUPER，主模型层 19–47，共 29 层及输出头；14,848 个专家 |
| Decode 专家驻留 | 24,576 个，约 44.4 GiB，全部在两卡显存中 |
| 显存预留 | 第一张卡 1,536 MiB，后续卡 768 MiB |
| MTP | 关闭：启动参数中没有 `--mtp`；保留 `--spec 3` 本身不会启用 MTP |
| 上下文 | `--max-context 262144` |
| KV | `--kv int8`，即 K8V8；`--kv-resident 20480`，其余通过 RAM 中的完整 KV 池访问 |
| Prefill | `--prefill 2048`；工作区临时借用专家缓存，结束后恢复 |
| 前缀缓存 | `--prompt-cache 6`，检查点位于 RAM；v2/v4 均已恢复开启 |
| 其他对照设置 | suffix draft、lookup chain、adaptive expert swaps 关闭 |
| Ngram / PLE | 保留原表，约 25.03 GiB，默认 `--ple-io direct`；SSD 按需读取，RAM 行缓存 |
| 视觉 | 原 F16 MMPROJ，由独立 Vulkan 进程在 Intel Iris Xe 上编码；每图最多 768 个视觉 token |
| 服务 | `0.0.0.0:8080`，保留 Web Chat/Monitor，API key 已启用 |

两卡显存需要按层分别分配。54 GiB 是容量之和，不是任意一张卡都能直接使用的统一显存池。
单序列 decode 依次执行两个阶段；prefill 可以让两卡处理不同分块形成流水线。

前缀缓存于 2026-10-08 晚间开启并实测：5,767-token 续聊复用 5,741 token，只新读入 26，
prefill 为 0.380 s；从旧消息分支也能复用。之前的吞吐对照保留关闭前缀缓存时的测量条件。
检查点设置及验证见 [按需启动文档](IQ_KT_MODEL_SWITCH.zh-CN.md#前缀缓存)。

## 完成的实现

1. **原生 IQ3_KT / IQ4_KT。** 接入 GGUF 扩展类型 154/155，保留原来的 ggml 依赖和类型枚举。
   独立模块处理逐行 scale、256 元素主块、尾块及四字节对齐，覆盖 160、320、640 等非整块宽度。
   Python/C++ 模型读取、打包、embedding、稠密投影、routed experts、PLE 和 prefill 均已接通。
2. **CUDA decode。** IQ3/IQ4 分别生成打包的整数码，直接交给 DP4A，减少逐值重建再打包的工作。
   专家和稠密 KT 投影共同受益；保留 Q8_1 激活、行分组、归约顺序和 scale。
3. **CUDA prefill。** 用多个 lane 并行重建相邻值，再以向量方式写出 FP16/BF16/FP32。
   继续使用已有矩阵乘法路径。专用 KT 量化 MMQ 和融合专家 prefill 尚未实现。
4. **CPU 回退。** 使用共享 trellis 表和 AVX2 整数点积，跨 token 复用解码工作；保留标量回退及
   `STRATA_KT_SCALAR=1` 对照。v2 全驻留配置的 decode 无需 CPU 专家计算；v4 会使用此回退。
5. **MTP 侧车。** 专家保持原始 KT 字节，必要的稠密投影转换为现有运行时要求的 Q8_0/BF16，F32 norm 保持原值。
   添加带布局信息的 `experts.kt` 完成标记及加载校验。功能可用，当前默认配置关闭 MTP。
6. **视觉和电源处理。** 增加独立 Vulkan 编码器、按设备名称选择、编码器专用环境变量，以及只保留词表元数据的
   GGUF 导出器。加入请求触发的 GPU 唤醒包装器，复用本机已有 `gpu_monitor.py`。
7. **构建与复核。** 完成 sm75/sm89 构建，统一 Windows CUDA runtime 链接配置；移除新 KT 文件对 CUDA 专用
   BF16 头的直接依赖。执行验证覆盖 Windows CUDA 和 x86 AVX2；HIP/SYCL 尚未执行验证。

主模型打包保留原始专家和 PLE 数据，只转换运行时要求的小型 BF16 投影，并记录转换。
当前 pack 记录了 388 个舍入转换的张量，共 1.31 GiB；因此“保留原量化专家”不代表整个执行链逐位等同于 IK。

## Decode 提速来自哪里

### 先解决了 4080 的低频状态

直接启动早期 Strata 测试时，4080 曾在整个 decode 期间停在 P8、核心/显存 405 MHz，只有约 4.9–5.0 tok/s。
用户原来的 `run_uncensored_server_ik.ps1` 经代理调用了 GPU 唤醒逻辑。复用该逻辑后，同配置、相同输出恢复到约
32 tok/s。这些低频结果已从正常性能对照中排除。

当前包装器在 engine READY、专家缓存分配完成后建立并保留 CUDA probe，在每次生成前执行一次短矩阵运算。
它不会持续与 decode 争抢计算；空闲 120 秒或正常退出时释放时钟设置。直接用 `python -m serve.server`
启动 JSON 配置会绕过这层包装。

### MTP 关闭后，重新分配显存和层数

在 72/3,570-token 输入、每次输出 256 token 的对照中，MTP 接受率约 54%，每窗口约产出 2.06 token，
但窗口耗时约 68–71 ms；不开 MTP 单 token 约 31 ms。因此这组实际工作负载中 MTP 更慢。
早期数数提示几乎 100% 的接受率不能代表普通长回答。

MTP 草稿层及裁剪输出头在 4080 上占约 1.54 GiB。释放后还要移动分层点，才能让新增显存容纳更多主模型专家：

| 优化前 CUDA 算子下的配置 | 专家驻留情况 | 3,570-token 输入后的 decode |
| --- | --- | ---: |
| Split 21，MTP 开 | 部分专家仍在 RAM | 28.84–29.49 tok/s |
| Split 21，MTP 关 | 仍有约 1.92 GiB 专家在 RAM | 31.68–32.23 tok/s |
| Split 20，MTP 关 | RAM 专家降至约 0.94 GiB | 33.80–33.89 tok/s |
| Split 19，MTP 关，后卡预留 768 MiB | 全部 24,576 个专家驻留 | 38.03–38.93 tok/s |

最终两个阶段均进入全驻留 CUDA graph 路径，decode 的 CPU 专家和 PCIe 专家计数为零。
CPU 调度、PLE 读取、跨卡状态交接和长上下文 KV 流送仍然存在。

### Packed KT 算子带来了可单独复现的收益

同一 K8V4/2048、Split 19、MTP 关闭、全驻留配置，交替运行旧/新二进制：
每个版本每种输入长度共四次有效测量，每次输出 256 token，另有预热。

| 输入 token | 原算子 decode 均值 | Packed 算子 decode 均值 | 增幅 |
| --- | ---: | ---: | ---: |
| 72 | 39.21 tok/s | 63.37 tok/s | 61.6% |
| 3,570 | 39.43 tok/s | 62.22 tok/s | 57.8% |

对应请求的全部输出 token 一致。另一次固定核心/显存频率的对照仍测得约 60.6–65.0% 提升，
支持收益来自算子变化，而非仅由新负载下的自动升频造成。后续 K8V8 prefill 测试的 256-token decode 约为
58.5 tok/s；视觉 HTTP 测试后的一次 204-token 文本输出为 59.3 tok/s。

### 如何与以前的 IK 比较

旧 IK 对照也已经把 49/49 层放在两张 GPU 上。其约 30 tok/s 的历史观察不能简单解释为“专家在 CPU”。
本次保留优化后的 IK 构建，使用同一输入 token 做对照，测得短输入 40.47–41.34 tok/s、
3,570-token 输入 37.49–37.50 tok/s。

IK 对照使用 48,128 容量、K8V4、ubatch 512；Strata 的容量、KV、分层和部分权重表示均有差异。
因此，约 30→60 tok/s 可以描述用户不同阶段的体验；本次 Strata 新旧算子对照测得约 58–65% 收益，
不能据此宣称所有 IK 工作负载都翻倍。全驻留之后，KT 重建/点积和执行调度仍会影响速度；
两套引擎各项机制的独立贡献尚未做完整消融。

## Decode 的耗时分析与后续空间

下面是 **packed 优化前**、3,570-token 输入的 GPU 分段结果，不能作为当前约 60 tok/s 版本的耗时占比。
开启 profiler 会改变重叠执行并增加读回；百分比以两卡 stage 合计 28.91 ms/token 为分母。

| 分组 | ms/token | 占比 |
| --- | ---: | ---: |
| Routed experts | 9.43 | 32.6% |
| Shared expert | 1.93 | 6.7% |
| QKV、query/indexer、gate、output 投影 | 8.50 | 29.4% |
| Hyperconnection、归一化、路由及包含 PLE 的区间 | 5.55 | 19.2% |
| GDN 递归/卷积、QSA attention/KV | 2.40 | 8.3% |
| 词表输出头和 token 选择 | 0.80 | 2.8% |
| 剩余等待、combine、层间间隙 | 0.32 | 1.1% |

专家与稠密投影合计约 69%，因此本次优先优化两者共用的 KT 算子。该旧版 profile 中，2080 的 19 层占
61.7% 的 GPU stage 时间，4080 的 29 层及输出头占 38.3%。容量合适的分层并不一定让两卡耗时均衡。
这些时间区间也无法单独区分算力、访存、kernel launch 或 host 提交等待，需要进一步工具测量。

当前没有证据表明已到硬件极限。下一轮应先重测优化后的短/长输入 profile，再选择目标：

1. 检查 2080 上剩余 KT kernel 成本、稠密投影、路由和混合操作，评估分层移动的计算收益与显存代价。
2. 对 prefill 评估专用 KT MMQ 或融合 grouped-expert 路径，减少完整 FP16 中间矩阵和逐专家调用。
3. 补实际 200K/260K 输入后的长输出、代表性任务质量和 MTP 接受率；再决定是否重新启用 MTP。

全驻留配置的短输入结果不支持把剩余瓶颈主要归给 USB4 专家传输，也不能用短输入的 8.3% attention 占比推断
实际 260K 输入的瓶颈。

## Prefill、CPU 和分块

48 个主模型层的 prefill 计算都分配在两张 GPU 上。CPU 负责提交、数据准备和 I/O。
日志中“少一个 token”来自最后一个 prompt token 进入首个 verifier 窗口，并不表示有一整层交给 CPU 计算。
Prefill 借用专家缓存后可以通过 streamed ring 提供权重；“decode 全驻留”不能推导为“prefill 无数据传输”。

`--prefill 2048` 表示每次批量处理的 prompt token 数，与 IK 的 ubatch 用途接近；两套实现的工作区和调度不同。
它不会把 256K 上下文截成 2048。2048 是当前日常部署的选择，4096 对更长输入另有收益和显存成本。

优化前，专家权重解码占被测 prefill 阶段时间的约 49–66%。并行解码器使跨设备列出的专家解码区间合计
从 6.44 s 降至 2.17 s，约减少 66%。这些设备时间存在流水线重叠，不能相加当作请求墙钟时间。
正常、不开 profiler 的 K8V8 对照如下，每格为两次有效测量的均值：

| 实际输入 token | 原解码器，2048 | 新解码器，2048 | 新解码器，4096 |
| --- | ---: | ---: | ---: |
| 3,570 | 413.3 tok/s | 641.7 tok/s | 647.6 tok/s |
| 12,150 | 634.2 tok/s | 1,012.6 tok/s | 1,158.1 tok/s |

固定 2048 分块，收益约 55–60%；12,150-token 输入再改为 4096，增加约 14.4%。
本地部分文件名写作 `16k`，实际 tokenizer 数为 12,150，本页采用实际数量。
在 3,570-token 输入上，新的 2048 prefill 仍低于前述 IK 对照的 805–824 tok/s；decode 的收益不代表所有阶段
都已超过 IK。

2048 分块的专家槽借用量随路径变化：纯文本对照约 1.24 GiB/卡，当前视觉路径约 1.30 GiB/卡；
4096 对照约 2.05 GiB/卡。Prefill 结束后恢复专家，测量包含准备和恢复成本。
固定分块的新旧解码器输出在测试中一致；2048→4096 曾使一个 256-token 回答从第 91 个输出 token 开始不同。
旧解码器使用 4096 的控制组又与新解码器/4096 一致，说明该差异跟随分块变化。

实际 **260,000-token、无 prefix cache、K8V8、4096 分块**测试耗时 **198.126 s**，即 **1,312.3 tok/s**。
它正确返回输入开头、中间、末尾放置的三个值。此前 K8V4/2048、旧解码器的同一检索输入为 388.187 s；
两次同时改变了 KV、分块和解码器，不能把整个差值归给其中一项。
新的长输入测试输出仅 8 token、耗时 181.0 ms，适合验证容量/检索，尚不足以报告持续满上下文 decode。

## KV、上下文和 ngram 放在哪里

`--kv int8` 已经量化 K 和 V，即 K8V8。Strata 的 K8V4 使用 INT8 K 和旋转 Q4_0 V；
它与 IK 的 K=q8_0/V=q4_0 具有相同位数分配思路，但存储布局和计算过程不同。
当前完整 KV 池在 RAM 中，每个 QSA 层保留 20,480 个 GPU resident cells；从 RAM 读取更早状态时仍保留完整上下文。

本模型的 12 个 QSA 层，仅计算 K/V 数据池：

| 262,144 容量、20,480 resident cells | K8V8 | K8V4 |
| --- | ---: | ---: |
| 每层每 cell 字节数 | 1,056 | 816 |
| 完整 host K/V 池合计 | 3,168 MiB | 2,448 MiB |
| 两卡 resident K/V 池合计 | 247.5 MiB | 191.25 MiB |

K8V4 节省 720 MiB host 池、56.25 MiB resident 池；还需另计页表、GDN 状态和 prefill 缓冲。
v2 专家已经全部放入显存，这 56.25 MiB 不会再增加专家数量。缩至 204,800 容量也不会缩小固定的
20,480-cell resident 池；目前没有必要为了本次已验证的分配放弃 256K 容量。

K8V8 的 V 有更多精度余量，当前内存预算允许保留它。尚未做足够的质量评测来量化其相对 K8V4 的收益。
IK 与 Strata 的激活量化也不同：本实现 CUDA 为 Q8_1，CPU 为 Q8_0，IK 为 Q8_2_X4。
KT 数值对照、若干输出一致和检索成功，都不能扩展成全任务“零质量损失”的保证。

**Ngram/PLE 的约 25.03 GiB 原表仍在 SSD。** 当前配置未覆盖默认 `--ple-io direct`，所以按需进行无缓冲读取，
并在 RAM 保存有上限的行缓存，默认 1,048,576 行。命中的行直接从缓存取，未命中再读 SSD。
PLE 没有关闭，也没有把整张表常驻 RAM。初次部署版本的 Windows 实现不支持 `--ple-io ram`；
随后合入的 0.1.40.3 已增加 `VirtualLock` 支持，本机复测确认整表锁定成功。
这是可选设置，当前日常配置仍为 `direct`；`mmap` 与整表锁定常驻也不同。

## 两种链路和更大的 IQ4_KT 模型

2080 Ti 实际连接为 PCIe 3.0 ×8，理论单向约 7.9 GB/s，本次测得 H2D 约 6.3 GB/s；
4080 SUPER 经 USB4，测得约 2.9 GB/s。约两倍的传输差异只在需要从 host 搬数据时发挥作用。
专家已经在 4080 显存里时，其读取无需经过 USB4。

有效传输成本取决于“实际搬运字节数 ÷ 有效带宽”。2080 单卡虽然链路较快，专家缓存更小、计算能力也不同；
因此保留双卡容量仍有价值。早期让 2080 承担更多 RAM 专家流送是缓存不足时的策略；最终 IQ3 配置通过关闭 MTP、
调整分层与预留，让 decode 的专家流送需求归零。单卡的初期功能 smoke 不足以建立最佳单卡性能排名。

用户提出的约 47G→54G 主权重变化尚无完整 IQ4_KT 主模型 A/B 实测。文件体积、专家体积和运行时显存预算需要
分别统计，尤其要区分 PLE、稠密投影、专家、KV、工作区和 MTP；不能用 GGUF 总体积推算驻留率。
若较大的专家越过当前有效显存预算，可能重新引入 CPU 回退或 host→GPU 流送，速度变化就不再与文件增幅成比例。
反过来，缓存未驻留比例也不等于路由访问的未命中率。

下一次比较完整 IQ4_KT 时，应先读取实际张量尺寸，重算每卡缓存与分层，再使用相同输入/输出长度、唤醒策略和 KV
设置测量 decode、prefill、专家未命中和传输。当前 IQ4_KT 的投影/MTP/数值验证不能替代这一步。

## 视觉、Web 和 API 的使用入口

本机 PowerShell 启动 v4：

```powershell
& B:/llama.cpp/Strata/run-strata-v4.ps1
```

| 用途 | 地址或位置 |
| --- | --- |
| 本机 Web | `http://127.0.0.1:8080/` |
| 局域网 Web | `http://10.0.0.3:8080/` |
| OpenAI 兼容 base URL | `http://10.0.0.3:8080/v1`；本机也可用 `127.0.0.1` |
| API key | 本机配置的 `api_key` 字段，另存于 `strata-vision-api-key.txt`；在 Web 设置和 harness 中填写 |
| 服务日志 | v2：`strata-vision-igpu.log`；v4：`strata-v4-vision-igpu.log` |

MMPROJ 已加载到独立的核显进程。编码器按 Intel 的完整设备名称选择设备，启动脚本清除容易随启动变化的
Vulkan 数字过滤；编码器自己的 `CUDA_VISIBLE_DEVICES=-1` 不会改变文本引擎的设备选择。
辅助进程单独测试期间，两张 NVIDIA 的显存读数未增加。新 1024×1024 图像编码约 11.5 s，
重复图像命中 embedding cache 后可省去编码；文本 prefill 仍执行。768 上限按每张图计算。

简单图表的 OCR、颜色、多图和图片后继续文本均已通过实际 HTTP 测试。该样本不足以衡量复杂文档、定位任务或
降低图像 token 上限后的质量。构建、词表导出、具体测试时延见 [视觉文档](IQ_KT_VISION.md)。

启动脚本、JSON 配置、API key、模型和构建产物属于本机部署文件，未提交到 Git。仓库包含通用编码器、服务支持和
`tools/serve_gpu_wake.py`；唤醒包装器依赖另外安装的 `gpu_monitor.py`，本机通过
`--power-module-dir B:/llama.cpp/proxy` 指定。其他机器需按文档构建、打包并提供自己的路径和凭据。
安装器和模型目录尚未加入这套 KT 配置。

## 验证和代码归档

提交前复核通过以下检查：

| 范围 | 结果 |
| --- | --- |
| Python | 共 48 项：打包 24、视觉/服务 20、词表导出 1、唤醒生命周期 3 |
| 独立 IK DLL 对照 | 37 组随机/真实行、829 个真实张量跨度；最大相对解码误差 1.73×10⁻⁷ |
| sm75 / sm89 | 两卡各 37 组；GEMV 最大相对误差 4.70×10⁻⁷；含尾块、偏移、步长、padding、embedding、grouped experts |
| FP16/BF16/FP32 输出 | 对照标量解码器，检查舍入、交错行和未覆盖区域 |
| CPU AVX2 | 37 组 × 1–9 token，gate/up/down 与标量结果完全一致 |
| MTP | 26 个稠密张量对照，最大相对舍入误差 0.00393；F32 norm 不变；3 个完整专家 blob 字节一致 |
| 构建 | 主程序、GPU/CPU parity 目标通过；视觉 Vulkan helper 构建及设备校验通过 |

核心实现已提交到 `feat/iqkt-vulkan-vision`：

- [`5120816`](https://github.com/Amarillys/Strata/commit/51208166f208cfc238a2af915b5cd7ab599e9ae4)：
  IQ3_KT/IQ4_KT 接入、CPU/CUDA 优化、MTP、数值验证及性能文档。
- [`b2c769a`](https://github.com/Amarillys/Strata/commit/b2c769ae620b9d094a1b1f26888c6a804342f037)：
  独立 Vulkan 视觉、词表导出、GPU 唤醒和服务测试。

本机复核日志在 `build-kt/commit-*.log`；性能与视觉原始记录在各专题列出的 `build-kt/` 文件中。
这些日志和模型数据不随代码发布。为保留正在运行的服务，提交前复核构建写入 `build-kt/review-bin/`，
未覆盖被占用的服务可执行文件；临时构建输出目录设置已恢复。

## v2 / v4 按需启动（2026-10-08）

本机已保留两套预设：v2 使用 256K / K8V8 / 19+29 层，v4 使用 200K / K8V8 / 21+27 层，
MTP 都关闭。`run-strata-v2.bat` 和 `run-strata-v4.bat` 会等待当前请求结束后切换模型，
共用 8080、原 API key、Web 和 Intel 核显视觉；`run-strata-stop.bat` 关闭服务。
原 `run-strata-vision-igpu.ps1` 仍选择 v2。

v4 已完成文本与图像 HTTP 验证，并保持在线供能力测试。启动方式、模型别名、缓存情况及
验证边界见 [按需启动文档](IQ_KT_MODEL_SWITCH.zh-CN.md)。

v4 全驻留所需 GPU 权重载荷比 v2 增加约 4.34 GiB；PLE 表缩小发生在 SSD，不能抵消显存增量。
当前流送比例 0.5 在三个候选中最快，但缓存仍沿用通用 profile、动态交换关闭、CPU 默认使用全部物理核。
这些是下一轮可验证的优化方向；进一步量化 KV 只能节省几十 MiB resident 数据池。
详细测量口径、内存账目和源码核对见 [v4 显存与优化分析](IQ_KT_V4_ANALYSIS.zh-CN.md)。
