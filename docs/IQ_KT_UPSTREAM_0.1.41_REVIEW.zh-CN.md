# 上游 0.1.41：新增 128 个提交的合入评估

2026-10-08。**建议合入验证，主要收益是服务修复、长历史分词和后续维护；没有证据表明当前
KT 双卡方案会因这次更新再次翻倍。** 本次完成了源码审查和无冲突合并模拟，尚未实际合并、
编译或部署 0.1.41。在线服务仍是 0.1.40.3 加本地 KT/Vulkan 变更。

本机当前为 v4 IQ3_KT、2080 Ti 22 GiB＋USB4 4080 SUPER 32 GiB、21＋27 层、200K、K8V8、
prefill 2048、前缀缓存 6、MTP 关闭。4080 的 13,824 个专家已经全驻留，2080 仍有约 3.04 GiB
专家在 RAM。部署与测试见 [启动说明](IQ_KT_MODEL_SWITCH.zh-CN.md)和 [v4 分析](IQ_KT_V4_ANALYSIS.zh-CN.md)。

## 仓库与合并检查

| 项目 | 本次确认值 |
| --- | --- |
| 本地分支 | `feat/iqkt-vulkan-vision` |
| 本地 HEAD | `0c9e09c53fa19450d0ddac8c9cedc0710da43c58` |
| 上游 remote | `origin`，`Niko1221/Strata` |
| 用户仓库 remote | `myfork`，`Amarillys/Strata` |
| 上次合并的上游基点 | `d5ea7133741e67743c0e886bb426c0ce8d69cf6c`，0.1.40.3 |
| 本次 fetch 后上游 tip | [`fb58e0d`](https://github.com/Niko1221/Strata/commit/fb58e0dbc8399662c0e47c76578c6e878b14f6cf)，0.1.41 |
| 相对上游 ahead / behind | 6 / 128 |
| 上游变更规模 | 237 个文件；包括较多文档、社区基准及特性分支合并提交 |
| 文本合并模拟 | `git merge-tree --write-tree --name-only --no-messages HEAD origin/main`，退出 0，无冲突 |
| 模拟生成的 tree | `5b0de3220dadc0e9e6784b2f64d38df685f36d19`，不是当前工作树或合并提交 |

本地已提交变更与上游有 **7 个重叠文件**：`CMakeLists.txt`、`serve/server.py`、
`serve/test_server.py`、`src/kernels/cpu/native_expert.cpp`、`src/kernels/cuda/iq_kernels.cu`、
`src/kernels/cuda/native_mmvq.cu`、`src/prefill/prefill.cpp`。
因此可以说这次没有文本冲突，但不能仅因 KT/Vulkan 独立文件未被修改，就认定运行兼容性风险极低。
模拟检查也不包含本地尚未提交的文档和预设切换工具；这些变更已保留，没有覆盖。

## 对当前机器最有价值的更新

| 更新 | 对本机的意义与范围 |
| --- | --- |
| Tokenizer piece 缓存，`f346dc6` | 适合 Agent 反复重发长历史。上游同 token IDs，89K 示例 0.32→0.07 s；这是 CPU 分词时间，不能算作 decode 或模型 prefill 的倍数提升。 |
| 工具名空格解析，`dd66796` | `<function= NAME>` 会正确解析成 `NAME`，直接改善工具调用兼容性。 |
| 请求等待死锁、backlog 256、请求体检查，`d9621d1` / `c34dd57` / `039f9e9` | 对排队与并发客户端有用；本机没有开启批处理，不能把多 slot 流水线收益算进当前单请求吞吐。 |
| 看门狗，`d793212` / `5aee72b` / `f2e07af` | 结合 CPU、磁盘及 GPU 活动区分冻结与慢计算，并给仍在读取文件层的请求时间。冻结时终止引擎，下一请求重新加载；不等于自动重试并保住失败中的回答。 |
| AVX-512 检测移出宽指令集文件，`680d499` | 12900HK 当前使用 AVX2，这类入口检测修复有实用价值，不是 KT 算子提速。 |
| 三处 Q8_1 激活写出修复，`7a0e302` / `e372d4f` | 避免溢出 scale 与整数码不一致、NaN 向后传播。是激活路径的数值健壮性修复，不是改变 KV 为 Q8；正常值上游报告保持原位模式。 |
| HC norm scratch 复用，`ccd657a` | 减少 split norm 中重复读取和计算，可列为本机候选收益；提交中的 0.58→0.23 ms 是更早的 split norm 对照，不是本次 scratch 改动的净收益。 |

多个 API key、自动 GPU 排序、多会话停放修复也已进入上游。本机使用一个 key、显式
`--layer-split 21`，多会话停放预算为 0；这些更新不能直接解释当前 decode 变化。
自动选卡顺序的改动会保留显式手工分层。

## Prefill：不能直接套用的倍数与可以再测的路径

### 无 P2P peer prefill 是另一种分工，而且当前 KT 尚不满足 MMQ 条件

`--peer-device` 把专家分布到第二张卡，主卡保留稠密层等计算；它与我们正在使用的
`--layer-split` 互斥。本机按层分割已经让两张 GPU 执行 prefill，并非之前都留在一张卡上。

新源码 `Prefill::set_peer` 在没有 P2P 时默认允许 host 通道；`STRATA_PF_PEER_HOST=0`
才会禁用。在 compact host 路径，`STRATA_PF_PEER_SUMS` 默认打开，先在 peer 汇总每个 token 的
加权结果再回传；`STRATA_PF_PEER_F16` 还默认允许 FP16 传输。这些是特定路径的流量优化，
归约位置和 FP16 传输也有数值边界。

更直接的限制是 `set_peer` 要求 `mmq_plan().any`。合成合并树中的
`src/prefill/moe_mmq.cu::supported` **没有 IQ3_KT / IQ4_KT（154 / 155）**；
当前 48 层专家的 down 都是 KT，因此没有满足该 MMQ 计划的专家层。
本地 KT prefill 仍走按行解码为 FP16 后矩阵乘法的路径。
所以这项上游 2～2.4 倍的结果既不适用于当前分层模式，也不能只加一个开关就用于本模型；
要使用它还需要 KT MMQ 等实现和独立验证。

### CPU 分担不会自动启用在本机分层方案上

0.1.41 只在 CUDA、单 GPU、无 batch slots、chunk 小于 1024 时默认启用 CPU prefill 分担。
Layer split 仍默认关闭。显式 `STRATA_PREFILL_CPU_SHARE=auto` 可以在分层方案上测，
显式开启时默认阈值为 3072，处理小于该阈值的 chunk；3072 不是不可调整的硬上限。
分层阶段共享 CPU pool 时一次只有一个阶段占用它。

本机当前 chunk 2048，可以把它作为合并后的单独 A/B 项目：v4 的部分专家位于 RAM，
它可能减少 GPU 流送，也可能增加 12900HK 的负担。其他量化、CPU 上的数字不能直接套给 KT AVX2。

### 文件读取优化有条件适用，GGUF 不能照搬解除映射的收益

`0b030b9` 新增 stager 批量 `copy_blobs`：文件路径默认 batch 8，RAM profile 默认 1；
真正批量 direct reads 还要求源已经选择 unbuffered I/O。RAM complement 中的专家继续直接复制。
`83a7dbc` 在 GGUF 抽样页面至少 90% 已驻留页缓存时，使用 RAM stager profile，避免过多线程与环槽开销。

本机日志明确是 **GGUF shards in place，no experts.bin**。`FileExpertSource::drop_mapping`
明确保留 GGUF 的映射，因为 embedding/PLE 仍需要它；自动关闭映射的部分针对独立 `experts.bin`。
因此批量读取和热页识别值得测，但不能把“解除 NTFS 映射后 1260→1669 tok/s”整组结果当作
本机预期。当前 decode 的未驻留专家已在锁页 RAM，文件层优化主要看 prefill 借槽后取回专家的开销。

### Embedding 缓冲复用在 2048 下是每阶段 20 MiB

`199fd1d` 让 half-output `bo` 复用 `emb`。按 `2048 × 2560 × 4` 计算，每阶段节省的
逻辑缓冲区为 **20 MiB**；320 MiB 对应 32768 行。
`55dee8e` 又保留旧预算计数，默认不因此改变自动 chunk 或专家槽借用数量。
`STRATA_EMB_REUSE_ACCOUNT=1` 才让规划器使用这部分预算；它不自动给 decode 增加一层或几百个专家。

## CUDA 优化与实验开关

| 项目 | 本机判断 |
| --- | --- |
| GDN chunked recurrence | 默认关闭；`=1` 要求至少 128 SM，2080 Ti 为 68、4080 SUPER 为 80，不满足。`=2` 可强制测，但上游 3060/5070 测量反而慢，且归约数值不同，不列首要优化。 |
| 按架构 MMVQ IL 表 | 上游代码明确写着 sm89 尚无实测专用表，`kIlArch` 目前只列 sm70。该表面向类型 23/12/13/14、2～4 列；本地 KT 在入口提前走 `kt_mmvq`，不能宣称它直接给 KT 单 token decode 加速 3～20%。 |
| Foresight swap | 默认关闭，保留原专家字节和模型路由，但另占每层 VRAM 槽。上游多机测试大多持平或变慢；当前 4080 已全驻留，本阶段没有需补换的专家，不宜先挤掉常驻缓存来启用它。 |
| Route-resident | 默认关闭，明确会替换被选中的专家并改变输出。不是精确缓存优化，“只有极微小质量损失”没有本模型证据。保持关闭。 |
| Stage pin | `fb58e0d` 已恢复默认关闭。上游发布检查在 IQ3_S、4096-token prompt 后发现 decode 损坏，怀疑异步复制完成前复用缓冲；保留关闭。此处没有证明 KT 同样损坏，也没有必要开启它去承担该已知回归。 |

0.1.40.4 的紧急修复针对 Pascal sm61，Volta 的表和 expert mode 8 又由
`STRATA_SM70_TABLE=1` 控制；这些不直接对应本机 sm75/sm89。
AMD 的 hipBLASLt、Intel 的 SYCL 采样/写回改进，也不会直接加速这里独立的 Intel Vulkan 视觉编码器。

## 建议的合入与验收范围

建议以 **`fb58e0d` 完整版本**准备合并，保留其 stage-pin 默认关闭的最终修复和相互依赖的数值修复。
没有文本冲突，整合维护成本较低；不应只挑仍默认开 pin 的中间提交。先沿用当前参数，
不要把升级与改变路由、开启 MTP 或改变 KV 混在同一次吞吐比较里。

正式部署前需要完成：

1. 隔离构建 sm75/sm89，保留现用二进制；复核 KT 两卡数值对照、CPU AVX2、服务及 Vulkan 接口。
2. 先确认同参数启动仍有 **9,142 + 13,824** 个专家、4080 全驻留图、200K/K8V8 和前缀缓存。
   当前捕获时 CUDA 最低余量约 75 MiB，已经运行验证；新版本的分配量仍需重新实测。
3. 同提示词、相同 GPU 唤醒方式、新旧交替测 prefill/decode；先默认配置，再单独测
   `STRATA_PREFILL_CPU_SHARE=auto` 等候选，记录缓存命中、CPU、频率和专家驻留数量。
4. 完成 204K 输入、持续输出、立即续聊缓存命中，以及核显图片/Web/API/原模型别名复核。
   长缓存对照应避免其他会话插入，当前没有多会话停放。

本次没有做这些新版本运行测试，也没有改动线上引擎或凭据。现有 21＋27 全驻留的测量结果
来自旧的 0.1.40.3＋KT 构建，不能算作 0.1.41 的升级收益。

主要源码依据（固定在本次上游 tip）：

- [Prefill：peer、CPU share、stager 和缓冲预算](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/src/prefill/prefill.cpp)
- [MMQ 格式范围](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/src/prefill/moe_mmq.cu)
- [专家文件源：drop_mapping / copy_blobs](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/src/core/expert_source.cpp)
- [MMVQ 架构表](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/src/kernels/cuda/native_mmvq.cu)
- [实验开关与上游实测边界](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/docs/DETAILS.md)
- [服务端实现](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/serve/server.py)
