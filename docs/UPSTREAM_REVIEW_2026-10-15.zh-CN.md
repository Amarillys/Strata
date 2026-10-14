# 上游 v0.1.44.1 更新评估（2026-10-15）

## 基线与范围

- 本地分支：`feat/ik-support-with-vulkan-vision`，HEAD `27a80e97`。
- 本地和上游的共同祖先：`a0116b27`，标签 `v0.1.43.1`。
- 本次 `git fetch origin` 后的上游：`296b59d2`，标签 `v0.1.44.1`。
- `git rev-list --left-right --count HEAD...origin/main`：本地独有 2，上游独有 282。
- 上游 282 中，250 个非合并提交、32 个合并提交；不等于 282 项独立功能。
- 上游相对共同祖先：358 文件，新增 47,208 行、删除 1,243 行。
- 初始工作区干净。只更新远程引用并做 Git 合并预检，没有合并到工作分支、构建、加载模型或重启服务。

`main` 分支还停在更老版本，其落后数量不是本次分析的基线。
原始更新范围：[GitHub compare](https://github.com/Niko1221/Strata/compare/a0116b2782a5b82622358c04a06496485d12b2d6...296b59d2)。
分析依据为本次 fetch 得到的提交、代码及随仓库提交的测量记录；网页 compare 未能通过浏览工具加载。

## 本机适用性

本地 `strata-active.json` 记录的预设为 `v4-lite`，指向
`strata-iq3kt-v4-lite-k8v8-pf2048-vision-igpu.json`。这是配置记录，不代表已核验正在运行的进程。
只提取了评估所需字段，没有输出密钥。

该预设为 IQ3_KT v4、双卡 layer split 19、prefill 2048、int8 KV、
max-context 307200、kv-resident 20480、conversation-pool-tokens 524288；
视觉设备为独立 Vulkan Intel Iris Xe，max_tokens 768。
配置中 `--spec 3` 存在，但没有 `--mtp`，因此不能把它当作已经开启 MTP。
`--adapt-every 0`、`--suffix-draft 0`、`--lookup-chain 0`，PCIe share 固定为 0.5。

## 优先引入

| 改动 | 提交 | 价值与限制 |
|---|---|---|
| 图片会话缓存按图片内容标识及网格匹配 | `cf1b0edf`、`ee6d8bac` | 编码器重启或缓存淘汰后重新编码同一图片，浮点 embedding 的变化不再单独导致会话缓存失效。适合我们的独立 Vulkan 编码器。需要服务器和引擎一起更新；不是单改 Python。 |
| Responses API 遵循配置的 reasoning_effort，保留 false | `5d08c753` | 修复 Responses 请求未指定等级时忽略配置的问题，并修复 false 在转换中丢失。若客户端使用 Responses，能避免意外使用更高思考等级；不是通用吞吐优化。包含 frontend 和回归测试，不能只移植 server 的一段。 |
| 长 prefill 在下一 chunk 给短请求让路 | `ac21812c` | 修复并发请求调度中晚让路或不让路的问题。适合大上下文和多客户端；提升响应公平性，不能理解成单请求 prefill 提速。 |
| stdin reader 的 QUIT 生命周期竞争 | `6339e861` | 稳定性修复，避免借用主线程局部状态造成退出竞争。 |
| 所有分卡的启动 VRAM 检查 | `b5217c8b` | 输出后续每张卡的显存余量；不足 512 MiB 时警告并建议预留值。只诊断，不拒绝启动或自动修改配置。 |
| 引擎启动失败区分 GPU runtime 和非法指令 | `3d5819df`、`d981a510`、`9aa674ed` | GPU runtime 启动故障按 2/4/8 秒重试；后续补丁避免把真正的崩溃错误当作可重试启动。应作为完整修复组引入。 |

图片缓存代码仍需验证图片更换、顺序更换和网格变化会拒绝错误的复用；不能只测重复图片命中。

## 值得做本机 A/B 的性能改动

### 合并解码状态提交

`6f3e3810`：用批量 kernel 提交多个 GDN 层的 conv history/state 和 QSA 层的 tail/indexer，
减少每窗口的提交节点。`STRATA_COMMIT_BATCH` 默认开启，`=0` 保留旧路径。

上游 R9700 + UD-Q4_K_XL 的提交 GPU 时间为 2.35 → 0.65 ms，
GDN 和 QSA 的 parity 检查记录为 bitwise。
这是提交阶段的测量，不是整机 decode 快了 3.6 倍。
实现不依赖 KT 权重解码，理论上适用于我们的模型，但 NVIDIA 双卡收益必须实测；
无 MTP 仍有验证窗口状态提交，不能因此直接排除。

### 双卡 MTP 流水线

`24d972e5`：流水线解码可把 MTP 放到第一张卡，减少它与最后一张卡 verifier 的争用，
并把 PCIe share 更新接入流水线循环。
默认开启条件针对 HIP gfx12，不会自动给我们的 NVIDIA 双卡开启。
需要 MTP、draft vocabulary 子集、合适的 pipeline 参数，且不能与 batch 任意混用。

当前预设没有 MTP，所以单纯合并该提交不会得到其流水线收益。
若后续测试，必须先验证本地 KT MTP sidecar，分别比较串行、流水线、MTP 第一卡，
同时看接受率、显存和长上下文正确性。旧版文档已有的 pipeline 加速不能算成这 282 个提交的新收益。
上游关于 8192 相对 16384 chunk 的 +14%/+19% 测量来自 gfx12 双卡/四卡；
我们的显式 prefill 2048 不受该默认规则调整影响。

### HC Q8 与 dense Q8 GEMV

`0a379a1f`、`b60aab1b`、`446bec51`、`2d47caa4`：完善 `STRATA_HC_Q8=1`，
包括加载时分配 scratch、跨 stream 排序、BF16 pack 的 Q8 转换以及单卡不保留 BF16 副本。
`945fee23` 改进部分 dense Q8 GEMV 的线程布局。

上游 dense GEMV 的 R9700 UD-Q4_K_XL 整体 decode +1.0–2.7%；
NVIDIA 的部分 kernel 测量来自 sm86。我们的 sm75/sm89 和 KT dense 路径不保证落在同一分支。
HC Q8 涉及数值变化；单卡省显存路径也不能直接套到双卡。
v4-lite 的实际 HC tensor 格式和已有 pack 转换必须先核验，再决定是否开启。

### opencode 固定系统前缀缓存

`d8a49433`：`env_to_user: true` 把 opencode system 中的 `<env>` 目录/日期块移到第一条 user 消息，
避免目录或日期改变导致长系统前缀缓存失效。默认关闭。
上游 RTX 5070 Ti Laptop + Swift IQ3_XXS 换目录新会话：11.9 → 1.1 秒，复用了 13,693 tokens。
仅对相应请求形状有效，改变消息位置，需用真实 agent 请求验证；不能当成所有 API 十倍加速。

## 可以保留代码，当前不建议启用

| 改动 | 原因 |
|---|---|
| `640ac438` prefill-head-alias | 明确拒绝多 GPU/layer split，并要求 MTP。上游小显存单卡长 prompt 有收益，但短增量请求更慢。当前预设不适用。 |
| `ca41a6b9`、`8488c06b` elastic vision | 编码器按需加载并借用文本显存。我们已经把编码放在 Intel Vulkan 独立设备上；其 NVIDIA 显存借用策略不应直接开启，且冷加载成本需评估。 |
| `1aa0abf6` arena adapt-async | 这次新增模式限定单 GPU。当前双卡且 adaptive 关闭，不会直接受益。 |
| `cb22a53e` Windows adaptive overlap 信号/flush | 修复 WDDM 提交延迟造成的等待；但上游明确说明 ADAPT_OVERLAP 不运行于 layer split。属于备用路径稳定性。 |
| `277be72a` foresight swap | 修复 adaptive exchange 后槽位可能持有错误专家字节的竞争。重要正确性修复，但只在启用 FS 槽位及相关交换路径时触发；当前 adaptive 关闭。 |
| `caf1856b`、`54b61dbb` UMWAIT | 减少闲置 CPU worker 自旋，需要 CPU WAITPKG 和 MSVC。不是保证 decode 加速，尚未核验本机 CPU 是否支持。 |
| `ab6bce05`、`55e12831`、`f087f780` pin-by-need | opt-in 的 pinned 内存布局改造，涉及 padded layout/shared arena。与 KT 行字节和 CPU fallback 一起验证后再启用。 |
| `7d7da881` prompt direct I/O、THP 改动 | Linux 路径，当前 Windows 服务没有直接收益。 |
| P40 sm61 反量化及 dense F32 | Tesla P40 特定路径，2080 Ti/4080 SUPER 不适用。 |
| SYCL Arc、HIP gfx11/gfx12 优化 | 不能作用于 Vulkan Intel 视觉 helper；Vulkan 与 SYCL 是不同后端。 |

## 新功能概览

- Apple Silicon Metal：包含 M4/M5 kernel、MTP、vision、流式专家 arena、安装和内存上限保护，是新增行数的重要来源。
- Qwen3.6-35B-A3B：模型 family/geometry 识别、独立引擎路径、混合 CPU/GPU 专家和 GGUF nextn MTP。
- NVFP4：新 pack、CPU AVX2/AVX512 专家、CUDA decode/prompt、混合投影格式、质量/带宽工具。需要对应模型，不是 IQ3_KT 原地升级。
- Unsloth K-quants/Q8：Q6_K 专家进入常规构建、K-quant MMQ 独立动态库、内存常驻策略和量化质量工具。
- 图片生成：Qwen-Image-2.1-Turbo、`/v1/images/generations`、可选 native DiT runner；与图片理解的 Vulkan helper 是不同功能。
- setup：重跑保留已安装模型大小、保留部分用户环境设置、MTP 共用目录保留最近使用的 draft subset 等。

## 合并预检与实施建议

执行 `git merge-tree --write-tree HEAD origin/main`，只生成 Git 对象，未改工作树和索引。
检测出 6 个冲突文件：

1. `CMakeLists.txt`：本地 KT 编译/测试源与上游新增后端、量化构建。
2. `include/strata/artifact/gguf_reader.hpp`：KT include 与上游 geometry/family include，应同时保留。
3. `src/core/mtp.cpp`：KT expert stride/分组计算与上游 MTP profiling stamps，应保留 KT 分支并补充 stamps。
4. `src/kernels/cpu/native_expert.cpp`：KT 与 NVFP4 include 都需保留；自动合并后的分派仍需检查。
5. `src/kernels/cuda/iq_kernels.cu`：KT 分派与上游量化 kernel 改造，需要逐段结合。
6. `src/prefill/prefill.cpp`：KT row decoding 与 NVFP4 tail 参数，应按格式分别处理，不能覆盖 KT down 路径。

`serve/server.py`、视觉 CMake/helper、native_dense、native_mmvq 等自动合并，
但自动合并只说明文本没有冲突，不证明运行正确。视觉独立环境 overrides 在合并结果中仍存在。

建议以 v0.1.44.1 为目标建独立集成分支/工作树，保留现有二进制、配置和 pack，
逐项解决上述冲突。完整集成的长期维护成本通常低于维护大量依赖交叉的 cherry-pick。
如果只做近期最小修复，则优先引入 API、图片缓存、调度/退出修复，连同依赖和测试移植。

验证顺序：

1. Windows CUDA 12.8、sm75/sm89 独立构建，检查原先 pinned ggml 依赖与新 strata_kq 等动态库要求。
2. 原有 KT CPU/GPU row、GEMV、grouped expert、prefill 和 MTP sidecar parity。
3. API reasoning effort、server/frontend、图片缓存、独立 Vulkan env/device 隔离测试。
4. 新旧二进制同配置比较 4K/32K/长 prompt、短增量续聊、文本/图片会话切换；关注缓存命中和首 token 延迟。
5. 分别 A/B COMMIT_BATCH、HC_Q8 等开关，每次只改一个因素；不要同时开启 MTP/pipeline/adaptive。
6. 大上下文及 512K pool 是不同容量指标，要验证会话换入换出、取消、pool 边界和内存占用，不能只用短 smoke 证明。

本次没有构建或运行新版本，因此“可合入”表示源码评估和候选优先级，不代表已通过本机性能或正确性验证。

## 后续执行：用户选定修复

用户随后授权合入长 prefill 让路、图片会话缓存、Responses 思考等级、退出竞争、
分卡显存诊断和启动错误处理。以 `27a80e97` 为起点，在原工作分支按以下顺序 cherry-pick，全部无冲突：

| 上游提交 | 本地提交 | 修复 |
|---|---|---|
| `ac21812c` | `7f2f04bd` | 长 prefill 让路 |
| `6339e861` | `ad485703` | stdin reader 生命周期 |
| `b5217c8b` | `8339be46` | 后续分卡显存余量与警告 |
| `3d5819df` | `1b04923c` | GPU runtime 启动重试 |
| `ee6d8bac` | `043f502d` | 图片状态不匹配诊断 |
| `cf1b0edf` | `fad7c440` | 图片内容 key + grid 缓存匹配 |
| `d981a510` | `a85be090` | 崩溃分类和 CPU ISA 诊断 |
| `9aa674ed` | `7f864537` | 区分 CPU 缺指令与引擎非法指令 |
| `5d08c753` | `9717a76b` | 各 API 的思考等级 |

长 prefill 改动是修复现有并发槽位调度：等待请求的 prompt 小于当前 prompt 一半时，
服务器每 0.5 秒的静默轮询也会请求 BYIELD，不再只等 PP 进度行。
引擎在可让路的下一 chunk 边界响应，长请求之后恢复；每个请求最多让路两次。
它不自动给原来的串行服务增加并发槽位。

保留本地 KT、Vulkan 编码器和 GPU wake 代码；未引入性能开关、其他模型或 setup 默认调整。
独立构建目录为 `build-upstream-fixes`，CUDA 12.8、sm75/sm89、原 pinned ggml。
没有替换运行预设或已部署二进制，没有重启服务。

验证记录：

- `python -B -m unittest serve.test_effort_paths serve.test_server serve.test_parallel serve.test_frontend tools.test_vision_vocab tools.test_serve_gpu_wake`：381 项，全部通过，3 项 Windows 上不适用的 POSIX signal 测试跳过。
- 重建并执行 `conversation_cache_test`、`conversation_split_failure_test`、`conversation_memory_test`、`conversation_file_test`、`session_save_reclaim_test`：5/5 通过。
- `build-upstream-fixes/Release/strata.exe` 独立 CUDA 构建成功（12.8，sm75/sm89），`--help` 启动检查通过。
- 启用 `STRATA_BUILD_CONVERSATION_TESTS` 后构建并执行 `conversation_validation_test`：1/1 通过，C++ 测试合计 6/6。首次构建命令尝试了未注册的测试目标，随后开启测试选项并完成验证；引擎自身编译成功。
- HIP 工具链未找到，SYCL 引擎文档要求 Linux/oneAPI，本机未做这两个后端的构建及运行验证。相关 SYCL 修复按上游原提交同步。
- 未运行真实模型、图片重新编码或长上下文集成测试，以上通过不能替代部署后的端到端验证。

## 第二批：状态提交与 opencode 前缀缓存

用户确认后，无冲突 cherry-pick 两个功能：

- `6f3e3810` → `3ac6a327`：批量 GDN/QSA 状态提交；沿用上游默认开启，`STRATA_COMMIT_BATCH=0` 可回到逐层提交。
- `d8a49433` → `3babe86f`：opencode `<env>` 移到首个 user 消息；`env_to_user` 默认关闭，未调整本机预设。

验证：

- `python -B -m unittest serve.test_env_to_user serve.test_frontend serve.test_effort_paths`：37/37 通过。
- 在原独立目录重建 `strata`、`gdn_parity`、`qsa_parity`，CUDA 12.8 sm75/sm89 构建通过。
- 分别在 CUDA_VISIBLE_DEVICES=0（2080 Ti）与 1（4080 SUPER）运行两个 parity 程序：4/4 通过，均为 0 failures。
- 每张卡的 `gdn_commit_layers` 对逐层提交为 bitwise（n_keep 1–8，5 层）；QSA `append_steps_layers` 在 none、linear 2、yarn 2 三种缩放下均与逐层提交 bit-identical。
- 日志：`build-upstream-fixes/python-second.log`、`build-second.log`、`gdn-gpu0.log`、`gdn-gpu1.log`、`qsa-gpu0.log`、`qsa-gpu1.log`。

这证明测试形状的新旧 kernel 提交结果一致，不代表已完成真实 KT 双卡服务端到端或性能 A/B。
HIP/SYCL 验证限制同上。未加载新模型、替换部署二进制或重启服务。
