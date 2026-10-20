# 上游 v0.1.45 更新评估（2026-10-18）

本次已下载上游代码，并保存为本地快照分支 `upstream/main-20261018`，HEAD 为
`477ac13cbe8e582c20484871ce459d20d82df09d`，与新拉取的标签 `v0.1.45` 一致。
上游发布页日期为 10 月 17 日；不能把 HEAD 提交的作者日期当作发布日期。

- 当前工作分支仍是 `feat/ik-support-with-vulkan-vision`，HEAD `2f10e4f7`。
- 与上游共同祖先为 `296b59d2`，即已合入的 `v0.1.44.1`。
- `HEAD...upstream/main-20261018`：本地独有 18，上游独有 264；上游包含 246 个非合并提交、18 个合并提交。
- 相对 v0.1.44.1：267 个文件，新增 30,785 行、删除 1,289 行；提交数量不等于独立功能数量。
- 原有未提交修改为 `docs/IQ_KT_VISION.md`、`serve/server.py`、`serve/test_server.py`，本次均未改写。

原仓库已有的 `.git/FETCH_HEAD` 和 `origin/main` reflog 无写权限。对象下载后，普通 fetch
仍未能更新 `origin/main`，因此该远端跟踪引用仍是 `296b59d2`。本次使用显式 refspec、
`--no-write-fetch-head --no-tags --refmap=` 把远端 main 拉到上述新分支，并单独拉取 v0.1.45；
成功后用 `rev-parse` 和 `describe --tags --exact-match` 复核。旧同名标签有差异，未强制覆盖。
网络访问仅在本次命令中停用失效代理并改用 OpenSSL TLS 后端，未修改持久 Git 配置或关闭证书校验。

依据：[版本发布说明](https://github.com/Niko1221/Strata/releases/tag/v0.1.45)、
[版本比较](https://github.com/Niko1221/Strata/compare/v0.1.44.1...v0.1.45)，以及已下载的提交和代码。
下面的性能数字来自上游记录，本机未运行性能测试。

本次只读取了活动预设和评估所需的配置字段。`strata-active.json` 指向 `v4-lite`，配置文件名是
`strata-iq3kt-v4-lite-k8v8-pf2048-vision-igpu.json`。其参数包含 layer split 19、prefill 2048、
int8 KV、max-context 307200、kv-resident 20480、会话缓存 12288 MiB / 4 slots；adaptive 关闭，
PCIe share 显式为 0.5，没有 `--mtp`。视觉 helper 路径为本地 Vulkan 构建，
但 `vision.device` 实际写的是 `NVIDIA GeForce RTX 2080 Ti`，图片上限 768 tokens；
不能因文件名带 igpu 就认定配置使用 Intel。活动预设的引擎为 `build-kt/prod-2f10e4f7/strata.exe`，
开启 `STRATA_PREFILL_KT_MMQ=1`，`--prompt-cache 5 --prompt-cache-every 0`。
按 `docs/IQ_KT_VISION.md` 的 10 月 15 日部署记录，这次切换已移除 `--conversation-pool-tokens 524288`；
当前分支也没有该会话池实现。以上为配置和部署记录，未核验正在运行的进程。

对本机比较值得关注的变化：

| 改动 | 提交 | 用途及限制 |
|---|---|---|
| 长对话只对共享前缀之后的文本重新分词 | `f13a870d` | 缓存此前 prompt 的 token 和可恢复边界，减少长历史每轮重复分词；属于服务器 CPU 开销优化，不能解释为模型 prefill 吞吐提升。 |
| checkpoint 从末尾向前分散保留 | `a7d4b754` | 新增默认关闭的 `--prompt-cache-spread`，避免 LRU 把恢复点都聚集在最近几轮。适合 agent 改写旧 tool 输出或较早历史，双卡路径也使用该策略；只改变已有 checkpoint 的淘汰，不会补建缺失的恢复点。当前 `--prompt-cache-every 0` 关闭定期间隔 checkpoint，需要结合实际历史评估。 |
| 新会话借用已停放会话的中间 checkpoint | `c9e571e0`、`2b296987` | agent 分支可借用共同前缀并保留原会话；但 `generate.cpp` 明确用 `stages.empty()` 限定单 GPU，当前双卡 layer split 不会借用，仍恢复整份分卡快照。到达 pin=N 的前缀仍走 take，保护原有 pin 语义。 |
| 并发续聊优先选持有本会话的 slot | `dd08a7d0` | 防止两个会话同时续聊时相互覆盖缓存，导致全量重读；主要作用于开启 parallel 的服务。 |
| 会话 K/V 预留内存收紧 | `53f72571` | 再次停放时每个 K/V buffer 最多保留其大小的八分之一，减少原来按 buffer 保留最多 16 MiB 的额外占用。 |
| 减少连续短回合重写完整快照 | `d9cb22d2` | 新增 `--conversation-cache-park-gain N`；默认 0，保持每回合停放。开启后以恢复时最多补读 N tokens 换取更少快照复制，需要按真实会话 A/B。 |
| tool call 流式解析、logit_bias 预处理 | `fc27296d`、`b145a4a2`、`b6dac807` | 减少重复扫描和重复整理参数的 CPU 时间。 |
| 双卡 auto split 修复 | `3b848837`、`b8844516` | 避免异构双卡搜索给其中一张卡分配极少层。当前手动 split 19 不受该自动规则影响。 |
| 第二 GPU 专家缓存的 P2P 探测 | `33150724` | 启动时双向精确复制检查通过才使用 peer access；这是 peer-expert tier 的检查，不能泛化为 layer split 全部路径的修复。 |

NVIDIA 性能改动需要按格式和设备判断：

| 项目 | 上游记录 | 本机判断 |
|---|---|---|
| `STRATA_SM75_PERF=1` | Turing 可选系列，作者 RTX 2080 Ti 报告 decode +5–6%、部分深上下文 prompt 最多 +30%；维护者没有 sm75 实卡复测。 | 对 2080 Ti 值得单独测试，4080 SUPER 不受这个总开关影响；其中 QSA、HC 等共享路径可能有用，专家格式相关的 kernel 不能直接套用 IQ3_KT。 |
| `STRATA_QSA_ROWS=1`、`STRATA_ATTN_DAILY=1` | 新的 QSA scorer / decode attention，默认关闭；跨机器收益有正有负。 | 不依赖 KT 专家行解码，适合作为后续单项 A/B 候选。 |
| `STRATA_EXPDEC=1` | 标准 IQ gate/up 的共享内存 kernel，默认关闭，文档包含 2080 Ti kernel 测量。 | 上游 `expdec_gu_ok` 只接受类型 16/17/18/21/22；本地 KT 在 grouped expert 入口提前分派到 `kt_expert_grouped`。该优化不能当作 KT 专家提速。 |
| `STRATA_DEQUANT_WIDE` | 默认覆盖扩展到 sm86，Q4_K/Q5_K/Q6_K/IQ4_XS dense 权重反量化。 | 本机是 sm75/sm89，不应将 sm86 默认调整的测量收益算到本机。 |
| `STRATA_PCIE_BALANCE=1` | 文档限定 CUDA、单 GPU、无 batch、无显式 `--pcie-frac`；A4000 +5.6%，3060/P100 略慢。 | 当前双卡且固定 PCIe share 0.5，不适用。 |
| resident RAM 预算保留 prompt 借用区 | `f2e142f4`：预算还有空间时，借出缓存槽对应的专家也留 RAM，减少回读文件。 | 仅在相应 resident-budget 路径有意义；不能只合并就预期当前配置加速。 |

服务器默认行为也有变化，升级不能只关注 kernel 的答案一致性：

- `a8091f4d`、`e8634865`：没有显式采样参数时，按模型卡补默认值。Flash-Next thinking 为
  temperature 1.0 / top_p 0.95 / top_k 20；非 thinking 为 0.7 / 0.8 / 20、presence_penalty 1.5。
  配置、共享设置和请求字段可覆盖；显式 temperature 0 仍然是 greedy。
- `d0e990ba`、`194c2688`：开启 thinking、没有显式思考预算且输出 cap 至少 1024 时，
  自动保留回答空间。例如 cap 8192 对应思考预算 6144，给回答留 2048；`answer_room: false`
  可关闭。Anthropic 的 `thinking.budget_tokens` 也作为预算处理。
- `e69cbfa5`：显式 `reasoning_effort: false` 可以覆盖已配置的默认等级。
- `a091cdf6`：Prometheus 吞吐指标按两次抓取之间的窗口取平均，空闲时归零。

其他主要更新：

- Qwen3.6-35B-A3B 增加图片、最多 8 路并发和 batch MTP；更大默认 prompt chunk 及新 kernel。
  上游 R9700 + UD-Q4_K_XL：decode 83.2 → 117.8 tok/s，4K prompt 3453 → 6220 tok/s；
  RTX 3060 混合模式仅约 +2% / +3%。这是 Qwen3.6 路径，不能套到当前 Flash-Next IQ3_KT。
- 多用户服务增加 layer split 下每组最多 8 rows、总 slots 可超过 8 的调度，以及复制、checkpoint、
  admission、MMQ pointer table 等可选优化。上游 4×R9700、IQ3_S 全专家驻显存、16 用户测到
  208 → 324 tok/s、首 token 中位数 20.6 → 2.0 秒；当前双卡配置需独立验证。
  `STRATA_SLOT_CKPT_MOVE=1` 在上游缓存复用测试中失败，上游明确不建议开启。
- pipelined batch 可选在独立线程读新 prompt（`STRATA_PIPED_CONCURRENT`），或分段让 decoding 继续
  （`STRATA_PIPED_READ_PIECE`）；不是默认开启，也不是单请求加速保证。
- setup 对不足 24 GB 的 NVIDIA 卡，新增图片配置优先使用 CPU encoder；已有配置不自动改动。
  修复 UD-Q5_K_XL metadata-only shard 检查、Windows AMD 子进程 PATH 和旧 CMakeCache 保留关闭选项。
- GGUF 全零 `attention.compress_ratios` 作为 converter 占位处理，修复部分 Uncensored 文件被拒绝。
- Mac 的小内存 prompt 分配、M1 Max kernel、超长 no-copy buffer 和监控更新；Intel Arc Q6_K、
  AMD 长上下文 top-k/scorer、RDNA2/3/4 及部分 APU 增加优化或可选支持。

已执行 `git merge-tree --write-tree --name-only HEAD upstream/main-20261018` 做源码合并预检，
只生成 Git 对象，未改索引或工作树。发现 1 个冲突文件：`src/kernels/cuda/iq_kernels.cu`，
冲突处是本地 `kt_kernels.hpp` 与上游新增 `sm75_series.hpp` include，应同时保留。
其余文本自动合并不代表 KT、双卡、视觉、缓存运行正确；该预检也不包含尚未提交的 3 个文件修改。

本地已有 live/checkpoint 精确前缀恢复、系统根固定、`strata_prefix` 固定文档前缀、图片内容 key、
双卡会话停放，以及恢复后复用未改 K/V、增量追加快照；这些不是 v0.1.45 新增。
本机优先评估 prompt 前缀分词、spread retention、buffer 预留收紧，再按实际切换负载评估 park-gain；
单 GPU 的借用机制与并发 slot 修复不应计入当前双卡串行配置的收益。原始分词器已有 20 万条 piece
缓存，因此新前缀分词进一步减少的是全 prompt 扫描、切分与拼装等工作，不能把旧路径说成每次完全重做 BPE。

后续集成应使用独立分支/工作树，保留 KT MMQ、视觉权重 host 支持和现有未提交修改；先验证 API、
缓存及 KT parity，再按单开关比较 Turing/QSA 优化。当前仅完成下载、源码审阅和合并预检，
未合并到服务分支、构建、加载模型、调整运行预设或重启服务。
