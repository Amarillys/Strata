# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).

## 本机工作约定

当前工作区是 `B:\llama.cpp`，实际项目根目录是 `B:\llama.cpp\Strata`。以下参数是
**2026-10-09 的部署快照**，用于接手维护；后续以用户要求、实际配置和运行状态为准。
先检查 `git status --short`，保留已有的源码、文档和 GPU 唤醒模块修改，不为整理工作区而重置它们。

### 入口与证据

| 位置 | 用途 |
| --- | --- |
| [src/program/generate.cpp](src/program/generate.cpp) | 引擎参数、请求协议、会话切换和运行指标 |
| [include/strata/program/conversation_pool.hpp](include/strata/program/conversation_pool.hpp) | 会话名额、总 token 预算、增长和淘汰策略 |
| [include/strata/core/conversation_cache.hpp](include/strata/core/conversation_cache.hpp) | 完整快照、按会话归属匹配和缓存管理 |
| [serve/server.py](serve/server.py)、[serve/conversation_pool.py](serve/conversation_pool.py) | API 路由、会话身份提取和指标 |
| [tools/switch_preset.py](tools/switch_preset.py)、[tools/serve_gpu_wake.py](tools/serve_gpu_wake.py) | 本机预设管理与 GPU 唤醒启动器 |
| [会话池说明](docs/CONVERSATION_POOL.zh-CN.md)、[工作记录](docs/CACHE_WORK_LOG_2026-10-09.zh-CN.md) | 当前功能、历史原因、实测和验证边界 |

生产预设目录由 `strata-models.json` 指定，活动记录为 `strata-active.json`。可用
`python tools/switch_preset.py status` 查看管理器识别到的服务，再核对 `/health`、`/metrics`、
引擎 `INFO` 和实际进程的 exe/参数。记录里的 PID、磁盘上的配置、二进制版本号都不能单独证明
正在运行哪一版代码；本地构建应同时记录二进制哈希。

当前 v4 配置为
`strata-iq3kt-v4-dual22-k8v4-pf2048-vision-igpu.json`，exe 为
`build-kt/pool-candidate/strata.exe`。旧 `build-kt/Release/strata.exe` 留作回退。
文件名中的 `k8v4` 是历史命名，实际参数为 `--kv int8`，即 K8V8。

2026-10-09 增加 `v4-lite` 独立预设：v4 稠密权重（含 387 个高精度投影）＋v2 IQ3_KT
专家＋v2 IQ4_KT PLE 表。配置为 `strata-iq3kt-v4-lite-k8v8-pf2048-vision-igpu.json`，
19＋29 分层、显存预留 960/512 MiB，沿用上述候选 exe 和下述缓存配置。原 v2/v4 预设保留，
实际活动选择以管理器、健康检查和运行进程为准。该 pack 直接引用原 v2/v4 GGUF，不能删除源分片。
来源、实测与质量边界见 [v4-lite 记录](docs/IQ_KT_V4_LITE_2026-10-09.zh-CN.md)。

### 缓存配置的含义

- 单会话上限 `--max-context 307200`；RoPE 为 `yarn`、scale `1.171875`。
  这扩展了原生 262144 上限，尚未验证完整 300K 长度的回答质量。
- 总池 `--conversation-pool-tokens 524288`，配合 `--conversation-cache-slots 4`：
  三个有 ID 通道和一个共享无 ID 通道，包含活动会话。推理仍为串行，池模式不与 `--batch` 同开。
- 快照预算 `--conversation-cache-mib 12288`，按需使用 RAM；物理可用内存门槛默认 2560 MiB。
  token 额度、快照字节额度和实际可用 RAM 都可能限制保留量，四个名额不保证四份满长上下文。
- `--prompt-cache 2 --prompt-cache-every 0`：池模式强制关闭周期检查点，至多保留根和最近轮次
  两份内部恢复状态。检查点数量不代表可容纳的 token 数，也不代表完整会话数量。
- 完整 KV 及循环状态通过 host 快照停放；`--kv-resident 20480` 是 GPU 常驻工作集。
  512K 是逻辑总量，不是单请求长度，也不是新实现的 GPU 分页分配器。
- 新会话首次请求、已淘汰会话、改写早期输入、匿名通道相互替换时仍可能 prefill。
  内存池不跨进程保留；当前池模式不支持 SAVE/RESTORE。

路由必须使用稳定的会话身份，例如 `session_id + thread_id`。每次变化的 `request_id`、
普通用户 ID、`prompt_cache_key` 不作为会话身份。支持的请求字段和头部见会话池说明。
同一客户端的主会话和子任务可能有不同的系统提示、工具或模板；ID 只选通道，复用还需要
实际 token、图片和控制状态匹配。Qwen 的 GDN 循环状态不能仅用最终 KV 回退到任意公共前缀。

日志只保留身份来源和散列；不要输出原始 ID、API key 或请求正文。历史请求没有记录元数据，
不能据此认定真实客户端传过 ID。用新请求的路由日志和 `/metrics.conversation_pool` 验证。
本地 Python API 调用沿用 `urllib.request.ProxyHandler({})`，凭据从配置读取，不写入命令行。

### 构建与服务变更

文档和只读排查不触发模型切换或服务重启。需要更新运行版本时，先构建可验证的候选，
再按照已有预设管理流程部署，保留失败时的回退路径。

- Windows 下不要覆盖正在运行的 exe。检查 CMake 缓存中的输出目录；本机 `build-kt` 的
  `CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE` 已指向 `build-kt/pool-candidate`，该候选目前正在使用，
  下次构建应选另一个输出目录。改完源码不等于运行服务已更新。
- 复用 `tools/switch_preset.py` 的 `Presets`：加锁、识别所属进程、检查端口和文件、等待请求
  完成后 drain/unload、启动、记录活动状态、验证健康。保留原启动器 argv/cwd 和 GPU 唤醒参数，
  不按进程名批量结束其他服务。
- 同一个 v4 已运行时，`switch("v4")` 不会重启。需要真正重启以加载新配置时，应明确执行上述
  完整流程；核对实际 exe、参数和 `INFO`，失败时还原配置并用原 argv/cwd 恢复。
- 本机访问入口通常为 `http://127.0.0.1:8080`；监听地址、端口和 API key 沿用现有部署。
  配置备份可存本机 `logs/`，其中可能含密钥，不提交或整份输出。

### 验证与记录

缓存或路由代码变更可在项目根目录执行以下相关检查；纯文档变更只需核对事实、链接和 diff。

```powershell
cmake -S . -B build-conversation-pool-tests -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=ON
cmake --build build-conversation-pool-tests --config Release --target conversation_cache_test conversation_split_failure_test conversation_memory_test conversation_file_test
ctest --test-dir build-conversation-pool-tests -C Release -R '^conversation_' --output-on-failure
python -m unittest serve.test_conversation_pool
python -m unittest serve.test_responses
```

涉及其他 API 或后端时补跑相应检查。GPU 回归应使用固定的系统提示和采样参数，覆盖
长 A → 短 B → 返回 A、匿名替换、第四个有 ID 会话进入、淘汰后返回；检查 cached/read token、
实际淘汰对象和回复。耗时必须说明是纯 prefill 还是包含请求、生成及指标读取。

原始证据放 `logs/<任务>-<日期>-<唯一后缀>/`，可公开且去掉敏感信息的结论放 `docs/`。
记录模型、硬件、参数、二进制哈希和未测范围；不把测试探针当作真实客户端验证。
截至上述日期，CUDA 构建及核心会话测试通过；一次 71 项混合测试有一项
`NoLeakedSampler` 线程计数失败，原因未定。完整 512K GPU 压力测试及 HIP/SYCL 构建仍未验证，
详见工作记录。后续修改应更新验证状态，不把未验证事项记为通过。
