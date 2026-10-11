# 按会话 ID 保留上下文（可选）

`--conversation-pool-tokens 524288` 启用总量 512K token 的会话池。与
`--conversation-cache-slots 4` 配合时，最多保留三个有 ID 的会话和一个无 ID 通道，
正在执行的会话也计入这四个通道和总 token 额度。无 ID 的请求共用一个通道。
这是上下文缓存数量，推理仍为串行；此模式不与 `--batch` 同时开启。

示例引擎参数：

```text
--max-context 307200
--conversation-cache-mib 12288
--conversation-cache-slots 4
--conversation-pool-tokens 524288
--prompt-cache 2
--prompt-cache-every 0
```

池按真实提示和生成的 token 增长，不预留每条请求的 `max_output_tokens`，也不会预先
占满 12 GiB。`conversation-cache-mib` 是包含 KV、循环状态、检查点和切换中快照的
另一条内存上限；实际物理内存不足时仍可提前淘汰或放弃保存。单会话仍受
`max-context` 限制，池的总量必须至少等于它。最大上下文的 RoPE 设置另行决定。

每个通道保存最新完整 KV 和运行状态。模式启用后，引擎强制关闭周期检查点，最多保留
系统提示边界和最近轮次边界两份恢复状态，以适应客户端重新渲染答案的情况。
这两份状态不限制上下文长度；该模型的循环层不能仅靠 KV 还原任意历史位置。

第四个新的有 ID 会话替换最久未使用的有 ID 会话，无 ID 通道的名额单独保留。
合计超过 512K 时按最近使用时间淘汰其他通道的整份上下文。无 ID 通道也受总容量限制。
同一个 ID 改写早期输入时，其旧分支被替换。重新访问已经淘汰的内容会重新 prefill。
重启会清空此内存池。

Python API 服务会把突然缩短的小请求转到共享无 ID 通道（混池），以免客户端用主会话 ID
发送辅助请求时覆盖长上下文。判定同时满足以下三项：

- 该 ID 上次留在有 ID 通道的输入至少为 3000 token。
- 本次完整输入严格小于 3000 token。
- 本次输入不超过上次输入的四分之一。

例如 `90589 → 502 → 91535`：中间的 502-token 请求走无 ID 通道，前后长请求仍走
原有 ID 通道。比较使用模板渲染后的输入 token 数，不含生成 token；同一次 API 请求的
思考续写沿用最初的输入长度。转走的小请求不更新该 ID 的比较基线，连续小请求也按此处理。
首次出现的 ID、普通短对话，以及不满足上述条件的请求沿用原路由。长度基线按最近使用
保留，数量不超过有 ID 名额，随引擎重启清空。日志用 `reason=short_input` 标明转流原因，
仅附输入长度和会话散列。主动缩短历史若满足同样条件，也会进入无 ID 通道。

## 会话标识

支持请求体的 `strata_conversation_id`、`conversation_id`、`session_id` / `thread_id`，
以及 `metadata`、`client_metadata` 中的对应字段。Codex 的
`client_metadata["x-codex-turn-metadata"]` 按 `session_id + thread_id` 区分主会话和子任务。
支持 `X-Conversation-ID`、`X-Session-ID` / `X-Thread-ID`、`X-Codex-Turn-Metadata` 请求头。
Claude Code 的 `metadata.user_id` 内嵌 session ID 也可识别。

`request_id`、普通用户 ID 和 `prompt_cache_key` 不作为会话 ID。请求 ID 通常每次不同，
缓存 key 可能由主会话和子任务共享。缺少有效会话标识时进入无 ID 通道。

ID 仅用于选择通道；复用仍要求 token、图片和控制向量设置完全匹配。不同通道不共享
可变的运行状态。新会话首次请求仍需处理自己的提示，扩大池不能从长会话的末尾恢复
任意早期公共前缀。此版本不把 ID 写入磁盘会话文件，池模式不支持 SAVE/RESTORE。

日志只记录 ID 来源和散列，不记录原始会话 ID 或提示内容。`GET /metrics` 的
`conversation_pool` 字段给出总额度、已用 token、通道散列、各通道 token 和淘汰次数。
未启用该选项时保留原有缓存匹配与保留策略。

## 验证

CPU 缓存策略测试包含 3+1 分配、生成过程超过 512K、按 ID 隔离、相同 ID 的 token
不匹配、替换与淘汰。`python -m unittest serve.test_conversation_pool` 验证元数据识别，
以及 Chat Completions、Responses 和 Messages 三条 HTTP 路径的传递。
GPU 验证结果另存于本机 `logs/` 下的对应诊断目录；未完成的后端验证不能视为通过。

2026-10-09，Windows / CUDA，RTX 2080 Ti 22 GiB + RTX 4080 SUPER 32 GiB，
`qwen3.8-flash-next-v4-iq3kt`、INT8 KV、layer split 22、prefill 2048，实测如下。
长会话与子任务使用相同的系统提示、不同的 `session_id + thread_id` 组合。

| 请求 | 输入 token | 复用 token | 补读 token | 脚本耗时 |
| --- | ---: | ---: | ---: | ---: |
| A 首次长请求 | 65,026 | 0 | 65,026 | 71.13 s |
| B 首次子任务 | 9,677 | 0 | 9,677 | 14.62 s |
| 返回 A | 65,048 | 65,028 | 20 | 0.91 s |
| 替换无 ID 请求后返回 A | 65,073 | 65,051 | 22 | 0.74 s |
| 第四个有 ID 会话进入后返回 A | 65,101 | 65,078 | 23 | 0.88 s |

耗时含一次 `/metrics` 读取。第四个有 ID 会话淘汰的是最久未使用的 B，随后访问 B
确实从头读取；当时最近访问的 A 和无 ID 通道都保留。两条短贪心请求的回复与修改前一致。
本次没有把 GPU 实测填满到 512K，512K 边界和生成过程增长由 CPU 策略测试覆盖。
本机没有 HIP / SYCL 编译器，未验证这两个后端的构建或运行。

### 2026-10-10：同 ID 小输入转混池

这次只修改 Python 路由，沿用 v4-lite 预设、INT8 KV、19＋29 分层、prefill 2048 和
512K 池。通过预设管理器等待请求完成后重启，保留原启动参数和 GPU 唤醒配置。
实际运行的 `build-kt/pool-candidate/strata.exe` 未改，SHA-256 为
`2947bb7c8f50e86cd1ae1b36aedb63527d3802c59800f8b90fc5ac72ef8c52c6`。

99 项 Python 测试通过：会话路由 20 项、Responses 40 项、请求统计／思考恢复／重启等待／
取消请求 39 项；现有 4 项核心会话测试也通过。覆盖 3000-token 严格边界、四分之一边界、
连续小请求、不同 ID、普通短对话、最近输入基线、重启清空，以及内部续写跨过 3000 token。
Chat Completions、Responses、Messages 均验证客户端不能伪造比较用的输入长度。

Windows / CUDA，RTX 2080 Ti 22 GiB＋RTX 4080 SUPER 约 32 GiB，合成实测如下；
短请求携带与 A 相同的请求头会话 ID。

| 请求 | 输入 token | 复用 token | 补读 token | prefill |
| --- | ---: | ---: | ---: | ---: |
| A 首次长请求 | 16,246 | 0 | 16,246 | 18.362 s |
| 同 ID 第一次小请求，转混池 | 496 | 0 | 496 | 2.795 s |
| 同 ID 第二次小请求，仍转混池 | 586 | 0 | 586 | 2.971 s |
| 无 ID 请求替换混池内容 | 47 | 0 | 47 | 0.630 s |
| 返回 A | 16,267 | 16,247 | 20 | 0.464 s |

两条小请求的日志均记录 `reason=short_input`；三次混池请求期间，A 通道保留的
16,247 token 未变。表中时间为 API 报告的引擎 `prompt_ms`，不含排队和生成。
测试期间有真实客户端请求穿插，因此没有额外创建有 ID 通道来重跑第四个 ID 淘汰测试；
该策略由既有核心测试覆盖，本次未改 C++ 淘汰逻辑。未做完整 512K 压力测试或 GPU 图像回归。
这些请求是合成探针，不能据此判断真实客户端某条小请求的用途。

本机原始证据、源码回退副本和部署校验在
`logs/short-input-pool-20261010-4ht_q4pi/`，其中 `gpu-results.json` 保存测量，
`gpu-events.log` 保存去掉请求内容的路由及引擎事件。
