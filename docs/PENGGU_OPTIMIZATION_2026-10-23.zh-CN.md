# 砰咕缓存与思考档位优化汇总（2026-10-23）

本次调整已部署到本机 Strata 和 `10.0.0.2` 的 penggu。目标是让群聊稳定历史可以跨轮复用，
避免一次性任务占用会话池，并减少日常调用不必要的长思考。本页记录最终状态；早期全局
user-tail 配置只作为对照实验条件保留，不应照抄上线。

## 最终配置

| 位置 | 已应用设置 | 用途 |
| --- | --- | --- |
| Strata 服务预设 | `cache_tail_user_ips: ["10.0.0.2"]` | 仅匹配实际 TCP 连接来源的检查点策略 |
| Strata 引擎参数 | `--conversation-cache-slots 8` | 原为4槽，减少多群和不同用途之间的争用 |
| Strata 引擎参数 | `--conversation-cache-mib 12288` | 会话池仍为12 GiB |
| Strata 引擎参数 | 移除全局 `--tail-role-token 846` | 未匹配来源沿用引擎默认策略 |
| penggu 的 qwen-iq3kt 模型 | `cachePolicy: "strata"` | 为一次性任务发送 Strata 的跳过停放扩展 |
| penggu 的 qwen-iq3kt 模型 | `reasoningEffort: "low"` | 插话判断与前情摘要使用 low |
| penggu 现有回复路径 | 主模型副本提升到 medium | 艾特回复、主动发言、插话第二步使用 medium |

未增加512-token硬上限。长期记忆仍使用原云端记忆模型。插话第二步只有原流程需要时才调用；
如果判断阶段的初稿被直接采用，初稿仍来自 low，不会额外增加一次 medium 调用。

实际运行配置可能包含 API key，不提交这些文件或配置备份。bot 模型对象中应合并的无密钥字段为：

```json
{
  "id": "qwen-iq3kt",
  "reasoningEffort": "low",
  "cachePolicy": "strata"
}
```

这是现有模型对象的字段片段，不能替代完整模型/供应商配置。

## IP 策略在哪里实现，是否进入二进制

这项功能同时涉及 Python 服务和 C++ 引擎，但具体 IP 不在二进制中。

| 层 | 实现文件 | 负责什么 | 改来源 IP 是否需重编译 |
| --- | --- | --- | --- |
| 本地 JSON 预设 | `cache_tail_user_ips` | 保存需要使用该策略的来源列表；当前为10.0.0.2 | 否，修改后按预设流程重启服务 |
| Python HTTP 服务 | `serve/client_cache.py`、`serve/server.py` | 读取真实 socket peer、匹配名单、剥离客户端伪造字段，生成内部请求参数 | 否 |
| 编译后的 C++ 引擎 | `src/program/generate.cpp` | 接收每请求 `tail_role=N`，据此选择检查点位置 | 此通用能力需要本次新编译的引擎 |

匹配请求的内部流程为：

```text
socket peer = 10.0.0.2
  -> Python 匹配 cache_tail_user_ips
  -> 当前 tokenizer 将 user 编为846
  -> 该条 GEN/GENI 带 tail_role=846
  -> C++ 在最后一条 user 消息之前保存检查点
```

该边界保留稳定群聊历史，把带当前时间的末尾触发通知排除在检查点外。请求仍完整执行，
不是删除通知或裁剪模型上下文。846由 tokenizer 得出，也没有作为固定模型 token 写进通用代码。

不匹配时不加覆盖参数；引擎每条请求重新取启动默认值，本次默认是-1。前一个 bot 请求的参数
不会继承给下一条 coding 请求。名单为空时功能关闭，旧引擎会在加载模型前被能力检查拒绝。

匹配不信任 X-Forwarded-For、X-Real-IP 或 JSON 里的来源声明。它区分主机，不识别应用：
10.0.0.2上的其他调用也匹配，代理场景匹配的是代理连接地址。当前不支持连续批处理引擎。
完整配置与接口约束见 [CLIENT_CACHE_POLICY.md](CLIENT_CACHE_POLICY.md)。

## 一次性任务与思考分工

penggu 对明确配置 `cachePolicy: "strata"` 的 Chat Completions 模型，在 context-recap、
memory-update及其重试、vlm-describe、emoji-describe请求中发送 `strata_checkpoint: false`。
它们仍能读取匹配的缓存，但不把本次输出停放进会话池。普通回复和 batch-judge保留停车行为。
其他供应商与未启用能力的模型不接收扩展，fallback按目标模型能力重新判断。

代价是一次性任务发生思考预算收尾续写时，可能需要重新读取输入；截图中2101输入、3307输出的
context-recap因此多读了约4.9秒。不过其主要耗时是约3090 tokens的思考及收尾。思考档位
测试后按用户要求将本地模型默认改为low，现有 `withReasoningEffort` 才能对回复路径明确传medium。
Strata全局默认没有因此改变。

## 验证结果与范围

- 固定五群序列中，6次重复请求输入复用率71.57%→99.20%，预填总耗时21.617→5.590秒。
  包含5次冷读的11请求总耗时61.279→46.779秒，11对简短回答相同。
  这是槽位与边界设置整体的合成测试，不能当成真实聊天平均收益。
- 真实来源交错测试中，10.0.0.2续轮复用约99.2%；本机Chat/Responses仍保留原检查点策略，
  即使伪造来源头或内部JSON字段也不会启用bot策略。7条合成回答一致。
- Python服务回归共372项：369通过、3跳过；bot缓存与连续对话测试共18项通过。
  bot类型检查仍有既存依赖声明、废弃模块等报错，本次llm.ts/types.ts无报错。
- 编译并部署CUDA版本；本机未构建或运行HIP/SYCL。二进制及CUDA依赖、哈希保留在Git忽略的
  `build-kt/penggu-cache-ip-20261023/`，旧生产二进制保留。
- 思考对比为9个合成群聊场景×3档，另6次补测。low/medium/xhigh平均耗时分别6.1/8.1/10.3秒。
  主要事实和推理题未见low特有错误，但语气有偏硬、格式有遗漏。512预算对照未触发上限，
  不足以证明复杂摘要强制截断仍可靠。
- bot配置参数检查覆盖batch-judge/context-recap=low，mention-reply/batch-compose/proactive=medium。
  14:23重启后PID15704，14:24确认登录成功、stderr为空。配置变更只有该模型reasoningEffort。

## 记录与维护

- [首轮缓存原因、测量与部署](PENGGU_CACHE_TUNING_2026-10-23.zh-CN.md)
- [按IP实现、真实来源验证及部署恢复](PENGGU_CACHE_IP_SCOPE_2026-10-23.zh-CN.md)
- [思考档位实验及用户确认后的应用](PENGGU_EFFORT_TEST_2026-10-23.zh-CN.md)
- 合成缓存、思考评测脚本：`tools/penggu_cache_bench.py`、`tools/penggu_effort_bench.py`。

关闭IP策略时，移除 `cache_tail_user_ips` 或置为空列表，并确认没有遗留全局tail-role开关，
按预设管理流程重启服务；不需要重编译。关闭bot跳过停放扩展时移除模型cachePolicy并重启bot。
改回思考档位时修改模型reasoningEffort并重启bot；SSH启动仍通过WMI调用start_bot.ps1脱离会话。

Git仅提交本次源码、测试、无密钥示例和报告。原有未提交文档、个人文件、含凭据的运行配置、
模型/二进制/原始结果与配置备份留在原处，不混入本次提交。
