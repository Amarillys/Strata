# 砰咕 bot 缓存优化与部署（2026-10-23）

本轮最终设置与维护方式见 [优化汇总](PENGGU_OPTIMIZATION_2026-10-23.zh-CN.md)。

后续更新：全局 user-tail 检查点已改为按客户端 IP 选择，当前配置与验证见
[IP 限定部署记录](PENGGU_CACHE_IP_SCOPE_2026-10-23.zh-CN.md)。下文保留首轮全局配置的测量条件。

已在本机 v4-lite 服务和 `Y:\project\penggu` 应用两处部署。bot 通过 SSH 在
`10.0.0.2` 的 `D:\project\penggu` 运行，与本机 Y: 共享目录是同一份代码。
本次可以正常访问共享目录及 SSH；此前只读分析中的访问限制已不再适用。

## 原因

- bot 真实请求日志中，优化前本轮 18 次成功调用涉及 5 个群：11 次 mention-reply、
  6 次 batch-judge、1 次 context-recap；输入 161,299 tokens，复用 53,328 tokens（33.06%）。
  这组自然流量只说明现场情况，不用它与下面的合成测试比较速度。
- mention-reply 在完整历史后附加带当前时间的触发通知；下一轮不保留上次通知。
  原引擎在最后的 assistant 头部之前保存检查点，因此该检查点包含上次通知，下一轮无法匹配，
  只能回到 system 检查点。历史中的 assistant 还包含展示用时间戳、清理后的文本，
  也不能直接续接引擎上一轮原始输出。
- batch-judge 附加同类通知，但只取最近 40 条聊天。窗口向前移动时，历史起点也变化；
  此时不能复用到完整历史末端。它与 mention-reply 的用途规则不同，system 也不同。
- 会话池原有 4 个停放槽位，多个群、不同用途和短探针会争用。12 GiB 预算尚有余量，
  不需要先扩内存预算。截图中的 VRAM HIT RATE 是专家权重命中率，提示词复用应看 REUSED。

## 已应用的改动

本机预设 `strata-iq3kt-v4-lite-k8v8-pf2048-vision-igpu.json`：

```text
--conversation-cache-slots 4 -> 8
新增 --tail-role-token 846
```

846 是此模型 tokenizer 的 `user` token。已检查当前引擎支持 `--tail-role-token`：
它把检查点放在最后一条 user 消息之前，恰好保留砰咕的稳定历史、排除动态通知。
这是已有引擎能力，无需重编译。请求文本、角色、群记忆、历史窗口及回复规则均未修改。

仍为 `--prompt-cache 5`、`--prompt-cache-every 0`、12 GiB 会话预算；不启用全局
`strata_prefix` pin 或周期快照。后者会增加额外快照成本，本轮明确的消息边界已解决完整历史
场景的重复读取。本配置按 user 角色选择边界，会作用于此服务的所有客户端；其他客户端仍
执行完整提示词，只是检查点位置改变。此 token ID 不可直接用于另一种 tokenizer。

运行时逐项核对引擎命令行，只有上述两项缓存设置改变。保持原生产二进制、307200 context、
INT8 KV、layer split 19、prefill 2048、关闭 MTP，以及实际运行的 840/540 MiB 显存预留。
专家槽位仍为 24,138。原运行适配器继续处理日志路径和显存预留。

bot 的 `plugin/positive-reply/types.ts` 增加可选 `cachePolicy: "strata"`；仅给本地
`qwen-iq3kt` 模型启用。`llm.ts` 为 context-recap、memory-update（含重试）、
vlm-describe、emoji-describe 的 Chat Completions 请求加 `strata_checkpoint: false`。
这些任务不续接输出，因此不应占用停放槽位。batch-judge、普通回复和主动发言继续保存缓存。
其他供应商、未启用该能力的模型以及 Responses 协议不接收这个扩展；fallback 按目标模型重新判断。
当前记忆模型使用云端供应商，不受这项扩展影响。

## 固定序列测试

脚本：`tools/penggu_cache_bench.py`。bot 暂停后，依次执行
`A1, A2, B1, C1, D1, E1, A3, B2, C2, D2, E2`。每次约 8K 输入，固定 system 与聊天历史，
后续轮次追加带时间戳的 assistant 和群友消息，并替换末尾通知。五个模拟群使用不同 system。
强制简短回答 OK、temperature=0、关闭思考。没有发送测试消息到 QQ 群。

同一生产二进制，旧配置一轮、新配置一轮；新配置前重启服务。每组各 11 次请求，
输入 token 数逐对一致，11 对回答均相同。以下是两个配置整体的测量，未单独拆分槽位数
与检查点位置的贡献，也不是对真实 QQ 回复质量的评测。

| 指标 | 原配置 | 新配置 |
| --- | ---: | ---: |
| 6 次重复请求输入复用率 | 71.57% | 99.20% |
| 6 次重复请求预填总耗时 | 21.617 s | 5.590 s |
| 6 次重复请求端到端总耗时 | 22.092 s | 6.060 s |
| 含 5 次冷请求的全组复用率 | 39.12% | 54.22% |
| 全组预填总耗时 | 60.384 s | 45.755 s |
| 全组端到端总耗时 | 61.279 s | 46.779 s |

重复请求预填减少 74.14%，全组端到端减少 23.66%。重复请求原来均复用 5,790 tokens，
现在复用 8,020 / 8,052 tokens。冷请求平均预填约 7.75 -> 8.03 s，单轮实验不能把
这部分差异归因于某一个设置；收益来自后续复用。没有声称解码速度因此提高。

新配置测试末尾停放 4 个条目，约 3.26 GiB，0 次淘汰、5 次恢复。五群测试未证明
8 个槽位在更高并发群数下足够；超过预算/槽位仍会淘汰。窗口滑动、system 改动、摘要更新、
表情清单轮换和首次访问新群仍可能回退到较早前缀或冷读，99.2% 不是自然流量承诺。

另做真实 HTTP 验证：聊天 A → 一次性摘要 → 聊天 A。停放计数为 11 → 12 → 12，
确认摘要没有被停放，A 恢复并复用 2,622 / 2,641 tokens，三个回答均为 OK。

## 检查与恢复

- 本机及 bot 主机均通过 `node --test test/strata-cache.test.cjs test/conversation-follow-up.test.cjs`，
  共 18 项。新增测试覆盖一次性任务、正常聊天、其他供应商、Responses 和 fallback。
- `npx --no-install tsc --noEmit` 仍被项目既有 node_modules 类型声明及
  `middleware/deprecated/` 等错误阻挡；本次修改的 llm.ts / types.ts 无报错。
- 部署使用预设管理器的 catalog 锁、空闲等待、卸载、启动和健康检查；部署失败自动恢复原配置与进程。
  `strata-active.json` 已更新。
- bot 11:23 暂停，11:27 经 WMI 脱离 SSH 会话启动，PID 6836，日志
  `data/logs/bot-20261023-112735.log` 确认 `login finished`，检查时 stderr 为 0 字节。
- 服务 wrapper PID 11444、engine PID 17200；`/health` 确认 loaded/images/API key 均启用，
  检查时 busy=false、queued=0，物理内存可用约 37.2 GiB。

本机测量与部署文件位于 `build-kt/penggu-cache-20261023/`：
`baseline.json`、`optimized.json`、`one-shot.json`、`deployment.json`、`final-service.json`。
这里只保存指标与合成回答哈希，不保存用户请求正文。该目录还包含部署前配置备份，
可能含密钥，不能提交或公开。两个仓库中原有的未提交改动均保留。
重启后的前 14 条服务请求是上述合成验证，监控累计数字包含它们；自然流量应从 bot
恢复时间之后单独统计。测试快照按正常缓存策略被后续群聊逐步替换。

回退时将本机预设的 conversation-cache-slots 改回 4、移除 tail-role-token 及其值，
按相同预设管理流程重启服务；如需关闭 bot 扩展，移除本地模型的 cachePolicy 并通过
`start_bot.ps1 -Action restart` 重启 bot（SSH 下仍经 WMI）。回退无需改模型、聊天记录或群记忆。
