# 砰咕缓存策略按 IP 限定（2026-10-23）

本文记录IP策略上线时的状态，后续思考分档已应用，当前全貌见
[优化汇总](PENGGU_OPTIMIZATION_2026-10-23.zh-CN.md)。

用户要求：只给来自 `10.0.0.2` 的砰咕请求使用“最后一条 user 消息之前保存检查点”，
其他来源的 coding 请求恢复引擎原有策略。

## 实现

- 本机预设删除全局 `--tail-role-token 846`，新增
  `"cache_tail_user_ips": ["10.0.0.2"]`。
- `serve/client_cache.py` 按 HTTP socket peer 匹配；不使用 X-Forwarded-For、X-Real-IP
  或请求正文选择来源。支持 IPv4-mapped IPv6 规范化，不接受主机名或网段。
- 服务把匹配请求传为内部 `tail_role=846`。846 由当前 tokenizer 对 user 的编码得出，
  不在通用服务代码中硬编码。来自 HTTP JSON 的内部字段始终被剥离或覆盖。
- 引擎每条 GEN/GENI 都重新初始化请求参数，无参数时回到启动默认值 `-1`，
  所以前一个 bot 请求不会改变下一条 coding 请求的检查点选择。
- Chat Completions、Anthropic Messages、Responses 的流式和非流式入口均经过同一策略。
  这是按主机区分：10.0.0.2 上的其他应用也适用；反向代理情况下看到的是代理连接地址。
- 缓存仍为 8 个槽位、12 GiB，周期快照仍关闭；本轮不改 bot 请求、提示词、思考档位或预算。

详细配置说明见 [CLIENT_CACHE_POLICY.md](CLIENT_CACHE_POLICY.md)。该功能默认关闭，
旧引擎不支持时在加载模型前拒绝该配置，不会静默退化成无效设置；暂不支持连续批处理。

## 构建与回归

从当前源代码构建 CUDA 引擎（MSVC 14.42、CUDA 13.0.2、Release、sm75/sm89/sm120）。
部署二进制和哈希保存在 `build-kt/penggu-cache-ip-20261023/`；旧生产二进制保留。
本轮仅验证本机 CUDA 后端，未构建或运行 HIP/SYCL。

`python -m unittest serve.test_server serve.test_responses serve.test_client_cache`：
372 项，369 通过、3 跳过。新增测试覆盖 IP 精确匹配、映射地址、伪造来源头/内部字段、
默认请求保持原样、三种 HTTP API 与流式入口、请求间不继承策略、配置与引擎能力校验。

首次切换缺少新二进制所需 CUDA 13 DLL，Windows 返回 0xC0000135，预设管理器自动恢复原服务。
随后补齐 cudart64_13.dll、cublas64_13.dll、cublasLt64_13.dll，记录哈希，
并将实际运行 `strata.exe --version` 加入部署前检查后重新切换。

## 上线与真实来源验证

新服务已上线：wrapper PID 20684、engine PID 21736；health 确认模型、图片入口和
API key 均启用。引擎报告 `tail_role_token=-1`、`request_tail_role=1`，
缓存仍为 8 槽 / 12288 MiB，expert slots 24138，其他启动参数与切换前一致。
砰咕于 14:04:47 通过 WMI 脱离 SSH 启动，PID 10776；14:05:21 验证登录成功，stderr 为空。

暂停 bot 自动调用期间，用相同结构的约 8K-token 合成群聊进行 7 次顺序请求，
只要求回复 `OK`，关闭思考、temperature=0、最多输出 8 tokens。所有输出哈希相同。
每组首轮 reused=0，续轮结果如下；这些是缓存机制测试，不代表真实聊天质量或平均延迟。

| 实际来源 / API | 续轮 | Prompt | Reused | 复用率 | Prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| 本机 / Chat Completions | 2 | 8086 | 5791 | 71.618% | 3633.0 ms |
| 10.0.0.2 / Chat Completions | 2 | 8086 | 8021 | 99.196% | 853.4 ms |
| 本机 / Responses | 2 | 8087 | 5792 | 71.621% | 3574.4 ms |
| 10.0.0.2 / Chat Completions | 3 | 8118 | 8053 | 99.199% | 976.9 ms |

本机两组续轮都带伪造的来源头和内部 JSON 字段，仍使用原检查点策略。
执行顺序为本机 Chat → 远端 Chat → 本机 Responses → 远端 Chat 第 3 轮，
验证原生引擎每请求恢复默认策略，以及交错调用后远端缓存仍可复用。
原始指标、探针、部署和健康记录在 `build-kt/penggu-cache-ip-20261023/`；
远端探针结果在 bot 的 `test/results/strata-cache-ip-20261023/`，不包含请求正文或凭据。

## 截图中长请求与当前思考档位

13:50:51 开始、2,101 prompt / 3,307 output 的请求，在 bot 调用日志中对应
13:52:06 完成的 `context-recap`。它压缩老群聊成前情摘要，最多保存 400 字。
当前调用传 `max_tokens=4096`，服务的 answer-room 机制给出 3,072-token 思考预算；
日志确认预算耗尽后关闭思考，约 3,090 tokens 为思考及收尾、约 217 tokens 为正文。
第一段生成耗时 63.913 秒，总服务耗时约 77 秒。bot 端计时还包含排队。

它发送 `strata_checkpoint: false`，没有停放入会话池；但内部收尾续写因未保留 live state，
又读取了 5,191 tokens（4.888 秒）。此前的“不占缓存槽位”优化有这部分续写代价，
本例主要耗时仍是思考生成。

定位截图时本地模型未配置 thinking/reasoningEffort，Strata 预设与共享设置也未覆盖，
当时实际采用模型模板默认 `xhigh`。bot 的 withReasoningEffort 只调整原先配了 reasoningEffort
的模型，因此注释中的“回复使用 medium”当时没有应用到此本地模型。

建议供后续对比：普通聊天/插话 low，复杂回复 medium，前情摘要 low + 512-token
reasoning_budget_tokens。这里的档位在本地模板中是思考指令，硬性上限是独立的预算字段。
该建议尚未应用，也未做 bot 回复/摘要质量对比；当前任务只改变缓存策略的来源范围。

后续补测见 [思考档位小样本对比](PENGGU_EFFORT_TEST_2026-10-23.zh-CN.md)：
9 个合成群聊场景、三档共 27 次，加 6 次补测。low 的主要事实与推理结果未见特有退化，
但语气有偏硬、格式有遗漏；回复倾向 medium。512-token 摘要对照未触发上限，
不能据此证明复杂摘要强制截断仍可靠。用户随后确认应用low判断/摘要、medium回复，
配置及重启结果见实验记录的“用户确认后应用配置”一节；没有添加512-token硬上限。
