# API 与工具调用解析补丁集成（2026-10-24）

本批从 `3853f0f4` 出发，合入目标为 `feat/ik-support-with-vulkan-vision`。
按用户要求，只引入 API 与工具解析修复：生产代码涉及 `serve/frontend.py`、
`serve/server.py` 和 `serve/responses.py`，并带入对应测试和说明。

## 上游来源与行为

| 上游提交 | 本地提交 | 修复 |
| --- | --- | --- |
| `70f42407` | `dd23f4bb` | Chat Completions、Anthropic 和 MCP 循环只返回请求声明的工具调用；未声明的调用保留为文本。允许已声明工具使用 `<=NAME>`、`<function:NAME>` 等变体标签。 |
| `8bf178fe` | `31edfb69` | 工具的非歧义别名统一回到声明名称并使用其参数类型；修复遗漏包装闭合标签、连续调用、字面量示例和截断调用的边界；Responses 中未完成的调用保持 incomplete。 |

两个补丁均通过 `cherry-pick -x` 保留来源，无合并冲突；第二个补丁也保留了
PR #3556 的共同作者。Responses 未声明工具时保留原有 Codex 压缩请求行为。
无法识别的调用会记录计数及少量工具名称，不记录完整参数。

`max_tokens=0/-1` 在本地基线已受支持，有既有回归覆盖，因此没有引入上游
先拒绝这两个值、随后恢复的来回改动。

## 保留已有的长参数扫描优化

直接应用第二个补丁后，新增的遗漏包装标签检查会在每次 feed 时执行
`function_end()`，重新扫描增长中的工具参数，绕过此前的增量扫描优化。
本地补充提交 `0cd40acf`：只在新增文本及跨块重叠区出现后续 `<tool_call>` 时
执行结构扫描；参数里的字面量标签及 recovery 模式继续走完整扫描。

扩展既有扫描回归，与无预检的参考解析器逐次比较 feed 和 finish 的输出事件，
覆盖跨块标签、遗漏包装闭合、连续调用、字面量示例及 recovery 开关；另检查
流式与非流式长参数期间不会重复执行结构扫描，且下一次调用会重置扫描位置。

本机合成解析微基准：Windows、Python 3.11.9，同一进程加载三版 frontend，
7 轮交替执行，每轮反转先后顺序，取毫秒中位数。正文为
`("normal text with UTF-8 cafe 咖啡\n" * 3000)`，包入 write 工具的字符串参数，
总计 105,078 UTF-8 字节，每次 feed 64 个字符，thinking 和 recovery 关闭。
每轮检查工具名称与最终参数一致。

| 解析方式 | 原基线 | 直接应用补丁 | 补充扫描预检后 |
| --- | ---: | ---: | ---: |
| 整体交付参数 | 4.797 ms | 34.863 ms | 5.858 ms |
| 流式交付参数 | 7.881 ms | 37.532 ms | 9.094 ms |

测量时同时有 mock 回归运行；这是合成解析开销，不包含模型或网络，
也不代表生产吞吐提升。脚本和输出位于集成工作树的
`logs/api-tools-parser-bench-20261024.py`、同名 `.txt`，不提交日志。

## 最终验证

```powershell
python -m unittest serve.test_model_card_sampling serve.test_web_sampling serve.test_server serve.test_effort_paths serve.test_frontend serve.test_parser_hold serve.test_parser_call_scan serve.test_parser_track serve.test_reasoning_tools serve.test_reasoning_rescue serve.test_tool_name_space serve.test_responses serve.test_structured serve.test_detok serve.test_native_tool_parser serve.test_tool_call_envelope serve.test_tool_call_unclosed serve.test_mcp serve.test_client_cache serve.test_keepawake serve.test_queue_keepalive serve.test_request_hardening
```

```text
Ran 564 tests in 162.319s
OK (skipped=8)
```

556 项通过、8 项跳过。覆盖三种 HTTP API、流式与非流式调用、MCP、
显式关闭思考、采样、输出预算、按 IP 缓存策略、请求校验和生命周期。
`git diff --check` 通过。

集成分支为 `integrate/upstream-api-tools-20261024`，工作树位于
`B:\llama.cpp\Strata-integrate-api-tools-20261024`；完整测试日志为
`logs/upstream-api-tools-tests-final-20261024.log`，不提交日志。
通过后使用 `git merge --ff-only integrate/upstream-api-tools-20261024` 合入目标分支。

本次更新源码，服务使用新代码需另按预设管理流程重启。没有修改 C++ 引擎、
模型预设、缓存参数、MTP/DFlash2、显卡配置或其他上游功能；原有未跟踪文件保留。
