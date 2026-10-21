# 小型服务层上游补丁集成（2026-10-21）

本批从服务分支 `ebd2ec5d` 出发，仅引入三个 Python 补丁。目标是修复显式关闭思考失效，
以及降低流式文本和长工具参数的解析开销。没有修改 C++、KT kernels、视觉配置、采样默认值或预设。

## 补丁范围

| 上游提交 | 集成提交 | 改动 |
| --- | --- | --- |
| `e69cbfa5` | `c8164671` | 请求显式 `reasoning_effort: false` 不再被配置或共享默认等级覆盖；仅缺省、None、空字符串补默认值。 |
| `b145a4a2` | `7908a397` | 流式标签的半截后缀匹配先查可能的起点，减少对普通文本的重复检查。 |
| `fc27296d` | `032206d8` | 工具调用闭合标签尚未出现时，只搜索新增文本和跨块重叠区，避免反复执行完整结构扫描。 |

三项均以 `cherry-pick -x` 保留上游来源，无合并冲突。解析改动保留既有事件语义；工具参数含有字面量
闭合标签时恢复完整结构扫描，recovery 模式也继续走原有结构扫描。

额外补充：

- 配置 high、共享 high、配置 low 加共享 high 三种默认来源，分别验证显式 False、字符串 none、
  None、空字符串和字段缺省，共 15 个 mock HTTP 场景，核对实际送给引擎的 prompt。
- 修正既有 `NoLeakedSampler` 测试的线程计数竞态。原测试比较全进程采样线程总数，其他测试的线程
  恰好结束时会误报。本次只跟踪当前服务新建的线程，并保证断言失败时仍关闭测试服务器；
  没有修改生产 telemetry 实现。

## 验证与本机测量

回归包含原来的服务层四个模块，以及 frontend、parser、工具调用恢复、Responses、structured output
和 detokenizer。上游解析测试覆盖标签跨块、字面量标签、多个调用、未完成调用及 recovery 分支，
并与原始解析策略逐次比较输出事件。

```powershell
python -m unittest serve.test_model_card_sampling serve.test_web_sampling serve.test_server serve.test_effort_paths serve.test_frontend serve.test_parser_hold serve.test_parser_call_scan serve.test_parser_track serve.test_reasoning_tools serve.test_reasoning_rescue serve.test_tool_name_space serve.test_responses serve.test_structured serve.test_detok
```

首轮 481 项在 139.364 秒内完成，只有上述线程计数测试失败（实际线程数 4，预期 5）；该用例单独运行通过。
修正测试隔离后，重新运行同一组用例，最终结果见本记录末尾。

同日 Python 解析微基准：Windows / Python 3.11.9，同一进程载入 `ebd2ec5d` 和集成后的 frontend，
每个场景交替执行前后版本各 7 次，记录耗时中位数。正文为
`("normal text with UTF-8 cafe 咖啡\n" * 3000)`，按 64 个字符分块调用 `OutputParser.feed()`，
最后调用 `finish()`。工具场景使用 `write` 的字符串参数 `text` 包裹该正文；thinking 和 recovery 关闭。
先验证全部输出事件的种类、文本、工具名称和参数相同，忽略随机生成的调用 ID。

| 合成输入 | UTF-8 字节数 | 原版本中位数 | 集成后中位数 |
| --- | ---: | ---: | ---: |
| 普通文本 | 105000 | 4.853 ms | 3.557 ms |
| 工具调用，整体交付参数 | 105078 | 31.622 ms | 4.017 ms |
| 工具调用，流式交付参数 | 105078 | 36.740 ms | 7.547 ms |

这是本机合成解析测量，期间另有 mock 回归测试运行；不包含模型推理或网络传输，不能当作生产吞吐、
首 token 延迟或所有工具负载的收益预测。没有加载真实模型或重启实际服务。

## Git 状态与合入方式

集成分支为 `integrate/upstream-serve-20261021`，工作树位于
`B:\llama.cpp\Strata-integrate-serve-20261021`。

初次集成时，主工作区已有文件的 Windows ACL 拒绝当前执行进程写入，包括 `.git/logs/HEAD`、服务分支 reflog
和 `serve/server.py`。尝试切换分支时因 reflog 写入失败而停止，随后改用独立工作树完成集成。
写权限检查仅打开文件、未写入内容；没有改 ACL、接管所有权或改全局 Git 配置。
当时主工作区保留 `ebd2ec5d` 及原有未跟踪评审记录和 `recycler/`，未合入此批补丁。

同日权限环境更新后，已在主工作区执行以下快进合并，服务源码分支
`feat/ik-support-with-vulkan-vision` 从 `ebd2ec5d` 前进到 `3631e17c`，没有冲突：

```powershell
git -C B:/llama.cpp/Strata -c safe.directory=B:/llama.cpp/Strata merge --ff-only integrate/upstream-serve-20261021
```

合入的源码和测试与下述 481 项回归的版本完全一致；此后的文档提交仅更新本节合入状态。
原有未跟踪评审记录和 `recycler/` 保留，没有推送远端。

这只更新源码。运行中的服务何时采用新代码，应另外按预设管理流程安排切换；不要启动集成工作树里的
临时服务来替代当前部署。

## 下一批建议

1. 独立引入 `f13a870d` 的长对话前缀分词，重点验证 token IDs 一致、特殊 token 字面量、图片、
   count_tokens 不污染缓存、上下文拒绝后的缓存清理和多会话交错；然后做本机长历史编码耗时对比。
2. 再评估 Windows 锁页预留和 C++ 缓存管理，保留 KT/双卡路径，并完成受影响后端构建及 GPU 回归。
3. Grammar 单独立项。大型 logit_bias 列表缓存、DSpark 和 ZOS 暂不混入这批小补丁。

## 最终回归结果

```text
Ran 481 tests in 140.159s
OK (skipped=8)
```

473 项通过、8 项跳过：3 项为 Windows 不适用的信号测试，5 项需要工作树中未配置的模型包 tokenizer。
最终日志保存在集成工作树的 `logs/upstream-serve-tests-final-20261021.log`，不提交日志。
`git diff --check` 通过。此次没有进行 C++ 构建或 GPU 性能验证。
