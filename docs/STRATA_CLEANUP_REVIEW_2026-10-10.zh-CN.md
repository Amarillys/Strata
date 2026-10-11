# Strata 目录清理建议（2026-10-10）

本轮只统计和核对依赖，没有删除文件。范围为 `B:\llama.cpp\Strata`，不包含旁边的独立仓库
或 `B:\models`。统计时共约 **24.12 GiB、25,926 个文件**；按逻辑文件字节计，不追随重解析点。
本轮没有发现硬链接路径，大文件也核对了实际文件 ID 和链接数。后续新日志可能使数字略有变化。

当前管理器识别到的活动预设是 **v4-lite**，8080 的 health 为 `ok / loaded=true`。
实际运行的引擎为 `build-kt/pool-candidate/strata.exe`，视觉引擎为
`build-vision-vulkan/bin/Release/strata-vision.exe`。此外还核对了 v2、v4 回退预设的路径。

## 优先清理：约 12.71 GiB

以下都是验证副本或发布包的展开副本，未被当前服务及三份预设引用。
新的 4070 Ti SUPER ZIP 已完成解压、全部文件校验、依赖检查和独立模型 pack 生成，相关验证副本可清理。

| 相对 Strata 根目录的路径 | 大小约 | 内容与保留条件 |
| --- | ---: | --- |
| `build-release-validation/` | 4.18 GiB | 旧发布的 Extracted Package、With Spaces、Final Smoke 等验包副本。正式发布 ZIP 保存在 dist/release。 |
| `logs/v4-lite-share-20261009-01/applied-pack-iq3/` 和 `applied-pack-iq4/` | 2.77 GiB | 专家补丁应用测试的两份 dense.bin；建议只清理这两个子目录，保留旁边的小日志和验证记录。 |
| `logs/rt4070 extracted 20261010/` | 2.11 GiB | 本次 4070 运行包的解压验证副本及新生成的 pack，未启动生产模型。 |
| `logs/v4-lite-gguf-20261009-01/portable-pack/` | 1.39 GiB | 独立 GGUF 导出后的 pack 验证副本；公开验证结果在 bench/results。 |
| `dist/candidate/` | 0.77 GiB | 旧候选运行包的展开目录；已有正式发布 ZIP。 |
| `dist/release/strata-0.1.41-iqkt.1-windows-x64-cuda13/` | 0.76 GiB | 旧正式运行包的展开副本，保留同目录的 `.zip` 和 `.zip.sha256`。 |
| `dist/portable-v4-lite-20261010/Strata-v4-lite-4070TiS-Windows-20261010/` | 0.72 GiB | 本次运行包的 staging 目录，保留同目录的 `.zip` 和 `.zip.sha256`。 |

这里的 dist 建议只指表中展开目录，**不是整个 dist/release 或 dist/portable-v4-lite-20261010**。
后者还放着要传给另一台电脑的最终 ZIP。

## 可重新生成的构建缓存：约 3.01 GiB

这四项不被当前生产预设使用。清理不会删除源码，但下次运行对应构建或测试前，需要重新配置、
编译或下载依赖。保留这些缓存可以省下后续重建时间。

| 路径 | 大小约 | 清理后的影响 |
| --- | ---: | --- |
| `build-release-tools/` | 1.23 GiB | CUDA 13 的打包工具和下载缓存；下次从源码制作发布引擎需重新准备。当前生产使用系统 CUDA 12.8。 |
| `build-release-cuda13/` | 0.69 GiB | CUDA 13 便携引擎的编译产物；正式引擎已在发布 ZIP 中。 |
| `build-release-vision/` | 0.48 GiB | 便携视觉引擎的编译产物。当前核显使用的是另一个 build-vision-vulkan 目录。 |
| `build-conversation-pool-tests/` | 0.61 GiB | 会话池测试的 CMake、依赖和可执行文件；源码测试位于 src/、serve/ 等目录，保留即可重建。 |

先清理上一表，再清理这四项，合计约 **15.73 GiB**。

## 由是否保留旧测试/回退决定

| 路径 | 大小约 | 条件 |
| --- | ---: | --- |
| `pack/iq3_xxs/` | 1.44 GiB | GSQ IQ3_XXS 的 pack，当前三份预设均未引用。不再做该模型的对照时可清理；重测需从相应 GGUF 重新打包。 |
| `build-kt/pre-upstream-0.1.40.3/` 与 `pre-upstream-0.1.41/` | 0.31 GiB | 早期版本备份。只有放弃这两版直接回退时再清理。 |
| `build-kt/review-bin/` | 0.11 GiB | 旧审查候选可执行文件；当前运行的是 pool-candidate。保留旧候选对照需求时留着。 |

其他大量小日志通常只有几 MiB，清理收益远小于上述 dense.bin 和发布副本。
`logs/` 还含运行日志、配置备份及原始测试证据，不建议整目录清空。

## 当前应保留

| 路径 | 大小约 | 用途 |
| --- | ---: | --- |
| `pack/iq3kt-v4-lite/` | 1.40 GiB | 当前 v4-lite 直接使用的 pack、tokenizer 和视觉词表。 |
| `build-kt/pool-candidate/` | 0.11 GiB | 当前 v4-lite 和 v4 预设使用的引擎。 |
| `build-vision-vulkan/bin/Release/` | 0.044 GiB | 当前正在运行的核显视觉引擎。暂不建议整删 build-vision-vulkan。 |
| `pack/iq3kt/`、`pack/iq3kt-v4/` | 2.86 GiB | v2、v4 回退预设仍引用。删除会使对应预设不可用。 |
| `build-kt/Release/` | 0.45 GiB | v2 预设引用的引擎及现有回退产物。不能整删 build-kt。 |
| 本次 `Strata-v4-lite-4070TiS-Windows-20261010.zip` | 0.466 GiB | 约 476.72 MiB，需要传输的最终运行包。 |
| `dist/release/` 中旧正式 ZIP 及 SHA256 | 0.480 GiB | 原始已验证发布档案；本次运行包由它派生。 |
| `src/`、`include/`、`serve/`、`tools/`、`docs/`、`bench/results/`、`data/`、`.git/` | 很小 | 源码、测试记录、专家 profile 和 Git 数据。部分成果尚未提交，不能按“未跟踪”当作临时文件清理。 |
| `strata-models.json`、`strata-active.json`、三份生产 JSON、启动脚本 | 很小 | 预设和启动入口，保留当前路径对应关系。 |

模型文件不在本轮目录统计内。本机现有 v4-lite 服务仍直接引用原 v2/v4 GGUF；已经导出新的独立
GGUF，并不意味着本机预设已切换。因此不要根据这份 Strata 清理表删除原模型分片。

## 4070 Ti SUPER 运行包

最终文件：

```text
dist/portable-v4-lite-20261010/Strata-v4-lite-4070TiS-Windows-20261010.zip
```

ZIP 为 **499,876,913 bytes（约 476.72 MiB）**，SHA256：

```text
8d1968281a8817cfaf372564be38c7ba7e7102fe3427f384d43a97407bea0792
```

使用方法见 [运行包说明](WINDOWS_V4_LITE_4070TI.zh-CN.md)，验证记录见
[4070 运行包结果](../bench/results/2026-10-10-portable-4070/README.md)。
目标机解压后双击 START.bat，首次填模型路径；启动完成再双击 BENCHMARK.bat。
