# 给已有 v4 的 Strata 用户：v4-lite 权重补丁

这个目录可以直接转交给已有同款 v4 GGUF 的用户。应用后得到：

**v4 稠密、输出头、共享专家及 387 个高精度小投影＋v2 IQ3_KT 路由专家＋原 v4 IQ3_KT PLE 大表。**

补丁带有需要的 v2 原始权重，接收者无需下载完整 v2。所有替换张量按字节复制，保留量化时的
scale、编码和行尾数据；不重新量化或校准。112 个与 v2 完全相同的专家张量直接复用接收者的 v4，
只附带第 0–5、38–47 层的 32 个 gate/up 张量。

## 需要传哪些文件

| 文件 | 大小约 | 用途 |
| --- | ---: | --- |
| `v2-experts-iq3kt.gguf` | 9.8047 GiB | 32 个需要替换的专家张量，必需 |
| `v4_lite_patch.py`、`gguf_reader.py` | 很小 | 安装与校验脚本，Python 3.10+，只用标准库 |
| `manifest.json`、`SHA256SUMS.txt` | 很小 | 来源、文件哈希、张量形状、类型与位置 |
| `README.zh-CN.md`、`AI_PROMPT.zh-CN.md`、两份 `LICENSE-*.txt` | 很小 | 操作说明、AI 提示词及脚本依赖的许可说明 |

这份默认分享包约 **9.80 GiB**，只需要传专家文件和上表全部小文件，**无需传 PLE**。
脚本默认 `--ple iq3`，保留接收者原 v4 的 PLE，对应此前的固定 IQ3 PLE 对照组。
与 v2 IQ4 PLE 组合相比，上一轮 128 个位置的 NLL 为 1.266321 对 1.262176；差异很小，
但这不是所有任务上的质量保证。

可选的 `v2-ple-iq4kt.gguf` 约 25.0341 GiB，单独保留在发送端，没有放进本分享目录。
若以后需要，可以另行取得该文件、放入本目录，再显式使用 `--ple iq4`，对应我们当前线上部署的
IQ4 PLE 组合。两份权重合计约 34.8388 GiB。本次生成的 manifest 保留了这个可选文件的校验信息；
SHA256SUMS.txt 只列本分享目录实际包含的文件。

补丁 GGUF 是供 Strata 索引读取的部分权重文件。运行时需要保留它与原 v4 两个分片；
它不能单独作为完整模型加载。补丁会增加磁盘占用，原 v4 文件不会缩小。
在我们 19＋29 分层的机器上，换专家减少约 2.7344 GiB GPU 权重载荷；接收者的实际缓存分配、
空闲显存和速度取决于自己的硬件与设置。

## 适用的 v4

原模型的两个分片为：

```text
Qwen3.8-Flash-Next-Uncensored-ik_llama-v4-IQ3_KT-00001-of-00002.gguf
Qwen3.8-Flash-Next-Uncensored-ik_llama-v4-IQ3_KT-00002-of-00002.gguf
```

完整文件 SHA256：

```text
60a3b982267f316e1d3f88b1194afe66feef2c460a7890473251be763ca25b21  shard 1
eed2a8880920f6b7f2311278198251f77b54e019a305d5ce5050cb05f01b754b  shard 2
```

接收者需要已有可运行的 v4 Strata 配置和 native pack。引擎需支持 IQ3_KT/IQ4_KT、
`native_experts.txt` v4 的逐角色来源列、`--native-dense-gguf` 及 `--ple-gguf`。
安装脚本还会核对 v4 pack 的稠密文件和规范化后的索引，保证沿用与本组合一致的稠密转换。
如果检查失败，应先排查文件或 pack 来源，不能跳过哈希检查强行使用。

## 应用

1. 把整个补丁目录放到 v4 模型目录下，例如
   `D:\Models\Qwen\strata-v4-lite-patch-20261009\`。补丁目录名保持无空格、逗号。
   Strata 的专家索引按相对 v4 第一分片的位置查找补丁，因此两者需要在同一磁盘。
   v4 所在目录的更上层路径可以含空格。
2. 在补丁目录打开 PowerShell，用接收者自己的 Strata Python 环境运行以下命令；把三个路径
   改成他本机的路径。输出目录、输出配置必须尚不存在，它们的父目录需要已存在。

```powershell
python .\v4_lite_patch.py apply --config "D:\Strata\strata-v4.json" --out-pack "D:\Strata\pack\iq3kt-v4-lite" --out-config "D:\Strata\strata-v4-lite.json"
```

脚本读取原配置的 `cwd`、`--native`、`--pack`。若原配置未记录实际工作目录，可用
`--cwd "D:\Strata"` 明确指定。默认完整验证需要顺序读取约 82 GiB 的 v4 与专家补丁文件，耗时取决于磁盘。
脚本逐块处理，不会把整个模型载入内存，也不需要空闲 GPU。

如果以后另行收到了可选 IQ4 PLE 文件，并希望使用它：

```powershell
python .\v4_lite_patch.py apply --config "D:\Strata\strata-v4.json" --out-pack "D:\Strata\pack\iq3kt-v4-lite-iq4ple" --out-config "D:\Strata\strata-v4-lite-iq4ple.json" --ple iq4
```

也可以只检查输入，暂不创建 pack/config：

```powershell
python .\v4_lite_patch.py apply --config "D:\Strata\strata-v4.json" --verify-only
```

安装产生独立 pack，约需另加 1.4 GiB 磁盘空间。它复制已验证的 v4 稠密文件，重写专家索引；
tokenizer 继续使用原配置中的 v4 tokenizer。新配置继承本机 exe、GPU、分层、显存预留、上下文、
KV、缓存、vision 与其他运行设置。以下来源会显式设置：

| 参数/内容 | 来源 |
| --- | --- |
| `--native`、`--native-head-gguf` | 原 v4 |
| 两次 `--native-dense-gguf` | 原 v4 两个分片 |
| `native_experts.txt` 的 112 个不变张量 | 原 v4 |
| `native_experts.txt` 的 32 个替换张量 | 补丁的专家 GGUF |
| `--ple-gguf` | 默认原 v4 IQ3 PLE；显式 `--ple iq4` 时为另行收到的 IQ4 PLE 补丁 |

新配置包含接收者自己的 API key 等私有设置，保存在本机配置位置，不能放回分享目录。
安装脚本只准备新文件；接收者按自己的预设管理流程登记并启用新配置，使用原 v4 配置即可回退。
有正在运行的服务时，按仓库指南识别所属进程、等待请求完成后切换，保留原启动器和唤醒参数。
迁移补丁或模型位置后，需要重新运行脚本生成对应路径的新 pack/config。

## 给对方 AI 的入口

把目录中的 `AI_PROMPT.zh-CN.md` 全文发给对方 AI，并告知补丁目录与本机 v4 配置路径。
它可以直接调用附带的脚本，核对来源后接入本机预设。

本补丁的验证重点是与已测试组合的权重字节、索引和来源一致。已有的小样本 NLL/KLD 结果
不构成对接收者机器速度或所有任务质量的保证；启用后的健康检查和简短生成应在接收者本机完成。
