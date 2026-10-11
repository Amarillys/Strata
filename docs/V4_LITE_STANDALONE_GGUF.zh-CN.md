# v4-lite 主模型＋独立 IQ4_KT PLE GGUF

这个目录是一套完整的两分片模型。复制两份 GGUF 到另一台电脑即可携带全部模型权重，
运行时无需原 v2、原 v4 或之前的专家补丁。

| 文件 | 内容 | 大小约 |
| --- | --- | ---: |
| `Qwen3.8-Flash-Next-V4-Lite-IQ3_KT-IQ4PLE-00001-of-00002.gguf` | 1,223 个张量：v4 稠密等非专家权重＋v2 全部 IQ3_KT 路由专家 | 49.52 GiB |
| `Qwen3.8-Flash-Next-V4-Lite-IQ3_KT-IQ4PLE-00002-of-00002.gguf` | 仅 PLE 大表，v2 IQ4_KT | 25.03 GiB |

两份合计约 **74.55 GiB**。第二份就是单独的 PLE GGUF，使用标准分片命名便于 Strata 自动发现。
两份放在同一目录，保留配套文件名。SHA256SUMS.txt 可用于传输后校验；manifest.json 记录每个张量的
原始来源和 SHA256，来源文件名仅用于追溯，不构成运行依赖。

保留的 v4 部分包括全部 387 个高精度小投影、共享专家、输出头、token embedding、PLE key/value
投影和其他非路由专家张量。48 层的 144 个路由专家张量全部来自 v2 IQ3_KT。PLE 表来自 v2 IQ4_KT，
type 155、shape [160, 320001536]。所有张量直接复制量化字节，没有重新量化或校准。

第一分片保留 v4 的 tokenizer、chat template、架构参数等原始 metadata 编码。只调整模型名称、
分片信息并增加组合说明；模型共 1,224 个张量，第二分片只有一个 PLE 张量。

## 文件大小与显存

**49.52 GiB 是主 GGUF 的磁盘大小。约 45 GiB 仅接近专家部分，不能当作完整模型的显存需求。**

| 统计对象 | 大小约 | 含义 |
| --- | ---: | --- |
| v4-lite 全部路由专家 | 44.41 GiB | 不含稠密权重、输出头等 |
| v4-lite 全部 GPU 权重载荷 | 49.45 GiB | 19＋29 分层，假设所有专家都驻留 GPU |
| 原版 v4 全部 GPU 权重载荷 | 52.2 GiB | v4-lite 的专家权重减少约 2.73 GiB |
| 本机同组合实测 NVML 峰值之和 | 52.09 GiB | 两卡的设备总占用，包含运行缓冲区和其他设备占用 |

上述 GPU 权重载荷不含 KV、工作区、计算图、分配粒度及驱动。此前本机同组合使用
RTX 2080 Ti 22 GiB＋RTX 4080 SUPER 32 GiB、19＋29 分层、960/512 MiB 预留，
两卡 NVML 峰值分别为 21,397.3125 / 31,945.640625 MiB；这是各卡峰值相加，
不代表仅模型权重的占用，也不是目标电脑的最低显存要求。仍有约 777 MiB 专家留在 RAM。

25.03 GiB 的 IQ4_KT PLE 大表沿用 Strata 的 SSD direct 路径，不需要整份装进显存。
目标电脑显存较小时可以让更多专家留在 RAM，速度取决于显存、内存与硬盘配置。
本次重新打包保持张量字节不变；相同引擎和设置下，不会因改成这套文件而额外增加权重显存。

## 在另一台电脑的 Strata 上使用

使用支持 Qwen3.8 Flash Next 的 IQ3_KT/IQ4_KT 格式的 Strata。先将两份 GGUF 放到同一目录，
然后在 Strata 项目目录、使用该项目的 Python 环境生成本机 pack。下面以 D 盘路径为例：

```powershell
python tools/iq_pack.py --gguf "D:\Models\v4-lite\Qwen3.8-Flash-Next-V4-Lite-IQ3_KT-IQ4PLE-00001-of-00002.gguf" --out "pack/v4-lite-portable" --compat-bf16
```

如果提示找不到 `llama.cpp's gguf-py`，先在当前 PowerShell 会话中指定该电脑已有的
`llama.cpp/gguf-py` 目录，再重试上述命令，例如：

```powershell
$env:STRATA_GGUF_PY = "D:\llama.cpp\gguf-py"
```

这是打包工具的 Python 依赖路径，与原 v2/v4 权重无关。通过 Strata setup 安装到
`third_party/llama.cpp/gguf-py` 的环境通常可以自动找到它。

`--compat-bf16` 是 Strata 对部分小投影使用的运行时转换，与我们已有 v4 pack 的流程一致；
它不会改写这两份 GGUF，也不会改变量化专家或 PLE 表。生成的 pack 约需额外 1.4 GiB，
tokenizer 会从主 GGUF 导出。不需要额外生成一份约 44 GiB 的 experts.bin。

新建本机预设时使用该 pack，以及下面的模型来源：

```text
--pack <新生成的 pack>
--native <00001-of-00002.gguf>
--native-head-gguf <00001-of-00002.gguf>
--native-dense-gguf <00001-of-00002.gguf>
--ple-gguf <00002-of-00002.gguf>
```

所有稠密张量都在第一分片，PLE 大表只在第二分片。标准分片信息也允许引擎从 `--native`
自动找到第二份；显式写出来源可方便核查配置。服务的 tokenizer 指向新 pack 下的 tokenizer 目录。

GPU、分层、显存预留、上下文、KV、缓存、端口与 API key 使用目标电脑自己的设置。
如果需要视觉输入，继续使用目标电脑原有的兼容视觉模型和 mmproj；本目录包含的是主语言模型。
通过目标电脑已有的预设管理流程启用，检查 health、实际模型路径和一次短文本生成。

## 已完成的检查

- 全部 1,224 个张量逐一核对形状、量化类型与源字节 SHA256，并读回核对两个完整文件 SHA256。
- Strata 实际 C++ GGUF 读取器通过架构、分片、张量边界和独立 IQ4_KT PLE 检查。
- 在只允许打开这两个新 GGUF、禁止子进程的 Python 审计中，成功生成 pack 和 tokenizer。
- 新 pack 的 dense.bin 与原 v4 逐字节一致，索引数据行与五份 tokenizer 文件 SHA256 一致；
  48 层的 144 个 IQ3_KT 专家张量都指向主分片中的正确偏移。

这次检查未切换本机生产服务，也未在目标电脑运行 GPU 推理。上面的显存数据来自此前相同
权重组合的本机测试，不是对新文件在目标电脑上的测量。

## 可以交给目标电脑 AI 的操作说明

> 我有这一目录的两份 Qwen3.8-Flash-Next v4-lite GGUF，第二分片仅包含 IQ4_KT PLE。
> 请先阅读目录内 README.zh-CN.md 和本机 AGENTS.md，使用现有 Strata 配置与硬件设置，
> 校验两个文件的 SHA256，在新目录运行 tools/iq_pack.py --compat-bf16 生成 pack 和 tokenizer，
> 建立独立预设并完成一次启动与生成检查。所有模型权重都在这两个 GGUF 内，不需要下载原 v2/v4。
> 主分片用于 native、head 和 dense，第二分片用于 ple-gguf。保留已有预设作为回退，
> 按本机服务管理流程切换，不输出 API key 或整份私有配置。
