# v4-lite 运行包：4070 Ti SUPER 16GB＋64GB DDR4

这是 Windows 10/11 64 位便携运行包，使用已验证的 Strata 0.1.41-iqkt.1 CUDA 引擎和匹配的
Python 服务，包含 RTX 40 系所需的原生 sm89 代码。模型为 v4 稠密权重（含 387 个高精度投影）＋
v2 IQ3_KT 专家＋独立 IQ4_KT PLE。

包内带 Python、CUDA 13 runtime、cuBLAS、必要的 VC 运行库、Web/API 和模型打包工具。
无需另装 Python、Git、Visual Studio 或 CUDA Toolkit，也不下载模型。
目标机需要支持 AVX2 的 CPU，以及支持 CUDA 13 的 NVIDIA R580 或更新驱动。

## 启动

1. 把 ZIP 完整解压到本机可写目录，例如 `D:\Strata-v4-lite\`，不要在压缩软件内启动。
2. 两份模型保留原名、放在同一 SSD 文件夹。双击 **START.bat**，首次填写第一分片路径
   或模型文件夹路径。第一次会在运行包下生成约 1.4 GiB 的 pack，随后加载模型。
3. 等待服务就绪后打开 `http://127.0.0.1:8080/`，可使用 Chat 和 Monitor。
   API base URL 为 `http://127.0.0.1:8080/v1`，模型名可填 `strata`。

两份模型分别为：

```text
Qwen3.8-Flash-Next-V4-Lite-IQ3_KT-IQ4PLE-00001-of-00002.gguf
Qwen3.8-Flash-Next-V4-Lite-IQ3_KT-IQ4PLE-00002-of-00002.gguf
```

第一份 49.52 GiB，第二份 25.03 GiB，仅包含 IQ4_KT PLE。原 v2/v4、专家补丁和开发机 pack
都不需要。设置时检查文件长度和完整 GGUF 头部；**VERIFY-MODEL.bat** 可以另外完整核对两个文件
SHA256，用于确认传输后的所有权重字节，可能需要几分钟。

首次在目标机生成新的 API key，保存在 `local/API_KEY.txt` 和 `strata-v4-lite.json` 中。
网页和客户端使用这个 key。默认仅本机可访问，不带开发机的 key、会话或私有配置。
Ctrl+C 或关闭启动窗口结束服务。初次加载大量专家到 RAM 时可能暂时卡顿，请等待引擎日志推进。

模型之外建议留至少 6 GiB 磁盘余量供运行包、pack、日志与校验解压使用。PLE 使用 SSD direct；
Windows 分页文件保持系统管理。64GB RAM 还要容纳 Windows、专家、KV 和文件缓存，测试时先关闭
占用大量内存/显存的软件，避免同时启动第二份服务。

## 测速

服务加载完成后，双击 **BENCHMARK.bat**。它依次执行一次短预热、短输入、较长输入首次处理、
相同较长输入再次处理，后面三组各请求最多 256 个输出 token。
较长输入约数千 token，以报告中的实际 token 数为准。若模型提前结束，报告会保留实际输出长度。

结果保存在 `results/benchmark-时间/` 下的 `results.json` 和 `README.md`，包括：

- 输入/输出 token、前缀缓存 token、首字时间（TTFT）、引擎 prefill 和 decode 速度。
- CPU 型号和核数、GPU 型号、系统内存大小。
- 测试期间设备显存峰值、最低可用 RAM、GPU 利用率及观察到的 PCIe 代数/宽度。

可以把这个结果目录带回来分析。报告不保存 API key、完整请求或模型回答正文。
显存是设备总占用，TTFT 包含请求开销；首次处理和缓存复用分别比较，不能混成一个速度。
这只是速度测试，不是质量评测或完整 64K 上下文测试。

**CHECK.bat** 只检查随包 Python 依赖、显卡和驱动，不加载模型。
启动失败时先看服务窗口和 `logs/strata-v4-lite-engine.log`。

## 起步参数

| 项目 | 默认设置 |
| --- | --- |
| GPU | NVIDIA 编号 0，单卡 |
| 上下文上限 | 65536（64K） |
| KV 精度 | int8，即 K8V8 |
| GPU 常驻 KV | 20480 |
| Prefill chunk | 1024 |
| 专家 GPU 缓存 | auto，按目标机实际余量分配 |
| 显存预留 | 1024 MiB |
| GPU 外的专家 | resident-experts，留 RAM |
| PLE 大表 | SSD direct，保留 IQ4_KT |
| CPU 亲和性 | auto，不复制开发机核号 |
| PCIe 流送比例 | 使用目标机启动时的带宽探测，不固定比例 |
| 前缀检查点 | 2，周期检查点关闭 |
| 视觉 / MTP 侧车 | 未启用 |

v4-lite 的全部 GPU 权重载荷约 49.45 GiB，其中专家约 44.41 GiB。16GB 显存下只能缓存其中一部分，
大量专家由 RAM/CPU 和 PCIe 路径参与运算。PCIe 4.0×16 比开发机实测的慢链路更有利，但 DDR4 带宽、
CPU 型号和实际缓存命中率都会影响结果，因此这里不预报目标机的 tok/s。

这份运行包使用已有发布引擎，未加入本机后来部署的 512K 多会话池和 GPU 唤醒启动器。
它保留普通前缀复用；先按单会话测试资源占用，再根据目标机实测决定是否扩展。

修改配置时先 Ctrl+C 停止自己的 START 窗口，再运行以下命令或编辑 `strata-v4-lite.json`。
已完成的 pack 会复用，改上下文或显存预留不会重写模型：

```powershell
# 显存紧张时降低预填充块、增加余量
.\python\python.exe -X utf8 tools/windows/lite_runtime.py configure --prefill 512 --reserve-mib 1536
# 系统内存紧张时先改为 32K 上下文
.\python\python.exe -X utf8 tools/windows/lite_runtime.py configure --context 32768
# 更换模型目录，仍要求同一套两分片
.\python\python.exe -X utf8 tools/windows/lite_runtime.py configure --model "D:\Models\v4-lite"
# 8080 已被其他服务占用时换端口
.\python\python.exe -X utf8 tools/windows/lite_runtime.py configure --port 8081
```

不用复制双卡的 layer-split、核显设备名或固定 PCIe 比例。显卡驱动版本不足时由目标机更新驱动；
这个包本身不安装驱动，也不修改系统设置。

`BUILD.json` 记录本包组成，`BASE-BUILD.json` 记录原引擎来源，`FILES.sha256` 用于校验程序文件。
模型摘要在 `data/v4-lite-model.json`。本包未在物理 4070 Ti SUPER / DDR4 目标机上跑过推理；
应以目标机首次启动和 BENCHMARK 输出作为该电脑的验证结果。
