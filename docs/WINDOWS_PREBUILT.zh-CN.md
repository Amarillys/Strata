# Windows 预编译包：v4 IQ3_KT / RTX 5080 16G

适用模型：`Qwen3.8-Flash-Next-Uncensored-ik_llama-v4-IQ3_KT-00001-of-00002.gguf`
及其第二分片。起步配置面向 RTX 5080 16 GiB、64 GiB DDR5、Intel i9-13900HX。

## 下载和依赖

从 [Amarillys/Strata Release](https://github.com/Amarillys/Strata/releases/tag/v0.1.41-iqkt.1) 下载名称包含
`windows-x64-cuda13.zip` 的压缩包。GitHub 自动生成的 `Source code` 不是预编译包。

包内包含 Strata 的 IQ3_KT / IQ4_KT 引擎、Vulkan 视觉编码器、Python 3.13.16、服务/Web 页面、
模型打包器、Python 依赖、CUDA 13.0.96 runtime 和 cuBLAS 13.1.0.3。无需安装 Python、
Visual Studio、CMake、Git 或完整 CUDA Toolkit。CUDA Toolkit 只在制作这个包时使用。

电脑端仍需要：

- Windows 10/11 x64，支持 AVX2 的处理器；13900HX 满足要求。
- 支持 CUDA 13 的 NVIDIA 显卡驱动，R580 或更新；建议使用显卡对应的当前正式驱动。
- 两份完整的 v4 GGUF，放在同一个本地 SSD 目录，保持原文件名。合计约 **72.52 GiB**；
  47.15 GiB 是其中 routed experts 的大小，不是全部下载量。
- 模型之外留出至少 8 GiB 磁盘余量供程序、生成的小投影包和日志使用。Windows 分页文件保持
  “系统管理”，并给所在磁盘留足空间；运行前关闭占用大量内存/显存的软件。
- 可选识图：匹配模型的 MMPROJ，以及 Intel 核显驱动。13900HX 通常是 Intel UHD，
  不能直接复制开发机的 Iris Xe 设备名。文本模式不需要 MMPROJ。

只装驱动，不必再下载 CUDA 13 Toolkit。许可证位于 `licenses/`、`third_party/ik_kt/` 和
Python 依赖的 `.dist-info` 目录。`BUILD.json` 记录编译来源及依赖版本，`FILES.sha256` 记录包内校验值。
模型来源：[ji-farthing 的 GGUF 仓库](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-Uncensored-ik-llama-GGUF)。
请选择文件名同时包含 `v4`、`IQ3_KT` 的两份分片；模型权重不包含在程序压缩包内。

## 三步启动

1. 完整解压压缩包，例如 `D:\Strata-v4\`。不要直接在压缩软件内部运行。
2. 双击 **CONFIGURE.bat**，粘贴第一分片 GGUF 的完整路径。脚本会校验分片、准备小投影和
   tokenizer，生成 `strata-v4.json`。专家与 PLE 保留原始 KT 数据，不会复制整套模型。
3. 双击 **START.bat**，等待加载完成。Web Chat/Monitor：`http://127.0.0.1:8080/`；
   OpenAI API base URL：`http://127.0.0.1:8080/v1`，模型名可填 `strata`。

本机随机生成的 API key 在 **local/API_KEY.txt**，也保存在 `strata-v4.json` 的 `api_key`。
在网页和 API 客户端中填入同一个值。每次启动会保留原 key；请勿上传这两个本机文件。
Ctrl+C 或关闭启动窗口结束服务。首次加载大量 RAM 专家需要时间，可在 Monitor 和
`logs/strata-v4-engine.log` 查看进度。

本次 64 GiB RAM 的受限缓存实测中，首次装载约 66 秒；装载期间文件缓存使可用 RAM
短暂降到约 0.2 GiB，完成后恢复到约 15 GiB。启动初期可能明显卡顿，应先关闭大型程序，
等待日志继续推进，避免同时启动第二个实例。日志显示 CUDA page-locking 被拒绝后，
仍可通过 Windows VirtualLock 成功驻留；请以最终 `resident RAM mode` 行判断是否成功。

`CHECK.bat` 检查 Python 依赖、GPU 枚举；命令行运行 `CHECK.bat --vision` 还会列出 Vulkan 设备。
它不下载模型，也不安装驱动。

## 默认配置及调整

起步模板在 `tools/windows/templates/v4-5080-16g.json`；实际修改 **strata-v4.json**，保存后重启。
这些是单卡起步设置，尚不是 5080 实机调优结果。

| 项目 | 默认值 | 作用 |
| --- | --- | --- |
| NVIDIA GPU | `"gpu": 0` | 单卡，无 layer split |
| 上下文容量 | `--max-context 65536` | 先验证 64K，再增加 |
| KV | `--kv int8` | 即 K8V8 |
| GPU KV | `--kv-resident 20480` | 完整 KV 池在 RAM；GPU 保留 resident 部分 |
| Prefill chunk | `--prefill 1024` | 减少小显存设备的峰值工作区；可实测 2048 |
| 专家显存缓存 | `--expert-cache auto` | 根据启动时实际余量分配 |
| 显存预留 | `--vram-reserve-mib 1024` | 桌面/图计算余量；内存不足时提高 |
| CPU 专家 | `--resident-experts` | 优先把 GPU 外的专家保留在 RAM，不足时报告回退 |
| CPU 线程 | `--pool-affinity auto` | 在混合架构 CPU 上自动选择 P 核，不硬编码本机核号 |
| PCIe 流送比例 | 不固定 `--pcie-frac` | 使用引擎的实际链路探测 |
| 前缀缓存 | `--prompt-cache 6` | RAM 检查点，用于多轮对话复用 |
| PLE / ngram | `--ple-io direct` | 约 20.27 GiB 大表留 SSD，行缓存留 RAM |
| MTP | 关闭 | 没有加载 MTP 侧车；`--spec 3` 本身不加载侧车 |
| 视觉 | 关闭，可另行启用 | 不占用 5080 的专家显存预算 |

16 GiB 无法容纳 v4 的约 52 GiB GPU 权重载荷。大部分专家会在 RAM，由 CPU 或 RAM→GPU 流送
参与计算，速度不能套用开发机 54 GiB 双卡的 40–60 tok/s。64 GiB 系统内存也需要同时容纳
专家、KV、前缀检查点和 Windows；请以启动日志是否完整驻留、实际可用 RAM 和分页活动判断。

常用调整可直接运行，不会重新复制权重；已完成且源文件未变的 pack 会复用：

```powershell
.\python\python.exe -X utf8 tools/windows/portable.py configure --context 131072
.\python\python.exe -X utf8 tools/windows/portable.py configure --context 65536 --prefill 2048
.\python\python.exe -X utf8 tools/windows/portable.py configure --kv k8v4
```

从 64K 开始测，确认内存余量后再尝试 128K；256K 不是这个单卡模板的默认承诺。
K8V4 可进一步减少 KV 占用，但不会减少专家权重。遇到显存不足可先将 prefill 降到 512、
显存预留提高到 1536；若 RAM 专家发生分页，优先减少同时运行的软件和上下文/缓存需求。

局域网访问：把 `strata-v4.json` 的 `host` 改成 `0.0.0.0`，保留生成的 `api_key`，按需允许
Windows 防火墙 TCP 8080。其他设备访问 `http://这台电脑的局域网IP:8080/`，API 后缀为 `/v1`。

## 可选：让核显识图

先完成文本启动。停止服务，准备匹配模型的 MMPROJ，再双击 **ENABLE-VISION.bat**，输入其路径。
脚本会寻找唯一的 Intel Vulkan GPU，以完整设备名写入配置，同时生成 KT 兼容的视觉词表。
没有找到 Intel GPU 时会停止配置，不会自动把视觉权重装到 5080。

也可手动指定 `CHECK.bat --vision` 列出的设备名称：

```powershell
.\python\python.exe -X utf8 tools/windows/portable.py vision `
  --mmproj "D:\Models\mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf" `
  --device "Intel(R) UHD Graphics"
```

以上设备名只是示例。启动日志应确认选择了 Intel；默认每张图最多 768 个视觉 token。
核显共享系统内存，开启视觉后也要检查 64 GiB RAM 余量。开发机已验证 Iris Xe 路径，
13900HX 的 UHD 型号、驱动和实际识图性能需要朋友端确认。

## 验证范围

这个包使用 CUDA 13.0.2 和 MSVC 2022 构建，包含原生 sm75、sm89、sm120 GPU 代码；
CPU 基线为 AVX2，视觉为单独的 Vulkan/AVX2 程序。已校验 DLL 闭包、随包 Python、
KT 数值对照和 v4 单卡启动。本次 89 个 PE 文件依赖检查、14 项启动器测试、两张本机 GPU
各 38 组 KT 对照、CPU AVX2 对照均通过。从带空格路径中解压后的实际 ZIP 完成 v4 打包、
生成、Web/API 鉴权和前缀缓存验证；核显 helper 也完成 512×512 图片编码。
详细条件见包内 `bench/results/2026-10-09-windows-cuda13/README.md`。

开发机没有 RTX 5080，**sm120 已编译，不等于完成 5080 实机吞吐或 64K 满上下文验证**。
受限缓存测试只用于检查此模板的运行路径。模型小投影仍有 `conversions.json` 记录的 BF16 等
转换，不能把 KT 字节保留理解成整条推理链与 IK 逐位一致。

## 复现发布包（仅维护者）

普通用户无需执行本节。源码分支为 `feat/iqkt-vulkan-vision`，ggml/llama.cpp 固定到
`3cf03257f219afbe7334045ff7c6a06ac68c627d`。需要 Python 3.11+、MSVC 2022、CMake、Ninja 和 Vulkan SDK。

```powershell
git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp checkout --detach 3cf03257f219afbe7334045ff7c6a06ac68c627d
python tools/windows/prepare_dependencies.py --out build-release-deps
# 在 VS x64 Developer PowerShell 中；把路径改成自己的目录。
$strataCuda = (Resolve-Path build-release-deps/cuda-13.0.2).Path
$strataGgml = (Resolve-Path third_party/llama.cpp).Path
$env:CUDA_PATH = $strataCuda
$env:PATH = "$strataCuda/bin;$strataCuda/bin/x64;$strataCuda/nvvm/bin/x64;$env:PATH"
cmake -S . -B build-release-cuda13 -G Ninja -DCMAKE_BUILD_TYPE=Release `
  "-DCMAKE_CUDA_COMPILER=$strataCuda/bin/nvcc.exe" "-DCUDAToolkit_ROOT=$strataCuda" `
  '-DCMAKE_CUDA_ARCHITECTURES=75;89;120' -DSTRATA_ENABLE_CUDA=ON -DSTRATA_PORTABLE=ON `
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared "-DSTRATA_GGML_DIR=$strataGgml" `
  -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF `
  -DGGML_AVX_VNNI=OFF -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF
cmake --build build-release-cuda13 --target strata strata-device kt_parity kt_cpu_parity --parallel 6
cmake -S tools/vision -B build-release-vision -G Ninja -DCMAKE_BUILD_TYPE=Release `
  "-DLLAMA_DIR=$strataGgml" -DSTRATA_VISION_CUDA=OFF -DSTRATA_VISION_VULKAN=ON `
  -DSTRATA_PORTABLE=ON -DGGML_OPENMP=OFF
cmake --build build-release-vision --target strata-vision --parallel 4
python tools/windows/package_release.py --build build-release-cuda13 --vision-build build-release-vision `
  --cuda build-release-deps/cuda-13.0.2 --downloads build-release-deps/downloads `
  --ggml third_party/llama.cpp --crt "$env:VCToolsRedistDir/x64/Microsoft.VC143.CRT" --out dist
```

请使用全新的 build 目录。发布器拒绝覆盖现有 staging 目录，要求 portable 编译、sm120 原生
代码、指定 ggml commit，检查非系统 DLL 缺失，并生成整个 ZIP 和包内文件的 SHA256。
依赖来自 NVIDIA、python.org、PyPI 的固定 URL，下载时检查锁文件中的 SHA256。
