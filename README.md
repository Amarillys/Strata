# Strata · IQ_KT 与异构双卡优化

本仓库是 [Niko1221/Strata](https://github.com/Niko1221/Strata) 的扩展分支，重点支持
**IK 的 IQ3_KT / IQ4_KT 量化模型、CUDA / AVX2 算子优化，以及独立 Vulkan 核显视觉编码**。
主要验证模型为 Qwen3.8-Flash-Next Uncensored 的 IQ3_KT v2 / v4。

Native IQ3_KT/IQ4_KT support, optimized CUDA/AVX2 kernels, and a separate Vulkan vision encoder.
For the original Strata introduction and installer, see [the upstream README](README_UPSTREAM.md).

当前分支：`feat/iqkt-vulkan-vision`。已合入上游 **v0.1.41 / `fb58e0d`**。
本页介绍本分支的实现和实测；完整过程见 [工作总结](docs/IQ_KT_SUMMARY.zh-CN.md)。

## 本分支增加了什么

| 改动 | 实现范围 |
| --- | --- |
| **原生 IQ3_KT / IQ4_KT** | 识别 IK GGUF 类型 154 / 155，处理逐行 scale、尾块和对齐；覆盖模型读取、打包、embedding、稠密投影、专家和 PLE。保留原 ggml 依赖与类型枚举。 |
| **KT decode 优化** | CUDA 直接重建打包整数码并执行 DP4A；CPU 使用共享 trellis 表和 AVX2 整数点积，跨 token 复用解码，保留标量对照。 |
| **KT prefill** | 多 lane 并行解码，向量写出 FP16 / BF16 / FP32，再接入现有矩阵乘法路径。 |
| **KT MTP 侧车** | 专家保留原始 KT 字节；转换运行时需要的 Q8_0 / BF16 投影，并校验格式、尺寸和完成标记。 |
| **Vulkan 视觉** | 独立编码进程、按设备名称选择 GPU、视觉词表导出和独立环境配置；本机由 Intel Iris Xe 处理原 F16 MMPROJ。 |
| **部署工具** | 请求触发 GPU 唤醒包装器，解决本机 eGPU 低频问题；v2 / v4 预设切换工具等待请求结束、检查进程归属，并在启动失败时恢复旧预设。 |

专家缓存、RAM KV 流送、按层多卡分工、前缀缓存和 Web/API 服务来自上游 Strata。
本分支在这些机制上接入 KT，并记录异构双卡的分层、缓存、MTP 和长上下文对照。

## 实测结果

测试机器：**Windows、i9-12900HK、64 GiB RAM、2080 Ti 22 GiB（PCIe 3.0 ×8）、
4080 SUPER 32 GiB（USB4）和 Intel Iris Xe**。以下数字针对这台机器，MTP 均关闭。

### KT 算子的独立收益

v2、相同分层/缓存/KV 设置、新旧算子交替运行，每次生成 256 token：

| 输入 token | 原 CUDA KT 算子 | Packed KT 算子 | 提升 |
| ---: | ---: | ---: | ---: |
| 72 | 39.21 tok/s | 63.37 tok/s | 61.6% |
| 3,570 | 39.43 tok/s | 62.22 tok/s | 57.8% |

对应输出 token 一致。另一次固定频率对照支持收益来自算子变化。
这是本分支算子的前后对照；与 IK、不同模型版本或不同上下文配置的结果不能直接等同。
条件及原始记录见 [性能分析](docs/IQ_KT_PERFORMANCE.md)。

### 当前 v4：256K / K8V8 / 22＋26 层

| 工作负载 | 输出 token | Prefill | Decode |
| --- | ---: | ---: | ---: |
| 3,305-token 输入 | 每次 256，共四次 | 未缓存均值 7.272 s | **41.43 tok/s** |
| 9,769-token 输入 | 每次 256，共四次 | 未缓存均值 13.045 s | **41.86 tok/s** |
| 261,618-token 长输入 | 384 | 320.104 s | **36.6 tok/s** |
| 262,024-token 立即续聊 | 10 | **0.364 s** | 39.7 tok/s |

长输入正确取回三个分散位置的校验串；续聊复用 **262,001 token**，只新读入 23 个，
总输入输出达到 **262,034 / 262,144**。这是近容量上限的检索和运行验证，不是通用质量评测。

同日同版本的 21＋27 / 200K 在短、中输入下为 42.84 / 43.82 tok/s。
当前方案慢约 **1.4–2.0 tok/s（3.3–4.5%）**，按容量偏好采用 256K。
上游 0.1.41 的独立版本 A/B 没有显示整体吞吐提升，不能把上述算子收益再次计入升级。

逐次计时、提示词与校验摘要：[256K 基准](bench/results/2026-10-08-v4-256k/README.md) ·
[0.1.41 升级基准](bench/results/2026-10-08-iqkt-upstream-0.1.41/README.md)。

## 已验证的双卡配置

| 项目 | v2 IQ3_KT | v4 IQ3_KT |
| --- | --- | --- |
| 2080 / 4080 层数 | 19 / 29 | 22 / 26 |
| 上下文容量 | 262,144 | 262,144 |
| KV / GPU resident cells | K8V8 / 20,480 | K8V8 / 20,480 |
| 2080 / 4080 常驻专家 | 9,728 / 14,848 | 9,090 / 13,312 |
| RAM 专家 | 0，全部专家在两卡显存 | 约 4.11 GiB；4080 负责的专家全部驻留 |
| PLE 大表 | 约 25.03 GiB，SSD direct | 约 20.27 GiB，SSD direct |

两套配置均使用 prefill 2048、RAM 前缀检查点 6、MTP 关闭和 Intel 核显视觉。
完整 KV 池位于 RAM，GPU 保留 resident 部分。两张卡的显存按阶段分配，不能视为统一显存池。
v4 提高了部分专家和稠密权重的精度，不能沿用 v2 全驻留和约 60 tok/s 的结论。
配置细节及权重分析见 [按需启动](docs/IQ_KT_MODEL_SWITCH.zh-CN.md)和 [v4 分析](docs/IQ_KT_V4_ANALYSIS.zh-CN.md)。

## 构建与使用

**KT 模型目前需要手动构建和打包，尚未接入上游一键安装菜单。**

```powershell
git clone --branch feat/iqkt-vulkan-vision https://github.com/Amarillys/Strata.git
cd Strata
```

1. 按 [KT 构建文档](docs/IQ_KT.md)准备指定 ggml 依赖，编译 `strata`、`kt_parity` 和
   `kt_cpu_parity`，使用 `tools/iq_pack.py` 生成模型包。文档中的本机路径需要按实际位置调整。
2. 需要图片输入时，按 [Vulkan 视觉文档](docs/IQ_KT_VISION.md)构建独立 helper，
   使用 `tools/vision_vocab.py` 导出视觉词表，并选择目标设备。
3. 按 [部署说明](docs/IQ_KT_MODEL_SWITCH.zh-CN.md)配置服务与模型预设。
   `tools/switch_preset.py` 提供受控切换；模型路径、API key、目录 JSON 和 `run-strata-*` 启动器是本机文件，
   不随源码分发。GPU 唤醒包装器还需要配置已有的 `gpu_monitor.py` 电源模块路径。

本机服务使用 **8080**，保留 Web Chat/Monitor、OpenAI 兼容 API 和原模型别名。
浏览器地址为 `http://127.0.0.1:8080`，API base URL 为 `http://127.0.0.1:8080/v1`；
局域网监听配置为 `0.0.0.0` 时启用 API key。
通用 Strata 安装与其他模型说明见 [原 README](README_UPSTREAM.md)及 [安装文档](docs/INSTALL.md)。

## 验证范围

- 最新 IK DLL 对照覆盖 **38 组行数据、974 段真实张量**，最大相对解码误差约 **1.73×10⁻⁷**；
  sm75 / sm89 的 KT CUDA 和 CPU 标量 / AVX2 对照通过。
- 0.1.41 服务测试 **595 项，6 项跳过，其余通过**；另有 **48 项工具测试**通过。
  文本、核显识图、前缀缓存、LAN/Web/API 和模型别名均完成运行检查。
- 执行验证覆盖 Windows CUDA、x86 AVX2 和 Intel Vulkan 视觉。尚未测试 KT 在 HIP/SYCL 上的执行
  或完整 IQ4_KT 主模型的吞吐；专用 KT MMQ 尚未实现，prefill 使用解码后的矩阵乘法路径。
- 专家和 PLE 保留原量化字节，但运行时要求的小投影存在记录在案的格式转换。
  数值对照、生成一致或一次长输入检索，不代表整个推理链与 IK 逐位一致或所有任务质量不变。

## 文档入口

| 文档 | 内容 |
| --- | --- |
| [工作总结](docs/IQ_KT_SUMMARY.zh-CN.md) | 实现、性能来源、配置与验证边界 |
| [KT 技术文档](docs/IQ_KT.md) | 格式布局、构建、打包、数值对照和 MTP 侧车 |
| [Decode 性能](docs/IQ_KT_PERFORMANCE.md) / [Prefill](docs/IQ_KT_PREFILL.md) | 算子优化、GPU 频率、分块和历史对照 |
| [Vulkan 视觉](docs/IQ_KT_VISION.md) | 核显编码、词表兼容与服务接入 |
| [v2 / v4 按需启动](docs/IQ_KT_MODEL_SWITCH.zh-CN.md) | 当前参数、8080、前缀缓存和预设切换 |
| [v4 显存与优化分析](docs/IQ_KT_V4_ANALYSIS.zh-CN.md) | 混合精度、KLD 来源、CPU/流送开销与后续方向 |
| [0.1.41 合并部署](docs/IQ_KT_UPSTREAM_0.1.41_MERGE.zh-CN.md) | 上游合入、构建、版本 A/B 和最终部署 |

## 上游与许可证

Strata 引擎和服务来自 [Niko1221/Strata](https://github.com/Niko1221/Strata)；
KT 布局与 trellis 重建参考 [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)，
底层 ggml 和视觉使用 [llama.cpp](https://github.com/ggml-org/llama.cpp)。
模型来自 Qwen 及相应 Uncensored/量化发布者，模型权重遵循各自许可证。

本仓库沿用 [MIT License](LICENSE)。上游原首页保存在 [README_UPSTREAM.md](README_UPSTREAM.md)，
其多语言说明、致谢与通用安装信息一并保留。
