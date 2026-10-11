# GSQ-RCO IQ3_XXS 的精度预算与 v4 对照

2026-10-09，读取用户指定的 `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` 两个本地 GGUF 分片，
与 Uncensored IQ3_KT v2/v4 比较。另取得作者发布说明和 IQ3_XXS 分配清单，固定到版本
[`ed59f92082b1e93c0e96d60a8b11aab089b52f09`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/ed59f92082b1e93c0e96d60a8b11aab089b52f09/README.md)。
本次只做文件与源码核算，没有重新加载模型、生成权重或改动在线服务。

**这套模型把大量专家 gate/up 压到 IQ2 系列，把部分 down 提到 IQ4_NL，并保留关键小投影为
BF16。大块 attention/GDN 使用逐张量的混合精度。IQ3_XXS 是预算档位名称。**

## 作者怎样决定分配

按[发布说明](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/ed59f92082b1e93c0e96d60a8b11aab089b52f09/README.md#quantization-procedure)，制作分为三步：

1. **GSQ 生成各档候选**：对每个待搜索张量，生成候选 GGUF 格式的量化版本。GSQ 是
   Gumbel-Softmax Quantization，优化各权重的网格指派和每组缩放系数。
2. **RCO 搜索组合**：Riemannian Constrained Optimization 在总容量约束下，根据任务损失
   搜索张量的精度组合。作者把搜索单元记为 304 个 dense 张量和 48 个按层组织的专家矩阵组。
3. **组装 GGUF**：将选中的量化张量写入标准 GGUF，部署时使用这些格式已有的推理算子。

作者的 transformer 权重预算目标为约 3.00 bpw，不含固定 PLE 大表；本地把所有非 PLE
张量的实存字节除以参数数量，得到约 2.992 bpw。专家部分单独为约 2.842 bpw。
目标值与这里的载荷口径分别记录，不把文件名解释成每个张量恰好 3 bit。

GGUF 元数据记录 `imatrix-qwen3.8-flash-next.gguf`、`calib_qwen3.8-flash-next.tokids`、
926 个条目、1000 个 chunks；v2/v4 记录的是另一套 imatrix 与语料。未取得这些校准文件内容，
不能根据 chunks 数量直接比较校准质量。本机没有复现 GSQ 优化过程或 RCO 搜索。

## 专家：主要省在 gate/up

三版均有 48 层、每层 512 个专家、每 token 激活 10 个专家；本地全部 1224 个张量名称和
形状对应一致。此文件保持完整专家数量，和另行发布的 256 专家 Coder 版不同。
gate/up 每张量形状为 `[2560, 640, 512]`，down 为 `[640, 2560, 512]`。

GSQ 的 gate/up 同层采用相同类型：

| gate/up 类型 | 层数 | 每个权重的实际存储位数 |
| --- | ---: | ---: |
| IQ2_XXS | 9 | 2.0625 |
| IQ2_XS | 10 | 2.3125 |
| IQ2_S | 10 | 2.5625 |
| IQ3_XXS | 6 | 3.0625 |
| IQ3_S | 13 | 3.4375 |

48 层中 **29 层 gate/up 使用 IQ2 系列**，其余 19 层使用 IQ3 系列，没有统一加位的前后层段。
down 则是 **30 层 Q2_0、18 层 IQ4_NL**。18 层为
`0, 2, 4, 28, 29, 30, 31, 32, 34, 35, 37, 38, 39, 42, 43, 44, 46, 47`。

down 每行宽度 640，无法整除 256，因此不能直接使用要求 256 元素整块的 Q6_K/IQ3_XXS
等格式。作者给它的搜索候选只有 Q2_0（64 元素块，2.25 bpw）和 IQ4_NL（32 元素块，
4.5 bpw）。这是该发布配方的候选集合，并不表示所有 GGUF 格式中只有这两种可表达该形状。
IQ3_KT 支持尾部布局，因此 v2/v4 可以在 down 上采用另一种精度选择。

与前面讨论的 v4 gate/up 升位直接对照，GSQ 的前 6 层如下：

| 层 | GSQ gate/up | GSQ down | v4 gate/up | v4 down |
| --- | --- | --- | --- | --- |
| 0、2、4 | IQ2_XS | IQ4_NL | IQ4_KT | IQ3_KT |
| 1、3 | IQ2_XXS | Q2_0 | IQ4_KT | IQ3_KT |
| 5 | IQ2_XS | Q2_0 | IQ4_KT | IQ3_KT |

第 38–47 层也混用 IQ2/IQ3 gate/up 和 Q2_0/IQ4_NL down。完整层号及类型见
[逐层分配表](../bench/results/2026-10-09-gsq-weight-allocation/expert-layers.md)。

相对 v4，专家共少 **7.1826 GiB**，其中 **6.6846 GiB 来自 gate/up**，down 只少
**0.4980 GiB**。GSQ 在部分 down 上用了更高位的表示，所以“很多 down 是 2-bit 格式”
并不意味着 down 总载荷也按同样比例降低。

## 稠密路径：小投影保留，大矩阵逐项分配

| 部分 | GSQ IQ3_XXS 存储类型 | v4 IQ3_KT 存储类型 |
| --- | --- | --- |
| HC 投影，含最后的 HC | BF16，共 290 个 | Q8_0 |
| SSM alpha/beta 小投影 | BF16，共 72 个 | Q8_0 |
| indexer Q/K 投影 | BF16，共 24 个 | BF16 |
| PLE key/value 投影 | BF16，共 2 个 | Q8_0 |
| 路由矩阵及 shared-expert gate | BF16 | F32，当前 Strata 打包为 BF16 |
| attention/GDN 大投影 | 按张量使用 IQ3_S、IQ4_XS、Q4_K、Q5_K、Q6_K | Q8_0 |
| shared-expert gate/up | 逐张量混合 IQ3_S 至 Q6_K | Q8_0 |
| shared-expert down | 40 层 IQ4_NL、8 层 Q2_0 | Q8_0 |
| token embedding | IQ3_S | Q8_0 |
| 输出头 | Q5_K | Q8_0 |
| PLE n-gram 大表 | IQ4_NL | IQ3_KT |

这里的 shared-expert gate 是 `ffn_gate_inp_shexp.weight`，与表中的 shared-expert gate/up
矩阵区分。规范化等 F32 张量和 PLE F16 卷积也保留，完整角色统计见
[analysis.json](../bench/results/2026-10-09-gsq-weight-allocation/analysis.json)。

大矩阵内部也存在明显区别：36 个 `attn_qkv` 中 14 个为 Q6_K，36 个 `ssm_out` 中 19 个
为 Q6_K，而 12 个全注意力 `attn_output` 中 10 个为 Q6_K；这些角色使用 Q6_K 的比例不同。
这体现了具体张量的预算取舍，不能据此证明这些类型对 Uncensored v4 同样最优。

此前方案一关注的 **387 个投影，在 GSQ 中全部为 BF16**。当前 Strata 对 v4 的这些 Q8_0
或 BF16 投影也使用 BF16 运行表示，因此保留其源精度不会按 GGUF 格式差异等比例增加运行
显存；它保留的信息量仍与把低位量化结果展开为 BF16 不同。

## 容量如何落到本机

以下固定为 **19＋29 分层**的 GPU 权重载荷，含阶段需要的额外 PLE 投影及路由权重。
不含 KV、状态、工作区、CUDA graph、驱动和预留，不等同于 NVML 显存读数。

| 项目 | v2 IQ3_KT | v4 IQ3_KT | GSQ IQ3_XXS |
| --- | ---: | ---: | ---: |
| 专家权重 | 44.41 GiB | 47.15 GiB | 39.97 GiB |
| 其他 GPU 权重 | 3.43 GiB | 5.04 GiB | 3.72 GiB |
| 合计 GPU 权重载荷 | 47.84 GiB | 52.19 GiB | 43.69 GiB |
| SSD 上的 PLE 大表 | 25.03 GiB | 20.27 GiB | 26.82 GiB |
| 两个 GGUF 文件合计 | 72.52 GiB | 72.52 GiB | 70.63 GiB |

GSQ 比 v4 的 GPU 权重载荷少 **8.50 GiB**，文件却只小约 **1.89 GiB**，因为其 PLE 表
比 v4 大约 **6.56 GiB**。作者固定 PLE 为 IQ4_NL，未纳入 RCO 搜索；本机该表放 SSD。
同一类型组合在 22＋26 下，GSQ 的两卡权重载荷约为 **17.45＋26.23 GiB**。

此前[同机性能测试](IQ3_XXS_VS_V4_2026-10-09.zh-CN.md)实际用 22＋26，GSQ 全部专家驻留
双卡，v4 仍有 4250 MiB 专家在 RAM。这是已有的运行证据；本次没有重测吞吐或质量。

## 对 v4 缩减的启示与边界

可借鉴的是：先保留关键小投影的精度，再让大矩阵和各层专家在明确预算下竞争精度；
gate/up 与 down 分开考虑，并把格式支持的行宽和实际 GPU 放置纳入预算。

GSQ 的低位系数经过其优化过程，RCO 的分配也依赖具体候选量化、校准数据和任务损失。
这些分配不能证明把 v4 对应张量直接降为同名格式就会保留同样质量。GSQ 使用原始 Qwen
模型，当前 v4 来自 Uncensored 源模型，也不能直接把两者的张量拼接当作同源量化消融。
作者对 GSQ 的质量报告以原始 BF16 为参照，本机未复现，不能与 v4 发布者的 KLD 数字直接排名。

本次第一分片的 **1223 个类型条目与作者分配清单完全匹配**；第二分片只有 PLE 表。
验证了 GGUF 数据边界与头部哈希，未计算整文件哈希，也未验证量化系数与远端逐字节相同。
[分析脚本](../bench/results/2026-10-09-gsq-weight-allocation/analyze.py)及
[作者分配清单副本](../bench/results/2026-10-09-gsq-weight-allocation/publisher-allocation.txt)供复核。
原始下载与核算记录在 `logs/gsq-weight-allocation-2026-10-09-172734/`。
