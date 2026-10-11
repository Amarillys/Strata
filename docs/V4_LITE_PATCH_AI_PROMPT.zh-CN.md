# 交给接收者 AI 的提示词

请帮我把本机已运行的 Qwen3.8-Flash-Next IQ3_KT v4 Strata 配置扩展为一个独立的 v4-lite 预设。
我已有原 v4 两个 GGUF 分片，没有 v2；补丁目录已经附带必要的 v2 权重与应用脚本。
请完成本机准备、检查、预设接入和简短生成验证，保留原 v4 作为回退。

目标是 **v4 稠密、输出头、共享专家和 387 个高精度小投影＋v2 IQ3_KT 专家＋原 v4 IQ3_KT PLE**。
我选择只传约 9.80 GiB 的专家补丁，不传约 25.03 GiB 的 v2 IQ4 PLE；原 v4 已有可用的 PLE。
先阅读本机 AGENTS.md 和补丁目录的 README.zh-CN.md，核对 git status 并保留已有修改。
优先使用补丁中的 v4_lite_patch.py，不需要重新实现量化器、下载完整 v2 或从 v4 重新量化。

先根据我提供的路径，或本机预设管理器的目录记录，找到补丁目录、活动 v4 配置及其 cwd、pack、
native GGUF、tokenizer、exe。只读取需要的字段，不输出 API key、整份配置或私人请求正文。
如果路径能从本机记录推断出来，继续处理即可。

补丁预期原 v4 的完整 SHA256：

```text
60a3b982267f316e1d3f88b1194afe66feef2c460a7890473251be763ca25b21  00001-of-00002.gguf
eed2a8880920f6b7f2311278198251f77b54e019a305d5ce5050cb05f01b754b  00002-of-00002.gguf
```

在补丁目录使用本机 Python 3.10+，替换下面的本机路径后运行：

```powershell
python .\v4_lite_patch.py apply --config "本机原v4配置.json" --out-pack "本机新pack目录" --out-config "本机新v4-lite配置.json"
```

脚本会自动完整校验 v4 与所选补丁权重，并检查本机 v4 稠密 pack；不要另加跳过检查的选项。
脚本默认 `--ple iq3`，也可以显式写上这个参数。**缺少 v2-ple-iq4kt.gguf 是这次交付的正常情况**，
请继续使用原 v4 的 IQ3_KT PLE，不要下载 PLE、不要禁用 PLE、不要自动改成 IQ4 PLE。
只有我以后明确要求并提供可选 PLE 文件时，才使用 `--ple iq4`。
如果原配置没有实际 cwd，补充 `--cwd`。补丁需要与 v4 在同一磁盘，相对路径不能含空格或逗号。

请核对这些实现要求：

- 32 个替换张量来自第 0–5、38–47 层的 gate/up，均为 v2 原始 IQ3_KT 字节。
- 另外 112 个专家张量已在发送端逐字节证明与 v2 相同，可以从经完整哈希验证的 v4 读取。
  专家索引应覆盖 48 层、144 个张量，合计 47,689,236,480 字节，所有角色为 type 154。
- 稠密 pack 保留 v4，包含 387 个高精度小投影。不要复制旧 experts.bin 来覆盖新的原生来源。
- `--native` 和输出头来源保持原 v4；显式重复两次 `--native-dense-gguf` 指定原 v4 两分片，
  防止 PLE 来源被自动加入稠密来源。
- `--ple-gguf` 指向原 v4 包含 per_layer_token_embd.weight 的分片，type 154、
  shape [160, 320001536]。原 v4 的 PLE key/value 小投影仍保留。
- 补丁 GGUF 是部分张量容器，不能当独立完整模型启动。所有原 v4 分片与选用的补丁文件都要保留。
- 引擎需要支持 IQ3_KT/IQ4_KT、native_experts.txt v4 的逐角色分片列及上述显式来源参数。
  若不支持，指出缺失点并按本机仓库指南处理，不要把检查失败当作安装成功。
- 如果 dense pack 哈希不符，先核对它是否确实由同款 v4 生成。可依据本机 tools/iq_pack.py
  的 --help，用 --gguf、--out、--compat-bf16 在新目录重建 v4 native pack，再重试。
  不要跳过校验，也不要从其他模型的 canonical pack 借用稠密权重。

应用脚本生成新配置后，请沿用我本机的 GPU、分层、显存预留、上下文、KV、缓存与 vision 设置。
不要照搬发送端的显卡编号、19＋29 分层或显存预留。生成的配置包含本机私有凭据，不要放入分享包。

接入本机已有的预设管理方式。若项目有 tools/switch_preset.py::Presets，就复用其加锁、归属检查、
drain/unload、启动、记录和恢复流程，保留原 launcher argv/cwd 与 GPU 唤醒参数。候选启动失败时，
恢复原 v4。不要按进程名批量杀服务，也不要覆盖运行中的 exe。

启用后检查实际运行 exe/参数、health.loaded、模型名称，做一次短文本生成。报告使用 IQ4 还是 IQ3 PLE、
32 个替换与 112 个复用专家的来源、是否保留 v4 稠密、配置位置及回退方式。只报告实际验证过的内容；
发送端的速度、NLL/KLD 与显存测量不能代替我的本机验证。
