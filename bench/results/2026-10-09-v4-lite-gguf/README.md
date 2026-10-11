# v4-lite with a separate IQ4_KT PLE GGUF

The export carries the existing v4-lite composition in two complete GGUF shards. The first holds all
v4 non-expert tensors and all 144 v2 IQ3_KT routed-expert tensors. The second holds only the v2 IQ4_KT
`per_layer_token_embd.weight`. No original v2/v4 shards or expert patch are required at runtime.

| File suffix | Tensors | Bytes | GiB |
| --- | ---: | ---: | ---: |
| `00001-of-00002.gguf` | 1,223 | 53,168,222,432 | 49.52 |
| `00002-of-00002.gguf` | 1 | 26,880,129,376 | 25.03 |
| Total | 1,224 | 80,048,351,808 | 74.55 |

The files are in `B:\models\llm\Qwen\Qwen3.8-Flash-Next\v4-lite-iq4ple-gguf-20261009`.
The export manifest records each tensor's source, shape, type, byte length, output offset and payload SHA256.
All tensor payloads and both complete files passed a read-back hash check. Metadata from v4, including
tokenizer and chat template, retains its original encoding except the name, split fields and composition note.

`reader_probe.cpp` uses Strata's real `GgufModel` without GPU initialization. It verified the two shards,
architecture, all tensor bounds, IQ3_KT experts in the first shard, and the sole IQ4_KT PLE tensor in the second.
The result is in [cpp-reader-validation.json](cpp-reader-validation.json).

`verify_portable_pack.py` generated tokenizer and pack data with a Python audit hook permitting only the two
new GGUFs and forbidding child processes. It used the existing local llama.cpp `gguf-py` dependency through
`STRATA_GGUF_PY`. An initial attempt without this path stopped because the dependency was not discoverable;
the same exported GGUFs passed once the existing dependency path was supplied.

The generated `dense.bin` has SHA256
`5b897f416728fcba4a24c8937a7fad71299d026c6c86dc6c77c922dad22cf7b0`, identical to the existing v4 pack.
Normalized index rows and all five text tokenizer files match v4. All 48 expert-index rows use IQ3_KT,
refer only to the new first shard, and match its exported tensor offsets and byte lengths.
The native pack is 1,496,017,231 bytes; no `experts.bin` is needed.
Details are in [portable-pack-validation.json](portable-pack-validation.json).

The exporter tests passed for metadata/payload preservation, corrupt-payload rejection and existing-file
protection. The new files were not loaded into the production GPU service, and no target-computer inference
or performance measurement was made. Prior VRAM measurements refer to the same tensor composition using
the original files; they do not establish a minimum VRAM requirement for another computer.

See [the deployment guide](../../../docs/V4_LITE_STANDALONE_GGUF.zh-CN.md) for the model paths, pack command,
disk-size versus VRAM accounting and target-machine instructions.
