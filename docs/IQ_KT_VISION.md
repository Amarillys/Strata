# IQ3_KT text with Intel Iris Xe vision

The [work summary (中文)](IQ_KT_SUMMARY.zh-CN.md) records the full implementation, performance analysis,
KV/ngram placement and validation limits. This page describes the current local launch configuration.

Verified on 2026-10-08: the existing F16 Qwen3.8-Flash-Next MMPROJ runs in a separate Vulkan process on
Intel Iris Xe. Strata reads its image embeddings using the existing GENI/M-RoPE path. The text model uses
2080 Ti + 4080 SUPER, layer split 19, K8V8, capacity 262144, 20480 resident KV cells, prefill 2048, and MTP off.

The local preset is `strata-iq3kt-dual19-int8-pf2048-vision-igpu.json`. It copies the text preset and adds
`--vision`, the encoder configuration, and a log file. Start it from PowerShell:

The machine-specific preset, `run-strata-vision-igpu.ps1`, model data and API key are local files excluded
from Git. The encoder, server support and `tools/serve_gpu_wake.py` are committed source; the wake wrapper
requires the separately installed `gpu_monitor.py` provider named by `--power-module-dir`.

```powershell
& B:/llama.cpp/Strata/run-strata-vision-igpu.ps1
# Optional different port:
& B:/llama.cpp/Strata/run-strata-vision-igpu.ps1 -Port 8097
```

Default browser address: `http://127.0.0.1:8080/`; OpenAI base URL: `http://127.0.0.1:8080/v1`.
The launcher now binds `0.0.0.0`, so this PC is also reachable at `http://10.0.0.3:8080/` on the LAN.
API requests require the key in the local preset's `api_key` field (also saved in `strata-vision-api-key.txt`).
Enter that key in the web client's API-key setting and in the harness. This follows the repository's LAN
setup requirement; a running server needs a restart after changing its bind address or authentication.
The script starts a separate Strata service. It does not run or change the existing IK server or proxy.
Ctrl+C stops the server, encoder and text engine and releases the temporary clock locks.
The verification services were stopped after testing.

## Device selection and GPU wake

Vulkan ordinals can change between launches. During the 0.1.40.3 merge check on 2026-10-08, Iris Xe was
listed at ordinal 1 and later at ordinal 2. A numeric filter from the earlier enumeration hid the Intel
device and prevented startup. The local preset now relies on the exact device description, with no numeric
`GGML_VK_VISIBLE_DEVICES` filter:

```json
{
  "gpu": true,
  "device": "Intel(R) Iris(R) Xe Graphics",
  "max_tokens": 768,
  "env": {"CUDA_VISIBLE_DEVICES": "-1"}
}
```

`device` accepts a backend name (such as Vulkan0) or its exact description. The helper logs the selected
device and passes it explicitly to the projector. The local launcher clears any inherited
`GGML_VK_VISIBLE_DEVICES` for its child process and restores the shell's previous value on exit. When
starting the server directly, unset that variable too: a description cannot select a device hidden by a
filter. The encoder-only CUDA setting does not change the text engine's two CUDA UUIDs.

The unfiltered, exact-name check loaded and warmed the new helper in 13.46 s and encoded a 512 chart in
2.945 s. Its 256 × 2560 FP32 embedding was finite. The two NVIDIA cards' memory use remained unchanged
before, after warm-up and after encoding; enumerating them did not put the projector on them.

The launcher uses `tools/serve_gpu_wake.py` and the existing `B:/llama.cpp/proxy/gpu_monitor.py` class.
It creates the retained Torch wake probe on the 4080 **after** engine READY, pulses before each generation,
and releases clocks after 120 seconds without a generation or on normal shutdown. The probe includes a
CUDA/Torch context; its total VRAM footprint is larger than its 512 KiB tensor. Starting the Python server
directly with the JSON preset bypasses this wake wrapper.

## Build and tokenizer compatibility

The vision helper is Vulkan-only, independently built against the same pinned upstream source as the
text engine (`Strata-ggml-kt`, commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`):

```powershell
cmake -S tools/vision -B build-vision-vulkan -G 'Visual Studio 17 2022' -A x64 `
  -DLLAMA_DIR=B:/llama.cpp/Strata-ggml-kt `
  -DSTRATA_VISION_VULKAN=ON -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=ON -DGGML_OPENMP=OFF
cmake --build build-vision-vulkan --config Release --target strata-vision -j 4
```

The merged upstream vision build defaults to `STRATA_PORTABLE=ON`; the explicit setting above keeps
the tested configuration clear. The main CUDA engine still uses its own non-portable build configuration.

The existing projector is
`B:/models/llm/Qwen/Qwen3.8-Flash-Next/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf` (907543296 bytes).
No conversion of its weights was needed.

Upstream GGUF rejects IK tensor types 154/155 even for vocab-only loading. `tools/vision_vocab.py` therefore
copies the original first shard's raw metadata into `pack/iq3kt/tokenizer/vision-vocab.gguf`, omits all tensors,
and drops `split.*` so the loader does not follow weight shards. All 60 retained metadata values matched
the original model exactly. The result is 10945312 bytes. The source model is read-only; the exporter refuses
to overwrite existing files. Recreate a missing vocabulary file with:

```powershell
python -B tools/vision_vocab.py `
  --model B:/models/llm/Qwen/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-Uncensored-ik_llama-IQ3_KT_v2-00001-of-00002.gguf `
  --output pack/iq3kt/tokenizer/vision-vocab.gguf
```

To inspect device numbering, remove `GGML_VK_VISIBLE_DEVICES` in that shell before running
`build-vision-vulkan/bin/Release/strata-vision.exe --list-devices`.

## Measurements

These are the original pre-merge measurements. The [0.1.40.3 merge report](IQ_KT_UPSTREAM_MERGE.zh-CN.md)
records the later build and HTTP revalidation.

Helper-only runs, with finite FP32 embeddings and verified SVE dimensions:

| Input | Image tokens | Encoding time |
|---|---:|---:|
| 512 × 512 chart | 256 | 3.792 s |
| 1024 × 1024 chart | 729 | 11.501 s |
| Same 1024 chart, encoded again without API caching | 729 | 11.503 s |

The 768 cap is per image. The square grid rounds the 1024 image to 27 × 27 = 729 tokens; it does not
pad every image to 768. Helper loading and warm-up took 16.1 s in this run. NVIDIA memory readings stayed
at 307 MiB (2080) and 31 MiB (4080) before, during and after the helper-only test.

Full HTTP streaming requests, temperature 0, thinking disabled, no prompt cache, no MTP:

| Request | Vision encode | Text prefill | Whole request | Result |
|---|---:|---:|---:|---|
| 512 chart | 3.131 s | 2.753 s | 6.155 s | TEST 421; Red, Green, Blue |
| 1024 chart | 11.507 s | 3.014 s | 14.762 s | TEST 421; Red, Green, Blue |
| Repeat identical 1024 chart | cache hit | 2.837 s | 3.061 s | Same answer |
| Two images, first cached and second new | 11.515 s for new image | 3.329 s | 14.985 s | Correct Run numbers: 1, 2 |

The repeated-image result reuses the vision embeddings, while text prefill still runs. The image requests
contain only 288–1022 total prompt tokens, so their prefill throughput should not be compared with the
earlier multi-thousand-token text benchmarks. The added latency for a fresh 1024 image is mostly its
11.5-second iGPU encode. These simple charts validate routing, OCR, colors and multiple images; they do
not establish accuracy on dense documents or grounding tasks at the 768-token cap.

A text request after all image tests generated 204 tokens at 59.3 tok/s by the engine/API timing, with
0.409 s prefill and 0.437 s to first text. This is one measured sample, not a matched vision-off speed
comparison. Engine INFO reported 24576 expert slots (9728 primary + 14848 secondary), `kv=int8`,
`context=262144`, and `mtp_max=0`. Startup confirmed 100% expert residency. The vision path borrows 719
expert slots per card during prefill and restores them for decode.

Validation included the new metadata-preservation and GPU-wake lifecycle tests, existing and extended
vision environment/argument/image/cache tests, helper build/device/embedding checks, and five real HTTP
requests. A deliberately wrong Vulkan ordinal was rejected when the preset required the Intel description.
The final description-based preset also passed a real encoder request through `serve.server.Vision`.

Local evidence: `build-kt/vision-encoder.json`, `vision-encoder.log`, `vision-http.json`,
`vision-http-test.log`, `vision-final-device-test.log`, and `strata-vision-igpu.log`.
