# IQ3_KT text with Intel Iris Xe vision

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

This build enumerated unfiltered Vulkan devices as 2080 Ti = 0, 4080 SUPER = 1, Iris Xe = 2. The old IK
script's `GGML_VK_VISIBLE_DEVICES=1` would select the 4080 here. The vision section therefore uses:

```json
{
  "gpu": true,
  "device": "Intel(R) Iris(R) Xe Graphics",
  "max_tokens": 768,
  "env": {"GGML_VK_VISIBLE_DEVICES": "2", "CUDA_VISIBLE_DEVICES": "-1"}
}
```

`device` accepts a backend name (such as Vulkan0) or its exact description. Requiring the Intel description
also makes an ordinal mismatch fail before loading the projector. The helper logs its selected device.
The encoder-only environment does not change the text engine's two CUDA UUIDs.

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
  -DSTRATA_VISION_VULKAN=ON -DSTRATA_VISION_CUDA=OFF -DGGML_OPENMP=OFF
cmake --build build-vision-vulkan --config Release --target strata-vision -j 4
```

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

## Iris Xe attention diagnosis (2026-10-15)

A helper-only comparison on this machine (i9-12900HK, two 32 GiB DDR5-4800 DIMMs,
Intel driver 32.0.101.6559) used one generated 1024 x 1024 chart, the same projector,
and the unchanged 768 image-token cap. Every run produced 729 output tokens, width 2560.
These are encoder times after READY, not HTTP latency or text prefill times:

| Backend | Attention | Encoding time |
|---|---|---:|
| Iris Xe Vulkan | auto (enabled) | 11.545, 11.521 s |
| Iris Xe Vulkan | off | 6.440, 6.449 s |
| CPU, 8 threads | off | 21.665 s |

Disabling flash attention reduced encoder time by about 44% (about 1.79x throughput).
Vulkan timestamps placed about 8.00 of 11.41 seconds in 27 FLASH_ATTN_EXT operations,
with head dimension 72 and 2916 internal image patches. The merger reduces four patches
to each output token. The current Intel driver does not advertise VK_KHR_cooperative_matrix;
the backend source therefore selects its scalar flash-attention path on this device.
The graph has one backend split in both modes; this does not suggest repeated GPU/CPU
transfers as the source of the delay.

A separate helper linked against the existing libraries added stage timers and INFO logs.
With attention off, file decode took 2.727 ms, preprocessing 18.937 ms, and encoding
6428.515 ms. Vulkan compute buffers increased from 85.47 to 617.27 MiB; the CPU buffer
was 8.59 MiB in both modes. This additional allocation is on the iGPU's shared system
memory, not either NVIDIA card. Loading remained a separate startup cost.

The Python server now accepts an optional vision-only override:

```json
{"vision": {"flash_attn": "off"}}
```

Add the field to the existing vision object; do not replace its other fields. Accepted
values are `auto`, `on`, and `off`. Omitting it preserves the previous helper arguments
and backend defaults. The helper already supports the corresponding `--flash-attn` flag.
No running service or local preset was changed during this diagnosis.

All measured embeddings were finite, but attention modes are not numerically identical:
their flattened embedding cosine similarity was 0.999392, with about 3.49% relative L2
difference using auto as the reference. This chart comparison does not validate OCR or
grounding accuracy. Fourteen server argument and vision lifecycle tests passed.
Diagnostic files, generated chart, per-operation timings and stage-timer source are in
`build-kt/igpu-diagnosis-20261015/`; the production encoder executable was not replaced.

### Follow-up: explicit local override and FP16 experiment

Following the user's instruction, all three catalog presets (`v2`, `v4`, `v4-lite`)
now explicitly set `vision.flash_attn` to `off`. The active v4-lite service was unloaded
through its API while no requests were in flight, then restarted with its original
command-line arguments. The local `run-strata.ps1` currently references an absent
`tools/switch_preset.py`; the restart therefore reused the running launcher's arguments
and working directory rather than invoking the broken switch entry.

The original projector has 110 BF16 tensors and 224 F32 tensors despite its F16 filename.
A separate test copy converted the BF16 tensors to FP16 without changing dimensions or
file size. There were no overflows; the maximum absolute weight conversion error was
2.98e-8, with 22,445 values rounding to zero out of 444,100,608 converted values.
With attention off this copy encoded the same chart in 5.813 and 5.817 seconds,
about 10% less time than the original weights. Its embedding cosine similarity to the
original off-mode result was 0.993637, with 11.27% relative L2 difference. The Vulkan
backend selects different accumulation paths for BF16 and FP16 weights, so weight
conversion error alone is not an accuracy check. No preset selects this experimental
copy; OCR/grounding quality has not been validated for it.

The restarted service reported v4-lite loaded with vision available, and its actual
encoder command line contained `--flash-attn off`. A generated 1024 x 1024 OCR/color
chart returned `TEST 421` and `Red, Green, Blue` in 10.051 s for the whole HTTP request
(759 prompt tokens, 14 generated tokens). Repeating the identical request returned
the same answer in 0.412 s with 752 prompt tokens cached. This is a simple smoke test,
not a dense-document quality evaluation or an isolated encode timing. The live service
was left running with the original projector and the attention override.

Helper-only token-cap tests with the original projector and attention off used the
same generated chart as the initial benchmark:

| Maximum image tokens | Actual output tokens | Encoding time |
|---|---:|---:|
| 768 | 729 | 6.440, 6.449 s |
| 512 | 484 | 3.679, 3.701 s |
| 256 | 256 | 1.534, 1.539 s |

Lower caps change the image detail the model receives. No OCR/grounding accuracy
comparison was run for these caps, and all local presets retain the 768 cap.
Evidence: `caps-results.json`, `http-fa-off.json`, `fp16-results.json`,
`fp16-conversion.json`, and `fp16-numerics.json` in the diagnostic directory.

A further FP16-copy run with `GGML_VK_DISABLE_F16=1` and attention off took 8.300
and 8.305 seconds. This disables FP16 arithmetic across the Vulkan backend rather
than changing only matrix accumulation precision, so it is not an isolated test of
FP16 weights with FP32 accumulation. It provided no speed benefit and was not enabled
in any preset. Evidence: `fp16-f32acc-results.json` and `fp16-f32acc.log` in the same
diagnostic directory (the filenames describe the experimental intent).

### Current local settings: 640-token cap (2026-10-15)

At the user's request, all three catalog presets now use `vision.flash_attn: "off"`
and `vision.max_tokens: 640`, superseding the 768 cap above. The original projector
is retained. The active v4-lite service was unloaded while idle and restarted with
its original launcher arguments; the live encoder command line was checked for
`--flash-attn off --max-tokens 640` on Intel Iris Xe.

The same HTTP OCR/color chart correctly returned TEST 421 and Red, Green, Blue.
The first whole request took 8.896 s (655 prompt tokens, 15 completion tokens), and
an identical repeat took 0.441 s with 648 prompt tokens cached. These are HTTP times,
not isolated encoder measurements. The service was left running and available.
Evidence: `build-kt/igpu-diagnosis-20261015/http-fa-off-640.json`.

### Current local settings: encoder on the 2080 Ti with weights in RAM (2026-10-15)

This supersedes the Iris Xe settings above for the v4-lite preset
(`strata-iq3kt-v4-lite-k8v8-pf2048-vision-igpu.json`); the v2 and v4 presets still
use the Iris Xe. The vision helper is now built with `-DSTRATA_VISION_WEIGHTS_HOST=ON`
(`build-vision-vulkan`; see `docs/DETAILS.md`), and the preset changed in four values:
`vision.device` `"NVIDIA GeForce RTX 2080 Ti"`, `vision.flash_attn` `"auto"`,
`vision.env.MTMD_WEIGHTS_HOST` `"cpu"`, and `--vram-reserve-mib` 960 -> 840. The
640-token cap is unchanged. Backup of the previous preset and the wrapper's command line:
`logs/vision-2080ti-20261015/` (the backup contains the API key).

Helper-only, same generated 1024 x 1024 chart, 625 image tokens: Iris Xe with attention
off took 5.29 s; the 2080 Ti took 0.22 s with the weights in VRAM (+963 MiB) and
0.43-0.46 s with them in RAM (+113 MiB), with bit-identical embeddings.

After the restart the engine kept all 9,728 of the 2080 Ti's experts (zero-doorbell
graph) and 14,428 on the 4080, as before; 732 MiB of VRAM were free on the 2080 Ti
with everything loaded (802 before). Live HTTP requests, temperature 0, thinking off:

| Request | Before (Iris Xe) | Now |
|---|---:|---:|
| New 1024 x 1024 chart, 655 prompt tokens | 8.896 s | 3.967 s, 3.570 s (two charts) |
| Same chart again | 0.441 s | 0.400 s |

Both charts were read correctly. Their prompt prefill took 3.05 and 2.78 s, now most of
the request. One text request (39 prompt tokens, 512 generated) decoded at 51.7 tok/s
with a 99.5% hit rate; this is one sample, not a matched before/after decode benchmark.
The encoder's first start with the newly linked helper spent 9.3 s compiling NVIDIA
shaders; later starts took about 2 s in helper-only tests.

The wrapper now runs from the scheduled task `Strata v4-lite server` (interactive
session, highest privileges, no time limit), which runs
`build-kt/vision-opt-20261015/start-server.ps1` with the previous command line and
working directory. To go back: restore `logs/vision-2080ti-20261015/config.before.json`
over the preset and restart the service. Evidence: `build-kt/vision-opt-20261015/`
(`README.md`, `device-results.json`, `restart.log`, `verify-2080ti.json`).

### Prefill follow-up on the same service (2026-10-15)

One restart per arm through the scheduled task, bracketed by the current preset before and after; prompts carry
unique nonces (no cached prefix); prompt-read seconds from the API timings, two requests each:

| Arm | 655-token image prompt | 2.6K | 4.6K | 13.7K | Decode tok/s (3 x 512) |
|---|---:|---:|---:|---:|---:|
| current preset (before) | 2.76, 2.76 | 6.53, 5.10 | 7.04, 7.05 | 15.54, 15.51 | 55.0 / 54.5 / 54.7 |
| `STRATA_PREFILL_HELP=1` | 2.98, 2.79 | 6.63, 5.24 | 7.24, 7.11 | 15.88, 15.63 | 54.8 / 54.3 / 55.0 |
| `--prefill 1024` | 2.97, 2.76 | 6.79, 5.90 | 9.07, 9.54 | 24.10, 23.87 | 55.3 / 54.7 / 54.6 |
| both | 2.92, 2.77 | 6.68, 6.10 | 9.21, 9.46 | 23.98, 23.94 | 55.1 / 54.8 / 54.6 |
| current preset (after) | 2.95, 2.78 | 6.44, 5.13 | 7.22, 7.08 | 15.83, 15.86 | 54.9 / 54.5 / 54.9 |

`STRATA_PREFILL_HELP` changed nothing measurable (its peer only computes MMQ rows; the KT prompt path expands experts
to FP16). `--prefill 1024` left the 4080's session buffers and its 14,428 experts unchanged (only the prompt loan
fell from 744 to 537 slots per card) and slowed 4.6K-13.7K prompts by 30-55%. Neither was kept; the preset was
restored byte for byte. One of ten chart answers gave only the number (`TEST 905`, colors omitted), reproducibly.

`STRATA_PREFILL_TIMING=1` (one extra restart) shows why a 655-token image prompt costs about as much as a 2,048-token
chunk: in one chunk nearly every expert of every layer is routed, and the KT prompt path expands each routed expert
to FP16 before its products. The 648-token image chunk took 1.03 s of GPU timeline on the 2080 Ti (dequant 318 ms)
and 1.99 s on the 4080 (dequant 792 ms, gate/up 649 ms, down 415 ms); a full 2,048-token chunk took about 2.5 s on
the 4080 (dequant ~900 ms). ik_llama.cpp has IQ3_KT MMQ kernels including the grouped-expert (`_id`) variant;
porting them is the remaining step `docs/IQ_KT_PREFILL.md` names, not done here.

Cost of a larger image-token cap, measured without changing the service: text prompts of 630 / 745 / 1,199 tokens
read in 2.32-2.36 / 2.40-2.55 / 3.24-3.51 s (unique nonces, three each, first 630-token outlier 3.29 s excluded); the
encoder with weights in RAM took 0.44 / 0.47 / 0.67-0.75 s at 625 / 729 / 1,024 image tokens and held 113 / 128 /
176 MiB on the 2080 Ti. A 1,024 cap would need `--vram-reserve-mib` lowered by another ~60 MiB to keep all 9,728
experts. Evidence: `build-kt/vision-opt-20261015/prefill-ab/` and `device-results.json`.

### Current local settings: 768-token cap (2026-10-15, 12:40)

At the user's request the v4-lite preset's `vision.max_tokens` is 768 again (backup of the 640 preset:
`logs/vision-2080ti-20261015/config.before-768.json`); `--vram-reserve-mib` stays 840. After the restart the
2080 Ti still holds all 9,728 experts (zero-doorbell graph), 14,428 on the 4080, 784 MiB free on the 2080 Ti.
Fresh 1024 x 1024 charts (759 prompt tokens) took 3.46-3.91 s per whole request (prompt read 2.86-3.10 s).
With the A/B prompt ("Read the TEST number and list rectangle colors ... Answer briefly.") every number was read
correctly, but 3 of 8 answers gave only the number; asked directly for the colors, 3 of 3 answers were correct.
At 640 the same prompt gave the number only in 1 of 10. These samples are too small to attribute the difference
to the cap. Evidence: `build-kt/vision-opt-20261015/prefill-ab/cap768*.log`.

### Current local settings: working-branch engine with KT MMQ (2026-10-15, 14:51)

At the user's request production now runs the working branch's engine (feat/ik-support-with-vulkan-vision at
2f10e4f7, copied to `build-kt/prod-2f10e4f7/strata.exe`) instead of `build-kt/pool-candidate/strata.exe`, which came
from the feat/iqkt-vulkan-vision line. The v4-lite preset changed in three places: that `exe`,
`env.STRATA_PREFILL_KT_MMQ=1` (docs/IQ_KT_PREFILL.md), and no `--conversation-pool-tokens 524288` (the working branch
has no conversation pool; the conversation cache settings are unchanged). Backup of the previous preset:
`logs/vision-2080ti-20261015/config.before-kt-mmq.json`.

After the restart: all 9,728 experts on the 2080 Ti (zero-doorbell graph), 782 MiB free there. A fresh chart was
read completely (TEST number and three colors) with a 2.15 s prompt read; 13,810-token needle prompts read in 13.0 s
(the first long prompt after the start), then 9.95 and 9.60 s, with all hidden numbers returned.
Evidence: `build-kt/vision-opt-20261015/prefill-ab/deploy-kt-mmq.log`.
