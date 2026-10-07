# IQ3_KT and IQ4_KT (local CUDA implementation)

The optimized KT kernels measured 58-63 decode tok/s on the local 2080 Ti + 4080 SUPER configuration,
with all experts resident and MTP off. See [the sustained comparisons](IQ_KT_PERFORMANCE.md) and
[prefill measurements](IQ_KT_PREFILL.md) for workloads, numerical checks and limits. The actual 260K input
check establishes capacity and retrieval on that test, not sustained full-context decode speed or broad quality.
The early 0.79 to 7.51 tok/s result below compares CPU adapters; it is not a gain over IK.

Validation covers Windows CUDA (sm75/sm89) and x86 AVX2. HIP execution has not been validated.

The engine can read IK's IQ3_KT (type 154) and IQ4_KT (155) weights without changing its upstream ggml enum or
dependency. These are row formats: each row has a float scale, partial 256-element blocks have a separate layout,
and rows are padded to four bytes. Treating them as ordinary fixed-size blocks loses the row scales and tails.

Implemented paths: Python/C++ GGUF sizes, native dense and embedding reads, CUDA Q8_1 matrix-vector products,
grouped routed experts, CPU Q8_0 expert fallback, PLE's 160-wide rows, and prefill through row decoding to FP16/BF16
and the existing matrix multiplication path. Routed gate/up and down must both use KT formats; their KT types may
differ. The CPU fallback supports this model's widths up to 2560/640.

`tools/mtp_ik_rt.py` prepares a single Qwen4Exp MTP layer with 512 experts and widths 2560/640. Experts retain their
original KT bytes. The existing MTP runtime requires dense Q8_0/BF16 projections, so these are converted and listed
in `conversions.json`; IK's F32 norms are preserved without adding another Gemma +1. The combined `eh_proj` is split
into embedding and hidden projections. The drafter uses the main model's embedding/output head. `experts.kt` is a
completion marker containing version, gate/up type, down type, embedding width, FF width, expert count and blob
size. The loader checks these against the model and checks the expert file size before reading it. Older engines
cannot load this sidecar because it intentionally has no legacy `experts.bin`.

## Reproduce on this Windows workspace

PowerShell, from `B:/llama.cpp/Strata`. The upstream ggml worktree is at
`3cf03257f219afbe7334045ff7c6a06ac68c627d`; KT reconstruction follows IK
`c68d4e3c85ba45f713187f0932b42bf6b34e79b8` (MIT). The configured compiler is CUDA 12.8 and Visual Studio 2022.

```powershell
$env:STRATA_GGUF_PY = 'B:/llama.cpp/Strata-ggml-kt/gguf-py'
$env:PYTHONUTF8 = '1'
$env:PATH = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin;' + $env:PATH
$model = 'B:/models/llm/Qwen/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-Uncensored-ik_llama-IQ3_KT_v2-00001-of-00002.gguf'
$mtpModel = 'B:/models/llm/Qwen/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-Uncensored-MTP-ik_llama-IQ4_KT.gguf'
cmake -S . -B build-kt -G 'Visual Studio 17 2022' -A x64 -DSTRATA_ENABLE_CUDA=ON '-DCMAKE_CUDA_ARCHITECTURES=75;89' -DSTRATA_GGML_DIR=B:/llama.cpp/Strata-ggml-kt -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
cmake --build build-kt --config Release --target strata kt_parity kt_cpu_parity -j 4
python -B tools/iq_pack.py --gguf $model --out pack/iq3kt --compat-bf16
python -B tools/mtp_ik_rt.py --gguf $mtpModel --out pack/mtp-iq4kt
```

`Shared` avoids linking both CUDA runtime variants on this toolchain. The main pack keeps experts and PLE in the
original GGUF shards. Only its required small BF16 projections are converted; the current pack records 388 rounded
tensors (1.31 GiB). No source model is rewritten.

```powershell
$oracle = 'B:/llama.cpp/stop-fix-20260930/bin/ggml.dll'
$cudaBin = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin'
$shard2 = $model.Replace('00001-of-00002', '00002-of-00002')
python -B tools/test_kt.py --oracle-dll $oracle --dll-dir $cudaBin --fixtures build-kt/kt.bin --gguf $model $shard2 $mtpModel
build-kt/Release/kt_parity.exe build-kt/kt.bin 0
build-kt/Release/kt_parity.exe build-kt/kt.bin 1
build-kt/Release/kt_cpu_parity.exe build-kt/kt.bin $model 0
build-kt/Release/kt_cpu_parity.exe build-kt/kt.bin $mtpModel 48
python -B tools/test_mtp_ik_rt.py --gguf $mtpModel --rt pack/mtp-iq4kt --oracle-dll $oracle --dll-dir $cudaBin
python -B tools/test_iq_pack.py
```

CUDA device ordinals can differ from `nvidia-smi` indices. For a single-card run use its UUID from
`nvidia-smi --query-gpu=index,name,uuid --format=csv` as `CUDA_VISIBLE_DEVICES`. With a token-ID prompt file:

```powershell
build-kt/Release/strata.exe --pack pack/iq3kt --native $model --tokens-file build-kt/prompt.txt --spec 3 --mtp pack/mtp-iq4kt --prefill 32 --max-context 256 --max-new 24 --expert-cache 2048 --expert-profile data/expert-profile.bin --resident-experts --check-logits --stats
```

## Validation on 2026-10-07

- 37 random/real row cases and 829 real tensor spans against the existing IK DLL. Maximum relative decode error:
  1.73e-7. Widths include 32, 160, 320, 640, 2560, 6144 and 10240.
- Both RTX 2080 Ti (sm75) and RTX 4080 SUPER (sm89) passed CPU/GPU decoding, Q8 activation GEMV, grouped experts
  with repeated tokens and permuted destinations, CPU expert fallback, PLE dispatch, embedding gathers, padded
  FP16 output, BF16 output and interleaved gate/up prefill output. Maximum GEMV relative error: 4.70e-7.
- MTP audit: sample rows from all 26 dense tensors against IK, maximum relative conversion error 0.00393;
  F32 norms unchanged; first/middle/last complete expert blobs byte-exact with the source GGUF.
- Main pack regression suite: 24 tests passed.
- Each card separately completed a 26-token counting prompt and 24 generated tokens with PLE and MTP enabled.
  Output token IDs were identical between cards. Each accepted 16/16 draft tokens on this simple prompt.
  Local logs: `build-kt/smoke-2080.log`, `build-kt/smoke-4080.log`, `build-kt/mtp-audit.log`.

The deliberately small cache held 2048 experts (3.70 GiB), with about 40.71 GiB in locked system RAM. These are
compatibility smoke runs, not tuned throughput benchmarks:

| GPU | Measured H2D | Prefill, 25 batched tokens | Decode, 24 tokens | MTP draft time |
| --- | ---: | ---: | ---: | ---: |
| RTX 2080 Ti 22 GiB | 6.3 GB/s | 9.80 tok/s | 0.79 tok/s | 4.116 ms/round |
| RTX 4080 SUPER 32 GiB | 2.9 GB/s | 7.53 tok/s | 0.80 tok/s | 2.747 ms/round |

## CPU optimization and dual-GPU checks

The CPU adapter now uses shared trellis tables (512 KiB IQ3, 256 KiB IQ4), AVX2 signed integer dots, and one decode
per weight group for all tokens. Both operands are widened to int16 before multiplication, so -128 activations
are handled without saturation or sign overflow. AVX2/F16C availability is checked before entry; the original
scalar path remains available on older CPUs and with `STRATA_KT_SCALAR=1`.

`kt_cpu_parity` checked all 37 fixtures at 1..9 tokens, including row ranges, tails and extreme int8 activations:
gate/up/down outputs exactly matched the scalar arithmetic. A single-thread microbenchmark on 16 real experts
on the Core i9-12900HK measured the following gate/up plus down times per expert (fixed Q8_0 activations, excluding
their quantization; the thread was not pinned, so use the full-model comparison for end-to-end gains):

| Format | Tokens per expert | Scalar | AVX2 |
| --- | ---: | ---: | ---: |
| IQ3_KT, main layer 0 | 1 | 25.972 ms | 1.552 ms |
| IQ3_KT, main layer 0 | 3 | 50.980 ms | 2.279 ms |
| IQ3_KT, main layer 0 | 8 | 98.613 ms | 4.346 ms |
| IQ4_KT, MTP layer 48 | 1 | 9.110 ms | 3.253 ms |
| IQ4_KT, MTP layer 48 | 3 | 26.790 ms | 3.732 ms |
| IQ4_KT, MTP layer 48 | 8 | 72.361 ms | 5.576 ms |

The same 2080 Ti / 2048-slot counting smoke improved from 0.79 to 7.51 decode tok/s, with identical output tokens
(`build-kt/smoke-2080-avx2.log`). The previous runs had no GPU streaming of misses because CUDA refused to pin the
40.71 GiB RAM complement; locking it into the Windows working set does not make it CUDA-addressable.

A dual-GPU run assigned layers 0..19 to the 2080 Ti, layers 20..47 and MTP to the 4080 SUPER, with
`--trim-stage-weights --expert-cache auto --vram-reserve-mib 1536 --max-context 262144 --kv int8 --kv-resident 20480`.
It cached 9721 + 13796 experts (42.50 GiB total), leaving 1.91 GiB of experts in RAM, successfully CUDA-pinned.
With `--prefill 512`, the counting prompt decoded at 31.0 tok/s (24 tokens, 16/16 drafts accepted); a Python
sum-of-even-numbers request decoded at 32.1 tok/s (30 tokens, 20/22 drafts accepted) and returned a correct function.
These are short requests with 256K capacity configured. Logs/config/results: `build-kt/dual-256k-*`.

### Actual 260,000-token input

The same 20-layer split with `--prefill 2048` completed a 260,000-token synthetic retrieval prompt with no cached
prefix. It contained a repeated neutral log sentence and three values near the start, token 130,000 and the end.
The model correctly returned `cedar, cobalt, quartz` and EOS. Prefill took 438.283 seconds (593.2 tok/s); its short
8-token reply took 336.1 ms (23.8 tok/s), accepting 6/6 MTP drafts. The host-pool decode cache counter reported
97.4% hits (this counter is not a whole-model hit rate for a layer split; see the performance comparison). KV streaming
reported 66.07% of 25,675 block reads in VRAM and 35.1 MiB read from RAM. The run used the original PLE table.

This validates an actual near-capacity input, KV streaming and MTP on the two cards. A repeated synthetic prompt and
an eight-token answer do not establish general long-context quality or sustained decode speed. The input,
configuration, protocol and result are in `build-kt/prompt-260k.txt` and `build-kt/dual-260k-*`.

### Moving all 4080 experts into VRAM

Changing the split to 21 assigned layers 0..20 to the 2080 Ti and 21..47 plus MTP to the 4080 SUPER. Its 13,824
experts all fit in the 4080's cache; the 2080 cached 9,687 experts, with the remaining 1.92 GiB successfully pinned
in RAM. The short counting/code outputs matched the 20-layer split. Decode was 32.6/29.7 tok/s respectively,
versus 31.0/32.1 at split 20: these small samples show no consistent speed advantage. The 21-layer split concentrates
uncached expert work on the faster host link, but it also assigns another layer to the slower GPU. KV still streams
on both cards. The 260,000-token test used split 20; split 21 has only been tested with short requests.

The local server configurations are `strata-iq3kt-dual.json` (split 20, tested at 260K input) and
`strata-iq3kt-dual21.json` (split 21). Both use a 262,144-token capacity, INT8 KV with 20,480 resident cells, 2048-token
prefill chunks, native KT MTP, and the two GPUs in UUID order. The frontend/server is the existing Strata server;
these tests used the engine's stdio protocol. From the repository directory:

```powershell
$env:PYTHONUTF8 = '1'
python -m serve.server --engine strata --config strata-iq3kt-dual.json --host 127.0.0.1 --port 8080
```

Use the `dual21` config to compare the alternate split. The benchmark processes exited after their requests; no
server is left running by the tests. Local configs, model packs and logs are generated artifacts, not source files.

Subsequent tuning on 2026-10-08 produced `strata-iq3kt-dual19-k8v4-256k.json`: MTP off, layers 0..18 on the 2080 Ti,
19..47 on the 4080 SUPER, all experts resident, K8V4 KV and 2048-token prefill. It passed the actual 260K retrieval
input with 669.8 tok/s prefill before the packed CUDA change. Subsequent packed KT row-dot kernels improve the
normal short/3.6K decode comparison from about 39 tok/s to 62-63 tok/s, with identical tested output tokens. See
[the measured comparison and wake-up requirement](IQ_KT_PERFORMANCE.md) and
[the decode stage breakdown](IQ_KT_DECODE_PROFILE.md). The earlier MTP smoke results above are retained as
compatibility evidence, not the recommended performance comparison.

## Remaining work

CUDA decode uses Q8_1 activations and CPU uses Q8_0, so arithmetic is not bit-identical to IK's Q8_2_X4 contract.
The KT CUDA adapter now generates packed codes directly for DP4A, with separate IQ3_KT/IQ4_KT kernels and
aligned seed/sign loads. It still repeats reconstruction for each token in a verification window; it does not
yet share reconstructed weights across tokens.
KT prefill currently decodes rows before matrix multiplication; dedicated quantized MMQ and fused KT prefill kernels
have not been implemented. Its decoder now reconstructs eight adjacent values per lane with vector output stores;
see the [prefill profile, numerical checks and measured gains](IQ_KT_PREFILL.md). The tested long-input preset is
`strata-iq3kt-dual19-int8-pf4096.json`: K8V8 and 4096-token chunks, actual 260K-input retrieval in 198.1 seconds
(1,312.3 tok/s prefill), keeping the full 262,144 capacity.

Long-form quality/acceptance and sustained performance require broader validation. HIP/SYCL builds and the
installer/model catalog have not been updated or validated. The optional
`--mtp-q4` head conversion still requires a head format recognized by upstream ggml. These results establish
working single-GPU execution and format correctness within the stated tests, not a production performance target.
