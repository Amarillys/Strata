# Local IQ_KT performance comparison, 2026-10-07/08

The current packed KT kernels reach 58-63 decode tok/s in the measurements below. The
[packed-kernel comparison](#packed-kt-cuda-row-dot-2026-10-08) isolates that change; subsequent
[prefill measurements](IQ_KT_PREFILL.md) cover the optimized row decoder and K8V8.

The first sections record the earlier configuration: disabling MTP and fitting all experts in VRAM brought
decode to 38-40 tok/s, close to the optimized IK baseline on those prompts, before the packed-kernel change.
The early counting/code requests were too short to establish sustained throughput. The earlier 0.79 to 7.51
tok/s result measures replacement of the initial scalar CPU fallback, not a gain over IK.

The subsequent [decode stage profile](IQ_KT_DECODE_PROFILE.md) identifies experts and dense projections as the
largest measured groups and distinguishes instrumented timings from the unprofiled throughput below.

## Test conditions

Core i9-12900HK, 64 GiB RAM, RTX 2080 Ti 22 GiB on PCIe 3.0 x8, RTX 4080 SUPER 32 GiB over USB4. Both engines use
the local Uncensored IQ3_KT_v2 GGUF; Strata's MTP sidecar is IQ4_KT. The two prompts contain 72 and 3,570 tokens:
a request for a 2,000-word technical manual about temperature, pressure, flow and anomaly detection, with one or
160 copies of the same measurement-log sentence. The exact text is in the saved benchmark configs. Each run has
one warm-up request followed by two measurements of each prompt. Every measured request generates 256 tokens,
greedily, without suppressing EOS. Values below exclude the warm-up and give the range of the two measurements.

Strata uses engine 0.1.40 with the local KT adapter, a 262,144-token capacity, INT8 KV, 20,480 resident KV cells,
512-token prefill, the same expert profile, auto-sized expert caches, trimmed per-stage weights, and a 1,536 MiB
VRAM reserve. Prompt caching, adaptive expert swaps, suffix drafts and lookup chaining are disabled. `--spec 3`
is kept for both arms; removing `--mtp` is the actual MTP-off switch. Setting `--mtp-max-t 1` still performs draft
catch-up work and is not an equivalent control.

The IK comparison uses `B:/llama.cpp/stop-fix-20260930/bin/llama-server.exe`, the preserved optimized build, with
Turing graphs, the IQ3_KT 640-wide one-warp/four-row specialization and MMQ cap 64 enabled. MTP is off. It uses
48,128 actual context capacity, K=q8_0/V=q4_0, ubatch 512, batch 1024, six CPU threads, layer split 29/20, and
49/49 GPU-offloaded layers. It receives the same prompt token IDs; reported prompt counts confirm full prefill
on each request. A temporary localhost server is used; the existing proxy/backend configuration is untouched.

This compares the available configurations on the same workload, not just their kernels: capacity, KV format,
layer order, dense-weight representation and CPU expert work differ. Outputs across engines/MTP settings are
not necessarily token-identical. One prompt family and two measured repetitions do not establish general
quality, MTP acceptance, or long-context throughput. The sustained tests here have at most 3,570 input tokens;
the separate 260K check in [IQ_KT.md](IQ_KT.md) remains a capacity/retrieval smoke test.

## The eGPU wake-up requirement

`B:/llama.cpp/run_uncensored_server_ik.ps1` calls the dynamic-boost proxy. Its actual wake-up implementation is in
`B:/llama.cpp/proxy/gpu_monitor.py`: set the GPU clock ranges, then run a small FP16 matrix-multiply pulse. The
manager retains its CUDA context and pulses again when a request starts. Launching Strata directly bypasses it.

Without that handling, telemetry caught the 4080 at P8, 405 MHz core and 405 MHz memory throughout decode. The
initial sustained MTP-off run managed only 4.9-5.0 tok/s. With the existing wake-up routine, the same configuration
and identical output tokens returned to roughly 32 tok/s. A one-time pulse in a process that immediately exits
was not reliable: a later MTP run fell back to P8 after loading. Those low-clock runs are excluded from the table.

The retained comparison invokes the existing manager before each request and keeps the probe context alive.
The probe initializes after engine cache allocation, preserving the same expert slot counts between MTP arms.
It does not execute continuously alongside decode. Sampled decode clocks were 1,500 MHz core/6,800 MHz memory
on the 2080, and 2,100-2,805 MHz core/11,251 MHz memory on the 4080. These are the script's operating ranges,
not a claim of identical fixed core clocks at every instant. Clock locks are released in cleanup.

## Sustained results

| Engine/configuration | 72-token input, decode tok/s | 3,570-token input, decode tok/s | 3,570-token prefill tok/s |
| --- | ---: | ---: | ---: |
| Optimized IK, MTP off | 40.47-41.34 | 37.49-37.50 | 805-824 |
| Strata, split 21, MTP off | 32.59-32.94 | 31.68-32.23 | 265-265 |
| Strata, split 21, MTP on | 29.99-30.16 | 28.84-29.49 | 249-249 |
| Strata, split 20, MTP off | 34.93-35.08 | 33.80-33.89 | 248-249 |
| Strata, split 19, MTP off | 37.54-37.62 | 36.28-37.18 | 235-236 |
| Strata, split 19, MTP off, all experts resident | 39.87-39.94 | 38.03-38.93 | 243-244 |

The MTP arm accepts 133-134 of 246 proposed tokens (about 54%), producing 256 tokens in 124 windows, or 2.06
tokens/window. A window takes about 68-71 ms, versus roughly 31 ms for one token without MTP. The resulting
MTP configuration is slower on this workload. The near-100% acceptance of the earlier counting prompt does not
generalize. Keeping MTP enabled solely because it accelerated that smoke prompt is not justified.

The prefill gap also remains: KT currently decodes weights to FP16 before GEMM, rather than using a dedicated
KT MMQ path. The KT CUDA row-dot adapter reconstructs weights for each token, so it does not yet implement
explicit reconstruction reuse across verification tokens. These are implementation limits and optimization
candidates, not a measured attribution of every millisecond of the end-to-end gap.

## Spending the freed MTP memory on experts

The draft layer uses 1,364 MiB and its pruned head another 212.9 MiB on the 4080. At split 21, all 13,824 experts
assigned to that card already fit. Simply removing MTP leaves that card with more free VRAM but still leaves
the 2080 with 9,687 cached experts. The total stays at 23,511, with 1.92 GiB of experts in RAM.

Moving one layer from the 2080 to the 4080 (split 20) while leaving MTP off raises the caches to 9,721 + 14,336
= 24,057 experts. All of the 4080's experts still fit; the remaining RAM complement falls to 0.94 GiB. Measured
decode rises to 34-35 tok/s. This demonstrates why disabling MTP should be paired with repartitioning for this
configuration. It does not eliminate the measured gap to IK or establish that every further layer move helps.

Moving a second layer (split 19) makes all 9,728 of the 2080's experts resident. The 4080 caches 14,474 of its
14,848 experts; the total becomes 24,202 and the RAM complement falls to 0.68 GiB. The remaining CPU expert
work now belongs to the 4080's layers. Despite its slower host link, this configuration is faster on the tested
workload: 36-38 tok/s, with far fewer misses and a fully resident first stage. Prefill gets slower as more layers
move to the 4080 in these tests. Cache tuning improved decode, not overall throughput in every phase.

Finally, keeping the first card's reserve at 1,536 MiB and setting `--vram-reserve-later-mib 768` allows all
14,848 experts on the 4080 to fit. Together with the 2080's 9,728, all 24,576 routed experts (44.4 GiB) are now
resident. Both stages use the zero-doorbell graph, and the decode CPU-expert and PCIe-expert counters are zero.
Steady decode reaches 39.9 tok/s on the short prompt and 38.0-38.9 tok/s after 3,570 input tokens. The final
configuration therefore closes most of the initial decode gap without MTP. It does not close the prefill gap.

With the wake probe present, NVML snapshots between these requests showed about 1,875 MiB free on the 2080 and
1,140 MiB free on the 4080 (these are snapshots, not a guaranteed minimum during every kernel or other workload).
No reserve was removed from the display card. The 256K capacity and resident-KV limit are unchanged, but this
specific all-resident preset has only been exercised up to 3,570 input tokens, not the earlier actual 260K input.

Saved local server presets:

- `strata-iq3kt-dual19-no-mtp.json`: the measured all-resident configuration, including 512-token prefill.
- `strata-iq3kt-dual19-no-mtp-reserve1536.json`: split 19 with the original reserve on both cards; 0.68 GiB of
  experts remain in RAM, and measured decode is 36-38 tok/s.

The config parser and engine settings were checked; these measurements use the engine's stdio interface, not
an HTTP deployment of these presets. No server was left running. The presets themselves do not install the
request-driven GPU wake-up routine; they must be used with the existing wake-up handling on this eGPU system.

The logged cache-hit fraction needs care in a layer split: a fully resident stage's zero-doorbell path does not
contribute the same host-pool counters as a stage with misses. For example, 53,760 lookups over 256 plain-decode
tokens describe 21 layers at ten experts/token, not all 48 layers. Do not label that ratio as a whole-model hit
rate or compare its denominator unchanged across splits. The slot counts and RAM sizes above cover both cards.

## Local evidence and reproduction

All generated artifacts are under `build-kt/` and are not committed model/source files:

- `decode-awake-mtp-{off,on}-config.json`, matching `.json`, `.log`, `.protocol` results.
- `decode-awake-mtp-off-split{20,19}-*` and `decode-awake-mtp-off-all-resident-*` for cache repartitioning.
- `decode-sustained-ik.json` and `.log` for the private IK run, including actual launch arguments and timings.
- `decode-telemetry.jsonl` for timestamped GPU clocks, P-state, utilization and power.
- `decode-sustained-mtp-off-*` and the interrupted `decode-boosted-mtp-on-*` are low-clock diagnostic runs,
  excluded from throughput comparisons. `decode-boosted-mtp-off-*` is the successful one-shot-wake control.
- `run-smoke.py`, `run-ik-private.py` and `with-gpu-boost.py` are local harnesses. They do not edit the proxy config.

For example, from the repository directory on this machine:

```powershell
$env:PYTHONUTF8 = '1'
# The harness's wake probe sees only the 4080. Its child engine gets both UUIDs from the JSON config.
$env:CUDA_VISIBLE_DEVICES = 'GPU-d04690c4-efd4-12e9-9f0d-473bb2fe64a9'
python -B build-kt/with-gpu-boost.py build-kt/run-smoke.py build-kt/decode-awake-mtp-off-config.json
python -B build-kt/with-gpu-boost.py build-kt/run-smoke.py build-kt/decode-awake-mtp-on-config.json
```

Each harness exits its own engine after the requests and releases the temporary clock settings. These are
benchmark controls, not an installation of automatic power management into Strata's web server.

## K8V4 and a larger prefill chunk, 2026-10-08

The original `--kv int8` already quantizes both K and V. Strata also supports `--kv k8v4`: INT8 K and rotated
Q4_0 V. This is the same 8-bit/4-bit allocation idea as IK's K=q8_0/V=q4_0, not the same storage layout or
bit-identical arithmetic. No KT kernel change was needed to enable it.

The model has 12 QSA layers. Their K/V storage costs 1,056 bytes/cell with INT8 and 816 with K8V4. At 262,144
capacity, the full host K/V pools therefore change from 3,168 to 2,448 MiB, saving 720 MiB. Because
`--kv-resident 20480` keeps only that many cells per QSA layer on the GPU, the resident K/V pools save just
56.25 MiB across both GPUs. These calculations exclude other state, page tables and prefill staging buffers.
All experts already fit, so the saving does not add more experts in this configuration.

Reducing capacity to 204,800 would reduce the K8V4 host pools to 1,912.5 MiB, but would not reduce the fixed
20,480-cell resident pools. It is not necessary for the tested K8V4 allocation and is not a demonstrated
prefill speed optimization.

The same 72/3,570-token prompts, 256-token outputs, clock handling and warm-up protocol were used. Normal
benchmarks below have `STRATA_VERIFY_PROFILE` disabled. All 24,576 experts remain resident during decode.

| KV | Prefill chunk | 3,570-token prefill tok/s | 72-token input, decode tok/s | 3,570-token input, decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| INT8 | 512 | 243-244 | 39.87-39.94 | 38.03-38.93 |
| K8V4 | 512 | 245.7-245.8 | 39.35-39.74 | 37.67-38.86 |
| INT8 | 2048 | 411.6-418.1 | 39.76-39.93 | 37.80-38.84 |
| K8V4 | 2048 | 410.7-417.1 | 38.63-39.44 | 37.95-39.04 |

The meaningful prefill gain comes from the larger chunk, not from changing KV precision. Chunk 2048 temporarily
borrows 719 expert slots (1.30 GiB) per card instead of 340 (0.61 GiB); the cache is restored for decode.
The short-prompt decode measurements do not establish a speed benefit for K8V4. Output text can differ across
KV formats; these timing comparisons do not establish equal quality across arbitrary inputs.

Evidence: `build-kt/kv-k8v4-256k-pf512-*`, `kv-int8-256k-pf2048-*`, `kv-k8v4-256k-pf2048-*`, and
`kv-telemetry.jsonl`. The original INT8/512 results are the all-resident benchmark above.

### Actual 260,000-token K8V4 input

The K8V4/2048 run then processed the existing `build-kt/prompt-260k.txt` with no cached prefix. It correctly
returned `cedar, cobalt, quartz` followed by EOS: the keys are near the beginning, middle and end of the input.
Prefill took 388.187 seconds, or 669.8 tok/s. The eight-token answer took 246.6 ms (32.4 tok/s), which is too
short to establish sustained long-context decode throughput. The logged stage's KV counter reported 62.31%
VRAM hits and 25.0 MiB read from RAM; this is not a whole-model aggregate. Decode required no CPU experts or
expert-weight streaming, and all experts were restored to VRAM after the larger prefill workspace was returned.

This is a long-input allocation/streaming/retrieval check, not a broad long-context quality evaluation. It shows
that reducing capacity to 200K is unnecessary for this tested setup. The 262,144-token capacity was retained.

Saved presets, validated through the server's argument and environment construction:

- `strata-iq3kt-dual19-k8v4-256k.json`: K8V4, 2048-token prefill, all experts resident, MTP off, tested at actual
  260K input through the engine's stdio interface.
- `strata-iq3kt-dual19-int8-pf2048.json`: the same larger chunk with original INT8 KV; its new comparison covers
  short and 3.6K inputs.

Neither preset installs GPU power handling into the web server. Continue using the request-time CUDA wake-up
routine on this eGPU system. Benchmark processes have exited and the temporary clock settings were released.

## Packed KT CUDA row dot, 2026-10-08

The initial CUDA adapter decoded 32 scalar integer codes, packed them again for DP4A, and selected IQ3 versus IQ4
inside the device path. The replacement specializes the two formats, generates four codes directly in packed
byte form, uses DP4A for the trellis byte sum, applies IQ3 signs with byte operations, and loads aligned seeds and
sign words. Dense KT projections and grouped experts both use it. The row grouping, warp reduction order, scales,
Q8_1 activations and gate/up activation are retained. Prefill's FP16 dequantization path is unchanged.

The earlier binary is preserved as `build-kt/Release/strata-pre-packed-kt.exe`; the optimized binary is
`build-kt/Release/strata.exe`. Alternating old/new/old/new runs used the K8V4/2048 preset above: MTP off, layer
split 19, all 24,576 experts resident, 262,144 context capacity, 20,480 resident KV cells, prompt caching and other
draft paths disabled. Each run included one warm-up, then two 72-token and two 3,570-token prompts, each generating
256 tokens. The following means/ranges exclude the warm-up and cover four measurements per input length/version.

| Input tokens | Earlier decode tok/s, mean (range) | Packed decode tok/s, mean (range) | Mean gain |
| --- | ---: | ---: | ---: |
| 72 | 39.21 (38.30-41.24) | 63.37 (62.79-63.96) | 61.6% |
| 3,570 | 39.43 (37.50-43.94) | 62.22 (60.03-64.27) | 57.8% |

Every generated token matched the earlier binary for the corresponding request, including warm-ups. The
37 independent IK-oracle fixture cases passed on sm75 and sm89, with maximum decode relative error 1.73e-7 and
GEMV error 4.70e-7, unchanged from the earlier tests. The CPU parity executable also passed its 37 fixtures over
1-9 tokens. This establishes numerical agreement for these cases, not arbitrary long-form quality equivalence.

Both versions used the retained request-time CUDA wake probe and the same clock ranges. Telemetry from the
second old run and both new runs recorded 2080 clocks at 1500/6800 MHz during sampled decode intervals and 4080
memory at 11251 MHz. The older kernel's 4080 core varied within approximately 2340-2790 MHz on measured requests;
the new kernel stayed around 2790 MHz. Thus the table measures the same power-handling policy, including its
workload-dependent frequency response, rather than perfectly equal core clocks.

A separate control locked the 2080 core to 1500 MHz and the 4080 core to 2100 MHz after every wake pulse, for
both binaries. Sampled decode clocks were exactly those values, with memory at 6800/11251 MHz. This used the
same five-request protocol; the table excludes warm-up and averages two measurements per length/version.

| Input tokens | Earlier fixed-clock decode tok/s | Packed fixed-clock decode tok/s | Mean gain |
| --- | ---: | ---: | ---: |
| 72 | 36.89 | 60.87 | 65.0% |
| 3,570 | 36.17 | 58.08 | 60.6% |

All output tokens again matched. Thus the gain persists with equal sampled core/memory clocks; it does not
depend on the new workload's higher automatic core frequency. These fixed clocks are a benchmark control,
not a change to the server preset or the user's power manager. The control's evidence is
`build-kt/packed-kt-{before,after}-fixed-*` plus the corresponding result, log and protocol files.

The 3,570-token prefill measurements were 412.1-419.9 tok/s before and 405.1-415.6 after. They show no prefill speedup
from this decode change. Dedicated KT MMQ/fused prefill remains unimplemented. No new actual 200K/260K-input
throughput claim is made by these short/3.6K tests; the previous 260K retrieval result belongs to the earlier binary.

Evidence: `build-kt/packed-kt-{before,after}-{a,b}-config.json` and matching `.log`, `.protocol`, `.json`, telemetry
files, `packed-kt-summary.json`, and `summarize-packed-kt.py`. `packed-kt-build.log` records the sm75/sm89 build.
The current K8V4 preset already selects the rebuilt executable. No model pack, proxy configuration or running
service was changed for this experiment. All benchmark engines exited and temporary GPU clock locks were released.

The subsequent [KT prefill optimization](IQ_KT_PREFILL.md) uses K8V8 and improves prefill while preserving this
MMVQ implementation. It includes separate normal/profile measurements and a larger-chunk comparison.
