# KT prefill measurements and optimization, 2026-10-08

The daily [vision deployment](IQ_KT_VISION.md) uses chunk 2048 and K8V8. Chunk 4096 below is a separately tested
long-input option. See the [work summary (中文)](IQ_KT_SUMMARY.zh-CN.md) for the combined results and current settings.

The initial KT prefill adapter spent much of its GPU timeline expanding expert weights to FP16. Replacing its
scalar decoder with a parallel packed decoder improves measured 2,048-chunk prefill from 413 to 642 tok/s at
3,570 input tokens, and from 634 to 1,013 tok/s at 12,150 tokens. The existing FP16 matrix products are retained;
this is not a new quantized MMQ implementation.

## Configuration and comparison

The model, machine and request-time CUDA wake routine are described in [IQ_KT_PERFORMANCE.md](IQ_KT_PERFORMANCE.md).
These measurements use K8V8 (`--kv int8`), capacity 262,144, 20,480 resident KV cells, MTP off, layers 0-18 on the
2080 Ti and 19-47 on the 4080 SUPER. All 24,576 experts are resident again for decode. Prompt caching and the
other draft paths are disabled. No proxy or running service configuration was changed.

Each normal run includes a 72-token warm-up, then two requests each at 3,570 and 12,150 tokens. One 3,570-token
request generates 256 output tokens to check sustained decode; the other requests generate 32. The longer prompt's
local request name is `16k-*`, but its actual tokenizer count is 12,150; the numbers here use that count.

| Actual input tokens | Old decoder, chunk 2048 | New decoder, chunk 2048 | New decoder, chunk 4096 |
| --- | ---: | ---: | ---: |
| 3,570 | 413.3 (412.4-414.2) | 641.7 (632.6-650.8) | 647.6 (644.3-650.9) |
| 12,150 | 634.2 (633.4-635.0) | 1012.6 (1011.5-1013.6) | 1158.1 (1143.8-1172.5) |

Values are mean prefill tok/s with the two measurements' range, excluding warm-up. With chunk 2048 held fixed,
the gains are 55.3% and 59.7%. Raising the new decoder's chunk to 4096 adds about 14.4% at 12,150 tokens; the small
3,570-token difference does not establish a benefit there. These runs use the same power-handling policy, not a
fixed-core-frequency control. Telemetry accompanies every run.

The 256-token decode check measured 58.08 tok/s before, 58.50 after, and 58.51 with chunk 4096, all using K8V8.
There is no observed loss of the preceding decode optimization in this check. This is not a sustained 260K-input
decode comparison.

## What the phase profile shows

`STRATA_PREFILL_TIMING=1` records CUDA events at phase boundaries. Intervals include stream waits and gaps while
the host submits work; they are not hardware counters isolating arithmetic or memory stalls. The two devices
pipeline successive chunks, so their timelines must not be summed and called request wall time. Extra events
also add overhead: the instrumented 3,570-token requests took 9.63 s before and 6.91 s after, versus about
8.64 s and 5.56 s in normal runs. Throughput claims above use the normal runs.

The profiler handles 3,569 tokens in the batched path; the request's last token uses the verifier. Its main stage
logs one timeline and its successor logs two chunks:

| Device / scope | Earlier GPU timeline ms | Earlier expert dequant ms | New GPU timeline ms | New expert dequant ms |
| --- | ---: | ---: | ---: | ---: |
| 2080 Ti, layers 0-18, full batched input | 5006 | 2456 | 4027 | 749 |
| 4080 SUPER, layers 19-47, chunk 2048 | 3195 | 2045 | 2131 | 718 |
| 4080 SUPER, layers 19-47, chunk 1521 | 2929 | 1935 | 1997 | 702 |

The expert-dequant intervals fall from 6.44 s summed across the stages to 2.17 s, about 66%. Before the change,
they occupied 49-66% of each listed timeline. FP16 gate/up and down products occupied about 20-21%. The profile
therefore justified optimizing the decoder before implementing a much larger MMQ adapter. Some new matrix-product
intervals are longer under instrumentation; because phase events also charge host submission gaps, this alone
does not show that cuBLAS arithmetic became slower.

## Implementation and numerical checks

[kt_kernels.cu](../src/kernels/cuda/kt_kernels.cu) now assigns four lanes to each 32-value group. Each lane
reconstructs eight adjacent codes using the packed DP4A trellis helper, preserving an IQ3 seed's eight recurrence
steps without duplicate work. Separate IQ3/IQ4 kernels handle full and tail blocks. Aligned outputs use vector
stores; scalar stores preserve arbitrary destination offsets and leading dimensions.

The weight scales, codes, FP16 rounding, cuBLAS products, Q8_1 decode activations, and MMVQ implementation are
unchanged. The path also serves KT embeddings and FP32/BF16 row conversion.

The 37 independent IK-oracle cases passed on both sm75 and sm89, with maximum FP32 decode relative error 1.73e-7
and GEMV error 4.70e-7. Added checks compare FP32, FP16 and BF16 output bits against the original scalar decoder,
including aligned and offset destinations, odd row strides, interleaved rows and untouched padding. The existing
tests also cover embedding gathers, tail widths, gate/up interleaving, and grouped experts.

All tested generated tokens match with chunk 2048 held fixed, in both normal and profiled runs. Changing to chunk
4096 changes one 256-token answer beginning at output token 90 (zero-based); its first 32 tokens and the other
short-output requests match. This chunk comparison must not be described as bit-identical output.
An additional control ran the old decoder with chunk 4096: its warm-up and the complete 256-token answer both
match the new decoder at chunk 4096 exactly. Thus the observed answer difference follows the chunk change in
this test, rather than the replacement decoder. Changing chunk size changes matrix and recurrence batching;
the control does not establish general output equivalence across chunk sizes.

## Actual 260,000-token input with K8V8

The optimized decoder and chunk 4096 processed the existing `build-kt/prompt-260k.txt` with no cached prefix.
Prefill took 198,126.3 ms, or 1,312.3 tok/s. The answer was `cedar, cobalt, quartz` followed by EOS, correctly
retrieving values placed near the beginning, middle and end. This is a repeated synthetic retrieval test, not
a broad long-context quality evaluation.

All routed experts were restored to VRAM for decode, with zero CPU expert and PCIe expert entries. The logged
stage's KV counter reported 62.74% VRAM hits and 24.7 MiB read from RAM; it is not a whole-model aggregate.
The eight-token answer took 181.0 ms, which is too short to establish sustained long-context decode throughput.

The new local preset is `strata-iq3kt-dual19-int8-pf4096.json`. Its engine arguments match this test: capacity
262,144, K8V8, 20,480 resident KV cells, MTP off, split 19, and 4096-token prefill. Keep using the existing CUDA
wake routine; the preset does not install automatic GPU power management. The 2048 preset remains available.
Both use the rebuilt `build-kt/Release/strata.exe`.

Chunk 4096 borrows 1,133 expert slots per card (about 2.05 GiB) for its workspace, versus 687 (1.24 GiB) in this
turn's chunk-2048 runs. These slots are restored at the end of prefill. The throughput measurements include the
request's preparation and restoration costs, not just the inner GEMMs.
The benchmark engines have exited and all temporary GPU clock locks have been released.

## Remaining work and evidence

The model still expands each selected expert to FP16 and calls its products separately. A dedicated KT MMQ or
fused grouped-expert path could avoid the full intermediate matrix and reduce launch count. Its activation
precision and numerical contract would need independent checks; the measurements here do not promise a speedup
for that unimplemented path. New profiling should guide further work after this decoder change.

Local artifacts under `build-kt/`:

- `prefill-kt-{before,after}-config.json`, results, logs and telemetry: normal comparisons.
- `prefill-kt-{before,after}-profile-*`: instrumented comparisons, including both GPUs' chunk timelines.
- `prefill-kt-pf4096-*`: larger-chunk measurements.
- `prefill-kt-old-pf4096-control-*`: the original decoder at the larger chunk, isolating the output difference.
- `prefill-kt-summary.json` and `summarize-prefill-kt.py`: timing/output comparisons and profile parsing.
- `prefill-kt-build.log`: sm75/sm89 build. The earlier binary is `Release/strata-pre-prefill-kt.exe` and its
  kernel source snapshot is `kt_kernels-pre-prefill.cu`; the rebuilt preset executable is `Release/strata.exe`.
