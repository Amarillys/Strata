# Where local KT decode time goes, 2026-10-08

This profile predates the packed KT CUDA optimization described in
[IQ_KT_PERFORMANCE.md](IQ_KT_PERFORMANCE.md#packed-kt-cuda-row-dot-2026-10-08).
Its stage percentages describe the earlier binary, not the faster replacement.

The all-resident, MTP-off configuration spends most of its measured GPU time on experts and dense projections.
CPU expert fallback and expert-weight streaming are no longer the main target. On these short/3.6K prompts,
attention/KV work is a relatively small share, so shrinking KV is not expected to transform decode throughput.

## Method and limits

The tested configuration is the INT8-KV, 262,144-capacity, 20,480-resident-KV preset from
[IQ_KT_PERFORMANCE.md](IQ_KT_PERFORMANCE.md): layers 0-18 on the 2080 Ti, 19-47 plus the head on the 4080 SUPER,
all 24,576 routed experts resident, MTP off, and 512-token prefill. Both GPUs use the existing request-time wake
routine. After a warm-up request, the profiler records 256 generated tokens for the 72- and 3,570-token prompts.

`STRATA_VERIFY_PROFILE=1` places GPU timestamps around verifier stages. It also disables the shared expert's
separate-stream overlap and reads profile data back. Therefore this is an instrumented breakdown, not the normal
throughput benchmark: the measured request takes 27.40 ms/token for the short prompt and 29.30 ms/token after
3,570 input tokens, versus roughly 25-26 ms/token without instrumentation. Stage intervals include their enclosed
kernel/dependency time; this is not a hardware-counter diagnosis separating memory stalls from arithmetic.

The table uses the 3,570-token request. Percentages divide by the two GPUs' reported stage sum, 28.91 ms/token.
The small difference from the 29.30 ms/token request time is not assigned to CPU computation. Summed printed
components can differ by a few hundredths of a millisecond because the logger rounds each component.

| Measured group | ms/token | Share of GPU stage sum |
| --- | ---: | ---: |
| Routed experts: gate/up, activation quantization, down | 9.43 | 32.6% |
| Shared expert and its activation quantization | 1.93 | 6.7% |
| QKV/query/indexer, gate and output projections | 8.50 | 29.4% |
| Hyperconnection mixing/normalization, routing and the PLE-containing segment | 5.55 | 19.2% |
| GDN recurrence/convolution and QSA attention/KV processing | 2.40 | 8.3% |
| Final vocabulary head and token selection | 0.80 | 2.8% |
| Remaining waits, combine and inter-layer gaps | 0.32 | 1.1% |

The short-prompt measurement has the same ranking: routed + shared experts 41.2%, projections 30.0%, mixing and
routing 17.8%, recurrence/attention 6.7%, head 2.9%. The profile's `hc0 norm` is a timing segment, not a pure norm
kernel measurement: the first read segment can include PLE work before the hyperconnection read. The logger
replaces `hc-read0` with its inner intervals when those stamps exist; the summary does not count both.

## Distribution across the two cards

| GPU | Main layers | GPU stage time, 3.6K prompt | Share |
| --- | ---: | ---: | ---: |
| RTX 2080 Ti | 19 | 17.83 ms/token | 61.7% |
| RTX 4080 SUPER | 29, plus output head | 11.08 ms/token | 38.3% |

For this single sequence, the stages run in order with a hand-off; the two cards' arithmetic capacity does not
simply add. The partition was chosen to make all experts fit. It is not a partition with equal execution times.
Moving another layer to the 4080 would require more weight memory there or a new cache tradeoff; this trace alone
does not establish that such a move would improve total throughput.

## What to optimize next

1. **KT expert and projection kernels, especially on the 2080.** Experts and projections together occupy about
   69% of this profile. The profiled [KT CUDA adapter](../src/kernels/cuda/kt_kernels.cu) reconstructed 32 integer
   codes in each lane before DP4A, with a generic row-dot path. That implementation has since been replaced with
   separate IQ3/IQ4 packed row-dot kernels. Model metadata confirms that layer 0's QKV, gate and output projections are IQ4_KT, and its routed
   experts are IQ3_KT. Optimizing only routed experts would miss the substantial dense-projection cost.
2. **Mixing and routing.** This is another sizable group, using Strata's existing fused/BF16 paths for the small
   converted projections. Any change needs to account for the segment's PLE work and existing fusion, not assume
   the entire 5.55 ms is a single norm or projection.
3. **KV changes are primarily a capacity measure at these input lengths.** All attention/recurrence work combined
   is 8.3% here, and only part of that is KV reads. A long-context profile can differ. The percentage is not a
   prediction for actual 200K/260K input, nor a claim that all of this time can be removed by quantization.

The normal decode log reports zero CPU expert work and zero PCIe expert entries for the all-resident preset.
CPU orchestration, token/PLE reads and cross-GPU state transfer still exist. These measurements do not support
blaming the remaining gap primarily on CPU experts or USB4 expert streaming. They also do not establish a
compute-bound versus memory-bound classification for the KT kernels; that needs kernel-level counters.

## Evidence

- `build-kt/decode-stage-profile-int8-all-resident-config.json`: exact configuration and prompts.
- Matching `.log`, `.json` and `.protocol`: GPU stage stamps, request times and generated output.
- `build-kt/decode-stage-profile-summary.json`: grouped totals for warm-up, short and 3.6K input.
- `build-kt/summarize-decode-stages.py`: aggregation from named log fields, without summing overlapping groups.
- `build-kt/kv-telemetry.jsonl`: GPU clock, P-state and memory samples during these tests.

No inference kernel was changed to produce this profile. The profiler process exits after its requests and the
temporary clock settings are released.
