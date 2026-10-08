# IQ_KT upstream 0.1.41 validation, 2026-10-08

Windows, i9-12900HK, 64 GiB RAM, RTX 2080 Ti 22 GiB (PCIe 3.0 x8) and RTX 4080 SUPER
32 GiB (USB4). Both builds include the local IQ_KT and Intel Iris Xe Vulkan vision support.

This comparison holds v4 IQ3_KT at split 21 / 27, 204800 context, K8V8, 20480 resident KV
cells, prefill 2048, prompt-cache 6, reserves 1024 / 128 MiB, pcie-frac 0.5 and MTP off.
All four runs retain the same request-triggered GPU wake wrapper and load the vision helper.

| HTTP input tokens | Old decode tok/s | New decode tok/s | Old cold prefill s | New cold prefill s |
| ---: | ---: | ---: | ---: | ---: |
| 3305 | 43.73 | 42.84 | 6.908 | 7.330 |
| 9769 | 43.16 | 43.82 | 12.055 | 12.423 |

The order is old A, new A, old B, new B. Each run has an excluded 32-token warmup and
two requests per input length, each generating 256 tokens. The second request reuses
the prefix. Decode means use total tokens / total decode time. Cold prefill means
include only the two uncached requests per version and input length.

There is no overall throughput improvement in these samples. All corresponding
output texts match; this is not a general task-quality assessment. Clock sampling
covers only the second A/B pair, including prefill, with dynamic core clocks.

- `results.json`: every request, engine residency, comparison, clock and memory summaries, binary hashes.
- `validation.json`: build/tests, new 204K retrieval/continuation, 1024-token generation, v2 smoke and initial deployment.
- `prompts.json`: complete synthetic short/medium HTTP messages and sampling parameters.

The initial deployment record reflects the 21 / 27, 200K configuration tested here.
Any later split/context experiment is reported separately. Private config files,
credentials, raw hardware logs and user conversations are excluded.

See [the Chinese deployment record](../../../docs/IQ_KT_UPSTREAM_0.1.41_MERGE.zh-CN.md)
and [the current launch presets](../../../docs/IQ_KT_MODEL_SWITCH.zh-CN.md).
