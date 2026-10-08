# v4 IQ3_KT: split 22 / 26, 256K, K8V8

2026-10-08, the same Windows i9-12900HK / 64 GiB / 2080 Ti 22 GiB / USB4 4080 SUPER
32 GiB system as the [upstream comparison](../2026-10-08-iqkt-upstream-0.1.41/README.md).
Both configurations use the merged 0.1.41 build. Only layer split and maximum context change.

| Setting | Baseline | Selected preset |
| --- | ---: | ---: |
| 2080 / 4080 layers | 21 / 27 | 22 / 26 |
| Context capacity | 204800 | 262144 |
| Resident experts, 2080 / 4080 | 9142 / 13824 | 9090 / 13312 |
| Expert complement in RAM | 3117 MiB | 4209 MiB |
| 3305-token input decode | 42.84 tok/s | 41.43 tok/s |
| 9769-token input decode | 43.82 tok/s | 41.86 tok/s |

Both retain K8V8, 20480 resident KV cells, prefill 2048, prompt-cache 6, reserves
1024 / 128 MiB, pcie-frac 0.5, MTP off, direct SSD PLE and Intel Iris Xe Vulkan vision.
The 4080 stage is fully resident in both configurations.

Each configuration has two independent loads and four 256-token decode samples per
input length, plus two uncached prefills. The baseline comes from the earlier same-day
`after-a` and `after-b` runs. This is a sequential configuration comparison, with the
same request-triggered GPU wake policy and dynamic core clocks; it is not an interleaved
configuration crossover. Means use total output tokens / total decode time.

The selected preset is 1.41 / 1.96 tok/s slower on these inputs (3.3% / 4.5%). Short
output text matches between configurations; medium output matches within each configuration
but differs between layer placements. These samples do not establish task-quality changes.

| Selected-preset validation | Input / output tokens | Prefill | Decode |
| --- | --- | ---: | ---: |
| Near-capacity retrieval and generation | 261618 / 384 | 320.104 s | 36.6 tok/s |
| Immediate continuation | 262024 / 10 | 0.364 s | 39.7 tok/s |
| Sustained short-input generation | 3305 / 1024 | 6.423 s | 39.7 tok/s |

All three embedded access codes were retrieved. The continuation reused 262001 tokens,
read only 23 new tokens, and correctly returned the last code. Input and output share
the 262144-token budget; the continuation reached 262034 total tokens.
Minimum NVML free memory during the long/sustained checks was 1157 MiB / 1518 MiB.
The 4080's CUDA free-memory reading after graph capture was 1059 MiB, a different measurement.

After validation, this configuration was deployed on the existing authenticated 8080
service. Text, original model alias, Web, LAN and Intel vision checks passed. The choice
follows the user's preference for full context at this measured speed cost. Split 21 / 27
with 256K was not tested, so 200K is not an established hard limit of that split.

- `results.json` contains configuration, comparison, individual requests, memory/clock
  summaries, long validation and final deployment checks.
- `long-prompt-recipe.json` specifies the synthetic near-capacity prompt, code positions,
  user-text hash and sampling parameters. Short/medium messages are in the preceding
  upstream comparison's `prompts.json`.

To reconstruct the long prompt, use `tools/strata_tokenizer.py::Tokenizer.from_gguf`
with the v4 model and encode with `parse_special=True`. Start with `head`, fill with repeated
`filler` tokens to each listed offset, then append the encoded string
`\nRecord {record} access code: {code}\n`. Fill to `target_tokens_before_http_template`
minus the encoded tail length and append `tail`. Decode, extract the user text from
`Read the diagnostic log below.` up to the following `<|im_end|>`, and submit it with
the recorded system message. This produces 261618 HTTP prompt tokens with the tested template.

No private configs, credentials or user conversations are included. One synthetic
near-capacity retrieval is not a general long-context quality benchmark, and comparisons
with 204K long runs cannot isolate split cost from different input lengths.
