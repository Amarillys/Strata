# IQ_KT sampled KLD, 2026-10-09

128 fixed positions from eight synthetic documents; 248,320 full-vocabulary logits per position.
The reference is quantized v4, not BF16. The IQ3 PLE arm was added after the user asked to compare it.

| Model | PLE table | Mean KL(v4 || model), nats | Top-1 agreement | Mean sampled NLL, nats |
| --- | --- | ---: | ---: | ---: |
| v4 reference | v4 IQ3_KT | 0 | 100% | 1.238878 |
| v2 | v2 IQ4_KT | 0.080085 | 89.0625% | 1.281316 |
| v4-lite | v2 IQ4_KT | 0.037340 | 93.7500% | 1.262176 |
| v4-lite-iq3ple | v4 IQ3_KT | 0.031595 | 92.1875% | 1.266321 |

Each model also repeated eight positions; repeat KL was zero for all four arms. All 544 dumps passed
shape, finite-value, and saved-file hash checks. Both lite arms used the same GPU expert residency.
The other models had different automatic residency, so their comparison includes CPU/GPU numerical effects.

The IQ3 PLE variant saves 4.7684 GiB of logical tensor payload relative to the IQ4 PLE combination.
It had lower mean reference KL, slightly higher sampled NLL, and two fewer Top-1 matches.
These are short synthetic text samples, not a general quality ranking or a standard perplexity benchmark.

The reference itself uses IQ3_KT PLE. Retaining its table can also retain its quantization errors;
an improved candidate may diverge from that reference. The IQ4 PLE lite arm had slightly better sampled
NLL despite higher reference KL, so KL and Top-1 agreement must not be used as a quality leaderboard.
Full v4 dense weights and experts plus v2 IQ4_KT PLE were **not tested**. That combination is a reasonable
quality-first candidate, with no measured claim that it is best. The present sample does not establish
whether IQ4 PLE's extra 4.7684 GiB is justified by general quality gains.

- [Chinese report](../../../docs/IQ_KT_KLD_2026-10-09.zh-CN.md): interpretation, per-category numbers,
  capacity, runtime settings, and limitations.
- [corpus.json](corpus.json): complete synthetic evaluation text, selected before inference.
- [summary.json](summary.json): all per-position results, category aggregates, and repeatability.
- [manifest.json](manifest.json): original three-arm configuration allowlist, binary and tokenizer hashes.
- [ple-variant-manifest.json](ple-variant-manifest.json): the added IQ3_KT PLE arm and its sole weight change.
- [capture-validation.json](capture-validation.json): dump checks, cache use, and actual expert residency.
- [runtime.json](runtime.json): completion and restoration evidence, with private launcher arguments omitted.

The native serve diagnostic writes one FP32 logits row per request. A temporary identity template keeps
each request equal to the intended text prefix; encoding it again must reproduce the original token IDs.
All requests use a cold prompt state. Comparison applies full-vocabulary stable log-softmax in FP64 at
temperature 1. Each sampled target is the real next token in the fixed document, not a generated token.

To reproduce, use a fresh directory under this repository's `logs/`, starting with the managed v4-lite
service running. These commands temporarily unload it and restore its original argv, cwd, and configuration:

```powershell
python bench/results/2026-10-09-iqkt-kld/quick_kld.py --out logs/iqkt-kld-new-run --run
python bench/results/2026-10-09-iqkt-kld/ple_variant.py --out logs/iqkt-kld-new-run --run
```

Omit `--run` for preparation only. The variant can be queued during the baseline with `--wait-for-base`.
Neither script edits production configs, merges GGUF files, or requantizes weights. Private configuration
copies and raw logits remain under `logs/iqkt-kld-20261009-quick-01/`. The restored production preset keeps
the original IQ4_KT PLE; the IQ3_KT PLE variant was evaluated without deploying it.
