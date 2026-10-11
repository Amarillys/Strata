# v2 dense + v4 experts + v4 IQ3_KT PLE

This arm reuses the saved v4 reference logits and all 128 positions from the
[previous KLD run](../2026-10-09-iqkt-kld/README.md). It adds 128 predictions and eight repeat checks.
The previous four models were not rerun, and their original result files were not overwritten.

| Model, with v4 IQ3_KT PLE throughout | Mean KL(v4 || model) | Top-1 agreement | Mean sampled NLL | GPU weight saving vs v4 |
| --- | ---: | ---: | ---: | ---: |
| Original v4 | 0 | 100% | 1.238878 | 0 |
| Previous: v4 dense + v2 experts | 0.031595 | 92.1875% | 1.266321 | 2.7344 GiB |
| New: v2 dense + v4 experts | 0.083597 | 89.8438% | 1.276728 | 1.6094 GiB |

The new arm kept all v4 experts: 112 IQ3_KT tensors and 32 IQ4_KT gate/up tensors. It used all v2 dense
weights, including the 387 small projections, PLE key/value projections, output head, and token embedding.
Only the PLE table came from v4. No weights were requantized or calibrated.

Overall sampled NLL was higher than in the previous v4-dense/v2-expert arm, with less GPU memory saved.
The tool/JSON subset improved, while Chinese, English, and code worsened. These small synthetic samples
do not establish a general quality ranking; reference KL is a distribution comparison, not a quality score.

All 136 new captures passed length, finite-value, and saved-file hash checks. No prompt cache was reused.
All eight repeats were byte-identical. The same binary and runtime settings were retained; automatic expert
residency changed, adding 834 GPU experts relative to the original v4 run and reducing RAM expert storage
from about 3.76 GiB to 2.16 GiB. Backend numerical differences remain part of cross-model comparisons.

- [Chinese report](../../../docs/IQ_KT_V2_DENSE_V4_EXPERTS_2026-10-09.zh-CN.md): full results, memory accounting, and limits.
- [summary.json](summary.json): previous four arms plus the new per-position results and aggregates.
- [composition.json](composition.json): source shard metadata, every expert extent, copied-file hashes, and weight accounting.
- [manifest.json](manifest.json): exact test arguments, binary hash, and links to the immutable baseline.
- [capture-validation.json](capture-validation.json): capture and residency checks.
- [runtime.json](runtime.json): loading, successful completion, and restoration evidence.
- [dense_variant.py](../2026-10-09-iqkt-kld/dense_variant.py): preparation, evaluation, and restoration script.

To reproduce, start with one managed Strata preset and use a fresh task directory under `logs/`:

```powershell
python bench/results/2026-10-09-iqkt-kld/dense_variant.py --base logs/iqkt-kld-20261009-quick-01 --out logs/iqkt-kld-v2dense-new --run
```

Omit `--run` to prepare and validate without switching models. Private config copies, launcher arguments,
the candidate pack, and raw logits remain under `logs/iqkt-kld-20261009-v2dense-v4experts-01/`.
The original v4-lite service was restored; the new arm was not deployed as a production preset.
