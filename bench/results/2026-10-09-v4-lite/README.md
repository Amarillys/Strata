# v4-lite local composition and benchmark

The hybrid uses v4 dense weights (including all 387 promoted projections), native
head, embedding and tokenizer; every routed expert and only the PLE lookup table
come from v2. Existing GGUF payloads are selected directly without requantization.

- `compose.py` creates an independent pack and ignored local config. It refuses
  to overwrite an existing candidate. `composition.json` records sources, copied
  file hashes, tensor offsets and all 387 included projections.
- `benchmark.py` uses the existing `Presets` manager, isolates requests on port
  18080, and restores the original launcher and service on completion or failure.
  It requires one managed v4 service before starting. Raw configs remain under
  `logs/.../private/` and must not be published.
- `summarize.py` exports synthetic results from completed runs. `summary.json`
  contains timing/residency aggregates; `results.json` contains per-request
  measurements; `runtime.json` records binary/configuration evidence;
  `quality.json` preserves the 13-case smoke results and generated answers.
- `deployment.json` records the final v4-lite service on port 8080, an API probe
  through the existing v4 model alias, and original-config/launcher checks.

Commands used from the repository root:

```powershell
python -B bench/results/2026-10-09-v4-lite/compose.py
python -B bench/results/2026-10-09-v4-lite/benchmark.py --out logs/v4-lite-2026-10-09-1810
python -B bench/results/2026-10-09-v4-lite/benchmark.py --out logs/v4-lite-reserve-2026-10-09-1830 --arms hybrid19r960l512
```

Use a new output directory for a rerun. A hybrid arm's `r` suffix overrides the
primary GPU reserve in MiB and `l` overrides the later GPU reserve. Without these
suffixes, the reserve values come from the original v4 config. These switches do
not change weights, context, KV format or matrix shapes.

Throughput uses fresh conversations, identical tokenized inputs, greedy sampling
with thinking disabled, and 256 generated tokens. Short/medium inputs have three
measured repetitions, 32K has two, 128K one. The long conversation is parked while
an unrelated short conversation runs, then resumed and checked for cached tokens
and three retrieval codes. Each arm also runs Chinese extraction, arithmetic,
code semantics, structured output and a synthetic function-call case.

The correctness set is small and is not an overall quality ranking. No BF16
reference NLL/KLD, full 300K request, complete 512K pool stress test, or new visual
image request was included. The existing Intel vision configuration is retained.
No C++/CUDA code or executable was modified for this composition.

See the [Chinese report](../../../docs/IQ_KT_V4_LITE_2026-10-09.zh-CN.md).
