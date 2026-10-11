# GSQ-RCO IQ3_XXS tensor allocation

2026-10-09. [Chinese analysis](../../../docs/GSQ_IQ3_XXS_WEIGHT_ALLOCATION.zh-CN.md).

`analyze.py` reads the two local shards for GSQ IQ3_XXS and Uncensored IQ3_KT
v2/v4, validates tensor bounds, and compares the 1223 shard-1 tensor types with
`publisher-allocation.txt`. The pinned allocation matches every local entry.
Shard 2 holds only the fixed IQ4_NL PLE table. All 1224 tensor names and shapes
match across models; tensor-value identity is not asserted.

`analysis.json` contains metadata, header hashes, type/role counts, tensor
extents, per-layer expert allocation, and GPU weight-payload accounting.
`expert-layers.md` gives the 48-layer expert comparison. Stage accounting reuses
the earlier IQ3_KT inventory and exactly reproduces its v2/v4 baselines.
GPU payloads exclude KV, state, workspaces, graphs, driver allocations and reserves.

Publisher revision: `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.
The original allocation and model card were fetched from
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09).
Download records and the analysis log are under
`logs/gsq-weight-allocation-2026-10-09-172734/`.

Run from the Strata repository root:

```powershell
python bench/results/2026-10-09-gsq-weight-allocation/analyze.py
```

No model was loaded or changed. GSQ/RCO optimization, calibration coverage,
publisher quality benchmarks, and whole-GGUF hashes were not reproduced.
