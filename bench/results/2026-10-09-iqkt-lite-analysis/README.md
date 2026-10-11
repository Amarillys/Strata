# IQ3_KT lightweight candidates: offline accounting

2026-10-09. [Analysis and recommendations](../../../docs/IQ_KT_V4_LITE_OPTIONS.zh-CN.md).

`analyze.py` reads the existing local v2/v4 GGUF headers and prepared pack indexes.
It does not create model weights, invoke CUDA, or change the running service.
`analysis.json` records the 387 candidate projection upgrades, layer-by-layer
expert savings, and GPU weight payloads for possible mixed representations.

Payloads exclude KV, runtime workspaces, graphs, driver allocations and reserves.
Candidate quality, speed and full GPU residency have not been measured.

`compare_experts.py` subsequently reads and directly compares the entire payload
of every same-type routed-expert tensor and router in both local GGUF versions,
recording per-tensor SHA-256 in `expert-identity.json`. All 112 IQ3_KT expert
tensors and all 48 F32 routers are byte-identical. The remaining 32 expert
gate/up tensors change from IQ3_KT to IQ4_KT and are inventoried by type and size.
The comparison read 69.6875 GiB in 76.079 seconds with bounded read buffers;
input file sizes and modification times were unchanged. It does not verify
whole-file, high-precision source or original imatrix hashes, or candidate quality.

Run from the repository root with:

```powershell
python bench/results/2026-10-09-iqkt-lite-analysis/compare_experts.py
```

The local execution log is `logs/iqkt-expert-identity-2026-10-09-164725/compare.log`.

`promoted-value-samples.json` additionally records decoded-value samples from
the 32 changed-type gate/up tensors: first, middle and last flattened rows of
each tensor, decoded using `tools/kt_quants.py`. All 32 tensors have numerical
differences, across 245,760 sampled positions per model. This is not a complete
value scan or a comparison against original BF16 values; it cannot establish
pre-quantization source identity or quantify error relative to BF16.
The raw record is `logs/iqkt-promoted-values-2026-10-09-165954/comparison.json`.
