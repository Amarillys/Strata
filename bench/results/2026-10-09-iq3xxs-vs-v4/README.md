# GSQ-RCO IQ3_XXS versus v4 IQ3_KT

Same-day comparison on 2026-10-09: i9-12900HK, RTX 2080 Ti 22 GiB and RTX 4080 SUPER
32 GiB, Windows, one Strata 0.1.41 CUDA binary. Same 22/26 split, K8V8, prefill 2048,
307200 context, 524288-token conversation pool, no MTP and no prompt reuse.

The [Chinese report](../../../docs/IQ3_XXS_VS_V4_2026-10-09.zh-CN.md) contains the
measurements, method, hardware, exact binary hash and limitations. Three 256-token
generation runs each at 3305, 9769 and 32768 prompt tokens; one at 128000, with a
cached continuation. The original v4 deployment was restored and rechecked.

`results.json` includes every measured request, warmups, continuation and restoration
checks. `runtime.json` holds sanitized configurations and runtime evidence.
`engine-timings.txt` and `prompt-recipes.json` provide timing lines and reproducible inputs.

`benchmark.py --out logs/<new-directory> --prepare-only` checks the local prerequisites.
Omit `--prepare-only` to run; this temporarily unloads the managed v4 service, uses
localhost port 18080, and restores the original launcher. The script uses credentials
from the local preset without putting them on the command line. Model paths and the
existing local preset are prerequisites; it does not download or switch a permanent default.
