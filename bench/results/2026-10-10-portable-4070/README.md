# Windows v4-lite runtime for RTX 4070 Ti SUPER

The package contains the unchanged Strata 0.1.41-iqkt.1 engine and matching server from the previously verified
Windows CUDA 13 release. It adds setup for the two exported v4-lite / IQ4_KT PLE GGUFs, a single-GPU 64K/K8V8
configuration, model integrity checks and a local speed-report tool. Python and CUDA runtime dependencies are
included. Model payloads, production credentials, private configs and production logs are excluded.

Archive: `dist/portable-v4-lite-20261010/Strata-v4-lite-4070TiS-Windows-20261010.zip`, 499,876,913 bytes.
SHA256: `8d1968281a8817cfaf372564be38c7ba7e7102fe3427f384d43a97407bea0792`.

The archive was extracted into a path with spaces. All 1,456 manifest members passed SHA256 checks; DLL imports
were checked for 88 PE files. With PATH restricted to the extracted bundle and Windows, the embedded Python
loaded the bundled dependencies and the device helper enumerated the local NVIDIA GPUs. The engine help showed
all flags required by the new template; this build returns 2 for usage output, which was accepted for that check.

Using only the embedded Python and the new GGUFs, CONFIGURE generated the local pack and tokenizer. `dense.bin`,
normalized index rows and all five tokenizer files matched the original v4 reference. All 48 expert-index rows
used IQ3_KT and matched the new main-shard offsets. The config explicitly selected the second shard for IQ4_KT PLE.
No `experts.bin` was generated. These checks are recorded in [runtime-validation.json](runtime-validation.json).

Seven focused tests passed on the development Python and again with the extracted embedded Python and packaged
modules. They cover missing or corrupt model files, split-path selection, configuration/key preservation,
wrong-PLE rejection, streamed timing extraction and reports without credentials or response text. Benchmark
protocol checks used synthetic responses; they are not a GPU throughput measurement.

No new model was loaded onto the production GPUs. The existing v4-lite service stayed healthy. There was no
physical RTX 4070 Ti SUPER / DDR4 target available, so the default configuration is a starting point and the
included BENCHMARK.bat must measure that computer's actual performance.

The runtime's base release does not include the later local 512K conversation-pool and GPU-wake changes.
Ordinary prefix caching is enabled. See [the user guide](../../../docs/WINDOWS_V4_LITE_4070TI.zh-CN.md).

The user also requested a storage review without deletion. Its snapshot and recommendations are in
[the cleanup review](../../../docs/STRATA_CLEANUP_REVIEW_2026-10-10.zh-CN.md) and
[storage-summary.json](storage-summary.json). The repeated release/pack validation copies total about 12.71 GiB;
rebuildable caches add about 3.01 GiB. No user files were deleted.
