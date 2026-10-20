# Windows CUDA 13 IQ_KT release validation

2026-10-09. This validates the downloadable Windows package for the IQ_KT fork,
including the v4 IQ3_KT startup template. It is not an RTX 5080 benchmark.

Build: upstream Strata 0.1.41 plus the fork's KT and Vulkan changes, CUDA Toolkit
13.0.2 / nvcc 13.0.88, CUDA runtime 13.0.96, cuBLAS 13.1.0.3, MSVC 14.42.34433.
The CUDA binary contains native sm75, sm89 and sm120 code. Both the text engine
and Vulkan helper use the portable AVX2 baseline. The ggml/llama.cpp source is
`3cf03257f219afbe7334045ff7c6a06ac68c627d`.

## Package and numerical checks

- Extracted the actual ZIP into a directory containing spaces. Removed system
  Python, CUDA Toolkit, Vulkan SDK and compiler directories from child-process
  paths. The embedded Python 3.13.16 imported the bundled dependencies, read the
  model, exported its tokenizer and completed a new v4 pack in 36.0 seconds.
- Examined the import tables of 89 PE binaries/extensions in the package; all
  non-system dependencies were present. The real engine loaded cuBLAS 13 and
  cuBLASLt 13 from the extracted package. NVIDIA's `nvcudart_hybrid64.dll` comes
  from the display driver, as intended, and is not redistributed.
- 14 launcher tests passed: configuration validation, preserved local keys,
  spaces/Unicode in argument lists, pack completion/reuse/source binding,
  Intel device selection and safe dependency archive paths.
- CUDA KT parity on both sm75 and sm89: 38 oracle fixtures each, maximum relative
  reconstruction error `1.73e-7`, maximum GEMV error `4.70e-7`.
- Portable CPU KT AVX2: 38 fixtures, 1–9 tokens, gate/up/down and partial row
  ranges exactly matched the scalar path; real v4 layer 0 also passed.
- Optional Vulkan helper: the bundled first-run tool selected Intel Iris Xe by
  description and exported a 60-field, tensor-free vocabulary. A 512×512 image
  encoded in 3.137 seconds after 14.880 seconds of startup; all 256×2560 FP32
  embedding values were finite and repeating it reused the cached file.
  This checks the encoder path, not the 13900HX UHD GPU or general vision quality.

## Single-GPU functional check

Hardware actually used: i9-12900HK, 64 GiB DDR5 and RTX 4080 SUPER 32 GiB via
USB4, NVIDIA driver 610.88. Only this GPU ran the text model. A reserve of
17,400 MiB emulated the cache budget of a 16 GiB card with a 1,024 MiB reserve;
it is not a hardware-enforced 16 GiB memory limit. Its link measured 2.9 GB/s,
so these timings must not be extrapolated to an RTX 5080 / 13900HX.

v4 IQ3_KT, capacity 65,536, K8V8, 20,480 resident KV cells, prefill 1,024,
6 prefix checkpoints, resident experts, PLE SSD direct, MTP off, CPU affinity
auto and measured PCIe share. The cache held 4,216 experts (8.08 GiB), with
39.83 GiB in RAM including the prefill loan backup. Decode reported zero expert
blob reads from files. Web, `/v1/models`, authenticated generation and rejection
of unauthenticated API calls passed.

| Request | Prompt / reused | Output | Prompt time | Result |
| --- | ---: | ---: | ---: | --- |
| Arithmetic | 28 / 0 | 3 | 2.942 s | `42` |
| Notes with a release code | 3,186 / 0 | 10 | 49.679 s | `STRATA-5080-PASS` |
| Same notes repeated | 3,186 / 3,179 | 10 | 0.409 s | Same code |

These short outputs were a function check, not a sustained decode benchmark.
The notes are 180 lines of `Record i: This package contains the application,
tokenizer and local runtime.`, followed by the code and a question asking for it.
Generation used temperature 0 and reasoning effort `none`.

NVML sampling every two seconds saw a peak of 14,510 MiB on the test GPU.
During loading, available RAM briefly reached about 0.17 GiB; after READY and
during requests it stayed around 15 GiB. The 39.83 GiB CUDA host-pinning request
was refused; Windows working-set locking / VirtualLock successfully kept the
resident complement. This startup memory pressure is documented for users.

The `STRATA_UNBUFFERED_LOAD=1` diagnostic was ignored without an explicit
`--resident-budget-gib`, as the engine log and source specify. It is not part of
the shipped template. Repeating the functional check with that ignored variable
again passed generation, authentication and prefix reuse; it did not solve the
startup memory peak. No change to the inference engine was made for packaging.

The original v4 dual-GPU 256K/K8V8 service was restored on port 8080 after tests.

## Limits

No physical RTX 5080 or i9-13900HX was available. sm120 compilation, portable CPU
code and dependency closure are checked, but 5080 throughput and full 64K input
remain unmeasured. Capacity 65,536 was initialized; the longest input in this
release check was 3,186 tokens. The earlier dual-card 256K tests concern a
different memory budget. The small dense projections still have the packer's
recorded BF16/F16/F32 conversions; this is not a claim of whole-engine bitwise
identity with IK or universal quality equivalence.
