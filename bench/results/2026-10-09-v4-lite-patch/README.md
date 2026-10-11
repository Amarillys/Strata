# Portable v4-lite expert patch

The recipient has the same two v4 GGUF shards and Strata, but no v2 model.
The delivered bundle contains 32 original v2 IQ3_KT gate/up tensors and reuses
112 experts proven byte-identical between v2 and v4. It keeps v4 dense weights,
including all 387 higher-precision projections, and defaults to the recipient's
original **v4 IQ3_KT PLE**. The optional 25 GiB IQ4 PLE is outside the shared bundle.

The shared expert GGUF is 10,527,705,568 bytes (9.80469 GiB), including its header.
The complete shared directory is 10,527,786,042 bytes, including scripts, manifests,
instructions and license files. It is located at:

```text
B:\models\llm\Qwen\Qwen3.8-Flash-Next\strata-v4-lite-patch-20261009
```

The source v4 files stay intact. Applying the patch creates a separate dense pack
and config, retaining the recipient's hardware and runtime settings. It reads both
v4 shards plus the patch, verifies full SHA256 hashes, and never requantizes.

Validation completed on 2026-10-09:

- Seven offline tests passed, covering byte-preserving streaming, KT row tails,
  corruption/truncation detection, refusal to overwrite an existing artifact,
  mixed-role shard paths and recipient config preservation.
- Both PLE choices were applied into independent local packs/configs. The IQ4
  application preceded the separation of the optional PLE file from the bundle.
- The final bundle, with IQ4 PLE absent, passed its default `--verify-only` command.
  A Python audit hook rejected attempts to open original v2 shards or the optional
  IQ4 PLE; none occurred. The script ran with `python -S`, without site packages.
- All 32 replacement payloads were independently hashed against the original v2
  tensors. The 112 reused experts retain the complete comparison evidence from
  the earlier same-day run, with unchanged source file sizes and modification times.
- All 144 index entries were checked against the actual selected GGUF tensor names,
  offsets, shapes, types and byte counts. The resulting expert payload is
  47,689,236,480 bytes, all IQ3_KT.
- Dense pack payload hashes matched v4; the config explicitly names both v4 dense
  shards and the original IQ3 PLE. Private settings stayed local and no API key is
  present in the shared text files.

This run validates packaging and weight identity. It did not load the candidate
onto GPUs or run another generation/quality benchmark. The production preset was
not switched; the final read-only service check is in `validation.json`.

- [Recipient instructions](../../../docs/V4_LITE_PATCH.zh-CN.md)
- [Prompt for the recipient's AI](../../../docs/V4_LITE_PATCH_AI_PROMPT.zh-CN.md)
- [Builder and installer](../../../tools/v4_lite_patch.py)
- [Offline tests](../../../tools/test_v4_lite_patch.py)
- [Bundle manifest](manifest.json)
- [Validation record](validation.json)
- [Final recipient audit](default-recipient-audit.json)
- [Checksums of delivered files](SHA256SUMS.txt)

Local test outputs are under `logs/v4-lite-share-20261009-01/`. Those local configs
inherit private credentials and are not part of the shareable bundle or this report.
