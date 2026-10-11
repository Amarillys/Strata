"""Generate a Strata pack using only the two exported GGUFs, then check its contents."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from v4_lite_patch import hash_file, normalized_index  # noqa: E402


def read_json(path):
    return Path(path).read_text(encoding="utf-8-sig")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gguf-py", type=Path)
    args = parser.parse_args()
    model_dir, out = args.model_dir.resolve(), args.out.resolve()
    manifest = json.loads(read_json(model_dir / "manifest.json"))
    files = manifest["files"]
    require(len(files) == 2, "Expected exactly two exported GGUFs")
    allowed = {(model_dir / f["filename"]).resolve() for f in files}
    seen = set()

    def audit(event, event_args):
        if event == "open" and isinstance(event_args[0], (str, bytes)):
            path = Path(os.fsdecode(event_args[0]))
            if path.suffix.lower() == ".gguf":
                path = path.resolve()
                require(path in allowed, "A tool tried to open a GGUF outside the new model: " + path.name)
                seen.add(path)
        elif event == "subprocess.Popen":
            raise RuntimeError("This verification runs both pack tools in-process; a child process is unexpected")

    sys.addaudithook(audit)
    if args.gguf_py:
        os.environ["STRATA_GGUF_PY"] = str(args.gguf_py.resolve())

    def run_tool(name, arguments):
        previous = sys.argv
        sys.argv = [str(ROOT / "tools" / name), *map(str, arguments)]
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise RuntimeError(f"{name} failed: {error.code}") from error
        finally:
            sys.argv = previous

    first = model_dir / files[0]["filename"]
    run_tool("strata_tokenizer.py", ["--gguf", first, "--out", out])
    run_tool("iq_pack.py", ["--gguf", first, "--out", out, "--compat-bf16"])
    require(seen == allowed, "The two exported model shards were not both opened")
    require(not (out / "experts.bin").exists(), "Native pack should not need experts.bin")

    reference = json.loads(read_json(ROOT / "bench/results/2026-10-09-v4-lite-patch/manifest.json"))["v4_pack"]
    checked_files = []
    for record in reference["files"]:
        path = out / record["filename"]
        digest = hash_file(path)
        require(path.stat().st_size == record["bytes"] and digest == record["sha256"],
                "Canonical v4 pack mismatch: " + path.name)
        checked_files.append({"filename": path.name, "bytes": path.stat().st_size, "sha256": digest})
    index_hash = hashlib.sha256(normalized_index(out / "index.txt").encode()).hexdigest()
    require(index_hash == reference["index_rows_sha256"], "Canonical v4 index rows differ")

    composition = json.loads(read_json(ROOT / "bench/results/2026-10-09-v4-lite/composition.json"))
    tokenizers = []
    for record in composition["copied_v4_files"]:
        name = record["file"]
        if name.startswith("tokenizer/") and not name.endswith("vision-vocab.gguf"):
            path = out / name
            digest = hash_file(path)
            require(path.stat().st_size == record["bytes"] and digest == record["v4_sha256"],
                    "Tokenizer mismatch: " + name)
            tokenizers.append({"filename": name, "sha256": digest, "bytes": path.stat().st_size})
    require(len(tokenizers) == 5, "Expected five tokenizer files")

    tensors = {record["name"]: record for record in files[0]["tensors"]}
    rows = [line.split() for line in (out / "native_experts.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]
    require(len(rows) == 48, "Expected 48 expert layers")
    total, count = 0, 0
    for expected_layer, row in enumerate(rows):
        require(len(row) == 8, "Experts unexpectedly refer to another shard")
        layer, gu_type, down_type, offset, blob, gate, up, down = map(int, row)
        require(layer == expected_layer and gu_type == down_type == 154,
                "Wrong layer or expert quantization")
        require(offset == total, "Expert blob layout is not contiguous")
        layer_bytes = 0
        for role, actual_offset in zip(("gate", "up", "down"), (gate, up, down)):
            tensor = tensors[f"blk.{layer}.ffn_{role}_exps.weight"]
            require(tensor["type_id"] == 154 and tensor["shape"][2] == 512,
                    "Wrong expert type or count")
            require(actual_offset == tensor["output_offset"], "Expert offset mismatch")
            layer_bytes += tensor["bytes"]
            count += 1
        require(blob * 512 == layer_bytes, "Expert blob size mismatch")
        total += layer_bytes
    require(total == 47689236480 and count == 144, "Wrong total expert payload")

    result = {
        "passed": True,
        "gguf_open_audit": "only the two exported GGUFs permitted; child processes forbidden",
        "opened_ggufs": sorted(path.name for path in seen),
        "original_models_required": False,
        "gguf_files": [{key: f[key] for key in ("filename", "bytes", "sha256")} for f in files],
        "canonical_v4_files": checked_files,
        "canonical_v4_index_rows_sha256": index_hash,
        "v4_tokenizer_files": tokenizers,
        "native_experts": {"layers": 48, "tensors": count, "experts_per_layer": 512,
                           "type": "IQ3_KT", "bytes": total, "all_offsets_match_export": True,
                           "all_in_first_shard": True,
                           "index_sha256": hash_file(out / "native_experts.txt")},
        "experts_bin_required": False,
        "pack_bytes": sum(path.stat().st_size for path in out.rglob("*") if path.is_file()),
        "gpu_inference_tested_with_new_files": False,
    }
    (out.parent / "portable-pack-validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
