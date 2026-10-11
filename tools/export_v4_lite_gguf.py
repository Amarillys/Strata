"""Export v4 dense + v2 IQ3_KT experts, with v2 IQ4_KT PLE as shard 2.

Copies quantized payloads without dequantization. Original metadata records are
preserved byte-for-byte except the model name and split bookkeeping. Both output
files are ordinary GGUF v3 shards; they do not reference the original models.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import time

from gguf_reader import GGUFFile, GGUF_MAGIC
from v4_lite_patch import CHUNK, PLE, check_profile, file_record, fingerprint, gguf_string, inventory, model, require, tensor_bytes

STEM = "Qwen3.8-Flash-Next-V4-Lite-IQ3_KT-IQ4PLE"
REPLACED_META = {"general.name", "split.no", "split.count", "split.tensors.count"}


def raw_metadata(g):
    """Keep GGUF scalar/array types, UTF-8 strings and float bits intact."""
    with g.path.open("rb") as f:
        header = f.read(g.data_start)
    f = io.BytesIO(header)
    magic, version, _, count = struct.unpack("<IIQQ", f.read(24))
    require(magic == GGUF_MAGIC and version == 3, "Expected GGUF v3")
    records = {}
    for _ in range(count):
        begin = f.tell()
        key = g._str(f)
        g._value(f)
        require(key not in records, f"Duplicate metadata key: {key}")
        records[key] = header[begin:f.tell()]
    return records


def meta_entry(key, value, type_id):
    value_bytes = gguf_string(value) if type_id == 8 else struct.pack({2: "<H", 4: "<I", 5: "<i"}[type_id], value)
    return gguf_string(key) + struct.pack("<I", type_id) + value_bytes


def shard_metadata(original, part, total_tensors):
    if part == 0:
        entries = {k: value for k, value in raw_metadata(original).items() if k not in REPLACED_META}
        title = "Qwen3.8 Flash Next V4 Lite - IQ3_KT experts, IQ4_KT PLE"
        entries["strata.composition"] = meta_entry("strata.composition", "v4 non-experts; v2 IQ3_KT experts; v2 IQ4_KT PLE", 8)
    else:
        entries = {"general.architecture": meta_entry("general.architecture", "qwen4exp", 8),
                   "general.file_type": meta_entry("general.file_type", 153, 4)}
        title = "Qwen3.8 Flash Next V4 Lite - IQ4_KT PLE only"
    entries["general.name"] = meta_entry("general.name", title, 8)
    entries["split.no"] = meta_entry("split.no", part, 2)
    entries["split.count"] = meta_entry("split.count", 2, 2)
    entries["split.tensors.count"] = meta_entry("split.tensors.count", total_tensors, 5)
    return entries


def make_header(items, metadata, alignment=32):
    require(alignment == 32, "This export uses the source model's 32-byte GGUF alignment")
    header = bytearray(struct.pack("<IIQQ", GGUF_MAGIC, 3, len(items), len(metadata)))
    header += b"".join(metadata.values())
    offset, offsets = 0, []
    names = set()
    for _, t in items:
        require(t.name not in names, f"Duplicate output tensor: {t.name}")
        names.add(t.name)
        offsets.append(offset)
        header += gguf_string(t.name) + struct.pack("<I", len(t.shape))
        header += struct.pack(f"<{len(t.shape)}Q", *t.shape) + struct.pack("<IQ", t.type_id, offset)
        offset = (offset + tensor_bytes(t) + alignment - 1) // alignment * alignment
    header += bytes(-len(header) % alignment)
    return bytes(header), offsets


def write_shard(path, items, metadata):
    path = Path(path).resolve()
    partial = path.with_name(path.name + ".partial")
    require(not path.exists() and not partial.exists(), f"Refusing to overwrite: {path}")
    header, offsets = make_header(items, metadata)
    expected_size = len(header) + offsets[-1] + tensor_bytes(items[-1][1])
    digest, records = hashlib.sha256(), []
    last = time.monotonic()
    print(f"WRITE {path.name}: {len(items)} tensors, {expected_size / 2**30:.4f} GiB", flush=True)
    with partial.open("xb") as dst:
        def write(data):
            dst.write(data)
            digest.update(data)
        write(header)
        for index, ((g, t), offset) in enumerate(zip(items, offsets)):
            write(bytes(len(header) + offset - dst.tell()))
            size = remaining = tensor_bytes(t)
            tensor_hash = hashlib.sha256()
            with g.path.open("rb") as src:
                src.seek(g.data_start + t.offset)
                while remaining:
                    data = src.read(min(CHUNK, remaining))
                    require(bool(data), f"Truncated source tensor: {t.name}")
                    write(data)
                    tensor_hash.update(data)
                    remaining -= len(data)
                    if time.monotonic() - last >= 15:
                        print(f"WRITE {path.name}: {dst.tell() / 2**30:.1f}/{expected_size / 2**30:.1f} GiB, tensor {index+1}/{len(items)}", flush=True)
                        last = time.monotonic()
            records.append({"name": t.name, "shape": t.shape, "type_id": t.type_id, "type": t.type_name,
                            "bytes": size, "output_offset": len(header) + offset,
                            "source_file": g.path.name, "source_offset": g.data_start + t.offset,
                            "payload_sha256": tensor_hash.hexdigest()})
        # A single-tensor PLE GGUF must end at its last row, without trailing padding.
        require(dst.tell() == expected_size, "Wrong output size")
        dst.flush()
        os.fsync(dst.fileno())
    partial.rename(path)
    return {"filename": path.name, "bytes": expected_size, "sha256": digest.hexdigest(),
            "header_bytes": len(header), "tensors": records}


def verify_shard(path, record):
    """One sequential pass checks the complete file and every selected payload."""
    path = Path(path)
    require(path.stat().st_size == record["bytes"], f"Wrong file size: {path.name}")
    g = GGUFFile(path)
    table = inventory([g])
    require(set(table) == {r["name"] for r in record["tensors"]}, "Output tensor names changed")
    require(len(table) == len(record["tensors"]), "Wrong output tensor count")
    digest = hashlib.sha256()
    last = time.monotonic()
    with path.open("rb") as f:
        def read(count, tensor_hash=None):
            while count:
                data = f.read(min(CHUNK, count))
                require(bool(data), "Truncated output while verifying")
                digest.update(data)
                if tensor_hash is not None:
                    tensor_hash.update(data)
                count -= len(data)
        for r in record["tensors"]:
            t = table[r["name"]][1]
            require(t.shape == r["shape"] and t.type_id == r["type_id"]
                    and g.data_start + t.offset == r["output_offset"] and tensor_bytes(t) == r["bytes"],
                    f"Wrong tensor directory: {t.name}")
            require(r["output_offset"] >= f.tell(), "Output tensor order changed")
            read(r["output_offset"] - f.tell())
            tensor_hash = hashlib.sha256()
            read(r["bytes"], tensor_hash)
            require(tensor_hash.hexdigest() == r["payload_sha256"], f"Payload mismatch: {t.name}")
            if time.monotonic() - last >= 15:
                print(f"VERIFY {path.name}: {f.tell() / 2**30:.1f}/{record['bytes'] / 2**30:.1f} GiB", flush=True)
                last = time.monotonic()
        require(f.tell() == record["bytes"] and not f.read(1), "Unexpected bytes after the last tensor")
    require(digest.hexdigest() == record["sha256"], f"File checksum mismatch: {path.name}")
    return g


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def export(a):
    out = Path(a.out).resolve()
    require(not out.exists() and out.parent.is_dir(), "Choose a new output directory inside an existing parent")
    v4files, v4 = model(a.v4)
    v2files, v2 = model(a.v2)
    check_profile(v4, v2)
    require(all(g.alignment == 32 for g in v4files + v2files), "Unexpected source alignment")
    before = {g.path: fingerprint(g.path) for g in v4files + v2files}
    # Preserve the original directory order: this also retains dense pack layout.
    main = [v2[t.name] if t.name.endswith("_exps.weight") else (g, t)
            for g in v4files for t in g.tensors if t.name != PLE]
    groups = [main, [v2[PLE]]]
    require(len(main) == 1223, "Wrong main tensor count")
    needed = sum(tensor_bytes(t) for group in groups for _, t in group)
    require(shutil.disk_usage(out.parent).free > needed + 256 * 2**20, "Insufficient free disk space")
    out.mkdir()
    records = []
    for part, items in enumerate(groups):
        path = out / f"{STEM}-{part+1:05d}-of-00002.gguf"
        records.append(write_shard(path, items, shard_metadata(v4files[0], part, 1224)))
    require(all(fingerprint(path) == value for path, value in before.items()), "A source model changed during export")
    checked = [verify_shard(out / r["filename"], r) for r in records]
    require([len(g.tensors) for g in checked] == [1223, 1], "Wrong shard distribution")
    for i, g in enumerate(checked):
        require(g.metadata["split.no"] == i and g.metadata["split.count"] == 2
                and g.metadata["split.tensors.count"] == 1224, "Wrong GGUF split metadata")
    source_meta, result_meta = raw_metadata(v4files[0]), raw_metadata(checked[0])
    preserved = [key for key in source_meta if key not in REPLACED_META]
    require(all(source_meta[key] == result_meta[key] for key in preserved), "Original metadata bytes changed")
    actual = inventory(checked)
    require(actual.keys() == v4.keys(), "An original tensor is missing")
    require(actual[PLE][0].path == checked[1].path and checked[1].tensors[0].offset == 0,
            "The second shard must hold only the PLE table at data offset zero")
    upgrades_path = Path(__file__).resolve().parents[1] / "bench/results/2026-10-09-iqkt-lite-analysis/analysis.json"
    upgrades = json.loads(upgrades_path.read_text(encoding="utf-8-sig"))["upgrades_387"]
    require(len(upgrades) == 387 and all(actual[r["tensor"]][1].type_name == r["v4_type"] for r in upgrades),
            "The 387 v4 projection formats were not preserved")
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(), "format": "gguf-v3-split-model",
        "composition": {"main": "all v4 non-experts; all v2 IQ3_KT routed experts", "ple": "v2 IQ4_KT"},
        "original_models_required_at_runtime": False, "requantized": False, "new_calibration": False,
        "source_files": {"v4": [file_record(g) for g in v4files], "v2": [file_record(g) for g in v2files]},
        "preserved_raw_metadata_keys": preserved, "v4_high_precision_projections": len(upgrades),
        "expert_tensors": 144, "expert_type": "IQ3_KT", "ple_type": "IQ4_KT",
        "tensor_payloads_and_complete_files_verified": True, "files": records,
        "export_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(out / "manifest.json", manifest)
    readme = Path(__file__).resolve().parents[1] / "docs/V4_LITE_STANDALONE_GGUF.zh-CN.md"
    shutil.copyfile(readme, out / "README.zh-CN.md")
    sums = {r["filename"]: r["sha256"] for r in records}
    for name in ("manifest.json", "README.zh-CN.md"):
        sums[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    with (out / "SHA256SUMS.txt").open("x", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"{value}  {name}\n" for name, value in sums.items()))
    print(json.dumps({"directory": str(out), "files": [{k: r[k] for k in ("filename", "bytes", "sha256")} for r in records],
                      "total_gib": sum(r["bytes"] for r in records) / 2**30, "all_1224_tensor_payloads_verified": True}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v4", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        export(a)
    except (ValueError, OSError, KeyError, struct.error) as e:
        ap.exit(1, f"Export refused: {e}\n")


if __name__ == "__main__":
    main()
