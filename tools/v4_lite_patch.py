"""Build/apply a portable, byte-preserving v4 -> v4-lite Strata overlay.

Python 3.10+ and the adjacent gguf_reader.py are sufficient; no numpy/GPU is used.
Application creates a new pack/config. It does not change a model, preset catalog,
or running service. The generated config inherits the recipient's private settings.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import time

from gguf_reader import GGUFFile, GGUF_MAGIC

CHUNK = 8 * 1024 * 1024
PLE = "per_layer_token_embd.weight"
LAYERS = tuple(range(6)) + tuple(range(38, 48))
REPLACEMENTS = tuple(f"blk.{l}.ffn_{r}_exps.weight" for l in LAYERS for r in ("gate", "up"))
EXPERT_FILE = "v2-experts-iq3kt.gguf"
PLE_FILE = "v2-ple-iq4kt.gguf"
FORMAT = "strata-v4-lite-overlay-1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def fingerprint(path):
    s = Path(path).stat()
    return s.st_size, s.st_mtime_ns


def hash_file(path):
    path = Path(path)
    before = fingerprint(path)
    h = hashlib.sha256()
    at, last = 0, time.monotonic()
    with path.open("rb") as f:
        while data := f.read(CHUNK):
            h.update(data)
            at += len(data)
            if time.monotonic() - last >= 15:
                print(f"HASH {path.name}: {at / 2**30:.1f}/{before[0] / 2**30:.1f} GiB", flush=True)
                last = time.monotonic()
    require(fingerprint(path) == before, f"File changed while hashing: {path.name}")
    return h.hexdigest()


def tensor_bytes(t):
    # Row geometry from tools/kt_quants.py (ik_llama MIT; license in the bundle).
    # Keeping only the byte-count formula avoids a numpy dependency in the installer.
    require(t.shape and all(d > 0 for d in t.shape), f"Invalid shape: {t.name}")
    if t.type_id in (154, 155):
        n = t.shape[0]
        require(n % 32 == 0, f"Invalid KT row width: {t.name}")
        tail = n % 256 // 32
        rb = (4 + n // 256 * (100 if t.type_id == 154 else 128)
              + ((tail + 1) // 2 + 12 * tail if t.type_id == 154 else 16 * tail) + 3) & ~3
        return math.prod(t.shape[1:]) * rb
    size = t.expected_bytes()
    require(size is not None, f"Unsupported tensor type: {t.name}, {t.type_id}")
    return size


def inventory(files):
    tensors = {}
    for g in files:
        end = g.data_start
        require(0 < g.alignment <= 4096 and g.alignment & (g.alignment - 1) == 0,
                f"Invalid GGUF alignment: {g.path.name}")
        require(end <= g.path.stat().st_size, f"Truncated GGUF header: {g.path.name}")
        for t in sorted(g.tensors, key=lambda t: t.offset):
            start = g.data_start + t.offset
            size = tensor_bytes(t)
            require(t.name not in tensors, f"Duplicate tensor: {t.name}")
            require(t.offset % g.alignment == 0 and start >= end,
                    f"Overlapping or unaligned tensor: {t.name}")
            require(start + size <= g.path.stat().st_size, f"Truncated tensor: {t.name}")
            tensors[t.name] = (g, t)
            end = start + size
    return tensors


def model(first):
    first = Path(first).resolve()
    require("00001-of-00002" in first.name, "Pass shard 00001-of-00002 of the original v4 GGUF")
    files = [GGUFFile(first.with_name(first.name.replace("00001-of-00002", f"{i:05d}-of-00002")))
             for i in (1, 2)]
    require(files[0].metadata.get("general.architecture") == "qwen4exp", "Wrong model architecture")
    tensors = inventory(files)
    require(len(tensors) == 1224, "Expected the unmodified 1,224-tensor model")
    return files, tensors


def file_record(g):
    size, mtime = fingerprint(g.path)
    with g.path.open("rb") as f:
        header = hashlib.sha256(f.read(g.data_start)).hexdigest()
    return {"filename": g.path.name, "bytes": size, "mtime_ns": mtime, "header_sha256": header}


def check_profile(v4, v2=None):
    for layer in range(48):
        for role in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{role}_exps.weight"
            t = v4[name][1]
            shape = [640, 2560, 512] if role == "down" else [2560, 640, 512]
            require(t.shape == shape, f"Unexpected expert geometry: {name}")
            require(t.type_id == (155 if name in REPLACEMENTS else 154), f"Not the supported v4: {name}")
            if v2 is not None:
                require(v2[name][1].type_id == 154, f"Not the supported v2 IQ3_KT expert: {name}")
    require(v4[PLE][1].shape == [160, 320001536] and v4[PLE][1].type_id == 154,
            "Expected the original v4 IQ3_KT PLE")
    if v2 is not None:
        require(v2.keys() == v4.keys() and all(v2[n][1].shape == v4[n][1].shape for n in v4),
                "v2/v4 tensor names or shapes differ")
        require(v2[PLE][1].type_id == 155, "Expected the v2 IQ4_KT PLE")


def payload_hash(g, t):
    h = hashlib.sha256()
    remain = tensor_bytes(t)
    with g.path.open("rb") as f:
        f.seek(g.data_start + t.offset)
        while remain:
            b = f.read(min(CHUNK, remain))
            require(bool(b), f"Truncated source: {t.name}")
            h.update(b)
            remain -= len(b)
    return h.hexdigest()


def identical_experts(v4files, v2files, v4, v2, evidence_path=None):
    prior = {}
    if evidence_path:
        evidence = read_json(evidence_path)
        for label, files in (("v4", v4files), ("v2", v2files)):
            old = {r["filename"]: r for r in evidence["files"][label]}
            for g in files:
                r = old.get(g.path.name, {})
                require(fingerprint(g.path) == (r.get("bytes"), r.get("mtime_ns")),
                        "Source changed since the complete expert comparison; omit --identity to compare again")
        prior = {r["tensor"]: r for r in evidence["tensors"] if r["group"] == "expert"}
    rows = []
    for name in sorted(n for n in v4 if n.endswith("_exps.weight") and n not in REPLACEMENTS):
        hashes = []
        for label, table in (("v4", v4), ("v2", v2)):
            g, t = table[name]
            if evidence_path:
                old = prior[name]
                r = old[label]
                require(old["result"] == "equal" and r["file"] == g.path.name
                        and r["offset"] == g.data_start + t.offset and r["bytes"] == tensor_bytes(t)
                        and r["shape"] == t.shape and r["type"] == t.type_name,
                        f"Expert identity evidence does not match: {name}")
                value = r["payload_sha256"]
            else:
                value = payload_hash(g, t)
            require(re.fullmatch(r"[0-9a-f]{64}", value), "Invalid payload hash")
            hashes.append(value)
        require(hashes[0] == hashes[1], f"A same-type expert changed; cannot omit it: {name}")
        rows.append({"name": name, "sha256": hashes[0], "bytes": tensor_bytes(v4[name][1])})
    require(len(rows) == 112, "Expected 112 byte-identical experts")
    return rows


def gguf_string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def overlay_header(items, title):
    meta = {"general.architecture": "qwen4exp", "general.name": title}
    header = bytearray(struct.pack("<IIQQ", GGUF_MAGIC, 3, len(items), len(meta)))
    for key, value in meta.items():
        header += gguf_string(key) + struct.pack("<I", 8) + gguf_string(value)
    offsets, offset = [], 0
    for _, t in items:
        offsets.append(offset)
        header += gguf_string(t.name) + struct.pack("<I", len(t.shape))
        header += struct.pack(f"<{len(t.shape)}Q", *t.shape) + struct.pack("<IQ", t.type_id, offset)
        offset = (offset + tensor_bytes(t) + 31) // 32 * 32
    header += bytes(-len(header) % 32)
    return bytes(header), offsets


def write_overlay(path, items, title):
    """Stream existing tensor bytes, including row metadata; never dequantize."""
    path = Path(path).resolve()
    partial = path.with_name(path.name + ".partial")
    require(not path.exists() and not partial.exists(), f"Output already exists: {path}")
    header, offsets = overlay_header(items, title)
    total_hash, records = hashlib.sha256(), []
    with partial.open("xb") as dst:
        def write(data):
            dst.write(data)
            total_hash.update(data)
        write(header)
        for (g, t), offset in zip(items, offsets):
            write(bytes(len(header) + offset - dst.tell()))
            remain, h = tensor_bytes(t), hashlib.sha256()
            print(f"COPY {t.name}: {remain / 2**30:.3f} GiB", flush=True)
            before = fingerprint(g.path)
            last, copied = time.monotonic(), 0
            with g.path.open("rb") as src:
                src.seek(g.data_start + t.offset)
                while remain:
                    data = src.read(min(CHUNK, remain))
                    require(bool(data), f"Truncated source: {t.name}")
                    write(data)
                    h.update(data)
                    copied += len(data)
                    remain -= len(data)
                    if time.monotonic() - last >= 15:
                        print(f"COPY {t.name}: {copied / 2**30:.1f} GiB written", flush=True)
                        last = time.monotonic()
            require(fingerprint(g.path) == before, f"Source changed while copying: {g.path.name}")
            records.append({"name": t.name, "shape": t.shape, "type_id": t.type_id,
                            "type": t.type_name, "offset": len(header) + offset,
                            "bytes": tensor_bytes(t), "sha256": h.hexdigest()})
        # No trailing padding: a one-tensor PLE file must end exactly at its last row.
        dst.flush()
        os.fsync(dst.fileno())
    partial.rename(path)
    return {"filename": path.name, "bytes": path.stat().st_size,
            "sha256": total_hash.hexdigest(), "tensors": records}


def normalized_index(path):
    return "\n".join(" ".join(line.split()) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
                     if line.strip() and not line.lstrip().startswith("#")) + "\n"


def pack_reference(pack):
    pack = Path(pack)
    require((pack / "dense.bin").is_file() and (pack / "index.txt").is_file(), "Missing v4 native pack")
    files = [{"filename": name, "bytes": (pack / name).stat().st_size, "sha256": hash_file(pack / name)}
             for name in ("dense.bin", "extra.bin", "embd.bin") if (pack / name).is_file()]
    return {"files": files, "index_rows_sha256": hashlib.sha256(normalized_index(pack / "index.txt").encode()).hexdigest()}


def build(a):
    out = Path(a.out).resolve()
    require(not out.exists(), "Bundle directory already exists; choose a fresh output")
    v4files, v4 = model(a.v4)
    v2files, v2 = model(a.v2)
    check_profile(v4, v2)
    source_stats = {g.path: fingerprint(g.path) for g in v4files + v2files}
    same = identical_experts(v4files, v2files, v4, v2, a.identity)
    cache = read_json(a.v4_hashes) if a.v4_hashes else None
    base_records = []
    for i, g in enumerate(v4files):
        record = file_record(g)
        if cache is not None:
            require(all(cache[i].get(k) == v for k, v in record.items()), "Stale v4 full-hash cache")
            record["sha256"] = cache[i]["sha256"]
        else:
            record["sha256"] = hash_file(g.path)
        base_records.append(record)
    canonical = pack_reference(a.v4_pack)
    names = (*REPLACEMENTS, PLE) if a.include_ple else REPLACEMENTS
    needed = sum(tensor_bytes(v2[n][1]) for n in names)
    require(out.parent.is_dir(), "Create the parent directory first")
    require(shutil.disk_usage(out.parent).free > needed + 256 * 2**20, "Insufficient free disk space for the bundle")
    out.mkdir()
    experts = write_overlay(out / EXPERT_FILE, [v2[n] for n in REPLACEMENTS], "v4-lite: v2 IQ3_KT expert replacements")
    weights = {"experts": experts}
    if a.include_ple:
        weights["ple_iq4"] = write_overlay(out / PLE_FILE, [v2[PLE]], "v4-lite: v2 IQ4_KT PLE table")
    here = Path(__file__).resolve().parent
    root = here.parent
    texts = {
        "v4_lite_patch.py": Path(__file__).resolve(),
        "gguf_reader.py": here / "gguf_reader.py",
        "README.zh-CN.md": root / "docs/V4_LITE_PATCH.zh-CN.md",
        "AI_PROMPT.zh-CN.md": root / "docs/V4_LITE_PATCH_AI_PROMPT.zh-CN.md",
        "LICENSE-Strata.txt": root / "LICENSE",
        "LICENSE-ik_kt.txt": root / "third_party/ik_kt/LICENSE",
    }
    for name, source in texts.items():
        shutil.copyfile(source, out / name)
    require(all(fingerprint(path) == stat for path, stat in source_stats.items()), "A model changed during export")
    manifest = {
        "format": FORMAT,
        "method": "Original v2 tensor bytes; no requantization or calibration. Reuse 112 byte-identical v4 experts.",
        "v4_shards": base_records,
        "v2_sources": [file_record(g) for g in v2files],
        "v4_pack": canonical,
        "unchanged_experts": same,
        "weights": weights,
        "default_ple": "iq3",
        "included_components": list(weights),
        "expected_expert_bytes": sum(tensor_bytes(v2[n][1]) for n in v2 if n.endswith("_exps.weight")),
        "components": {"dense": "v4, including all 387 higher-precision projections", "experts": "v2 IQ3_KT",
                       "ple_iq4": "v2 IQ4_KT (optional)", "ple_iq3": "original v4 IQ3_KT (default)"},
    }
    write_json(out / "manifest.json", manifest)
    sums = {r["filename"]: r["sha256"] for r in weights.values()}
    sums.update({name: hash_file(out / name) for name in (*texts, "manifest.json")})
    with (out / "SHA256SUMS.txt").open("x", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"{value}  {name}\n" for name, value in sums.items()))
    print(json.dumps({"bundle": str(out), "experts_gib": experts["bytes"] / 2**30,
                      "included_iq4_ple_gib": weights.get("ple_iq4", {}).get("bytes", 0) / 2**30,
                      "default_ple": "iq3", "v2_not_needed_by_recipient": True}), flush=True)


def check_file(path, record):
    path = Path(path)
    require(path.is_file() and path.stat().st_size == record["bytes"], f"Missing or wrong-size file: {path.name}")
    print(f"VERIFY {path.name}", flush=True)
    require(hash_file(path) == record["sha256"], f"SHA256 mismatch: {path.name}")


def check_overlay(bundle, record):
    name = record["filename"]
    require(name in (EXPERT_FILE, PLE_FILE), "Unexpected overlay filename")
    path = Path(bundle) / name
    check_file(path, record)
    g = GGUFFile(path)
    tensors = inventory([g])
    require(len(tensors) == len(record["tensors"]), "Overlay tensor count mismatch")
    require(set(tensors) == {r["name"] for r in record["tensors"]}, "Overlay tensor names mismatch")
    for r in record["tensors"]:
        t = tensors[r["name"]][1]
        require(t.type_id == r["type_id"] and t.shape == r["shape"]
                and g.data_start + t.offset == r["offset"] and tensor_bytes(t) == r["bytes"],
                f"Overlay directory mismatch: {t.name}")
    if set(tensors) == {PLE}:
        t = tensors[PLE][1]
        require(t.offset == 0 and g.data_start + tensor_bytes(t) == path.stat().st_size,
                "The PLE file must end exactly at its last row")
    return tensors


def argument(args, flag):
    at = [i for i, x in enumerate(args) if x == flag]
    require(len(at) == 1 and at[0] + 1 < len(args), f"Config must have exactly one {flag} value")
    return args[at[0] + 1]


def set_values(args, flag, values):
    result, i = [], 0
    while i < len(args):
        if args[i] == flag:
            require(i + 1 < len(args), f"Missing value for {flag}")
            i += 2
        else:
            result.append(args[i])
            i += 1
    for value in values:
        result.extend((flag, str(value)))
    return result


def source_name(path, native):
    path, native = Path(path).resolve(), Path(native).resolve()
    if path == native:
        return ""
    try:
        rel = Path(os.path.relpath(path, native.parent)).as_posix()
    except ValueError:
        raise ValueError("Place the bundle on the same drive as v4, preferably in its model directory") from None
    require(not any(c.isspace() for c in rel) and "," not in rel,
            "The expert index cannot contain spaces/commas: put the bundle beside v4 in a simply named folder")
    return rel


def expert_index(selected, native, layers=48):
    lines = ["# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard | gate,up,down]",
             "# v4-lite overlay; unchanged experts use v4, replacements use the bundled v2 bytes."]
    total, records = 0, []
    for layer in range(layers):
        items = [selected[f"blk.{layer}.ffn_{r}_exps.weight"] for r in ("gate", "up", "down")]
        n_expert = items[0][1].shape[2]
        require(all(t.type_id == 154 and t.shape[2] == n_expert for _, t in items), "All resulting experts must be IQ3_KT")
        require(items[0][1].shape == items[1][1].shape
                and items[2][1].shape[:2] == list(reversed(items[0][1].shape[:2])), "Inconsistent expert geometry")
        sizes = [tensor_bytes(t) for _, t in items]
        require(all(size % n_expert == 0 for size in sizes), "Invalid per-expert byte count")
        blob = sum(sizes) // n_expert
        offsets = [g.data_start + t.offset for g, t in items]
        sources = [source_name(g.path, native) for g, _ in items]
        columns = [layer, 154, 154, total, blob, *offsets]
        if any(sources):
            columns.append(sources[0] if len(set(sources)) == 1 else ",".join(sources))
        lines.append(" ".join(map(str, columns)))
        for (g, t), size, offset, source in zip(items, sizes, offsets, sources):
            records.append({"name": t.name, "source": source or Path(native).name,
                            "offset": offset, "bytes": size, "type_id": t.type_id})
        total += sum(sizes)
    return "\n".join(lines) + "\n", records, total


def patched_config(cfg, cwd, pack, native_files, head_file, ple_file, out_config, ple_kind):
    result = copy.deepcopy(cfg)
    args = [x for x in result["args"] if x != "--no-ple"]
    for flag, values in {"--pack": [pack], "--native": [native_files[0]],
                         "--native-dense-gguf": native_files, "--native-head-gguf": [head_file],
                         "--ple-gguf": [ple_file]}.items():
        args = set_values(args, flag, values)
    result["args"] = args
    result["cwd"] = str(cwd)
    result["tokenizer"] = str((cwd / cfg["tokenizer"]).resolve())
    result["model_name"] = f"qwen3.8-flash-next-v4-lite-{ple_kind}ple"
    result["aliases"] = [result["model_name"]]
    result["log"] = str(cwd / "logs" / (out_config.stem + ".log"))
    return result


def apply(a):
    bundle = Path(a.bundle or Path(__file__).resolve().parent).resolve()
    manifest = read_json(bundle / "manifest.json")
    require(manifest.get("format") == FORMAT, "Unsupported patch format")
    cfg_path = Path(a.config).resolve()
    cfg = read_json(cfg_path)
    require(isinstance(cfg.get("args"), list) and all(isinstance(x, str) for x in cfg["args"]), "Config args must be a string list")
    cwd = (cfg_path.parent / (a.cwd or cfg.get("cwd", "."))).resolve()
    first = (cwd / argument(cfg["args"], "--native")).resolve()
    old_pack = (cwd / argument(cfg["args"], "--pack")).resolve()
    out = Path(a.out_pack).resolve() if a.out_pack else None
    out_config = Path(a.out_config).resolve() if a.out_config else None
    if not a.verify_only:
        require(out is not None and out_config is not None, "--out-pack and --out-config are required")
        require(not out.exists() and not out_config.exists(), "Refusing to overwrite an existing pack/config")
        require(out.parent.is_dir() and out_config.parent.is_dir(), "Output parent directories must exist")
        require(not out_config.is_relative_to(out) and not out.is_relative_to(bundle)
                and not out_config.is_relative_to(bundle), "Keep local outputs outside the shareable bundle")
    v4files, v4 = model(first)
    check_profile(v4)
    require(len(manifest["v4_shards"]) == 2, "Invalid base-shard manifest")
    # Cheap header/size checks precede the complete hashes.
    for g, record in zip(v4files, manifest["v4_shards"]):
        current = file_record(g)
        require(all(current[k] == record[k] for k in ("bytes", "header_sha256")), "This is a different v4 GGUF; patch refused")
    for g, record in zip(v4files, manifest["v4_shards"]):
        check_file(g.path, record)
    reference = pack_reference(old_pack)
    require(reference == manifest["v4_pack"],
            "The v4 dense pack differs. Rebuild it from this v4 with tools/iq_pack.py --compat-bf16, then retry; do not bypass this check")
    replaced = check_overlay(bundle, manifest["weights"]["experts"])
    require(set(replaced) == set(REPLACEMENTS), "The patch must contain exactly the 32 changed gate/up tensors")
    for name, (_, t) in replaced.items():
        require(t.type_id == 154 and t.shape == v4[name][1].shape, f"Wrong expert replacement: {name}")
    unchanged = {r["name"] for r in manifest["unchanged_experts"]}
    require(unchanged == {n for n in v4 if n.endswith("_exps.weight") and n not in replaced}, "Incomplete unchanged-expert inventory")
    if a.ple == "iq4":
        require("ple_iq4" in manifest["weights"], "This bundle has no IQ4 PLE component; use iq3 or obtain an IQ4 PLE bundle")
        ple = check_overlay(bundle, manifest["weights"]["ple_iq4"])
        require(set(ple) == {PLE} and ple[PLE][1].type_id == 155 and ple[PLE][1].shape == v4[PLE][1].shape,
                "Wrong IQ4_KT PLE replacement")
        ple_file = ple[PLE][0].path
    else:
        ple_file = v4[PLE][0].path
    selected = {**v4, **replaced}
    text, expert_records, total = expert_index(selected, first)
    require(total == manifest["expected_expert_bytes"], "Resulting expert byte count mismatch")
    result = {"verified": True, "ple": a.ple, "expert_tensors": len(expert_records),
              "replaced_expert_tensors": len(replaced), "reused_v4_expert_tensors": len(unchanged),
              "expert_bytes": total, "service_changed": False}
    if a.verify_only:
        print(json.dumps(result), flush=True)
        return
    require(shutil.disk_usage(out.parent).free > sum(r["bytes"] for r in reference["files"]) + 64 * 2**20,
            "Insufficient disk space for a separate dense pack")
    new_cfg = patched_config(cfg, cwd, out, [g.path for g in v4files], v4["output.weight"][0].path,
                             ple_file, out_config, a.ple)
    require((Path(new_cfg["tokenizer"]) / "vocab.json").is_file(), "Configured v4 tokenizer is missing")
    out.mkdir()
    for record in reference["files"]:
        name = record["filename"]
        require(name in ("dense.bin", "extra.bin", "embd.bin"), "Unexpected dense pack file")
        shutil.copyfile(old_pack / name, out / name)
        check_file(out / name, record)
    shutil.copyfile(old_pack / "index.txt", out / "index.txt")
    require(hashlib.sha256(normalized_index(out / "index.txt").encode()).hexdigest() == reference["index_rows_sha256"],
            "Copied dense index failed validation")
    write_json(out / "patch-provenance.json", {**result, "bundle": str(bundle),
               "manifest_sha256": hash_file(bundle / "manifest.json"), "v4_shards": [str(g.path) for g in v4files],
               "dense_source": str(old_pack), "ple_source": str(ple_file), "experts": expert_records})
    with (out / "native_experts.txt").open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    write_json(out_config, new_cfg)
    print(json.dumps({**result, "pack": str(out), "config": str(out_config),
                      "note": "The new config contains your local private settings. Keep it out of the shared bundle."}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="Sender only: export original v2 tensor bytes")
    for flag in ("v4", "v2", "v4-pack", "out"):
        b.add_argument("--" + flag, required=True)
    b.add_argument("--identity", help="Optional existing complete expert comparison; source stats must still match")
    b.add_argument("--v4-hashes", help="Optional full v4 hash records; source stats and headers must still match")
    b.add_argument("--include-ple", action="store_true", help="Also export the optional 25 GiB v2 IQ4_KT PLE table")
    a = sub.add_parser("apply", help="Recipient: validate and create an independent pack/config")
    a.add_argument("--bundle", help="Default: the directory containing this script")
    a.add_argument("--config", required=True, help="Recipient's existing v4 Strata JSON config")
    a.add_argument("--cwd", help="Override the working directory only if absent/wrong in the original config")
    a.add_argument("--out-pack")
    a.add_argument("--out-config")
    a.add_argument("--ple", choices=("iq3", "iq4"), default="iq3", help="Default iq3 keeps v4's table; iq4 needs the optional PLE file")
    a.add_argument("--verify-only", action="store_true", help="Validate all inputs without creating a pack/config")
    args = ap.parse_args()
    try:
        (build if args.command == "build" else apply)(args)
    except (ValueError, OSError, KeyError, struct.error) as e:
        ap.exit(1, f"Patch refused: {e}\n")


if __name__ == "__main__":
    main()
