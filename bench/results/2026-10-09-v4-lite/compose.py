"""Create a local hybrid pack without requantizing or changing its source models.

The generated config contains the existing local API key and stays git-ignored.
Only the provenance manifest is suitable for publication. This script does not
change the preset catalog or contact/restart the service.
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from gguf_reader import GGUFFile
from switch_preset import Presets, argument, resolved

OUT = Path(__file__).resolve().parent
PACK = ROOT / "pack/iq3kt-v4-lite"
CONFIG = ROOT / "strata-iq3kt-v4-lite-k8v8-pf2048-vision-igpu.json"
PLE = "per_layer_token_embd.weight"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def set_values(args, key, values):
    result = []
    it = iter(args)
    for value in it:
        if value == key:
            next(it)
        else:
            result.append(value)
    for value in values:
        result += [key, str(value)]
    return result


def model(first):
    files, tensors, records = {}, {}, []
    for i in (1, 2):
        path = Path(str(first).replace("00001-of-00002", f"{i:05d}-of-00002"))
        g = GGUFFile(path)
        files[path.name] = g
        for t in g.tensors:
            size = t.expected_bytes()
            if size is None or g.data_start + t.offset + size > path.stat().st_size:
                raise RuntimeError(f"Invalid tensor extent: {path.name}: {t.name}")
            if t.name in tensors:
                raise RuntimeError(f"Duplicate tensor: {t.name}")
            tensors[t.name] = (g, t)
        with path.open("rb") as f:
            header_hash = hashlib.sha256(f.read(g.data_start)).hexdigest()
        stat = path.stat()
        records.append({"filename": path.name, "bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns, "header_sha256": header_hash,
                        "full_file_sha256": None})
    if len(tensors) != 1224:
        raise RuntimeError("Unexpected model tensor count")
    return files, tensors, records


def main():
    if PACK.exists() or CONFIG.exists():
        raise RuntimeError("Candidate already exists; refusing to overwrite it")
    p = Presets(ROOT / "strata-models.json")
    first = {name: resolved(ROOT, argument(p.configs[name]["args"], "--native")) for name in ("v2", "v4")}
    packs = {name: resolved(ROOT, argument(p.configs[name]["args"], "--pack")) for name in ("v2", "v4")}
    models = {name: model(path) for name, path in first.items()}
    v2files, v2, _ = models["v2"]
    _, v4, _ = models["v4"]
    if v2.keys() != v4.keys() or any(v2[n][1].shape != v4[n][1].shape for n in v4):
        raise RuntimeError("The source models do not have matching tensor shapes")
    evidence_path = ROOT / "bench/results/2026-10-09-iqkt-lite-analysis/expert-identity.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    # The full comparison is already available; require unchanged file stats before reusing it.
    for name in ("v2", "v4"):
        prior = evidence["files"][name]
        for record in models[name][2]:
            old = next(row for row in prior if row["filename"] == record["filename"])
            if old["bytes"] != record["bytes"] or old["mtime_ns"] != record["mtime_ns"]:
                raise RuntimeError("A source changed since the full expert comparison")

    rows = ["# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard | gate,up,down]",
            "# Hybrid: all 144 expert tensors are read from explicit v2 shards; --native remains v4."]
    expert_records = []
    total = 0
    seen = set()
    for line in (packs["v2"] / "native_experts.txt").read_text(encoding="utf-8").splitlines():
        a = line.split()
        if not a or a[0].startswith("#"):
            continue
        if len(a) not in (8, 9):
            raise RuntimeError("Unexpected native expert index row")
        layer, gu, down, packed_offset, blob, *offsets = map(int, a[:8])
        if layer in seen or gu != 154 or down != 154 or packed_offset != total:
            raise RuntimeError("Invalid v2 expert layout")
        seen.add(layer)
        sources = a[8].split(",") if len(a) == 9 else [""]
        if len(sources) == 1:
            sources *= 3
        if len(sources) != 3:
            raise RuntimeError("Invalid role sources")
        sources = [name or first["v2"].name for name in sources]
        layer_bytes = 0
        for role, source, offset in zip(("gate", "up", "down"), sources, offsets):
            if source not in v2files:
                raise RuntimeError("An expert source is not one of the two v2 shards")
            name = f"blk.{layer}.ffn_{role}_exps.weight"
            g, t = v2[name]
            if g.path.name != source or g.data_start + t.offset != offset or t.type_id != 154:
                raise RuntimeError(f"Expert source/offset/type mismatch: {name}")
            size = t.expected_bytes()
            layer_bytes += size
            expert_records.append({"tensor": name, "file": source, "offset": offset,
                                   "bytes": size, "type": t.type_name, "shape": t.shape})
        if layer_bytes != blob * 512:
            raise RuntimeError("Expert blob size does not match the source tensors")
        total += layer_bytes
        shard_column = sources[0] if len(set(sources)) == 1 else ",".join(sources)
        rows.append(" ".join(a[:8] + [shard_column]))
    if seen != set(range(48)) or len(expert_records) != 144 or total != 47689236480:
        raise RuntimeError("Incomplete v2 expert inventory")

    analysis = json.loads((ROOT / "bench/results/2026-10-09-iqkt-lite-analysis/analysis.json").read_text(encoding="utf-8"))
    upgrades = analysis["upgrades_387"]
    if len(upgrades) != 387 or any(v4[r["tensor"]][1].type_name != r["v4_type"] for r in upgrades):
        raise RuntimeError("The v4 projection inventory changed")
    # Copy, never hardlink: candidate edits must not mutate the original v4 pack.
    PACK.mkdir(parents=True)
    copied = []
    for source in sorted(packs["v4"].rglob("*")):
        if not source.is_file() or source.name == "native_experts.txt":
            continue
        rel = source.relative_to(packs["v4"])
        target = PACK / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hash, target_hash = digest(source), digest(target)
        if source_hash != target_hash:
            raise RuntimeError(f"Copy hash mismatch: {rel}")
        copied.append({"file": rel.as_posix(), "bytes": source.stat().st_size,
                       "v4_sha256": source_hash, "hybrid_sha256": target_hash})
    index = PACK / "native_experts.txt"
    index.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ple_file, ple_tensor = v2[PLE]
    if ple_tensor.type_name != "IQ4_KT":
        raise RuntimeError("Expected the v2 IQ4_KT PLE table")

    cfg = copy.deepcopy(p.configs["v4"])
    cfg["model_name"] = "qwen3.8-flash-next-v4-lite-iq3kt"
    cfg["aliases"] = ["qwen3.8-flash-next-iq3kt", "qwen3.8-flash-next-v4-iq3kt"]
    cfg["tokenizer"] = str(PACK / "tokenizer")
    cfg["layer_split"] = "19"
    cfg["log"] = str(ROOT / "logs/strata-v4-lite.log")
    if "vision" in cfg:
        cfg["vision"]["model"] = str(PACK / "tokenizer/vision-vocab.gguf")
    for key, values in {
        "--pack": [PACK], "--native": [first["v4"]],
        "--native-dense-gguf": [g.path for g in models["v4"][0].values()],
        "--native-head-gguf": [first["v4"]], "--ple-gguf": [ple_file.path],
        "--layer-split": [19],
    }.items():
        cfg["args"] = set_values(cfg["args"], key, values)
    write_json(CONFIG, cfg)
    v4_expert_bytes = sum(t.expected_bytes() for _, t in v4.values() if t.name.endswith("_exps.weight"))
    manifest = {
        "created": datetime.datetime.now().astimezone().isoformat(),
        "name": "v4-lite", "pack": str(PACK), "config": str(CONFIG),
        "method": "Byte-preserving source selection; no requantization or new calibration.",
        "sources": {name: models[name][2] for name in ("v2", "v4")},
        "dense_source": "v4", "native_dense_sources": [g.path.name for g in models["v4"][0].values()],
        "head_tokenizer_embedding_source": "v4", "ple_projection_source": "v4",
        "expert_source": "v2", "expert_count": len(expert_records), "expert_bytes": total,
        "expert_saving_vs_v4_bytes": v4_expert_bytes - total,
        "ple_table": {"source": "v2", "file": ple_file.path.name, "type": ple_tensor.type_name,
                      "shape": ple_tensor.shape, "offset": ple_file.data_start + ple_tensor.offset,
                      "bytes": ple_tensor.expected_bytes(),
                      "increase_vs_v4_bytes": ple_tensor.expected_bytes() - v4[PLE][1].expected_bytes()},
        "copied_v4_files": copied, "native_experts_sha256": digest(index),
        "expert_tensors": expert_records, "included_v4_upgrades": upgrades,
        "full_expert_comparison_evidence": str(evidence_path.relative_to(ROOT)),
        "full_expert_comparison_sha256": digest(evidence_path),
        "gpu_weight_payload_splits": analysis["variants"]["v4_iq3_experts"]["splits"],
        "limitations": ["No full GGUF checksums; headers, tensor extents and prior file stats checked.",
                        "No inference/quality/residency results are implied by this composition manifest.",
                        "The hybrid references original GGUFs; originals must remain at their paths."]}
    write_json(PACK / "hybrid-provenance.json", manifest)
    write_json(OUT / "composition.json", manifest)
    p.paths["v4-lite"] = CONFIG
    p.configs["v4-lite"] = cfg
    p.validate("v4-lite")
    print(json.dumps({"pack": str(PACK), "config": str(CONFIG), "experts_checked": len(expert_records),
                      "v4_upgrades": len(upgrades), "copied_files": len(copied),
                      "expert_saving_gib": (v4_expert_bytes - total) / 2**30,
                      "ple_increase_gib": manifest["ple_table"]["increase_vs_v4_bytes"] / 2**30}, ensure_ascii=False))


if __name__ == "__main__":
    main()
