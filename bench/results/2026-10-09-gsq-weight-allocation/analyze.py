"""Read GGUF headers and existing Strata pack indexes; do not load model weights.

Compare GSQ-RCO IQ3_XXS with local Uncensored IQ3_KT v2/v4 and validate the
publisher's pinned allocation against local tensor storage types. Runtime
accounting reuses the previously checked native-dense/stage-trimming model.
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gguf_reader import GGUFFile

spec = importlib.util.spec_from_file_location(
    "lite_accounting", HERE.parent / "2026-10-09-iqkt-lite-analysis/analyze.py")
accounting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accounting)

MODEL_DIR = Path("B:/models/llm/Qwen/Qwen3.8-Flash-Next")
MODELS = {
    "gsq_iq3_xxs": ("*GSQ-RCO-IQ3_XXS-*.gguf", "pack/iq3_xxs"),
    "v2": ("*ik_llama-IQ3_KT_v2-*.gguf", "pack/iq3kt"),
    "v4": ("*ik_llama-v4-IQ3_KT-*.gguf", "pack/iq3kt-v4"),
}
PUBLISHER_REVISION = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
PUBLISHER_BASE = "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/"
PUBLISHER_ALLOCATION = "tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.rco-allocation.txt"


def read_model(tag, pattern, pack):
    ts, idx, files = accounting.load(tag, pattern, pack)
    inventory = accounting.inventory(ts, idx)
    tensor_sources, metadata = {}, {}
    for f in files:
        path = MODEL_DIR / f["filename"]
        gguf = GGUFFile(path)
        with path.open("rb") as handle:
            f["header_sha256"] = hashlib.sha256(handle.read(gguf.data_start)).hexdigest()
        f["data_start"] = gguf.data_start
        f["tensor_count"] = len(gguf.tensors)
        for key, value in gguf.metadata.items():
            if not key.startswith(("tokenizer.", "split.")) and not isinstance(value, list):
                if key == "quantize.imatrix.file":
                    value = Path(value).name
                metadata[key] = value
        ordered = sorted(gguf.tensors, key=lambda t: t.offset)
        for i, t in enumerate(ordered):
            size = t.expected_bytes()
            end = (gguf.data_start + ordered[i + 1].offset
                   if i + 1 < len(ordered) else f["bytes"])
            assert size is not None and gguf.data_start + t.offset + size <= end
            tensor_sources[t.name] = {"file": path.name, "absolute_offset": gguf.data_start + t.offset}
    groups = {}
    for name, tensor in ts.items():
        role = accounting.role(name)
        group = groups.setdefault(role, {"count": 0, "bytes": 0, "types": {}})
        group["count"] += 1
        group["bytes"] += tensor.expected_bytes()
        kind = group["types"].setdefault(tensor.type_name, {"count": 0, "bytes": 0, "layers": []})
        kind["count"] += 1
        kind["bytes"] += tensor.expected_bytes()
        if accounting.layer(name) >= 0:
            kind["layers"].append(accounting.layer(name))
    for group in groups.values():
        for kind in group["types"].values():
            kind["layers"].sort()
    expert_elements = sum(t.elements for n, t in ts.items() if n.endswith("_exps.weight"))
    transformer_names = [n for n in ts if n != "per_layer_token_embd.weight"]
    transformer_elements = sum(ts[n].elements for n in transformer_names)
    transformer_bytes = sum(ts[n].expected_bytes() for n in transformer_names)
    result = {
        "files": files, "metadata": metadata, "pack": pack,
        "type_counts": dict(collections.Counter(t.type_name for t in ts.values())),
        "roles": groups, "accounting": inventory,
        "expert_effective_bpw": 8 * inventory["expert_bytes"] / expert_elements,
        "transformer_payload_bpw": 8 * transformer_bytes / transformer_elements,
        "tensors": [{"name": n, "shape": t.shape, "type": t.type_name,
                     "bytes": t.expected_bytes(), **tensor_sources[n]} for n, t in sorted(ts.items())],
    }
    return ts, result


def main():
    loaded = {tag: read_model(tag, *args) for tag, args in MODELS.items()}
    gsq = loaded["gsq_iq3_xxs"][0]
    assert len(gsq) == 1224
    for tag in ("v2", "v4"):
        other = loaded[tag][0]
        assert gsq.keys() == other.keys()
        assert all(gsq[n].shape == other[n].shape for n in gsq)
    assert all(gsq[f"blk.{i}.ffn_gate_exps.weight"].type_id ==
               gsq[f"blk.{i}.ffn_up_exps.weight"].type_id for i in range(48))
    allocation_file = HERE / "publisher-allocation.txt"
    allocation = {}
    for line in allocation_file.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, kind = line.split(":", 1)
            assert name not in allocation
            allocation[name] = kind.strip()
    assert len(allocation) == 1223
    assert gsq.keys() - allocation.keys() == {"per_layer_token_embd.weight"}
    mismatches = [{"tensor": n, "publisher": kind, "local": gsq[n].type_name}
                  for n, kind in allocation.items() if gsq[n].type_name != kind]
    assert not mismatches
    previous = json.loads((HERE.parent / "2026-10-09-iqkt-lite-analysis/analysis.json").read_text(encoding="utf-8"))
    for tag in ("v2", "v4"):
        assert loaded[tag][1]["accounting"] == previous["variants"][tag]
    projection_types = collections.Counter(gsq[r["tensor"]].type_name for r in previous["upgrades_387"])
    assert projection_types == {"BF16": 387}
    per_layer = []
    for i in range(48):
        row = {"layer": i}
        for tag, (ts, _) in loaded.items():
            row[tag] = {"gate_up": ts[f"blk.{i}.ffn_gate_exps.weight"].type_name,
                        "down": ts[f"blk.{i}.ffn_down_exps.weight"].type_name,
                        "expert_bytes": sum(ts[f"blk.{i}.ffn_{role}_exps.weight"].expected_bytes()
                                            for role in ("gate", "up", "down"))}
        per_layer.append(row)
    result = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": __doc__, "models": {tag: data for tag, (_, data) in loaded.items()},
        "publisher": {
            "revision": PUBLISHER_REVISION,
            "readme_url": PUBLISHER_BASE + PUBLISHER_REVISION + "/README.md",
            "allocation_url": PUBLISHER_BASE + PUBLISHER_REVISION + "/" + PUBLISHER_ALLOCATION,
            "allocation_sha256": hashlib.sha256(allocation_file.read_bytes()).hexdigest(),
            "allocation_entries_compared": len(allocation), "allocation_mismatches": mismatches,
        },
        "checks": ["All 1224 tensor names and shapes match across models",
                   "All 1223 publisher allocation entries match local GSQ shard-1 types",
                   "The remaining shard-2 tensor is per_layer_token_embd.weight",
                   "v2/v4 GPU accounting exactly matches the earlier independent inventory",
                   "All 387 previously considered projection upgrades are BF16 in GSQ"],
        "per_layer_experts": per_layer,
        "not_verified": ["whole-file hashes or cross-model weight identity",
                         "GSQ/RCO search or per-tensor training reproduced locally",
                         "original calibration data or imatrix contents",
                         "publisher benchmark results reproduced locally"],
    }
    (HERE / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = ["# GSQ IQ3_XXS 与 v4 IQ3_KT 的逐层专家分配", "",
             "层号从 0 开始。每个 gate/up 类型同时用于该层两个张量，每个张量覆盖 512 个专家。",
             "v2 的全部 gate、up、down 均为 IQ3_KT。此表由 analyze.py 读取本地 GGUF 头部生成。", "",
             "| 层 | GSQ gate/up | GSQ down | v4 gate/up | v4 down |",
             "| ---: | --- | --- | --- | --- |"]
    for row in per_layer:
        a, b = row["gsq_iq3_xxs"], row["v4"]
        table.append(f"| {row['layer']} | {a['gate_up']} | {a['down']} | {b['gate_up']} | {b['down']} |")
    (HERE / "expert-layers.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    for tag, (_, data) in loaded.items():
        inv = data["accounting"]
        print(json.dumps({"model": tag,
                          "expert_GiB": inv["expert_bytes"] / 2**30,
                          "gpu_weight_payload_GiB_split19": sum(s["weight_payload_bytes"] for s in inv["splits"]["19"]) / 2**30,
                          "PLE_GiB": data["roles"]["per_layer_token_embd.weight"]["bytes"] / 2**30,
                          "file_GiB": sum(f["bytes"] for f in data["files"]) / 2**30,
                          "expert_effective_bpw": data["expert_effective_bpw"],
                          "transformer_payload_bpw": data["transformer_payload_bpw"]}))
    print(json.dumps({"checks": result["checks"], "publisher_revision": PUBLISHER_REVISION}))


if __name__ == "__main__":
    main()
