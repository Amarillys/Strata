"""Read-only, complete payload comparison of v2/v4 routed experts and routers.

Unlike the earlier five-window probe, every byte of every same-type tensor in
scope is compared. Different-type tensors are inventoried, not requantized.
The GGUFs are opened read-only; this script does not contact the live service.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from gguf_reader import GGUFFile

PATTERNS = {
    "v2": "*ik_llama-IQ3_KT_v2-*.gguf",
    "v4": "*ik_llama-v4-IQ3_KT-*.gguf",
}


def inventory(directory, pattern):
    tensors, files = {}, []
    for path in sorted(directory.glob(pattern)):
        gguf = GGUFFile(path)
        stat = path.stat()
        metadata = {k: v for k, v in gguf.metadata.items()
                    if k.startswith("quantize.imatrix.") or k in (
                        "general.name", "general.architecture",
                        "general.quantization_version", "general.file_type")}
        # Keep the useful calibration filename without the publisher's home path.
        if "quantize.imatrix.file" in metadata:
            metadata["quantize.imatrix.file"] = Path(metadata["quantize.imatrix.file"]).name
        files.append({"filename": path.name, "bytes": stat.st_size,
                      "mtime_ns": stat.st_mtime_ns, "metadata": metadata})
        ordered = sorted(gguf.tensors, key=lambda t: t.offset)
        for i, tensor in enumerate(ordered):
            size = tensor.expected_bytes()
            assert size is not None and size > 0, tensor.name
            start = gguf.data_start + tensor.offset
            end_limit = (gguf.data_start + ordered[i + 1].offset
                         if i + 1 < len(ordered) else stat.st_size)
            assert start + size <= end_limit, tensor.name
            assert tensor.name not in tensors, tensor.name
            tensors[tensor.name] = {
                "file": path.name, "offset": start, "bytes": size,
                "shape": tensor.shape, "type": tensor.type_name,
            }
    assert len(files) == 2 and len(tensors) == 1224
    return tensors, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path,
                        default=Path("B:/models/llm/Qwen/Qwen3.8-Flash-Next"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("expert-identity.json"))
    args = parser.parse_args()
    started = time.monotonic()
    loaded = {tag: inventory(args.model_dir, pattern) for tag, pattern in PATTERNS.items()}
    v2, v4 = loaded["v2"][0], loaded["v4"][0]
    assert v2.keys() == v4.keys()
    assert all(v2[n]["shape"] == v4[n]["shape"] for n in v2)
    names = [n for n in v2 if n.endswith(("_exps.weight", ".ffn_gate_inp.weight"))]
    names.sort(key=lambda n: (v2[n]["file"], v2[n]["offset"]))
    assert len(names) == 192  # 48 layers * (gate, up, down, router)
    results, counts = [], collections.Counter()
    bytes_read = 0
    last_progress = started
    with contextlib.ExitStack() as stack:
        handles = {f["filename"]: stack.enter_context(
            (args.model_dir / f["filename"]).open("rb", buffering=0))
            for _, files in loaded.values() for f in files}
        for name in names:
            a, b = v2[name], v4[name]
            group = "expert" if name.endswith("_exps.weight") else "router"
            row = {"tensor": name, "group": group, "v2": dict(a), "v4": dict(b)}
            if a["type"] != b["type"]:
                row["result"] = "different_type"
            else:
                assert a["bytes"] == b["bytes"]
                h2, h4 = handles[a["file"]], handles[b["file"]]
                h2.seek(a["offset"])
                h4.seek(b["offset"])
                sha2, sha4 = hashlib.sha256(), hashlib.sha256()
                offset, first_difference = 0, None
                while offset < a["bytes"]:
                    size = min(4 * 1024 * 1024, a["bytes"] - offset)
                    data2, data4 = h2.read(size), h4.read(size)
                    assert len(data2) == len(data4) == size, name
                    sha2.update(data2)
                    sha4.update(data4)
                    if data2 != data4 and first_difference is None:
                        first_difference = offset + next(
                            i for i, (x, y) in enumerate(zip(data2, data4)) if x != y)
                    offset += size
                    bytes_read += 2 * size
                row["v2"]["payload_sha256"] = sha2.hexdigest()
                row["v4"]["payload_sha256"] = sha4.hexdigest()
                row["result"] = "equal" if first_difference is None else "different_payload"
                if first_difference is not None:
                    row["first_different_payload_byte"] = first_difference
            counts[f"{group}.{row['result']}"] += 1
            results.append(row)
            now = time.monotonic()
            if now - last_progress >= 15:
                print(json.dumps({"progress": f"{len(results)}/{len(names)}",
                                  "read_GiB": round(bytes_read / 2**30, 3),
                                  "counts": dict(counts)}), flush=True)
                last_progress = now
    for _, files in loaded.values():
        for f in files:
            stat = (args.model_dir / f["filename"]).stat()
            assert (stat.st_size, stat.st_mtime_ns) == (f["bytes"], f["mtime_ns"])
    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": __doc__, "model_directory": str(args.model_dir),
        "files": {tag: files for tag, (_, files) in loaded.items()},
        "counts": dict(counts), "read_bytes": bytes_read,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "file_sizes_and_mtimes_unchanged": True,
        "equal_expert_bytes_per_model": sum(r["v2"]["bytes"] for r in results
            if r["group"] == "expert" and r["result"] == "equal"),
        "tensors": results,
        "not_measured": ["whole-GGUF checksums", "original imatrix checksums",
                         "high-precision source identity", "candidate model quality"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("tensors", "files", "method")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
