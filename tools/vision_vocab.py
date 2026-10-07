"""Export a vocab-only GGUF for strata-vision from an IK KT model's first shard.

Upstream gguf rejects IK tensor types even for vocab_only loads. Copy all metadata
byte for byte, except split.* (there are no weight shards to load), and no tensors.
The original model is opened read-only; an existing output is never overwritten.
"""
from __future__ import annotations

import argparse
import pathlib
import struct

from gguf_reader import GGUFFile, GGUF_MAGIC


def export_vocab(source: pathlib.Path, output: pathlib.Path) -> dict:
    source, output = pathlib.Path(source), pathlib.Path(output)
    reader = GGUFFile.__new__(GGUFFile)
    entries, dropped = [], []
    alignment = 32
    with source.open("rb") as f:
        magic, version, _, n_kv = struct.unpack("<IIQQ", f.read(24))
        if magic != GGUF_MAGIC or version != 3:
            raise ValueError("expected a little-endian GGUF v3")
        for _ in range(n_kv):
            start = f.tell()
            key = reader._str(f)
            value = reader._value(f)
            end = f.tell()
            if key.startswith("split."):
                dropped.append(key)
                continue
            if key == "general.alignment":
                alignment = int(value)
            f.seek(start)
            entries.append(f.read(end - start))
    if alignment <= 0 or alignment & (alignment - 1) or alignment > 1 << 20:
        raise ValueError(f"invalid GGUF alignment: {alignment}")
    # Exclusive creation also protects the source when output names it (or a hard link).
    with output.open("xb") as f:
        f.write(struct.pack("<IIQQ", GGUF_MAGIC, 3, 0, len(entries)))
        for entry in entries:
            f.write(entry)
        f.write(bytes((-f.tell()) % alignment))
    return {"output": str(output), "metadata": len(entries), "dropped": dropped,
            "bytes": output.stat().st_size}


if __name__ == "__main__":
    import json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=pathlib.Path, required=True, help="original model's first GGUF shard")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    a = ap.parse_args()
    print(json.dumps(export_vocab(a.model, a.output), ensure_ascii=False))
