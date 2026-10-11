"""Small, offline checks for lossless metadata and separated GGUF payloads."""
from pathlib import Path
import struct
import tempfile
import unittest

from gguf_reader import GGUFFile, TensorInfo
import export_v4_lite_gguf as e


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-gguf-export-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.assertTrue(self.root.name.startswith("strata-gguf-export-"))
        self.temp.cleanup()

    def fixture(self):
        entries = {
            "general.architecture": e.meta_entry("general.architecture", "qwen4exp", 8),
            "general.name": e.meta_entry("general.name", "original", 8),
            "general.file_type": e.meta_entry("general.file_type", 152, 4),
            "rope.scale": e.gguf_string("rope.scale") + struct.pack("<If", 6, 1.171875),
            "tokenizer.ggml.tokens": e.gguf_string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2)
                                     + e.gguf_string("中文") + e.gguf_string("<|test|>"),
            "split.no": e.meta_entry("split.no", 0, 2),
            "split.count": e.meta_entry("split.count", 1, 2),
            "split.tensors.count": e.meta_entry("split.tensors.count", 3, 5),
        }
        specs = [TensorInfo("norm.weight", [4], 0, "F32", 0),
                 TensorInfo("blk.0.ffn_gate_exps.weight", [256, 2, 2], 154, "IQ3_KT", 32),
                 TensorInfo(e.PLE, [160, 3], 155, "IQ4_KT", 448)]
        # Build the source header independently of make_header/write_shard.
        head = bytearray(struct.pack("<IIQQ", e.GGUF_MAGIC, 3, len(specs), len(entries)))
        head += b"".join(entries.values())
        offset = 0
        payloads = {}
        for t in specs:
            t.offset = offset
            head += e.gguf_string(t.name) + struct.pack("<I", len(t.shape))
            head += struct.pack(f"<{len(t.shape)}Q", *t.shape) + struct.pack("<IQ", t.type_id, offset)
            data = bytes((i * 31 + len(t.name)) % 256 for i in range(e.tensor_bytes(t)))
            payloads[t.name] = data
            offset = (offset + len(data) + 31) // 32 * 32
        head += bytes(-len(head) % 32)
        path = self.root / "source.gguf"
        with path.open("wb") as f:
            f.write(head)
            for t in specs:
                f.write(bytes(len(head) + t.offset - f.tell()))
                f.write(payloads[t.name])
        return GGUFFile(path), entries, payloads

    def test_split_preserves_metadata_types_tokens_and_tensor_payloads(self):
        source, entries, payloads = self.fixture()
        table = e.inventory([source])
        main = self.root / "test-00001-of-00002.gguf"
        ple = self.root / "test-00002-of-00002.gguf"
        a = e.write_shard(main, [table[n] for n in payloads if n != e.PLE], e.shard_metadata(source, 0, 3))
        b = e.write_shard(ple, [table[e.PLE]], e.shard_metadata(source, 1, 3))
        checked = [e.verify_shard(main, a), e.verify_shard(ple, b)]
        raw = e.raw_metadata(checked[0])
        for key, value in entries.items():
            if key not in e.REPLACED_META:
                self.assertEqual(raw[key], value)
        self.assertEqual(checked[0].metadata["tokenizer.ggml.tokens"], ["中文", "<|test|>"])
        for part, g in enumerate(checked):
            self.assertEqual(g.metadata["split.no"], part)
            self.assertEqual(g.metadata["split.count"], 2)
            self.assertEqual(g.metadata["split.tensors.count"], 3)
            with g.path.open("rb") as f:
                for t in g.tensors:
                    f.seek(g.data_start+t.offset)
                    self.assertEqual(f.read(e.tensor_bytes(t)), payloads[t.name])
        self.assertEqual(ple.stat().st_size, checked[1].data_start + 252)

    def test_corrupt_payload_fails_verification(self):
        source, _, _ = self.fixture()
        path = self.root / "output.gguf"
        table = e.inventory([source])
        record = e.write_shard(path, [table[e.PLE]], e.shard_metadata(source, 1, 3))
        with path.open("r+b") as f:
            f.seek(-1, 2)
            value = f.read(1)
            f.seek(-1, 2)
            f.write(bytes([value[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "Payload mismatch"):
            e.verify_shard(path, record)

    def test_existing_outputs_are_preserved(self):
        source, _, _ = self.fixture()
        path = self.root / "keep.gguf"
        path.write_bytes(b"existing")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            e.write_shard(path, [(source, source.tensors[0])], e.shard_metadata(source, 0, 3))
        self.assertEqual(path.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
