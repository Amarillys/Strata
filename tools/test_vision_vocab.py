"""The upstream vocab loader must never see KT tensor descriptors or split references."""
import pathlib
import struct
import tempfile
import unittest

from gguf_reader import GGUFFile, GGUF_MAGIC
from vision_vocab import export_vocab


def string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


class VocabExport(unittest.TestCase):
    def test_exact_metadata_without_kt_or_splits(self):
        entries = [string("general.architecture") + struct.pack("<I", 8) + string("qwen3next"),
                   string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) + string("猫") + string("<|image_pad|>"),
                   string("general.alignment") + struct.pack("<II", 4, 64),
                   string("tokenizer.ggml.scores") + struct.pack("<IIQff", 9, 6, 2, 0.5, -2.5)]
        splits = [string("split.count") + struct.pack("<IH", 2, 2),
                  string("split.no") + struct.pack("<IH", 2, 0)]
        tensor = string("blk.0.experts") + struct.pack("<IQQIQ", 2, 256, 4, 154, 0)
        raw = struct.pack("<IIQQ", GGUF_MAGIC, 3, 1, 6) + b"".join(entries + splits) + tensor
        with tempfile.TemporaryDirectory() as d:
            src, dst = pathlib.Path(d) / "in.gguf", pathlib.Path(d) / "out.gguf"
            src.write_bytes(raw)
            export_vocab(src, dst)
            got = GGUFFile(dst)
            original = GGUFFile(src)
            self.assertEqual(got.tensors, [])
            self.assertEqual(got.metadata, {k: v for k, v in original.metadata.items() if not k.startswith("split.")})
            self.assertEqual(dst.read_bytes()[24:24 + sum(map(len, entries))], b"".join(entries))
            self.assertEqual(dst.stat().st_size % 64, 0)
            self.assertEqual(src.read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                export_vocab(src, src)
            with self.assertRaises(FileExistsError):
                export_vocab(src, dst)
            self.assertEqual(src.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
