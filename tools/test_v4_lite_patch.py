"""Offline checks for streamed GGUF overlays and recipient config isolation."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from gguf_reader import GGUFFile, TensorInfo
import v4_lite_patch as patch


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-patch-")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.assertTrue(self.root.name.startswith("strata-patch-"))
        self.tmp.cleanup()

    def source(self, specs):
        items = [(None, TensorInfo(name, shape, typ, {154: "IQ3_KT", 155: "IQ4_KT"}[typ], 0))
                 for name, shape, typ in specs]
        header, offsets = patch.overlay_header(items, "synthetic source")
        path = self.root / "source.gguf"
        expected = {}
        with path.open("wb") as f:
            f.write(header)
            for (_, t), offset in zip(items, offsets):
                f.write(bytes(len(header) + offset - f.tell()))
                data = bytes((i * 37 + len(t.name)) % 256 for i in range(patch.tensor_bytes(t)))
                f.write(data)
                expected[t.name] = data
        return patch.inventory([GGUFFile(path)]), expected

    def test_kt_geometry_including_160_column_tail(self):
        for typ, shape, size in [(154, [2560, 640, 512], 328990720),
                                 (154, [640, 2560, 512], 335544320),
                                 (155, [160, 320001536], 26880129024)]:
            self.assertEqual(patch.tensor_bytes(TensorInfo("t", shape, typ, "KT", 0)), size)

    def test_stream_keeps_every_source_byte_and_rebases_offsets(self):
        tensors, expected = self.source([("other", [256, 2], 154),
                                         ("gate", [256, 3, 2], 154), ("up", [256, 3, 2], 154)])
        path = self.root / patch.EXPERT_FILE
        record = patch.write_overlay(path, [tensors["up"], tensors["gate"]], "expert patch")
        result = patch.check_overlay(self.root, record)
        self.assertEqual(set(result), {"up", "gate"})
        for name, (g, t) in result.items():
            with path.open("rb") as f:
                f.seek(g.data_start + t.offset)
                self.assertEqual(f.read(patch.tensor_bytes(t)), expected[name])
        self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_ple_has_no_trailing_padding_even_with_short_tail(self):
        tensors, expected = self.source([(patch.PLE, [160, 3], 155)])
        path = self.root / patch.PLE_FILE
        record = patch.write_overlay(path, [tensors[patch.PLE]], "PLE")
        g, t = patch.check_overlay(self.root, record)[patch.PLE]
        self.assertEqual(path.stat().st_size, g.data_start + 252)
        self.assertTrue(path.read_bytes().endswith(expected[patch.PLE]))
        self.assertEqual(t.offset, 0)

    def test_same_length_corruption_is_rejected(self):
        tensors, _ = self.source([("gate", [256, 2], 154)])
        path = self.root / patch.EXPERT_FILE
        record = patch.write_overlay(path, [tensors["gate"]], "test")
        with path.open("r+b") as f:
            f.seek(-1, 2)
            old = f.read(1)
            f.seek(-1, 2)
            f.write(bytes([old[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            patch.check_overlay(self.root, record)

    def test_truncated_source_and_existing_output_are_rejected(self):
        tensors, _ = self.source([("gate", [256, 2], 154)])
        path = self.root / patch.EXPERT_FILE
        path.write_bytes(b"keep this file")
        with self.assertRaisesRegex(ValueError, "already exists"):
            patch.write_overlay(path, [tensors["gate"]], "test")
        self.assertEqual(path.read_bytes(), b"keep this file")
        source = self.root / "source.gguf"
        with source.open("r+b") as f:
            f.truncate(source.stat().st_size - 1)
        with self.assertRaisesRegex(ValueError, "Truncated tensor"):
            patch.inventory([GGUFFile(source)])

    def test_mixed_role_paths_and_contiguous_expert_layout(self):
        native = self.root / "base with spaces" / "v4-1.gguf"
        other = native.with_name("v4-2.gguf")
        overlay = native.parent / "patch" / patch.EXPERT_FILE
        selected = {}
        for layer in range(2):
            for i, role in enumerate(("gate", "up", "down")):
                name = f"blk.{layer}.ffn_{role}_exps.weight"
                shape = [32, 256, 2] if role == "down" else [256, 32, 2]
                file = overlay if layer == 0 and role != "down" else native if i == 0 else other
                selected[name] = (SimpleNamespace(path=file, data_start=256),
                                  TensorInfo(name, shape, 154, "IQ3_KT", i * 65536))
        text, rows, size = patch.expert_index(selected, native, layers=2)
        data = [line.split() for line in text.splitlines() if not line.startswith("#")]
        self.assertEqual(len(rows), 6)
        self.assertEqual(data[0][8], f"patch/{patch.EXPERT_FILE},patch/{patch.EXPERT_FILE},v4-2.gguf")
        self.assertEqual(data[1][8], ",v4-2.gguf,v4-2.gguf")
        self.assertEqual(int(data[1][3]), int(data[0][4]) * 2)
        self.assertEqual(size, sum(int(line[4]) * 2 for line in data))
        with self.assertRaisesRegex(ValueError, "spaces/commas"):
            patch.source_name(native.parent / "bad folder" / patch.EXPERT_FILE, native)

    def test_config_preserves_machine_settings_and_replaces_all_dense_sources(self):
        cwd = self.root
        cfg = {"api_key": "test-private-placeholder", "gpu": [1, 0], "layer_split": "23",
               "tokenizer": "v4-tokenizer", "env": {"setting": "kept"}, "vision": {"model": "local-vision"},
               "args": ["--pack", "old", "--native", "old-gguf", "--native-dense-gguf", "wrong-ple",
                        "--native-dense-gguf", "old-dense", "--ple-gguf", "old-ple", "--no-ple",
                        "--gpu-reserve", "111", "--layer-split", "23"]}
        before = copy.deepcopy(cfg)
        native = [cwd / "v4-1.gguf", cwd / "v4-2.gguf"]
        result = patch.patched_config(cfg, cwd, cwd / "new-pack", native, native[0], cwd / patch.PLE_FILE,
                                      cwd / "new-config.json", "iq4")
        self.assertEqual(cfg, before)
        for key in ("api_key", "gpu", "layer_split", "env", "vision"):
            self.assertEqual(result[key], cfg[key])
        self.assertEqual([result["args"][i+1] for i, x in enumerate(result["args"]) if x == "--native-dense-gguf"],
                         list(map(str, native)))
        self.assertNotIn("--no-ple", result["args"])
        self.assertEqual(patch.argument(result["args"], "--gpu-reserve"), "111")
        self.assertEqual(patch.argument(result["args"], "--ple-gguf"), str(cwd / patch.PLE_FILE))


if __name__ == "__main__":
    unittest.main()
