"""Offline tests for the release's config, pack ownership and launch boundaries."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portable as P
from prepare_dependencies import archive_path


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strata release ")
        self.root = Path(self.tmp.name)
        template = self.root / "tools/windows/templates/v4-5080-16g.json"
        template.parent.mkdir(parents=True)
        shutil.copy2(P.ROOT / "tools/windows/templates/v4-5080-16g.json", template)
        self.cfg = P.read_json(template)
        self.cfg["api_key"] = "unit-test-only-key"
        self.first = self.root / "模型 文件-00001-of-00002.gguf"
        self.first.write_bytes(b"fixture")
        self.source = {"schema": 1, "shards": [{"name": self.first.name, "bytes": 7}]}

    def tearDown(self):
        self.tmp.cleanup()

    def options(self, **overrides):
        values = dict(config="strata-v4.json", model=str(self.first), pack=None, context=None,
                      prefill=None, kv=None, reserve_mib=None, gpu=None, host=None, port=None)
        values.update(overrides)
        return argparse.Namespace(**values)

    def markers(self, pack):
        for name in P.MARKERS:
            path = pack / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")

    def test_template_defaults_and_no_personal_device_or_mtp(self):
        P.validate_config(self.cfg)
        self.assertEqual(P.value_of(self.cfg["args"], "--max-context"), "65536")
        self.assertEqual(P.value_of(self.cfg["args"], "--kv"), "int8")
        self.assertNotIn("vision", self.cfg)
        self.assertNotIn("--mtp-dir", self.cfg["args"])
        self.assertNotIn("CUDA_VISIBLE_DEVICES", self.cfg["env"])

    def test_bad_budget_and_port_rejected(self):
        for flag, value in (("--max-context", 262145), ("--prefill", "auto"), ("--kv-resident", 1)):
            cfg = json.loads(json.dumps(self.cfg))
            P.set_arg(cfg["args"], flag, value)
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                P.validate_config(cfg)
        self.cfg["port"] = 0
        with self.assertRaises(ValueError):
            P.validate_config(self.cfg)

    def test_lan_without_key_is_rejected(self):
        self.cfg.update(host="0.0.0.0", api_key=" ")
        with self.assertRaises(ValueError):
            P.validate_config(self.cfg)

    def test_configure_keeps_key_and_custom_settings_on_repeat(self):
        with mock.patch.object(P, "inspect_model", return_value=self.source), \
                mock.patch.object(P, "prepare_pack"), contextlib.redirect_stdout(io.StringIO()) as out:
            P.configure(self.options(host="0.0.0.0", port=8099), self.root)
            cfg1 = P.read_json(self.root / "strata-v4.json")
            P.configure(self.options(context=131072), self.root)
            cfg2 = P.read_json(self.root / "strata-v4.json")
        self.assertEqual(cfg1["api_key"], cfg2["api_key"])
        self.assertNotIn(cfg1["api_key"], out.getvalue())
        self.assertEqual(cfg2["host"], "0.0.0.0")
        self.assertEqual(cfg2["port"], 8099)
        self.assertEqual(P.value_of(cfg2["args"], "--max-context"), "131072")
        self.assertEqual(P.value_of(cfg2["args"], "--native"), self.first.name)

    def test_pack_command_preserves_unicode_and_spaces(self):
        pack = self.root / "pack with spaces"
        with mock.patch.object(P.subprocess, "run", side_effect=lambda *a, **kw: self.markers(pack)) as run:
            P.prepare_pack(self.first, pack, self.source, self.root)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--gguf") + 1], str(self.first))
        self.assertIn("--compat-bf16", command)
        self.assertNotIn("--experts-bin", command)
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_completed_owned_pack_is_reused(self):
        pack = self.root / "pack"
        self.markers(pack)
        P.write_json(pack / "portable-source.json", self.source)
        with mock.patch.object(P.subprocess, "run") as run:
            P.prepare_pack(self.first, pack, self.source, self.root)
        run.assert_not_called()

    def test_incomplete_pack_can_resume_with_same_source(self):
        pack = self.root / "pack"
        P.write_json(pack / "portable-source.json", self.source)
        with mock.patch.object(P.subprocess, "run", side_effect=lambda *a, **kw: self.markers(pack)) as run:
            P.prepare_pack(self.first, pack, self.source, self.root)
        run.assert_called_once()

    def test_unowned_and_mismatched_pack_are_not_overwritten(self):
        pack = self.root / "pack"
        self.markers(pack)
        with self.assertRaises(ValueError), mock.patch.object(P.subprocess, "run") as run:
            P.prepare_pack(self.first, pack, self.source, self.root)
        run.assert_not_called()
        P.write_json(pack / "portable-source.json", {"schema": 1, "shards": []})
        with self.assertRaises(ValueError):
            P.prepare_pack(self.first, pack, self.source, self.root)

    def test_second_shard_rejected_before_packing(self):
        with self.assertRaisesRegex(ValueError, "第一分片"):
            P.inspect_model(self.first.with_name("v4-00002-of-00002.gguf"))

    def test_intel_selection_uses_description(self):
        output = "Vulkan0: NVIDIA RTX 5080\nVulkan3: Intel(R) UHD Graphics\nCPU: Intel i9\n"
        self.assertEqual(P.intel_device(output), "Intel(R) UHD Graphics")

    def test_missing_or_ambiguous_intel_does_not_choose_nvidia(self):
        for output in ("Vulkan0: NVIDIA RTX 5080\nCPU: Intel i9", "Vulkan0: Intel A\nVulkan1: Intel B"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                P.intel_device(output)

    def test_runtime_environment_ignores_external_python(self):
        with mock.patch.dict(os.environ, {"PYTHONHOME": "bad", "PYTHONPATH": "bad"}):
            env = P.runtime_env(self.root)
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertTrue(env["PATH"].startswith(str(self.root / "engine")))

    def test_start_passes_port_and_config_without_shell(self):
        self.markers(self.root / "packs/v4-iq3kt")
        (self.root / "engine").mkdir()
        (self.root / "engine/strata.exe").write_bytes(b"fixture")
        P.set_arg(self.cfg["args"], "--native", self.first.name)
        self.cfg["port"] = 8099
        P.write_json(self.root / "strata-v4.json", self.cfg)
        with mock.patch.object(P.subprocess, "call", return_value=0) as run:
            P.start(argparse.Namespace(config="strata-v4.json", no_browser=True), self.root)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--port") + 1], "8099")
        self.assertNotIn("--open", command)
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_dependency_zip_paths_cannot_escape(self):
        for path in ("../bad", "/bad", "C:/bad", "root/../../bad", "root\\bad"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                archive_path(path)


if __name__ == "__main__":
    unittest.main()
