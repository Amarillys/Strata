"""Checks for split-model integrity, local configuration and the portable speed report."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lite_runtime as L
import benchmark_portable as B


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata lite 路径 ")
        self.root = Path(self.temp.name)
        path = self.root / L.TEMPLATE
        path.parent.mkdir(parents=True)
        shutil.copyfile(L.ROOT / L.TEMPLATE, path)
        self.models = self.root / "模型 files"
        self.models.mkdir()
        files = []
        for part in (1, 2):
            path = self.models / f"lite-{part:05d}-of-00002.gguf"
            data = b"GGUF" + bytes([part]) * 12
            path.write_bytes(data)
            files.append({"filename": path.name, "bytes": len(data), "header_bytes": 4,
                          "header_sha256": hashlib.sha256(data[:4]).hexdigest(),
                          "sha256": hashlib.sha256(data).hexdigest()})
        L.P.write_json(self.root / L.CONTRACT, {"files": files})
        self.first = self.models / files[0]["filename"]
        self.ple = self.models / files[1]["filename"]

    def tearDown(self):
        self.temp.cleanup()

    def options(self, **values):
        args = dict(config=L.CONFIG, model=str(self.models), pack=None, context=None, prefill=None,
                    reserve_mib=None, kv=None, gpu=None, port=None, host=None, no_browser=True)
        args.update(values)
        return argparse.Namespace(**args)

    def configured(self, **values):
        with mock.patch.object(L.P, "inspect_model", return_value={"schema": 1}), \
                mock.patch.object(L.P, "prepare_pack"), contextlib.redirect_stdout(io.StringIO()):
            L.configure(self.options(**values), self.root)
        return L.P.read_json(self.root / L.CONFIG)

    def test_directory_and_first_shard_resolve_same_two_files(self):
        self.assertEqual(L.model_files(str(self.models), self.root), [self.first, self.ple])
        self.assertEqual(L.model_files(str(self.first), self.root), [self.first, self.ple])
        with self.assertRaises(ValueError):
            L.model_files(str(self.ple), self.root)

    def test_missing_ple_refused_before_config_or_pack(self):
        self.ple.unlink()
        with mock.patch.object(L.P, "prepare_pack") as prepare, self.assertRaises(ValueError):
            L.configure(self.options(), self.root)
        prepare.assert_not_called()
        self.assertFalse((self.root / L.CONFIG).exists())

    def test_header_and_full_payload_corruption_are_detected(self):
        self.ple.write_bytes(b"BAD!" + b"\x02" * 12)
        with self.assertRaises(ValueError):
            L.model_files(str(self.first), self.root)
        self.ple.write_bytes(b"GGUF" + b"\x03" * 12)
        L.model_files(str(self.first), self.root)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            L.model_files(str(self.first), self.root, full=True)

    def test_reconfigure_preserves_key_and_port_and_sets_ple(self):
        first = self.configured(port=8099)
        second = self.configured(context=32768)
        self.assertEqual(first["api_key"], second["api_key"])
        self.assertEqual(second["port"], 8099)
        self.assertEqual(L.P.value_of(second["args"], "--max-context"), "32768")
        self.assertEqual(L.P.local_path(L.P.value_of(second["args"], "--ple-gguf"), self.root), self.ple)
        self.assertNotIn("--layer-split", second["args"])
        self.assertNotIn("--pcie-frac", second["args"])

    def test_wrong_ple_path_refused_before_start(self):
        cfg = self.configured()
        L.P.set_arg(cfg["args"], "--ple-gguf", L.P.value_of(cfg["args"], "--native"))
        L.P.write_json(self.root / L.CONFIG, cfg)
        with mock.patch.object(L.P, "start") as start, self.assertRaises(ValueError):
            L.start(self.options(), self.root)
        start.assert_not_called()


def stream():
    values = [{"choices": [{"delta": {"role": "assistant"}}]},
              {"choices": [{"delta": {"content": "private-fixture-response"}}]},
              {"choices": [{"delta": {}, "finish_reason": "length"}],
               "usage": {"prompt_tokens": 100, "completion_tokens": 16,
                         "prompt_tokens_details": {"cached_tokens": 80}},
               "timings": {"cache_n": 80, "prompt_n": 20, "prompt_ms": 400,
                           "prompt_per_second": 50, "predicted_n": 16,
                           "predicted_ms": 2000, "predicted_per_second": 8}}]
    return io.BytesIO(b": heartbeat\n\n" + b"".join(b"data: " + json.dumps(v).encode() + b"\n\n" for v in values)
                      + b"data: [DONE]\n\n")


class BenchmarkTests(unittest.TestCase):
    def test_sse_keeps_token_counts_and_engine_rates_without_text(self):
        opener = mock.Mock()
        opener.open.return_value = stream()
        result = B.completion(opener, "http://127.0.0.1:8089", "private-fixture-key", "strata",
                              "private-fixture-prompt", 16)
        self.assertEqual(result["cached_tokens"], 80)
        self.assertEqual(result["completion_tokens"], 16)
        self.assertEqual(result["timings"]["predicted_per_second"], 8)
        for private in ("private-fixture-key", "private-fixture-response", "private-fixture-prompt"):
            self.assertNotIn(private, json.dumps(result))

    def test_report_writes_all_cases_without_credentials_or_content(self):
        with tempfile.TemporaryDirectory(prefix="strata report ") as tmp:
            root = Path(tmp)
            cfg = L.P.read_json(L.ROOT / L.TEMPLATE)
            cfg["api_key"] = "private-fixture-key"
            L.P.write_json(root / L.CONFIG, cfg)
            opener = mock.Mock()
            opener.open.side_effect = lambda req, **kw: (
                io.BytesIO(json.dumps({"loaded": True, "model": cfg["model_name"]}).encode())
                if req.full_url.endswith("/health") else stream())
            with mock.patch.object(B.urllib.request, "build_opener", return_value=opener), \
                    mock.patch.object(B, "Sampler") as sampler, contextlib.redirect_stdout(io.StringIO()):
                sampler.return_value.hardware = {"gpu": "fixture"}
                sampler.return_value.finish.return_value = {"samples": 1}
                self.assertEqual(B.run(argparse.Namespace(config=L.CONFIG, tokens=16), root), 0)
            report = next((root / "results").glob("*/results.json"))
            text = report.read_text(encoding="utf-8")
            self.assertEqual(len(json.loads(text)["cases"]), 4)
            self.assertNotIn("private-fixture-key", text)
            self.assertNotIn("private-fixture-response", text)
            self.assertTrue(report.with_name("README.md").is_file())


if __name__ == "__main__":
    unittest.main()
