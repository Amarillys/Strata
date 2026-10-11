"""Add the v4 IQ3_KT PLE variant to an existing quick_kld run, then restore production."""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

import quick_kld as q
from gguf_reader import GGUFFile

VARIANT = "v4-lite-iq3ple"


def prepare(p, out):
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    samples = json.loads((out / "samples.json").read_text(encoding="utf-8"))
    if q.digest(out / "samples.json") != manifest["samples_sha256"]:
        raise RuntimeError("The baseline samples changed")
    first = Path(q.argument(p.configs["v4"]["args"], "--native"))
    candidates = [Path(str(first).replace("00001-of-00002", f"{i:05d}-of-00002")) for i in (1, 2)]
    matches = [(path, tensor) for path in candidates for tensor in GGUFFile(path).tensors
               if tensor.name == "per_layer_token_embd.weight"]
    if len(matches) != 1 or matches[0][1].type_name != "IQ3_KT":
        raise RuntimeError("Expected one v4 IQ3_KT PLE tensor")
    ple_path, tensor = matches[0]
    cfg = json.loads((out / "private/kld-v4-lite.json").read_text(encoding="utf-8"))
    previous_ple = q.argument(cfg["args"], "--ple-gguf")
    cfg["args"] = q.flag(cfg["args"], "--ple-gguf", ple_path)
    cfg["model_name"] = "kld-" + VARIANT
    cfg["log"] = str(out / f"engine-{VARIANT}.log")
    cfg["env"]["STRATA_DUMP_FIRST_LOGITS"] = str(out / "logits" / VARIANT / "row")
    (out / "logits" / VARIANT).mkdir(exist_ok=True)
    if list((out / "logits" / VARIANT).glob("row.*")):
        raise RuntimeError("Variant logits already exist")
    name = "kld-" + VARIANT
    path = out / "private" / (name + ".json")
    q.write_json(path, cfg)
    p.paths[name], p.configs[name] = path, cfg
    p.validate(name)
    if q.digest(cfg["exe"]) != manifest["binary_sha256"]:
        raise RuntimeError("The engine changed since the baseline")
    details = {"variant": VARIANT, "base": "v4-lite", "sole_weight_change": "per_layer_token_embd.weight",
               "previous_ple": previous_ple, "ple_source": str(ple_path), "ple_type": tensor.type_name,
               "ple_bytes": tensor.expected_bytes(), "ple_shape": tensor.shape,
               "args": cfg["args"], "binary_sha256": manifest["binary_sha256"],
               "samples_sha256": manifest["samples_sha256"], "script_sha256": q.digest(Path(__file__))}
    q.write_json(out / "ple-variant-manifest.json", details)
    print("PREPARED PLE VARIANT " + json.dumps({"type": tensor.type_name, "size_gib": tensor.expected_bytes() / 2**30}), flush=True)
    return samples, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--wait-for-base", action="store_true")
    args = ap.parse_args()
    os.chdir(q.ROOT)
    out = args.out.resolve()
    if not out.is_relative_to(q.ROOT / "logs"):
        ap.error("Output must be inside this repository's logs directory")
    p = q.Presets(q.ROOT / "strata-models.json")
    samples, manifest = prepare(p, out)
    if not args.run:
        return
    if args.wait_for_base:
        print("Waiting for the three baseline models to finish and restore production...", flush=True)
        deadline = time.monotonic() + 3600
        while not (out / "completion.json").exists():
            if (out / "failure.json").exists() or time.monotonic() >= deadline:
                raise RuntimeError("The baseline did not finish successfully; no variant service was started")
            time.sleep(1)
    elif not (out / "completion.json").exists():
        raise RuntimeError("Finish the baseline run first")
    current = None
    with p.lock():
        running = p.running()
        if len(running) != 1 or running[0].name != "v4-lite":
            raise RuntimeError("Expected the restored v4-lite service; nothing was stopped")
        old = running[0]
        for arm in q.ARMS:
            if q.digest(p.paths[arm]) != manifest["original_sha256"][arm]:
                raise RuntimeError("A production configuration changed")
        original_logs = p.logs
        p.check_port(running)
        p.port = q.PORT
        p.check_port(running)
        q.write_json(out / "private/ple-original-launcher.json", {"name": old.name, "argv": old.argv,
                      "cwd": old.cwd, "port": old.port, "pid": old.process.pid})
        q.write_json(out / "ple-before.json", q.snapshot(p, old))
        changed = False
        try:
            p.drain(old)
            changed = True
            p.record(None)
            p.logs = out / "launchers"
            name = "kld-" + VARIANT
            argv = q.flag(q.flag(q.flag(old.argv, "--config", p.paths[name]), "--host", "127.0.0.1"), "--port", q.PORT)
            start = time.perf_counter()
            current = p.launch(name, previous=q.Running(name, old.process, argv, old.cwd, q.PORT))
            p.record(current)
            q.write_json(out / "ple-variant-load.json", {"load_seconds": time.perf_counter() - start, **q.snapshot(p, current)})
            q.run_arm(p, VARIANT, samples, out, manifest["vocab"])
        except BaseException as e:
            q.write_json(out / "ple-variant-failure.json", {"type": type(e).__name__, "message": str(e)})
            raise
        finally:
            if changed:
                try:
                    if current is not None:
                        try:
                            p.drain(current)
                        except BaseException:
                            p.terminate_owned(current)
                finally:
                    p.logs, p.port = original_logs, old.port
                    restored = p.launch(old.name, previous=old)
                    p.record(restored)
                    response = p.api(old.port, "v1/chat/completions", {**q.SAMPLING, "max_tokens": 8,
                                     "messages": [{"role": "user", "content": "只回复一个字：好"}]})
                    state = q.snapshot(p, restored)
                    state["launcher_argv_preserved"] = restored.argv == old.argv
                    state["launcher_cwd_preserved"] = restored.cwd == old.cwd
                    state["production_files_unchanged"] = all(q.digest(p.paths[k]) == manifest["original_sha256"][k] for k in q.ARMS)
                    state["catalog_unchanged"] = q.digest(p.catalog_path) == manifest["original_sha256"]["catalog"]
                    state["warmup_completion_tokens"] = response.get("usage", {}).get("completion_tokens")
                    q.write_json(out / "ple-restored.json", state)
                    print("RESTORED " + json.dumps(state["health"]), flush=True)
                    if not all(state[k] for k in ("launcher_argv_preserved", "launcher_cwd_preserved", "production_files_unchanged", "catalog_unchanged")):
                        raise RuntimeError("Restoration did not preserve the original settings")
    shutil.copy2(out / "summary.json", out / "summary-initial-three.json")
    q.ARMS = (*q.ARMS, VARIANT)
    result = q.compare(out, samples, manifest["vocab"])
    for category, aggregate in result["by_category"].items():
        rows = [row for row in result["rows"] if category == "all" or row["category"] == category]
        aggregate["iq3_ple_kl_below_v2_fraction"] = float(np.mean([
            row["models"][VARIANT]["kl_v4_to_model"] < row["models"]["v2"]["kl_v4_to_model"] for row in rows]))
        aggregate["iq3_minus_iq4_ple_mean_nll"] = float(np.mean([
            row["models"][VARIANT]["nll"] - row["models"]["v4-lite"]["nll"] for row in rows]))
    q.write_json(out / "summary.json", result)
    q.write_json(out / "ple-variant-completion.json", {"completed_at": datetime.datetime.now().astimezone().isoformat(),
                                                     "requests": len(samples) + 8, "restored": True})


if __name__ == "__main__":
    main()
