"""Evaluate v2 dense weights + v4 experts + v4 IQ3_KT PLE against saved v4 logits."""
from __future__ import annotations

import argparse
import copy
import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

import quick_kld as q

VARIANT = "v2dense-v4experts-iq3ple"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


composition = module("lite_composition", q.ROOT / "bench/results/2026-10-09-v4-lite/compose.py")
accounting = module("iqkt_accounting", q.ROOT / "bench/results/2026-10-09-iqkt-lite-analysis/analyze.py")


def pack_index(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            result[parts[0]] = {"bytes": int(parts[6]), "kind": int(parts[2]), "bits": int(parts[9])}
    return result


def source_free(args):
    result = []
    i = 0
    while i < len(args):
        if args[i] in q.SOURCE_FLAGS:
            i += 2
        else:
            result.append(args[i])
            i += 1
    return result


def prepare(p, base, out):
    if not (base / "ple-variant-completion.json").is_file():
        raise RuntimeError("Expected the completed four-arm baseline")
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    samples = json.loads((base / "samples.json").read_text(encoding="utf-8"))
    if q.digest(base / "samples.json") != manifest["samples_sha256"]:
        raise RuntimeError("Saved baseline samples changed")
    if q.digest(q.HERE / "corpus.json") != manifest["corpus_sha256"]:
        raise RuntimeError("The evaluation corpus changed")
    reference_records = json.loads((base / "requests-v4.json").read_text(encoding="utf-8"))
    for record in reference_records:
        if q.digest(base / record["file"]) != record["sha256"]:
            raise RuntimeError("A saved reference logits file changed")
    for directory in (out, out / "private", out / "launchers", out / "logits" / VARIANT):
        directory.mkdir(parents=True, exist_ok=True)
    if list((out / "logits" / VARIANT).glob("row.*")):
        raise RuntimeError("Candidate logits already exist; use a fresh output directory")
    first = {tag: (q.ROOT / q.argument(p.configs[tag]["args"], "--native")).resolve() for tag in ("v2", "v4")}
    packs = {tag: (q.ROOT / q.argument(p.configs[tag]["args"], "--pack")).resolve() for tag in ("v2", "v4")}
    if first["v2"].parent != first["v4"].parent:
        raise RuntimeError("Expert shard columns require both models in the same directory")
    models = {tag: composition.model(path) for tag, path in first.items()}
    files4, tensors4, _ = models["v4"]
    _, tensors2, _ = models["v2"]
    if tensors2.keys() != tensors4.keys() or any(tensors2[n][1].shape != tensors4[n][1].shape for n in tensors4):
        raise RuntimeError("Source model tensor shapes differ")
    prior = json.loads((q.ROOT / "bench/results/2026-10-09-v4-lite/composition.json").read_text(encoding="utf-8"))
    for tag in models:
        for record in models[tag][2]:
            previous = next(row for row in prior["sources"][tag] if row["filename"] == record["filename"])
            if any(record[key] != previous[key] for key in ("bytes", "mtime_ns", "header_sha256")):
                raise RuntimeError("A source GGUF changed since the previous comparison")
    for record in prior["copied_v4_files"]:
        if q.digest(packs["v4"] / record["file"]) != record["v4_sha256"]:
            raise RuntimeError("A reference v4 pack file changed")

    rows = ["# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard | gate,up,down]",
            "# v2 dense / v4 experts: every expert role explicitly names its original v4 shard."]
    experts = []
    total = 0
    seen = set()
    for line in (packs["v4"] / "native_experts.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if len(parts) not in (8, 9):
            raise RuntimeError("Unexpected expert index layout")
        layer, gu, down, offset, blob, *role_offsets = map(int, parts[:8])
        if layer in seen or gu not in (154, 155) or down != 154 or offset != total:
            raise RuntimeError("Invalid v4 expert layout")
        seen.add(layer)
        sources = parts[8].split(",") if len(parts) == 9 else [""]
        if len(sources) == 1:
            sources *= 3
        if len(sources) != 3:
            raise RuntimeError("Invalid expert role source list")
        sources = [source or first["v4"].name for source in sources]
        layer_bytes = 0
        for role, source, raw_offset in zip(("gate", "up", "down"), sources, role_offsets):
            name = f"blk.{layer}.ffn_{role}_exps.weight"
            gguf, tensor = tensors4[name]
            if source not in files4 or gguf.path.name != source or gguf.data_start + tensor.offset != raw_offset:
                raise RuntimeError("Expert shard or offset mismatch: " + name)
            if tensor.type_id != (down if role == "down" else gu):
                raise RuntimeError("Expert quantization type mismatch: " + name)
            layer_bytes += tensor.expected_bytes()
            experts.append({"tensor": name, "file": source, "type": tensor.type_name, "offset": raw_offset,
                            "bytes": tensor.expected_bytes(), "shape": tensor.shape})
        if layer_bytes != blob * 512:
            raise RuntimeError("Expert extent mismatch")
        total += layer_bytes
        rows.append(" ".join(parts[:8] + [sources[0] if len(set(sources)) == 1 else ",".join(sources)]))
    if seen != set(range(48)) or len(experts) != 144 or total != 50625249280:
        raise RuntimeError("Incomplete v4 expert inventory")
    if sum(row["type"] == "IQ4_KT" for row in experts) != 32:
        raise RuntimeError("Expected 32 higher-precision v4 expert tensors")

    pack = out / "pack"
    if pack.exists() and not (out / "composition.json").exists():
        raise RuntimeError("Refusing to overwrite a pack without a completed preparation manifest")
    pack.mkdir(exist_ok=True)
    copied = []
    for source in sorted(packs["v2"].rglob("*")):
        if not source.is_file() or source.name == "native_experts.txt":
            continue
        rel = source.relative_to(packs["v2"])
        target = pack / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        source_hash = q.digest(source)
        if not target.exists():
            shutil.copyfile(source, target)
        if q.digest(target) != source_hash:
            raise RuntimeError("Candidate pack file differs from v2: " + str(rel))
        copied.append({"file": rel.as_posix(), "bytes": source.stat().st_size, "source_v2_sha256": source_hash})
    expert_text = "\n".join(rows) + "\n"
    index = pack / "native_experts.txt"
    if index.exists() and index.read_text(encoding="utf-8") != expert_text:
        raise RuntimeError("Existing candidate expert index differs")
    index.write_text(expert_text, encoding="utf-8")
    shutil.copy2(base / "samples.json", out / "samples.json")
    tokenizer = out / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    for source in (base / "tokenizer").iterdir():
        if source.is_file():
            shutil.copy2(source, tokenizer / source.name)
    for name, checksum in manifest["tokenizer_sha256"]["v2"].items():
        if q.digest(tokenizer / name) != checksum or q.digest(packs["v2"] / "tokenizer" / name) != checksum:
            raise RuntimeError("Tokenizer differs from the baseline")
    gguf_ple, ple = tensors4["per_layer_token_embd.weight"]
    if ple.type_name != "IQ3_KT":
        raise RuntimeError("Expected the reference v4 IQ3_KT PLE")
    mixed = {name: tensor for name, (_, tensor) in tensors2.items()}
    mixed.update({name: tensor for name, (_, tensor) in tensors4.items()
                  if name.endswith("_exps.weight") or name == "per_layer_token_embd.weight"})
    candidate_inventory = accounting.inventory(mixed, pack_index(packs["v2"] / "index.txt"))
    v4_inventory = accounting.inventory({name: tensor for name, (_, tensor) in tensors4.items()},
                                        pack_index(packs["v4"] / "index.txt"))
    savings = [a["weight_payload_bytes"] - b["weight_payload_bytes"]
               for a, b in zip(v4_inventory["splits"]["19"], candidate_inventory["splits"]["19"])]
    comp = {"name": VARIANT, "method": "Source selection only; no requantization or calibration.",
            "sources": {tag: model[2] for tag, model in models.items()},
            "dense_source": "v2, including all 387 small projections", "head_embedding_source": "v2",
            "ple_projection_source": "v2", "ple_table_source": "v4", "ple_type": ple.type_name,
            "expert_source": "v4", "expert_bytes": total, "expert_tensors": experts, "copied_files": copied,
            "expert_index_sha256": q.digest(index), "v4_inventory": v4_inventory,
            "candidate_inventory": candidate_inventory, "gpu_weight_saving_bytes_split19": savings,
            "logical_gguf_payload_saving_bytes": v4_inventory["tensor_payload_bytes"] - candidate_inventory["tensor_payload_bytes"],
            "note": "GPU weight savings exclude KV, workspaces and graphs; automatic expert caching can consume the freed space."}
    q.write_json(out / "composition.json", comp)
    cfg = json.loads((base / "private/kld-v4.json").read_text(encoding="utf-8"))
    cfg["api_key"] = p.key
    common_args = source_free(cfg["args"])
    cfg["args"] = common_args + ["--pack", str(pack), "--native", str(first["v2"])]
    for gguf in models["v2"][0].values():
        cfg["args"] += ["--native-dense-gguf", str(gguf.path)]
    cfg["args"] += ["--native-head-gguf", str(tensors2["output.weight"][0].path), "--ple-gguf", str(gguf_ple.path)]
    if source_free(cfg["args"]) != source_free(manifest["configs"]["v4"]["args"]):
        raise RuntimeError("Runtime settings differ from the saved reference")
    cfg["model_name"] = "kld-" + VARIANT
    cfg["tokenizer"] = str(tokenizer)
    cfg["log"] = str(out / f"engine-{VARIANT}.log")
    cfg["env"]["STRATA_DUMP_FIRST_LOGITS"] = str(out / "logits" / VARIANT / "row")
    name = "kld-" + VARIANT
    path = out / "private" / (name + ".json")
    q.write_json(path, cfg)
    p.paths[name], p.configs[name] = path, cfg
    p.validate(name)
    if q.digest(cfg["exe"]) != manifest["binary_sha256"]:
        raise RuntimeError("Engine binary differs from the saved reference")
    info = {"variant": VARIANT, "baseline_run": str(base), "baseline_manifest_sha256": q.digest(base / "manifest.json"),
            "baseline_summary_sha256": q.digest(base / "summary.json"), "samples_sha256": manifest["samples_sha256"],
            "binary_sha256": manifest["binary_sha256"], "vocab": manifest["vocab"], "positions": len(samples),
            "script_sha256": q.digest(Path(__file__)), "args": cfg["args"],
            "original_sha256": {tag: q.digest(p.paths[tag]) for tag in q.ARMS}, "catalog_sha256": q.digest(p.catalog_path)}
    q.write_json(out / "manifest.json", info)
    print("PREPARED " + json.dumps({"variant": VARIANT, "positions": len(samples), "expert_tensors": len(experts),
          "gpu_weight_saving_gib": sum(savings) / 2**30,
          "logical_gguf_saving_gib": comp["logical_gguf_payload_saving_bytes"] / 2**30}), flush=True)
    return samples, info


def compare(base, out, samples, info):
    result = copy.deepcopy(json.loads((base / "summary.json").read_text(encoding="utf-8")))
    if q.digest(base / "summary.json") != info["baseline_summary_sha256"]:
        raise RuntimeError("Baseline result changed during the test")
    for i, (sample, row) in enumerate(zip(samples, result["rows"])):
        if sample["id"] != row["id"] or sample["target"] != row["target"]:
            raise RuntimeError("Sample order differs from the reference")
        ref = q.log_softmax(q.raw_row(base / "logits/v4" / f"row.{i}", info["vocab"]))
        test = q.log_softmax(q.raw_row(out / "logits" / VARIANT / f"row.{i}", info["vocab"]))
        row["models"][VARIANT] = {"kl_v4_to_model": q.kl(ref, test), "top1_matches_v4": bool(ref.argmax() == test.argmax()),
                                   "nll": float(-test[sample["target"]]), "top1_token": int(test.argmax())}
    for category, group in result["by_category"].items():
        rows = [row for row in result["rows"] if category == "all" or row["category"] == category]
        metrics = [row["models"][VARIANT] for row in rows]
        nll = float(np.mean([m["nll"] for m in metrics]))
        group["models"][VARIANT] = {"kl_v4_to_model": q.stats([m["kl_v4_to_model"] for m in metrics]),
            "top1_agreement_v4": float(np.mean([m["top1_matches_v4"] for m in metrics])),
            "mean_nll_nats": nll, "exp_mean_sampled_nll": float(np.exp(nll)),
            "delta_nll_vs_v4": nll - group["models"]["v4"]["mean_nll_nats"]}
    repeat, matches = [], []
    for j, i in enumerate(range(0, len(samples), 16)):
        a = q.log_softmax(q.raw_row(out / "logits" / VARIANT / f"row.{i}", info["vocab"]))
        b = q.log_softmax(q.raw_row(out / "logits" / VARIANT / f"row.{len(samples) + j}", info["vocab"]))
        repeat.append(q.kl(a, b))
        matches.append(bool(a.argmax() == b.argmax()))
    result["repeatability"][VARIANT] = {"positions": len(repeat), "kl": q.stats(repeat), "top1_agreement": float(np.mean(matches))}
    result["added_variant"] = VARIANT
    result["baseline_run"] = str(base)
    q.write_json(out / "summary.json", result)
    print("RESULT " + json.dumps({"candidate": result["by_category"]["all"]["models"][VARIANT],
                                  "repeatability": result["repeatability"][VARIANT]}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=q.ROOT / "logs/iqkt-kld-20261009-quick-01")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    os.chdir(q.ROOT)
    base, out = args.base.resolve(), args.out.resolve()
    if not out.is_relative_to(q.ROOT / "logs") or out == q.ROOT / "logs" or out == base:
        ap.error("Use a separate task directory under logs/")
    p = q.Presets(q.ROOT / "strata-models.json")
    samples, info = prepare(p, base, out)
    if not args.run:
        return
    current = None
    with p.lock():
        running = p.running()
        if len(running) != 1:
            raise RuntimeError("Expected one managed service; no model was stopped")
        old = running[0]
        original_logs = p.logs
        p.port = old.port
        p.check_port(running)
        p.port = q.PORT
        p.check_port(running)
        q.write_json(out / "private/original-launcher.json", {"name": old.name, "argv": old.argv,
                     "cwd": old.cwd, "port": old.port, "pid": old.process.pid})
        q.write_json(out / "before.json", q.snapshot(p, old))
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
            q.write_json(out / "load.json", {"load_seconds": time.perf_counter() - start, **q.snapshot(p, current)})
            q.run_arm(p, VARIANT, samples, out, info["vocab"])
        except BaseException as e:
            q.write_json(out / "failure.json", {"type": type(e).__name__, "message": str(e)})
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
                    state["production_files_unchanged"] = all(q.digest(p.paths[k]) == info["original_sha256"][k] for k in q.ARMS)
                    state["catalog_unchanged"] = q.digest(p.catalog_path) == info["catalog_sha256"]
                    state["warmup_completion_tokens"] = response.get("usage", {}).get("completion_tokens")
                    q.write_json(out / "restored.json", state)
                    print("RESTORED " + json.dumps(state["health"]), flush=True)
                    if not all(state[k] for k in ("launcher_argv_preserved", "launcher_cwd_preserved", "production_files_unchanged", "catalog_unchanged")):
                        raise RuntimeError("Restoration did not preserve the original settings")
    compare(base, out, samples, info)
    q.write_json(out / "completion.json", {"completed_at": datetime.datetime.now().astimezone().isoformat(),
                                         "requests": len(samples) + 8, "restored": True, "reference_reused": True})


if __name__ == "__main__":
    main()
