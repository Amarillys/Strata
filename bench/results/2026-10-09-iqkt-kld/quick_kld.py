"""Sample full-vocabulary native logits, preserving and restoring the managed service.

All evaluation text is synthetic and checked in. Private configs and launcher
arguments remain under logs/. No engine or production configuration is edited.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from switch_preset import Presets, Running, argument
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate

ARMS = ("v4", "v2", "v4-lite")
PORT = 18080
SOURCE_FLAGS = {"--pack", "--native", "--native-dense-gguf", "--native-head-gguf", "--ple-gguf", "--embd-gguf"}
SAMPLING = {"temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0,
            "repeat_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0,
            "penalty_last_n": 0, "seed": 1729, "max_tokens": 1,
            "chat_template_kwargs": {"enable_thinking": False}}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def flag(args, name, value):
    args = list(args)
    while name in args:
        i = args.index(name)
        del args[i:i + 2]
    return args + [name, str(value)]


def load_tokenizer(path):
    path = Path(path)
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for token, i in vocab.items():
        tokens[i] = token
    return Tokenizer(tokens, (path / "merges.txt").read_text(encoding="utf-8").split("\n"),
                     json.loads((path / "token_type.json").read_text(encoding="utf-8")))


def prepare(p, out):
    for directory in (out, out / "private", out / "launchers", out / "logits", out / "tokenizer"):
        directory.mkdir(parents=True, exist_ok=True)
    original = {name: digest(p.paths[name]) for name in ARMS}
    original["catalog"] = digest(p.catalog_path)
    tokenizer_hashes = {}
    for name in ARMS:
        tokenizer_hashes[name] = {f: digest(Path(p.configs[name]["tokenizer"]) / f)
                                  for f in ("vocab.json", "merges.txt", "token_type.json")}
    if any(tokenizer_hashes[name] != tokenizer_hashes["v4"] for name in ARMS):
        raise RuntimeError("Tokenizer files differ between models")
    for name in tokenizer_hashes["v4"]:
        shutil.copy2(Path(p.configs["v4"]["tokenizer"]) / name, out / "tokenizer" / name)
    (out / "tokenizer/chat_template.jinja").write_text("{{ messages[0]['content'] }}", encoding="utf-8")
    tok = load_tokenizer(out / "tokenizer")
    tpl = ChatTemplate(out / "tokenizer/chat_template.jinja")
    corpus = json.loads((HERE / "corpus.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261009)
    samples, documents = [], []
    for doc in corpus:
        ids = tok.encode(doc["text"], parse_special=True)
        valid = []
        for n in range(32, min(len(ids), 513)):
            try:
                prefix = tok.decode(ids[:n], errors="strict")
            except UnicodeDecodeError:
                continue
            if tok.encode(prefix, parse_special=True) == ids[:n]:
                valid.append((n, prefix))
        if len(valid) < 16:
            raise RuntimeError("Not enough valid token boundaries in " + doc["id"])
        indices = sorted(rng.choice(len(valid), size=16, replace=False).tolist())
        for j in indices:
            n, prefix = valid[j]
            if tpl.render([{"role": "user", "content": prefix}], enable_thinking=False) != prefix:
                raise RuntimeError("The raw evaluation template changed the prefix")
            samples.append({"id": f"{doc['id']}:{n}", "document": doc["id"], "category": doc["category"],
                            "prefix": prefix, "tokens": ids[:n], "target": ids[n], "position": n})
        documents.append({"id": doc["id"], "category": doc["category"], "tokens": len(ids), "positions": 16})
    write_json(out / "samples.json", samples)
    configs = {}
    for arm in ARMS:
        cfg = copy.deepcopy(p.configs["v4"])
        args = list(cfg["args"])
        for name in SOURCE_FLAGS:
            while name in args:
                i = args.index(name)
                del args[i:i + 2]
        source = p.configs[arm]["args"]
        for i, value in enumerate(source):
            if value in SOURCE_FLAGS:
                args.extend(source[i:i + 2])
        for name, value in {"--layer-split": 19, "--vram-reserve-mib": 1024,
                            "--vram-reserve-later-mib": 768, "--prompt-cache": 0,
                            "--prompt-cache-every": 0, "--conversation-cache-mib": 0,
                            "--conversation-pool-tokens": 0}.items():
            args = flag(args, name, value)
        args = [a for a in args if a != "--vision"]
        cfg.pop("vision", None)
        cfg["args"] = args
        cfg["layer_split"] = "19"
        cfg["host"] = "127.0.0.1"
        cfg["model_name"] = "kld-" + arm
        cfg["aliases"] = []
        cfg["tokenizer"] = str(out / "tokenizer")
        cfg["log"] = str(out / f"engine-{arm}.log")
        cfg.setdefault("env", {})["STRATA_DUMP_FIRST_LOGITS"] = str(out / "logits" / arm / "row")
        (out / "logits" / arm).mkdir(exist_ok=True)
        name = "kld-" + arm
        path = out / "private" / f"{name}.json"
        write_json(path, cfg)
        p.paths[name], p.configs[name] = path, cfg
        p.validate(name)
        configs[arm] = {k: cfg[k] for k in ("exe", "args", "layer_split", "tokenizer")}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()
    manifest = {"prepared_at": datetime.datetime.now().astimezone().isoformat(), "arms": ARMS,
                "reference": "v4", "vocab": len(tok.tokens), "samples": len(samples), "documents": documents,
                "corpus_sha256": digest(HERE / "corpus.json"), "samples_sha256": digest(out / "samples.json"),
                "script_sha256": digest(Path(__file__)), "original_sha256": original,
                "tokenizer_sha256": tokenizer_hashes, "binary_sha256": digest(cfg["exe"]),
                "configs": configs, "sampling": SAMPLING,
                "hardware": {"gpus": gpu, "system_ram_gib": psutil.virtual_memory().total / 2**30},
                "note": "128 fixed raw-text prefixes, full-vocabulary FP32 logits, float64 log-softmax at temperature 1; "
                        "same runtime settings and layer split, automatic expert residency can differ by weight size; "
                        "synthetic short-context sample, not a standard perplexity benchmark or BF16 fidelity test."}
    write_json(out / "manifest.json", manifest)
    print("PREPARED " + json.dumps({"samples": len(samples), "vocab": manifest["vocab"],
                                    "prefix_tokens": [min(s["position"] for s in samples), max(s["position"] for s in samples)]}), flush=True)
    return samples, manifest


def raw_row(path, vocab):
    if path.stat().st_size != vocab * 4:
        raise RuntimeError("Wrong logits file size: " + str(path))
    row = np.fromfile(path, dtype="<f4")
    if not np.isfinite(row).all():
        raise RuntimeError("Nonfinite logits in " + str(path))
    return row


def log_softmax(row):
    row = row.astype(np.float64)
    row -= row.max()
    return row - math.log(np.exp(row).sum())


def kl(reference, candidate):
    value = float(np.dot(np.exp(reference), reference - candidate))
    if value < -1e-9 or not math.isfinite(value):
        raise RuntimeError("Invalid KL value")
    return max(0.0, value)


def stats(values):
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def compare(out, samples, vocab):
    rows = []
    for i, sample in enumerate(samples):
        distributions = {arm: log_softmax(raw_row(out / "logits" / arm / f"row.{i}", vocab)) for arm in ARMS}
        ref = distributions["v4"]
        row = {k: sample[k] for k in ("id", "document", "category", "position", "target")}
        row["models"] = {arm: {"kl_v4_to_model": kl(ref, lp), "top1_matches_v4": bool(lp.argmax() == ref.argmax()),
                               "nll": float(-lp[sample["target"]]), "top1_token": int(lp.argmax())}
                         for arm, lp in distributions.items()}
        rows.append(row)
    aggregate = {}
    for category in ("all", *sorted({s["category"] for s in samples})):
        subset = [r for r in rows if category == "all" or r["category"] == category]
        aggregate[category] = {"positions": len(subset), "models": {}}
        for arm in ARMS:
            m = [r["models"][arm] for r in subset]
            nll = float(np.mean([r["nll"] for r in m]))
            aggregate[category]["models"][arm] = {
                "kl_v4_to_model": stats([r["kl_v4_to_model"] for r in m]),
                "top1_agreement_v4": float(np.mean([r["top1_matches_v4"] for r in m])),
                "mean_nll_nats": nll, "exp_mean_sampled_nll": math.exp(nll),
                "delta_nll_vs_v4": nll - float(np.mean([r["models"]["v4"]["nll"] for r in subset]))}
        aggregate[category]["lite_kl_below_v2_fraction"] = float(np.mean([
            r["models"]["v4-lite"]["kl_v4_to_model"] < r["models"]["v2"]["kl_v4_to_model"] for r in subset]))
    repeatability = {}
    for arm in ARMS:
        values, agreements = [], []
        for j, i in enumerate(range(0, len(samples), 16)):
            a = log_softmax(raw_row(out / "logits" / arm / f"row.{i}", vocab))
            b = log_softmax(raw_row(out / "logits" / arm / f"row.{len(samples) + j}", vocab))
            values.append(kl(a, b))
            agreements.append(bool(a.argmax() == b.argmax()))
        repeatability[arm] = {"positions": len(values), "kl": stats(values), "top1_agreement": float(np.mean(agreements))}
    result = {"reference": "v4", "units": "nats", "positions": len(samples), "by_category": aggregate,
              "repeatability": repeatability, "rows": rows}
    write_json(out / "summary.json", result)
    print("RESULT " + json.dumps({"all": aggregate["all"], "repeatability": repeatability}), flush=True)
    return result


def run_arm(p, arm, samples, out, vocab):
    records = []
    work = list(enumerate(samples)) + [(i, samples[i]) for i in range(0, len(samples), 16)]
    for ordinal, (i, sample) in enumerate(work):
        path = out / "logits" / arm / f"row.{ordinal}"
        if path.exists():
            raise RuntimeError("Refusing to overwrite logits: " + str(path))
        start = time.perf_counter()
        response = p.api(PORT, "v1/chat/completions", {**SAMPLING, "model": "kld-" + arm,
                         "messages": [{"role": "user", "content": sample["prefix"]}]})
        if response.get("usage", {}).get("prompt_tokens") != sample["position"]:
            raise RuntimeError("API changed the input token count at " + sample["id"])
        raw_row(path, vocab)
        record = {"ordinal": ordinal, "sample": sample["id"], "repeat": ordinal >= len(samples),
                  "seconds": time.perf_counter() - start, "file": str(path.relative_to(out)),
                  "sha256": digest(path), "usage": response.get("usage")}
        records.append(record)
        write_json(out / f"requests-{arm}.json", records)
        if ordinal == 0 or (ordinal + 1) % 16 == 0 or ordinal + 1 == len(work):
            print("PROGRESS " + json.dumps({"model": arm, "done": ordinal + 1, "total": len(work),
                                             "last_seconds": round(record["seconds"], 3)}), flush=True)


def snapshot(p, running):
    engines = []
    for child in running.process.children(recursive=True):
        try:
            if Path(child.exe()).name.lower() == "strata.exe":
                engines.append({"pid": child.pid, "exe": child.exe(), "argv": child.cmdline()})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    h = p.api(running.port, "health")
    health = {k: h[k] for k in ("status", "loaded", "model", "max_context", "version") if k in h}
    return {"preset": running.name, "launcher_pid": running.process.pid, "port": running.port,
            "health": health, "engines": engines}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run", action="store_true", help="run models after preparation; otherwise only validate inputs")
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "logs"):
        ap.error("Output must be inside this repository's logs directory")
    if args.compare_only:
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        compare(out, json.loads((out / "samples.json").read_text(encoding="utf-8")), manifest["vocab"])
        return
    if list((out / "logits").glob("*/row.*")):
        ap.error("Use a fresh output directory; logits already exist")
    p = Presets(ROOT / "strata-models.json")
    samples, manifest = prepare(p, out)
    # Numerical sanity check before interrupting the service.
    a = log_softmax(np.array([1, 2, 3], dtype=np.float32))
    if kl(a, a) != 0 or not math.isclose(float(np.exp(a).sum()), 1):
        raise RuntimeError("KL normalization check failed")
    if not args.run:
        return
    current = None
    with p.lock():
        running = p.running()
        if len(running) != 1 or running[0].name != "v4-lite":
            raise RuntimeError("Expected the existing managed v4-lite service; nothing was stopped")
        old = running[0]
        original_logs = p.logs
        p.check_port(running)
        p.port = PORT
        p.check_port(running)
        write_json(out / "private/original-launcher.json", {"name": old.name, "argv": old.argv, "cwd": old.cwd,
                                                             "port": old.port, "pid": old.process.pid})
        write_json(out / "before.json", snapshot(p, old))
        changed = False
        try:
            p.drain(old)
            changed = True
            p.record(None)
            p.logs = out / "launchers"
            loads = []
            for arm in ARMS:
                name = "kld-" + arm
                argv = flag(flag(flag(old.argv, "--config", p.paths[name]), "--host", "127.0.0.1"), "--port", PORT)
                start = time.perf_counter()
                current = p.launch(name, previous=Running(name, old.process, argv, old.cwd, PORT))
                p.record(current)
                loads.append({"model": arm, "load_seconds": time.perf_counter() - start, **snapshot(p, current)})
                write_json(out / "loads.json", loads)
                print("LOADED " + arm, flush=True)
                run_arm(p, arm, samples, out, manifest["vocab"])
                p.drain(current)
                current = None
                p.record(None)
        except BaseException as e:
            write_json(out / "failure.json", {"type": type(e).__name__, "message": str(e)})
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
                    response = p.api(old.port, "v1/chat/completions", {**SAMPLING, "max_tokens": 8,
                                     "messages": [{"role": "user", "content": "只回复一个字：好"}]})
                    state = snapshot(p, restored)
                    state["launcher_argv_preserved"] = restored.argv == old.argv
                    state["launcher_cwd_preserved"] = restored.cwd == old.cwd
                    state["production_files_unchanged"] = all(digest(p.paths[k]) == manifest["original_sha256"][k] for k in ARMS)
                    state["catalog_unchanged"] = digest(p.catalog_path) == manifest["original_sha256"]["catalog"]
                    state["warmup_completion_tokens"] = response.get("usage", {}).get("completion_tokens")
                    write_json(out / "restored.json", state)
                    print("RESTORED " + json.dumps(state["health"]), flush=True)
                    if not all(state[k] for k in ("launcher_argv_preserved", "launcher_cwd_preserved", "production_files_unchanged", "catalog_unchanged")):
                        raise RuntimeError("Restoration did not preserve the original settings")
    compare(out, samples, manifest["vocab"])
    write_json(out / "completion.json", {"completed_at": datetime.datetime.now().astimezone().isoformat(),
                                         "requests_per_model": len(samples) + len(range(0, len(samples), 16)), "restored": True})


if __name__ == "__main__":
    main()
