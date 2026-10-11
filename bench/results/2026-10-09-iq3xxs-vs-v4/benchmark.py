"""Local, authenticated A/B benchmark; restores the original managed service.

Run from the Strata root with --out pointing under logs/. --prepare-only checks
the inputs and renders prompts without unloading the production model. Private
config copies stay under logs/; never publish that directory wholesale.
"""
from __future__ import annotations

import argparse
import collections
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

import psutil
import pynvml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate
from switch_preset import Presets, Running, argument

MODEL_DIR = Path("B:/models/llm/Qwen/Qwen3.8-Flash-Next")
XXS = MODEL_DIR / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
BASE_PROMPTS = ROOT / "bench/results/2026-10-08-iqkt-upstream-0.1.41/prompts.json"
LONG_RECIPE = ROOT / "bench/results/2026-10-08-v4-256k/long-prompt-recipe.json"
BENCH_PORT = 18080
SAMPLING = {"temperature": 0, "top_p": 1, "top_k": 1, "seed": 424242,
            "chat_template_kwargs": {"enable_thinking": False}}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def flag(args, key, value):
    result = list(args)
    if key in result:
        result[result.index(key) + 1] = str(value)
    else:
        result += [key, str(value)]
    return result


def make_prompts():
    base = json.loads(BASE_PROMPTS.read_text(encoding="utf-8"))
    recipe = json.loads(LONG_RECIPE.read_text(encoding="utf-8"))
    tok = Tokenizer.from_gguf(XXS)
    templates = {name: ChatTemplate(ROOT / pack / "tokenizer/chat_template.jinja")
                 for name, pack in (("v4", "pack/iq3kt-v4"), ("iq3_xxs", "pack/iq3_xxs"))}
    prompts = {"short": base["short_messages"], "medium": base["medium_messages"]}
    recipes = {}
    for label, target in (("long32k", 32768), ("long128k", 128000)):
        head = recipe["head"].split("<|im_start|>user\n", 1)[1]
        tail = recipe["tail"].split("<|im_end|>", 1)[0]
        filler = tok.encode(recipe["filler"])
        desired = target - 30
        for _ in range(5):
            ids = tok.encode(head)
            placements = []

            def fill_to(n):
                remaining = max(0, n - len(ids))
                ids.extend((filler * (remaining // len(filler) + 1))[:remaining])

            for row, fraction in zip(recipe["placements"], (0.20, 0.50, 0.85)):
                fill_to(int(desired * fraction))
                placements.append({"record": row["record"], "code": row["code"],
                                   "user_token_offset": len(ids)})
                ids.extend(tok.encode(f"\nRecord {row['record']} access code: {row['code']}\n"))
            tail_ids = tok.encode(tail)
            fill_to(desired - len(tail_ids))
            ids.extend(tail_ids)
            messages = [{"role": "system", "content": recipe["system"]},
                        {"role": "user", "content": tok.decode(ids)}]
            actual = len(tok.encode(templates["v4"].render(messages, enable_thinking=False),
                                    parse_special=True))
            if actual == target:
                break
            desired += target - actual
        if actual != target:
            raise RuntimeError(f"Could not construct {target} tokens: {actual}")
        prompts[label] = messages
        recipes[label] = {"target_http_tokens": target, "user_tokens": desired,
                          "head": head, "tail": tail, "filler": recipe["filler"],
                          "placements": placements}
    checks = {}
    for label, messages in prompts.items():
        rendered = {name: tpl.render(messages, enable_thinking=False) for name, tpl in templates.items()}
        if rendered["v4"] != rendered["iq3_xxs"]:
            raise RuntimeError(f"Different rendered text for {label}; comparison needs review")
        ids = tok.encode(rendered["v4"], parse_special=True)
        checks[label] = {"prompt_tokens": len(ids),
                         "render_sha256": hashlib.sha256(rendered["v4"].encode()).hexdigest(),
                         "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                         "identical_render_between_models": True}
    return prompts, recipes, checks


def model_summary(first):
    files = []
    for i in (1, 2):
        path = Path(str(first).replace("00001-of-00002", f"{i:05d}-of-00002"))
        g = GGUFFile(path)
        invalid = [t.name for t in g.tensors if t.expected_bytes() is None or
                   g.data_start + t.offset + t.expected_bytes() > path.stat().st_size]
        if invalid:
            raise RuntimeError(f"Truncated or unsupported model tensors: {invalid}")
        with path.open("rb") as f:
            header_hash = hashlib.sha256(f.read(g.data_start)).hexdigest()
        files.append({"filename": path.name, "bytes": path.stat().st_size,
                      "header_sha256": header_hash, "full_file_sha256": None,
                      "tensor_types": dict(collections.Counter(t.type_name for t in g.tensors)),
                      "expert_types": dict(collections.Counter(t.type_name for t in g.tensors if "_exps." in t.name)),
                      "expert_bytes": sum(t.expected_bytes() for t in g.tensors if "_exps." in t.name)})
    return {"files": files, "total_bytes": sum(f["bytes"] for f in files)}


class Telemetry:
    def __init__(self, path):
        pynvml.nvmlInit()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        self.file = Path(path).open("w", encoding="utf-8")
        self.samples = []
        self.phase = "setup"
        self.pid = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        gpus = []
        for i, h in enumerate(self.handles):
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpus.append({"index": i, "memory_used_mib": mem.used / 2**20,
                         "memory_free_mib": mem.free / 2**20, "gpu_util": util.gpu,
                         "temperature_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                         "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
                         "graphics_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS),
                         "memory_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
                         "pcie_gen": pynvml.nvmlDeviceGetCurrPcieLinkGeneration(h),
                         "pcie_width": pynvml.nvmlDeviceGetCurrPcieLinkWidth(h)})
        mem = psutil.virtual_memory()
        proc = {}
        if self.pid:
            try:
                p = psutil.Process(self.pid)
                m = p.memory_info()
                proc = {"pid": self.pid, "rss_mib": m.rss / 2**20,
                        "private_mib": getattr(m, "private", 0) / 2**20,
                        "cpu_times": p.cpu_times()._asdict()}
            except psutil.Error:
                pass
        return {"time": time.time(), "phase": self.phase, "gpus": gpus,
                "ram_used_gib": mem.used / 2**30, "ram_available_gib": mem.available / 2**30,
                "engine_process": proc}

    def run(self):
        while not self.stop_event.is_set():
            try:
                row = self.sample()
                self.samples.append(row)
                self.file.write(json.dumps(row) + "\n")
                self.file.flush()
            except Exception as e:
                self.file.write(json.dumps({"time": time.time(), "telemetry_error": str(e)}) + "\n")
                self.file.flush()
            self.stop_event.wait(0.5)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.file.close()
        pynvml.nvmlShutdown()

    def summary(self, phase):
        rows = [s for s in self.samples if s["phase"] == phase]
        if not rows:
            return {"sample_count": 0}
        return {"sample_count": len(rows), "ram_peak_gib": max(s["ram_used_gib"] for s in rows),
                "ram_min_available_gib": min(s["ram_available_gib"] for s in rows),
                "gpus": [{"index": i,
                          "peak_used_mib": max(s["gpus"][i]["memory_used_mib"] for s in rows),
                          "min_free_mib": min(s["gpus"][i]["memory_free_mib"] for s in rows),
                          "max_temperature_c": max(s["gpus"][i]["temperature_c"] for s in rows),
                          "median_graphics_mhz": statistics.median(s["gpus"][i]["graphics_mhz"] for s in rows),
                          "median_power_w": statistics.median(s["gpus"][i]["power_w"] for s in rows)}
                         for i in range(len(self.handles))]}


class Runner:
    def __init__(self, manager, out, telemetry, prompts, checks):
        self.p, self.out, self.tel = manager, out, telemetry
        self.prompts, self.checks = prompts, checks
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.results = []
        self.loads = []
        self.current = None

    def snapshot(self, r):
        metric = self.p.api(r.port, "metrics")
        children = []
        for child in r.process.children(recursive=True):
            if child.name().lower() == "strata.exe":
                self.tel.pid = child.pid
                children.append({"pid": child.pid, "exe": child.exe(), "argv": child.cmdline()})
        return {"health": self.p.api(r.port, "health"), "engine": metric["engine"],
                "hardware_static": metric["hardware_static"], "engine_processes": children,
                "hardware_snapshot": self.tel.sample()}

    def request(self, model, case, repetition, port, *, phase="measured", messages=None,
                max_tokens=256, session=None, expect_fresh=True):
        label = f"{model}-{case}-{phase}-{repetition}"
        session = session or label
        messages = self.prompts[case] if messages is None else messages
        body = {**SAMPLING, "messages": messages, "max_tokens": max_tokens,
                "stream": True, "stream_options": {"include_usage": True}, "session_id": session}
        request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                   data=json.dumps(body).encode("utf-8"),
                   headers={"Authorization": "Bearer " + self.p.key, "Content-Type": "application/json"})
        before = self.p.api(port, "metrics")["totals"]["requests"]
        self.tel.phase = label
        print(f"REQUEST {label}", flush=True)
        started = time.perf_counter()
        first = None
        first_kind = None
        usage = None
        finish = None
        text_parts, reasoning = [], []
        saw_done = False
        with self.opener.open(request, timeout=900) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(f"Streaming API error: {event['error']}")
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    for key, dest in (("content", text_parts), ("reasoning_content", reasoning)):
                        value = delta.get(key)
                        if value:
                            if first is None:
                                first = time.perf_counter() - started
                                first_kind = key
                            dest.append(value)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        total = time.perf_counter() - started
        if not saw_done or not usage:
            raise RuntimeError(f"Incomplete streaming response for {label}")
        for _ in range(50):
            metric = self.p.api(port, "metrics")
            if metric["totals"]["requests"] > before:
                break
            time.sleep(0.1)
        if metric["totals"]["requests"] != before + 1:
            raise RuntimeError(f"Unexpected concurrent/missing request at {label}")
        timing = metric["requests"][0]
        output = "".join(text_parts)
        row = {"model": model, "case": case, "phase": phase, "repetition": repetition,
               "label": label, "ttft_s": first, "first_delta_kind": first_kind,
               "client_total_s": total, "finish": finish, "usage": usage,
               "timing": timing, "prefill_tok_s": timing["prompt_read"] * 1000 / timing["prompt_ms"],
               "decode_tok_s": timing["engine_generated"] * 1000 / timing["decode_ms"],
               "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
               "reasoning_characters": sum(map(len, reasoning)), "memory": self.tel.summary(label)}
        if case.startswith("long"):
            row["retrieval_codes"] = {code: code in output for code in
                                      ("amber-orchid-7391", "cobalt-sparrow-5826", "jade-harbor-4168")}
        if expect_fresh and timing.get("reused", 0) != 0:
            row["invalid_reason"] = "Unexpected prefix reuse"
        if case in self.checks and phase in ("measured", "warmup", "restored_check"):
            if usage["prompt_tokens"] != self.checks[case]["prompt_tokens"]:
                row["invalid_reason"] = "Unexpected prompt token count"
        if phase == "measured" and usage["completion_tokens"] != max_tokens:
            row["early_eos"] = True
        write_json(self.out / "responses" / f"{label}.json",
                   {"content": output, "reasoning_content": "".join(reasoning), "result": row})
        self.results.append(row)
        write_json(self.out / "results.json", self.results)
        print(json.dumps({k: row[k] for k in ("label", "ttft_s", "client_total_s", "prefill_tok_s", "decode_tok_s")}
                         | {"prompt": usage["prompt_tokens"], "output": usage["completion_tokens"],
                            "reused": timing.get("reused", 0), "recall": row.get("retrieval_codes")}), flush=True)
        if row.get("invalid_reason"):
            raise RuntimeError(f"{label}: {row['invalid_reason']}")
        return row, output, session

    def suite(self, model, port):
        self.request(model, "short", 0, port, phase="warmup")
        last = None
        for case, reps in (("short", 3), ("medium", 3), ("long32k", 3), ("long128k", 1)):
            for rep in range(1, reps + 1):
                last = self.request(model, case, rep, port)
        _, output, session = last
        followup = self.prompts["long128k"] + [{"role": "assistant", "content": output},
                    {"role": "user", "content": "Repeat only the exact access codes from records A, B and C, in order."}]
        self.request(model, "long128k-followup", 1, port, phase="followup", messages=followup,
                     max_tokens=64, session=session, expect_fresh=False)


def prepare(p, out):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("private", "responses", "launchers"):
        (out / name).mkdir(exist_ok=True)
    base = p.configs["v4"]
    prompts, recipes, checks = make_prompts()
    configs = {}
    for label in ("v4", "iq3_xxs"):
        cfg = copy.deepcopy(base)
        cfg["host"] = "127.0.0.1"
        cfg["model_name"] = "benchmark-" + label
        cfg["aliases"] = []
        cfg["log"] = str(out / f"engine-{label}.log")
        if label == "iq3_xxs":
            cfg["args"] = flag(cfg["args"], "--pack", str(ROOT / "pack/iq3_xxs"))
            cfg["args"] = flag(cfg["args"], "--native", str(XXS))
            cfg["tokenizer"] = str(ROOT / "pack/iq3_xxs/tokenizer")
        name = "bench-" + label
        path = out / "private" / f"{name}.json"
        write_json(path, cfg)
        p.paths[name] = path
        p.configs[name] = cfg
        p.validate(name)
        configs[label] = {k: v for k, v in cfg.items() if k != "api_key"}
    write_json(out / "synthetic-prompts.json", prompts)
    write_json(out / "prompt-recipes.json", {"checks": checks, "long": recipes,
                                            "base_prompts": str(BASE_PROMPTS.relative_to(ROOT)),
                                            "sampling": SAMPLING})
    manifest = {"started": datetime.datetime.now().astimezone().isoformat(),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                "binary": {"path": base["exe"], "sha256": digest(base["exe"])},
                "os": platform.platform(), "ram_gib": psutil.virtual_memory().total / 2**30,
                "configs": configs, "sampling": SAMPLING, "prompt_checks": checks,
                "models": {"v4": model_summary(argument(base["args"], "--native")),
                           "iq3_xxs": model_summary(XXS)},
                "production_config_sha256": digest(p.paths["v4"]),
                "catalog_sha256": digest(p.catalog_path),
                "tokenizer_render_identical_for_all_benchmark_prompts": True,
                "note": "GGUF tensor extents checked; header hashes only, no full model checksum."}
    write_json(out / "manifest.json", manifest)
    print("PREPARED " + json.dumps(checks), flush=True)
    return prompts, checks, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "logs"):
        ap.error("--out must be under this repository's logs directory")
    if (out / "results.json").exists():
        raise RuntimeError("This output directory already has results; choose a new directory")
    os.chdir(ROOT)
    p = Presets(ROOT / "strata-models.json")
    prompts, checks, manifest = prepare(p, out)
    if args.prepare_only:
        return
    tel = Telemetry(out / "telemetry.jsonl")
    runner = Runner(p, out, tel, prompts, checks)
    tel.start()
    try:
        with p.lock():
            running = p.running()
            if len(running) != 1 or running[0].name != "v4":
                raise RuntimeError("Expected exactly one managed v4 service before the benchmark")
            old = running[0]
            original_logs = p.logs
            write_json(out / "private/original-launcher.json",
                       {"name": old.name, "pid": old.process.pid, "argv": old.argv, "cwd": old.cwd, "port": old.port})
            p.check_port(running)
            p.port = BENCH_PORT
            p.check_port(running)
            write_json(out / "before.json", runner.snapshot(old))
            changed = False
            try:
                p.drain(old)
                changed = True
                p.record(None)
                p.logs = out / "launchers"
                for label in ("v4", "iq3_xxs"):
                    name = "bench-" + label
                    argv = flag(flag(flag(old.argv, "--config", p.paths[name]), "--host", "127.0.0.1"),
                                "--port", BENCH_PORT)
                    launch_spec = Running(name, old.process, argv, old.cwd, BENCH_PORT)
                    tel.phase = "load-" + label
                    started = time.perf_counter()
                    runner.current = p.launch(name, previous=launch_spec)
                    elapsed = time.perf_counter() - started
                    p.record(runner.current)
                    snap = runner.snapshot(runner.current)
                    load = {"model": label, "load_seconds": elapsed, **snap}
                    runner.loads.append(load)
                    write_json(out / "loads.json", runner.loads)
                    print("LOADED " + json.dumps({"model": label, "seconds": elapsed, "engine": snap["engine"]}), flush=True)
                    runner.suite(label, BENCH_PORT)
                    write_json(out / f"after-{label}.json", runner.snapshot(runner.current))
                    p.drain(runner.current)
                    runner.current = None
                    tel.pid = None
                    p.record(None)
            finally:
                if changed:
                    if runner.current is not None:
                        p.drain(runner.current)
                        runner.current = None
                    p.logs = original_logs
                    p.port = old.port
                    tel.phase = "restore-v4"
                    restored = p.launch(old.name, previous=old)
                    p.record(restored)
                    state = runner.snapshot(restored)
                    state["production_config_unchanged"] = digest(p.paths["v4"]) == manifest["production_config_sha256"]
                    state["catalog_unchanged"] = digest(p.catalog_path) == manifest["catalog_sha256"]
                    state["launcher_argv_preserved"] = restored.argv == old.argv
                    write_json(out / "restored.json", state)
                    print("RESTORED " + json.dumps(state["health"]), flush=True)
                    # Only synthetic requests; restores an ordinary response after a fresh load.
                    runner.request("v4-restored", "short", 0, old.port, phase="warmup")
                    runner.request("v4-restored", "short", 1, old.port, phase="restored_check")
    except BaseException as e:
        write_json(out / "failure.json", {"type": type(e).__name__, "message": str(e), "time": time.time()})
        raise
    finally:
        tel.close()
    write_json(out / "completion.json", {"completed": datetime.datetime.now().astimezone().isoformat(),
                                         "requests": len(runner.results), "restored": True})
    print("COMPLETE " + str(out), flush=True)


if __name__ == "__main__":
    main()
