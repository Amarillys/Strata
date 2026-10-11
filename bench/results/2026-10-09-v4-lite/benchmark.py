"""Compare the hybrid, a matched v2 and current v4; always restore the original service.

All requests are synthetic. Private configs/launcher arguments stay under logs/.
Uses the existing preset manager for ownership, locking, draining and rollback.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compose import CONFIG, set_values

spec = importlib.util.spec_from_file_location("common_benchmark", ROOT / "bench/results/2026-10-09-iq3xxs-vs-v4/benchmark.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
from switch_preset import Presets, Running, argument
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate

PORT = 18080
QUALITY = [
    {"id": "inventory", "prompt": "仓库最初有 240 件货。先发出全部的 15%，再到货 38 件，再发出当时库存的 25%。只输出 JSON：{\"remaining\": 数字}。", "expected": {"remaining": 181.5}},
    {"id": "chinese_extract", "prompt": "根据以下记录只输出 JSON，字段为 confirmed（已确认姓名，按记录顺序）、total（已确认金额总和）：王敏：确认，128 元；李明：取消，299 元；张华：确认，76 元；赵宁：待定，55 元。", "expected": {"confirmed": ["王敏", "张华"], "total": 204}},
    {"id": "python_defaults", "prompt": "Python 3 代码：\ndef f(x, a=[]):\n    a.append(x)\n    return a.copy()\nr1=f(1)\nr2=f(2)\nr3=f(3, [])\nr4=f(4)\n只输出 JSON 对象，键为 r1、r2、r3、r4，值为对应变量最终值。", "expected": {"r1": [1], "r2": [1, 2], "r3": [3], "r4": [1, 2, 4]}},
    {"id": "python_closures", "prompt": "Python 3：\nfs=[lambda: i for i in range(3)]\ngs=[lambda i=i: i for i in range(3)]\n只输出 JSON：{\"fs\": [f() for f in fs] 的结果, \"gs\": [g() for g in gs] 的结果}。", "expected": {"fs": [2, 2, 2], "gs": [0, 1, 2]}},
    {"id": "sql_null", "prompt": "SQL 表 t(x) 有四行：NULL、0、0、2。查询 SELECT COUNT(*),COUNT(x),COUNT(DISTINCT x),SUM(x) FROM t。只输出 JSON 数组，按查询列顺序列出结果。", "expected": [4, 3, 2, 2]},
    {"id": "half_open_intervals", "prompt": "半开区间 [1,4)、[4,7)、[3,5)、[10,12)。把有交集或端点相接的区间合并，再求并集的总长度。只输出 JSON：{\"intervals\": [[起点,终点],...], \"length\":数字}，区间升序。", "expected": {"intervals": [[1, 7], [10, 12]], "length": 8}},
    {"id": "percent", "prompt": "一件商品先涨价 20%，再在新价格上降价 20%。最终价格为 192 元。只输出 JSON：{\"original\":原价,\"net_change_percent\":最终相对原价的百分比变化}。下降用负数，不带百分号。", "expected": {"original": 200, "net_change_percent": -4}},
    {"id": "network", "prompt": "IPv4 地址 192.168.7.129/26 所在子网，按传统规则排除网络地址和广播地址。只输出 JSON：{\"network\":网络地址字符串,\"broadcast\":广播地址字符串,\"usable_hosts\":可用主机数}。", "expected": {"network": "192.168.7.128", "broadcast": "192.168.7.191", "usable_hosts": 62}},
    {"id": "date_rollover", "prompt": "上海时间（UTC+8）2026 年 10 月 10 日 00:30，对应 UTC 的日期和时间是什么？只输出 JSON：{\"utc\":\"YYYY-MM-DD HH:MM\"}。", "expected": {"utc": "2026-10-09 16:30"}},
    {"id": "stable_unique", "prompt": "从序列 [3,1,3,2,1,0,2,5,0] 中去重，保留每个数第一次出现的顺序。只输出 JSON 数组。", "expected": [3, 1, 2, 0, 5]},
    {"id": "graph", "prompt": "有向图的边及非负权重：A→B:4，A→C:1，C→B:2，B→D:1，C→D:5。求 A 到 D 的最短路径。只输出 JSON：{\"path\":[节点字符串,...],\"cost\":数字}。", "expected": {"path": ["A", "C", "B", "D"], "cost": 4}},
    {"id": "json_types", "prompt": "按要求构造 JSON：name 是字符串 Lin，active 是布尔真，count 是整数零，note 是 null，tags 是空数组。只输出该对象，不添加其他字段。", "expected": {"name": "Lin", "active": True, "count": 0, "note": None, "tags": []}},
]


def quality(runner, model, port, out):
    scores = []
    for case in QUALITY:
        messages = [{"role": "system", "content": "请准确解答，只输出用户要求的结果。"},
                    {"role": "user", "content": case["prompt"]}]
        row, answer, _ = runner.request(model, "quality-" + case["id"], 1, port, phase="quality",
                                       messages=messages, max_tokens=384)
        try:
            parsed = json.loads(answer.strip())
            passed = parsed == case["expected"] and type(parsed) is type(case["expected"])
        except (ValueError, TypeError):
            passed = False
        scores.append({"case": case["id"], "passed": passed, "answer": answer,
                       "expected": case["expected"], "finish": row["finish"]})
    tool = {"type": "function", "function": {"name": "get_weather", "description": "查询指定城市指定日期的天气。",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                "required": ["city", "date", "unit"], "additionalProperties": False}}}
    before = runner.p.api(port, "metrics")["totals"]["requests"]
    result = runner.p.api(port, "v1/chat/completions", {**common.SAMPLING,
        "messages": [{"role": "system", "content": "今天是 2026-10-09。需要天气时调用工具。"},
                     {"role": "user", "content": "查一下杭州明天的天气，用摄氏度，city 字段使用中文城市名。"}],
        "tools": [tool], "tool_choice": "auto", "max_tokens": 256,
        "session_id": model + "-weather-tool"})
    message = result["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    tool_pass = False
    if len(calls) == 1 and calls[0]["function"]["name"] == "get_weather":
        try:
            tool_pass = json.loads(calls[0]["function"]["arguments"]) == {
                "city": "杭州", "date": "2026-10-10", "unit": "celsius"}
        except (ValueError, TypeError):
            pass
    after = runner.p.api(port, "metrics")["totals"]["requests"]
    if after != before + 1:
        raise RuntimeError("Concurrent or missing tool request")
    scores.append({"case": "tool_arguments", "passed": tool_pass, "message": message})
    data = {"model": model, "passed": sum(r["passed"] for r in scores), "total": len(scores),
            "cases": scores, "scope": "Small synthetic correctness smoke set, not a model quality ranking."}
    common.write_json(out / f"quality-{model}.json", data)
    print("QUALITY " + json.dumps({k: data[k] for k in ("model", "passed", "total")}), flush=True)


def suite(runner, label, port):
    runner.request(label, "short", 0, port, phase="warmup")
    last = None
    for case, reps in (("short", 3), ("medium", 3), ("long32k", 2), ("long128k", 1)):
        for rep in range(1, reps + 1):
            last = runner.request(label, case, rep, port)
    _, output, session = last
    # Park the long conversation, run an unrelated short one, then return to it.
    runner.request(label, "short", 1, port, phase="cache-interleave", max_tokens=64)
    followup = runner.prompts["long128k"] + [{"role": "assistant", "content": output},
                {"role": "user", "content": "Repeat only the exact access codes from records A, B and C, in order."}]
    row, _, _ = runner.request(label, "long128k-followup", 1, port, phase="followup", messages=followup,
                              max_tokens=64, session=session, expect_fresh=False)
    if row["timing"].get("reused", 0) < 128000 or not all(row["retrieval_codes"].values()):
        raise RuntimeError("Long conversation restore/retrieval check failed")


def prepare(p, out, arms):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("private", "responses", "launchers"):
        (out / name).mkdir(exist_ok=True)
    hybrid = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    prompts, recipes, checks = common.make_prompts()
    configs = {}
    for label in arms:
        hybrid_match = re.fullmatch(r"hybrid(\d+)(?:r(\d+))?(?:l(\d+))?", label)
        if hybrid_match:
            cfg = copy.deepcopy(hybrid)
            split = int(hybrid_match[1])
            for group, key in ((2, "--vram-reserve-mib"), (3, "--vram-reserve-later-mib")):
                reserve = hybrid_match[group] or argument(p.configs["v4"]["args"], key)
                cfg["args"] = set_values(cfg["args"], key, [int(reserve)])
        elif label.startswith("v2matched"):
            cfg = copy.deepcopy(p.configs["v4"])
            split = int(label.removeprefix("v2matched"))
            for key in ("--pack", "--native"):
                cfg["args"] = set_values(cfg["args"], key, [argument(p.configs["v2"]["args"], key)])
            cfg["tokenizer"] = p.configs["v2"]["tokenizer"]
        elif label == "v4current22":
            cfg = copy.deepcopy(p.configs["v4"])
            split = 22
        else:
            raise ValueError("Unknown arm " + label)
        cfg["host"] = "127.0.0.1"
        cfg["model_name"] = "benchmark-" + label
        cfg["aliases"] = []
        cfg["log"] = str(out / f"engine-{label}.log")
        cfg["layer_split"] = str(split)
        cfg["args"] = set_values(cfg["args"], "--layer-split", [split])
        name = "bench-" + label
        path = out / "private" / f"{name}.json"
        common.write_json(path, cfg)
        p.paths[name], p.configs[name] = path, cfg
        p.validate(name)
        # Allowlist: no credentials, arbitrary environment values or full configs in public exports.
        configs[label] = {k: cfg[k] for k in ("exe", "cwd", "tokenizer", "model_name", "layer_split", "args")}
    # Confirm both tokenizers and templates produce exactly the same benchmark inputs.
    template_checks = {}
    for name in ("v2", "v4"):
        cfg = p.configs[name]
        tok = Tokenizer.from_gguf(Path(argument(cfg["args"], "--native")))
        tpl = ChatTemplate(Path(cfg["tokenizer"]) / "chat_template.jinja")
        template_checks[name] = {}
        for case, messages in prompts.items():
            rendered = tpl.render(messages, enable_thinking=False)
            ids = tok.encode(rendered, parse_special=True)
            token_hash = common.hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            if token_hash != checks[case]["token_ids_sha256"]:
                raise RuntimeError("Tokenized prompts differ between models")
            template_checks[name][case] = token_hash
    common.write_json(out / "synthetic-prompts.json", prompts)
    common.write_json(out / "prompt-recipes.json", {"checks": checks, "long": recipes, "sampling": common.SAMPLING})
    common.write_json(out / "quality-prompts.json", QUALITY)
    manifest = {"started": datetime.datetime.now().astimezone().isoformat(), "arms": arms,
                "binary": {"path": p.configs["v4"]["exe"], "sha256": common.digest(p.configs["v4"]["exe"])},
                "configs": configs, "sampling": common.SAMPLING, "tokenizer_checks": template_checks,
                "production_config_sha256": common.digest(p.paths["v4"]),
                "v2_config_sha256": common.digest(p.paths["v2"]), "catalog_sha256": common.digest(p.catalog_path),
                "composition_sha256": common.digest(HERE / "composition.json"),
                "note": "v2/hybrid share v4 binary, context, KV, sampling and split; current v4 retains split22."}
    common.write_json(out / "manifest.json", manifest)
    print("PREPARED " + json.dumps({"arms": arms, "tokens": {k: v["prompt_tokens"] for k, v in checks.items()}}), flush=True)
    return prompts, checks, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arms", nargs="+", default=["hybrid19", "v2matched19", "v4current22"])
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--screen-only", action="store_true", help="Only warmup and three short requests per arm")
    args = ap.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "logs") or (out / "results.json").exists():
        ap.error("Use a new directory under this repository's logs/")
    os.chdir(ROOT)
    p = Presets(ROOT / "strata-models.json")
    prompts, checks, manifest = prepare(p, out, args.arms)
    if args.prepare_only:
        return
    tel = common.Telemetry(out / "telemetry.jsonl")
    runner = common.Runner(p, out, tel, prompts, checks)
    tel.start()
    try:
        with p.lock():
            running = p.running()
            if len(running) != 1 or running[0].name != "v4":
                raise RuntimeError("Expected one managed v4 service")
            old = running[0]
            original_logs = p.logs
            common.write_json(out / "private/original-launcher.json", {
                "name": old.name, "pid": old.process.pid, "argv": old.argv, "cwd": old.cwd, "port": old.port})
            p.check_port(running)
            p.port = PORT
            p.check_port(running)
            common.write_json(out / "before.json", runner.snapshot(old))
            changed = False
            try:
                p.drain(old)
                changed = True
                p.record(None)
                p.logs = out / "launchers"
                for label in args.arms:
                    name = "bench-" + label
                    argv = common.flag(common.flag(common.flag(old.argv, "--config", p.paths[name]),
                                       "--host", "127.0.0.1"), "--port", PORT)
                    tel.phase = "load-" + label
                    started = time.perf_counter()
                    runner.current = p.launch(name, previous=Running(name, old.process, argv, old.cwd, PORT))
                    p.record(runner.current)
                    snap = runner.snapshot(runner.current)
                    runner.loads.append({"model": label, "load_seconds": time.perf_counter() - started, **snap})
                    common.write_json(out / "loads.json", runner.loads)
                    print("LOADED " + json.dumps({"model": label, "engine": snap["engine"],
                            "gpu_memory": snap["hardware_snapshot"]["gpus"]}), flush=True)
                    if args.screen_only:
                        runner.request(label, "short", 0, PORT, phase="warmup")
                        for rep in range(1, 4):
                            runner.request(label, "short", rep, PORT)
                    else:
                        suite(runner, label, PORT)
                        quality(runner, label, PORT, out)
                    common.write_json(out / f"after-{label}.json", runner.snapshot(runner.current))
                    p.drain(runner.current)
                    runner.current = None
                    tel.pid = None
                    p.record(None)
            finally:
                if changed:
                    try:
                        if runner.current is not None:
                            try:
                                p.drain(runner.current)
                            except BaseException:
                                # This is our isolated benchmark child, never an unrelated process.
                                p.terminate_owned(runner.current)
                            runner.current = None
                    finally:
                        p.logs = original_logs
                        p.port = old.port
                        tel.phase = "restore-v4"
                        restored = p.launch(old.name, previous=old)
                        p.record(restored)
                        state = runner.snapshot(restored)
                        state["production_config_unchanged"] = common.digest(p.paths["v4"]) == manifest["production_config_sha256"]
                        state["v2_config_unchanged"] = common.digest(p.paths["v2"]) == manifest["v2_config_sha256"]
                        state["catalog_unchanged"] = common.digest(p.catalog_path) == manifest["catalog_sha256"]
                        state["launcher_argv_preserved"] = restored.argv == old.argv
                        common.write_json(out / "restored.json", state)
                        print("RESTORED " + json.dumps(state["health"]), flush=True)
                        runner.request("v4-restored", "short", 0, old.port, phase="warmup", max_tokens=64)
    except BaseException as e:
        common.write_json(out / "failure.json", {"type": type(e).__name__, "message": str(e), "time": time.time()})
        raise
    finally:
        tel.close()
    common.write_json(out / "completion.json", {"completed": datetime.datetime.now().astimezone().isoformat(),
                                               "requests": len(runner.results), "restored": True})
    print("COMPLETE " + str(out), flush=True)


if __name__ == "__main__":
    main()
