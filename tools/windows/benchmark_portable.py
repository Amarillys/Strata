"""Measure this bundle's local Strata server; save timings and hardware, never credentials or chat text."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import portable as P


class Sampler:
    def __init__(self, gpu_index):
        import psutil
        from serve.telemetry import gpu_reader, _cpu_name
        self.ps, self.gpu = psutil, gpu_reader(gpu_index)
        self.hardware = {"cpu": _cpu_name(), "physical_cores": psutil.cpu_count(logical=False),
                         "logical_threads": psutil.cpu_count(), "ram_gib": psutil.virtual_memory().total / 2**30,
                         "gpu_index": gpu_index, "gpu": self.gpu.name() if self.gpu.ok() else None}
        self.samples, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.is_set():
            memory = self.ps.virtual_memory()
            sample = {"ram_available_gib": memory.available / 2**30}
            if self.gpu.ok():
                values = self.gpu.read()
                sample.update({k: values.get(k) for k in
                               ("mem_used", "mem_total", "util", "power", "pcie_gen", "pcie_width")})
            self.samples.append(sample)
            self.stop.wait(0.5)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=2)
        result = {"samples": len(self.samples), "interval_seconds": 0.5}
        for source, target, operation, scale in (
                ("ram_available_gib", "minimum_available_ram_gib", min, 1),
                ("mem_used", "peak_device_vram_mib", max, 2**20),
                ("mem_total", "device_vram_total_mib", max, 2**20),
                ("util", "peak_gpu_utilization_percent", max, 1),
                ("power", "peak_gpu_watts", max, 1),
                ("pcie_gen", "maximum_observed_pcie_generation", max, 1),
                ("pcie_width", "maximum_observed_pcie_width", max, 1)):
            values = [s[source] for s in self.samples if isinstance(s.get(source), (int, float))]
            result[target] = operation(values) / scale if values else None
        return result


def completion(opener, base, key, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "seed": 20261010,
            "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(base + "/v1/chat/completions",
                                    data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    start, first = time.perf_counter(), None
    usage, timings, finish, output = {}, {}, None, hashlib.sha256()
    with opener.open(request, timeout=900) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                break
            item = json.loads(data)
            if item.get("error"):
                raise RuntimeError("服务在测试请求中返回错误；请检查本机引擎日志")
            if item.get("usage"):
                usage = item["usage"]
            if item.get("timings"):
                timings = item["timings"]
            for choice in item.get("choices", []):
                delta = choice.get("delta") or {}
                content = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if content:
                    if first is None:
                        first = time.perf_counter()
                    output.update(content.encode("utf-8"))
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    if not usage or not timings or first is None:
        raise RuntimeError("响应缺少内容或引擎计时；请确认使用此运行包并查看日志")
    timing_keys = ("cache_n", "prompt_n", "prompt_ms", "prompt_per_second",
                   "predicted_n", "predicted_ms", "predicted_per_second")
    return {"prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "output_sha256": output.hexdigest(), "ttft_seconds": first - start,
            "elapsed_seconds": time.perf_counter() - start, "finish_reason": finish,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "timings": {k: timings.get(k) for k in timing_keys}}


def run(a, root=ROOT):
    cfg = P.read_json(P.local_path(a.config, root))
    P.validate_config(cfg)
    base = "http://127.0.0.1:" + str(cfg["port"])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(base + "/health", headers={"Authorization": "Bearer " + cfg["api_key"]})
    with opener.open(request, timeout=10) as response:
        health = json.load(response)
    if not health.get("loaded") or health.get("model") != cfg["model_name"]:
        raise ValueError("此端口未运行配置指定的 v4-lite，请先用 START.bat 启动并等待加载完成")
    output_dir = root / "results" / datetime.datetime.now().strftime("benchmark-%Y%m%d-%H%M%S-%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    prompt = "请从 1 开始连续列出到 2000 的整数，用逗号分隔，不要省略、不要解释。"
    paragraph = ("The GPU holds the frequently used experts. Other experts stay in system memory. "
                 "Transfers over PCI Express and CPU execution share the remaining work. "
                 "Memory bandwidth and cache residency affect throughput.\n")
    medium = "以下是背景资料：\n" + paragraph * 48 + "\n请忽略背景资料，" + prompt
    cases = [("warmup", "只输出 6 乘以 7 的结果。", 16), ("short", prompt, a.tokens),
             ("longer_first", medium, a.tokens), ("longer_repeat", medium, a.tokens)]
    fields = ("--max-context", "--prefill", "--kv", "--kv-resident", "--expert-cache", "--vram-reserve-mib",
              "--prompt-cache", "--pool-affinity", "--ple-io")
    report = {"created_at": datetime.datetime.now().astimezone().isoformat(), "model": cfg["model_name"],
              "settings": {flag: P.value_of(cfg["args"], flag) for flag in fields},
              "sampling": {"temperature": 0, "seed": 20261010, "reasoning_effort": "none",
                           "max_tokens": a.tokens}, "cases": [],
              "limits": "NVML is device-wide. TTFT includes HTTP overhead. Check cached_tokens; warmup is separate. "
                        "This is a speed probe, not a quality or full-context benchmark."}
    success = True
    for label, text, count in cases:
        print(f"测试 {label}，最多生成 {count} token...", flush=True)
        sampler = Sampler(cfg["gpu"])
        sampler.start()
        try:
            result = completion(opener, base, cfg["api_key"], cfg["model_name"], text, count)
        except (OSError, ValueError, RuntimeError) as error:
            result = {"error_type": type(error).__name__}
            success = False
        finally:
            memory = sampler.finish()
        report["hardware"] = sampler.hardware
        report["cases"].append({"case": label, **result, "memory": memory})
        P.write_json(output_dir / "results.json", report)
        if not success:
            print("测试中止，请检查服务窗口；已有统计已保存。", flush=True)
            break
        print(f"  输入 {result['prompt_tokens']}，缓存 {result['cached_tokens']}，"
              f"TTFT {result['ttft_seconds']:.2f}s，decode {result['timings']['predicted_per_second']} tok/s",
              flush=True)
    lines = ["# Strata v4-lite target-machine benchmark", "", report["limits"], "",
             "| Case | Input | Cached | Output | TTFT s | Prefill tok/s | Decode tok/s |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["cases"]:
        if "error_type" in row:
            lines += [f"| {row['case']} | {row['error_type']} | | | | | |"]
        else:
            t = row["timings"]
            lines += [f"| {row['case']} | {row['prompt_tokens']} | {row['cached_tokens']} | "
                      f"{row['completion_tokens']} | {row['ttft_seconds']:.3f} | "
                      f"{t['prompt_per_second']} | {t['predicted_per_second']} |"]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("结果已保存：" + str(output_dir), flush=True)
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="strata-v4-lite.json")
    parser.add_argument("--tokens", type=int, default=256)
    a = parser.parse_args()
    if not 16 <= a.tokens <= 4096:
        parser.error("--tokens must be 16..4096")
    try:
        return run(a)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"无法测速（{type(error).__name__}）：请先启动 START.bat，确认模型已加载。", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
