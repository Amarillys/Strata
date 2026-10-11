"""Export only synthetic benchmark evidence; never copy private configs/logs."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    results, loads, runs, quality = [], [], [], []
    for run in args.runs:
        if not (run / "completion.json").exists():
            raise RuntimeError(f"Run not complete: {run}")
        results.extend(read(run / "results.json"))
        loads.extend(read(run / "loads.json"))
        manifest = read(run / "manifest.json")
        runs.append({"directory": str(run), "manifest": manifest, "completion": read(run / "completion.json")})
        quality.extend(read(p) for p in sorted(run.glob("quality-*.json")) if p.name != "quality-prompts.json")
    groups = collections.defaultdict(list)
    for row in results:
        if row["phase"] == "measured":
            if row.get("invalid_reason") or row.get("early_eos"):
                raise RuntimeError("Invalid or unequal-length throughput measurement")
            groups[row["model"], row["case"]].append(row)
    summary = []
    for (model, case), rows in groups.items():
        item = {"model": model, "case": case, "repetitions": len(rows),
                "prompt_tokens": rows[0]["usage"]["prompt_tokens"],
                "completion_tokens": rows[0]["usage"]["completion_tokens"]}
        for key in ("decode_tok_s", "prefill_tok_s", "ttft_s", "client_total_s"):
            vals = [r[key] for r in rows]
            item[key] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
        summary.append(item)
    memory = []
    for load in loads:
        model = load["model"]
        measured = [r for r in results if r["model"] == model and r.get("memory", {}).get("sample_count", 0)]
        gpus = []
        for i in range(len(load["hardware_snapshot"]["gpus"])):
            values = [r["memory"]["gpus"][i] for r in measured]
            gpus.append({"index": i, "peak_used_mib": max(r["peak_used_mib"] for r in values),
                         "min_free_mib": min(r["min_free_mib"] for r in values)})
        memory.append({"model": model, "expert_slots": load["engine"]["expert_slots"],
                       "expert_slots_primary": load["engine"]["expert_slots_primary"],
                       "expert_total": 24576, "expert_cache_mib": load["engine"]["expert_cache_mib"],
                       "expert_ram_mib": load["engine"]["arena_mib"], "gpus": gpus})
    data = {"throughput": summary, "residency": memory,
            "quality": [{k: q[k] for k in ("model", "passed", "total", "scope")} for q in quality],
            "method": "Median of fresh, fixed-input, 256-token requests; warmups excluded.",
            "limits": ["Synthetic correctness/recall probes do not establish overall quality superiority.",
                       "No BF16 reference NLL/KLD, full 300K request, or 512K pool stress test.",
                       "Current v4 uses split22; v2 and hybrid use split19; see each runtime config."]}
    write(HERE / "summary.json", data)
    write(HERE / "results.json", results)
    write(HERE / "runtime.json", {"runs": runs, "loads": loads})
    write(HERE / "quality.json", quality)
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
