"""Offline setup and launch for the v4-lite / IQ4_KT PLE Windows bundle."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portable as P

ROOT = P.ROOT
CONFIG = "strata-v4-lite.json"
TEMPLATE = "tools/windows/templates/v4-lite-4070tis-16g.json"
CONTRACT = "data/v4-lite-model.json"


def model_files(value, root=ROOT, full=False):
    contract = P.read_json(root / CONTRACT)
    first = P.local_path(value, root)
    if first.is_dir():
        first /= contract["files"][0]["filename"]
    if first.name != contract["files"][0]["filename"]:
        raise ValueError("请选择本次 v4-lite 的 00001-of-00002.gguf，或它所在的文件夹")
    paths = []
    for record in contract["files"]:
        path = first.with_name(record["filename"])
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError("模型文件缺失或长度不符，请检查传输是否完成：" + path.name)
        with path.open("rb") as stream:
            header = hashlib.sha256(stream.read(record["header_bytes"])).hexdigest()
        if header != record["header_sha256"]:
            raise ValueError("模型头部不匹配：" + path.name)
        if full:
            print("完整 SHA256 校验：" + path.name, flush=True)
            digest, done, previous = hashlib.sha256(), 0, time.monotonic()
            with path.open("rb") as stream:
                while data := stream.read(16 * 2**20):
                    digest.update(data)
                    done += len(data)
                    if time.monotonic() - previous >= 10:
                        print(f"  {100 * done / record['bytes']:.1f}%", flush=True)
                        previous = time.monotonic()
            if digest.hexdigest() != record["sha256"]:
                raise ValueError("SHA256 不匹配，请重新复制：" + path.name)
            print("  SHA256 通过", flush=True)
        paths.append(path)
    return paths


def configure(a, root=ROOT):
    config = P.local_path(a.config, root)
    cfg = P.read_json(config if config.exists() else root / TEMPLATE)
    prior = P.value_of(cfg["args"], "--native")
    value = getattr(a, "model", None) or (prior if prior != "__MODEL_FIRST_SHARD__" else None)
    if not value:
        print("两份 v4-lite GGUF 放在同一 SSD 文件夹，保留原文件名。")
        value = input("第一分片路径或模型文件夹（可以拖入）: ").strip()
    if not value:
        raise ValueError("未提供模型路径")
    first, ple = model_files(value, root)
    pack = P.local_path(getattr(a, "pack", None) or P.value_of(cfg["args"], "--pack"), root)
    for flag, path in (("--native", first), ("--native-head-gguf", first),
                       ("--native-dense-gguf", first), ("--ple-gguf", ple), ("--pack", pack)):
        P.set_arg(cfg["args"], flag, P.portable_path(path, root))
    cfg["tokenizer"] = P.portable_path(pack / "tokenizer", root)
    for attr, flag in (("context", "--max-context"), ("prefill", "--prefill"),
                       ("reserve_mib", "--vram-reserve-mib"), ("kv", "--kv")):
        if getattr(a, attr, None) is not None:
            P.set_arg(cfg["args"], flag, getattr(a, attr))
    for attr in ("gpu", "port", "host"):
        if getattr(a, attr, None) is not None:
            cfg[attr] = getattr(a, attr)
    if not cfg.get("api_key"):
        cfg["api_key"] = "sk-strata-" + secrets.token_urlsafe(32)
    P.validate_config(cfg)
    source = P.inspect_model(first)
    P.prepare_pack(first, pack, source, root)
    for folder in ("logs", "local"):
        (root / folder).mkdir(exist_ok=True)
    P.write_json(config, cfg)
    (root / "local/API_KEY.txt").write_text(cfg["api_key"] + "\n", encoding="utf-8")
    print(f"配置完成：{config.name}；API key 保存在 local/API_KEY.txt。")
    print(f"上下文 {P.value_of(cfg['args'], '--max-context')}，KV {P.value_of(cfg['args'], '--kv')}，"
          f"prefill {P.value_of(cfg['args'], '--prefill')}。")
    print("专家来自主分片，IQ4_KT PLE 来自第二分片；无需原 v2/v4。", flush=True)
    return 0


def start(a, root=ROOT):
    if not P.local_path(a.config, root).is_file():
        configure(a, root)
    cfg = P.read_json(P.local_path(a.config, root))
    first, ple = model_files(P.value_of(cfg["args"], "--native"), root)
    expected = (("--native-head-gguf", first), ("--native-dense-gguf", first), ("--ple-gguf", ple))
    if any(P.local_path(P.value_of(cfg["args"], flag), root) != path for flag, path in expected):
        raise ValueError("权重来源不匹配，请重跑 CONFIGURE.bat")
    return P.start(a, root)


def verify(a, root=ROOT):
    value = a.model
    if not value:
        path = P.local_path(a.config, root)
        if path.is_file():
            value = P.value_of(P.read_json(path)["args"], "--native")
        else:
            value = input("第一分片路径或模型文件夹: ").strip()
    model_files(value, root, full=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    configure_parser = sub.add_parser("configure", help="填写模型路径并生成 pack")
    configure_parser.add_argument("--model")
    configure_parser.add_argument("--pack")
    configure_parser.add_argument("--context", type=int)
    configure_parser.add_argument("--prefill", type=int)
    configure_parser.add_argument("--reserve-mib", type=int)
    configure_parser.add_argument("--kv", choices=("int8", "k8v4", "fp16", "q4_0"))
    configure_parser.add_argument("--gpu", type=int)
    configure_parser.add_argument("--port", type=int)
    configure_parser.add_argument("--host")
    run = sub.add_parser("start", help="首次配置后启动 Web/API")
    run.add_argument("--no-browser", action="store_true")
    check = sub.add_parser("check", help="检查随包依赖与显卡驱动，不加载模型")
    check.set_defaults(vision=False)
    checksum = sub.add_parser("verify", help="完整校验两份 GGUF，不加载模型")
    checksum.add_argument("--model")
    for command in (configure_parser, run, checksum):
        command.add_argument("--config", default=CONFIG)
    a = parser.parse_args(argv)
    os.chdir(ROOT)
    try:
        if a.command == "check":
            return P.check(a)
        return {"configure": configure, "start": start, "verify": verify}[a.command](a)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Strata: " + str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
