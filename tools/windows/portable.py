"""Offline first-run and launch support for the Windows IQ_KT release.

The release keeps this file at tools/windows/portable.py. Paths in strata-v4.json
are relative to the extracted release, including after moving that directory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "strata-v4.json"
MARKERS = ("native_experts.txt", "dense.bin", "index.txt", "tokenizer/vocab.json",
           "tokenizer/merges.txt", "tokenizer/token_type.json", "tokenizer/chat_template.jinja")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def local_path(value: str | Path, root: Path = ROOT) -> Path:
    path = Path(str(value).strip().strip('"')).expanduser()
    # Keep symlink names: the shard suffix is needed to find the other GGUF files.
    return path.absolute() if path.is_absolute() else (root / path).absolute()


def portable_path(path: Path, root: Path = ROOT) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def value_of(args: list[str], flag: str) -> str:
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        raise ValueError(f"配置缺少 {flag}") from None


def set_arg(args: list[str], flag: str, value: str | int) -> None:
    if flag in args:
        i = args.index(flag)
        if i + 1 == len(args):
            raise ValueError(f"配置中的 {flag} 缺少值")
        args[i + 1] = str(value)
    else:
        args.extend([flag, str(value)])


def runtime_env(root: Path = ROOT) -> dict:
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    env["PATH"] = os.pathsep.join([str(root / "engine"), str(root / "python"), env.get("PATH", "")])
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["STRATA_GGUF_PY"] = str(root / "third_party" / "llama.cpp" / "gguf-py")
    return env


def validate_config(cfg: dict) -> None:
    args = cfg.get("args")
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        raise ValueError("args 必须是字符串数组")
    for flag, lo, hi in (("--max-context", 4096, 262144), ("--prefill", 256, 8192),
                         ("--kv-resident", 20480, 262144), ("--vram-reserve-mib", 0, 131072),
                         ("--prompt-cache", 0, 64)):
        try:
            n = int(value_of(args, flag))
        except ValueError:
            raise ValueError(f"{flag} 必须是 {lo}..{hi} 的整数") from None
        if not lo <= n <= hi:
            raise ValueError(f"{flag} 必须在 {lo}..{hi} 范围")
    if value_of(args, "--kv") not in ("int8", "k8v4", "fp16", "q4_0"):
        raise ValueError("--kv 可选 int8、k8v4、fp16、q4_0；K8V8 对应 int8")
    if not isinstance(cfg.get("gpu"), int) or isinstance(cfg["gpu"], bool) or cfg["gpu"] < 0:
        raise ValueError("这个模板的 gpu 是单张 NVIDIA 卡的非负编号")
    port = cfg.get("port", 8080)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port 必须是 1..65535 的整数")
    if not isinstance(cfg.get("host"), str) or not cfg["host"].strip():
        raise ValueError("host 不能为空")
    if not isinstance(cfg.get("api_key"), str) or not cfg["api_key"].strip():
        raise ValueError("需要本机 API key；先运行 CONFIGURE.bat")
    if "--layer-split" in args or "--peer-device" in args:
        raise ValueError("这是单卡模板，请移除双卡分层/peer 参数")


def inspect_model(first: Path) -> dict:
    match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name, re.I)
    if match and int(match.group(1)) != 1:
        raise ValueError("请选择 -00001-of-00002.gguf 第一分片")
    sys.path.insert(0, str(ROOT / "tools"))
    from iq_pack import Model
    model = Model(first)
    metadata = model.files[0].metadata
    arch = metadata.get("general.architecture")
    if arch != "qwen4exp" or metadata.get(f"{arch}.block_count") != 48:
        raise ValueError("此模板用于 48 层 Qwen3.8-Flash-Next (qwen4exp)")
    if not any(t.type_name == "IQ3_KT" for g in model.files for t in g.tensors):
        raise ValueError("模型中没有 IQ3_KT 张量，请检查文件")
    shards = []
    for path in model.paths:
        stat = path.stat()
        with path.open("rb") as stream:
            head_hash = hashlib.sha256(stream.read(1024 * 1024)).hexdigest()
        shards.append({"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                       "header_sha256": head_hash})
    return {"schema": 1, "shards": shards}


def prepare_pack(first: Path, pack: Path, source: dict, root: Path = ROOT) -> None:
    receipt = pack / "portable-source.json"
    if receipt.exists() and read_json(receipt) != source:
        raise ValueError("此 pack 属于其他模型或文件已变化，请用 --pack 指定新目录")
    complete = all((pack / p).is_file() for p in MARKERS)
    if complete and receipt.exists():
        print("复用已完成的模型包。", flush=True)
        return
    if pack.exists() and any(pack.iterdir()) and not receipt.exists():
        raise ValueError("pack 目录已有其他文件，请用 --pack 指定一个空目录")
    pack.mkdir(parents=True, exist_ok=True)
    write_json(receipt, source)  # An interrupted first run may resume, but never with another model.
    print("准备模型：保留专家和 PLE 的原始 KT 数据，转换运行时需要的小投影。", flush=True)
    subprocess.run([sys.executable, "-X", "utf8", "-u", str(root / "tools" / "iq_pack.py"),
                    "--gguf", str(first), "--out", str(pack), "--compat-bf16"],
                   cwd=root, env=runtime_env(root), check=True)
    if not all((pack / p).is_file() for p in MARKERS):
        raise RuntimeError("模型包没有完整生成；修复错误后可重跑 CONFIGURE.bat")


def configure(a, root: Path = ROOT) -> int:
    path = local_path(a.config, root)
    cfg = read_json(path) if path.exists() else read_json(root / "tools/windows/templates/v4-5080-16g.json")
    cfg = copy.deepcopy(cfg)
    prior = value_of(cfg["args"], "--native")
    model = a.model or (prior if prior != "__MODEL_FIRST_SHARD__" else None)
    if not model:
        print("把 v4 两份 GGUF 放在同一 SSD 文件夹中，保持原文件名。")
        model = input("第一分片 GGUF 完整路径（可拖入文件）: ").strip()
    if not model:
        raise ValueError("未提供模型路径")
    first = local_path(model, root)
    pack = local_path(a.pack or value_of(cfg["args"], "--pack"), root)
    set_arg(cfg["args"], "--native", portable_path(first, root))
    set_arg(cfg["args"], "--pack", portable_path(pack, root))
    cfg["tokenizer"] = portable_path(pack / "tokenizer", root)
    for attr, flag in (("context", "--max-context"), ("prefill", "--prefill"),
                       ("kv", "--kv"), ("reserve_mib", "--vram-reserve-mib")):
        if getattr(a, attr) is not None:
            set_arg(cfg["args"], flag, getattr(a, attr))
    for attr in ("gpu", "host", "port"):
        if getattr(a, attr) is not None:
            cfg[attr] = getattr(a, attr)
    if not cfg.get("api_key"):
        cfg["api_key"] = "sk-strata-" + secrets.token_urlsafe(32)
    validate_config(cfg)
    source = inspect_model(first)
    print(f"模型文件合计 {sum(s['bytes'] for s in source['shards']) / 2**30:.2f} GiB；不会复制整套权重。")
    prepare_pack(first, pack, source, root)
    for folder in ("logs", "local"):
        (root / folder).mkdir(exist_ok=True)
    if cfg.get("vision"):
        # A new model needs its own vocabulary, even when keeping the same MMPROJ.
        cfg["vision"]["model"] = ensure_vision_vocab(first, pack, root)
    write_json(path, cfg)
    (root / "local/API_KEY.txt").write_text(cfg["api_key"] + "\n", encoding="utf-8")
    print(f"\n配置完成：{path.name}\nAPI key：local/API_KEY.txt（请勿公开）")
    print(f"上下文 {value_of(cfg['args'], '--max-context')}，KV {value_of(cfg['args'], '--kv')}，"
          f"prefill {value_of(cfg['args'], '--prefill')}。")
    print("双击 START.bat 启动；Web 和 Monitor 与 API 共用端口 " + str(cfg["port"]) + "。")
    return 0


def ensure_vision_vocab(first: Path, pack: Path, root: Path = ROOT) -> str:
    dest = pack / "tokenizer/vision-vocab.gguf"
    if not dest.exists():
        subprocess.run([sys.executable, "-X", "utf8", str(root / "tools/vision_vocab.py"),
                        "--model", str(first), "--output", str(dest)],
                       cwd=root, env=runtime_env(root), check=True)
    return portable_path(dest, root)


def intel_device(output: str) -> str:
    matches = [description.strip() for backend, description in
               (line.split(":", 1) for line in output.splitlines() if ":" in line)
               if backend.strip().startswith("Vulkan") and "intel" in description.lower()]
    if len(matches) != 1:
        raise ValueError("未找到唯一的 Intel Vulkan 核显；运行 CHECK.bat --vision 查看设备，"
                         "再用 ENABLE-VISION.bat --device \"完整设备名\"")
    return matches[0]


def enable_vision(a, root: Path = ROOT) -> int:
    path = local_path(a.config, root)
    cfg = read_json(path)
    validate_config(cfg)
    mmproj = a.mmproj or input("MMPROJ GGUF 完整路径（留空取消）: ").strip()
    if not mmproj:
        return 0
    mmproj = local_path(mmproj, root)
    if not mmproj.is_file():
        raise ValueError(f"找不到 MMPROJ：{mmproj}")
    env = runtime_env(root)
    env.pop("GGML_VK_VISIBLE_DEVICES", None)
    exe = root / "engine/strata-vision.exe"
    listing = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", env=env, check=True)
    device = a.device or intel_device(listing.stdout)
    choices = [part.strip() for line in listing.stdout.splitlines() if ":" in line
               for part in line.split(":", 1)]
    if device not in choices:
        raise ValueError(f"视觉设备不可用：{device}")
    first = local_path(value_of(cfg["args"], "--native"), root)
    pack = local_path(value_of(cfg["args"], "--pack"), root)
    cfg["vision"] = {"exe": "engine/strata-vision.exe", "mmproj": portable_path(mmproj, root),
                     "model": ensure_vision_vocab(first, pack, root), "gpu": True, "device": device,
                     "max_tokens": 768, "env": {"CUDA_VISIBLE_DEVICES": "-1"}}
    if "--vision" not in cfg["args"]:
        cfg["args"].append("--vision")
    write_json(path, cfg)
    print("视觉设备：" + device + "。重启 START.bat 后生效。")
    return 0


def start(a, root: Path = ROOT) -> int:
    path = local_path(a.config, root)
    if not path.is_file():
        raise ValueError("请先运行 CONFIGURE.bat，填写模型路径")
    cfg = read_json(path)
    validate_config(cfg)
    if not local_path(cfg["exe"], root).is_file():
        raise ValueError("缺少 engine/strata.exe，请完整解压预编译包")
    pack = local_path(value_of(cfg["args"], "--pack"), root)
    missing = [str(pack / p) for p in MARKERS if not (pack / p).is_file()]
    if missing:
        raise ValueError("模型包不完整，请重跑 CONFIGURE.bat：" + missing[0])
    if not local_path(value_of(cfg["args"], "--native"), root).is_file():
        raise ValueError("原 GGUF 已移动，请重跑 CONFIGURE.bat --model \"新的第一分片路径\"")
    for folder in ("logs", "local"):
        (root / folder).mkdir(exist_ok=True)
    (root / "local/API_KEY.txt").write_text(cfg["api_key"] + "\n", encoding="utf-8")
    print(f"Web/Monitor: http://127.0.0.1:{cfg['port']}/\nAPI: http://127.0.0.1:{cfg['port']}/v1")
    print("API key 保存在 local/API_KEY.txt；Ctrl+C 结束服务。", flush=True)
    env = runtime_env(root)
    if cfg.get("vision"):
        env.pop("GGML_VK_VISIBLE_DEVICES", None)
    cmd = [sys.executable, "-X", "utf8", "-u", "-m", "serve.server", "--engine", "strata",
           "--config", str(path), "--port", str(cfg["port"])]
    if not a.no_browser:
        cmd.append("--open")
    return subprocess.call(cmd, cwd=root, env=env)


def check(a, root: Path = ROOT) -> int:
    print("Python:", sys.version.split()[0], "(64 bit)" if sys.maxsize > 2**32 else "(32 bit)")
    for name in ("numpy", "regex", "jinja2", "PIL", "psutil"):
        module = importlib.import_module(name)
        print(name + ":", getattr(module, "__version__", "OK"))
    import psutil
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.total / 2**30:.1f} GiB total, {mem.available / 2**30:.1f} GiB available")
    env = runtime_env(root)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    result = subprocess.run([str(root / "engine/strata-device.exe"), "--list-devices"], env=env, cwd=root,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode:
        return result.returncode
    if not re.search(r"^device \d+:", result.stdout, re.M):
        raise RuntimeError("CUDA 未检测到 GPU；请检查 NVIDIA 驱动是否支持 CUDA 13")
    if a.vision:
        env.pop("GGML_VK_VISIBLE_DEVICES", None)
        return subprocess.call([str(root / "engine/strata-vision.exe"), "--list-devices"], env=env, cwd=root)
    print("驱动须支持 CUDA 13（NVIDIA R580 或更新）；无需安装 CUDA Toolkit。")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    conf = commands.add_parser("configure", help="prepare v4 and create the local config")
    conf.add_argument("--model")
    conf.add_argument("--pack")
    conf.add_argument("--context", type=int)
    conf.add_argument("--prefill", type=int)
    conf.add_argument("--kv", choices=("int8", "k8v4", "fp16", "q4_0"))
    conf.add_argument("--reserve-mib", type=int)
    conf.add_argument("--gpu", type=int)
    conf.add_argument("--host")
    conf.add_argument("--port", type=int)
    run = commands.add_parser("start", help="start the local Web/API server")
    run.add_argument("--no-browser", action="store_true")
    vision = commands.add_parser("vision", help="enable optional Intel Vulkan image input")
    vision.add_argument("--mmproj")
    vision.add_argument("--device")
    for p in (conf, run, vision):
        p.add_argument("--config", default=DEFAULT_CONFIG)
    chk = commands.add_parser("check", help="check the bundled runtime and GPU drivers")
    chk.add_argument("--vision", action="store_true")
    a = parser.parse_args(argv)
    os.chdir(ROOT)
    try:
        return {"configure": configure, "start": start, "vision": enable_vision, "check": check}[a.command](a)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("\nStrata: " + str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
