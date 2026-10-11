"""Make the 4070 Ti SUPER v4-lite runtime from the verified Windows IQ_KT release, without model payloads."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from package_release import audit_imports, digest
from prepare_dependencies import archive_path

ROOT = Path(__file__).resolve().parents[2]
BASE_SHA256 = "3cfef042824a250dae3195ff685e4db3bbeb4bad190f029cc12c935b0acc8605"
STEM = "Strata-v4-lite-4070TiS-Windows-20261010"
TOOLS = {"tools/iq_pack.py", "tools/gguf_reader.py", "tools/kt_quants.py", "tools/_paths.py",
         "tools/strata_tokenizer.py", "tools/windows/portable.py"}


def included(name):
    return (name in TOOLS or name in ("LICENSE", "data/expert-profile.bin", "third_party/ik_kt/LICENSE")
            or name.startswith(("python/", "serve/", "licenses/", "third_party/llama.cpp/gguf-py/"))
            or (name.startswith("engine/") and name != "engine/strata-vision.exe"))


def launcher(command, pause=True):
    tail = "pause" if pause else 'if not "%strata_result%"=="0" pause'
    return ("@echo off\nsetlocal\nchcp 65001 >nul\npushd \"%~dp0\"\n"
            '"python\\python.exe" -X utf8 -u ' + command + " %*\n"
            'set "strata_result=%errorlevel%"\n' + tail + "\npopd\nexit /b %strata_result%\n").replace("\n", "\r\n")


def package(a):
    if digest(a.base) != BASE_SHA256:
        raise ValueError("The base ZIP does not match the verified IQ_KT release")
    stage = a.out / STEM
    stage.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(a.base) as source:
        prefix = source.namelist()[0].split("/", 1)[0] + "/"
        expected = {line.split("  ", 1)[1]: line.split("  ", 1)[0]
                    for line in source.read(prefix + "FILES.sha256").decode("utf-8").splitlines()}
        base_build = json.loads(source.read(prefix + "BUILD.json"))
        if "89" not in base_build["sass_architectures"] or base_build["cpu_baseline"] != "AVX2":
            raise ValueError("The release must include native sm89 and the AVX2 CPU baseline")
        for record in source.infolist():
            if record.is_dir() or not record.filename.startswith(prefix):
                continue
            rel = record.filename[len(prefix):]
            if not included(rel):
                continue
            target = stage / archive_path(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            value = hashlib.sha256()
            with source.open(record) as stream, target.open("xb") as output:
                while data := stream.read(8 * 2**20):
                    output.write(data)
                    value.update(data)
            if value.hexdigest() != expected.get(rel):
                raise ValueError("Base member checksum mismatch: " + rel)
    for rel in ("tools/windows/lite_runtime.py", "tools/windows/benchmark_portable.py",
                "tools/windows/templates/v4-lite-4070tis-16g.json"):
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    readme = ROOT / "docs/WINDOWS_V4_LITE_4070TI.zh-CN.md"
    shutil.copyfile(readme, stage / "README.zh-CN.md")
    shutil.copyfile(readme, stage / "README.txt")
    commands = {"START.bat": ("tools\\windows\\lite_runtime.py start", False),
                "CONFIGURE.bat": ("tools\\windows\\lite_runtime.py configure", True),
                "CHECK.bat": ("tools\\windows\\lite_runtime.py check", True),
                "VERIFY-MODEL.bat": ("tools\\windows\\lite_runtime.py verify", True),
                "BENCHMARK.bat": ("tools\\windows\\benchmark_portable.py", True)}
    for name, (command, pause) in commands.items():
        (stage / name).write_bytes(launcher(command, pause).encode("utf-8"))
    exported = json.loads((a.model_dir / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for file in exported["files"]:
        path = a.model_dir / file["filename"]
        if path.stat().st_size != file["bytes"]:
            raise ValueError("Exported model size changed")
        record = {k: file[k] for k in ("filename", "bytes", "sha256", "header_bytes")}
        with path.open("rb") as stream:
            record["header_sha256"] = hashlib.sha256(stream.read(file["header_bytes"])).hexdigest()
        records.append(record)
    model_contract = {"model": "v4-lite with IQ4_KT PLE", "files": records,
                      "expert_type": "IQ3_KT", "ple_type": "IQ4_KT", "routed_expert_tensors": 144,
                      "v4_high_precision_projections": 387}
    (stage / "data/v4-lite-model.json").write_text(json.dumps(model_contract, indent=2) + "\n", encoding="utf-8")
    (stage / "BASE-BUILD.json").write_text(json.dumps(base_build, indent=2) + "\n", encoding="utf-8")
    imports = audit_imports(stage)
    (stage / "DLL_IMPORTS.json").write_text(json.dumps(imports, indent=2) + "\n", encoding="utf-8")
    build = {"package": STEM, "created_at": datetime.datetime.now().astimezone().isoformat(),
             "base_release": base_build["release"], "base_archive_sha256": BASE_SHA256,
             "engine_sha256": digest(stage / "engine/strata.exe"),
             "engine_and_server_unmodified_from_base_release": True,
             "source_commit": base_build["source_commit"], "cpu_baseline": "AVX2",
             "native_cuda_architectures": base_build["sass_architectures"],
             "minimum_nvidia_driver_branch": 580,
             "target": {"os": "Windows 10/11 x64", "gpu": "RTX 4070 Ti SUPER 16 GiB",
                        "ram": "64 GiB DDR4", "pcie": "4.0 x16"},
             "model_payloads_included": False, "python_included": True,
             "cuda_runtime_included": True, "vision_enabled": False,
             "target_gpu_inference_verified": False,
             "launcher_source_sha256": {rel: digest(ROOT / rel) for rel in
                                        ("tools/windows/lite_runtime.py", "tools/windows/benchmark_portable.py",
                                         "tools/windows/templates/v4-lite-4070tis-16g.json")}}
    (stage / "BUILD.json").write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")
    if list(stage.rglob("*.gguf")) or (stage / "strata-v4-lite.json").exists():
        raise ValueError("Unexpected model or private configuration in clean staging")
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    (stage / "FILES.sha256").write_text("".join(digest(p) + "  " + p.relative_to(stage).as_posix() + "\n"
                                                for p in files), encoding="utf-8")
    archive = a.out / (STEM + ".zip")
    print(f"DLL imports verified for {len(imports)} PE files; writing ZIP...", flush=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            out.write(path, STEM + "/" + path.relative_to(stage).as_posix())
    sha = digest(archive)
    archive.with_suffix(".zip.sha256").write_text(sha + "  " + archive.name + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive.resolve()), "bytes": archive.stat().st_size,
                      "sha256": sha, "pe_files_audited": len(imports)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    package(parser.parse_args())
