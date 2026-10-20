"""Create the self-contained Windows CUDA IQ_KT release from portable builds.

See docs/WINDOWS_PREBUILT.zh-CN.md for the build and package commands.
Only an explicit allowlist of repository files enters the archive. Local configs,
model packs, keys, Python installations and build logs are never copied.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from prepare_dependencies import archive_path, digest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/hip"))
from dll_closure import pe_imports

SYSTEM_DLLS = set("""kernel32.dll kernelbase.dll user32.dll gdi32.dll advapi32.dll shell32.dll ole32.dll
oleaut32.dll shlwapi.dll ws2_32.dll bcrypt.dll crypt32.dll ntdll.dll secur32.dll rpcrt4.dll psapi.dll
dbghelp.dll version.dll winmm.dll dnsapi.dll iphlpapi.dll msvcrt.dll comdlg32.dll setupapi.dll
winspool.drv comctl32.dll ucrtbase.dll dwmapi.dll normaliz.dll ncrypt.dll cryptbase.dll wtsapi32.dll
wintrust.dll powrprof.dll cfgmgr32.dll imm32.dll winhttp.dll mswsock.dll uxtheme.dll gdiplus.dll
msimg32.dll dhcpcsvc.dll hid.dll cabinet.dll netapi32.dll mpr.dll userenv.dll propsys.dll
authz.dll usp10.dll bcryptprimitives.dll cryptsp.dll dbgcore.dll win32u.dll pdh.dll
nvcuda.dll nvml.dll vulkan-1.dll""".split())
TOOLS = ("iq_pack.py", "gguf_reader.py", "kt_quants.py", "_paths.py", "strata_tokenizer.py", "vision_vocab.py")
CUDA_DLLS = ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll")
GGML_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd).decode("utf-8").strip()


def cache_values(build: Path) -> dict:
    values = {}
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line and ":" in line.split("=", 1)[0]:
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    return values


def executable(build: Path, name: str) -> Path:
    candidates = (build / name, build / "Release" / name, build / "bin" / name, build / "bin/Release" / name)
    return next(p for p in candidates if p.is_file())


def copy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def install_python(stage: Path, downloads: Path, lock: dict) -> None:
    for record in [lock["python"], *lock["wheels"]]:
        source = downloads / record["filename"]
        if digest(source) != record["sha256"]:
            raise ValueError("SHA256 mismatch: " + source.name)
        base = stage / "python" if record is lock["python"] else stage / "python/Lib/site-packages"
        with zipfile.ZipFile(source) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                rel = archive_path(item.filename)
                if rel.parts[0].endswith(".data"):
                    if rel.parts[1] not in ("purelib", "platlib"):
                        continue  # CLI scripts and C headers are not needed by this runtime.
                    rel = Path(*rel.parts[2:])
                dest = base / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as stream, dest.open("wb") as output:
                    shutil.copyfileobj(stream, output)
    version = lock["python"]["version"].split(".")
    short = "".join(version[:2])
    (stage / "python" / f"python{short}._pth").write_text(
        f"python{short}.zip\n.\nLib/site-packages\n..\n../tools\n../third_party/llama.cpp/gguf-py\nimport site\n",
        encoding="utf-8")


def audit_imports(stage: Path) -> dict:
    files = [p for p in stage.rglob("*") if p.is_file() and p.suffix.lower() in (".exe", ".dll", ".pyd")]
    bundled = {p.name.lower() for p in files}
    result, missing = {}, set()
    for file in files:
        names = pe_imports(file)
        result[file.relative_to(stage).as_posix()] = names
        for dep in names:
            name = dep.lower()
            if name not in bundled and name not in SYSTEM_DLLS and not name.startswith(("api-ms-", "ext-ms-")):
                missing.add((file.name, dep))
    if missing:
        raise RuntimeError("unbundled non-system imports: " + repr(sorted(missing)))
    return result


def make_package(a) -> Path:
    cfg, vcfg = cache_values(a.build), cache_values(a.vision_build)
    for name, values in (("engine", cfg), ("vision", vcfg)):
        if values.get("STRATA_PORTABLE") != "ON" or values.get("GGML_NATIVE") != "OFF":
            raise ValueError(name + " must be STRATA_PORTABLE=ON / GGML_NATIVE=OFF")
        if any(values.get(flag) == "ON" for flag in ("GGML_AVX512", "GGML_AVX512_VBMI", "GGML_AVX512_VNNI",
                                                     "GGML_AVX512_BF16", "GGML_AVX_VNNI", "GGML_AMX_TILE",
                                                     "GGML_AMX_INT8", "GGML_AMX_BF16")):
            raise ValueError(name + " ggml CPU baseline must not exceed AVX2")
    archs = cfg["CMAKE_CUDA_ARCHITECTURES"].split(";")
    if "120" not in archs or cfg.get("CMAKE_CUDA_RUNTIME_LIBRARY") != "Shared":
        raise ValueError("build requires sm120 and the shared CUDA runtime")
    if vcfg.get("STRATA_VISION_CUDA") != "OFF" or vcfg.get("STRATA_VISION_VULKAN") != "ON":
        raise ValueError("vision must be Vulkan-only")
    if git("rev-parse", "HEAD", cwd=a.ggml) != GGML_COMMIT:
        raise ValueError("ggml checkout must be the pinned commit " + GGML_COMMIT)
    binary = executable(a.build, "strata.exe")
    dump = subprocess.check_output([str(a.cuda / "bin/cuobjdump.exe"), "--list-elf", str(binary)], text=True)
    sass_archs = sorted(set(re.findall(r"sm_(\d+)", dump)), key=int)
    if not set(archs) <= set(sass_archs):
        raise ValueError("missing native GPU code: " + repr(sass_archs))
    nvcc = subprocess.check_output([str(a.cuda / "bin/nvcc.exe"), "--version"], text=True).strip()
    if not re.search(r"release 13\.", nvcc):
        raise ValueError("CUDA 13 is required")
    stem = f"strata-{a.version}-windows-x64-cuda13"
    stage = a.out / stem
    stage.mkdir(parents=True, exist_ok=False)  # Never recursively remove an existing directory.
    for name in ("strata.exe", "strata-device.exe"):
        copy(executable(a.build, name), stage / "engine" / name)
    copy(executable(a.vision_build, "strata-vision.exe"), stage / "engine/strata-vision.exe")
    for name in CUDA_DLLS:
        copy(a.cuda / "bin/x64" / name, stage / "engine" / name)
    runtime = json.loads((ROOT / "tools/windows/runtime-lock.json").read_text(encoding="utf-8"))
    install_python(stage, a.downloads, runtime)
    for source in a.crt.glob("*.dll"):
        copy(source, stage / "engine" / source.name)
        # Python embeds its own current vcruntime: do not downgrade it to the build compiler's copy.
        if not (stage / "python" / source.name).exists():
            copy(source, stage / "python" / source.name)
    for rel in git("ls-files", "serve").splitlines():
        p = Path(rel)
        if p.name.startswith("test_") or "fixtures" in p.parts or p.name in ("mcp_fake_server.py", "chat_golden.json"):
            continue
        copy(ROOT / p, stage / p)
    for name in TOOLS:
        copy(ROOT / "tools" / name, stage / "tools" / name)
    for rel in ("tools/windows/portable.py", "tools/windows/templates/v4-5080-16g.json",
                "tools/windows/runtime-lock.json", "tools/windows/cuda-redist-lock.json",
                "data/expert-profile.bin", "LICENSE", "third_party/ik_kt/LICENSE",
                "docs/WINDOWS_PREBUILT.zh-CN.md", "bench/results/2026-10-09-windows-cuda13/README.md",
                "bench/results/2026-10-09-windows-cuda13/validation.json"):
        copy(ROOT / rel, stage / rel)
    for source in (ROOT / "tools/windows/launchers").glob("*.bat"):
        (stage / source.name).write_bytes(source.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8"))
    copy(ROOT / "docs/WINDOWS_PREBUILT.zh-CN.md", stage / "README.txt")
    shutil.copytree(a.ggml / "gguf-py/gguf", stage / "third_party/llama.cpp/gguf-py/gguf",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    copy(a.ggml / "LICENSE", stage / "licenses/llama.cpp-LICENSE")
    copy(stage / "python/LICENSE.txt", stage / "licenses/Python-LICENSE.txt")
    for component in ("cuda_cudart", "libcublas"):
        copy(a.cuda / "licenses" / component / "LICENSE", stage / "licenses" / component / "LICENSE")
    (stage / "licenses/NOTICE.txt").write_text(
        "Strata and its KT extensions: see /LICENSE and /third_party/ik_kt/LICENSE.\n"
        "ggml / llama.cpp / gguf-py: see llama.cpp-LICENSE.\n"
        "NVIDIA CUDA runtime and cuBLAS: unmodified official redistributables under the included NVIDIA EULAs.\n"
        "Microsoft Visual C++ runtime: Visual Studio 2022 Distributable Code, app-local runtime DLLs.\n"
        "https://learn.microsoft.com/visualstudio/releases/2022/redistribution\n"
        "Python: included PSF license; wheel licenses and notices remain in python/Lib/site-packages/*.dist-info.\n"
        "Web font: serve/web/fonts/OFL.txt. Model weights and graphics drivers are not included.\n",
        encoding="utf-8")
    imports = audit_imports(stage)
    (stage / "DLL_IMPORTS.json").write_text(json.dumps(imports, indent=2) + "\n", encoding="utf-8")
    manifest = {"release": a.version, "backend": "cuda", "platform": "windows-x64",
                "built_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source_commit": git("rev-parse", "HEAD"), "source_dirty": bool(git("status", "--porcelain")),
                "upstream_commit": "fb58e0dbc8399662c0e47c76578c6e878b14f6cf", "ggml_commit": GGML_COMMIT,
                "cuda_toolkit": "13.0.2", "nvcc": nvcc, "sass_architectures": sass_archs,
                "cpu_baseline": "AVX2", "msvc_runtime": "static engine; app-local redistributables for dependencies",
                "vision": "Vulkan only, portable AVX2", "minimum_nvidia_driver_branch": 580,
                "python": runtime["python"], "wheels": runtime["wheels"],
                "cuda_sources": json.loads((ROOT / "tools/windows/cuda-redist-lock.json").read_text()),
                "validation": "See README.txt. Native sm120 code included; no physical RTX 5080 was available."}
    (stage / "BUILD.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    lines = [digest(p) + "  " + p.relative_to(stage).as_posix() for p in files]
    (stage / "FILES.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive = a.out / (stem + ".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            zipped.write(path, stem + "/" + path.relative_to(stage).as_posix())
    archive.with_suffix(".zip.sha256").write_text(digest(archive) + "  " + archive.name + "\n", encoding="utf-8")
    print(archive, f"({archive.stat().st_size / 2**20:.1f} MiB)")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("build", "vision-build", "cuda", "downloads", "ggml", "crt", "out"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--version", default="0.1.41-iqkt.1")
    make_package(parser.parse_args())
