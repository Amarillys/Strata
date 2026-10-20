"""Download the locked official CUDA toolchain and Python runtime; no installation.

python tools/windows/prepare_dependencies.py --out build-release-deps
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive_path(name: str, strip: int = 0) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or any(":" in p or "\\" in p for p in relative.parts):
        raise ValueError("unsafe archive path: " + name)
    return Path(*relative.parts[strip:])


def fetch(record: dict, cache: Path) -> Path:
    dest = cache / record["filename"]
    if dest.exists() and digest(dest) == record["sha256"]:
        return dest
    cache.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + ".part")
    print("Downloading", record["filename"], flush=True)
    with urllib.request.urlopen(record["url"], timeout=60) as source, temp.open("wb") as output:
        shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    if digest(temp) != record["sha256"]:
        raise ValueError("SHA256 mismatch: " + record["filename"])
    temp.replace(dest)
    return dest


def prepare(out: Path) -> None:
    cuda = json.loads((HERE / "cuda-redist-lock.json").read_text(encoding="utf-8"))
    runtime = json.loads((HERE / "runtime-lock.json").read_text(encoding="utf-8"))
    cache, toolkit = out / "downloads", out / "cuda-13.0.2"
    for record in [runtime["python"], *runtime["wheels"], *cuda]:
        fetch(record, cache)
    for record in cuda:
        with zipfile.ZipFile(cache / record["filename"]) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                rel = archive_path(item.filename, strip=1)
                dest = toolkit / rel
                if len(rel.parts) == 1 and any(s in rel.name.lower() for s in ("license", "eula", "notice")):
                    dest = toolkit / "licenses" / record["component"] / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, dest.open("wb") as output:
                    shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    print("CUDA root:", toolkit.absolute())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build-release-deps"))
    prepare(parser.parse_args().out)
