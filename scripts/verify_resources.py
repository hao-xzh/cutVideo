from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "resources" / "manifest.json"


def platform_key() -> str:
    machine = platform.machine().lower()
    if sys.platform == "win32" and machine in {"amd64", "x86_64"}:
        return "windows-x86_64"
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return "macos-arm64"
    raise SystemExit(f"Unsupported build platform: {sys.platform}/{machine}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-unpinned", action="store_true")
    args = parser.parse_args()
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    key = platform_key()
    errors: list[str] = []

    for required in (
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "resources" / "icons" / "app-icon.png",
        ROOT / "resources" / "icons" / "app-icon.ico",
        ROOT / "resources" / "icons" / "app-icon.icns",
        ROOT / "resources" / "licenses" / "LGPL-3.0.txt",
        ROOT
        / "resources"
        / "licenses"
        / "python-packages"
        / "pypinyin-0.55.0.dist-info"
        / "LICENSE.txt",
    ):
        if not required.is_file():
            errors.append(f"missing release asset: {required}")

    build_info = ROOT / "resources" / (
        "FFMPEG_BUILDINFO_WINDOWS.txt"
        if key == "windows-x86_64"
        else "FFMPEG_BUILDINFO_MACOS.txt"
    )
    if not build_info.is_file():
        errors.append(f"missing FFmpeg build configuration: {build_info}")

    suffix = ".exe" if key.startswith("windows") else ""
    binary_entry = data["binaries"][key]
    license_value = binary_entry.get("license")
    if not license_value and not args.allow_unpinned:
        errors.append(f"missing binary license entry: {key}")
    elif license_value and not (ROOT / "resources" / license_value).is_file():
        errors.append(f"missing binary license file: {license_value}")
    if not binary_entry.get("source_code") and not args.allow_unpinned:
        errors.append(f"missing corresponding source URL: {key}")
    for name in ("ffmpeg", "ffprobe"):
        path = ROOT / "resources" / "bin" / key / f"{name}{suffix}"
        if not path.is_file():
            errors.append(f"missing binary: {path}")
            continue
        if key == "macos-arm64" and not path.stat().st_mode & 0o111:
            errors.append(f"binary is not executable: {path}")
        expected = binary_entry.get(name)
        if not expected and not args.allow_unpinned:
            errors.append(f"missing SHA-256 in manifest for {key}/{name}")
        elif expected and sha256_file(path).lower() != expected.lower():
            errors.append(f"SHA-256 mismatch: {path}")

    if key == "macos-arm64":
        lame = binary_entry.get("lgpl_dependencies", {}).get("libmp3lame", {})
        for field in ("file", "sha256", "version", "source_code", "license"):
            if not lame.get(field):
                errors.append(f"missing libmp3lame manifest field: {field}")
        if lame.get("file"):
            lame_path = ROOT / "resources" / "bin" / key / lame["file"]
            if not lame_path.is_file():
                errors.append(f"missing libmp3lame dylib: {lame_path}")
            elif lame.get("sha256") and sha256_file(lame_path).lower() != lame["sha256"].lower():
                errors.append(f"SHA-256 mismatch: {lame_path}")
        if lame.get("license") and not (ROOT / "resources" / lame["license"]).is_file():
            errors.append(f"missing libmp3lame license file: {lame['license']}")

    for name, entry in data["models"].items():
        model_dir = ROOT / "resources" / entry["path"]
        if not model_dir.is_dir() or not any(model_dir.iterdir()):
            errors.append(f"missing model: {model_dir}")
        if not entry.get("license") and not args.allow_unpinned:
            errors.append(f"missing model license entry: {name}")
        elif entry.get("license") and not (ROOT / "resources" / entry["license"]).is_file():
            errors.append(f"missing model license file: {entry['license']}")
        expected = entry.get("sha256")
        if not expected and not args.allow_unpinned:
            errors.append(f"missing model tree SHA-256: {name}")
        elif expected and model_dir.is_dir() and sha256_tree(model_dir).lower() != expected.lower():
            errors.append(f"model tree SHA-256 mismatch: {name}")

    if errors:
        print("Resource verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Resources verified for {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
