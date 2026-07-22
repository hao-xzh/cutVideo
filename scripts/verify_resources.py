from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument(
        "--resource-root",
        type=Path,
        default=ROOT / "resources",
        help="resource directory to verify (defaults to the source tree)",
    )
    args = parser.parse_args()
    resource_root = args.resource_root.expanduser().resolve()
    manifest_path = resource_root / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = platform_key()
    errors: list[str] = []

    required_assets = [
        ROOT / "THIRD_PARTY_NOTICES.md",
        resource_root / "icons" / "app-icon.png",
        resource_root / "icons" / "app-icon.ico",
        resource_root / "icons" / "app-icon.icns",
        resource_root / "backgrounds" / "workspace-dark.png",
        resource_root / "backgrounds" / "workspace-light.png",
        resource_root / "licenses" / "LGPL-3.0.txt",
        resource_root
        / "licenses"
        / "python-packages"
        / "pypinyin-0.55.0.dist-info"
        / "LICENSE.txt",
    ]
    if key == "macos-arm64":
        required_assets.extend(
            resource_root / "licenses" / "python-packages" / package / license_path
            for package, license_path in (
                ("mlx_qwen3_asr-0.3.5.dist-info", "licenses/LICENSE"),
                ("mlx-0.29.4.dist-info", "licenses/LICENSE"),
                ("mlx_metal-0.29.4.dist-info", "licenses/LICENSE"),
                ("regex-2026.7.19.dist-info", "licenses/LICENSE.txt"),
            )
        )
    for required in required_assets:
        if not required.is_file():
            errors.append(f"missing release asset: {required}")

    selftest = data.get("selftest", {})
    if not isinstance(selftest, dict):
        errors.append("invalid selftest manifest entry")
    else:
        selftest_audio = selftest.get("audio")
        selftest_sha = selftest.get("sha256")
        selftest_transcript = selftest.get("transcript")
        selftest_source = selftest.get("source")
        selftest_license = selftest.get("license")
        if not isinstance(selftest_audio, str) or not selftest_audio:
            errors.append("missing selftest audio path")
        else:
            selftest_path = resource_root / selftest_audio
            if not selftest_path.is_file():
                errors.append(f"missing selftest audio: {selftest_path}")
            elif not isinstance(selftest_sha, str) or not selftest_sha:
                errors.append("missing selftest audio SHA-256")
            elif sha256_file(selftest_path).lower() != selftest_sha.lower():
                errors.append(f"SHA-256 mismatch: {selftest_path}")
        if not isinstance(selftest_transcript, str) or not selftest_transcript.strip():
            errors.append("missing selftest transcript")
        if not isinstance(selftest_source, str) or not selftest_source.strip():
            errors.append("missing selftest source")
        if not isinstance(selftest_license, str) or not selftest_license.strip():
            errors.append("missing selftest license")
        elif not (resource_root / selftest_license).is_file():
            errors.append(f"missing selftest license file: {selftest_license}")

    build_info = resource_root / (
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
    elif license_value and not (resource_root / license_value).is_file():
        errors.append(f"missing binary license file: {license_value}")
    if not binary_entry.get("source_code") and not args.allow_unpinned:
        errors.append(f"missing corresponding source URL: {key}")
    for name in ("ffmpeg", "ffprobe"):
        path = resource_root / "bin" / key / f"{name}{suffix}"
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
            lame_path = resource_root / "bin" / key / lame["file"]
            if not lame_path.is_file():
                errors.append(f"missing libmp3lame dylib: {lame_path}")
            elif lame.get("sha256") and sha256_file(lame_path).lower() != lame["sha256"].lower():
                errors.append(f"SHA-256 mismatch: {lame_path}")
        if lame.get("license") and not (resource_root / lame["license"]).is_file():
            errors.append(f"missing libmp3lame license file: {lame['license']}")

    required_models = (
        {"qwen3-asr-0.6b-4bit", "qwen3-forced-aligner-0.6b-4bit"}
        if key == "macos-arm64"
        else {"fa-zh", "paraformer-zh", "fsmn-vad"}
    )
    models = data.get("models", {})
    if not isinstance(models, dict):
        errors.append("invalid models manifest entry")
        models = {}
    for name in sorted(required_models):
        entry = models.get(name)
        if not isinstance(entry, dict):
            errors.append(f"missing model manifest entry: {name}")
            continue
        model_dir = resource_root / entry["path"]
        if not model_dir.is_dir() or not any(model_dir.iterdir()):
            errors.append(f"missing model: {model_dir}")
        if not entry.get("license") and not args.allow_unpinned:
            errors.append(f"missing model license entry: {name}")
        elif entry.get("license") and not (resource_root / entry["license"]).is_file():
            errors.append(f"missing model license file: {entry['license']}")
        expected = entry.get("sha256")
        if not expected and not args.allow_unpinned:
            errors.append(f"missing model tree SHA-256: {name}")
        elif expected and model_dir.is_dir() and sha256_tree(model_dir).lower() != expected.lower():
            errors.append(f"model tree SHA-256 mismatch: {name}")
        if name.startswith("qwen3-") and model_dir.is_dir():
            for filename in (
                "config.json",
                "merges.txt",
                "quantization_config.json",
                "tokenizer_config.json",
                "vocab.json",
                "weights.safetensors",
            ):
                if not (model_dir / filename).is_file():
                    errors.append(f"missing local-only model file: {model_dir / filename}")

    if errors:
        print("Resource verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Resources verified for {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
