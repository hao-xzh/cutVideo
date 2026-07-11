from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path


class ResourceError(RuntimeError):
    """Raised when an offline runtime resource is missing or unsupported."""


@dataclass(frozen=True, slots=True)
class RuntimeResources:
    root: Path
    ffmpeg: Path | None
    ffprobe: Path | None
    fa_model: Path | None
    asr_model: Path | None
    vad_model: Path | None

    @property
    def has_ffmpeg(self) -> bool:
        return bool(self.ffmpeg and self.ffmpeg.is_file() and self.ffprobe and self.ffprobe.is_file())

    @property
    def has_models(self) -> bool:
        return all(
            path is not None and path.is_dir()
            for path in (self.fa_model, self.asr_model, self.vad_model)
        )

    def missing(self) -> list[str]:
        missing: list[str] = []
        if not self.ffmpeg or not self.ffmpeg.is_file():
            missing.append("ffmpeg")
        if not self.ffprobe or not self.ffprobe.is_file():
            missing.append("ffprobe")
        for label, path in (
            ("fa-zh", self.fa_model),
            ("paraformer-zh", self.asr_model),
            ("fsmn-vad", self.vad_model),
        ):
            if not path or not path.is_dir():
                missing.append(label)
        return missing


def platform_key() -> str:
    machine = platform.machine().lower()
    if sys.platform == "win32" and machine in {"amd64", "x86_64"}:
        return "windows-x86_64"
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return "macos-arm64"
    return f"{sys.platform}-{machine}"


def _resource_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    if value := os.environ.get("CUTVIDEO_RESOURCE_ROOT"):
        candidates.append(Path(value).expanduser())

    if frozen_root := getattr(sys, "_MEIPASS", None):
        candidates.append(Path(frozen_root) / "resources")

    executable_dir = Path(sys.executable).resolve().parent
    candidates.extend((executable_dir / "resources", executable_dir.parent / "Resources" / "resources"))

    source_root = Path(__file__).resolve().parents[2]
    candidates.append(source_root / "resources")
    return candidates


def find_resource_root() -> Path:
    for candidate in _resource_root_candidates():
        if (candidate / "manifest.json").is_file():
            return candidate.resolve()
    return _resource_root_candidates()[0].resolve()


def _existing_file(value: str | None, fallback: Path) -> Path | None:
    path = Path(value).expanduser() if value else fallback
    return path.resolve() if path.is_file() else None


def _existing_dir(value: str | None, fallback: Path) -> Path | None:
    path = Path(value).expanduser() if value else fallback
    return path.resolve() if path.is_dir() else None


def discover_resources() -> RuntimeResources:
    root = find_resource_root()
    key = platform_key()
    suffix = ".exe" if sys.platform == "win32" else ""
    binary_root = root / "bin" / key
    model_root = Path(os.environ.get("CUTVIDEO_MODEL_ROOT", root / "models")).expanduser()
    return RuntimeResources(
        root=root,
        ffmpeg=_existing_file(os.environ.get("CUTVIDEO_FFMPEG"), binary_root / f"ffmpeg{suffix}"),
        ffprobe=_existing_file(
            os.environ.get("CUTVIDEO_FFPROBE"), binary_root / f"ffprobe{suffix}"
        ),
        fa_model=_existing_dir(None, model_root / "fa-zh"),
        asr_model=_existing_dir(None, model_root / "paraformer-zh"),
        vad_model=_existing_dir(None, model_root / "fsmn-vad"),
    )


def load_manifest(root: Path | None = None) -> dict[str, object]:
    manifest = (root or find_resource_root()) / "manifest.json"
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceError(f"无法读取离线资源清单：{manifest}") from exc
