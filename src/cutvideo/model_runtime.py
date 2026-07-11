"""Offline-only helpers for loading and invoking the bundled speech models.

The application deliberately keeps FunASR as an optional, lazily imported
dependency.  This module is the single boundary at which a model is loaded:
only existing local directories are accepted and all supported model hubs are
put into offline mode before importing FunASR.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .subprocess_options import hidden_subprocess_kwargs


class ModelUnavailableError(RuntimeError):
    """Raised when an offline model or its local audio runtime is unavailable."""


_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "MODELSCOPE_OFFLINE": "1",
    "FUNASR_DISABLE_UPDATE": "1",
}


def configure_offline_environment() -> None:
    """Permanently disable supported model-hub network access for this process.

    CutVideo is an offline application, so restoring these variables after a
    model call would be surprising and could allow a later lazy load to reach a
    model hub.  Explicit assignment also overrides a conflicting parent-shell
    value such as ``HF_HUB_OFFLINE=0``.
    """

    os.environ.update(_OFFLINE_ENVIRONMENT)


@contextmanager
def _normalized_windows_dll_directories() -> Iterator[None]:
    """Normalize PyInstaller's escaped package paths while importing Torch.

    PyTorch 2.13 enumerates its DLL directory during import. In a frozen
    Windows process its package ``__file__`` can contain repeated separators,
    which ``os.add_dll_directory`` rejects with WinError 206 even though the
    normalized path is short and valid.
    """

    add_directory = getattr(os, "add_dll_directory", None)
    if os.name != "nt" or add_directory is None:
        yield
        return

    def normalized(path: str | bytes | os.PathLike[str] | os.PathLike[bytes]):
        return add_directory(os.path.normpath(os.fspath(path)))

    os.add_dll_directory = normalized  # type: ignore[attr-defined,assignment]
    try:
        yield
    finally:
        os.add_dll_directory = add_directory  # type: ignore[attr-defined,assignment]


def require_local_model(path: str | Path, label: str) -> Path:
    """Resolve *path* and reject anything other than a readable local directory."""

    candidate = Path(path).expanduser()
    if not candidate.is_dir():
        raise ModelUnavailableError(f"本地模型不可用 ({label}): {candidate}")
    return candidate.resolve()


def load_funasr_model(
    *,
    model_path: str | Path,
    label: str,
    vad_model_path: str | Path | None = None,
    device: str = "cpu",
    model_factory: Callable[..., Any] | None = None,
    extra_options: Mapping[str, Any] | None = None,
) -> Any:
    """Create a FunASR ``AutoModel`` from local paths only.

    ``model_factory`` is an intentional injection point for tests and for a
    frozen application that exposes AutoModel through a small compatibility
    shim.  No model name accepted by a remote registry is ever passed here.
    """

    local_model = require_local_model(model_path, label)
    local_vad = (
        require_local_model(vad_model_path, "fsmn-vad")
        if vad_model_path is not None
        else None
    )
    configure_offline_environment()

    if model_factory is None:
        try:
            with _normalized_windows_dll_directories():
                from funasr import AutoModel  # type: ignore[import-not-found]
        except (ImportError, OSError) as exc:
            raise ModelUnavailableError("FunASR 本地运行库不可用") from exc
        model_factory = AutoModel

    options: dict[str, Any] = {
        "model": str(local_model),
        "device": device,
        "disable_update": True,
        "disable_pbar": True,
        "disable_log": True,
    }
    if local_vad is not None:
        options["vad_model"] = str(local_vad)
    if extra_options:
        options.update(extra_options)

    # Never let caller options replace a verified local path or enable update
    # checks.  These assignments are deliberately after ``extra_options``.
    options["model"] = str(local_model)
    options["disable_update"] = True
    options["disable_pbar"] = True
    options["disable_log"] = True
    if local_vad is not None:
        options["vad_model"] = str(local_vad)
    try:
        return model_factory(**options)
    except Exception as exc:  # FunASR backends expose several exception types.
        raise ModelUnavailableError(f"无法加载本地模型 ({label}): {exc}") from exc


def _extract_wave_window(
    source: Path,
    output: Path,
    start_ms: int,
    end_ms: int,
) -> None:
    """Dependency-free PCM WAV slicing used when FFmpeg is not configured."""

    try:
        with wave.open(str(source), "rb") as reader:
            sample_rate = reader.getframerate()
            start_frame = max(0, round(start_ms * sample_rate / 1000))
            end_frame = min(reader.getnframes(), round(end_ms * sample_rate / 1000))
            if end_frame <= start_frame:
                raise ModelUnavailableError("模型音频窗口为空")
            reader.setpos(start_frame)
            frames = reader.readframes(end_frame - start_frame)
            params = reader.getparams()
        with wave.open(str(output), "wb") as writer:
            writer.setnchannels(params.nchannels)
            writer.setsampwidth(params.sampwidth)
            writer.setframerate(params.framerate)
            writer.setcomptype(params.comptype, params.compname)
            writer.writeframes(frames)
    except (EOFError, OSError, wave.Error) as exc:
        raise ModelUnavailableError(
            "非 WAV 输入需要配置随应用捆绑的 FFmpeg，不能使用网络或外部解码服务"
        ) from exc


def _extract_ffmpeg_window(
    source: Path,
    output: Path,
    start_ms: int,
    end_ms: int,
    ffmpeg_path: Path,
) -> None:
    duration_ms = end_ms - start_ms
    command = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-ss",
        f"{start_ms / 1000:.6f}",
        "-t",
        f"{duration_ms / 1000:.6f}",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(output),
    ]
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        **hidden_subprocess_kwargs(),
    )
    if completed.returncode:
        error = completed.stderr.decode("utf-8", "replace").strip()
        raise ModelUnavailableError(f"FFmpeg 无法准备模型音频窗口: {error}")


@contextmanager
def local_audio_window(
    audio_path: str | Path,
    *,
    start_ms: int,
    end_ms: int,
    ffmpeg_path: str | Path | None = None,
) -> Iterator[Path]:
    """Yield a temporary, local WAV containing exactly the requested window."""

    source = Path(audio_path).expanduser()
    if not source.is_file():
        raise ModelUnavailableError(f"音频文件不可用: {source}")
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("invalid model audio window")

    with tempfile.TemporaryDirectory(prefix="cutvideo-model-") as directory:
        output = Path(directory) / "window.wav"
        if ffmpeg_path is not None:
            ffmpeg = Path(ffmpeg_path).expanduser()
            if not ffmpeg.is_file():
                raise ModelUnavailableError(f"本地 FFmpeg 不可用: {ffmpeg}")
            _extract_ffmpeg_window(source, output, start_ms, end_ms, ffmpeg.resolve())
        else:
            _extract_wave_window(source, output, start_ms, end_ms)
        if not output.is_file() or output.stat().st_size <= 44:
            raise ModelUnavailableError("模型音频窗口解码后为空")
        yield output


__all__ = [
    "ModelUnavailableError",
    "configure_offline_environment",
    "load_funasr_model",
    "local_audio_window",
    "require_local_model",
]
