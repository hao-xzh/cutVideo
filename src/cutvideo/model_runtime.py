"""Offline-only helpers for loading and invoking the bundled speech models.

The application deliberately keeps FunASR as an optional, lazily imported
dependency.  This module is the single boundary at which a model is loaded:
only existing local directories are accepted and all supported model hubs are
put into offline mode before importing FunASR.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import threading
import time
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
_MODEL_EXECUTION_LOCK = threading.RLock()
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[object, ...], Any] = {}


def _cancelled(cancel: object | None) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(cancel)


@contextmanager
def model_execution_guard(cancel: object | None = None) -> Iterator[None]:
    """Serialize heavyweight model sessions across both editor workspaces."""

    while not _MODEL_EXECUTION_LOCK.acquire(timeout=0.1):
        if _cancelled(cancel):
            raise ModelUnavailableError("模型任务已取消")
    try:
        yield
    finally:
        _MODEL_EXECUTION_LOCK.release()


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


def _freeze_model_option(value: object) -> object:
    """Turn nested model options into a stable, hashable cache-key value."""

    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (str(key), _freeze_model_option(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_model_option(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_model_option(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def load_funasr_model(
    *,
    model_path: str | Path,
    label: str,
    vad_model_path: str | Path | None = None,
    device: str = "cpu",
    model_factory: Callable[..., Any] | None = None,
    extra_options: Mapping[str, Any] | None = None,
) -> Any:
    """Load or reuse a FunASR ``AutoModel`` from local paths only.

    ``model_factory`` is an intentional injection point for tests and for a
    frozen application that exposes AutoModel through a small compatibility
    shim. Injected factories deliberately bypass the process cache. No model
    name accepted by a remote registry is ever passed here.
    """

    local_model = require_local_model(model_path, label)
    local_vad = (
        require_local_model(vad_model_path, "fsmn-vad")
        if vad_model_path is not None
        else None
    )
    configure_offline_environment()

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

    if model_factory is None:
        cache_key = (
            str(local_model),
            str(local_vad) if local_vad is not None else "",
            label,
            device,
            _freeze_model_option(extra_options or {}),
        )
        # Model deserialization is both slow and memory-heavy. Holding this
        # lock through construction prevents two workspaces from loading the
        # same ~GB-scale weights at once. Inference itself is serialized by
        # ``model_execution_guard`` because FunASR mutates runtime kwargs.
        with _MODEL_CACHE_LOCK:
            if cache_key in _MODEL_CACHE:
                return _MODEL_CACHE[cache_key]
            try:
                with _normalized_windows_dll_directories():
                    from funasr import AutoModel  # type: ignore[import-not-found]
            except (ImportError, OSError) as exc:
                raise ModelUnavailableError("FunASR 本地运行库不可用") from exc
            try:
                model = AutoModel(**options)
            except Exception as exc:  # FunASR exposes several exception types.
                raise ModelUnavailableError(f"无法加载本地模型 ({label}): {exc}") from exc
            _MODEL_CACHE[cache_key] = model
            return model

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
    cancel: object | None = None,
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
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
            shell=False,
            **hidden_subprocess_kwargs(),
        )
        while process.poll() is None:
            if _cancelled(cancel):
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise ModelUnavailableError("模型音频准备已取消")
            time.sleep(0.05)
        if process.returncode:
            stderr_file.seek(0)
            error = stderr_file.read().decode("utf-8", "replace").strip()
            raise ModelUnavailableError(f"FFmpeg 无法准备模型音频窗口: {error}")


@contextmanager
def local_audio_window(
    audio_path: str | Path,
    *,
    start_ms: int,
    end_ms: int,
    ffmpeg_path: str | Path | None = None,
    cancel: object | None = None,
) -> Iterator[Path]:
    """Yield a temporary, local WAV containing exactly the requested window."""

    source = Path(audio_path).expanduser()
    if not source.is_file():
        raise ModelUnavailableError(f"音频文件不可用: {source}")
    if (
        isinstance(start_ms, bool)
        or isinstance(end_ms, bool)
        or not isinstance(start_ms, (int, float))
        or not isinstance(end_ms, (int, float))
        or not math.isfinite(float(start_ms))
        or not math.isfinite(float(end_ms))
        or float(start_ms) != int(start_ms)
        or float(end_ms) != int(end_ms)
    ):
        raise ValueError("model audio window must use finite integer milliseconds")
    start_ms = int(start_ms)
    end_ms = int(end_ms)
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("invalid model audio window")

    if source.suffix.casefold() == ".wav" and start_ms == 0:
        try:
            with wave.open(str(source), "rb") as reader:
                duration_ms = reader.getnframes() * 1000 / reader.getframerate()
                model_ready = (
                    reader.getcomptype() == "NONE"
                    and reader.getnchannels() == 1
                    and reader.getsampwidth() == 2
                    and reader.getframerate() == 16_000
                )
            if end_ms + 1 >= duration_ms and (ffmpeg_path is None or model_ready):
                yield source.resolve()
                return
        except (EOFError, OSError, wave.Error, ZeroDivisionError):
            pass

    with tempfile.TemporaryDirectory(prefix="cutvideo-model-") as directory:
        output = Path(directory) / "window.wav"
        if ffmpeg_path is not None:
            ffmpeg = Path(ffmpeg_path).expanduser()
            if not ffmpeg.is_file():
                raise ModelUnavailableError(f"本地 FFmpeg 不可用: {ffmpeg}")
            _extract_ffmpeg_window(
                source,
                output,
                start_ms,
                end_ms,
                ffmpeg.resolve(),
                cancel,
            )
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
    "model_execution_guard",
    "require_local_model",
]
