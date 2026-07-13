"""FFmpeg discovery and safe, sample-exact export commands."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .subprocess_options import hidden_subprocess_kwargs


class FFmpegError(RuntimeError):
    pass


class FFmpegNotFoundError(FFmpegError):
    pass


class FFmpegCancelledError(FFmpegError):
    pass


class FFmpegProcessError(FFmpegError):
    def __init__(self, argv: Iterable[str], returncode: int, stderr: str):
        self.argv = tuple(str(value) for value in argv)
        self.returncode = int(returncode)
        self.stderr = stderr.strip()
        detail = self.stderr or "no diagnostic output"
        super().__init__(f"FFmpeg command failed ({self.returncode}): {detail}")


@dataclass(frozen=True, slots=True)
class FFmpegTools:
    ffmpeg: Path
    ffprobe: Path


@dataclass(frozen=True, slots=True)
class ExportResult:
    wav_path: Path
    mp3_path: Path
    command: tuple[str, ...]
    kept_samples: int
    removed_samples: int


@dataclass(frozen=True, slots=True)
class PreviewResult:
    original_wav_path: Path
    edited_wav_path: Path
    command: tuple[str, ...]


def _executable_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _platform_tags() -> tuple[str, ...]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows":
        os_tags = ("windows", "win")
    elif system == "darwin":
        os_tags = ("macos", "darwin", "mac")
    else:
        os_tags = (system, "linux") if system != "linux" else ("linux",)
    if machine in {"amd64", "x86_64", "x64"}:
        arch_tags = ("x64", "x86_64", "amd64")
    elif machine in {"arm64", "aarch64"}:
        arch_tags = ("arm64", "aarch64")
    else:
        arch_tags = (machine,)
    return tuple(f"{os_tag}-{arch}" for os_tag in os_tags for arch in arch_tags)


def _resource_roots(explicit: str | Path | None) -> list[Path]:
    roots: list[Path] = []
    if explicit is not None:
        roots.append(Path(explicit).expanduser())
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        roots.append(Path(frozen_root))
    package_file = Path(__file__).resolve()
    roots.extend((package_file.parents[2], package_file.parent, Path.cwd()))
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _candidate_binaries(root: Path, program: str) -> list[Path]:
    executable = _executable_name(program)
    candidates: list[Path] = []
    bases = (root, root / "resources", root / "Resources")
    for base in bases:
        candidates.append(base / executable)
        candidates.append(base / "bin" / executable)
        for tag in _platform_tags():
            candidates.append(base / "bin" / tag / executable)
    return candidates


def _resolve_program(
    program: str,
    explicit: str | Path | None,
    roots: list[Path],
) -> Path | None:
    if explicit:
        value = Path(explicit).expanduser()
        if value.is_dir():
            value = value / _executable_name(program)
        if value.is_file():
            return value.resolve()
        # A plain executable name supplied by an environment variable may be
        # resolvable through PATH.
        found = shutil.which(str(explicit))
        if found:
            return Path(found).resolve()
        return None
    for root in roots:
        for candidate in _candidate_binaries(root, program):
            if candidate.is_file():
                return candidate.resolve()
    found = shutil.which(program)
    return Path(found).resolve() if found else None


def discover_ffmpeg(
    *,
    resource_root: str | Path | None = None,
    ffmpeg_path: str | Path | None = None,
    ffprobe_path: str | Path | None = None,
) -> FFmpegTools:
    """Locate bundled tools first, then fall back to PATH.

    ``CUTVIDEO_FFMPEG`` and ``CUTVIDEO_FFPROBE`` override bundled discovery.
    If only ffmpeg is explicitly configured, a sibling ffprobe is preferred.
    """

    roots = _resource_roots(resource_root)
    ffmpeg_explicit = ffmpeg_path or os.environ.get("CUTVIDEO_FFMPEG")
    ffprobe_explicit = ffprobe_path or os.environ.get("CUTVIDEO_FFPROBE")
    ffmpeg = _resolve_program("ffmpeg", ffmpeg_explicit, roots)
    if ffprobe_explicit is None and ffmpeg is not None:
        sibling = ffmpeg.with_name(_executable_name("ffprobe"))
        if sibling.is_file():
            ffprobe_explicit = sibling
    ffprobe = _resolve_program("ffprobe", ffprobe_explicit, roots)
    if ffmpeg is None or ffprobe is None:
        missing = ", ".join(
            name for name, value in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)) if value is None
        )
        raise FFmpegNotFoundError(
            f"找不到 {missing}。请安装应用自带资源，或设置 CUTVIDEO_FFMPEG/CUTVIDEO_FFPROBE。"
        )
    return FFmpegTools(ffmpeg=ffmpeg, ffprobe=ffprobe)


def _coerce_interval(interval: object) -> tuple[int, int]:
    if isinstance(interval, dict):
        return int(interval["start_sample"]), int(interval["end_sample"])
    if hasattr(interval, "start_sample") and hasattr(interval, "end_sample"):
        return int(interval.start_sample), int(interval.end_sample)  # type: ignore[attr-defined]
    try:
        start, end = interval  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise TypeError("interval must have start_sample/end_sample or be a pair") from exc
    return int(start), int(end)


def _merge(intervals: Iterable[object], total_samples: int) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    for interval in intervals:
        start, end = _coerce_interval(interval)
        start = max(0, start)
        end = min(total_samples, end)
        if end > start:
            normalized.append((start, end))
    normalized.sort()
    merged: list[tuple[int, int]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _kept_ranges(
    intervals: Iterable[object],
    total_samples: int,
    *,
    window_start: int = 0,
    window_end: int | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    start_limit = max(0, int(window_start))
    end_limit = total_samples if window_end is None else min(total_samples, int(window_end))
    if end_limit <= start_limit:
        raise ValueError("preview/export window is empty")
    cuts = _merge(intervals, total_samples)
    kept: list[tuple[int, int]] = []
    cursor = start_limit
    for start, end in cuts:
        if end <= start_limit or start >= end_limit:
            continue
        clipped_start = max(start_limit, start)
        clipped_end = min(end_limit, end)
        if clipped_start > cursor:
            kept.append((cursor, clipped_start))
        cursor = max(cursor, clipped_end)
    if cursor < end_limit:
        kept.append((cursor, end_limit))
    if not kept:
        raise ValueError("cut intervals remove the complete audio window")
    return cuts, kept


def _edit_filter_parts(
    kept: list[tuple[int, int]],
    sample_rate: int,
    *,
    source_label: str = "0:a",
    output_label: str = "edited",
    fade_ms: float = 8.0,
    prefix: str = "k",
) -> tuple[list[str], int]:
    if sample_rate <= 0 or fade_ms < 0:
        raise ValueError("invalid sample rate or fade duration")
    parts: list[str] = []
    source_labels: list[str]
    if len(kept) == 1:
        source_labels = [source_label]
    else:
        source_labels = [f"{prefix}src{i}" for i in range(len(kept))]
        parts.append(
            f"[{source_label}]asplit={len(kept)}"
            + "".join(f"[{label}]" for label in source_labels)
        )
    segment_labels: list[str] = []
    lengths: list[int] = []
    for index, ((start, end), source) in enumerate(zip(kept, source_labels, strict=True)):
        label = f"{prefix}{index}"
        parts.append(
            f"[{source}]atrim=start_sample={start}:end_sample={end},"
            f"asetpts=PTS-STARTPTS[{label}]"
        )
        segment_labels.append(label)
        lengths.append(end - start)

    if len(segment_labels) == 1:
        parts.append(f"[{segment_labels[0]}]anull[{output_label}]")
        return parts, lengths[0]

    requested_fade = round(sample_rate * fade_ms / 1000.0)
    current_label = segment_labels[0]
    current_length = lengths[0]
    for index in range(1, len(segment_labels)):
        next_length = lengths[index]
        fade_samples = min(requested_fade, current_length, next_length)
        join_label = output_label if index == len(segment_labels) - 1 else f"{prefix}join{index}"
        if fade_samples > 0:
            parts.append(
                f"[{current_label}][{segment_labels[index]}]"
                f"acrossfade=ns={fade_samples}:c1=qsin:c2=qsin[{join_label}]"
            )
            current_length += next_length - fade_samples
        else:
            parts.append(
                f"[{current_label}][{segment_labels[index]}]concat=n=2:v=0:a=1[{join_label}]"
            )
            current_length += next_length
        current_label = join_label
    return parts, current_length


def build_export_filter(
    intervals: Iterable[object],
    total_samples: int,
    sample_rate: int,
    *,
    fade_ms: float = 8.0,
) -> tuple[str, int, int]:
    """Return ``filter_complex``, output frames, and removed input frames."""

    cuts, kept = _kept_ranges(intervals, total_samples)
    parts, kept_after_fades = _edit_filter_parts(kept, sample_rate, fade_ms=fade_ms)
    parts.append("[edited]asplit=2[wavout][mp3out]")
    removed = sum(end - start for start, end in cuts)
    return ";".join(parts), kept_after_fades, removed


def build_export_command(
    audio_path: str | Path,
    intervals: Iterable[object],
    wav_path: str | Path,
    mp3_path: str | Path,
    *,
    total_samples: int,
    sample_rate: int,
    channels: int,
    tools: FFmpegTools,
    fade_ms: float = 8.0,
    mp3_bitrate: str = "192k",
    overwrite: bool = True,
    include_progress: bool = False,
) -> tuple[list[str], int, int]:
    if channels <= 0:
        raise ValueError("channels must be positive")
    filter_text, kept_samples, removed_samples = build_export_filter(
        intervals,
        total_samples,
        sample_rate,
        fade_ms=fade_ms,
    )
    mp3_sample_rate = _compatible_mp3_sample_rate(sample_rate)
    mp3_channels = min(channels, 2)
    argv = [
        str(tools.ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-i",
        str(Path(audio_path)),
        "-filter_complex",
        filter_text,
    ]
    if include_progress:
        argv.extend(["-progress", "pipe:1", "-nostats"])
    argv.extend(
        [
            "-map",
            "[wavout]",
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            str(Path(wav_path)),
            "-map",
            "[mp3out]",
            "-c:a",
            "libmp3lame",
            "-b:a",
            mp3_bitrate,
            "-ar",
            str(mp3_sample_rate),
            "-ac",
            str(mp3_channels),
            str(Path(mp3_path)),
        ]
    )
    return argv, kept_samples, removed_samples


def _compatible_mp3_sample_rate(sample_rate: int) -> int:
    supported = (8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000)
    if sample_rate in supported:
        return sample_rate
    if sample_rate > 48_000:
        if sample_rate % 44_100 == 0:
            return 44_100
        return 48_000
    return min(supported, key=lambda value: (abs(value - sample_rate), -value))


def _validate_audio_output(path: Path, tools: FFmpegTools) -> None:
    if not path.is_file() or path.stat().st_size <= 44:
        raise FFmpegError(f"FFmpeg output is missing or empty: {path}")
    argv = [
        str(tools.ffprobe),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        shell=False,
        **hidden_subprocess_kwargs(),
    )
    if completed.returncode or b"audio" not in completed.stdout:
        raise FFmpegProcessError(
            argv,
            completed.returncode,
            completed.stderr.decode("utf-8", "replace") or "output has no audio stream",
        )


def _commit_output_pair(
    temporary: tuple[Path, Path],
    final: tuple[Path, Path],
    *,
    overwrite: bool,
) -> None:
    token = uuid.uuid4().hex
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for target in final:
            if target.exists():
                if not overwrite:
                    raise FileExistsError(target)
                backup = target.with_name(f".{target.name}.{token}.bak")
                os.replace(target, backup)
                backups[target] = backup
        try:
            for source, target in zip(temporary, final, strict=True):
                os.replace(source, target)
                committed.append(target)
        except Exception:
            for target in committed:
                with suppress(OSError):
                    target.unlink()
            for target, backup in backups.items():
                if backup.exists():
                    os.replace(backup, target)
            raise
        for backup in backups.values():
            with suppress(OSError):
                backup.unlink()
    except Exception:
        for target, backup in backups.items():
            if backup.exists() and not target.exists():
                with suppress(OSError):
                    os.replace(backup, target)
        raise


def _cancelled(cancel: Callable[[], bool] | object | None) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(cancel)


def _run_with_progress(
    argv: list[str],
    *,
    expected_duration_seconds: float | None = None,
    progress_cb: Callable[[float], None] | None = None,
    cancel: Callable[[], bool] | object | None = None,
) -> None:
    # Progress key/value lines are read from stdout; stderr is file-backed to
    # prevent pipe deadlocks.  shell=False and an argv list are mandatory.
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            **hidden_subprocess_kwargs(),
        )
        assert process.stdout is not None
        if progress_cb:
            progress_cb(0.0)
        try:
            while True:
                if _cancelled(cancel):
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise FFmpegCancelledError("FFmpeg export was cancelled")
                line = process.stdout.readline()
                if line:
                    key, separator, value = line.strip().partition("=")
                    if (
                        separator
                        and progress_cb
                        and expected_duration_seconds
                        and key in {"out_time_us", "out_time_ms"}
                    ):
                        # Modern ffmpeg documents out_time_us.  Some older
                        # builds misleadingly use out_time_ms for the same
                        # microsecond value.
                        with suppress(ValueError):
                            progress_cb(
                                max(
                                    0.0,
                                    min(0.99, float(value) / 1_000_000 / expected_duration_seconds),
                                )
                            )
                elif process.poll() is not None:
                    break
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        if returncode:
            stderr_file.seek(0)
            error = stderr_file.read().decode("utf-8", "replace")
            raise FFmpegProcessError(argv, returncode, error)
        if progress_cb:
            progress_cb(1.0)


def export_audio(
    audio_path: str | Path,
    intervals: Iterable[object],
    wav_path: str | Path,
    mp3_path: str | Path,
    *,
    info: object | None = None,
    total_samples: int | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    tools: FFmpegTools | None = None,
    resource_root: str | Path | None = None,
    fade_ms: float = 8.0,
    mp3_bitrate: str = "192k",
    overwrite: bool = True,
    progress_cb: Callable[[float], None] | None = None,
    cancel: Callable[[], bool] | object | None = None,
) -> ExportResult:
    """Create PCM16 WAV and 192 kbps MP3 in one FFmpeg process."""

    if info is None and (total_samples is None or sample_rate is None or channels is None):
        from .audio import probe_audio

        info = probe_audio(audio_path, tools=tools, resource_root=resource_root, cancel=cancel)
    if info is not None:
        total_samples = int(info.total_samples)  # type: ignore[attr-defined]
        sample_rate = int(info.sample_rate)  # type: ignore[attr-defined]
        channels = int(info.channels)  # type: ignore[attr-defined]
    assert total_samples is not None and sample_rate is not None and channels is not None
    resolved_tools = tools or discover_ffmpeg(resource_root=resource_root)
    wav_output = Path(wav_path).expanduser().resolve()
    mp3_output = Path(mp3_path).expanduser().resolve()
    input_path = Path(audio_path).expanduser().resolve()
    if wav_output == mp3_output:
        raise ValueError("WAV and MP3 output paths must be different")
    if input_path in {wav_output, mp3_output}:
        raise ValueError("an output path must not overwrite the input audio")
    if not overwrite and (wav_output.exists() or mp3_output.exists()):
        raise FileExistsError("an output file already exists")
    wav_output.parent.mkdir(parents=True, exist_ok=True)
    mp3_output.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary_wav = wav_output.with_name(f".{wav_output.stem}.{token}.tmp.wav")
    temporary_mp3 = mp3_output.with_name(f".{mp3_output.stem}.{token}.tmp.mp3")
    argv, kept_samples, removed_samples = build_export_command(
        audio_path,
        intervals,
        temporary_wav,
        temporary_mp3,
        total_samples=total_samples,
        sample_rate=sample_rate,
        channels=channels,
        tools=resolved_tools,
        fade_ms=fade_ms,
        mp3_bitrate=mp3_bitrate,
        overwrite=overwrite,
        include_progress=True,
    )
    try:
        _run_with_progress(
            argv,
            expected_duration_seconds=kept_samples / sample_rate,
            progress_cb=progress_cb,
            cancel=cancel,
        )
        _validate_audio_output(temporary_wav, resolved_tools)
        _validate_audio_output(temporary_mp3, resolved_tools)
        _commit_output_pair(
            (temporary_wav, temporary_mp3),
            (wav_output, mp3_output),
            overwrite=overwrite,
        )
    finally:
        for temporary in (temporary_wav, temporary_mp3):
            with suppress(OSError):
                temporary.unlink()
    return ExportResult(wav_output, mp3_output, tuple(argv), kept_samples, removed_samples)


def build_preview_command(
    audio_path: str | Path,
    intervals: Iterable[object],
    original_wav_path: str | Path,
    edited_wav_path: str | Path,
    *,
    start_sample: int,
    end_sample: int,
    total_samples: int,
    sample_rate: int,
    channels: int,
    tools: FFmpegTools,
    fade_ms: float = 8.0,
    output_sample_rate: int = 48_000,
    output_channels: int = 2,
    overwrite: bool = True,
) -> list[str]:
    _cuts, kept = _kept_ranges(
        intervals,
        total_samples,
        window_start=start_sample,
        window_end=end_sample,
    )
    # Split once for original vs edited; the edit helper performs a further
    # split when multiple kept pieces are present.
    parts = ["[0:a]asplit=2[originalsrc][editsrc]"]
    parts.append(
        f"[originalsrc]atrim=start_sample={int(start_sample)}:end_sample={int(end_sample)},"
        "asetpts=PTS-STARTPTS[original]"
    )
    edit_parts, _length = _edit_filter_parts(
        kept,
        sample_rate,
        source_label="editsrc",
        output_label="edited",
        fade_ms=fade_ms,
        prefix="p",
    )
    parts.extend(edit_parts)
    return [
        str(tools.ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-i",
        str(Path(audio_path)),
        "-filter_complex",
        ";".join(parts),
        "-map",
        "[original]",
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(output_sample_rate),
        "-ac",
        str(output_channels),
        str(Path(original_wav_path)),
        "-map",
        "[edited]",
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(output_sample_rate),
        "-ac",
        str(output_channels),
        str(Path(edited_wav_path)),
    ]


def generate_preview(
    audio_path: str | Path,
    intervals: Iterable[object],
    original_wav_path: str | Path,
    edited_wav_path: str | Path,
    *,
    start_sample: int,
    end_sample: int,
    info: object,
    tools: FFmpegTools | None = None,
    resource_root: str | Path | None = None,
    fade_ms: float = 8.0,
    overwrite: bool = True,
    cancel: Callable[[], bool] | object | None = None,
) -> PreviewResult:
    """Render original and edited local PCM16 WAV previews together."""

    resolved_tools = tools or discover_ffmpeg(resource_root=resource_root)
    original_output = Path(original_wav_path).expanduser().resolve()
    edited_output = Path(edited_wav_path).expanduser().resolve()
    original_output.parent.mkdir(parents=True, exist_ok=True)
    edited_output.parent.mkdir(parents=True, exist_ok=True)
    argv = build_preview_command(
        audio_path,
        intervals,
        original_output,
        edited_output,
        start_sample=start_sample,
        end_sample=end_sample,
        total_samples=int(info.total_samples),  # type: ignore[attr-defined]
        sample_rate=int(info.sample_rate),  # type: ignore[attr-defined]
        channels=int(info.channels),  # type: ignore[attr-defined]
        tools=resolved_tools,
        fade_ms=fade_ms,
        overwrite=overwrite,
    )
    _run_with_progress(argv, cancel=cancel)
    _validate_audio_output(original_output, resolved_tools)
    _validate_audio_output(edited_output, resolved_tools)
    return PreviewResult(original_output, edited_output, tuple(argv))


# Alternate name used in a few UI sketches.
create_preview = generate_preview
