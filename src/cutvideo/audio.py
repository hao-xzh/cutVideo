"""Audio inspection and sample-domain editing helpers.

All timestamps in CutVideo are derived from decoded PCM frames.  Container
metadata (and, in particular, the duration reported by Windows Explorer) is
never used as the authoritative clock.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ffmpeg import (
    FFmpegCancelledError,
    FFmpegProcessError,
    FFmpegTools,
    discover_ffmpeg,
)
from .subprocess_options import hidden_subprocess_kwargs

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool] | object


@dataclass(frozen=True, slots=True)
class AudioInfo:
    """Properties of the first audio stream.

    ``total_samples`` means PCM frames per channel, not the total number of
    interleaved floating-point values.
    """

    path: Path
    sample_rate: int
    channels: int
    total_samples: int
    duration_seconds: float
    codec: str | None = None

    @property
    def frames(self) -> int:
        return self.total_samples


@dataclass(frozen=True, slots=True)
class CutInterval:
    start_sample: int
    end_sample: int
    text: str = ""
    status: str = "approved"

    def __post_init__(self) -> None:
        if self.start_sample < 0:
            raise ValueError("start_sample must be non-negative")
        if self.end_sample <= self.start_sample:
            raise ValueError("end_sample must be greater than start_sample")

    @property
    def length(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True, slots=True)
class WaveformEnvelope:
    """A mono min/max/RMS envelope for one waveform zoom level."""

    sample_rate: int
    block_size: int
    minimum: np.ndarray
    maximum: np.ndarray
    rms: np.ndarray

    @property
    def points(self) -> int:
        return int(self.minimum.size)


def _cancelled(cancel: CancelCheck | None) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(cancel)


def _report(progress_cb: ProgressCallback | None, value: float) -> None:
    if progress_cb is not None:
        progress_cb(max(0.0, min(1.0, float(value))))


def _resolve_tools(
    tools: FFmpegTools | None,
    *,
    resource_root: str | Path | None = None,
    ffmpeg_path: str | Path | None = None,
    ffprobe_path: str | Path | None = None,
) -> FFmpegTools:
    return tools or discover_ffmpeg(
        resource_root=resource_root,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
    )


def _probe_stream(path: Path, tools: FFmpegTools) -> tuple[int, int, str | None, float | None]:
    argv = [
        str(tools.ffprobe),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,codec_name,duration:format=duration",
        "-of",
        "json",
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
    if completed.returncode:
        raise FFmpegProcessError(argv, completed.returncode, completed.stderr.decode("utf-8", "replace"))
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
        stream = payload["streams"][0]
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
    except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise FFmpegProcessError(argv, completed.returncode, "ffprobe did not return a usable audio stream") from exc
    if sample_rate <= 0 or channels <= 0:
        raise FFmpegProcessError(argv, completed.returncode, "invalid sample rate or channel count")
    duration_value = stream.get("duration") or payload.get("format", {}).get("duration")
    try:
        estimated_duration = float(duration_value) if duration_value is not None else None
    except (TypeError, ValueError):
        estimated_duration = None
    return sample_rate, channels, stream.get("codec_name"), estimated_duration


def _decoded_byte_chunks(
    path: Path,
    tools: FFmpegTools,
    *,
    audio_filter: str | None = None,
    cancel: CancelCheck | None = None,
    chunk_size: int = 1024 * 1024,
) -> Iterator[bytes]:
    argv = [
        str(tools.ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
    ]
    if audio_filter:
        argv.extend(["-af", audio_filter])
    argv.extend(["-c:a", "pcm_f32le", "-f", "f32le", "-"])

    # stderr is backed by a temporary file so a verbose decoder can never
    # deadlock while stdout is consumed incrementally.
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            shell=False,
            **hidden_subprocess_kwargs(),
        )
        assert process.stdout is not None
        try:
            while True:
                if _cancelled(cancel):
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise FFmpegCancelledError("audio decoding was cancelled")
                chunk = process.stdout.read(chunk_size)
                if not chunk:
                    break
                yield chunk
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        if returncode:
            stderr_file.seek(0)
            error = stderr_file.read().decode("utf-8", "replace")
            raise FFmpegProcessError(argv, returncode, error)


def probe_audio(
    path: str | Path,
    *,
    tools: FFmpegTools | None = None,
    resource_root: str | Path | None = None,
    ffmpeg_path: str | Path | None = None,
    ffprobe_path: str | Path | None = None,
    progress_cb: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
) -> AudioInfo:
    """Probe an audio file and count all decoded f32 PCM frames.

    ffprobe supplies stream shape and codec only.  Duration is computed from
    the exact number of bytes emitted by ffmpeg's decoder.
    """

    audio_path = Path(path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    resolved_tools = _resolve_tools(
        tools,
        resource_root=resource_root,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
    )
    sample_rate, channels, codec, estimated_duration = _probe_stream(audio_path, resolved_tools)
    bytes_per_frame = channels * np.dtype("<f4").itemsize
    byte_count = 0
    estimated_bytes = (
        int(estimated_duration * sample_rate * bytes_per_frame)
        if estimated_duration and estimated_duration > 0
        else 0
    )
    _report(progress_cb, 0.0)
    for chunk in _decoded_byte_chunks(audio_path, resolved_tools, cancel=cancel):
        byte_count += len(chunk)
        if estimated_bytes:
            _report(progress_cb, min(0.99, byte_count / estimated_bytes))
    if byte_count % bytes_per_frame:
        raise ValueError("ffmpeg emitted a partial PCM frame")
    total_samples = byte_count // bytes_per_frame
    if total_samples <= 0:
        raise ValueError(f"audio stream contains no decoded samples: {audio_path}")
    _report(progress_cb, 1.0)
    return AudioInfo(
        path=audio_path,
        sample_rate=sample_rate,
        channels=channels,
        total_samples=total_samples,
        duration_seconds=total_samples / sample_rate,
        codec=codec,
    )


def decode_f32(
    path: str | Path,
    *,
    info: AudioInfo | None = None,
    start_sample: int = 0,
    end_sample: int | None = None,
    tools: FFmpegTools | None = None,
    resource_root: str | Path | None = None,
    cancel: CancelCheck | None = None,
) -> np.ndarray:
    """Decode a (preferably short) exact sample range to ``frames x channels``."""

    audio_path = Path(path).expanduser().resolve()
    resolved_tools = _resolve_tools(tools, resource_root=resource_root)
    if info is None:
        sample_rate, channels, _codec, _duration = _probe_stream(audio_path, resolved_tools)
    else:
        sample_rate, channels = info.sample_rate, info.channels
    del sample_rate  # the filter below operates directly in sample units
    start = max(0, int(start_sample))
    if end_sample is not None and int(end_sample) <= start:
        return np.empty((0, channels), dtype=np.float32)
    filter_text = f"atrim=start_sample={start}"
    if end_sample is not None:
        filter_text += f":end_sample={int(end_sample)}"
    chunks = list(
        _decoded_byte_chunks(
            audio_path,
            resolved_tools,
            audio_filter=filter_text,
            cancel=cancel,
        )
    )
    raw = b"".join(chunks)
    frame_bytes = channels * 4
    if len(raw) % frame_bytes:
        raise ValueError("ffmpeg emitted a partial PCM frame")
    return np.frombuffer(raw, dtype="<f4").reshape((-1, channels)).copy()


def _as_mono(samples: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError("samples must be one-dimensional or frames x channels")
    if array.shape[1] == 0:
        return np.empty(array.shape[0], dtype=np.float32)
    return np.mean(array, axis=1, dtype=np.float32)


def compute_waveform_envelope(
    samples: np.ndarray | Sequence[float],
    sample_rate: int,
    block_size: int,
) -> WaveformEnvelope:
    """Compute one min/max/RMS level, padding no samples into the result."""

    if sample_rate <= 0 or block_size <= 0:
        raise ValueError("sample_rate and block_size must be positive")
    mono = _as_mono(samples)
    if mono.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return WaveformEnvelope(sample_rate, block_size, empty, empty.copy(), empty.copy())
    blocks = math.ceil(mono.size / block_size)
    minimum = np.empty(blocks, dtype=np.float32)
    maximum = np.empty(blocks, dtype=np.float32)
    rms = np.empty(blocks, dtype=np.float32)
    for index in range(blocks):
        part = mono[index * block_size : (index + 1) * block_size]
        minimum[index] = np.min(part)
        maximum[index] = np.max(part)
        rms[index] = math.sqrt(float(np.mean(np.square(part, dtype=np.float64))))
    return WaveformEnvelope(sample_rate, block_size, minimum, maximum, rms)


def _downsample_envelope(level: WaveformEnvelope) -> WaveformEnvelope:
    count = math.ceil(level.points / 2)
    minimum = np.empty(count, dtype=np.float32)
    maximum = np.empty(count, dtype=np.float32)
    rms = np.empty(count, dtype=np.float32)
    for index in range(count):
        start = index * 2
        stop = min(start + 2, level.points)
        minimum[index] = np.min(level.minimum[start:stop])
        maximum[index] = np.max(level.maximum[start:stop])
        rms[index] = math.sqrt(float(np.mean(np.square(level.rms[start:stop], dtype=np.float64))))
    return WaveformEnvelope(level.sample_rate, level.block_size * 2, minimum, maximum, rms)


def build_waveform_envelopes(
    samples: np.ndarray | Sequence[float],
    sample_rate: int,
    *,
    base_block_size: int = 256,
    max_levels: int = 12,
    minimum_points: int = 64,
) -> list[WaveformEnvelope]:
    """Build successively coarser waveform envelopes for smooth zooming."""

    if max_levels <= 0:
        raise ValueError("max_levels must be positive")
    levels = [compute_waveform_envelope(samples, sample_rate, base_block_size)]
    while len(levels) < max_levels and levels[-1].points > minimum_points:
        levels.append(_downsample_envelope(levels[-1]))
    return levels


def read_waveform_envelopes(
    path: str | Path,
    *,
    info: AudioInfo,
    tools: FFmpegTools | None = None,
    resource_root: str | Path | None = None,
    target_points: int = 8_000,
    max_levels: int = 8,
    progress_cb: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
) -> list[WaveformEnvelope]:
    """Decode and bucket a large file without retaining its full PCM payload."""

    if target_points <= 0:
        raise ValueError("target_points must be positive")
    audio_path = Path(path).expanduser().resolve()
    resolved_tools = _resolve_tools(tools, resource_root=resource_root)
    block_size = max(1, math.ceil(info.total_samples / target_points))
    minimum: list[float] = []
    maximum: list[float] = []
    rms: list[float] = []
    pending = np.empty(0, dtype=np.float32)
    processed = 0
    frame_bytes = info.channels * 4
    byte_remainder = b""
    _report(progress_cb, 0.0)
    for chunk in _decoded_byte_chunks(audio_path, resolved_tools, cancel=cancel):
        raw = byte_remainder + chunk
        usable = len(raw) - (len(raw) % frame_bytes)
        byte_remainder = raw[usable:]
        if not usable:
            continue
        frames = np.frombuffer(raw[:usable], dtype="<f4").reshape((-1, info.channels))
        mono = _as_mono(frames)
        if pending.size:
            mono = np.concatenate((pending, mono))
        complete = (mono.size // block_size) * block_size
        if complete:
            shaped = mono[:complete].reshape((-1, block_size))
            minimum.extend(np.min(shaped, axis=1).tolist())
            maximum.extend(np.max(shaped, axis=1).tolist())
            rms.extend(np.sqrt(np.mean(np.square(shaped, dtype=np.float64), axis=1)).tolist())
        pending = mono[complete:].copy()
        processed += frames.shape[0]
        _report(progress_cb, min(0.99, processed / info.total_samples))
    if byte_remainder:
        raise ValueError("ffmpeg emitted a partial PCM frame")
    if pending.size:
        minimum.append(float(np.min(pending)))
        maximum.append(float(np.max(pending)))
        rms.append(math.sqrt(float(np.mean(np.square(pending, dtype=np.float64)))))
    base = WaveformEnvelope(
        info.sample_rate,
        block_size,
        np.asarray(minimum, dtype=np.float32),
        np.asarray(maximum, dtype=np.float32),
        np.asarray(rms, dtype=np.float32),
    )
    levels = [base]
    while len(levels) < max_levels and levels[-1].points > 1:
        levels.append(_downsample_envelope(levels[-1]))
    _report(progress_cb, 1.0)
    return levels


# Singular spelling is convenient for callers that only display one zoom level.
def read_waveform_envelope(*args, **kwargs) -> WaveformEnvelope:
    return read_waveform_envelopes(*args, **kwargs)[0]


def refine_boundary(
    samples: np.ndarray | Sequence[float],
    predicted_sample: int,
    sample_rate: int,
    *,
    search_ms: float = 60.0,
    lower_bound: int = 0,
    upper_bound: int | None = None,
    energy_window_ms: float = 2.0,
) -> int:
    """Move a predicted cut to a nearby quiet zero crossing.

    The scoring deliberately favors low local RMS over distance.  This avoids
    cutting a phoneme merely because a louder zero crossing is one sample
    closer to the model prediction.
    """

    if sample_rate <= 0 or search_ms < 0:
        raise ValueError("invalid sample rate or search window")
    mono = _as_mono(samples)
    if mono.size == 0:
        return 0
    high_limit = mono.size if upper_bound is None else min(mono.size, int(upper_bound))
    low_limit = max(0, int(lower_bound))
    if high_limit <= low_limit:
        raise ValueError("upper_bound must be greater than lower_bound")
    predicted = int(np.clip(int(predicted_sample), low_limit, high_limit - 1))
    radius = max(0, round(sample_rate * search_ms / 1000.0))
    start = max(low_limit, predicted - radius)
    stop = min(high_limit, predicted + radius + 1)
    segment = mono[start:stop].astype(np.float64, copy=False)
    if segment.size == 1:
        return start

    energy_radius = max(1, round(sample_rate * energy_window_ms / 2000.0))
    squared = np.square(segment)
    kernel = np.ones(energy_radius * 2 + 1, dtype=np.float64)
    sums = np.convolve(squared, kernel, mode="same")
    counts = np.convolve(np.ones(segment.size, dtype=np.float64), kernel, mode="same")
    local_rms = np.sqrt(sums / counts)
    peak_rms = float(np.max(local_rms)) or 1.0
    peak_amp = float(np.max(np.abs(segment))) or 1.0

    crossing = np.zeros(segment.size, dtype=bool)
    crossing[segment == 0] = True
    signs = np.signbit(segment)
    crossing[1:] |= signs[1:] != signs[:-1]
    distance = np.abs(np.arange(start, stop) - predicted) / max(1, radius)
    score = (
        0.68 * (local_rms / peak_rms)
        + 0.20 * (np.abs(segment) / peak_amp)
        + 0.08 * distance
        + 0.04 * (~crossing)
    )
    best = int(np.argmin(score))
    return start + best


def _coerce_interval(interval: object) -> tuple[int, int]:
    if isinstance(interval, CutInterval):
        return interval.start_sample, interval.end_sample
    if isinstance(interval, dict):
        return int(interval["start_sample"]), int(interval["end_sample"])
    try:
        start, end = interval  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise TypeError("interval must have start_sample/end_sample or be a pair") from exc
    return int(start), int(end)


def merge_intervals(
    intervals: Iterable[object],
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> list[tuple[int, int]]:
    """Clamp, sort, and merge overlapping or touching sample intervals."""

    if maximum is not None and maximum < minimum:
        raise ValueError("maximum must not be less than minimum")
    normalized: list[tuple[int, int]] = []
    for interval in intervals:
        start, end = _coerce_interval(interval)
        start = max(int(minimum), start)
        if maximum is not None:
            end = min(int(maximum), end)
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


def write_cut_list_csv(
    path: str | Path,
    intervals: Iterable[object],
    sample_rate: int,
) -> Path:
    """Write an Excel-friendly UTF-8 cut list with sample-exact boundaries."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "start_sample",
                "end_sample",
                "start_seconds",
                "end_seconds",
                "duration_seconds",
                "text",
                "status",
            ],
        )
        writer.writeheader()
        for index, interval in enumerate(intervals, 1):
            start, end = _coerce_interval(interval)
            writer.writerow(
                {
                    "index": index,
                    "start_sample": start,
                    "end_sample": end,
                    "start_seconds": f"{start / sample_rate:.6f}",
                    "end_seconds": f"{end / sample_rate:.6f}",
                    "duration_seconds": f"{(end - start) / sample_rate:.6f}",
                    "text": getattr(interval, "text", "") if not isinstance(interval, dict) else interval.get("text", ""),
                    "status": getattr(interval, "status", "") if not isinstance(interval, dict) else interval.get("status", ""),
                }
            )
    return output


# Backwards/short-form aliases used by the UI layer.
build_waveform_envelope = compute_waveform_envelope
generate_waveform_envelopes = build_waveform_envelopes
write_cut_csv = write_cut_list_csv
