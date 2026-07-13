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
BOUNDARY_SEARCH_MS = 60.0
BOUNDARY_MAX_EXPAND_MS = 20.0
BOUNDARY_ZERO_CROSSING_MS = 8.0
BOUNDARY_GUARD_SAFETY_MS = 5.0
BOUNDARY_MIN_EDGE_CONFIDENCE = 0.25


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
class BoundaryRefinement:
    start_sample: int
    end_sample: int
    expanded: bool
    requires_review: bool
    diagnostics: dict[str, object]


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


def probe_audio_with_waveform(
    path: str | Path,
    *,
    tools: FFmpegTools | None = None,
    resource_root: str | Path | None = None,
    points_per_second: int = 200,
    minimum_points: int = 12_000,
    maximum_points: int = 600_000,
    progress_cb: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
) -> tuple[AudioInfo, WaveformEnvelope]:
    """Count exact PCM frames and build the review waveform in one decode pass."""

    if points_per_second <= 0 or minimum_points <= 0 or maximum_points < minimum_points:
        raise ValueError("invalid waveform density")
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    resolved_tools = _resolve_tools(tools, resource_root=resource_root)
    sample_rate, channels, codec, estimated_duration = _probe_stream(audio_path, resolved_tools)
    estimated_seconds = estimated_duration if estimated_duration and estimated_duration > 0 else 60.0
    target_points = min(
        maximum_points,
        max(minimum_points, round(estimated_seconds * points_per_second)),
    )
    block_size = max(1, math.ceil(estimated_seconds * sample_rate / target_points))
    minimum: list[float] = []
    maximum: list[float] = []
    rms: list[float] = []
    pending = np.empty(0, dtype=np.float32)
    processed = 0
    frame_bytes = channels * 4
    byte_remainder = b""
    estimated_frames = max(1, round(estimated_seconds * sample_rate))
    _report(progress_cb, 0.0)
    for chunk in _decoded_byte_chunks(audio_path, resolved_tools, cancel=cancel):
        raw = byte_remainder + chunk
        usable = len(raw) - len(raw) % frame_bytes
        byte_remainder = raw[usable:]
        if not usable:
            continue
        frames = np.frombuffer(raw[:usable], dtype="<f4").reshape((-1, channels))
        mono = _as_mono(frames)
        if pending.size:
            mono = np.concatenate((pending, mono))
        complete = mono.size // block_size * block_size
        if complete:
            shaped = mono[:complete].reshape((-1, block_size))
            minimum.extend(np.min(shaped, axis=1).tolist())
            maximum.extend(np.max(shaped, axis=1).tolist())
            rms.extend(np.sqrt(np.mean(np.square(shaped, dtype=np.float64), axis=1)).tolist())
        pending = mono[complete:].copy()
        processed += frames.shape[0]
        _report(progress_cb, min(0.99, processed / estimated_frames))
    if byte_remainder:
        raise ValueError("ffmpeg emitted a partial PCM frame")
    if processed <= 0:
        raise ValueError(f"audio stream contains no decoded samples: {audio_path}")
    if pending.size:
        minimum.append(float(np.min(pending)))
        maximum.append(float(np.max(pending)))
        rms.append(math.sqrt(float(np.mean(np.square(pending, dtype=np.float64)))))
    envelope = WaveformEnvelope(
        sample_rate,
        block_size,
        np.asarray(minimum, dtype=np.float32),
        np.asarray(maximum, dtype=np.float32),
        np.asarray(rms, dtype=np.float32),
    )
    while envelope.points > maximum_points:
        envelope = _downsample_envelope(envelope)
    info = AudioInfo(
        path=audio_path,
        sample_rate=sample_rate,
        channels=channels,
        total_samples=processed,
        duration_seconds=processed / sample_rate,
        codec=codec,
    )
    _report(progress_cb, 1.0)
    return info, envelope


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


def _boundary_signals(
    samples: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        signal = array.astype(np.float64, copy=False)
        return np.abs(signal), signal
    if array.ndim != 2:
        raise ValueError("samples must be one-dimensional or frames x channels")
    if array.shape[0] == 0 or array.shape[1] == 0:
        empty = np.empty(array.shape[0], dtype=np.float64)
        return empty, empty.copy()
    channels = array.astype(np.float64, copy=False)
    # Energy is computed before channel mixing so opposite-phase stereo cannot
    # erase speech.  A single stable channel is then used only for sign changes.
    energy = np.sqrt(np.mean(np.square(channels), axis=1))
    channel_rms = np.sqrt(np.mean(np.square(channels), axis=0))
    zero_signal = channels[:, int(np.argmax(channel_rms))]
    return energy, zero_signal


def _detect_energy_edge(
    energy: np.ndarray,
    predicted: int,
    sample_rate: int,
    *,
    kind: str,
    lower_bound: int,
    upper_bound: int,
) -> tuple[int, str, float, float | None]:
    frame_size = max(2, round(sample_rate * 0.005))
    hop_size = max(1, round(sample_rate * 0.0025))
    scan_start = max(0, lower_bound - frame_size)
    scan_end = min(energy.size, upper_bound + frame_size + 1)
    if scan_end - scan_start < frame_size:
        return predicted, "model", 0.0, None
    starts = np.arange(scan_start, scan_end - frame_size + 1, hop_size, dtype=np.int64)
    rms = np.asarray(
        [
            math.sqrt(float(np.mean(np.square(energy[start : start + frame_size]))))
            for start in starts
        ],
        dtype=np.float64,
    )
    peak = float(np.max(rms)) if rms.size else 0.0
    if peak <= 1e-9:
        return predicted, "model", 0.0, None
    noise = float(np.percentile(rms, 20))
    threshold = max(noise * (10 ** (10 / 20)), peak * (10 ** (-35 / 20)), 1e-9)
    voiced = rms >= threshold
    stable = np.zeros(voiced.size, dtype=bool)
    for index in range(max(0, voiced.size - 1)):
        stable[index] = bool(voiced[index] and voiced[index + 1])

    candidates: list[int] = []
    if kind == "start":
        for index, active in enumerate(stable):
            if active and (index == 0 or not stable[index - 1]):
                candidates.append(int(starts[index]))
    elif kind == "end":
        for index in range(1, stable.size):
            if stable[index - 1] and not stable[index]:
                candidates.append(int(starts[index] + frame_size))
        if stable.size and stable[-1]:
            candidates.append(int(starts[-1] + frame_size))
    else:
        raise ValueError("kind must be 'start' or 'end'")
    candidates = [value for value in candidates if lower_bound <= value <= upper_bound]
    if not candidates:
        return predicted, "model", 0.0, None
    edge = min(candidates, key=lambda value: abs(value - predicted))
    confidence = max(0.0, min(1.0, (peak - threshold) / peak))
    threshold_db = 20 * math.log10(max(1e-12, threshold / peak))
    return edge, "energy", confidence, threshold_db


def _strict_zero_crossing(
    signal: np.ndarray,
    predicted: int,
    sample_rate: int,
    *,
    lower_bound: int,
    upper_bound: int,
    search_ms: float,
) -> int | None:
    if signal.size == 0:
        return None
    radius = max(0, round(sample_rate * search_ms / 1000.0))
    start = max(0, lower_bound, predicted - radius)
    stop = min(signal.size - 1, upper_bound, predicted + radius)
    if stop < start:
        return None
    segment = signal[start : stop + 1].astype(np.float64, copy=False)
    crossing = np.zeros(segment.size, dtype=bool)
    crossing[segment == 0] = True
    if segment.size > 1:
        signs = np.signbit(segment)
        crossing[1:] |= signs[1:] != signs[:-1]
    candidate_offsets = np.flatnonzero(crossing)
    if not candidate_offsets.size:
        return None
    energy_radius = max(1, round(sample_rate * 0.0025))
    kernel = np.ones(energy_radius * 2 + 1, dtype=np.float64)
    local_rms = np.sqrt(
        np.convolve(np.square(segment), kernel, mode="same")
        / np.convolve(np.ones(segment.size, dtype=np.float64), kernel, mode="same")
    )
    peak = float(np.max(local_rms)) or 1.0
    candidates = start + candidate_offsets
    score = 0.75 * (local_rms[candidate_offsets] / peak) + 0.25 * (
        np.abs(candidates - predicted) / max(1, radius)
    )
    return int(candidates[int(np.argmin(score))])


def refine_boundary(
    samples: np.ndarray | Sequence[float],
    predicted_sample: int,
    sample_rate: int,
    *,
    search_ms: float = BOUNDARY_SEARCH_MS,
    lower_bound: int = 0,
    upper_bound: int | None = None,
    energy_window_ms: float = 2.0,
) -> int:
    """Return a real nearby zero crossing, otherwise keep the prediction."""

    del energy_window_ms  # retained for API compatibility
    if sample_rate <= 0 or not math.isfinite(search_ms) or search_ms < 0:
        raise ValueError("invalid sample rate or search window")
    _energy, signal = _boundary_signals(samples)
    if signal.size == 0:
        return 0
    high_limit = signal.size if upper_bound is None else min(signal.size, int(upper_bound))
    low_limit = max(0, int(lower_bound))
    if high_limit <= low_limit:
        raise ValueError("upper_bound must be greater than lower_bound")
    predicted = int(np.clip(int(predicted_sample), low_limit, high_limit - 1))
    crossing = _strict_zero_crossing(
        signal,
        predicted,
        sample_rate,
        lower_bound=low_limit,
        upper_bound=high_limit - 1,
        search_ms=search_ms,
    )
    return predicted if crossing is None else crossing


def refine_cut_boundaries(
    samples: np.ndarray | Sequence[float],
    predicted_start_sample: int,
    predicted_end_sample: int,
    sample_rate: int,
    *,
    left_guard_sample: int | None = None,
    right_guard_sample: int | None = None,
    vad_start_sample: int | None = None,
    vad_end_sample: int | None = None,
    search_ms: float = BOUNDARY_SEARCH_MS,
    max_expand_ms: float = BOUNDARY_MAX_EXPAND_MS,
    zero_crossing_ms: float = BOUNDARY_ZERO_CROSSING_MS,
    guard_safety_ms: float = BOUNDARY_GUARD_SAFETY_MS,
    sample_offset: int = 0,
) -> BoundaryRefinement:
    """Refine a cut using speech edges, neighbor guards and strict zero crossings."""

    if sample_rate <= 0 or any(
        not math.isfinite(value) or value < 0
        for value in (search_ms, max_expand_ms, zero_crossing_ms, guard_safety_ms)
    ):
        raise ValueError("invalid boundary refinement parameters")
    energy, zero_signal = _boundary_signals(samples)
    sample_count = int(energy.size)
    if sample_count < 2:
        return BoundaryRefinement(
            0,
            max(1, sample_count),
            False,
            True,
            {
                "fallback": "empty_pcm",
                "evidence_complete": False,
                "review_reasons": ["empty_pcm"],
            },
        )
    predicted_start = max(0, min(sample_count - 1, int(predicted_start_sample)))
    predicted_end = max(predicted_start + 1, min(sample_count, int(predicted_end_sample)))
    search_samples = round(sample_rate * search_ms / 1000.0)
    expand_samples = round(sample_rate * max_expand_ms / 1000.0)
    safety_samples = round(sample_rate * guard_safety_ms / 1000.0)

    start_minimum = predicted_start
    if left_guard_sample is not None:
        start_minimum = max(
            0,
            predicted_start - expand_samples,
            int(left_guard_sample) + safety_samples,
        )
    start_maximum = min(sample_count - 1, predicted_start + search_samples, predicted_end - 1)
    start_guard_conflict = start_maximum < start_minimum

    def choose_edge(
        kind: str,
        predicted: int,
        lower: int,
        upper: int,
        vad_edge: int | None,
    ) -> tuple[int, str, float, float | None]:
        if (
            vad_edge is not None
            and lower <= int(vad_edge) <= upper
            and abs(int(vad_edge) - predicted) <= search_samples
        ):
            return int(vad_edge), "vad", 1.0, None
        return _detect_energy_edge(
            energy,
            predicted,
            sample_rate,
            kind=kind,
            lower_bound=lower,
            upper_bound=upper,
        )

    if start_guard_conflict:
        start_edge = refined_start = predicted_start
        start_method = "guard_conflict"
        start_confidence = 0.0
        start_threshold_db = None
        start_crossing = None
    else:
        start_edge, start_method, start_confidence, start_threshold_db = choose_edge(
            "start",
            predicted_start,
            start_minimum,
            start_maximum,
            vad_start_sample,
        )
        start_edge_supported = (
            start_method in {"vad", "energy"}
            and start_confidence >= BOUNDARY_MIN_EDGE_CONFIDENCE
        )
        start_crossing = (
            _strict_zero_crossing(
                zero_signal,
                start_edge,
                sample_rate,
                lower_bound=start_minimum,
                upper_bound=start_maximum,
                search_ms=zero_crossing_ms,
            )
            if start_edge_supported
            else None
        )
        refined_start = (
            start_crossing
            if start_crossing is not None
            else (start_edge if start_edge_supported else predicted_start)
        )

    end_minimum = max(refined_start + 1, predicted_end - search_samples)
    end_maximum = predicted_end
    if right_guard_sample is not None:
        end_maximum = min(
            sample_count,
            predicted_end + expand_samples,
            int(right_guard_sample) - safety_samples,
        )
    end_guard_conflict = end_maximum < end_minimum
    if end_guard_conflict:
        end_edge = refined_end = predicted_end
        end_method = "guard_conflict"
        end_confidence = 0.0
        end_threshold_db = None
        end_crossing = None
    else:
        end_edge, end_method, end_confidence, end_threshold_db = choose_edge(
            "end",
            predicted_end,
            end_minimum,
            end_maximum,
            vad_end_sample,
        )
        end_edge_supported = (
            end_method in {"vad", "energy"}
            and end_confidence >= BOUNDARY_MIN_EDGE_CONFIDENCE
        )
        end_crossing = (
            _strict_zero_crossing(
                zero_signal,
                min(sample_count - 1, end_edge),
                sample_rate,
                lower_bound=min(sample_count - 1, end_minimum),
                upper_bound=min(sample_count - 1, end_maximum),
                search_ms=zero_crossing_ms,
            )
            if end_edge_supported
            else None
        )
        refined_end = (
            end_crossing
            if end_crossing is not None
            else (end_edge if end_edge_supported else predicted_end)
        )
    invalid_interval = not 0 <= refined_start < refined_end <= sample_count
    if invalid_interval:
        refined_start, refined_end = predicted_start, predicted_end
        start_method = end_method = "model"
        start_crossing = end_crossing = None
    expanded = refined_start < predicted_start or refined_end > predicted_end
    review_reasons: list[str] = []
    if start_guard_conflict or end_guard_conflict:
        review_reasons.append("adjacent_guard_conflict")
    if start_method not in {"vad", "energy"} or start_confidence < BOUNDARY_MIN_EDGE_CONFIDENCE:
        review_reasons.append("start_speech_edge_missing")
    if end_method not in {"vad", "energy"} or end_confidence < BOUNDARY_MIN_EDGE_CONFIDENCE:
        review_reasons.append("end_speech_edge_missing")
    if start_crossing is None:
        review_reasons.append("start_zero_crossing_missing")
    if end_crossing is None:
        review_reasons.append("end_zero_crossing_missing")
    if invalid_interval:
        review_reasons.append("refined_interval_invalid")
    if expanded:
        review_reasons.append("boundary_expanded")
    review_reasons = list(dict.fromkeys(review_reasons))
    evidence_complete = not review_reasons

    def absolute(value: int | None) -> int | None:
        return None if value is None else sample_offset + int(value)

    diagnostics: dict[str, object] = {
        "sequence": ["speech_edge", "adjacent_guard", "limited_expansion", "zero_crossing"],
        "search_ms": float(search_ms),
        "max_expand_ms": float(max_expand_ms),
        "zero_crossing_ms": float(zero_crossing_ms),
        "guard_safety_ms": float(guard_safety_ms),
        "stereo_energy_mode": "channel_rms_before_mix",
        "expanded": expanded,
        "evidence_complete": evidence_complete,
        "guard_conflict": start_guard_conflict or end_guard_conflict,
        "review_reasons": review_reasons,
        "left_guard_sample": absolute(left_guard_sample),
        "right_guard_sample": absolute(right_guard_sample),
        "start": {
            "predicted_sample": absolute(predicted_start),
            "speech_edge_sample": absolute(start_edge),
            "final_sample": absolute(refined_start),
            "delta_samples": refined_start - predicted_start,
            "method": start_method,
            "confidence": round(start_confidence, 4),
            "energy_threshold_db": (
                round(start_threshold_db, 3) if start_threshold_db is not None else None
            ),
            "zero_crossing_applied": start_crossing is not None,
        },
        "end": {
            "predicted_sample": absolute(predicted_end),
            "speech_edge_sample": absolute(end_edge),
            "final_sample": absolute(refined_end),
            "delta_samples": refined_end - predicted_end,
            "method": end_method,
            "confidence": round(end_confidence, 4),
            "energy_threshold_db": (
                round(end_threshold_db, 3) if end_threshold_db is not None else None
            ),
            "zero_crossing_applied": end_crossing is not None,
        },
    }
    return BoundaryRefinement(
        refined_start,
        refined_end,
        expanded,
        not evidence_complete,
        diagnostics,
    )


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
