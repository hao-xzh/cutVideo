"""Continuous, timestamp-preserving ASR over overlapping local windows."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .alignment import (
    AlignmentCancelledError,
    FunASRAsrAligner,
    RecognizedToken,
    normalize_with_mapping,
    voice_ranges_from_model_result,
)
from .model_runtime import PreparedAudioWindowCache, local_audio_window

_FIRST_CORE_WINDOW_MS = 5_000
_CORE_WINDOW_MS = 12_000
_CONTEXT_MS = 1_200
_TIMESTAMP_EPSILON_MS = 5.0
_MAX_CROSS_WINDOW_TIMELINE_REPAIR_MS = 500.0
_SINGLE_TOKEN_DUPLICATE_BOUNDARY_TOLERANCE_MS = 40.0
_PHRASE_DUPLICATE_BOUNDARY_TOLERANCE_MS = 160.0
_MIN_PHRASE_DUPLICATE_TOKENS = 4
_MAX_DUPLICATE_PREFIX_TOKENS = 96
_COARSE_TIMESTAMP_PRECISIONS = frozenset({"segment", "unknown"})
_WINDOW_MATCH_MARGIN_MS = 250.0
_WINDOW_MATCH_MAX_CENTER_DISTANCE_MS = 650.0
_WINDOW_MATCH_MAX_INTERVAL_GAP_MS = 300.0
_QWEN_SAMPLE_RATE = 16_000
_QWEN_FIRST_TARGET_MS = 5_000
_QWEN_FIRST_SEARCH_MS = 1_500
_QWEN_NEXT_TARGET_MS = 20_000
_QWEN_NEXT_SEARCH_MS = 3_000
_QWEN_MIN_FIRST_MS = 3_000
_QWEN_MIN_NEXT_MS = 12_000
_QWEN_MAX_CHUNK_MS = 30_000
_QWEN_ENERGY_WINDOW_MS = 200
_QWEN_ENERGY_STEP_MS = 50
_SENTENCE_BOUNDARY_CHARACTERS = frozenset("。！？!?；;")
_CLAUSE_BOUNDARY_CHARACTERS = frozenset("，,：:")


@dataclass(frozen=True, slots=True)
class ProgressiveRecognitionChunk:
    """One newly committed, chronological portion of a transcription."""

    tokens: tuple[RecognizedToken, ...]
    sequence: int
    total_sequences: int
    committed_until_ms: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class ProgressiveRecognitionResult:
    """The authoritative token timeline assembled from all emitted chunks."""

    tokens: tuple[RecognizedToken, ...]
    voice_ranges: tuple[tuple[float, float], ...]
    chunk_count: int
    deduplicated_token_count: int = 0
    inference_device: str = "unknown"
    sentence_boundary_indexes: tuple[int, ...] = ()
    clause_boundary_indexes: tuple[int, ...] = ()


ProgressCallback = Callable[[float, str], None]
ChunkCallback = Callable[[ProgressiveRecognitionChunk], None]


def _recognition_text(result: object) -> str:
    if isinstance(result, Mapping):
        for key in ("text", "sentence", "transcript"):
            value = result.get(key)
            if value is not None:
                return str(value)
    for name in ("text", "sentence", "transcript"):
        value = getattr(result, name, None)
        if value is not None:
            return str(value)
    return result if isinstance(result, str) else ""


def _semantic_boundary_offsets(text: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Map model punctuation to offsets in the punctuation-free token stream."""

    sentence_offsets: set[int] = set()
    clause_offsets: set[int] = set()
    normalized_count = 0
    for character in text:
        normalized_count += len(normalize_with_mapping(character).text)
        if normalized_count <= 0:
            continue
        if character in _SENTENCE_BOUNDARY_CHARACTERS:
            sentence_offsets.add(normalized_count)
        elif character in _CLAUSE_BOUNDARY_CHARACTERS:
            clause_offsets.add(normalized_count)
    return tuple(sorted(sentence_offsets)), tuple(sorted(clause_offsets))


def _cancelled(cancel: object | None) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(cancel)


def _raise_if_cancelled(cancel: object | None) -> None:
    if _cancelled(cancel):
        raise AlignmentCancelledError("progressive recognition was cancelled")


def _shift_token(token: RecognizedToken, offset_ms: int) -> RecognizedToken:
    return RecognizedToken(
        token.text,
        token.start_ms + offset_ms,
        token.end_ms + offset_ms,
        token.confidence,
        token.confidence_available,
        token.timestamp_precision,
    )


def _tokens_for_core(
    tokens: Sequence[RecognizedToken],
    *,
    core_start_ms: int,
    core_end_ms: int,
    final: bool,
) -> list[tuple[int, RecognizedToken]]:
    selected: list[tuple[int, RecognizedToken]] = []
    for index, token in enumerate(tokens):
        midpoint = (token.start_ms + token.end_ms) / 2
        if midpoint < core_start_ms:
            continue
        if midpoint < core_end_ms or (final and midpoint <= core_end_ms):
            selected.append((index, token))
    return selected


def _token_midpoint(token: RecognizedToken) -> float:
    return (token.start_ms + token.end_ms) / 2.0


def _token_interval_gap(left: RecognizedToken, right: RecognizedToken) -> float:
    return max(0.0, max(left.start_ms, right.start_ms) - min(left.end_ms, right.end_ms))


def _overlap_token_matches(
    previous: Sequence[RecognizedToken],
    incoming: Sequence[RecognizedToken],
    *,
    overlap_start_ms: float,
    overlap_end_ms: float,
) -> list[tuple[int, int]]:
    """Align identical text in the acoustic overlap of adjacent windows.

    Maximising match count first and minimising timestamp distance second keeps
    repeated natural speech aligned to its own occurrence.  This is stronger
    evidence than comparing only the rendered suffix and prefix.
    """

    if overlap_end_ms <= overlap_start_ms:
        return []

    def in_overlap(token: RecognizedToken) -> bool:
        return (
            token.end_ms >= overlap_start_ms - _WINDOW_MATCH_MARGIN_MS
            and token.start_ms <= overlap_end_ms + _WINDOW_MATCH_MARGIN_MS
        )

    left = [(index, token) for index, token in enumerate(previous) if in_overlap(token)]
    right = [(index, token) for index, token in enumerate(incoming) if in_overlap(token)]
    if not left or not right:
        return []

    # Each cell is ``(matched token count, negative accumulated centre drift)``.
    # One additional exact character always outranks any timestamp tie-break.
    scores: list[list[tuple[int, float]]] = [
        [(0, 0.0) for _ in range(len(right) + 1)] for _ in range(len(left) + 1)
    ]
    directions: list[list[str]] = [
        ["" for _ in range(len(right) + 1)] for _ in range(len(left) + 1)
    ]
    for left_index in range(1, len(left) + 1):
        directions[left_index][0] = "up"
    for right_index in range(1, len(right) + 1):
        directions[0][right_index] = "left"

    for left_index in range(1, len(left) + 1):
        previous_token = left[left_index - 1][1]
        for right_index in range(1, len(right) + 1):
            incoming_token = right[right_index - 1][1]
            candidates: list[tuple[tuple[int, float], str]] = [
                (scores[left_index - 1][right_index], "up"),
                (scores[left_index][right_index - 1], "left"),
            ]
            center_distance = abs(
                _token_midpoint(previous_token) - _token_midpoint(incoming_token)
            )
            if (
                previous_token.text == incoming_token.text
                and center_distance <= _WINDOW_MATCH_MAX_CENTER_DISTANCE_MS
                and _token_interval_gap(previous_token, incoming_token)
                <= _WINDOW_MATCH_MAX_INTERVAL_GAP_MS
            ):
                prior_count, prior_distance = scores[left_index - 1][right_index - 1]
                candidates.append(
                    ((prior_count + 1, prior_distance - center_distance), "match")
                )
            score, direction = max(
                candidates,
                key=lambda item: (
                    item[0],
                    item[1] == "match",
                    item[1] == "up",
                ),
            )
            scores[left_index][right_index] = score
            directions[left_index][right_index] = direction

    matches: list[tuple[int, int]] = []
    left_index = len(left)
    right_index = len(right)
    while left_index and right_index:
        direction = directions[left_index][right_index]
        if direction == "match":
            matches.append((left[left_index - 1][0], right[right_index - 1][0]))
            left_index -= 1
            right_index -= 1
        elif direction == "up":
            left_index -= 1
        else:
            right_index -= 1
    matches.reverse()
    return matches


def _match_run_is_same_acoustics(
    previous: Sequence[RecognizedToken],
    incoming: Sequence[RecognizedToken],
    run: Sequence[tuple[int, int]],
) -> bool:
    """Reject ambiguous one-off text matches while accepting stable phrases."""

    if not run:
        return False
    center_distances = [
        abs(_token_midpoint(previous[left]) - _token_midpoint(incoming[right]))
        for left, right in run
    ]
    interval_gaps = [
        _token_interval_gap(previous[left], incoming[right]) for left, right in run
    ]
    if len(run) == 1:
        left, right = run[0]
        return (
            center_distances[0] <= _SINGLE_TOKEN_DUPLICATE_BOUNDARY_TOLERANCE_MS
            and min(previous[left].end_ms, incoming[right].end_ms)
            > max(previous[left].start_ms, incoming[right].start_ms)
        )
    if len(run) == 2:
        return (
            sum(center_distances) / 2.0 <= 180.0
            and max(center_distances) <= 280.0
            and max(interval_gaps) <= 100.0
        )
    return (
        sum(center_distances) / len(center_distances) <= 350.0
        and max(center_distances) <= _WINDOW_MATCH_MAX_CENTER_DISTANCE_MS
        and max(interval_gaps) <= _WINDOW_MATCH_MAX_INTERVAL_GAP_MS
    )


def _duplicate_incoming_indexes(
    previous: Sequence[RecognizedToken],
    incoming: Sequence[RecognizedToken],
    *,
    overlap_start_ms: float,
    overlap_end_ms: float,
    core_boundary_ms: float,
) -> set[int]:
    """Return incoming tokens already committed by the preceding core."""

    matches = _overlap_token_matches(
        previous,
        incoming,
        overlap_start_ms=overlap_start_ms,
        overlap_end_ms=overlap_end_ms,
    )
    duplicates: set[int] = set()
    run_start = 0
    while run_start < len(matches):
        run_end = run_start + 1
        while run_end < len(matches):
            previous_pair = matches[run_end - 1]
            next_pair = matches[run_end]
            if (
                next_pair[0] != previous_pair[0] + 1
                or next_pair[1] != previous_pair[1] + 1
            ):
                break
            run_end += 1
        run = matches[run_start:run_end]
        if _match_run_is_same_acoustics(previous, incoming, run):
            for previous_index, incoming_index in run:
                if (
                    _token_midpoint(previous[previous_index]) < core_boundary_ms
                    and _token_midpoint(incoming[incoming_index]) >= core_boundary_ms
                ):
                    duplicates.add(incoming_index)
        run_start = run_end
    return duplicates


def _clip_tokens_to_core(
    tokens: Sequence[RecognizedToken],
    *,
    core_start_ms: int,
    core_end_ms: int,
) -> tuple[RecognizedToken, ...]:
    """Keep committed timestamps inside the non-overlapping core interval."""

    clipped: list[RecognizedToken] = []
    for token in tokens:
        start_ms = max(float(core_start_ms), token.start_ms)
        end_ms = min(float(core_end_ms), token.end_ms)
        if end_ms <= start_ms:
            raise ValueError("渐进识别 token 无法裁入所属核心窗口")
        if start_ms == token.start_ms and end_ms == token.end_ms:
            clipped.append(token)
            continue
        clipped.append(
            RecognizedToken(
                token.text,
                start_ms,
                end_ms,
                token.confidence,
                token.confidence_available,
                "unknown",
            )
        )
    return tuple(clipped)


def _duplicate_prefix_length(
    committed: Sequence[RecognizedToken],
    incoming: Sequence[RecognizedToken],
) -> int:
    """Find a timestamp-overlapping text suffix repeated by the next window.

    A one-token text match is intentionally stricter than a phrase match.  A
    loose adjacency tolerance would turn a genuine repeated word (for example
    ``好好``) into one token at a core boundary.
    """

    # Sentence-level backends can give every character in one short window the
    # same coarse interval.  Cover a complete short-window sentence so shared
    # acoustic context cannot be rendered twice at the boundary.
    maximum = min(
        _MAX_DUPLICATE_PREFIX_TOKENS,
        len(committed),
        len(incoming),
    )
    for count in range(maximum, 0, -1):
        existing = committed[-count:]
        candidate = incoming[:count]
        pairs = tuple(zip(existing, candidate, strict=True))
        if not all(left.text == right.text for left, right in pairs):
            continue
        tolerance_ms = (
            _SINGLE_TOKEN_DUPLICATE_BOUNDARY_TOLERANCE_MS
            if count == 1
            else _PHRASE_DUPLICATE_BOUNDARY_TOLERANCE_MS
        )
        timeline_matches = all(
            min(left.end_ms, right.end_ms) > max(left.start_ms, right.start_ms)
            and abs(left.start_ms - right.start_ms) <= tolerance_ms
            and abs(left.end_ms - right.end_ms) <= tolerance_ms
            for left, right in pairs
        )
        if not timeline_matches:
            continue
        if any(
            token.timestamp_precision in _COARSE_TIMESTAMP_PRECISIONS
            for pair in pairs
            for token in pair
        ):
            # Coarse repeated text is fundamentally ambiguous at any length.
            # A visible duplicate is safer than deleting genuine speech.
            return 0
        if 1 < count < _MIN_PHRASE_DUPLICATE_TOKENS:
            # Never collapse a genuine adjacent short repetition such as
            # ``好好`` or ``可以可以`` merely because its coarse intervals
            # overlap at the core boundary.
            return 0
        return count
    return 0


def _append_monotonic_tokens(
    committed: list[RecognizedToken],
    incoming: Sequence[RecognizedToken],
) -> tuple[RecognizedToken, ...]:
    """Append a window core while preserving the persisted timeline contract."""

    appended: list[RecognizedToken] = []
    for token in incoming:
        candidate = token
        if committed:
            previous = committed[-1]
            end_adjustment = max(0.0, previous.end_ms - candidate.end_ms)
            # Coarse tokens may span the core boundary and therefore have an
            # early start while their end still advances.  Reject only when the
            # complete interval moves backwards beyond plausible VAD overlap.
            if end_adjustment > _MAX_CROSS_WINDOW_TIMELINE_REPAIR_MS:
                raise ValueError(
                    "渐进识别时间轴跨窗口回退超过 500 毫秒；已停止写入，避免生成错位音频标注"
                )
            start_ms = max(previous.start_ms, candidate.start_ms)
            end_ms = max(previous.end_ms, candidate.end_ms)
            if end_ms <= start_ms:
                end_ms = start_ms + _TIMESTAMP_EPSILON_MS
            if start_ms != candidate.start_ms or end_ms != candidate.end_ms:
                candidate = RecognizedToken(
                    candidate.text,
                    start_ms,
                    end_ms,
                    candidate.confidence,
                    candidate.confidence_available,
                    "unknown",
                )
        committed.append(candidate)
        appended.append(candidate)
    return tuple(appended)


def _ranges_for_core(
    ranges: Sequence[tuple[float, float]],
    *,
    core_start_ms: int,
    core_end_ms: int,
) -> list[tuple[float, float]]:
    clipped: list[tuple[float, float]] = []
    for start_ms, end_ms in ranges:
        start = max(float(core_start_ms), start_ms)
        end = min(float(core_end_ms), end_ms)
        if end > start:
            clipped.append((start, end))
    return clipped


def _ranges_from_tokens(tokens: Sequence[RecognizedToken]) -> list[tuple[float, float]]:
    if not tokens:
        return []
    ranges: list[tuple[float, float]] = []
    start = tokens[0].start_ms
    end = tokens[0].end_ms
    for token in tokens[1:]:
        if token.start_ms - end >= 500.0:
            ranges.append((start, end))
            start = token.start_ms
        end = max(end, token.end_ms)
    ranges.append((start, end))
    return ranges


def _merge_ranges(
    ranges: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    merged: list[tuple[float, float]] = []
    for start_ms, end_ms in sorted(ranges):
        if merged and start_ms <= merged[-1][1] + _TIMESTAMP_EPSILON_MS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ms))
        else:
            merged.append((start_ms, end_ms))
    return tuple(merged)


def _qwen_local_energy_cut(audio: object, start_sample: int, *, first: bool) -> int:
    import numpy as np

    waveform = np.asarray(audio, dtype=np.float32)
    total_samples = len(waveform)
    maximum_samples = _QWEN_MAX_CHUNK_MS * _QWEN_SAMPLE_RATE // 1000
    if total_samples - start_sample <= maximum_samples:
        return total_samples

    target_ms = _QWEN_FIRST_TARGET_MS if first else _QWEN_NEXT_TARGET_MS
    search_ms = _QWEN_FIRST_SEARCH_MS if first else _QWEN_NEXT_SEARCH_MS
    minimum_ms = _QWEN_MIN_FIRST_MS if first else _QWEN_MIN_NEXT_MS
    nominal = start_sample + target_ms * _QWEN_SAMPLE_RATE // 1000
    search_start = max(
        start_sample + minimum_ms * _QWEN_SAMPLE_RATE // 1000,
        nominal - search_ms * _QWEN_SAMPLE_RATE // 1000,
    )
    search_end = min(
        start_sample + maximum_samples,
        nominal + search_ms * _QWEN_SAMPLE_RATE // 1000,
        total_samples,
    )
    window_samples = max(1, _QWEN_ENERGY_WINDOW_MS * _QWEN_SAMPLE_RATE // 1000)
    step_samples = max(1, _QWEN_ENERGY_STEP_MS * _QWEN_SAMPLE_RATE // 1000)
    best_cut = min(nominal, search_end)
    best_energy = math.inf
    for window_start in range(
        search_start,
        max(search_start + 1, search_end - window_samples + 1),
        step_samples,
    ):
        window = waveform[window_start : window_start + window_samples]
        if not len(window):
            continue
        energy = float(np.mean(window * window))
        if energy < best_energy:
            best_energy = energy
            best_cut = window_start + len(window) // 2
    return max(start_sample + 1, min(best_cut, total_samples))


def _qwen_chunk_bounds(audio: object) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start_sample = 0
    total_samples = len(audio)  # type: ignore[arg-type]
    while start_sample < total_samples:
        end_sample = _qwen_local_energy_cut(
            audio,
            start_sample,
            first=not bounds,
        )
        bounds.append((start_sample, end_sample))
        start_sample = end_sample
    return bounds


def _recognize_audio_progressively_qwen_mlx(
    *,
    audio_path: Path,
    duration_ms: int,
    asr_model_path: str | Path,
    ffmpeg_path: Path,
    progress_cb: ProgressCallback | None,
    chunk_cb: ChunkCallback | None,
    cancel: object | None,
) -> ProgressiveRecognitionResult:
    from .qwen_mlx import QwenMlxAsrAligner

    recognizer = QwenMlxAsrAligner(asr_model_path)
    committed: list[RecognizedToken] = []
    voice_ranges: list[tuple[float, float]] = []
    sentence_boundary_indexes: set[int] = set()
    clause_boundary_indexes: set[int] = set()
    try:
        if progress_cb is not None:
            progress_cb(0.0, "正在准备首段音频与 GPU 识别模型…")
        with local_audio_window(
            audio_path,
            start_ms=0,
            end_ms=duration_ms,
            ffmpeg_path=ffmpeg_path,
            cancel=cancel,
        ) as prepared:
            audio = recognizer.load_prepared_audio(prepared)
        # MLX streams are thread-local.  Model loading and inference must stay
        # on this same worker thread; preloading on a helper thread makes the
        # cached GPU stream unusable during decoding.
        recognizer.preload()
        _raise_if_cancelled(cancel)
        bounds = _qwen_chunk_bounds(audio)
        total_sequences = len(bounds)
        for sequence, (start_sample, end_sample) in enumerate(bounds):
            _raise_if_cancelled(cancel)
            start_ms = round(start_sample * 1000 / _QWEN_SAMPLE_RATE)
            end_ms = round(end_sample * 1000 / _QWEN_SAMPLE_RATE)
            if end_ms <= start_ms:
                end_ms = start_ms + 1
            if progress_cb is not None:
                progress_cb(
                    start_sample / max(1, len(audio)),  # type: ignore[arg-type]
                    f"正在持续识别 {sequence + 1}/{total_sequences}…",
                )
            tokens, raw_result = recognizer.recognize_audio_with_result(
                audio=audio[start_sample:end_sample],  # type: ignore[index]
                time_offset_ms=start_ms,
                window_end_ms=end_ms,
            )
            committed_before_chunk = len(committed)
            appended = _append_monotonic_tokens(committed, tokens)
            sentence_offsets, clause_offsets = _semantic_boundary_offsets(
                _recognition_text(raw_result)
            )
            sentence_boundary_indexes.update(
                committed_before_chunk + offset
                for offset in sentence_offsets
                # A local ASR window almost always adds terminal punctuation,
                # even when its low-energy cut lands inside a sentence.  Only
                # punctuation followed by more speech in the same window is
                # strong semantic evidence; the final boundary is unnecessary.
                if 0 < offset < len(appended)
            )
            clause_boundary_indexes.update(
                committed_before_chunk + offset
                for offset in clause_offsets
                if 0 < offset < len(appended)
            )
            voice_ranges.extend(_ranges_from_tokens(appended))
            if chunk_cb is not None and appended:
                chunk_cb(
                    ProgressiveRecognitionChunk(
                        appended,
                        sequence + 1,
                        total_sequences,
                        end_ms,
                        duration_ms,
                    )
                )
            if progress_cb is not None:
                progress_cb(
                    end_sample / max(1, len(audio)),  # type: ignore[arg-type]
                    f"已持续识别到 {end_ms / 1000:.1f} 秒",
                )
        return ProgressiveRecognitionResult(
            tokens=tuple(committed),
            voice_ranges=_merge_ranges(voice_ranges),
            chunk_count=len(bounds),
            deduplicated_token_count=0,
            inference_device=recognizer.device,
            sentence_boundary_indexes=tuple(sorted(sentence_boundary_indexes)),
            clause_boundary_indexes=tuple(sorted(clause_boundary_indexes)),
        )
    finally:
        recognizer.release()


def recognize_audio_progressively(
    *,
    audio_path: str | Path,
    duration_ms: int,
    asr_model_path: str | Path,
    vad_model_path: str | Path | None,
    ffmpeg_path: str | Path,
    progress_cb: ProgressCallback | None = None,
    chunk_cb: ChunkCallback | None = None,
    cancel: object | None = None,
    device: str | None = None,
    first_core_window_ms: int = _FIRST_CORE_WINDOW_MS,
    core_window_ms: int = _CORE_WINDOW_MS,
    context_ms: int = _CONTEXT_MS,
    backend: str = "funasr",
) -> ProgressiveRecognitionResult:
    """Recognize front-to-back and publish each stable core immediately.

    Every inference window includes acoustic context on both sides, while only
    its non-overlapping core is committed.  This keeps later windows flowing at
    a steady cadence without duplicating their shared context.  Extracting
    each source window with FFmpeg input seeking prevents later chunks from
    becoming progressively slower on long compressed audio.
    """

    if (
        duration_ms <= 0
        or first_core_window_ms <= 0
        or core_window_ms <= 0
        or context_ms < 0
    ):
        raise ValueError("invalid progressive recognition window")
    source = Path(audio_path).expanduser().resolve(strict=True)
    resolved_ffmpeg = Path(ffmpeg_path).expanduser().resolve(strict=True)
    if backend == "qwen-mlx":
        return _recognize_audio_progressively_qwen_mlx(
            audio_path=source,
            duration_ms=duration_ms,
            asr_model_path=asr_model_path,
            ffmpeg_path=resolved_ffmpeg,
            progress_cb=progress_cb,
            chunk_cb=chunk_cb,
            cancel=cancel,
        )
    if backend != "funasr":
        raise ValueError(f"unknown progressive recognition backend: {backend}")
    if vad_model_path is None:
        raise ValueError("FunASR progressive recognition requires a VAD model")
    first_core_end = min(duration_ms, first_core_window_ms)
    remaining_ms = max(0, duration_ms - first_core_end)
    total_sequences = 1 + math.ceil(remaining_ms / core_window_ms)
    recognizer = FunASRAsrAligner(
        asr_model_path,
        vad_model_path,
        ffmpeg_path=None,
        device=device,
    )
    committed: list[RecognizedToken] = []
    raw_committed: list[RecognizedToken] = []
    voice_ranges: list[tuple[float, float]] = []
    previous_window_tokens: tuple[RecognizedToken, ...] = ()
    previous_window_start = 0
    previous_window_end = 0
    deduplicated_token_count = 0

    windows: list[tuple[int, int, int, int]] = []
    for sequence in range(total_sequences):
        if sequence == 0:
            core_start = 0
            core_end = first_core_end
        else:
            core_start = first_core_end + (sequence - 1) * core_window_ms
            core_end = min(duration_ms, core_start + core_window_ms)
        windows.append(
            (
                core_start,
                core_end,
                max(0, core_start - context_ms),
                min(duration_ms, core_end + context_ms),
            )
        )

    # Audio decoding and MPS/CUDA inference use different hardware.  Keep one
    # bounded window decoded ahead so the accelerator does not sit idle while
    # FFmpeg starts and seeks the next compressed segment.  CPU inference stays
    # sequential to avoid competing with FFmpeg for the same cores.
    accelerated = not recognizer.device.casefold().startswith("cpu")
    executor: ThreadPoolExecutor | None = None
    prepared_future: Future[Path] | None = None
    window_cache = PreparedAudioWindowCache(
        source,
        ffmpeg_path=resolved_ffmpeg,
        cancel=cancel,
        fast_seek=True,
    )
    if accelerated:
        try:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="cutvideo-audio-prefetch",
            )
            prepared_future = executor.submit(
                window_cache.get,
                windows[0][2],
                windows[0][3],
            )
            # If the idle application warmup has not finished yet, deserialize
            # the model while the first compressed-audio window is decoded.
            recognizer.preload()
            if recognizer.device.casefold().startswith("cpu"):
                executor.shutdown(wait=True, cancel_futures=True)
                executor = None
                prepared_future = None
        except Exception:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            window_cache.close()
            raise

    try:
        for sequence, (core_start, core_end, window_start, window_end) in enumerate(windows):
            _raise_if_cancelled(cancel)
            if progress_cb is not None:
                progress_cb(
                    core_start / duration_ms,
                    f"正在持续识别 {sequence + 1}/{total_sequences}…",
                )
            if prepared_future is not None:
                window = prepared_future.result()
            else:
                window = window_cache.get(window_start, window_end)
            if executor is not None and sequence + 1 < total_sequences:
                next_window = windows[sequence + 1]
                prepared_future = executor.submit(
                    window_cache.get,
                    next_window[2],
                    next_window[3],
                )
            else:
                prepared_future = None
            window_duration = window_end - window_start
            relative_tokens, raw_result = recognizer.recognize_prepared_with_result(
                audio_path=window,
                time_offset_ms=0,
                window_end_ms=window_duration,
            )
            if executor is not None and recognizer.device.casefold().startswith("cpu"):
                executor.shutdown(wait=True, cancel_futures=True)
                executor = None
                prepared_future = None
            _raise_if_cancelled(cancel)
            absolute_tokens = tuple(
                _shift_token(token, window_start) for token in relative_tokens
            )
            indexed_core_tokens = _tokens_for_core(
                absolute_tokens,
                core_start_ms=core_start,
                core_end_ms=core_end,
                final=sequence + 1 == total_sequences,
            )
            overlap_duplicates: set[int] = set()
            if previous_window_tokens:
                overlap_duplicates = _duplicate_incoming_indexes(
                    previous_window_tokens,
                    absolute_tokens,
                    overlap_start_ms=max(previous_window_start, window_start),
                    overlap_end_ms=min(previous_window_end, window_end),
                    core_boundary_ms=core_start,
                )
            core_tokens = [
                token
                for token_index, token in indexed_core_tokens
                if token_index not in overlap_duplicates
            ]
            deduplicated_token_count += len(indexed_core_tokens) - len(core_tokens)
            duplicate_prefix = _duplicate_prefix_length(raw_committed, core_tokens)
            unique_raw_tokens = core_tokens[duplicate_prefix:]
            deduplicated_token_count += duplicate_prefix
            committed_core_tokens = _clip_tokens_to_core(
                unique_raw_tokens,
                core_start_ms=core_start,
                core_end_ms=core_end,
            )
            appended = _append_monotonic_tokens(committed, committed_core_tokens)
            raw_committed.extend(unique_raw_tokens)
            raw_ranges = voice_ranges_from_model_result(
                raw_result,
                time_offset_ms=window_start,
                window_end_ms=window_end,
            )
            core_ranges = _ranges_for_core(
                raw_ranges,
                core_start_ms=core_start,
                core_end_ms=core_end,
            )
            voice_ranges.extend(core_ranges or _ranges_from_tokens(appended))
            if chunk_cb is not None and appended:
                chunk_cb(
                    ProgressiveRecognitionChunk(
                        appended,
                        sequence + 1,
                        total_sequences,
                        core_end,
                        duration_ms,
                    )
                )
            if progress_cb is not None:
                progress_cb(
                    core_end / duration_ms,
                    f"已持续识别到 {core_end / 1000:.1f} 秒",
                )
            previous_window_tokens = absolute_tokens
            previous_window_start = window_start
            previous_window_end = window_end
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        window_cache.close()

    return ProgressiveRecognitionResult(
        tokens=tuple(committed),
        voice_ranges=_merge_ranges(voice_ranges),
        chunk_count=total_sequences,
        deduplicated_token_count=deduplicated_token_count,
        inference_device=recognizer.device,
    )


__all__ = [
    "ProgressiveRecognitionChunk",
    "ProgressiveRecognitionResult",
    "recognize_audio_progressively",
]
