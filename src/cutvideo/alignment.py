"""Offline transcript-to-audio alignment and conservative cut suggestions.

The public entry point, :func:`align_transcript`, intentionally consumes the
DOCX/parser and audio-info objects by attribute rather than importing their
types.  It can therefore be reused by the UI, project migration code and small
test doubles without introducing a storage-layer dependency.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pypinyin import Style, pinyin

from .model_runtime import (
    ModelUnavailableError,
    PreparedAudioWindowCache,
    evict_funasr_model,
    load_funasr_model,
    local_audio_window,
    model_execution_guard,
    model_inference_device,
    require_local_model,
    resolve_inference_device,
)

_LOGGER = logging.getLogger(__name__)

SHORT_HIGHLIGHT = "short_highlight"
REPEATED_CONTEXT = "repeated_context"
INSUFFICIENT_COVERAGE = "insufficient_coverage"
BOUNDARY_DISAGREEMENT = "boundary_disagreement_over_120ms"
FALLBACK_ALIGNMENT = "fallback_character_ratio"
FORCE_ALIGNER_UNAVAILABLE = "force_aligner_unavailable"
ASR_ALIGNER_UNAVAILABLE = "asr_aligner_unavailable"
DOCUMENT_TIME_FALLBACK = "document_time_search_fallback"
FORCE_CONTEXT_UNSTABLE = "local_context_alignment_unstable"
LOCAL_RECOGNITION_UNAVAILABLE = "local_audio_recognition_refinement_unavailable"
ALIGNMENT_AMBIGUOUS = "alignment_candidate_margin_too_small"
COARSE_TIMESTAMP = "coarse_timestamp_not_character_level"
TIMESTAMP_ANOMALY = "timestamp_validation_failed"
MODEL_CONFIDENCE_UNAVAILABLE = "model_character_confidence_unavailable"
LOW_MODEL_CONFIDENCE = "low_model_character_confidence"
BOUNDARY_GUARD_UNAVAILABLE = "adjacent_character_guard_unavailable"
BOUNDARY_EXPANDED = "boundary_expanded_for_speech_edge"
BOUNDARY_REFINEMENT_UNCERTAIN = "boundary_refinement_evidence_incomplete"

# Stored in every project so results produced by an older alignment strategy
# are never silently reused after the audio-first pipeline changes.
ALIGNMENT_PIPELINE_PURPOSE = "alignment_strategy"
ALIGNMENT_PIPELINE_NAME = "audio-text-primary"
ALIGNMENT_PIPELINE_VERSION = "9"

# A highlighted phrase should be one locally continuous piece of speech.  A
# larger hole usually means that sparse ASR matches from two different spoken
# passages were accidentally joined.  Paragraph/context lookup deliberately
# does not use this limit because those wider search ranges may contain pauses.
MAX_HIGHLIGHT_INTERNAL_GAP_MS = 1_500.0

STATUS_NEEDS_REVIEW = "needs_review"
STATUS_AUTO_APPROVED = "auto_approved"

TIMESTAMP_PRECISION_CHARACTER = "character"
TIMESTAMP_PRECISION_TOKEN = "token"
TIMESTAMP_PRECISION_SEGMENT = "segment"
TIMESTAMP_PRECISION_UNKNOWN = "unknown"
_PRECISION_RANK = {
    TIMESTAMP_PRECISION_CHARACTER: 0,
    TIMESTAMP_PRECISION_TOKEN: 1,
    TIMESTAMP_PRECISION_SEGMENT: 2,
    TIMESTAMP_PRECISION_UNKNOWN: 3,
}
_TIMESTAMP_EPSILON_MS = 5.0
_MIN_TIMESTAMP_SPAN_MS = 5.0
_MODEL_WINDOW_BOUNDARY_TOLERANCE_MS = 120.0
_VAD_GAP_MS = 500.0
_MIN_ALIGNMENT_MARGIN = 0.12
_LOW_MODEL_CONFIDENCE_THRESHOLD = 0.50
_SPOKEN_FILLER_CHARACTERS = frozenset({"呃", "嗯", "啊", "哦", "诶", "唉", "额"})


class AlignmentCancelledError(RuntimeError):
    """Raised when alignment is cancelled between paragraph windows."""


def _is_cancelled(cancel: object | None) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(cancel)

_TAG_RE = re.compile(r"<[^>]+>")
_MISSING = object()


@dataclass(frozen=True, slots=True)
class NormalizedText:
    """Search/alignment text plus a lossless map back to original offsets."""

    original: str
    text: str
    normalized_to_original: tuple[int, ...]

    def normalized_range(self, original_start: int, original_end: int) -> tuple[int, int] | None:
        """Return the normalized half-open range covered by an original range."""

        if original_start < 0 or original_end < original_start or original_end > len(self.original):
            raise ValueError("original range is outside the text")
        indexes = [
            index
            for index, original_index in enumerate(self.normalized_to_original)
            if original_start <= original_index < original_end
        ]
        if not indexes:
            return None
        return indexes[0], indexes[-1] + 1


def normalize_with_mapping(text: str) -> NormalizedText:
    """Normalize Chinese/Latin text for matching while retaining source indexes.

    Unicode compatibility forms and Latin case are folded.  Whitespace,
    punctuation, symbols and formatting/control characters are ignored, as
    they are normally absent or unstable in ASR output.  Every emitted code
    point retains the index of the original code point that produced it.
    """

    normalized: list[str] = []
    mapping: list[int] = []
    for original_index, character in enumerate(text):
        for folded in unicodedata.normalize("NFKC", character).casefold():
            category = unicodedata.category(folded)
            if folded.isspace() or category[0] in {"P", "S", "C", "Z"}:
                continue
            normalized.append(folded)
            mapping.append(original_index)
    return NormalizedText(text, "".join(normalized), tuple(mapping))


# A short alias reads naturally at call sites and is kept public for plugins.
normalize_text = normalize_with_mapping


def _is_han_character(character: str) -> bool:
    if len(character) != 1:
        return False
    codepoint = ord(character)
    return any(
        start <= codepoint <= end
        for start, end in (
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2FA1F),
            (0x30000, 0x323AF),
        )
    )


def _phonetic_keys(text: str) -> tuple[str, ...]:
    """Return one context-aware, tone-free matching key per normalized character."""

    readings = pinyin(
        text,
        style=Style.NORMAL,
        heteronym=False,
        errors=lambda value: list(value),
        strict=False,
    )
    if len(readings) != len(text):
        # The error callback above normally preserves a one-to-one shape for
        # non-Han runs.  Keep a defensive literal fallback: a shape mismatch
        # must never shift timestamp indexes.
        return tuple(f"literal:{character}" for character in text)
    return tuple(
        f"han:{reading[0]}"
        if _is_han_character(character) and reading
        else f"literal:{character}"
        for character, reading in zip(text, readings, strict=True)
    )


@dataclass(frozen=True, slots=True)
class _DpEntry:
    score: float
    previous_i: int
    previous_j: int
    previous_rank: int
    operation: str


@dataclass(frozen=True, slots=True)
class _ChunkAlignment:
    best_pairs: tuple[tuple[int, int, float], ...]
    second_pairs: tuple[tuple[int, int, float], ...]
    best_score: float
    second_score: float | None
    forced_partition: bool = False


@dataclass(frozen=True, slots=True)
class _CharacterAlignment:
    pairs: tuple[tuple[int, int, float], ...]
    margins: Mapping[int, float]
    ambiguity_margin: float
    forced_partition: bool


def _character_match(
    reference_character: str,
    observed_character: str,
    reference_key: str,
    observed_key: str,
    *,
    allow_phonetic_fallback: bool,
) -> tuple[float, float]:
    if reference_character == observed_character:
        return 2.0, 1.0
    if (
        allow_phonetic_fallback
        and _is_han_character(reference_character)
        and _is_han_character(observed_character)
        and reference_key == observed_key
    ):
        return 1.35, 0.88
    return -1.15, 0.0


def _reference_deletion_penalty(reference_text: str, index: int) -> float:
    """Discount a duplicated manuscript character spoken only once."""

    character = reference_text[index]
    duplicated = (
        (index > 0 and reference_text[index - 1] == character)
        or (index + 1 < len(reference_text) and reference_text[index + 1] == character)
    )
    return 0.22 if duplicated else 0.75


def _observed_insertion_penalty(observed_text: str, index: int) -> float:
    """Treat common hesitation particles as normal spoken realization."""

    return 0.18 if observed_text[index] in _SPOKEN_FILLER_CHARACTERS else 0.55


def _reference_segment_indexes(transcript: str, normalized: NormalizedText) -> tuple[int, ...]:
    segment_at_original: list[int] = []
    segment = 0
    for character in transcript:
        segment_at_original.append(segment)
        if character in {"\n", "\r"}:
            segment += 1
    return tuple(
        segment_at_original[index] if index < len(segment_at_original) else segment
        for index in normalized.normalized_to_original
    )


def _unique_monotonic_anchors(
    reference_text: str,
    observed_text: str,
    *,
    width: int = 4,
) -> list[tuple[int, int, int]]:
    if min(len(reference_text), len(observed_text)) < width:
        return []
    reference_positions: dict[str, list[int]] = {}
    observed_positions: dict[str, list[int]] = {}
    for index in range(len(reference_text) - width + 1):
        reference_positions.setdefault(reference_text[index : index + width], []).append(index)
    for index in range(len(observed_text) - width + 1):
        observed_positions.setdefault(observed_text[index : index + width], []).append(index)
    candidates = sorted(
        (positions[0], observed_positions[key][0], width)
        for key, positions in reference_positions.items()
        if len(positions) == 1 and len(observed_positions.get(key, ())) == 1
    )
    if not candidates:
        return []

    tails: list[int] = []
    tail_candidates: list[int] = []
    previous = [-1] * len(candidates)
    for candidate_index, (_reference, observed, _width) in enumerate(candidates):
        position = bisect_left(tails, observed)
        if position == len(tails):
            tails.append(observed)
            tail_candidates.append(candidate_index)
        else:
            tails[position] = observed
            tail_candidates[position] = candidate_index
        if position:
            previous[candidate_index] = tail_candidates[position - 1]

    selected: list[tuple[int, int, int]] = []
    candidate_index = tail_candidates[-1]
    while candidate_index >= 0:
        selected.append(candidates[candidate_index])
        candidate_index = previous[candidate_index]
    selected.reverse()

    non_overlapping: list[tuple[int, int, int]] = []
    reference_end = observed_end = 0
    for reference, observed, anchor_width in selected:
        if reference < reference_end or observed < observed_end:
            continue
        non_overlapping.append((reference, observed, anchor_width))
        reference_end = reference + anchor_width
        observed_end = observed + anchor_width
    return non_overlapping


def _trace_dp_path(
    cells: Mapping[tuple[int, int], tuple[_DpEntry, ...]],
    end_i: int,
    end_j: int,
    rank: int,
    reference_text: str,
    observed_text: str,
    reference_keys: Sequence[str],
    observed_keys: Sequence[str],
    *,
    reference_offset: int,
    observed_offset: int,
    allow_phonetic_fallback: bool,
) -> tuple[tuple[int, int, float], ...]:
    pairs: list[tuple[int, int, float]] = []
    i, j = end_i, end_j
    while i or j:
        entries = cells.get((i, j), ())
        if rank >= len(entries):
            break
        entry = entries[rank]
        if entry.operation == "match":
            _score, confidence = _character_match(
                reference_text[i - 1],
                observed_text[j - 1],
                reference_keys[i - 1],
                observed_keys[j - 1],
                allow_phonetic_fallback=allow_phonetic_fallback,
            )
            if confidence > 0:
                pairs.append(
                    (reference_offset + i - 1, observed_offset + j - 1, confidence)
                )
        i, j, rank = entry.previous_i, entry.previous_j, entry.previous_rank
    pairs.reverse()
    return tuple(pairs)


def _dp_align_chunk(
    reference_text: str,
    observed_text: str,
    *,
    reference_offset: int,
    observed_offset: int,
    allow_phonetic_fallback: bool,
    reference_segments: Sequence[int],
    observed_segments: Sequence[int],
) -> _ChunkAlignment:
    reference_length = len(reference_text)
    observed_length = len(observed_text)
    if not reference_length:
        return _ChunkAlignment(
            (),
            (),
            -sum(
                _observed_insertion_penalty(observed_text, index)
                for index in range(observed_length)
            ),
            None,
        )
    if not observed_length:
        return _ChunkAlignment(
            (),
            (),
            -sum(
                _reference_deletion_penalty(reference_text, index)
                for index in range(reference_length)
            ),
            None,
        )

    band = max(32, abs(reference_length - observed_length) + 24)
    estimated_cells = (reference_length + 1) * min(observed_length + 1, 2 * band + 3)
    if estimated_cells > 280_000 and reference_length > 1 and observed_length > 1:
        reference_midpoint = reference_length // 2
        observed_midpoint = max(
            1,
            min(
                observed_length - 1,
                round(observed_length * reference_midpoint / reference_length),
            ),
        )
        left = _dp_align_chunk(
            reference_text[:reference_midpoint],
            observed_text[:observed_midpoint],
            reference_offset=reference_offset,
            observed_offset=observed_offset,
            allow_phonetic_fallback=allow_phonetic_fallback,
            reference_segments=reference_segments[:reference_midpoint],
            observed_segments=observed_segments[:observed_midpoint],
        )
        right = _dp_align_chunk(
            reference_text[reference_midpoint:],
            observed_text[observed_midpoint:],
            reference_offset=reference_offset + reference_midpoint,
            observed_offset=observed_offset + observed_midpoint,
            allow_phonetic_fallback=allow_phonetic_fallback,
            reference_segments=reference_segments[reference_midpoint:],
            observed_segments=observed_segments[observed_midpoint:],
        )
        best_pairs = left.best_pairs + right.best_pairs
        alternatives: list[tuple[float, tuple[tuple[int, int, float], ...]]] = []
        if left.second_score is not None:
            alternatives.append(
                (left.second_score + right.best_score, left.second_pairs + right.best_pairs)
            )
        if right.second_score is not None:
            alternatives.append(
                (left.best_score + right.second_score, left.best_pairs + right.second_pairs)
            )
        alternatives.sort(key=lambda item: item[0], reverse=True)
        return _ChunkAlignment(
            best_pairs,
            alternatives[0][1] if alternatives else (),
            left.best_score + right.best_score,
            alternatives[0][0] if alternatives else None,
            True,
        )

    reference_keys = (
        _phonetic_keys(reference_text)
        if allow_phonetic_fallback
        else tuple(f"literal:{character}" for character in reference_text)
    )
    observed_keys = (
        _phonetic_keys(observed_text)
        if allow_phonetic_fallback
        else tuple(f"literal:{character}" for character in observed_text)
    )
    reference_segment_count = max(reference_segments, default=0) + 1
    observed_segment_count = max(observed_segments, default=0) + 1
    cells: dict[tuple[int, int], tuple[_DpEntry, ...]] = {
        (0, 0): (_DpEntry(0.0, 0, 0, 0, ""),)
    }
    for i in range(reference_length + 1):
        center = round(i * observed_length / reference_length)
        minimum_j = max(0, center - band)
        maximum_j = min(observed_length, center + band)
        if i == 0:
            minimum_j = 0
        if i == reference_length:
            maximum_j = observed_length
        for j in range(minimum_j, maximum_j + 1):
            if i == 0 and j == 0:
                continue
            possibilities: list[_DpEntry] = []
            if i and j:
                score, _confidence = _character_match(
                    reference_text[i - 1],
                    observed_text[j - 1],
                    reference_keys[i - 1],
                    observed_keys[j - 1],
                    allow_phonetic_fallback=allow_phonetic_fallback,
                )
                if reference_segment_count > 1 and observed_segment_count > 1:
                    reference_position = reference_segments[i - 1] / (reference_segment_count - 1)
                    observed_position = observed_segments[j - 1] / (observed_segment_count - 1)
                    score -= 0.20 * abs(reference_position - observed_position)
                for rank, previous in enumerate(cells.get((i - 1, j - 1), ())):
                    possibilities.append(
                        _DpEntry(previous.score + score, i - 1, j - 1, rank, "match")
                    )
            if i:
                for rank, previous in enumerate(cells.get((i - 1, j), ())):
                    possibilities.append(
                        _DpEntry(
                            previous.score
                            - _reference_deletion_penalty(reference_text, i - 1),
                            i - 1,
                            j,
                            rank,
                            "delete",
                        )
                    )
            if j:
                for rank, previous in enumerate(cells.get((i, j - 1), ())):
                    possibilities.append(
                        _DpEntry(
                            previous.score
                            - _observed_insertion_penalty(observed_text, j - 1),
                            i,
                            j - 1,
                            rank,
                            "insert",
                        )
                    )
            possibilities.sort(key=lambda entry: entry.score, reverse=True)
            selected: list[_DpEntry] = []
            seen: set[tuple[int, int, int, str]] = set()
            for entry in possibilities:
                identity = (
                    entry.previous_i,
                    entry.previous_j,
                    entry.previous_rank,
                    entry.operation,
                )
                if identity in seen:
                    continue
                seen.add(identity)
                selected.append(entry)
                if len(selected) == 2:
                    break
            if selected:
                cells[(i, j)] = tuple(selected)

    terminal = cells.get((reference_length, observed_length), ())
    if not terminal:
        return _ChunkAlignment((), (), float("-inf"), None, True)
    best_pairs = _trace_dp_path(
        cells,
        reference_length,
        observed_length,
        0,
        reference_text,
        observed_text,
        reference_keys,
        observed_keys,
        reference_offset=reference_offset,
        observed_offset=observed_offset,
        allow_phonetic_fallback=allow_phonetic_fallback,
    )
    second_pairs = (
        _trace_dp_path(
            cells,
            reference_length,
            observed_length,
            1,
            reference_text,
            observed_text,
            reference_keys,
            observed_keys,
            reference_offset=reference_offset,
            observed_offset=observed_offset,
            allow_phonetic_fallback=allow_phonetic_fallback,
        )
        if len(terminal) > 1
        else ()
    )
    return _ChunkAlignment(
        best_pairs,
        second_pairs,
        terminal[0].score,
        terminal[1].score if len(terminal) > 1 else None,
    )


def _monotonic_character_alignment(
    reference_text: str,
    observed_text: str,
    *,
    allow_phonetic_fallback: bool,
    reference_segments: Sequence[int] | None = None,
    observed_segments: Sequence[int] | None = None,
) -> _CharacterAlignment:
    if not reference_text or not observed_text:
        return _CharacterAlignment((), {}, 0.0, False)
    ref_segments = tuple(reference_segments or (0,) * len(reference_text))
    obs_segments = tuple(observed_segments or (0,) * len(observed_text))
    anchors = _unique_monotonic_anchors(reference_text, observed_text)
    chunks: list[_ChunkAlignment] = []
    anchor_pairs: list[tuple[int, int, float]] = []
    reference_cursor = observed_cursor = 0
    for reference_anchor, observed_anchor, width in anchors:
        chunks.append(
            _dp_align_chunk(
                reference_text[reference_cursor:reference_anchor],
                observed_text[observed_cursor:observed_anchor],
                reference_offset=reference_cursor,
                observed_offset=observed_cursor,
                allow_phonetic_fallback=allow_phonetic_fallback,
                reference_segments=ref_segments[reference_cursor:reference_anchor],
                observed_segments=obs_segments[observed_cursor:observed_anchor],
            )
        )
        anchor_pairs.extend(
            (reference_anchor + offset, observed_anchor + offset, 1.0)
            for offset in range(width)
        )
        reference_cursor = reference_anchor + width
        observed_cursor = observed_anchor + width
    chunks.append(
        _dp_align_chunk(
            reference_text[reference_cursor:],
            observed_text[observed_cursor:],
            reference_offset=reference_cursor,
            observed_offset=observed_cursor,
            allow_phonetic_fallback=allow_phonetic_fallback,
            reference_segments=ref_segments[reference_cursor:],
            observed_segments=obs_segments[observed_cursor:],
        )
    )

    best_pairs = list(anchor_pairs)
    for chunk in chunks:
        best_pairs.extend(chunk.best_pairs)
    best_pairs.sort(key=lambda match: (match[0], match[1]))

    ambiguous_chunk: _ChunkAlignment | None = None
    ambiguity_margin = 1.0
    for chunk in chunks:
        if chunk.second_score is None or not chunk.second_pairs:
            continue
        best_map = {reference: observed for reference, observed, _score in chunk.best_pairs}
        second_map = {
            reference: observed for reference, observed, _score in chunk.second_pairs
        }
        if best_map == second_map:
            # Two paths that only reorder equivalent insert/delete operations
            # do not represent a competing spoken occurrence.
            continue
        raw_margin = max(0.0, chunk.best_score - chunk.second_score)
        normalized_margin = min(1.0, raw_margin / max(1.0, 2.0 * len(chunk.best_pairs)))
        if normalized_margin < ambiguity_margin:
            ambiguity_margin = normalized_margin
            ambiguous_chunk = chunk

    margins: dict[int, float] = {}
    if ambiguous_chunk is not None:
        best_map = {reference: observed for reference, observed, _score in ambiguous_chunk.best_pairs}
        second_map = {
            reference: observed for reference, observed, _score in ambiguous_chunk.second_pairs
        }
        for reference in set(best_map) | set(second_map):
            if best_map.get(reference) != second_map.get(reference):
                margins[reference] = ambiguity_margin
    return _CharacterAlignment(
        tuple(best_pairs),
        margins,
        ambiguity_margin if margins else 1.0,
        any(chunk.forced_partition for chunk in chunks),
    )


def _matching_character_pairs(
    reference_text: str,
    observed_text: str,
    *,
    allow_phonetic_fallback: bool,
) -> list[tuple[int, int, float]]:
    """Map text with anchored, monotonic dynamic programming.

    Unique exact 4-character anchors partition long recordings.  Gaps use a
    weighted monotonic DP that explicitly models ASR insertions, omissions and
    Chinese homophones instead of relying on a full-text heuristic matcher.
    """

    return list(
        _monotonic_character_alignment(
            reference_text,
            observed_text,
            allow_phonetic_fallback=allow_phonetic_fallback,
        ).pairs
    )


@dataclass(frozen=True, slots=True)
class TimedSpan:
    """An absolute millisecond interval mapped to normalized transcript offsets."""

    normalized_start: int
    normalized_end: int
    start_ms: float
    end_ms: float
    confidence: float = 1.0
    model_confidence: float | None = None
    timestamp_precision: str = TIMESTAMP_PRECISION_CHARACTER
    alignment_margin: float = 1.0

    def __post_init__(self) -> None:
        if self.normalized_start < 0 or self.normalized_end <= self.normalized_start:
            raise ValueError("invalid normalized span")
        if (
            not math.isfinite(self.start_ms)
            or not math.isfinite(self.end_ms)
            or self.start_ms < 0
            or self.end_ms <= self.start_ms
        ):
            raise ValueError("invalid timed span")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("timed span confidence must be between zero and one")
        if self.model_confidence is not None and (
            not math.isfinite(self.model_confidence)
            or not 0.0 <= self.model_confidence <= 1.0
        ):
            raise ValueError("model confidence must be between zero and one")
        if self.timestamp_precision not in _PRECISION_RANK:
            raise ValueError("invalid timestamp precision")
        if not math.isfinite(self.alignment_margin) or not 0.0 <= self.alignment_margin <= 1.0:
            raise ValueError("alignment margin must be between zero and one")


@dataclass(frozen=True, slots=True)
class AlignmentTrack:
    """One model's normalized-reference alignment for a paragraph."""

    spans: tuple[TimedSpan, ...]
    engine: str
    coverage: float
    mean_model_confidence: float | None = None
    minimum_model_confidence: float | None = None
    low_confidence_ratio: float | None = None
    timestamp_precision: str = TIMESTAMP_PRECISION_UNKNOWN
    timestamp_valid: bool = True
    ambiguity_margin: float = 1.0
    diagnostics: tuple[str, ...] = ()
    vad_ranges: tuple[tuple[float, float], ...] = ()
    model_confidence_coverage: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.coverage) or not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be between zero and one")
        for value in (
            self.mean_model_confidence,
            self.minimum_model_confidence,
            self.low_confidence_ratio,
            self.model_confidence_coverage,
        ):
            if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError("track confidence metrics must be between zero and one")
        if self.timestamp_precision not in _PRECISION_RANK:
            raise ValueError("invalid track timestamp precision")
        if not isinstance(self.timestamp_valid, bool):
            raise ValueError("track timestamp validity must be boolean")
        if not isinstance(self.diagnostics, tuple) or any(
            not isinstance(item, str) for item in self.diagnostics
        ):
            raise ValueError("track diagnostics must be a tuple of strings")
        if not math.isfinite(self.ambiguity_margin) or not 0 <= self.ambiguity_margin <= 1:
            raise ValueError("track ambiguity margin must be between zero and one")
        previous_start = previous_end = -1.0
        previous_normalized = -1
        for span in self.spans:
            if span.normalized_start < previous_normalized:
                raise ValueError("alignment spans must be text-monotonic")
            if (
                span.start_ms + _TIMESTAMP_EPSILON_MS < previous_start
                or span.end_ms + _TIMESTAMP_EPSILON_MS < previous_end
            ):
                raise ValueError("alignment spans must be time-monotonic")
            previous_normalized = span.normalized_start
            previous_start = max(previous_start, span.start_ms)
            previous_end = max(previous_end, span.end_ms)
        previous_vad_start = previous_vad_end = -1.0
        for start_ms, end_ms in self.vad_ranges:
            if (
                not math.isfinite(start_ms)
                or not math.isfinite(end_ms)
                or start_ms < 0
                or end_ms <= start_ms
            ):
                raise ValueError("invalid VAD range")
            if (
                start_ms + _TIMESTAMP_EPSILON_MS < previous_vad_start
                or end_ms + _TIMESTAMP_EPSILON_MS < previous_vad_end
            ):
                raise ValueError("VAD ranges must be time-monotonic")
            previous_vad_start = max(previous_vad_start, start_ms)
            previous_vad_end = max(previous_vad_end, end_ms)


@dataclass(frozen=True, slots=True)
class AlignmentRange:
    start_ms: float
    end_ms: float
    coverage: float
    mean_model_confidence: float | None
    minimum_model_confidence: float | None
    low_confidence_ratio: float | None
    timestamp_precision: str
    ambiguity_margin: float
    timestamp_valid: bool
    model_confidence_coverage: float
    vad_start_ms: float | None = None
    vad_end_ms: float | None = None

    def __getitem__(self, index: int | slice) -> object:
        return (self.start_ms, self.end_ms, self.coverage)[index]


@dataclass(frozen=True, slots=True)
class RecognizedToken:
    """One ASR character with an absolute millisecond interval."""

    text: str
    start_ms: float
    end_ms: float
    confidence: float = 1.0
    confidence_available: bool = False
    timestamp_precision: str = TIMESTAMP_PRECISION_UNKNOWN

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("recognized token text must not be empty")
        if (
            not math.isfinite(self.start_ms)
            or not math.isfinite(self.end_ms)
            or self.start_ms < 0
            or self.end_ms <= self.start_ms
        ):
            raise ValueError("invalid recognized token interval")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("recognized token confidence must be between zero and one")
        if not isinstance(self.confidence_available, bool):
            raise ValueError("recognized token confidence availability must be boolean")
        if self.timestamp_precision not in _PRECISION_RANK:
            raise ValueError("invalid recognized token timestamp precision")


@runtime_checkable
class ForceAligner(Protocol):
    """Replaceable forced-aligner interface used by :func:`align_transcript`."""

    def align(
        self,
        *,
        audio_path: str | Path,
        transcript: str,
        window_start_ms: int,
        window_end_ms: int,
    ) -> AlignmentTrack:
        """Align *transcript* and return absolute-millisecond spans."""


@runtime_checkable
class AsrAligner(Protocol):
    """Replaceable ASR timestamp interface used as an independent check."""

    def align(
        self,
        *,
        audio_path: str | Path,
        transcript: str,
        window_start_ms: int,
        window_end_ms: int,
    ) -> AlignmentTrack:
        """Recognize the window and map its timestamps onto *transcript*."""


@dataclass(slots=True)
class AlignmentCandidate:
    paragraph_index: int
    highlight_index: int
    highlighted_text: str
    proposed_start_sample: int
    proposed_end_sample: int
    confidence: float
    reasons: list[str] = field(default_factory=list)
    requires_review: bool = True
    status: str = STATUS_NEEDS_REVIEW
    diagnostics: dict[str, object] = field(default_factory=dict)
    left_guard_sample: int | None = None
    right_guard_sample: int | None = None
    speech_start_sample: int | None = None
    speech_end_sample: int | None = None

    # Project/UI code historically uses "suggested" and "text".  Keeping
    # aliases here avoids duplicating an otherwise identical transport type.
    @property
    def suggested_start_sample(self) -> int:
        return self.proposed_start_sample

    @property
    def suggested_end_sample(self) -> int:
        return self.proposed_end_sample

    @property
    def text(self) -> str:
        return self.highlighted_text


def _get(value: object, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if default is not _MISSING:
        return default
    raise AttributeError(f"missing required field: {'/'.join(names)}")


@dataclass(frozen=True, slots=True)
class _ObservedCharacter:
    text: str
    start_ms: float
    end_ms: float
    model_confidence: float | None
    timestamp_precision: str
    segment_index: int


def _probability(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def _token_text(value: object) -> str:
    if isinstance(value, Mapping):
        return str(
            value.get("text", value.get("token", value.get("word", value.get("value", ""))))
        )
    return str(value)


def _token_confidence(value: object) -> float | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("confidence", "score", "probability", "prob"):
        if key in value and (confidence := _probability(value[key])) is not None:
            return confidence
    return None


def _item_confidences(item: Mapping[str, Any], count: int) -> list[float | None]:
    for key in (
        "token_confidence",
        "token_confidences",
        "confidences",
        "scores",
        "confidence",
        "score",
    ):
        value = item.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            confidences = [_probability(part) for part in value]
            if len(confidences) == count:
                return confidences
        elif count == 1 and (confidence := _probability(value)) is not None:
            # A sentence-level scalar must not be replicated and presented as
            # independent character confidence.  Only a one-token result makes
            # this value character-local.
            return [confidence]
    return [None] * count


def _expand_timed_unit(
    token: str,
    start_ms: float,
    end_ms: float,
    *,
    model_confidence: float | None,
    timestamp_precision: str,
    segment_index: int,
) -> list[_ObservedCharacter]:
    normalized = normalize_with_mapping(_TAG_RE.sub("", token)).text
    if not normalized:
        return []
    precision = (
        TIMESTAMP_PRECISION_CHARACTER
        if len(normalized) == 1 and timestamp_precision == TIMESTAMP_PRECISION_CHARACTER
        else timestamp_precision
    )
    character_confidence = model_confidence if len(normalized) == 1 else None
    # A word/segment timestamp is deliberately attached to every contained
    # character as one shared interval.  Never split it evenly and present the
    # resulting boundaries as real character timestamps.
    return [
        _ObservedCharacter(
            character,
            start_ms,
            end_ms,
            character_confidence,
            precision,
            segment_index,
        )
        for character in normalized
    ]


def _numeric_pair(value: object) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        try:
            return float(_get(value, "start", "start_ms", "begin")), float(
                _get(value, "end", "end_ms", "stop")
            )
        except (AttributeError, TypeError, ValueError):
            return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    numbers: list[float] = []
    for part in value:
        if isinstance(part, bool):
            continue
        try:
            numbers.append(float(part))
        except (TypeError, ValueError):
            continue
    if len(numbers) < 2:
        return None
    return numbers[-2], numbers[-1]


def _result_dicts(result: object) -> list[Mapping[str, Any]]:
    if isinstance(result, Mapping):
        return [result]
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        dictionaries: list[Mapping[str, Any]] = []
        for item in result:
            dictionaries.extend(_result_dicts(item))
        return dictionaries
    return []


def _validated_timestamp_pairs(
    raw_timestamps: object,
    *,
    multiplier: float,
    window_start_ms: float,
    window_end_ms: float | None,
) -> tuple[list[tuple[int, float, float]], list[str]]:
    if not isinstance(raw_timestamps, Sequence) or isinstance(raw_timestamps, (str, bytes)):
        return [], []
    pairs: list[tuple[int, float, float]] = []
    diagnostics: list[str] = []
    previous_start = previous_end = -1.0
    maximum_relative = (
        None if window_end_ms is None else max(0.0, window_end_ms - window_start_ms)
    )
    for index, raw in enumerate(raw_timestamps):
        pair = _numeric_pair(raw)
        if pair is None:
            diagnostics.append(f"timestamp_pair_invalid:{index}")
            continue
        relative_start = pair[0] * multiplier
        relative_end = pair[1] * multiplier
        if (
            not math.isfinite(relative_start)
            or not math.isfinite(relative_end)
            or relative_start < -_MODEL_WINDOW_BOUNDARY_TOLERANCE_MS
            or relative_end - relative_start < _MIN_TIMESTAMP_SPAN_MS
        ):
            diagnostics.append(f"timestamp_interval_invalid:{index}")
            continue
        if (
            maximum_relative is not None
            and relative_end
            > maximum_relative + _MODEL_WINDOW_BOUNDARY_TOLERANCE_MS
        ):
            diagnostics.append(f"timestamp_outside_window:{index}")
            continue
        relative_start = max(0.0, relative_start)
        if maximum_relative is not None:
            relative_end = min(maximum_relative, relative_end)
        if relative_end - relative_start < _MIN_TIMESTAMP_SPAN_MS:
            diagnostics.append(f"timestamp_collapsed_after_clamp:{index}")
            continue
        if (
            relative_start + _TIMESTAMP_EPSILON_MS < previous_start
            or relative_end + _TIMESTAMP_EPSILON_MS < previous_end
        ):
            diagnostics.append(f"timestamp_not_monotonic:{index}")
            continue
        absolute_start = relative_start + window_start_ms
        absolute_end = relative_end + window_start_ms
        if absolute_end <= absolute_start:
            diagnostics.append(f"timestamp_collapsed_after_clamp:{index}")
            continue
        pairs.append((index, absolute_start, absolute_end))
        previous_start = max(previous_start, relative_start)
        previous_end = max(previous_end, relative_end)
    return pairs, diagnostics


def voice_ranges_from_model_result(
    result: object,
    *,
    time_offset_ms: float = 0.0,
    timestamps_in_seconds: bool = False,
    window_end_ms: float | None = None,
) -> tuple[tuple[float, float], ...]:
    """Extract validated, absolute VAD/sentence ranges from a FunASR result."""

    if (
        isinstance(time_offset_ms, bool)
        or not math.isfinite(float(time_offset_ms))
        or time_offset_ms < 0
        or (
            window_end_ms is not None
            and (
                isinstance(window_end_ms, bool)
                or not math.isfinite(float(window_end_ms))
                or window_end_ms <= time_offset_ms
            )
        )
    ):
        raise ValueError("invalid model timestamp window")
    if not isinstance(timestamps_in_seconds, bool):
        raise ValueError("timestamps_in_seconds must be boolean")
    multiplier = 1000.0 if timestamps_in_seconds else 1.0
    ranges: list[tuple[float, float]] = []
    for item in _result_dicts(result):
        for key in ("value", "segments", "vad", "sentence_info"):
            raw_ranges = item.get(key)
            validated, _diagnostics = _validated_timestamp_pairs(
                raw_ranges,
                multiplier=multiplier,
                window_start_ms=time_offset_ms,
                window_end_ms=window_end_ms,
            )
            ranges.extend((start_ms, end_ms) for _index, start_ms, end_ms in validated)
    ranges.sort()
    merged: list[tuple[float, float]] = []
    for start_ms, end_ms in ranges:
        if merged and start_ms <= merged[-1][1] + _TIMESTAMP_EPSILON_MS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ms))
        else:
            merged.append((start_ms, end_ms))
    return tuple(merged)


def _segment_index_for_interval(
    start_ms: float,
    end_ms: float,
    vad_ranges: Sequence[tuple[float, float]],
    fallback_segment: int,
) -> int:
    midpoint = (start_ms + end_ms) / 2
    for index, (vad_start, vad_end) in enumerate(vad_ranges):
        if vad_start - _TIMESTAMP_EPSILON_MS <= midpoint <= vad_end + _TIMESTAMP_EPSILON_MS:
            return index
    return fallback_segment


def _fa_serialized_tokens(text: str) -> list[str]:
    """Extract non-silence tokens from fa-zh's ``token start end;`` text."""

    tokens: list[str] = []
    for record in text.split(";"):
        fields = record.strip().rsplit(maxsplit=2)
        if len(fields) != 3:
            continue
        token, start, end = fields
        try:
            float(start)
            float(end)
        except ValueError:
            continue
        if token.strip():
            tokens.append(token)
    return tokens


def _observed_characters(
    result: object,
    reference: NormalizedText,
    *,
    time_offset_ms: float,
    timestamps_in_seconds: bool,
    forced_reference: bool = False,
    window_end_ms: float | None = None,
) -> tuple[
    list[_ObservedCharacter],
    tuple[str, ...],
    bool,
    tuple[tuple[float, float], ...],
]:
    observed: list[_ObservedCharacter] = []
    diagnostics: list[str] = []
    multiplier = 1000.0 if timestamps_in_seconds else 1.0
    vad_ranges = voice_ranges_from_model_result(
        result,
        time_offset_ms=time_offset_ms,
        timestamps_in_seconds=timestamps_in_seconds,
        window_end_ms=window_end_ms,
    )
    fallback_segment = 0
    previous_unit_end: float | None = None

    def append_unit(
        token: str,
        start_ms: float,
        end_ms: float,
        confidence: float | None,
        precision: str,
    ) -> None:
        nonlocal fallback_segment, previous_unit_end
        if (
            not vad_ranges
            and previous_unit_end is not None
            and start_ms - previous_unit_end >= _VAD_GAP_MS
        ):
            fallback_segment += 1
        segment_index = _segment_index_for_interval(
            start_ms,
            end_ms,
            vad_ranges,
            fallback_segment,
        )
        observed.extend(
            _expand_timed_unit(
                token,
                start_ms,
                end_ms,
                model_confidence=confidence,
                timestamp_precision=precision,
                segment_index=segment_index,
            )
        )
        previous_unit_end = end_ms if previous_unit_end is None else max(previous_unit_end, end_ms)

    for item in _result_dicts(result):
        raw_timestamps = item.get("timestamp", item.get("timestamps", item.get("time_stamp")))
        if raw_timestamps is None:
            diagnostics.append("timestamps_missing")
        pairs, pair_diagnostics = _validated_timestamp_pairs(
            raw_timestamps,
            multiplier=multiplier,
            window_start_ms=time_offset_ms,
            window_end_ms=window_end_ms,
        )
        diagnostics.extend(pair_diagnostics)
        if not pairs:
            # Some FunASR versions expose only segment-level information. When
            # character timestamps are present they take precedence; consuming
            # both shapes would duplicate the recognized text and corrupt the
            # reference mapping.
            sentence_info = item.get("sentence_info")
            if isinstance(sentence_info, Sequence) and not isinstance(
                sentence_info, (str, bytes)
            ):
                sentence_pairs, sentence_diagnostics = _validated_timestamp_pairs(
                    sentence_info,
                    multiplier=multiplier,
                    window_start_ms=time_offset_ms,
                    window_end_ms=window_end_ms,
                )
                diagnostics.extend(sentence_diagnostics)
                for sentence_index, start_ms, end_ms in sentence_pairs:
                    sentence = sentence_info[sentence_index]
                    if not isinstance(sentence, Mapping):
                        continue
                    text = str(sentence.get("text", ""))
                    confidence = next(
                        (
                            value
                            for key in ("confidence", "score", "probability", "prob")
                            if (value := _probability(sentence.get(key))) is not None
                        ),
                        None,
                    )
                    append_unit(
                        text,
                        start_ms,
                        end_ms,
                        confidence,
                        TIMESTAMP_PRECISION_SEGMENT,
                    )
            continue

        raw_tokens = item.get("tokens", item.get("token"))
        text = str(item.get("text", item.get("sentence", item.get("transcript", ""))))
        serialized_tokens = _fa_serialized_tokens(text) if forced_reference else []
        raw_pair_count = (
            len(raw_timestamps)
            if isinstance(raw_timestamps, Sequence)
            and not isinstance(raw_timestamps, (str, bytes))
            else 0
        )
        confidences = _item_confidences(item, raw_pair_count)
        token_confidences: list[float | None] = [None] * raw_pair_count
        if forced_reference and raw_pair_count == len(reference.text):
            # fa-zh returns authoritative token timestamps but serializes its
            # human-readable ``text`` as ``token start end;`` records.  That
            # field must not be normalized as recognized prose: its timing
            # digits otherwise create false alignment coverage and move
            # every boundary.  The model contract states that ``timestamp``
            # excludes silence and is aligned one-to-one with input tokens.
            tokens = list(reference.text)
            precisions = [TIMESTAMP_PRECISION_CHARACTER] * raw_pair_count
        elif forced_reference and len(serialized_tokens) == raw_pair_count:
            # Tokenization can legitimately group Latin words or multiple
            # characters.  In that case parse the fa-zh serialization and let
            # the normal monotonic matcher map those real tokens.
            tokens = serialized_tokens
            precisions = [
                TIMESTAMP_PRECISION_CHARACTER
                if len(normalize_with_mapping(token).text) == 1
                else TIMESTAMP_PRECISION_TOKEN
                for token in tokens
            ]
        elif isinstance(raw_tokens, Sequence) and not isinstance(raw_tokens, (str, bytes)):
            tokens = [_token_text(token) for token in raw_tokens]
            token_confidences = [_token_confidence(token) for token in raw_tokens]
            token_confidences.extend([None] * max(0, raw_pair_count - len(token_confidences)))
            precisions = [
                TIMESTAMP_PRECISION_CHARACTER
                if len(normalize_with_mapping(token).text) == 1
                else TIMESTAMP_PRECISION_TOKEN
                for token in tokens
            ]
        else:
            whitespace_tokens = _TAG_RE.sub("", text).split()
            if len(whitespace_tokens) == raw_pair_count:
                tokens = whitespace_tokens
                precisions = [
                    TIMESTAMP_PRECISION_CHARACTER
                    if len(normalize_with_mapping(token).text) == 1
                    else TIMESTAMP_PRECISION_TOKEN
                    for token in tokens
                ]
            else:
                normalized_output = normalize_with_mapping(_TAG_RE.sub("", text)).text
                if not normalized_output and forced_reference:
                    normalized_output = reference.text
                if not normalized_output:
                    # An ASR result without recognized text is not evidence
                    # that the supplied DOCX reference was spoken.  In
                    # particular, never manufacture coverage merely because a
                    # backend happened to return timestamp pairs.
                    continue
                if len(normalized_output) == raw_pair_count:
                    tokens = list(normalized_output)
                    precisions = [TIMESTAMP_PRECISION_CHARACTER] * raw_pair_count
                else:
                    # Last-resort grouping for wordpiece/English results whose
                    # token list was not returned by this FunASR version.
                    tokens = []
                    for index in range(raw_pair_count):
                        left = round(index * len(normalized_output) / raw_pair_count)
                        right = round((index + 1) * len(normalized_output) / raw_pair_count)
                        tokens.append(normalized_output[left:right])
                    precisions = [TIMESTAMP_PRECISION_TOKEN] * raw_pair_count

        for pair_index, start_ms, end_ms in pairs:
            if pair_index >= len(tokens):
                diagnostics.append(f"timestamp_without_token:{pair_index}")
                continue
            confidence = (
                token_confidences[pair_index]
                if pair_index < len(token_confidences)
                and token_confidences[pair_index] is not None
                else confidences[pair_index]
            )
            append_unit(
                tokens[pair_index],
                start_ms,
                end_ms,
                confidence,
                precisions[pair_index],
            )
    if not observed and not diagnostics:
        diagnostics.append("timestamped_tokens_unavailable")
    return observed, tuple(dict.fromkeys(diagnostics)), not diagnostics, vad_ranges


def _alignment_track_from_observed(
    observed: Sequence[_ObservedCharacter],
    transcript: str,
    *,
    engine: str,
    forced_reference: bool,
    timestamp_diagnostics: Sequence[str],
    timestamp_valid: bool,
    vad_ranges: Sequence[tuple[float, float]],
    timestamp_unit: str,
) -> AlignmentTrack:
    reference = normalize_with_mapping(transcript)
    if not reference.text:
        return AlignmentTrack((), engine, 0.0)
    observed_text = "".join(character.text for character in observed)
    alignment = _monotonic_character_alignment(
        reference.text,
        observed_text,
        allow_phonetic_fallback=not forced_reference,
        reference_segments=_reference_segment_indexes(transcript, reference),
        observed_segments=tuple(character.segment_index for character in observed),
    )
    spans: list[TimedSpan] = []
    covered: set[int] = set()
    for normalized_index, observed_index, confidence in alignment.pairs:
        character = observed[observed_index]
        spans.append(
            TimedSpan(
                normalized_index,
                normalized_index + 1,
                character.start_ms,
                character.end_ms,
                confidence,
                character.model_confidence,
                character.timestamp_precision,
                alignment.margins.get(normalized_index, 1.0),
            )
        )
        covered.add(normalized_index)
    spans.sort(key=lambda span: (span.normalized_start, span.start_ms))
    model_confidences = [
        span.model_confidence for span in spans if span.model_confidence is not None
    ]
    precision = max(
        (span.timestamp_precision for span in spans),
        key=lambda value: _PRECISION_RANK[value],
        default=TIMESTAMP_PRECISION_UNKNOWN,
    )
    diagnostics = list(timestamp_diagnostics)
    diagnostics.append(f"timestamp_unit:{timestamp_unit}")
    diagnostics.append("matching_strategy:anchored_monotonic_dp")
    if alignment.forced_partition:
        diagnostics.append("dp_forced_partition")
    if not vad_ranges and not forced_reference:
        diagnostics.append("vad_ranges_unavailable")
    return AlignmentTrack(
        tuple(spans),
        engine,
        len(covered) / len(reference.text),
        sum(model_confidences) / len(model_confidences) if model_confidences else None,
        min(model_confidences) if model_confidences else None,
        (
            sum(value < _LOW_MODEL_CONFIDENCE_THRESHOLD for value in model_confidences)
            / len(model_confidences)
            if model_confidences
            else None
        ),
        precision,
        timestamp_valid,
        alignment.ambiguity_margin,
        tuple(dict.fromkeys(diagnostics)),
        vad_ranges,
        len(model_confidences) / len(spans) if spans else 0.0,
    )


def alignment_track_from_model_result(
    result: object,
    transcript: str,
    *,
    engine: str,
    time_offset_ms: float = 0.0,
    timestamps_in_seconds: bool = False,
    forced_reference: bool = False,
    window_end_ms: float | None = None,
) -> AlignmentTrack:
    """Convert common FunASR result shapes into normalized-reference spans."""

    if (
        isinstance(time_offset_ms, bool)
        or not math.isfinite(float(time_offset_ms))
        or time_offset_ms < 0
        or (
            window_end_ms is not None
            and (
                isinstance(window_end_ms, bool)
                or not math.isfinite(float(window_end_ms))
                or window_end_ms <= time_offset_ms
            )
        )
    ):
        raise ValueError("invalid model timestamp window")
    if not isinstance(timestamps_in_seconds, bool):
        raise ValueError("timestamps_in_seconds must be boolean")
    reference = normalize_with_mapping(transcript)
    if not reference.text:
        return AlignmentTrack((), engine, 0.0)
    observed, timestamp_diagnostics, timestamp_valid, vad_ranges = _observed_characters(
        result,
        reference,
        time_offset_ms=time_offset_ms,
        timestamps_in_seconds=timestamps_in_seconds,
        forced_reference=forced_reference,
        window_end_ms=window_end_ms,
    )
    return _alignment_track_from_observed(
        observed,
        transcript,
        engine=engine,
        forced_reference=forced_reference,
        timestamp_diagnostics=timestamp_diagnostics,
        timestamp_valid=timestamp_valid,
        vad_ranges=vad_ranges,
        timestamp_unit="seconds" if timestamps_in_seconds else "milliseconds",
    )


def alignment_track_from_recognition_tokens(
    tokens: Sequence[RecognizedToken],
    transcript: str,
    *,
    engine: str,
    vad_ranges: Sequence[tuple[float, float]] = (),
) -> AlignmentTrack:
    """Map an already streamed token timeline onto one reference transcript."""

    observed: list[_ObservedCharacter] = []
    fallback_segment = 0
    previous_end: float | None = None
    normalized_ranges = tuple(
        sorted(
            (float(start_ms), float(end_ms))
            for start_ms, end_ms in vad_ranges
            if math.isfinite(start_ms)
            and math.isfinite(end_ms)
            and start_ms >= 0
            and end_ms > start_ms
        )
    )
    for token in tokens:
        if previous_end is not None and token.start_ms - previous_end >= _VAD_GAP_MS:
            fallback_segment += 1
        segment_index = _segment_index_for_interval(
            token.start_ms,
            token.end_ms,
            normalized_ranges,
            fallback_segment,
        )
        observed.extend(
            _expand_timed_unit(
                token.text,
                token.start_ms,
                token.end_ms,
                model_confidence=(
                    token.confidence if token.confidence_available else None
                ),
                timestamp_precision=token.timestamp_precision,
                segment_index=segment_index,
            )
        )
        previous_end = (
            token.end_ms if previous_end is None else max(previous_end, token.end_ms)
        )
    return _alignment_track_from_observed(
        observed,
        transcript,
        engine=engine,
        forced_reference=False,
        timestamp_diagnostics=(),
        timestamp_valid=True,
        vad_ranges=normalized_ranges,
        timestamp_unit="milliseconds",
    )


def recognition_tokens_from_model_result(
    result: object,
    *,
    time_offset_ms: float = 0.0,
    timestamps_in_seconds: bool = False,
    window_end_ms: float | None = None,
) -> tuple[RecognizedToken, ...]:
    """Convert a raw FunASR result into timestamped display characters."""

    observed, _diagnostics, _timestamp_valid, _vad_ranges = _observed_characters(
        result,
        normalize_with_mapping(""),
        time_offset_ms=time_offset_ms,
        timestamps_in_seconds=timestamps_in_seconds,
        window_end_ms=window_end_ms,
    )
    tokens = [
        RecognizedToken(
            character.text,
            character.start_ms,
            character.end_ms,
            character.model_confidence if character.model_confidence is not None else 0.0,
            character.model_confidence is not None,
            character.timestamp_precision,
        )
        for character in observed
    ]
    # Preserve the model's transcript order.  Adjacent FunASR VAD slices may
    # overlap by a few milliseconds; sorting those characters by timestamp can
    # silently swap the spoken text at a slice boundary.  The standalone audio
    # workspace normalizes the small overlap while keeping this semantic order.
    return tuple(tokens)


def _refinement_groups(
    tokens: Sequence[RecognizedToken],
    *,
    gap_ms: float,
    max_window_span_ms: float,
) -> list[tuple[int, int]]:
    """Split tokens into locally continuous speech chunks for re-alignment."""

    groups: list[tuple[int, int]] = []
    group_start = 0
    group_first_start = tokens[0].start_ms
    previous_end = tokens[0].end_ms
    for index in range(1, len(tokens)):
        token = tokens[index]
        if (
            token.start_ms - previous_end >= gap_ms
            or token.end_ms - group_first_start > max_window_span_ms
        ):
            groups.append((group_start, index))
            group_start = index
            group_first_start = token.start_ms
        previous_end = max(previous_end, token.end_ms)
    groups.append((group_start, len(tokens)))
    return groups


def _group_already_character_precise(group: Sequence[RecognizedToken]) -> bool:
    """Return True when every token already has a distinct character interval."""

    if not group:
        return True
    if any(token.timestamp_precision != TIMESTAMP_PRECISION_CHARACTER for token in group):
        return False
    if len(group) < 2:
        return True
    # Shared identical intervals mean the character label is lying about
    # granularity, so a forced-alignment pass is still worthwhile.
    first = group[0]
    return not all(
        math.isclose(token.start_ms, first.start_ms)
        and math.isclose(token.end_ms, first.end_ms)
        for token in group[1:]
    )


def refine_recognition_tokens(
    tokens: Sequence[RecognizedToken],
    *,
    force_aligner: ForceAligner,
    audio_path: str | Path,
    audio_end_ms: float,
    window_padding_ms: int = 200,
    max_window_span_ms: float = 30_000.0,
    max_boundary_shift_ms: float = 350.0,
    minimum_coverage: float = 0.80,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel: object | None = None,
) -> tuple[tuple[RecognizedToken, ...], dict[str, object]]:
    """Sharpen shared word/segment intervals into per-character timestamps.

    A recognizer may emit only a coarse interval for a whole local sentence;
    every contained character then shares that interval, which makes a
    mid-token selection look badly misplaced.  This pass re-runs the selected
    local forced-alignment backend on each continuous speech chunk with the
    recognized text itself as the reference, yielding per-character boundaries
    without inventing any text.  A
    refined boundary is accepted only when the chunk was well covered, the new
    interval stays near the original one, and token order stays monotonic.
    """

    refinement_engine = (
        "qwen3-asr-mlx-forced-character-refinement"
        if type(force_aligner).__name__ == "QwenMlxForceAligner"
        else "fa-zh-character-refinement"
    )
    if not tokens:
        return (), {"engine": refinement_engine, "group_count": 0}
    if window_padding_ms < 0 or not math.isfinite(audio_end_ms) or audio_end_ms <= 0:
        raise ValueError("invalid refinement window parameters")
    if not math.isfinite(max_boundary_shift_ms) or max_boundary_shift_ms < 0:
        raise ValueError("max_boundary_shift_ms must be non-negative and finite")
    if not math.isfinite(minimum_coverage) or not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be between zero and one")
    groups = _refinement_groups(
        tokens,
        gap_ms=_VAD_GAP_MS,
        max_window_span_ms=max_window_span_ms,
    )
    refined: list[RecognizedToken] = list(tokens)
    refined_token_count = 0
    refined_group_count = 0
    failed_group_count = 0
    previous_start = -math.inf
    previous_end = -math.inf

    def accept(index: int, start_ms: float, end_ms: float, precision: str) -> bool:
        nonlocal previous_start, previous_end, refined_token_count
        original = tokens[index]
        if (
            end_ms <= start_ms
            or start_ms < original.start_ms - max_boundary_shift_ms
            or end_ms > original.end_ms + max_boundary_shift_ms
            or start_ms + _TIMESTAMP_EPSILON_MS < previous_start
            or end_ms + _TIMESTAMP_EPSILON_MS < previous_end
        ):
            return False
        refined[index] = RecognizedToken(
            original.text,
            start_ms,
            end_ms,
            original.confidence,
            original.confidence_available,
            precision,
        )
        previous_start = max(previous_start, start_ms)
        previous_end = max(previous_end, end_ms)
        refined_token_count += 1
        return True

    if progress_cb is not None:
        progress_cb(0, len(groups))
    skipped_precise_group_count = 0
    ffmpeg_path = getattr(force_aligner, "ffmpeg_path", None)
    stack = ExitStack()
    window_cache: PreparedAudioWindowCache | None = None
    if Path(audio_path).expanduser().is_file():
        window_cache = stack.enter_context(
            PreparedAudioWindowCache(
                audio_path,
                ffmpeg_path=ffmpeg_path,
                cancel=cancel,
            )
        )
    try:
      for group_index, (left, right) in enumerate(groups):
        if _is_cancelled(cancel):
            raise AlignmentCancelledError("token refinement was cancelled")
        group = tokens[left:right]
        offsets: list[int] = []
        lengths: list[int] = []
        base = 0
        for token in group:
            length = len(normalize_with_mapping(token.text).text)
            offsets.append(base)
            lengths.append(length)
            base += length
        keep_group_originals = True
        if base > 0 and _group_already_character_precise(group):
            # paraformer sometimes already emits true per-character intervals.
            # Re-running fa-zh on those chunks only burns CPU.
            skipped_precise_group_count += 1
            previous_start = max(previous_start, *(token.start_ms for token in group))
            previous_end = max(previous_end, *(token.end_ms for token in group))
            if progress_cb is not None:
                progress_cb(group_index + 1, len(groups))
            continue
        if base > 0:
            window_start = max(0, int(min(token.start_ms for token in group)) - window_padding_ms)
            window_end = min(
                int(math.ceil(audio_end_ms)),
                int(math.ceil(max(token.end_ms for token in group))) + window_padding_ms,
            )
            track = (
                _invoke_aligner(
                    force_aligner,
                    engine=refinement_engine,
                    audio_path=audio_path,
                    transcript="".join(token.text for token in group),
                    window_start_ms=window_start,
                    window_end_ms=window_end,
                    reference_length=base,
                    window_cache=window_cache,
                )
                if window_end > window_start
                else None
            )
            if (
                track is not None
                and track.timestamp_valid
                and track.coverage >= minimum_coverage
            ):
                spans_by_index: dict[int, TimedSpan] = {}
                for span in track.spans:
                    for index in range(span.normalized_start, span.normalized_end):
                        spans_by_index.setdefault(index, span)
                keep_group_originals = False
                refined_group_count += 1
                for token_offset, token in enumerate(group):
                    index = left + token_offset
                    covering = [
                        spans_by_index[position]
                        for position in range(
                            offsets[token_offset],
                            offsets[token_offset] + lengths[token_offset],
                        )
                        if position in spans_by_index
                    ]
                    accepted = False
                    if len(covering) == lengths[token_offset] and covering:
                        precision = (
                            TIMESTAMP_PRECISION_CHARACTER
                            if lengths[token_offset] == 1
                            and all(
                                span.timestamp_precision == TIMESTAMP_PRECISION_CHARACTER
                                for span in covering
                            )
                            else max(
                                (span.timestamp_precision for span in covering),
                                key=lambda value: _PRECISION_RANK[value],
                            )
                        )
                        accepted = accept(
                            index,
                            min(span.start_ms for span in covering),
                            max(span.end_ms for span in covering),
                            precision,
                        )
                    if not accepted:
                        previous_start = max(previous_start, token.start_ms)
                        previous_end = max(previous_end, token.end_ms)
            else:
                failed_group_count += 1
        if keep_group_originals:
            previous_start = max(previous_start, *(token.start_ms for token in group))
            previous_end = max(previous_end, *(token.end_ms for token in group))
        if progress_cb is not None:
            progress_cb(group_index + 1, len(groups))
    finally:
        stack.close()
    diagnostics: dict[str, object] = {
        "engine": refinement_engine,
        "group_count": len(groups),
        "refined_group_count": refined_group_count,
        "failed_group_count": failed_group_count,
        "skipped_precise_group_count": skipped_precise_group_count,
        "refined_token_count": refined_token_count,
        "unchanged_token_count": len(tokens) - refined_token_count,
        "window_padding_ms": window_padding_ms,
        "max_boundary_shift_ms": max_boundary_shift_ms,
        "minimum_coverage": minimum_coverage,
    }
    return tuple(refined), diagnostics


class FunASRForceAligner:
    """Local ``fa-zh`` adapter.  The model is loaded lazily and never downloaded."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        ffmpeg_path: str | Path | None = None,
        device: str | None = None,
        model_factory: Any | None = None,
        timestamps_in_seconds: bool = False,
    ) -> None:
        self.model_path = require_local_model(model_path, "fa-zh")
        self.ffmpeg_path = Path(ffmpeg_path).resolve() if ffmpeg_path is not None else None
        self.device = resolve_inference_device(device)
        self.model_factory = model_factory
        self.timestamps_in_seconds = timestamps_in_seconds
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            model = load_funasr_model(
                model_path=self.model_path,
                label="fa-zh",
                device=self.device,
                model_factory=self.model_factory,
            )
            self.device = model_inference_device(model, self.device)
            self._model = model
        return self._model

    def _generate(self, **kwargs: Any) -> object:
        try:
            with model_execution_guard():
                return self._get_model().generate(**kwargs)
        except Exception as accelerator_error:
            if self.model_factory is not None or not self.device.casefold().startswith("mps"):
                raise
            _LOGGER.warning(
                "fa-zh MPS inference failed; retrying this session on CPU: %s",
                accelerator_error,
            )
            if self._model is not None:
                evict_funasr_model(self._model)
            self.device = "cpu"
            self._model = None
            with model_execution_guard():
                return self._get_model().generate(**kwargs)

    def align(
        self,
        *,
        audio_path: str | Path,
        transcript: str,
        window_start_ms: int,
        window_end_ms: int,
    ) -> AlignmentTrack:
        with local_audio_window(
            audio_path,
            start_ms=window_start_ms,
            end_ms=window_end_ms,
            ffmpeg_path=self.ffmpeg_path,
        ) as window:
            return self.align_prepared(
                audio_path=window,
                transcript=transcript,
                time_offset_ms=window_start_ms,
                window_end_ms=window_end_ms,
            )

    def align_prepared(
        self,
        *,
        audio_path: str | Path,
        transcript: str,
        time_offset_ms: int,
        window_end_ms: int,
    ) -> AlignmentTrack:
        """Force-align against an already materialized local audio window."""

        prepared = Path(audio_path).expanduser()
        prepared_path = prepared.resolve() if prepared.exists() else prepared
        try:
            result = self._generate(
                input=(str(prepared_path), transcript),
                data_type=("sound", "text"),
                disable_pbar=True,
                disable_log=True,
            )
        except Exception as exc:
            raise ModelUnavailableError(f"fa-zh 本地推理失败: {exc}") from exc
        return alignment_track_from_model_result(
            result,
            transcript,
            engine="fa-zh",
            time_offset_ms=time_offset_ms,
            timestamps_in_seconds=self.timestamps_in_seconds,
            forced_reference=True,
            window_end_ms=window_end_ms,
        )


class FunASRAsrAligner:
    """Local ``paraformer-zh`` plus ``fsmn-vad`` timestamp adapter."""

    def __init__(
        self,
        model_path: str | Path,
        vad_model_path: str | Path,
        *,
        ffmpeg_path: str | Path | None = None,
        device: str | None = None,
        model_factory: Any | None = None,
        timestamps_in_seconds: bool = False,
        max_vad_segment_ms: int = 30_000,
    ) -> None:
        self.model_path = require_local_model(model_path, "paraformer-zh")
        self.vad_model_path = require_local_model(vad_model_path, "fsmn-vad")
        self.ffmpeg_path = Path(ffmpeg_path).resolve() if ffmpeg_path is not None else None
        self.device = resolve_inference_device(device)
        self.model_factory = model_factory
        self.timestamps_in_seconds = timestamps_in_seconds
        self.max_vad_segment_ms = max(1_000, int(max_vad_segment_ms))
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            model = load_funasr_model(
                model_path=self.model_path,
                vad_model_path=self.vad_model_path,
                label="paraformer-zh",
                device=self.device,
                model_factory=self.model_factory,
                extra_options={
                    # FunASR's own current CLI uses 30 s VAD chunks.  Keeping
                    # long-form recognition in bounded speech segments reduces
                    # timestamp drift and memory use without consulting Word
                    # anchors.
                    "vad_kwargs": {
                        "max_single_segment_time": self.max_vad_segment_ms,
                    }
                },
            )
            self.device = model_inference_device(model, self.device)
            self._model = model
        return self._model

    def _generate(self, **kwargs: Any) -> object:
        try:
            with model_execution_guard():
                return self._get_model().generate(**kwargs)
        except Exception as accelerator_error:
            if self.model_factory is not None or not self.device.casefold().startswith("mps"):
                raise
            _LOGGER.warning(
                "paraformer-zh MPS inference failed; retrying this session on CPU: %s",
                accelerator_error,
            )
            if self._model is not None:
                evict_funasr_model(self._model)
            self.device = "cpu"
            self._model = None
            with model_execution_guard():
                return self._get_model().generate(**kwargs)

    def preload(self) -> None:
        """Load the local ASR/VAD pair without running an inference."""

        self._get_model()

    def warmup(self) -> None:
        """Load the model and compile its first real inference path while idle."""

        self.preload()
        example = self.model_path / "example" / "asr_example.wav"
        if not example.is_file():
            return
        self._generate(
            input=str(example),
            batch_size_s=300,
            use_itn=False,
            pred_timestamp=True,
            disable_pbar=True,
            disable_log=True,
        )

    def align(
        self,
        *,
        audio_path: str | Path,
        transcript: str,
        window_start_ms: int,
        window_end_ms: int,
    ) -> AlignmentTrack:
        with local_audio_window(
            audio_path,
            start_ms=window_start_ms,
            end_ms=window_end_ms,
            ffmpeg_path=self.ffmpeg_path,
        ) as window:
            return self.align_prepared(
                audio_path=window,
                transcript=transcript,
                time_offset_ms=window_start_ms,
                window_end_ms=window_end_ms,
            )

    def align_prepared(
        self,
        *,
        audio_path: str | Path,
        transcript: str,
        time_offset_ms: int,
        window_end_ms: int,
    ) -> AlignmentTrack:
        """Map timestamps from an already materialized local audio window."""

        prepared = Path(audio_path).expanduser()
        prepared_path = prepared.resolve() if prepared.exists() else prepared
        try:
            result = self._generate(
                input=str(prepared_path),
                batch_size_s=300,
                use_itn=False,
                pred_timestamp=True,
                disable_pbar=True,
                disable_log=True,
            )
        except Exception as exc:
            raise ModelUnavailableError(f"paraformer-zh 本地推理失败: {exc}") from exc
        return alignment_track_from_model_result(
            result,
            transcript,
            engine="paraformer-zh+fsmn-vad",
            time_offset_ms=time_offset_ms,
            timestamps_in_seconds=self.timestamps_in_seconds,
            window_end_ms=window_end_ms,
        )

    def recognize(
        self,
        *,
        audio_path: str | Path,
        window_start_ms: int,
        window_end_ms: int,
    ) -> tuple[RecognizedToken, ...]:
        """Recognize one audio window without requiring a reference transcript."""

        tokens, _result = self.recognize_with_result(
            audio_path=audio_path,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
        )
        return tokens

    def recognize_with_result(
        self,
        *,
        audio_path: str | Path,
        window_start_ms: int,
        window_end_ms: int,
    ) -> tuple[tuple[RecognizedToken, ...], object]:
        """Return display tokens together with raw local model segmentation metadata."""

        with local_audio_window(
            audio_path,
            start_ms=window_start_ms,
            end_ms=window_end_ms,
            ffmpeg_path=self.ffmpeg_path,
        ) as window:
            return self.recognize_prepared_with_result(
                audio_path=window,
                time_offset_ms=window_start_ms,
                window_end_ms=window_end_ms,
            )

    def recognize_prepared_with_result(
        self,
        *,
        audio_path: str | Path,
        time_offset_ms: int,
        window_end_ms: int,
    ) -> tuple[tuple[RecognizedToken, ...], object]:
        """Recognize an already materialized local audio window.

        Progressive ASR has already decoded its bounded MP3/M4A window to a
        temporary WAV.  Accepting that exact file avoids copying the same PCM
        into a second temporary directory before every inference.
        """

        prepared = Path(audio_path).expanduser()
        prepared_path = prepared.resolve() if prepared.exists() else prepared
        try:
            result = self._generate(
                input=str(prepared_path),
                batch_size_s=300,
                use_itn=False,
                pred_timestamp=True,
                disable_pbar=True,
                disable_log=True,
            )
        except Exception as exc:
            raise ModelUnavailableError(f"paraformer-zh 本地转写失败: {exc}") from exc
        return (
            recognition_tokens_from_model_result(
                result,
                time_offset_ms=time_offset_ms,
                timestamps_in_seconds=self.timestamps_in_seconds,
                window_end_ms=window_end_ms,
            ),
            result,
        )


def _coerce_track(value: object, engine: str, reference_length: int) -> AlignmentTrack:
    if isinstance(value, AlignmentTrack):
        return value
    raw_spans = _get(value, "spans", default=value)
    if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, (str, bytes)):
        raise TypeError("aligner result must contain a sequence of spans")
    spans: list[TimedSpan] = []
    covered: set[int] = set()
    for raw in raw_spans:
        if isinstance(raw, TimedSpan):
            span = raw
        else:
            span = TimedSpan(
                int(_get(raw, "normalized_start", "start_char", "text_start")),
                int(_get(raw, "normalized_end", "end_char", "text_end")),
                float(_get(raw, "start_ms")),
                float(_get(raw, "end_ms")),
                float(_get(raw, "confidence", default=1.0)),
                (
                    float(model_confidence)
                    if (model_confidence := _get(raw, "model_confidence", default=None))
                    is not None
                    else None
                ),
                str(
                    _get(
                        raw,
                        "timestamp_precision",
                        default=TIMESTAMP_PRECISION_UNKNOWN,
                    )
                ),
                float(_get(raw, "alignment_margin", default=1.0)),
            )
        spans.append(span)
        covered.update(range(span.normalized_start, span.normalized_end))
    supplied_coverage = _get(value, "coverage", default=None)
    coverage = (
        float(supplied_coverage)
        if supplied_coverage is not None
        else min(1.0, len(covered) / max(1, reference_length))
    )
    model_confidences = [
        span.model_confidence for span in spans if span.model_confidence is not None
    ]
    precision = max(
        (span.timestamp_precision for span in spans),
        key=lambda item: _PRECISION_RANK[item],
        default=TIMESTAMP_PRECISION_UNKNOWN,
    )
    timestamp_valid = _get(value, "timestamp_valid", default=True)
    if not isinstance(timestamp_valid, bool):
        raise ValueError("timestamp_valid must be boolean")
    return AlignmentTrack(
        tuple(spans),
        str(_get(value, "engine", default=engine)),
        coverage,
        float(mean_confidence)
        if (mean_confidence := _get(value, "mean_model_confidence", default=None))
        is not None
        else (sum(model_confidences) / len(model_confidences) if model_confidences else None),
        float(minimum_confidence)
        if (minimum_confidence := _get(value, "minimum_model_confidence", default=None))
        is not None
        else (min(model_confidences) if model_confidences else None),
        float(low_ratio)
        if (low_ratio := _get(value, "low_confidence_ratio", default=None)) is not None
        else None,
        str(_get(value, "timestamp_precision", default=precision)),
        timestamp_valid,
        float(_get(value, "ambiguity_margin", default=1.0)),
        tuple(str(item) for item in _get(value, "diagnostics", default=())),
        tuple(
            (float(start), float(end))
            for start, end in _get(value, "vad_ranges", default=())
        ),
        float(
            _get(
                value,
                "model_confidence_coverage",
                default=(len(model_confidences) / len(spans) if spans else 0.0),
            )
        ),
    )


def _invoke_aligner(
    aligner: object | None,
    *,
    engine: str,
    audio_path: str | Path,
    transcript: str,
    window_start_ms: int,
    window_end_ms: int,
    reference_length: int,
    window_cache: PreparedAudioWindowCache | None = None,
) -> AlignmentTrack | None:
    if aligner is None:
        return None
    try:
        if window_cache is not None and hasattr(aligner, "align_prepared"):
            prepared = window_cache.get(window_start_ms, window_end_ms)
            result = aligner.align_prepared(
                audio_path=prepared,
                transcript=transcript,
                time_offset_ms=window_start_ms,
                window_end_ms=window_end_ms,
            )
        else:
            result = aligner.align(
                audio_path=audio_path,
                transcript=transcript,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
            )
        return _coerce_track(result, engine, reference_length)
    except ModelUnavailableError:
        # One corrupt paragraph or missing optional runtime must never result in
        # an unsafe automatic cut.  The caller records an explicit fallback.
        return None
    except (TypeError, ValueError, OverflowError) as exc:
        # Preserve a machine-readable rejection instead of silently degrading a
        # malformed timestamp result to an apparently healthy empty alignment.
        return AlignmentTrack(
            (),
            engine,
            0.0,
            timestamp_precision=TIMESTAMP_PRECISION_UNKNOWN,
            timestamp_valid=False,
            diagnostics=(f"aligner_result_rejected:{type(exc).__name__}",),
        )
    except Exception:
        return None


def _range_from_track(
    track: AlignmentTrack | None,
    normalized_start: int,
    normalized_end: int,
    *,
    cap_by_track_coverage: bool = True,
    max_internal_gap_ms: float | None = None,
) -> AlignmentRange | None:
    if track is None or normalized_end <= normalized_start:
        return None
    intervals: list[tuple[int, int, float, float]] = []
    selected_spans: list[TimedSpan] = []
    covered: set[int] = set()
    for span in track.spans:
        left = max(normalized_start, span.normalized_start)
        right = min(normalized_end, span.normalized_end)
        if right <= left:
            continue
        width = span.normalized_end - span.normalized_start
        duration = span.end_ms - span.start_ms
        relative_left = (left - span.normalized_start) / width
        relative_right = (right - span.normalized_start) / width
        intervals.append(
            (
                left,
                right,
                span.start_ms + relative_left * duration,
                span.start_ms + relative_right * duration,
            )
        )
        selected_spans.append(span)
        covered.update(range(left, right))
    if not intervals:
        return None
    intervals.sort(key=lambda interval: (interval[0], interval[1], interval[2]))
    if max_internal_gap_ms is not None:
        if max_internal_gap_ms < 0:
            raise ValueError("max_internal_gap_ms must not be negative")
        previous_end_ms = intervals[0][3]
        for _left, _right, start_ms, end_ms in intervals[1:]:
            if start_ms - previous_end_ms > max_internal_gap_ms:
                return None
            previous_end_ms = max(previous_end_ms, end_ms)
    local_coverage = len(covered) / (normalized_end - normalized_start)
    coverage = min(local_coverage, track.coverage) if cap_by_track_coverage else local_coverage
    start_ms = min(interval[2] for interval in intervals)
    end_ms = max(interval[3] for interval in intervals)
    model_confidences = [
        span.model_confidence
        for span in selected_spans
        if span.model_confidence is not None
    ]
    precision = max(
        (span.timestamp_precision for span in selected_spans),
        key=lambda value: _PRECISION_RANK[value],
        default=track.timestamp_precision,
    )
    overlapping_vad = [
        (vad_start, vad_end)
        for vad_start, vad_end in track.vad_ranges
        if vad_end >= start_ms and vad_start <= end_ms
    ]
    return AlignmentRange(
        start_ms,
        end_ms,
        coverage,
        sum(model_confidences) / len(model_confidences) if model_confidences else None,
        min(model_confidences) if model_confidences else None,
        (
            sum(value < _LOW_MODEL_CONFIDENCE_THRESHOLD for value in model_confidences)
            / len(model_confidences)
            if model_confidences
            else None
        ),
        precision,
        min((span.alignment_margin for span in selected_spans), default=track.ambiguity_margin),
        track.timestamp_valid,
        len(model_confidences) / len(selected_spans) if selected_spans else 0.0,
        min((item[0] for item in overlapping_vad), default=None),
        max((item[1] for item in overlapping_vad), default=None),
    )


def _neighbor_guards_from_track(
    track: AlignmentTrack | None,
    normalized_start: int,
    normalized_end: int,
) -> tuple[float | None, float | None]:
    if track is None:
        return None, None
    preceding = [
        span
        for span in track.spans
        if span.normalized_end <= normalized_start
    ]
    following = [
        span
        for span in track.spans
        if span.normalized_start >= normalized_end
    ]
    left_guard = max((span.end_ms for span in preceding), default=None)
    right_guard = min((span.start_ms for span in following), default=None)
    if left_guard is not None and right_guard is not None and right_guard <= left_guard:
        return None, None
    return left_guard, right_guard


def _count_occurrences(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    count = 0
    start = 0
    while (position := haystack.find(needle, start)) >= 0:
        count += 1
        start = position + 1
    return count


def _context_is_repeated(
    paragraph: NormalizedText,
    highlight: object,
    marked: str,
    *,
    corpus: str | None = None,
) -> bool:
    target = normalize_with_mapping(marked).text
    search_text = paragraph.text if corpus is None else corpus
    if not target or _count_occurrences(search_text, target) <= 1:
        return False
    before = normalize_with_mapping(str(_get(highlight, "context_before", default=""))).text[-8:]
    after = normalize_with_mapping(str(_get(highlight, "context_after", default=""))).text[:8]
    contextual = before + target + after
    # A unique surrounding phrase disambiguates repeated target words.
    return not contextual or _count_occurrences(search_text, contextual) != 1


def _fallback_range(
    paragraph: NormalizedText,
    original_start: int,
    original_end: int,
    core_start_ms: float,
    core_end_ms: float,
) -> tuple[float, float]:
    normalized_range = paragraph.normalized_range(original_start, original_end)
    duration = max(0.0, core_end_ms - core_start_ms)
    if normalized_range is not None and paragraph.text:
        left, right = normalized_range
        denominator = len(paragraph.text)
    else:
        left, right = original_start, original_end
        denominator = max(1, len(paragraph.original))
    return (
        core_start_ms + duration * left / denominator,
        core_start_ms + duration * right / denominator,
    )


def _to_sample_range(
    start_ms: float,
    end_ms: float,
    *,
    sample_rate: int,
    total_samples: int,
) -> tuple[int, int]:
    start = max(0, min(total_samples, round(start_ms * sample_rate / 1000)))
    end = max(0, min(total_samples, round(end_ms * sample_rate / 1000)))
    if end <= start:
        if total_samples <= 0:
            return 0, 0
        start = min(start, total_samples - 1)
        end = start + 1
    return start, end


def _context_excerpt(
    paragraph: NormalizedText,
    normalized_start: int,
    normalized_end: int,
    radius: int,
) -> tuple[str, int, int, int, int] | None:
    """Return a short original-text excerpt and local normalized target range."""

    if not paragraph.text or not paragraph.normalized_to_original:
        return None
    left = max(0, normalized_start - radius)
    right = min(len(paragraph.text), normalized_end + radius)
    if right <= left:
        return None
    original_left = paragraph.normalized_to_original[left]
    original_right = paragraph.normalized_to_original[right - 1] + 1
    excerpt = paragraph.original[original_left:original_right]
    excerpt_normalized = normalize_with_mapping(excerpt)
    target = excerpt_normalized.normalized_range(
        paragraph.normalized_to_original[normalized_start] - original_left,
        paragraph.normalized_to_original[normalized_end - 1] + 1 - original_left,
    )
    if target is None:
        return None
    return excerpt, target[0], target[1], left, right


def _average_ranges(
    ranges: Sequence[AlignmentRange],
) -> AlignmentRange:
    count = len(ranges)
    mean_confidences = [
        item.mean_model_confidence
        for item in ranges
        if item.mean_model_confidence is not None
    ]
    minimum_confidences = [
        item.minimum_model_confidence
        for item in ranges
        if item.minimum_model_confidence is not None
    ]
    low_ratios = [item.low_confidence_ratio for item in ranges if item.low_confidence_ratio is not None]
    return AlignmentRange(
        sum(item.start_ms for item in ranges) / count,
        sum(item.end_ms for item in ranges) / count,
        min(item.coverage for item in ranges),
        sum(mean_confidences) / len(mean_confidences) if mean_confidences else None,
        min(minimum_confidences) if minimum_confidences else None,
        max(low_ratios) if low_ratios else None,
        max(ranges, key=lambda item: _PRECISION_RANK[item.timestamp_precision]).timestamp_precision,
        min(item.ambiguity_margin for item in ranges),
        all(item.timestamp_valid for item in ranges),
        min(item.model_confidence_coverage for item in ranges),
        min(
            (item.vad_start_ms for item in ranges if item.vad_start_ms is not None),
            default=None,
        ),
        max(
            (item.vad_end_ms for item in ranges if item.vad_end_ms is not None),
            default=None,
        ),
    )


def _range_diagnostics(value: AlignmentRange | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "start_ms": round(value.start_ms, 3),
        "end_ms": round(value.end_ms, 3),
        "coverage": round(value.coverage, 4),
        "mean_model_confidence": (
            round(value.mean_model_confidence, 4)
            if value.mean_model_confidence is not None
            else None
        ),
        "minimum_model_confidence": (
            round(value.minimum_model_confidence, 4)
            if value.minimum_model_confidence is not None
            else None
        ),
        "low_confidence_ratio": (
            round(value.low_confidence_ratio, 4)
            if value.low_confidence_ratio is not None
            else None
        ),
        "timestamp_precision": value.timestamp_precision,
        "timestamp_valid": value.timestamp_valid,
        "model_confidence_coverage": round(value.model_confidence_coverage, 4),
        "alignment_margin": round(value.ambiguity_margin, 4),
        "vad_start_ms": (
            round(value.vad_start_ms, 3) if value.vad_start_ms is not None else None
        ),
        "vad_end_ms": round(value.vad_end_ms, 3) if value.vad_end_ms is not None else None,
    }


def align_transcript(
    parsed: object,
    audio_info: object,
    force_aligner: ForceAligner | None = None,
    asr_aligner: AsrAligner | None = None,
    *,
    window_padding_ms: int = 800,
    local_recognition_padding_ms: int = 5_000,
    boundary_tolerance_ms: float = 120.0,
    minimum_force_coverage: float = 0.80,
    minimum_asr_coverage: float = 0.65,
    auto_approve_threshold: float = 0.82,
    global_asr_track: AlignmentTrack | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel: object | None = None,
) -> list[AlignmentCandidate]:
    """Map DOCX yellow ranges onto the decoded PCM sample timeline.

    The primary path recognizes the complete audio once, monotonically maps
    that observed speech back onto the complete DOCX text, and derives every
    candidate from those absolute timestamps.  DOCX time anchors are used only
    when the real speech mapping fails, and such results are always review-only.
    ``fa-zh`` is a secondary boundary check inside the paragraph window found
    by ASR.  Agreeing engines are averaged; if two forced-alignment context
    sizes agree but their edges disagree with ASR, that sharper suggestion is
    allowed only as a review-required candidate and can never auto-approve.
    """

    if window_padding_ms < 0:
        raise ValueError("window_padding_ms must not be negative")
    if local_recognition_padding_ms < 0:
        raise ValueError("local_recognition_padding_ms must not be negative")
    if not math.isfinite(boundary_tolerance_ms) or boundary_tolerance_ms <= 0:
        raise ValueError("boundary_tolerance_ms must be positive and finite")
    for label, value in (
        ("minimum_force_coverage", minimum_force_coverage),
        ("minimum_asr_coverage", minimum_asr_coverage),
        ("auto_approve_threshold", auto_approve_threshold),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must be between zero and one")
    sample_rate = int(_get(audio_info, "sample_rate", "samplerate"))
    total_samples_value = _get(
        audio_info,
        "total_samples",
        "sample_count",
        "frames",
        default=None,
    )
    if total_samples_value is None:
        duration_seconds = float(_get(audio_info, "duration_seconds"))
        total_samples = round(duration_seconds * sample_rate)
    else:
        total_samples = int(total_samples_value)
    if sample_rate <= 0 or total_samples <= 0:
        raise ValueError("audio_info must describe a non-empty decoded PCM timeline")
    duration_ms = total_samples * 1000.0 / sample_rate
    audio_path = _get(audio_info, "path", "audio_path", "source_path", default="")

    paragraphs_value = _get(parsed, "paragraphs", default=parsed)
    if not isinstance(paragraphs_value, Sequence) or isinstance(paragraphs_value, (str, bytes)):
        raise TypeError("parsed transcript must contain a paragraph sequence")
    paragraphs = list(paragraphs_value)

    # Joining with newlines keeps paragraph boundaries readable for model
    # diagnostics while normalization removes those separators.  The bases
    # therefore map directly into the one global normalized reference.
    paragraph_texts = [str(_get(paragraph, "text")) for paragraph in paragraphs]
    normalized_paragraphs = [normalize_with_mapping(text) for text in paragraph_texts]
    global_normalized_text = "".join(item.text for item in normalized_paragraphs)
    paragraph_bases: list[int] = []
    normalized_total = 0
    for normalized in normalized_paragraphs:
        paragraph_bases.append(normalized_total)
        normalized_total += len(normalized.text)
    global_transcript = "\n".join(paragraph_texts)

    # This is the decisive search step: actual recognized audio across the
    # full decoded PCM duration.  No Word anchor is involved in this call.
    audio_end_ms = max(1, (total_samples * 1000 + sample_rate - 1) // sample_rate)
    ffmpeg_path = next(
        (
            path
            for aligner in (force_aligner, asr_aligner)
            if (path := getattr(aligner, "ffmpeg_path", None)) is not None
        ),
        None,
    )
    stack = ExitStack()
    window_cache: PreparedAudioWindowCache | None = None
    if str(audio_path) and Path(str(audio_path)).expanduser().is_file():
        window_cache = stack.enter_context(
            PreparedAudioWindowCache(
                audio_path,
                ffmpeg_path=ffmpeg_path,
                cancel=cancel,
            )
        )
    try:
        return _align_transcript_with_window_cache(
            paragraphs=paragraphs,
            paragraph_texts=paragraph_texts,
            normalized_paragraphs=normalized_paragraphs,
            paragraph_bases=paragraph_bases,
            global_transcript=global_transcript,
            global_normalized_text=global_normalized_text,
            normalized_total=normalized_total,
            audio_path=audio_path,
            audio_end_ms=audio_end_ms,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            total_samples=total_samples,
            force_aligner=force_aligner,
            asr_aligner=asr_aligner,
            global_asr_track=global_asr_track,
            window_padding_ms=window_padding_ms,
            local_recognition_padding_ms=local_recognition_padding_ms,
            boundary_tolerance_ms=boundary_tolerance_ms,
            minimum_force_coverage=minimum_force_coverage,
            minimum_asr_coverage=minimum_asr_coverage,
            auto_approve_threshold=auto_approve_threshold,
            progress_cb=progress_cb,
            cancel=cancel,
            window_cache=window_cache,
        )
    finally:
        stack.close()


def _align_transcript_with_window_cache(
    *,
    paragraphs: list[object],
    paragraph_texts: list[str],
    normalized_paragraphs: list[NormalizedText],
    paragraph_bases: list[int],
    global_transcript: str,
    global_normalized_text: str,
    normalized_total: int,
    audio_path: object,
    audio_end_ms: int,
    duration_ms: float,
    sample_rate: int,
    total_samples: int,
    force_aligner: ForceAligner | None,
    asr_aligner: AsrAligner | None,
    global_asr_track: AlignmentTrack | None,
    window_padding_ms: int,
    local_recognition_padding_ms: int,
    boundary_tolerance_ms: float,
    minimum_force_coverage: float,
    minimum_asr_coverage: float,
    auto_approve_threshold: float,
    progress_cb: Callable[[int, int], None] | None,
    cancel: object | None,
    window_cache: PreparedAudioWindowCache | None,
) -> list[AlignmentCandidate]:
    candidates: list[AlignmentCandidate] = []
    if global_asr_track is None:
        global_asr_track = _invoke_aligner(
            asr_aligner,
            engine="paraformer-zh+fsmn-vad-global",
            audio_path=audio_path,
            transcript=global_transcript,
            window_start_ms=0,
            window_end_ms=audio_end_ms,
            reference_length=normalized_total,
            window_cache=window_cache,
        )
    total_paragraphs = len(paragraphs)
    if progress_cb is not None:
        progress_cb(0, total_paragraphs)
    for position, paragraph in enumerate(paragraphs):
        if _is_cancelled(cancel):
            raise AlignmentCancelledError("alignment was cancelled")
        paragraph_index = int(_get(paragraph, "index", default=position))
        text = paragraph_texts[position]
        highlights_value = _get(paragraph, "highlights", default=())
        highlights = list(highlights_value)
        if not highlights:
            continue
        normalized = normalized_paragraphs[position]
        paragraph_base = paragraph_bases[position]
        paragraph_end = paragraph_base + len(normalized.text)
        anchor_ms = float(_get(paragraph, "anchor_ms", "start_ms"))
        next_anchor = (
            float(_get(paragraphs[position + 1], "anchor_ms", "start_ms"))
            if position + 1 < len(paragraphs)
            else duration_ms
        )
        core_start_ms = max(0.0, min(duration_ms, anchor_ms))
        core_end_ms = max(core_start_ms, min(duration_ms, next_anchor))
        observed_paragraph_range = _range_from_track(
            global_asr_track,
            paragraph_base,
            paragraph_end,
            cap_by_track_coverage=False,
        )
        has_observed_paragraph = (
            observed_paragraph_range is not None
            and observed_paragraph_range[2] >= 0.20
        )
        if has_observed_paragraph:
            assert observed_paragraph_range is not None
            # Forced alignment refines/checks the paragraph where the actual
            # recognized words put it, not where the Word timestamp claims it
            # should be.
            window_start_ms = max(
                0,
                int(observed_paragraph_range[0]) - local_recognition_padding_ms,
            )
            window_end_ms = min(
                audio_end_ms,
                int(observed_paragraph_range[1] + 0.999999)
                + local_recognition_padding_ms,
            )
        else:
            # Last-resort search hint.  Every candidate produced through this
            # path carries DOCUMENT_TIME_FALLBACK and cannot auto-approve.
            window_start_ms = max(0, int(core_start_ms) - window_padding_ms)
            window_end_ms = min(
                audio_end_ms,
                int(core_end_ms + 0.999999) + window_padding_ms,
            )
        if window_end_ms <= window_start_ms:
            window_end_ms = min(int(round(duration_ms)), window_start_ms + 1)

        fallback_force_track = None
        if not has_observed_paragraph:
            fallback_force_track = _invoke_aligner(
                force_aligner,
                engine="fa-zh-anchor-fallback",
                audio_path=audio_path,
                transcript=text,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                reference_length=len(normalized.text),
                window_cache=window_cache,
            )
        # When the global audio text cannot locate a paragraph at all, retain
        # one anchor-window ASR attempt strictly as a review-only fallback.
        anchor_asr_track = None
        if not has_observed_paragraph:
            anchor_asr_track = _invoke_aligner(
                asr_aligner,
                engine="paraformer-zh+fsmn-vad-anchor-fallback",
                audio_path=audio_path,
                transcript=text,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                reference_length=len(normalized.text),
                window_cache=window_cache,
            )

        for highlight_index, highlight in enumerate(highlights):
            if _is_cancelled(cancel):
                raise AlignmentCancelledError("alignment was cancelled")
            original_start = int(_get(highlight, "start", "start_offset", "char_start"))
            original_end = int(_get(highlight, "end", "end_offset", "char_end"))
            if original_start < 0 or original_end <= original_start or original_end > len(text):
                raise ValueError(
                    f"paragraph {paragraph_index} highlight {highlight_index} has invalid offsets"
                )
            marked = str(_get(highlight, "text", default=text[original_start:original_end]))
            normalized_range = normalized.normalized_range(original_start, original_end)
            if normalized_range is None:
                # Parser-produced transcripts already filter these, but the
                # duck-typed public API may receive external paragraph objects.
                # Never turn punctuation-only formatting into an audio cut.
                continue
            force_range = None
            force_track_available = False
            force_context_unstable = False
            force_context_consensus = False
            observed_context_insufficient = False
            local_asr_track = anchor_asr_track
            has_local_refinement = False
            context_track = anchor_asr_track if not has_observed_paragraph else global_asr_track
            context_base = 0 if not has_observed_paragraph else paragraph_base
            if normalized_range is None:
                asr_range = None
                normalized_length = 0
            else:
                normalized_start, normalized_end = normalized_range
                normalized_length = normalized_end - normalized_start
                if has_observed_paragraph:
                    global_asr_range = _range_from_track(
                        global_asr_track,
                        paragraph_base + normalized_start,
                        paragraph_base + normalized_end,
                        cap_by_track_coverage=False,
                        max_internal_gap_ms=MAX_HIGHLIGHT_INTERNAL_GAP_MS,
                    )
                    coarse_search_range = global_asr_range
                    if coarse_search_range is None:
                        coarse_context_left = max(0, normalized_start - 24)
                        coarse_context_right = min(
                            len(normalized.text),
                            normalized_end + 24,
                        )
                        coarse_search_range = _range_from_track(
                            global_asr_track,
                            paragraph_base + coarse_context_left,
                            paragraph_base + coarse_context_right,
                            cap_by_track_coverage=False,
                        )
                    if coarse_search_range is None and observed_paragraph_range is not None:
                        approximate_start, approximate_end = _fallback_range(
                            normalized,
                            original_start,
                            original_end,
                            observed_paragraph_range[0],
                            observed_paragraph_range[1],
                        )
                        coarse_search_range = (approximate_start, approximate_end, 0.0)

                    if coarse_search_range is not None:
                        refinement_start = max(0, int(coarse_search_range[0]) - 6_000)
                        refinement_end = min(
                            audio_end_ms,
                            int(coarse_search_range[1] + 0.999999) + 6_000,
                        )
                        if refinement_end > refinement_start:
                            local_asr_track = _invoke_aligner(
                                asr_aligner,
                                engine="paraformer-zh+fsmn-vad-highlight-refinement",
                                audio_path=audio_path,
                                transcript=text,
                                window_start_ms=refinement_start,
                                window_end_ms=refinement_end,
                                reference_length=len(normalized.text),
                                window_cache=window_cache,
                            )
                            if _is_cancelled(cancel):
                                raise AlignmentCancelledError("alignment was cancelled")
                    local_asr_range = _range_from_track(
                        local_asr_track,
                        normalized_start,
                        normalized_end,
                        cap_by_track_coverage=False,
                        max_internal_gap_ms=MAX_HIGHLIGHT_INTERNAL_GAP_MS,
                    )
                    has_local_refinement = (
                        local_asr_range is not None
                        and local_asr_range[2] >= minimum_asr_coverage
                    )
                    asr_range = local_asr_range or global_asr_range
                    local_context_range = _range_from_track(
                        local_asr_track,
                        max(0, normalized_start - 24),
                        min(len(normalized.text), normalized_end + 24),
                        cap_by_track_coverage=False,
                    )
                    if local_context_range is not None and local_context_range[2] >= 0.20:
                        context_track = local_asr_track
                        context_base = 0
                    context_slices = (
                        (max(0, normalized_start - 8), normalized_start),
                        (normalized_end, min(len(normalized.text), normalized_end + 8)),
                    )
                    available_context = 0
                    supported_context = 0.0
                    for context_left, context_right in context_slices:
                        context_length = context_right - context_left
                        if context_length <= 0:
                            continue
                        available_context += context_length
                        mapped_context = _range_from_track(
                            context_track,
                            context_base + context_left,
                            context_base + context_right,
                            cap_by_track_coverage=False,
                        )
                        if mapped_context is not None:
                            supported_context += context_length * mapped_context[2]
                    observed_context_insufficient = (
                        available_context < 4
                        or supported_context + 1e-9 < min(6, available_context)
                    )
                else:
                    asr_range = _range_from_track(
                        anchor_asr_track,
                        normalized_start,
                        normalized_end,
                        max_internal_gap_ms=MAX_HIGHLIGHT_INTERNAL_GAP_MS,
                    )

                if has_observed_paragraph and context_track is not None:
                    # The dedicated timestamp model is most stable on a short,
                    # matching excerpt.  Run two context sizes around the
                    # ASR-located target; agreement across those passes guards
                    # against a forced aligner inventing positions for text
                    # that was not actually spoken.
                    contextual_ranges: list[AlignmentRange] = []
                    seen_windows: set[tuple[str, int, int]] = set()
                    for context_radius in (10, 20):
                        if _is_cancelled(cancel):
                            raise AlignmentCancelledError("alignment was cancelled")
                        excerpt_data = _context_excerpt(
                            normalized,
                            normalized_start,
                            normalized_end,
                            context_radius,
                        )
                        if excerpt_data is None:
                            continue
                        (
                            excerpt,
                            excerpt_target_start,
                            excerpt_target_end,
                            context_start,
                            context_end,
                        ) = excerpt_data
                        observed_context = _range_from_track(
                            context_track,
                            context_base + context_start,
                            context_base + context_end,
                            cap_by_track_coverage=False,
                        )
                        if observed_context is None or observed_context[2] < 0.35:
                            continue
                        local_window_start = max(0, int(observed_context[0]) - 450)
                        local_window_end = min(
                            audio_end_ms,
                            int(observed_context[1] + 0.999999) + 450,
                        )
                        if local_window_end <= local_window_start:
                            continue
                        window_key = (excerpt, local_window_start, local_window_end)
                        if window_key in seen_windows:
                            continue
                        seen_windows.add(window_key)
                        excerpt_normalized_length = len(normalize_with_mapping(excerpt).text)
                        local_force_track = _invoke_aligner(
                            force_aligner,
                            engine=f"fa-zh-context-{context_radius}",
                            audio_path=audio_path,
                            transcript=excerpt,
                            window_start_ms=local_window_start,
                            window_end_ms=local_window_end,
                            reference_length=excerpt_normalized_length,
                            window_cache=window_cache,
                        )
                        if local_force_track is not None:
                            force_track_available = True
                        local_force_range = _range_from_track(
                            local_force_track,
                            excerpt_target_start,
                            excerpt_target_end,
                            max_internal_gap_ms=MAX_HIGHLIGHT_INTERNAL_GAP_MS,
                        )
                        if local_force_range is not None:
                            contextual_ranges.append(local_force_range)

                    if contextual_ranges:
                        force_range = contextual_ranges[0]
                    if len(contextual_ranges) >= 2:
                        context_difference = max(
                            max(
                                abs(left[0] - right[0]),
                                abs(left[1] - right[1]),
                            )
                            for left in contextual_ranges
                            for right in contextual_ranges
                        )
                        force_context_unstable = context_difference > 80.0
                        if not force_context_unstable:
                            force_context_consensus = True
                            force_range = _average_ranges(contextual_ranges)
                else:
                    force_track_available = fallback_force_track is not None
                    force_range = _range_from_track(
                        fallback_force_track,
                        normalized_start,
                        normalized_end,
                        max_internal_gap_ms=MAX_HIGHLIGHT_INTERNAL_GAP_MS,
                    )

            fallback = force_range is None and asr_range is None
            used_character_force_boundary = False
            if asr_range is not None and force_range is not None:
                preliminary_difference = max(
                    abs(force_range[0] - asr_range[0]),
                    abs(force_range[1] - asr_range[1]),
                )
                if preliminary_difference <= boundary_tolerance_ms:
                    if (
                        force_context_consensus
                        and asr_range.timestamp_precision != TIMESTAMP_PRECISION_CHARACTER
                        and force_range.timestamp_precision == TIMESTAMP_PRECISION_CHARACTER
                    ):
                        # The ASR interval only has word/segment granularity, so
                        # its edges systematically overshoot the highlighted
                        # characters.  Two agreeing forced-alignment context
                        # windows provide genuine per-character edges inside
                        # the ASR-confirmed location; averaging with the wider
                        # word span would only drag the cut outward again.
                        proposed_start_ms, proposed_end_ms = force_range[:2]
                        used_character_force_boundary = True
                    else:
                        # Both timestamp mechanisms have similar published
                        # error scales.  Their midpoint reduces one-sided
                        # boundary bias while the actual-audio ASR still
                        # determines the location.
                        proposed_start_ms = (asr_range[0] + force_range[0]) / 2
                        proposed_end_ms = (asr_range[1] + force_range[1]) / 2
                elif force_context_consensus and has_local_refinement:
                    # Two differently sized, audio-located transcript windows
                    # agree with each other.  That stable dedicated timestamp
                    # estimate is a better character edge than the broader
                    # ASR token span, while the disagreement still forces
                    # human review.
                    proposed_start_ms, proposed_end_ms = force_range[:2]
                else:
                    proposed_start_ms, proposed_end_ms = asr_range[:2]
            elif asr_range is not None:
                # The recognized audio is authoritative.  Forced alignment is
                # only an independent precision/consistency check.
                proposed_start_ms, proposed_end_ms = asr_range[:2]
            elif force_range is not None:
                proposed_start_ms, proposed_end_ms = force_range[:2]
            else:
                fallback_start_ms = (
                    observed_paragraph_range[0]
                    if has_observed_paragraph and observed_paragraph_range is not None
                    else core_start_ms
                )
                fallback_end_ms = (
                    observed_paragraph_range[1]
                    if has_observed_paragraph and observed_paragraph_range is not None
                    else core_end_ms
                )
                proposed_start_ms, proposed_end_ms = _fallback_range(
                    normalized,
                    original_start,
                    original_end,
                    fallback_start_ms,
                    fallback_end_ms,
                )

            # Never push a model result back into the Word anchor interval.
            # Only the decoded PCM boundaries are absolute constraints.
            proposed_start_ms = max(0.0, min(duration_ms, proposed_start_ms))
            proposed_end_ms = max(0.0, min(duration_ms, proposed_end_ms))
            start_sample, end_sample = _to_sample_range(
                proposed_start_ms,
                proposed_end_ms,
                sample_rate=sample_rate,
                total_samples=total_samples,
            )

            active_asr_track = (
                local_asr_track
                if has_local_refinement
                else (global_asr_track if has_observed_paragraph else anchor_asr_track)
            )
            if has_local_refinement and local_asr_track is not None:
                guard_track = local_asr_track
                guard_start = normalized_start
                guard_end = normalized_end
            elif has_observed_paragraph and global_asr_track is not None:
                guard_track = global_asr_track
                guard_start = paragraph_base + normalized_start
                guard_end = paragraph_base + normalized_end
            else:
                guard_track = anchor_asr_track
                guard_start = normalized_start
                guard_end = normalized_end
            left_guard_ms, right_guard_ms = _neighbor_guards_from_track(
                guard_track,
                guard_start,
                guard_end,
            )
            left_guard_sample = (
                max(0, min(total_samples, round(left_guard_ms * sample_rate / 1000)))
                if left_guard_ms is not None
                else None
            )
            right_guard_sample = (
                max(0, min(total_samples, round(right_guard_ms * sample_rate / 1000)))
                if right_guard_ms is not None
                else None
            )
            if left_guard_sample is not None and left_guard_sample >= end_sample:
                left_guard_sample = None
            if right_guard_sample is not None and right_guard_sample <= start_sample:
                right_guard_sample = None

            short = normalized_length <= 2
            repeated = _context_is_repeated(
                normalized,
                highlight,
                marked,
                corpus=global_normalized_text,
            )
            force_coverage = force_range.coverage if force_range is not None else 0.0
            asr_coverage = asr_range.coverage if asr_range is not None else 0.0
            insufficient = (
                force_range is not None and force_coverage < minimum_force_coverage
            ) or (asr_range is not None and asr_coverage < minimum_asr_coverage)
            insufficient = insufficient or (force_track_available and force_range is None)
            insufficient = insufficient or (
                active_asr_track is not None and asr_range is None
            )
            insufficient = insufficient or observed_context_insufficient
            insufficient = insufficient or (
                force_aligner is not None
                and has_observed_paragraph
                and force_range is None
            )
            if force_range is not None and asr_range is not None:
                boundary_difference: float | None = max(
                    abs(force_range.start_ms - asr_range.start_ms),
                    abs(force_range.end_ms - asr_range.end_ms),
                )
            else:
                boundary_difference = None
            disagreement = (
                boundary_difference is not None
                and boundary_difference > boundary_tolerance_ms
            )
            timestamp_anomaly = any(
                item is not None and not item.timestamp_valid
                for item in (asr_range, force_range)
            ) or (active_asr_track is not None and not active_asr_track.timestamp_valid)
            coarse_timestamp = any(
                item is not None
                and item.timestamp_precision != TIMESTAMP_PRECISION_CHARACTER
                for item in (asr_range, force_range)
            )
            if used_character_force_boundary:
                # The cut edges come from agreeing per-character forced
                # alignment; the coarser ASR word span only confirmed the
                # location and no longer defines any boundary.
                coarse_timestamp = False
            ambiguity_margin = min(
                (
                    item.ambiguity_margin
                    for item in (asr_range, force_range)
                    if item is not None
                ),
                default=0.0,
            )
            dp_forced_partition = bool(
                active_asr_track is not None
                and "dp_forced_partition" in active_asr_track.diagnostics
            )
            ambiguous = asr_range is not None and (
                ambiguity_margin < _MIN_ALIGNMENT_MARGIN or dp_forced_partition
            )
            model_confidence_unavailable = (
                asr_range is not None and asr_range.model_confidence_coverage < 1.0
            )
            low_model_confidence = bool(
                asr_range is not None
                and asr_range.mean_model_confidence is not None
                and (
                    asr_range.mean_model_confidence < 0.72
                    or (
                        asr_range.minimum_model_confidence is not None
                        and asr_range.minimum_model_confidence < 0.40
                    )
                    or (
                        asr_range.low_confidence_ratio is not None
                        and asr_range.low_confidence_ratio > 0.20
                    )
                )
            )
            guard_unavailable = left_guard_sample is None or right_guard_sample is None
            candidate_span_ms = max(0.0, proposed_end_ms - proposed_start_ms)
            duration_anomaly = (
                candidate_span_ms > max(4_000.0, normalized_length * 1_200.0)
                or (
                    normalized_length >= 3
                    and candidate_span_ms < normalized_length * 15.0
                )
            )
            timestamp_anomaly = timestamp_anomaly or duration_anomaly

            reasons: list[str] = []
            if fallback:
                reasons.append(FALLBACK_ALIGNMENT)
            if not has_observed_paragraph:
                reasons.append(DOCUMENT_TIME_FALLBACK)
            if short:
                reasons.append(SHORT_HIGHLIGHT)
            if repeated:
                reasons.append(REPEATED_CONTEXT)
            if ambiguous:
                reasons.append(ALIGNMENT_AMBIGUOUS)
            if insufficient:
                reasons.append(INSUFFICIENT_COVERAGE)
            if disagreement:
                reasons.append(BOUNDARY_DISAGREEMENT)
            if force_context_unstable:
                reasons.append(FORCE_CONTEXT_UNSTABLE)
            if timestamp_anomaly:
                reasons.append(TIMESTAMP_ANOMALY)
            if coarse_timestamp:
                reasons.append(COARSE_TIMESTAMP)
            if model_confidence_unavailable:
                reasons.append(MODEL_CONFIDENCE_UNAVAILABLE)
            if low_model_confidence:
                reasons.append(LOW_MODEL_CONFIDENCE)
            if guard_unavailable:
                reasons.append(BOUNDARY_GUARD_UNAVAILABLE)
            if has_observed_paragraph and not has_local_refinement:
                reasons.append(LOCAL_RECOGNITION_UNAVAILABLE)
            if force_aligner is None:
                reasons.append(FORCE_ALIGNER_UNAVAILABLE)
            if active_asr_track is None:
                reasons.append(ASR_ALIGNER_UNAVAILABLE)

            if fallback:
                confidence = 0.15
            elif force_range is not None and asr_range is not None:
                confidence = 0.55 + 0.25 * force_coverage + 0.12 * asr_coverage
                if not disagreement and boundary_difference is not None:
                    confidence += 0.08 * max(
                        0.0, 1.0 - boundary_difference / boundary_tolerance_ms
                    )
            elif force_range is not None:
                confidence = 0.45 + 0.30 * force_coverage
            else:
                confidence = 0.25 + 0.30 * asr_coverage
            if (
                asr_range is not None
                and asr_range.mean_model_confidence is not None
                and not model_confidence_unavailable
            ):
                confidence = 0.72 * confidence + 0.28 * asr_range.mean_model_confidence
            elif model_confidence_unavailable:
                confidence = min(confidence, 0.69)
            if short:
                confidence -= 0.20
            if repeated:
                confidence -= 0.12
            if ambiguous:
                confidence -= 0.20
            if insufficient:
                confidence -= 0.20
            if disagreement:
                confidence -= 0.25
            if force_context_unstable:
                confidence -= 0.12
            if coarse_timestamp:
                confidence -= 0.18
            if low_model_confidence:
                confidence -= 0.18
            if guard_unavailable:
                confidence -= 0.08
            if timestamp_anomaly:
                confidence = min(confidence, 0.20)
            confidence = round(max(0.0, min(0.99, confidence)), 4)

            auto_approved = (
                force_range is not None
                and asr_range is not None
                and not fallback
                and not short
                and not repeated
                and not ambiguous
                and not insufficient
                and not disagreement
                and not force_context_unstable
                and not timestamp_anomaly
                and not coarse_timestamp
                and not model_confidence_unavailable
                and not low_model_confidence
                and not guard_unavailable
                and has_observed_paragraph
                and has_local_refinement
                and confidence >= auto_approve_threshold
            )
            speech_start_sample = (
                max(
                    0,
                    min(total_samples, round(asr_range.vad_start_ms * sample_rate / 1000)),
                )
                if asr_range is not None and asr_range.vad_start_ms is not None
                else None
            )
            speech_end_sample = (
                max(
                    0,
                    min(total_samples, round(asr_range.vad_end_ms * sample_rate / 1000)),
                )
                if asr_range is not None and asr_range.vad_end_ms is not None
                else None
            )
            diagnostics: dict[str, object] = {
                "timestamp_unit": "ms",
                "matching_strategy": "anchored_monotonic_dp",
                "source_highlight_index": int(
                    _get(highlight, "source_highlight_index", default=highlight_index)
                ),
                "force_alignment": _range_diagnostics(force_range),
                "asr_alignment": _range_diagnostics(asr_range),
                "boundary_disagreement_ms": (
                    round(boundary_difference, 3)
                    if boundary_difference is not None
                    else None
                ),
                "candidate_span_ms": round(candidate_span_ms, 3),
                "used_character_force_boundary": used_character_force_boundary,
                "alignment_ambiguity_margin": round(ambiguity_margin, 4),
                "dp_forced_partition": dp_forced_partition,
                "left_guard_sample": left_guard_sample,
                "right_guard_sample": right_guard_sample,
                "initial_start_sample": start_sample,
                "initial_end_sample": end_sample,
                "asr_track_diagnostics": (
                    list(active_asr_track.diagnostics) if active_asr_track is not None else []
                ),
                "global_asr_track_diagnostics": (
                    list(global_asr_track.diagnostics) if global_asr_track is not None else []
                ),
                "local_asr_track_diagnostics": (
                    list(local_asr_track.diagnostics) if local_asr_track is not None else []
                ),
            }
            candidates.append(
                AlignmentCandidate(
                    paragraph_index=paragraph_index,
                    highlight_index=highlight_index,
                    highlighted_text=marked,
                    proposed_start_sample=start_sample,
                    proposed_end_sample=end_sample,
                    confidence=confidence,
                    reasons=list(dict.fromkeys(reasons)),
                    requires_review=not auto_approved,
                    status=STATUS_AUTO_APPROVED if auto_approved else STATUS_NEEDS_REVIEW,
                    diagnostics=diagnostics,
                    left_guard_sample=left_guard_sample,
                    right_guard_sample=right_guard_sample,
                    speech_start_sample=speech_start_sample,
                    speech_end_sample=speech_end_sample,
                )
            )
        if progress_cb is not None:
            progress_cb(position + 1, total_paragraphs)
    return candidates


__all__ = [
    "ALIGNMENT_AMBIGUOUS",
    "ALIGNMENT_PIPELINE_NAME",
    "ALIGNMENT_PIPELINE_PURPOSE",
    "ALIGNMENT_PIPELINE_VERSION",
    "ASR_ALIGNER_UNAVAILABLE",
    "BOUNDARY_EXPANDED",
    "BOUNDARY_GUARD_UNAVAILABLE",
    "BOUNDARY_REFINEMENT_UNCERTAIN",
    "BOUNDARY_DISAGREEMENT",
    "COARSE_TIMESTAMP",
    "DOCUMENT_TIME_FALLBACK",
    "FALLBACK_ALIGNMENT",
    "FORCE_CONTEXT_UNSTABLE",
    "FORCE_ALIGNER_UNAVAILABLE",
    "INSUFFICIENT_COVERAGE",
    "LOCAL_RECOGNITION_UNAVAILABLE",
    "LOW_MODEL_CONFIDENCE",
    "MAX_HIGHLIGHT_INTERNAL_GAP_MS",
    "MODEL_CONFIDENCE_UNAVAILABLE",
    "REPEATED_CONTEXT",
    "SHORT_HIGHLIGHT",
    "STATUS_AUTO_APPROVED",
    "STATUS_NEEDS_REVIEW",
    "TIMESTAMP_ANOMALY",
    "TIMESTAMP_PRECISION_CHARACTER",
    "TIMESTAMP_PRECISION_SEGMENT",
    "TIMESTAMP_PRECISION_TOKEN",
    "TIMESTAMP_PRECISION_UNKNOWN",
    "AlignmentCancelledError",
    "AlignmentCandidate",
    "AlignmentRange",
    "AlignmentTrack",
    "RecognizedToken",
    "AsrAligner",
    "ForceAligner",
    "FunASRAsrAligner",
    "FunASRForceAligner",
    "NormalizedText",
    "TimedSpan",
    "align_transcript",
    "alignment_track_from_model_result",
    "alignment_track_from_recognition_tokens",
    "normalize_text",
    "normalize_with_mapping",
    "recognition_tokens_from_model_result",
    "refine_recognition_tokens",
    "voice_ranges_from_model_result",
]
