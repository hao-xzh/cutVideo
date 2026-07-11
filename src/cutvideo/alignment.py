"""Offline transcript-to-audio alignment and conservative cut suggestions.

The public entry point, :func:`align_transcript`, intentionally consumes the
DOCX/parser and audio-info objects by attribute rather than importing their
types.  It can therefore be reused by the UI, project migration code and small
test doubles without introducing a storage-layer dependency.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pypinyin import Style, pinyin

from .model_runtime import (
    ModelUnavailableError,
    load_funasr_model,
    local_audio_window,
    require_local_model,
)

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

# Stored in every project so results produced by an older alignment strategy
# are never silently reused after the audio-first pipeline changes.
ALIGNMENT_PIPELINE_PURPOSE = "alignment_strategy"
ALIGNMENT_PIPELINE_NAME = "audio-text-primary"
ALIGNMENT_PIPELINE_VERSION = "5"

# A highlighted phrase should be one locally continuous piece of speech.  A
# larger hole usually means that sparse ASR matches from two different spoken
# passages were accidentally joined.  Paragraph/context lookup deliberately
# does not use this limit because those wider search ranges may contain pauses.
MAX_HIGHLIGHT_INTERNAL_GAP_MS = 1_500.0

STATUS_NEEDS_REVIEW = "needs_review"
STATUS_AUTO_APPROVED = "auto_approved"


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


def _matching_character_pairs(
    reference_text: str,
    observed_text: str,
    *,
    allow_phonetic_fallback: bool,
) -> list[tuple[int, int, float]]:
    """Map observed characters monotonically, preferring exact text anchors.

    Chinese homophones are considered only inside gaps left by exact matching
    blocks.  This prevents a phonetic match from displacing already reliable
    literal context or making Latin/non-Han text fuzzy.
    """

    exact_matcher = SequenceMatcher(None, reference_text, observed_text, autojunk=False)
    matches: list[tuple[int, int, float]] = []
    reference_keys = _phonetic_keys(reference_text) if allow_phonetic_fallback else ()
    observed_keys = _phonetic_keys(observed_text) if allow_phonetic_fallback else ()
    previous_reference_end = 0
    previous_observed_end = 0

    for reference_start, observed_start, size in exact_matcher.get_matching_blocks():
        if (
            allow_phonetic_fallback
            and reference_start > previous_reference_end
            and observed_start > previous_observed_end
        ):
            phonetic_matcher = SequenceMatcher(
                None,
                reference_keys[previous_reference_end:reference_start],
                observed_keys[previous_observed_end:observed_start],
                autojunk=False,
            )
            for local_reference, local_observed, phonetic_size in (
                phonetic_matcher.get_matching_blocks()
            ):
                for offset in range(phonetic_size):
                    reference_index = previous_reference_end + local_reference + offset
                    observed_index = previous_observed_end + local_observed + offset
                    reference_character = reference_text[reference_index]
                    observed_character = observed_text[observed_index]
                    if reference_character == observed_character:
                        confidence = 1.0
                    elif _is_han_character(reference_character) and _is_han_character(
                        observed_character
                    ):
                        confidence = 0.9
                    else:
                        # Identical phonetic keys for non-Han text are literal
                        # keys and should therefore only match equal characters.
                        continue
                    matches.append((reference_index, observed_index, confidence))

        for offset in range(size):
            matches.append((reference_start + offset, observed_start + offset, 1.0))
        previous_reference_end = reference_start + size
        previous_observed_end = observed_start + size

    matches.sort(key=lambda match: (match[0], match[1]))
    return matches


@dataclass(frozen=True, slots=True)
class TimedSpan:
    """An absolute millisecond interval mapped to normalized transcript offsets."""

    normalized_start: int
    normalized_end: int
    start_ms: float
    end_ms: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.normalized_start < 0 or self.normalized_end <= self.normalized_start:
            raise ValueError("invalid normalized span")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("invalid timed span")


@dataclass(frozen=True, slots=True)
class AlignmentTrack:
    """One model's normalized-reference alignment for a paragraph."""

    spans: tuple[TimedSpan, ...]
    engine: str
    coverage: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be between zero and one")


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


def _partition_token(token: str, start_ms: float, end_ms: float) -> list[tuple[str, float, float]]:
    normalized = normalize_with_mapping(_TAG_RE.sub("", token)).text
    if not normalized:
        return []
    width = (end_ms - start_ms) / len(normalized)
    return [
        (character, start_ms + index * width, start_ms + (index + 1) * width)
        for index, character in enumerate(normalized)
    ]


def _numeric_pair(value: object) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        try:
            return float(_get(value, "start", "start_ms")), float(
                _get(value, "end", "end_ms")
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
) -> list[tuple[str, float, float]]:
    observed: list[tuple[str, float, float]] = []
    multiplier = 1000.0 if timestamps_in_seconds else 1.0

    for item in _result_dicts(result):
        raw_timestamps = item.get("timestamp", item.get("timestamps", item.get("time_stamp")))
        pairs = (
            [pair for raw in raw_timestamps if (pair := _numeric_pair(raw)) is not None]
            if isinstance(raw_timestamps, Sequence)
            and not isinstance(raw_timestamps, (str, bytes))
            else []
        )
        if not pairs:
            # Some FunASR versions expose only segment-level information. When
            # character timestamps are present they take precedence; consuming
            # both shapes would duplicate the recognized text and corrupt the
            # reference mapping.
            sentence_info = item.get("sentence_info")
            if isinstance(sentence_info, Sequence) and not isinstance(
                sentence_info, (str, bytes)
            ):
                for sentence in sentence_info:
                    if not isinstance(sentence, Mapping):
                        continue
                    pair = _numeric_pair(sentence)
                    text = str(sentence.get("text", ""))
                    if pair is not None:
                        observed.extend(
                            _partition_token(
                                text,
                                pair[0] * multiplier + time_offset_ms,
                                pair[1] * multiplier + time_offset_ms,
                            )
                        )
            continue

        raw_tokens = item.get("tokens", item.get("token"))
        text = str(item.get("text", item.get("sentence", item.get("transcript", ""))))
        serialized_tokens = _fa_serialized_tokens(text) if forced_reference else []
        if forced_reference and len(pairs) == len(reference.text):
            # fa-zh returns authoritative token timestamps but serializes its
            # human-readable ``text`` as ``token start end;`` records.  That
            # field must not be normalized as recognized prose: its timing
            # digits otherwise create false SequenceMatcher coverage and move
            # every boundary.  The model contract states that ``timestamp``
            # excludes silence and is aligned one-to-one with input tokens.
            tokens = list(reference.text)
        elif forced_reference and len(serialized_tokens) == len(pairs):
            # Tokenization can legitimately group Latin words or multiple
            # characters.  In that case parse the fa-zh serialization and let
            # the normal reference SequenceMatcher map those real tokens.
            tokens = serialized_tokens
        elif isinstance(raw_tokens, Sequence) and not isinstance(raw_tokens, (str, bytes)):
            tokens = [str(token) for token in raw_tokens]
        else:
            whitespace_tokens = _TAG_RE.sub("", text).split()
            if len(whitespace_tokens) == len(pairs):
                tokens = whitespace_tokens
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
                if len(normalized_output) == len(pairs):
                    tokens = list(normalized_output)
                else:
                    # Last-resort grouping for wordpiece/English results whose
                    # token list was not returned by this FunASR version.
                    tokens = []
                    for index in range(len(pairs)):
                        left = round(index * len(normalized_output) / len(pairs))
                        right = round((index + 1) * len(normalized_output) / len(pairs))
                        tokens.append(normalized_output[left:right])

        for token, pair in zip(tokens, pairs, strict=False):
            observed.extend(
                _partition_token(
                    token,
                    pair[0] * multiplier + time_offset_ms,
                    pair[1] * multiplier + time_offset_ms,
                )
            )
    return observed


def alignment_track_from_model_result(
    result: object,
    transcript: str,
    *,
    engine: str,
    time_offset_ms: float = 0.0,
    timestamps_in_seconds: bool = False,
    forced_reference: bool = False,
) -> AlignmentTrack:
    """Convert common FunASR result shapes into normalized-reference spans."""

    reference = normalize_with_mapping(transcript)
    if not reference.text:
        return AlignmentTrack((), engine, 0.0)
    observed = _observed_characters(
        result,
        reference,
        time_offset_ms=time_offset_ms,
        timestamps_in_seconds=timestamps_in_seconds,
        forced_reference=forced_reference,
    )
    observed_text = "".join(character for character, _start, _end in observed)
    spans: list[TimedSpan] = []
    covered: set[int] = set()
    for normalized_index, observed_index, confidence in _matching_character_pairs(
        reference.text,
        observed_text,
        allow_phonetic_fallback=not forced_reference,
    ):
        _character, start_ms, end_ms = observed[observed_index]
        if end_ms <= start_ms or start_ms < 0:
            continue
        spans.append(
            TimedSpan(
                normalized_index,
                normalized_index + 1,
                start_ms,
                end_ms,
                confidence,
            )
        )
        covered.add(normalized_index)
    spans.sort(key=lambda span: (span.normalized_start, span.start_ms))
    return AlignmentTrack(tuple(spans), engine, len(covered) / len(reference.text))


class FunASRForceAligner:
    """Local ``fa-zh`` adapter.  The model is loaded lazily and never downloaded."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        ffmpeg_path: str | Path | None = None,
        device: str = "cpu",
        model_factory: Any | None = None,
        timestamps_in_seconds: bool = False,
    ) -> None:
        self.model_path = require_local_model(model_path, "fa-zh")
        self.ffmpeg_path = Path(ffmpeg_path).resolve() if ffmpeg_path is not None else None
        self.device = device
        self.model_factory = model_factory
        self.timestamps_in_seconds = timestamps_in_seconds
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = load_funasr_model(
                model_path=self.model_path,
                label="fa-zh",
                device=self.device,
                model_factory=self.model_factory,
            )
        return self._model

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
            try:
                result = self._get_model().generate(
                    input=(str(window), transcript),
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
            time_offset_ms=window_start_ms,
            timestamps_in_seconds=self.timestamps_in_seconds,
            forced_reference=True,
        )


class FunASRAsrAligner:
    """Local ``paraformer-zh`` plus ``fsmn-vad`` timestamp adapter."""

    def __init__(
        self,
        model_path: str | Path,
        vad_model_path: str | Path,
        *,
        ffmpeg_path: str | Path | None = None,
        device: str = "cpu",
        model_factory: Any | None = None,
        timestamps_in_seconds: bool = False,
        max_vad_segment_ms: int = 30_000,
    ) -> None:
        self.model_path = require_local_model(model_path, "paraformer-zh")
        self.vad_model_path = require_local_model(vad_model_path, "fsmn-vad")
        self.ffmpeg_path = Path(ffmpeg_path).resolve() if ffmpeg_path is not None else None
        self.device = device
        self.model_factory = model_factory
        self.timestamps_in_seconds = timestamps_in_seconds
        self.max_vad_segment_ms = max(1_000, int(max_vad_segment_ms))
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = load_funasr_model(
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
        return self._model

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
            try:
                result = self._get_model().generate(
                    input=str(window),
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
            time_offset_ms=window_start_ms,
            timestamps_in_seconds=self.timestamps_in_seconds,
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
            )
        spans.append(span)
        covered.update(range(span.normalized_start, span.normalized_end))
    supplied_coverage = _get(value, "coverage", default=None)
    coverage = (
        float(supplied_coverage)
        if supplied_coverage is not None
        else min(1.0, len(covered) / max(1, reference_length))
    )
    return AlignmentTrack(tuple(spans), str(_get(value, "engine", default=engine)), coverage)


def _invoke_aligner(
    aligner: object | None,
    *,
    engine: str,
    audio_path: str | Path,
    transcript: str,
    window_start_ms: int,
    window_end_ms: int,
    reference_length: int,
) -> AlignmentTrack | None:
    if aligner is None:
        return None
    try:
        result = aligner.align(
            audio_path=audio_path,
            transcript=transcript,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
        )
        return _coerce_track(result, engine, reference_length)
    except Exception:
        # One corrupt paragraph or missing optional runtime must never result in
        # an unsafe automatic cut.  The caller records an explicit fallback.
        return None


def _range_from_track(
    track: AlignmentTrack | None,
    normalized_start: int,
    normalized_end: int,
    *,
    cap_by_track_coverage: bool = True,
    max_internal_gap_ms: float | None = None,
) -> tuple[float, float, float] | None:
    if track is None or normalized_end <= normalized_start:
        return None
    intervals: list[tuple[int, int, float, float]] = []
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
    return (
        min(interval[2] for interval in intervals),
        max(interval[3] for interval in intervals),
        coverage,
    )


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
    ranges: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float]:
    count = len(ranges)
    return (
        sum(item[0] for item in ranges) / count,
        sum(item[1] for item in ranges) / count,
        min(item[2] for item in ranges),
    )


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
    candidates: list[AlignmentCandidate] = []

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
    global_asr_track = _invoke_aligner(
        asr_aligner,
        engine="paraformer-zh+fsmn-vad-global",
        audio_path=audio_path,
        transcript=global_transcript,
        window_start_ms=0,
        window_end_ms=audio_end_ms,
        reference_length=normalized_total,
    )
    if global_asr_track is not None and not global_asr_track.spans:
        global_asr_track = None

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
            )

        for highlight_index, highlight in enumerate(highlights):
            original_start = int(_get(highlight, "start", "start_offset", "char_start"))
            original_end = int(_get(highlight, "end", "end_offset", "char_end"))
            if original_start < 0 or original_end <= original_start or original_end > len(text):
                raise ValueError(
                    f"paragraph {paragraph_index} highlight {highlight_index} has invalid offsets"
                )
            marked = str(_get(highlight, "text", default=text[original_start:original_end]))
            normalized_range = normalized.normalized_range(original_start, original_end)
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
                            )
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
                    contextual_ranges: list[tuple[float, float, float]] = []
                    seen_windows: set[tuple[str, int, int]] = set()
                    for context_radius in (10, 20):
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
            if asr_range is not None and force_range is not None:
                preliminary_difference = max(
                    abs(force_range[0] - asr_range[0]),
                    abs(force_range[1] - asr_range[1]),
                )
                if preliminary_difference <= boundary_tolerance_ms:
                    # Both timestamp mechanisms have similar published error
                    # scales.  Their midpoint reduces one-sided boundary bias
                    # while the actual-audio ASR still determines the location.
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

            short = normalized_length <= 2
            repeated = _context_is_repeated(
                normalized,
                highlight,
                marked,
                corpus=global_normalized_text,
            )
            force_coverage = force_range[2] if force_range is not None else 0.0
            asr_coverage = asr_range[2] if asr_range is not None else 0.0
            insufficient = (
                force_range is not None and force_coverage < minimum_force_coverage
            ) or (asr_range is not None and asr_coverage < minimum_asr_coverage)
            insufficient = insufficient or (
                force_track_available and force_range is None
            ) or (
                (local_asr_track if has_local_refinement else global_asr_track) is not None
                and asr_range is None
            )
            insufficient = insufficient or observed_context_insufficient
            insufficient = insufficient or (
                force_aligner is not None
                and has_observed_paragraph
                and force_range is None
            )
            if force_range is not None and asr_range is not None:
                boundary_difference = max(
                    abs(force_range[0] - asr_range[0]),
                    abs(force_range[1] - asr_range[1]),
                )
            else:
                boundary_difference = float("inf")
            disagreement = (
                force_range is not None
                and asr_range is not None
                and boundary_difference > boundary_tolerance_ms
            )

            reasons: list[str] = []
            if fallback:
                reasons.append(FALLBACK_ALIGNMENT)
            if not has_observed_paragraph:
                reasons.append(DOCUMENT_TIME_FALLBACK)
            if short:
                reasons.append(SHORT_HIGHLIGHT)
            if repeated:
                reasons.append(REPEATED_CONTEXT)
            if insufficient:
                reasons.append(INSUFFICIENT_COVERAGE)
            if disagreement:
                reasons.append(BOUNDARY_DISAGREEMENT)
            if force_context_unstable:
                reasons.append(FORCE_CONTEXT_UNSTABLE)
            if has_observed_paragraph and not has_local_refinement:
                reasons.append(LOCAL_RECOGNITION_UNAVAILABLE)
            if force_aligner is None:
                reasons.append(FORCE_ALIGNER_UNAVAILABLE)
            if (local_asr_track if has_local_refinement else global_asr_track) is None:
                reasons.append(ASR_ALIGNER_UNAVAILABLE)

            if fallback:
                confidence = 0.15
            elif force_range is not None and asr_range is not None:
                confidence = 0.55 + 0.25 * force_coverage + 0.12 * asr_coverage
                if not disagreement:
                    confidence += 0.08 * max(
                        0.0, 1.0 - boundary_difference / boundary_tolerance_ms
                    )
            elif force_range is not None:
                confidence = 0.45 + 0.30 * force_coverage
            else:
                confidence = 0.25 + 0.30 * asr_coverage
            if short:
                confidence -= 0.20
            if repeated:
                confidence -= 0.12
            if insufficient:
                confidence -= 0.20
            if disagreement:
                confidence -= 0.25
            if force_context_unstable:
                confidence -= 0.12
            confidence = round(max(0.0, min(0.99, confidence)), 4)

            auto_approved = (
                force_range is not None
                and asr_range is not None
                and not fallback
                and not short
                and not repeated
                and not insufficient
                and not disagreement
                and not force_context_unstable
                and has_observed_paragraph
                and has_local_refinement
                and confidence >= auto_approve_threshold
            )
            candidates.append(
                AlignmentCandidate(
                    paragraph_index=paragraph_index,
                    highlight_index=highlight_index,
                    highlighted_text=marked,
                    proposed_start_sample=start_sample,
                    proposed_end_sample=end_sample,
                    confidence=confidence,
                    reasons=reasons,
                    requires_review=not auto_approved,
                    status=STATUS_AUTO_APPROVED if auto_approved else STATUS_NEEDS_REVIEW,
                )
            )
        if progress_cb is not None:
            progress_cb(position + 1, total_paragraphs)
    return candidates


__all__ = [
    "ALIGNMENT_PIPELINE_NAME",
    "ALIGNMENT_PIPELINE_PURPOSE",
    "ALIGNMENT_PIPELINE_VERSION",
    "ASR_ALIGNER_UNAVAILABLE",
    "BOUNDARY_DISAGREEMENT",
    "DOCUMENT_TIME_FALLBACK",
    "FALLBACK_ALIGNMENT",
    "FORCE_CONTEXT_UNSTABLE",
    "FORCE_ALIGNER_UNAVAILABLE",
    "INSUFFICIENT_COVERAGE",
    "LOCAL_RECOGNITION_UNAVAILABLE",
    "MAX_HIGHLIGHT_INTERNAL_GAP_MS",
    "REPEATED_CONTEXT",
    "SHORT_HIGHLIGHT",
    "STATUS_AUTO_APPROVED",
    "STATUS_NEEDS_REVIEW",
    "AlignmentCancelledError",
    "AlignmentCandidate",
    "AlignmentTrack",
    "AsrAligner",
    "ForceAligner",
    "FunASRAsrAligner",
    "FunASRForceAligner",
    "NormalizedText",
    "TimedSpan",
    "align_transcript",
    "alignment_track_from_model_result",
    "normalize_text",
    "normalize_with_mapping",
]
