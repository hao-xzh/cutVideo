"""MLX-backed Qwen3-ASR adapters for local macOS Apple Silicon inference."""

from __future__ import annotations

import gc
import importlib
import math
import platform
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .alignment import (
    TIMESTAMP_PRECISION_CHARACTER,
    TIMESTAMP_PRECISION_SEGMENT,
    TIMESTAMP_PRECISION_TOKEN,
    TIMESTAMP_PRECISION_UNKNOWN,
    AlignmentTrack,
    RecognizedToken,
    TimedSpan,
    alignment_track_from_recognition_tokens,
    normalize_with_mapping,
)
from .model_runtime import (
    ModelUnavailableError,
    local_audio_window,
    model_execution_guard,
    require_local_model,
)

_CACHE_LIMIT_BYTES = 512 * 1024 * 1024
# One 16 kHz PCM sample.  This is only a structural fallback for the aligner's
# quantised zero-length labels; repaired timestamps are explicitly downgraded
# to ``unknown`` and therefore still require boundary verification.
_MIN_SPAN_MS = 1000.0 / 16_000.0
_ASR_ENGINE = "qwen3-asr-mlx"
_FORCE_ENGINE = "qwen3-asr-mlx-forced"


def _get(value: object, *names: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _validate_window(start_ms: object, end_ms: object) -> tuple[int, int]:
    if (
        isinstance(start_ms, bool)
        or isinstance(end_ms, bool)
        or not isinstance(start_ms, (int, float))
        or not isinstance(end_ms, (int, float))
        or not math.isfinite(float(start_ms))
        or not math.isfinite(float(end_ms))
        or int(start_ms) != float(start_ms)
        or int(end_ms) != float(end_ms)
    ):
        raise ValueError("Qwen MLX 音频窗口必须使用整数毫秒")
    start = int(start_ms)
    end = int(end_ms)
    if start < 0 or end <= start:
        raise ValueError("Qwen MLX 音频窗口无效")
    return start, end


def _configure_mlx_runtime() -> Any:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ModelUnavailableError("Qwen MLX 仅支持 macOS arm64")
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except Exception as exc:
        raise ModelUnavailableError("MLX 运行库不可用，请先安装本地 MLX 依赖") from exc
    try:
        metal = mx.metal
        if not bool(metal.is_available()):
            raise ModelUnavailableError("MLX Metal GPU 不可用")
        mx.set_default_device(mx.gpu)
        set_cache_limit = getattr(mx, "set_cache_limit", None)
        if not callable(set_cache_limit):
            set_cache_limit = getattr(metal, "set_cache_limit", None)
        if callable(set_cache_limit):
            set_cache_limit(_CACHE_LIMIT_BYTES)
    except ModelUnavailableError:
        raise
    except Exception as exc:
        raise ModelUnavailableError(f"MLX Metal 初始化失败: {exc}") from exc
    return mx


def _load_qwen_module() -> Any:
    _configure_mlx_runtime()
    try:
        module = importlib.import_module("mlx_qwen3_asr")
    except Exception as exc:
        raise ModelUnavailableError("mlx_qwen3_asr 本地运行库不可用") from exc
    _set_holder_capacity(1)
    return module


def _load_forced_aligner_class() -> type[Any]:
    _load_qwen_module()
    try:
        module = importlib.import_module("mlx_qwen3_asr.forced_aligner")
    except Exception as exc:
        raise ModelUnavailableError("Qwen MLX 强制对齐运行库不可用") from exc
    aligner_type = getattr(module, "ForcedAligner", None)
    if not isinstance(aligner_type, type):
        raise ModelUnavailableError("Qwen MLX 强制对齐器不可用")
    return aligner_type


def _cache_holders() -> tuple[object, ...]:
    holders: list[object] = []
    for module_name, holder_name in (
        ("mlx_qwen3_asr.load_models", "_ModelHolder"),
        ("mlx_qwen3_asr.tokenizer", "_TokenizerHolder"),
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        holder = getattr(module, holder_name, None)
        if holder is not None:
            holders.append(holder)
    return tuple(holders)


def _set_holder_capacity(capacity: int) -> None:
    for holder in _cache_holders():
        setter = getattr(holder, "set_cache_capacity", None)
        if callable(setter):
            try:
                setter(capacity)
            except Exception as exc:
                raise ModelUnavailableError("Qwen MLX 模型缓存配置失败") from exc


def _clear_holder(holder: object) -> None:
    for method_name in ("clear", "clear_cache", "reset", "release"):
        method = getattr(holder, method_name, None)
        if callable(method):
            with suppress(Exception):
                method()
    for attribute in ("_cache", "cache", "_models", "models", "_tokenizers", "tokenizers"):
        cache = getattr(holder, attribute, None)
        if hasattr(cache, "clear"):
            with suppress(Exception):
                cache.clear()


def _release_qwen_runtime() -> None:
    for holder in _cache_holders():
        _clear_holder(holder)
    with suppress(Exception):
        _set_holder_capacity(0)
    try:
        mx = importlib.import_module("mlx.core")
        clear_cache = getattr(mx, "clear_cache", None)
        if not callable(clear_cache):
            clear_cache = getattr(mx.metal, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
    except Exception:
        pass
    gc.collect()


def _result_text(result: object) -> str:
    if isinstance(result, Mapping):
        for key in ("text", "sentence", "transcript"):
            value = result.get(key)
            if value is not None:
                return str(value)
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        return "".join(_result_text(item) for item in result)
    for name in ("text", "sentence", "transcript"):
        value = getattr(result, name, None)
        if value is not None:
            return str(value)
    return str(result) if isinstance(result, str) else ""


def _display_characters(text: str) -> tuple[str, ...]:
    # Punctuation and whitespace have no acoustic interval.  Keeping the same
    # normalisation contract as the existing editor ensures every displayed
    # token can be mapped back to speech.
    return tuple(normalize_with_mapping(text).text)


def _sentence_info_result(
    text: str,
    window_duration_ms: float,
    raw_result: object,
) -> dict[str, object]:
    return {
        "text": text,
        "sentence_info": [
            {
                "text": text,
                "start": 0.0,
                "end": window_duration_ms,
            }
        ],
        "raw": raw_result,
    }


def _uniform_tokens(
    text: str,
    *,
    time_offset_ms: int,
    window_end_ms: int,
) -> tuple[RecognizedToken, ...]:
    characters = _display_characters(text)
    if not characters:
        return ()
    duration = max(float(window_end_ms - time_offset_ms), _MIN_SPAN_MS)
    token_count = len(characters)
    tokens: list[RecognizedToken] = []
    previous_end = float(time_offset_ms)
    for index, character in enumerate(characters):
        start = float(time_offset_ms) + duration * index / token_count
        end = float(time_offset_ms) + duration * (index + 1) / token_count
        if index:
            start = max(start, previous_end)
        if index == token_count - 1:
            end = float(window_end_ms)
        if end <= start:
            end = min(float(window_end_ms), start + _MIN_SPAN_MS)
        if end <= start:
            break
        tokens.append(
            RecognizedToken(
                character,
                start,
                end,
                0.0,
                False,
                TIMESTAMP_PRECISION_SEGMENT,
            )
        )
        previous_end = end
    return tuple(tokens)


def _normal_word_text(value: object) -> str:
    return str(_get(value, "text", "word", "token", "value", default=""))


def _word_times_ms(
    value: object,
    *,
    window_duration_ms: float,
) -> tuple[float, float] | None:
    start_value = _get(value, "start", "start_ms", "begin", "start_time", default=None)
    end_value = _get(value, "end", "end_ms", "stop", "end_time", default=None)
    try:
        start = float(start_value)
        end = float(end_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end):
        return None
    if (
        start <= window_duration_ms / 1000.0 + 1.0
        and end <= window_duration_ms / 1000.0 + 1.0
    ):
        start *= 1000.0
        end *= 1000.0
    return start, end


def _word_confidence(value: object) -> float | None:
    raw = _get(value, "confidence", "score", "probability", "prob", default=None)
    if isinstance(raw, bool):
        return None
    try:
        confidence = float(raw)
    except (TypeError, ValueError):
        return None
    return confidence if math.isfinite(confidence) and 0.0 <= confidence <= 1.0 else None


def _aligned_words(result: object) -> Sequence[object]:
    for name in ("words", "aligned_words", "segments", "spans"):
        value = _get(result, name, default=None)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return value
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        return result
    return ()


def _audio_to_numpy(audio: object) -> object:
    try:
        import numpy as np
    except Exception as exc:
        raise ModelUnavailableError("NumPy 运行库不可用，无法读取 Qwen MLX 音频") from exc
    return np.asarray(audio, dtype=np.float32)


def _load_audio_numpy(audio_path: Path) -> object:
    try:
        audio_module = importlib.import_module("mlx_qwen3_asr.audio")
        load_audio = audio_module.load_audio
        loaded = load_audio(str(audio_path))
    except Exception as exc:
        raise ModelUnavailableError(f"Qwen MLX 音频读取失败: {exc}") from exc
    audio = loaded[0] if isinstance(loaded, tuple) and loaded else loaded
    return _audio_to_numpy(audio)


@dataclass(slots=True)
class _ForceSpanDraft:
    normalized_start: int
    normalized_end: int
    start_ms: float
    end_ms: float
    model_confidence: float | None
    timestamp_precision: str
    repaired: bool = False


def _repair_force_intervals(
    drafts: list[_ForceSpanDraft],
    *,
    window_start_ms: float,
    window_end_ms: float,
) -> tuple[int, int]:
    """Make quantised aligner intervals positive and non-overlapping.

    Qwen's forced aligner uses an 80 ms timestamp grid, so a quickly spoken
    CJK character can legitimately be returned as ``start == end``.  Assign a
    single PCM sample without expanding into both neighbours.  When no free
    gap exists, borrow that sample from the longer adjacent interval and mark
    every changed interval as uncertain.
    """

    repaired_count = 0
    borrowed_count = 0
    for index, draft in enumerate(drafts):
        draft.start_ms = min(window_end_ms, max(window_start_ms, draft.start_ms))
        draft.end_ms = min(window_end_ms, max(window_start_ms, draft.end_ms))
        if index and draft.start_ms < drafts[index - 1].end_ms:
            draft.start_ms = drafts[index - 1].end_ms
            draft.repaired = True
        if draft.end_ms < draft.start_ms:
            draft.end_ms = draft.start_ms
            draft.repaired = True

    for index, draft in enumerate(drafts):
        if draft.end_ms > draft.start_ms:
            continue
        repaired_count += 1
        left = drafts[index - 1] if index else None
        right = drafts[index + 1] if index + 1 < len(drafts) else None
        lower = left.end_ms if left is not None else window_start_ms
        upper = right.start_ms if right is not None else window_end_ms
        point = min(upper, max(lower, draft.start_ms))
        if upper - lower >= _MIN_SPAN_MS:
            start = min(max(lower, point - _MIN_SPAN_MS / 2.0), upper - _MIN_SPAN_MS)
            draft.start_ms = start
            draft.end_ms = start + _MIN_SPAN_MS
            draft.repaired = True
            continue

        left_slack = (
            left.end_ms - left.start_ms - _MIN_SPAN_MS if left is not None else -1.0
        )
        right_slack = (
            right.end_ms - right.start_ms - _MIN_SPAN_MS if right is not None else -1.0
        )
        if right is not None and right_slack >= _MIN_SPAN_MS and right_slack >= left_slack:
            draft.start_ms = right.start_ms
            draft.end_ms = right.start_ms + _MIN_SPAN_MS
            right.start_ms = draft.end_ms
            draft.repaired = True
            right.repaired = True
            borrowed_count += 1
        elif left is not None and left_slack >= _MIN_SPAN_MS:
            draft.end_ms = left.end_ms
            draft.start_ms = left.end_ms - _MIN_SPAN_MS
            left.end_ms = draft.start_ms
            draft.repaired = True
            left.repaired = True
            borrowed_count += 1
        else:
            # This requires every neighbouring interval to be shorter than two
            # PCM samples.  Leave it invalid so the caller can drop coverage
            # instead of fabricating overlapping audio.
            continue

    return repaired_count, borrowed_count


def _build_force_track(
    result: object,
    transcript: str,
    *,
    time_offset_ms: int,
    window_end_ms: int,
) -> AlignmentTrack:
    reference = normalize_with_mapping(transcript)
    if not reference.text:
        return AlignmentTrack((), _FORCE_ENGINE, 0.0, diagnostics=("coverage:0.000",))

    window_duration_ms = float(window_end_ms - time_offset_ms)
    cursor = 0
    covered: set[int] = set()
    drafts: list[_ForceSpanDraft] = []
    diagnostics: list[str] = []
    dropped = 0

    for raw_word in _aligned_words(result):
        text = _normal_word_text(raw_word)
        normalized = normalize_with_mapping(text).text
        if not normalized:
            dropped += 1
            continue
        normalized_start = reference.text.find(normalized, cursor)
        if normalized_start < 0:
            dropped += 1
            continue
        normalized_end = normalized_start + len(normalized)
        cursor = normalized_end

        relative_times = _word_times_ms(raw_word, window_duration_ms=window_duration_ms)
        if relative_times is None:
            dropped += 1
            continue
        relative_start, relative_end = relative_times
        start_ms = min(
            float(window_end_ms),
            max(float(time_offset_ms), float(time_offset_ms) + relative_start),
        )
        end_ms = min(
            float(window_end_ms),
            max(float(time_offset_ms), float(time_offset_ms) + relative_end),
        )
        precision = (
            TIMESTAMP_PRECISION_CHARACTER
            if len(normalized) == 1
            else TIMESTAMP_PRECISION_TOKEN
        )
        drafts.append(
            _ForceSpanDraft(
                normalized_start,
                normalized_end,
                start_ms,
                end_ms,
                _word_confidence(raw_word),
                precision,
            )
        )

    zero_repaired, borrowed_count = _repair_force_intervals(
        drafts,
        window_start_ms=float(time_offset_ms),
        window_end_ms=float(window_end_ms),
    )
    spans: list[TimedSpan] = []
    previous_end = float(time_offset_ms)
    unrepaired_count = 0
    for draft in drafts:
        if draft.end_ms <= draft.start_ms or draft.start_ms < previous_end:
            dropped += 1
            unrepaired_count += 1
            continue
        precision = (
            TIMESTAMP_PRECISION_UNKNOWN
            if draft.repaired
            else draft.timestamp_precision
        )
        spans.append(
            TimedSpan(
                draft.normalized_start,
                draft.normalized_end,
                draft.start_ms,
                draft.end_ms,
                1.0,
                draft.model_confidence,
                precision,
                0.0 if draft.repaired else 1.0,
            )
        )
        covered.update(range(draft.normalized_start, draft.normalized_end))
        previous_end = draft.end_ms

    coverage = len(covered) / len(reference.text)
    model_confidences = [
        span.model_confidence for span in spans if span.model_confidence is not None
    ]
    diagnostics.extend(
        (
            f"coverage:{coverage:.3f}",
            f"zero_repair:{zero_repaired}",
            f"borrowed_sample:{borrowed_count}",
            f"unrepaired:{unrepaired_count}",
            f"drop:{dropped}",
        )
    )
    return AlignmentTrack(
        tuple(spans),
        _FORCE_ENGINE,
        coverage,
        sum(model_confidences) / len(model_confidences) if model_confidences else None,
        min(model_confidences) if model_confidences else None,
        None,
        (
            TIMESTAMP_PRECISION_UNKNOWN
            if zero_repaired
            else TIMESTAMP_PRECISION_CHARACTER
            if spans
            and all(
                span.timestamp_precision == TIMESTAMP_PRECISION_CHARACTER
                for span in spans
            )
            else TIMESTAMP_PRECISION_TOKEN
            if spans
            else TIMESTAMP_PRECISION_SEGMENT
        ),
        bool(spans) and unrepaired_count == 0,
        0.0 if zero_repaired else 1.0,
        tuple(diagnostics),
        (),
        len(model_confidences) / len(spans) if spans else 0.0,
    )


class QwenMlxAsrAligner:
    """Local Qwen3-ASR MLX recognizer with segment-level display timestamps."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        ffmpeg_path: str | Path | None = None,
        language: str = "Chinese",
    ) -> None:
        self.model_path = require_local_model(model_path, "qwen3-asr-mlx")
        _configure_mlx_runtime()
        self.ffmpeg_path = Path(ffmpeg_path).expanduser().resolve() if ffmpeg_path else None
        self.language = language
        self.device = "metal-gpu"
        self._module: Any | None = None

    def _get_module(self) -> Any:
        if self._module is None:
            self._module = _load_qwen_module()
        return self._module

    def preload(self) -> None:
        with model_execution_guard():
            self._get_module()
            model_module = importlib.import_module("mlx_qwen3_asr.load_models")
            holder = getattr(model_module, "_ModelHolder", None)
            get_model = getattr(holder, "get", None)
            if not callable(get_model):
                raise ModelUnavailableError("Qwen MLX 模型加载入口不可用")
            get_model(str(self.model_path))

    def warmup(self) -> None:
        self.preload()

    def load_prepared_audio(self, audio_path: str | Path) -> object:
        prepared = Path(audio_path).expanduser()
        if not prepared.is_file():
            raise ModelUnavailableError(f"音频文件不可用: {prepared}")
        return _load_audio_numpy(prepared.resolve())

    def _transcribe(self, audio: object) -> object:
        try:
            with model_execution_guard():
                module = self._get_module()
                transcribe = getattr(module, "transcribe", None)
                if not callable(transcribe):
                    raise ModelUnavailableError("Qwen MLX 转写入口不可用")
                return transcribe(
                    audio,
                    model=str(self.model_path),
                    language=self.language,
                    return_chunks=False,
                    verbose=False,
                )
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(f"Qwen MLX 本地转写失败: {exc}") from exc

    def recognize_prepared_with_result(
        self,
        *,
        audio_path: str | Path,
        time_offset_ms: int,
        window_end_ms: int,
    ) -> tuple[tuple[RecognizedToken, ...], object]:
        time_offset_ms, window_end_ms = _validate_window(time_offset_ms, window_end_ms)
        prepared = Path(audio_path).expanduser()
        if not prepared.is_file():
            raise ModelUnavailableError(f"音频文件不可用: {prepared}")
        prepared_path = prepared.resolve()
        result = self._transcribe(prepared_path)
        text = _result_text(result)
        tokens = _uniform_tokens(
            text,
            time_offset_ms=time_offset_ms,
            window_end_ms=window_end_ms,
        )
        return (
            tokens,
            _sentence_info_result(text, window_end_ms - time_offset_ms, result),
        )

    def recognize_audio_with_result(
        self,
        *,
        audio: object,
        time_offset_ms: int,
        window_end_ms: int,
    ) -> tuple[tuple[RecognizedToken, ...], object]:
        """Recognize an already decoded 16 kHz mono waveform."""

        time_offset_ms, window_end_ms = _validate_window(time_offset_ms, window_end_ms)
        result = self._transcribe(audio)
        text = _result_text(result)
        tokens = _uniform_tokens(
            text,
            time_offset_ms=time_offset_ms,
            window_end_ms=window_end_ms,
        )
        return (
            tokens,
            _sentence_info_result(text, window_end_ms - time_offset_ms, result),
        )

    def recognize_with_result(
        self,
        *,
        audio_path: str | Path,
        window_start_ms: int,
        window_end_ms: int,
    ) -> tuple[tuple[RecognizedToken, ...], object]:
        window_start_ms, window_end_ms = _validate_window(window_start_ms, window_end_ms)
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

    def recognize(
        self,
        *,
        audio_path: str | Path,
        window_start_ms: int,
        window_end_ms: int,
    ) -> tuple[RecognizedToken, ...]:
        tokens, _result = self.recognize_with_result(
            audio_path=audio_path,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
        )
        return tokens

    def align_prepared(
        self,
        *,
        audio_path: str | Path,
        transcript: str,
        time_offset_ms: int,
        window_end_ms: int,
    ) -> AlignmentTrack:
        tokens, raw_result = self.recognize_prepared_with_result(
            audio_path=audio_path,
            time_offset_ms=time_offset_ms,
            window_end_ms=window_end_ms,
        )
        vad_ranges = ()
        try:
            from .alignment import voice_ranges_from_model_result

            vad_ranges = voice_ranges_from_model_result(
                raw_result,
                time_offset_ms=time_offset_ms,
                window_end_ms=window_end_ms,
            )
        except Exception:
            vad_ranges = ()
        try:
            return alignment_track_from_recognition_tokens(
                tokens,
                transcript,
                engine=_ASR_ENGINE,
                vad_ranges=vad_ranges,
            )
        except Exception as exc:
            raise ModelUnavailableError(f"Qwen MLX 识别结果映射失败: {exc}") from exc

    def align(
        self,
        *,
        audio_path: str | Path,
        transcript: str,
        window_start_ms: int,
        window_end_ms: int,
    ) -> AlignmentTrack:
        window_start_ms, window_end_ms = _validate_window(window_start_ms, window_end_ms)
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

    def release(self) -> None:
        self._module = None
        _release_qwen_runtime()


class QwenMlxForceAligner:
    """Local Qwen3-ASR MLX forced-alignment adapter."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        ffmpeg_path: str | Path | None = None,
        language: str = "Chinese",
    ) -> None:
        self.model_path = require_local_model(model_path, "qwen3-asr-mlx-forced")
        _configure_mlx_runtime()
        self.ffmpeg_path = Path(ffmpeg_path).expanduser().resolve() if ffmpeg_path else None
        self.language = language
        self.device = "metal-gpu"
        self._aligner: Any | None = None

    def _get_aligner(self) -> Any:
        if self._aligner is None:
            aligner_type = _load_forced_aligner_class()
            try:
                self._aligner = aligner_type(model=str(self.model_path), device=self.device)
            except TypeError:
                try:
                    self._aligner = aligner_type(str(self.model_path))
                except TypeError:
                    self._aligner = aligner_type(model_path=str(self.model_path))
        return self._aligner

    def align_prepared(
        self,
        *,
        audio_path: str | Path,
        transcript: str,
        time_offset_ms: int,
        window_end_ms: int,
    ) -> AlignmentTrack:
        time_offset_ms, window_end_ms = _validate_window(time_offset_ms, window_end_ms)
        prepared = Path(audio_path).expanduser()
        if not prepared.is_file():
            raise ModelUnavailableError(f"音频文件不可用: {prepared}")
        prepared_path = prepared.resolve()
        try:
            with model_execution_guard():
                audio = _load_audio_numpy(prepared_path)
                result = self._get_aligner().align(
                    audio,
                    transcript,
                    self.language,
                )
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(f"Qwen MLX 强制对齐失败: {exc}") from exc
        try:
            return _build_force_track(
                result,
                transcript,
                time_offset_ms=time_offset_ms,
                window_end_ms=window_end_ms,
            )
        except Exception as exc:
            raise ModelUnavailableError(f"Qwen MLX 强制对齐结果转换失败: {exc}") from exc

    def align(
        self,
        *,
        audio_path: str | Path,
        transcript: str,
        window_start_ms: int,
        window_end_ms: int,
    ) -> AlignmentTrack:
        window_start_ms, window_end_ms = _validate_window(window_start_ms, window_end_ms)
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

    def release(self) -> None:
        self._aligner = None
        _release_qwen_runtime()


__all__ = [
    "QwenMlxAsrAligner",
    "QwenMlxForceAligner",
]
