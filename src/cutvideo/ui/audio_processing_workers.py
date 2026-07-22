"""Background operations for the standalone transcript-driven audio workspace."""

from __future__ import annotations

import hashlib
import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from ..alignment import (
    ForceAligner,
    FunASRForceAligner,
    RecognizedToken,
    normalize_with_mapping,
    refine_recognition_tokens,
)
from ..audio import (
    BOUNDARY_GUARD_SAFETY_MS,
    BOUNDARY_MAX_EXPAND_MS,
    BOUNDARY_SEARCH_MS,
    BOUNDARY_ZERO_CROSSING_MS,
    AudioInfo,
    WaveformEnvelope,
    decode_f32,
    probe_audio_metadata,
    probe_audio_with_waveform,
    refine_cut_boundaries,
)
from ..audio_processing import (
    TRANSCRIPT_SEGMENTATION_STRATEGY,
    AudioProcessingProject,
    TranscriptToken,
    default_audio_processing_project_path,
    infer_transcript_segment_starts,
    load_audio_processing_project,
    save_audio_processing_project,
)
from ..ffmpeg import FFmpegTools, discover_ffmpeg, export_audio, generate_preview
from ..model_runtime import model_execution_guard, resolve_inference_device
from ..progressive_asr import (
    ProgressiveRecognitionChunk,
    recognize_audio_progressively,
)
from ..project import AudioInfo as ProjectAudioInfo
from ..project import SourceFile
from ..resources import (
    RuntimeResources,
    discover_resources,
    find_resource_root,
    platform_key,
)
from .workers import (
    AsrTranscriptPartial,
    CancelToken,
    CandidatePreviewResult,
    Operation,
    ProgressReporter,
    _commit_artifacts,
)

_MAX_CROSS_VAD_TIMELINE_REPAIR_MS = 500.0
_ANNOTATION_FORCE_PADDING_MS = 350
_ANNOTATION_FORCE_MAX_SHIFT_MS = 400.0
_ANNOTATION_FORCE_MIN_COVERAGE = 0.80


def _force_aligner_for_resources(
    resources: RuntimeResources,
    *,
    ffmpeg_path: str | Path,
) -> ForceAligner | None:
    if resources.has_qwen_models and resources.qwen_force_model is not None:
        from ..qwen_mlx import QwenMlxForceAligner

        return QwenMlxForceAligner(
            resources.qwen_force_model,
            ffmpeg_path=ffmpeg_path,
        )
    if resources.fa_model is not None:
        return FunASRForceAligner(
            resources.fa_model,
            ffmpeg_path=ffmpeg_path,
            device=resolve_inference_device(),
        )
    return None


def _release_aligner(aligner: object) -> None:
    release = getattr(aligner, "release", None)
    if callable(release):
        release()


def _voice_ranges_ms_from_diagnostics(
    diagnostics: Mapping[str, object] | None,
) -> tuple[tuple[float, float], ...]:
    if not isinstance(diagnostics, Mapping):
        return ()
    raw = diagnostics.get("voice_ranges_ms", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    ranges: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 2:
            continue
        try:
            start_ms = float(item[0])
            end_ms = float(item[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(start_ms) and math.isfinite(end_ms) and end_ms > start_ms:
            ranges.append((start_ms, end_ms))
    return tuple(ranges)


def _boundary_indexes_from_diagnostics(
    diagnostics: Mapping[str, object] | None,
    key: str,
    *,
    token_count: int,
) -> tuple[int, ...]:
    if not isinstance(diagnostics, Mapping):
        return ()
    raw = diagnostics.get(key, ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(
        sorted(
            {
                value
                for value in raw
                if isinstance(value, int)
                and not isinstance(value, bool)
                and 0 < value < token_count
            }
        )
    )


def _vad_samples_covering_interval(
    voice_ranges_ms: Sequence[tuple[float, float]],
    *,
    start_sample: int,
    end_sample: int,
    sample_rate: int,
) -> tuple[int | None, int | None]:
    if sample_rate <= 0 or end_sample <= start_sample:
        return None, None
    start_ms = start_sample * 1000.0 / sample_rate
    end_ms = end_sample * 1000.0 / sample_rate
    midpoint = (start_ms + end_ms) / 2.0
    for vad_start_ms, vad_end_ms in voice_ranges_ms:
        if vad_start_ms - 1e-6 <= midpoint <= vad_end_ms + 1e-6:
            return (
                max(0, round(vad_start_ms * sample_rate / 1000.0)),
                max(1, round(vad_end_ms * sample_rate / 1000.0)),
            )
    return None, None


def _force_align_annotation_bounds(
    *,
    force_aligner: ForceAligner,
    audio_path: str | Path,
    selected_text: str,
    start_sample: int,
    end_sample: int,
    sample_rate: int,
    total_samples: int,
    left_guard_sample: int | None,
    right_guard_sample: int | None,
) -> tuple[int, int, dict[str, object]] | None:
    """Re-align the selected annotation text inside a short local window."""

    normalized = normalize_with_mapping(selected_text).text
    if not normalized or sample_rate <= 0:
        return None
    audio_end_ms = max(1, math.ceil(total_samples * 1000 / sample_rate))
    original_start_ms = start_sample * 1000.0 / sample_rate
    original_end_ms = end_sample * 1000.0 / sample_rate
    window_start_ms = max(0, int(original_start_ms) - _ANNOTATION_FORCE_PADDING_MS)
    window_end_ms = min(
        audio_end_ms,
        int(math.ceil(original_end_ms)) + _ANNOTATION_FORCE_PADDING_MS,
    )
    if window_end_ms <= window_start_ms:
        return None
    try:
        track = force_aligner.align(
            audio_path=audio_path,
            transcript=selected_text,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
        )
    except Exception:
        return None
    if (
        not track.timestamp_valid
        or track.coverage < _ANNOTATION_FORCE_MIN_COVERAGE
        or not track.spans
    ):
        return None
    aligned_start_ms = min(span.start_ms for span in track.spans)
    aligned_end_ms = max(span.end_ms for span in track.spans)
    if (
        aligned_end_ms <= aligned_start_ms
        or aligned_start_ms < original_start_ms - _ANNOTATION_FORCE_MAX_SHIFT_MS
        or aligned_end_ms > original_end_ms + _ANNOTATION_FORCE_MAX_SHIFT_MS
    ):
        return None
    safety = max(1, round(sample_rate * BOUNDARY_GUARD_SAFETY_MS / 1000.0))
    refined_start = max(0, min(total_samples - 1, round(aligned_start_ms * sample_rate / 1000.0)))
    refined_end = max(
        refined_start + 1,
        min(total_samples, round(aligned_end_ms * sample_rate / 1000.0)),
    )
    if left_guard_sample is not None:
        refined_start = max(refined_start, int(left_guard_sample) + safety)
    if right_guard_sample is not None:
        refined_end = min(refined_end, int(right_guard_sample) - safety)
    if not 0 <= refined_start < refined_end <= total_samples:
        return None
    return (
        refined_start,
        refined_end,
        {
            "engine": type(force_aligner).__name__,
            "inference_device": str(getattr(force_aligner, "device", "unknown")),
            "coverage": round(track.coverage, 4),
            "window_start_ms": window_start_ms,
            "window_end_ms": window_end_ms,
            "aligned_start_ms": round(aligned_start_ms, 3),
            "aligned_end_ms": round(aligned_end_ms, 3),
        },
    )


def _source_stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        getattr(stat, "st_ino", 0),
    )


@dataclass(slots=True)
class AudioProcessingAnalysisResult:
    project: AudioProcessingProject
    project_path: Path
    audio_info: AudioInfo
    tools: FFmpegTools
    waveform_envelope: WaveformEnvelope | None = None
    source_matches: bool = True
    refinement_pending: bool = False


@dataclass(slots=True)
class AudioProcessingRefinementResult:
    project: AudioProcessingProject
    project_path: Path
    updated_token_count: int
    character_refinement: dict[str, object] | None = None


@dataclass(slots=True)
class AudioProcessingExportResult:
    wav_path: Path
    mp3_path: Path
    project_path: Path
    kept_samples: int
    removed_samples: int


@dataclass(slots=True)
class AnnotationBoundaryRefinementResult:
    annotation_id: str
    initial_start_sample: int
    initial_end_sample: int
    refined_start_sample: int
    refined_end_sample: int
    evidence_complete: bool
    diagnostics: dict[str, object]


def _normalize_recognized_token_timeline(
    tokens: Sequence[RecognizedToken],
    *,
    audio_end_ms: float,
) -> tuple[tuple[RecognizedToken, ...], dict[str, object]]:
    """Keep transcript order while repairing cross-VAD timestamp overlap.

    FunASR joins independently decoded VAD slices whose look-back/look-ahead
    regions can overlap.  Their text order is authoritative, but a token at the
    next slice boundary can therefore start or end slightly before its
    predecessor.  Project storage intentionally requires a monotonic timeline.
    Clamp only the regressing edge, retain overlaps, and downgrade corrected
    timestamps so the UI never presents a repaired interval as character-exact.
    """

    if not tokens:
        return (), {
            "strategy": "preserve_model_order_clamp_cross_vad_overlap",
            "input_token_count": 0,
            "adjusted_token_count": 0,
        }
    if not math.isfinite(audio_end_ms) or audio_end_ms <= 0:
        raise ValueError("invalid recognition timeline duration")

    end_limit = float(audio_end_ms)
    minimum_span = min(1.0, end_limit)
    maximum_start = max(0.0, end_limit - minimum_span)
    previous_start = 0.0
    previous_end = 0.0
    normalized: list[RecognizedToken] = []
    adjusted_count = 0
    start_adjustment_count = 0
    end_adjustment_count = 0
    maximum_start_adjustment_ms = 0.0
    maximum_end_adjustment_ms = 0.0

    for token in tokens:
        bounded_start = min(max(0.0, token.start_ms), maximum_start)
        bounded_end = min(end_limit, max(token.end_ms, bounded_start + minimum_span))
        start_ms = max(previous_start, bounded_start)
        end_ms = min(
            end_limit,
            max(previous_end, bounded_end, start_ms + minimum_span),
        )
        # ``start_ms`` can only reach ``end_limit`` when the input duration is
        # sub-millisecond.  Keep even that degenerate fixture representable.
        if end_ms <= start_ms:
            start_ms = maximum_start
            end_ms = end_limit

        start_adjustment = abs(start_ms - token.start_ms)
        end_adjustment = abs(end_ms - token.end_ms)
        # A coarse word/segment token may legitimately start well before the
        # preceding character while still advancing its end.  Only a large end
        # regression proves that the whole timeline moved backwards.
        if end_adjustment > _MAX_CROSS_VAD_TIMELINE_REPAIR_MS:
            raise ValueError(
                "识别时间轴跨段回退超过 500 毫秒；已停止写入，避免生成错位音频标注"
            )
        adjusted = start_adjustment > 1e-6 or end_adjustment > 1e-6
        if adjusted:
            adjusted_count += 1
            start_adjustment_count += start_adjustment > 1e-6
            end_adjustment_count += end_adjustment > 1e-6
            maximum_start_adjustment_ms = max(
                maximum_start_adjustment_ms,
                start_adjustment,
            )
            maximum_end_adjustment_ms = max(
                maximum_end_adjustment_ms,
                end_adjustment,
            )
        normalized.append(
            RecognizedToken(
                token.text,
                start_ms,
                end_ms,
                token.confidence,
                token.confidence_available,
                "unknown" if adjusted else token.timestamp_precision,
            )
        )
        previous_start = start_ms
        previous_end = end_ms

    return tuple(normalized), {
        "strategy": "preserve_model_order_clamp_cross_vad_overlap",
        "input_token_count": len(tokens),
        "adjusted_token_count": adjusted_count,
        "start_adjustment_count": start_adjustment_count,
        "end_adjustment_count": end_adjustment_count,
        "precision_downgraded_count": adjusted_count,
        "maximum_start_adjustment_ms": round(maximum_start_adjustment_ms, 3),
        "maximum_end_adjustment_ms": round(maximum_end_adjustment_ms, 3),
    }


def _segment_starts_from_voice_ranges(
    tokens: list[TranscriptToken],
    voice_ranges: Sequence[tuple[float, float]],
    sample_rate: int,
    *,
    sentence_boundary_indexes: Sequence[int] = (),
    clause_boundary_indexes: Sequence[int] = (),
) -> list[int]:
    return infer_transcript_segment_starts(
        tokens,
        sample_rate,
        voice_ranges_ms=voice_ranges,
        sentence_boundary_indexes=sentence_boundary_indexes,
        clause_boundary_indexes=clause_boundary_indexes,
    )


def _transcript_tokens_from_recognized(
    recognized: Sequence[RecognizedToken],
    *,
    sample_rate: int,
    total_samples: int,
) -> list[TranscriptToken]:
    tokens: list[TranscriptToken] = []
    for item in recognized:
        start = max(
            0,
            min(total_samples - 1, round(item.start_ms * sample_rate / 1000)),
        )
        end = max(
            start + 1,
            min(total_samples, round(item.end_ms * sample_rate / 1000)),
        )
        tokens.append(
            TranscriptToken(
                item.text,
                start,
                end,
                item.confidence,
                item.confidence_available,
                item.timestamp_precision,
            )
        )
    return tokens


def _recognized_tokens_from_transcript(
    tokens: Sequence[TranscriptToken],
    *,
    sample_rate: int,
) -> tuple[RecognizedToken, ...]:
    return tuple(
        RecognizedToken(
            item.text,
            item.start_sample * 1000 / sample_rate,
            item.end_sample * 1000 / sample_rate,
            item.confidence,
            item.confidence_available,
            item.timestamp_precision,
        )
        for item in tokens
    )


def make_audio_processing_analysis_operation(
    audio_path: str,
    partial_cb: Callable[[AsrTranscriptPartial], None] | None = None,
) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> AudioProcessingAnalysisResult:
        source_path = Path(audio_path).expanduser().resolve(strict=True)
        resources = discover_resources()
        if not resources.has_ffmpeg:
            raise FileNotFoundError("缺少本平台的 FFmpeg 离线资源")
        recognition_backend = resources.recognition_backend
        if recognition_backend == "qwen-mlx":
            recognition_model = resources.qwen_asr_model
            recognition_vad_model = None
        elif recognition_backend == "funasr":
            recognition_model = resources.asr_model
            recognition_vad_model = resources.vad_model
        else:
            raise FileNotFoundError("缺少可用的本地离线识别模型")
        assert recognition_model is not None
        tools = FFmpegTools(resources.ffmpeg, resources.ffprobe)  # type: ignore[arg-type]
        inference_device = resolve_inference_device()
        report(0.03, "正在核对音频文件…")
        initial_stamp = _source_stamp(source_path)
        waveform_envelope: WaveformEnvelope | None = None
        report(0.05, "正在快速读取音频时长…")
        try:
            planning_info = probe_audio_metadata(source_path, tools=tools)
        except (ValueError, OSError):
            # Unusual containers without duration metadata retain the safe
            # exact-decode path.  Common formats start ASR immediately after
            # the cheap ffprobe call above.
            planning_info, waveform_envelope = probe_audio_with_waveform(
                source_path,
                tools=tools,
                progress_cb=lambda value: report(
                    0.05 + value * 0.16,
                    "正在建立真实样本时间轴…",
                ),
                cancel=token,
            )
        token.raise_if_cancelled()
        planning_duration_ms = max(
            1,
            math.ceil(planning_info.duration_seconds * 1000),
        )
        emitted_token_count = 0

        def publish_chunk(chunk: ProgressiveRecognitionChunk) -> None:
            nonlocal emitted_token_count
            emitted_token_count += len(chunk.tokens)
            if partial_cb is not None:
                partial_cb(
                    AsrTranscriptPartial(
                        "".join(item.text for item in chunk.tokens),
                        chunk.sequence,
                        chunk.total_sequences,
                        chunk.committed_until_ms,
                        chunk.duration_ms,
                        emitted_token_count,
                    )
                )

        progressive = recognize_audio_progressively(
            audio_path=source_path,
            duration_ms=planning_duration_ms,
            asr_model_path=recognition_model,
            vad_model_path=recognition_vad_model,
            ffmpeg_path=tools.ffmpeg,
            progress_cb=lambda value, message: report(
                0.08 + value * 0.72,
                message,
            ),
            chunk_cb=publish_chunk,
            cancel=token,
            device=inference_device,
            backend=recognition_backend,
        )
        actual_inference_device = (
            progressive.inference_device
            if progressive.inference_device != "unknown"
            else inference_device
        )
        recognized = progressive.tokens
        if not recognized:
            raise ValueError("没有识别出可编辑文字；请确认音频包含清晰人声")
        if emitted_token_count != len(recognized):
            raise ValueError("流式转写与最终文本不一致；已停止保存，避免生成错位项目")
        if _source_stamp(source_path) != initial_stamp:
            raise ValueError("识别过程中音频文件发生变化，请重新开始")

        report(0.82, "文字已流式输出，正在建立精确 PCM 时间轴…")
        source = SourceFile.from_path(source_path)
        if _source_stamp(source_path) != initial_stamp:
            raise ValueError("计算文件指纹时音频发生变化，请重新开始")
        if waveform_envelope is None:
            info, waveform_envelope = probe_audio_with_waveform(
                source_path,
                tools=tools,
                progress_cb=lambda value: report(
                    0.82 + value * 0.10,
                    "文字已输出，正在生成完整波形…",
                ),
                cancel=token,
            )
        else:
            info = planning_info
        token.raise_if_cancelled()
        duration_ms = max(1, math.ceil(info.total_samples * 1000 / info.sample_rate))
        recognized, timeline_before_refinement = _normalize_recognized_token_timeline(
            recognized,
            audio_end_ms=duration_ms,
        )
        # Deliver the streamed text as soon as ASR and the PCM timeline are
        # ready.  Character sharpening continues separately; text-based cut
        # actions stay gated until it completes so a coarse draft cannot be
        # mistaken for a verified audio boundary.
        refinement_pending = (
            resources.qwen_force_model is not None or resources.fa_model is not None
        ) and bool(recognized)
        tokens = _transcript_tokens_from_recognized(
            recognized,
            sample_rate=info.sample_rate,
            total_samples=info.total_samples,
        )
        report(0.94, "正在按人声停顿建立分段…")
        voice_ranges = tuple(
            (max(0.0, start_ms), min(float(duration_ms), end_ms))
            for start_ms, end_ms in progressive.voice_ranges
            if start_ms < duration_ms and end_ms > 0
        )
        segment_starts = _segment_starts_from_voice_ranges(
            tokens,
            voice_ranges,
            info.sample_rate,
            sentence_boundary_indexes=progressive.sentence_boundary_indexes,
            clause_boundary_indexes=progressive.clause_boundary_indexes,
        )
        if not source.matches_file(source_path):
            raise ValueError("识别过程中音频文件发生变化，请重新开始")
        precision_counts: dict[str, int] = {}
        for item in tokens:
            precision_counts[item.timestamp_precision] = (
                precision_counts.get(item.timestamp_precision, 0) + 1
            )
        confidence_count = sum(item.confidence_available for item in tokens)
        project = AudioProcessingProject(
            audio=source,
            audio_info=ProjectAudioInfo(
                sample_rate=info.sample_rate,
                channels=info.channels,
                total_samples=info.total_samples,
                format_name=source_path.suffix.lstrip(".").lower(),
                codec_name=info.codec or "",
            ),
            tokens=tokens,
            segment_starts=segment_starts,
            analysis_diagnostics={
                "timestamp_unit": "pcm_sample",
                "accepted_timestamp_count": len(tokens),
                "timestamp_precision_counts": precision_counts,
                "raw_character_confidence_count": confidence_count,
                "raw_character_confidence_coverage": round(
                    confidence_count / len(tokens),
                    4,
                ),
                "vad_range_count": len(voice_ranges),
                "voice_ranges_ms": [
                    [round(start_ms, 3), round(end_ms, 3)]
                    for start_ms, end_ms in voice_ranges
                ],
                "coarse_timestamps_are_not_interpolated": True,
                "recognition_delivery": (
                    "progressive_local_energy_chunks"
                    if recognition_backend == "qwen-mlx"
                    else "progressive_overlapping_windows"
                ),
                "recognition_backend": recognition_backend,
                "recognition_chunk_count": progressive.chunk_count,
                "recognition_streamed_token_count": emitted_token_count,
                "recognition_deduplicated_token_count": (
                    progressive.deduplicated_token_count
                ),
                "recognition_sentence_boundary_indexes": list(
                    progressive.sentence_boundary_indexes
                ),
                "recognition_clause_boundary_indexes": list(
                    progressive.clause_boundary_indexes
                ),
                "semantic_segmentation_source": (
                    "qwen_raw_punctuation_v1"
                    if progressive.sentence_boundary_indexes
                    or progressive.clause_boundary_indexes
                    else "acoustic_pause_fallback"
                ),
                "recognition_sentence_boundary_count": len(
                    progressive.sentence_boundary_indexes
                ),
                "recognition_clause_boundary_count": len(
                    progressive.clause_boundary_indexes
                ),
                "segmentation_strategy": TRANSCRIPT_SEGMENTATION_STRATEGY,
                "segmentation_uses_vad_and_token_gaps": True,
                "segmentation_uses_model_punctuation": bool(
                    progressive.sentence_boundary_indexes
                    or progressive.clause_boundary_indexes
                ),
                "inference_device": actual_inference_device,
                "character_refinement_status": (
                    "pending" if refinement_pending else "skipped"
                ),
                "timeline_normalization": {
                    "before_character_refinement": timeline_before_refinement,
                },
            },
            output_directory=str(source_path.parent),
        )
        project_path = default_audio_processing_project_path(source_path)
        report(
            0.97,
            "文字已输出，正在准备逐字边界…"
            if refinement_pending
            else "转写与逐字边界已就绪…",
        )
        save_audio_processing_project(project, project_path)
        report(
            1.0,
            "文字已输出，正在后台完成逐字对齐"
            if refinement_pending
            else "转写与逐字边界已就绪",
        )
        return AudioProcessingAnalysisResult(
            project,
            project_path,
            info,
            tools,
            waveform_envelope=waveform_envelope,
            refinement_pending=refinement_pending,
        )

    return operation


def make_audio_processing_refinement_operation(
    project: AudioProcessingProject,
    project_path: str | Path,
    *,
    audio_info: AudioInfo,
    tools: FFmpegTools,
) -> Operation:
    def operation(
        token: CancelToken,
        report: ProgressReporter,
    ) -> AudioProcessingRefinementResult:
        resources = discover_resources()
        force_aligner = _force_aligner_for_resources(
            resources,
            ffmpeg_path=tools.ffmpeg,
        )
        if force_aligner is None:
            raise FileNotFoundError("缺少本地逐字时间戳模型")
        source_path = Path(project.audio.path).expanduser().resolve(strict=True)
        if not project.audio.matches_file(source_path):
            raise ValueError("精修开始前原音频已变化，请重新识别")
        duration_ms = max(
            1,
            math.ceil(audio_info.total_samples * 1000 / audio_info.sample_rate),
        )
        draft_tokens = _recognized_tokens_from_transcript(
            project.tokens,
            sample_rate=audio_info.sample_rate,
        )
        report(0.05, "文字已输出，正在后台细化逐字时间戳…")
        try:
            with model_execution_guard(token):
                refined, character_refinement = refine_recognition_tokens(
                    draft_tokens,
                    force_aligner=force_aligner,
                    audio_path=source_path,
                    audio_end_ms=duration_ms,
                    window_padding_ms=(0 if resources.has_qwen_models else 200),
                    max_boundary_shift_ms=(
                        30_000.0 if resources.has_qwen_models else 350.0
                    ),
                    progress_cb=lambda done, total: report(
                        0.05 + 0.85 * done / max(1, total),
                        f"正在后台细化逐字时间戳 {done}/{total}…",
                    ),
                    cancel=token,
                )
            actual_inference_device = str(
                getattr(force_aligner, "device", "unknown")
            )
        finally:
            _release_aligner(force_aligner)
        refined, timeline_after_refinement = _normalize_recognized_token_timeline(
            refined,
            audio_end_ms=duration_ms,
        )
        token.raise_if_cancelled()
        if not project.audio.matches_file(source_path):
            raise ValueError("精修过程中原音频已变化，请重新识别")
        transcript_tokens = _transcript_tokens_from_recognized(
            refined,
            sample_rate=audio_info.sample_rate,
            total_samples=audio_info.total_samples,
        )
        working = AudioProcessingProject.from_dict(project.to_dict())
        diagnostics = {
            **(character_refinement or {}),
            "inference_device": actual_inference_device,
        }
        timeline = dict(working.analysis_diagnostics.get("timeline_normalization", {}))
        timeline["after_character_refinement"] = timeline_after_refinement
        working.analysis_diagnostics["timeline_normalization"] = timeline
        working.analysis_diagnostics["inference_device"] = actual_inference_device
        updated = working.apply_character_timestamp_refinement(
            transcript_tokens,
            diagnostics=diagnostics,
        )
        working.segment_starts = infer_transcript_segment_starts(
            working.tokens,
            audio_info.sample_rate,
            voice_ranges_ms=_voice_ranges_ms_from_diagnostics(
                working.analysis_diagnostics
            ),
            sentence_boundary_indexes=_boundary_indexes_from_diagnostics(
                working.analysis_diagnostics,
                "recognition_sentence_boundary_indexes",
                token_count=len(working.tokens),
            ),
            clause_boundary_indexes=_boundary_indexes_from_diagnostics(
                working.analysis_diagnostics,
                "recognition_clause_boundary_indexes",
                token_count=len(working.tokens),
            ),
        )
        working.analysis_diagnostics["segmentation_strategy"] = (
            TRANSCRIPT_SEGMENTATION_STRATEGY
        )
        report(0.96, "正在保存细化后的时间戳…")
        destination = Path(project_path).expanduser().resolve()
        save_audio_processing_project(working, destination)
        report(1.0, "逐字时间戳已细化完成")
        return AudioProcessingRefinementResult(
            working,
            destination,
            updated,
            diagnostics,
        )

    return operation


def make_audio_processing_load_operation(project_path: str) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> AudioProcessingAnalysisResult:
        token.raise_if_cancelled()
        path = Path(project_path).expanduser().resolve(strict=True)
        report(0.10, "正在打开音频处理项目…")
        project = load_audio_processing_project(path)
        if (
            project.analysis_diagnostics.get("segmentation_strategy")
            != TRANSCRIPT_SEGMENTATION_STRATEGY
        ):
            sentence_boundary_indexes = _boundary_indexes_from_diagnostics(
                project.analysis_diagnostics,
                "recognition_sentence_boundary_indexes",
                token_count=len(project.tokens),
            )
            clause_boundary_indexes = _boundary_indexes_from_diagnostics(
                project.analysis_diagnostics,
                "recognition_clause_boundary_indexes",
                token_count=len(project.tokens),
            )
            project.segment_starts = infer_transcript_segment_starts(
                project.tokens,
                project.audio_info.sample_rate,
                voice_ranges_ms=_voice_ranges_ms_from_diagnostics(
                    project.analysis_diagnostics
                ),
                sentence_boundary_indexes=sentence_boundary_indexes,
                clause_boundary_indexes=clause_boundary_indexes,
            )
            project.analysis_diagnostics["segmentation_strategy"] = (
                TRANSCRIPT_SEGMENTATION_STRATEGY
            )
            if (
                project.analysis_diagnostics.get("recognition_backend") == "qwen-mlx"
                and not sentence_boundary_indexes
                and not clause_boundary_indexes
            ):
                project.analysis_diagnostics[
                    "semantic_segmentation_upgrade_required"
                ] = True
        source_matches = project.audio.matches_file(project.audio.path)
        tools = discover_ffmpeg(resource_root=find_resource_root())
        if source_matches:
            info, waveform_envelope = probe_audio_with_waveform(
                project.audio.path,
                tools=tools,
                cancel=token,
            )
            if (
                info.sample_rate != project.audio_info.sample_rate
                or info.total_samples != project.audio_info.total_samples
            ):
                raise ValueError("项目音频的 PCM 样本时间轴不匹配")
        else:
            waveform_envelope = None
            info = AudioInfo(
                path=Path(project.audio.path),
                sample_rate=project.audio_info.sample_rate,
                channels=project.audio_info.channels,
                total_samples=project.audio_info.total_samples,
                duration_seconds=project.audio_info.duration_seconds,
                codec=project.audio_info.codec_name or None,
            )
        refinement_pending = bool(
            source_matches
            and project.tokens
            and project.analysis_diagnostics.get("character_refinement_status")
            != "completed"
        )
        if refinement_pending:
            project.analysis_diagnostics["character_refinement_status"] = "pending"
        report(1.0, "项目已打开")
        return AudioProcessingAnalysisResult(
            project,
            path,
            info,
            tools,
            waveform_envelope=waveform_envelope,
            source_matches=source_matches,
            refinement_pending=refinement_pending,
        )

    return operation


def make_audio_processing_relink_operation(
    project_path: str | Path,
    audio_path: str,
) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> AudioProcessingAnalysisResult:
        path = Path(project_path).expanduser().resolve(strict=True)
        project = load_audio_processing_project(path)
        old_audio_parent = Path(project.audio.path).parent
        old_output = Path(project.output_directory)
        report(0.10, "正在核对重新选择的音频指纹…")
        project.audio.relink(audio_path)
        if old_output == old_audio_parent:
            project.output_directory = str(Path(project.audio.path).parent)
        tools = discover_ffmpeg(resource_root=find_resource_root())
        report(0.45, "正在核对音频时间轴…")
        info, waveform_envelope = probe_audio_with_waveform(
            project.audio.path,
            tools=tools,
            cancel=token,
        )
        if (
            info.sample_rate != project.audio_info.sample_rate
            or info.total_samples != project.audio_info.total_samples
        ):
            raise ValueError("重新选择的音频 PCM 时间轴与项目不匹配")
        save_audio_processing_project(project, path)
        refinement_pending = bool(
            project.tokens
            and project.analysis_diagnostics.get("character_refinement_status")
            != "completed"
        )
        if refinement_pending:
            project.analysis_diagnostics["character_refinement_status"] = "pending"
        report(1.0, "音频已重新关联")
        return AudioProcessingAnalysisResult(
            project,
            path,
            info,
            tools,
            waveform_envelope=waveform_envelope,
            source_matches=True,
            refinement_pending=refinement_pending,
        )

    return operation


def make_annotation_boundary_refinement_operation(
    *,
    audio_path: str,
    info: AudioInfo,
    tools: FFmpegTools,
    annotation_id: str,
    start_sample: int,
    end_sample: int,
    left_guard_sample: int | None = None,
    right_guard_sample: int | None = None,
    selected_text: str = "",
    voice_ranges_ms: Sequence[tuple[float, float]] = (),
    enable_force_realign: bool = True,
) -> Operation:
    """Snap one delete annotation via verified timestamps and acoustic edges."""

    def operation(token: CancelToken, report: ProgressReporter) -> AnnotationBoundaryRefinementResult:
        report(0.05, "正在细化删除范围的文字/声学边界…")
        unrefined = AnnotationBoundaryRefinementResult(
            annotation_id,
            start_sample,
            end_sample,
            start_sample,
            end_sample,
            False,
            {"fallback": "refinement_window_unavailable"},
        )
        if end_sample - start_sample < 3:
            return unrefined
        working_start = start_sample
        working_end = end_sample
        force_diagnostics: dict[str, object] | None = None
        if (
            enable_force_realign
            and selected_text.strip()
            and platform_key() != "macos-arm64"
        ):
            resources = discover_resources()
            # Qwen already aligned the complete transcript with neighbouring
            # context.  Re-aligning only the selected phrase inside a padded
            # window would let adjacent speech pull its timestamps outward.
            # Keep that globally anchored interval and only apply the acoustic
            # edge pass below.  The legacy FunASR path still needs this local
            # sharpening step.
            if not resources.has_qwen_models and resources.fa_model is not None:
                report(0.15, "正在对选中文字做局部逐字对齐…")
                force_aligner = FunASRForceAligner(
                    resources.fa_model,
                    ffmpeg_path=tools.ffmpeg,
                    device=resolve_inference_device(),
                )
                aligned = _force_align_annotation_bounds(
                    force_aligner=force_aligner,
                    audio_path=audio_path,
                    selected_text=selected_text,
                    start_sample=working_start,
                    end_sample=working_end,
                    sample_rate=info.sample_rate,
                    total_samples=info.total_samples,
                    left_guard_sample=left_guard_sample,
                    right_guard_sample=right_guard_sample,
                )
                token.raise_if_cancelled()
                if aligned is not None:
                    working_start, working_end, force_diagnostics = aligned
        vad_start_sample, vad_end_sample = _vad_samples_covering_interval(
            voice_ranges_ms,
            start_sample=working_start,
            end_sample=working_end,
            sample_rate=info.sample_rate,
        )
        radius = max(
            1,
            round(
                info.sample_rate
                * (BOUNDARY_SEARCH_MS + BOUNDARY_MAX_EXPAND_MS + BOUNDARY_ZERO_CROSSING_MS)
                / 1000
            ),
        )
        clip_start = max(0, working_start - radius)
        clip_end = min(info.total_samples, working_end + radius)
        if clip_end <= clip_start:
            return unrefined
        report(0.55, "正在按声学边界贴齐删除范围…")
        samples = decode_f32(
            audio_path,
            info=info,
            start_sample=clip_start,
            end_sample=clip_end,
            tools=tools,
            cancel=token,
        )
        token.raise_if_cancelled()
        if not samples.size:
            return unrefined
        refinement = refine_cut_boundaries(
            samples,
            working_start - clip_start,
            working_end - clip_start,
            info.sample_rate,
            left_guard_sample=(
                left_guard_sample - clip_start if left_guard_sample is not None else None
            ),
            right_guard_sample=(
                right_guard_sample - clip_start if right_guard_sample is not None else None
            ),
            vad_start_sample=(
                vad_start_sample - clip_start if vad_start_sample is not None else None
            ),
            vad_end_sample=(
                vad_end_sample - clip_start if vad_end_sample is not None else None
            ),
            sample_offset=clip_start,
        )
        refined_start = clip_start + refinement.start_sample
        refined_end = clip_start + refinement.end_sample
        if not 0 <= refined_start < refined_end <= info.total_samples:
            return unrefined
        diagnostics: dict[str, object] = dict(refinement.diagnostics)
        if force_diagnostics is not None:
            diagnostics["force_realign"] = force_diagnostics
            diagnostics["pre_acoustic_start_sample"] = working_start
            diagnostics["pre_acoustic_end_sample"] = working_end
        if vad_start_sample is not None and vad_end_sample is not None:
            diagnostics["vad_start_sample"] = vad_start_sample
            diagnostics["vad_end_sample"] = vad_end_sample
        report(1.0, "删除边界已按文字对齐与声学证据细化")
        return AnnotationBoundaryRefinementResult(
            annotation_id,
            start_sample,
            end_sample,
            refined_start,
            refined_end,
            not refinement.requires_review,
            diagnostics,
        )

    return operation


def make_audio_processing_preview_operation(
    project: AudioProcessingProject,
    *,
    start_sample: int,
    end_sample: int,
    mode: str,
    tools: FFmpegTools,
    preview_directory: str | Path,
) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> CandidatePreviewResult:
        if mode not in {"original", "selection", "edited"}:
            raise ValueError(f"未知试听模式: {mode}")
        if not 0 <= start_sample < end_sample <= project.audio_info.total_samples:
            raise ValueError("试听框选范围无效")
        source_path = Path(project.audio.path)
        if not source_path.is_file() or source_path.stat().st_size != project.audio.size_bytes:
            raise ValueError("原音频内容发生变化")
        directory = Path(preview_directory)
        directory.mkdir(parents=True, exist_ok=True)
        state_key = hashlib.sha256(
            repr(
                (
                    start_sample,
                    end_sample,
                    [(item.start_sample, item.end_sample) for item in project.deletion_intervals],
                )
            ).encode("utf-8")
        ).hexdigest()[:12]
        original = directory / f"audio-processing-{state_key}-original.wav"
        selection = directory / f"audio-processing-{state_key}-selection.wav"
        selection_copy = directory / f"audio-processing-{state_key}-selection-copy.wav"
        edited = directory / f"audio-processing-{state_key}-edited.wav"
        padding = round(project.audio_info.sample_rate * 3.0)
        context_start = max(0, start_sample - padding)
        context_end = min(project.audio_info.total_samples, end_sample + padding)
        preview_intervals = [*project.deletion_intervals, (start_sample, end_sample)]
        if mode in {"original", "edited"} and not (original.is_file() and edited.is_file()):
            report(0.10, "正在生成前后 3 秒试听…")
            generate_preview(
                project.audio.path,
                preview_intervals,
                original,
                edited,
                start_sample=context_start,
                end_sample=context_end,
                info=project.audio_info,
                tools=tools,
                cancel=token,
            )
        elif mode == "selection" and not selection.is_file():
            report(0.10, "正在生成删除范围试听…")
            generate_preview(
                project.audio.path,
                [],
                selection,
                selection_copy,
                start_sample=start_sample,
                end_sample=end_sample,
                info=project.audio_info,
                tools=tools,
                cancel=token,
            )
        report(1.0, "试听已就绪")
        return CandidatePreviewResult(original, selection, edited)

    return operation


def make_audio_processing_export_operation(
    project: AudioProcessingProject,
    project_path: Path,
    *,
    tools: FFmpegTools,
) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> AudioProcessingExportResult:
        snapshot = AudioProcessingProject.from_dict(project.to_dict())
        unresolved = [item for item in snapshot.annotations if item.review_required]
        if unresolved:
            raise ValueError(f"仍有 {len(unresolved)} 个删除边界尚未复核，已拒绝导出")
        if not snapshot.audio.matches_file(snapshot.audio.path):
            raise ValueError("导出前检测到原音频内容发生变化")
        source_stat = Path(snapshot.audio.path).stat()
        source_stamp = (source_stat.st_size, source_stat.st_mtime_ns, source_stat.st_ctime_ns)
        output_directory = Path(snapshot.output_directory).expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        stem = Path(snapshot.audio.path).stem
        wav_path = output_directory / f"{stem}_处理完成.wav"
        mp3_path = output_directory / f"{stem}_处理完成.mp3"
        snapshot.output_directory = str(output_directory)
        exported_project = output_directory / f"{stem}.audioprocess.json"
        with tempfile.TemporaryDirectory(
            dir=output_directory,
            prefix=".cutvideo-audio-export-",
        ) as staging_directory:
            staging = Path(staging_directory)
            staged_wav = staging / wav_path.name
            staged_mp3 = staging / mp3_path.name
            staged_project = staging / exported_project.name
            report(0.03, "正在导出处理后的 WAV 与 MP3…")
            result = export_audio(
                snapshot.audio.path,
                snapshot.deletion_intervals,
                staged_wav,
                staged_mp3,
                info=snapshot.audio_info,
                tools=tools,
                progress_cb=lambda value: report(
                    0.03 + value * 0.90,
                    "正在导出处理后的音频…",
                ),
                cancel=token,
            )
            save_audio_processing_project(snapshot, staged_project)
            current_stat = Path(snapshot.audio.path).stat()
            current_stamp = (
                current_stat.st_size,
                current_stat.st_mtime_ns,
                current_stat.st_ctime_ns,
            )
            if current_stamp != source_stamp:
                raise ValueError("导出过程中原音频发生变化，未替换任何输出文件")
            _commit_artifacts(
                (
                    (staged_wav, wav_path),
                    (staged_mp3, mp3_path),
                    (staged_project, exported_project),
                )
            )
        if project_path.resolve() != exported_project.resolve():
            with suppress(Exception):
                save_audio_processing_project(snapshot, project_path)
        report(1.0, "导出完成")
        return AudioProcessingExportResult(
            result.wav_path,
            result.mp3_path,
            exported_project,
            result.kept_samples,
            result.removed_samples,
        )

    return operation


__all__ = [
    "AnnotationBoundaryRefinementResult",
    "AudioProcessingAnalysisResult",
    "AudioProcessingExportResult",
    "AudioProcessingRefinementResult",
    "make_annotation_boundary_refinement_operation",
    "make_audio_processing_analysis_operation",
    "make_audio_processing_export_operation",
    "make_audio_processing_load_operation",
    "make_audio_processing_preview_operation",
    "make_audio_processing_refinement_operation",
    "make_audio_processing_relink_operation",
]
