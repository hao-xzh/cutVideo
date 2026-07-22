"""Cancellable background operations used by the Qt desktop UI."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from ..alignment import (
    ALIGNMENT_PIPELINE_NAME,
    ALIGNMENT_PIPELINE_PURPOSE,
    ALIGNMENT_PIPELINE_VERSION,
    BOUNDARY_EXPANDED,
    BOUNDARY_REFINEMENT_UNCERTAIN,
    SHORT_HIGHLIGHT,
    STATUS_NEEDS_REVIEW,
    AlignmentCandidate,
    AsrAligner,
    ForceAligner,
    FunASRAsrAligner,
    FunASRForceAligner,
    align_transcript,
    alignment_track_from_recognition_tokens,
    normalize_with_mapping,
)
from ..audio import (
    BOUNDARY_MAX_EXPAND_MS,
    BOUNDARY_SEARCH_MS,
    BOUNDARY_ZERO_CROSSING_MS,
    AudioInfo,
    CutInterval,
    WaveformEnvelope,
    decode_f32,
    probe_audio,
    probe_audio_with_waveform,
    read_waveform_envelopes,
    refine_cut_boundaries,
    write_cut_list_csv,
)
from ..docx_parser import ParsedTranscript, parse_docx
from ..ffmpeg import (
    FFmpegCancelledError,
    FFmpegTools,
    discover_ffmpeg,
    export_audio,
    generate_preview,
)
from ..model_runtime import ModelUnavailableError, model_execution_guard
from ..progressive_asr import (
    ProgressiveRecognitionChunk,
    recognize_audio_progressively,
)
from ..project import (
    AudioInfo as ProjectAudioInfo,
)
from ..project import (
    CandidateStatus,
    CutCandidate,
    ExportOptions,
    ModelInfo,
    ProjectV1,
    SourceFile,
    load_project,
    save_project,
)
from ..resources import RuntimeResources, discover_resources, load_manifest

ProgressReporter = Callable[[float, str], None]
Operation = Callable[["CancelToken", ProgressReporter], object]
FileStamp = tuple[int, int, int, int]


class TaskCancelled(RuntimeError):
    """Raised cooperatively when a UI background operation is cancelled."""


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_set():
            raise TaskCancelled("操作已取消")


class TaskSignals(QObject):
    progress = Signal(int, str)
    result = Signal(object)
    error = Signal(str)
    cancelled = Signal()
    finished = Signal()


@dataclass(frozen=True, slots=True)
class AsrTranscriptPartial:
    """A real, newly committed ASR delta safe to display before finalization."""

    delta_text: str
    sequence: int
    total_sequences: int
    committed_until_ms: int
    total_duration_ms: int
    token_count: int


class BackgroundTask(QRunnable):
    """Run an operation on ``QThreadPool`` and marshal results to the GUI thread."""

    def __init__(self, operation: Operation) -> None:
        super().__init__()
        self.operation = operation
        self.token = CancelToken()
        self.signals = TaskSignals()
        self.setAutoDelete(True)

    def cancel(self) -> None:
        self.token.cancel()

    @Slot()
    def run(self) -> None:
        def report(value: float, message: str = "") -> None:
            percent = round(max(0.0, min(1.0, float(value))) * 100)
            self.signals.progress.emit(percent, message)

        try:
            value = self.operation(self.token, report)
            self.token.raise_if_cancelled()
        except (TaskCancelled, FFmpegCancelledError):
            self.signals.cancelled.emit()
        except Exception as exc:  # UI boundary: present a readable diagnostic.
            if self.token.is_set():
                self.signals.cancelled.emit()
            else:
                detail = str(exc).strip() or type(exc).__name__
                self.signals.error.emit(detail)
        else:
            self.signals.result.emit(value)
        finally:
            self.signals.finished.emit()


@dataclass(slots=True)
class PreflightResult:
    transcript: ParsedTranscript
    audio_info: AudioInfo
    resources: RuntimeResources
    tools: FFmpegTools
    audio_source: SourceFile
    document_source: SourceFile
    audio_stamp: FileStamp
    document_stamp: FileStamp
    waveform_envelope: WaveformEnvelope
    reused_project: ProjectV1 | None = None
    reused_project_path: Path | None = None


@dataclass(slots=True)
class AnalysisResult:
    project: ProjectV1
    project_path: Path
    recognized_text: str = ""


@dataclass(slots=True)
class LoadedProjectResult:
    project: ProjectV1
    project_path: Path
    audio_info: AudioInfo
    source_matches: dict[str, bool]


@dataclass(slots=True)
class ExportTaskResult:
    wav_path: Path
    mp3_path: Path
    csv_path: Path
    project_path: Path
    kept_samples: int
    removed_samples: int


@dataclass(slots=True)
class CandidatePreviewResult:
    original_wav_path: Path
    selection_wav_path: Path
    edited_wav_path: Path


def _copy_source(source: SourceFile) -> SourceFile:
    return SourceFile(source.path, source.sha256, source.size_bytes)


def _same_source(left: SourceFile, right: SourceFile) -> bool:
    return left.size_bytes == right.size_bytes and left.sha256 == right.sha256


def _file_stamp(path: str | Path) -> FileStamp:
    stat = Path(path).expanduser().stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, getattr(stat, "st_ino", 0))


def _fingerprint_stable_source(path: str | Path) -> tuple[SourceFile, FileStamp]:
    before = _file_stamp(path)
    source = SourceFile.from_path(path)
    after = _file_stamp(source.path)
    if before != after:
        raise ValueError("计算文件指纹时输入发生变化，请重试")
    return source, after


def _require_preflight_sources(
    preflight: PreflightResult,
    stage: str,
    *,
    verify_digest: bool = False,
) -> None:
    unchanged = (
        _file_stamp(preflight.audio_source.path) == preflight.audio_stamp
        and _file_stamp(preflight.document_source.path) == preflight.document_stamp
    )
    if unchanged and verify_digest:
        unchanged = preflight.audio_source.matches_file(
            preflight.audio_source.path
        ) and preflight.document_source.matches_file(preflight.document_source.path)
    if not unchanged:
        raise ValueError(f"{stage}输入文件发生变化，请重新预检")


def _manifest_model_info(
    resources: RuntimeResources,
    purpose: str,
    name: str,
) -> ModelInfo:
    try:
        models = load_manifest(resources.root).get("models", {})
        entry = models.get(name, {}) if isinstance(models, dict) else {}
        revision_value = entry.get("revision")
        digest_value = entry.get("sha256")
        revision = revision_value if isinstance(revision_value, str) else "bundled-local"
        digest = digest_value if isinstance(digest_value, str) else ""
    except Exception:
        revision = "bundled-local"
        digest = ""
    return ModelInfo(purpose, name, revision, digest)


def _runtime_aligners(
    resources: RuntimeResources,
    *,
    ffmpeg_path: str | Path,
) -> tuple[ForceAligner | None, AsrAligner | None]:
    if resources.has_qwen_models:
        from ..qwen_mlx import QwenMlxAsrAligner, QwenMlxForceAligner

        assert resources.qwen_asr_model is not None
        assert resources.qwen_force_model is not None
        return (
            QwenMlxForceAligner(
                resources.qwen_force_model,
                ffmpeg_path=ffmpeg_path,
            ),
            QwenMlxAsrAligner(
                resources.qwen_asr_model,
                ffmpeg_path=ffmpeg_path,
            ),
        )
    force_aligner: ForceAligner | None = None
    asr_aligner: AsrAligner | None = None
    if resources.fa_model is not None:
        force_aligner = FunASRForceAligner(
            resources.fa_model,
            ffmpeg_path=ffmpeg_path,
        )
    if resources.asr_model is not None and resources.vad_model is not None:
        asr_aligner = FunASRAsrAligner(
            resources.asr_model,
            resources.vad_model,
            ffmpeg_path=ffmpeg_path,
        )
    return force_aligner, asr_aligner


def _release_model_adapter(adapter: object | None) -> None:
    release = getattr(adapter, "release", None)
    if callable(release):
        release()


def _uses_current_alignment_strategy(project: ProjectV1) -> bool:
    return any(
        model.purpose == ALIGNMENT_PIPELINE_PURPOSE
        and model.name == ALIGNMENT_PIPELINE_NAME
        and model.version == ALIGNMENT_PIPELINE_VERSION
        for model in project.models
    )


def make_recognition_warmup_operation() -> Operation:
    """Load and prime the shared ASR/VAD model during the first idle moment."""

    def operation(token: CancelToken, report: ProgressReporter) -> bool:
        token.raise_if_cancelled()
        resources = discover_resources()
        report(0.05, "正在后台预热连续识别模型…")
        if resources.has_qwen_models and resources.qwen_asr_model is not None:
            from ..qwen_mlx import QwenMlxAsrAligner

            recognizer = QwenMlxAsrAligner(resources.qwen_asr_model)
        elif resources.asr_model is not None and resources.vad_model is not None:
            recognizer = FunASRAsrAligner(
                resources.asr_model,
                resources.vad_model,
                ffmpeg_path=None,
            )
        else:
            return False
        recognizer.warmup()
        token.raise_if_cancelled()
        report(1.0, "连续识别模型已预热")
        return True

    return operation


def make_preflight_operation(audio_path: str, document_path: str) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> PreflightResult:
        report(0.01, "正在固定输入文件指纹…")
        audio_source, audio_stamp = _fingerprint_stable_source(audio_path)
        document_source, document_stamp = _fingerprint_stable_source(document_path)
        token.raise_if_cancelled()
        report(0.04, "正在读取 Word 标注…")
        transcript = parse_docx(document_path)
        token.raise_if_cancelled()
        resources = discover_resources()
        tools = discover_ffmpeg(resource_root=resources.root)
        report(0.08, "正在解码音频并建立准确时间轴…")
        info, waveform_envelope = probe_audio_with_waveform(
            audio_path,
            tools=tools,
            progress_cb=lambda value: report(0.08 + value * 0.76, "正在解码音频并建立准确时间轴…"),
            cancel=token,
        )
        transcript.validate_audio_duration(info.total_samples * 1000 // info.sample_rate)
        token.raise_if_cancelled()
        if _file_stamp(audio_path) != audio_stamp or _file_stamp(document_path) != document_stamp:
            raise ValueError("预检过程中输入文件发生变化，请重新开始预检")

        reused: ProjectV1 | None = None
        reused_path: Path | None = None
        default_project = Path(audio_path).with_suffix(".cutvideo.json")
        if default_project.is_file():
            report(0.87, "正在核对已有项目…")
            try:
                candidate = load_project(default_project)
                stored_shape = (
                    candidate.audio_info.sample_rate,
                    candidate.audio_info.channels,
                    candidate.audio_info.total_samples,
                )
                fresh_shape = (info.sample_rate, info.channels, info.total_samples)
                if (
                    _same_source(candidate.audio, audio_source)
                    and _same_source(candidate.document, document_source)
                    and stored_shape == fresh_shape
                    and _uses_current_alignment_strategy(candidate)
                ):
                    old_audio_parent = Path(candidate.audio.path).parent
                    old_output = Path(candidate.export_options.output_directory)
                    candidate.audio.relink(audio_source.path)
                    candidate.document.relink(document_source.path)
                    if old_output == old_audio_parent:
                        candidate.export_options.output_directory = str(
                            Path(audio_source.path).parent
                        )
                    save_project(candidate, default_project)
                    reused = candidate
                    reused_path = default_project.resolve()
            except Exception:
                # A stale/corrupt sidecar must not make otherwise valid inputs unusable.
                reused = None
        report(1.0, "预检完成")
        return PreflightResult(
            transcript,
            info,
            resources,
            tools,
            audio_source,
            document_source,
            audio_stamp,
            document_stamp,
            waveform_envelope,
            reused,
            reused_path,
        )

    return operation


def make_analysis_operation(
    preflight: PreflightResult,
    partial_cb: Callable[[AsrTranscriptPartial], None] | None = None,
) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> AnalysisResult:
        token.raise_if_cancelled()
        _require_preflight_sources(preflight, "自动分析开始前")
        report(0.03, "正在准备本地对齐模型…")
        resources = preflight.resources
        force_aligner, asr_aligner = _runtime_aligners(
            resources,
            ffmpeg_path=preflight.tools.ffmpeg,
        )
        models: list[ModelInfo] = [
            ModelInfo(
                ALIGNMENT_PIPELINE_PURPOSE,
                ALIGNMENT_PIPELINE_NAME,
                ALIGNMENT_PIPELINE_VERSION,
            )
        ]
        if resources.has_qwen_models:
            models.append(
                _manifest_model_info(
                    resources,
                    "primary_recognition_timeline",
                    "qwen3-asr-0.6b-4bit",
                )
            )
            models.append(
                _manifest_model_info(
                    resources,
                    "forced_alignment",
                    "qwen3-forced-aligner-0.6b-4bit",
                )
            )
        else:
            if force_aligner is not None:
                models.append(_manifest_model_info(resources, "forced_alignment", "fa-zh"))
            if asr_aligner is not None:
                models.append(
                    _manifest_model_info(
                        resources,
                        "primary_recognition_timeline",
                        "paraformer-zh",
                    )
                )
                models.append(
                    _manifest_model_info(resources, "voice_activity_detection", "fsmn-vad")
                )
        duration_ms = max(
            1,
            math.ceil(
                preflight.audio_info.total_samples
                * 1000
                / preflight.audio_info.sample_rate
            ),
        )
        global_asr_track = None
        recognized_text = ""
        progressive_chunk_count = 0
        recognition_inference_device = "unavailable"
        if asr_aligner is not None:
            recognition_backend = resources.recognition_backend
            recognition_model = (
                resources.qwen_asr_model
                if recognition_backend == "qwen-mlx"
                else resources.asr_model
            )
            recognition_vad_model = (
                None
                if recognition_backend == "qwen-mlx"
                else resources.vad_model
            )
            assert recognition_model is not None
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

            try:
                progressive = recognize_audio_progressively(
                    audio_path=preflight.audio_info.path,
                    duration_ms=duration_ms,
                    asr_model_path=recognition_model,
                    vad_model_path=recognition_vad_model,
                    ffmpeg_path=preflight.tools.ffmpeg,
                    progress_cb=lambda value, message: report(
                        0.08 + 0.38 * value,
                        message,
                    ),
                    chunk_cb=publish_chunk,
                    cancel=token,
                    backend=recognition_backend,
                )
            except ModelUnavailableError:
                token.raise_if_cancelled()
                asr_aligner = None
                report(0.46, "连续识别不可用，将生成全部需复核的安全估算…")
            else:
                recognized_text = "".join(item.text for item in progressive.tokens)
                progressive_chunk_count = progressive.chunk_count
                recognition_inference_device = progressive.inference_device
                global_transcript = "\n".join(
                    paragraph.text for paragraph in preflight.transcript.paragraphs
                )
                global_asr_track = alignment_track_from_recognition_tokens(
                    progressive.tokens,
                    global_transcript,
                    engine=(
                        "qwen3-asr-0.6b-mlx-progressive"
                        if recognition_backend == "qwen-mlx"
                        else "paraformer-zh+fsmn-vad-progressive"
                    ),
                    vad_ranges=progressive.voice_ranges,
                )

        try:
            with model_execution_guard(token):
                report(0.48, "正在用流式识别结果匹配 Word 文字…")
                alignments = align_transcript(
                    preflight.transcript,
                    preflight.audio_info,
                    force_aligner,
                    asr_aligner,
                    global_asr_track=global_asr_track,
                    progress_cb=lambda done, total: report(
                        0.48 + 0.22 * (done / max(1, total)),
                        f"正在按真实语音定位段落 {done}/{total}…",
                    ),
                    cancel=token,
                )
        finally:
            _release_model_adapter(force_aligner)
            _release_model_adapter(asr_aligner)
        token.raise_if_cancelled()

        count = max(1, len(alignments))
        for index, alignment in enumerate(alignments):
            token.raise_if_cancelled()
            _refine_alignment(alignment, preflight.audio_info, preflight.tools, token)
            report(
                0.72 + 0.17 * (index + 1) / count,
                f"正在优化切点 {index + 1}/{len(alignments)}…",
            )

        project_candidates: list[CutCandidate] = []
        for alignment in alignments:
            paragraph = preflight.transcript.paragraphs[alignment.paragraph_index]
            highlight = paragraph.highlights[alignment.highlight_index]
            candidate = CutCandidate.from_alignment(
                alignment,
                context_before=highlight.context_before,
                context_after=highlight.context_after,
            )
            candidate.source_start_char = highlight.start
            candidate.source_end_char = highlight.end
            # Every yellow Word highlight is a human decision.  Alignment
            # confidence prioritises the queue and exposes risk, but must never
            # silently turn a marked passage into an approved deletion.
            candidate.review_required = True
            candidate.status = CandidateStatus.NEEDS_REVIEW
            if (
                len(normalize_with_mapping(candidate.text).text) <= 2
                and SHORT_HIGHLIGHT not in candidate.reasons
            ):
                candidate.reasons.append(SHORT_HIGHLIGHT)
            project_candidates.append(candidate)

        token.raise_if_cancelled()
        report(0.91, "正在创建项目记录并计算文件指纹…")
        _require_preflight_sources(preflight, "自动分析过程中", verify_digest=True)
        storage_info = ProjectAudioInfo(
            sample_rate=preflight.audio_info.sample_rate,
            channels=preflight.audio_info.channels,
            total_samples=preflight.audio_info.total_samples,
            format_name=Path(preflight.audio_info.path).suffix.lstrip(".").lower(),
            codec_name=preflight.audio_info.codec or "",
        )
        project = ProjectV1(
            audio=_copy_source(preflight.audio_source),
            document=_copy_source(preflight.document_source),
            audio_info=storage_info,
            candidates=project_candidates,
            models=models,
            analysis_diagnostics={
                "alignment_pipeline": {
                    "name": ALIGNMENT_PIPELINE_NAME,
                    "version": ALIGNMENT_PIPELINE_VERSION,
                    "matching_strategy": "anchored_monotonic_dp",
                    "timestamp_unit": "ms",
                    "recognition_delivery": "progressive_overlapping_windows",
                    "recognition_chunk_count": progressive_chunk_count,
                    "asr_inference_device": recognition_inference_device,
                    "fa_inference_device": (
                        force_aligner.device if force_aligner is not None else "unavailable"
                    ),
                },
                "skipped_highlights": preflight.transcript.skipped_highlights,
                "candidate_count": len(project_candidates),
                "auto_approved_count": sum(
                    item.status is CandidateStatus.AUTO_APPROVED
                    for item in project_candidates
                ),
                "review_required_count": sum(item.needs_review for item in project_candidates),
            },
            export_options=ExportOptions(
                output_directory=str(Path(preflight.audio_source.path).parent)
            ),
        )
        project.validate()
        project_path = Path(preflight.audio_info.path).with_suffix(".cutvideo.json")
        save_project(project, project_path)
        report(1.0, "分析完成")
        return AnalysisResult(project, project_path.resolve(), recognized_text)

    return operation


def _refine_alignment(
    alignment: AlignmentCandidate,
    info: AudioInfo,
    tools: FFmpegTools,
    token: CancelToken,
) -> None:
    radius = max(
        1,
        round(
            info.sample_rate
            * (BOUNDARY_SEARCH_MS + BOUNDARY_MAX_EXPAND_MS + BOUNDARY_ZERO_CROSSING_MS)
            / 1000
        ),
    )
    clip_start = max(0, alignment.proposed_start_sample - radius)
    clip_end = min(info.total_samples, alignment.proposed_end_sample + radius)
    if clip_end <= clip_start:
        return
    samples = decode_f32(
        info.path,
        info=info,
        start_sample=clip_start,
        end_sample=clip_end,
        tools=tools,
        cancel=token,
    )
    token.raise_if_cancelled()
    if not samples.size:
        return
    local_start = alignment.proposed_start_sample - clip_start
    local_end = alignment.proposed_end_sample - clip_start
    if local_end - local_start < 3:
        return
    refinement = refine_cut_boundaries(
        samples,
        local_start,
        local_end,
        info.sample_rate,
        left_guard_sample=(
            alignment.left_guard_sample - clip_start
            if alignment.left_guard_sample is not None
            else None
        ),
        right_guard_sample=(
            alignment.right_guard_sample - clip_start
            if alignment.right_guard_sample is not None
            else None
        ),
        vad_start_sample=(
            alignment.speech_start_sample - clip_start
            if alignment.speech_start_sample is not None
            else None
        ),
        vad_end_sample=(
            alignment.speech_end_sample - clip_start
            if alignment.speech_end_sample is not None
            else None
        ),
        sample_offset=clip_start,
    )
    refined_start = clip_start + refinement.start_sample
    refined_end = clip_start + refinement.end_sample
    if 0 <= refined_start < refined_end <= info.total_samples:
        alignment.proposed_start_sample = refined_start
        alignment.proposed_end_sample = refined_end
        alignment.diagnostics["boundary_refinement"] = refinement.diagnostics
        if refinement.requires_review:
            alignment.requires_review = True
            alignment.status = STATUS_NEEDS_REVIEW
            if BOUNDARY_REFINEMENT_UNCERTAIN not in alignment.reasons:
                alignment.reasons.append(BOUNDARY_REFINEMENT_UNCERTAIN)
        if refinement.expanded and BOUNDARY_EXPANDED not in alignment.reasons:
            alignment.reasons.append(BOUNDARY_EXPANDED)


def make_waveform_operation(info: AudioInfo, tools: FFmpegTools) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> WaveformEnvelope:
        # Keep roughly 5 ms per bucket so a ±2 s review window exposes
        # hundreds of meaningful waveform samples instead of a few coarse bars.
        target_points = min(600_000, max(12_000, round(info.duration_seconds * 200)))
        levels = read_waveform_envelopes(
            info.path,
            info=info,
            tools=tools,
            target_points=target_points,
            max_levels=1,
            progress_cb=lambda value: report(value, "正在生成波形…"),
            cancel=token,
        )
        token.raise_if_cancelled()
        return levels[0]

    return operation


def make_load_project_operation(project_path: str) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> LoadedProjectResult:
        report(0.05, "正在读取项目…")
        project = load_project(project_path)
        if not _uses_current_alignment_strategy(project):
            raise ValueError(
                "该项目使用旧版 Word 时间主导的切点算法，不能安全复用。"
                "请重新选择原音频和 Word，并执行自动分析。"
            )
        token.raise_if_cancelled()
        matches = project.verify_source_files()
        report(0.45, "正在核对音频时间轴…")
        info = AudioInfo(
            path=Path(project.audio.path),
            sample_rate=project.audio_info.sample_rate,
            channels=project.audio_info.channels,
            total_samples=project.audio_info.total_samples,
            duration_seconds=project.audio_info.duration_seconds,
            codec=project.audio_info.codec_name or None,
        )
        report(1.0, "项目已打开")
        return LoadedProjectResult(project, Path(project_path).resolve(), info, matches)

    return operation


def make_relink_project_operation(
    project_path: Path,
    audio_path: str,
    document_path: str,
) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> LoadedProjectResult:
        report(0.04, "正在核对重新选择的文件…")
        project = load_project(project_path)
        if not _uses_current_alignment_strategy(project):
            raise ValueError(
                "该项目使用旧版 Word 时间主导的切点算法，不能安全复用。"
                "请重新选择原音频和 Word，并执行自动分析。"
            )
        token.raise_if_cancelled()
        if not project.audio.matches_file(project.audio.path):
            old_audio_parent = Path(project.audio.path).parent
            old_output = Path(project.export_options.output_directory)
            project.relink_source("audio", audio_path)
            if old_output == old_audio_parent:
                project.export_options.output_directory = str(Path(project.audio.path).parent)
        if not project.document.matches_file(project.document.path):
            project.relink_source("document", document_path)
        token.raise_if_cancelled()
        save_project(project, project_path)
        info = AudioInfo(
            path=Path(project.audio.path),
            sample_rate=project.audio_info.sample_rate,
            channels=project.audio_info.channels,
            total_samples=project.audio_info.total_samples,
            duration_seconds=project.audio_info.duration_seconds,
            codec=project.audio_info.codec_name or None,
        )
        report(1.0, "文件已重新关联")
        return LoadedProjectResult(
            project,
            project_path.resolve(),
            info,
            {"audio": True, "document": True},
        )

    return operation


def make_save_project_operation(project: ProjectV1, project_path: Path) -> Operation:
    # Snapshot before entering another thread, so rapid UI edits cannot race JSON traversal.
    snapshot = ProjectV1.from_dict(project.to_dict())

    def operation(token: CancelToken, report: ProgressReporter) -> Path:
        token.raise_if_cancelled()
        result = save_project(snapshot, project_path)
        report(1.0, "项目已保存")
        return result

    return operation


def make_preview_operation(
    project: ProjectV1,
    info: AudioInfo,
    tools: FFmpegTools,
    candidate_id: str,
    mode: str,
    output_directory: Path,
) -> Operation:
    candidates = [CutCandidate.from_dict(item, index) for index, item in enumerate(project.to_dict()["candidates"])]

    def operation(token: CancelToken, report: ProgressReporter) -> CandidatePreviewResult:
        if mode not in {"original", "selection", "edited"}:
            raise ValueError(f"未知试听模式: {mode}")
        selected = [
            CutInterval(
                candidate.effective_start_sample,
                candidate.effective_end_sample,
                candidate.text,
                candidate.status.value,
            )
            for candidate in candidates
            if candidate.is_selected
        ]
        current = next(candidate for candidate in candidates if candidate.id == candidate_id)
        if not any(
            interval.start_sample == current.effective_start_sample
            and interval.end_sample == current.effective_end_sample
            for interval in selected
        ):
            selected.append(
                CutInterval(
                    current.effective_start_sample,
                    current.effective_end_sample,
                    current.text,
                    current.status.value,
                )
            )
        padding = info.sample_rate * 2
        window_start = max(0, current.effective_start_sample - padding)
        window_end = min(info.total_samples, current.effective_end_sample + padding)
        state_key = hashlib.sha256(
            repr(
                (
                    candidate_id,
                    current.effective_start_sample,
                    current.effective_end_sample,
                    [(item.start_sample, item.end_sample) for item in selected],
                    project.export_options.crossfade_ms,
                )
            ).encode("utf-8")
        ).hexdigest()[:12]
        original = output_directory / f"{candidate_id}-{state_key}-original.wav"
        edited = output_directory / f"{candidate_id}-{state_key}-edited.wav"
        selection_padding = round(info.sample_rate * 0.120)
        selection_start = max(0, current.effective_start_sample - selection_padding)
        selection_end = min(info.total_samples, current.effective_end_sample + selection_padding)
        selection = output_directory / f"{candidate_id}-{state_key}-selection.wav"
        selection_copy = output_directory / f"{candidate_id}-{state_key}-selection-copy.wav"
        if mode in {"original", "edited"} and not (original.is_file() and edited.is_file()):
            report(0.10, "正在生成前后对比试听…")
            generate_preview(
                info.path,
                selected,
                original,
                edited,
                start_sample=window_start,
                end_sample=window_end,
                info=info,
                tools=tools,
                fade_ms=project.export_options.crossfade_ms,
                cancel=token,
            )
        elif mode == "selection" and not selection.is_file():
            report(0.10, "正在生成待删范围试听…")
            generate_preview(
                info.path,
                [],
                selection,
                selection_copy,
                start_sample=selection_start,
                end_sample=selection_end,
                info=info,
                tools=tools,
                fade_ms=project.export_options.crossfade_ms,
                cancel=token,
            )
        report(1.0, "试听片段已生成")
        return CandidatePreviewResult(original, selection, edited)

    return operation


def make_export_operation(
    project: ProjectV1,
    project_path: Path,
    output_directory: Path,
    tools: FFmpegTools,
) -> Operation:
    snapshot = ProjectV1.from_dict(project.to_dict())

    def operation(token: CancelToken, report: ProgressReporter) -> ExportTaskResult:
        if not snapshot.ready_to_export:
            raise ValueError("仍有待复核项目，暂不能导出")
        token.raise_if_cancelled()
        report(0.01, "正在核对输入文件和音频时间轴…")
        if not snapshot.audio.matches_file(snapshot.audio.path):
            raise ValueError("原音频内容已变化，请重新分析或重新关联项目")
        if not snapshot.document.matches_file(snapshot.document.path):
            raise ValueError("Word 标注文档内容已变化，请重新分析或重新关联项目")
        current_info = probe_audio(
            snapshot.audio.path,
            tools=tools,
            progress_cb=lambda value: report(
                0.01 + value * 0.11, "正在核对输入文件和音频时间轴…"
            ),
            cancel=token,
        )
        expected_shape = (
            snapshot.audio_info.sample_rate,
            snapshot.audio_info.channels,
            snapshot.audio_info.total_samples,
        )
        actual_shape = (
            current_info.sample_rate,
            current_info.channels,
            current_info.total_samples,
        )
        if actual_shape != expected_shape:
            raise ValueError("音频 PCM 时间轴与项目记录不一致，请重新分析")
        if not snapshot.audio.matches_file(
            snapshot.audio.path
        ) or not snapshot.document.matches_file(snapshot.document.path):
            raise ValueError("核对过程中输入文件发生变化，请重新分析")
        verified_audio_stamp = _file_stamp(snapshot.audio.path)
        verified_document_stamp = _file_stamp(snapshot.document.path)
        token.raise_if_cancelled()
        output_directory.mkdir(parents=True, exist_ok=True)
        source = Path(snapshot.audio.path)
        stem = source.stem
        wav_path = output_directory / f"{stem}_剪辑完成.wav"
        mp3_path = output_directory / f"{stem}_剪辑完成.mp3"
        csv_path = output_directory / f"{stem}_切点.csv"
        exported_project_path = output_directory / f"{stem}.cutvideo.json"
        intervals = [
            CutInterval(
                candidate.effective_start_sample,
                candidate.effective_end_sample,
                candidate.text,
                candidate.status.value,
            )
            for candidate in snapshot.selected_candidates
        ]
        snapshot.export_options.output_directory = str(output_directory.resolve())
        with tempfile.TemporaryDirectory(
            dir=output_directory,
            prefix=".cutvideo-export-",
        ) as staging_directory:
            staging = Path(staging_directory)
            staged_wav = staging / wav_path.name
            staged_mp3 = staging / mp3_path.name
            staged_csv = staging / csv_path.name
            staged_project = staging / exported_project_path.name
            result = export_audio(
                snapshot.audio.path,
                intervals,
                staged_wav,
                staged_mp3,
                total_samples=snapshot.audio_info.total_samples,
                sample_rate=snapshot.audio_info.sample_rate,
                channels=snapshot.audio_info.channels,
                tools=tools,
                fade_ms=snapshot.export_options.crossfade_ms,
                mp3_bitrate=f"{snapshot.export_options.mp3_bitrate_kbps}k",
                progress_cb=lambda value: report(
                    0.12 + value * 0.78, "正在导出 WAV 和 MP3…"
                ),
                cancel=token,
            )
            token.raise_if_cancelled()
            report(0.92, "正在写入切点清单和项目记录…")
            write_cut_list_csv(staged_csv, intervals, snapshot.audio_info.sample_rate)
            save_project(snapshot, staged_project)
            token.raise_if_cancelled()
            if (
                _file_stamp(snapshot.audio.path) != verified_audio_stamp
                or _file_stamp(snapshot.document.path) != verified_document_stamp
            ):
                raise ValueError("导出过程中输入文件发生变化，未替换任何输出文件")
            _commit_artifacts(
                (
                    (staged_wav, wav_path),
                    (staged_mp3, mp3_path),
                    (staged_csv, csv_path),
                    (staged_project, exported_project_path),
                )
            )
        if exported_project_path.resolve() != project_path.resolve():
            # The exported project is already safely committed. Updating the
            # working sidecar is useful but is not part of the four-file output
            # transaction and must not make a successful export look failed.
            with suppress(Exception):
                save_project(snapshot, project_path)
        report(1.0, "导出完成")
        return ExportTaskResult(
            wav_path.resolve(),
            mp3_path.resolve(),
            csv_path.resolve(),
            exported_project_path.resolve(),
            result.kept_samples,
            result.removed_samples,
        )

    return operation


def _commit_artifacts(pairs: tuple[tuple[Path, Path], ...]) -> None:
    """Atomically replace a same-filesystem group with rollback on failure."""

    targets = [target for _staged, target in pairs]
    if len(set(targets)) != len(targets):
        raise ValueError("导出目标路径重复")
    for staged, target in pairs:
        if not staged.is_file() or staged.is_symlink():
            raise ValueError(f"导出临时文件无效: {staged}")
        if target.exists() and (not target.is_file() or target.is_symlink()):
            raise ValueError(f"导出目标不是普通文件: {target}")

    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for _staged, target in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.with_name(f".{target.name}.{uuid4().hex}.bak")
                os.replace(target, backup)
                backups[target] = backup
        for staged, target in pairs:
            os.replace(staged, target)
            committed.append(target)
    except Exception:
        for target in reversed(committed):
            with suppress(OSError):
                target.unlink(missing_ok=True)
        for target, backup in reversed(tuple(backups.items())):
            if backup.exists():
                with suppress(OSError):
                    os.replace(backup, target)
        raise
    else:
        for backup in backups.values():
            with suppress(OSError):
                backup.unlink(missing_ok=True)


__all__ = [
    "AnalysisResult",
    "AsrTranscriptPartial",
    "BackgroundTask",
    "CancelToken",
    "CandidatePreviewResult",
    "ExportTaskResult",
    "LoadedProjectResult",
    "PreflightResult",
    "TaskCancelled",
    "make_analysis_operation",
    "make_export_operation",
    "make_load_project_operation",
    "make_preflight_operation",
    "make_preview_operation",
    "make_recognition_warmup_operation",
    "make_relink_project_operation",
    "make_save_project_operation",
    "make_waveform_operation",
]
