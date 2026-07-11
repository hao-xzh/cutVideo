"""Cancellable background operations used by the Qt desktop UI."""

from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from ..alignment import (
    ALIGNMENT_PIPELINE_NAME,
    ALIGNMENT_PIPELINE_PURPOSE,
    ALIGNMENT_PIPELINE_VERSION,
    SHORT_HIGHLIGHT,
    AlignmentCandidate,
    FunASRAsrAligner,
    FunASRForceAligner,
    align_transcript,
    normalize_with_mapping,
)
from ..audio import (
    AudioInfo,
    CutInterval,
    WaveformEnvelope,
    decode_f32,
    probe_audio,
    read_waveform_envelopes,
    refine_boundary,
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
    reused_project: ProjectV1 | None = None
    reused_project_path: Path | None = None


@dataclass(slots=True)
class AnalysisResult:
    project: ProjectV1
    project_path: Path


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


def _require_preflight_sources(preflight: PreflightResult, stage: str) -> None:
    if not preflight.audio_source.matches_file(
        preflight.audio_source.path
    ) or not preflight.document_source.matches_file(preflight.document_source.path):
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


def _uses_current_alignment_strategy(project: ProjectV1) -> bool:
    return any(
        model.purpose == ALIGNMENT_PIPELINE_PURPOSE
        and model.name == ALIGNMENT_PIPELINE_NAME
        and model.version == ALIGNMENT_PIPELINE_VERSION
        for model in project.models
    )


def make_preflight_operation(audio_path: str, document_path: str) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> PreflightResult:
        report(0.01, "正在固定输入文件指纹…")
        audio_source = SourceFile.from_path(audio_path)
        document_source = SourceFile.from_path(document_path)
        token.raise_if_cancelled()
        report(0.04, "正在读取 Word 标注…")
        transcript = parse_docx(document_path)
        token.raise_if_cancelled()
        resources = discover_resources()
        tools = discover_ffmpeg(resource_root=resources.root)
        report(0.08, "正在解码音频并建立准确时间轴…")
        info = probe_audio(
            audio_path,
            tools=tools,
            progress_cb=lambda value: report(0.08 + value * 0.76, "正在解码音频并建立准确时间轴…"),
            cancel=token,
        )
        transcript.validate_audio_duration(info.total_samples * 1000 // info.sample_rate)
        token.raise_if_cancelled()
        if not audio_source.matches_file(audio_path) or not document_source.matches_file(
            document_path
        ):
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
            reused,
            reused_path,
        )

    return operation


def make_analysis_operation(preflight: PreflightResult) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> AnalysisResult:
        token.raise_if_cancelled()
        _require_preflight_sources(preflight, "自动分析开始前")
        report(0.03, "正在准备本地对齐模型…")
        resources = preflight.resources
        force_aligner = None
        asr_aligner = None
        models: list[ModelInfo] = [
            ModelInfo(
                ALIGNMENT_PIPELINE_PURPOSE,
                ALIGNMENT_PIPELINE_NAME,
                ALIGNMENT_PIPELINE_VERSION,
            )
        ]
        if resources.fa_model is not None:
            force_aligner = FunASRForceAligner(
                resources.fa_model,
                ffmpeg_path=preflight.tools.ffmpeg,
            )
            models.append(_manifest_model_info(resources, "forced_alignment", "fa-zh"))
        if resources.asr_model is not None and resources.vad_model is not None:
            asr_aligner = FunASRAsrAligner(
                resources.asr_model,
                resources.vad_model,
                ffmpeg_path=preflight.tools.ffmpeg,
            )
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
        report(0.10, "正在识别整段真实语音并匹配 Word 文字…")
        alignments = align_transcript(
            preflight.transcript,
            preflight.audio_info,
            force_aligner,
            asr_aligner,
            progress_cb=lambda done, total: report(
                0.10 + 0.60 * (done / max(1, total)),
                f"正在按真实语音定位段落 {done}/{total}…",
            ),
            cancel=token,
        )
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
            # Storage remains conservative even if a future alignment adapter
            # accidentally marks a one/two-character item as automatic.
            if len(normalize_with_mapping(candidate.text).text) <= 2:
                candidate.review_required = True
                candidate.status = CandidateStatus.NEEDS_REVIEW
                if SHORT_HIGHLIGHT not in candidate.reasons:
                    candidate.reasons.append(SHORT_HIGHLIGHT)
            project_candidates.append(candidate)

        token.raise_if_cancelled()
        report(0.91, "正在创建项目记录并计算文件指纹…")
        _require_preflight_sources(preflight, "自动分析过程中")
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
            export_options=ExportOptions(
                output_directory=str(Path(preflight.audio_source.path).parent)
            ),
        )
        project.validate()
        project_path = Path(preflight.audio_info.path).with_suffix(".cutvideo.json")
        save_project(project, project_path)
        report(1.0, "分析完成")
        return AnalysisResult(project, project_path.resolve())

    return operation


def _refine_alignment(
    alignment: AlignmentCandidate,
    info: AudioInfo,
    tools: FFmpegTools,
    token: CancelToken,
) -> None:
    radius = max(1, round(info.sample_rate * 0.060))
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
    mono = np.mean(samples, axis=1, dtype=np.float32)
    local_start = alignment.proposed_start_sample - clip_start
    local_end = alignment.proposed_end_sample - clip_start
    if local_end - local_start < 3:
        return
    refined_start_local = refine_boundary(
        mono,
        local_start,
        info.sample_rate,
        search_ms=60.0,
        lower_bound=local_start,
        upper_bound=max(local_start + 1, local_end),
    )
    refined_end_local = refine_boundary(
        mono,
        local_end,
        info.sample_rate,
        search_ms=60.0,
        lower_bound=min(refined_start_local + 1, len(mono) - 1),
        upper_bound=min(len(mono), local_end + 1),
    )
    refined_start = clip_start + refined_start_local
    refined_end = clip_start + refined_end_local
    if 0 <= refined_start < refined_end <= info.total_samples:
        alignment.proposed_start_sample = refined_start
        alignment.proposed_end_sample = refined_end


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
    output_directory: Path,
) -> Operation:
    candidates = [CutCandidate.from_dict(item, index) for index, item in enumerate(project.to_dict()["candidates"])]

    def operation(token: CancelToken, report: ProgressReporter) -> CandidatePreviewResult:
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
        original = output_directory / f"{candidate_id}-original.wav"
        edited = output_directory / f"{candidate_id}-edited.wav"
        report(0.05, "正在生成试听片段…")
        full_result = generate_preview(
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
        token.raise_if_cancelled()
        selection_padding = round(info.sample_rate * 0.120)
        selection_start = max(0, current.effective_start_sample - selection_padding)
        selection_end = min(info.total_samples, current.effective_end_sample + selection_padding)
        selection = output_directory / f"{candidate_id}-selection.wav"
        selection_copy = output_directory / f"{candidate_id}-selection-copy.wav"
        report(0.72, "正在生成待删片段试听…")
        selection_result = generate_preview(
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
        return CandidatePreviewResult(
            full_result.original_wav_path,
            selection_result.original_wav_path,
            full_result.edited_wav_path,
        )

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
            if not snapshot.audio.matches_file(
                snapshot.audio.path
            ) or not snapshot.document.matches_file(snapshot.document.path):
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
    "make_relink_project_operation",
    "make_save_project_operation",
    "make_waveform_operation",
]
