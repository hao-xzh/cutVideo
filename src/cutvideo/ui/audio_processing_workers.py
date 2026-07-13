"""Background operations for the standalone transcript-driven audio workspace."""

from __future__ import annotations

import hashlib
import math
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from ..alignment import FunASRAsrAligner, voice_ranges_from_model_result
from ..audio import AudioInfo, WaveformEnvelope, probe_audio_with_waveform
from ..audio_processing import (
    AudioProcessingProject,
    TranscriptToken,
    default_audio_processing_project_path,
    infer_transcript_segment_starts,
    load_audio_processing_project,
    save_audio_processing_project,
)
from ..ffmpeg import FFmpegTools, export_audio, generate_preview
from ..model_runtime import local_audio_window, model_execution_guard
from ..project import AudioInfo as ProjectAudioInfo
from ..project import SourceFile
from ..resources import discover_resources
from .workers import (
    CancelToken,
    CandidatePreviewResult,
    Operation,
    ProgressReporter,
    _commit_artifacts,
)


@dataclass(slots=True)
class AudioProcessingAnalysisResult:
    project: AudioProcessingProject
    project_path: Path
    audio_info: AudioInfo
    tools: FFmpegTools
    waveform_envelope: WaveformEnvelope | None = None
    source_matches: bool = True


@dataclass(slots=True)
class AudioProcessingExportResult:
    wav_path: Path
    mp3_path: Path
    project_path: Path
    kept_samples: int
    removed_samples: int


def _segment_starts_from_voice_ranges(
    tokens: list[TranscriptToken],
    voice_ranges: Sequence[tuple[float, float]],
    sample_rate: int,
) -> list[int]:
    if not tokens or not voice_ranges or sample_rate <= 0:
        return []
    boundaries = [
        (left[1] + right[0]) / 2
        for left, right in zip(voice_ranges, voice_ranges[1:], strict=False)
    ]
    starts = [0]
    search_from = 1
    for boundary_ms in boundaries:
        for index in range(search_from, len(tokens)):
            midpoint_ms = (
                (tokens[index].start_sample + tokens[index].end_sample)
                * 500
                / sample_rate
            )
            if midpoint_ms >= boundary_ms:
                starts.append(index)
                search_from = index + 1
                break
    return sorted(set(starts))


def make_audio_processing_analysis_operation(audio_path: str) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> AudioProcessingAnalysisResult:
        source_path = Path(audio_path).expanduser().resolve(strict=True)
        resources = discover_resources()
        if not resources.has_ffmpeg:
            raise FileNotFoundError("缺少本平台的 FFmpeg 离线资源")
        if resources.asr_model is None or resources.vad_model is None:
            raise FileNotFoundError("缺少 paraformer-zh 或 fsmn-vad 离线识别模型")
        tools = FFmpegTools(resources.ffmpeg, resources.ffprobe)  # type: ignore[arg-type]
        report(0.03, "正在核对音频指纹…")
        source = SourceFile.from_path(source_path)
        report(0.08, "正在解码音频并建立真实样本时间轴…")
        info, waveform_envelope = probe_audio_with_waveform(
            source_path,
            tools=tools,
            progress_cb=lambda value: report(0.08 + value * 0.20, "正在核对音频时长…"),
            cancel=token,
        )
        token.raise_if_cancelled()
        report(0.30, "正在完整识别音频文字与时间戳…")
        duration_ms = max(1, math.ceil(info.total_samples * 1000 / info.sample_rate))
        report(0.29, "正在准备可复用的模型音频缓存…")
        with model_execution_guard(token), local_audio_window(
            source_path,
            start_ms=0,
            end_ms=duration_ms,
            ffmpeg_path=tools.ffmpeg,
            cancel=token,
        ) as model_audio:
            recognizer = FunASRAsrAligner(
                resources.asr_model,
                resources.vad_model,
                ffmpeg_path=None,
            )
            report(0.34, "正在完整识别音频文字与时间戳…")
            recognized, raw_result = recognizer.recognize_with_result(
                audio_path=model_audio,
                window_start_ms=0,
                window_end_ms=duration_ms,
            )
        token.raise_if_cancelled()
        tokens: list[TranscriptToken] = []
        for item in recognized:
            start = max(0, min(info.total_samples - 1, round(item.start_ms * info.sample_rate / 1000)))
            end = max(start + 1, min(info.total_samples, round(item.end_ms * info.sample_rate / 1000)))
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
        if not tokens:
            raise ValueError("没有识别出可编辑文字；请确认音频包含清晰人声")
        report(0.88, "正在按人声停顿建立分段…")
        voice_ranges = voice_ranges_from_model_result(
            raw_result,
            window_end_ms=duration_ms,
        )
        segment_starts = _segment_starts_from_voice_ranges(
            tokens, voice_ranges, info.sample_rate
        )
        if not segment_starts:
            segment_starts = infer_transcript_segment_starts(tokens, info.sample_rate)
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
                "coarse_timestamps_are_not_interpolated": True,
            },
            output_directory=str(source_path.parent),
        )
        project_path = default_audio_processing_project_path(source_path)
        report(0.96, "正在保存音频处理项目…")
        save_audio_processing_project(project, project_path)
        report(1.0, "完整转写已就绪")
        return AudioProcessingAnalysisResult(
            project,
            project_path,
            info,
            tools,
            waveform_envelope=waveform_envelope,
        )

    return operation


def make_audio_processing_load_operation(project_path: str) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> AudioProcessingAnalysisResult:
        token.raise_if_cancelled()
        path = Path(project_path).expanduser().resolve(strict=True)
        report(0.10, "正在打开音频处理项目…")
        project = load_audio_processing_project(path)
        source_matches = project.audio.matches_file(project.audio.path)
        resources = discover_resources()
        if not resources.has_ffmpeg:
            raise FileNotFoundError("缺少本平台的 FFmpeg 离线资源")
        tools = FFmpegTools(resources.ffmpeg, resources.ffprobe)  # type: ignore[arg-type]
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
        report(1.0, "项目已打开")
        return AudioProcessingAnalysisResult(
            project,
            path,
            info,
            tools,
            waveform_envelope=waveform_envelope,
            source_matches=source_matches,
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
        resources = discover_resources()
        if not resources.has_ffmpeg:
            raise FileNotFoundError("缺少本平台的 FFmpeg 离线资源")
        tools = FFmpegTools(resources.ffmpeg, resources.ffprobe)  # type: ignore[arg-type]
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
        report(1.0, "音频已重新关联")
        return AudioProcessingAnalysisResult(
            project,
            path,
            info,
            tools,
            waveform_envelope=waveform_envelope,
            source_matches=True,
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
    "AudioProcessingAnalysisResult",
    "AudioProcessingExportResult",
    "make_audio_processing_analysis_operation",
    "make_audio_processing_export_operation",
    "make_audio_processing_load_operation",
    "make_audio_processing_preview_operation",
    "make_audio_processing_relink_operation",
]
