"""Background operations for the standalone transcript-driven audio workspace."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ..alignment import FunASRAsrAligner
from ..audio import AudioInfo, probe_audio
from ..audio_processing import (
    AudioProcessingProject,
    TranscriptToken,
    default_audio_processing_project_path,
    load_audio_processing_project,
    save_audio_processing_project,
)
from ..ffmpeg import FFmpegTools, export_audio, generate_preview
from ..project import AudioInfo as ProjectAudioInfo
from ..project import SourceFile
from ..resources import discover_resources
from .workers import CancelToken, CandidatePreviewResult, Operation, ProgressReporter


@dataclass(slots=True)
class AudioProcessingAnalysisResult:
    project: AudioProcessingProject
    project_path: Path
    audio_info: AudioInfo
    tools: FFmpegTools


@dataclass(slots=True)
class AudioProcessingExportResult:
    wav_path: Path
    mp3_path: Path
    project_path: Path
    kept_samples: int
    removed_samples: int


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
        info = probe_audio(
            source_path,
            tools=tools,
            progress_cb=lambda value: report(0.08 + value * 0.20, "正在核对音频时长…"),
            cancel=token,
        )
        token.raise_if_cancelled()
        report(0.30, "正在完整识别音频文字与时间戳…")
        recognizer = FunASRAsrAligner(
            resources.asr_model,
            resources.vad_model,
            ffmpeg_path=tools.ffmpeg,
        )
        duration_ms = max(1, math.ceil(info.total_samples * 1000 / info.sample_rate))
        recognized = recognizer.recognize(
            audio_path=source_path,
            window_start_ms=0,
            window_end_ms=duration_ms,
        )
        token.raise_if_cancelled()
        tokens: list[TranscriptToken] = []
        for item in recognized:
            start = max(0, min(info.total_samples - 1, round(item.start_ms * info.sample_rate / 1000)))
            end = max(start + 1, min(info.total_samples, round(item.end_ms * info.sample_rate / 1000)))
            tokens.append(TranscriptToken(item.text, start, end, item.confidence))
        if not tokens:
            raise ValueError("没有识别出可编辑文字；请确认音频包含清晰人声")
        if not source.matches_file(source_path):
            raise ValueError("识别过程中音频文件发生变化，请重新开始")
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
            output_directory=str(source_path.parent),
        )
        project_path = default_audio_processing_project_path(source_path)
        report(0.96, "正在保存音频处理项目…")
        save_audio_processing_project(project, project_path)
        report(1.0, "完整转写已就绪")
        return AudioProcessingAnalysisResult(project, project_path, info, tools)

    return operation


def make_audio_processing_load_operation(project_path: str) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> AudioProcessingAnalysisResult:
        token.raise_if_cancelled()
        path = Path(project_path).expanduser().resolve(strict=True)
        report(0.10, "正在打开音频处理项目…")
        project = load_audio_processing_project(path)
        if not project.audio.matches_file(project.audio.path):
            raise ValueError("项目记录的原音频已移动或内容发生变化")
        resources = discover_resources()
        if not resources.has_ffmpeg:
            raise FileNotFoundError("缺少本平台的 FFmpeg 离线资源")
        tools = FFmpegTools(resources.ffmpeg, resources.ffprobe)  # type: ignore[arg-type]
        info = probe_audio(project.audio.path, tools=tools, cancel=token)
        if (
            info.sample_rate != project.audio_info.sample_rate
            or info.total_samples != project.audio_info.total_samples
        ):
            raise ValueError("项目音频的 PCM 样本时间轴不匹配")
        report(1.0, "项目已打开")
        return AudioProcessingAnalysisResult(project, path, info, tools)

    return operation


def make_audio_processing_preview_operation(
    project: AudioProcessingProject,
    *,
    start_sample: int,
    end_sample: int,
    tools: FFmpegTools,
    preview_directory: str | Path,
) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> CandidatePreviewResult:
        if not 0 <= start_sample < end_sample <= project.audio_info.total_samples:
            raise ValueError("试听框选范围无效")
        if not project.audio.matches_file(project.audio.path):
            raise ValueError("原音频内容发生变化")
        directory = Path(preview_directory)
        directory.mkdir(parents=True, exist_ok=True)
        original = directory / "audio-processing-selection.wav"
        edited = directory / "audio-processing-edited.wav"
        report(0.10, "正在生成本地试听…")
        generate_preview(
            project.audio.path,
            project.deletion_intervals,
            original,
            edited,
            start_sample=start_sample,
            end_sample=end_sample,
            info=project.audio_info,
            tools=tools,
            cancel=token,
        )
        report(1.0, "试听已就绪")
        return CandidatePreviewResult(original, original, edited)

    return operation


def make_audio_processing_export_operation(
    project: AudioProcessingProject,
    project_path: Path,
    *,
    tools: FFmpegTools,
) -> Operation:
    def operation(token: CancelToken, report: ProgressReporter) -> AudioProcessingExportResult:
        snapshot = AudioProcessingProject.from_dict(project.to_dict())
        if not snapshot.audio.matches_file(snapshot.audio.path):
            raise ValueError("导出前检测到原音频内容发生变化")
        output_directory = Path(snapshot.output_directory).expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        stem = Path(snapshot.audio.path).stem
        wav_path = output_directory / f"{stem}_处理完成.wav"
        mp3_path = output_directory / f"{stem}_处理完成.mp3"
        report(0.03, "正在导出处理后的 WAV 与 MP3…")
        result = export_audio(
            snapshot.audio.path,
            snapshot.deletion_intervals,
            wav_path,
            mp3_path,
            info=snapshot.audio_info,
            tools=tools,
            progress_cb=lambda value: report(0.03 + value * 0.92, "正在导出处理后的音频…"),
            cancel=token,
        )
        snapshot.output_directory = str(output_directory)
        exported_project = output_directory / f"{stem}.audioprocess.json"
        save_audio_processing_project(snapshot, exported_project)
        if project_path.resolve() != exported_project.resolve():
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
]
