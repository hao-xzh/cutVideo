from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from . import __version__
from .alignment import (
    ALIGNMENT_PIPELINE_NAME,
    ALIGNMENT_PIPELINE_PURPOSE,
    ALIGNMENT_PIPELINE_VERSION,
    AlignmentCandidate,
    FunASRAsrAligner,
    FunASRForceAligner,
    align_transcript,
)
from .audio import AudioInfo as DecodedAudioInfo
from .audio import decode_f32, probe_audio, refine_boundary
from .docx_parser import ParsedTranscript, parse_docx
from .ffmpeg import FFmpegTools
from .project import (
    AudioInfo as ProjectAudioInfo,
)
from .project import (
    CutCandidate,
    ExportOptions,
    ModelInfo,
    ProjectV1,
    SourceFile,
)
from .resources import RuntimeResources, discover_resources, load_manifest

ProgressCallback = Callable[[float, str], None]


class AnalysisCancelledError(RuntimeError):
    pass


@dataclass(slots=True)
class AnalysisResult:
    transcript: ParsedTranscript
    audio_info: DecodedAudioInfo
    project: ProjectV1
    resources: RuntimeResources
    warnings: list[str]


def _cancelled(cancel: Event | Callable[[], bool] | None) -> bool:
    if cancel is None:
        return False
    return bool(cancel()) if callable(cancel) else cancel.is_set()


def _report(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback:
        callback(max(0.0, min(1.0, float(fraction))), message)


def _candidate_context(
    parsed: ParsedTranscript, candidate: AlignmentCandidate
) -> tuple[str, str]:
    paragraph = parsed.paragraphs[candidate.paragraph_index]
    highlight = paragraph.highlights[candidate.highlight_index]
    return highlight.context_before, highlight.context_after


def _model_record(resources: RuntimeResources, purpose: str, name: str) -> ModelInfo:
    try:
        models = load_manifest(resources.root).get("models", {})
        entry = models.get(name, {}) if isinstance(models, dict) else {}
        revision = entry.get("revision")
        digest = entry.get("sha256")
        return ModelInfo(
            purpose,
            name,
            revision if isinstance(revision, str) else "bundled-local",
            digest if isinstance(digest, str) else "",
        )
    except Exception:
        return ModelInfo(purpose, name, "bundled-local")


def _build_aligners(
    resources: RuntimeResources, warnings: list[str]
) -> tuple[FunASRForceAligner | None, FunASRAsrAligner | None]:
    ffmpeg = resources.ffmpeg
    force = None
    asr = None
    if resources.fa_model and ffmpeg:
        force = FunASRForceAligner(resources.fa_model, ffmpeg_path=ffmpeg, device="cpu")
    else:
        warnings.append("缺少 fa-zh，本次强制对齐将使用保守回退，所有切点都需要复核。")
    if resources.asr_model and resources.vad_model and ffmpeg:
        asr = FunASRAsrAligner(
            resources.asr_model,
            resources.vad_model,
            ffmpeg_path=ffmpeg,
            device="cpu",
        )
    else:
        warnings.append("缺少 paraformer-zh/fsmn-vad，无法建立整段真实语音时间线。")
    return force, asr


def _refine_candidate_boundaries(
    candidates: list[AlignmentCandidate],
    audio_info: DecodedAudioInfo,
    tools: FFmpegTools,
    *,
    progress_cb: ProgressCallback | None,
    cancel: Event | Callable[[], bool] | None,
) -> None:
    """Snap proposed cuts to quiet zero crossings without loading the whole file."""

    radius = max(1, round(audio_info.sample_rate * 0.065))
    total = len(candidates)
    for index, candidate in enumerate(candidates):
        if _cancelled(cancel):
            raise AnalysisCancelledError("analysis was cancelled")
        window_start = max(0, candidate.proposed_start_sample - radius)
        window_end = min(audio_info.total_samples, candidate.proposed_end_sample + radius)
        pcm = decode_f32(
            audio_info.path,
            info=audio_info,
            start_sample=window_start,
            end_sample=window_end,
            tools=tools,
            cancel=cancel,
        )
        if pcm.size:
            local_start = candidate.proposed_start_sample - window_start
            local_end = candidate.proposed_end_sample - window_start
            if local_end - local_start < 3:
                continue
            refined_start = refine_boundary(
                pcm,
                local_start,
                audio_info.sample_rate,
                search_ms=60,
                # Automatic acoustic snapping may only shrink the proposed
                # deletion.  Expanding toward retained speech without a
                # neighboring-token bound could silently remove a syllable.
                lower_bound=local_start,
                upper_bound=max(1, local_end),
            )
            refined_end = refine_boundary(
                pcm,
                local_end,
                audio_info.sample_rate,
                search_ms=60,
                lower_bound=min(refined_start + 1, len(pcm) - 1),
                upper_bound=min(len(pcm), local_end + 1),
            )
            if refined_end > refined_start:
                candidate.proposed_start_sample = window_start + refined_start
                candidate.proposed_end_sample = window_start + refined_end
        _report(
            progress_cb,
            0.94 + (index + 1) / max(1, total) * 0.04,
            f"正在优化切口 {index + 1}/{total}…",
        )


def analyze_pair(
    audio_path: str | Path,
    document_path: str | Path,
    *,
    resources: RuntimeResources | None = None,
    progress_cb: ProgressCallback | None = None,
    cancel: Event | Callable[[], bool] | None = None,
) -> AnalysisResult:
    """Run the complete offline preflight and conservative alignment pipeline."""

    resource_set = resources or discover_resources()
    if not resource_set.has_ffmpeg:
        raise FileNotFoundError("缺少本平台的 ffmpeg/ffprobe 离线资源")
    tools = FFmpegTools(resource_set.ffmpeg, resource_set.ffprobe)  # type: ignore[arg-type]

    _report(progress_cb, 0.005, "正在固定输入文件指纹…")
    audio_source = SourceFile.from_path(audio_path)
    document_source = SourceFile.from_path(document_path)
    _report(progress_cb, 0.01, "正在读取 Word 标注…")
    parsed = parse_docx(document_path)
    if _cancelled(cancel):
        raise AnalysisCancelledError("analysis was cancelled")

    _report(progress_cb, 0.05, "正在解码音频并建立样本时间轴…")
    audio_info = probe_audio(
        audio_path,
        tools=tools,
        progress_cb=(
            (lambda value: _report(progress_cb, 0.05 + value * 0.20, "正在核对音频时长…"))
            if progress_cb
            else None
        ),
        cancel=cancel,
    )
    parsed.validate_audio_duration(round(audio_info.duration_seconds * 1000))
    if _cancelled(cancel):
        raise AnalysisCancelledError("analysis was cancelled")

    warnings: list[str] = []
    force_aligner, asr_aligner = _build_aligners(resource_set, warnings)
    _report(progress_cb, 0.26, "正在识别整段真实语音并匹配 Word 文字…")

    def alignment_progress(done: int, total: int) -> None:
        ratio = done / total if total else 1.0
        _report(
            progress_cb,
            0.26 + ratio * 0.68,
            f"正在按真实语音定位段落 {done}/{total}…",
        )

    aligned = align_transcript(
        parsed,
        audio_info,
        force_aligner,
        asr_aligner,
        progress_cb=alignment_progress,
        cancel=cancel,
    )
    _refine_candidate_boundaries(
        aligned,
        audio_info,
        tools,
        progress_cb=progress_cb,
        cancel=cancel,
    )
    if _cancelled(cancel):
        raise AnalysisCancelledError("analysis was cancelled")

    candidates: list[CutCandidate] = []
    for candidate in aligned:
        before, after = _candidate_context(parsed, candidate)
        stored = CutCandidate.from_alignment(
            candidate,
            candidate_id=f"p{candidate.paragraph_index:03d}-h{candidate.highlight_index:02d}",
            context_before=before,
            context_after=after,
        )
        highlight = parsed.paragraphs[candidate.paragraph_index].highlights[
            candidate.highlight_index
        ]
        stored.source_start_char = highlight.start
        stored.source_end_char = highlight.end
        candidates.append(stored)

    models: list[ModelInfo] = [
        ModelInfo(
            ALIGNMENT_PIPELINE_PURPOSE,
            ALIGNMENT_PIPELINE_NAME,
            ALIGNMENT_PIPELINE_VERSION,
        )
    ]
    if force_aligner:
        models.append(_model_record(resource_set, "forced_alignment", "fa-zh"))
    if asr_aligner:
        models.append(
            _model_record(
                resource_set,
                "primary_recognition_timeline",
                "paraformer-zh",
            )
        )
        models.append(_model_record(resource_set, "voice_activity_detection", "fsmn-vad"))
    project_audio = ProjectAudioInfo(
        sample_rate=audio_info.sample_rate,
        channels=audio_info.channels,
        total_samples=audio_info.total_samples,
        format_name=Path(audio_path).suffix.lstrip(".").lower(),
        codec_name=audio_info.codec or "",
    )
    if not audio_source.matches_file(audio_source.path) or not document_source.matches_file(
        document_source.path
    ):
        raise ValueError("自动分析过程中输入文件发生变化，请重新开始")
    project = ProjectV1(
        audio=SourceFile(audio_source.path, audio_source.sha256, audio_source.size_bytes),
        document=SourceFile(
            document_source.path,
            document_source.sha256,
            document_source.size_bytes,
        ),
        audio_info=project_audio,
        candidates=candidates,
        models=models,
        export_options=ExportOptions(
            output_directory=str(Path(audio_source.path).parent)
        ),
        app_version=__version__,
    )
    project.validate()
    _report(progress_cb, 1.0, "分析完成")
    return AnalysisResult(parsed, audio_info, project, resource_set, warnings)


def expected_output_paths(audio_path: str | Path, output_directory: str | Path) -> dict[str, Path]:
    source = Path(audio_path)
    output = Path(output_directory)
    stem = source.stem
    return {
        "wav": output / f"{stem}_剪辑完成.wav",
        "mp3": output / f"{stem}_剪辑完成.mp3",
        "project": output / f"{stem}.cutvideo.json",
        "cuts": output / f"{stem}_切点.csv",
    }


__all__ = [
    "AnalysisCancelledError",
    "AnalysisResult",
    "analyze_pair",
    "expected_output_paths",
]
