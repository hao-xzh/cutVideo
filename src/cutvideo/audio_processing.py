"""Project model for transcript-driven editing of one standalone audio file."""

from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

from . import __version__
from .project import AudioInfo, ProjectIOError, ProjectValidationError, SourceFile

AUDIO_PROCESSING_SCHEMA: Final = "cutvideo.audio_processing"
AUDIO_PROCESSING_VERSION: Final = 1
DEFAULT_SEGMENT_SILENCE_SECONDS: Final = 0.5
DEFAULT_MAX_SEGMENT_SECONDS: Final = 30.0
LOW_RAW_CONFIDENCE_THRESHOLD: Final = 0.40


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_diagnostic_value(value: object, label: str, *, depth: int = 0) -> None:
    """Keep persisted diagnostics finite, JSON-safe and reasonably bounded."""

    if depth > 8:
        raise ProjectValidationError(f"{label} 嵌套过深")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProjectValidationError(f"{label} 不能包含 NaN 或无穷值")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_diagnostic_value(item, f"{label}[{index}]", depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProjectValidationError(f"{label} 的键必须是字符串")
            _validate_diagnostic_value(item, f"{label}.{key}", depth=depth + 1)
        return
    raise ProjectValidationError(f"{label} 包含不可序列化的诊断值")


def _annotation_review_metadata(
    tokens: list[TranscriptToken],
) -> tuple[bool, list[str], dict[str, Any]]:
    precisions = sorted({token.timestamp_precision for token in tokens})
    confidences = [token.confidence for token in tokens if token.confidence_available]
    # A model timestamp is only an initial content location.  Even a genuine
    # character timestamp is not yet an acoustically refined cut boundary.
    reasons: list[str] = ["boundary_precision_unverified"]
    if any(token.timestamp_precision != "character" for token in tokens):
        reasons.append("coarse_timestamp")
    if len(confidences) != len(tokens):
        reasons.append("model_confidence_unavailable")
    if confidences and min(confidences) < LOW_RAW_CONFIDENCE_THRESHOLD:
        reasons.append("low_model_confidence")
    diagnostics: dict[str, Any] = {
        "boundary_source": "model_timestamp",
        "timestamp_precisions": precisions,
        "selected_token_count": len(tokens),
        "raw_confidence_available_count": len(confidences),
        "raw_confidence_mean": (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        ),
        "raw_confidence_minimum": round(min(confidences), 4) if confidences else None,
    }
    return True, reasons, diagnostics


@dataclass(frozen=True, slots=True)
class TranscriptToken:
    text: str
    start_sample: int
    end_sample: int
    confidence: float = 1.0
    confidence_available: bool = False
    timestamp_precision: str = "unknown"

    def validate(self, total_samples: int) -> None:
        if not self.text:
            raise ProjectValidationError("转写文字不能为空")
        if not 0 <= self.start_sample < self.end_sample <= total_samples:
            raise ProjectValidationError("转写文字的样本范围无效")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ProjectValidationError("转写置信度必须位于 0 到 1")
        if not isinstance(self.confidence_available, bool):
            raise ProjectValidationError("转写置信度可用状态必须是布尔值")
        if self.timestamp_precision not in {"character", "token", "segment", "unknown"}:
            raise ProjectValidationError("转写时间戳精度无效")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "confidence": self.confidence,
            "confidence_available": self.confidence_available,
            "timestamp_precision": self.timestamp_precision,
        }

    @classmethod
    def from_dict(cls, value: object) -> TranscriptToken:
        if not isinstance(value, dict):
            raise ProjectValidationError("转写 token 必须是对象")
        try:
            return cls(
                text=str(value["text"]),
                start_sample=int(value["start_sample"]),
                end_sample=int(value["end_sample"]),
                confidence=float(value.get("confidence", 1.0)),
                confidence_available=value.get("confidence_available", False),
                timestamp_precision=str(value.get("timestamp_precision", "unknown")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectValidationError("转写 token 字段无效") from exc


def infer_transcript_segment_starts(
    tokens: list[TranscriptToken],
    sample_rate: int,
    *,
    silence_seconds: float = DEFAULT_SEGMENT_SILENCE_SECONDS,
    max_segment_seconds: float = DEFAULT_MAX_SEGMENT_SECONDS,
) -> list[int]:
    """Infer stable speech-line starts for projects created before VAD persistence."""

    if not tokens or sample_rate <= 0:
        return []
    silence_samples = round(sample_rate * silence_seconds)
    max_segment_samples = round(sample_rate * max_segment_seconds)
    starts = [0]
    segment_start_sample = tokens[0].start_sample
    for index in range(1, len(tokens)):
        token = tokens[index]
        gap = token.start_sample - tokens[index - 1].end_sample
        if gap >= silence_samples or token.start_sample - segment_start_sample >= max_segment_samples:
            starts.append(index)
            segment_start_sample = token.start_sample
    return starts


@dataclass(slots=True)
class AudioAnnotation:
    id: str
    operation: str
    text: str
    token_start: int
    token_end: int
    start_sample: int
    end_sample: int
    review_required: bool = True
    review_reasons: list[str] = field(
        default_factory=lambda: ["boundary_precision_unverified"]
    )
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate(self, token_count: int, total_samples: int) -> None:
        if not self.id:
            raise ProjectValidationError("标注 id 不能为空")
        if self.operation != "delete":
            raise ProjectValidationError(f"暂不支持的标注操作: {self.operation}")
        if not self.text:
            raise ProjectValidationError("标注文字不能为空")
        if not 0 <= self.token_start < self.token_end <= token_count:
            raise ProjectValidationError("标注 token 范围无效")
        if not 0 <= self.start_sample < self.end_sample <= total_samples:
            raise ProjectValidationError("标注音频范围无效")
        if not isinstance(self.review_required, bool):
            raise ProjectValidationError("标注复核状态必须是布尔值")
        if not isinstance(self.review_reasons, list) or any(
            not isinstance(item, str) or not item for item in self.review_reasons
        ):
            raise ProjectValidationError("标注复核原因必须是非空字符串数组")
        if self.review_required and not self.review_reasons:
            raise ProjectValidationError("待复核标注必须保存复核原因")
        if not isinstance(self.diagnostics, dict):
            raise ProjectValidationError("标注诊断信息必须是对象")
        _validate_diagnostic_value(self.diagnostics, "annotation.diagnostics")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operation": self.operation,
            "text": self.text,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "review_required": self.review_required,
            "review_reasons": list(self.review_reasons),
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value: object) -> AudioAnnotation:
        if not isinstance(value, dict):
            raise ProjectValidationError("音频标注必须是对象")
        legacy_review = "review_required" not in value
        try:
            raw_reasons = value.get(
                "review_reasons",
                ["legacy_boundary_precision_unknown"] if legacy_review else [],
            )
            raw_diagnostics = value.get("diagnostics", {})
            if not isinstance(raw_reasons, list) or not isinstance(raw_diagnostics, dict):
                raise TypeError
            return cls(
                id=str(value["id"]),
                operation=str(value["operation"]),
                text=str(value["text"]),
                token_start=int(value["token_start"]),
                token_end=int(value["token_end"]),
                start_sample=int(value["start_sample"]),
                end_sample=int(value["end_sample"]),
                review_required=value.get("review_required", True),
                review_reasons=list(raw_reasons),
                diagnostics=dict(raw_diagnostics),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectValidationError("音频标注字段无效") from exc


@dataclass(slots=True)
class AudioProcessingProject:
    audio: SourceFile
    audio_info: AudioInfo
    tokens: list[TranscriptToken]
    segment_starts: list[int] = field(default_factory=list)
    annotations: list[AudioAnnotation] = field(default_factory=list)
    analysis_diagnostics: dict[str, Any] = field(default_factory=dict)
    output_directory: str = ""
    app_version: str = __version__
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def validate(self) -> None:
        self.audio.validate("audio")
        self.audio_info.validate()
        if not self.tokens:
            raise ProjectValidationError("完整音频没有识别出可编辑文字")
        previous_start = -1
        previous_end = -1
        for token in self.tokens:
            token.validate(self.audio_info.total_samples)
            if token.start_sample < previous_start or token.end_sample < previous_end:
                raise ProjectValidationError("转写 token 时间必须递增")
            previous_start = token.start_sample
            previous_end = token.end_sample
        if not self.segment_starts:
            self.segment_starts = infer_transcript_segment_starts(
                self.tokens, self.audio_info.sample_rate
            )
        if (
            not self.segment_starts
            or self.segment_starts[0] != 0
            or self.segment_starts != sorted(set(self.segment_starts))
            or self.segment_starts[-1] >= len(self.tokens)
        ):
            raise ProjectValidationError("人声分段起点无效")
        ids: set[str] = set()
        for annotation in self.annotations:
            annotation.validate(len(self.tokens), self.audio_info.total_samples)
            if annotation.id in ids:
                raise ProjectValidationError("标注 id 重复")
            ids.add(annotation.id)
        if not isinstance(self.analysis_diagnostics, dict):
            raise ProjectValidationError("音频分析诊断信息必须是对象")
        _validate_diagnostic_value(
            self.analysis_diagnostics,
            "analysis_diagnostics",
        )
        if not self.output_directory:
            raise ProjectValidationError("导出目录不能为空")

    @property
    def deletion_intervals(self) -> list[AudioAnnotation]:
        return sorted(self.annotations, key=lambda item: (item.start_sample, item.end_sample))

    def add_delete_annotation(self, token_start: int, token_end: int) -> AudioAnnotation:
        if not 0 <= token_start < token_end <= len(self.tokens):
            raise ProjectValidationError("请先选择一段识别文字")
        start_sample = self.tokens[token_start].start_sample
        end_sample = self.tokens[token_end - 1].end_sample
        merged = [
            item
            for item in self.annotations
            if item.start_sample <= end_sample and item.end_sample >= start_sample
        ]
        if merged:
            token_start = min(token_start, *(item.token_start for item in merged))
            token_end = max(token_end, *(item.token_end for item in merged))
            start_sample = min(start_sample, *(item.start_sample for item in merged))
            end_sample = max(end_sample, *(item.end_sample for item in merged))
            annotation_id = merged[0].id
            merged_ids = {item.id for item in merged}
            self.annotations = [item for item in self.annotations if item.id not in merged_ids]
        else:
            annotation_id = f"d-{uuid4().hex[:12]}"
        review_required, review_reasons, diagnostics = _annotation_review_metadata(
            self.tokens[token_start:token_end]
        )
        annotation = AudioAnnotation(
            id=annotation_id,
            operation="delete",
            text="".join(item.text for item in self.tokens[token_start:token_end]),
            token_start=token_start,
            token_end=token_end,
            start_sample=start_sample,
            end_sample=end_sample,
            review_required=review_required,
            review_reasons=review_reasons,
            diagnostics=diagnostics,
        )
        self.annotations.append(annotation)
        self.annotations.sort(key=lambda item: (item.start_sample, item.end_sample))
        self.updated_at = _now()
        return annotation

    def mark_annotation_boundary_reviewed(
        self,
        annotation_id: str,
        *,
        source: str,
    ) -> bool:
        """Persist an explicit human boundary confirmation or adjustment."""

        if not source.strip():
            raise ProjectValidationError("边界复核来源不能为空")
        annotation = next(
            (item for item in self.annotations if item.id == annotation_id),
            None,
        )
        if annotation is None:
            return False
        annotation.review_required = False
        annotation.review_reasons = []
        annotation.diagnostics = {
            **annotation.diagnostics,
            "boundary_source": source,
            "manual_review_confirmed": True,
            "reviewed_at": _now(),
        }
        self.updated_at = _now()
        return True

    def remove_annotation(self, annotation_id: str) -> bool:
        before = len(self.annotations)
        self.annotations = [item for item in self.annotations if item.id != annotation_id]
        if len(self.annotations) != before:
            self.updated_at = _now()
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": AUDIO_PROCESSING_SCHEMA,
            "version": AUDIO_PROCESSING_VERSION,
            "app_version": self.app_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "audio": self.audio.to_dict(),
            "audio_info": self.audio_info.to_dict(),
            "tokens": [item.to_dict() for item in self.tokens],
            "segment_starts": self.segment_starts,
            "annotations": [item.to_dict() for item in self.annotations],
            "analysis_diagnostics": dict(self.analysis_diagnostics),
            "output_directory": self.output_directory,
        }

    @classmethod
    def from_dict(cls, value: object) -> AudioProcessingProject:
        if not isinstance(value, dict):
            raise ProjectValidationError("音频处理项目必须是对象")
        if value.get("schema") != AUDIO_PROCESSING_SCHEMA:
            raise ProjectValidationError("不是音频处理项目文件")
        if value.get("version") != AUDIO_PROCESSING_VERSION:
            raise ProjectValidationError("暂不支持此音频处理项目版本")
        try:
            project = cls(
                audio=SourceFile.from_dict(value["audio"], "audio"),
                audio_info=AudioInfo.from_dict(value["audio_info"]),
                tokens=[TranscriptToken.from_dict(item) for item in value["tokens"]],
                segment_starts=[int(item) for item in value.get("segment_starts", [])],
                annotations=[
                    AudioAnnotation.from_dict(item) for item in value.get("annotations", [])
                ],
                analysis_diagnostics=dict(value.get("analysis_diagnostics", {})),
                output_directory=str(value["output_directory"]),
                app_version=str(value.get("app_version", "")),
                created_at=str(value.get("created_at", "")),
                updated_at=str(value.get("updated_at", "")),
            )
        except (KeyError, TypeError) as exc:
            raise ProjectValidationError("音频处理项目缺少必要字段") from exc
        project.validate()
        return project


def default_audio_processing_project_path(audio_path: str | Path) -> Path:
    return Path(audio_path).with_suffix(".audioprocess.json")


def save_audio_processing_project(
    project: AudioProcessingProject,
    path: str | Path,
) -> Path:
    project.updated_at = _now()
    payload = json.dumps(project.to_dict(), ensure_ascii=False, indent=2) + "\n"
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise ProjectIOError(f"无法保存音频处理项目: {destination}: {exc}") from exc
    return destination


def load_audio_processing_project(path: str | Path) -> AudioProcessingProject:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectIOError(f"无法读取音频处理项目: {source}: {exc}") from exc
    return AudioProcessingProject.from_dict(value)


__all__ = [
    "AUDIO_PROCESSING_SCHEMA",
    "AUDIO_PROCESSING_VERSION",
    "AudioAnnotation",
    "AudioProcessingProject",
    "TranscriptToken",
    "infer_transcript_segment_starts",
    "default_audio_processing_project_path",
    "load_audio_processing_project",
    "save_audio_processing_project",
]
