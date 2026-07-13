"""Versioned, validated project records for the offline editor.

The JSON file is the durable boundary between analysis, manual review and
export.  Paths are convenient hints only; SHA-256 fingerprints are the source
identity so a moved audio or DOCX can be safely re-linked.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

from . import __version__

PROJECT_SCHEMA: Final = "cutvideo.project"
PROJECT_VERSION: Final = 1
DEFAULT_APP_VERSION: Final = __version__
_HASH_CHUNK_BYTES: Final = 1024 * 1024


class ProjectError(Exception):
    """Base class for project persistence errors."""


class ProjectValidationError(ProjectError, ValueError):
    """Raised when project data violates the V1 schema or invariants."""


class ProjectIOError(ProjectError, OSError):
    """Raised when a project cannot be read or atomically written."""


class CandidateStatus(StrEnum):
    """Review state of a proposed deletion."""

    AUTO_APPROVED = "auto_approved"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    SKIPPED = "skipped"


def sha256_file(path: str | Path, *, chunk_size: int = _HASH_CHUNK_BYTES) -> str:
    """Return a lowercase SHA-256 digest without loading the file into memory."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    source = Path(path).expanduser()
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise ProjectIOError(f"无法计算文件 SHA-256: {source}: {exc}") from exc
    return digest.hexdigest()


# A friendly alias for callers that naturally search for "hash_file".
hash_file = sha256_file


@dataclass(slots=True)
class SourceFile:
    path: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_path(cls, path: str | Path) -> SourceFile:
        source = Path(path).expanduser()
        try:
            resolved = source.resolve(strict=True)
            if not resolved.is_file():
                raise ProjectIOError(f"输入路径不是文件: {resolved}")
            size = resolved.stat().st_size
        except ProjectIOError:
            raise
        except OSError as exc:
            raise ProjectIOError(f"无法读取输入文件: {source}: {exc}") from exc
        return cls(path=str(resolved), sha256=sha256_file(resolved), size_bytes=size)

    @property
    def name(self) -> str:
        return Path(self.path).name

    def validate(self, label: str = "source") -> None:
        _nonempty_string(self.path, f"{label}.path")
        digest = _nonempty_string(self.sha256, f"{label}.sha256")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ProjectValidationError(f"{label}.sha256 必须是 64 位小写十六进制")
        _nonnegative_int(self.size_bytes, f"{label}.size_bytes")

    def matches_file(self, path: str | Path) -> bool:
        """Return whether *path* is exactly this source, by size and SHA-256."""

        candidate = Path(path).expanduser()
        try:
            if not candidate.is_file() or candidate.stat().st_size != self.size_bytes:
                return False
            return sha256_file(candidate) == self.sha256
        except (OSError, ProjectIOError):
            return False

    def relink(self, path: str | Path) -> None:
        """Update the path only after fingerprinting the replacement file."""

        candidate = Path(path).expanduser()
        if not self.matches_file(candidate):
            raise ProjectValidationError(f"文件内容与项目记录不匹配: {candidate}")
        self.path = str(candidate.resolve())

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, value: object, label: str) -> SourceFile:
        data = _mapping(value, label)
        source = cls(
            path=_string_field(data, "path", label),
            sha256=_string_field(data, "sha256", label),
            size_bytes=_int_field(data, "size_bytes", label),
        )
        source.validate(label)
        return source


@dataclass(slots=True)
class AudioInfo:
    sample_rate: int
    channels: int
    total_samples: int
    format_name: str = ""
    codec_name: str = ""

    @property
    def duration_ms(self) -> int:
        return self.total_samples * 1000 // self.sample_rate

    @property
    def duration_seconds(self) -> float:
        return self.total_samples / self.sample_rate

    def validate(self) -> None:
        _positive_int(self.sample_rate, "audio_info.sample_rate")
        _positive_int(self.channels, "audio_info.channels")
        _nonnegative_int(self.total_samples, "audio_info.total_samples")
        _string(self.format_name, "audio_info.format_name")
        _string(self.codec_name, "audio_info.codec_name")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "total_samples": self.total_samples,
            "duration_ms": self.duration_ms,
            "format_name": self.format_name,
            "codec_name": self.codec_name,
        }

    @classmethod
    def from_dict(cls, value: object) -> AudioInfo:
        data = _mapping(value, "audio_info")
        result = cls(
            sample_rate=_int_field(data, "sample_rate", "audio_info"),
            channels=_int_field(data, "channels", "audio_info"),
            total_samples=_int_field(data, "total_samples", "audio_info"),
            format_name=_optional_string_field(data, "format_name", "audio_info", ""),
            codec_name=_optional_string_field(data, "codec_name", "audio_info", ""),
        )
        result.validate()
        stored_duration = data.get("duration_ms")
        if stored_duration is not None and (
            isinstance(stored_duration, bool)
            or not isinstance(stored_duration, int)
            or stored_duration != result.duration_ms
        ):
            raise ProjectValidationError("audio_info.duration_ms 与 PCM 样本时间轴不一致")
        return result


@dataclass(slots=True)
class ModelInfo:
    purpose: str
    name: str
    version: str
    artifact_sha256: str = ""

    def validate(self, index: int | None = None) -> None:
        prefix = f"models[{index}]" if index is not None else "model"
        _nonempty_string(self.purpose, f"{prefix}.purpose")
        _nonempty_string(self.name, f"{prefix}.name")
        _nonempty_string(self.version, f"{prefix}.version")
        if self.artifact_sha256:
            digest = _nonempty_string(self.artifact_sha256, f"{prefix}.artifact_sha256")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ProjectValidationError(
                    f"{prefix}.artifact_sha256 必须是 64 位小写十六进制"
                )

    def to_dict(self) -> dict[str, str]:
        return {
            "purpose": self.purpose,
            "name": self.name,
            "version": self.version,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_dict(cls, value: object, index: int) -> ModelInfo:
        label = f"models[{index}]"
        data = _mapping(value, label)
        model = cls(
            purpose=_string_field(data, "purpose", label),
            name=_string_field(data, "name", label),
            version=_string_field(data, "version", label),
            artifact_sha256=_optional_string_field(data, "artifact_sha256", label, ""),
        )
        model.validate(index)
        return model


@dataclass(slots=True)
class ExportOptions:
    output_directory: str
    wav_enabled: bool = True
    mp3_enabled: bool = True
    mp3_bitrate_kbps: int = 192
    crossfade_ms: int = 8
    zero_crossing_search_ms: int = 60

    def validate(self) -> None:
        _nonempty_string(self.output_directory, "export_options.output_directory")
        _boolean(self.wav_enabled, "export_options.wav_enabled")
        _boolean(self.mp3_enabled, "export_options.mp3_enabled")
        if not self.wav_enabled and not self.mp3_enabled:
            raise ProjectValidationError("至少需要启用 WAV 或 MP3 中的一种导出格式")
        _positive_int(self.mp3_bitrate_kbps, "export_options.mp3_bitrate_kbps")
        _nonnegative_int(self.crossfade_ms, "export_options.crossfade_ms")
        _nonnegative_int(
            self.zero_crossing_search_ms,
            "export_options.zero_crossing_search_ms",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_directory": self.output_directory,
            "wav_enabled": self.wav_enabled,
            "mp3_enabled": self.mp3_enabled,
            "mp3_bitrate_kbps": self.mp3_bitrate_kbps,
            "crossfade_ms": self.crossfade_ms,
            "zero_crossing_search_ms": self.zero_crossing_search_ms,
        }

    @classmethod
    def from_dict(cls, value: object) -> ExportOptions:
        data = _mapping(value, "export_options")
        result = cls(
            output_directory=_string_field(data, "output_directory", "export_options"),
            wav_enabled=_optional_bool_field(data, "wav_enabled", "export_options", True),
            mp3_enabled=_optional_bool_field(data, "mp3_enabled", "export_options", True),
            mp3_bitrate_kbps=_optional_int_field(
                data, "mp3_bitrate_kbps", "export_options", 192
            ),
            crossfade_ms=_optional_int_field(data, "crossfade_ms", "export_options", 8),
            zero_crossing_search_ms=_optional_int_field(
                data,
                "zero_crossing_search_ms",
                "export_options",
                60,
            ),
        )
        result.validate()
        return result


@dataclass(slots=True)
class CutCandidate:
    id: str
    paragraph_index: int
    highlight_index: int
    text: str
    context_before: str
    context_after: str
    suggested_start_sample: int
    suggested_end_sample: int
    confidence: float
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    status: CandidateStatus = CandidateStatus.NEEDS_REVIEW
    review_required: bool = True
    final_start_sample: int | None = None
    final_end_sample: int | None = None
    source_start_char: int | None = None
    source_end_char: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, CandidateStatus):
            try:
                self.status = CandidateStatus(_enum_value(self.status))
            except (TypeError, ValueError) as exc:
                raise ProjectValidationError(f"未知候选状态: {self.status!r}") from exc
        # Keep a private copy: callers often pass a tuple or a model-owned list.
        self.reasons = list(self.reasons)
        self.diagnostics = dict(self.diagnostics)

    @classmethod
    def from_alignment(
        cls,
        alignment: object,
        *,
        candidate_id: str | None = None,
        context_before: str = "",
        context_after: str = "",
    ) -> CutCandidate:
        """Build from ``alignment.AlignmentCandidate`` without importing ML code."""

        paragraph_index = _object_int(alignment, "paragraph_index")
        highlight_index = _object_int(alignment, "highlight_index")
        requires_review = bool(getattr(alignment, "requires_review", False))
        raw_status = _enum_value(getattr(alignment, "status", ""))
        if raw_status == "auto_approved":
            status = CandidateStatus.AUTO_APPROVED
        elif raw_status == "approved":
            status = CandidateStatus.APPROVED
        elif raw_status == "skipped":
            status = CandidateStatus.SKIPPED
        else:
            status = CandidateStatus.NEEDS_REVIEW
            requires_review = True
        text = getattr(alignment, "text", None)
        if text is None:
            text = getattr(alignment, "highlighted_text", None)
        start = getattr(alignment, "suggested_start_sample", None)
        if start is None:
            start = getattr(alignment, "proposed_start_sample", None)
        end = getattr(alignment, "suggested_end_sample", None)
        if end is None:
            end = getattr(alignment, "proposed_end_sample", None)

        result = cls(
            id=candidate_id or f"p{paragraph_index:04d}-h{highlight_index:03d}",
            paragraph_index=paragraph_index,
            highlight_index=highlight_index,
            text=text,
            context_before=context_before,
            context_after=context_after,
            suggested_start_sample=start,
            suggested_end_sample=end,
            confidence=getattr(alignment, "confidence", None),
            reasons=list(getattr(alignment, "reasons", [])),
            diagnostics=dict(getattr(alignment, "diagnostics", {})),
            status=status,
            review_required=requires_review,
        )
        result.validate()
        return result

    @property
    def is_selected(self) -> bool:
        return self.status in {CandidateStatus.AUTO_APPROVED, CandidateStatus.APPROVED}

    @property
    def needs_review(self) -> bool:
        return self.status is CandidateStatus.NEEDS_REVIEW

    @property
    def effective_start_sample(self) -> int:
        return (
            self.final_start_sample
            if self.final_start_sample is not None
            else self.suggested_start_sample
        )

    @property
    def effective_end_sample(self) -> int:
        return (
            self.final_end_sample
            if self.final_end_sample is not None
            else self.suggested_end_sample
        )

    def approve(self, start_sample: int | None = None, end_sample: int | None = None) -> None:
        if (start_sample is None) != (end_sample is None):
            raise ProjectValidationError("人工切点必须同时提供开始和结束样本")
        self.final_start_sample = start_sample
        self.final_end_sample = end_sample
        self.status = CandidateStatus.APPROVED
        self.validate()

    def skip(self) -> None:
        self.status = CandidateStatus.SKIPPED

    def reset(self) -> None:
        self.final_start_sample = None
        self.final_end_sample = None
        self.status = (
            CandidateStatus.NEEDS_REVIEW
            if self.review_required
            else CandidateStatus.AUTO_APPROVED
        )

    def validate(self, index: int | None = None, total_samples: int | None = None) -> None:
        label = f"candidates[{index}]" if index is not None else "candidate"
        _nonempty_string(self.id, f"{label}.id")
        _nonnegative_int(self.paragraph_index, f"{label}.paragraph_index")
        _nonnegative_int(self.highlight_index, f"{label}.highlight_index")
        _nonempty_string(self.text, f"{label}.text")
        _string(self.context_before, f"{label}.context_before")
        _string(self.context_after, f"{label}.context_after")
        _sample_interval(
            self.suggested_start_sample,
            self.suggested_end_sample,
            f"{label}.suggested",
        )
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ProjectValidationError(f"{label}.confidence 必须是数字")
        if not math.isfinite(float(self.confidence)) or not 0 <= float(self.confidence) <= 1:
            raise ProjectValidationError(f"{label}.confidence 必须在 0 到 1 之间")
        self.confidence = float(self.confidence)
        if not isinstance(self.reasons, list) or any(not isinstance(item, str) for item in self.reasons):
            raise ProjectValidationError(f"{label}.reasons 必须是字符串数组")
        _diagnostic_mapping(self.diagnostics, f"{label}.diagnostics")
        if not isinstance(self.status, CandidateStatus):
            try:
                self.status = CandidateStatus(_enum_value(self.status))
            except (TypeError, ValueError) as exc:
                raise ProjectValidationError(f"{label}.status 无效") from exc
        _boolean(self.review_required, f"{label}.review_required")
        if self.review_required and self.status is CandidateStatus.AUTO_APPROVED:
            raise ProjectValidationError(
                f"{label} 标记为必须复核时不能处于自动通过状态"
            )
        if (self.source_start_char is None) != (self.source_end_char is None):
            raise ProjectValidationError(f"{label} 的原文开始和结束字符必须同时存在")
        if self.source_start_char is not None and self.source_end_char is not None:
            _sample_interval(
                self.source_start_char,
                self.source_end_char,
                f"{label}.source_char",
            )
        if (self.final_start_sample is None) != (self.final_end_sample is None):
            raise ProjectValidationError(f"{label} 的人工开始和结束样本必须同时存在")
        if self.final_start_sample is not None and self.final_end_sample is not None:
            _sample_interval(
                self.final_start_sample,
                self.final_end_sample,
                f"{label}.final",
            )
        if total_samples is not None and self.effective_end_sample > total_samples:
            raise ProjectValidationError(f"{label} 的结束样本超出音频时间轴")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "paragraph_index": self.paragraph_index,
            "highlight_index": self.highlight_index,
            "text": self.text,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "suggested_start_sample": self.suggested_start_sample,
            "suggested_end_sample": self.suggested_end_sample,
            "final_start_sample": self.final_start_sample,
            "final_end_sample": self.final_end_sample,
            "source_start_char": self.source_start_char,
            "source_end_char": self.source_end_char,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "diagnostics": dict(self.diagnostics),
            "status": self.status.value,
            "review_required": self.review_required,
        }

    @classmethod
    def from_dict(cls, value: object, index: int) -> CutCandidate:
        label = f"candidates[{index}]"
        data = _mapping(value, label)
        candidate = cls(
            id=_string_field(data, "id", label),
            paragraph_index=_int_field(data, "paragraph_index", label),
            highlight_index=_int_field(data, "highlight_index", label),
            text=_string_field(data, "text", label),
            context_before=_optional_string_field(data, "context_before", label, ""),
            context_after=_optional_string_field(data, "context_after", label, ""),
            suggested_start_sample=_int_field(data, "suggested_start_sample", label),
            suggested_end_sample=_int_field(data, "suggested_end_sample", label),
            final_start_sample=_optional_nullable_int_field(
                data, "final_start_sample", label
            ),
            final_end_sample=_optional_nullable_int_field(data, "final_end_sample", label),
            source_start_char=_optional_nullable_int_field(
                data, "source_start_char", label
            ),
            source_end_char=_optional_nullable_int_field(data, "source_end_char", label),
            confidence=_number_field(data, "confidence", label),
            reasons=_optional_string_list_field(data, "reasons", label),
            diagnostics=dict(_diagnostic_mapping(data.get("diagnostics", {}), f"{label}.diagnostics")),
            status=_string_field(data, "status", label),
            review_required=_optional_bool_field(data, "review_required", label, True),
        )
        candidate.validate(index)
        return candidate


@dataclass(slots=True)
class ProjectV1:
    audio: SourceFile
    document: SourceFile
    audio_info: AudioInfo
    candidates: list[CutCandidate] = field(default_factory=list)
    models: list[ModelInfo] = field(default_factory=list)
    analysis_diagnostics: dict[str, Any] = field(default_factory=dict)
    export_options: ExportOptions = field(
        default_factory=lambda: ExportOptions(output_directory=".")
    )
    app_version: str = DEFAULT_APP_VERSION
    project_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())
    schema_version: int = PROJECT_VERSION

    @classmethod
    def create(
        cls,
        audio_path: str | Path,
        document_path: str | Path,
        audio_info: AudioInfo,
        *,
        candidates: list[CutCandidate] | None = None,
        models: list[ModelInfo] | None = None,
        analysis_diagnostics: Mapping[str, Any] | None = None,
        output_directory: str | Path | None = None,
        app_version: str = DEFAULT_APP_VERSION,
    ) -> ProjectV1:
        audio = SourceFile.from_path(audio_path)
        document = SourceFile.from_path(document_path)
        output = Path(output_directory) if output_directory is not None else Path(audio.path).parent
        result = cls(
            audio=audio,
            document=document,
            audio_info=audio_info,
            candidates=list(candidates or []),
            models=list(models or []),
            analysis_diagnostics=dict(analysis_diagnostics or {}),
            export_options=ExportOptions(output_directory=str(output.resolve())),
            app_version=app_version,
        )
        result.validate()
        return result

    @property
    def audio_file(self) -> SourceFile:
        return self.audio

    @property
    def document_file(self) -> SourceFile:
        return self.document

    @property
    def unresolved_candidates(self) -> list[CutCandidate]:
        return [candidate for candidate in self.candidates if candidate.needs_review]

    @property
    def ready_to_export(self) -> bool:
        return not self.unresolved_candidates

    @property
    def selected_candidates(self) -> list[CutCandidate]:
        return [candidate for candidate in self.candidates if candidate.is_selected]

    def touch(self) -> None:
        self.updated_at = _now_iso()

    def relink_source(self, kind: str, path: str | Path) -> None:
        if kind == "audio":
            self.audio.relink(path)
        elif kind in {"document", "docx"}:
            self.document.relink(path)
        else:
            raise ValueError("kind must be 'audio' or 'document'")
        self.touch()

    def verify_source_files(self) -> dict[str, bool]:
        return {
            "audio": self.audio.matches_file(self.audio.path),
            "document": self.document.matches_file(self.document.path),
        }

    def validate(self, *, verify_files: bool = False) -> None:
        if self.schema_version != PROJECT_VERSION:
            raise ProjectValidationError(
                f"不支持项目版本 {self.schema_version!r}，当前仅支持 V{PROJECT_VERSION}"
            )
        self.audio.validate("inputs.audio")
        self.document.validate("inputs.document")
        self.audio_info.validate()
        _nonempty_string(self.app_version, "app_version")
        _nonempty_string(self.project_id, "project_id")
        _valid_iso_datetime(self.created_at, "created_at")
        _valid_iso_datetime(self.updated_at, "updated_at")
        if not isinstance(self.models, list):
            raise ProjectValidationError("models 必须是数组")
        purposes: set[str] = set()
        for index, model in enumerate(self.models):
            if not isinstance(model, ModelInfo):
                raise ProjectValidationError(f"models[{index}] 类型无效")
            model.validate(index)
            if model.purpose in purposes:
                raise ProjectValidationError(f"模型用途重复: {model.purpose}")
            purposes.add(model.purpose)
        _diagnostic_mapping(self.analysis_diagnostics, "analysis_diagnostics")
        if not isinstance(self.candidates, list):
            raise ProjectValidationError("candidates 必须是数组")
        ids: set[str] = set()
        locations: set[tuple[int, int]] = set()
        for index, candidate in enumerate(self.candidates):
            if not isinstance(candidate, CutCandidate):
                raise ProjectValidationError(f"candidates[{index}] 类型无效")
            candidate.validate(index, self.audio_info.total_samples)
            if candidate.id in ids:
                raise ProjectValidationError(f"候选 id 重复: {candidate.id}")
            ids.add(candidate.id)
            location = (candidate.paragraph_index, candidate.highlight_index)
            if location in locations:
                raise ProjectValidationError(f"候选原文位置重复: {location}")
            locations.add(location)
        if not isinstance(self.export_options, ExportOptions):
            raise ProjectValidationError("export_options 类型无效")
        self.export_options.validate()
        if verify_files:
            missing = [name for name, matches in self.verify_source_files().items() if not matches]
            if missing:
                raise ProjectValidationError(
                    f"输入文件缺失或内容不匹配: {', '.join(missing)}"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": PROJECT_SCHEMA,
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "app_version": self.app_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "inputs": {
                "audio": self.audio.to_dict(),
                "document": self.document.to_dict(),
            },
            "audio_info": self.audio_info.to_dict(),
            "models": [model.to_dict() for model in self.models],
            "analysis_diagnostics": dict(self.analysis_diagnostics),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "export_options": self.export_options.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> ProjectV1:
        data = _mapping(value, "project")
        if data.get("schema") != PROJECT_SCHEMA:
            raise ProjectValidationError(f"未知项目 schema: {data.get('schema')!r}")
        version = _int_field(data, "schema_version", "project")
        if version != PROJECT_VERSION:
            raise ProjectValidationError(
                f"不支持项目版本 {version}，当前仅支持 V{PROJECT_VERSION}"
            )
        inputs = _mapping(_required(data, "inputs", "project"), "inputs")
        models_data = _list_field(data, "models", "project")
        candidates_data = _list_field(data, "candidates", "project")
        project = cls(
            audio=SourceFile.from_dict(_required(inputs, "audio", "inputs"), "inputs.audio"),
            document=SourceFile.from_dict(
                _required(inputs, "document", "inputs"), "inputs.document"
            ),
            audio_info=AudioInfo.from_dict(_required(data, "audio_info", "project")),
            models=[ModelInfo.from_dict(item, index) for index, item in enumerate(models_data)],
            analysis_diagnostics=dict(
                _diagnostic_mapping(data.get("analysis_diagnostics", {}), "analysis_diagnostics")
            ),
            candidates=[
                CutCandidate.from_dict(item, index)
                for index, item in enumerate(candidates_data)
            ],
            export_options=ExportOptions.from_dict(
                _required(data, "export_options", "project")
            ),
            app_version=_string_field(data, "app_version", "project"),
            project_id=_string_field(data, "project_id", "project"),
            created_at=_string_field(data, "created_at", "project"),
            updated_at=_string_field(data, "updated_at", "project"),
            schema_version=version,
        )
        project.validate()
        return project


def save_project(project: ProjectV1, path: str | Path) -> Path:
    """Validate and atomically replace a UTF-8 V1 project JSON file."""

    if not isinstance(project, ProjectV1):
        raise TypeError("project must be ProjectV1")
    target = Path(path).expanduser()
    try:
        parent = target.parent.resolve()
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProjectIOError(f"无法创建项目目录 {target.parent}: {exc}") from exc

    project.touch()
    payload = project.to_dict()
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
        _sync_directory(parent)
    except OSError as exc:
        raise ProjectIOError(f"无法原子保存项目 {target}: {exc}") from exc
    finally:
        if temporary_name is not None:
            with suppress(OSError):
                Path(temporary_name).unlink(missing_ok=True)
    return target.resolve()


def load_project(path: str | Path, *, verify_files: bool = False) -> ProjectV1:
    source = Path(path).expanduser()
    try:
        raw = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ProjectIOError(f"无法读取项目 {source}: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProjectValidationError(f"项目 JSON 已损坏: {exc}") from exc
    project = ProjectV1.from_dict(value)
    project.validate(verify_files=verify_files)
    return project


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _sync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _enum_value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _object_int(value: object, name: str) -> int:
    result = getattr(value, name, None)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ProjectValidationError(f"alignment.{name} 必须是整数")
    return result


def _sample_interval(start: object, end: object, label: str) -> None:
    _nonnegative_int(start, f"{label}_start_sample")
    _nonnegative_int(end, f"{label}_end_sample")
    if start >= end:
        raise ProjectValidationError(f"{label} 的结束样本必须晚于开始样本")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectValidationError(f"{label} 必须是 JSON 对象")
    return value


def _diagnostic_mapping(value: object, label: str) -> Mapping[str, Any]:
    data = _mapping(value, label)
    _validate_diagnostic_value(data, label, depth=0)
    return data


def _validate_diagnostic_value(value: object, label: str, *, depth: int) -> None:
    if depth > 8:
        raise ProjectValidationError(f"{label} 嵌套过深")
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProjectValidationError(f"{label} 不能包含 NaN 或无穷大")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_diagnostic_value(item, f"{label}[{index}]", depth=depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProjectValidationError(f"{label} 的键必须是字符串")
            _validate_diagnostic_value(item, f"{label}.{key}", depth=depth + 1)
        return
    raise ProjectValidationError(f"{label} 只能包含 JSON 基础类型")


def _required(data: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in data:
        raise ProjectValidationError(f"{label} 缺少字段 {key}")
    return data[key]


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ProjectValidationError(f"{label} 必须是字符串")
    return value


def _nonempty_string(value: object, label: str) -> str:
    result = _string(value, label)
    if not result.strip():
        raise ProjectValidationError(f"{label} 不能为空")
    return result


def _string_field(data: Mapping[str, Any], key: str, label: str) -> str:
    return _string(_required(data, key, label), f"{label}.{key}")


def _optional_string_field(
    data: Mapping[str, Any], key: str, label: str, default: str
) -> str:
    return _string(data.get(key, default), f"{label}.{key}")


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ProjectValidationError(f"{label} 必须是布尔值")
    return value


def _optional_bool_field(
    data: Mapping[str, Any], key: str, label: str, default: bool
) -> bool:
    return _boolean(data.get(key, default), f"{label}.{key}")


def _int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProjectValidationError(f"{label} 必须是整数")
    return value


def _int_field(data: Mapping[str, Any], key: str, label: str) -> int:
    return _int(_required(data, key, label), f"{label}.{key}")


def _optional_int_field(
    data: Mapping[str, Any], key: str, label: str, default: int
) -> int:
    return _int(data.get(key, default), f"{label}.{key}")


def _optional_nullable_int_field(
    data: Mapping[str, Any], key: str, label: str
) -> int | None:
    value = data.get(key)
    return None if value is None else _int(value, f"{label}.{key}")


def _number_field(data: Mapping[str, Any], key: str, label: str) -> float:
    value = _required(data, key, label)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectValidationError(f"{label}.{key} 必须是数字")
    return float(value)


def _list_field(data: Mapping[str, Any], key: str, label: str) -> list[Any]:
    value = _required(data, key, label)
    if not isinstance(value, list):
        raise ProjectValidationError(f"{label}.{key} 必须是数组")
    return value


def _optional_string_list_field(
    data: Mapping[str, Any], key: str, label: str
) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ProjectValidationError(f"{label}.{key} 必须是字符串数组")
    return list(value)


def _positive_int(value: object, label: str) -> int:
    result = _int(value, label)
    if result <= 0:
        raise ProjectValidationError(f"{label} 必须大于 0")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    result = _int(value, label)
    if result < 0:
        raise ProjectValidationError(f"{label} 不能为负数")
    return result


def _valid_iso_datetime(value: object, label: str) -> None:
    text = _nonempty_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProjectValidationError(f"{label} 不是有效 ISO-8601 时间") from exc
    if parsed.tzinfo is None:
        raise ProjectValidationError(f"{label} 必须包含时区")


__all__ = [
    "AudioInfo",
    "CandidateStatus",
    "CutCandidate",
    "ExportOptions",
    "ModelInfo",
    "PROJECT_SCHEMA",
    "PROJECT_VERSION",
    "ProjectError",
    "ProjectIOError",
    "ProjectV1",
    "ProjectValidationError",
    "SourceFile",
    "hash_file",
    "load_project",
    "save_project",
    "sha256_file",
]
