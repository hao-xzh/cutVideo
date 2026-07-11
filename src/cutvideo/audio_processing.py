"""Project model for transcript-driven editing of one standalone audio file."""

from __future__ import annotations

import json
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


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class TranscriptToken:
    text: str
    start_sample: int
    end_sample: int
    confidence: float = 1.0

    def validate(self, total_samples: int) -> None:
        if not self.text:
            raise ProjectValidationError("转写文字不能为空")
        if not 0 <= self.start_sample < self.end_sample <= total_samples:
            raise ProjectValidationError("转写文字的样本范围无效")
        if not 0.0 <= self.confidence <= 1.0:
            raise ProjectValidationError("转写置信度必须位于 0 到 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "confidence": self.confidence,
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
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectValidationError("转写 token 字段无效") from exc


@dataclass(slots=True)
class AudioAnnotation:
    id: str
    operation: str
    text: str
    token_start: int
    token_end: int
    start_sample: int
    end_sample: int

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operation": self.operation,
            "text": self.text,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
        }

    @classmethod
    def from_dict(cls, value: object) -> AudioAnnotation:
        if not isinstance(value, dict):
            raise ProjectValidationError("音频标注必须是对象")
        try:
            return cls(
                id=str(value["id"]),
                operation=str(value["operation"]),
                text=str(value["text"]),
                token_start=int(value["token_start"]),
                token_end=int(value["token_end"]),
                start_sample=int(value["start_sample"]),
                end_sample=int(value["end_sample"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectValidationError("音频标注字段无效") from exc


@dataclass(slots=True)
class AudioProcessingProject:
    audio: SourceFile
    audio_info: AudioInfo
    tokens: list[TranscriptToken]
    annotations: list[AudioAnnotation] = field(default_factory=list)
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
        for token in self.tokens:
            token.validate(self.audio_info.total_samples)
            if token.start_sample < previous_start:
                raise ProjectValidationError("转写 token 时间必须递增")
            previous_start = token.start_sample
        ids: set[str] = set()
        for annotation in self.annotations:
            annotation.validate(len(self.tokens), self.audio_info.total_samples)
            if annotation.id in ids:
                raise ProjectValidationError("标注 id 重复")
            ids.add(annotation.id)
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
        annotation = AudioAnnotation(
            id=annotation_id,
            operation="delete",
            text="".join(item.text for item in self.tokens[token_start:token_end]),
            token_start=token_start,
            token_end=token_end,
            start_sample=start_sample,
            end_sample=end_sample,
        )
        self.annotations.append(annotation)
        self.annotations.sort(key=lambda item: (item.start_sample, item.end_sample))
        self.updated_at = _now()
        return annotation

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
            "annotations": [item.to_dict() for item in self.annotations],
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
                annotations=[
                    AudioAnnotation.from_dict(item) for item in value.get("annotations", [])
                ],
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
    "default_audio_processing_project_path",
    "load_audio_processing_project",
    "save_audio_processing_project",
]
