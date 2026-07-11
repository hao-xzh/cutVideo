from __future__ import annotations

import json
from pathlib import Path

import pytest

from cutvideo.project import (
    AudioInfo,
    CandidateStatus,
    CutCandidate,
    ExportOptions,
    ModelInfo,
    ProjectV1,
    ProjectValidationError,
    SourceFile,
    load_project,
    save_project,
    sha256_file,
)


def _candidate(*, status: CandidateStatus = CandidateStatus.NEEDS_REVIEW) -> CutCandidate:
    return CutCandidate(
        id="p0001-h000",
        paragraph_index=1,
        highlight_index=0,
        text="删除",
        context_before="需要",
        context_after="的内容",
        suggested_start_sample=4_800,
        suggested_end_sample=9_600,
        confidence=0.72,
        reasons=["片段较短"],
        status=status,
        review_required=True,
    )


def _project(tmp_path: Path) -> ProjectV1:
    audio_path = tmp_path / "中文 音频.mp3"
    document_path = tmp_path / "原文.docx"
    audio_path.write_bytes(b"audio-data")
    document_path.write_bytes(b"document-data")
    return ProjectV1(
        audio=SourceFile.from_path(audio_path),
        document=SourceFile.from_path(document_path),
        audio_info=AudioInfo(
            sample_rate=48_000,
            channels=2,
            total_samples=480_000,
            format_name="mp3",
            codec_name="mp3",
        ),
        candidates=[_candidate()],
        models=[
            ModelInfo(purpose="forced_alignment", name="fa-zh", version="1.0"),
            ModelInfo(purpose="asr", name="paraformer-zh", version="1.0"),
        ],
        export_options=ExportOptions(output_directory=str(tmp_path)),
    )


def test_sha256_source_matching_and_safe_relink(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    moved = tmp_path / "moved.bin"
    wrong = tmp_path / "wrong.bin"
    first.write_bytes(b"abc")
    moved.write_bytes(b"abc")
    wrong.write_bytes(b"abd")

    source = SourceFile.from_path(first)

    assert sha256_file(first) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert source.matches_file(moved)
    assert not source.matches_file(wrong)
    source.relink(moved)
    assert Path(source.path) == moved.resolve()
    with pytest.raises(ProjectValidationError, match="不匹配"):
        source.relink(wrong)


def test_project_round_trip_preserves_unicode_samples_models_and_options(tmp_path: Path) -> None:
    project = _project(tmp_path)
    output = tmp_path / "示例.cutvideo.json"

    saved_path = save_project(project, output)
    loaded = load_project(saved_path, verify_files=True)

    assert loaded.to_dict() == project.to_dict()
    assert loaded.audio_info.duration_ms == 10_000
    assert loaded.candidates[0].text == "删除"
    assert loaded.models[0].name == "fa-zh"
    assert loaded.export_options.mp3_bitrate_kbps == 192
    assert "删除" in output.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".示例.cutvideo.json.*.tmp"))


def test_review_workflow_blocks_export_until_approved_or_skipped(tmp_path: Path) -> None:
    project = _project(tmp_path)
    candidate = project.candidates[0]

    assert not project.ready_to_export
    candidate.approve(5_000, 9_000)
    assert project.ready_to_export
    assert candidate.is_selected
    assert (candidate.effective_start_sample, candidate.effective_end_sample) == (5_000, 9_000)

    candidate.reset()
    assert candidate.status is CandidateStatus.NEEDS_REVIEW
    candidate.skip()
    assert project.ready_to_export
    assert not candidate.is_selected


def test_validation_rejects_out_of_timeline_and_duplicate_candidates(tmp_path: Path) -> None:
    project = _project(tmp_path)
    project.candidates[0].suggested_end_sample = project.audio_info.total_samples + 1
    with pytest.raises(ProjectValidationError, match="超出音频时间轴"):
        project.validate()

    project = _project(tmp_path)
    project.candidates.append(_candidate())
    with pytest.raises(ProjectValidationError, match="候选 id 重复"):
        project.validate()


def test_invalid_project_does_not_replace_existing_file(tmp_path: Path) -> None:
    project = _project(tmp_path)
    output = tmp_path / "project.cutvideo.json"
    save_project(project, output)
    original = output.read_bytes()
    project.candidates[0].confidence = 2.0

    with pytest.raises(ProjectValidationError, match="0 到 1"):
        save_project(project, output)

    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".project.cutvideo.json.*.tmp"))


def test_load_rejects_unknown_version_and_corrupt_json(tmp_path: Path) -> None:
    project = _project(tmp_path)
    output = tmp_path / "project.cutvideo.json"
    save_project(project, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectValidationError, match="不支持项目版本"):
        load_project(output)

    output.write_text("{broken", encoding="utf-8")
    with pytest.raises(ProjectValidationError, match="JSON 已损坏"):
        load_project(output)


def test_verify_files_detects_changed_source(tmp_path: Path) -> None:
    project = _project(tmp_path)
    output = tmp_path / "project.cutvideo.json"
    save_project(project, output)
    Path(project.audio.path).write_bytes(b"changed")

    with pytest.raises(ProjectValidationError, match="audio"):
        load_project(output, verify_files=True)


def test_audio_duration_is_derived_from_pcm_samples() -> None:
    info = AudioInfo(sample_rate=48_000, channels=2, total_samples=120_377_088)

    assert info.duration_ms == 2_507_856
    assert info.duration_seconds == pytest.approx(2_507.856)


def test_required_review_cannot_masquerade_as_auto_approved() -> None:
    candidate = _candidate(status=CandidateStatus.AUTO_APPROVED)

    with pytest.raises(ProjectValidationError, match="必须复核"):
        candidate.validate()


def test_source_character_offsets_survive_project_round_trip(tmp_path: Path) -> None:
    project = _project(tmp_path)
    project.candidates[0].source_start_char = 12
    project.candidates[0].source_end_char = 14

    loaded = ProjectV1.from_dict(project.to_dict())

    assert loaded.candidates[0].source_start_char == 12
    assert loaded.candidates[0].source_end_char == 14
