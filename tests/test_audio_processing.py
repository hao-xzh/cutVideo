from __future__ import annotations

from pathlib import Path

from cutvideo.audio_processing import (
    AudioProcessingProject,
    TranscriptToken,
    load_audio_processing_project,
    save_audio_processing_project,
)
from cutvideo.project import AudioInfo, SourceFile


def _project(tmp_path: Path) -> AudioProcessingProject:
    audio = tmp_path / "访谈.wav"
    audio.write_bytes(b"fixture-audio")
    return AudioProcessingProject(
        audio=SourceFile.from_path(audio),
        audio_info=AudioInfo(1_000, 1, 10_000, "wav", "pcm_s16le"),
        tokens=[
            TranscriptToken("你", 1_000, 1_200),
            TranscriptToken("好", 1_200, 1_450),
            TranscriptToken("世", 2_000, 2_250),
            TranscriptToken("界", 2_250, 2_500),
        ],
        output_directory=str(tmp_path),
    )


def test_audio_processing_project_round_trip(tmp_path: Path) -> None:
    project = _project(tmp_path)
    annotation = project.add_delete_annotation(0, 2)
    annotation.start_sample = 980
    annotation.end_sample = 1_480
    path = tmp_path / "访谈.audioprocess.json"

    save_audio_processing_project(project, path)
    loaded = load_audio_processing_project(path)

    assert "".join(item.text for item in loaded.tokens) == "你好世界"
    assert loaded.annotations[0].text == "你好"
    assert loaded.annotations[0].start_sample == 980
    assert loaded.deletion_intervals[0].operation == "delete"


def test_overlapping_delete_annotations_are_merged(tmp_path: Path) -> None:
    project = _project(tmp_path)
    first = project.add_delete_annotation(0, 2)
    second = project.add_delete_annotation(1, 3)

    assert len(project.annotations) == 1
    assert second.id == first.id
    assert second.text == "你好世"
    assert (second.token_start, second.token_end) == (0, 3)
    assert project.remove_annotation(second.id)
    assert project.annotations == []
