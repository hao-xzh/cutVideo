from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import cutvideo.ui.workers as workers
from cutvideo.alignment import (
    ALIGNMENT_PIPELINE_NAME,
    ALIGNMENT_PIPELINE_PURPOSE,
    ALIGNMENT_PIPELINE_VERSION,
)
from cutvideo.audio import AudioInfo, WaveformEnvelope
from cutvideo.ffmpeg import ExportResult, FFmpegTools
from cutvideo.project import (
    AudioInfo as ProjectAudioInfo,
)
from cutvideo.project import CandidateStatus, CutCandidate, ModelInfo, ProjectV1
from cutvideo.resources import RuntimeResources
from cutvideo.ui.workers import CancelToken


def _fake_envelope() -> WaveformEnvelope:
    zeros = np.zeros(4, dtype=np.float32)
    return WaveformEnvelope(48_000, 256, zeros, zeros.copy(), zeros.copy())


def _ready_project(tmp_path: Path) -> ProjectV1:
    audio = tmp_path / "source.mp3"
    document = tmp_path / "source.docx"
    audio.write_bytes(b"original-audio")
    document.write_bytes(b"original-document")
    candidate = CutCandidate(
        id="p0000-h000",
        paragraph_index=0,
        highlight_index=0,
        text="删除",
        context_before="前",
        context_after="后",
        suggested_start_sample=10,
        suggested_end_sample=20,
        confidence=0.5,
        status=CandidateStatus.SKIPPED,
        review_required=True,
    )
    return ProjectV1.create(
        audio,
        document,
        ProjectAudioInfo(48_000, 2, 48_000, "mp3", "mp3"),
        candidates=[candidate],
        models=[
            ModelInfo(
                ALIGNMENT_PIPELINE_PURPOSE,
                ALIGNMENT_PIPELINE_NAME,
                ALIGNMENT_PIPELINE_VERSION,
            )
        ],
    )


def test_preflight_rejects_file_changed_while_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "source.mp3"
    document = tmp_path / "source.docx"
    audio.write_bytes(b"first")
    document.write_bytes(b"document")

    class Transcript:
        def validate_audio_duration(self, _duration_ms: int) -> None:
            return None

    def fake_probe(*_args: object, **_kwargs: object) -> tuple[AudioInfo, WaveformEnvelope]:
        audio.write_bytes(b"changed")
        return AudioInfo(audio, 48_000, 2, 48_000, 1.0, "mp3"), _fake_envelope()

    resources = RuntimeResources(tmp_path, tmp_path / "ffmpeg", tmp_path / "ffprobe", None, None, None)
    tools = FFmpegTools(tmp_path / "ffmpeg", tmp_path / "ffprobe")
    monkeypatch.setattr(workers, "parse_docx", lambda _path: Transcript())
    monkeypatch.setattr(workers, "probe_audio_with_waveform", fake_probe)
    monkeypatch.setattr(workers, "discover_resources", lambda: resources)
    monkeypatch.setattr(workers, "discover_ffmpeg", lambda **_kwargs: tools)

    operation = workers.make_preflight_operation(str(audio), str(document))
    with pytest.raises(ValueError, match="发生变化"):
        operation(CancelToken(), lambda _value, _message: None)


def test_export_rejects_changed_source_before_writing(tmp_path: Path) -> None:
    project = _ready_project(tmp_path)
    Path(project.audio.path).write_bytes(b"replacement")
    output = tmp_path / "output"
    operation = workers.make_export_operation(
        project,
        tmp_path / "source.cutvideo.json",
        output,
        FFmpegTools(tmp_path / "ffmpeg", tmp_path / "ffprobe"),
    )

    with pytest.raises(ValueError, match="原音频内容已变化"):
        operation(CancelToken(), lambda _value, _message: None)

    assert not output.exists()


def test_preflight_reuses_sidecar_after_all_inputs_are_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "original"
    moved = tmp_path / "moved"
    original.mkdir()
    moved.mkdir()
    project = _ready_project(original)
    sidecar = original / "source.cutvideo.json"
    from cutvideo.project import save_project

    save_project(project, sidecar)
    moved_audio = Path(project.audio.path).replace(moved / "source.mp3")
    moved_document = Path(project.document.path).replace(moved / "source.docx")
    moved_sidecar = sidecar.replace(moved / sidecar.name)

    class Transcript:
        def validate_audio_duration(self, _duration_ms: int) -> None:
            return None

    resources = RuntimeResources(tmp_path, tmp_path / "ffmpeg", tmp_path / "ffprobe", None, None, None)
    tools = FFmpegTools(tmp_path / "ffmpeg", tmp_path / "ffprobe")
    monkeypatch.setattr(workers, "parse_docx", lambda _path: Transcript())
    monkeypatch.setattr(
        workers,
        "probe_audio_with_waveform",
        lambda *_args, **_kwargs: (
            AudioInfo(moved_audio, 48_000, 2, 48_000, 1.0, "mp3"),
            _fake_envelope(),
        ),
    )
    monkeypatch.setattr(workers, "discover_resources", lambda: resources)
    monkeypatch.setattr(workers, "discover_ffmpeg", lambda **_kwargs: tools)

    result = workers.make_preflight_operation(
        str(moved_audio), str(moved_document)
    )(CancelToken(), lambda _value, _message: None)

    assert result.reused_project is not None
    assert Path(result.reused_project.audio.path) == moved_audio
    assert Path(result.reused_project.document.path) == moved_document
    assert Path(result.reused_project.export_options.output_directory) == moved
    assert result.reused_project_path == moved_sidecar


def test_preflight_does_not_reuse_previous_alignment_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _ready_project(tmp_path)
    project.models = []
    from cutvideo.project import save_project

    sidecar = Path(project.audio.path).with_suffix(".cutvideo.json")
    save_project(project, sidecar)

    class Transcript:
        def validate_audio_duration(self, _duration_ms: int) -> None:
            return None

    resources = RuntimeResources(
        tmp_path,
        tmp_path / "ffmpeg",
        tmp_path / "ffprobe",
        None,
        None,
        None,
    )
    tools = FFmpegTools(tmp_path / "ffmpeg", tmp_path / "ffprobe")
    monkeypatch.setattr(workers, "parse_docx", lambda _path: Transcript())
    monkeypatch.setattr(
        workers,
        "probe_audio_with_waveform",
        lambda *_args, **_kwargs: (
            AudioInfo(Path(project.audio.path), 48_000, 2, 48_000, 1.0, "mp3"),
            _fake_envelope(),
        ),
    )
    monkeypatch.setattr(workers, "discover_resources", lambda: resources)
    monkeypatch.setattr(workers, "discover_ffmpeg", lambda **_kwargs: tools)

    result = workers.make_preflight_operation(
        project.audio.path,
        project.document.path,
    )(CancelToken(), lambda _value, _message: None)

    assert result.reused_project is None


def test_open_project_rejects_previous_alignment_strategy(tmp_path: Path) -> None:
    project = _ready_project(tmp_path)
    project.models = []
    from cutvideo.project import save_project

    sidecar = tmp_path / "source.cutvideo.json"
    save_project(project, sidecar)

    with pytest.raises(ValueError, match="旧版 Word 时间主导"):
        workers.make_load_project_operation(str(sidecar))(
            CancelToken(),
            lambda _value, _message: None,
        )


def test_export_rejects_same_length_change_during_pcm_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _ready_project(tmp_path)
    source = Path(project.audio.path)

    def changing_probe(*_args: object, **_kwargs: object) -> AudioInfo:
        source.write_bytes(source.read_bytes()[::-1])
        return AudioInfo(source, 48_000, 2, 48_000, 1.0, "mp3")

    monkeypatch.setattr(workers, "probe_audio", changing_probe)
    output = tmp_path / "output"
    operation = workers.make_export_operation(
        project,
        tmp_path / "source.cutvideo.json",
        output,
        FFmpegTools(tmp_path / "ffmpeg", tmp_path / "ffprobe"),
    )

    with pytest.raises(ValueError, match="核对过程中"):
        operation(CancelToken(), lambda _value, _message: None)

    assert not output.exists()


def test_export_rejects_change_before_group_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _ready_project(tmp_path)
    source = Path(project.audio.path)
    monkeypatch.setattr(
        workers,
        "probe_audio",
        lambda *_args, **_kwargs: AudioInfo(source, 48_000, 2, 48_000, 1.0, "mp3"),
    )

    def changing_export(
        _source: str | Path,
        _intervals: object,
        wav_path: Path,
        mp3_path: Path,
        **_kwargs: object,
    ) -> ExportResult:
        wav_path.write_bytes(b"wav")
        mp3_path.write_bytes(b"mp3")
        source.write_bytes(source.read_bytes()[::-1])
        return ExportResult(wav_path, mp3_path, (), 47_990, 10)

    monkeypatch.setattr(workers, "export_audio", changing_export)
    output = tmp_path / "output"
    operation = workers.make_export_operation(
        project,
        tmp_path / "source.cutvideo.json",
        output,
        FFmpegTools(tmp_path / "ffmpeg", tmp_path / "ffprobe"),
    )

    with pytest.raises(ValueError, match="导出过程中"):
        operation(CancelToken(), lambda _value, _message: None)

    assert output.is_dir()
    assert not list(output.iterdir())


def test_relink_moves_default_output_directory_with_audio(tmp_path: Path) -> None:
    original = tmp_path / "original"
    moved = tmp_path / "moved"
    original.mkdir()
    moved.mkdir()
    project = _ready_project(original)
    from cutvideo.project import save_project

    project_path = tmp_path / "source.cutvideo.json"
    save_project(project, project_path)
    moved_audio = Path(project.audio.path).replace(moved / "source.mp3")
    moved_document = Path(project.document.path).replace(moved / "source.docx")

    result = workers.make_relink_project_operation(
        project_path, str(moved_audio), str(moved_document)
    )(CancelToken(), lambda _value, _message: None)

    assert Path(result.project.audio.path) == moved_audio
    assert Path(result.project.document.path) == moved_document
    assert Path(result.project.export_options.output_directory) == moved


def test_artifact_group_rolls_back_every_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = []
    targets = []
    for index in range(4):
        source = tmp_path / f"staged-{index}"
        target = tmp_path / f"target-{index}"
        source.write_text(f"new-{index}", encoding="utf-8")
        target.write_text(f"old-{index}", encoding="utf-8")
        staged.append(source)
        targets.append(target)

    real_replace = os.replace
    failed = False

    def flaky_replace(source: str | Path, target: str | Path) -> None:
        nonlocal failed
        if Path(source) == staged[2] and Path(target) == targets[2] and not failed:
            failed = True
            raise OSError("disk full")
        real_replace(source, target)

    monkeypatch.setattr(workers.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="disk full"):
        workers._commit_artifacts(tuple(zip(staged, targets, strict=True)))

    assert [path.read_text(encoding="utf-8") for path in targets] == [
        "old-0",
        "old-1",
        "old-2",
        "old-3",
    ]
    assert not list(tmp_path.glob(".*.bak"))


def test_artifact_group_rejects_directory_collision_without_moving_it(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staged.wav"
    target = tmp_path / "output.wav"
    staged.write_bytes(b"new")
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="不是普通文件"):
        workers._commit_artifacts(((staged, target),))

    assert marker.read_text(encoding="utf-8") == "old"
    assert staged.read_bytes() == b"new"
    assert not list(tmp_path.glob(".*.bak"))


def test_long_audio_waveform_uses_five_millisecond_buckets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = AudioInfo(
        tmp_path / "audio.mp3",
        48_000,
        2,
        round(2_507.8 * 48_000),
        2_507.8,
        "mp3",
    )
    captured: dict[str, int] = {}
    envelope = WaveformEnvelope(
        48_000,
        240,
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )

    def fake_read(*_args: object, **kwargs: object) -> list[WaveformEnvelope]:
        captured["target_points"] = int(kwargs["target_points"])
        return [envelope]

    monkeypatch.setattr(workers, "read_waveform_envelopes", fake_read)
    operation = workers.make_waveform_operation(
        info, FFmpegTools(tmp_path / "ffmpeg", tmp_path / "ffprobe")
    )

    assert operation(CancelToken(), lambda _value, _message: None) is envelope
    assert captured["target_points"] == round(info.duration_seconds * 200)
