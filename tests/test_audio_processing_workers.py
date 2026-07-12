from __future__ import annotations

from pathlib import Path

import cutvideo.ui.audio_processing_workers as workers
from cutvideo.alignment import RecognizedToken
from cutvideo.audio import AudioInfo
from cutvideo.audio_processing import AudioProcessingProject, TranscriptToken
from cutvideo.ffmpeg import ExportResult, FFmpegTools
from cutvideo.project import AudioInfo as ProjectAudioInfo
from cutvideo.project import SourceFile
from cutvideo.resources import RuntimeResources
from cutvideo.ui.workers import CancelToken


def test_audio_processing_analysis_builds_sample_exact_transcript_project(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "standalone.wav"
    audio.write_bytes(b"audio-fixture")
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffprobe = tmp_path / "ffprobe.exe"
    ffmpeg.write_bytes(b"bin")
    ffprobe.write_bytes(b"bin")
    asr = tmp_path / "asr"
    vad = tmp_path / "vad"
    asr.mkdir()
    vad.mkdir()
    resources = RuntimeResources(tmp_path, ffmpeg, ffprobe, None, asr, vad)

    class FakeRecognizer:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def recognize(self, **_kwargs) -> tuple[RecognizedToken, ...]:
            return (
                RecognizedToken("你", 1_000, 1_200),
                RecognizedToken("好", 1_200, 1_500),
            )

    monkeypatch.setattr(workers, "discover_resources", lambda: resources)
    monkeypatch.setattr(
        workers,
        "probe_audio",
        lambda *_args, **_kwargs: AudioInfo(audio, 1_000, 1, 5_000, 5.0, "pcm"),
    )
    monkeypatch.setattr(workers, "FunASRAsrAligner", FakeRecognizer)

    result = workers.make_audio_processing_analysis_operation(str(audio))(
        CancelToken(),
        lambda _value, _message: None,
    )

    assert [item.text for item in result.project.tokens] == ["你", "好"]
    assert [(item.start_sample, item.end_sample) for item in result.project.tokens] == [
        (1_000, 1_200),
        (1_200, 1_500),
    ]
    assert result.project_path.name == "standalone.audioprocess.json"
    assert result.project_path.is_file()


def test_audio_processing_export_uses_delete_annotations_and_expected_names(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "standalone.wav"
    audio.write_bytes(b"audio-fixture")
    project = AudioProcessingProject(
        audio=SourceFile.from_path(audio),
        audio_info=ProjectAudioInfo(1_000, 1, 5_000, "wav", "pcm"),
        tokens=[TranscriptToken("删", 1_000, 1_200), TranscriptToken("除", 1_200, 1_500)],
        output_directory=str(tmp_path),
    )
    project.add_delete_annotation(0, 2)
    project_path = tmp_path / "source.audioprocess.json"
    captured: dict[str, object] = {}

    def fake_export(audio_path, intervals, wav_path, mp3_path, **_kwargs):
        captured["audio"] = audio_path
        captured["intervals"] = list(intervals)
        Path(wav_path).write_bytes(b"wav")
        Path(mp3_path).write_bytes(b"mp3")
        return ExportResult(Path(wav_path), Path(mp3_path), (), 4_500, 500)

    monkeypatch.setattr(workers, "export_audio", fake_export)
    result = workers.make_audio_processing_export_operation(
        project,
        project_path,
        tools=FFmpegTools(tmp_path / "ffmpeg", tmp_path / "ffprobe"),
    )(CancelToken(), lambda _value, _message: None)

    assert result.wav_path.name == "standalone_处理完成.wav"
    assert result.mp3_path.name == "standalone_处理完成.mp3"
    assert result.project_path.name == "standalone.audioprocess.json"
    assert len(captured["intervals"]) == 1


def test_audio_processing_preview_keeps_three_seconds_of_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "standalone.wav"
    audio.write_bytes(b"audio-fixture")
    project = AudioProcessingProject(
        audio=SourceFile.from_path(audio),
        audio_info=ProjectAudioInfo(1_000, 1, 10_000, "wav", "pcm"),
        tokens=[TranscriptToken("删", 4_000, 4_200), TranscriptToken("除", 4_200, 4_500)],
        output_directory=str(tmp_path),
    )
    project.add_delete_annotation(0, 2)
    calls: list[dict[str, object]] = []

    def fake_preview(audio_path, intervals, original, edited, **kwargs):
        calls.append(
            {
                "audio": audio_path,
                "intervals": list(intervals),
                "original": Path(original),
                "edited": Path(edited),
                **kwargs,
            }
        )

    monkeypatch.setattr(workers, "generate_preview", fake_preview)
    result = workers.make_audio_processing_preview_operation(
        project,
        start_sample=4_000,
        end_sample=4_500,
        tools=FFmpegTools(tmp_path / "ffmpeg", tmp_path / "ffprobe"),
        preview_directory=tmp_path / "preview",
    )(CancelToken(), lambda _value, _message: None)

    assert len(calls) == 2
    assert (calls[0]["start_sample"], calls[0]["end_sample"]) == (1_000, 7_500)
    assert calls[0]["intervals"][-1] == (4_000, 4_500)
    assert (calls[1]["start_sample"], calls[1]["end_sample"]) == (4_000, 4_500)
    assert calls[1]["intervals"] == []
    assert result.original_wav_path.name == "audio-processing-original.wav"
    assert result.selection_wav_path.name == "audio-processing-selection.wav"
