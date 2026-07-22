from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import cutvideo.ui.audio_processing_workers as workers
from cutvideo.alignment import RecognizedToken
from cutvideo.audio import AudioInfo
from cutvideo.audio_processing import AudioProcessingProject, TranscriptToken
from cutvideo.ffmpeg import ExportResult, FFmpegTools
from cutvideo.progressive_asr import ProgressiveRecognitionResult
from cutvideo.project import AudioInfo as ProjectAudioInfo
from cutvideo.project import SourceFile
from cutvideo.resources import RuntimeResources
from cutvideo.ui.workers import CancelToken


@contextmanager
def _passthrough_audio_window(audio_path, **_kwargs):
    yield Path(audio_path)


def _fake_environment(tmp_path: Path, *, with_fa: bool = False) -> tuple[Path, RuntimeResources]:
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
    fa = None
    if with_fa:
        fa = tmp_path / "fa"
        fa.mkdir()
    return audio, RuntimeResources(tmp_path, ffmpeg, ffprobe, fa, asr, vad)


def _fake_progressive_result() -> ProgressiveRecognitionResult:
    return ProgressiveRecognitionResult(
        (
            RecognizedToken("你", 1_000, 1_200),
            RecognizedToken("好", 1_200, 1_500),
        ),
        ((900.0, 1_600.0),),
        1,
    )


def test_audio_processing_analysis_builds_sample_exact_transcript_project(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio, resources = _fake_environment(tmp_path)

    monkeypatch.setattr(workers, "discover_resources", lambda: resources)
    monkeypatch.setattr(workers, "resolve_inference_device", lambda: "cpu")
    monkeypatch.setattr(
        workers,
        "probe_audio_metadata",
        lambda *_args, **_kwargs: AudioInfo(audio, 1_000, 1, 5_000, 5.0, "pcm"),
    )
    monkeypatch.setattr(
        workers,
        "probe_audio_with_waveform",
        lambda *_args, **_kwargs: (AudioInfo(audio, 1_000, 1, 5_000, 5.0, "pcm"), None),
    )
    monkeypatch.setattr(
        workers,
        "recognize_audio_progressively",
        lambda **_kwargs: _fake_progressive_result(),
    )

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
    assert result.project.segment_starts == [0]
    assert result.refinement_pending is False
    assert result.project.analysis_diagnostics["character_refinement_status"] == "skipped"
    assert "character_refinement" not in result.project.analysis_diagnostics
    assert result.project.analysis_diagnostics["voice_ranges_ms"] == [[900.0, 1_600.0]]


def test_audio_processing_analysis_defers_character_refinement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio, resources = _fake_environment(tmp_path, with_fa=True)

    monkeypatch.setattr(workers, "discover_resources", lambda: resources)
    monkeypatch.setattr(workers, "resolve_inference_device", lambda: "cpu")
    monkeypatch.setattr(
        workers,
        "probe_audio_metadata",
        lambda *_args, **_kwargs: AudioInfo(audio, 1_000, 1, 5_000, 5.0, "pcm"),
    )
    monkeypatch.setattr(
        workers,
        "probe_audio_with_waveform",
        lambda *_args, **_kwargs: (AudioInfo(audio, 1_000, 1, 5_000, 5.0, "pcm"), None),
    )
    monkeypatch.setattr(
        workers,
        "recognize_audio_progressively",
        lambda **_kwargs: _fake_progressive_result(),
    )

    def fail_if_refine(*_args, **_kwargs):
        raise AssertionError("draft analysis must not run fa-zh inline")

    monkeypatch.setattr(workers, "refine_recognition_tokens", fail_if_refine)

    result = workers.make_audio_processing_analysis_operation(str(audio))(
        CancelToken(),
        lambda _value, _message: None,
    )

    assert result.refinement_pending is True
    assert result.project.analysis_diagnostics["character_refinement_status"] == "pending"
    assert [(item.start_sample, item.end_sample) for item in result.project.tokens] == [
        (1_000, 1_200),
        (1_200, 1_500),
    ]


@contextmanager
def _passthrough_guard(*_args, **_kwargs):
    yield


def test_audio_processing_refinement_applies_character_timestamps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio, resources = _fake_environment(tmp_path, with_fa=True)
    project = AudioProcessingProject(
        audio=SourceFile.from_path(audio),
        audio_info=ProjectAudioInfo(1_000, 1, 5_000, "wav", "pcm"),
        tokens=[
            TranscriptToken("你", 1_000, 1_200, timestamp_precision="token"),
            TranscriptToken("好", 1_200, 1_500, timestamp_precision="token"),
        ],
        analysis_diagnostics={"character_refinement_status": "pending"},
        output_directory=str(tmp_path),
    )
    project_path = tmp_path / "standalone.audioprocess.json"

    monkeypatch.setattr(workers, "discover_resources", lambda: resources)
    monkeypatch.setattr(workers, "resolve_inference_device", lambda: "cpu")
    monkeypatch.setattr(workers, "FunASRForceAligner", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(workers, "model_execution_guard", _passthrough_guard)

    def fake_refine(tokens, **_kwargs):
        refined = tuple(
            RecognizedToken(
                token.text,
                token.start_ms + 50,
                token.end_ms - 50,
                token.confidence,
                token.confidence_available,
                "character",
            )
            for token in tokens
        )
        return refined, {"engine": "fa-zh-character-refinement", "refined_token_count": 2}

    monkeypatch.setattr(workers, "refine_recognition_tokens", fake_refine)

    result = workers.make_audio_processing_refinement_operation(
        project,
        project_path,
        audio_info=AudioInfo(audio, 1_000, 1, 5_000, 5.0, "pcm"),
        tools=FFmpegTools(tmp_path / "ffmpeg", tmp_path / "ffprobe"),
    )(CancelToken(), lambda _value, _message: None)

    assert [(item.start_sample, item.end_sample) for item in result.project.tokens] == [
        (1_050, 1_150),
        (1_250, 1_450),
    ]
    assert all(item.timestamp_precision == "character" for item in result.project.tokens)
    assert result.updated_token_count == 2
    assert result.project.analysis_diagnostics["character_refinement_status"] == "completed"
    refinement = result.project.analysis_diagnostics["character_refinement"]
    assert refinement["refined_token_count"] == 2
    assert project_path.is_file()


def test_vad_ranges_create_transcript_line_starts() -> None:
    tokens = [
        TranscriptToken("甲", 1_000, 1_200),
        TranscriptToken("乙", 1_200, 1_400),
        TranscriptToken("丙", 2_100, 2_300),
        TranscriptToken("丁", 2_300, 2_500),
    ]

    starts = workers._segment_starts_from_voice_ranges(
        tokens,
        ((900.0, 1_500.0), (2_000.0, 2_600.0)),
        1_000,
    )

    assert starts == [0, 2]


def test_voice_ranges_are_read_from_analysis_diagnostics() -> None:
    ranges = workers._voice_ranges_ms_from_diagnostics(
        {"voice_ranges_ms": [[100.0, 500.0], [800, 1200], ["bad"], [1]]}
    )
    assert ranges == ((100.0, 500.0), (800.0, 1200.0))


def test_annotation_boundary_refinement_uses_force_realign_and_vad(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "standalone.wav"
    audio.write_bytes(b"audio-fixture")
    info = AudioInfo(audio, 1_000, 1, 5_000, 5.0, "pcm")
    fa = tmp_path / "fa"
    fa.mkdir()
    resources = RuntimeResources(
        tmp_path,
        tmp_path / "ffmpeg",
        tmp_path / "ffprobe",
        fa,
        tmp_path / "asr",
        tmp_path / "vad",
    )
    (tmp_path / "ffmpeg").write_bytes(b"bin")
    (tmp_path / "ffprobe").write_bytes(b"bin")
    (tmp_path / "asr").mkdir()
    (tmp_path / "vad").mkdir()

    class _Force:
        def align(self, **kwargs):
            assert kwargs["transcript"] == "你好"
            from cutvideo.alignment import AlignmentTrack, TimedSpan

            return AlignmentTrack(
                (
                    TimedSpan(0, 1, 1_050.0, 1_150.0, 1.0, None, "character", 1.0),
                    TimedSpan(1, 2, 1_150.0, 1_450.0, 1.0, None, "character", 1.0),
                ),
                "fa-zh",
                1.0,
                timestamp_precision="character",
                timestamp_valid=True,
            )

    captured: dict[str, object] = {}

    def fake_refine(samples, predicted_start, predicted_end, sample_rate, **kwargs):
        captured.update(kwargs)
        from cutvideo.audio import BoundaryRefinement

        return BoundaryRefinement(
            predicted_start,
            predicted_end,
            False,
            False,
            {"start_method": "vad", "end_method": "energy"},
        )

    monkeypatch.setattr(workers, "discover_resources", lambda: resources)
    monkeypatch.setattr(workers, "resolve_inference_device", lambda: "cpu")
    monkeypatch.setattr(workers, "FunASRForceAligner", lambda *_a, **_k: _Force())
    monkeypatch.setattr(workers, "refine_cut_boundaries", fake_refine)
    monkeypatch.setattr(
        workers,
        "decode_f32",
        lambda *_a, **_k: __import__("numpy").zeros((800, 1), dtype="float32"),
    )

    result = workers.make_annotation_boundary_refinement_operation(
        audio_path=str(audio),
        info=info,
        tools=FFmpegTools(tmp_path / "ffmpeg", tmp_path / "ffprobe"),
        annotation_id="d-force",
        start_sample=1_000,
        end_sample=1_500,
        selected_text="你好",
        voice_ranges_ms=((900.0, 1_600.0),),
    )(CancelToken(), lambda _value, _message: None)

    assert result.annotation_id == "d-force"
    assert result.diagnostics["force_realign"]["engine"] == "fa-zh-annotation-realign"
    assert result.diagnostics["vad_start_sample"] == 900
    assert result.diagnostics["vad_end_sample"] == 1_600
    assert captured["vad_start_sample"] is not None
    assert captured["vad_end_sample"] is not None


def test_annotation_boundary_refinement_snaps_to_silence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "standalone.wav"
    audio.write_bytes(b"audio-fixture")
    sample_rate = 8_000
    total = sample_rate * 2
    info = AudioInfo(audio, sample_rate, 1, total, 2.0, "pcm")

    speech_start = sample_rate // 2
    speech_end = sample_rate
    captured: dict[str, int] = {}

    def fake_decode(_path, *, info, start_sample, end_sample, tools=None, cancel=None):
        captured["start"] = start_sample
        captured["end"] = end_sample
        times = np.arange(start_sample, end_sample)
        samples = np.zeros(times.size, dtype=np.float32)
        voiced = (times >= speech_start) & (times < speech_end)
        samples[voiced] = 0.5 * np.sin(
            2 * math.pi * 220 * times[voiced] / info.sample_rate
        ).astype(np.float32)
        return samples.reshape(-1, 1)

    monkeypatch.setattr(workers, "decode_f32", fake_decode)

    predicted_start = speech_start - 200
    predicted_end = speech_end + 200
    result = workers.make_annotation_boundary_refinement_operation(
        audio_path=str(audio),
        info=info,
        tools=FFmpegTools(tmp_path / "ffmpeg", tmp_path / "ffprobe"),
        annotation_id="d-test",
        start_sample=predicted_start,
        end_sample=predicted_end,
    )(CancelToken(), lambda _value, _message: None)

    assert result.annotation_id == "d-test"
    assert result.initial_start_sample == predicted_start
    assert result.initial_end_sample == predicted_end
    assert captured["start"] < predicted_start < predicted_end < captured["end"]
    assert 0 <= result.refined_start_sample < result.refined_end_sample <= total
    # The refined start moves toward the actual speech onset instead of the
    # 200-sample-early model prediction.
    assert result.refined_start_sample > predicted_start
    assert isinstance(result.diagnostics, dict)


def test_annotation_boundary_refinement_keeps_tiny_ranges_unchanged(tmp_path: Path) -> None:
    audio = tmp_path / "standalone.wav"
    audio.write_bytes(b"audio-fixture")
    info = AudioInfo(audio, 8_000, 1, 16_000, 2.0, "pcm")

    result = workers.make_annotation_boundary_refinement_operation(
        audio_path=str(audio),
        info=info,
        tools=FFmpegTools(tmp_path / "ffmpeg", tmp_path / "ffprobe"),
        annotation_id="d-tiny",
        start_sample=100,
        end_sample=102,
    )(CancelToken(), lambda _value, _message: None)

    assert (result.refined_start_sample, result.refined_end_sample) == (100, 102)
    assert result.evidence_complete is False


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
    annotation = project.add_delete_annotation(0, 2)
    project.mark_annotation_boundary_reviewed(annotation.id, source="test_fixture")
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
    tools = FFmpegTools(tmp_path / "ffmpeg", tmp_path / "ffprobe")
    original_result = workers.make_audio_processing_preview_operation(
        project,
        start_sample=4_000,
        end_sample=4_500,
        mode="original",
        tools=tools,
        preview_directory=tmp_path / "preview",
    )(CancelToken(), lambda _value, _message: None)
    selection_result = workers.make_audio_processing_preview_operation(
        project,
        start_sample=4_000,
        end_sample=4_500,
        mode="selection",
        tools=tools,
        preview_directory=tmp_path / "preview",
    )(CancelToken(), lambda _value, _message: None)

    assert len(calls) == 2
    assert (calls[0]["start_sample"], calls[0]["end_sample"]) == (1_000, 7_500)
    assert calls[0]["intervals"][-1] == (4_000, 4_500)
    assert (calls[1]["start_sample"], calls[1]["end_sample"]) == (4_000, 4_500)
    assert calls[1]["intervals"] == []
    assert original_result.original_wav_path.name.endswith("-original.wav")
    assert selection_result.selection_wav_path.name.endswith("-selection.wav")
