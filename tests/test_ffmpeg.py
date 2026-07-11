from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from cutvideo.audio import probe_audio
from cutvideo.ffmpeg import (
    FFmpegProcessError,
    FFmpegTools,
    build_export_command,
    build_export_filter,
    build_preview_command,
    discover_ffmpeg,
    export_audio,
)

try:
    REAL_TOOLS: FFmpegTools | None = discover_ffmpeg(
        resource_root=Path(__file__).resolve().parents[1] / "resources"
    )
except Exception:
    REAL_TOOLS = None


def _touch_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_discovery_honors_environment_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suffix = ".exe" if os.name == "nt" else ""
    ffmpeg = tmp_path / f"ffmpeg{suffix}"
    ffprobe = tmp_path / f"ffprobe{suffix}"
    _touch_executable(ffmpeg)
    _touch_executable(ffprobe)
    monkeypatch.setenv("CUTVIDEO_FFMPEG", str(ffmpeg))
    monkeypatch.setenv("CUTVIDEO_FFPROBE", str(ffprobe))
    assert discover_ffmpeg() == FFmpegTools(ffmpeg.resolve(), ffprobe.resolve())


def test_export_filter_merges_cuts_and_uses_eight_ms_acrossfade() -> None:
    filter_text, kept, removed = build_export_filter(
        [(1_000, 2_000), (1_500, 2_500), (7_000, 8_000)],
        total_samples=10_000,
        sample_rate=48_000,
    )
    assert "atrim=start_sample=0:end_sample=1000" in filter_text
    assert "atrim=start_sample=2500:end_sample=7000" in filter_text
    assert "atrim=start_sample=8000:end_sample=10000" in filter_text
    assert filter_text.count("acrossfade=ns=384:c1=qsin:c2=qsin") == 2
    assert filter_text.endswith("[edited]asplit=2[wavout][mp3out]")
    assert removed == 2_500
    assert kept == 10_000 - 2_500 - 2 * 384


def test_export_command_is_one_argv_command_with_two_mapped_outputs(tmp_path: Path) -> None:
    tools = FFmpegTools(Path("/safe/ffmpeg"), Path("/safe/ffprobe"))
    command, _kept, _removed = build_export_command(
        tmp_path / "input;not-a-shell.mp3",
        [(100, 200)],
        tmp_path / "out.wav",
        tmp_path / "out.mp3",
        total_samples=1_000,
        sample_rate=48_000,
        channels=2,
        tools=tools,
    )
    assert command[0] == str(tools.ffmpeg)
    assert command.count("-filter_complex") == 1
    assert command.count("-map") == 2
    assert "pcm_s16le" in command
    assert "libmp3lame" in command
    assert "192k" in command
    assert str(tmp_path / "input;not-a-shell.mp3") in command


def test_export_command_preserves_wav_shape_but_makes_mp3_compatible(tmp_path: Path) -> None:
    command, _kept, _removed = build_export_command(
        tmp_path / "input.wav",
        [(100, 200)],
        tmp_path / "out.wav",
        tmp_path / "out.mp3",
        total_samples=10_000,
        sample_rate=96_000,
        channels=4,
        tools=FFmpegTools(Path("/safe/ffmpeg"), Path("/safe/ffprobe")),
    )
    wav_index = command.index(str(tmp_path / "out.wav"))
    mp3_index = command.index(str(tmp_path / "out.mp3"))
    assert command[wav_index - 4 : wav_index] == ["-ar", "96000", "-ac", "4"]
    assert command[mp3_index - 4 : mp3_index] == ["-ar", "48000", "-ac", "2"]


def test_preview_command_uses_device_friendly_pcm_shape(tmp_path: Path) -> None:
    original = tmp_path / "original.wav"
    edited = tmp_path / "edited.wav"
    command = build_preview_command(
        tmp_path / "input.wav",
        [],
        original,
        edited,
        start_sample=0,
        end_sample=9_600,
        total_samples=9_600,
        sample_rate=96_000,
        channels=4,
        tools=FFmpegTools(Path("/safe/ffmpeg"), Path("/safe/ffprobe")),
    )

    original_index = command.index(str(original))
    edited_index = command.index(str(edited))
    assert command[original_index - 4 : original_index] == ["-ar", "48000", "-ac", "2"]
    assert command[edited_index - 4 : edited_index] == ["-ar", "48000", "-ac", "2"]


@pytest.mark.skipif(REAL_TOOLS is None, reason="FFmpeg not installed")
def test_real_ffmpeg_duration_comes_from_decoded_sample_count(tmp_path: Path) -> None:
    assert REAL_TOOLS is not None
    tools = REAL_TOOLS
    source = tmp_path / "source.wav"
    sample_rate = 8_000
    frames = 1_003
    tone = (np.sin(2 * np.pi * 220 * np.arange(frames) / sample_rate) * 10_000).astype("<i2")
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(tone.tobytes())
    info = probe_audio(source, tools=tools)
    assert info.total_samples == frames
    assert info.duration_seconds == pytest.approx(frames / sample_rate)


@pytest.mark.skipif(REAL_TOOLS is None, reason="FFmpeg not installed")
def test_real_ffmpeg_exports_wav_and_mp3_in_one_call(tmp_path: Path) -> None:
    assert REAL_TOOLS is not None
    tools = REAL_TOOLS
    source = tmp_path / "source.wav"
    rate = 8_000
    frames = 8_000
    tone = (np.sin(2 * np.pi * 220 * np.arange(frames) / rate) * 10_000).astype("<i2")
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(tone.tobytes())
    info = probe_audio(source, tools=tools)
    result = export_audio(
        source,
        [(3_000, 4_000)],
        tmp_path / "完成.wav",
        tmp_path / "完成.mp3",
        info=info,
        tools=tools,
    )
    assert result.wav_path.stat().st_size > 44
    assert result.mp3_path.stat().st_size > 0


@pytest.mark.skipif(REAL_TOOLS is None, reason="FFmpeg not installed")
def test_real_ffmpeg_downsamples_high_rate_multichannel_only_for_mp3(tmp_path: Path) -> None:
    assert REAL_TOOLS is not None
    source = tmp_path / "source-96k-4ch.wav"
    rate = 96_000
    frames = 9_600
    mono = (np.sin(2 * np.pi * 440 * np.arange(frames) / rate) * 8_000).astype("<i2")
    interleaved = np.repeat(mono[:, None], 4, axis=1).reshape(-1)
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(4)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(interleaved.tobytes())
    info = probe_audio(source, tools=REAL_TOOLS)
    result = export_audio(
        source,
        [(2_000, 2_500)],
        tmp_path / "out.wav",
        tmp_path / "out.mp3",
        info=info,
        tools=REAL_TOOLS,
    )
    wav_info = probe_audio(result.wav_path, tools=REAL_TOOLS)
    mp3_info = probe_audio(result.mp3_path, tools=REAL_TOOLS)
    assert (wav_info.sample_rate, wav_info.channels) == (96_000, 4)
    assert (mp3_info.sample_rate, mp3_info.channels) == (48_000, 2)


def test_failed_export_preserves_existing_outputs_and_cleans_partials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cutvideo.ffmpeg as module

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    wav_output = tmp_path / "out.wav"
    mp3_output = tmp_path / "out.mp3"
    wav_output.write_bytes(b"old-wav")
    mp3_output.write_bytes(b"old-mp3")

    def fail(argv: list[str], **_kwargs: object) -> None:
        for value in argv:
            path = Path(value)
            if ".tmp." in path.name and path.suffix in {".wav", ".mp3"}:
                path.write_bytes(b"partial")
        raise FFmpegProcessError(argv, 1, "simulated failure")

    monkeypatch.setattr(module, "_run_with_progress", fail)
    with pytest.raises(FFmpegProcessError):
        export_audio(
            source,
            [(100, 200)],
            wav_output,
            mp3_output,
            total_samples=1_000,
            sample_rate=48_000,
            channels=2,
            tools=FFmpegTools(Path("ffmpeg"), Path("ffprobe")),
        )
    assert wav_output.read_bytes() == b"old-wav"
    assert mp3_output.read_bytes() == b"old-mp3"
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_export_rejects_overwriting_the_input(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    with pytest.raises(ValueError, match="input"):
        export_audio(
            source,
            [(100, 200)],
            source,
            tmp_path / "out.mp3",
            total_samples=1_000,
            sample_rate=48_000,
            channels=2,
            tools=FFmpegTools(Path("ffmpeg"), Path("ffprobe")),
        )
