from __future__ import annotations

import wave

import pytest
from PySide6.QtMultimedia import QAudio

from cutvideo.ui.audio_player import AudioPlaybackError, _is_natural_completion, read_pcm16_wav


def test_read_pcm16_wav(tmp_path) -> None:
    path = tmp_path / "preview.wav"
    frames = b"\x01\x00\xff\x7f" * 12
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(48_000)
        writer.writeframes(frames)

    result = read_pcm16_wav(path)

    assert result.sample_rate == 48_000
    assert result.channels == 2
    assert result.frames == frames


def test_read_pcm16_wav_rejects_non_pcm16(tmp_path) -> None:
    path = tmp_path / "preview.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(1)
        writer.setframerate(8_000)
        writer.writeframes(b"\x80" * 10)

    with pytest.raises(AudioPlaybackError, match="PCM 16-bit"):
        read_pcm16_wav(path)


def test_short_buffer_underrun_is_a_normal_completion() -> None:
    assert _is_natural_completion(QAudio.Error.UnderrunError, at_end=True)
    assert _is_natural_completion(QAudio.Error.NoError, at_end=True)
    assert not _is_natural_completion(QAudio.Error.UnderrunError, at_end=False)
    assert not _is_natural_completion(QAudio.Error.IOError, at_end=True)
