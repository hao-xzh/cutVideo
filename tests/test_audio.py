from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
import pytest

from cutvideo.audio import (
    CutInterval,
    build_waveform_envelopes,
    compute_waveform_envelope,
    merge_intervals,
    probe_audio,
    refine_boundary,
    write_cut_list_csv,
)
from cutvideo.ffmpeg import discover_ffmpeg

SAMPLE_AUDIO = Path(os.environ.get("CUTVIDEO_SAMPLE_AUDIO", "__missing_sample_audio__.mp3"))


def test_merge_intervals_sorts_clamps_and_joins_touching_ranges() -> None:
    assert merge_intervals([(20, 30), (-5, 3), (3, 8), (25, 40), (99, 120)], maximum=100) == [
        (0, 8),
        (20, 40),
        (99, 100),
    ]


def test_waveform_envelope_handles_partial_final_block_and_stereo() -> None:
    samples = np.array(
        [[-1.0, -1.0], [0.5, 0.5], [1.0, 1.0], [-0.25, -0.25], [0.75, 0.75]],
        dtype=np.float32,
    )
    envelope = compute_waveform_envelope(samples, 48_000, 2)
    np.testing.assert_allclose(envelope.minimum, [-1.0, -0.25, 0.75])
    np.testing.assert_allclose(envelope.maximum, [0.5, 1.0, 0.75])
    assert envelope.rms[-1] == 0.75
    assert envelope.block_size == 2


def test_waveform_envelopes_form_a_multiresolution_pyramid() -> None:
    samples = np.linspace(-1, 1, 128, dtype=np.float32)
    levels = build_waveform_envelopes(
        samples,
        16_000,
        base_block_size=4,
        max_levels=4,
        minimum_points=1,
    )
    assert [level.block_size for level in levels] == [4, 8, 16, 32]
    assert [level.points for level in levels] == [32, 16, 8, 4]
    assert levels[1].minimum[0] == levels[0].minimum[:2].min()


def test_refine_boundary_prefers_nearby_silent_zero_crossing() -> None:
    sample_rate = 1_000
    x = np.arange(300)
    samples = np.sin(2 * math.pi * x / 20).astype(np.float32)
    samples[135:151] = 0
    refined = refine_boundary(samples, 120, sample_rate, search_ms=40, energy_window_ms=4)
    assert 135 <= refined <= 150


def test_refine_boundary_respects_protected_bounds() -> None:
    samples = np.zeros(100, dtype=np.float32)
    assert refine_boundary(samples, 10, 1_000, search_ms=60, lower_bound=30, upper_bound=50) == 30


def test_cut_csv_is_utf8_and_sample_exact(tmp_path: Path) -> None:
    output = write_cut_list_csv(
        tmp_path / "切点.csv",
        [CutInterval(480, 960, "删除", "approved")],
        48_000,
    )
    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["start_seconds"] == "0.010000"
    assert rows[0]["text"] == "删除"


@pytest.mark.skipif(not SAMPLE_AUDIO.is_file(), reason="real sample audio is unavailable")
def test_real_sample_uses_gapless_decoded_pcm_timeline() -> None:
    tools = discover_ffmpeg(resource_root=Path(__file__).resolve().parents[1] / "resources")
    info = probe_audio(SAMPLE_AUDIO, tools=tools)

    assert (info.sample_rate, info.channels) == (48_000, 2)
    assert info.total_samples == 120_374_411
    assert info.duration_seconds == pytest.approx(2507.8002291666667)
