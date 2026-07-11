from __future__ import annotations

from pathlib import Path

from cutvideo.pipeline import expected_output_paths


def test_expected_output_paths_are_stable(tmp_path: Path) -> None:
    stem = "\u8bbf\u8c08"
    outputs = expected_output_paths(f"{stem}.mp3", tmp_path)
    assert outputs["wav"].name == f"{stem}_\u526a\u8f91\u5b8c\u6210.wav"
    assert outputs["mp3"].name == f"{stem}_\u526a\u8f91\u5b8c\u6210.mp3"
    assert outputs["project"].name == f"{stem}.cutvideo.json"
    assert outputs["cuts"].name == f"{stem}_\u5207\u70b9.csv"
