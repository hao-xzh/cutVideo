"""Offline release-package diagnostics used after freezing the application."""

from __future__ import annotations

import json
import os
import subprocess
import traceback
from pathlib import Path
from typing import Any

from .alignment import FunASRAsrAligner, FunASRForceAligner
from .audio import probe_audio
from .ffmpeg import discover_ffmpeg
from .model_runtime import configure_offline_environment
from .resources import discover_resources, load_manifest
from .subprocess_options import hidden_subprocess_kwargs


def _verify_preview_backend(report: dict[str, Any]) -> None:
    # Import after model inference so Torch's native DLLs load before Qt on
    # Windows.  The editor plays FFmpeg-generated PCM through QAudioSink and
    # intentionally does not depend on QMediaPlayer codec plugins.
    from .ui.audio_player import PcmWavPlayer

    report["preview_backend"] = PcmWavPlayer.__name__


def _run_self_test(*, include_models: bool) -> dict[str, Any]:
    configure_offline_environment()
    resources = discover_resources()
    missing = resources.missing()
    if missing:
        raise RuntimeError(f"missing offline resources: {', '.join(missing)}")
    assert resources.ffmpeg is not None
    assert resources.ffprobe is not None
    tools = discover_ffmpeg(resource_root=resources.root)
    version = subprocess.run(
        [str(tools.ffmpeg), "-hide_banner", "-version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
        **hidden_subprocess_kwargs(),
    )
    if version.returncode:
        raise RuntimeError(f"bundled FFmpeg failed: {version.stderr.strip()}")
    report: dict[str, Any] = {
        "resource_root": str(resources.root),
        "platform_manifest": load_manifest(resources.root).get("schema_version"),
        "ffmpeg_version": version.stdout.splitlines()[0] if version.stdout else "",
        "models_tested": False,
    }
    if not include_models:
        _verify_preview_backend(report)
        return report

    assert resources.fa_model is not None
    assert resources.asr_model is not None
    assert resources.vad_model is not None
    transcript = "欢迎大家来到魔搭社区进行体验"
    force_audio = resources.fa_model / "example" / "asr_example.wav"
    force_info = probe_audio(force_audio, tools=tools)
    force_track = FunASRForceAligner(
        resources.fa_model,
        ffmpeg_path=tools.ffmpeg,
    ).align(
        audio_path=force_audio,
        transcript=transcript,
        window_start_ms=0,
        window_end_ms=max(1, round(force_info.duration_seconds * 1000)),
    )
    if not force_track.spans:
        raise RuntimeError("fa-zh inference returned no timestamp spans")

    asr_audio = resources.asr_model / "example" / "asr_example.wav"
    asr_info = probe_audio(asr_audio, tools=tools)
    asr_track = FunASRAsrAligner(
        resources.asr_model,
        resources.vad_model,
        ffmpeg_path=tools.ffmpeg,
    ).align(
        audio_path=asr_audio,
        transcript=transcript,
        window_start_ms=0,
        window_end_ms=max(1, round(asr_info.duration_seconds * 1000)),
    )
    if not asr_track.spans:
        raise RuntimeError("paraformer-zh/fsmn-vad inference returned no timestamp spans")
    report.update(
        {
            "models_tested": True,
            "fa_span_count": len(force_track.spans),
            "fa_coverage": force_track.coverage,
            "asr_span_count": len(asr_track.spans),
            "asr_coverage": asr_track.coverage,
        }
    )
    _verify_preview_backend(report)
    return report


def run_self_test_cli(arguments: list[str]) -> int:
    include_models = "--self-test-models" in arguments
    report_path = Path(
        os.environ.get("CUTVIDEO_SELFTEST_REPORT", "cutvideo-selftest.json")
    ).expanduser()
    payload: dict[str, Any]
    try:
        result = _run_self_test(include_models=include_models)
    except Exception as exc:
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    else:
        payload = {"ok": True, **result}
        exit_code = 0
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return exit_code


__all__ = ["run_self_test_cli"]
