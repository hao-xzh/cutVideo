from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

from cutvideo.model_store import (
    QWEN_MODEL_NAMES,
    prepare_qwen_models,
    qwen_models_ready,
)

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = ROOT / "resources"
MODEL_ROOT = RESOURCE_ROOT / "models"


def _publish_qwen_pair(source_root: Path) -> None:
    """Replace only the Qwen pair while preserving cached Windows models."""

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    backup_root = RESOURCE_ROOT / f".qwen-model-backup-{os.getpid()}-{uuid.uuid4().hex}"
    backup_root.mkdir(parents=True)
    moved_new: list[str] = []
    moved_old: list[str] = []
    try:
        for name in QWEN_MODEL_NAMES:
            source = source_root / name
            destination = MODEL_ROOT / name
            backup = backup_root / name
            if destination.exists():
                os.replace(destination, backup)
                moved_old.append(name)
            os.replace(source, destination)
            moved_new.append(name)
    except Exception:
        for name in reversed(moved_new):
            destination = MODEL_ROOT / name
            if destination.exists():
                os.replace(destination, source_root / name)
        for name in reversed(moved_old):
            backup = backup_root / name
            if backup.exists():
                os.replace(backup, MODEL_ROOT / name)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the pinned Qwen/MLX model pair used for macOS release self-test",
    )
    parser.add_argument("--accept-model-licenses", action="store_true", required=True)
    args = parser.parse_args()
    del args

    def report(value: float, message: str) -> None:
        print(f"[{round(value * 100):3d}%] {message}", flush=True)

    if qwen_models_ready(MODEL_ROOT, RESOURCE_ROOT, full=True):
        source = "existing"
        downloaded = False
        migrated = False
    else:
        fetch_root = RESOURCE_ROOT / f".qwen-model-fetch-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            result = prepare_qwen_models(
                resource_root=RESOURCE_ROOT,
                target_root=fetch_root,
                progress=report,
            )
            _publish_qwen_pair(fetch_root)
            if not qwen_models_ready(MODEL_ROOT, RESOURCE_ROOT, full=True):
                raise RuntimeError("Qwen models failed verification after source-tree publish")
            source = result.source
            downloaded = result.downloaded
            migrated = result.migrated
        finally:
            shutil.rmtree(fetch_root, ignore_errors=True)
    print(
        json.dumps(
            {
                "model_root": str(MODEL_ROOT),
                "source": source,
                "downloaded": downloaded,
                "migrated": migrated,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
