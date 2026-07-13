from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "resources" / "models"
MODELS = {
    "fa-zh": (
        "iic/speech_timestamp_prediction-v1-16k-offline",
        "ed6abeae67748098b6055978a17eefd2fa0f9fd6",
    ),
    "paraformer-zh": (
        "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "6059253cbfda9ea43d6c6f198d38854f09adb298",
    ),
    "fsmn-vad": (
        "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "f9a8b8274674755d925277e27063869038d41515",
    ),
}


def tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch release-time FunASR model assets")
    parser.add_argument("models", nargs="*", metavar="MODEL")
    parser.add_argument("--accept-model-licenses", action="store_true", required=True)
    args = parser.parse_args()
    unknown = sorted(set(args.models) - set(MODELS))
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}; choose from {', '.join(sorted(MODELS))}")

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install the project ml extra before fetching models") from exc

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict[str, str]] = {}
    selected = args.models or list(MODELS)
    for name in selected:
        model_id, revision = MODELS[name]
        destination = MODEL_ROOT / name
        print(f"Downloading {model_id}@{revision} -> {destination}")
        resolved = Path(
            snapshot_download(
                model_id,
                local_dir=str(destination),
                revision=revision,
            )
        )
        report[name] = {
            "model_id": model_id,
            "revision": revision,
            "resolved_path": str(resolved.resolve()),
            "tree_sha256": tree_hash(destination),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
