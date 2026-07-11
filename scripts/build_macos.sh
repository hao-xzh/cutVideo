#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-.venv/bin/python}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
if [[ "$("$PYTHON" -c 'import platform, sys; print(sys.version_info[:2] == (3, 11) and sys.version_info.releaselevel == "final" and platform.machine() == "arm64")')" != "True" ]]; then
  echo "macOS release builds require a final native arm64 Python 3.11 runtime." >&2
  exit 1
fi
if [[ ! -x resources/bin/macos-arm64/ffmpeg || ! -x resources/bin/macos-arm64/ffprobe ]]; then
  bash scripts/build_macos_ffmpeg.sh
fi
VERIFY_ARGS=(scripts/verify_resources.py)
if [[ "${ALLOW_UNPINNED:-0}" == "1" ]]; then
  VERIFY_ARGS+=(--allow-unpinned)
fi
"$PYTHON" "${VERIFY_ARGS[@]}"

"$PYTHON" -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --onedir \
  --name=CutVideo \
  --icon="$ROOT/resources/icons/app-icon.icns" \
  --paths=src \
  --additional-hooks-dir="$ROOT/scripts/pyinstaller-hooks" \
  --collect-all=pypinyin \
  --exclude-module=modelscope \
  --exclude-module=transformers \
  --exclude-module=huggingface_hub \
  --exclude-module=torch._dynamo \
  --exclude-module=torch._inductor \
  --exclude-module=torch.onnx \
  --exclude-module=torchvision \
  --exclude-module=sklearn \
  --exclude-module=umap \
  --exclude-module=pynndescent \
  --exclude-module=matplotlib \
  --exclude-module=cv2 \
  --add-data="$ROOT/resources:resources" \
  --add-data="$ROOT/THIRD_PARTY_NOTICES.md:." \
  --add-data="$ROOT/README.md:." \
  --distpath=dist \
  --workpath=build/pyinstaller-macos \
  --specpath=build \
  src/cutvideo_launcher.py

APP="dist/CutVideo.app"
file "$APP/Contents/MacOS/CutVideo" | grep -q "arm64"
if [[ -n "${APPLE_CODESIGN_IDENTITY:-}" ]]; then
  codesign --force --deep --options runtime --timestamp --sign "$APPLE_CODESIGN_IDENTITY" "$APP"
else
  codesign --force --deep --sign - "$APP"
fi
codesign --verify --deep --strict --verbose=2 "$APP"
REPORT="${TMPDIR:-/tmp}/cutvideo-selftest-$$.json"
CUTVIDEO_SELFTEST_REPORT="$REPORT" "$APP/Contents/MacOS/CutVideo" --self-test-models
cat "$REPORT"
rm -f "$REPORT"
STAGE="dist/dmg-root"
rm -rf "$STAGE"
mkdir -p "$STAGE"
ditto "$APP" "$STAGE/CutVideo.app"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname CutVideo -srcfolder "$STAGE" -ov -format UDZO dist/CutVideo.dmg
