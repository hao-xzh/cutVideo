#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-.venv/bin/python}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"
export SDKROOT="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
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

# macOS DMGs are deliberately code-only.  Models live in Application Support
# and are migrated from an older full bundle or downloaded once at runtime.
# The source-tree models remain available for the frozen external-model smoke
# test below, but no model weight may enter the app bundle.
RESOURCE_STAGE="$ROOT/build/resources-macos"
rm -rf "$RESOURCE_STAGE"
ditto "$ROOT/resources" "$RESOURCE_STAGE"
rm -rf "$RESOURCE_STAGE/models"
mkdir -p "$RESOURCE_STAGE/models"
"$PYTHON" scripts/verify_resources.py \
  --resource-root "$RESOURCE_STAGE" \
  --model-mode download-only

"$PYTHON" -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --onedir \
  --name=CutVideo \
  --icon="$ROOT/resources/icons/app-icon.icns" \
  --paths=src \
  --additional-hooks-dir="$ROOT/scripts/pyinstaller-hooks" \
  --hidden-import=mlx_qwen3_asr \
  --collect-all=pypinyin \
  --exclude-module=funasr \
  --exclude-module=modelscope \
  --exclude-module=transformers \
  --exclude-module=huggingface_hub \
  --exclude-module=torch \
  --exclude-module=torchaudio \
  --exclude-module=torch._dynamo \
  --exclude-module=torch._inductor \
  --exclude-module=torch.onnx \
  --exclude-module=torchvision \
  --exclude-module=sklearn \
  --exclude-module=umap \
  --exclude-module=pynndescent \
  --exclude-module=matplotlib \
  --exclude-module=cv2 \
  --add-data="$RESOURCE_STAGE:resources" \
  --add-data="$ROOT/THIRD_PARTY_NOTICES.md:." \
  --add-data="$ROOT/README.md:." \
  --distpath=dist \
  --workpath=build/pyinstaller-macos \
  --specpath=build \
  src/cutvideo_launcher.py

APP="dist/CutVideo.app"
APP_VERSION="$(PYTHONPATH="$ROOT/src" "$PYTHON" -c 'from cutvideo import __version__; print(__version__)')"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $APP_VERSION" "$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $APP_VERSION" "$APP/Contents/Info.plist" \
  || /usr/libexec/PlistBuddy -c "Add :CFBundleVersion string $APP_VERSION" "$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :LSMinimumSystemVersion $MACOSX_DEPLOYMENT_TARGET" "$APP/Contents/Info.plist" \
  || /usr/libexec/PlistBuddy -c "Add :LSMinimumSystemVersion string $MACOSX_DEPLOYMENT_TARGET" "$APP/Contents/Info.plist"
file "$APP/Contents/MacOS/CutVideo" | grep -q "arm64"
if find "$APP/Contents" -type f \( -name '*.safetensors' -o -name '*.onnx' \) | grep -q .; then
  echo "code-only macOS bundle unexpectedly contains model weights" >&2
  exit 1
fi
"$PYTHON" scripts/audit_macos_bundle.py "$APP" --maximum-minos "$MACOSX_DEPLOYMENT_TARGET"
if [[ -n "${APPLE_CODESIGN_IDENTITY:-}" ]]; then
  codesign --force --deep --options runtime --timestamp --sign "$APPLE_CODESIGN_IDENTITY" "$APP"
else
  codesign --force --deep --sign - "$APP"
fi
codesign --verify --deep --strict --verbose=2 "$APP"
REPORT="${TMPDIR:-/tmp}/cutvideo-selftest-$$.json"
CUTVIDEO_MODEL_ROOT="$ROOT/resources/models" \
  CUTVIDEO_SELFTEST_REPORT="$REPORT" \
  "$APP/Contents/MacOS/CutVideo" --self-test-models
cat "$REPORT"
rm -f "$REPORT"
STAGE="dist/dmg-root"
rm -rf "$STAGE"
mkdir -p "$STAGE"
PKG="dist/CutVideo.pkg"
PKG_ROOT="$ROOT/build/pkg-root"
rm -rf "$PKG_ROOT"
mkdir -p "$PKG_ROOT/Applications"
ditto "$APP" "$PKG_ROOT/Applications/CutVideo.app"
PKG_ARGS=(
  --root "$PKG_ROOT"
  --install-location /
  --component-plist "$ROOT/scripts/macos-component.plist"
  --scripts "$ROOT/scripts/macos-pkg-scripts"
  --identifier com.cutvideo.app
  --version "$APP_VERSION"
  --ownership recommended
)
if [[ -n "${APPLE_INSTALLER_IDENTITY:-}" ]]; then
  PKG_ARGS+=(--sign "$APPLE_INSTALLER_IDENTITY")
fi
pkgbuild "${PKG_ARGS[@]}" "$PKG"
pkgutil --check-signature "$PKG" || [[ -z "${APPLE_INSTALLER_IDENTITY:-}" ]]
PKG_AUDIT="$ROOT/build/pkg-audit"
rm -rf "$PKG_AUDIT"
pkgutil --expand-full "$PKG" "$PKG_AUDIT"
if find "$PKG_AUDIT" -type f \( -name '*.safetensors' -o -name '*.onnx' \) | grep -q .; then
  echo "code-only installer unexpectedly contains model weights" >&2
  exit 1
fi
if ! find "$PKG_AUDIT" -type f -name migrate-models -perm -111 | grep -q . \
  || ! grep -R -q 'migrate-models' "$PKG_AUDIT"; then
  echo "installer is missing the executable legacy-model migration hook" >&2
  exit 1
fi
ditto "$PKG" "$STAGE/安装 CutVideo.pkg"
if [[ "$(find "$STAGE" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')" != "1" ]]; then
  echo "DMG staging root must contain exactly one installer" >&2
  exit 1
fi
hdiutil create -volname CutVideo -srcfolder "$STAGE" -ov -format UDZO dist/CutVideo.dmg
hdiutil verify dist/CutVideo.dmg
if [[ "${KEEP_MACOS_APP_BUNDLE:-0}" != "1" ]]; then
  rm -rf "$STAGE" "$APP"
fi
rm -rf "$RESOURCE_STAGE"
rm -rf "$PKG_ROOT"
rm -rf "$PKG_AUDIT"
