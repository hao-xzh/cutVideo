#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This script must run natively on an Apple Silicon Mac." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="8.1.2"
ARCHIVE="ffmpeg-${VERSION}.tar.xz"
URL="https://ffmpeg.org/releases/${ARCHIVE}"
EXPECTED_SHA256="464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c"
BUILD_ROOT="$ROOT/.build/ffmpeg-macos-arm64"
SOURCE_ROOT="$BUILD_ROOT/ffmpeg-${VERSION}"
OUTPUT_ROOT="$ROOT/resources/bin/macos-arm64"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"
export SDKROOT="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
PYTHON_BIN="$(cd "$ROOT" && "${PYTHON:-python3}" -c 'import sys; print(sys.executable)')"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required to build the LGPL libmp3lame dependency." >&2
  exit 1
fi
brew list pkg-config >/dev/null 2>&1 || brew install pkg-config

mkdir -p "$BUILD_ROOT"
if [[ ! -f "$BUILD_ROOT/$ARCHIVE" ]]; then
  curl --fail --location --proto '=https' --tlsv1.2 "$URL" -o "$BUILD_ROOT/$ARCHIVE"
fi
ACTUAL_SHA256="$(shasum -a 256 "$BUILD_ROOT/$ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "FFmpeg source SHA-256 mismatch." >&2
  exit 1
fi

rm -rf "$SOURCE_ROOT" "$BUILD_ROOT/install"
tar -xf "$BUILD_ROOT/$ARCHIVE" -C "$BUILD_ROOT"
LAME_PREFIX="${CUTVIDEO_LAME_PREFIX:-$ROOT/.build/lame-macos-arm64/install}"
if [[ ! -f "$LAME_PREFIX/lib/libmp3lame.0.dylib" ]]; then
  bash "$ROOT/scripts/build_macos_lame.sh"
fi
export PKG_CONFIG_PATH="$LAME_PREFIX/lib/pkgconfig"

cd "$SOURCE_ROOT"
./configure \
  --prefix="$BUILD_ROOT/install" \
  --arch=arm64 \
  --target-os=darwin \
  --cc="$(xcrun --find clang)" \
  --disable-autodetect \
  --disable-doc \
  --disable-debug \
  --disable-network \
  --disable-ffplay \
  --disable-shared \
  --enable-static \
  --enable-libmp3lame \
  --extra-cflags="-I$LAME_PREFIX/include -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET" \
  --extra-ldflags="-L$LAME_PREFIX/lib -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET"
make -j"$(sysctl -n hw.logicalcpu)"
make install

mkdir -p "$OUTPUT_ROOT"
install -m 0755 "$BUILD_ROOT/install/bin/ffmpeg" "$OUTPUT_ROOT/ffmpeg"
install -m 0755 "$BUILD_ROOT/install/bin/ffprobe" "$OUTPUT_ROOT/ffprobe"

# Keep libmp3lame dynamically replaceable to satisfy the LGPL relinking path.
# Relocate the Homebrew dylib beside each executable that references it.
dylib_name=""
for binary in ffmpeg ffprobe; do
  dependency="$(otool -L "$OUTPUT_ROOT/$binary" | awk '/libmp3lame.*dylib/{print $1; exit}')"
  if [[ -n "$dependency" ]]; then
    dylib_name="$(basename "$dependency")"
    dependency_source="$dependency"
    if [[ ! -f "$dependency_source" ]]; then
      dependency_source="$LAME_PREFIX/lib/$dylib_name"
    fi
    if [[ ! -f "$dependency_source" ]]; then
      echo "Unable to locate the dynamically linked libmp3lame dependency." >&2
      exit 1
    fi
    cp "$dependency_source" "$OUTPUT_ROOT/$dylib_name"
    chmod 0755 "$OUTPUT_ROOT/$dylib_name"
    install_name_tool -change "$dependency" "@executable_path/$dylib_name" "$OUTPUT_ROOT/$binary"
    otool -L "$OUTPUT_ROOT/$binary" | grep -Fq "@executable_path/$dylib_name"
  fi
done
if [[ -z "$dylib_name" ]]; then
  echo "FFmpeg did not dynamically link libmp3lame; refusing an LGPL-incompatible release." >&2
  exit 1
fi
for license_path in "$LAME_PREFIX/share/doc/lame/COPYING" "$LAME_PREFIX/COPYING"; do
  if [[ -f "$license_path" ]]; then
    cp "$license_path" "$ROOT/resources/licenses/LGPL-LAME.txt"
    break
  fi
done
if [[ ! -f "$ROOT/resources/licenses/LGPL-LAME.txt" ]]; then
  echo "Homebrew LAME license file was not found." >&2
  exit 1
fi

"$OUTPUT_ROOT/ffmpeg" -hide_banner -version > "$ROOT/resources/FFMPEG_BUILDINFO_MACOS.txt"
"$OUTPUT_ROOT/ffmpeg" -hide_banner -buildconf >> "$ROOT/resources/FFMPEG_BUILDINFO_MACOS.txt"
printf '\nDynamically linked LAME dependency:\n' >> "$ROOT/resources/FFMPEG_BUILDINFO_MACOS.txt"
otool -L "$OUTPUT_ROOT/ffmpeg" | grep 'libmp3lame' >> "$ROOT/resources/FFMPEG_BUILDINFO_MACOS.txt"

FFMPEG_HASH="$(shasum -a 256 "$OUTPUT_ROOT/ffmpeg" | awk '{print $1}')"
FFPROBE_HASH="$(shasum -a 256 "$OUTPUT_ROOT/ffprobe" | awk '{print $1}')"
LAME_HASH="$(shasum -a 256 "$OUTPUT_ROOT/$dylib_name" | awk '{print $1}')"
LAME_VERSION="3.100"
ROOT="$ROOT" FFMPEG_HASH="$FFMPEG_HASH" FFPROBE_HASH="$FFPROBE_HASH" \
  LAME_FILE="$dylib_name" LAME_HASH="$LAME_HASH" LAME_VERSION="$LAME_VERSION" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
path = root / "resources" / "manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))
entry = manifest["binaries"]["macos-arm64"]
entry.update(
    {
        "ffmpeg": os.environ["FFMPEG_HASH"],
        "ffprobe": os.environ["FFPROBE_HASH"],
        "source": "https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz",
        "source_code": "https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz",
        "archive_sha256": "464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c",
        "license": "licenses/LGPL-2.1.txt",
        "version": "8.1.2",
        "variant": "macos-arm64-lgpl-static+dynamic-libmp3lame",
        "lgpl_dependencies": {
            "libmp3lame": {
                "file": os.environ["LAME_FILE"],
                "sha256": os.environ["LAME_HASH"],
                "version": os.environ["LAME_VERSION"],
                "source_code": "https://sourceforge.net/projects/lame/files/lame/",
                "license": "licenses/LGPL-LAME.txt",
            }
        },
    }
)
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

echo "macOS arm64 FFmpeg resources created and pinned in resources/manifest.json"
