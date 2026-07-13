#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This script must run natively on an Apple Silicon Mac." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="3.100"
ARCHIVE="lame-${VERSION}.tar.gz"
URL="https://downloads.sourceforge.net/project/lame/lame/${VERSION}/${ARCHIVE}"
EXPECTED_SHA256="ddfe36cab873794038ae2c1210557ad34857a4b6bdc515785d1da9e175b1da1e"
BUILD_ROOT="$ROOT/.build/lame-macos-arm64"
SOURCE_ROOT="$BUILD_ROOT/lame-${VERSION}"
INSTALL_ROOT="$BUILD_ROOT/install"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"
export SDKROOT="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"

mkdir -p "$BUILD_ROOT"
if [[ ! -f "$BUILD_ROOT/$ARCHIVE" ]]; then
  curl --fail --location --proto '=https' --tlsv1.2 "$URL" -o "$BUILD_ROOT/$ARCHIVE"
fi
ACTUAL_SHA256="$(shasum -a 256 "$BUILD_ROOT/$ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "LAME source SHA-256 mismatch." >&2
  exit 1
fi

rm -rf "$SOURCE_ROOT" "$INSTALL_ROOT"
tar -xzf "$BUILD_ROOT/$ARCHIVE" -C "$BUILD_ROOT"
cd "$SOURCE_ROOT"
export CFLAGS="-O2 -arch arm64 -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET"
export LDFLAGS="-arch arm64 -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET"
./configure \
  --prefix="$INSTALL_ROOT" \
  --disable-frontend \
  --disable-decoder \
  --disable-static \
  --enable-shared
make -j"$(sysctl -n hw.logicalcpu)"
make install
cp COPYING "$ROOT/resources/licenses/LGPL-LAME.txt"
echo "LAME ${VERSION} created at: $INSTALL_ROOT"
