#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--accept-model-licenses" || $# -ne 1 ]]; then
  echo "Usage: bash scripts/bootstrap_macos.sh --accept-model-licenses" >&2
  echo "Review the two Qwen Apache-2.0 model cards before accepting." >&2
  exit 2
fi
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This script must run natively on an Apple Silicon Mac." >&2
  exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "Install Homebrew from https://brew.sh/ and run this command again." >&2
  exit 1
fi
if ! xcode-select -p >/dev/null 2>&1 || ! xcrun --find clang >/dev/null 2>&1; then
  echo "Install Xcode Command Line Tools with: xcode-select --install" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MACOSX_DEPLOYMENT_TARGET=15.0

brew list uv >/dev/null 2>&1 || brew install uv
uv python install 3.11
PYTHON311="$(uv python find 3.11)"
if [[ "$("$PYTHON311" -c 'import platform, sys; print(sys.version_info[:2] == (3, 11) and sys.version_info.releaselevel == "final" and platform.machine() == "arm64")')" != "True" ]]; then
  echo "A final Python 3.11 runtime is required." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python "$PYTHON311" .venv
fi
if [[ "$(.venv/bin/python -c 'import platform, sys; print(sys.version_info[:2] == (3, 11) and sys.version_info.releaselevel == "final" and platform.machine() == "arm64")')" != "True" ]]; then
  echo "Existing .venv is not native arm64 Python 3.11; remove it and run again." >&2
  exit 1
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -c constraints-release.txt -e '.[dev,ml]'
.venv/bin/python scripts/fetch_qwen_models.py --accept-model-licenses
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests scripts
PYTHON=.venv/bin/python bash scripts/build_macos.sh

echo "macOS installer created at: $ROOT/dist/CutVideo.dmg"
