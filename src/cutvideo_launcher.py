"""Top-level launcher used by standalone packagers.

Keeping the executable entry point outside the ``cutvideo`` package preserves
the package context for all relative imports when the program is frozen.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence


def _ensure_windowed_standard_streams() -> None:
    """Give console-oriented dependencies harmless streams in a GUI build."""

    if not getattr(sys, "frozen", False):
        return
    if sys.stdin is None:
        sys.stdin = open(os.devnull, encoding="utf-8")  # noqa: SIM115
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115


def frozen_main(argv: Sequence[str] | None = None) -> int:
    """Dispatch diagnostics before Qt and keep Windows DLL load order stable.

    Some Windows PyTorch builds can crash when their native libraries are
    initialized after Qt has already loaded its own runtime libraries.  The
    frozen GUI therefore initializes Torch before importing ``cutvideo.app``.
    Model self-tests do not need Qt at all and bypass it completely.
    """

    arguments = list(argv) if argv is not None else list(sys.argv)
    _ensure_windowed_standard_streams()
    if "--self-test" in arguments or "--self-test-models" in arguments:
        from cutvideo.selftest import run_self_test_cli

        return run_self_test_cli(arguments)

    if sys.platform == "win32" and getattr(sys, "frozen", False):
        import torch  # noqa: F401  # Native runtime must load before PySide6/Qt.

    from cutvideo.app import main

    return main(arguments)


if __name__ == "__main__":
    raise SystemExit(frozen_main())
