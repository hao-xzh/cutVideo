"""Cross-platform subprocess options for a quiet desktop application."""

from __future__ import annotations

import subprocess
import sys
from typing import Any


def hidden_subprocess_kwargs() -> dict[str, Any]:
    """Return flags that prevent console tools from flashing a window.

    FFmpeg and FFprobe are console executables.  A windowed parent process on
    Windows otherwise gives each invocation a short-lived console window.
    Other platforms must not receive Windows-only ``creationflags``.
    """

    if sys.platform != "win32":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}


__all__ = ["hidden_subprocess_kwargs"]
