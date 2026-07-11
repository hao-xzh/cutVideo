from __future__ import annotations

import subprocess
import sys

from cutvideo.subprocess_options import hidden_subprocess_kwargs


def test_windows_subprocesses_never_create_a_console(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    options = hidden_subprocess_kwargs()

    expected = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    assert options["creationflags"] & expected


def test_non_windows_subprocesses_receive_no_windows_flags(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")

    assert hidden_subprocess_kwargs() == {}
