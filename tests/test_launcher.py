from __future__ import annotations

import builtins
import sys
from types import ModuleType

from cutvideo_launcher import frozen_main


def test_frozen_model_selftest_bypasses_torch_and_qt(monkeypatch) -> None:
    imported: list[str] = []
    original_import = builtins.__import__
    selftest = ModuleType("cutvideo.selftest")
    selftest.run_self_test_cli = lambda arguments: 17  # type: ignore[attr-defined]

    def tracking_import(name, *args, **kwargs):
        imported.append(name)
        if name == "cutvideo.selftest":
            return selftest
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    assert frozen_main(["CutVideo.exe", "--self-test-models"]) == 17
    assert "torch" not in imported
    assert "cutvideo.app" not in imported


def test_frozen_windows_gui_imports_torch_before_qt_app(monkeypatch) -> None:
    imported: list[str] = []
    original_import = builtins.__import__
    torch_module = ModuleType("torch")
    app_module = ModuleType("cutvideo.app")
    app_module.main = lambda arguments: 23  # type: ignore[attr-defined]

    def tracking_import(name, *args, **kwargs):
        imported.append(name)
        if name == "torch":
            return torch_module
        if name == "cutvideo.app":
            return app_module
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracking_import)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    assert frozen_main(["CutVideo.exe"]) == 23
    assert imported.index("torch") < imported.index("cutvideo.app")
