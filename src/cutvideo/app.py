"""Application entry point."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .model_runtime import configure_offline_environment
from .resources import find_resource_root
from .ui.main_window import MainWindow
from .ui.theme import APP_STYLESHEET


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Return the process QApplication, creating it when necessary."""

    configure_offline_environment()
    existing = QApplication.instance()
    if existing is not None:
        return existing
    app = QApplication(list(argv) if argv is not None else sys.argv)
    QCoreApplication.setOrganizationName("CutVideo")
    QCoreApplication.setApplicationName("离线 Word 黄标音频剪辑器")
    QCoreApplication.setApplicationVersion("0.1.0")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    icon_path = find_resource_root() / "icons" / "app-icon.png"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))
    return app


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else list(sys.argv)
    if "--self-test" in arguments or "--self-test-models" in arguments:
        from .selftest import run_self_test_cli

        return run_self_test_cli(arguments)
    app = create_application(arguments)
    window = MainWindow()
    window.show()
    return app.exec()


__all__ = ["create_application", "main"]
