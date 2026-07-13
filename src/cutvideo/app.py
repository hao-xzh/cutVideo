"""Application entry point."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from . import __version__
from .model_runtime import configure_offline_environment
from .resources import find_resource_root
from .ui.main_window import MainWindow
from .ui.settings import preferred_theme
from .ui.theme import stylesheet_for_theme


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Return the process QApplication, creating it when necessary."""

    configure_offline_environment()
    existing = QApplication.instance()
    if existing is not None:
        return existing
    app = QApplication(list(argv) if argv is not None else sys.argv)
    QCoreApplication.setOrganizationName("CutVideo")
    QCoreApplication.setApplicationName("离线 Word 黄标音频剪辑器")
    QCoreApplication.setApplicationVersion(__version__)
    app.setStyle("Fusion")
    theme = preferred_theme()
    app.setProperty("theme", theme)
    app.setStyleSheet(stylesheet_for_theme(theme))
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
