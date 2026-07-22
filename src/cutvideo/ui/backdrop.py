"""Theme-aware ambient artwork behind the desktop workspaces."""

from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from ..resources import find_resource_root
from .theme import THEME_DARK, current_theme, theme_color


class ThemedBackdrop(QWidget):
    """Paint the packaged light or dark artwork without distorting its aspect ratio."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("applicationShell")
        self.setAutoFillBackground(False)
        self._pixmaps: dict[str, QPixmap] = {}

    def refresh_theme(self) -> None:
        """Schedule a repaint after the application theme property changes."""

        self.update()

    def _pixmap(self, theme: str) -> QPixmap:
        cached = self._pixmaps.get(theme)
        if cached is not None:
            return cached
        filename = "workspace-dark.png" if theme == THEME_DARK else "workspace-light.png"
        path = find_resource_root() / "backgrounds" / filename
        pixmap = QPixmap(str(path)) if path.is_file() else QPixmap()
        self._pixmaps[theme] = pixmap
        return pixmap

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor(theme_color("bg")))

        pixmap = self._pixmap(current_theme())
        if pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            return

        source_width = float(pixmap.width())
        source_height = float(pixmap.height())
        target_ratio = self.width() / self.height()
        source_ratio = source_width / source_height
        if source_ratio > target_ratio:
            crop_width = source_height * target_ratio
            source = QRectF((source_width - crop_width) / 2.0, 0.0, crop_width, source_height)
        else:
            crop_height = source_width / target_ratio
            source = QRectF(0.0, (source_height - crop_height) / 2.0, source_width, crop_height)
        painter.drawPixmap(QRectF(self.rect()), pixmap, source)


__all__ = ["ThemedBackdrop"]
