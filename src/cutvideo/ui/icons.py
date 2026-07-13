"""SVG icon helpers shared by the desktop workspaces."""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QObject, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QAbstractButton, QToolButton, QWidget

from ..resources import find_resource_root
from .theme import theme_color


@lru_cache(maxsize=64)
def ui_icon(name: str) -> QIcon:
    """Load a packaged monochrome SVG icon by its extension-free name."""

    path = find_resource_root() / "icons" / "ui" / f"{name}.svg"
    return QIcon(str(path)) if path.is_file() else QIcon()


def set_button_icon(
    button: QAbstractButton,
    name: str,
    *,
    size: int = 17,
    tooltip: str | None = None,
) -> None:
    """Apply an SVG icon and optional accessible tooltip to a button."""

    button.setIcon(ui_icon(name))
    button.setIconSize(QSize(size, size))
    if tooltip:
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)


def make_icon_button(
    name: str,
    tooltip: str,
    *,
    parent: QWidget | None = None,
    size: int = 32,
    icon_size: int = 17,
) -> QToolButton:
    """Return a compact icon-only button with a readable accessible name."""

    button = QToolButton(parent)
    button.setObjectName("iconButton")
    button.setFixedSize(size, size)
    button.setAutoRaise(True)
    set_button_icon(button, name, size=icon_size, tooltip=tooltip)
    return button


class ButtonSpinner(QObject):
    """Animate a restrained busy glyph inside one existing button.

    The button keeps its text, geometry and place in the layout.  Stopping the
    indicator restores the exact icon and accessibility description that were
    present before the task started.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._button: QAbstractButton | None = None
        self._idle_icon = QIcon()
        self._idle_icon_size = QSize(17, 17)
        self._idle_accessible_description = ""
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._advance)

    @property
    def is_active(self) -> bool:
        return self._button is not None

    def start(self, button: QAbstractButton) -> None:
        if self._button is button:
            return
        self.stop()
        self._button = button
        self._idle_icon = button.icon()
        self._idle_icon_size = button.iconSize()
        self._idle_accessible_description = button.accessibleDescription()
        self._angle = 0
        button.setProperty("busy", True)
        button.setAccessibleDescription("正在准备试听")
        self._render_frame()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        button = self._button
        if button is not None:
            button.setIcon(self._idle_icon)
            button.setIconSize(self._idle_icon_size)
            button.setProperty("busy", False)
            button.setAccessibleDescription(self._idle_accessible_description)
        self._button = None

    def _advance(self) -> None:
        self._angle = (self._angle + 24) % 360
        self._render_frame()

    def _render_frame(self) -> None:
        button = self._button
        if button is None:
            return
        logical_size = max(
            15,
            self._idle_icon_size.width(),
            self._idle_icon_size.height(),
        )
        pixmap = QPixmap(logical_size, logical_size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(2.2, 2.2, logical_size - 4.4, logical_size - 4.4)

        track = QColor(theme_color("accent"))
        track.setAlpha(48)
        track_pen = QPen(track)
        track_pen.setWidthF(1.8)
        painter.setPen(track_pen)
        painter.drawEllipse(bounds)

        active_pen = QPen(QColor(theme_color("accent")))
        active_pen.setWidthF(1.9)
        active_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(active_pen)
        center = bounds.center()
        painter.translate(center)
        painter.rotate(self._angle)
        painter.translate(-center)
        painter.drawArc(bounds, 35 * 16, 250 * 16)
        painter.end()

        icon = QIcon()
        icon.addPixmap(pixmap, QIcon.Mode.Normal, QIcon.State.Off)
        icon.addPixmap(pixmap, QIcon.Mode.Disabled, QIcon.State.Off)
        button.setIcon(icon)
        button.setIconSize(QSize(logical_size, logical_size))


__all__ = ["ButtonSpinner", "make_icon_button", "set_button_icon", "ui_icon"]
