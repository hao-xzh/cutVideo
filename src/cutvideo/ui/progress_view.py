"""Compact, theme-aware progress presentation for background tasks."""

from __future__ import annotations

import time

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .theme import theme_color


class _SmoothProgressBar(QProgressBar):
    """A fine, custom-painted bar with a restrained one-way shimmer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRange(0, 100)
        self.setTextVisible(False)
        self.setFixedHeight(4)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._target = 0
        self._active = False
        self._shimmer_x = -36.0
        self._shimmer_timer = QTimer(self)
        self._shimmer_timer.setInterval(32)
        self._shimmer_timer.timeout.connect(self._advance_shimmer)
        self._animation = QPropertyAnimation(self, b"displayValue", self)
        self._animation.setDuration(220)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

    def _get_display_value(self) -> int:
        return self.value()

    def _set_display_value(self, value: int) -> None:
        # Both incoming targets and animation frames are clamped monotonically.
        self.setValue(max(self.value(), min(100, int(round(value)))))
        self.update()

    displayValue = Property(int, _get_display_value, _set_display_value)

    @property
    def target(self) -> int:
        return self._target

    def set_target(self, value: int) -> None:
        value = max(0, min(100, int(value)))
        if value <= self._target:
            return
        self._target = value
        self._animation.stop()
        self._animation.setStartValue(self.value())
        self._animation.setEndValue(value)
        self._animation.start()

    def reset_target(self) -> None:
        self._animation.stop()
        self._target = 0
        self.setValue(0)
        self._shimmer_x = -36.0
        self.update()

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if self._active and self.isVisible():
            if not self._shimmer_timer.isActive():
                self._shimmer_timer.start()
        else:
            self._shimmer_timer.stop()
        self.update()

    def _advance_shimmer(self) -> None:
        self._shimmer_x += 2.4
        if self._shimmer_x > self.width() + 36:
            self._shimmer_x = -36.0
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if self._active:
            self._shimmer_timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._shimmer_timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(self.rect()).adjusted(0, 0.5, 0, -0.5)
        radius = bounds.height() / 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme_color("progress_track")))
        painter.drawRoundedRect(bounds, radius, radius)

        fraction = self.value() / 100
        if fraction <= 0:
            return
        fill = QRectF(bounds)
        fill.setWidth(bounds.width() * fraction)
        path = QPainterPath()
        path.addRoundedRect(bounds, radius, radius)
        painter.save()
        painter.setClipPath(path)
        painter.fillRect(fill, QColor(theme_color("accent")))
        if self._active:
            shine = QLinearGradient(self._shimmer_x - 18, 0, self._shimmer_x + 18, 0)
            shine.setColorAt(0.0, QColor(255, 255, 255, 0))
            shine.setColorAt(0.5, QColor(255, 255, 255, 70))
            shine.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.fillRect(fill, shine)
        painter.restore()


class TaskProgressView(QWidget):
    """A compact status row and progress bar for cancellable background work."""

    cancelRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("taskProgressView")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._active = False
        self._started_at = 0.0
        self._message = ""
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1_000)
        self._elapsed_timer.timeout.connect(self._render_message)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 2)
        root.setSpacing(3)
        row = QHBoxLayout()
        row.setSpacing(8)

        self.message_label = QLabel(" ")
        self.message_label.setObjectName("mutedLabel")
        self.message_label.setAccessibleName("任务阶段")
        self.message_label.setToolTip("当前任务阶段")
        self.message_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        self.percentage_label = QLabel("0%")
        self.percentage_label.setObjectName("mutedLabel")
        self.percentage_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.percentage_label.setMinimumWidth(34)
        self.percentage_label.setAccessibleName("任务进度")
        self.percentage_label.setToolTip("当前任务进度")

        self.cancel_button = QPushButton("取消")
        self.cancel_button.setObjectName("cancelTaskButton")
        self.cancel_button.setProperty("compact", True)
        self.cancel_button.setProperty("danger", True)
        self.cancel_button.setToolTip("取消当前任务")
        self.cancel_button.setAccessibleName("取消任务")
        self.cancel_button.clicked.connect(self.cancelRequested)

        row.addWidget(self.message_label, 1)
        row.addWidget(self.percentage_label)
        row.addWidget(self.cancel_button)

        self.progress_bar = _SmoothProgressBar(self)
        self.progress_bar.setAccessibleName("任务进度条")
        self.progress_bar.setToolTip("当前任务进度")
        root.addLayout(row)
        root.addWidget(self.progress_bar)
        self.set_cancel_enabled(True)
        self.setVisible(False)

    def start(self, message: str = "") -> None:
        self._active = True
        self._started_at = time.monotonic()
        self._message = str(message)
        self.progress_bar.reset_target()
        self._set_percentage(0)
        self._render_message()
        self.progress_bar.set_active(True)
        self._elapsed_timer.start()
        self.show()

    def set_progress(self, value: float | int, message: str | None = None) -> None:
        numeric = float(value)
        # Existing task signals use integer percentages; accept fractional
        # normalized values as a convenience without turning ``1%`` into 100%.
        if isinstance(value, float) and 0 <= numeric <= 1:
            numeric *= 100
        target = max(0, min(100, round(numeric)))
        self.progress_bar.set_target(target)
        self._set_percentage(self.progress_bar.target)
        if message is not None:
            self.set_message(message)

    def set_message(self, message: str) -> None:
        self._message = str(message)
        self._render_message()

    def finish(self) -> None:
        self._active = False
        self._started_at = 0.0
        self._elapsed_timer.stop()
        self.progress_bar.set_active(False)
        self.hide()

    def set_cancel_enabled(self, enabled: bool) -> None:
        self.cancel_button.setEnabled(bool(enabled))

    def _set_percentage(self, value: int) -> None:
        self.percentage_label.setText(f"{max(0, min(100, int(value)))}%")

    def _render_message(self) -> None:
        message = self._message or " "
        if self._active and self._started_at > 0:
            elapsed = max(0.0, time.monotonic() - self._started_at)
            if elapsed >= 1:
                message = f"{message} · 已用 {_format_duration(elapsed)}"
        self.message_label.setText(message)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if self._active:
            self.progress_bar.set_active(True)

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.progress_bar.set_active(False)
        super().hideEvent(event)


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, remaining = divmod(total, 60)
    if minutes:
        return f"{minutes}分 {remaining}秒"
    return f"{remaining}秒"


__all__ = ["TaskProgressView"]
