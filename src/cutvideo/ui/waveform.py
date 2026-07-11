"""Interactive waveform review widget with draggable cut boundaries."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from ..audio import WaveformEnvelope


class WaveformWidget(QWidget):
    """Draw an envelope and support boundary editing, panning and zooming."""

    boundariesChanged = Signal(int, int)
    boundaryDragFinished = Signal(int, int)
    viewChanged = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("waveformWidget")
        self.setMinimumHeight(190)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._envelope: WaveformEnvelope | None = None
        self._total_samples = 0
        self._view_start = 0
        self._view_end = 1
        self._selection_start = 0
        self._selection_end = 0
        self._dragging: str | None = None
        self._pan_press_x = 0.0
        self._pan_view_start = 0
        self._pan_view_end = 1
        self._placeholder = "完成分析后将在这里显示局部波形"
        self.setToolTip("拖动空白波形可前后平移；滚轮缩放；橙色边界可直接拖动。")

    @property
    def selection(self) -> tuple[int, int]:
        return self._selection_start, self._selection_end

    @property
    def view_range(self) -> tuple[int, int]:
        """Return the visible PCM sample interval."""

        return self._view_start, self._view_end

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def clear(self) -> None:
        self._envelope = None
        self._total_samples = 0
        self._view_start = 0
        self._view_end = 1
        self._selection_start = 0
        self._selection_end = 0
        self._dragging = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        self.viewChanged.emit(0, 0)

    def set_envelope(self, envelope: WaveformEnvelope, total_samples: int) -> None:
        self._envelope = envelope
        self._total_samples = max(0, int(total_samples))
        self._set_view_range(0, max(1, self._total_samples))
        self.update()

    def set_selection(
        self,
        start_sample: int,
        end_sample: int,
        *,
        center: bool = True,
        padding_seconds: float = 2.0,
    ) -> None:
        total = max(1, self._total_samples)
        start = max(0, min(total - 1, int(start_sample)))
        end = max(start + 1, min(total, int(end_sample)))
        self._selection_start = start
        self._selection_end = end
        if center and self._envelope is not None:
            self.focus_selection(padding_seconds=padding_seconds)
        self.update()

    def set_view(self, start_sample: int, end_sample: int) -> None:
        """Set the visible range while keeping it inside the decoded audio."""

        if self._envelope is None or self._total_samples <= 0:
            return
        self._set_view_range(start_sample, end_sample)

    def set_scroll_position(self, position: float) -> None:
        """Pan to a normalized position in the currently available scroll range."""

        if self._envelope is None or self._total_samples <= 0:
            return
        span = self._view_end - self._view_start
        available = max(0, self._total_samples - span)
        start = round(max(0.0, min(1.0, float(position))) * available)
        self._set_view_range(start, start + span)

    def pan_samples(self, delta_samples: int) -> None:
        """Move the visible window without changing its duration."""

        if self._envelope is None or self._total_samples <= 0:
            return
        delta = int(delta_samples)
        self._set_view_range(self._view_start + delta, self._view_end + delta)

    def pan_by_fraction(self, fraction: float) -> None:
        """Move by a fraction of the current visible duration."""

        span = self._view_end - self._view_start
        self.pan_samples(round(span * float(fraction)))

    def zoom(self, factor: float, *, anchor_sample: int | None = None) -> None:
        """Scale the visible duration around an optional audio sample."""

        if self._envelope is None or self._total_samples <= 0 or factor <= 0:
            return
        old_span = max(1, self._view_end - self._view_start)
        anchor = (
            (self._view_start + self._view_end) // 2
            if anchor_sample is None
            else max(0, min(self._total_samples, int(anchor_sample)))
        )
        anchor_ratio = max(0.0, min(1.0, (anchor - self._view_start) / old_span))
        new_span = round(old_span * float(factor))
        new_span = max(self._minimum_view_span(), min(self._total_samples, new_span))
        new_start = round(anchor - anchor_ratio * new_span)
        self._set_view_range(new_start, new_start + new_span)

    def zoom_in(self) -> None:
        self.zoom(0.72)

    def zoom_out(self) -> None:
        self.zoom(1.0 / 0.72)

    def focus_selection(self, *, padding_seconds: float = 2.0) -> None:
        """Show the complete selection with useful editable context on both sides."""

        if self._envelope is None or self._total_samples <= 0:
            return
        padding = max(
            round(self._envelope.sample_rate * max(0.0, float(padding_seconds))),
            (self._selection_end - self._selection_start) // 2,
        )
        self._set_view_range(
            self._selection_start - padding,
            self._selection_end + padding,
        )

    def _minimum_view_span(self) -> int:
        if self._envelope is None:
            return 1
        # 250 ms is precise enough for syllable boundaries while leaving enough
        # waveform context to understand what is being dragged.
        return min(
            max(1, self._total_samples),
            max(1, round(self._envelope.sample_rate * 0.25)),
        )

    def _set_view_range(self, start_sample: int, end_sample: int) -> None:
        total = max(1, self._total_samples)
        requested_start = int(start_sample)
        requested_end = int(end_sample)
        span = max(self._minimum_view_span(), requested_end - requested_start)
        span = min(total, span)
        start = max(0, min(total - span, requested_start))
        end = start + span
        if (start, end) == (self._view_start, self._view_end):
            return
        self._view_start = start
        self._view_end = end
        self.update()
        self.viewChanged.emit(start, end)

    def _sample_to_x(self, sample: int) -> float:
        width = max(1, self.width() - 2)
        duration = max(1, self._view_end - self._view_start)
        return 1.0 + (sample - self._view_start) * width / duration

    def _x_to_sample(self, x: float, *, clamp_to_view: bool = True) -> int:
        width = max(1, self.width() - 2)
        ratio = (x - 1.0) / width
        if clamp_to_view:
            ratio = max(0.0, min(1.0, ratio))
        return round(self._view_start + ratio * (self._view_end - self._view_start))

    def _boundary_distance(self, x: float, sample: int) -> float:
        if sample < self._view_start or sample > self._view_end:
            return math.inf
        return abs(x - self._sample_to_x(sample))

    def _keep_dragged_sample_visible(self, sample: int) -> None:
        span = self._view_end - self._view_start
        margin = max(1, round(span * 0.06))
        if sample < self._view_start + margin:
            self.pan_samples(sample - (self._view_start + margin))
        elif sample > self._view_end - margin:
            self.pan_samples(sample - (self._view_end - margin))

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#f8fafc"))
        painter.setPen(QPen(QColor("#dce2e9"), 1))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 7, 7)

        if self._envelope is None or not self._envelope.points or self._total_samples <= 0:
            painter.setPen(QColor("#7b8798"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return

        top = 14.0
        bottom = float(self.height() - 24)
        middle = (top + bottom) / 2
        amplitude = max(1.0, (bottom - top) / 2)
        painter.setPen(QPen(QColor("#e2e7ed"), 1))
        painter.drawLine(QPointF(1, middle), QPointF(self.width() - 2, middle))

        envelope = self._envelope
        first = max(0, self._view_start // envelope.block_size)
        last = min(
            envelope.points,
            math.ceil(self._view_end / envelope.block_size) + 1,
        )
        if last > first:
            point_indexes, visible_minimum, visible_maximum = _waveform_buckets(
                envelope,
                first,
                last,
                max_points=max(64, self.width() * 2),
            )
            path = QPainterPath()
            for offset, point_index in enumerate(point_indexes):
                sample = int(point_index) * envelope.block_size
                x = self._sample_to_x(sample)
                maximum = max(-1.0, min(1.0, float(visible_maximum[offset])))
                minimum = max(-1.0, min(1.0, float(visible_minimum[offset])))
                y_top = middle - maximum * amplitude
                y_bottom = middle - minimum * amplitude
                path.moveTo(x, y_top)
                path.lineTo(x, y_bottom)
            painter.setPen(QPen(QColor("#5276a8"), 1.15))
            painter.drawPath(path)

        selection_left = self._sample_to_x(self._selection_start)
        selection_right = self._sample_to_x(self._selection_end)
        visible_left = max(1.0, selection_left)
        visible_right = min(float(self.width() - 2), selection_right)
        if visible_right > visible_left:
            painter.fillRect(
                int(visible_left),
                int(top),
                max(1, int(visible_right - visible_left)),
                int(bottom - top),
                QColor(224, 153, 66, 45),
            )
        edge_pen = QPen(QColor("#c8782d"), 2)
        painter.setPen(edge_pen)
        if self._view_start <= self._selection_start <= self._view_end:
            painter.drawLine(QPointF(selection_left, top), QPointF(selection_left, bottom))
        if self._view_start <= self._selection_end <= self._view_end:
            painter.drawLine(QPointF(selection_right, top), QPointF(selection_right, bottom))

        painter.setPen(QColor("#697586"))
        left_seconds = self._view_start / envelope.sample_rate
        right_seconds = self._view_end / envelope.sample_rate
        painter.drawText(8, self.height() - 7, _format_seconds(left_seconds))
        right_text = _format_seconds(right_seconds)
        metrics = painter.fontMetrics()
        painter.drawText(
            self.width() - metrics.horizontalAdvance(right_text) - 8, self.height() - 7, right_text
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() is not Qt.MouseButton.LeftButton or self._envelope is None:
            return super().mousePressEvent(event)
        x = event.position().x()
        distance_start = self._boundary_distance(x, self._selection_start)
        distance_end = self._boundary_distance(x, self._selection_end)
        if min(distance_start, distance_end) <= 12:
            self._dragging = "start" if distance_start <= distance_end else "end"
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            event.accept()
            return
        self._dragging = "pan"
        self._pan_press_x = x
        self._pan_view_start = self._view_start
        self._pan_view_end = self._view_end
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging is None:
            if self._envelope is not None:
                near_edge = (
                    min(
                        self._boundary_distance(event.position().x(), self._selection_start),
                        self._boundary_distance(event.position().x(), self._selection_end),
                    )
                    <= 10
                )
                self.setCursor(
                    Qt.CursorShape.SizeHorCursor if near_edge else Qt.CursorShape.OpenHandCursor
                )
            return super().mouseMoveEvent(event)
        if self._dragging == "pan":
            width = max(1, self.width() - 2)
            span = self._pan_view_end - self._pan_view_start
            delta = round(-(event.position().x() - self._pan_press_x) * span / width)
            self._set_view_range(
                self._pan_view_start + delta,
                self._pan_view_end + delta,
            )
            event.accept()
            return
        sample = self._x_to_sample(event.position().x(), clamp_to_view=False)
        sample = max(0, min(self._total_samples, sample))
        if self._dragging == "start":
            self._selection_start = max(
                0,
                min(self._selection_end - 1, sample),
            )
        else:
            self._selection_end = min(
                self._total_samples,
                max(self._selection_start + 1, sample),
            )
        dragged_sample = self._selection_start if self._dragging == "start" else self._selection_end
        self._keep_dragged_sample_visible(dragged_sample)
        self.boundariesChanged.emit(self._selection_start, self._selection_end)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging is not None and event.button() is Qt.MouseButton.LeftButton:
            finished_boundary_drag = self._dragging in {"start", "end"}
            self._dragging = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            if finished_boundary_drag:
                self.boundaryDragFinished.emit(
                    self._selection_start,
                    self._selection_end,
                )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._envelope is None or self._total_samples <= 0:
            return super().wheelEvent(event)
        delta = event.angleDelta()
        if delta.x() or event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            amount = delta.x() if delta.x() else delta.y()
            self.pan_by_fraction(-amount / 120.0 * 0.12)
        elif delta.y():
            steps = delta.y() / 120.0
            self.zoom(
                0.82**steps,
                anchor_sample=self._x_to_sample(event.position().x()),
            )
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._envelope is None:
            return super().keyPressEvent(event)
        if event.key() == Qt.Key.Key_Left:
            self.pan_by_fraction(-0.12)
        elif event.key() == Qt.Key.Key_Right:
            self.pan_by_fraction(0.12)
        elif event.key() == Qt.Key.Key_PageUp:
            self.pan_by_fraction(-0.8)
        elif event.key() == Qt.Key.Key_PageDown:
            self.pan_by_fraction(0.8)
        elif event.key() in {Qt.Key.Key_Plus, Qt.Key.Key_Equal}:
            self.zoom_in()
        elif event.key() == Qt.Key.Key_Minus:
            self.zoom_out()
        elif event.key() == Qt.Key.Key_0:
            self.focus_selection()
        else:
            return super().keyPressEvent(event)
        event.accept()


def _format_seconds(value: float) -> str:
    minutes, seconds = divmod(max(0.0, value), 60.0)
    return f"{int(minutes):02d}:{seconds:05.2f}"


def _waveform_buckets(
    envelope: WaveformEnvelope,
    first: int,
    last: int,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel-bound a visible envelope while preserving extrema in each bucket."""

    first = max(0, min(envelope.points, int(first)))
    last = max(first, min(envelope.points, int(last)))
    count = last - first
    if count <= 0 or max_points <= 0:
        empty_int = np.empty(0, dtype=np.int64)
        empty_float = np.empty(0, dtype=np.float32)
        return empty_int, empty_float, empty_float
    step = max(1, math.ceil(count / max_points))
    starts = np.arange(0, count, step, dtype=np.int64)
    minimum = np.minimum.reduceat(envelope.minimum[first:last], starts)
    maximum = np.maximum.reduceat(envelope.maximum[first:last], starts)
    return first + starts, minimum, maximum


__all__ = ["WaveformWidget"]
