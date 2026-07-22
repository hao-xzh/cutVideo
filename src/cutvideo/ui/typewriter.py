"""Typewriter-style text appending for GUI-thread ``QTextEdit`` updates."""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QTextEdit

_CURSOR_END = getattr(QTextCursor, "End", QTextCursor.MoveOperation.End)


class TypewriterTextController(QObject):
    """Reveal already received text deltas into a read-only ``QTextEdit``.

    The controller does not cross thread boundaries.  Call its public methods
    from GUI slots, then let the internal timer append text on the GUI thread.
    """

    # Keep ordinary ASR deltas visible long enough to bridge the next model
    # window.  Only a genuinely large backlog accelerates, so the transcript
    # feels continuous instead of flashing one whole window and then pausing.
    _INTERVAL_MS = 45
    _MAX_CHARS_PER_TICK = 24

    def __init__(self, text_edit: QTextEdit, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._text_edit = text_edit
        self._pending_chunks: deque[str] = deque()
        self._pending_length = 0
        self._timer = QTimer(self)
        self._timer.setInterval(self._INTERVAL_MS)
        self._timer.timeout.connect(self._append_next_chunk)
        self._text_edit.setReadOnly(True)

    @property
    def is_active(self) -> bool:
        return self._timer.isActive()

    def start(self, placeholder: str) -> None:
        self.stop()
        self._text_edit.setReadOnly(True)
        self._text_edit.setPlaceholderText(placeholder)
        self._text_edit.clear()

    def append_text(self, delta: str) -> None:
        if not delta:
            return
        reveal_immediately = self._pending_length == 0 and not self._timer.isActive()
        self._pending_chunks.append(delta)
        self._pending_length += len(delta)
        if reveal_immediately:
            self._append_next_chunk()
        if self._pending_length and not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._pending_chunks.clear()
        self._pending_length = 0

    def replace_text(self, text: str) -> None:
        visible = self._text_edit.toPlainText()
        if text.startswith(visible):
            self.stop()
            self.append_text(text[len(visible) :])
            self._move_to_end()
            return
        self.stop()
        self._text_edit.setPlainText(text)
        self._move_to_end()

    def clear(self) -> None:
        self.stop()
        self._text_edit.clear()

    def _append_next_chunk(self) -> None:
        if not self._pending_length:
            self._timer.stop()
            return

        chunk_size = self._chunk_size(self._pending_length)
        chunk = self._take_pending(chunk_size)

        cursor = self._text_edit.textCursor()
        cursor.movePosition(_CURSOR_END)
        cursor.insertText(chunk)
        self._text_edit.setTextCursor(cursor)
        self._text_edit.ensureCursorVisible()

        if not self._pending_length:
            self._timer.stop()

    def _take_pending(self, character_count: int) -> str:
        parts: list[str] = []
        remaining = min(character_count, self._pending_length)
        self._pending_length -= remaining
        while remaining:
            head = self._pending_chunks.popleft()
            if len(head) <= remaining:
                parts.append(head)
                remaining -= len(head)
                continue
            parts.append(head[:remaining])
            self._pending_chunks.appendleft(head[remaining:])
            remaining = 0
        return "".join(parts)

    def _move_to_end(self) -> None:
        cursor = self._text_edit.textCursor()
        cursor.movePosition(_CURSOR_END)
        self._text_edit.setTextCursor(cursor)
        self._text_edit.ensureCursorVisible()

    @classmethod
    def _chunk_size(cls, backlog_length: int) -> int:
        if backlog_length >= 2_400:
            return cls._MAX_CHARS_PER_TICK
        if backlog_length >= 1_200:
            return 16
        if backlog_length >= 600:
            return 10
        if backlog_length >= 360:
            return 6
        if backlog_length >= 240:
            return 4
        if backlog_length >= 160:
            return 3
        if backlog_length >= 80:
            return 2
        return 1


__all__ = ["TypewriterTextController"]
