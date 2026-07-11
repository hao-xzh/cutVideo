"""Path input that accepts a single local file by drag and drop."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QMimeData, Signal
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent
from PySide6.QtWidgets import QLineEdit, QWidget


class FileDropLineEdit(QLineEdit):
    fileDropped = Signal(str)

    def __init__(
        self,
        accepted_suffixes: tuple[str, ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.accepted_suffixes = tuple(item.casefold() for item in accepted_suffixes)
        self.setAcceptDrops(True)

    def accepts_path(self, path: str | Path) -> bool:
        candidate = Path(path)
        return candidate.is_file() and candidate.suffix.casefold() in self.accepted_suffixes

    def _path_from_mime(self, mime: QMimeData) -> Path | None:
        if not mime.hasUrls():
            return None
        local_paths = [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]
        if len(local_paths) != 1 or not self.accepts_path(local_paths[0]):
            return None
        return local_paths[0].resolve()

    def _set_drop_active(self, active: bool) -> None:
        if self.property("dropActive") == active:
            return
        self.setProperty("dropActive", active)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._path_from_mime(event.mimeData()) is not None:
            event.acceptProposedAction()
            self._set_drop_active(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self._set_drop_active(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        self._set_drop_active(False)
        path = self._path_from_mime(event.mimeData())
        if path is None:
            event.ignore()
            return
        value = str(path)
        self.setText(value)
        self.fileDropped.emit(value)
        event.acceptProposedAction()


__all__ = ["FileDropLineEdit"]
