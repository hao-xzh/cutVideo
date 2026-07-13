"""Full-workspace file import landing views."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QMimeData, QSize, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .icons import set_button_icon, ui_icon

_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
_DOCUMENT_SUFFIXES = {".docx"}


class ImportLandingView(QWidget):
    """Large drag-and-drop entry shown before an editor workspace is needed."""

    audioBrowseRequested = Signal()
    documentBrowseRequested = Signal()
    projectBrowseRequested = Signal()
    audioDropped = Signal(str)
    documentDropped = Signal(str)

    def __init__(self, mode: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if mode not in {"word", "audio"}:
            raise ValueError(f"unsupported import mode: {mode}")
        self.mode = mode
        self.setObjectName("importLanding")
        self.setAcceptDrops(True)
        self._build_ui()
        self.set_paths()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(34, 30, 34, 34)
        root.addStretch(1)

        surface = QWidget()
        surface.setObjectName("importDropSurface")
        surface.setMaximumWidth(880)
        surface.setMinimumHeight(470)
        surface.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.drop_surface = surface
        content = QVBoxLayout(surface)
        content.setContentsMargins(46, 42, 46, 38)
        content.setSpacing(12)

        icon = QLabel()
        icon.setObjectName("importHeroIcon")
        icon.setPixmap(ui_icon("upload").pixmap(QSize(58, 58)))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.addWidget(icon)

        word_mode = self.mode == "word"
        kicker = QLabel("WORD WORKFLOW" if word_mode else "TRANSCRIPT EDITOR")
        kicker.setObjectName("importKicker")
        kicker.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("拖入音频与标注文档" if word_mode else "拖入一段音频开始")
        title.setObjectName("importTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle = QLabel(
            "同时拖入音频和 .docx，或分别选择文件"
            if word_mode
            else "导入后进入完整转写、波形标注与试听工作台"
        )
        subtitle.setObjectName("importSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        content.addWidget(kicker)
        content.addWidget(title)
        content.addWidget(subtitle)
        content.addSpacing(12)

        file_row = QHBoxLayout()
        file_row.setSpacing(10)
        self.audio_button = _file_button("audio", "选择音频")
        self.audio_button.clicked.connect(lambda _checked=False: self.audioBrowseRequested.emit())
        file_row.addWidget(self.audio_button, 1)
        self.document_button: QPushButton | None = None
        if word_mode:
            self.document_button = _file_button("document", "选择 Word")
            self.document_button.clicked.connect(
                lambda _checked=False: self.documentBrowseRequested.emit()
            )
            file_row.addWidget(self.document_button, 1)
        content.addLayout(file_row)

        self.state_label = QLabel()
        self.state_label.setObjectName("importStateLabel")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.addWidget(self.state_label)

        formats = QLabel(
            "MP3 · WAV · M4A · AAC · FLAC  +  DOCX"
            if word_mode
            else "MP3 · WAV · M4A · AAC · FLAC"
        )
        formats.setObjectName("importFormats")
        formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.addWidget(formats)
        content.addStretch(1)

        project_button = QPushButton("打开已有项目")
        project_button.setObjectName("importProjectButton")
        project_button.setProperty("secondary", True)
        set_button_icon(project_button, "project", size=17)
        project_button.clicked.connect(lambda _checked=False: self.projectBrowseRequested.emit())
        self.project_button = project_button
        project_row = QHBoxLayout()
        project_row.addStretch(1)
        project_row.addWidget(project_button)
        project_row.addStretch(1)
        content.addLayout(project_row)

        privacy = QLabel("文件不会上传 · 识别与剪辑均在本机完成")
        privacy.setObjectName("importPrivacy")
        privacy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.addWidget(privacy)

        center = QHBoxLayout()
        center.addStretch(1)
        center.addWidget(surface, 8)
        center.addStretch(1)
        root.addLayout(center, 8)
        root.addStretch(1)

    def set_paths(self, audio_path: str = "", document_path: str = "") -> None:
        """Reflect selected inputs while the landing page remains visible."""

        audio_source = Path(audio_path).expanduser() if audio_path else None
        audio_ready = bool(
            audio_source
            and audio_source.is_file()
            and audio_source.suffix.casefold() in _AUDIO_SUFFIXES
        )
        audio_name = audio_source.name if audio_ready and audio_source is not None else ""
        _set_file_button(self.audio_button, "音频", audio_name)
        document_source = Path(document_path).expanduser() if document_path else None
        document_ready = bool(
            document_source
            and document_source.is_file()
            and document_source.suffix.casefold() in _DOCUMENT_SUFFIXES
        )
        document_name = (
            document_source.name
            if document_ready and document_source is not None
            else ""
        )
        if self.document_button is not None:
            _set_file_button(self.document_button, "Word", document_name)

        invalid_audio = bool(audio_path) and not audio_ready
        invalid_document = bool(document_path) and not document_ready
        if invalid_audio or (self.mode == "word" and invalid_document):
            message = "文件不存在或格式不受支持"
        elif self.mode == "audio":
            message = "音频已就绪" if audio_name else "可拖放文件到窗口任意位置"
        elif audio_name and document_name:
            message = "音频与 Word 已就绪"
        elif audio_name:
            message = "音频已选择，还需要 Word 文档"
        elif document_name:
            message = "Word 已选择，还需要音频"
        else:
            message = "可一次拖入两个文件"
        self.state_label.setText(message)

    def set_busy(self, busy: bool) -> None:
        for button in (self.audio_button, self.document_button, self.project_button):
            if button is not None:
                button.setEnabled(not busy)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
            self._set_drop_active(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self._set_drop_active(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        self._set_drop_active(False)
        paths = self._paths_from_mime(event.mimeData())
        if not paths:
            event.ignore()
            return
        audio = next((path for path in paths if path.suffix.casefold() in _AUDIO_SUFFIXES), None)
        document = next(
            (path for path in paths if path.suffix.casefold() in _DOCUMENT_SUFFIXES),
            None,
        )
        if audio is not None:
            self.audioDropped.emit(str(audio))
        if self.mode == "word" and document is not None:
            self.documentDropped.emit(str(document))
        event.acceptProposedAction()

    def _paths_from_mime(self, mime: QMimeData) -> list[Path]:
        if not mime.hasUrls():
            return []
        allowed = _AUDIO_SUFFIXES | (_DOCUMENT_SUFFIXES if self.mode == "word" else set())
        paths = [
            Path(url.toLocalFile()).resolve()
            for url in mime.urls()
            if url.isLocalFile() and Path(url.toLocalFile()).is_file()
        ]
        return [path for path in paths if path.suffix.casefold() in allowed]

    def _set_drop_active(self, active: bool) -> None:
        if self.drop_surface.property("dropActive") == active:
            return
        self.drop_surface.setProperty("dropActive", active)
        self.drop_surface.style().unpolish(self.drop_surface)
        self.drop_surface.style().polish(self.drop_surface)


def _file_button(icon_name: str, text: str) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("importFileButton")
    button.setMinimumHeight(62)
    set_button_icon(button, icon_name, size=23)
    return button


def _set_file_button(button: QPushButton, label: str, filename: str) -> None:
    button.setText(f"{label}  ·  {filename}" if filename else f"选择{label}")
    ready = bool(filename)
    if button.property("ready") == ready:
        return
    button.setProperty("ready", ready)
    button.style().unpolish(button)
    button.style().polish(button)


__all__ = ["ImportLandingView"]
