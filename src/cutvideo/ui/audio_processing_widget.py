"""Standalone full-transcript audio annotation and editing workspace."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from PySide6.QtCore import QDir, QPoint, QSignalBlocker, Qt, QTemporaryDir, QThreadPool, QTimer
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QKeySequence,
    QShortcut,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..audio import AudioInfo, WaveformEnvelope
from ..audio_processing import (
    AudioAnnotation,
    AudioProcessingProject,
    save_audio_processing_project,
)
from ..ffmpeg import FFmpegTools
from .audio_player import AudioPlaybackError, PcmWavPlayer
from .audio_processing_workers import (
    AudioProcessingAnalysisResult,
    AudioProcessingExportResult,
    make_audio_processing_analysis_operation,
    make_audio_processing_export_operation,
    make_audio_processing_load_operation,
    make_audio_processing_preview_operation,
)
from .waveform import WaveformWidget
from .workers import BackgroundTask, CandidatePreviewResult, make_waveform_operation

_SCROLL_STEPS = 100_000


class AudioProcessingWidget(QWidget):
    """Full audio transcription, region annotations, preview and export."""

    def __init__(self, thread_pool: QThreadPool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("audioProcessingWorkspace")
        self.thread_pool = thread_pool
        self.project: AudioProcessingProject | None = None
        self.project_path: Path | None = None
        self.audio_info: AudioInfo | None = None
        self.tools: FFmpegTools | None = None
        self.waveform_envelope: WaveformEnvelope | None = None
        self._active_task: BackgroundTask | None = None
        self._token_document_ranges: list[tuple[int, int]] = []
        self._editing_annotation_id: str | None = None
        self._updating = False
        self._shortcuts: list[QShortcut] = []
        template = str(Path(QDir.tempPath()) / "cutvideo-audio-processing-XXXXXX")
        self._preview_directory = QTemporaryDir(template)
        self._player = PcmWavPlayer(self)
        self._player.failed.connect(self._playback_failed)
        self._player.active_changed.connect(lambda _active: self._refresh_controls())
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(450)
        self._autosave_timer.timeout.connect(self._autosave)
        self._build_ui()
        self._connect_signals()
        self._refresh_controls()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 12)
        root.setSpacing(12)

        header = QFrame()
        header.setObjectName("headerCard")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 13, 18, 13)
        heading = QVBoxLayout()
        heading.setSpacing(2)
        title = QLabel("音频处理")
        title.setObjectName("titleLabel")
        subtitle = QLabel("完整离线转写后，选择文字建立删除标注；文字、波形与试听范围实时同步")
        subtitle.setObjectName("subtitleLabel")
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header_layout.addLayout(heading, 1)
        shortcut_hint = QLabel("Delete / Backspace / Ctrl+B  标注删除")
        shortcut_hint.setObjectName("shortcutBadge")
        header_layout.addWidget(shortcut_hint)
        root.addWidget(header)

        input_card = QFrame()
        input_card.setObjectName("inputCard")
        input_layout = QGridLayout(input_card)
        input_layout.setContentsMargins(18, 13, 18, 13)
        input_layout.setHorizontalSpacing(10)
        input_layout.addWidget(QLabel("音频"), 0, 0)
        self.audio_path_edit = QLineEdit()
        self.audio_path_edit.setObjectName("processingAudioPathEdit")
        self.audio_path_edit.setPlaceholderText("选择一个需要单独处理的 MP3、WAV、M4A/AAC 或 FLAC")
        self.audio_browse_button = QPushButton("选择音频…")
        self.audio_browse_button.setObjectName("processingAudioBrowseButton")
        self.open_project_button = QPushButton("打开项目…")
        self.open_project_button.setObjectName("processingOpenProjectButton")
        self.transcribe_button = QPushButton("完整识别")
        self.transcribe_button.setObjectName("processingTranscribeButton")
        self.transcribe_button.setProperty("primary", True)
        input_layout.addWidget(self.audio_path_edit, 0, 1)
        input_layout.addWidget(self.audio_browse_button, 0, 2)
        input_layout.addWidget(self.open_project_button, 0, 3)
        input_layout.addWidget(self.transcribe_button, 0, 4)
        self.summary_label = QLabel("选择音频后执行完整识别")
        self.summary_label.setObjectName("mutedLabel")
        input_layout.addWidget(self.summary_label, 1, 1, 1, 4)
        input_layout.setColumnStretch(1, 1)
        root.addWidget(input_card)

        self.progress_container = QWidget()
        progress_layout = QHBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(4, 0, 4, 0)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("mutedLabel")
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setProperty("danger", True)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar, 1)
        progress_layout.addWidget(self.cancel_button)
        self.progress_container.setVisible(False)
        root.addWidget(self.progress_container)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("audioProcessingSplitter")
        splitter.setChildrenCollapsible(False)

        transcript_card = QFrame()
        transcript_card.setObjectName("reviewCard")
        transcript_layout = QVBoxLayout(transcript_card)
        transcript_layout.setContentsMargins(14, 13, 14, 14)
        transcript_heading = QHBoxLayout()
        transcript_title = QLabel("完整转写")
        transcript_title.setObjectName("sectionTitle")
        self.selection_label = QLabel("拖选文字后建立标注")
        self.selection_label.setObjectName("mutedLabel")
        transcript_heading.addWidget(transcript_title)
        transcript_heading.addStretch(1)
        transcript_heading.addWidget(self.selection_label)
        transcript_layout.addLayout(transcript_heading)
        self.transcript_edit = QTextEdit()
        self.transcript_edit.setObjectName("processingTranscriptEdit")
        self.transcript_edit.setReadOnly(True)
        self.transcript_edit.setAcceptRichText(False)
        self.transcript_edit.setPlaceholderText("完整识别完成后，文字会在这里按语音停顿分段显示")
        self.transcript_edit.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        transcript_layout.addWidget(self.transcript_edit, 1)
        transcript_actions = QHBoxLayout()
        self.add_delete_button = QPushButton("标注为删除")
        self.add_delete_button.setObjectName("processingAddDeleteButton")
        self.add_delete_button.setProperty("primary", True)
        self.preview_text_button = QPushButton("试听选中文字")
        self.preview_text_button.setObjectName("processingPreviewTextButton")
        transcript_actions.addWidget(self.add_delete_button)
        transcript_actions.addWidget(self.preview_text_button)
        transcript_actions.addStretch(1)
        transcript_layout.addLayout(transcript_actions)
        splitter.addWidget(transcript_card)

        detail_card = QFrame()
        detail_card.setObjectName("reviewCard")
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(14, 13, 14, 14)
        detail_layout.setSpacing(8)
        detail_title = QLabel("范围与删除标注")
        detail_title.setObjectName("sectionTitle")
        detail_layout.addWidget(detail_title)
        waveform_hint = QLabel("拖动波形框选范围；拖橙色边界精调；Alt+拖动平移；滚轮缩放")
        waveform_hint.setObjectName("mutedLabel")
        detail_layout.addWidget(waveform_hint)
        self.waveform = WaveformWidget()
        self.waveform.setObjectName("processingWaveform")
        self.waveform.set_selection_drawing_enabled(True)
        self.waveform.set_placeholder("完成完整识别后将在这里显示可框选波形")
        detail_layout.addWidget(self.waveform, 1)

        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("视野"))
        self.waveform_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.waveform_scrollbar.setObjectName("processingWaveformScrollBar")
        self.waveform_scrollbar.setRange(0, 0)
        self.zoom_out_button = QPushButton("缩小")
        self.zoom_in_button = QPushButton("放大")
        self.focus_button = QPushButton("定位范围")
        for button in (self.zoom_out_button, self.zoom_in_button, self.focus_button):
            button.setProperty("compact", True)
        view_layout.addWidget(self.waveform_scrollbar, 1)
        view_layout.addWidget(self.zoom_out_button)
        view_layout.addWidget(self.zoom_in_button)
        view_layout.addWidget(self.focus_button)
        detail_layout.addLayout(view_layout)

        boundary_layout = QGridLayout()
        self.start_spin = _time_spin("processingStartSpin")
        self.end_spin = _time_spin("processingEndSpin")
        boundary_layout.addWidget(QLabel("范围开始"), 0, 0)
        boundary_layout.addWidget(QLabel("范围结束"), 0, 1)
        boundary_layout.addWidget(self.start_spin, 1, 0)
        boundary_layout.addWidget(self.end_spin, 1, 1)
        detail_layout.addLayout(boundary_layout)

        preview_layout = QHBoxLayout()
        self.preview_selection_button = QPushButton("试听框选")
        self.preview_edited_button = QPushButton("试听删除后")
        self.stop_button = QPushButton("停止")
        preview_layout.addWidget(self.preview_selection_button)
        preview_layout.addWidget(self.preview_edited_button)
        preview_layout.addWidget(self.stop_button)
        preview_layout.addStretch(1)
        detail_layout.addLayout(preview_layout)

        annotation_heading = QHBoxLayout()
        annotation_title = QLabel("删除标注")
        annotation_title.setObjectName("sectionTitle")
        self.annotation_summary = QLabel("0 项")
        self.annotation_summary.setObjectName("mutedLabel")
        annotation_heading.addWidget(annotation_title)
        annotation_heading.addStretch(1)
        annotation_heading.addWidget(self.annotation_summary)
        detail_layout.addLayout(annotation_heading)
        self.annotation_table = QTableWidget(0, 3)
        self.annotation_table.setObjectName("processingAnnotationTable")
        self.annotation_table.setHorizontalHeaderLabels(["文字", "范围", "操作"])
        self.annotation_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.annotation_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.annotation_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.annotation_table.verticalHeader().setVisible(False)
        annotation_header = self.annotation_table.horizontalHeader()
        annotation_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        annotation_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        annotation_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        detail_layout.addWidget(self.annotation_table, 1)
        annotation_actions = QHBoxLayout()
        self.remove_annotation_button = QPushButton("取消所选标注")
        self.remove_annotation_button.setObjectName("processingRemoveAnnotationButton")
        annotation_actions.addStretch(1)
        annotation_actions.addWidget(self.remove_annotation_button)
        detail_layout.addLayout(annotation_actions)
        splitter.addWidget(detail_card)
        splitter.setSizes([760, 560])
        root.addWidget(splitter, 1)

        export_card = QFrame()
        export_card.setObjectName("exportCard")
        export_layout = QHBoxLayout(export_card)
        export_layout.setContentsMargins(16, 11, 16, 11)
        export_title = QLabel("导出位置")
        export_title.setObjectName("sectionTitle")
        self.output_edit = QLineEdit()
        self.output_edit.setObjectName("processingOutputPathEdit")
        self.output_browse_button = QPushButton("选择目录…")
        self.export_button = QPushButton("导出处理完成音频")
        self.export_button.setObjectName("processingExportButton")
        self.export_button.setProperty("primary", True)
        export_layout.addWidget(export_title)
        export_layout.addWidget(self.output_edit, 1)
        export_layout.addWidget(self.output_browse_button)
        export_layout.addWidget(self.export_button)
        root.addWidget(export_card)

    def _connect_signals(self) -> None:
        self.audio_browse_button.clicked.connect(self._choose_audio)
        self.audio_path_edit.textChanged.connect(lambda _value: self._refresh_controls())
        self.open_project_button.clicked.connect(self._choose_project)
        self.transcribe_button.clicked.connect(self._start_transcription)
        self.cancel_button.clicked.connect(self._cancel_task)
        self.transcript_edit.selectionChanged.connect(self._text_selection_changed)
        self.transcript_edit.customContextMenuRequested.connect(self._show_transcript_menu)
        self.add_delete_button.clicked.connect(self._add_delete_annotation)
        self.preview_text_button.clicked.connect(lambda: self._start_preview("selection"))
        self.annotation_table.currentCellChanged.connect(self._annotation_row_changed)
        self.remove_annotation_button.clicked.connect(self._remove_selected_annotation)
        self.waveform.boundariesChanged.connect(self._waveform_selection_changed)
        self.waveform.boundaryDragFinished.connect(self._waveform_drag_finished)
        self.waveform.viewChanged.connect(self._waveform_view_changed)
        self.waveform_scrollbar.valueChanged.connect(self._waveform_scroll_changed)
        self.zoom_out_button.clicked.connect(self.waveform.zoom_out)
        self.zoom_in_button.clicked.connect(self.waveform.zoom_in)
        self.focus_button.clicked.connect(self.waveform.focus_selection)
        self.start_spin.valueChanged.connect(self._spin_selection_changed)
        self.end_spin.valueChanged.connect(self._spin_selection_changed)
        self.preview_selection_button.clicked.connect(lambda: self._start_preview("selection"))
        self.preview_edited_button.clicked.connect(lambda: self._start_preview("edited"))
        self.stop_button.clicked.connect(self._player.stop)
        self.output_browse_button.clicked.connect(self._choose_output)
        self.output_edit.textChanged.connect(self._output_changed)
        self.export_button.clicked.connect(self._start_export)
        for sequence in ("Delete", "Backspace", "Ctrl+B", "Meta+B"):
            shortcut = QShortcut(QKeySequence(sequence), self.transcript_edit)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(self._add_delete_annotation)
            self._shortcuts.append(shortcut)

    def _choose_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择需要处理的音频",
            self.audio_path_edit.text(),
            "音频文件 (*.mp3 *.wav *.m4a *.aac *.flac);;所有文件 (*)",
        )
        if path:
            self.audio_path_edit.setText(path)

    def _choose_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开音频处理项目",
            self.audio_path_edit.text(),
            "音频处理项目 (*.audioprocess.json);;JSON 文件 (*.json)",
        )
        if path:
            self._start_task(make_audio_processing_load_operation(path), self._analysis_ready)

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择导出目录", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _start_transcription(self) -> None:
        path = self.audio_path_edit.text().strip()
        if not path:
            QMessageBox.information(self, "请选择音频", "请先选择一个需要单独处理的音频文件。")
            return
        self._start_task(make_audio_processing_analysis_operation(path), self._analysis_ready)

    def _analysis_ready(self, value: object) -> None:
        assert isinstance(value, AudioProcessingAnalysisResult)
        self.project = value.project
        self.project_path = value.project_path
        self.audio_info = value.audio_info
        self.tools = value.tools
        self.audio_path_edit.setText(value.project.audio.path)
        self.output_edit.setText(value.project.output_directory)
        self._render_transcript()
        self._populate_annotations()
        duration = _format_sample(value.audio_info.total_samples, value.audio_info.sample_rate)
        self.summary_label.setText(
            f"{len(value.project.tokens)} 个带时间戳文字 · 音频 {duration} · "
            f"{len(value.project.annotations)} 个删除标注"
        )
        self.waveform.clear()
        self.waveform.set_placeholder("正在后台生成完整波形…")
        QTimer.singleShot(
            0,
            lambda: self._start_task(
                make_waveform_operation(value.audio_info, value.tools),
                self._waveform_ready,
            ),
        )

    def _waveform_ready(self, value: object) -> None:
        assert isinstance(value, WaveformEnvelope)
        if self.audio_info is None:
            return
        self.waveform_envelope = value
        self.waveform.set_envelope(value, self.audio_info.total_samples)
        if self.project and self.project.annotations:
            self.annotation_table.selectRow(0)
        elif self.project and self.project.tokens:
            first = self.project.tokens[0]
            self.waveform.set_selection(first.start_sample, first.end_sample)
        self._refresh_controls()

    def _render_transcript(self) -> None:
        if self.project is None:
            return
        parts: list[str] = []
        ranges: list[tuple[int, int]] = []
        position = 0
        line_characters = 0
        previous_end = 0
        sample_rate = self.project.audio_info.sample_rate
        for index, token in enumerate(self.project.tokens):
            gap = (token.start_sample - previous_end) / sample_rate if index else 0.0
            if index and (gap >= 0.75 or line_characters >= 42):
                parts.append("\n")
                position += 1
                line_characters = 0
            start = position
            parts.append(token.text)
            position += len(token.text)
            ranges.append((start, position))
            line_characters += len(token.text)
            previous_end = token.end_sample
        self._token_document_ranges = ranges
        with QSignalBlocker(self.transcript_edit):
            self.transcript_edit.setPlainText("".join(parts))
        self._refresh_transcript_formats()

    def _selected_token_range(self) -> tuple[int, int] | None:
        cursor = self.transcript_edit.textCursor()
        start, end = cursor.selectionStart(), cursor.selectionEnd()
        if end <= start:
            return None
        indexes = [
            index
            for index, (left, right) in enumerate(self._token_document_ranges)
            if right > start and left < end
        ]
        return (indexes[0], indexes[-1] + 1) if indexes else None

    def _tokens_for_waveform_selection(self) -> tuple[int, int] | None:
        if self.project is None:
            return None
        start, end = self.waveform.selection
        indexes = [
            index
            for index, token in enumerate(self.project.tokens)
            if token.end_sample > start and token.start_sample < end
        ]
        return (indexes[0], indexes[-1] + 1) if indexes else None

    def _text_selection_changed(self) -> None:
        if self._updating or self.project is None:
            return
        token_range = self._selected_token_range()
        self._editing_annotation_id = None
        self.annotation_table.clearSelection()
        if token_range is None:
            self.selection_label.setText("拖选文字后建立标注")
            self._refresh_controls()
            return
        left, right = token_range
        start = self.project.tokens[left].start_sample
        end = self.project.tokens[right - 1].end_sample
        self.selection_label.setText(
            f"已选择 {right - left} 字 · {_format_sample(start, self.project.audio_info.sample_rate)}"
            f" – {_format_sample(end, self.project.audio_info.sample_rate)}"
        )
        if self.waveform_envelope is not None:
            self.waveform.set_selection(start, end)
        self._set_spin_samples(start, end)
        self._refresh_controls()

    def _select_tokens(self, start: int, end: int) -> None:
        if not (0 <= start < end <= len(self._token_document_ranges)):
            return
        cursor = self.transcript_edit.textCursor()
        cursor.setPosition(self._token_document_ranges[start][0])
        cursor.setPosition(self._token_document_ranges[end - 1][1], QTextCursor.MoveMode.KeepAnchor)
        self._updating = True
        try:
            self.transcript_edit.setTextCursor(cursor)
            self.transcript_edit.ensureCursorVisible()
        finally:
            self._updating = False

    def _add_delete_annotation(self) -> None:
        if self.project is None:
            return
        token_range = self._selected_token_range() or self._tokens_for_waveform_selection()
        if token_range is None:
            QMessageBox.information(self, "没有选择范围", "请先拖选转写文字或在波形上框选一段语音。")
            return
        annotation = self.project.add_delete_annotation(*token_range)
        waveform_start, waveform_end = self.waveform.selection
        if self._selected_token_range() is None and waveform_end > waveform_start:
            annotation.start_sample = waveform_start
            annotation.end_sample = waveform_end
        self._populate_annotations(select_id=annotation.id)
        self._refresh_transcript_formats()
        self._schedule_autosave()

    def _remove_selected_annotation(self) -> None:
        if self.project is None or self._editing_annotation_id is None:
            return
        if self.project.remove_annotation(self._editing_annotation_id):
            self._editing_annotation_id = None
            self._populate_annotations()
            self._refresh_transcript_formats()
            self._schedule_autosave()

    def _populate_annotations(self, *, select_id: str | None = None) -> None:
        self.annotation_table.setRowCount(0)
        if self.project is None:
            return
        self.annotation_table.setRowCount(len(self.project.annotations))
        selected_row = -1
        for row, annotation in enumerate(self.project.annotations):
            text_item = QTableWidgetItem(annotation.text)
            text_item.setData(Qt.ItemDataRole.UserRole, annotation.id)
            range_text = (
                f"{_format_sample(annotation.start_sample, self.project.audio_info.sample_rate)} – "
                f"{_format_sample(annotation.end_sample, self.project.audio_info.sample_rate)}"
            )
            self.annotation_table.setItem(row, 0, text_item)
            self.annotation_table.setItem(row, 1, QTableWidgetItem(range_text))
            operation = QTableWidgetItem("删除")
            operation.setForeground(QColor("#b85d20"))
            self.annotation_table.setItem(row, 2, operation)
            if annotation.id == select_id:
                selected_row = row
        self.annotation_summary.setText(f"{len(self.project.annotations)} 项")
        if selected_row >= 0:
            self.annotation_table.selectRow(selected_row)
        self.summary_label.setText(
            f"{len(self.project.tokens)} 个带时间戳文字 · {len(self.project.annotations)} 个删除标注"
        )
        self._refresh_controls()

    def _annotation_row_changed(self, row: int, _column: int, _old_row: int, _old_column: int) -> None:
        annotation = self._annotation_for_row(row)
        self._editing_annotation_id = annotation.id if annotation else None
        if annotation is None:
            self._refresh_controls()
            return
        self._select_tokens(annotation.token_start, annotation.token_end)
        if self.waveform_envelope is not None:
            self.waveform.set_selection(annotation.start_sample, annotation.end_sample)
        self._set_spin_samples(annotation.start_sample, annotation.end_sample)
        self._refresh_controls()

    def _annotation_for_row(self, row: int) -> AudioAnnotation | None:
        if self.project is None or row < 0:
            return None
        item = self.annotation_table.item(row, 0)
        annotation_id = str(item.data(Qt.ItemDataRole.UserRole)) if item else ""
        return next((item for item in self.project.annotations if item.id == annotation_id), None)

    def _editing_annotation(self) -> AudioAnnotation | None:
        if self.project is None or self._editing_annotation_id is None:
            return None
        return next(
            (item for item in self.project.annotations if item.id == self._editing_annotation_id),
            None,
        )

    def _waveform_selection_changed(self, start: int, end: int) -> None:
        if self._updating or self.project is None:
            return
        self._set_spin_samples(start, end)
        annotation = self._editing_annotation()
        if annotation is not None:
            annotation.start_sample = start
            annotation.end_sample = end
            self._populate_annotations(select_id=annotation.id)

    def _waveform_drag_finished(self, _start: int, _end: int) -> None:
        if self._editing_annotation() is not None:
            self._schedule_autosave()

    def _set_spin_samples(self, start: int, end: int) -> None:
        if self.project is None:
            return
        rate = self.project.audio_info.sample_rate
        self._updating = True
        try:
            with QSignalBlocker(self.start_spin), QSignalBlocker(self.end_spin):
                self.start_spin.setMaximum(self.project.audio_info.total_samples * 1000 / rate)
                self.end_spin.setMaximum(self.project.audio_info.total_samples * 1000 / rate)
                self.start_spin.setValue(start * 1000 / rate)
                self.end_spin.setValue(end * 1000 / rate)
        finally:
            self._updating = False

    def _spin_selection_changed(self) -> None:
        if self._updating or self.project is None:
            return
        rate = self.project.audio_info.sample_rate
        start = round(self.start_spin.value() * rate / 1000)
        end = round(self.end_spin.value() * rate / 1000)
        if end <= start:
            return
        self.waveform.set_selection(start, end, center=False)
        annotation = self._editing_annotation()
        if annotation is not None:
            annotation.start_sample = start
            annotation.end_sample = end
            self._populate_annotations(select_id=annotation.id)
            self._schedule_autosave()

    def _show_transcript_menu(self, position: QPoint) -> None:
        menu = QMenu(self.transcript_edit)
        add_action = QAction("标注为删除", menu)
        add_action.setShortcut(QKeySequence("Ctrl+B"))
        add_action.triggered.connect(self._add_delete_annotation)
        preview_action = QAction("试听选中文字", menu)
        preview_action.triggered.connect(lambda: self._start_preview("selection"))
        remove_action = QAction("取消所选删除标注", menu)
        remove_action.setEnabled(self._editing_annotation() is not None)
        remove_action.triggered.connect(self._remove_selected_annotation)
        menu.addAction(add_action)
        menu.addAction(preview_action)
        menu.addSeparator()
        menu.addAction(remove_action)
        menu.addSeparator()
        copy_action = menu.addAction("复制文字")
        copy_action.triggered.connect(self.transcript_edit.copy)
        menu.exec(self.transcript_edit.mapToGlobal(position))

    def _refresh_transcript_formats(self) -> None:
        if self.project is None:
            self.transcript_edit.setExtraSelections([])
            return
        selections: list[QTextEdit.ExtraSelection] = []
        for annotation in self.project.annotations:
            if not (0 <= annotation.token_start < annotation.token_end <= len(self._token_document_ranges)):
                continue
            selection = QTextEdit.ExtraSelection()
            cursor = self.transcript_edit.textCursor()
            cursor.setPosition(self._token_document_ranges[annotation.token_start][0])
            cursor.setPosition(
                self._token_document_ranges[annotation.token_end - 1][1],
                QTextCursor.MoveMode.KeepAnchor,
            )
            selection.cursor = cursor
            formatting = QTextCharFormat()
            formatting.setBackground(QBrush(QColor(244, 184, 127, 115)))
            formatting.setForeground(QBrush(QColor("#9b4b18")))
            formatting.setFontStrikeOut(True)
            selection.format = formatting
            selections.append(selection)
        self.transcript_edit.setExtraSelections(selections)

    def _start_preview(self, mode: str) -> None:
        if self.project is None or self.tools is None or not self._preview_directory.isValid():
            return
        token_range = self._selected_token_range()
        if token_range is not None and self.waveform_envelope is None:
            left, right = token_range
            start = self.project.tokens[left].start_sample
            end = self.project.tokens[right - 1].end_sample
        else:
            start, end = self.waveform.selection
        if end <= start:
            QMessageBox.information(self, "没有试听范围", "请先选择文字或在波形上框选范围。")
            return
        self._player.stop()
        self._start_task(
            make_audio_processing_preview_operation(
                self.project,
                start_sample=start,
                end_sample=end,
                tools=self.tools,
                preview_directory=self._preview_directory.path(),
            ),
            lambda value: self._preview_ready(value, mode),
        )

    def _preview_ready(self, value: object, mode: str) -> None:
        assert isinstance(value, CandidatePreviewResult)
        path = value.edited_wav_path if mode == "edited" else value.selection_wav_path
        try:
            self._player.play(path)
        except AudioPlaybackError as exc:
            self._playback_failed(str(exc))

    def _playback_failed(self, message: str) -> None:
        QMessageBox.warning(self, "无法试听", message or "音频设备播放失败，请检查系统输出设备。")

    def _start_export(self) -> None:
        if self.project is None or self.project_path is None or self.tools is None:
            return
        self.project.output_directory = self.output_edit.text().strip()
        self._start_task(
            make_audio_processing_export_operation(
                self.project,
                self.project_path,
                tools=self.tools,
            ),
            self._export_ready,
        )

    def _export_ready(self, value: object) -> None:
        assert isinstance(value, AudioProcessingExportResult)
        QMessageBox.information(
            self,
            "导出完成",
            f"已生成：\n{value.wav_path.name}\n{value.mp3_path.name}\n\n"
            f"共删除 {value.removed_samples} 个 PCM 样本。",
        )

    def _output_changed(self, value: str) -> None:
        if self.project is not None:
            self.project.output_directory = value
            self._schedule_autosave()

    def _schedule_autosave(self) -> None:
        if self.project is not None and self.project_path is not None:
            self._autosave_timer.start()

    def _autosave(self) -> None:
        if self.project is None or self.project_path is None:
            return
        with suppress(Exception):
            save_audio_processing_project(self.project, self.project_path)

    def _start_task(self, operation, callback) -> None:  # type: ignore[no-untyped-def]
        if self._active_task is not None:
            return
        task = BackgroundTask(operation)
        self._active_task = task
        self.progress_container.setVisible(True)
        self.progress_bar.setValue(0)
        task.signals.progress.connect(self._task_progress)
        task.signals.result.connect(callback)
        task.signals.error.connect(lambda message: QMessageBox.warning(self, "操作失败", message))
        task.signals.cancelled.connect(
            lambda: self.summary_label.setText("操作已取消，可重新开始")
        )
        task.signals.finished.connect(self._task_finished)
        self.thread_pool.start(task)
        self._refresh_controls()

    def _task_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def _task_finished(self) -> None:
        self._active_task = None
        self.progress_container.setVisible(False)
        self._refresh_controls()

    def _cancel_task(self) -> None:
        if self._active_task is not None:
            self._active_task.cancel()

    def _waveform_view_changed(self, start: int, end: int) -> None:
        if self.waveform.total_samples <= 0:
            self.waveform_scrollbar.setRange(0, 0)
            return
        span = max(1, end - start)
        available = max(0, self.waveform.total_samples - span)
        with QSignalBlocker(self.waveform_scrollbar):
            self.waveform_scrollbar.setRange(0, _SCROLL_STEPS if available else 0)
            self.waveform_scrollbar.setValue(
                round(start / available * _SCROLL_STEPS) if available else 0
            )
            self.waveform_scrollbar.setPageStep(max(1, round(span / self.waveform.total_samples * _SCROLL_STEPS)))

    def _waveform_scroll_changed(self, value: int) -> None:
        if self.waveform_scrollbar.maximum() > 0:
            self.waveform.set_scroll_position(value / self.waveform_scrollbar.maximum())

    def _refresh_controls(self) -> None:
        busy = self._active_task is not None
        ready = self.project is not None
        has_selection = self._selected_token_range() is not None if ready else False
        waveform_ready = self.waveform_envelope is not None
        self.audio_browse_button.setEnabled(not busy)
        self.open_project_button.setEnabled(not busy)
        self.transcribe_button.setEnabled(not busy and bool(self.audio_path_edit.text().strip()))
        self.add_delete_button.setEnabled(not busy and ready and (has_selection or waveform_ready))
        self.preview_text_button.setEnabled(not busy and ready and has_selection)
        self.preview_selection_button.setEnabled(not busy and waveform_ready)
        self.preview_edited_button.setEnabled(not busy and waveform_ready)
        self.stop_button.setEnabled(self._player.is_active)
        self.remove_annotation_button.setEnabled(not busy and self._editing_annotation() is not None)
        self.start_spin.setEnabled(not busy and waveform_ready)
        self.end_spin.setEnabled(not busy and waveform_ready)
        self.export_button.setEnabled(not busy and ready and bool(self.project.annotations))

    def shutdown(self) -> None:
        self._player.stop()
        if self._active_task is not None:
            self._active_task.cancel()
        self._autosave()


def _time_spin(name: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setObjectName(name)
    spin.setDecimals(3)
    spin.setRange(0.0, 99_999_999.0)
    spin.setSingleStep(20.0)
    spin.setSuffix(" ms")
    return spin


def _format_sample(sample: int, sample_rate: int) -> str:
    milliseconds = max(0, sample) * 1000 / max(1, sample_rate)
    minutes, remainder = divmod(milliseconds, 60_000)
    return f"{int(minutes):02d}:{remainder / 1000:06.3f}"


__all__ = ["AudioProcessingWidget"]
