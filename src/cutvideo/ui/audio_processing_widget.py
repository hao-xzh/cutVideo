"""Standalone full-transcript audio annotation and editing workspace."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from PySide6.QtCore import (
    QDir,
    QPoint,
    QSignalBlocker,
    Qt,
    QTemporaryDir,
    QThreadPool,
    QTimer,
    Signal,
)
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
    QAbstractButton,
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
    QPushButton,
    QScrollBar,
    QSplitter,
    QStackedWidget,
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
    make_audio_processing_relink_operation,
)
from .file_drop_edit import FileDropLineEdit
from .icons import ButtonSpinner, make_icon_button, set_button_icon
from .import_view import ImportLandingView
from .progress_view import TaskProgressView
from .settings import dialog_start, remember_dialog_path
from .theme import theme_color
from .waveform import WaveformWidget
from .workers import BackgroundTask, CandidatePreviewResult, make_waveform_operation

_SCROLL_STEPS = 100_000
_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
_REVIEW_REASON_LABELS = {
    "coarse_timestamp": "仅有词/句级时间",
    "model_confidence_unavailable": "模型未提供逐字置信度",
    "low_model_confidence": "局部识别置信度较低",
    "legacy_boundary_precision_unknown": "旧项目未保存时间戳精度",
    "boundary_precision_unverified": "边界精度尚未核验",
}


class AudioProcessingWidget(QWidget):
    """Full audio transcription, region annotations, preview and export."""

    statusMessage = Signal(str)

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
        self._task_uses_progress = True
        self._preview_spinner = ButtonSpinner(self)
        self._token_document_ranges: list[tuple[int, int]] = []
        self._segment_label_ranges: list[tuple[int, int]] = []
        self._editing_annotation_id: str | None = None
        self._updating = False
        self._setting_audio_path = False
        self._shortcuts: list[QShortcut] = []
        self._undo_history: list[list[dict[str, object]]] = []
        self._redo_history: list[list[dict[str, object]]] = []
        self._annotation_edit_baseline: list[dict[str, object]] | None = None
        self._annotation_edit_active = False
        self._foreground_generation = 0
        self._source_generation = 0
        template = str(Path(QDir.tempPath()) / "cutvideo-audio-processing-XXXXXX")
        self._preview_directory = QTemporaryDir(template)
        self._player = PcmWavPlayer(self)
        self._player.failed.connect(self._playback_failed)
        self._player.finished.connect(lambda: self.statusMessage.emit("试听完成"))
        self._player.active_changed.connect(lambda _active: self._refresh_controls())
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(450)
        self._autosave_timer.timeout.connect(self._autosave)
        self._build_ui()
        self._connect_signals()
        self._refresh_controls()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.workspace_stack = QStackedWidget()
        self.workspace_stack.setObjectName("processingWorkspaceStack")
        self.import_view = ImportLandingView("audio")
        self.workspace_stack.addWidget(self.import_view)
        editor = QWidget()
        editor.setObjectName("processingEditor")
        self.workspace_stack.addWidget(editor)
        outer.addWidget(self.workspace_stack)

        root = QVBoxLayout(editor)
        root.setContentsMargins(16, 8, 16, 12)
        root.setSpacing(10)

        input_card = QFrame()
        input_card.setObjectName("inputCard")
        input_layout = QGridLayout(input_card)
        input_layout.setContentsMargins(12, 9, 12, 9)
        input_layout.setHorizontalSpacing(10)
        input_layout.addWidget(QLabel("音频"), 0, 0)
        self.audio_path_edit = FileDropLineEdit((".mp3", ".wav", ".m4a", ".aac", ".flac"))
        self.audio_path_edit.setObjectName("processingAudioPathEdit")
        self.audio_path_edit.setPlaceholderText("拖入单独音频，或点击右侧选择文件")
        self.audio_browse_button = make_icon_button(
            "folder-open",
            "选择需要处理的音频",
        )
        self.audio_browse_button.setObjectName("processingAudioBrowseButton")
        self.open_project_button = QPushButton("打开项目…")
        self.open_project_button.setObjectName("processingOpenProjectButton")
        set_button_icon(self.open_project_button, "project")
        self.transcribe_button = QPushButton("完整识别")
        self.transcribe_button.setObjectName("processingTranscribeButton")
        self.transcribe_button.setProperty("primary", True)
        set_button_icon(self.transcribe_button, "audio-primary")
        input_layout.addWidget(self.audio_path_edit, 0, 1)
        input_layout.addWidget(self.audio_browse_button, 0, 2)
        input_layout.addWidget(self.open_project_button, 0, 3)
        input_layout.addWidget(self.transcribe_button, 0, 4)
        self.summary_label = QLabel("选择音频后执行完整识别")
        self.summary_label.setObjectName("mutedLabel")
        input_layout.addWidget(self.summary_label, 1, 1, 1, 4)
        input_layout.setColumnStretch(1, 1)
        root.addWidget(input_card)

        self.task_progress = TaskProgressView()
        self.progress_container = self.task_progress
        self.progress_label = self.task_progress.message_label
        self.progress_bar = self.task_progress.progress_bar
        self.cancel_button = self.task_progress.cancel_button
        root.addWidget(self.task_progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("audioProcessingSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        transcript_card = QFrame()
        transcript_card.setObjectName("reviewCard")
        transcript_layout = QVBoxLayout(transcript_card)
        transcript_layout.setContentsMargins(10, 9, 10, 10)
        transcript_heading = QHBoxLayout()
        transcript_title = QLabel("完整转写")
        transcript_title.setObjectName("sectionTitle")
        self.selection_label = QLabel("拖选文字后建立标注")
        self.selection_label.setObjectName("mutedLabel")
        self.transcript_search = QLineEdit()
        self.transcript_search.setObjectName("processingTranscriptSearch")
        self.transcript_search.setPlaceholderText("搜索转写")
        self.transcript_search.setMaximumWidth(170)
        self.transcript_search_button = make_icon_button("search", "查找下一处转写文字")
        transcript_heading.addWidget(transcript_title)
        transcript_heading.addStretch(1)
        transcript_heading.addWidget(self.selection_label)
        transcript_heading.addWidget(self.transcript_search)
        transcript_heading.addWidget(self.transcript_search_button)
        transcript_layout.addLayout(transcript_heading)
        self.transcript_edit = QTextEdit()
        self.transcript_edit.setObjectName("processingTranscriptEdit")
        self.transcript_edit.setReadOnly(True)
        self.transcript_edit.setAcceptRichText(False)
        self.transcript_edit.setPlaceholderText("完整识别完成后，文字会在这里按语音停顿分段显示")
        self.transcript_edit.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        transcript_layout.addWidget(self.transcript_edit, 1)
        self.add_delete_button = QPushButton("标记删除")
        self.add_delete_button.setObjectName("processingAddDeleteButton")
        self.add_delete_button.setProperty("primary", True)
        set_button_icon(self.add_delete_button, "delete")
        self.add_delete_button.setToolTip("将当前文字或波形范围标记为删除")
        self.preview_text_button = QPushButton("试听文字")
        self.preview_text_button.setObjectName("processingPreviewTextButton")
        set_button_icon(self.preview_text_button, "play")
        self.preview_text_button.setToolTip("试听当前选中文字对应的音频")
        splitter.addWidget(transcript_card)

        detail_card = QFrame()
        detail_card.setObjectName("reviewCard")
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(10, 9, 10, 10)
        detail_layout.setSpacing(8)
        detail_title = QLabel("范围与删除标注")
        detail_title.setObjectName("sectionTitle")
        detail_layout.addWidget(detail_title)
        waveform_hint = QLabel(
            "文字决定标注内容，橙色时间范围决定实际删除音频；拖边界精调，Alt+拖动平移"
        )
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
        self.zoom_out_button = make_icon_button("zoom-out", "缩小波形视野")
        self.zoom_in_button = make_icon_button("zoom-in", "放大波形视野")
        self.focus_button = make_icon_button("focus", "定位当前选择范围")
        view_layout.addWidget(self.waveform_scrollbar, 1)
        view_layout.addWidget(self.zoom_out_button)
        view_layout.addWidget(self.zoom_in_button)
        view_layout.addWidget(self.focus_button)
        detail_layout.addLayout(view_layout)

        boundary_layout = QGridLayout()
        self.start_spin = _time_spin("processingStartSpin")
        self.end_spin = _time_spin("processingEndSpin")
        boundary_layout.addWidget(QLabel("删除 / 试听开始"), 0, 0)
        boundary_layout.addWidget(QLabel("删除 / 试听结束"), 0, 1)
        boundary_layout.addWidget(self.start_spin, 1, 0)
        boundary_layout.addWidget(self.end_spin, 1, 1)
        detail_layout.addLayout(boundary_layout)

        self.preview_original_button = QPushButton("原音 ±3s")
        set_button_icon(self.preview_original_button, "play")
        self.preview_original_button.setToolTip("试听删除范围前后各 3 秒的原始音频")
        self.preview_selection_button = QPushButton("删除段")
        set_button_icon(self.preview_selection_button, "play")
        self.preview_selection_button.setToolTip("只试听将要删除的音频范围")
        self.preview_edited_button = QPushButton("删除后")
        set_button_icon(self.preview_edited_button, "play")
        self.preview_edited_button.setToolTip("试听移除当前删除范围后的衔接效果")
        self.stop_button = make_icon_button("stop", "停止试听", size=30)

        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)
        action_layout.addWidget(self.add_delete_button)
        action_layout.addWidget(self.preview_text_button)
        action_layout.addStretch(1)
        action_layout.addWidget(self.preview_original_button)
        action_layout.addWidget(self.preview_selection_button)
        action_layout.addWidget(self.preview_edited_button)
        action_layout.addWidget(self.stop_button)
        detail_layout.addLayout(action_layout)

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
        self.annotation_table.setHorizontalHeaderLabels(["文字", "范围", "状态"])
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
        self.confirm_boundary_button = QPushButton("确认边界")
        self.confirm_boundary_button.setObjectName("processingConfirmBoundaryButton")
        set_button_icon(self.confirm_boundary_button, "check")
        self.remove_annotation_button = QPushButton("取消所选标注")
        self.remove_annotation_button.setObjectName("processingRemoveAnnotationButton")
        annotation_actions.addStretch(1)
        annotation_actions.addWidget(self.confirm_boundary_button)
        annotation_actions.addWidget(self.remove_annotation_button)
        detail_layout.addLayout(annotation_actions)
        splitter.addWidget(detail_card)
        splitter.setSizes([520, 800])
        root.addWidget(splitter, 1)

        export_card = QFrame()
        export_card.setObjectName("exportCard")
        export_layout = QHBoxLayout(export_card)
        export_layout.setContentsMargins(12, 8, 12, 8)
        export_title = QLabel("导出位置")
        export_title.setObjectName("sectionTitle")
        self.output_edit = QLineEdit()
        self.output_edit.setObjectName("processingOutputPathEdit")
        self.output_browse_button = make_icon_button(
            "folder-open",
            "选择导出目录",
        )
        self.export_button = QPushButton("导出处理完成音频")
        self.export_button.setObjectName("processingExportButton")
        self.export_button.setProperty("primary", True)
        set_button_icon(self.export_button, "export")
        export_layout.addWidget(export_title)
        export_layout.addWidget(self.output_edit, 1)
        export_layout.addWidget(self.output_browse_button)
        export_layout.addWidget(self.export_button)
        root.addWidget(export_card)

    def _connect_signals(self) -> None:
        self.import_view.audioBrowseRequested.connect(self._choose_audio)
        self.import_view.projectBrowseRequested.connect(self._choose_project)
        self.import_view.audioDropped.connect(self._set_imported_audio)
        self.audio_browse_button.clicked.connect(self._choose_audio)
        self.audio_path_edit.textChanged.connect(self._audio_path_changed)
        self.audio_path_edit.fileDropped.connect(
            lambda path: self.statusMessage.emit(f"已拖入音频：{Path(path).name}，可以开始完整识别")
        )
        self.open_project_button.clicked.connect(self._choose_project)
        self.transcribe_button.clicked.connect(self._start_transcription)
        self.task_progress.cancelRequested.connect(self._cancel_task)
        self.transcript_edit.selectionChanged.connect(self._text_selection_changed)
        self.transcript_edit.customContextMenuRequested.connect(self._show_transcript_menu)
        self.transcript_search.returnPressed.connect(self._find_next_transcript_match)
        self.transcript_search_button.clicked.connect(self._find_next_transcript_match)
        self.add_delete_button.clicked.connect(self._add_delete_annotation)
        self.preview_text_button.clicked.connect(
            lambda: self._start_preview("selection", self.preview_text_button)
        )
        self.annotation_table.currentCellChanged.connect(self._annotation_row_changed)
        self.confirm_boundary_button.clicked.connect(self._confirm_selected_boundary)
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
        self.start_spin.editingFinished.connect(self._annotation_edit_finished)
        self.end_spin.editingFinished.connect(self._annotation_edit_finished)
        self.preview_original_button.clicked.connect(
            lambda: self._start_preview("original", self.preview_original_button)
        )
        self.preview_selection_button.clicked.connect(
            lambda: self._start_preview("selection", self.preview_selection_button)
        )
        self.preview_edited_button.clicked.connect(
            lambda: self._start_preview("edited", self.preview_edited_button)
        )
        self.stop_button.clicked.connect(self._stop_playback)
        self.output_browse_button.clicked.connect(self._choose_output)
        self.output_edit.textChanged.connect(self._output_changed)
        self.export_button.clicked.connect(self._start_export)
        for sequence in ("Delete", "Backspace", "Ctrl+B", "Meta+B"):
            shortcut = QShortcut(QKeySequence(sequence), self.transcript_edit)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(self._add_delete_annotation)
            self._shortcuts.append(shortcut)
        for sequence, callback in (
            ("Ctrl+F", lambda: self.transcript_search.setFocus()),
            ("Meta+F", lambda: self.transcript_search.setFocus()),
            ("Ctrl+Z", self._undo_annotation_edit),
            ("Meta+Z", self._undo_annotation_edit),
            ("Ctrl+Shift+Z", self._redo_annotation_edit),
            ("Meta+Shift+Z", self._redo_annotation_edit),
        ):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    def _find_next_transcript_match(self) -> None:
        query = self.transcript_search.text().strip()
        if not query or self.project is None:
            return
        if not self.transcript_edit.find(query):
            cursor = self.transcript_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            self.transcript_edit.setTextCursor(cursor)
            if not self.transcript_edit.find(query):
                self.statusMessage.emit(f"未找到：{query}")
                return
        self.transcript_edit.setFocus()
        self.statusMessage.emit(f"已定位：{query}")

    def _set_imported_audio(self, path: str) -> None:
        remember_dialog_path("processing_audio", path)
        self.audio_path_edit.setText(path)
        self.statusMessage.emit(f"已选择音频：{Path(path).name}，可以开始完整识别")

    def _audio_path_changed(self, value: str) -> None:
        if self._setting_audio_path:
            return
        self._foreground_generation += 1
        self._source_generation += 1
        if self._active_task is not None:
            self._active_task.cancel()
        self._player.stop()
        current_source = ""
        if self.project is not None:
            current_source = self.project.audio.path
        elif self.audio_info is not None:
            current_source = str(self.audio_info.path)
        if current_source and Path(value).expanduser() != Path(current_source).expanduser():
            self._clear_project_for_new_source()
        self.import_view.set_paths(value)
        self._sync_workspace_page()
        self._refresh_controls()

    def _clear_project_for_new_source(self) -> None:
        self._player.stop()
        self.project = None
        self.project_path = None
        self.audio_info = None
        self.tools = None
        self.waveform_envelope = None
        self._token_document_ranges.clear()
        self._segment_label_ranges.clear()
        self._editing_annotation_id = None
        self._undo_history.clear()
        self._redo_history.clear()
        self._annotation_edit_baseline = None
        self._annotation_edit_active = False
        self.transcript_edit.clear()
        self.annotation_table.setRowCount(0)
        self.waveform.clear()
        self.output_edit.clear()
        self.summary_label.setText("选择音频后执行完整识别")
        self.selection_label.setText("拖选文字后建立标注")
        self.annotation_summary.setText("0 项")
        with QSignalBlocker(self.start_spin), QSignalBlocker(self.end_spin):
            self.start_spin.setValue(0.0)
            self.end_spin.setValue(0.0)

    def _sync_workspace_page(self, *, force_editor: bool = False) -> None:
        path = self.audio_path_edit.text().strip()
        self.import_view.set_paths(path)
        source = Path(path).expanduser()
        supported = source.is_file() and source.suffix.casefold() in _AUDIO_SUFFIXES
        show_editor = force_editor or self.project is not None or supported
        self.workspace_stack.setCurrentIndex(1 if show_editor else 0)

    def _choose_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择需要处理的音频",
            dialog_start("processing_audio", self.audio_path_edit.text()),
            "音频文件 (*.mp3 *.wav *.m4a *.aac *.flac);;所有文件 (*)",
        )
        if path:
            self._set_imported_audio(path)

    def _choose_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开音频处理项目",
            dialog_start("processing_project", self.audio_path_edit.text()),
            "音频处理项目 (*.audioprocess.json);;JSON 文件 (*.json)",
        )
        if path:
            remember_dialog_path("processing_project", path)
            self._sync_workspace_page(force_editor=True)
            self._start_task(make_audio_processing_load_operation(path), self._analysis_ready)

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择导出目录",
            dialog_start("processing_output", self.output_edit.text()),
        )
        if path:
            remember_dialog_path("processing_output", path)
            self.output_edit.setText(path)

    def _start_transcription(self) -> None:
        path = self.audio_path_edit.text().strip()
        source = Path(path).expanduser()
        if not source.is_file() or source.suffix.casefold() not in _AUDIO_SUFFIXES:
            QMessageBox.information(
                self,
                "请选择音频",
                "请选择一个存在且格式受支持的音频文件。",
            )
            return
        self._start_task(make_audio_processing_analysis_operation(path), self._analysis_ready)
        self.statusMessage.emit("正在完整识别音频，所有处理均在本机完成…")

    def _analysis_ready(self, value: object) -> None:
        assert isinstance(value, AudioProcessingAnalysisResult)
        if not value.source_matches:
            QMessageBox.information(
                self,
                "重新关联原音频",
                "项目记录的音频已移动或内容发生变化。请选择同一份原音频，程序会按 SHA-256 指纹核对。",
            )
            replacement, _ = QFileDialog.getOpenFileName(
                self,
                "重新选择原音频",
                str(value.project_path.parent),
                "音频文件 (*.mp3 *.wav *.m4a *.aac *.flac);;所有文件 (*)",
            )
            if replacement:
                QTimer.singleShot(
                    0,
                    lambda: self._start_task(
                        make_audio_processing_relink_operation(
                            value.project_path,
                            replacement,
                        ),
                        self._analysis_ready,
                    ),
                )
            else:
                self.statusMessage.emit("打开项目已取消：需要重新关联原音频")
            return
        self._source_generation += 1
        source_generation = self._source_generation
        self.project = value.project
        self._undo_history.clear()
        self._redo_history.clear()
        self._annotation_edit_active = False
        self.project_path = value.project_path
        self.audio_info = value.audio_info
        self.tools = value.tools
        self._setting_audio_path = True
        try:
            self.audio_path_edit.setText(value.project.audio.path)
        finally:
            self._setting_audio_path = False
        self.import_view.set_paths(value.project.audio.path)
        self._sync_workspace_page(force_editor=True)
        self.output_edit.setText(value.project.output_directory)
        self._render_transcript()
        self._populate_annotations()
        duration = _format_sample(value.audio_info.total_samples, value.audio_info.sample_rate)
        self.summary_label.setText(
            f"{len(value.project.tokens)} 个带时间戳文字 · "
            f"{len(value.project.segment_starts)} 个人声段 · 音频 {duration} · "
            f"{len(value.project.annotations)} 个删除标注"
        )
        self.statusMessage.emit(
            f"完整识别完成：{len(value.project.tokens)} 个时间戳文字，可选择文字建立标注"
        )
        self.waveform.clear()
        expected_path = str(value.audio_info.path)
        if value.waveform_envelope is not None:
            self._waveform_ready(
                value.waveform_envelope,
                expected_path=expected_path,
                source_generation=source_generation,
            )
        else:
            self.waveform.set_placeholder("正在后台生成完整波形…")

            def waveform_ready(envelope: object) -> None:
                self._waveform_ready(
                    envelope,
                    expected_path=expected_path,
                    source_generation=source_generation,
                )

            QTimer.singleShot(
                0,
                lambda: self._start_task(
                    make_waveform_operation(value.audio_info, value.tools),
                    waveform_ready,
                ),
            )

    def _waveform_ready(
        self,
        value: object,
        *,
        expected_path: str | None = None,
        source_generation: int | None = None,
    ) -> None:
        assert isinstance(value, WaveformEnvelope)
        if (
            self.audio_info is None
            or (source_generation is not None and source_generation != self._source_generation)
            or (
                expected_path is not None
                and str(self.audio_info.path) != expected_path
            )
        ):
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
        segment_label_ranges: list[tuple[int, int]] = []
        segment_starts = set(self.project.segment_starts)
        for index, token in enumerate(self.project.tokens):
            if index in segment_starts:
                if index:
                    parts.append("\n")
                    position += 1
                label = f"[{_format_sample(token.start_sample, self.project.audio_info.sample_rate)}] "
                label_start = position
                parts.append(label)
                position += len(label)
                segment_label_ranges.append((label_start, position))
            start = position
            parts.append(token.text)
            position += len(token.text)
            ranges.append((start, position))
        self._token_document_ranges = ranges
        self._segment_label_ranges = segment_label_ranges
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
        selected_tokens = self.project.tokens[left:right]
        start = selected_tokens[0].start_sample
        end = selected_tokens[-1].end_sample
        precision_uncertain = any(
            token.timestamp_precision != "character" or not token.confidence_available
            for token in selected_tokens
        )
        precision_notice = " · 边界需波形确认"
        self.selection_label.setText(
            f"已选择 {right - left} 字 · {_format_sample(start, self.project.audio_info.sample_rate)}"
            f" – {_format_sample(end, self.project.audio_info.sample_rate)}{precision_notice}"
        )
        if precision_uncertain:
            self.statusMessage.emit(
                f"已选择 {right - left} 个文字；模型没有可靠逐字切点，标注后需在波形中确认边界"
            )
        else:
            self.statusMessage.emit(
                f"已选择 {right - left} 个文字；逐字时间仅作初始建议，标注后需试听或确认边界"
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
        text_token_range = self._selected_token_range()
        token_range = text_token_range or self._tokens_for_waveform_selection()
        if token_range is None:
            QMessageBox.information(self, "没有选择范围", "请先拖选转写文字或在波形上框选一段语音。")
            return
        self._record_annotation_edit()
        waveform_start, waveform_end = self.waveform.selection
        previous_annotations = list(self.project.annotations)
        token_start, token_end = token_range
        recognized_start = self.project.tokens[token_start].start_sample
        recognized_end = self.project.tokens[token_end - 1].end_sample
        overlapping = [
            item
            for item in previous_annotations
            if item.start_sample <= recognized_end and item.end_sample >= recognized_start
        ]
        annotation = self.project.add_delete_annotation(*token_range)
        if self.waveform_envelope is not None and waveform_end > waveform_start:
            annotation.start_sample = min(
                waveform_start,
                *(item.start_sample for item in overlapping),
            ) if overlapping else waveform_start
            annotation.end_sample = max(
                waveform_end,
                *(item.end_sample for item in overlapping),
            ) if overlapping else waveform_end
        if text_token_range is None:
            self.project.mark_annotation_boundary_reviewed(
                annotation.id,
                source="manual_waveform_selection",
            )
        self._populate_annotations(select_id=annotation.id)
        self._refresh_transcript_formats()
        self._schedule_autosave()
        if annotation.review_required:
            self.statusMessage.emit(
                f"已建立删除标注：{annotation.text}；模型时间仅是初始建议，请试听并确认波形边界"
            )
        else:
            self.statusMessage.emit(
                f"已建立删除标注：{annotation.text}；可在右侧拖动边界并反复试听"
            )

    def _confirm_selected_boundary(self) -> None:
        if self.project is None:
            return
        annotation = self._editing_annotation()
        if annotation is None:
            return
        self._record_annotation_edit()
        if self.project.mark_annotation_boundary_reviewed(
            annotation.id,
            source="manual_confirmation",
        ):
            self._populate_annotations(select_id=annotation.id)
            self._reset_annotation_baseline()
            self._schedule_autosave()
            self.statusMessage.emit("已确认删除边界，可继续导出")

    def _remove_selected_annotation(self) -> None:
        if self.project is None or self._editing_annotation_id is None:
            return
        self._record_annotation_edit()
        if self.project.remove_annotation(self._editing_annotation_id):
            self._editing_annotation_id = None
            self._populate_annotations()
            self._refresh_transcript_formats()
            self._schedule_autosave()
            self.statusMessage.emit("已取消所选删除标注")

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
            state = QTableWidgetItem("需复核" if annotation.review_required else "已确认")
            state.setForeground(
                QColor(theme_color("accent" if annotation.review_required else "success"))
            )
            if annotation.review_required:
                reason_text = "；".join(
                    _REVIEW_REASON_LABELS.get(reason, reason)
                    for reason in annotation.review_reasons
                )
                state.setToolTip(f"{reason_text}。试听后拖动边界，或点击“确认边界”。")
            self.annotation_table.setItem(row, 2, state)
            if annotation.id == select_id:
                selected_row = row
        self.annotation_summary.setText(f"{len(self.project.annotations)} 项")
        if selected_row >= 0:
            self.annotation_table.selectRow(selected_row)
        removed_samples = sum(
            item.end_sample - item.start_sample for item in self.project.deletion_intervals
        )
        removed_duration = _format_sample(
            removed_samples,
            self.project.audio_info.sample_rate,
        )
        review_count = sum(item.review_required for item in self.project.annotations)
        review_text = f" · {review_count} 项待复核" if review_count else ""
        self.annotation_summary.setText(
            f"{len(self.project.annotations)} 项 · 删除 {removed_duration}{review_text}"
        )
        self.summary_label.setText(
            f"{len(self.project.tokens)} 个带时间戳文字 · "
            f"{len(self.project.segment_starts)} 个人声段 · "
            f"{len(self.project.annotations)} 个删除标注 · 删除 {removed_duration}"
            f"{review_text}"
        )
        self._refresh_controls()

    def _annotation_row_changed(self, row: int, _column: int, _old_row: int, _old_column: int) -> None:
        annotation = self._annotation_for_row(row)
        self._editing_annotation_id = annotation.id if annotation else None
        self._reset_annotation_baseline()
        if annotation is None:
            self._refresh_controls()
            return
        self._select_tokens(annotation.token_start, annotation.token_end)
        if self.waveform_envelope is not None:
            self.waveform.set_selection(annotation.start_sample, annotation.end_sample)
        self._set_spin_samples(annotation.start_sample, annotation.end_sample)
        if annotation.review_required:
            reason_text = "；".join(
                _REVIEW_REASON_LABELS.get(reason, reason)
                for reason in annotation.review_reasons
            )
            self.statusMessage.emit(f"此标注需复核：{reason_text}")
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

    def _annotation_snapshot(self) -> list[dict[str, object]]:
        if self.project is None:
            return []
        return [item.to_dict() for item in self.project.annotations]

    def _record_annotation_edit(self) -> None:
        if self.project is None:
            return
        if self._annotation_edit_active:
            return
        snapshot = self._annotation_edit_baseline or self._annotation_snapshot()
        self._undo_history.append(snapshot)
        self._undo_history = self._undo_history[-100:]
        self._redo_history.clear()
        self._annotation_edit_baseline = None
        self._annotation_edit_active = True

    def _reset_annotation_baseline(self) -> None:
        self._annotation_edit_baseline = self._annotation_snapshot() if self.project else None
        self._annotation_edit_active = False

    def _restore_annotation_snapshot(self, snapshot: list[dict[str, object]]) -> None:
        if self.project is None:
            return
        self.project.annotations = [AudioAnnotation.from_dict(item) for item in snapshot]
        self._editing_annotation_id = None
        self._populate_annotations()
        self._refresh_transcript_formats()
        self._schedule_autosave()
        self._reset_annotation_baseline()

    def _undo_annotation_edit(self) -> None:
        if self.project is None or not self._undo_history:
            return
        self._redo_history.append(self._annotation_snapshot())
        self._restore_annotation_snapshot(self._undo_history.pop())
        self.statusMessage.emit("已撤销上一步标注操作")

    def _redo_annotation_edit(self) -> None:
        if self.project is None or not self._redo_history:
            return
        self._undo_history.append(self._annotation_snapshot())
        self._restore_annotation_snapshot(self._redo_history.pop())
        self.statusMessage.emit("已重做标注操作")

    def _update_annotation_row(self, annotation: AudioAnnotation) -> None:
        if self.project is None:
            return
        for row in range(self.annotation_table.rowCount()):
            item = self.annotation_table.item(row, 0)
            if item is None or item.data(Qt.ItemDataRole.UserRole) != annotation.id:
                continue
            range_item = self.annotation_table.item(row, 1)
            if range_item is not None:
                range_item.setText(
                    f"{_format_sample(annotation.start_sample, self.project.audio_info.sample_rate)} – "
                    f"{_format_sample(annotation.end_sample, self.project.audio_info.sample_rate)}"
                )
            return

    def _waveform_selection_changed(self, start: int, end: int) -> None:
        if self._updating or self.project is None:
            return
        self._set_spin_samples(start, end)
        annotation = self._editing_annotation()
        if annotation is not None:
            self._record_annotation_edit()
            annotation.start_sample = start
            annotation.end_sample = end
            self._update_annotation_row(annotation)

    def _waveform_drag_finished(self, _start: int, _end: int) -> None:
        if self._editing_annotation() is not None:
            annotation = self._editing_annotation()
            assert annotation is not None
            assert self.project is not None
            self.project.mark_annotation_boundary_reviewed(
                annotation.id,
                source="manual_waveform_adjustment",
            )
            self._populate_annotations(select_id=annotation.id)
            self._reset_annotation_baseline()
            self._schedule_autosave()
            self.statusMessage.emit("删除标注边界已调整并自动保存")
        else:
            self.statusMessage.emit("删除 / 试听范围已更新")

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
            self._record_annotation_edit()
            annotation.start_sample = start
            annotation.end_sample = end
            self._update_annotation_row(annotation)
            self._schedule_autosave()

    def _annotation_edit_finished(self) -> None:
        annotation = self._editing_annotation()
        if annotation is not None:
            assert self.project is not None
            self.project.mark_annotation_boundary_reviewed(
                annotation.id,
                source="manual_numeric_adjustment",
            )
            self._populate_annotations(select_id=annotation.id)
            self._reset_annotation_baseline()
            self._schedule_autosave()

    def _show_transcript_menu(self, position: QPoint) -> None:
        menu = QMenu(self.transcript_edit)
        add_action = QAction("标注为删除", menu)
        add_action.setShortcut(QKeySequence("Ctrl+B"))
        add_action.triggered.connect(self._add_delete_annotation)
        preview_action = QAction("试听选中文字", menu)
        preview_action.triggered.connect(
            lambda: self._start_preview("selection", self.preview_text_button)
        )
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
        for start, end in self._segment_label_ranges:
            selection = QTextEdit.ExtraSelection()
            cursor = self.transcript_edit.textCursor()
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            selection.cursor = cursor
            formatting = QTextCharFormat()
            formatting.setForeground(QBrush(QColor(theme_color("muted"))))
            formatting.setFontFixedPitch(True)
            selection.format = formatting
            selections.append(selection)
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
            highlight_background = QColor(theme_color("accent"))
            highlight_background.setAlpha(64)
            formatting.setBackground(QBrush(highlight_background))
            formatting.setForeground(QBrush(QColor(theme_color("highlight_text"))))
            formatting.setFontStrikeOut(True)
            selection.format = formatting
            selections.append(selection)
        self.transcript_edit.setExtraSelections(selections)

    def refresh_theme(self) -> None:
        """Refresh colors painted outside the application stylesheet."""

        if self.project is not None:
            self._populate_annotations(select_id=self._editing_annotation_id)
            self._refresh_transcript_formats()
        self.waveform.update()

    def _start_preview(
        self,
        mode: str,
        trigger_button: QAbstractButton | None = None,
    ) -> None:
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
        self.statusMessage.emit("正在生成本地试听，不会上传音频…")
        if trigger_button is None:
            trigger_button = {
                "original": self.preview_original_button,
                "selection": self.preview_selection_button,
                "edited": self.preview_edited_button,
            }.get(mode)
        self._start_task(
            make_audio_processing_preview_operation(
                self.project,
                start_sample=start,
                end_sample=end,
                mode=mode,
                tools=self.tools,
                preview_directory=self._preview_directory.path(),
            ),
            lambda value: self._preview_ready(value, mode),
            show_progress=False,
            busy_button=trigger_button,
        )

    def _preview_ready(self, value: object, mode: str) -> None:
        assert isinstance(value, CandidatePreviewResult)
        paths = {
            "original": value.original_wav_path,
            "selection": value.selection_wav_path,
            "edited": value.edited_wav_path,
        }
        path = paths[mode]
        try:
            self._player.play(path)
        except AudioPlaybackError as exc:
            self._playback_failed(str(exc))
        else:
            messages = {
                "original": "正在试听原音（删除范围前后各 3 秒）",
                "selection": "正在试听将要删除的音频范围",
                "edited": "正在试听删除后效果（删除范围前后各 3 秒）",
            }
            self.statusMessage.emit(messages[mode])

    def _playback_failed(self, message: str) -> None:
        self.statusMessage.emit("试听失败，请检查系统音频输出设备")
        QMessageBox.warning(self, "无法试听", message or "音频设备播放失败，请检查系统输出设备。")

    def _start_export(self) -> None:
        if self.project is None or self.project_path is None or self.tools is None:
            return
        pending_review = [item for item in self.project.annotations if item.review_required]
        if pending_review:
            QMessageBox.information(
                self,
                "仍有边界需要复核",
                f"还有 {len(pending_review)} 个删除范围尚未完成波形或人工确认。\n\n"
                "请逐项试听后拖动边界，或点击“确认边界”，再执行导出。",
            )
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
        self.statusMessage.emit(f"音频处理导出完成：{value.wav_path.parent}")

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

    def _start_task(
        self,
        operation,
        callback,
        *,
        show_progress: bool = True,
        busy_button: QAbstractButton | None = None,
    ) -> None:  # type: ignore[no-untyped-def]
        if self._active_task is not None:
            return
        task = BackgroundTask(operation)
        self._active_task = task
        self._foreground_generation += 1
        generation = self._foreground_generation
        self._task_uses_progress = show_progress
        if show_progress:
            self.task_progress.start("正在准备本地处理…")
            task.signals.progress.connect(self._task_progress)
        else:
            self.task_progress.finish()
            if busy_button is not None:
                self._preview_spinner.start(busy_button)
            task.signals.progress.connect(self._task_status)
        task.signals.result.connect(
            lambda value: self._deliver_task_result(
                task,
                generation,
                callback,
                value,
            )
        )
        task.signals.error.connect(self._task_error)
        task.signals.cancelled.connect(self._task_cancelled)
        task.signals.finished.connect(lambda: self._task_finished(task))
        self.thread_pool.start(task)
        self._refresh_controls()

    def _task_status(self, _value: int, message: str) -> None:
        if message:
            self.statusMessage.emit(message)

    def _deliver_task_result(
        self,
        task: BackgroundTask,
        generation: int,
        callback,
        value: object,
    ) -> None:  # type: ignore[no-untyped-def]
        if self._active_task is task and generation == self._foreground_generation:
            callback(value)

    def _task_progress(self, value: int, message: str) -> None:
        self.task_progress.set_progress(value, message)
        if message:
            self.statusMessage.emit(message)

    def _task_error(self, message: str) -> None:
        self.statusMessage.emit(f"操作失败：{message}")
        QMessageBox.warning(self, "操作失败", message)

    def _task_cancelled(self) -> None:
        self.summary_label.setText("操作已取消，可重新开始")
        self.statusMessage.emit("操作已取消")

    def _stop_playback(self) -> None:
        self._player.stop()
        self.statusMessage.emit("试听已停止")

    def _task_finished(self, task: BackgroundTask) -> None:
        if self._active_task is not task:
            return
        self._active_task = None
        self._preview_spinner.stop()
        if self._task_uses_progress:
            self.task_progress.finish()
        self._task_uses_progress = True
        self._sync_workspace_page()
        self._refresh_controls()

    def _cancel_task(self) -> None:
        if self._active_task is not None:
            self.task_progress.set_message("正在取消 · 若模型正在推理，将在当前片段结束后停止…")
            self.task_progress.set_cancel_enabled(False)
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
        self.import_view.set_busy(busy)
        ready = self.project is not None
        has_selection = self._selected_token_range() is not None if ready else False
        waveform_ready = self.waveform_envelope is not None
        self.audio_path_edit.setEnabled(not busy)
        self.audio_browse_button.setEnabled(not busy)
        self.open_project_button.setEnabled(not busy)
        self.task_progress.set_cancel_enabled(busy)
        audio_source = Path(self.audio_path_edit.text().strip()).expanduser()
        supported_audio = (
            audio_source.is_file() and audio_source.suffix.casefold() in _AUDIO_SUFFIXES
        )
        self.transcribe_button.setEnabled(not busy and supported_audio)
        self.add_delete_button.setEnabled(not busy and ready and (has_selection or waveform_ready))
        self.preview_text_button.setEnabled(not busy and ready and has_selection)
        self.preview_original_button.setEnabled(not busy and waveform_ready)
        self.preview_selection_button.setEnabled(not busy and waveform_ready)
        self.preview_edited_button.setEnabled(not busy and waveform_ready)
        self.stop_button.setEnabled(self._player.is_active)
        self.remove_annotation_button.setEnabled(not busy and self._editing_annotation() is not None)
        editing_annotation = self._editing_annotation()
        self.confirm_boundary_button.setEnabled(
            not busy and editing_annotation is not None and editing_annotation.review_required
        )
        self.start_spin.setEnabled(not busy and waveform_ready)
        self.end_spin.setEnabled(not busy and waveform_ready)
        pending_review = bool(
            ready and any(item.review_required for item in self.project.annotations)
        )
        self.export_button.setEnabled(
            not busy and ready and bool(self.project.annotations) and not pending_review
        )
        self.export_button.setToolTip(
            "请先确认所有待复核删除边界" if pending_review else "导出已确认的删除结果"
        )

    def shutdown(self) -> None:
        self._player.stop()
        self._preview_spinner.stop()
        if self._active_task is not None:
            self._active_task.cancel()
        self._autosave()


def _time_spin(name: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
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
