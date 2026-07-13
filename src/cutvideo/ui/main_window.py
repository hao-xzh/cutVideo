"""Main desktop workflow window."""

from __future__ import annotations

import html
from collections import Counter
from contextlib import suppress
from pathlib import Path

from PySide6.QtCore import (
    QDir,
    QEasingCurve,
    QPropertyAnimation,
    QSignalBlocker,
    Qt,
    QTemporaryDir,
    QThreadPool,
    QTimer,
    QUrl,
)
from PySide6.QtGui import QCloseEvent, QColor, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..audio import AudioInfo, WaveformEnvelope
from ..ffmpeg import FFmpegTools, discover_ffmpeg
from ..project import CandidateStatus, CutCandidate, ProjectV1, save_project
from ..resources import discover_resources
from .audio_player import AudioPlaybackError, PcmWavPlayer
from .audio_processing_widget import AudioProcessingWidget
from .file_drop_edit import FileDropLineEdit
from .icons import ButtonSpinner, make_icon_button, set_button_icon, ui_icon
from .import_view import ImportLandingView
from .progress_view import TaskProgressView
from .settings import dialog_start, remember_dialog_path, remember_theme
from .theme import (
    THEME_DARK,
    THEME_LIGHT,
    current_theme,
    stylesheet_for_theme,
    theme_color,
)
from .usage_guide import UsageGuideDialog
from .waveform import WaveformWidget
from .workers import (
    AnalysisResult,
    BackgroundTask,
    CandidatePreviewResult,
    ExportTaskResult,
    LoadedProjectResult,
    PreflightResult,
    make_analysis_operation,
    make_export_operation,
    make_load_project_operation,
    make_preflight_operation,
    make_preview_operation,
    make_relink_project_operation,
    make_waveform_operation,
)

_WAVEFORM_SCROLL_STEPS = 100_000
_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
_DOCUMENT_SUFFIXES = {".docx"}

STATUS_LABELS = {
    CandidateStatus.AUTO_APPROVED: "自动通过",
    CandidateStatus.NEEDS_REVIEW: "需复核",
    CandidateStatus.APPROVED: "已批准",
    CandidateStatus.SKIPPED: "已保留",
}

REASON_LABELS = {
    "short_highlight": "文字较短（1–2 字）",
    "repeated_context": "上下文重复",
    "insufficient_coverage": "对齐覆盖不足",
    "boundary_disagreement_over_120ms": "双模型边界差异超过 120 ms",
    "fallback_character_ratio": "模型不可用，按文字比例估算",
    "force_aligner_unavailable": "强制对齐模型不可用",
    "asr_aligner_unavailable": "真实语音识别模型不可用",
    "document_time_search_fallback": "真实语音未定位到该段，仅用 Word 时间辅助搜索",
    "local_context_alignment_unstable": "两次局部文字边界对齐不稳定",
    "local_audio_recognition_refinement_unavailable": "局部真实语音重识别未能可靠定位",
    "alignment_candidate_margin_too_small": "重复或近似语句候选差距过小",
    "coarse_timestamp_not_character_level": "模型只提供词级或句级时间戳",
    "timestamp_validation_failed": "模型时间戳存在单位、顺序或跨度异常",
    "model_character_confidence_unavailable": "模型未提供可验证的逐字置信度",
    "low_model_character_confidence": "模型原始逐字置信度偏低",
    "adjacent_character_guard_unavailable": "缺少相邻字符边界保护",
    "boundary_expanded_for_speech_edge": "为保护语音起止点已有限向外扩张",
    "boundary_refinement_evidence_incomplete": "切口缺少完整语音边缘或零交叉证据",
}


class MainWindow(QMainWindow):
    """Chinese desktop UI for import, analysis, review and export."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mainWindow")
        self.setWindowTitle("CutVideo · 离线音频工作台")
        self.resize(1360, 900)
        self.setMinimumSize(1100, 720)

        self.thread_pool = QThreadPool.globalInstance()
        self._active_task: BackgroundTask | None = None
        self._foreground_uses_progress = True
        self._preview_spinner = ButtonSpinner(self)
        self._auxiliary_tasks: set[BackgroundTask] = set()
        self._preflight: PreflightResult | None = None
        self.project: ProjectV1 | None = None
        self.project_path: Path | None = None
        self.audio_info: AudioInfo | None = None
        self.tools: FFmpegTools | None = None
        self.waveform_envelope: WaveformEnvelope | None = None
        self._selected_row = -1
        self._updating_boundaries = False
        self._setting_paths = False
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(450)
        self._autosave_timer.timeout.connect(self._autosave)
        preview_template = str(Path(QDir.tempPath()) / "cutvideo-preview-XXXXXX")
        self._preview_directory = QTemporaryDir(preview_template)
        self._audio_player = PcmWavPlayer(self)
        self._audio_player.finished.connect(self._playback_finished)
        self._audio_player.failed.connect(self._playback_failed)
        self._audio_player.active_changed.connect(lambda _active: self._refresh_controls())
        self._preview_queue: list[str] = []
        self._preview_queue_active = False
        self._undo_history: list[dict[str, object]] = []
        self._redo_history: list[dict[str, object]] = []
        self._edit_baseline: dict[str, object] | None = None
        self._edit_active = False
        self._shortcuts: list[QShortcut] = []
        self._theme_name = current_theme()
        self._theme_animation: QPropertyAnimation | None = None
        self._theme_overlay: QLabel | None = None
        self._workspace_animation: QPropertyAnimation | None = None
        self._workspace_overlay: QLabel | None = None
        self._foreground_generation = 0
        self._source_generation = 0

        self._build_ui()
        self._connect_signals()
        self._install_shortcuts()
        self._set_step(1)
        self._refresh_controls()
        self.statusBar().setSizeGripEnabled(False)
        # Primary state is presented inline; a permanent bottom status strip
        # adds visual noise to the compact workstation layout.
        self.statusBar().hide()

    def _build_ui(self) -> None:
        shell = QWidget(self)
        shell.setObjectName("applicationShell")
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        sidebar = QFrame(shell)
        sidebar.setObjectName("workspaceSidebar")
        sidebar.setFixedWidth(88)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 14, 10, 12)
        sidebar_layout.setSpacing(8)

        brand = QLabel(sidebar)
        brand.setObjectName("brandMark")
        application = QApplication.instance()
        brand_icon = application.windowIcon() if application is not None else ui_icon("cut-accent")
        brand.setPixmap(brand_icon.pixmap(30, 30))
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand.setToolTip("CutVideo 离线音频工作台")
        brand.setAccessibleName("CutVideo")
        brand.setFixedHeight(42)
        sidebar_layout.addWidget(brand)
        sidebar_layout.addSpacing(18)

        self.workspace_navigation = QButtonGroup(self)
        self.workspace_navigation.setExclusive(True)
        self.word_workspace_button = _make_workspace_button(
            "Word 剪辑",
            "document-accent",
            "Word 黄标剪辑工作区",
        )
        self.audio_workspace_button = _make_workspace_button(
            "音频处理",
            "audio",
            "完整转写与音频处理工作区",
        )
        self.workspace_navigation.addButton(self.word_workspace_button, 0)
        self.workspace_navigation.addButton(self.audio_workspace_button, 1)
        self.word_workspace_button.setChecked(True)
        sidebar_layout.addWidget(self.word_workspace_button)
        sidebar_layout.addWidget(self.audio_workspace_button)
        sidebar_layout.addStretch(1)
        shell_layout.addWidget(sidebar)

        workspace_area = QWidget(shell)
        workspace_area.setObjectName("workspaceArea")
        workspace_layout = QVBoxLayout(workspace_area)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)

        self.help_button = make_icon_button(
            "help",
            "使用说明与快捷键（F1）",
            size=34,
        )
        self.help_button.setObjectName("helpButton")
        self.help_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_button = make_icon_button(
            "sun",
            "切换到明亮模式",
            size=34,
        )
        self.theme_button.setObjectName("themeButton")
        self.theme_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_theme_button()

        utility_bar = QWidget(workspace_area)
        utility_bar.setObjectName("utilityBar")
        utility_bar.setFixedHeight(44)
        utility_layout = QHBoxLayout(utility_bar)
        utility_layout.setContentsMargins(0, 5, 12, 3)
        utility_layout.setSpacing(2)
        utility_layout.addStretch(1)
        utility_layout.addWidget(self.help_button)
        utility_layout.addWidget(self.theme_button)
        workspace_layout.addWidget(utility_bar)

        # Kept as ``workspace_tabs`` for compatibility with existing shortcuts/plugins.
        self.workspace_tabs = QStackedWidget(workspace_area)
        self.workspace_tabs.setObjectName("workspaceStack")
        workspace_layout.addWidget(self.workspace_tabs, 1)
        shell_layout.addWidget(workspace_area, 1)
        self.setCentralWidget(shell)

        self.word_workspace_stack = QStackedWidget()
        self.word_workspace_stack.setObjectName("wordWorkspaceStack")
        self.word_import_view = ImportLandingView("word")
        self.word_workspace_stack.addWidget(self.word_import_view)
        viewport = QScrollArea()
        viewport.setObjectName("mainScrollArea")
        viewport.setWidgetResizable(True)
        viewport.setFrameShape(QFrame.Shape.NoFrame)
        self.word_workspace_stack.addWidget(viewport)
        self.workspace_tabs.addWidget(self.word_workspace_stack)
        central = QWidget()
        central.setObjectName("centralWidget")
        central.setMinimumSize(940, 600)
        viewport.setWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 8, 16, 12)
        root.setSpacing(10)

        input_card = QFrame()
        input_card.setObjectName("inputCard")
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(12, 10, 12, 9)
        input_layout.setSpacing(8)

        file_row = QHBoxLayout()
        file_row.setSpacing(18)
        audio_row = QHBoxLayout()
        audio_row.setSpacing(8)
        audio_label = QLabel("音频")
        audio_label.setObjectName("fieldLabel")
        audio_label.setFixedWidth(34)
        self.audio_path_edit = FileDropLineEdit((".mp3", ".wav", ".m4a", ".aac", ".flac"))
        self.audio_path_edit.setObjectName("audioPathEdit")
        self.audio_path_edit.setPlaceholderText("拖入音频，或点击右侧选择文件")
        self.audio_browse_button = make_icon_button(
            "folder-open",
            "选择音频文件",
        )
        self.audio_browse_button.setObjectName("audioBrowseButton")
        audio_row.addWidget(audio_label)
        audio_row.addWidget(self.audio_path_edit, 1)
        audio_row.addWidget(self.audio_browse_button)
        file_row.addLayout(audio_row, 1)

        document_row = QHBoxLayout()
        document_row.setSpacing(8)
        document_label = QLabel("Word")
        document_label.setObjectName("fieldLabel")
        document_label.setFixedWidth(38)
        self.docx_path_edit = FileDropLineEdit((".docx",))
        self.docx_path_edit.setObjectName("docxPathEdit")
        self.docx_path_edit.setPlaceholderText("拖入带黄色标记的 Word，或点击右侧选择文件")
        self.docx_browse_button = make_icon_button(
            "folder-open",
            "选择 Word 标注文档",
        )
        self.docx_browse_button.setObjectName("docxBrowseButton")
        document_row.addWidget(document_label)
        document_row.addWidget(self.docx_path_edit, 1)
        document_row.addWidget(self.docx_browse_button)
        file_row.addLayout(document_row, 1)
        input_layout.addLayout(file_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.workflow_step_label = QLabel("输入准备")
        self.workflow_step_label.setObjectName("workflowStepLabel")
        self.input_summary_label = QLabel("尚未预检")
        self.input_summary_label.setObjectName("mutedLabel")
        self.input_summary_label.setWordWrap(True)
        action_row.addWidget(self.workflow_step_label)
        action_row.addWidget(self.input_summary_label, 1)

        self.open_project_button = QPushButton("打开项目…")
        self.open_project_button.setObjectName("openProjectButton")
        self.open_project_button.setProperty("secondary", True)
        self.open_project_button.setMinimumWidth(112)
        set_button_icon(self.open_project_button, "project")
        self.preflight_button = QPushButton("开始预检")
        self.preflight_button.setObjectName("preflightButton")
        self.preflight_button.setMinimumWidth(106)
        set_button_icon(self.preflight_button, "check")
        self.analyze_button = QPushButton("自动分析")
        self.analyze_button.setObjectName("analyzeButton")
        self.analyze_button.setProperty("primary", True)
        self.analyze_button.setMinimumWidth(110)
        set_button_icon(self.analyze_button, "audio-primary")
        action_row.addWidget(self.open_project_button)
        action_row.addWidget(self.preflight_button)
        action_row.addWidget(self.analyze_button)
        input_layout.addLayout(action_row)
        root.addWidget(input_card)

        self.task_progress = TaskProgressView()
        self.progress_container = self.task_progress
        self.progress_label = self.task_progress.message_label
        self.progress_bar = self.task_progress.progress_bar
        self.cancel_button = self.task_progress.cancel_button
        root.addWidget(self.task_progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("reviewSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        list_card = QFrame()
        list_card.setObjectName("reviewCard")
        list_layout = QVBoxLayout(list_card)
        list_layout.setContentsMargins(10, 9, 10, 10)
        list_layout.setSpacing(8)
        list_heading = QHBoxLayout()
        list_title = QLabel("候选切点")
        list_title.setObjectName("sectionTitle")
        self.review_summary_label = QLabel("等待分析")
        self.review_summary_label.setObjectName("mutedLabel")
        self.candidate_filter = QComboBox()
        self.candidate_filter.setObjectName("candidateFilter")
        self.candidate_filter.addItems(["全部切点", "仅需复核", "仅已删除", "仅已保留"])
        list_heading.addWidget(list_title)
        list_heading.addStretch(1)
        list_heading.addWidget(self.candidate_filter)
        list_heading.addWidget(self.review_summary_label)
        list_layout.addLayout(list_heading)
        self.candidate_table = QTableWidget(0, 6)
        self.candidate_table.setObjectName("candidateTable")
        self.candidate_table.setHorizontalHeaderLabels(
            ["文字", "段落", "时间", "置信度", "状态", "原因"]
        )
        self.candidate_table.setAlternatingRowColors(True)
        self.candidate_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.candidate_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.candidate_table.verticalHeader().setVisible(False)
        header_view = self.candidate_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        list_layout.addWidget(self.candidate_table, 1)
        splitter.addWidget(list_card)

        detail_card = QFrame()
        detail_card.setObjectName("reviewCard")
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(10, 9, 10, 10)
        detail_layout.setSpacing(9)
        detail_title = QLabel("切点复核")
        detail_title.setObjectName("sectionTitle")
        detail_layout.addWidget(detail_title)
        self.context_label = QLabel("从左侧选择一个候选项")
        self.context_label.setObjectName("hintLabel")
        self.context_label.setWordWrap(True)
        self.context_label.setTextFormat(Qt.TextFormat.RichText)
        self.context_label.setMinimumHeight(36)
        detail_layout.addWidget(self.context_label)
        waveform_hint = QLabel("拖动空白波形可平移；滚轮缩放；橙色边界可直接拖动")
        waveform_hint.setObjectName("mutedLabel")
        detail_layout.addWidget(waveform_hint)
        self.waveform = WaveformWidget()
        detail_layout.addWidget(self.waveform, 1)

        waveform_view_layout = QHBoxLayout()
        waveform_view_layout.setSpacing(7)
        waveform_view_layout.addWidget(QLabel("视野"))
        self.waveform_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.waveform_scrollbar.setObjectName("waveformScrollBar")
        self.waveform_scrollbar.setAccessibleName("波形视野位置")
        self.waveform_scrollbar.setToolTip("拖动可查看当前切点前后更远的波形")
        self.waveform_scrollbar.setRange(0, 0)
        waveform_view_layout.addWidget(self.waveform_scrollbar, 1)
        self.waveform_zoom_out_button = make_icon_button(
            "zoom-out",
            "缩小波形视野",
        )
        self.waveform_zoom_out_button.setObjectName("waveformZoomOutButton")
        self.waveform_zoom_in_button = make_icon_button(
            "zoom-in",
            "放大波形以精调切点",
        )
        self.waveform_zoom_in_button.setObjectName("waveformZoomInButton")
        self.waveform_focus_button = make_icon_button(
            "focus",
            "定位当前切点",
        )
        self.waveform_focus_button.setObjectName("waveformFocusButton")
        waveform_view_layout.addWidget(self.waveform_zoom_out_button)
        waveform_view_layout.addWidget(self.waveform_zoom_in_button)
        waveform_view_layout.addWidget(self.waveform_focus_button)
        detail_layout.addLayout(waveform_view_layout)

        boundary_layout = QGridLayout()
        boundary_layout.setHorizontalSpacing(9)
        self.start_spin = _make_time_spin("startTimeSpin")
        self.end_spin = _make_time_spin("endTimeSpin")
        boundary_layout.addWidget(QLabel("开始切点"), 0, 0)
        start_controls = QHBoxLayout()
        start_controls.setSpacing(5)
        start_controls.addWidget(self.start_spin, 1)
        self.start_minus_button = _make_nudge_button("−20 ms", "开始切点提前 20 毫秒")
        self.start_plus_button = _make_nudge_button("+20 ms", "开始切点延后 20 毫秒")
        start_controls.addWidget(self.start_minus_button)
        start_controls.addWidget(self.start_plus_button)
        boundary_layout.addLayout(start_controls, 1, 0)
        boundary_layout.addWidget(QLabel("结束切点"), 0, 1)
        end_controls = QHBoxLayout()
        end_controls.setSpacing(5)
        end_controls.addWidget(self.end_spin, 1)
        self.end_minus_button = _make_nudge_button("−20 ms", "结束切点提前 20 毫秒")
        self.end_plus_button = _make_nudge_button("+20 ms", "结束切点延后 20 毫秒")
        end_controls.addWidget(self.end_minus_button)
        end_controls.addWidget(self.end_plus_button)
        boundary_layout.addLayout(end_controls, 1, 1)
        detail_layout.addLayout(boundary_layout)

        preview_layout = QHBoxLayout()
        self.preview_original_button = QPushButton("听原句")
        self.preview_original_button.setObjectName("previewOriginalButton")
        set_button_icon(self.preview_original_button, "play")
        self.preview_selection_button = QPushButton("只听待删")
        self.preview_selection_button.setObjectName("previewSelectionButton")
        set_button_icon(self.preview_selection_button, "play")
        self.preview_edited_button = QPushButton("听剪后")
        self.preview_edited_button.setObjectName("previewEditedButton")
        set_button_icon(self.preview_edited_button, "play")
        self.stop_preview_button = make_icon_button("stop", "停止试听")
        self.stop_preview_button.setObjectName("stopPreviewButton")
        preview_layout.addWidget(self.preview_original_button)
        preview_layout.addWidget(self.preview_selection_button)
        preview_layout.addWidget(self.preview_edited_button)
        preview_layout.addWidget(self.stop_preview_button)
        preview_layout.addStretch(1)
        detail_layout.addLayout(preview_layout)

        review_actions = QHBoxLayout()
        self.reset_button = QPushButton("恢复建议")
        self.reset_button.setObjectName("resetCandidateButton")
        set_button_icon(self.reset_button, "reset")
        self.skip_button = QPushButton("保留此处")
        self.skip_button.setObjectName("skipCandidateButton")
        self.approve_button = QPushButton("确认删除")
        self.approve_button.setObjectName("approveCandidateButton")
        self.approve_button.setProperty("primary", True)
        set_button_icon(self.approve_button, "delete")
        review_actions.addWidget(self.reset_button)
        review_actions.addStretch(1)
        review_actions.addWidget(self.skip_button)
        review_actions.addWidget(self.approve_button)
        detail_layout.addLayout(review_actions)
        splitter.addWidget(detail_card)
        splitter.setSizes([500, 820])
        root.addWidget(splitter, 1)

        export_card = QFrame()
        export_card.setObjectName("exportCard")
        export_layout = QHBoxLayout(export_card)
        export_layout.setContentsMargins(12, 8, 12, 8)
        export_title = QLabel("导出位置")
        export_title.setObjectName("sectionTitle")
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setObjectName("outputPathEdit")
        self.output_path_edit.setPlaceholderText("默认与源音频相同目录")
        self.output_browse_button = make_icon_button(
            "folder-open",
            "选择导出目录",
        )
        self.output_browse_button.setObjectName("outputBrowseButton")
        self.review_all_button = QPushButton("顺序试听已批准切点")
        self.review_all_button.setObjectName("reviewAllButton")
        set_button_icon(self.review_all_button, "play")
        self.export_button = QPushButton("导出 WAV + MP3")
        self.export_button.setObjectName("exportButton")
        self.export_button.setProperty("primary", True)
        set_button_icon(self.export_button, "export")
        self.export_button.setMinimumWidth(150)
        export_layout.addWidget(export_title)
        export_layout.addWidget(self.output_path_edit, 1)
        export_layout.addWidget(self.output_browse_button)
        export_layout.addWidget(self.review_all_button)
        export_layout.addWidget(self.export_button)
        root.addWidget(export_card)

        # Compatibility aliases are intentionally simple and useful to UI tests/plugins.
        self.audio_edit = self.audio_path_edit
        self.document_edit = self.docx_path_edit
        self.analysis_button = self.analyze_button

        processing_viewport = QScrollArea()
        processing_viewport.setObjectName("audioProcessingScrollArea")
        processing_viewport.setWidgetResizable(True)
        processing_viewport.setFrameShape(QFrame.Shape.NoFrame)
        self.audio_processing_widget = AudioProcessingWidget(self.thread_pool)
        self.audio_processing_widget.setMinimumSize(940, 600)
        processing_viewport.setWidget(self.audio_processing_widget)
        self.workspace_tabs.addWidget(processing_viewport)

    def _connect_signals(self) -> None:
        self.word_import_view.audioBrowseRequested.connect(self._choose_audio)
        self.word_import_view.documentBrowseRequested.connect(self._choose_document)
        self.word_import_view.projectBrowseRequested.connect(self._choose_project)
        self.word_import_view.audioDropped.connect(self._set_imported_audio)
        self.word_import_view.documentDropped.connect(self._set_imported_document)
        self.audio_browse_button.clicked.connect(self._choose_audio)
        self.audio_path_edit.fileDropped.connect(
            lambda path: self.statusBar().showMessage(f"已拖入音频：{Path(path).name}")
        )
        self.docx_browse_button.clicked.connect(self._choose_document)
        self.docx_path_edit.fileDropped.connect(
            lambda path: self.statusBar().showMessage(f"已拖入 Word：{Path(path).name}")
        )
        self.output_browse_button.clicked.connect(self._choose_output_directory)
        self.open_project_button.clicked.connect(self._choose_project)
        self.preflight_button.clicked.connect(self._start_preflight)
        self.analyze_button.clicked.connect(self._start_analysis)
        self.task_progress.cancelRequested.connect(self._cancel_active_task)
        self.candidate_table.currentCellChanged.connect(self._candidate_row_changed)
        self.candidate_filter.currentIndexChanged.connect(self._apply_candidate_filter)
        self.start_spin.valueChanged.connect(self._boundary_spin_changed)
        self.end_spin.valueChanged.connect(self._boundary_spin_changed)
        self.start_spin.editingFinished.connect(self._reset_edit_baseline)
        self.end_spin.editingFinished.connect(self._reset_edit_baseline)
        self.waveform.boundariesChanged.connect(self._waveform_boundaries_changed)
        self.waveform.boundaryDragFinished.connect(self._waveform_boundary_drag_finished)
        self.waveform.viewChanged.connect(self._waveform_view_changed)
        self.waveform_scrollbar.valueChanged.connect(self._waveform_scroll_changed)
        self.waveform_zoom_out_button.clicked.connect(self.waveform.zoom_out)
        self.waveform_zoom_in_button.clicked.connect(self.waveform.zoom_in)
        self.waveform_focus_button.clicked.connect(self.waveform.focus_selection)
        self.start_minus_button.clicked.connect(lambda: self._nudge_boundary("start", -20.0))
        self.start_plus_button.clicked.connect(lambda: self._nudge_boundary("start", 20.0))
        self.end_minus_button.clicked.connect(lambda: self._nudge_boundary("end", -20.0))
        self.end_plus_button.clicked.connect(lambda: self._nudge_boundary("end", 20.0))
        self.approve_button.clicked.connect(self._approve_candidate)
        self.skip_button.clicked.connect(self._skip_candidate)
        self.reset_button.clicked.connect(self._reset_candidate)
        self.preview_original_button.clicked.connect(
            lambda: self._start_preview("original", trigger_button=self.preview_original_button)
        )
        self.preview_selection_button.clicked.connect(
            lambda: self._start_preview(
                "selection",
                trigger_button=self.preview_selection_button,
            )
        )
        self.preview_edited_button.clicked.connect(
            lambda: self._start_preview("edited", trigger_button=self.preview_edited_button)
        )
        self.stop_preview_button.clicked.connect(lambda: self._stop_preview())
        self.review_all_button.clicked.connect(self._start_review_all)
        self.export_button.clicked.connect(self._start_export)
        self.audio_path_edit.textChanged.connect(self._inputs_changed)
        self.docx_path_edit.textChanged.connect(self._inputs_changed)
        self.audio_processing_widget.statusMessage.connect(self.statusBar().showMessage)
        self.workspace_navigation.idClicked.connect(self._switch_workspace)
        self.workspace_tabs.currentChanged.connect(self._workspace_changed)
        self.help_button.clicked.connect(self._show_usage_guide)
        self.theme_button.clicked.connect(self._toggle_theme)

    def _install_shortcuts(self) -> None:
        shortcuts = {
            "Space": lambda: (
                self._stop_preview() if self._audio_player.is_active else self._start_preview("edited")
            ),
            "Return": self._approve_candidate,
            "Ctrl+Return": self._approve_candidate,
            "Ctrl+K": self._skip_candidate,
            "Meta+K": self._skip_candidate,
            "Ctrl+R": self._reset_candidate,
            "Meta+R": self._reset_candidate,
            "J": lambda: self._move_candidate(1),
            "K": lambda: self._move_candidate(-1),
            "Ctrl+Z": self._undo_edit,
            "Meta+Z": self._undo_edit,
            "Ctrl+Shift+Z": self._redo_edit,
            "Meta+Shift+Z": self._redo_edit,
        }
        for sequence, callback in shortcuts.items():
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(
                lambda callback=callback: (
                    callback() if self.workspace_tabs.currentIndex() == 0 else None
                )
            )
            self._shortcuts.append(shortcut)

        help_shortcut = QShortcut(QKeySequence("F1"), self)
        help_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        help_shortcut.activated.connect(self._show_usage_guide)
        self._shortcuts.append(help_shortcut)

        workspace_shortcuts = {
            "Ctrl+1": 0,
            "Meta+1": 0,
            "Ctrl+2": 1,
            "Meta+2": 1,
        }
        for sequence, index in workspace_shortcuts.items():
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(
                lambda index=index: self._switch_workspace(index)
            )
            self._shortcuts.append(shortcut)
        for sequence, offset in (("Ctrl+Tab", 1), ("Ctrl+Shift+Tab", -1)):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(
                lambda offset=offset: self._cycle_workspace(offset)
            )
            self._shortcuts.append(shortcut)

    def _show_usage_guide(self) -> None:
        dialog = UsageGuideDialog(self)
        dialog.exec()

    def _update_theme_button(self) -> None:
        if self._theme_name == THEME_DARK:
            icon_name = "sun"
            tooltip = "切换到明亮模式"
        else:
            icon_name = "moon"
            tooltip = "切换到黑夜模式"
        set_button_icon(self.theme_button, icon_name, tooltip=tooltip)

    def _toggle_theme(self) -> None:
        if self._theme_animation is not None:
            return
        app = QApplication.instance()
        if app is None:
            return

        snapshot = self.grab()
        overlay = QLabel(self)
        overlay.setObjectName("themeTransitionOverlay")
        overlay.setPixmap(snapshot)
        overlay.setScaledContents(True)
        overlay.setGeometry(self.rect())
        overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        opacity = QGraphicsOpacityEffect(overlay)
        overlay.setGraphicsEffect(opacity)
        overlay.show()
        overlay.raise_()
        self._theme_overlay = overlay
        self.theme_button.setEnabled(False)

        self._theme_name = THEME_LIGHT if self._theme_name == THEME_DARK else THEME_DARK
        app.setProperty("theme", self._theme_name)
        app.setStyleSheet(stylesheet_for_theme(self._theme_name))
        remember_theme(self._theme_name)
        self._update_theme_button()
        self._refresh_theme_colors()
        overlay.raise_()

        animation = QPropertyAnimation(opacity, b"opacity", self)
        animation.setDuration(280)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        animation.finished.connect(self._finish_theme_transition)
        self._theme_animation = animation
        animation.start()

    def _finish_theme_transition(self) -> None:
        if self._theme_overlay is not None:
            self._theme_overlay.hide()
            self._theme_overlay.setGraphicsEffect(None)
            self._theme_overlay.deleteLater()
        self._theme_overlay = None
        self._theme_animation = None
        self.theme_button.setEnabled(True)

    def _refresh_theme_colors(self) -> None:
        if self.project is not None and self.audio_info is not None:
            for row, candidate in enumerate(self.project.candidates):
                self._update_candidate_row(row, candidate)
            self._show_selected_candidate(center=False)
        self.waveform.update()
        self.audio_processing_widget.refresh_theme()

    def _set_imported_audio(self, path: str) -> None:
        remember_dialog_path("word_audio", path)
        self.audio_path_edit.setText(path)
        self.statusBar().showMessage(f"已选择音频：{Path(path).name}")

    def _set_imported_document(self, path: str) -> None:
        remember_dialog_path("word_document", path)
        self.docx_path_edit.setText(path)
        self.statusBar().showMessage(f"已选择 Word：{Path(path).name}")

    def _sync_word_workspace_page(self, *, force_editor: bool = False) -> None:
        audio = self.audio_path_edit.text().strip()
        document = self.docx_path_edit.text().strip()
        self.word_import_view.set_paths(audio, document)
        inputs_ready = _is_supported_file(audio, _AUDIO_SUFFIXES) and _is_supported_file(
            document,
            _DOCUMENT_SUFFIXES,
        )
        show_editor = force_editor or self.project is not None or inputs_ready
        self.word_workspace_stack.setCurrentIndex(1 if show_editor else 0)

    def _record_edit_baseline(self) -> None:
        if self.project is None:
            return
        if self._edit_active:
            return
        if self._edit_baseline is None:
            self._edit_baseline = self.project.to_dict()
        self._undo_history.append(self._edit_baseline)
        self._undo_history = self._undo_history[-100:]
        self._redo_history.clear()
        self._edit_baseline = None
        self._edit_active = True

    def _reset_edit_baseline(self) -> None:
        self._edit_baseline = self.project.to_dict() if self.project is not None else None
        self._edit_active = False

    def _restore_history_snapshot(self, snapshot: dict[str, object]) -> None:
        selected_row = self._selected_row
        self.project = ProjectV1.from_dict(snapshot)
        self._populate_candidates()
        if self.project.candidates:
            row = max(0, min(selected_row, len(self.project.candidates) - 1))
            self.candidate_table.selectRow(row)
            self.candidate_table.setCurrentCell(row, 0)
        self._schedule_autosave()
        self._reset_edit_baseline()
        self._refresh_controls()

    def _undo_edit(self) -> None:
        if self.project is None or not self._undo_history:
            return
        self._redo_history.append(self.project.to_dict())
        self._restore_history_snapshot(self._undo_history.pop())
        self.statusBar().showMessage("已撤销上一步复核操作")

    def _redo_edit(self) -> None:
        if self.project is None or not self._redo_history:
            return
        self._undo_history.append(self.project.to_dict())
        self._restore_history_snapshot(self._redo_history.pop())
        self.statusBar().showMessage("已重做复核操作")

    def _move_candidate(self, offset: int) -> None:
        if self.project is None or not self.project.candidates or self._active_task is not None:
            return
        row = (self._selected_row + offset) % len(self.project.candidates)
        self.candidate_table.selectRow(row)
        self.candidate_table.setCurrentCell(row, 0)
        self.candidate_table.scrollToItem(self.candidate_table.item(row, 0))

    def _switch_workspace(self, index: int) -> None:
        if index == self.workspace_tabs.currentIndex():
            return
        self._finish_workspace_transition()
        previous = self.workspace_tabs.currentWidget()
        snapshot = previous.grab() if previous is not None else None
        self.workspace_tabs.setCurrentIndex(index)
        if snapshot is None:
            return
        overlay = QLabel(self.workspace_tabs)
        overlay.setObjectName("workspaceTransitionOverlay")
        overlay.setPixmap(snapshot)
        overlay.setScaledContents(True)
        overlay.setGeometry(self.workspace_tabs.rect())
        overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        effect = QGraphicsOpacityEffect(overlay)
        overlay.setGraphicsEffect(effect)
        overlay.show()
        overlay.raise_()
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(180)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(self._finish_workspace_transition)
        self._workspace_overlay = overlay
        self._workspace_animation = animation
        animation.start()

    def _cycle_workspace(self, offset: int) -> None:
        count = self.workspace_tabs.count()
        if count:
            self._switch_workspace((self.workspace_tabs.currentIndex() + offset) % count)

    def _finish_workspace_transition(self) -> None:
        if self._workspace_animation is not None:
            self._workspace_animation.stop()
        if self._workspace_overlay is not None:
            self._workspace_overlay.hide()
            self._workspace_overlay.setGraphicsEffect(None)
            self._workspace_overlay.deleteLater()
        self._workspace_animation = None
        self._workspace_overlay = None
        self.workspace_tabs.update()

    def _workspace_changed(self, index: int) -> None:
        self.word_workspace_button.setChecked(index == 0)
        self.audio_workspace_button.setChecked(index == 1)
        set_button_icon(
            self.word_workspace_button,
            "document-accent" if index == 0 else "document",
            size=22,
            tooltip="Word 黄标剪辑工作区",
        )
        set_button_icon(
            self.audio_workspace_button,
            "audio-accent" if index == 1 else "audio",
            size=22,
            tooltip="完整转写与音频处理工作区",
        )
        if index == 1:
            self.statusBar().showMessage("音频处理 · 拖入一段音频，完整识别后可选择文字或框选波形")
        elif self.project is not None:
            self.statusBar().showMessage("Word 黄标剪辑 · 项目已就绪")
        else:
            self.statusBar().showMessage("Word 黄标剪辑 · 可直接拖入音频与 Word")

    def _choose_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频",
            dialog_start("word_audio", self.audio_path_edit.text()),
            "音频文件 (*.mp3 *.wav *.m4a *.aac *.flac);;所有文件 (*)",
        )
        if path:
            self._set_imported_audio(path)

    def _choose_document(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Word 标注文档",
            dialog_start("word_document", self.docx_path_edit.text()),
            "Word 文档 (*.docx)",
        )
        if path:
            self._set_imported_document(path)

    def _choose_output_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择导出目录",
            dialog_start("word_output", self.output_path_edit.text() or self.audio_path_edit.text()),
        )
        if path:
            remember_dialog_path("word_output", path)
            self.output_path_edit.setText(path)
            if self.project is not None:
                self.project.export_options.output_directory = path
                self._schedule_autosave()

    def _choose_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 CutVideo 项目",
            dialog_start("word_project", self.audio_path_edit.text()),
            "CutVideo 项目 (*.cutvideo.json);;JSON 文件 (*.json)",
        )
        if path:
            remember_dialog_path("word_project", path)
            self.open_project(path)

    def open_project(self, path: str) -> None:
        self._sync_word_workspace_page(force_editor=True)
        self._start_foreground(
            make_load_project_operation(path),
            self._project_loaded,
            "正在打开项目…",
        )

    def _inputs_changed(self) -> None:
        if self._setting_paths:
            return
        self._foreground_generation += 1
        self._source_generation += 1
        if self._active_task is not None:
            self._active_task.cancel()
        self._stop_preview()
        for task in tuple(self._auxiliary_tasks):
            task.cancel()
        self._preflight = None
        self.project = None
        self.project_path = None
        self.audio_info = None
        self.waveform_envelope = None
        self.candidate_table.setRowCount(0)
        self.waveform.clear()
        self.input_summary_label.setText("输入已更改，请重新预检")
        self.review_summary_label.setText("等待分析")
        self._set_step(1)
        audio_text = self.audio_path_edit.text().strip()
        document_text = self.docx_path_edit.text().strip()
        audio_ready = _is_supported_file(audio_text, _AUDIO_SUFFIXES)
        document_ready = _is_supported_file(document_text, _DOCUMENT_SUFFIXES)
        if audio_ready and document_ready:
            self.statusBar().showMessage("音频与 Word 已就绪，可以开始预检")
        elif (audio_text and not audio_ready) or (document_text and not document_ready):
            self.statusBar().showMessage("文件不存在或格式不受支持，请重新选择")
        elif audio_ready:
            self.statusBar().showMessage("音频已就绪，请拖入或选择 Word 文档")
        elif document_ready:
            self.statusBar().showMessage("Word 已就绪，请拖入或选择音频")
        else:
            self.statusBar().showMessage("可直接拖入音频与 Word 文档")
        self._sync_word_workspace_page()
        self._refresh_controls()

    def _start_preflight(self) -> None:
        audio = self.audio_path_edit.text().strip()
        document = self.docx_path_edit.text().strip()
        self._start_foreground(
            make_preflight_operation(audio, document),
            self._preflight_completed,
            "正在预检…",
        )

    def _preflight_completed(self, result: object) -> None:
        assert isinstance(result, PreflightResult)
        self._preflight = result
        self.audio_info = result.audio_info
        self.tools = result.tools
        transcript = result.transcript
        duration = _format_ms(result.audio_info.total_samples, result.audio_info.sample_rate)
        missing_models = [
            item for item in result.resources.missing() if item not in {"ffmpeg", "ffprobe"}
        ]
        summary = (
            f"{len(transcript.paragraphs)} 个时间锚点 · "
            f"{len(transcript.highlights)} 处黄色标记 · "
            f"{transcript.highlighted_char_count} 个标记字符 · 音频 {duration}"
        )
        if missing_models:
            summary += " · 本地模型不完整，将生成全部需复核的安全估算"
        self.input_summary_label.setText(summary)
        self.output_path_edit.setText(str(Path(result.audio_info.path).parent))
        self.statusBar().showMessage("预检完成，可以开始自动分析")
        self._set_step(2)
        if result.reused_project is not None and result.reused_project_path is not None:
            self._accept_project(
                result.reused_project,
                result.reused_project_path,
                result.audio_info,
                result.waveform_envelope,
            )
            self.statusBar().showMessage("已按文件指纹复用已有分析项目")
        self._refresh_controls()

    def _start_analysis(self) -> None:
        if self._preflight is None:
            return
        self._set_step(2)
        self._start_foreground(
            make_analysis_operation(self._preflight),
            self._analysis_completed,
            "正在分析，请保持程序运行…",
        )

    def _analysis_completed(self, result: object) -> None:
        assert isinstance(result, AnalysisResult)
        assert self._preflight is not None
        self._accept_project(
            result.project,
            result.project_path,
            self._preflight.audio_info,
            self._preflight.waveform_envelope,
        )
        reason_counts = Counter(
            reason for candidate in result.project.candidates for reason in candidate.reasons
        )
        unresolved = len(result.project.unresolved_candidates)
        diagnostics = (
            f"分析完成 · {len(result.project.candidates)} 个切点 · {unresolved} 个需复核"
        )
        fallback_count = reason_counts.get("document_time_search_fallback", 0)
        local_failure_count = reason_counts.get(
            "local_audio_recognition_refinement_unavailable",
            0,
        )
        if fallback_count or local_failure_count:
            diagnostics += f" · 时间回退 {fallback_count} · 局部识别不足 {local_failure_count}"
        self.statusBar().showMessage(diagnostics)
        self.input_summary_label.setText(f"{self.input_summary_label.text()} · {diagnostics}")

    def _project_loaded(self, result: object) -> None:
        assert isinstance(result, LoadedProjectResult)
        if not all(result.source_matches.values()):
            replacements = self._request_relink_paths(result)
            if replacements is None:
                self.statusBar().showMessage("打开项目已取消：需要重新关联输入文件")
                return
            audio_path, document_path = replacements
            QTimer.singleShot(
                0,
                lambda: self._start_foreground(
                    make_relink_project_operation(
                        result.project_path,
                        audio_path,
                        document_path,
                    ),
                    self._project_loaded,
                    "正在重新关联文件…",
                ),
            )
            return
        resources = discover_resources()
        self.tools = discover_ffmpeg(resource_root=resources.root)
        self._preflight = None
        self._setting_paths = True
        try:
            self.audio_path_edit.setText(result.project.audio.path)
            self.docx_path_edit.setText(result.project.document.path)
        finally:
            self._setting_paths = False
        self._accept_project(result.project, result.project_path, result.audio_info)
        self.input_summary_label.setText(
            f"已打开项目 · 音频 {_format_ms(result.audio_info.total_samples, result.audio_info.sample_rate)}"
        )
        self.statusBar().showMessage("项目已打开，已有分析结果已复用")

    def _request_relink_paths(
        self,
        result: LoadedProjectResult,
    ) -> tuple[str, str] | None:
        QMessageBox.information(
            self,
            "重新关联输入文件",
            "项目记录的输入文件已移动或内容不匹配。请重新选择同一份文件，程序会按 SHA-256 指纹核对。",
        )
        audio_path = result.project.audio.path
        document_path = result.project.document.path
        if not result.source_matches.get("audio", False):
            audio_path, _ = QFileDialog.getOpenFileName(
                self,
                "重新选择原音频",
                str(result.project_path.parent),
                "音频文件 (*.mp3 *.wav *.m4a *.aac *.flac);;所有文件 (*)",
            )
            if not audio_path:
                return None
        if not result.source_matches.get("document", False):
            document_path, _ = QFileDialog.getOpenFileName(
                self,
                "重新选择 Word 标注文档",
                str(result.project_path.parent),
                "Word 文档 (*.docx)",
            )
            if not document_path:
                return None
        return audio_path, document_path

    def _accept_project(
        self,
        project: ProjectV1,
        project_path: Path,
        info: AudioInfo,
        envelope: WaveformEnvelope | None = None,
    ) -> None:
        self._source_generation += 1
        source_generation = self._source_generation
        self.project = project
        self._sync_word_workspace_page(force_editor=True)
        self._undo_history.clear()
        self._redo_history.clear()
        self._edit_active = False
        self.project_path = project_path
        self.audio_info = info
        # Do not leave a previous project's waveform interactive while the new
        # audio envelope is being built in the background.
        self.waveform_envelope = envelope
        self.waveform.clear()
        if self.tools is None:
            resources = discover_resources()
            self.tools = discover_ffmpeg(resource_root=resources.root)
        self.output_path_edit.setText(project.export_options.output_directory)
        self._populate_candidates()
        if envelope is not None:
            self.waveform.set_envelope(envelope, info.total_samples)
            self._show_selected_candidate(center=True)
        else:
            self.waveform.set_placeholder("正在后台生成波形…")
            assert self.tools is not None
            self._start_auxiliary(
                make_waveform_operation(info, self.tools),
                lambda result, expected=str(info.path), generation=source_generation: self._waveform_ready(
                    expected,
                    generation,
                    result,
                ),
                report_errors=False,
            )
        self._set_step(4 if project.ready_to_export else 3)
        self._refresh_controls()

    def _waveform_ready(
        self,
        expected_path: str,
        source_generation: int,
        result: object,
    ) -> None:
        if (
            not isinstance(result, WaveformEnvelope)
            or source_generation != self._source_generation
            or self.audio_info is None
            or str(self.audio_info.path) != expected_path
        ):
            return
        self.waveform_envelope = result
        self.waveform.set_envelope(result, self.audio_info.total_samples)
        self._show_selected_candidate(center=True)
        self._refresh_controls()

    def _populate_candidates(self) -> None:
        self.candidate_table.setRowCount(0)
        if self.project is None:
            return
        self.candidate_table.setRowCount(len(self.project.candidates))
        for row, candidate in enumerate(self.project.candidates):
            self._update_candidate_row(row, candidate)
        if self.project.candidates:
            unresolved = next(
                (
                    index
                    for index, candidate in enumerate(self.project.candidates)
                    if candidate.needs_review
                ),
                0,
            )
            self.candidate_table.selectRow(unresolved)
            self.candidate_table.setCurrentCell(unresolved, 0)
        self._update_review_summary()
        self._apply_candidate_filter()

    def _apply_candidate_filter(self, _index: int | None = None) -> None:
        if self.project is None:
            return
        mode = self.candidate_filter.currentIndex()
        first_visible = -1
        for row, candidate in enumerate(self.project.candidates):
            visible = (
                mode == 0
                or (mode == 1 and candidate.needs_review)
                or (mode == 2 and candidate.is_selected)
                or (mode == 3 and candidate.status is CandidateStatus.SKIPPED)
            )
            self.candidate_table.setRowHidden(row, not visible)
            if visible and first_visible < 0:
                first_visible = row
        if (
            self._selected_row >= 0
            and self.candidate_table.isRowHidden(self._selected_row)
            and first_visible >= 0
        ):
            self.candidate_table.selectRow(first_visible)
            self.candidate_table.setCurrentCell(first_visible, 0)

    def _update_candidate_row(self, row: int, candidate: CutCandidate) -> None:
        assert self.audio_info is not None
        start = _format_ms(candidate.effective_start_sample, self.audio_info.sample_rate)
        end = _format_ms(candidate.effective_end_sample, self.audio_info.sample_rate)
        values = (
            candidate.text,
            str(candidate.paragraph_index + 1),
            f"{start} – {end}",
            f"{candidate.confidence * 100:.0f}%",
            STATUS_LABELS[candidate.status],
            "；".join(_reason_text(reason) for reason in candidate.reasons),
        )
        for column, value in enumerate(values):
            item = self.candidate_table.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self.candidate_table.setItem(row, column, item)
            item.setText(value)
            item.setToolTip(value)
            if column == 4:
                if candidate.needs_review:
                    item.setForeground(QColor(theme_color("accent")))
                    item.setBackground(QColor(theme_color("accent_soft")))
                elif candidate.status is CandidateStatus.SKIPPED:
                    item.setForeground(QColor(theme_color("status_neutral_text")))
                    item.setBackground(QColor(theme_color("status_neutral_bg")))
                else:
                    item.setForeground(QColor(theme_color("success")))
                    item.setBackground(QColor(theme_color("success_soft")))

    def _candidate_row_changed(
        self,
        current_row: int,
        _current_column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        self._selected_row = current_row
        self._reset_edit_baseline()
        self._show_selected_candidate(center=True)
        candidate = self._current_candidate()
        if candidate is not None and self.project is not None:
            self.statusBar().showMessage(
                f"候选 {current_row + 1}/{len(self.project.candidates)}：{candidate.text}"
            )
        self._refresh_controls()

    def _current_candidate(self) -> CutCandidate | None:
        if (
            self.project is None
            or self._selected_row < 0
            or self._selected_row >= len(self.project.candidates)
        ):
            return None
        return self.project.candidates[self._selected_row]

    def _show_selected_candidate(self, *, center: bool) -> None:
        candidate = self._current_candidate()
        if candidate is None or self.audio_info is None:
            self.context_label.setText("从左侧选择一个候选项")
            return
        context = (
            html.escape(candidate.context_before)
            + '<span style="color:'
            + theme_color("highlight_text")
            + ";background:"
            + theme_color("highlight_bg")
            + ';font-weight:700;">'
            + html.escape(candidate.text)
            + "</span>"
            + html.escape(candidate.context_after)
        )
        self.context_label.setText(context)
        start_ms = candidate.effective_start_sample * 1000 / self.audio_info.sample_rate
        end_ms = candidate.effective_end_sample * 1000 / self.audio_info.sample_rate
        maximum_ms = self.audio_info.total_samples * 1000 / self.audio_info.sample_rate
        self._updating_boundaries = True
        try:
            with QSignalBlocker(self.start_spin), QSignalBlocker(self.end_spin):
                self.start_spin.setMaximum(maximum_ms)
                self.end_spin.setMaximum(maximum_ms)
                self.start_spin.setValue(start_ms)
                self.end_spin.setValue(end_ms)
        finally:
            self._updating_boundaries = False
        if self.waveform_envelope is not None:
            self.waveform.set_selection(
                candidate.effective_start_sample,
                candidate.effective_end_sample,
                center=center,
            )

    def _boundary_spin_changed(self) -> None:
        if self._updating_boundaries or self.audio_info is None:
            return
        candidate = self._current_candidate()
        if candidate is None:
            return
        self._record_edit_baseline()
        start = round(self.start_spin.value() * self.audio_info.sample_rate / 1000)
        end = round(self.end_spin.value() * self.audio_info.sample_rate / 1000)
        start = max(0, min(self.audio_info.total_samples - 1, start))
        end = max(start + 1, min(self.audio_info.total_samples, end))
        self._updating_boundaries = True
        try:
            with QSignalBlocker(self.start_spin), QSignalBlocker(self.end_spin):
                self.start_spin.setValue(start * 1000 / self.audio_info.sample_rate)
                self.end_spin.setValue(end * 1000 / self.audio_info.sample_rate)
        finally:
            self._updating_boundaries = False
        candidate.final_start_sample = start
        candidate.final_end_sample = end
        self.waveform.set_selection(start, end, center=False)
        self._update_candidate_row(self._selected_row, candidate)
        self._schedule_autosave()

    def _waveform_boundaries_changed(self, start: int, end: int) -> None:
        if self.audio_info is None:
            return
        candidate = self._current_candidate()
        if candidate is None:
            return
        self._record_edit_baseline()
        candidate.final_start_sample = start
        candidate.final_end_sample = end
        self._updating_boundaries = True
        try:
            with QSignalBlocker(self.start_spin), QSignalBlocker(self.end_spin):
                self.start_spin.setValue(start * 1000 / self.audio_info.sample_rate)
                self.end_spin.setValue(end * 1000 / self.audio_info.sample_rate)
        finally:
            self._updating_boundaries = False
        self._update_candidate_row(self._selected_row, candidate)

    def _waveform_boundary_drag_finished(self, _start: int, _end: int) -> None:
        self._reset_edit_baseline()
        self._schedule_autosave()
        self.statusBar().showMessage("切点边界已调整并自动保存")

    def _waveform_view_changed(self, start: int, end: int) -> None:
        total = self.waveform.total_samples
        if total <= 0 or end <= start:
            with QSignalBlocker(self.waveform_scrollbar):
                self.waveform_scrollbar.setRange(0, 0)
                self.waveform_scrollbar.setPageStep(_WAVEFORM_SCROLL_STEPS)
                self.waveform_scrollbar.setValue(0)
            return
        span = min(total, end - start)
        page_step = max(1, round(_WAVEFORM_SCROLL_STEPS * span / total))
        maximum = max(0, _WAVEFORM_SCROLL_STEPS - page_step)
        available_samples = max(0, total - span)
        value = round(start / available_samples * maximum) if available_samples and maximum else 0
        local_pan_samples = min(available_samples, max(1, round(span * 0.12)))
        single_step = (
            max(1, round(local_pan_samples / available_samples * maximum))
            if available_samples and maximum
            else 1
        )
        with QSignalBlocker(self.waveform_scrollbar):
            self.waveform_scrollbar.setRange(0, maximum)
            self.waveform_scrollbar.setPageStep(page_step)
            self.waveform_scrollbar.setSingleStep(single_step)
            self.waveform_scrollbar.setValue(value)

    def _waveform_scroll_changed(self, value: int) -> None:
        maximum = self.waveform_scrollbar.maximum()
        position = value / maximum if maximum else 0.0
        self.waveform.set_scroll_position(position)

    def _nudge_boundary(self, boundary: str, delta_ms: float) -> None:
        if boundary == "start":
            maximum = max(0.0, self.end_spin.value() - 0.001)
            self.start_spin.setValue(max(0.0, min(maximum, self.start_spin.value() + delta_ms)))
        else:
            minimum = self.start_spin.value() + 0.001
            self.end_spin.setValue(max(minimum, self.end_spin.value() + delta_ms))

    def _approve_candidate(self) -> None:
        candidate = self._current_candidate()
        if candidate is None or self.audio_info is None:
            return
        start = round(self.start_spin.value() * self.audio_info.sample_rate / 1000)
        end = round(self.end_spin.value() * self.audio_info.sample_rate / 1000)
        if end <= start:
            QMessageBox.warning(self, "切点无效", "结束切点必须晚于开始切点。")
            return
        self._record_edit_baseline()
        candidate.approve(start, end)
        self.statusBar().showMessage(f"已确认删除：{candidate.text}")
        self._candidate_status_changed(advance=True)

    def _skip_candidate(self) -> None:
        candidate = self._current_candidate()
        if candidate is None:
            return
        self._record_edit_baseline()
        candidate.skip()
        self.statusBar().showMessage(f"已保留：{candidate.text}")
        self._candidate_status_changed(advance=True)

    def _reset_candidate(self) -> None:
        candidate = self._current_candidate()
        if candidate is None:
            return
        self._record_edit_baseline()
        candidate.reset()
        self.statusBar().showMessage(f"已恢复建议切点：{candidate.text}")
        self._candidate_status_changed(advance=False)
        self._show_selected_candidate(center=True)

    def _candidate_status_changed(self, *, advance: bool) -> None:
        candidate = self._current_candidate()
        if candidate is None:
            return
        self._update_candidate_row(self._selected_row, candidate)
        self._update_review_summary()
        self._schedule_autosave()
        self._refresh_controls()
        self._reset_edit_baseline()
        if self.project is not None and self.project.ready_to_export:
            self.statusBar().showMessage("全部候选项已处理，可以导出")
            self._set_step(4)
        elif advance:
            self._select_next_unresolved()

    def _select_next_unresolved(self) -> None:
        if self.project is None or not self.project.unresolved_candidates:
            return
        total = len(self.project.candidates)
        for offset in range(1, total + 1):
            row = (self._selected_row + offset) % total
            if self.project.candidates[row].needs_review:
                self.candidate_table.selectRow(row)
                self.candidate_table.setCurrentCell(row, 0)
                self.candidate_table.scrollToItem(self.candidate_table.item(row, 0))
                return

    def _update_review_summary(self) -> None:
        if self.project is None:
            self.review_summary_label.setText("等待分析")
            return
        unresolved = len(self.project.unresolved_candidates)
        total = len(self.project.candidates)
        selected = len(self.project.selected_candidates)
        removed_samples = sum(
            candidate.effective_end_sample - candidate.effective_start_sample
            for candidate in self.project.selected_candidates
        )
        removed_text = (
            _format_ms(removed_samples, self.project.audio_info.sample_rate)
            if removed_samples
            else "00:00.000"
        )
        if unresolved:
            self.review_summary_label.setText(
                f"{total} 项 · {unresolved} 待复核 · {selected} 已删除 · {removed_text}"
            )
        else:
            self.review_summary_label.setText(
                f"{total} 项 · 已全部处理 · 删除 {selected} 项 / {removed_text}"
            )

    def _start_preview(
        self,
        mode: str,
        candidate_id: str | None = None,
        *,
        trigger_button: QAbstractButton | None = None,
    ) -> None:
        candidate = self._current_candidate()
        if candidate_id is not None and self.project is not None:
            candidate = next(
                (item for item in self.project.candidates if item.id == candidate_id),
                None,
            )
        if (
            candidate is None
            or self.project is None
            or self.audio_info is None
            or self.tools is None
            or not self._preview_directory.isValid()
        ):
            return
        self._stop_preview(clear_queue=candidate_id is None)
        output = Path(self._preview_directory.path())
        if trigger_button is None:
            trigger_button = (
                self.review_all_button
                if candidate_id is not None
                else {
                    "original": self.preview_original_button,
                    "selection": self.preview_selection_button,
                    "edited": self.preview_edited_button,
                }.get(mode)
            )

        def completed(value: object) -> None:
            assert isinstance(value, CandidatePreviewResult)
            paths = {
                "original": value.original_wav_path,
                "selection": value.selection_wav_path,
                "edited": value.edited_wav_path,
            }
            self._play_file(paths[mode])

        self._start_foreground(
            make_preview_operation(
                self.project,
                self.audio_info,
                self.tools,
                candidate.id,
                mode,
                output,
            ),
            completed,
            "正在准备试听…",
            show_progress=False,
            busy_button=trigger_button,
        )

    def _start_review_all(self) -> None:
        if self.project is None:
            return
        self._stop_preview()
        self._preview_queue = [candidate.id for candidate in self.project.selected_candidates]
        if not self._preview_queue:
            QMessageBox.information(self, "暂无切点", "当前没有已批准的删除切点。")
            return
        self._preview_queue_active = True
        self.statusBar().showMessage(f"准备顺序试听 {len(self._preview_queue)} 个已批准切点")
        self._advance_preview_queue()

    def _advance_preview_queue(self) -> None:
        if not self._preview_queue_active:
            return
        if not self._preview_queue:
            self._preview_queue_active = False
            self.statusBar().showMessage("已顺序试听全部已批准切点")
            return
        candidate_id = self._preview_queue.pop(0)
        if self.project is not None:
            row = next(
                (
                    index
                    for index, candidate in enumerate(self.project.candidates)
                    if candidate.id == candidate_id
                ),
                -1,
            )
            if row >= 0:
                self.candidate_table.selectRow(row)
                self.candidate_table.setCurrentCell(row, 0)
        self._start_preview(
            "edited",
            candidate_id,
            trigger_button=self.review_all_button,
        )

    def _play_file(self, path: Path) -> None:
        try:
            self._audio_player.play(path)
        except AudioPlaybackError as exc:
            QMessageBox.warning(self, "无法试听", str(exc))
            self._preview_queue_active = False
            self._preview_queue.clear()
            return
        self.statusBar().showMessage(f"正在试听：{path.name}")

    def _playback_finished(self) -> None:
        if self._preview_queue_active:
            self.statusBar().showMessage("当前切点试听完成，准备下一项…")
            QTimer.singleShot(120, self._advance_preview_queue)
        else:
            self.statusBar().showMessage("试听完成")

    def _playback_failed(self, message: str) -> None:
        self._preview_queue_active = False
        self._preview_queue.clear()
        self.statusBar().showMessage("试听失败，请检查系统音频输出设备")
        QMessageBox.warning(self, "无法试听", message)

    def _stop_preview(self, *, clear_queue: bool = True) -> None:
        if clear_queue:
            self._preview_queue_active = False
            self._preview_queue.clear()
        self._audio_player.stop()
        if clear_queue:
            self.statusBar().showMessage("试听已停止")

    def _start_export(self) -> None:
        if (
            self.project is None
            or self.project_path is None
            or self.tools is None
            or not self.project.ready_to_export
        ):
            return
        output_text = self.output_path_edit.text().strip()
        output = Path(output_text) if output_text else Path(self.project.audio.path).parent
        self.project.export_options.output_directory = str(output.resolve())
        self._start_foreground(
            make_export_operation(self.project, self.project_path, output, self.tools),
            self._export_completed,
            "正在导出…",
        )

    def _export_completed(self, result: object) -> None:
        assert isinstance(result, ExportTaskResult)
        self._set_step(5)
        self.statusBar().showMessage(f"导出完成：{result.wav_path.parent}")
        message = (
            f"已生成：\n{result.wav_path.name}\n{result.mp3_path.name}\n"
            f"{result.csv_path.name}\n{result.project_path.name}"
        )
        box = QMessageBox(QMessageBox.Icon.Information, "导出完成", message, parent=self)
        open_button = box.addButton("打开所在目录", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is open_button:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(result.wav_path.parent)))

    def _schedule_autosave(self) -> None:
        if self.project is not None and self.project_path is not None:
            self._autosave_timer.start()

    def _autosave(self) -> None:
        if self.project is None or self.project_path is None:
            return
        try:
            # Project JSON is small; saving it in order on the GUI thread keeps
            # an older background snapshot from overwriting a newer edit.
            save_project(self.project, self.project_path)
        except Exception as exc:
            self.statusBar().showMessage(f"项目自动保存失败：{exc}", 5000)
        else:
            self.statusBar().showMessage("项目已自动保存", 2000)

    def _start_foreground(
        self,
        operation,
        on_result,
        initial_message: str,
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
        self._foreground_uses_progress = show_progress
        if show_progress:
            task.signals.progress.connect(self._task_progress)
        else:
            self.task_progress.finish()
            if busy_button is not None:
                self._preview_spinner.start(busy_button)
            task.signals.progress.connect(self._task_status)
        task.signals.result.connect(
            lambda value: self._deliver_foreground_result(
                task,
                generation,
                on_result,
                value,
            )
        )
        task.signals.error.connect(self._task_error)
        task.signals.cancelled.connect(self._task_cancelled)
        task.signals.finished.connect(lambda: self._foreground_finished(task))
        if show_progress:
            self.task_progress.start(initial_message)
        self.statusBar().showMessage(initial_message)
        self._refresh_controls()
        self.thread_pool.start(task)

    def _task_status(self, _value: int, message: str) -> None:
        if message:
            self.statusBar().showMessage(message)

    def _deliver_foreground_result(
        self,
        task: BackgroundTask,
        generation: int,
        callback,
        value: object,
    ) -> None:  # type: ignore[no-untyped-def]
        if self._active_task is task and generation == self._foreground_generation:
            callback(value)

    def _start_auxiliary(
        self,
        operation,
        on_result,
        *,
        report_errors: bool,
        on_finished=None,
    ) -> None:  # type: ignore[no-untyped-def]
        task = BackgroundTask(operation)
        self._auxiliary_tasks.add(task)
        task.signals.result.connect(on_result)
        if report_errors:
            task.signals.error.connect(self._task_error)
        task.signals.finished.connect(lambda: self._auxiliary_finished(task, on_finished))
        self.thread_pool.start(task)

    def _task_progress(self, value: int, message: str) -> None:
        if message:
            self.statusBar().showMessage(message)
        self.task_progress.set_progress(value, message)

    def _task_error(self, message: str) -> None:
        self._preview_queue_active = False
        self._preview_queue.clear()
        self.statusBar().showMessage("操作失败")
        QMessageBox.critical(self, "操作失败", message)

    def _task_cancelled(self) -> None:
        self._preview_queue_active = False
        self._preview_queue.clear()
        self.statusBar().showMessage("操作已取消")

    def _foreground_finished(self, task: BackgroundTask) -> None:
        if self._active_task is task:
            self._active_task = None
        self._preview_spinner.stop()
        if self._foreground_uses_progress:
            self.task_progress.finish()
        self._foreground_uses_progress = True
        self._sync_word_workspace_page()
        self._refresh_controls()

    def _auxiliary_finished(self, task: BackgroundTask, on_finished=None) -> None:  # type: ignore[no-untyped-def]
        self._auxiliary_tasks.discard(task)
        if on_finished is not None:
            on_finished()

    def _cancel_active_task(self) -> None:
        if self._active_task is not None:
            self.task_progress.set_message("正在取消 · 若模型正在推理，将在当前片段结束后停止…")
            self.task_progress.set_cancel_enabled(False)
            self._active_task.cancel()

    def _refresh_controls(self) -> None:
        busy = self._active_task is not None
        self.word_import_view.set_busy(busy)
        has_inputs = _is_supported_file(
            self.audio_path_edit.text(),
            _AUDIO_SUFFIXES,
        ) and _is_supported_file(
            self.docx_path_edit.text(),
            _DOCUMENT_SUFFIXES,
        )
        has_project = self.project is not None
        has_candidate = self._current_candidate() is not None
        has_waveform = self.waveform_envelope is not None
        self.audio_path_edit.setEnabled(not busy)
        self.docx_path_edit.setEnabled(not busy)
        self.audio_browse_button.setEnabled(not busy)
        self.docx_browse_button.setEnabled(not busy)
        self.open_project_button.setEnabled(not busy)
        self.preflight_button.setEnabled(not busy and has_inputs)
        self.analyze_button.setEnabled(not busy and self._preflight is not None)
        self.task_progress.set_cancel_enabled(busy)
        self.candidate_table.setEnabled(not busy and has_project)
        waveform_enabled = not busy and has_candidate and has_waveform
        self.waveform.setEnabled(waveform_enabled)
        self.waveform_scrollbar.setEnabled(waveform_enabled)
        self.waveform_zoom_out_button.setEnabled(waveform_enabled)
        self.waveform_zoom_in_button.setEnabled(waveform_enabled)
        self.waveform_focus_button.setEnabled(waveform_enabled)
        for control in (
            self.start_spin,
            self.end_spin,
            self.start_minus_button,
            self.start_plus_button,
            self.end_minus_button,
            self.end_plus_button,
            self.preview_original_button,
            self.preview_selection_button,
            self.preview_edited_button,
            self.reset_button,
            self.skip_button,
            self.approve_button,
        ):
            control.setEnabled(not busy and has_candidate)
        self.stop_preview_button.setEnabled(self._audio_player.is_active)
        self.output_path_edit.setEnabled(not busy and has_project)
        self.output_browse_button.setEnabled(not busy and has_project)
        self.review_all_button.setEnabled(
            not busy and self.project is not None and bool(self.project.selected_candidates)
        )
        self.export_button.setEnabled(
            not busy and self.project is not None and self.project.ready_to_export
        )

    def _set_step(self, completed_step: int) -> None:
        labels = {
            1: "输入准备",
            2: "分析准备",
            3: "逐项复核",
            4: "导出准备",
            5: "处理完成",
        }
        self.workflow_step_label.setText(labels.get(completed_step, "处理中"))
        self.workflow_step_label.setProperty(
            "complete",
            completed_step >= 5,
        )
        self.workflow_step_label.style().unpolish(self.workflow_step_label)
        self.workflow_step_label.style().polish(self.workflow_step_label)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._autosave_timer.stop()
        self._stop_preview()
        self._preview_spinner.stop()
        self.audio_processing_widget.shutdown()
        if self._active_task is not None:
            self._active_task.cancel()
        for task in tuple(self._auxiliary_tasks):
            task.cancel()
        if self.project is not None and self.project_path is not None:
            # The JSON is small and this final synchronous snapshot prevents a
            # last sub-450 ms review action from being lost on an intentional close.
            with suppress(Exception):
                save_project(self.project, self.project_path)
        super().closeEvent(event)


def _make_time_spin(name: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
    spin.setObjectName(name)
    spin.setDecimals(3)
    spin.setRange(0.0, 86_400_000.0)
    spin.setSingleStep(1.0)
    spin.setSuffix(" ms")
    spin.setKeyboardTracking(False)
    spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return spin


def _make_workspace_button(text: str, icon_name: str, tooltip: str) -> QToolButton:
    button = QToolButton()
    button.setObjectName("workspaceNavigationButton")
    button.setProperty("nav", True)
    button.setCheckable(True)
    button.setAutoRaise(True)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
    button.setText(text)
    button.setFixedHeight(62)
    button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    set_button_icon(button, icon_name, size=22, tooltip=tooltip)
    return button


def _is_supported_file(path: str, suffixes: set[str]) -> bool:
    source = Path(path.strip()).expanduser()
    return source.is_file() and source.suffix.casefold() in suffixes


def _make_nudge_button(text: str, tooltip: str) -> QPushButton:
    button = QPushButton(text)
    button.setToolTip(tooltip)
    button.setFixedWidth(74)
    return button


def _format_ms(sample: int, sample_rate: int) -> str:
    milliseconds = sample * 1000 / sample_rate
    total_seconds = milliseconds / 1000
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    return f"{minutes:02d}:{seconds:06.3f}"


def _reason_text(reason: str) -> str:
    return REASON_LABELS.get(reason, reason.replace("_", " "))


__all__ = ["MainWindow"]
