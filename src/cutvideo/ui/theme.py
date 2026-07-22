"""Paired daylight and night themes for the ambient audio workspaces."""

from __future__ import annotations

from string import Template

from PySide6.QtWidgets import QApplication

THEME_DARK = "dark"
THEME_LIGHT = "light"

_PALETTES: dict[str, dict[str, str]] = {
    THEME_DARK: {
        "bg": "#0d131a",
        "chrome": "#101720",
        "surface": "#151d27",
        "surface_alt": "#202a35",
        "panel": "rgba(10, 18, 26, 96)",
        "panel_strong": "rgba(9, 16, 23, 122)",
        "sidebar_bg": "rgba(8, 13, 19, 242)",
        "workspace_wash": "rgba(9, 15, 22, 42)",
        "drop_surface": "rgba(6, 13, 20, 82)",
        "drop_border": "rgba(177, 198, 211, 137)",
        "field": "rgba(10, 16, 23, 136)",
        "field_focus": "rgba(20, 31, 41, 196)",
        "hover": "#293542",
        "pressed": "#0b1118",
        "divider": "#1b2631",
        "hairline": "rgba(117, 137, 156, 66)",
        "text": "#c7d0d9",
        "text_strong": "#f4f7fa",
        "muted": "#8996a4",
        "muted_low": "#64717e",
        "disabled_text": "#59636f",
        "disabled_bg": "rgba(20, 27, 35, 209)",
        "accent": "#f0a04a",
        "accent_hover": "#ffb45f",
        "accent_pressed": "#d98935",
        "accent_text": "#1a140e",
        "accent_disabled": "#403426",
        "accent_disabled_text": "#7a6e60",
        "accent_soft": "rgba(240, 160, 74, 33)",
        "selection": "rgba(54, 109, 128, 122)",
        "transcript_selection": "#e8bf4a",
        "transcript_selection_text": "#1f1808",
        "table_alt": "rgba(29, 39, 49, 55)",
        "table_header": "rgba(28, 38, 48, 92)",
        "progress_track": "#26313d",
        "scrollbar": "#3d4654",
        "scrollbar_hover": "#5a6575",
        "danger": "#ef8b88",
        "danger_bg": "rgba(92, 39, 43, 97)",
        "success": "#5ecf9e",
        "success_soft": "rgba(39, 101, 73, 61)",
        "waveform_bg": "#0a1118",
        "waveform": "#5bc4cf",
        "waveform_label": "#71808e",
        "highlight_bg": "#3a2d1c",
        "highlight_text": "#ffc078",
        "status_neutral_bg": "rgba(57, 71, 85, 117)",
        "status_neutral_text": "#93a0b0",
    },
    THEME_LIGHT: {
        "bg": "#eef2f5",
        "chrome": "#f7f9fb",
        "surface": "#ffffff",
        "surface_alt": "#e9eef2",
        "panel": "rgba(255, 255, 255, 70)",
        "panel_strong": "rgba(255, 255, 255, 104)",
        "sidebar_bg": "rgba(248, 250, 252, 240)",
        "workspace_wash": "rgba(245, 248, 250, 36)",
        "drop_surface": "rgba(248, 252, 253, 56)",
        "drop_border": "rgba(86, 112, 129, 102)",
        "field": "rgba(241, 245, 247, 138)",
        "field_focus": "rgba(233, 242, 246, 205)",
        "hover": "#dfe7ec",
        "pressed": "#d4dde4",
        "divider": "#d7e0e6",
        "hairline": "rgba(75, 99, 116, 51)",
        "text": "#374552",
        "text_strong": "#111a22",
        "muted": "#687783",
        "muted_low": "#87949e",
        "disabled_text": "#9da8b0",
        "disabled_bg": "rgba(229, 234, 238, 224)",
        "accent": "#df8a31",
        "accent_hover": "#ee9b44",
        "accent_pressed": "#c97827",
        "accent_text": "#1a140e",
        "accent_disabled": "#ead8c4",
        "accent_disabled_text": "#a48c74",
        "accent_soft": "rgba(223, 138, 49, 33)",
        "selection": "rgba(165, 207, 218, 133)",
        "transcript_selection": "#f0cb5a",
        "transcript_selection_text": "#2a2008",
        "table_alt": "rgba(239, 244, 247, 70)",
        "table_header": "rgba(233, 239, 243, 112)",
        "progress_track": "#d9e0e7",
        "scrollbar": "#b8c1cc",
        "scrollbar_hover": "#929eaa",
        "danger": "#c24a4a",
        "danger_bg": "rgba(220, 91, 91, 31)",
        "success": "#1a8a64",
        "success_soft": "rgba(38, 153, 110, 31)",
        "waveform_bg": "#f8fbfc",
        "waveform": "#2d91a0",
        "waveform_label": "#6d7883",
        "highlight_bg": "#ffe8d0",
        "highlight_text": "#8c4a0e",
        "status_neutral_bg": "rgba(202, 213, 221, 143)",
        "status_neutral_text": "#576370",
    },
}

_STYLESHEET = Template(
    """
    QWidget {
        color: $text;
        font-family: "SF Pro Text", "PingFang SC", "Segoe UI", sans-serif;
        font-size: 13px;
    }
    QMainWindow {
        background: $bg;
    }
    QWidget#applicationShell, QWidget#workspaceArea,
    QWidget#audioProcessingWorkspace, QWidget#wordWorkspaceStack,
    QWidget#processingWorkspaceStack, QStackedWidget#workspaceStack,
    QWidget#importLanding, QStackedWidget,
    QScrollArea#mainScrollArea, QScrollArea#audioProcessingScrollArea {
        background: transparent;
        border: none;
    }
    QWidget#windowDragRegion, QWidget#rightWorkspaceInset {
        background: transparent;
        border: none;
    }
    QWidget#centralWidget, QWidget#processingEditor {
        background: $workspace_wash;
    }
    QScrollArea#mainScrollArea > QWidget > QWidget,
    QScrollArea#audioProcessingScrollArea > QWidget > QWidget {
        background: transparent;
    }
    QFrame#workspaceSidebar {
        background: $sidebar_bg;
        border: none;
        border-right: 1px solid $hairline;
    }
    QWidget#sidebarBrand, QWidget#sidebarUtility, QLabel#brandMark { background: transparent; }
    QLabel#brandName {
        color: $text_strong;
        font-size: 14px;
        font-weight: 750;
    }

    QToolButton#workspaceNavigationButton {
        min-width: 108px;
        min-height: 50px;
        padding: 0 10px;
        color: $muted;
        background: transparent;
        border: none;
        border-left: 3px solid transparent;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 650;
        text-align: left;
    }
QToolButton#workspaceNavigationButton:hover {
    color: $text_strong;
    background: $surface_alt;
}
    QToolButton#workspaceNavigationButton:checked {
        color: $accent;
        background: $accent_soft;
        border-left: 3px solid $accent;
    }
QToolButton#workspaceNavigationButton:focus {
    color: $text_strong;
    background: $surface_alt;
}
    QToolButton#workspaceNavigationButton:checked:focus {
        color: $accent;
        background: $accent_soft;
        border-left: 3px solid $accent;
    }

    QFrame#inputCard {
        background: transparent;
        border: none;
        border-bottom: 1px solid $hairline;
        border-radius: 0;
    }
    QFrame#reviewCard {
        background: $panel;
        border: none;
        border-radius: 0;
    }
    QFrame#transcriptPane {
        background: transparent;
        border: none;
        border-bottom: 1px solid $hairline;
    }
    QFrame#exportCard {
        background: $panel_strong;
        border: none;
        border-top: 1px solid $hairline;
        border-radius: 0;
    }
QLabel#mutedLabel, QLabel#hintLabel { color: $muted; }
QLabel#fieldLabel {
    color: $muted;
    font-size: 12px;
    font-weight: 600;
}
QLabel#workspaceHeaderTitle {
    min-width: 44px;
    color: $accent;
    font-size: 16px;
    font-weight: 750;
}
QLabel#headerStatusDot, QLabel#headerStatus {
    color: $accent;
    font-size: 12px;
    font-weight: 650;
}
QLabel#headerSummary {
    color: $text;
    font-size: 12px;
    font-weight: 550;
}
    QLabel#sectionTitle {
        color: $text_strong;
        background: transparent;
        border-left: 3px solid $accent;
        padding-left: 8px;
        font-size: 14px;
        font-weight: 750;
    }
QLabel#workflowStepLabel {
    color: $accent;
    background: transparent;
    border: none;
    font-size: 11px;
    font-weight: 700;
}
QLabel#workflowStepLabel[complete="true"] { color: $success; }

QLineEdit, QDoubleSpinBox, QComboBox {
    min-height: 32px;
    padding: 0 10px;
    color: $text_strong;
    background: $field;
    border: 1px solid $hairline;
    border-radius: 4px;
    selection-background-color: $selection;
}
QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {
    background: $field_focus;
    border: 1px solid $accent;
}
QFrame#inputCard QLineEdit {
    min-height: 36px;
}
QLineEdit#processingAudioPathEdit {
    color: $text_strong;
    background: transparent;
    border: none;
    border-radius: 0;
    font-weight: 650;
}
QLineEdit#processingAudioPathEdit:focus {
    background: $field;
    border: 1px solid $hairline;
    border-radius: 4px;
}
QLineEdit[dropActive="true"] {
    color: $text_strong;
    background: $selection;
}
QLineEdit:disabled, QDoubleSpinBox:disabled {
    color: $disabled_text;
    background: $disabled_bg;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background: $surface;
    color: $text;
    border: 1px solid $hairline;
    border-radius: 4px;
    selection-background-color: $selection;
    outline: none;
    padding: 4px;
}
QTextEdit#processingTranscriptEdit,
QTextEdit#wordStreamingTranscriptEdit {
    color: $text;
    background: $panel;
    border: none;
    border-radius: 0;
    padding: 12px;
    selection-background-color: $transcript_selection;
    selection-color: $transcript_selection_text;
    font-size: 14px;
}

QPushButton {
    min-height: 32px;
    padding: 0 14px;
    color: $text;
    background: $surface_alt;
    border: 1px solid $hairline;
    border-radius: 4px;
    font-weight: 600;
    font-size: 13px;
}
QPushButton:hover {
    color: $text_strong;
    background: $hover;
}
QPushButton:pressed {
    background: $pressed;
}
QPushButton:disabled {
    color: $disabled_text;
    background: $disabled_bg;
}
QPushButton[compact="true"] {
    min-height: 28px;
    padding: 0 10px;
    font-size: 12px;
    border-radius: 3px;
}
QLabel#controlCaption {
    color: $muted;
    font-size: 12px;
    font-weight: 650;
}
QFrame#previewControlGroup {
    background: $field;
    border: 1px solid $hairline;
    border-radius: 7px;
}
QPushButton[previewAction="true"] {
    min-height: 29px;
    padding: 0 12px;
    color: $muted;
    background: transparent;
    border: none;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 650;
}
QPushButton[previewAction="true"]:hover {
    color: $text_strong;
    background: $hover;
}
QPushButton[previewAction="true"]:pressed {
    background: $pressed;
}
QPushButton[previewAction="true"]:disabled {
    color: $disabled_text;
    background: transparent;
    border: none;
}
QPushButton[previewAction="true"][playing="true"] {
    color: $accent;
    background: $accent_soft;
    border: none;
}
QPushButton[primary="true"] {
    color: $accent_text;
    background: $accent;
    border: 1px solid $accent;
    border-radius: 4px;
    font-weight: 700;
}
QPushButton[primary="true"]:hover {
    background: $accent_hover;
}
QPushButton[primary="true"]:pressed {
    background: $accent_pressed;
}
QPushButton[primary="true"]:disabled {
    color: $accent_disabled_text;
    background: $accent_disabled;
}
QPushButton[danger="true"] {
    color: $danger;
    background: $danger_bg;
    border: 1px solid $danger;
}
QPushButton[danger="true"]:hover {
    background: $hover;
}
QPushButton[accentOutline="true"] {
    color: $accent;
    background: transparent;
    border: 1px solid $accent;
}
QPushButton[accentOutline="true"]:hover {
    color: $accent_hover;
    background: $accent_soft;
    border: 1px solid $accent_hover;
}
QPushButton[accentOutline="true"]:disabled {
    color: $disabled_text;
    background: transparent;
    border: 1px solid $hairline;
}
QPushButton[secondary="true"] {
    color: $muted;
    background: transparent;
    border: 1px solid transparent;
}
QPushButton[secondary="true"]:hover {
    color: $text_strong;
    background: $surface_alt;
}
QFrame#inputCard QPushButton {
    min-height: 36px;
}

QToolButton#iconButton, QToolButton#helpButton, QToolButton#themeButton {
    color: $muted;
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 0;
}
QToolButton#iconButton:hover, QToolButton#helpButton:hover, QToolButton#themeButton:hover,
QToolButton#iconButton:focus, QToolButton#helpButton:focus, QToolButton#themeButton:focus {
    color: $accent;
    background: $surface_alt;
}
QToolButton#iconButton:pressed, QToolButton#helpButton:pressed,
QToolButton#themeButton:pressed {
    background: $pressed;
}
QToolButton#iconButton:disabled {
    background: transparent;
}

QWidget#taskProgressView { background: transparent; border: none; }

QTableWidget {
    color: $text;
    background: $panel;
    alternate-background-color: $table_alt;
    border: none;
    border-radius: 0;
    gridline-color: transparent;
    selection-background-color: $selection;
    selection-color: $text_strong;
    outline: none;
}
QTableWidget::item {
    padding: 5px 10px;
    border-bottom: 1px solid $hairline;
}
QTableWidget::item:selected {
    background: $selection;
    color: $text_strong;
}
QHeaderView::section {
    color: $muted;
    background: $table_header;
    border: none;
    border-bottom: 1px solid $hairline;
    padding: 8px 10px;
    font-size: 11px;
    font-weight: 700;
}
QHeaderView::section:disabled {
    color: $disabled_text;
    background: $table_header;
    border: none;
    border-bottom: 1px solid $hairline;
}
QTableCornerButton::section {
    background: $table_header;
    border: none;
}
QTableCornerButton::section:disabled {
    background: $table_header;
    border: none;
}

QProgressBar {
    min-height: 4px;
    max-height: 4px;
    background: $progress_track;
    border: none;
    border-radius: 2px;
}
QProgressBar::chunk {
    background: $accent;
    border-radius: 2px;
}

QScrollBar:horizontal {
    min-height: 8px;
    max-height: 8px;
    margin: 0;
    background: transparent;
    border: none;
}
QScrollBar::handle:horizontal {
    min-width: 36px;
    background: $scrollbar;
    border: none;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover { background: $scrollbar_hover; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
QScrollBar:vertical {
    min-width: 8px;
    max-width: 8px;
    margin: 0;
    background: transparent;
    border: none;
}
QScrollBar::handle:vertical {
    min-height: 36px;
    background: $scrollbar;
    border: none;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover { background: $scrollbar_hover; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }

QSplitter::handle:horizontal {
    background: transparent;
    border-left: 1px solid $hairline;
    width: 8px;
}
QSplitter::handle:vertical {
    background: transparent;
    border-top: 1px solid $hairline;
    height: 8px;
}
QStatusBar {
    color: $muted;
    background: $chrome;
    border: none;
    padding-left: 8px;
}
QToolTip {
    color: $text_strong;
    background: $surface_alt;
    border: 1px solid $hairline;
    border-radius: 3px;
    padding: 6px 8px;
}
QMessageBox { background: $surface; }

QWidget#importIntroPane, QWidget#importDropColumn { background: transparent; }
QWidget#importDropSurface {
    background-color: $drop_surface;
    border: 1px dashed $drop_border;
    border-radius: 6px;
}
QWidget#importDropSurface[dropActive="true"] {
    background: $accent_soft;
    border: 2px solid $accent;
}
QLabel#importKicker {
    color: $accent;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 0.8px;
}
QLabel#importTitle {
    color: $text_strong;
    font-size: 30px;
    font-weight: 750;
}
QLabel#importSubtitle { color: $muted; font-size: 14px; }
QLabel#importDropTitle {
    color: $text_strong;
    font-size: 16px;
    font-weight: 700;
}
QLabel#importStateLabel {
    color: $accent;
    font-size: 12px;
    font-weight: 650;
}
QLabel#importFormats {
    color: $muted_low;
    font-size: 11px;
}
QLabel#importPrivacy {
    color: $text;
    font-size: 12px;
}
QLabel#importPrivacyIcon { background: transparent; }
QLabel#importFooterPrivacy {
    color: $muted;
    background: transparent;
    font-size: 11px;
}
QPushButton#importFileButton {
    min-height: 58px;
    padding: 0 16px;
    color: $text;
    background: $field;
    border: 1px solid $hairline;
    border-radius: 4px;
    text-align: left;
    font-size: 13px;
}
QPushButton#importFileButton:hover {
    color: $text_strong;
    background: $field_focus;
}
QPushButton#importFileButton[ready="true"] {
    color: $success;
    background: $success_soft;
    border: 1px solid $success;
}
QPushButton#importProjectButton {
    min-height: 32px;
    padding: 0 12px;
    border: 1px solid transparent;
}

QDialog#usageGuideDialog { background: $bg; }
QScrollArea#usageGuideScroll, QWidget#usageGuideContent { background: transparent; }
QWidget#guideWorkflowPane, QWidget#guideShortcutPane {
    background: $surface;
    border: none;
    border-radius: 12px;
}
QLabel#guideTitle {
    color: $text_strong;
    font-size: 20px;
    font-weight: 700;
}
QLabel#guideSubtitle, QLabel#guideFooterNote { color: $muted; }
QLabel#guideKeyHint, QLabel#guideShortcutKey, QLabel#guideAccuracyTip {
    color: $text;
    background: $field;
    border: none;
    border-radius: 6px;
    padding: 5px 8px;
}
QLabel#guideKeyHint, QLabel#guideShortcutKey { font-size: 10px; }
QLabel#guideSectionTitle {
    color: $text_strong;
    font-size: 11px;
    font-weight: 750;
    padding-top: 2px;
}
QLabel#guideStepNumber {
    color: $accent;
    font-size: 10px;
    font-weight: 750;
    padding-top: 2px;
}
QLabel#guideStepCopy, QLabel#guideShortcutDescription { color: $text; }
QPushButton#guideCloseButton { min-width: 86px; }
QLabel#themeTransitionOverlay, QLabel#workspaceTransitionOverlay { background: transparent; }
"""
)


def normalize_theme(value: object) -> str:
    return THEME_LIGHT if str(value).casefold() == THEME_LIGHT else THEME_DARK


def stylesheet_for_theme(theme: str) -> str:
    """Render the complete application stylesheet for a named theme."""

    return _STYLESHEET.substitute(_PALETTES[normalize_theme(theme)])


def current_theme() -> str:
    app = QApplication.instance()
    return normalize_theme(app.property("theme") if app is not None else THEME_DARK)


def theme_color(role: str, theme: str | None = None) -> str:
    """Return one semantic color for custom-painted widgets and rich text."""

    palette = _PALETTES[normalize_theme(theme or current_theme())]
    try:
        return palette[role]
    except KeyError as exc:
        raise KeyError(f"unknown theme color role: {role}") from exc


APP_STYLESHEET = stylesheet_for_theme(THEME_DARK)

__all__ = [
    "APP_STYLESHEET",
    "THEME_DARK",
    "THEME_LIGHT",
    "current_theme",
    "normalize_theme",
    "stylesheet_for_theme",
    "theme_color",
]
