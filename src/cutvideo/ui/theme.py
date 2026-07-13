"""Dual compact themes for the professional audio workspaces."""

from __future__ import annotations

from string import Template

from PySide6.QtWidgets import QApplication

THEME_DARK = "dark"
THEME_LIGHT = "light"

_PALETTES: dict[str, dict[str, str]] = {
    THEME_DARK: {
        "bg": "#101317",
        "chrome": "#13171c",
        "surface": "#191e24",
        "surface_alt": "#222830",
        "field": "#0d1115",
        "field_focus": "#202b33",
        "hover": "#303841",
        "pressed": "#0d1014",
        "text": "#d8dee6",
        "text_strong": "#f2f5f8",
        "muted": "#8b96a2",
        "muted_low": "#68737e",
        "disabled_text": "#58616b",
        "disabled_bg": "#171b20",
        "accent": "#ff9f43",
        "accent_hover": "#ffb469",
        "accent_pressed": "#e88931",
        "accent_text": "#1c1a18",
        "accent_disabled": "#3b3129",
        "accent_disabled_text": "#7c7168",
        "accent_soft": "#34271d",
        "selection": "#284151",
        "transcript_selection": "#f1c84b",
        "transcript_selection_text": "#241c08",
        "table_alt": "#151a1f",
        "progress_track": "#293039",
        "scrollbar": "#46515c",
        "scrollbar_hover": "#687481",
        "danger": "#ff9b98",
        "danger_bg": "#2a1b1e",
        "success": "#67d7b0",
        "success_soft": "#173029",
        "waveform_bg": "#0d1014",
        "waveform": "#65c7d0",
        "waveform_label": "#7f8a95",
        "highlight_bg": "#3a2a1c",
        "highlight_text": "#ffbd7a",
        "status_neutral_bg": "#222830",
        "status_neutral_text": "#9aa5b1",
    },
    THEME_LIGHT: {
        "bg": "#f2f4f7",
        "chrome": "#ffffff",
        "surface": "#ffffff",
        "surface_alt": "#e9edf2",
        "field": "#edf1f5",
        "field_focus": "#e2edf4",
        "hover": "#dde3e9",
        "pressed": "#d4dbe2",
        "text": "#303840",
        "text_strong": "#151a1f",
        "muted": "#687481",
        "muted_low": "#8b96a1",
        "disabled_text": "#a5adb5",
        "disabled_bg": "#e8ebef",
        "accent": "#f28b2e",
        "accent_hover": "#ff9f43",
        "accent_pressed": "#d9751f",
        "accent_text": "#211d19",
        "accent_disabled": "#ead8c8",
        "accent_disabled_text": "#a48c78",
        "accent_soft": "#fff0e2",
        "selection": "#cfe4ef",
        "transcript_selection": "#f4ce57",
        "transcript_selection_text": "#2b2209",
        "table_alt": "#f7f8fa",
        "progress_track": "#d9e0e6",
        "scrollbar": "#b8c1ca",
        "scrollbar_hover": "#929eaa",
        "danger": "#b74343",
        "danger_bg": "#f8e6e6",
        "success": "#168466",
        "success_soft": "#dff2eb",
        "waveform_bg": "#f7f9fb",
        "waveform": "#2d91a0",
        "waveform_label": "#6d7883",
        "highlight_bg": "#ffe8d2",
        "highlight_text": "#8c480d",
        "status_neutral_bg": "#e8edf2",
        "status_neutral_text": "#57636f",
    },
}

_STYLESHEET = Template(
    """
QWidget {
    color: $text;
    font-size: 13px;
}
QMainWindow, QWidget#applicationShell, QWidget#workspaceArea,
QWidget#centralWidget, QWidget#audioProcessingWorkspace,
QWidget#wordWorkspaceStack, QWidget#processingWorkspaceStack,
QStackedWidget#workspaceStack,
QWidget#importLanding, QStackedWidget,
QScrollArea#mainScrollArea, QScrollArea#audioProcessingScrollArea {
    background: $bg;
}
QFrame#workspaceSidebar { background: $chrome; border: none; }
QWidget#utilityBar { background: $bg; border: none; }
QLabel#brandMark { background: transparent; }
QToolButton#workspaceNavigationButton {
    min-width: 64px;
    min-height: 62px;
    padding: 6px 2px 5px 2px;
    color: $muted;
    background: transparent;
    border: none;
    border-radius: 9px;
    font-size: 11px;
    font-weight: 600;
}
QToolButton#workspaceNavigationButton:hover { color: $text_strong; background: $surface_alt; }
QToolButton#workspaceNavigationButton:checked { color: $accent; background: $surface_alt; }
QToolButton#workspaceNavigationButton:focus { color: $text_strong; background: $surface_alt; }
QToolButton#workspaceNavigationButton:checked:focus { color: $accent; }
QFrame#inputCard, QFrame#reviewCard {
    background: $surface;
    border: none;
    border-radius: 10px;
}
QFrame#exportCard { background: transparent; border: none; border-radius: 0; }
QLabel#mutedLabel, QLabel#hintLabel { color: $muted; }
QLabel#fieldLabel { color: $muted; font-size: 12px; font-weight: 650; }
QLabel#sectionTitle { color: $text_strong; font-size: 13px; font-weight: 700; }
QLabel#workflowStepLabel {
    color: $accent;
    background: transparent;
    border: none;
    font-size: 11px;
    font-weight: 750;
}
QLabel#workflowStepLabel[complete="true"] { color: $success; }
QLineEdit, QDoubleSpinBox, QComboBox {
    min-height: 30px;
    padding: 0 9px;
    color: $text_strong;
    background: $field;
    border: none;
    border-radius: 6px;
    selection-background-color: $selection;
}
QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus { background: $field_focus; }
QLineEdit[dropActive="true"] { color: $text_strong; background: $selection; }
QLineEdit:disabled, QDoubleSpinBox:disabled { color: $disabled_text; background: $disabled_bg; }
QTextEdit#processingTranscriptEdit {
    color: $text;
    background: $field;
    border: none;
    border-radius: 7px;
    padding: 12px;
    selection-background-color: $transcript_selection;
    selection-color: $transcript_selection_text;
    font-size: 14px;
}
QPushButton {
    min-height: 30px;
    padding: 0 12px;
    color: $text;
    background: $surface_alt;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}
QPushButton:hover { color: $text_strong; background: $hover; }
QPushButton:pressed { background: $pressed; }
QPushButton:disabled { color: $disabled_text; background: $disabled_bg; }
QPushButton[compact="true"] { min-height: 28px; padding: 0 9px; font-size: 12px; }
QPushButton[primary="true"] {
    color: $accent_text;
    background: $accent;
    border-radius: 7px;
    font-weight: 750;
}
QPushButton[primary="true"]:hover { background: $accent_hover; }
QPushButton[primary="true"]:pressed { background: $accent_pressed; }
QPushButton[primary="true"]:disabled { color: $accent_disabled_text; background: $accent_disabled; }
QPushButton[danger="true"] { color: $danger; background: $danger_bg; }
QPushButton[secondary="true"] { color: $muted; background: transparent; }
QPushButton[secondary="true"]:hover { color: $text_strong; background: $surface_alt; }
QToolButton#iconButton, QToolButton#helpButton, QToolButton#themeButton {
    color: $muted;
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 0;
}
QToolButton#iconButton:hover, QToolButton#helpButton:hover, QToolButton#themeButton:hover,
QToolButton#iconButton:focus, QToolButton#helpButton:focus, QToolButton#themeButton:focus {
    color: $accent;
    background: $surface_alt;
}
QToolButton#iconButton:pressed, QToolButton#helpButton:pressed,
QToolButton#themeButton:pressed { background: $pressed; }
QToolButton#iconButton:disabled { background: transparent; }
QWidget#taskProgressView { background: transparent; border: none; }
QTableWidget {
    color: $text;
    background: $bg;
    alternate-background-color: $table_alt;
    border: none;
    border-radius: 7px;
    gridline-color: transparent;
    selection-background-color: $selection;
    selection-color: $text_strong;
    outline: none;
}
QHeaderView::section {
    color: $muted;
    background: $surface_alt;
    border: none;
    padding: 7px 6px;
    font-size: 11px;
    font-weight: 700;
}
QProgressBar {
    min-height: 4px;
    max-height: 4px;
    background: $progress_track;
    border: none;
    border-radius: 2px;
}
QProgressBar::chunk { background: $accent; border-radius: 2px; }
QScrollBar:horizontal { min-height: 9px; max-height: 9px; margin: 0; background: $bg; border: none; }
QScrollBar::handle:horizontal { min-width: 40px; background: $scrollbar; border: none; border-radius: 4px; }
QScrollBar::handle:horizontal:hover { background: $scrollbar_hover; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
QScrollBar:vertical { min-width: 9px; max-width: 9px; margin: 0; background: $bg; border: none; }
QScrollBar::handle:vertical { min-height: 40px; background: $scrollbar; border: none; border-radius: 4px; }
QScrollBar::handle:vertical:hover { background: $scrollbar_hover; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QSplitter::handle:horizontal { background: $bg; width: 8px; }
QStatusBar { color: $muted; background: $chrome; border: none; padding-left: 6px; }
QToolTip { color: $text_strong; background: $surface_alt; border: none; border-radius: 4px; padding: 5px; }
QMessageBox { background: $surface; }

QWidget#importDropSurface {
    background: transparent;
    border: none;
    border-radius: 0;
}
QWidget#importDropSurface[dropActive="true"] { background: $accent_soft; border-radius: 12px; }
QLabel#importKicker {
    color: $accent;
    font-size: 10px;
    font-weight: 800;
}
QLabel#importTitle { color: $text_strong; font-size: 27px; font-weight: 750; }
QLabel#importSubtitle { color: $muted; font-size: 14px; }
QLabel#importStateLabel { color: $accent; font-size: 12px; font-weight: 650; }
QLabel#importFormats, QLabel#importPrivacy { color: $muted_low; font-size: 11px; }
QPushButton#importFileButton {
    min-height: 62px;
    padding: 0 18px;
    color: $text;
    background: $surface;
    border: none;
    border-radius: 9px;
    text-align: left;
    font-size: 13px;
}
QPushButton#importFileButton:hover { color: $text_strong; background: $field_focus; }
QPushButton#importFileButton[ready="true"] { color: $success; background: $success_soft; }
QPushButton#importProjectButton { min-height: 30px; padding: 0 10px; }

QDialog#usageGuideDialog { background: $bg; }
QScrollArea#usageGuideScroll, QWidget#usageGuideContent { background: transparent; }
QWidget#guideWorkflowPane, QWidget#guideShortcutPane {
    background: $surface;
    border: none;
    border-radius: 8px;
}
QLabel#guideTitle { color: $text_strong; font-size: 22px; font-weight: 750; }
QLabel#guideSubtitle, QLabel#guideFooterNote { color: $muted; }
QLabel#guideKeyHint, QLabel#guideShortcutKey, QLabel#guideAccuracyTip {
    color: $text;
    background: $field;
    border: none;
    border-radius: 5px;
    padding: 5px 8px;
}
QLabel#guideKeyHint, QLabel#guideShortcutKey {
    font-size: 10px;
}
QLabel#guideSectionTitle { color: $text_strong; font-size: 11px; font-weight: 800; padding-top: 3px; }
QLabel#guideStepNumber { color: $accent; font-size: 10px; font-weight: 800; padding-top: 2px; }
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
