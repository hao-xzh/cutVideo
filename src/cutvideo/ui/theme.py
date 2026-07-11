"""Restrained, high-contrast application theme."""

from __future__ import annotations

APP_STYLESHEET = """
QWidget {
    color: #243043;
    font-family: "Segoe UI", "Microsoft YaHei UI", "PingFang SC", sans-serif;
    font-size: 13px;
}
QMainWindow, QWidget#centralWidget, QWidget#audioProcessingWorkspace {
    background: #f4f6f8;
}
QTabWidget#workspaceTabs::pane {
    border: none;
    background: #f4f6f8;
}
QTabWidget#workspaceTabs QTabBar::tab {
    min-width: 150px;
    min-height: 35px;
    padding: 0 18px;
    color: #657287;
    background: #edf1f5;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}
QTabWidget#workspaceTabs QTabBar::tab:selected {
    color: #245ec7;
    background: #ffffff;
    border-bottom-color: #2f68d2;
}
QFrame#headerCard, QFrame#inputCard, QFrame#reviewCard, QFrame#exportCard {
    background: #ffffff;
    border: 1px solid #dfe4ea;
    border-radius: 10px;
}
QLabel#titleLabel {
    color: #152033;
    font-size: 22px;
    font-weight: 650;
}
QLabel#subtitleLabel, QLabel#mutedLabel, QLabel#hintLabel {
    color: #697586;
}
QLabel#shortcutBadge {
    color: #55647a;
    background: #f0f3f7;
    border: 1px solid #dbe1e8;
    border-radius: 6px;
    padding: 6px 10px;
    font-family: "Cascadia Mono", "SFMono-Regular", monospace;
}
QLabel#sectionTitle {
    color: #1c283b;
    font-size: 15px;
    font-weight: 650;
}
QLabel[step="true"] {
    color: #728096;
    background: #eef1f5;
    border-radius: 12px;
    padding: 4px 11px;
    font-weight: 600;
}
QLabel[stepState="active"] {
    color: #245ec7;
    background: #e9f0ff;
}
QLabel[stepState="done"] {
    color: #287253;
    background: #e8f5ef;
}
QLineEdit, QDoubleSpinBox, QComboBox {
    min-height: 32px;
    padding: 0 9px;
    background: #ffffff;
    border: 1px solid #cfd6df;
    border-radius: 6px;
    selection-background-color: #bfd2fb;
}
QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #3d73d8;
}
QLineEdit:disabled, QDoubleSpinBox:disabled {
    color: #8c96a5;
    background: #f3f5f7;
}
QTextEdit#processingTranscriptEdit {
    color: #263348;
    background: #ffffff;
    border: 1px solid #d9e0e8;
    border-radius: 7px;
    padding: 14px;
    selection-background-color: #bfd2fb;
    selection-color: #1b2a40;
    font-size: 15px;
}
QTextEdit#processingTranscriptEdit:focus {
    border-color: #7d9edb;
}
QPushButton {
    min-height: 32px;
    padding: 0 14px;
    color: #263348;
    background: #ffffff;
    border: 1px solid #cbd3dd;
    border-radius: 6px;
    font-weight: 550;
}
QPushButton:hover {
    background: #f3f6fa;
    border-color: #aeb9c7;
}
QPushButton:pressed {
    background: #e9edf2;
}
QPushButton:disabled {
    color: #9aa3b0;
    background: #f0f2f4;
    border-color: #e0e4e8;
}
QPushButton[compact="true"] {
    min-height: 27px;
    padding: 0 10px;
    font-weight: 550;
}
QPushButton[primary="true"] {
    color: #ffffff;
    background: #2f68d2;
    border-color: #2f68d2;
    font-weight: 650;
}
QPushButton[primary="true"]:hover { background: #285dbd; }
QPushButton[primary="true"]:pressed { background: #2453a8; }
QPushButton[primary="true"]:disabled {
    color: #9aa3b0;
    background: #e9edf2;
    border-color: #dfe4ea;
}
QPushButton[danger="true"] {
    color: #a14237;
    background: #fff8f6;
    border-color: #e4c1bb;
}
QTableWidget {
    background: #ffffff;
    alternate-background-color: #f8fafc;
    border: 1px solid #dfe4ea;
    border-radius: 7px;
    gridline-color: #edf0f3;
    selection-background-color: #e7efff;
    selection-color: #1e2c41;
    outline: none;
}
QHeaderView::section {
    color: #596579;
    background: #f4f6f9;
    border: none;
    border-bottom: 1px solid #dfe4ea;
    padding: 8px 7px;
    font-weight: 650;
}
QProgressBar {
    min-height: 7px;
    max-height: 7px;
    background: #e8ecf1;
    border: none;
    border-radius: 3px;
}
QProgressBar::chunk {
    background: #3b72d6;
    border-radius: 3px;
}
QScrollBar:horizontal {
    min-height: 10px;
    max-height: 10px;
    margin: 0;
    background: #edf1f5;
    border: none;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    min-width: 34px;
    background: #aeb8c6;
    border: none;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover { background: #8997aa; }
QScrollBar:horizontal:disabled { background: #f1f3f6; }
QScrollBar::handle:horizontal:disabled { background: #d9dfe7; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
    height: 0;
    background: transparent;
    border: none;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
}
QSplitter::handle { background: transparent; width: 10px; }
QStatusBar { color: #627087; background: #f4f6f8; }
QToolTip {
    color: #ffffff;
    background: #28354a;
    border: none;
    padding: 5px;
}
"""

__all__ = ["APP_STYLESHEET"]
