"""In-app quick guide for the two CutVideo workspaces."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class UsageGuideDialog(QDialog):
    """Show concise workflows, accuracy tips and active keyboard shortcuts."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("usageGuideDialog")
        self.setWindowTitle("使用说明与快捷键")
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setModal(True)
        self.resize(900, 660)
        self.setMinimumSize(720, 520)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(16)

        header = QHBoxLayout()
        heading = QVBoxLayout()
        heading.setSpacing(3)
        title = QLabel("使用说明")
        title.setObjectName("guideTitle")
        subtitle = QLabel("从左侧选择工作区，再把自动结果当作可试听、可调整的初稿。")
        subtitle.setObjectName("guideSubtitle")
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addLayout(heading)
        header.addStretch(1)
        hint = QLabel("F1  随时打开")
        hint.setObjectName("guideKeyHint")
        header.addWidget(hint, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(header)

        scroll = QScrollArea()
        scroll.setObjectName("usageGuideScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("usageGuideContent")
        columns = QHBoxLayout(content)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(14)
        columns.addWidget(_workflow_pane(), 11)
        columns.addWidget(_shortcut_pane(), 9)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        footer = QHBoxLayout()
        note = QLabel("橙色范围决定实际删除的音频；不确定的切点请先听“剪后”。")
        note.setObjectName("guideFooterNote")
        note.setWordWrap(True)
        close_button = QPushButton("知道了")
        close_button.setObjectName("guideCloseButton")
        close_button.setProperty("primary", True)
        close_button.setDefault(True)
        close_button.clicked.connect(self.accept)
        footer.addWidget(note, 1)
        footer.addWidget(close_button)
        root.addLayout(footer)


def _workflow_pane() -> QWidget:
    pane = QWidget()
    pane.setObjectName("guideWorkflowPane")
    layout = QVBoxLayout(pane)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(10)

    _add_section_title(layout, "WORD 黄标剪辑")
    _add_step(layout, "01", "导入", "在整页导入区拖入音频和黄色高亮 Word，再执行预检。")
    _add_step(layout, "02", "分析", "点击“自动分析”，本地模型会查找高亮文字并给出删除范围。")
    _add_step(layout, "03", "复核", "逐条听原句、待删和剪后；拖动橙色边界可精调时间。")
    _add_step(layout, "04", "导出", "将不该删除的项目设为保留，确认后导出 WAV 与 MP3。")

    _add_section_title(layout, "音频处理")
    _add_step(layout, "01", "完整识别", "整页拖入音频后执行完整识别，转写文本会与波形联动。")
    _add_step(layout, "02", "建立标注", "选中文字后标注删除，或直接在波形上拖出时间范围。")
    _add_step(layout, "03", "精调试听", "文字说明删什么，橙色范围决定从音频中实际删多少。")
    _add_step(layout, "04", "导出", "检查删除标注与总时长，再导出处理完成的音频。")

    accuracy_tip = QLabel(
        "定位更准：黄色高亮尽量是一段完整话语，不要跨说话人；长音频可在 Word 中加入"
        "“发言人 张三 12:34”这类时间锚点。"
    )
    accuracy_tip.setObjectName("guideAccuracyTip")
    accuracy_tip.setWordWrap(True)
    layout.addWidget(accuracy_tip)
    layout.addStretch(1)
    return pane


def _shortcut_pane() -> QWidget:
    pane = QWidget()
    pane.setObjectName("guideShortcutPane")
    layout = QVBoxLayout(pane)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(3)

    _add_shortcut_group(
        layout,
        "通用",
        (
            ("F1", "打开使用说明"),
            ("☀  /  ☾", "右上角切换明暗主题"),
            ("Ctrl / ⌘  1 / 2", "切换左侧工作区"),
            ("Ctrl / ⌘  Z", "撤销当前工作区的编辑"),
            ("Ctrl / ⌘  Shift Z", "重做当前工作区的编辑"),
        ),
    )
    _add_shortcut_group(
        layout,
        "WORD 黄标剪辑",
        (
            ("Space", "试听剪后 / 停止"),
            ("Enter", "确认删除当前切点"),
            ("J  /  K", "下一个 / 上一个切点"),
            ("Ctrl / ⌘  K", "保留当前切点"),
            ("Ctrl / ⌘  R", "恢复自动建议"),
        ),
    )
    _add_shortcut_group(
        layout,
        "音频处理",
        (
            ("Del / ⌫ / Ctrl / ⌘ B", "将选中文字标注为删除"),
            ("Ctrl / ⌘  F", "搜索转写"),
        ),
    )
    _add_shortcut_group(
        layout,
        "波形获得焦点后",
        (
            ("←  /  →", "左右平移"),
            ("+  /  −", "放大 / 缩小"),
            ("0", "回到当前选择范围"),
        ),
    )
    layout.addStretch(1)
    return pane


def _add_section_title(layout: QVBoxLayout, text: str) -> None:
    label = QLabel(text)
    label.setObjectName("guideSectionTitle")
    layout.addWidget(label)


def _add_step(layout: QVBoxLayout, number: str, title: str, detail: str) -> None:
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(8)
    number_label = QLabel(number)
    number_label.setObjectName("guideStepNumber")
    number_label.setFixedWidth(25)
    copy = QLabel(f"{title}  {detail}")
    copy.setObjectName("guideStepCopy")
    copy.setWordWrap(True)
    row_layout.addWidget(number_label, 0, Qt.AlignmentFlag.AlignTop)
    row_layout.addWidget(copy, 1)
    layout.addWidget(row)


def _add_shortcut_group(
    layout: QVBoxLayout,
    title: str,
    rows: Iterable[tuple[str, str]],
) -> None:
    _add_section_title(layout, title)
    for keys, description in rows:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(9)
        key_label = QLabel(keys)
        key_label.setObjectName("guideShortcutKey")
        key_label.setMinimumWidth(126)
        description_label = QLabel(description)
        description_label.setObjectName("guideShortcutDescription")
        description_label.setWordWrap(True)
        row_layout.addWidget(key_label, 0, Qt.AlignmentFlag.AlignTop)
        row_layout.addWidget(description_label, 1)
        layout.addWidget(row)


__all__ = ["UsageGuideDialog"]
