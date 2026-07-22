from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QDir, QMimeData, QPoint, Qt, QUrl
from PySide6.QtGui import QTextCursor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSplitter

from cutvideo.app import create_application
from cutvideo.audio import WaveformEnvelope
from cutvideo.audio_processing import AudioProcessingProject, TranscriptToken
from cutvideo.project import AudioInfo as ProjectAudioInfo
from cutvideo.project import SourceFile
from cutvideo.ui.file_drop_edit import FileDropLineEdit
from cutvideo.ui.main_window import MainWindow
from cutvideo.ui.waveform import WaveformWidget, _waveform_buckets


def test_main_window_constructs_with_core_workflow_controls() -> None:
    app = create_application(["cutvideo-test"])
    assert isinstance(app, QApplication)
    window = MainWindow()
    try:
        assert window.windowTitle() == "CutVideo · 离线音频工作台"
        assert window.workspace_tabs.count() == 2
        assert window.word_workspace_button.text() == "Word 剪辑"
        assert window.audio_workspace_button.text() == "音频处理"
        assert window.audio_processing_widget.add_delete_button.text() == "标记删除"
        assert window.audio_processing_widget.preview_original_button.text() == "原音 ±3s"
        assert window.audio_processing_widget.preview_selection_button.text() == "删除段"
        assert not app.windowIcon().isNull()
        assert window.centralWidget().objectName() == "applicationShell"
        assert window.workspace_tabs.objectName() == "workspaceStack"
        assert Path(window._preview_directory.path()).parent == Path(QDir.tempPath())
        assert window.audio_path_edit.objectName() == "audioPathEdit"
        assert window.docx_path_edit.objectName() == "docxPathEdit"
        assert window.preflight_button.objectName() == "preflightButton"
        assert window.analyze_button.objectName() == "analyzeButton"
        assert window.candidate_table.columnCount() == 6
        assert window.preview_original_button.text() == "听原句"
        assert window.preview_selection_button.text() == "只听待删"
        assert window.preview_edited_button.text() == "听剪后"
        assert window.approve_button.text() == "确认删除"
        assert window.skip_button.text() == "保留此处"
        assert window.start_minus_button.text() == "−20 ms"
        assert window.end_plus_button.text() == "+20 ms"
        assert window.waveform_zoom_out_button.objectName() == "waveformZoomOutButton"
        assert window.waveform_zoom_in_button.objectName() == "waveformZoomInButton"
        assert window.waveform_focus_button.objectName() == "waveformFocusButton"
        assert window.waveform_scrollbar.objectName() == "waveformScrollBar"
        assert window.review_all_button.objectName() == "reviewAllButton"
        assert window.export_button.objectName() == "exportButton"
        assert not window.export_button.isEnabled()
        # Progress views stay hidden until a background task starts.
        assert window.progress_container.isHidden()
        assert window.audio_processing_widget.progress_container.isHidden()
    finally:
        window.close()


def test_file_drop_edit_accepts_one_matching_local_file(tmp_path: Path) -> None:
    create_application(["cutvideo-test"])
    audio = tmp_path / "dragged.MP3"
    audio.write_bytes(b"audio")
    document = tmp_path / "notes.txt"
    document.write_text("text", encoding="utf-8")
    edit = FileDropLineEdit((".mp3", ".wav"))
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(audio))])

    assert edit.accepts_path(audio)
    assert edit._path_from_mime(mime) == audio.resolve()
    assert not edit.accepts_path(document)


def test_progress_controls_do_not_move_either_workspace_layout() -> None:
    app = create_application(["cutvideo-test"])
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    try:
        app.processEvents()
        word_splitter = window.findChild(QSplitter, "reviewSplitter")
        processing_splitter = window.findChild(QSplitter, "audioProcessingSplitter")
        assert word_splitter is not None
        assert processing_splitter is not None
        word_before = word_splitter.geometry()
        window.progress_bar.setVisible(True)
        window.cancel_button.setVisible(True)
        app.processEvents()
        assert window.findChild(QSplitter, "reviewSplitter").geometry() == word_before

        window.workspace_tabs.setCurrentIndex(1)
        app.processEvents()
        processing_before = processing_splitter.geometry()
        workspace = window.audio_processing_widget
        workspace.progress_bar.setVisible(True)
        workspace.cancel_button.setVisible(True)
        app.processEvents()
        assert processing_splitter.geometry() == processing_before
    finally:
        window.close()


def test_audio_processing_text_selection_creates_delete_annotation(tmp_path: Path) -> None:
    create_application(["cutvideo-test"])
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    window = MainWindow()
    try:
        workspace = window.audio_processing_widget
        workspace.project = AudioProcessingProject(
            audio=SourceFile.from_path(source),
            audio_info=ProjectAudioInfo(1_000, 1, 5_000, "wav", "pcm_s16le"),
            tokens=[
                TranscriptToken("你", 1_000, 1_200),
                TranscriptToken("好", 1_200, 1_450),
                TranscriptToken("啊", 1_450, 1_700),
            ],
            segment_starts=[0, 2],
            output_directory=str(tmp_path),
        )
        workspace.project_path = tmp_path / "source.audioprocess.json"
        workspace.waveform_envelope = _interactive_envelope()
        workspace.waveform.set_envelope(workspace.waveform_envelope, 5_000)
        workspace._render_transcript()
        assert workspace.transcript_edit.toPlainText() == "[00:01.000] 你好\n[00:01.450] 啊"
        assert len(workspace._segment_label_ranges) == 2
        cursor = workspace.transcript_edit.textCursor()
        cursor.setPosition(workspace._token_document_ranges[0][0])
        cursor.setPosition(
            workspace._token_document_ranges[1][1],
            QTextCursor.MoveMode.KeepAnchor,
        )
        workspace.transcript_edit.setTextCursor(cursor)
        workspace.waveform.set_selection(900, 1_550, center=False)

        workspace._add_delete_annotation()

        assert len(workspace.project.annotations) == 1
        assert workspace.project.annotations[0].text == "你好"
        assert (
            workspace.project.annotations[0].start_sample,
            workspace.project.annotations[0].end_sample,
        ) == (900, 1_550)
        assert workspace.annotation_table.rowCount() == 1
        transcript_card = workspace.transcript_edit.parentWidget()
        detail_card = workspace.waveform.parentWidget()
        assert not transcript_card.isAncestorOf(workspace.add_delete_button)
        assert detail_card.isAncestorOf(workspace.add_delete_button)
    finally:
        window.close()


def test_waveform_widget_keeps_valid_selection() -> None:
    create_application(["cutvideo-test"])
    widget = WaveformWidget()
    widget.set_selection(20, 10)
    start, end = widget.selection
    assert 0 <= start < end


def _interactive_envelope() -> WaveformEnvelope:
    minimum = np.full(1_000, -0.2, dtype=np.float32)
    maximum = np.full(1_000, 0.2, dtype=np.float32)
    return WaveformEnvelope(
        sample_rate=1_000,
        block_size=10,
        minimum=minimum,
        maximum=maximum,
        rms=np.full(1_000, 0.1, dtype=np.float32),
    )


def test_waveform_view_can_pan_zoom_and_return_to_selection() -> None:
    create_application(["cutvideo-test"])
    widget = WaveformWidget()
    widget.set_envelope(_interactive_envelope(), 10_000)
    widget.set_selection(4_000, 4_500)

    assert widget.view_range == (2_000, 6_500)
    assert widget.selection == (4_000, 4_500)

    widget.pan_by_fraction(0.5)
    assert widget.view_range == (4_250, 8_750)
    assert widget.selection == (4_000, 4_500)

    widget.set_scroll_position(1.0)
    assert widget.view_range == (5_500, 10_000)

    old_span = widget.view_range[1] - widget.view_range[0]
    widget.zoom_in()
    assert widget.view_range[1] - widget.view_range[0] < old_span

    widget.focus_selection()
    assert widget.view_range == (2_000, 6_500)


def test_dragging_boundary_near_edge_keeps_the_view_stable() -> None:
    # The pixel-to-sample mapping must not change mid-drag: auto-panning while
    # a boundary is being dragged causes visible jitter, so the view stays
    # fixed and panning remains an explicit operation.
    create_application(["cutvideo-test"])
    widget = WaveformWidget()
    widget.resize(600, 200)
    widget.set_envelope(_interactive_envelope(), 10_000)
    widget.set_selection(4_000, 4_500)
    widget.show()
    try:
        end_x = round(widget._sample_to_x(4_500))
        QTest.mousePress(
            widget,
            Qt.MouseButton.LeftButton,
            pos=QPoint(end_x, 100),
        )
        QTest.mouseMove(widget, QPoint(590, 100))
        QTest.mouseRelease(
            widget,
            Qt.MouseButton.LeftButton,
            pos=QPoint(590, 100),
        )

        assert widget.selection[1] > 6_300
        assert widget.view_range == (2_000, 6_500)
    finally:
        widget.close()


def test_waveform_can_draw_a_new_selection_in_audio_processing_mode() -> None:
    create_application(["cutvideo-test"])
    widget = WaveformWidget()
    widget.resize(600, 200)
    widget.set_envelope(_interactive_envelope(), 10_000)
    widget.set_selection_drawing_enabled(True)
    widget.show()
    try:
        QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=QPoint(120, 100))
        QTest.mouseMove(widget, QPoint(360, 100))
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=QPoint(360, 100))
        start, end = widget.selection
        assert 0 <= start < end <= 10_000
        assert end - start > 1_000
    finally:
        widget.close()


def test_main_window_scrollbar_tracks_normalized_waveform_view() -> None:
    create_application(["cutvideo-test"])
    window = MainWindow()
    try:
        window.waveform.set_envelope(_interactive_envelope(), 10_000)
        window.waveform.set_selection(4_000, 4_500)

        assert window.waveform_scrollbar.maximum() > 0
        assert window.waveform_scrollbar.singleStep() < window.waveform_scrollbar.pageStep()
        window.waveform_scrollbar.setValue(window.waveform_scrollbar.maximum())
        assert window.waveform.view_range == (5_500, 10_000)

        window.waveform_zoom_in_button.setEnabled(True)
        old_span = window.waveform.view_range[1] - window.waveform.view_range[0]
        window.waveform_zoom_in_button.click()
        assert window.waveform.view_range[1] - window.waveform.view_range[0] < old_span
    finally:
        window.close()


def test_full_duration_waveform_is_pixel_bounded_without_losing_peaks() -> None:
    minimum = np.zeros(500_000, dtype=np.float32)
    maximum = np.zeros(500_000, dtype=np.float32)
    minimum[123_456] = -0.9
    maximum[345_678] = 0.8
    envelope = WaveformEnvelope(48_000, 240, minimum, maximum, np.zeros_like(minimum))

    indexes, visible_minimum, visible_maximum = _waveform_buckets(
        envelope, 0, envelope.points, max_points=2_000
    )

    assert len(indexes) <= 2_000
    assert visible_minimum.min() == pytest.approx(-0.9)
    assert visible_maximum.max() == pytest.approx(0.8)
