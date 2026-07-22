from __future__ import annotations

import os
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from cutvideo.docx_parser import DocxParseError, parse_docx

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"
SAMPLE_DOCX = Path(os.environ.get("CUTVIDEO_SAMPLE_DOCX", "__missing_sample_document__.docx"))


def _write_docx(
    path: Path,
    paragraphs: list[list[tuple[str, str | None]]],
) -> Path:
    document = ElementTree.Element(f"{W}document")
    body = ElementTree.SubElement(document, f"{W}body")
    for runs in paragraphs:
        paragraph = ElementTree.SubElement(body, f"{W}p")
        for text, highlight in runs:
            run = ElementTree.SubElement(paragraph, f"{W}r")
            if highlight is not None:
                properties = ElementTree.SubElement(run, f"{W}rPr")
                marked = ElementTree.SubElement(properties, f"{W}highlight")
                marked.set(f"{W}val", highlight)
            text_node = ElementTree.SubElement(run, f"{W}t")
            text_node.text = text
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "word/document.xml",
            ElementTree.tostring(document, encoding="utf-8", xml_declaration=True),
        )
    return path


def test_parses_strict_anchors_and_merges_adjacent_yellow_runs(tmp_path: Path) -> None:
    source = _write_docx(
        tmp_path / "fixture.docx",
        [
            [("2026年06月10日 14:27", None)],
            [("发言人", None), ("   00:00", None)],
            [("前文", None), ("删除", "yellow"), ("片段", "yellow"), ("后文", None)],
            [("第二段", None), ("删", "yellow"), ("尾", None)],
            [("发言人 00:10", None)],
            [("普通内容", None), ("红色不删", "red"), ("结束", None)],
        ],
    )

    transcript = parse_docx(source)

    assert len(transcript.paragraphs) == 2
    first = transcript.paragraphs[0]
    assert first.index == 0
    assert first.anchor_ms == 0
    assert first.text == "前文删除片段后文\n第二段删尾"
    assert [(span.start, span.end, span.text) for span in first.highlights] == [
        (2, 6, "删除片段"),
        (12, 13, "删"),
    ]
    assert first.highlights[0].context_before == "前文"
    assert first.highlights[0].context_after.startswith("后文\n第二段")
    assert transcript.paragraphs[1].highlights == []
    assert transcript.highlighted_char_count == 5


@pytest.mark.parametrize("anchor", ["发言人 0:1", "前缀 发言人 00:01", "发言人 00:60"])
def test_rejects_documents_without_an_exact_anchor(tmp_path: Path, anchor: str) -> None:
    source = _write_docx(tmp_path / "invalid.docx", [[(anchor, None)], [("正文", None)]])

    with pytest.raises(DocxParseError, match="未找到"):
        parse_docx(source)


def test_rejects_non_increasing_anchors(tmp_path: Path) -> None:
    source = _write_docx(
        tmp_path / "backwards.docx",
        [
            [("发言人 00:10", None)],
            [("正文一", None)],
            [("发言人 00:09", None)],
            [("正文二", None)],
        ],
    )

    with pytest.raises(DocxParseError, match="必须晚于"):
        parse_docx(source)


def test_rejects_anchor_without_body_and_blank_highlight(tmp_path: Path) -> None:
    no_body = _write_docx(tmp_path / "no_body.docx", [[("发言人 00:00", None)]])
    with pytest.raises(DocxParseError, match="没有正文"):
        parse_docx(no_body)

    blank = _write_docx(
        tmp_path / "blank.docx",
        [[("发言人 00:00", None)], [("正文", None), ("  ", "yellow")]],
    )
    with pytest.raises(DocxParseError, match="空白的黄色标记"):
        parse_docx(blank)


def test_rejects_corrupt_docx(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.docx"
    source.write_bytes(b"not a zip file")

    with pytest.raises(DocxParseError, match="无法读取"):
        parse_docx(source)


def test_validates_last_anchor_against_decoded_duration(tmp_path: Path) -> None:
    source = _write_docx(
        tmp_path / "duration.docx",
        [
            [("发言人 00:00", None)],
            [("一", None)],
            [("发言人 00:10", None)],
            [("二", None)],
        ],
    )
    transcript = parse_docx(source)

    transcript.validate_audio_duration(10_000)
    with pytest.raises(DocxParseError, match="超出音频时长"):
        transcript.validate_audio_duration(9_999)


@pytest.mark.skipif(not SAMPLE_DOCX.is_file(), reason="可选真实样例未通过环境变量提供")
def test_real_supermarket_fixture_regression() -> None:
    transcript = parse_docx(SAMPLE_DOCX)
    highlights = transcript.highlights

    assert len(transcript.paragraphs) == 60
    assert len(highlights) == 38
    assert sum(len(span.text) for span in highlights) == 208
    assert sum(len(span.text) <= 2 for span in highlights) == 15
