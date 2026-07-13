"""Parse the transcript convention used by cutVideo directly from DOCX OOXML.

The parser deliberately does not depend on :mod:`python-docx`.  A DOCX file is a
ZIP container and the information needed here lives in ``word/document.xml``.
Keeping this module small and dependency-free also makes preflight validation
available before the (much larger) speech models are loaded.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

WORDPROCESSINGML: Final = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W: Final = f"{{{WORDPROCESSINGML}}}"
DOCUMENT_XML: Final = "word/document.xml"
MAX_DOCUMENT_XML_BYTES: Final = 64 * 1024 * 1024
DEFAULT_CONTEXT_CHARS: Final = 12

# This intentionally does not match a date or a timestamp embedded in ordinary
# prose.  Elapsed hours may be 00-99, while minutes must be 00-59.
_ANCHOR_RE: Final = re.compile(
    r"^[\t \u00a0\u3000]*\u53d1\u8a00\u4eba"
    r"(?:[\t \u00a0\u3000]+(?P<speaker>[^\d:\r\n]{1,40}?))?"
    r"[\t \u00a0\u3000]+(?:"
    r"(?P<hour>[0-9]{1,3}):(?P<hour_minute>[0-5][0-9]):(?P<hour_second>[0-5][0-9])"
    r"|(?P<minute>[0-9]{1,3}):(?P<second>[0-5][0-9]))"
    r"[\t \u00a0\u3000]*$"
)


class DocxParseError(ValueError):
    """Raised when a DOCX is unreadable or violates the transcript contract."""


@dataclass(frozen=True, slots=True)
class HighlightSpan:
    """One contiguous yellow range, with offsets relative to paragraph text."""

    start: int
    end: int
    text: str
    context_before: str
    context_after: str
    source_highlight_index: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class TranscriptParagraph:
    """Transcript text belonging to one ``发言人 HH:MM`` anchor."""

    index: int
    anchor_ms: int
    text: str
    highlights: list[HighlightSpan] = field(default_factory=list)
    # OOXML paragraph indexes are useful in diagnostics but are intentionally
    # optional so callers can construct this public type with the five core
    # fields only.
    source_paragraph_index: int | None = None
    anchor_source_paragraph_index: int | None = None
    skipped_highlights: list[dict[str, object]] = field(default_factory=list)

    @property
    def anchor_seconds(self) -> float:
        return self.anchor_ms / 1000.0


@dataclass(frozen=True, slots=True)
class ParsedTranscript:
    """A validated transcript in document order."""

    paragraphs: list[TranscriptParagraph]
    source_path: str = ""

    @property
    def highlights(self) -> list[HighlightSpan]:
        return [span for paragraph in self.paragraphs for span in paragraph.highlights]

    @property
    def highlighted_char_count(self) -> int:
        return sum(span.length for span in self.highlights)

    @property
    def skipped_highlights(self) -> list[dict[str, object]]:
        diagnostics: list[dict[str, object]] = []
        for paragraph in self.paragraphs:
            diagnostics.extend(
                {"paragraph_index": paragraph.index, **item}
                for item in paragraph.skipped_highlights
            )
        return diagnostics

    def validate_audio_duration(self, duration_ms: int) -> None:
        """Ensure all anchors fit inside a decoded audio timeline."""

        if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
            raise TypeError("duration_ms must be an integer")
        if duration_ms < 0:
            raise ValueError("duration_ms must not be negative")
        if self.paragraphs and self.paragraphs[-1].anchor_ms > duration_ms:
            anchor = self.paragraphs[-1]
            raise DocxParseError(
                f"第 {anchor.index + 1} 个时间锚点 ({_format_ms(anchor.anchor_ms)}) "
                f"超出音频时长 ({_format_ms(duration_ms)})"
            )


@dataclass(slots=True)
class _RawBodyParagraph:
    source_index: int
    text: str
    highlights: list[tuple[int, int]]


@dataclass(slots=True)
class _RawAnchor:
    source_index: int
    anchor_ms: int
    body: list[_RawBodyParagraph] = field(default_factory=list)


def parse_docx(path: str | Path) -> ParsedTranscript:
    """Parse and validate a highlighted transcript DOCX.

    Only a whole paragraph matching ``发言人 HH:MM`` is an anchor.  All
    non-empty body paragraphs until the next anchor are retained and joined by
    a newline.  Yellow ``w:highlight`` runs that touch are merged.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise DocxParseError(f"DOCX 文件不存在或不可读取: {source}")

    root = _read_document_xml(source)
    body = root.find(f"{W}body")
    if body is None:
        raise DocxParseError("DOCX 缺少 word/document.xml 中的 w:body")

    anchors: list[_RawAnchor] = []
    current: _RawAnchor | None = None
    for source_index, paragraph in enumerate(body.iter(f"{W}p")):
        text, highlights = _read_paragraph(paragraph)
        anchor_ms = _parse_anchor(text)
        if anchor_ms is not None:
            if anchors and anchor_ms <= anchors[-1].anchor_ms:
                raise DocxParseError(
                    f"第 {len(anchors) + 1} 个时间锚点 {_format_ms(anchor_ms)} "
                    f"必须晚于前一个锚点 {_format_ms(anchors[-1].anchor_ms)}"
                )
            current = _RawAnchor(source_index=source_index, anchor_ms=anchor_ms)
            anchors.append(current)
            continue

        # Formatting-only and truly empty paragraphs do not form transcript
        # content.  Preserve whitespace inside non-empty prose for exact offsets.
        if current is not None and text.strip():
            current.body.append(
                _RawBodyParagraph(
                    source_index=source_index,
                    text=text,
                    highlights=highlights,
                )
            )

    if not anchors:
        raise DocxParseError("未找到格式为“发言人 HH:MM”的时间锚点")

    paragraphs = [_build_paragraph(index, anchor) for index, anchor in enumerate(anchors)]
    return ParsedTranscript(paragraphs=paragraphs, source_path=str(source.resolve()))


def _read_document_xml(source: Path) -> ElementTree.Element:
    try:
        with ZipFile(source) as archive:
            try:
                info = archive.getinfo(DOCUMENT_XML)
            except KeyError as exc:
                raise DocxParseError(f"DOCX 缺少 {DOCUMENT_XML}") from exc
            if info.file_size > MAX_DOCUMENT_XML_BYTES:
                raise DocxParseError(
                    f"{DOCUMENT_XML} 过大 ({info.file_size} bytes)，拒绝解析"
                )
            xml = archive.read(info)
    except DocxParseError:
        raise
    except (BadZipFile, OSError, RuntimeError) as exc:
        raise DocxParseError(f"无法读取 DOCX: {exc}") from exc

    try:
        return ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise DocxParseError(f"DOCX XML 已损坏: {exc}") from exc


def _read_paragraph(paragraph: ElementTree.Element) -> tuple[str, list[tuple[int, int]]]:
    chunks: list[str] = []
    spans: list[tuple[int, int]] = []
    offset = 0

    for run in paragraph.iter(f"{W}r"):
        run_text = _read_run_text(run)
        if not run_text:
            continue
        start = offset
        chunks.append(run_text)
        offset += len(run_text)

        highlight = run.find(f"{W}rPr/{W}highlight")
        if highlight is None:
            continue
        value = highlight.get(f"{W}val", "").casefold()
        if value != "yellow":
            continue

        if spans and spans[-1][1] == start:
            spans[-1] = (spans[-1][0], offset)
        else:
            spans.append((start, offset))

    return "".join(chunks), spans


def _read_run_text(run: ElementTree.Element) -> str:
    chunks: list[str] = []
    for node in run.iter():
        if node.tag == f"{W}t":
            chunks.append(node.text or "")
        elif node.tag == f"{W}tab":
            chunks.append("\t")
        elif node.tag in {f"{W}br", f"{W}cr"}:
            chunks.append("\n")
        elif node.tag == f"{W}noBreakHyphen":
            chunks.append("\N{NON-BREAKING HYPHEN}")
        elif node.tag == f"{W}softHyphen":
            chunks.append("\N{SOFT HYPHEN}")
    return "".join(chunks)


def _parse_anchor(text: str) -> int | None:
    match = _ANCHOR_RE.fullmatch(text)
    if match is None:
        return None
    if match.group("hour") is not None:
        hours = int(match.group("hour"))
        minutes = int(match.group("hour_minute"))
        seconds = int(match.group("hour_second"))
        return (hours * 3600 + minutes * 60 + seconds) * 1000
    minutes = int(match.group("minute"))
    seconds = int(match.group("second"))
    return (minutes * 60 + seconds) * 1000


def _build_paragraph(index: int, anchor: _RawAnchor) -> TranscriptParagraph:
    if not anchor.body:
        raise DocxParseError(
            f"第 {index + 1} 个时间锚点 {_format_ms(anchor.anchor_ms)} 后没有正文"
        )

    text_chunks: list[str] = []
    raw_spans: list[tuple[int, int]] = []
    offset = 0
    for body_index, body in enumerate(anchor.body):
        if body_index:
            text_chunks.append("\n")
            offset += 1
        text_chunks.append(body.text)
        raw_spans.extend((offset + start, offset + end) for start, end in body.highlights)
        offset += len(body.text)

    text = "".join(text_chunks)
    highlights: list[HighlightSpan] = []
    skipped_highlights: list[dict[str, object]] = []
    for source_highlight_index, (start, end) in enumerate(raw_spans):
        marked = text[start:end]
        if not marked.strip():
            raise DocxParseError(
                f"第 {index + 1} 段存在空白的黄色标记（字符 {start}:{end}）"
            )
        if not _contains_alignable_text(marked):
            skipped_highlights.append(
                {
                    "source_highlight_index": source_highlight_index,
                    "start": start,
                    "end": end,
                    "text": marked,
                    "reason": "punctuation_only_highlight",
                }
            )
            continue
        highlights.append(
            HighlightSpan(
                start=start,
                end=end,
                text=marked,
                context_before=text[max(0, start - DEFAULT_CONTEXT_CHARS) : start],
                context_after=text[end : end + DEFAULT_CONTEXT_CHARS],
                source_highlight_index=source_highlight_index,
            )
        )

    return TranscriptParagraph(
        index=index,
        anchor_ms=anchor.anchor_ms,
        text=text,
        highlights=highlights,
        source_paragraph_index=anchor.body[0].source_index,
        anchor_source_paragraph_index=anchor.source_index,
        skipped_highlights=skipped_highlights,
    )


def _contains_alignable_text(text: str) -> bool:
    """Return whether a yellow span contains speech-matchable characters."""

    for character in unicodedata.normalize("NFKC", text):
        category = unicodedata.category(character)
        if not character.isspace() and category[0] not in {"P", "S", "C", "Z"}:
            return True
    return False


def _format_ms(milliseconds: int) -> str:
    total_seconds = milliseconds // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


__all__ = [
    "DocxParseError",
    "HighlightSpan",
    "ParsedTranscript",
    "TranscriptParagraph",
    "parse_docx",
]
