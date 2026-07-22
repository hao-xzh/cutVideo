from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cutvideo.alignment import (
    ASR_ALIGNER_UNAVAILABLE,
    BOUNDARY_DISAGREEMENT,
    COARSE_TIMESTAMP,
    FALLBACK_ALIGNMENT,
    FORCE_ALIGNER_UNAVAILABLE,
    INSUFFICIENT_COVERAGE,
    MAX_HIGHLIGHT_INTERNAL_GAP_MS,
    REPEATED_CONTEXT,
    SHORT_HIGHLIGHT,
    STATUS_AUTO_APPROVED,
    STATUS_NEEDS_REVIEW,
    TIMESTAMP_PRECISION_TOKEN,
    AlignmentTrack,
    FunASRAsrAligner,
    RecognizedToken,
    TimedSpan,
    _range_from_track,
    align_transcript,
    alignment_track_from_model_result,
    normalize_with_mapping,
    recognition_tokens_from_model_result,
    refine_recognition_tokens,
)
from cutvideo.model_runtime import ModelUnavailableError, load_funasr_model


@dataclass
class Highlight:
    start: int
    end: int
    text: str
    context_before: str = ""
    context_after: str = ""


@dataclass
class Paragraph:
    index: int
    anchor_ms: int
    text: str
    highlights: list[Highlight] = field(default_factory=list)


@dataclass
class Parsed:
    paragraphs: list[Paragraph]


@dataclass
class Audio:
    path: Path
    sample_rate: int = 1_000
    total_samples: int = 10_000


class RecordingAligner:
    def __init__(self, *, delta_ms: float = 0, coverage: float = 1.0) -> None:
        self.delta_ms = delta_ms
        self.coverage = coverage
        self.calls: list[tuple[int, int]] = []

    def align(
        self,
        *,
        audio_path: str | Path,
        transcript: str,
        window_start_ms: int,
        window_end_ms: int,
    ) -> AlignmentTrack:
        del audio_path
        self.calls.append((window_start_ms, window_end_ms))
        normalized = normalize_with_mapping(transcript).text
        spans = tuple(
            TimedSpan(
                index,
                index + 1,
                1_000 + index * 100 + self.delta_ms,
                1_100 + index * 100 + self.delta_ms,
                model_confidence=0.9,
            )
            for index in range(len(normalized))
        )
        return AlignmentTrack(spans, "fake", self.coverage)


def _fixture(highlight: Highlight, text: str = "甲乙删除词丙丁") -> tuple[Parsed, Audio]:
    return (
        Parsed(
            [
                Paragraph(0, 1_000, text, [highlight]),
                Paragraph(1, 5_000, "收尾"),
            ]
        ),
        Audio(Path("not-read-by-fakes.wav")),
    )


def test_normalization_preserves_original_indexes() -> None:
    normalized = normalize_with_mapping("Ａ，你 A-B！")

    assert normalized.text == "a你ab"
    assert normalized.normalized_to_original == (0, 2, 4, 6)
    assert normalized.normalized_range(2, 5) == (1, 3)
    assert normalized.normalized_range(1, 2) is None


def test_two_agreeing_models_auto_approve_and_use_pcm_samples() -> None:
    parsed, audio = _fixture(Highlight(2, 5, "删除词", "甲乙", "丙丁"))
    force = RecordingAligner()
    asr = RecordingAligner(delta_ms=40)

    candidates = align_transcript(parsed, audio, force, asr)

    # ASR first finds the paragraph globally, then re-recognizes its local
    # waveform to remove long-form timestamp drift.
    assert asr.calls == [(0, 10_000), (0, 7_540)]
    assert force.calls == [(590, 2_190)]
    assert len(candidates) == 1
    candidate = candidates[0]
    # Location comes from actual speech; agreeing independent boundaries are
    # averaged to reduce one-sided timestamp bias.
    assert (candidate.proposed_start_sample, candidate.proposed_end_sample) == (1_220, 1_520)
    assert candidate.status == STATUS_AUTO_APPROVED
    assert candidate.requires_review is False
    assert candidate.confidence >= 0.82
    assert candidate.reasons == []
    assert candidate.suggested_start_sample == candidate.proposed_start_sample
    assert candidate.text == "删除词"


def test_model_windows_are_padded_and_clipped_to_decoded_duration() -> None:
    parsed = Parsed(
        [
            Paragraph(0, 100, "甲删除词", [Highlight(1, 4, "删除词")]),
            Paragraph(1, 1_000, "结尾"),
        ]
    )
    audio = Audio(Path("unused.wav"), sample_rate=1_000, total_samples=1_200)
    force = RecordingAligner()

    align_transcript(parsed, audio, force_aligner=force)

    assert force.calls == [(0, 1_200)]


def test_global_asr_boundary_is_not_clamped_to_document_anchor() -> None:
    class LateSpeechAligner(RecordingAligner):
        def align(self, **kwargs: object) -> AlignmentTrack:
            self.calls.append(
                (int(kwargs["window_start_ms"]), int(kwargs["window_end_ms"]))
            )
            normalized = normalize_with_mapping(str(kwargs["transcript"])).text
            return AlignmentTrack(
                tuple(
                    TimedSpan(index, index + 1, 7_000 + index * 100, 7_100 + index * 100)
                    for index in range(len(normalized))
                ),
                "late-real-speech",
                1.0,
            )

    parsed, audio = _fixture(Highlight(2, 5, "删除词", "甲乙", "丙丁"))
    asr = LateSpeechAligner()

    candidate = align_transcript(parsed, audio, asr_aligner=asr)[0]

    assert asr.calls == [(0, 10_000), (1_200, 10_000)]
    assert (candidate.proposed_start_sample, candidate.proposed_end_sample) == (7_200, 7_500)
    assert candidate.proposed_start_sample > 5_000


def test_local_recognition_replaces_coarse_long_form_timestamp() -> None:
    class CoarseThenFineAligner:
        def __init__(self) -> None:
            self.calls = 0

        def align(self, **kwargs: object) -> AlignmentTrack:
            self.calls += 1
            normalized = normalize_with_mapping(str(kwargs["transcript"])).text
            base = 3_000 if self.calls == 1 else 7_000
            return AlignmentTrack(
                tuple(
                    TimedSpan(index, index + 1, base + index * 100, base + (index + 1) * 100)
                    for index in range(len(normalized))
                ),
                "coarse" if self.calls == 1 else "fine",
                1.0,
            )

    parsed, audio = _fixture(Highlight(2, 5, "删除词", "甲乙", "丙丁"))
    asr = CoarseThenFineAligner()

    candidate = align_transcript(parsed, audio, asr_aligner=asr)[0]

    assert asr.calls == 2
    assert (candidate.proposed_start_sample, candidate.proposed_end_sample) == (7_200, 7_500)


def test_document_anchor_shift_does_not_move_audio_text_match() -> None:
    highlight = Highlight(2, 5, "删除词", "甲乙", "丙丁")
    first, audio = _fixture(highlight)
    shifted, _ = _fixture(highlight)
    shifted.paragraphs[0].anchor_ms = 3_500
    shifted.paragraphs[1].anchor_ms = 8_500

    first_candidate = align_transcript(
        first,
        audio,
        asr_aligner=RecordingAligner(delta_ms=40),
    )[0]
    shifted_candidate = align_transcript(
        shifted,
        audio,
        asr_aligner=RecordingAligner(delta_ms=40),
    )[0]

    assert (
        first_candidate.proposed_start_sample,
        first_candidate.proposed_end_sample,
    ) == (
        shifted_candidate.proposed_start_sample,
        shifted_candidate.proposed_end_sample,
    )


def test_forced_alignment_uses_two_short_audio_located_contexts() -> None:
    text = "前" * 25 + "删除词" + "后" * 25
    parsed = Parsed(
        [
            Paragraph(0, 100, text, [Highlight(25, 28, "删除词")]),
            Paragraph(1, 9_000, "收尾"),
        ]
    )
    audio = Audio(Path("unused.wav"))
    asr = RecordingAligner()

    class StableContextAligner:
        def __init__(self) -> None:
            self.transcripts: list[str] = []

        def align(self, **kwargs: object) -> AlignmentTrack:
            transcript = str(kwargs["transcript"])
            self.transcripts.append(transcript)
            normalized = normalize_with_mapping(transcript).text
            target = normalized.index("删除词")
            spans = tuple(
                TimedSpan(
                    index,
                    index + 1,
                    3_500 + (index - target) * 100,
                    3_600 + (index - target) * 100,
                )
                for index in range(len(normalized))
            )
            return AlignmentTrack(spans, "stable-context", 1.0)

    force = StableContextAligner()
    candidate = align_transcript(parsed, audio, force, asr)[0]

    assert len(force.transcripts) == 2
    assert all("删除词" in transcript for transcript in force.transcripts)
    assert [len(normalize_with_mapping(item).text) for item in force.transcripts] == [23, 43]
    assert all(len(item) < len(text) for item in force.transcripts)
    assert (candidate.proposed_start_sample, candidate.proposed_end_sample) == (3_500, 3_800)


def test_two_stable_context_passes_supply_review_boundary_when_asr_edges_disagree() -> None:
    text = "前" * 25 + "删除词" + "后" * 25
    parsed = Parsed(
        [
            Paragraph(0, 100, text, [Highlight(25, 28, "删除词")]),
            Paragraph(1, 9_000, "收尾"),
        ]
    )

    class StableContextAligner:
        def align(self, **kwargs: object) -> AlignmentTrack:
            normalized = normalize_with_mapping(str(kwargs["transcript"])).text
            target = normalized.index("删除词")
            return AlignmentTrack(
                tuple(
                    TimedSpan(
                        index,
                        index + 1,
                        3_500 + (index - target) * 100,
                        3_600 + (index - target) * 100,
                    )
                    for index in range(len(normalized))
                ),
                "stable-context",
                1.0,
            )

    candidate = align_transcript(
        parsed,
        Audio(Path("unused.wav")),
        StableContextAligner(),
        RecordingAligner(delta_ms=400),
    )[0]

    assert BOUNDARY_DISAGREEMENT in candidate.reasons
    assert candidate.requires_review
    assert (candidate.proposed_start_sample, candidate.proposed_end_sample) == (3_500, 3_800)


def test_missing_models_use_low_confidence_character_ratio_fallback() -> None:
    text = "甲乙删除词丙丁"
    parsed = Parsed([Paragraph(0, 0, text, [Highlight(2, 5, "删除词")])])
    audio = Audio(Path("unused.mp3"))

    candidate = align_transcript(parsed, audio)[0]

    assert candidate.proposed_start_sample == round(10_000 * 2 / 7)
    assert candidate.proposed_end_sample == round(10_000 * 5 / 7)
    # Base fallback confidence 0.15 minus the missing-neighbor-guard penalty.
    assert candidate.confidence == 0.07
    assert candidate.status == STATUS_NEEDS_REVIEW
    assert candidate.requires_review is True
    assert FALLBACK_ALIGNMENT in candidate.reasons
    assert FORCE_ALIGNER_UNAVAILABLE in candidate.reasons
    assert ASR_ALIGNER_UNAVAILABLE in candidate.reasons


def test_short_repeated_and_disagreeing_ranges_always_require_review() -> None:
    # Short highlight rule is independent of otherwise perfect model output.
    parsed, audio = _fixture(Highlight(2, 4, "删除", "甲乙", "词丙"))
    candidate = align_transcript(parsed, audio, RecordingAligner(), RecordingAligner())[0]
    assert SHORT_HIGHLIGHT in candidate.reasons
    assert candidate.requires_review

    repeated_text = "开场删除词中间删除词结尾"
    repeated = Highlight(2, 5, "删除词")
    parsed, audio = _fixture(repeated, repeated_text)
    candidate = align_transcript(parsed, audio, RecordingAligner(), RecordingAligner())[0]
    assert REPEATED_CONTEXT in candidate.reasons
    assert candidate.requires_review

    parsed, audio = _fixture(Highlight(2, 5, "删除词", "甲乙", "丙丁"))
    candidate = align_transcript(
        parsed,
        audio,
        RecordingAligner(),
        RecordingAligner(delta_ms=121),
    )[0]
    assert BOUNDARY_DISAGREEMENT in candidate.reasons
    assert candidate.requires_review


def test_unique_context_disambiguates_repeated_target() -> None:
    text = "开场删除词中间删除词结尾"
    parsed, audio = _fixture(Highlight(2, 5, "删除词", "开场", "中间"), text)

    candidate = align_transcript(parsed, audio, RecordingAligner(), RecordingAligner())[0]

    assert REPEATED_CONTEXT not in candidate.reasons
    assert candidate.status == STATUS_AUTO_APPROVED


def test_same_context_repeated_in_another_paragraph_requires_review() -> None:
    text = "开场删除词结束"
    parsed = Parsed(
        [
            Paragraph(0, 0, text, [Highlight(2, 5, "删除词", "开场", "结束")]),
            Paragraph(1, 5_000, text),
        ]
    )

    candidate = align_transcript(
        parsed,
        Audio(Path("unused.wav")),
        RecordingAligner(),
        RecordingAligner(),
    )[0]

    assert REPEATED_CONTEXT in candidate.reasons
    assert candidate.status == STATUS_NEEDS_REVIEW


def test_low_model_coverage_requires_review() -> None:
    parsed, audio = _fixture(Highlight(2, 5, "删除词", "甲乙", "丙丁"))

    candidate = align_transcript(
        parsed,
        audio,
        RecordingAligner(coverage=0.5),
        RecordingAligner(),
    )[0]

    assert INSUFFICIENT_COVERAGE in candidate.reasons
    assert candidate.requires_review


def test_missing_local_span_is_reported_as_insufficient_coverage() -> None:
    class MissingHighlightAligner(RecordingAligner):
        def align(self, **kwargs: object) -> AlignmentTrack:
            full = super().align(**kwargs)
            return AlignmentTrack(full.spans[:2], "partial", 1.0)

    parsed, audio = _fixture(Highlight(2, 5, "删除词", "甲乙", "丙丁"))
    candidate = align_transcript(
        parsed,
        audio,
        MissingHighlightAligner(),
        RecordingAligner(),
    )[0]

    assert INSUFFICIENT_COVERAGE in candidate.reasons
    assert candidate.requires_review


def test_model_exception_falls_back_without_automatic_approval() -> None:
    class BrokenAligner:
        def align(self, **kwargs: object) -> AlignmentTrack:
            del kwargs
            raise ModelUnavailableError("model missing")

    parsed, audio = _fixture(Highlight(2, 5, "删除词"))
    candidate = align_transcript(parsed, audio, BrokenAligner(), BrokenAligner())[0]

    assert FALLBACK_ALIGNMENT in candidate.reasons
    assert candidate.status == STATUS_NEEDS_REVIEW


def test_funasr_result_is_normalized_and_mapped_to_reference() -> None:
    result = [{"text": "你，好 A", "timestamp": [[0, 100], [100, 200], [200, 300]]}]

    track = alignment_track_from_model_result(
        result,
        "你，好 A！",
        engine="fake",
        time_offset_ms=500,
    )

    assert track.coverage == 1.0
    assert [(span.normalized_start, span.start_ms, span.end_ms) for span in track.spans] == [
        (0, 500, 600),
        (1, 600, 700),
        (2, 700, 800),
    ]


def test_asr_homophone_substitution_keeps_highlight_character_timestamp() -> None:
    # Paraformer commonly emits a plausible homophone (卖) for the DOCX word
    # (买).  Exact context remains authoritative while the one-character gap is
    # recovered with a tone-free pinyin key.
    result = [
        {
            "text": "他去超市卖菜",
            "timestamp": [[0, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]],
        }
    ]

    track = alignment_track_from_model_result(
        result,
        "他去超市买菜",
        engine="paraformer-test",
    )

    assert track.coverage == 1.0
    highlighted_span = next(span for span in track.spans if span.normalized_start == 4)
    assert (highlighted_span.start_ms, highlighted_span.end_ms) == (400, 500)
    assert 0.80 <= highlighted_span.confidence < 1.0


def test_raw_recognition_tokens_keep_absolute_timestamps() -> None:
    tokens = recognition_tokens_from_model_result(
        [{"text": "你好", "timestamp": [[0, 100], [100, 240]]}],
        time_offset_ms=2_000,
    )

    assert [(item.text, item.start_ms, item.end_ms) for item in tokens] == [
        ("你", 2_000, 2_100),
        ("好", 2_100, 2_240),
    ]


def test_highlight_range_rejects_sparse_matches_across_large_audio_hole() -> None:
    track = AlignmentTrack(
        (
            TimedSpan(0, 1, 1_000, 1_100),
            TimedSpan(1, 2, 4_000, 4_100),
        ),
        "sparse-asr",
        1.0,
    )

    # Wide paragraph/context lookup may legitimately span pauses.
    wide = _range_from_track(track, 0, 2)
    assert wide is not None
    assert (wide.start_ms, wide.end_ms, wide.coverage) == (1_000, 4_100, 1.0)
    # A single yellow deletion must never silently include the intervening
    # 2.9 seconds just because one character matched on each side.
    assert (
        _range_from_track(
            track,
            0,
            2,
            max_internal_gap_ms=MAX_HIGHLIGHT_INTERNAL_GAP_MS,
        )
        is None
    )


def test_character_timestamps_take_precedence_over_sentence_info() -> None:
    result = [
        {
            "text": "甲乙",
            "timestamp": [[10, 20], [20, 30]],
            "sentence_info": [{"text": "甲乙", "start": 100, "end": 200}],
        }
    ]

    track = alignment_track_from_model_result(result, "甲乙", engine="fake")

    assert [(span.start_ms, span.end_ms) for span in track.spans] == [
        (10, 20),
        (20, 30),
    ]


def test_empty_asr_text_never_borrows_document_reference() -> None:
    track = alignment_track_from_model_result(
        [{"text": "", "timestamp": [[10, 20], [20, 30]]}],
        "甲乙",
        engine="empty-asr",
    )

    assert track.coverage == 0.0
    assert track.spans == ()


def test_fa_serialized_text_uses_authoritative_reference_tokens() -> None:
    transcript = "\u4e00\u4e2a\u4e1c\u592a\u5e73\u6d0b"
    result = [
        {
            "text": (
                " 0.000 0.380;\u4e00 0.380 0.560;\u4e2a 0.560 0.800;"
                "\u4e1c 0.800 0.980;\u592a 0.980 1.140;\u5e73 1.140 1.260;"
                "\u6d0b 1.260 1.440; 1.440 1.600;"
            ),
            "timestamp": [
                [380, 560],
                [560, 800],
                [800, 980],
                [980, 1140],
                [1140, 1260],
                [1260, 1440],
            ],
        }
    ]

    track = alignment_track_from_model_result(
        result,
        transcript,
        engine="fa-zh",
        forced_reference=True,
    )

    assert track.coverage == 1.0
    assert [(span.start_ms, span.end_ms) for span in track.spans] == [
        (380, 560),
        (560, 800),
        (800, 980),
        (980, 1140),
        (1140, 1260),
        (1260, 1440),
    ]


def test_fa_serialized_text_parses_grouped_tokens_when_counts_differ() -> None:
    result = [
        {
            "text": " 0.000 0.100;\u4e00\u4e2a 0.100 0.500;\u897f 0.500 0.700;",
            "timestamp": [[100, 500], [500, 700]],
        }
    ]

    track = alignment_track_from_model_result(
        result,
        "\u4e00\u4e2a\u897f",
        engine="fa-zh",
        forced_reference=True,
    )

    assert track.coverage == 1.0
    # A grouped token deliberately shares its real interval instead of being
    # split evenly into fabricated per-character timestamps.
    assert [(span.start_ms, span.end_ms) for span in track.spans] == [
        (100, 500),
        (100, 500),
        (500, 700),
    ]
    assert [span.timestamp_precision for span in track.spans] == [
        "token",
        "token",
        "character",
    ]


def test_funasr_loader_accepts_only_local_paths_and_forces_offline(tmp_path: Path) -> None:
    model = tmp_path / "fa-zh"
    model.mkdir()
    captured: dict[str, object] = {}

    def factory(**options: object) -> object:
        captured.update(options)
        return object()

    loaded = load_funasr_model(
        model_path=model,
        label="fa-zh",
        device="cpu",
        model_factory=factory,
        extra_options={"disable_update": False},
    )

    assert loaded is not None
    assert captured["model"] == str(model.resolve())
    assert captured["device"] == "cpu"
    assert captured["disable_update"] is True
    assert captured["disable_pbar"] is True
    assert captured["disable_log"] is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["MODELSCOPE_OFFLINE"] == "1"

    with pytest.raises(ModelUnavailableError):
        load_funasr_model(
            model_path="remote/model-name",
            label="fa-zh",
            model_factory=factory,
        )


def test_resolve_inference_device_honors_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cutvideo.model_runtime as runtime

    monkeypatch.setenv("CUTVIDEO_INFERENCE_DEVICE", "cpu")
    assert runtime.resolve_inference_device() == "cpu"
    assert runtime.resolve_inference_device("cuda:0") == "cuda:0"
    monkeypatch.delenv("CUTVIDEO_INFERENCE_DEVICE", raising=False)
    monkeypatch.setattr(runtime, "prefer_inference_device", lambda: "mps")
    assert runtime.resolve_inference_device() == "mps"


def test_prepared_audio_window_cache_reuses_identical_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wave

    import cutvideo.model_runtime as runtime

    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\x00\x00" * 16_000)

    writes: list[tuple[int, int]] = []

    def fake_write(src, output, *, start_ms, end_ms, **_kwargs):
        writes.append((start_ms, end_ms))
        output.write_bytes(b"RIFF" + b"\x00" * 44)
        return output.resolve()

    monkeypatch.setattr(runtime, "_write_local_audio_window", fake_write)
    with runtime.PreparedAudioWindowCache(source) as cache:
        first = cache.get(0, 500)
        second = cache.get(0, 500)
        third = cache.get(500, 1_000)

    assert first == second
    assert first != third
    assert writes == [(0, 500), (500, 1_000)]
    assert cache.hit_count == 1
    assert cache.miss_count == 2


@pytest.mark.skipif(os.name != "nt", reason="Windows DLL search path regression")
def test_frozen_torch_dll_paths_are_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    import cutvideo.model_runtime as runtime

    captured: list[str] = []

    def fake_add_directory(path: str) -> object:
        captured.append(path)
        return object()

    monkeypatch.setattr(runtime.os, "add_dll_directory", fake_add_directory)
    with runtime._normalized_windows_dll_directories():
        runtime.os.add_dll_directory(r"D:\\workspace\\cutVideo\\torch\\lib")

    assert captured == [r"D:\workspace\cutVideo\torch\lib"]


class _CharacterRefiner:
    """Fake fa-zh returning genuine per-character intervals inside each window."""

    def __init__(self, *, base_ms: float = 1_000.0, step_ms: float = 300.0) -> None:
        self.base_ms = base_ms
        self.step_ms = step_ms
        self.calls: list[tuple[str, int, int]] = []

    def align(
        self,
        *,
        audio_path: str | Path,
        transcript: str,
        window_start_ms: int,
        window_end_ms: int,
    ) -> AlignmentTrack:
        del audio_path
        self.calls.append((transcript, window_start_ms, window_end_ms))
        normalized = normalize_with_mapping(transcript).text
        spans = tuple(
            TimedSpan(
                index,
                index + 1,
                self.base_ms + index * self.step_ms,
                self.base_ms + (index + 1) * self.step_ms,
            )
            for index in range(len(normalized))
        )
        return AlignmentTrack(spans, "fa-zh", 1.0)


def test_refinement_sharpens_shared_word_interval_into_characters() -> None:
    tokens = tuple(
        RecognizedToken(text, 1_000, 1_900, 0.9, True, "token")
        for text in "你好吗"
    )
    refiner = _CharacterRefiner()

    refined, diagnostics = refine_recognition_tokens(
        tokens,
        force_aligner=refiner,
        audio_path=Path("unused.wav"),
        audio_end_ms=5_000,
    )

    assert [(item.start_ms, item.end_ms) for item in refined] == [
        (1_000, 1_300),
        (1_300, 1_600),
        (1_600, 1_900),
    ]
    assert all(item.timestamp_precision == "character" for item in refined)
    assert all(item.confidence_available for item in refined)
    assert diagnostics["refined_token_count"] == 3
    assert diagnostics["failed_group_count"] == 0
    # The forced reference is the recognized text itself inside one padded window.
    assert refiner.calls == [("你好吗", 800, 2_100)]


def test_refinement_skips_groups_that_are_already_character_precise() -> None:
    tokens = (
        RecognizedToken("你", 1_000, 1_200, 0.9, True, "character"),
        RecognizedToken("好", 1_200, 1_400, 0.9, True, "character"),
        RecognizedToken("吗", 1_400, 1_600, 0.9, True, "character"),
    )
    refiner = _CharacterRefiner()

    refined, diagnostics = refine_recognition_tokens(
        tokens,
        force_aligner=refiner,
        audio_path=Path("unused.wav"),
        audio_end_ms=5_000,
    )

    assert refined == tokens
    assert refiner.calls == []
    assert diagnostics["skipped_precise_group_count"] == 1
    assert diagnostics["refined_token_count"] == 0


def test_refinement_still_runs_when_character_label_shares_one_interval() -> None:
    tokens = tuple(
        RecognizedToken(text, 1_000, 1_900, 0.9, True, "character")
        for text in "你好吗"
    )
    refiner = _CharacterRefiner()

    refined, diagnostics = refine_recognition_tokens(
        tokens,
        force_aligner=refiner,
        audio_path=Path("unused.wav"),
        audio_end_ms=5_000,
    )

    assert [(item.start_ms, item.end_ms) for item in refined] == [
        (1_000, 1_300),
        (1_300, 1_600),
        (1_600, 1_900),
    ]
    assert refiner.calls == [("你好吗", 800, 2_100)]
    assert diagnostics["skipped_precise_group_count"] == 0


def test_refinement_rejects_boundaries_far_from_original_interval() -> None:
    tokens = tuple(
        RecognizedToken(text, 1_000, 1_900, 0.9, True, "token")
        for text in "你好吗"
    )
    # A refinement result four seconds away is not credible for these tokens.
    refiner = _CharacterRefiner(base_ms=5_000.0)

    refined, diagnostics = refine_recognition_tokens(
        tokens,
        force_aligner=refiner,
        audio_path=Path("unused.wav"),
        audio_end_ms=10_000,
    )

    assert [(item.start_ms, item.end_ms) for item in refined] == [
        (1_000, 1_900),
        (1_000, 1_900),
        (1_000, 1_900),
    ]
    assert all(item.timestamp_precision == "token" for item in refined)
    assert diagnostics["refined_token_count"] == 0


def test_refinement_splits_groups_on_speech_gaps() -> None:
    tokens = (
        RecognizedToken("甲", 1_000, 1_400, 0.9, True, "token"),
        RecognizedToken("乙", 1_000, 1_400, 0.9, True, "token"),
        RecognizedToken("丙", 4_000, 4_400, 0.9, True, "token"),
    )

    class WindowRelativeRefiner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, int]] = []

        def align(self, *, audio_path, transcript, window_start_ms, window_end_ms):
            del audio_path
            self.calls.append((transcript, window_start_ms, window_end_ms))
            normalized = normalize_with_mapping(transcript).text
            base = window_start_ms + 200
            spans = tuple(
                TimedSpan(index, index + 1, base + index * 200, base + (index + 1) * 200)
                for index in range(len(normalized))
            )
            return AlignmentTrack(spans, "fa-zh", 1.0)

    refiner = WindowRelativeRefiner()
    refined, diagnostics = refine_recognition_tokens(
        tokens,
        force_aligner=refiner,
        audio_path=Path("unused.wav"),
        audio_end_ms=10_000,
    )

    assert [call[0] for call in refiner.calls] == ["甲乙", "丙"]
    assert diagnostics["group_count"] == 2
    assert [(item.start_ms, item.end_ms) for item in refined] == [
        (1_000, 1_200),
        (1_200, 1_400),
        (4_000, 4_200),
    ]


def test_coarse_asr_with_character_consensus_uses_forced_boundaries() -> None:
    text = "前" * 25 + "删除词" + "后" * 25
    parsed = Parsed(
        [
            Paragraph(0, 100, text, [Highlight(25, 28, "删除词")]),
            Paragraph(1, 9_000, "收尾"),
        ]
    )

    class TokenPrecisionAsr(RecordingAligner):
        def align(self, **kwargs: object) -> AlignmentTrack:
            self.calls.append(
                (int(kwargs["window_start_ms"]), int(kwargs["window_end_ms"]))
            )
            normalized = normalize_with_mapping(str(kwargs["transcript"])).text
            spans = tuple(
                TimedSpan(
                    index,
                    index + 1,
                    1_000 + index * 100 + self.delta_ms,
                    1_100 + index * 100 + self.delta_ms,
                    timestamp_precision=TIMESTAMP_PRECISION_TOKEN,
                )
                for index in range(len(normalized))
            )
            return AlignmentTrack(
                spans,
                "fake-token-precision",
                self.coverage,
                timestamp_precision=TIMESTAMP_PRECISION_TOKEN,
            )

    class StableContextAligner:
        def align(self, **kwargs: object) -> AlignmentTrack:
            normalized = normalize_with_mapping(str(kwargs["transcript"])).text
            target = normalized.index("删除词")
            return AlignmentTrack(
                tuple(
                    TimedSpan(
                        index,
                        index + 1,
                        3_500 + (index - target) * 100,
                        3_600 + (index - target) * 100,
                    )
                    for index in range(len(normalized))
                ),
                "stable-context",
                1.0,
            )

    candidate = align_transcript(
        parsed,
        Audio(Path("unused.wav")),
        StableContextAligner(),
        TokenPrecisionAsr(delta_ms=40),
    )[0]

    # Both engines agree within tolerance, but the ASR interval only has word
    # granularity, so the consensus per-character forced boundary is adopted
    # instead of the midpoint with the wider word span.
    assert (candidate.proposed_start_sample, candidate.proposed_end_sample) == (3_500, 3_800)
    assert candidate.diagnostics["used_character_force_boundary"] is True
    assert COARSE_TIMESTAMP not in candidate.reasons


def test_asr_adapter_explicitly_requests_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cutvideo.alignment as module

    model_dir = tmp_path / "asr"
    vad_dir = tmp_path / "vad"
    model_dir.mkdir()
    vad_dir.mkdir()
    calls: list[dict[str, object]] = []
    factory_options: dict[str, object] = {}

    class FakeModel:
        def generate(self, **options: object) -> object:
            calls.append(options)
            return [{"text": "\u7532", "timestamp": [[0, 100]]}]

    @contextmanager
    def fake_window(*_args: object, **_kwargs: object):
        yield tmp_path / "window.wav"

    def model_factory(**options: object) -> FakeModel:
        factory_options.update(options)
        return FakeModel()

    monkeypatch.setattr(module, "local_audio_window", fake_window)
    aligner = FunASRAsrAligner(
        model_dir,
        vad_dir,
        model_factory=model_factory,
    )
    track = aligner.align(
        audio_path=tmp_path / "source.wav",
        transcript="\u7532",
        window_start_ms=0,
        window_end_ms=100,
    )

    assert calls[0]["pred_timestamp"] is True
    assert calls[0]["use_itn"] is False
    assert factory_options["vad_kwargs"] == {"max_single_segment_time": 30_000}
    assert track.coverage == 1.0

    recognized = aligner.recognize(
        audio_path=tmp_path / "source.wav",
        window_start_ms=200,
        window_end_ms=400,
    )
    assert [(item.text, item.start_ms, item.end_ms) for item in recognized] == [
        ("甲", 200, 300)
    ]
