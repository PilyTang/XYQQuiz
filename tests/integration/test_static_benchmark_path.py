from __future__ import annotations

import numpy as np
import pytest

from xyq_quiz.acceptance.benchmark import (
    _ObservedLayoutDetector,
    _run_static_round,
)
from xyq_quiz.acceptance.fixtures import (
    FixtureKind,
    Provenance,
    RecognitionFixture,
)
from xyq_quiz.capture.models import CapturedFrame, Rect
from xyq_quiz.recognition.models import (
    DetectedLayout,
    RecognitionResult,
    RecognitionTimings,
)


LAYOUT = DetectedLayout(
    question_rect=Rect(0, 0, 10, 10),
    option_rects=(
        Rect(10, 0, 10, 10),
        Rect(20, 0, 10, 10),
        Rect(30, 0, 10, 10),
        Rect(40, 0, 10, 10),
    ),
    anchor_scores=(1.0, 1.0),
    profile_name="benchmark-test",
)


class LayoutDetector:
    def detect(self, _image: np.ndarray) -> DetectedLayout:
        return LAYOUT


class MarkerLayoutDetector:
    def detect(self, image: np.ndarray) -> DetectedLayout | None:
        return None if int(image[0, 0, 0]) == 0 else LAYOUT


class CountingPipeline:
    def __init__(self) -> None:
        self.calls = 0
        self.frame_ids: list[int] = []

    def recognize(
        self,
        frame: CapturedFrame,
        generation_id: int,
    ) -> RecognitionResult:
        self.calls += 1
        self.frame_ids.append(frame.frame_id)
        return RecognitionResult(
            generation_id=generation_id,
            frame_id=frame.frame_id,
            question_text="题目",
            option_texts=("甲", "乙", "丙", "丁"),
            official_answer="乙",
            question_score=100.0,
            question_runner_up_score=0.0,
            option_score=100.0,
            option_runner_up_score=0.0,
            high_confidence=True,
            option_index=1,
            overlay_rect=LAYOUT.option_rects[1],
            timings=RecognitionTimings(2.0, 3.0, 1.0, 6.0),
            source_id="fixture-1",
        )


def test_each_static_round_uses_fresh_coordinator_cache_but_same_pipeline() -> None:
    case = RecognitionFixture(
        file="keju-static.png",
        kind=FixtureKind.POSITIVE,
        expected_source_id="fixture-1",
        expected_option_index=1,
        window_size=(52, 12),
        dpi=(96, 96),
        provenance=Provenance.WEB,
        human_verified=True,
        sha256="0" * 64,
    )
    image = np.zeros((12, 52, 3), dtype=np.uint8)
    image[:10, :50] = np.arange(50, dtype=np.uint8)[None, :, None]
    detector = _ObservedLayoutDetector(LayoutDetector())
    pipeline = CountingPipeline()

    first, next_frame_id = _run_static_round(
        (case,),
        {case.file: image},
        detector,
        pipeline,
        next_frame_id=1,
    )
    second, _next_frame_id = _run_static_round(
        (case,),
        {case.file: image},
        detector,
        pipeline,
        next_frame_id=next_frame_id,
    )

    assert pipeline.calls == 2
    assert len(first) == len(second) == 1
    assert first[0].end_to_end_ms > 0
    assert second[0].end_to_end_ms > 0
    assert first[0].pipeline_total_ms == second[0].pipeline_total_ms == 6.0


def test_second_negative_cannot_reuse_previous_dialog_missing_terminal() -> None:
    first = RecognitionFixture(
        file="non-keju-first.png",
        kind=FixtureKind.NEGATIVE,
        expected_source_id=None,
        expected_option_index=None,
        window_size=(52, 12),
        dpi=(96, 96),
        provenance=Provenance.WEB,
        human_verified=True,
        sha256="0" * 64,
    )
    second = RecognitionFixture(
        file="non-keju-second.png",
        kind=FixtureKind.NEGATIVE,
        expected_source_id=None,
        expected_option_index=None,
        window_size=(52, 12),
        dpi=(96, 96),
        provenance=Provenance.WEB,
        human_verified=True,
        sha256="1" * 64,
    )
    no_layout = np.zeros((12, 52, 3), dtype=np.uint8)
    has_layout = np.zeros((12, 52, 3), dtype=np.uint8)
    has_layout[:10, :50] = np.arange(1, 51, dtype=np.uint8)[None, :, None]
    detector = _ObservedLayoutDetector(MarkerLayoutDetector())
    pipeline = CountingPipeline()

    with pytest.raises(RuntimeError, match="negative fixture produced overlay"):
        _run_static_round(
            (first, second),
            {first.file: no_layout, second.file: has_layout},
            detector,
            pipeline,
            next_frame_id=1,
        )

    assert pipeline.calls == 1
    assert pipeline.frame_ids == [4]
