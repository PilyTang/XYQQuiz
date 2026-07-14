from __future__ import annotations

import numpy as np

from xyq_quiz.capture.models import CapturedFrame, Rect
from xyq_quiz.knowledge.matcher import QuestionMatcher
from xyq_quiz.knowledge.models import QuestionRecord, normalize_text
from xyq_quiz.knowledge.store import QuestionBank
from xyq_quiz.recognition.models import DetectedLayout, OCRText
from xyq_quiz.recognition.layout import MultiProfileLayoutDetector
from xyq_quiz.recognition.pipeline import RecognitionPipeline


class FakeLayoutDetector:
    def detect(self, _image: np.ndarray) -> DetectedLayout:
        return DetectedLayout(
            Rect(0, 0, 2, 2),
            (Rect(2, 0, 2, 2), Rect(4, 0, 2, 2), Rect(6, 0, 2, 2), Rect(8, 0, 2, 2)),
            (1.0, 1.0),
        )


class FakeOCREngine:
    answers = {1: "题目", 2: "甲", 3: "乙", 4: "丙", 5: "丁"}

    def recognize(self, image: np.ndarray) -> OCRText:
        return OCRText(self.answers[int(image[0, 0, 0])], 1.0, 0.0)


def make_frame() -> CapturedFrame:
    image = np.zeros((2, 10, 3), dtype=np.uint8)
    for marker in range(1, 6):
        image[:, (marker - 1) * 2 : marker * 2] = marker
    return CapturedFrame.create(1, 1, image)


def test_recognition_result_exposes_matched_source_id() -> None:
    record = QuestionRecord("source-42", "题目", "乙", "题目")
    matcher = QuestionMatcher(
        QuestionBank((record,)),
        question_score=90,
        question_gap=5,
        option_score=90,
    )
    pipeline = RecognitionPipeline(
        FakeLayoutDetector(),
        FakeOCREngine(),
        matcher,
    )
    try:
        result = pipeline.recognize(make_frame(), generation_id=1)
    finally:
        pipeline.close()

    assert result.source_id == "source-42"


def test_recognition_matches_only_question_after_fixed_game_prompt() -> None:
    record = QuestionRecord("source-42", "题目", "乙", "题目")
    matcher = QuestionMatcher(QuestionBank((record,)), 92, 5, 90)
    engine = FakeOCREngine()
    engine.answers = {
        **engine.answers,
        1: "御前科举大赛第20关：这一关考的是古代常识。题目：题目",
    }
    pipeline = RecognitionPipeline(FakeLayoutDetector(), engine, matcher)
    try:
        result = pipeline.recognize(make_frame(), generation_id=1)
    finally:
        pipeline.close()

    assert result.high_confidence is True
    assert result.source_id == "source-42"


def test_recognition_strips_fixed_prompt_when_ocr_misreads_prompt_prefix() -> None:
    question = "左图为北宋年间的风俗画作品，它的名字是"
    record = QuestionRecord("source-picture", question, "乙", normalize_text(question))
    matcher = QuestionMatcher(QuestionBank((record,)), 92, 5, 90)
    engine = FakeOCREngine()
    engine.answers = {
        **engine.answers,
        1: f"彻前科举大赛第6关：这一关考的是书画艺术。题目：{question}",
    }
    pipeline = RecognitionPipeline(FakeLayoutDetector(), engine, matcher)
    try:
        result = pipeline.recognize(make_frame(), generation_id=1)
    finally:
        pipeline.close()

    assert result.question_score == 100
    assert result.source_id == "source-picture"


def test_question_body_containing_prompt_word_is_not_truncated() -> None:
    question = "关于题目：二字的说法"
    record = QuestionRecord("source-43", question, "乙", question)
    engine = FakeOCREngine()
    engine.answers = {**engine.answers, 1: question}
    pipeline = RecognitionPipeline(
        FakeLayoutDetector(),
        engine,
        QuestionMatcher(QuestionBank((record,)), 92, 5, 90),
    )
    try:
        result = pipeline.recognize(make_frame(), generation_id=1)
    finally:
        pipeline.close()

    assert result.high_confidence is True
    assert result.source_id == "source-43"


def test_ambiguous_layout_pipeline_never_runs_ocr_or_returns_overlay() -> None:
    first = DetectedLayout(
        Rect(0, 0, 2, 2),
        (Rect(2, 0, 2, 2),) * 4,
        (0.98, 0.98),
    )
    second = DetectedLayout(
        Rect(1, 0, 2, 2),
        (Rect(3, 0, 2, 2),) * 4,
        (0.98, 0.98),
    )

    class StaticDetector:
        def __init__(self, layout: DetectedLayout) -> None:
            self.layout = layout

        def detect(self, _frame: np.ndarray) -> DetectedLayout:
            return self.layout

    class RejectOCR:
        def recognize(self, _image: np.ndarray) -> OCRText:
            raise AssertionError("ambiguous layout must not start OCR")

    detector = MultiProfileLayoutDetector(
        (("first", StaticDetector(first)), ("second", StaticDetector(second)))
    )
    record = QuestionRecord("source-42", "题目", "乙", "题目")
    pipeline = RecognitionPipeline(
        detector,
        RejectOCR(),
        QuestionMatcher(QuestionBank((record,)), 92, 5, 90),
    )
    try:
        result = pipeline.recognize(make_frame(), generation_id=1)
    finally:
        pipeline.close()

    assert result.high_confidence is False
    assert result.overlay_rect is None


def test_option_crops_are_upscaled_for_small_game_text() -> None:
    class RecordingOCR(FakeOCREngine):
        def __init__(self) -> None:
            self.shapes: list[tuple[int, int]] = []

        def recognize(self, image: np.ndarray) -> OCRText:
            self.shapes.append(image.shape[:2])
            return super().recognize(
                image if image.shape[:2] == (2, 2) else image[::3, ::3]
            )

    record = QuestionRecord("source-42", "题目", "乙", "题目")
    engine = RecordingOCR()
    pipeline = RecognitionPipeline(
        FakeLayoutDetector(),
        engine,
        QuestionMatcher(QuestionBank((record,)), 92, 5, 90),
    )
    try:
        pipeline.recognize(make_frame(), generation_id=1)
    finally:
        pipeline.close()

    assert sorted(engine.shapes) == [(2, 2), (6, 6), (6, 6), (6, 6), (6, 6)]
