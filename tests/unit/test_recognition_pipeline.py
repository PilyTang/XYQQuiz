from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from xyq_quiz.capture.models import CapturedFrame, Rect
from xyq_quiz.knowledge.matcher import QuestionMatcher
from xyq_quiz.knowledge.models import OptionMatch, QuestionMatch, QuestionRecord
from xyq_quiz.knowledge.store import QuestionBank
from xyq_quiz.recognition.models import DetectedLayout, OCRText
from xyq_quiz.recognition.ocr import OCRRole
from xyq_quiz.recognition.pipeline import RecognitionPipeline


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "questions-small.json"


class FakeLayoutDetector:
    def __init__(self, layout: DetectedLayout | None) -> None:
        self.layout = layout

    def detect(self, _image: np.ndarray) -> DetectedLayout | None:
        return self.layout


class MarkerOCR:
    def __init__(
        self,
        answers: Mapping[int, str],
        barrier: threading.Barrier | None = None,
        delay: float = 0.0,
    ) -> None:
        self.answers = answers
        self.barrier = barrier
        self.delay = delay
        self.thread_ids: set[int] = set()
        self.calls = 0
        self._lock = threading.Lock()

    def recognize(self, image: np.ndarray) -> OCRText:
        marker = int(image[0, 0, 0])
        with self._lock:
            self.calls += 1
            self.thread_ids.add(threading.get_ident())
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        if self.delay:
            time.sleep(self.delay)
        return OCRText(self.answers[marker], 0.99, 1.0)


class GatedOCR(MarkerOCR):
    def __init__(self, answers: Mapping[int, str]) -> None:
        super().__init__(answers)
        self.all_started = threading.Event()
        self.release = threading.Event()

    def recognize(self, image: np.ndarray) -> OCRText:
        marker = int(image[0, 0, 0])
        with self._lock:
            self.calls += 1
            self.thread_ids.add(threading.get_ident())
            if self.calls == 1:
                self.all_started.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("OCR was not released")
        return OCRText(self.answers[marker], 0.99, 1.0)


class RoleAwareMarkerOCR(MarkerOCR):
    def __init__(self, answers: Mapping[int, str]) -> None:
        super().__init__(answers)
        self.roles: list[OCRRole] = []
        self.raw_shapes: list[tuple[int, ...]] = []
        self.fallback_shapes: list[tuple[int, ...]] = []

    def recognize_region(
        self,
        image: np.ndarray,
        role: OCRRole,
        *,
        fallback_image: np.ndarray,
    ) -> OCRText:
        self.roles.append(role)
        self.raw_shapes.append(image.shape)
        self.fallback_shapes.append(fallback_image.shape)
        return super().recognize(image)


class AliasMatcher:
    def __init__(self, answer: str, mapped: Mapping[str, int]) -> None:
        self.record = QuestionRecord("alias", "别名题", answer, "别名题")
        self.mapped = mapped
        self.map_calls: list[str] = []

    def match_question(self, _text: str) -> QuestionMatch:
        return QuestionMatch(100.0, 0.0, self.record)

    def map_answer(
        self,
        answer: str,
        _options: Sequence[str],
    ) -> OptionMatch | None:
        self.map_calls.append(answer)
        option_index = self.mapped.get(answer)
        if option_index is None:
            return None
        return OptionMatch(100.0, 0.0, option_index)

    def is_high_confidence(
        self,
        question: QuestionMatch | None,
        option: OptionMatch | None,
    ) -> bool:
        return question is not None and option is not None


class ReplacingMatcher(AliasMatcher):
    def __init__(
        self,
        answer: str,
        mapped: Mapping[str, int],
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        super().__init__(answer, mapped)
        self.entered = entered
        self.release = release

    def match_question(self, text: str) -> QuestionMatch:
        result = super().match_question(text)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None and not self.release.wait(timeout=2):
            raise TimeoutError("matcher was not released")
        return result


@pytest.fixture
def layout() -> DetectedLayout:
    return DetectedLayout(
        question_rect=Rect(0, 0, 10, 10),
        option_rects=(
            Rect(10, 0, 10, 10),
            Rect(20, 0, 10, 10),
            Rect(30, 0, 10, 10),
            Rect(40, 0, 10, 10),
        ),
        anchor_scores=(1.0, 1.0),
    )


@pytest.fixture
def frame() -> CapturedFrame:
    image = np.zeros((10, 50, 3), dtype=np.uint8)
    for marker in range(1, 6):
        image[:, (marker - 1) * 10 : marker * 10] = marker
    return CapturedFrame.create(27, 123_456, image)


def test_pipeline_default_uses_one_executor_thread_for_all_five_crops(
    layout: DetectedLayout,
    frame: CapturedFrame,
) -> None:
    matcher = QuestionMatcher(QuestionBank.load(FIXTURE_PATH), 92, 5, 90)
    ocr = MarkerOCR(
        {1: "梦幻西游中有多少个种族", 2: "2", 3: "3", 4: "4", 5: "5"},
        delay=0.03,
    )
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), ocr, matcher)
    try:
        result = pipeline.recognize(frame, 4)
    finally:
        pipeline.close()

    assert result.generation_id == 4
    assert result.frame_id == 27
    assert result.question_text == "梦幻西游中有多少个种族"
    assert result.option_texts == ("2", "3", "4", "5")
    assert result.official_answer == "3"
    assert result.high_confidence is True
    assert result.option_index == 1
    assert result.overlay_rect == layout.option_rects[1]
    assert len(ocr.thread_ids) == 1
    assert result.timings.ocr_ms >= 0
    assert result.timings.match_ms >= 0
    assert result.timings.total_ms >= result.timings.ocr_ms


def test_pipeline_recognizes_three_option_xiangshi_layout() -> None:
    three_option_layout = DetectedLayout(
        question_rect=Rect(0, 0, 10, 10),
        option_rects=(
            Rect(10, 0, 10, 10),
            Rect(20, 0, 10, 10),
            Rect(30, 0, 10, 10),
        ),
        anchor_scores=(1.0,),
    )
    image = np.zeros((10, 40, 3), dtype=np.uint8)
    for marker in range(1, 5):
        image[:, (marker - 1) * 10 : marker * 10] = marker
    three_option_frame = CapturedFrame.create(28, 123_457, image)
    matcher = QuestionMatcher(QuestionBank.load(FIXTURE_PATH), 92, 5, 90)
    ocr = MarkerOCR(
        {1: "梦幻西游中有多少个种族", 2: "2", 3: "3", 4: "4"}
    )
    pipeline = RecognitionPipeline(
        FakeLayoutDetector(three_option_layout), ocr, matcher
    )
    try:
        result = pipeline.recognize(three_option_frame, 5)
    finally:
        pipeline.close()

    assert result.option_texts == ("2", "3", "4")
    assert result.high_confidence is True
    assert result.option_index == 1
    assert result.overlay_rect == three_option_layout.option_rects[1]


def test_pipeline_does_not_expose_multi_worker_configuration(
    layout: DetectedLayout,
) -> None:
    matcher = QuestionMatcher(QuestionBank.load(FIXTURE_PATH), 92, 5, 90)

    with pytest.raises(TypeError, match="ocr_workers"):
        RecognitionPipeline(
            FakeLayoutDetector(layout),
            MarkerOCR({}),
            matcher,
            ocr_workers=2,
        )


def test_warm_up_and_recognition_share_the_same_executor_thread(
    layout: DetectedLayout,
    frame: CapturedFrame,
) -> None:
    matcher = QuestionMatcher(QuestionBank.load(FIXTURE_PATH), 92, 5, 90)
    ocr = MarkerOCR(
        {255: "", 1: "梦幻西游中有多少个种族", 2: "2", 3: "3", 4: "4", 5: "5"},
    )
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), ocr, matcher)
    try:
        pipeline.warm_up()
        pipeline.recognize(frame, 4)
    finally:
        pipeline.close()

    assert ocr.calls == 6
    assert len(ocr.thread_ids) == 1


def test_pipeline_uses_explicit_question_and_option_fast_paths_with_safe_fallbacks(
    layout: DetectedLayout,
    frame: CapturedFrame,
) -> None:
    matcher = QuestionMatcher(QuestionBank.load(FIXTURE_PATH), 92, 5, 90)
    ocr = RoleAwareMarkerOCR(
        {1: "梦幻西游中有多少个种族", 2: "2", 3: "3", 4: "4", 5: "5"}
    )
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), ocr, matcher)
    try:
        result = pipeline.recognize(frame, 4)
    finally:
        pipeline.close()

    assert result.high_confidence is True
    assert ocr.roles == [OCRRole.QUESTION] + [OCRRole.OPTION] * 4
    assert ocr.raw_shapes == [(10, 10, 3)] * 5
    assert ocr.fallback_shapes == [(10, 10, 3)] + [(30, 30, 3)] * 4
    assert [crop.shape for crop in pipeline.latest_crops()] == [
        (10, 10, 3),
        (30, 30, 3),
        (30, 30, 3),
        (30, 30, 3),
        (30, 30, 3),
    ]


def test_warm_up_preheats_role_aware_rec_only_business_path(
    layout: DetectedLayout,
) -> None:
    matcher = QuestionMatcher(QuestionBank.load(FIXTURE_PATH), 92, 5, 90)
    ocr = RoleAwareMarkerOCR({255: ""})
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), ocr, matcher)
    try:
        pipeline.warm_up()
    finally:
        pipeline.close()

    assert ocr.roles == [OCRRole.QUESTION]
    assert ocr.raw_shapes == ocr.fallback_shapes
    assert ocr.raw_shapes == [(96, 384, 3)]


def test_warm_up_is_idempotent(layout: DetectedLayout) -> None:
    matcher = QuestionMatcher(QuestionBank.load(FIXTURE_PATH), 92, 5, 90)
    ocr = MarkerOCR({255: ""})
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), ocr, matcher)
    try:
        pipeline.warm_up()
        pipeline.warm_up()
    finally:
        pipeline.close()

    assert ocr.calls == 1


def test_warm_up_after_close_fails_clearly(layout: DetectedLayout) -> None:
    matcher = QuestionMatcher(QuestionBank.load(FIXTURE_PATH), 92, 5, 90)
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), MarkerOCR({}), matcher)
    pipeline.close()

    with pytest.raises(RuntimeError, match="closed"):
        pipeline.warm_up()


def test_close_waits_for_started_warm_up_without_executor_race(
    layout: DetectedLayout,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    class BlockingWarmOCR:
        def recognize(self, _image: np.ndarray) -> OCRText:
            entered.set()
            if not release.wait(timeout=2):
                raise TimeoutError("warm-up was not released")
            return OCRText("warm", 1.0, 1.0)

    matcher = QuestionMatcher(QuestionBank.load(FIXTURE_PATH), 92, 5, 90)
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), BlockingWarmOCR(), matcher)
    warmer = threading.Thread(
        target=lambda: _capture_thread_error(pipeline.warm_up, errors)
    )
    closer = threading.Thread(
        target=lambda: _capture_thread_error(pipeline.close, errors)
    )
    warmer.start()
    assert entered.wait(timeout=0.5)
    closer.start()
    time.sleep(0.05)
    assert closer.is_alive()

    release.set()
    warmer.join(timeout=2)
    closer.join(timeout=2)

    assert not warmer.is_alive()
    assert not closer.is_alive()
    assert errors == []


def _capture_thread_error(operation, errors: list[BaseException]) -> None:
    try:
        operation()
    except BaseException as exc:
        errors.append(exc)


def test_pipeline_exposes_only_last_existing_recognition_crops(
    layout: DetectedLayout,
    frame: CapturedFrame,
) -> None:
    matcher = QuestionMatcher(QuestionBank.load(FIXTURE_PATH), 92, 5, 90)
    ocr = MarkerOCR(
        {1: "梦幻西游中有多少个种族", 2: "2", 3: "3", 4: "4", 5: "5"}
    )
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), ocr, matcher)
    try:
        assert pipeline.latest_crops() == ()
        pipeline.recognize(frame, 4)
        crops = pipeline.latest_crops()
    finally:
        pipeline.close()

    assert len(crops) == 5
    assert [int(crop[0, 0, 0]) for crop in crops] == [1, 2, 3, 4, 5]
    assert all(not crop.flags.writeable for crop in crops)
    crops[0].setflags(write=True)
    crops[0][0, 0, 0] = 99
    assert int(pipeline.latest_crops()[0][0, 0, 0]) == 1


def test_pipeline_returns_uncertain_without_overlay(
    layout: DetectedLayout,
    frame: CapturedFrame,
) -> None:
    matcher = QuestionMatcher(QuestionBank.load(FIXTURE_PATH), 92, 5, 90)
    ocr = MarkerOCR({1: "无法识别", 2: "甲", 3: "乙", 4: "丙", 5: "丁"})
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), ocr, matcher)
    try:
        result = pipeline.recognize(frame, 5)
    finally:
        pipeline.close()

    assert result.high_confidence is False
    assert result.option_index is None
    assert result.overlay_rect is None


def test_ascii_comma_aliases_must_converge_on_one_option(
    layout: DetectedLayout,
    frame: CapturedFrame,
) -> None:
    matcher = AliasMatcher(" 蝎子, ,蝎子精 ", {"蝎子": 1, "蝎子精": 1})
    ocr = MarkerOCR({1: "别名题", 2: "甲", 3: "蝎子精", 4: "丙", 5: "丁"})
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), ocr, matcher)
    try:
        result = pipeline.recognize(frame, 6)
    finally:
        pipeline.close()

    assert matcher.map_calls == ["蝎子", "蝎子精"]
    assert result.high_confidence is True
    assert result.option_index == 1
    assert result.overlay_rect == layout.option_rects[1]


def test_ascii_comma_aliases_mapping_to_different_options_are_uncertain(
    layout: DetectedLayout,
    frame: CapturedFrame,
) -> None:
    matcher = AliasMatcher("蝎子,蝎子精", {"蝎子": 0, "蝎子精": 1})
    ocr = MarkerOCR({1: "别名题", 2: "蝎子", 3: "蝎子精", 4: "丙", 5: "丁"})
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), ocr, matcher)
    try:
        result = pipeline.recognize(frame, 7)
    finally:
        pipeline.close()

    assert matcher.map_calls == ["蝎子", "蝎子精"]
    assert result.high_confidence is False
    assert result.option_index is None
    assert result.overlay_rect is None


def test_ascii_comma_aliases_all_failing_are_uncertain(
    layout: DetectedLayout,
    frame: CapturedFrame,
) -> None:
    matcher = AliasMatcher("蝎子,蝎子精", {})
    ocr = MarkerOCR({1: "别名题", 2: "甲", 3: "乙", 4: "丙", 5: "丁"})
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), ocr, matcher)
    try:
        result = pipeline.recognize(frame, 8)
    finally:
        pipeline.close()

    assert matcher.map_calls == ["蝎子", "蝎子精"]
    assert result.high_confidence is False
    assert result.option_index is None
    assert result.overlay_rect is None


def test_chinese_enumeration_comma_is_not_an_alias_separator(
    layout: DetectedLayout,
    frame: CapturedFrame,
) -> None:
    matcher = AliasMatcher("蝎子、蝎子精", {})
    ocr = MarkerOCR({1: "别名题", 2: "蝎子", 3: "蝎子精", 4: "丙", 5: "丁"})
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), ocr, matcher)
    try:
        result = pipeline.recognize(frame, 9)
    finally:
        pipeline.close()

    assert matcher.map_calls == ["蝎子、蝎子精"]
    assert result.high_confidence is False
    assert result.overlay_rect is None


def test_layout_failure_skips_ocr_and_returns_uncertain(frame: CapturedFrame) -> None:
    matcher = QuestionMatcher(QuestionBank.load(FIXTURE_PATH), 92, 5, 90)
    ocr = MarkerOCR({})
    pipeline = RecognitionPipeline(FakeLayoutDetector(None), ocr, matcher)
    try:
        result = pipeline.recognize(frame, 10)
    finally:
        pipeline.close()

    assert ocr.calls == 0
    assert result.high_confidence is False
    assert result.overlay_rect is None


def test_replace_matcher_applies_to_next_recognition(
    layout: DetectedLayout,
    frame: CapturedFrame,
) -> None:
    old = AliasMatcher("旧答案", {"旧答案": 0})
    new = AliasMatcher("新答案", {"新答案": 1})
    ocr = MarkerOCR({1: "题目", 2: "旧答案", 3: "新答案", 4: "丙", 5: "丁"})
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), ocr, old)
    try:
        first = pipeline.recognize(frame, 11)
        pipeline.replace_matcher(new)
        second = pipeline.recognize(frame, 12)
    finally:
        pipeline.close()

    assert first.official_answer == "旧答案"
    assert first.option_index == 0
    assert second.official_answer == "新答案"
    assert second.option_index == 1


def test_running_recognition_uses_one_matcher_snapshot(
    layout: DetectedLayout,
    frame: CapturedFrame,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    old = ReplacingMatcher("旧答案", {"旧答案": 0}, entered, release)
    new = AliasMatcher("新答案", {"新答案": 1})
    ocr = MarkerOCR({1: "题目", 2: "旧答案", 3: "新答案", 4: "丙", 5: "丁"})
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), ocr, old)
    result_holder: list[object] = []
    worker = threading.Thread(
        target=lambda: result_holder.append(pipeline.recognize(frame, 13))
    )
    try:
        worker.start()
        assert entered.wait(timeout=2)
        pipeline.replace_matcher(new)
        release.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
        running_result = result_holder[0]
        next_result = pipeline.recognize(frame, 14)
    finally:
        release.set()
        worker.join(timeout=2)
        pipeline.close()

    assert running_result.official_answer == "旧答案"
    assert running_result.option_index == 0
    assert next_result.official_answer == "新答案"
    assert next_result.option_index == 1


def test_matcher_snapshot_is_taken_before_ocr_starts(
    layout: DetectedLayout,
    frame: CapturedFrame,
) -> None:
    old = AliasMatcher("旧答案", {"旧答案": 0})
    new = AliasMatcher("新答案", {"新答案": 1})
    ocr = GatedOCR({1: "题目", 2: "旧答案", 3: "新答案", 4: "丙", 5: "丁"})
    pipeline = RecognitionPipeline(FakeLayoutDetector(layout), ocr, old)
    results: list[object] = []
    worker = threading.Thread(
        target=lambda: results.append(pipeline.recognize(frame, 15))
    )
    try:
        worker.start()
        assert ocr.all_started.wait(timeout=2)
        pipeline.replace_matcher(new)
        ocr.release.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
        running_result = results[0]
        next_result = pipeline.recognize(frame, 16)
    finally:
        ocr.release.set()
        worker.join(timeout=2)
        pipeline.close()

    assert running_result.official_answer == "旧答案"
    assert running_result.option_index == 0
    assert next_result.official_answer == "新答案"
    assert next_result.option_index == 1
