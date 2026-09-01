from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np
import pytest

from xyq_quiz.capture.models import CapturedFrame
from xyq_quiz.knowledge.matcher import QuestionMatcher
from xyq_quiz.knowledge.teacher_bank import load_teacher_bank
from xyq_quiz.knowledge.teacher_matcher import TeacherIconMatcher
from xyq_quiz.knowledge.updater import load_current_generation
from xyq_quiz.recognition.models import ActivityKind, ConfidenceLevel
from xyq_quiz.recognition.ocr import RapidOCREngine
from xyq_quiz.recognition.pipeline import RecognitionPipeline
from xyq_quiz.recognition.teachers_day_layout import TeacherLayoutDetector
from xyq_quiz.runtime.state import RuntimeStore


ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests/fixtures/teachers_day"
CASES = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
PROFILE = ROOT / "data/layouts/teachers-day.json"


def read_image(path):
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def feedback_frame(case, reverse=False, missing=False):
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    frame = np.full((768, 1024, 3), 160, np.uint8)

    def put(raw, image, padding=0):
        x, y, w, h = raw
        x, y = x+100-padding, y+100-padding
        frame[y:y+h+2*padding, x:x+w+2*padding] = cv2.resize(image, (w+2*padding, h+2*padding))

    for anchor in profile["anchors"].values():
        put(anchor["rect"], read_image(PROFILE.parent / anchor["template_path"]))
    put(profile["icon_rect"], read_image(FIXTURES / case["files"]["icon"]), padding=3)
    for rect, letter in zip(profile["option_rects"], "dcba" if reverse else "abcd"):
        if not (missing and letter == "a"):
            put(rect, read_image(FIXTURES / case["files"][f"option-{letter}"]))
    return CapturedFrame.create(1, time.monotonic_ns(), frame)


@pytest.fixture(scope="module")
def pipeline():
    bank = load_current_generation(ROOT / "data").question_bank
    value = RecognitionPipeline(
        TeacherLayoutDetector(PROFILE), RapidOCREngine(),
        QuestionMatcher(bank, 92, 5, 90),
        teacher_matcher=TeacherIconMatcher(load_teacher_bank(ROOT / "data/teachers_day")),
    )
    value.warm_up()
    yield value
    value.close()


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"feedback-{case['sample']}")
@pytest.mark.parametrize("reverse", [False, True])
def test_real_feedback_ocr_and_answer_mapping(pipeline, case, reverse):
    frame = feedback_frame(case, reverse=reverse)
    result = pipeline.recognize(frame, 1)
    expected = 3-case["option_index"] if reverse else case["option_index"]
    assert result.activity_kind is ActivityKind.TEACHERS_DAY
    assert result.option_texts == tuple(case["options"][::-1] if reverse else case["options"])
    assert result.official_answer == case["answer"]
    assert result.option_index == expected
    assert result.confidence_level is ConfidenceLevel.HIGH
    store = RuntimeStore()
    generation = store.begin_question("feedback", 1, frame_size=(1024, 768))
    assert store.complete(generation, result)
    assert store.snapshot().overlay is not None
    assert store.snapshot().option_index == expected


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"feedback-{case['sample']}")
def test_unreadable_distractor_still_blocks_answer(pipeline, case):
    result = pipeline.recognize(feedback_frame(case, missing=True), 1)
    assert result.option_index is None
    assert result.confidence_level is ConfidenceLevel.NONE
