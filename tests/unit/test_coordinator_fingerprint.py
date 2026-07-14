from __future__ import annotations

import cv2
import numpy as np
import pytest

from xyq_quiz.capture.models import Rect
from xyq_quiz.recognition.models import DetectedLayout
from xyq_quiz.runtime.coordinator import (
    _quiz_cache_identity,
    _quiz_stability_signature,
    _same_quiz_identity,
)


LAYOUT = DetectedLayout(
    question_rect=Rect(280, 180, 620, 120),
    option_rects=(
        Rect(300, 340, 500, 70),
        Rect(300, 430, 500, 70),
        Rect(300, 520, 500, 70),
        Rect(300, 610, 500, 70),
    ),
    anchor_scores=(1.0, 1.0),
    profile_name="real-scale-test",
)


def _base_frame() -> np.ndarray:
    frame = np.full((900, 1440, 3), (60, 85, 110), dtype=np.uint8)
    for rect in (LAYOUT.question_rect, *LAYOUT.option_rects):
        frame[
            rect.y : rect.y + rect.height,
            rect.x : rect.x + rect.width,
        ] = (228, 224, 216)
    cv2.putText(
        frame,
        "QUESTION TEXT",
        (310, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(frame, "O", (760, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2)
    cv2.putText(frame, "A", (820, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2)
    for index, rect in enumerate(LAYOUT.option_rects):
        cv2.putText(
            frame,
            f"OPTION {index}",
            (rect.x + 18, rect.y + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )
    return frame


def _with_small_change(region: str) -> np.ndarray:
    frame = _base_frame()
    if region == "question_digit":
        cv2.putText(frame, "1", (850, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (20, 20, 20), 1)
    elif region == "question_punctuation":
        cv2.circle(frame, (860, 275), 2, (20, 20, 20), -1)
    elif region == "question_stroke":
        cv2.line(frame, (870, 266), (870, 276), (20, 20, 20), 1)
    elif region == "question_o_to_q":
        cv2.line(frame, (777, 246), (786, 255), (20, 20, 20), 1)
    elif region == "question_overlay_digit":
        cv2.putText(frame, "1", (829, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (20, 20, 20), 1)
    elif region == "question_overlay_stroke":
        cv2.line(frame, (832, 230), (832, 244), (20, 20, 20), 1)
    elif region == "question_single_pixel":
        frame[274, 889] = (20, 20, 20)
    else:
        option_index = int(region.removeprefix("option_"))
        rect = LAYOUT.option_rects[option_index]
        cv2.putText(
            frame,
            "1",
            (rect.x + rect.width - 22, rect.y + rect.height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (20, 20, 20),
            1,
        )
    return frame


@pytest.mark.parametrize(
    "region",
    [
        "question_digit",
        "question_punctuation",
        "question_stroke",
        "question_o_to_q",
        "question_overlay_digit",
        "question_overlay_stroke",
        "question_single_pixel",
        "option_0",
        "option_1",
        "option_2",
        "option_3",
    ],
)
def test_cache_identity_changes_for_small_visible_text_change(region: str) -> None:
    assert not _same_quiz_identity(
        _quiz_cache_identity(_base_frame(), LAYOUT),
        _quiz_cache_identity(_with_small_change(region), LAYOUT),
    )


def test_cache_identity_ignores_unrelated_background_change() -> None:
    original = _base_frame()
    changed = original.copy()
    changed[20:140, 1000:1300] = (10, 220, 40)

    assert _same_quiz_identity(
        _quiz_cache_identity(original, LAYOUT),
        _quiz_cache_identity(changed, LAYOUT),
    )


def test_cache_identity_recomputes_after_light_jpeg_compression() -> None:
    original = _base_frame()
    encoded, payload = cv2.imencode(".jpg", original, [cv2.IMWRITE_JPEG_QUALITY, 98])
    assert encoded
    compressed = cv2.imdecode(payload, cv2.IMREAD_COLOR)

    assert not _same_quiz_identity(
        _quiz_cache_identity(original, LAYOUT),
        _quiz_cache_identity(compressed, LAYOUT),
    )


def test_stability_signature_is_stable_under_light_jpeg_compression() -> None:
    original = _base_frame()
    encoded, payload = cv2.imencode(".jpg", original, [cv2.IMWRITE_JPEG_QUALITY, 98])
    assert encoded
    compressed = cv2.imdecode(payload, cv2.IMREAD_COLOR)

    assert _quiz_stability_signature(
        original,
        LAYOUT,
    ) == _quiz_stability_signature(compressed, LAYOUT)
