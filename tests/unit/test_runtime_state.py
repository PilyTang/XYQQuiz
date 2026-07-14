from __future__ import annotations

import threading
import time

import pytest

from xyq_quiz.capture.models import Rect
from xyq_quiz.recognition.models import RecognitionResult, RecognitionTimings
from xyq_quiz.runtime.state import RuntimePhase, RuntimeStore


def recognition_result(
    generation_id: int,
    *,
    high_confidence: bool = True,
) -> RecognitionResult:
    return RecognitionResult(
        generation_id=generation_id,
        frame_id=20,
        question_text="题目",
        option_texts=("甲", "乙", "丙", "丁"),
        official_answer="乙" if high_confidence else None,
        question_score=99.0 if high_confidence else 60.0,
        question_runner_up_score=10.0,
        option_score=98.0 if high_confidence else 50.0,
        option_runner_up_score=12.0,
        high_confidence=high_confidence,
        option_index=1 if high_confidence else None,
        overlay_rect=Rect(20, 10, 20, 10) if high_confidence else None,
        timings=RecognitionTimings(1.0, 2.0, 3.0, 6.0),
    )


@pytest.fixture
def store() -> RuntimeStore:
    return RuntimeStore()


def test_stale_generation_cannot_overwrite_new_question(store: RuntimeStore) -> None:
    old = store.begin_question("old-hash", frame_id=10, frame_size=(100, 50))
    new = store.begin_question("new-hash", frame_id=11, frame_size=(100, 50))

    assert store.complete(old, recognition_result(old)) is False

    snapshot = store.snapshot()
    assert snapshot.generation_id == new
    assert snapshot.phase is RuntimePhase.RECOGNIZING
    assert snapshot.question_text == ""
    assert snapshot.overlay is None


def test_uncertain_result_has_no_overlay(store: RuntimeStore) -> None:
    generation = store.begin_question("hash", 20, frame_size=(100, 50))

    assert store.complete(
        generation,
        recognition_result(generation, high_confidence=False),
    )

    snapshot = store.snapshot()
    assert snapshot.phase is RuntimePhase.UNCERTAIN
    assert snapshot.overlay is None
    assert snapshot.question_text == "题目"


def test_answered_result_publishes_normalized_overlay(store: RuntimeStore) -> None:
    generation = store.begin_question("hash", 20, frame_size=(100, 50))
    store.complete(generation, recognition_result(generation))

    snapshot = store.snapshot()
    assert snapshot.phase is RuntimePhase.ANSWERED
    assert snapshot.overlay == pytest.approx((0.2, 0.2, 0.2, 0.2))
    assert snapshot.option_index == 1
    assert snapshot.timings == RecognitionTimings(1.0, 2.0, 3.0, 6.0)


def test_clear_event_is_published_within_state_transition(store: RuntimeStore) -> None:
    generation = store.begin_question("hash", 30, frame_size=(100, 50))
    store.complete(generation, recognition_result(generation))
    before = store.snapshot()

    cleared_generation = store.clear_question("dialog_missing")

    snapshot = store.snapshot()
    assert cleared_generation == before.generation_id + 1
    assert snapshot.version == before.version + 1
    assert snapshot.overlay is None
    assert snapshot.phase is RuntimePhase.MONITORING
    assert snapshot.question_text == ""
    assert snapshot.clear_monotonic_ns is not None
    assert snapshot.message == "dialog_missing"


def test_wait_after_uses_condition_notification(store: RuntimeStore) -> None:
    initial = store.snapshot()
    received = []
    waiting = threading.Event()

    def waiter() -> None:
        waiting.set()
        received.append(store.wait_after(initial.version, timeout=1.0))

    thread = threading.Thread(target=waiter)
    thread.start()
    assert waiting.wait(timeout=0.2)
    store.clear_question("changed")
    thread.join(timeout=0.2)

    assert not thread.is_alive()
    assert received[0] is not None
    assert received[0].version > initial.version


def test_stale_completion_does_not_publish_or_wake_waiter(store: RuntimeStore) -> None:
    old = store.begin_question("old", 1, frame_size=(100, 50))
    store.begin_question("new", 2, frame_size=(100, 50))
    version = store.snapshot().version

    assert store.complete(old, recognition_result(old)) is False
    started = time.monotonic()
    assert store.wait_after(version, timeout=0.02) is None
    assert time.monotonic() - started >= 0.015
