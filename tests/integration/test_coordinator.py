from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pytest

import xyq_quiz.runtime.coordinator as coordinator_module

from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.models import (
    CapturedFrame,
    CapturePhase,
    CaptureStatus,
    Rect,
)
from xyq_quiz.recognition.models import (
    DetectedLayout,
    RecognitionResult,
    RecognitionTimings,
)
from xyq_quiz.recognition.layout import (
    _AnchorMatch,
    LayoutProfile,
    MultiProfileLayoutDetector,
    TemplateLayoutDetector,
)
from xyq_quiz.runtime.coordinator import RecognitionCoordinator
from xyq_quiz.runtime.state import RuntimePhase, RuntimeStore


LAYOUT = DetectedLayout(
    question_rect=Rect(0, 0, 9, 8),
    option_rects=(
        Rect(0, 8, 9, 2),
        Rect(9, 8, 9, 2),
        Rect(18, 8, 9, 2),
        Rect(27, 8, 9, 2),
    ),
    anchor_scores=(1.0, 1.0),
)


class FakeCaptureService:
    def __init__(self) -> None:
        self._status = CaptureStatus(CapturePhase.CAPTURING)
        self._lock = threading.Lock()

    def status(self) -> CaptureStatus:
        with self._lock:
            return self._status

    def set_phase(self, phase: CapturePhase, message: str | None = None) -> None:
        with self._lock:
            self._status = CaptureStatus(phase, message=message)


class FakeLayoutDetector:
    def __init__(self) -> None:
        self.present = True
        self.layout = LAYOUT
        self.on_detect = None
        self.calls = 0
        self._condition = threading.Condition()

    def detect(self, _image: np.ndarray) -> DetectedLayout | None:
        with self._condition:
            self.calls += 1
            if self.on_detect is not None:
                self.on_detect(self.calls)
            self._condition.notify_all()
            return self.layout if self.present else None

    def wait_after(self, calls: int, timeout: float = 0.5) -> None:
        with self._condition:
            reached = self._condition.wait_for(
                lambda: self.calls > calls,
                timeout=timeout,
            )
        if not reached:
            raise AssertionError("layout detector did not receive the frame")


class GatedPipeline:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.started: defaultdict[int, threading.Event] = defaultdict(threading.Event)
        self.finished: defaultdict[int, threading.Event] = defaultdict(threading.Event)
        self.gates: defaultdict[int, threading.Event] = defaultdict(threading.Event)
        self._lock = threading.Lock()

    def recognize(
        self,
        frame: CapturedFrame,
        generation_id: int,
    ) -> RecognitionResult:
        marker = int(frame.bgr[0, 9, 0])
        with self._lock:
            self.calls.append(marker)
        self.started[marker].set()
        if not self.gates[marker].wait(timeout=2):
            raise TimeoutError(f"pipeline gate {marker} was not released")
        recognized = result(generation_id, frame.frame_id, marker)
        self.finished[marker].set()
        return recognized


class ImmediatePipeline(GatedPipeline):
    def recognize(
        self,
        frame: CapturedFrame,
        generation_id: int,
    ) -> RecognitionResult:
        marker = int(frame.bgr[0, 9, 0])
        with self._lock:
            self.calls.append(marker)
        self.started[marker].set()
        return result(generation_id, frame.frame_id, marker)


class RecordingExecutor:
    def __init__(self) -> None:
        self.futures: list[Future[RecognitionResult]] = []
        self.shutdown_calls: list[tuple[bool, bool]] = []
        self.submit_calls = 0

    def submit(self, _function, *_args) -> Future[RecognitionResult]:
        self.submit_calls += 1
        future: Future[RecognitionResult] = Future()
        self.futures.append(future)
        return future

    def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
    ) -> None:
        self.shutdown_calls.append((wait, cancel_futures))
        if cancel_futures:
            for future in self.futures:
                future.cancel()


def result(generation: int, frame_id: int, marker: int) -> RecognitionResult:
    return RecognitionResult(
        generation_id=generation,
        frame_id=frame_id,
        question_text=f"题目-{marker}",
        option_texts=("甲", "乙", "丙", "丁"),
        official_answer="乙",
        question_score=99.0,
        question_runner_up_score=1.0,
        option_score=99.0,
        option_runner_up_score=1.0,
        high_confidence=True,
        option_index=1,
        overlay_rect=LAYOUT.option_rects[1],
        timings=RecognitionTimings(1.0, 2.0, 3.0, 6.0),
    )


def frame(frame_id: int, marker: int) -> CapturedFrame:
    image = np.zeros((10, 36, 3), dtype=np.uint8)
    # A seeded question pattern gives each marker a stable dHash.
    rng = np.random.default_rng(marker)
    image[:8, :9] = rng.integers(0, 256, size=(8, 9, 1), dtype=np.uint8)
    image[0, 9] = marker
    return CapturedFrame.create(frame_id, time.monotonic_ns(), image)


def frame_with_option_change(
    frame_id: int,
    marker: int,
    option_index: int,
) -> CapturedFrame:
    captured = frame(frame_id, marker)
    image = captured.bgr.copy()
    rect = LAYOUT.option_rects[option_index]
    rng = np.random.default_rng(10_000 + option_index)
    image[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width] = (
        rng.integers(0, 256, size=(rect.height, rect.width, 1), dtype=np.uint8)
    )
    return CapturedFrame.create(frame_id, time.monotonic_ns(), image)


def frame_with_background_change(frame_id: int, marker: int) -> CapturedFrame:
    captured = frame(frame_id, marker)
    image = np.zeros((12, 36, 3), dtype=np.uint8)
    image[:10] = captured.bgr
    image[10:] = 255
    return CapturedFrame.create(frame_id, time.monotonic_ns(), image)


def wait_until(predicate, timeout: float = 0.5) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError("condition was not reached before timeout")
        time.sleep(min(0.005, remaining))


def make_coordinator(pipeline):
    capture = FakeCaptureService()
    hub = LatestFrameHub()
    detector = FakeLayoutDetector()
    store = RuntimeStore()
    coordinator = RecognitionCoordinator(capture, hub, detector, pipeline, store)
    return capture, hub, detector, pipeline, store, coordinator


def publish_detected(
    hub: LatestFrameHub,
    detector: FakeLayoutDetector,
    captured_frame: CapturedFrame,
) -> None:
    calls = detector.calls
    hub.publish(captured_frame)
    detector.wait_after(calls)


def test_changed_hash_invalidates_running_generation_before_old_result() -> None:
    _capture, hub, detector, pipeline, store, coordinator = make_coordinator(
        GatedPipeline()
    )
    coordinator.start()
    try:
        publish_detected(hub, detector, frame(1, 1))
        publish_detected(hub, detector, frame(2, 1))
        assert pipeline.started[1].wait(timeout=0.5)
        old_generation = store.snapshot().generation_id

        changed_at = time.monotonic()
        publish_detected(hub, detector, frame(3, 2))
        wait_until(lambda: store.snapshot().generation_id > old_generation)
        cleared = store.snapshot()
        assert time.monotonic() - changed_at < 0.1
        assert cleared.overlay is None
        assert cleared.phase is RuntimePhase.MONITORING

        publish_detected(hub, detector, frame(4, 2))
        pipeline.gates[1].set()
        assert pipeline.started[2].wait(timeout=0.5)
        assert store.snapshot().question_text != "题目-1"
        pipeline.gates[2].set()
        wait_until(lambda: store.snapshot().question_text == "题目-2")
        assert store.snapshot().phase is RuntimePhase.ANSWERED
    finally:
        pipeline.gates[1].set()
        pipeline.gates[2].set()
        coordinator.stop()


def test_same_hash_is_recognized_once_and_cached_after_return() -> None:
    _capture, hub, detector, pipeline, store, coordinator = make_coordinator(
        ImmediatePipeline()
    )
    coordinator.start()
    try:
        publish_detected(hub, detector, frame(1, 3))
        publish_detected(hub, detector, frame(2, 3))
        wait_until(lambda: store.snapshot().phase is RuntimePhase.ANSWERED)
        assert pipeline.calls == [3]

        publish_detected(hub, detector, frame(3, 3))
        publish_detected(hub, detector, frame(4, 3))
        assert pipeline.calls == [3]

        detector.present = False
        publish_detected(hub, detector, frame(5, 0))
        wait_until(lambda: store.snapshot().phase is RuntimePhase.MONITORING)
        detector.present = True
        publish_detected(hub, detector, frame(6, 3))
        publish_detected(hub, detector, frame(7, 3))
        wait_until(lambda: store.snapshot().phase is RuntimePhase.ANSWERED)
        assert pipeline.calls == [3]
    finally:
        coordinator.stop()


def test_background_change_outside_all_rois_reuses_recognition() -> None:
    _capture, hub, detector, pipeline, _store, coordinator = make_coordinator(
        ImmediatePipeline()
    )
    coordinator.start()
    try:
        publish_detected(hub, detector, frame(1, 3))
        publish_detected(hub, detector, frame(2, 3))
        wait_until(lambda: pipeline.calls == [3])

        publish_detected(hub, detector, frame_with_background_change(3, 3))
        time.sleep(0.05)

        assert pipeline.calls == [3]
    finally:
        coordinator.stop()


def test_change_in_any_option_roi_invalidates_cached_recognition() -> None:
    _capture, hub, detector, pipeline, _store, coordinator = make_coordinator(
        ImmediatePipeline()
    )
    coordinator.start()
    try:
        publish_detected(hub, detector, frame(1, 3))
        publish_detected(hub, detector, frame(2, 3))
        wait_until(lambda: pipeline.calls == [3])

        publish_detected(hub, detector, frame_with_option_change(3, 3, 2))
        publish_detected(hub, detector, frame_with_option_change(4, 3, 2))
        wait_until(lambda: len(pipeline.calls) == 2)

        assert pipeline.calls == [3, 3]
    finally:
        coordinator.stop()


def test_cache_identity_blocks_reuse_when_stability_signature_collides(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        coordinator_module,
        "_quiz_stability_signature",
        lambda _frame, _layout: "same-stability-signature",
    )
    _capture, hub, detector, pipeline, _store, coordinator = make_coordinator(
        ImmediatePipeline()
    )
    coordinator.start()
    try:
        publish_detected(hub, detector, frame(1, 3))
        publish_detected(hub, detector, frame(2, 3))
        wait_until(lambda: pipeline.calls == [3])

        publish_detected(hub, detector, frame_with_option_change(3, 3, 1))
        publish_detected(hub, detector, frame_with_option_change(4, 3, 1))
        wait_until(lambda: len(pipeline.calls) == 2)
    finally:
        coordinator.stop()


def test_cache_invalidation_discards_late_old_knowledge_result() -> None:
    _capture, hub, detector, pipeline, _store, coordinator = make_coordinator(
        GatedPipeline()
    )
    coordinator.start()
    try:
        publish_detected(hub, detector, frame(1, 7))
        publish_detected(hub, detector, frame(2, 7))
        assert pipeline.started[7].wait(timeout=0.5)

        coordinator.invalidate_cache()
        pipeline.gates[7].set()
        assert pipeline.finished[7].wait(timeout=0.5)

        detector.present = False
        publish_detected(hub, detector, frame(3, 0))
        detector.present = True
        publish_detected(hub, detector, frame(4, 7))
        publish_detected(hub, detector, frame(5, 7))
        wait_until(lambda: pipeline.calls.count(7) == 2)
    finally:
        pipeline.gates[7].set()
        coordinator.stop()


def test_capture_empty_clears_and_never_calls_pipeline() -> None:
    capture, hub, _detector, pipeline, store, coordinator = make_coordinator(
        ImmediatePipeline()
    )
    capture.set_phase(CapturePhase.CAPTURE_EMPTY)
    coordinator.start()
    try:
        hub.publish(frame(1, 4))
        hub.publish(frame(2, 4))
        wait_until(lambda: store.snapshot().phase is RuntimePhase.CAPTURE_EMPTY)
        assert store.snapshot().overlay is None
        assert pipeline.calls == []
    finally:
        coordinator.stop()


def test_capture_becoming_empty_during_confirmation_skips_pipeline() -> None:
    capture, hub, detector, pipeline, store, coordinator = make_coordinator(
        ImmediatePipeline()
    )
    detector.on_detect = lambda calls: (
        capture.set_phase(CapturePhase.CAPTURE_EMPTY)
        if calls == 2
        else None
    )
    coordinator.start()
    try:
        publish_detected(hub, detector, frame(1, 8))
        publish_detected(hub, detector, frame(2, 8))
        wait_until(lambda: store.snapshot().phase is RuntimePhase.CAPTURE_EMPTY)

        assert pipeline.calls == []
        assert store.snapshot().overlay is None
    finally:
        coordinator.stop()


def test_stop_does_not_wait_for_running_recognition_and_discards_late_result() -> None:
    _capture, hub, detector, pipeline, store, coordinator = make_coordinator(
        GatedPipeline()
    )
    coordinator.start()
    publish_detected(hub, detector, frame(1, 9))
    publish_detected(hub, detector, frame(2, 9))
    assert pipeline.started[9].wait(timeout=0.5)
    stop_finished = threading.Event()
    stop_error: list[BaseException] = []

    def stop_coordinator() -> None:
        try:
            coordinator.stop()
        except BaseException as exc:
            stop_error.append(exc)
        finally:
            stop_finished.set()

    stopper = threading.Thread(target=stop_coordinator)
    stopper.start()
    try:
        assert stop_finished.wait(timeout=0.15)
        assert stop_error == []
        stopped = store.snapshot()
        assert stopped.phase is RuntimePhase.WAITING_FOR_WINDOW
        assert stopped.overlay is None

        pipeline.gates[9].set()
        assert pipeline.finished[9].wait(timeout=0.5)
        assert store.snapshot() == stopped
    finally:
        pipeline.gates[9].set()
        stopper.join(timeout=2.5)
        coordinator.stop()


def test_start_is_idempotent_while_running_but_rejected_after_stop() -> None:
    capture = FakeCaptureService()
    hub = LatestFrameHub()
    detector = FakeLayoutDetector()
    pipeline = GatedPipeline()
    store = RuntimeStore()
    executors: list[ThreadPoolExecutor] = []

    def executor_factory() -> ThreadPoolExecutor:
        executor = ThreadPoolExecutor(max_workers=1)
        executors.append(executor)
        return executor

    coordinator = RecognitionCoordinator(
        capture,
        hub,
        detector,
        pipeline,
        store,
        executor_factory=executor_factory,
    )
    coordinator.start()
    coordinator.start()
    assert len(executors) == 1
    publish_detected(hub, detector, frame(1, 12))
    publish_detected(hub, detector, frame(2, 12))
    assert pipeline.started[12].wait(timeout=0.5)

    coordinator.stop(timeout=0.1)
    stopped = store.snapshot()
    try:
        try:
            coordinator.start()
        except RuntimeError as exc:
            assert "cannot be restarted after stop" in str(exc)
        else:
            raise AssertionError("stopped coordinator was restarted")
        assert len(executors) == 1
        assert pipeline.calls == [12]

        pipeline.gates[12].set()
        assert pipeline.finished[12].wait(timeout=0.5)
        assert store.snapshot() == stopped
    finally:
        pipeline.gates[12].set()
        coordinator.stop(timeout=0.5)


def test_worker_timeout_still_shuts_down_executor_and_cancels_pending() -> None:
    capture = FakeCaptureService()
    hub = LatestFrameHub()
    detector = FakeLayoutDetector()
    pipeline = ImmediatePipeline()
    store = RuntimeStore()
    executor = RecordingExecutor()
    detector_entered = threading.Event()
    detector_release = threading.Event()

    def block_third_detection(calls: int) -> None:
        if calls == 3:
            detector_entered.set()
            detector_release.wait(timeout=2)

    detector.on_detect = block_third_detection
    coordinator = RecognitionCoordinator(
        capture,
        hub,
        detector,
        pipeline,
        store,
        executor_factory=lambda: executor,
    )
    coordinator.start()
    try:
        publish_detected(hub, detector, frame(1, 10))
        publish_detected(hub, detector, frame(2, 10))
        wait_until(lambda: executor.submit_calls == 1)
        assert executor.futures[0].done() is False

        hub.publish(frame(3, 11))
        assert detector_entered.wait(timeout=0.5)
        started = time.monotonic()
        try:
            coordinator.stop(timeout=0.03)
        except RuntimeError as exc:
            assert "did not stop within 0.03 seconds" in str(exc)
        else:
            raise AssertionError("blocked coordinator worker did not time out")
        assert time.monotonic() - started < 0.2
        assert executor.shutdown_calls == [(False, True)]
        assert executor.futures[0].cancelled()
        try:
            coordinator.start()
        except RuntimeError as exc:
            assert "cannot be restarted after stop" in str(exc)
        else:
            raise AssertionError("stopping coordinator was restarted")
        assert executor.submit_calls == 1

        detector_release.set()
        coordinator.stop(timeout=0.5)
        assert executor.submit_calls == 1
        assert store.snapshot().phase is RuntimePhase.WAITING_FOR_WINDOW
    finally:
        detector_release.set()
        coordinator.stop(timeout=0.5)


def test_layout_disappearance_publishes_clear_within_100ms() -> None:
    _capture, hub, detector, _pipeline, store, coordinator = make_coordinator(
        ImmediatePipeline()
    )
    coordinator.start()
    try:
        publish_detected(hub, detector, frame(1, 5))
        publish_detected(hub, detector, frame(2, 5))
        wait_until(lambda: store.snapshot().phase is RuntimePhase.ANSWERED)

        detector.present = False
        changed_at = time.monotonic()
        publish_detected(hub, detector, frame(3, 5))
        wait_until(lambda: store.snapshot().phase is RuntimePhase.MONITORING)

        assert time.monotonic() - changed_at < 0.1
        assert store.snapshot().overlay is None
        assert store.snapshot().clear_monotonic_ns is not None
    finally:
        coordinator.stop()


def test_layout_coordinate_change_clears_and_invalidates_cached_answer() -> None:
    _capture, hub, detector, pipeline, store, coordinator = make_coordinator(
        ImmediatePipeline()
    )
    coordinator.start()
    try:
        publish_detected(hub, detector, frame(1, 6))
        publish_detected(hub, detector, frame(2, 6))
        wait_until(lambda: store.snapshot().phase is RuntimePhase.ANSWERED)
        answered_generation = store.snapshot().generation_id
        detector.layout = DetectedLayout(
            question_rect=LAYOUT.question_rect,
            option_rects=(
                Rect(0, 7, 9, 2),
                Rect(9, 7, 9, 2),
                Rect(18, 7, 9, 2),
                Rect(27, 7, 9, 2),
            ),
            anchor_scores=(0.99, 0.99),
        )

        publish_detected(hub, detector, frame(3, 6))

        snapshot = store.snapshot()
        assert snapshot.generation_id > answered_generation
        assert snapshot.phase is RuntimePhase.MONITORING
        assert snapshot.overlay is None
        assert snapshot.message == "layout_changed"

        publish_detected(hub, detector, frame(4, 6))
        wait_until(lambda: store.snapshot().phase is RuntimePhase.ANSWERED)
        assert pipeline.calls == [6, 6]
        assert store.snapshot().overlay is not None
    finally:
        coordinator.stop()


def test_initial_missing_layout_publishes_clear_transition() -> None:
    _capture, hub, detector, _pipeline, store, coordinator = make_coordinator(
        ImmediatePipeline()
    )
    detector.present = False
    coordinator.start()
    try:
        publish_detected(hub, detector, frame(1, 7))

        snapshot = store.snapshot()
        assert snapshot.phase is RuntimePhase.MONITORING
        assert snapshot.message == "dialog_missing"
        assert snapshot.clear_monotonic_ns is not None
    finally:
        coordinator.stop()


def test_ambiguous_multi_profile_layout_clears_without_pipeline_overlay() -> None:
    class StaticDetector:
        def __init__(self, layout: DetectedLayout) -> None:
            self.layout = layout

        def detect(self, _image: np.ndarray) -> DetectedLayout:
            return self.layout

    other = DetectedLayout(
        question_rect=Rect(1, 0, 9, 8),
        option_rects=LAYOUT.option_rects,
        anchor_scores=(1.0, 1.0),
    )
    detector = MultiProfileLayoutDetector(
        (("first", StaticDetector(LAYOUT)), ("second", StaticDetector(other)))
    )
    capture = FakeCaptureService()
    hub = LatestFrameHub()
    pipeline = ImmediatePipeline()
    store = RuntimeStore()
    coordinator = RecognitionCoordinator(capture, hub, detector, pipeline, store)
    coordinator.start()
    try:
        hub.publish(frame(1, 7))
        wait_until(lambda: store.snapshot().message == "dialog_missing")
        assert store.snapshot().overlay is None
        assert pipeline.calls == []
    finally:
        coordinator.stop()


def test_invalid_anchor_fit_does_not_stop_coordinator_and_overlay_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pattern = np.arange(64, dtype=np.uint8).reshape(8, 8)
    assert cv2.imwrite(str(tmp_path / "one.png"), pattern)
    assert cv2.imwrite(str(tmp_path / "two.png"), np.rot90(pattern))
    profile_path = tmp_path / "layout.json"
    profile_path.write_text(
        json.dumps({
            "reference_size": [100, 80],
            "question_rect": [0.0, 0.0, 0.2, 0.2],
            "option_rects": [
                [0.0, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2],
                [0.4, 0.2, 0.2, 0.2],
                [0.6, 0.2, 0.2, 0.2],
            ],
            "anchors": [
                {
                    "reference_rect": [0.1, 0.1, 0.08, 0.1],
                    "search_rect": [0, 0, 1, 1],
                    "template_path": "one.png",
                    "threshold": 0.8,
                },
                {
                    "reference_rect": [0.8, 0.7, 0.08, 0.1],
                    "search_rect": [0, 0, 1, 1],
                    "template_path": "two.png",
                    "threshold": 0.8,
                },
            ],
        }),
        encoding="utf-8",
    )
    detector = TemplateLayoutDetector(LayoutProfile.load(profile_path))
    invalid = (
        _AnchorMatch(0.99, 150, 16, 16, 16),
        _AnchorMatch(0.99, 20, 112, 16, 16),
    )
    valid = (
        _AnchorMatch(0.99, 20, 16, 16, 16),
        _AnchorMatch(0.99, 160, 112, 16, 16),
    )
    queued = iter((*invalid, *valid, *valid))
    monkeypatch.setattr(
        detector,
        "_match_anchor",
        lambda *_args, **_kwargs: next(queued),
    )
    capture = FakeCaptureService()
    hub = LatestFrameHub()
    pipeline = ImmediatePipeline()
    store = RuntimeStore()
    coordinator = RecognitionCoordinator(capture, hub, detector, pipeline, store)

    def large_frame(frame_id: int) -> CapturedFrame:
        image = np.zeros((160, 200, 3), dtype=np.uint8)
        image[0, 9] = 7
        return CapturedFrame.create(frame_id, time.monotonic_ns(), image)

    coordinator.start()
    try:
        hub.publish(large_frame(1))
        wait_until(lambda: store.snapshot().message == "dialog_missing")
        assert store.snapshot().overlay is None

        hub.publish(large_frame(2))
        time.sleep(0.02)
        hub.publish(large_frame(3))
        wait_until(lambda: store.snapshot().phase is RuntimePhase.ANSWERED)
        assert store.snapshot().overlay is not None
        assert pipeline.calls == [7]
    finally:
        coordinator.stop()


def test_cache_is_bounded_to_128_entries() -> None:
    _capture, hub, detector, pipeline, store, coordinator = make_coordinator(
        ImmediatePipeline()
    )
    coordinator.start()
    try:
        next_frame_id = 1
        for marker in range(1, 131):
            publish_detected(hub, detector, frame(next_frame_id, marker))
            publish_detected(hub, detector, frame(next_frame_id + 1, marker))
            next_frame_id += 2
            wait_until(lambda marker=marker: marker in pipeline.calls)

        detector.present = False
        publish_detected(hub, detector, frame(next_frame_id, 0))
        wait_until(lambda: store.snapshot().phase is RuntimePhase.MONITORING)
        detector.present = True
        publish_detected(hub, detector, frame(next_frame_id + 1, 1))
        publish_detected(hub, detector, frame(next_frame_id + 2, 1))
        wait_until(lambda: pipeline.calls.count(1) == 2)
        assert pipeline.calls.count(1) == 2
    finally:
        coordinator.stop()
