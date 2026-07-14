from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
import threading

import numpy as np
import pytest

from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.models import (
    CapturedFrame,
    CapturePhase,
    Rect,
    WindowTarget,
)
from xyq_quiz.capture.service import CaptureService, STATUS_EVENT_CAPACITY
from xyq_quiz.capture.wgc import WGCCapture, WGCCaptureStats
from xyq_quiz.config import AppConfig


class FakeFinder:
    def __init__(self, target: WindowTarget) -> None:
        self._condition = threading.Condition()
        self._target = target
        self._present = True
        self.calls = 0

    def __call__(self) -> list[WindowTarget]:
        with self._condition:
            self.calls += 1
            self._condition.notify_all()
            return [self._target] if self._present else []

    def set_present(self, present: bool) -> None:
        with self._condition:
            self._present = present
            self._condition.notify_all()

    def wait_for_calls(self, expected: int) -> None:
        with self._condition:
            assert self._condition.wait_for(
                lambda: self.calls >= expected,
                timeout=1,
            )


class BlockingFinder:
    def __init__(self, target: WindowTarget) -> None:
        self._target = target
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self) -> list[WindowTarget]:
        self.entered.set()
        assert self.release.wait(timeout=1)
        return [self._target]


class FakeWGC:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: CapturedFrame | None = None
        self._last_read_frame_id = 0
        self._stats_reads = 0
        self._running = False
        self._hwnd: int | None = None
        self.start_calls: list[int] = []
        self.close_count = 0

    def start(
        self,
        hwnd: int,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        with self._condition:
            if cancelled is not None and cancelled():
                return False
            self.start_calls.append(hwnd)
            self._running = True
            self._hwnd = hwnd
            self._condition.notify_all()
        return True

    def latest(self) -> CapturedFrame | None:
        with self._condition:
            frame = self._latest
            if frame is not None:
                self._last_read_frame_id = max(
                    self._last_read_frame_id,
                    frame.frame_id,
                )
                self._condition.notify_all()
            return frame

    def stats(self) -> WGCCaptureStats:
        with self._condition:
            self._stats_reads += 1
            self._condition.notify_all()
            return WGCCaptureStats(
                frame_count=0 if self._latest is None else self._latest.frame_id,
                frame_age_ms=None,
                content_change_count=0,
                running=self._running,
                hwnd=self._hwnd,
            )

    def close(self) -> None:
        with self._condition:
            self.close_count += 1
            self._running = False
            self._hwnd = None
            self._latest = None
            self._condition.notify_all()

    def wait_until_started(self) -> None:
        with self._condition:
            assert self._condition.wait_for(
                lambda: self._running and self._stats_reads > 0,
                timeout=1,
            )

    def wait_until_closed(self) -> None:
        with self._condition:
            assert self._condition.wait_for(
                lambda: self.close_count == 1,
                timeout=1,
            )

    def emit_black(self, count: int) -> None:
        self._emit(count, pixel_value=0)

    def emit_color(self) -> None:
        self._emit(1, pixel_value=64)

    def disappear(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()

    def _emit(self, count: int, pixel_value: int) -> None:
        for _ in range(count):
            with self._condition:
                assert self._condition.wait_for(lambda: self._running, timeout=1)
                frame_id = 1 if self._latest is None else self._latest.frame_id + 1
                self._latest = CapturedFrame.create(
                    frame_id=frame_id,
                    captured_at_ns=frame_id * 100,
                    bgr=np.full((80, 100, 3), pixel_value, dtype=np.uint8),
                )
                self._condition.notify_all()
                assert self._condition.wait_for(
                    lambda: self._last_read_frame_id >= frame_id,
                    timeout=1,
                )
                stats_reads_after_read = self._stats_reads
                assert self._condition.wait_for(
                    lambda: self._stats_reads > stats_reads_after_read,
                    timeout=1,
                )


class CountingFactory:
    def __init__(self, capture: FakeWGC) -> None:
        self._capture = capture
        self.calls = 0

    def __call__(self) -> FakeWGC:
        self.calls += 1
        return self._capture


class BlockingStartWGC(FakeWGC):
    def __init__(self) -> None:
        super().__init__()
        self.start_entered = threading.Event()
        self.closed_during_start = False

    def start(
        self,
        hwnd: int,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        with self._condition:
            if cancelled is not None and cancelled():
                return False
            self.start_calls.append(hwnd)
            self._hwnd = hwnd
            self.start_entered.set()
            self.closed_during_start = self._condition.wait_for(
                lambda: self.close_count > 0,
                timeout=0.25,
            )
            return False


class NativeControl:
    def __init__(self) -> None:
        self.stop_count = 0

    def stop(self) -> None:
        self.stop_count += 1


class NativeSession:
    def __init__(self) -> None:
        self.control = NativeControl()

    def event(self, callback: Callable[..., None]) -> Callable[..., None]:
        return callback

    def start_free_threaded(self) -> NativeControl:
        return self.control


class NativeSessionFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.sessions: list[NativeSession] = []

    def __call__(self, **_kwargs: object) -> NativeSession:
        self.calls += 1
        session = NativeSession()
        self.sessions.append(session)
        return session


class PausedBeforeNativeStartWGC(WGCCapture):
    def __init__(self) -> None:
        self.session_factory = NativeSessionFactory()
        super().__init__(factory=self.session_factory)
        self.before_native_start = threading.Event()
        self.release_native_start = threading.Event()
        self.closed = threading.Event()
        self.close_count = 0

    def start(
        self,
        hwnd: int,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        self.before_native_start.set()
        assert self.release_native_start.wait(timeout=1)
        if cancelled is None:
            return super().start(hwnd)
        return super().start(hwnd, cancelled=cancelled)

    def close(self) -> None:
        super().close()
        self.close_count += 1
        self.closed.set()


def config() -> AppConfig:
    return AppConfig()


def target() -> WindowTarget:
    return WindowTarget(
        hwnd=0x2011D0,
        title="梦幻西游 ONLINE",
        process_id=4321,
        process_name="mhtab.exe",
        class_name="MHXYMainFrame",
        rect=Rect(10, 20, 1292, 1023),
    )


def build_service(
    *,
    hub: LatestFrameHub | None = None,
) -> tuple[CaptureService, LatestFrameHub, FakeWGC, FakeFinder, CountingFactory]:
    shared_hub = hub or LatestFrameHub()
    fake_wgc = FakeWGC()
    fake_finder = FakeFinder(target())
    factory = CountingFactory(fake_wgc)
    service = CaptureService(config(), shared_hub, fake_finder, factory)
    return service, shared_hub, fake_wgc, fake_finder, factory


def run_started(
    check: Callable[[CaptureService, LatestFrameHub, FakeWGC, FakeFinder], None],
) -> None:
    service, hub, fake_wgc, fake_finder, factory = build_service()
    service.start()
    try:
        fake_wgc.wait_until_started()
        check(service, hub, fake_wgc, fake_finder)
        assert factory.calls == 1
    finally:
        service.stop()


def test_service_reports_empty_after_ten_black_frames() -> None:
    def check(
        service: CaptureService,
        hub: LatestFrameHub,
        fake_wgc: FakeWGC,
        _fake_finder: FakeFinder,
    ) -> None:
        fake_wgc.emit_black(10)

        assert service.status().phase is CapturePhase.CAPTURE_EMPTY
        latest = hub.snapshot()
        assert latest is not None and latest.frame_id == 10

    run_started(check)


def test_service_recovers_when_nonblack_frame_arrives() -> None:
    def check(
        service: CaptureService,
        _hub: LatestFrameHub,
        fake_wgc: FakeWGC,
        _fake_finder: FakeFinder,
    ) -> None:
        fake_wgc.emit_black(10)
        assert service.status().phase is CapturePhase.CAPTURE_EMPTY

        fake_wgc.emit_color()

        assert service.status().phase is CapturePhase.CAPTURING

    run_started(check)


def test_service_records_capture_phase_transitions_without_per_frame_duplicates() -> None:
    def check(
        service: CaptureService,
        _hub: LatestFrameHub,
        fake_wgc: FakeWGC,
        _fake_finder: FakeFinder,
    ) -> None:
        fake_wgc.emit_black(10)
        fake_wgc.emit_color()

        assert [status.phase for status in service.status_events()] == [
            CapturePhase.WAITING_FOR_WINDOW,
            CapturePhase.CAPTURING,
            CapturePhase.CAPTURE_EMPTY,
            CapturePhase.CAPTURING,
        ]

    run_started(check)


def test_status_event_cursor_returns_only_transitions_after_sampling_start() -> None:
    def check(
        service: CaptureService,
        _hub: LatestFrameHub,
        fake_wgc: FakeWGC,
        _fake_finder: FakeFinder,
    ) -> None:
        cursor = service.status_event_cursor()
        fake_wgc.emit_black(10)
        fake_wgc.emit_color()

        window = service.status_events_after(cursor)

        assert window.gap is False
        assert [event.status.phase for event in window.events] == [
            CapturePhase.CAPTURE_EMPTY,
            CapturePhase.CAPTURING,
        ]
        assert [event.sequence for event in window.events] == sorted(
            event.sequence for event in window.events
        )

    run_started(check)


def test_status_sampling_start_atomically_contains_cursor_and_current_status() -> None:
    def check(
        service: CaptureService,
        _hub: LatestFrameHub,
        fake_wgc: FakeWGC,
        _fake_finder: FakeFinder,
    ) -> None:
        start = service.status_sampling_start()
        with pytest.raises(FrozenInstanceError):
            start.cursor = 999  # type: ignore[misc]
        fake_wgc.emit_black(10)

        assert start.cursor < service.status_event_cursor()
        assert start.current_status.phase is CapturePhase.CAPTURING
        assert service.status_events_after(start.cursor).events[0].status.phase \
            is CapturePhase.CAPTURE_EMPTY

    run_started(check)


def test_status_event_history_is_bounded_and_reports_overflow_gap() -> None:
    service, _hub, _fake_wgc, _finder, _factory = build_service()
    old_cursor = service.status_event_cursor()
    for index in range(STATUS_EVENT_CAPACITY + 50):
        service._set_status(
            CapturePhase.CAPTURING if index % 2 else CapturePhase.ERROR,
            target(),
            str(index),
        )

    window = service.status_events_after(old_cursor)

    assert len(service.status_events()) == STATUS_EVENT_CAPACITY
    assert len(window.events) == STATUS_EVENT_CAPACITY
    assert window.gap is True


def test_service_closes_capture_and_resumes_discovery_when_window_disappears() -> None:
    service, _hub, fake_wgc, fake_finder, factory = build_service()
    service.start()
    try:
        fake_wgc.wait_until_started()
        fake_finder.set_present(False)
        fake_wgc.disappear()
        fake_finder.wait_for_calls(2)

        status = service.status()
        assert status.phase is CapturePhase.WAITING_FOR_WINDOW
        assert status.target is None
        assert fake_wgc.close_count == 1
        assert factory.calls == 1
    finally:
        service.stop()

    assert fake_wgc.close_count == 1


def test_stop_closes_owned_wgc_once() -> None:
    service, _hub, fake_wgc, _fake_finder, factory = build_service()
    service.start()
    fake_wgc.wait_until_started()

    service.stop()
    service.stop()

    assert fake_wgc.close_count == 1
    assert factory.calls == 1


def test_service_exposes_native_arrival_and_content_change_stats() -> None:
    service, _hub, fake_wgc, _finder, _factory = build_service()
    service.start()
    try:
        fake_wgc.wait_until_started()
        fake_wgc.emit_color()

        stats = service.capture_stats()

        assert stats.frame_count == 1
        assert stats.running is True
        assert stats.hwnd == target().hwnd
    finally:
        service.stop()


def test_stop_invalidates_wgc_start_before_joining_worker() -> None:
    fake_wgc = BlockingStartWGC()
    factory = CountingFactory(fake_wgc)
    service = CaptureService(config(), LatestFrameHub(), FakeFinder(target()), factory)
    service.start()
    assert fake_wgc.start_entered.wait(timeout=1)

    service.stop()

    assert fake_wgc.closed_during_start is True
    assert fake_wgc.close_count == 1
    assert factory.calls == 1


def test_stop_prevents_wgc_start_after_blocked_finder_returns() -> None:
    fake_wgc = FakeWGC()
    finder = BlockingFinder(target())
    factory = CountingFactory(fake_wgc)
    service = CaptureService(config(), LatestFrameHub(), finder, factory)
    stop_errors: list[BaseException] = []
    stop_returned = threading.Event()

    def stop_service() -> None:
        try:
            service.stop()
        except BaseException as exc:
            stop_errors.append(exc)
        finally:
            stop_returned.set()

    service.start()
    assert finder.entered.wait(timeout=1)
    stop_thread = threading.Thread(target=stop_service)
    stop_thread.start()
    try:
        fake_wgc.wait_until_closed()
        assert stop_returned.is_set() is False
        finder.release.set()
        assert stop_returned.wait(timeout=1)
        stop_thread.join(timeout=1)
    finally:
        finder.release.set()
        stop_thread.join(timeout=1)

    worker = service._worker
    assert stop_errors == []
    assert stop_thread.is_alive() is False
    assert worker is not None and worker.is_alive() is False
    assert fake_wgc.stats().running is False
    assert fake_wgc.start_calls == []
    assert service.status().phase is not CapturePhase.CAPTURING
    assert fake_wgc.close_count == 1
    assert factory.calls == 1


def test_stop_cancels_start_before_wgc_reserves_its_generation() -> None:
    capture = PausedBeforeNativeStartWGC()
    service = CaptureService(
        config(),
        LatestFrameHub(),
        FakeFinder(target()),
        lambda: capture,
    )
    stop_errors: list[BaseException] = []
    stop_returned = threading.Event()

    def stop_service() -> None:
        try:
            service.stop()
        except BaseException as exc:
            stop_errors.append(exc)
        finally:
            stop_returned.set()

    service.start()
    assert capture.before_native_start.wait(timeout=1)
    stop_thread = threading.Thread(target=stop_service)
    stop_thread.start()
    try:
        assert capture.closed.wait(timeout=1)
        assert stop_returned.is_set() is False
        capture.release_native_start.set()
        assert stop_returned.wait(timeout=1)
        stop_thread.join(timeout=1)
    finally:
        capture.release_native_start.set()
        stop_thread.join(timeout=1)

    worker = service._worker
    assert stop_errors == []
    assert stop_thread.is_alive() is False
    assert worker is not None and worker.is_alive() is False
    assert capture.stats().running is False
    assert capture.session_factory.calls == 0
    assert capture.close_count == 1
    assert service.status().phase is not CapturePhase.CAPTURING
