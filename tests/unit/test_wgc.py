from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading
import time
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from xyq_quiz.capture.wgc import WGCCapture


@dataclass
class FakeFrame:
    frame_buffer: NDArray[np.uint8]
    content_size_changed: bool = False


class FakeControl:
    def __init__(self) -> None:
        self.stop_count = 0

    def stop(self) -> None:
        self.stop_count += 1


class FakeSession:
    def __init__(self) -> None:
        self._callbacks: dict[str, Callable[..., None]] = {}
        self.started = False
        self.control = FakeControl()
        self.callback_control = FakeControl()

    def event(self, callback: Callable[..., None]) -> Callable[..., None]:
        self._callbacks[callback.__name__] = callback
        return callback

    def start_free_threaded(self) -> FakeControl:
        self.started = True
        return self.control

    def emit(
        self,
        frame_buffer: NDArray[np.uint8],
        *,
        content_size_changed: bool = False,
    ) -> None:
        self._callbacks["on_frame_arrived"](
            FakeFrame(frame_buffer, content_size_changed),
            self.callback_control,
        )


class FakeFactory:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []
        self.calls: list[dict[str, Any]] = []

    @property
    def session(self) -> FakeSession:
        return self.sessions[-1]

    def __call__(self, **kwargs: Any) -> FakeSession:
        self.calls.append(kwargs)
        session = FakeSession()
        self.sessions.append(session)
        return session


class BlockingSession(FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.start_entered = threading.Event()
        self.start_release = threading.Event()

    def start_free_threaded(self) -> FakeControl:
        self.started = True
        self.start_entered.set()
        if not self.start_release.wait(timeout=1):
            raise TimeoutError("test did not release capture startup")
        return self.control


class BlockingFactory(FakeFactory):
    def __init__(self) -> None:
        super().__init__()
        self.session_created = threading.Event()

    def __call__(self, **kwargs: Any) -> BlockingSession:
        self.calls.append(kwargs)
        session = BlockingSession()
        self.sessions.append(session)
        self.session_created.set()
        return session


class FirstBlockingFactory(BlockingFactory):
    def __call__(self, **kwargs: Any) -> FakeSession:
        self.calls.append(kwargs)
        session = BlockingSession() if not self.sessions else FakeSession()
        self.sessions.append(session)
        self.session_created.set()
        return session


class BlockingConstructionFactory(FakeFactory):
    def __init__(self) -> None:
        super().__init__()
        self.construction_entered = threading.Event()
        self.construction_release = threading.Event()
        self.prepared_session = FakeSession()

    def __call__(self, **kwargs: Any) -> FakeSession:
        self.calls.append(kwargs)
        self.construction_entered.set()
        if not self.construction_release.wait(timeout=1):
            raise TimeoutError("test did not release capture construction")
        self.sessions.append(self.prepared_session)
        return self.prepared_session


class RaisingBlockingSession(FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.start_entered = threading.Event()
        self.start_release = threading.Event()

    def start_free_threaded(self) -> FakeControl:
        self.started = True
        self.start_entered.set()
        if not self.start_release.wait(timeout=1):
            raise TimeoutError("test did not release failing startup")
        raise RuntimeError("first capture startup failed")


class FirstRaisingFactory(BlockingFactory):
    def __call__(self, **kwargs: Any) -> FakeSession:
        self.calls.append(kwargs)
        session = RaisingBlockingSession() if not self.sessions else FakeSession()
        self.sessions.append(session)
        self.session_created.set()
        return session


@pytest.fixture
def fake_factory() -> FakeFactory:
    return FakeFactory()


def test_wgc_publishes_latest_immutable_frame(fake_factory: FakeFactory) -> None:
    capture = WGCCapture(factory=fake_factory)
    capture.start(0x2011D0)

    bgra = np.zeros((10, 20, 4), dtype=np.uint8)
    bgra[:, :, 0] = 11
    bgra[:, :, 1] = 22
    bgra[:, :, 2] = 33
    bgra[:, :, 3] = 255
    fake_factory.session.emit(bgra)

    frame = capture.latest()
    assert frame is not None and frame.frame_id == 1
    assert frame.bgr.shape == (10, 20, 3)
    assert frame.bgr.flags.c_contiguous is True
    assert frame.bgr.flags.writeable is False
    assert tuple(frame.bgr[0, 0]) == (11, 22, 33)
    assert capture.stats().frame_count == 1


def test_wgc_does_not_wait_or_fallback_when_no_frame(fake_factory: FakeFactory) -> None:
    capture = WGCCapture(factory=fake_factory)
    capture.start(123)

    started = time.perf_counter()
    assert capture.latest() is None
    assert time.perf_counter() - started < 0.01


def test_wgc_start_is_idempotent_and_replaces_changed_hwnd(
    fake_factory: FakeFactory,
) -> None:
    capture = WGCCapture(factory=fake_factory)

    capture.start(123)
    first = fake_factory.session
    capture.start(123)
    assert len(fake_factory.sessions) == 1

    capture.start(456)
    assert first.control.stop_count == 1
    assert len(fake_factory.sessions) == 2
    assert fake_factory.calls == [
        {"window_hwnd": 123, "cursor_capture": False, "draw_border": False},
        {"window_hwnd": 456, "cursor_capture": False, "draw_border": False},
    ]
    assert fake_factory.session.started is True
    assert capture.stats().hwnd == 456


def test_wgc_stats_track_content_changes_age_and_close(
    fake_factory: FakeFactory,
) -> None:
    capture = WGCCapture(factory=fake_factory)
    capture.start(123)
    session = fake_factory.session
    session.emit(
        np.zeros((2, 3, 4), dtype=np.uint8),
        content_size_changed=True,
    )

    stats = capture.stats()
    assert stats.frame_count == 1
    assert stats.frame_age_ms is not None and stats.frame_age_ms >= 0
    assert stats.content_change_count == 1
    assert stats.running is True
    assert stats.hwnd == 123

    capture.close()
    assert session.control.stop_count == 1
    assert capture.latest() is None
    assert capture.stats().running is False
    assert capture.stats().hwnd is None


def test_wgc_stops_control_returned_after_close_during_start() -> None:
    factory = BlockingFactory()
    capture = WGCCapture(factory=factory)
    start_thread = threading.Thread(target=capture.start, args=(123,))

    start_thread.start()
    assert factory.session_created.wait(timeout=1)
    session = factory.session
    assert isinstance(session, BlockingSession)
    assert session.start_entered.wait(timeout=1)
    capture.close()
    session.start_release.set()
    start_thread.join(timeout=1)

    assert start_thread.is_alive() is False
    assert session.control.stop_count == 1
    assert capture.stats().running is False


def test_wgc_stops_late_control_and_ignores_late_frame_after_replacement() -> None:
    factory = FirstBlockingFactory()
    capture = WGCCapture(factory=factory)
    first_start = threading.Thread(target=capture.start, args=(123,))

    first_start.start()
    assert factory.session_created.wait(timeout=1)
    first = factory.session
    assert isinstance(first, BlockingSession)
    assert first.start_entered.wait(timeout=1)

    capture.start(456)
    second = factory.session
    second.emit(np.ones((2, 3, 4), dtype=np.uint8))
    current = capture.latest()
    assert current is not None

    first.start_release.set()
    first_start.join(timeout=1)
    assert first_start.is_alive() is False
    assert first.control.stop_count == 1

    first.emit(np.zeros((2, 3, 4), dtype=np.uint8))
    assert first.callback_control.stop_count == 1
    assert capture.latest() is current
    assert capture.stats().running is True
    assert capture.stats().hwnd == 456

    capture.close()
    assert second.control.stop_count == 1


def test_wgc_close_invalidates_start_blocked_before_capture_publication() -> None:
    factory = BlockingConstructionFactory()
    capture = WGCCapture(factory=factory)
    start_thread = threading.Thread(target=capture.start, args=(123,))

    start_thread.start()
    assert factory.construction_entered.wait(timeout=1)
    capture.close()
    factory.construction_release.set()
    start_thread.join(timeout=1)

    assert start_thread.is_alive() is False
    assert factory.prepared_session.control.stop_count == 1
    assert capture.stats().running is False
    assert capture.stats().hwnd is None


def test_wgc_stale_start_exception_does_not_close_newer_capture() -> None:
    factory = FirstRaisingFactory()
    capture = WGCCapture(factory=factory)
    errors: list[BaseException] = []

    def start_first() -> None:
        try:
            capture.start(123)
        except BaseException as exc:
            errors.append(exc)

    first_start = threading.Thread(target=start_first)
    first_start.start()
    assert factory.session_created.wait(timeout=1)
    first = factory.session
    assert isinstance(first, RaisingBlockingSession)
    assert first.start_entered.wait(timeout=1)

    capture.start(456)
    second = factory.session
    first.start_release.set()
    first_start.join(timeout=1)

    assert first_start.is_alive() is False
    assert len(errors) == 1
    assert str(errors[0]) == "first capture startup failed"
    assert second.control.stop_count == 0
    assert capture.stats().running is True
    assert capture.stats().hwnd == 456

    capture.close()
    assert second.control.stop_count == 1
