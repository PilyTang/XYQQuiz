from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import threading

import numpy as np

from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.models import (
    CapturedFrame,
    CapturePhase,
    CaptureStatus,
    WindowTarget,
)
from xyq_quiz.capture.wgc import WGCCapture, WGCCaptureStats
from xyq_quiz.capture.windowing import enumerate_windows, select_window
from xyq_quiz.config import AppConfig


WindowFinder = Callable[[], list[WindowTarget]]
WGCCaptureFactory = Callable[[], WGCCapture]

# At most one transition per 10 ms still retains 5.12 seconds of history;
# acceptance fails closed if a longer burst overwrites its sampling cursor.
STATUS_EVENT_CAPACITY = 512


@dataclass(frozen=True, slots=True)
class CaptureStatusEvent:
    sequence: int
    status: CaptureStatus


@dataclass(frozen=True, slots=True)
class CaptureStatusEventWindow:
    events: tuple[CaptureStatusEvent, ...]
    gap: bool


@dataclass(frozen=True, slots=True)
class CaptureStatusSamplingStart:
    cursor: int
    current_status: CaptureStatus


class CaptureService:
    def __init__(
        self,
        config: AppConfig,
        hub: LatestFrameHub,
        window_finder: WindowFinder = enumerate_windows,
        wgc_factory: WGCCaptureFactory = WGCCapture,
    ) -> None:
        self._config = config
        self._hub = hub
        self._window_finder = window_finder
        self._wgc = wgc_factory()
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._lifecycle_generation = 0
        self._stop_event = threading.Event()
        self._start_cancelled = threading.Event()
        self._worker: threading.Thread | None = None
        self._status = CaptureStatus(CapturePhase.WAITING_FOR_WINDOW)
        self._status_event_sequence = 0
        self._status_events: deque[CaptureStatusEvent] = deque(
            [CaptureStatusEvent(0, self._status)],
            maxlen=STATUS_EVENT_CAPACITY,
        )
        self._wgc_needs_close = True

    def start(self) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if self._worker is not None and self._worker.is_alive():
                    return
                self._lifecycle_generation += 1
                generation = self._lifecycle_generation
                self._stop_event.clear()
                start_cancelled = threading.Event()
                self._start_cancelled = start_cancelled
                self._status = CaptureStatus(CapturePhase.WAITING_FOR_WINDOW)
                self._record_status_event_locked(self._status)
                self._wgc_needs_close = True
                worker = threading.Thread(
                    target=self._run,
                    args=(generation, start_cancelled),
                    name="xyq-capture-service",
                    daemon=True,
                )
                self._worker = worker
            worker.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop_event.set()
            self._start_cancelled.set()
            self._lifecycle_generation += 1
            self._close_wgc_once()
            self._set_status(CapturePhase.WAITING_FOR_WINDOW)
            with self._lock:
                worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2)
            if worker.is_alive():
                raise RuntimeError("capture worker did not stop within two seconds")

    def status(self) -> CaptureStatus:
        with self._lock:
            return self._status

    def capture_stats(self) -> WGCCaptureStats:
        """Return native callback counts without fabricating repeated frames."""
        return self._wgc.stats()

    def status_events(self) -> tuple[CaptureStatus, ...]:
        """Snapshot every phase/target transition for continuous acceptance."""
        with self._lock:
            return tuple(event.status for event in self._status_events)

    def status_event_cursor(self) -> int:
        with self._lock:
            return self._status_event_sequence

    def status_sampling_start(self) -> CaptureStatusSamplingStart:
        """Atomically bind an event cursor to the status visible at that cursor."""
        with self._lock:
            return CaptureStatusSamplingStart(
                cursor=self._status_event_sequence,
                current_status=self._status,
            )

    def status_events_after(self, cursor: int) -> CaptureStatusEventWindow:
        with self._lock:
            oldest_sequence = self._status_events[0].sequence
            return CaptureStatusEventWindow(
                events=tuple(
                    event for event in self._status_events
                    if event.sequence > cursor
                ),
                gap=cursor < oldest_sequence - 1,
            )

    def _run(
        self,
        generation: int,
        start_cancelled: threading.Event,
    ) -> None:
        target: WindowTarget | None = None
        last_frame_id = 0
        black_frame_count = 0
        frame_poll_interval = 1 / self._config.capture.preview_fps

        try:
            while self._generation_is_active(generation):
                if target is None:
                    target = select_window(
                        self._window_finder(),
                        self._config.window.process_names,
                        self._config.window.class_names,
                    )
                    if target is None:
                        self._stop_event.wait(timeout=0.5)
                        continue

                    with self._lifecycle_lock:
                        if not self._generation_is_active_locked(generation):
                            return
                        self._mark_wgc_open()
                    started = self._wgc.start(
                        target.hwnd,
                        cancelled=start_cancelled.is_set,
                    )
                    with self._lifecycle_lock:
                        if not self._generation_is_active_locked(generation):
                            return
                        if not started:
                            raise RuntimeError("WGC capture did not start")
                        self._set_status(CapturePhase.CAPTURING, target)
                    last_frame_id = 0
                    black_frame_count = 0
                    continue

                if not self._wgc.stats().running:
                    with self._lifecycle_lock:
                        if not self._generation_is_active_locked(generation):
                            return
                        self._close_wgc_once()
                        target = None
                        last_frame_id = 0
                        black_frame_count = 0
                        self._set_status(CapturePhase.WAITING_FOR_WINDOW)
                    continue

                frame = self._wgc.latest()
                if frame is not None and frame.frame_id > last_frame_id:
                    last_frame_id = frame.frame_id
                    self._hub.publish(frame)
                    if _is_black_frame(frame):
                        black_frame_count += 1
                    else:
                        black_frame_count = 0

                    phase = (
                        CapturePhase.CAPTURE_EMPTY
                        if black_frame_count >= self._config.capture.black_frame_count
                        else CapturePhase.CAPTURING
                    )
                    with self._lifecycle_lock:
                        if not self._generation_is_active_locked(generation):
                            return
                        self._set_status(phase, target)

                self._stop_event.wait(timeout=frame_poll_interval)
        except Exception as exc:
            with self._lifecycle_lock:
                if self._generation_is_active_locked(generation):
                    self._set_status(CapturePhase.ERROR, target, str(exc))

    def _generation_is_active(self, generation: int) -> bool:
        with self._lifecycle_lock:
            return self._generation_is_active_locked(generation)

    def _generation_is_active_locked(self, generation: int) -> bool:
        return (
            self._lifecycle_generation == generation
            and not self._stop_event.is_set()
        )

    def _set_status(
        self,
        phase: CapturePhase,
        target: WindowTarget | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            status = CaptureStatus(phase, target, message)
            self._status = status
            self._record_status_event_locked(status)

    def _record_status_event_locked(self, status: CaptureStatus) -> None:
        if self._status_events[-1].status == status:
            return
        self._status_event_sequence += 1
        self._status_events.append(
            CaptureStatusEvent(self._status_event_sequence, status)
        )

    def _mark_wgc_open(self) -> None:
        with self._lock:
            self._wgc_needs_close = True

    def _close_wgc_once(self) -> None:
        with self._lock:
            if not self._wgc_needs_close:
                return
            self._wgc_needs_close = False
        self._wgc.close()


def _is_black_frame(frame: CapturedFrame) -> bool:
    image = frame.bgr
    if image.ndim != 3 or image.size == 0:
        return True

    height, width = image.shape[:2]
    rows = np.linspace(0, height - 1, num=min(64, height), dtype=np.intp)
    columns = np.linspace(0, width - 1, num=min(64, width), dtype=np.intp)
    sample = image[np.ix_(rows, columns)]
    return float(sample.mean()) < 2.0 and float(sample.std()) < 1.0
