from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading
import time
from typing import Any

import numpy as np

from xyq_quiz.capture.models import CapturedFrame


class WGCCaptureUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WGCCaptureStats:
    frame_count: int
    frame_age_ms: float | None
    content_change_count: int
    running: bool
    hwnd: int | None


class WGCCapture:
    def __init__(self, factory: Callable[..., Any] | None = None) -> None:
        self._factory = factory
        self._lock = threading.Lock()
        self._capture: Any | None = None
        self._capture_control: Any | None = None
        self._latest: CapturedFrame | None = None
        self._frame_count = 0
        self._last_signature: int | None = None
        self._content_change_count = 0
        self._running = False
        self._hwnd: int | None = None
        self._generation = 0

    def start(
        self,
        hwnd: int,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        with self._lock:
            if cancelled is not None and cancelled():
                return False
            if self._running and self._hwnd == hwnd:
                return True

            previous_capture = self._capture
            previous_control = self._capture_control
            self._generation += 1
            generation = self._generation
            self._reset_state_locked()
        self._stop_session(previous_capture, previous_control)

        capture: Any | None = None
        try:
            factory = self._factory or _windows_capture_factory()
            capture = factory(
                window_hwnd=hwnd,
                cursor_capture=False,
                draw_border=False,
            )

            @capture.event
            def on_frame_arrived(frame: Any, capture_control: Any) -> None:
                with self._lock:
                    accepting_frames = (
                        self._generation == generation
                        and self._running
                        and self._capture is capture
                    )
                if not accepting_frames:
                    capture_control.stop()
                    return

                raw_bgra = frame.frame_buffer
                if raw_bgra is None or raw_bgra.ndim != 3 or raw_bgra.shape[2] < 4:
                    return

                bgr = np.ascontiguousarray(raw_bgra[:, :, :3])
                height, width = bgr.shape[:2]
                sample = bgr[:: max(1, height // 64), :: max(1, width // 64)]
                signature = int(sample.sum(dtype=np.uint64) % 1_000_000_007)
                captured_at_ns = time.perf_counter_ns()

                with self._lock:
                    if (
                        self._generation != generation
                        or not self._running
                        or self._capture is not capture
                    ):
                        capture_control.stop()
                        return
                    self._frame_count += 1
                    if self._last_signature != signature:
                        self._last_signature = signature
                        self._content_change_count += 1
                    self._latest = CapturedFrame.create(
                        frame_id=self._frame_count,
                        captured_at_ns=captured_at_ns,
                        bgr=bgr,
                    )

            @capture.event
            def on_closed(*_args: Any) -> None:
                with self._lock:
                    if self._generation == generation and self._capture is capture:
                        self._running = False

            with self._lock:
                if self._generation == generation:
                    self._capture = capture
                    self._hwnd = hwnd
                    self._running = True

            control = capture.start_free_threaded()
        except Exception:
            self._cleanup_failed_start(generation, capture)
            raise

        with self._lock:
            capture_is_current = (
                self._generation == generation and self._capture is capture
            )
            if capture_is_current:
                self._capture_control = control
        if not capture_is_current and control is not None:
            control.stop()
        return capture_is_current

    def latest(self) -> CapturedFrame | None:
        with self._lock:
            return self._latest

    def stats(self) -> WGCCaptureStats:
        now_ns = time.perf_counter_ns()
        with self._lock:
            age_ms = (
                None
                if self._latest is None
                else max(0.0, (now_ns - self._latest.captured_at_ns) / 1_000_000)
            )
            return WGCCaptureStats(
                frame_count=self._frame_count,
                frame_age_ms=age_ms,
                content_change_count=self._content_change_count,
                running=self._running,
                hwnd=self._hwnd,
            )

    def close(self) -> None:
        with self._lock:
            capture = self._capture
            capture_control = self._capture_control
            self._generation += 1
            self._reset_state_locked()
        self._stop_session(capture, capture_control)

    def _cleanup_failed_start(self, generation: int, capture: Any | None) -> None:
        with self._lock:
            if self._generation != generation:
                return
            current_capture = self._capture
            current_control = self._capture_control
            self._generation += 1
            self._reset_state_locked()
        cleanup_capture = current_capture if current_capture is not None else capture
        self._stop_session(cleanup_capture, current_control)

    def _reset_state_locked(self) -> None:
        self._capture = None
        self._capture_control = None
        self._latest = None
        self._frame_count = 0
        self._last_signature = None
        self._content_change_count = 0
        self._running = False
        self._hwnd = None

    @staticmethod
    def _stop_session(capture: Any | None, capture_control: Any | None) -> None:
        target = capture_control if capture_control is not None else capture
        if target is not None and hasattr(target, "stop"):
            target.stop()


def _windows_capture_factory() -> Callable[..., Any]:
    try:
        from windows_capture import WindowsCapture
    except Exception as exc:  # pragma: no cover - environment dependent
        raise WGCCaptureUnavailable(
            "Install WGC support with: python -m pip install windows-capture"
        ) from exc
    return WindowsCapture
