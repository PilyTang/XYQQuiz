from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading
import time
from typing import Any

import cv2
import numpy as np

from xyq_quiz.capture.models import CapturedFrame
from xyq_quiz.capture.video import LatestVideoHub


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
    def __init__(
        self,
        factory: Callable[..., Any] | None = None,
        *,
        minimum_update_interval_ms: int | None = None,
        video_hub: LatestVideoHub | None = None,
        preview_width: int = 1024,
        preview_fps: int | None = None,
        recognition_fps: int | None = None,
    ) -> None:
        if (
            minimum_update_interval_ms is not None
            and (
                not isinstance(minimum_update_interval_ms, int)
                or isinstance(minimum_update_interval_ms, bool)
                or minimum_update_interval_ms <= 0
            )
        ):
            raise ValueError("minimum_update_interval_ms must be a positive integer")
        if (
            not isinstance(preview_width, int)
            or isinstance(preview_width, bool)
            or preview_width <= 0
        ):
            raise ValueError("preview_width must be a positive integer")
        if (
            preview_fps is not None
            and (
                not isinstance(preview_fps, int)
                or isinstance(preview_fps, bool)
                or preview_fps <= 0
            )
        ):
            raise ValueError("preview_fps must be a positive integer")
        if (
            recognition_fps is not None
            and (
                not isinstance(recognition_fps, int)
                or isinstance(recognition_fps, bool)
                or recognition_fps <= 0
            )
        ):
            raise ValueError("recognition_fps must be a positive integer")
        self._factory = factory
        self._minimum_update_interval_ms = minimum_update_interval_ms
        self._video_hub = video_hub
        self._preview_width = preview_width
        self._preview_interval_ns = (
            None if preview_fps is None else 1_000_000_000 // preview_fps
        )
        self._recognition_interval_ns = (
            None if recognition_fps is None else 1_000_000_000 // recognition_fps
        )
        if self._video_hub is not None:
            self._video_hub.set_pixel_format("i420")
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
        self._last_preview_ns = 0
        self._last_recognition_ns = 0

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
            capture_options: dict[str, Any] = {
                "window_hwnd": hwnd,
                "cursor_capture": False,
                "draw_border": False,
            }
            if self._minimum_update_interval_ms is not None:
                # Ask Windows Graphics Capture to throttle before Python copies
                # the full BGRA frame.  Polling the already-copied latest frame
                # later cannot recover this CPU cost.
                capture_options["minimum_update_interval"] = (
                    self._minimum_update_interval_ms
                )
            capture = factory(**capture_options)

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

                captured_at_ns = time.perf_counter_ns()
                height, width = raw_bgra.shape[:2]

                with self._lock:
                    if (
                        self._generation != generation
                        or not self._running
                        or self._capture is not capture
                    ):
                        capture_control.stop()
                        return
                    self._frame_count += 1
                    frame_id = self._frame_count
                    recognition_due = (
                        self._recognition_interval_ns is None
                        or self._last_recognition_ns == 0
                        or captured_at_ns - self._last_recognition_ns
                        >= self._recognition_interval_ns
                    )
                    if recognition_due:
                        self._last_recognition_ns = captured_at_ns
                    preview_due = self._video_hub is not None and (
                        self._preview_interval_ns is None
                        or self._last_preview_ns == 0
                        or captured_at_ns - self._last_preview_ns
                        >= self._preview_interval_ns
                    )
                    if preview_due:
                        self._last_preview_ns = captured_at_ns

                # The native ``minimum_update_interval`` hint can be ignored,
                # so reject over-rate callbacks before sampling or copying the
                # full frame.  Preview and recognition keep independent clocks.
                if not recognition_due and not preview_due:
                    return

                sample = raw_bgra[
                    :: max(1, height // 64),
                    :: max(1, width // 64),
                    :3,
                ]
                signature = int(sample.sum(dtype=np.uint64) % 1_000_000_007)
                with self._lock:
                    if (
                        self._generation != generation
                        or not self._running
                        or self._capture is not capture
                    ):
                        capture_control.stop()
                        return
                    if self._last_signature != signature:
                        self._last_signature = signature
                        self._content_change_count += 1

                if self._video_hub is not None and preview_due:
                    preview_width = min(width, self._preview_width)
                    preview_width = max(2, preview_width - preview_width % 2)
                    preview_height = max(
                        2,
                        round(height * preview_width / width),
                    )
                    preview_height -= preview_height % 2
                    preview = raw_bgra[:, :, :4]
                    if (preview_width, preview_height) != (width, height):
                        preview = cv2.resize(
                            preview,
                            (preview_width, preview_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    preview_i420 = cv2.cvtColor(
                        preview,
                        cv2.COLOR_BGRA2YUV_I420,
                    )
                    self._video_hub.publish(
                        frame_id=frame_id,
                        timestamp_us=captured_at_ns // 1_000,
                        width=preview_width,
                        height=preview_height,
                        key_frame=True,
                        payload=preview_i420.tobytes(),
                    )

                if recognition_due:
                    bgr = cv2.cvtColor(raw_bgra, cv2.COLOR_BGRA2BGR)
                    with self._lock:
                        if (
                            self._generation == generation
                            and self._running
                            and self._capture is capture
                        ):
                            self._latest = CapturedFrame.create(
                                frame_id=frame_id,
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
        self._last_preview_ns = 0
        self._last_recognition_ns = 0

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
