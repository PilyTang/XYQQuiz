from __future__ import annotations

import threading

from xyq_quiz.capture.models import CapturedFrame


class LatestFrameHub:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: CapturedFrame | None = None

    def publish(self, frame: CapturedFrame) -> None:
        with self._condition:
            self._latest = frame
            self._condition.notify_all()

    def snapshot(self) -> CapturedFrame | None:
        with self._condition:
            return self._latest

    def wait_after(self, frame_id: int, timeout: float) -> CapturedFrame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._latest is not None
                and self._latest.frame_id > frame_id,
                timeout=timeout,
            )
            if self._latest is not None and self._latest.frame_id > frame_id:
                return self._latest
            return None
