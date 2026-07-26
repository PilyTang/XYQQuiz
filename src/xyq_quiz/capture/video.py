from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Callable


@dataclass(frozen=True, slots=True)
class EncodedVideoFrame:
    sequence: int
    frame_id: int
    timestamp_us: int
    width: int
    height: int
    key_frame: bool
    payload: bytes


@dataclass(frozen=True, slots=True)
class EncodedVideoWindow:
    frames: tuple[EncodedVideoFrame, ...]
    gap: bool


class LatestVideoHub:
    """Small latest-frame video buffer with explicit format transitions."""

    def __init__(
        self,
        *,
        capacity: int = 90,
        request_key_frame: Callable[[], None] | None = None,
        pixel_format: str = "nv12",
    ) -> None:
        if capacity < 2:
            raise ValueError("video hub capacity must be at least two")
        if pixel_format not in {"nv12", "bgra", "i420"}:
            raise ValueError("unsupported preview pixel format")
        self._condition = threading.Condition()
        self._frames: deque[EncodedVideoFrame] = deque(maxlen=capacity)
        self._sequence = 0
        self._request_key_frame = request_key_frame
        self._pixel_format = pixel_format

    @property
    def pixel_format(self) -> str:
        with self._condition:
            return self._pixel_format

    def set_pixel_format(self, value: str) -> None:
        if value not in {"nv12", "bgra", "i420"}:
            raise ValueError("unsupported preview pixel format")
        with self._condition:
            if self._pixel_format == value:
                return
            self._pixel_format = value
            self._frames.clear()
            self._condition.notify_all()

    def set_key_frame_requester(self, callback: Callable[[], None] | None) -> None:
        with self._condition:
            self._request_key_frame = callback

    def publish(
        self,
        *,
        frame_id: int,
        timestamp_us: int,
        width: int,
        height: int,
        key_frame: bool,
        payload: bytes,
    ) -> EncodedVideoFrame:
        if not payload:
            raise ValueError("video frame payload must not be empty")
        with self._condition:
            self._sequence += 1
            frame = EncodedVideoFrame(
                sequence=self._sequence,
                frame_id=frame_id,
                timestamp_us=timestamp_us,
                width=width,
                height=height,
                key_frame=key_frame,
                payload=bytes(payload),
            )
            self._frames.append(frame)
            self._condition.notify_all()
            return frame

    def wait_after(self, sequence: int, timeout: float) -> EncodedVideoWindow:
        with self._condition:
            self._condition.wait_for(
                lambda: bool(self._frames) and self._frames[-1].sequence > sequence,
                timeout=timeout,
            )
            if not self._frames or self._frames[-1].sequence <= sequence:
                return EncodedVideoWindow((), False)
            oldest = self._frames[0].sequence
            gap = sequence < oldest - 1
            if gap:
                key_index = next(
                    (
                        index
                        for index in range(len(self._frames) - 1, -1, -1)
                        if self._frames[index].key_frame
                    ),
                    None,
                )
                frames = (
                    tuple(list(self._frames)[key_index:])
                    if key_index is not None
                    else ()
                )
            else:
                frames = tuple(frame for frame in self._frames if frame.sequence > sequence)
            return EncodedVideoWindow(frames, gap)

    def request_key_frame(self) -> None:
        with self._condition:
            callback = self._request_key_frame
        if callback is not None:
            callback()

    def clear(self) -> None:
        with self._condition:
            self._frames.clear()
            self._condition.notify_all()


__all__ = ["EncodedVideoFrame", "EncodedVideoWindow", "LatestVideoHub"]
