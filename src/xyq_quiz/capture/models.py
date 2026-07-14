from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("rectangle dimensions must be positive")

    def normalized(
        self,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float, float, float]:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        return (
            self.x / frame_width,
            self.y / frame_height,
            self.width / frame_width,
            self.height / frame_height,
        )


@dataclass(frozen=True, slots=True)
class WindowTarget:
    hwnd: int
    title: str
    process_id: int
    process_name: str
    class_name: str
    rect: Rect


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    frame_id: int
    captured_at_ns: int
    bgr: NDArray[np.uint8]

    @classmethod
    def create(
        cls,
        frame_id: int,
        captured_at_ns: int,
        bgr: NDArray[np.uint8],
    ) -> CapturedFrame:
        bgr.setflags(write=False)
        return cls(frame_id, captured_at_ns, bgr)


class CapturePhase(StrEnum):
    WAITING_FOR_WINDOW = "WAITING_FOR_WINDOW"
    CAPTURING = "CAPTURING"
    CAPTURE_EMPTY = "CAPTURE_EMPTY"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class CaptureStatus:
    phase: CapturePhase
    target: WindowTarget | None = None
    message: str | None = None
