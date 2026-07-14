from __future__ import annotations

import numpy as np

from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.models import CapturedFrame


def frame(frame_id: int) -> CapturedFrame:
    return CapturedFrame.create(
        frame_id=frame_id,
        captured_at_ns=frame_id * 100,
        bgr=np.full((4, 6, 3), frame_id, dtype=np.uint8),
    )


def test_hub_keeps_only_latest_frame() -> None:
    hub = LatestFrameHub()

    hub.publish(frame(1))
    hub.publish(frame(2))

    latest = hub.snapshot()
    assert latest is not None and latest.frame_id == 2
    waited = hub.wait_after(1, 0.01)
    assert waited is not None and waited.frame_id == 2


def test_wait_after_times_out_without_queue_growth() -> None:
    hub = LatestFrameHub()
    hub.publish(frame(3))

    assert hub.wait_after(3, 0.01) is None
