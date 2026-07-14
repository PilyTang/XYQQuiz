import numpy as np

from xyq_quiz.capture.models import CapturedFrame, Rect


def test_rect_normalizes_against_frame_size() -> None:
    assert Rect(100, 50, 200, 100).normalized(1000, 500) == (
        0.1,
        0.1,
        0.2,
        0.2,
    )


def test_captured_frame_is_read_only() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    frame = CapturedFrame.create(7, 123456, image)
    assert frame.frame_id == 7
    assert frame.bgr.flags.writeable is False
