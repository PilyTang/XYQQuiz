from __future__ import annotations

from xyq_quiz.capture.video import LatestVideoHub


def _publish(hub: LatestVideoHub, frame_id: int, *, key_frame: bool = True) -> None:
    hub.publish(
        frame_id=frame_id,
        timestamp_us=frame_id * 1_000,
        width=4,
        height=2,
        key_frame=key_frame,
        payload=bytes((frame_id,)),
    )


def test_video_hub_returns_every_new_frame_without_a_gap() -> None:
    hub = LatestVideoHub(capacity=4)
    _publish(hub, 1)
    _publish(hub, 2)

    window = hub.wait_after(0, 0)

    assert window.gap is False
    assert [frame.frame_id for frame in window.frames] == [1, 2]


def test_video_hub_recovers_a_gap_from_the_latest_complete_frame() -> None:
    hub = LatestVideoHub(capacity=2)
    _publish(hub, 1)
    _publish(hub, 2)
    _publish(hub, 3)

    window = hub.wait_after(0, 0)

    assert window.gap is True
    assert [frame.frame_id for frame in window.frames] == [3]


def test_video_hub_requests_a_refresh_through_the_current_callback() -> None:
    calls: list[str] = []
    hub = LatestVideoHub(request_key_frame=lambda: calls.append("first"))
    hub.request_key_frame()
    hub.set_key_frame_requester(lambda: calls.append("second"))
    hub.request_key_frame()

    assert calls == ["first", "second"]


def test_video_hub_switches_pixel_format_and_discards_old_frames() -> None:
    hub = LatestVideoHub()
    _publish(hub, 1)

    hub.set_pixel_format("bgra")

    assert hub.pixel_format == "bgra"
    assert hub.wait_after(0, 0).frames == ()

    hub.set_pixel_format("i420")
    assert hub.pixel_format == "i420"


def test_video_hub_rejects_unknown_pixel_format() -> None:
    try:
        LatestVideoHub(pixel_format="rgb")
    except ValueError as error:
        assert "unsupported preview pixel format" in str(error)
    else:
        raise AssertionError("unknown pixel format was accepted")
