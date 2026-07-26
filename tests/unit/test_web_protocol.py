from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from xyq_quiz.web.protocol import (
    encode_bgra_packet,
    encode_frame_packet,
    encode_i420_packet,
    encode_nv12_packet,
)


def test_frame_packet_has_big_endian_id_and_jpeg() -> None:
    packet = encode_frame_packet(42, b"\xff\xd8jpeg")

    assert int.from_bytes(packet[:8], "big") == 42
    assert packet[8:] == b"\xff\xd8jpeg"


@pytest.mark.parametrize("frame_id", [-1, 2**64])
def test_frame_packet_rejects_ids_outside_unsigned_64_bit(frame_id: int) -> None:
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        encode_frame_packet(frame_id, b"jpeg")


def test_i420_packet_has_metadata_and_planar_payload() -> None:
    payload = bytes(range(24))
    packet = encode_i420_packet(
        frame_id=7,
        timestamp_us=123,
        width=4,
        height=4,
        i420=payload,
    )

    assert int.from_bytes(packet[0:8], "big") == 7
    assert int.from_bytes(packet[8:16], "big", signed=True) == 123
    assert int.from_bytes(packet[16:20], "big") == 4
    assert int.from_bytes(packet[20:24], "big") == 4
    assert packet[24:] == payload


def test_nv12_packet_keeps_native_flags_header() -> None:
    payload = bytes(range(12))
    packet = encode_nv12_packet(
        frame_id=9,
        timestamp_us=456,
        width=4,
        height=2,
        nv12=payload,
    )

    assert packet[0] == 1
    assert int.from_bytes(packet[1:9], "big") == 9
    assert int.from_bytes(packet[9:17], "big", signed=True) == 456
    assert packet[25:] == payload


def test_bgra_packet_keeps_raw_video_header_and_payload() -> None:
    payload = bytes(range(32))
    packet = encode_bgra_packet(
        frame_id=10,
        timestamp_us=789,
        width=4,
        height=2,
        bgra=payload,
    )

    assert packet[0] == 1
    assert int.from_bytes(packet[1:9], "big") == 10
    assert int.from_bytes(packet[9:17], "big", signed=True) == 789
    assert int.from_bytes(packet[17:21], "big") == 4
    assert int.from_bytes(packet[21:25], "big") == 2
    assert packet[25:] == payload


def test_bgra_packet_rejects_wrong_payload_size() -> None:
    with pytest.raises(ValueError, match="BGRA payload size"):
        encode_bgra_packet(
            frame_id=1,
            timestamp_us=1,
            width=2,
            height=2,
            bgra=b"too short",
        )


def test_testclient_warning_filter_is_exact_and_anchored() -> None:
    project = Path(__file__).parents[2] / "pyproject.toml"
    config = tomllib.loads(project.read_text(encoding="utf-8"))

    assert config["tool"]["pytest"]["ini_options"]["filterwarnings"] == [
        "ignore:^Using `httpx` with `starlette\\.testclient` is deprecated; "
        "install `httpx2` instead\\.$:"
        "starlette.exceptions.StarletteDeprecationWarning:"
        "^fastapi\\.testclient$"
    ]
