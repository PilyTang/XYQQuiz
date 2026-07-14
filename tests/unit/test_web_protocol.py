from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from xyq_quiz.web.protocol import encode_frame_packet


def test_frame_packet_has_big_endian_id_and_jpeg() -> None:
    packet = encode_frame_packet(42, b"\xff\xd8jpeg")

    assert int.from_bytes(packet[:8], "big") == 42
    assert packet[8:] == b"\xff\xd8jpeg"


@pytest.mark.parametrize("frame_id", [-1, 2**64])
def test_frame_packet_rejects_ids_outside_unsigned_64_bit(frame_id: int) -> None:
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        encode_frame_packet(frame_id, b"jpeg")


def test_testclient_warning_filter_is_exact_and_anchored() -> None:
    project = Path(__file__).parents[2] / "pyproject.toml"
    config = tomllib.loads(project.read_text(encoding="utf-8"))

    assert config["tool"]["pytest"]["ini_options"]["filterwarnings"] == [
        "ignore:^Using `httpx` with `starlette\\.testclient` is deprecated; "
        "install `httpx2` instead\\.$:"
        "starlette.exceptions.StarletteDeprecationWarning:"
        "^fastapi\\.testclient$"
    ]
