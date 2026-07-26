from __future__ import annotations


def encode_frame_packet(frame_id: int, jpeg: bytes) -> bytes:
    """Encode an unsigned 64-bit big-endian frame id followed by JPEG bytes."""
    if not 0 <= frame_id < 2**64:
        raise ValueError("frame_id must fit in an unsigned 64-bit integer")
    return frame_id.to_bytes(8, "big", signed=False) + jpeg


def encode_h264_packet(
    *,
    frame_id: int,
    timestamp_us: int,
    width: int,
    height: int,
    key_frame: bool,
    h264: bytes,
) -> bytes:
    if not 0 <= frame_id < 2**64:
        raise ValueError("frame_id must fit in an unsigned 64-bit integer")
    if not -(2**63) <= timestamp_us < 2**63:
        raise ValueError("timestamp_us must fit in a signed 64-bit integer")
    if not 0 < width < 2**32 or not 0 < height < 2**32:
        raise ValueError("video dimensions must fit in unsigned 32-bit integers")
    if not h264:
        raise ValueError("H.264 payload must not be empty")
    flags = 1 if key_frame else 0
    return b"".join(
        (
            bytes((flags,)),
            frame_id.to_bytes(8, "big", signed=False),
            timestamp_us.to_bytes(8, "big", signed=True),
            width.to_bytes(4, "big", signed=False),
            height.to_bytes(4, "big", signed=False),
            h264,
        )
    )


def encode_i420_packet(
    *,
    frame_id: int,
    timestamp_us: int,
    width: int,
    height: int,
    i420: bytes,
) -> bytes:
    if not 0 <= frame_id < 2**64:
        raise ValueError("frame_id must fit in an unsigned 64-bit integer")
    if not -(2**63) <= timestamp_us < 2**63:
        raise ValueError("timestamp_us must fit in a signed 64-bit integer")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("I420 dimensions must be positive even integers")
    if len(i420) != width * height * 3 // 2:
        raise ValueError("I420 payload size does not match its dimensions")
    return b"".join(
        (
            frame_id.to_bytes(8, "big", signed=False),
            timestamp_us.to_bytes(8, "big", signed=True),
            width.to_bytes(4, "big", signed=False),
            height.to_bytes(4, "big", signed=False),
            i420,
        )
    )


def encode_nv12_packet(
    *,
    frame_id: int,
    timestamp_us: int,
    width: int,
    height: int,
    nv12: bytes,
) -> bytes:
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("NV12 dimensions must be positive even integers")
    if len(nv12) != width * height * 3 // 2:
        raise ValueError("NV12 payload size does not match its dimensions")
    # Keep the leading flags byte used by the native stream contract.  A raw
    # plane is independently displayable, so bit 0 is always set.
    return encode_h264_packet(
        frame_id=frame_id,
        timestamp_us=timestamp_us,
        width=width,
        height=height,
        key_frame=True,
        h264=nv12,
    )


def encode_bgra_packet(
    *,
    frame_id: int,
    timestamp_us: int,
    width: int,
    height: int,
    bgra: bytes,
) -> bytes:
    if width <= 0 or height <= 0:
        raise ValueError("BGRA dimensions must be positive integers")
    if len(bgra) != width * height * 4:
        raise ValueError("BGRA payload size does not match its dimensions")
    # Raw hardware-preview frames share the native 25-byte metadata header.
    return encode_h264_packet(
        frame_id=frame_id,
        timestamp_us=timestamp_us,
        width=width,
        height=height,
        key_frame=True,
        h264=bgra,
    )


__all__ = [
    "encode_frame_packet",
    "encode_bgra_packet",
    "encode_h264_packet",
    "encode_i420_packet",
    "encode_nv12_packet",
]
