from __future__ import annotations


def encode_frame_packet(frame_id: int, jpeg: bytes) -> bytes:
    """Encode an unsigned 64-bit big-endian frame id followed by JPEG bytes."""
    if not 0 <= frame_id < 2**64:
        raise ValueError("frame_id must fit in an unsigned 64-bit integer")
    return frame_id.to_bytes(8, "big", signed=False) + jpeg


__all__ = ["encode_frame_packet"]
