"""
Sidecar stat packet format.

Per-frame UDP packet from the video encoding pipeline, 8 bytes little-endian:
  u32  frame_size     (bytes)
  u16  fps_x10        (fps * 10, e.g. 1200 = 120.0 fps)
  u8   frame_type     (0=P, 1=I)
  u8   reserved
"""

import struct

STAT_FMT = "<IHBx"
STAT_SIZE = struct.calcsize(STAT_FMT)  # 8 bytes


def unpack_stat(data: bytes) -> dict:
    """Unpack a sidecar stat packet into a dict."""
    frame_size, fps_x10, frame_type = struct.unpack(STAT_FMT, data[:STAT_SIZE])
    return {
        "frame_size": frame_size,
        "fps": fps_x10 / 10.0,
        "frame_type": frame_type,
    }


def pack_stat(frame_size: int, fps: float, frame_type: int = 0) -> bytes:
    """Pack a sidecar stat packet."""
    return struct.pack(STAT_FMT, frame_size, int(fps * 10), frame_type)
