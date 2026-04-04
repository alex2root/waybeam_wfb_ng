"""Tests for sidecar stat packet protocol."""

import struct
import pytest

from fec_controller.protocol import STAT_FMT, STAT_SIZE, pack_stat, unpack_stat


class TestStatPacket:

    def test_size_is_8_bytes(self):
        assert STAT_SIZE == 8

    def test_pack_unpack_roundtrip(self):
        data = pack_stat(frame_size=5000, fps=120.0, frame_type=1)
        assert len(data) == STAT_SIZE
        stat = unpack_stat(data)
        assert stat["frame_size"] == 5000
        assert stat["fps"] == 120.0
        assert stat["frame_type"] == 1

    def test_pack_unpack_p_frame(self):
        data = pack_stat(frame_size=3000, fps=60.0, frame_type=0)
        stat = unpack_stat(data)
        assert stat["frame_size"] == 3000
        assert stat["fps"] == 60.0
        assert stat["frame_type"] == 0

    def test_fps_encoding_precision(self):
        """fps_x10 encoding: 120.5 fps -> 1205 -> 120.5"""
        data = pack_stat(frame_size=1000, fps=120.5)
        stat = unpack_stat(data)
        assert stat["fps"] == pytest.approx(120.5)

    def test_max_frame_size(self):
        data = pack_stat(frame_size=0xFFFFFFFF, fps=30.0)
        stat = unpack_stat(data)
        assert stat["frame_size"] == 0xFFFFFFFF

    def test_little_endian_format(self):
        """Verify little-endian byte order."""
        data = pack_stat(frame_size=1, fps=0.1, frame_type=0)
        # u32 LE: 01 00 00 00, u16 LE: 01 00, u8: 00, pad: 00
        assert data[0] == 1
        assert data[1] == 0

    def test_unpack_ignores_extra_bytes(self):
        data = pack_stat(frame_size=5000, fps=60.0) + b"\xff\xff\xff\xff"
        stat = unpack_stat(data)
        assert stat["frame_size"] == 5000
