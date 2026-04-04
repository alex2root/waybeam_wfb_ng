"""Tests for FECControllerService."""

import asyncio
import socket
import pytest

from fec_controller.config import ControllerConfig
from fec_controller.protocol import pack_stat
from fec_controller.service import FECControllerService


class TestServiceStatHandling:

    def test_handle_stat_updates_controller(self):
        config = ControllerConfig()
        service = FECControllerService(config, dry_run=True)

        data = pack_stat(frame_size=5000, fps=120.0, frame_type=0)
        service._handle_stat(data)

        assert service._frame_count == 1
        assert service.controller.get_current() is not None

    def test_handle_stat_ignores_short_packets(self):
        config = ControllerConfig()
        service = FECControllerService(config, dry_run=True)

        service._handle_stat(b"\x00\x00")  # Too short
        assert service._frame_count == 0

    def test_handle_stat_counts_frames(self):
        config = ControllerConfig()
        service = FECControllerService(config, dry_run=True)

        for i in range(10):
            data = pack_stat(frame_size=5000, fps=120.0)
            service._handle_stat(data)

        assert service._frame_count == 10

    def test_dry_run_does_not_send(self):
        config = ControllerConfig()
        service = FECControllerService(config, dry_run=True)

        # wfb_tx socket should not be connected in dry_run
        data = pack_stat(frame_size=5000, fps=120.0)
        service._handle_stat(data)
        assert service.wfb_tx._sock is None


class TestServiceIntegration:

    @pytest.mark.asyncio
    async def test_udp_listener_receives_packets(self):
        """Full integration: send UDP stat packets to the service."""
        config = ControllerConfig()
        service = FECControllerService(
            config, stat_port=0, dry_run=True  # OS picks port
        )

        loop = asyncio.get_event_loop()

        # Start listener on ephemeral port
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _TestStatProtocol(service._handle_stat),
            local_addr=("127.0.0.1", 0),
        )
        actual_port = transport.get_extra_info("sockname")[1]

        try:
            # Send stat packets
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            for _ in range(5):
                data = pack_stat(frame_size=8000, fps=60.0, frame_type=0)
                sender.sendto(data, ("127.0.0.1", actual_port))
            sender.close()

            # Give asyncio a moment to process
            await asyncio.sleep(0.05)

            assert service._frame_count == 5
            p = service.controller.get_current()
            assert p is not None
            assert p.k > 0
        finally:
            transport.close()


class _TestStatProtocol(asyncio.DatagramProtocol):
    def __init__(self, handler):
        self._handler = handler

    def datagram_received(self, data, addr):
        self._handler(data)
