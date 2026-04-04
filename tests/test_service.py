"""Tests for FECControllerService and FPSEstimator."""

import asyncio
import socket
import pytest

from fec_controller.config import ControllerConfig
from fec_controller.protocol import pack_stat, pack_frame, FRAME_TYPE_I
from fec_controller.service import FECControllerService, FPSEstimator


class TestFPSEstimator:

    def test_default_fps_before_data(self):
        est = FPSEstimator(default_fps=60.0)
        assert est.fps == 60.0

    def test_single_sample_returns_default(self):
        est = FPSEstimator(default_fps=60.0)
        fps = est.update(1_000_000)
        assert fps == 60.0

    def test_two_samples_gives_fps(self):
        est = FPSEstimator(alpha=1.0)  # instant tracking
        est.update(0)
        fps = est.update(16_667)  # ~60 fps interval in us
        assert fps == pytest.approx(60.0, rel=0.01)

    def test_120fps_estimation(self):
        est = FPSEstimator(alpha=1.0)
        est.update(0)
        fps = est.update(8_333)  # ~120 fps interval in us
        assert fps == pytest.approx(120.0, rel=0.01)

    def test_ewma_smoothing(self):
        est = FPSEstimator(alpha=0.05)
        # Feed many 60fps frames, then check convergence
        for i in range(200):
            est.update(i * 16_667)
        assert est.fps == pytest.approx(60.0, rel=0.02)

    def test_ignores_zero_interval(self):
        est = FPSEstimator(default_fps=30.0)
        est.update(1000)
        fps = est.update(1000)  # same timestamp = zero interval
        assert fps == 30.0  # stays at default


class TestServiceSidecarMode:

    def test_handle_sidecar_frame(self):
        config = ControllerConfig()
        service = FECControllerService(
            config, dry_run=True, sidecar_mode=True
        )

        # Simulate several frames with encoder trailer
        for i in range(10):
            data = pack_frame(
                frame_ready_us=i * 16_667,  # ~60fps
                frame_size_bytes=5000,
                seq_count=4,
            )
            service._handle_sidecar_frame(data)

        assert service._frame_count == 10
        assert service.controller.get_current() is not None

    def test_sidecar_frame_uses_frame_size_bytes(self):
        config = ControllerConfig()
        service = FECControllerService(
            config, dry_run=True, sidecar_mode=True
        )

        # Send frames with specific frame_size_bytes in encoder trailer
        for i in range(5):
            data = pack_frame(
                frame_ready_us=i * 8_333,  # ~120fps
                frame_size_bytes=20000,
                seq_count=14,
            )
            service._handle_sidecar_frame(data)

        p = service.controller.get_current()
        assert p is not None
        # With 20000 byte frames at headroom 1.05, k should be ~15
        assert p.k >= 10

    def test_handle_packet_routes_sidecar(self):
        config = ControllerConfig()
        service = FECControllerService(
            config, dry_run=True, sidecar_mode=True
        )

        data = pack_frame(
            frame_ready_us=100_000,
            frame_size_bytes=5000,
        )
        service._handle_packet(data)
        assert service._frame_count == 1

    def test_handle_packet_ignores_non_frame(self):
        """Non-FRAME sidecar messages should be ignored."""
        from fec_controller.protocol import pack_subscribe
        config = ControllerConfig()
        service = FECControllerService(
            config, dry_run=True, sidecar_mode=True
        )

        service._handle_packet(pack_subscribe())
        assert service._frame_count == 0


class TestServiceLegacyMode:

    def test_handle_stat_updates_controller(self):
        config = ControllerConfig()
        service = FECControllerService(
            config, dry_run=True, sidecar_mode=False
        )

        data = pack_stat(frame_size=5000, fps=120.0, frame_type=0)
        service._handle_stat(data)

        assert service._frame_count == 1
        assert service.controller.get_current() is not None

    def test_handle_stat_ignores_short_packets(self):
        config = ControllerConfig()
        service = FECControllerService(
            config, dry_run=True, sidecar_mode=False
        )

        service._handle_stat(b"\x00\x00")
        assert service._frame_count == 0

    def test_handle_packet_routes_legacy(self):
        config = ControllerConfig()
        service = FECControllerService(
            config, dry_run=True, sidecar_mode=False
        )

        data = pack_stat(frame_size=5000, fps=120.0)
        service._handle_packet(data)
        assert service._frame_count == 1

    def test_dry_run_does_not_send(self):
        config = ControllerConfig()
        service = FECControllerService(
            config, dry_run=True, sidecar_mode=False
        )

        data = pack_stat(frame_size=5000, fps=120.0)
        service._handle_stat(data)
        assert service.wfb_tx._sock is None


class TestServiceIntegration:

    @pytest.mark.asyncio
    async def test_udp_listener_receives_sidecar_frames(self):
        """Full integration: send sidecar FRAME packets to the service via UDP."""
        config = ControllerConfig()
        service = FECControllerService(
            config, stat_port=0, dry_run=True, sidecar_mode=True
        )

        loop = asyncio.get_event_loop()

        from fec_controller.service import _SidecarProtocol
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _SidecarProtocol(service._handle_packet),
            local_addr=("127.0.0.1", 0),
        )
        actual_port = transport.get_extra_info("sockname")[1]

        try:
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            for i in range(5):
                data = pack_frame(
                    frame_ready_us=i * 16_667,
                    frame_size_bytes=8000,
                    frame_type=0,
                )
                sender.sendto(data, ("127.0.0.1", actual_port))
            sender.close()

            await asyncio.sleep(0.05)

            assert service._frame_count == 5
            p = service.controller.get_current()
            assert p is not None
            assert p.k > 0
        finally:
            transport.close()
