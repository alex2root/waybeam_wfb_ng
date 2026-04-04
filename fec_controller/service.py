"""
Async FEC controller service.

Supports two input modes:
  1. Sidecar mode (default): subscribes to the venc sidecar, receives FRAME
     messages, derives fps from frame_ready_us intervals, uses frame_size_bytes
     from the encoder trailer when available.
  2. Legacy stat mode: receives 8-byte stat packets with explicit fps field.

Designed to run as a waybeam-hub module.
"""

import time
import asyncio
import logging

from fec_controller.config import ControllerConfig
from fec_controller.controller import FECController, FECParams
from fec_controller.protocol import (
    STAT_SIZE,
    FRAME_BASE_SIZE,
    SIDECAR_MAGIC,
    SIDECAR_VERSION,
    MSG_FRAME,
    HDR_SIZE,
    SidecarFrame,
    pack_subscribe,
    parse_frame,
    parse_header,
    unpack_stat,
)
from fec_controller.wfb_control import WfbTxControl

log = logging.getLogger("fec_ctrl")

# Resubscribe interval (sidecar TTL is 5s, refresh well before)
_SUBSCRIBE_INTERVAL_S = 2.0


class _SidecarProtocol(asyncio.DatagramProtocol):
    """UDP protocol for sidecar FRAME messages."""

    def __init__(self, handler):
        self._handler = handler
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        self._handler(data)


class FPSEstimator:
    """Estimate fps from frame_ready_us intervals using EWMA."""

    def __init__(self, alpha: float = 0.05, default_fps: float = 60.0):
        self._alpha = alpha
        self._default_fps = default_fps
        self._last_frame_ready_us: int | None = None
        self._avg_interval_us: float | None = None

    def update(self, frame_ready_us: int) -> float:
        """Feed a frame_ready_us timestamp, return estimated fps."""
        if self._last_frame_ready_us is not None and frame_ready_us > self._last_frame_ready_us:
            interval_us = frame_ready_us - self._last_frame_ready_us
            if interval_us > 0:
                if self._avg_interval_us is None:
                    self._avg_interval_us = float(interval_us)
                else:
                    self._avg_interval_us = (
                        self._alpha * interval_us
                        + (1.0 - self._alpha) * self._avg_interval_us
                    )
        self._last_frame_ready_us = frame_ready_us

        if self._avg_interval_us and self._avg_interval_us > 0:
            return 1_000_000.0 / self._avg_interval_us
        return self._default_fps

    @property
    def fps(self) -> float:
        if self._avg_interval_us and self._avg_interval_us > 0:
            return 1_000_000.0 / self._avg_interval_us
        return self._default_fps


class FECControllerService:
    """Async service: receives sidecar stats, drives FEC updates."""

    def __init__(
        self,
        config: ControllerConfig,
        stat_port: int = 5610,
        wfb_control_host: str = "127.0.0.1",
        wfb_control_port: int = 0,
        dry_run: bool = False,
        sidecar_mode: bool = True,
        sidecar_host: str = "",
        sidecar_port: int = 0,
    ):
        self.config = config
        self.stat_port = stat_port
        self.dry_run = dry_run
        self.sidecar_mode = sidecar_mode
        self.sidecar_host = sidecar_host
        self.sidecar_port = sidecar_port

        self.controller = FECController(config)
        self.wfb_tx = WfbTxControl(wfb_control_host, wfb_control_port)
        self.fps_estimator = FPSEstimator(alpha=config.ewma_alpha)

        self._frame_count = 0
        self._last_log_time = 0.0
        self._sidecar_transport = None

    def _apply_update(self, params: FECParams) -> None:
        if self.dry_run:
            log.info(
                "[DRY RUN] set_fec k=%d n=%d (redun=%.0f%% timeout=%dms "
                "hroom=%.2f avg=%.0fB pkts=%d)",
                params.k,
                params.n,
                params.redundancy * 100,
                params.fec_timeout_ms,
                params.headroom,
                params.avg_frame_size,
                params.packets_per_frame,
            )
        else:
            self.wfb_tx.send_fec(params.k, params.n)

    def _handle_sidecar_frame(self, data: bytes) -> None:
        """Process a sidecar FRAME message."""
        frame = parse_frame(data)
        if frame is None:
            return

        self._frame_count += 1

        # Derive fps from frame_ready_us timing intervals
        fps = self.fps_estimator.update(frame.frame_ready_us)

        # Use encoder trailer frame_size if available, otherwise estimate
        # from seq_count * MTU (rough, but usable for base-only frames)
        if frame.has_enc_info and frame.frame_size_bytes > 0:
            frame_size = frame.frame_size_bytes
        else:
            frame_size = frame.seq_count * self.config.mtu

        result = self.controller.update(frame_size, fps)
        if result is not None:
            self._apply_update(result)

        self._periodic_log()

    def _handle_stat(self, data: bytes) -> None:
        """Process a legacy 8-byte stat packet."""
        if len(data) < STAT_SIZE:
            return

        stat = unpack_stat(data)
        self._frame_count += 1

        result = self.controller.update(stat["frame_size"], stat["fps"])
        if result is not None:
            self._apply_update(result)

        self._periodic_log()

    def _handle_packet(self, data: bytes) -> None:
        """Route incoming packet to the right handler based on content."""
        if self.sidecar_mode:
            # In sidecar mode, only process FRAME messages
            hdr = parse_header(data)
            if hdr is not None and hdr[2] == MSG_FRAME:
                self._handle_sidecar_frame(data)
        else:
            self._handle_stat(data)

    def _periodic_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log_time >= 2.0:
            p = self.controller.get_current()
            if p:
                hr = self.controller.headroom_tracker.headroom
                fps = self.fps_estimator.fps if self.sidecar_mode else (self.controller.current_fps or 0)
                log.info(
                    "status: frames=%d avg=%.0fB hroom=%.2f k=%d n=%d "
                    "redun=%.0f%% fps=%.1f updates=%d",
                    self._frame_count,
                    self.controller.avg_frame_size,
                    hr,
                    p.k,
                    p.n,
                    p.redundancy * 100,
                    fps,
                    self.controller.update_count,
                )
            self._last_log_time = now

    async def _subscribe_loop(self) -> None:
        """Periodically send SUBSCRIBE to the venc sidecar."""
        if not self.sidecar_host or not self.sidecar_port:
            log.warning("Sidecar mode enabled but no sidecar host/port configured")
            return

        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sub_msg = pack_subscribe()
        addr = (self.sidecar_host, self.sidecar_port)

        log.info("Subscribing to sidecar at %s:%d every %.0fs",
                 self.sidecar_host, self.sidecar_port, _SUBSCRIBE_INTERVAL_S)

        try:
            while True:
                try:
                    sock.sendto(sub_msg, addr)
                except OSError as e:
                    log.warning("Failed to send SUBSCRIBE: %s", e)
                await asyncio.sleep(_SUBSCRIBE_INTERVAL_S)
        finally:
            sock.close()

    async def run(self) -> None:
        """Main async loop - listen for sidecar or stat packets."""
        mode_str = "sidecar" if self.sidecar_mode else "legacy stat"
        log.info("FEC controller starting (%s mode) on UDP :%d", mode_str, self.stat_port)
        log.info("wfb_tx control: %s:%d", self.wfb_tx.host, self.wfb_tx.port)
        log.info("MTU=%d dry_run=%s", self.config.mtu, self.dry_run)

        if not self.dry_run:
            self.wfb_tx.connect()

        loop = asyncio.get_event_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _SidecarProtocol(self._handle_packet),
            local_addr=("0.0.0.0", self.stat_port),
        )
        self._sidecar_transport = transport

        log.info("Listening for %s packets...", mode_str)

        try:
            if self.sidecar_mode and self.sidecar_host and self.sidecar_port:
                # Run subscribe loop concurrently
                sub_task = asyncio.create_task(self._subscribe_loop())
                try:
                    await asyncio.Event().wait()
                finally:
                    sub_task.cancel()
            else:
                await asyncio.Event().wait()
        finally:
            transport.close()
            self.wfb_tx.close()
