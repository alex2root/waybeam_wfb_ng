"""
Async FEC controller service.

Receives sidecar stat packets via UDP and drives FEC updates to wfb_tx.
Designed to run as a waybeam-hub module.
"""

import time
import asyncio
import logging

from fec_controller.config import ControllerConfig
from fec_controller.controller import FECController, FECParams
from fec_controller.protocol import STAT_SIZE, unpack_stat
from fec_controller.wfb_control import WfbTxControl

log = logging.getLogger("fec_ctrl")


class _StatProtocol(asyncio.DatagramProtocol):
    def __init__(self, handler):
        self._handler = handler

    def datagram_received(self, data, addr):
        self._handler(data)


class FECControllerService:
    """Async service: receives sidecar stats, drives FEC updates."""

    def __init__(
        self,
        config: ControllerConfig,
        stat_port: int = 5610,
        wfb_control_host: str = "127.0.0.1",
        wfb_control_port: int = 0,
        dry_run: bool = False,
    ):
        self.config = config
        self.stat_port = stat_port
        self.dry_run = dry_run

        self.controller = FECController(config)
        self.wfb_tx = WfbTxControl(wfb_control_host, wfb_control_port)

        self._frame_count = 0
        self._last_log_time = 0.0

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

    def _handle_stat(self, data: bytes) -> None:
        if len(data) < STAT_SIZE:
            return

        stat = unpack_stat(data)
        self._frame_count += 1

        result = self.controller.update(stat["frame_size"], stat["fps"])
        if result is not None:
            self._apply_update(result)

        now = time.monotonic()
        if now - self._last_log_time >= 2.0:
            p = self.controller.get_current()
            if p:
                hr = self.controller.headroom_tracker.headroom
                log.info(
                    "status: frames=%d avg=%.0fB hroom=%.2f k=%d n=%d "
                    "redun=%.0f%% updates=%d",
                    self._frame_count,
                    self.controller.avg_frame_size,
                    hr,
                    p.k,
                    p.n,
                    p.redundancy * 100,
                    self.controller.update_count,
                )
            self._last_log_time = now

    async def run(self) -> None:
        """Main async loop - listen for stat packets."""
        log.info("FEC controller starting on UDP :%d", self.stat_port)
        log.info("wfb_tx control: %s:%d", self.wfb_tx.host, self.wfb_tx.port)
        log.info("MTU=%d dry_run=%s", self.config.mtu, self.dry_run)

        if not self.dry_run:
            self.wfb_tx.connect()

        loop = asyncio.get_event_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _StatProtocol(self._handle_stat),
            local_addr=("0.0.0.0", self.stat_port),
        )

        log.info("Listening for sidecar stat packets...")

        try:
            await asyncio.Event().wait()
        finally:
            transport.close()
            self.wfb_tx.close()
