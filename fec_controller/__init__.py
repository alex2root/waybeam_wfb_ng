"""
waybeam-hub adaptive FEC controller for wfb-ng.

Dynamically adjusts wfb-ng FEC parameters (k/n) based on real-time
video frame statistics from the encoding pipeline.

Architecture:
  video encoder -> sidecar (per-frame UDP stats) -> fec_controller -> wfb_tx (UDP control port)
"""

from fec_controller.protocol import (
    STAT_FMT,
    STAT_SIZE,
    FRAME_BASE_SIZE,
    FRAME_EXT_SIZE,
    SidecarFrame,
    pack_stat,
    unpack_stat,
    pack_subscribe,
    pack_frame,
    pack_frame_base,
    parse_frame,
    parse_header,
)
from fec_controller.config import ControllerConfig
from fec_controller.headroom import HeadroomTracker
from fec_controller.controller import FECParams, FECController
from fec_controller.wfb_control import WfbTxControl
from fec_controller.service import FECControllerService, FPSEstimator

__all__ = [
    "STAT_FMT",
    "STAT_SIZE",
    "FRAME_BASE_SIZE",
    "FRAME_EXT_SIZE",
    "SidecarFrame",
    "pack_stat",
    "unpack_stat",
    "pack_subscribe",
    "pack_frame",
    "pack_frame_base",
    "parse_frame",
    "parse_header",
    "ControllerConfig",
    "HeadroomTracker",
    "FECParams",
    "FECController",
    "WfbTxControl",
    "FECControllerService",
    "FPSEstimator",
]
