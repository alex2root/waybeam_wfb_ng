"""Simulation utilities for testing FEC controller without real hardware."""

import random

from fec_controller.config import ControllerConfig
from fec_controller.controller import FECController


def simulate_stream(
    fps: int = 120,
    base_frame_size: int = 5000,
    i_frame_multiplier: float = 1.15,
    gop_interval: int = 30,
    duration_s: float = 8.0,
    bitrate_events: list | None = None,
) -> None:
    if bitrate_events is None:
        bitrate_events = [(3.0, base_frame_size * 3), (6.0, base_frame_size)]

    sim_time = [0.0]
    controller = FECController(
        config=ControllerConfig(ewma_alpha=0.05),
        time_fn=lambda: sim_time[0],
    )

    frame_period = 1.0 / fps
    total_frames = int(duration_s * fps)

    print(
        f"\nSimulating {fps}fps, {duration_s}s, base={base_frame_size}B, "
        f"I-mult={i_frame_multiplier}x"
    )
    for t, sz in bitrate_events:
        print(f"  Event at {t}s: base -> {sz}B")

    hdr = (
        f"{'Time':>6s}  {'T':>1s}  {'FrmSize':>8s}  {'EWMA':>8s}  "
        f"{'Hroom':>5s}  {'k':>3s}  {'n':>3s}  {'n-k':>3s}  "
        f"{'Redun':>5s}  {'T/O':>3s}  {'Pkt':>3s}"
    )
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    current_base = base_frame_size
    event_idx = 0

    for i in range(total_frames):
        t = i * frame_period
        sim_time[0] = t

        while event_idx < len(bitrate_events) and t >= bitrate_events[event_idx][0]:
            current_base = bitrate_events[event_idx][1]
            event_idx += 1

        is_iframe = i % gop_interval == 0
        jitter = random.gauss(1.0, 0.03)
        frame_size = int(
            current_base * (i_frame_multiplier if is_iframe else 1.0) * jitter
        )
        ftype = "I" if is_iframe else "P"

        result = controller.update(frame_size, fps)

        near_event = any(abs(t - ev[0]) < frame_period * 2 for ev in bitrate_events)
        if result is not None or i % fps == 0 or near_event:
            p = result or controller.get_current()
            if p:
                marker = " <<<" if result else ""
                hr = controller.headroom_tracker.headroom
                print(
                    f"{t:6.2f}  {ftype:>1s}  {frame_size:8d}  "
                    f"{controller.avg_frame_size:8.0f}  "
                    f"{hr:5.2f}  "
                    f"{p.k:3d}  {p.n:3d}  {p.n - p.k:3d}  "
                    f"{p.redundancy:4.0%}  {p.fec_timeout_ms:3d}  "
                    f"{p.packets_per_frame:3d}{marker}"
                )


def print_reference_table() -> None:
    controller = FECController()

    print(f"\nReference table (MTU={controller.cfg.mtu}, headroom=1.15)")
    hdr = (
        f"{'FrameSize':>10s}  {'Pkts':>4s}  {'k':>3s}  {'n':>3s}  "
        f"{'n-k':>3s}  {'Redun':>5s}  {'T/O@60':>6s}  {'T/O@120':>7s}  "
        f"{'Effic':>5s}"
    )
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    for size in [
        500, 1000, 1446, 2000, 3000, 5000, 8000, 12000, 20000, 30000, 44000, 60000
    ]:
        p60 = controller.compute_params(float(size), 60, 1.15)
        p120 = controller.compute_params(float(size), 120, 1.15)
        print(
            f"{size:10d}  {p60.packets_per_frame:4d}  "
            f"{p60.k:3d}  {p60.n:3d}  {p60.n - p60.k:3d}  "
            f"{p60.redundancy:4.0%}  {p60.fec_timeout_ms:4d}  "
            f"{p120.fec_timeout_ms:5d}  {p60.k / p60.n:4.0%}"
        )
