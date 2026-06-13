#!/usr/bin/env python3
"""probe_bridge.py -- forward probe JSONL records up the uplink.

Reads probe_log.py's stdout (one JSON record per line) and sends each
{"type":"probe"} record as a single UDP datagram to the GS uplink wfb_tx
udp_in (default 127.0.0.1:6600). The records ride the SAME tunnel as the
video rx_ant stats to the vehicle's link_controller :5801, which demuxes
them by "type". Non-probe / comment lines are skipped.

Natural probe_log rate is ~1-3 rec/s, well under the uplink's capacity;
--max-pps is a backstop so a misconfigured feeder can't overrun the uplink
UDP input (which drops silently above capacity).

    probe_log.py --rung 50:3:5850 --by-mcs | probe_bridge.py --to 127.0.0.1:6600
"""
import argparse
import socket
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", default="127.0.0.1:6600",
                    help="host:port of the uplink wfb_tx udp_in (default 127.0.0.1:6600)")
    ap.add_argument("--max-pps", type=float, default=10.0,
                    help="rate cap as an overrun backstop (default 10)")
    a = ap.parse_args()

    host, _, port = a.to.partition(":")
    port = int(port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    min_gap = 1.0 / a.max_pps if a.max_pps > 0 else 0.0

    sent = 0
    last = 0.0
    for line in sys.stdin:
        line = line.strip()
        if not line or line[0] == "#":
            continue
        # tolerate both compact and spaced JSON
        if '"type":"probe"' not in line.replace(" ", ""):
            continue
        if min_gap:
            now = time.monotonic()
            wait = min_gap - (now - last)
            if wait > 0:
                time.sleep(wait)
        sock.sendto(line.encode(), (host, port))
        last = time.monotonic()
        sent += 1
        if sent % 10 == 0:
            sys.stderr.write(f"[bridge] {sent} probe records -> {a.to}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
