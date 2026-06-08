#!/usr/bin/env python3
"""UDP tee/splitter for wfb_rx -Y JSON stats.

Sits between the wfb stats producer and its real consumer so live link
telemetry can be captured / analysed locally WITHOUT disturbing the existing
upstream path (the back-channel that ships stats to the air unit).

Wiring with the waybeam ground supervisor (gs_supervisor):

  Today the supervisor forwards wfb_rx -Y stats to a single `stats_out`
  (config: 127.0.0.1:6600 = the uplink wfb_tx udp_in). Point that at this tee
  instead, and let the tee fan the stream back out:

      config stats_out:  127.0.0.1:6650   (this tee's --listen)

      wfb_rx -Y -> supervisor stats_drain -> :6650 (tee)
                                                |
                +-------------------------------+-------------------------+
                v                               v                         v
        --forward 127.0.0.1:6600        --tap 127.0.0.1:6700      --jsonl real_wfb.jsonl
        (wfb_tx udp_in; upstream,        (live Tier-1 pipeline,    (capture for training/
         byte-for-byte unchanged)         wfb_pipeline.py)          schema validation)

  Forwarding happens FIRST, before any local work, so the added latency on the
  real back-channel is a couple of syscalls (microseconds).

Examples:
  # capture a real sample to JSONL while keeping the upstream path intact
  python3 telemetry/wfb_stats_tee.py --listen 6650 \
      --forward 127.0.0.1:6600 --jsonl telemetry/real_wfb.jsonl

  # also fan a live copy to the analysis pipeline
  python3 telemetry/wfb_stats_tee.py --listen 6650 \
      --forward 127.0.0.1:6600 --tap 127.0.0.1:6700 --jsonl telemetry/real_wfb.jsonl

Stdlib only (matches the telemetry tooling guardrail).
"""
from __future__ import annotations

import argparse
import socket
import sys
import time


def parse_hostport(s: str) -> tuple[str, int]:
    host, _, port = s.rpartition(":")
    if not port:
        raise ValueError(f"expected HOST:PORT, got {s!r}")
    return (host or "127.0.0.1", int(port))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", type=int, default=6650, help="UDP port to receive stats on")
    ap.add_argument("--bind", default="127.0.0.1", help="address to bind --listen")
    ap.add_argument("--forward", default="",
                    help="HOST:PORT real downstream to relay verbatim (e.g. wfb_tx udp_in 127.0.0.1:6600)")
    ap.add_argument("--tap", action="append", default=[], metavar="HOST:PORT",
                    help="extra HOST:PORT to receive a copy (repeatable; e.g. the analysis pipeline)")
    ap.add_argument("--jsonl", default="", help="append each datagram as one line to this file")
    ap.add_argument("--stats-every", type=float, default=5.0,
                    help="seconds between throughput log lines (0 = silent)")
    args = ap.parse_args()

    fwd = parse_hostport(args.forward) if args.forward else None
    taps = [parse_hostport(t) for t in args.tap]

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((args.bind, args.listen))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # line-buffered append; one datagram == one JSON object == one line.
    jf = open(args.jsonl, "ab", buffering=0) if args.jsonl else None

    dests = (["forward " + str(fwd)] if fwd else []) + ["tap " + str(t) for t in taps]
    print(f"[tee] listening udp {args.bind}:{args.listen} -> "
          f"{', '.join(dests) or '(no relay)'}"
          f"{'  jsonl=' + args.jsonl if jf else ''}", file=sys.stderr, flush=True)

    n = 0
    nbytes = 0
    last = time.monotonic()
    try:
        while True:
            data, _ = rx.recvfrom(65535)
            # relay FIRST to minimise added latency on the real back-channel
            if fwd:
                try:
                    tx.sendto(data, fwd)
                except OSError:
                    pass
            for t in taps:
                try:
                    tx.sendto(data, t)
                except OSError:
                    pass
            if jf:
                jf.write(data if data.endswith(b"\n") else data + b"\n")
            n += 1
            nbytes += len(data)
            if args.stats_every > 0:
                now = time.monotonic()
                if now - last >= args.stats_every:
                    rate = n / (now - last)
                    print(f"[tee] {n} datagrams ({nbytes} B) in last "
                          f"{now - last:.1f}s -> {rate:.1f}/s", file=sys.stderr, flush=True)
                    n = 0
                    nbytes = 0
                    last = now
    except KeyboardInterrupt:
        print("\n[tee] stopped", file=sys.stderr)
    finally:
        if jf:
            jf.close()


if __name__ == "__main__":
    main()
