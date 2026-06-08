#!/usr/bin/env python3
"""Probe-PER logger (ground side): per-rung packet-error-rate from probe streams.

Reads the `-Y` rx_ant JSON that a probe `wfb_rx_native` emits for each MCS rung
(one UDP stats port per rung) and turns it into windowed PER records that the
gemma4 closed loop can ingest alongside `wfb_review_queue` output. This is the
measured-headroom signal PROBE_PER_SPEC.md calls for.

Bench-validated facts this relies on (2026-06-07, see memory wfb-manual-probe-setup):
  * the probe MUST mirror the video PHY (`-B 20 -S 1 -L 1`) and vary only MCS,
    or LDPC-less high rungs read a falsely pessimistic cliff;
  * with FEC 1/1 AND a paced feed (<=20 pps/rung) wfb's own counters are exact:
    `uniq + lost == sent`, so PER = lost / (uniq + lost) is trustworthy. We log
    `accounted` so a consumer can spot a feed overrun (accounted < sent => some
    packets were dropped at the TX UDP input, not on air -> PER understated).

Each emitted record (one per rung per window):
  {"type":"probe","ts_ms":..,"radio_port":P,"mcs":M,"per":0.0..1.0,
   "recv":uniq,"lost":lost,"accounted":uniq+lost,"rssi":dBm|null,"snr":dB|null,
   "window_s":W}

    # one rung:
    python3 probe_log.py --rung 51:3:5851
    # boundary probe (cur=2 on 50, cur+1=3 on 51), 1s windows:
    python3 probe_log.py --rung 50:2:5850 --rung 51:3:5851 --window-s 1.0 --out probe.jsonl
"""
from __future__ import annotations

import argparse
import json
import select
import socket
import sys
import time


def best_ant(rec: dict):
    """Best-RSSI antenna's (rssi, snr) -- RSSI-primary like the controller."""
    ants = [a for a in rec.get("ant", [])
            if isinstance(a.get("rssi"), dict) and a["rssi"].get("avg") is not None]
    if not ants:
        return None, None
    b = max(ants, key=lambda a: a["rssi"]["avg"])
    snr = b["snr"]["avg"] if isinstance(b.get("snr"), dict) and b["snr"].get("avg") is not None else None
    return float(b["rssi"]["avg"]), (float(snr) if snr is not None else None)


def _rec_mcs(rec: dict):
    """The MCS reported in this -Y record's ant[] block (None if absent)."""
    m = None
    for a in rec.get("ant", []):
        if "mcs" in a:
            m = a["mcs"]
    return m


class Rung:
    """Accumulator for one MCS rung's probe stats over a sliding window.

    Default: one record per window labelled with the last seen mcs. With
    ``by_mcs`` (swept-TX mode), counters are bucketed by the *received* mcs of
    each -Y record, so a window that straddles a ``set_radio`` step emits one
    clean record per mcs instead of mislabelling the whole window.
    """

    def __init__(self, radio_port: int, mcs: int, stats_port: int, by_mcs: bool = False):
        self.radio_port = radio_port
        self.mcs = mcs                  # nominal probed MCS (from the launcher)
        self.stats_port = stats_port
        self.by_mcs = by_mcs
        self.win_start_ms = None
        self.reset()

    def reset(self):
        self.uniq = self.lost = 0
        self.rssi = self.snr = None
        self.seen_mcs = None
        self.buckets = {}               # mcs -> {uniq,lost,rssi,snr} (by_mcs mode)

    def add(self, rec: dict):
        pk = rec.get("pkt", {})
        u = int(pk.get("uniq", 0) or 0)
        l = int(pk.get("lost", 0) or 0)
        r, s = best_ant(rec)
        m = _rec_mcs(rec)
        if self.by_mcs:
            key = m if m is not None else self.mcs
            b = self.buckets.get(key)
            if b is None:
                b = self.buckets[key] = {"uniq": 0, "lost": 0, "rssi": None, "snr": None}
            b["uniq"] += u
            b["lost"] += l
            if r is not None:
                b["rssi"], b["snr"] = r, s
        else:
            self.uniq += u
            self.lost += l
            if r is not None:
                self.rssi, self.snr = r, s          # last good reading in the window
            if m is not None:
                self.seen_mcs = m

    @staticmethod
    def _record(ts_ms, radio_port, mcs, uniq, lost, rssi, snr, window_s) -> dict:
        acc = uniq + lost
        per = (lost / acc) if acc else None         # None = no traffic this window
        return {
            "type": "probe", "ts_ms": ts_ms,
            "radio_port": radio_port,
            "mcs": mcs,
            "per": round(per, 4) if per is not None else None,
            "recv": uniq, "lost": lost, "accounted": acc,
            "rssi": rssi, "snr": snr, "window_s": window_s,
        }

    def emit_all(self, ts_ms: int, window_s: float) -> list:
        """One or more probe records for the window just elapsed."""
        if not self.by_mcs:
            mcs = self.seen_mcs if self.seen_mcs is not None else self.mcs
            return [self._record(ts_ms, self.radio_port, mcs,
                                 self.uniq, self.lost, self.rssi, self.snr, window_s)]
        return [self._record(ts_ms, self.radio_port, mcs,
                             b["uniq"], b["lost"], b["rssi"], b["snr"], window_s)
                for mcs, b in sorted(self.buckets.items())]


def parse_rung(spec: str, by_mcs: bool = False) -> Rung:
    # "radio_port:mcs:stats_port"
    rp, mcs, sp = spec.split(":")
    return Rung(int(rp), int(mcs), int(sp), by_mcs=by_mcs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", action="append", required=True, metavar="RP:MCS:STATSPORT",
                    help="one per probed rung (repeatable)")
    ap.add_argument("--window-s", type=float, default=1.0,
                    help="PER averaging window (s); 1.0 ~= controller sample cadence")
    ap.add_argument("--by-mcs", action="store_true",
                    help="bucket each rung by the RECEIVED mcs (for a single swept TX): "
                         "emit one clean record per mcs per window")
    ap.add_argument("--out", help="append JSONL here (default stdout)")
    ap.add_argument("--bind", default="127.0.0.1")
    args = ap.parse_args()

    rungs = [parse_rung(s, by_mcs=args.by_mcs) for s in args.rung]
    by_fd = {}
    for r in rungs:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((args.bind, r.stats_port))
        s.setblocking(False)
        by_fd[s.fileno()] = (s, r)

    out = open(args.out, "a") if args.out else sys.stdout
    poller = select.poll()
    for fd in by_fd:
        poller.register(fd, select.POLLIN)
    win_ms = int(args.window_s * 1000)
    print(f"# probe_log: {len(rungs)} rung(s), {args.window_s}s windows"
          f"{' (by-mcs)' if args.by_mcs else ''} -> {args.out or 'stdout'}",
          file=sys.stderr, flush=True)

    tally = {}      # mcs -> [uniq, lost] cumulative, for the exit summary

    def write_records(recs):
        for rec in recs:
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            t = tally.setdefault(rec["mcs"], [0, 0])
            t[0] += rec["recv"]
            t[1] += rec["lost"]
        out.flush()

    try:
        while True:
            for fd, _ev in poller.poll(200):
                s, rung = by_fd[fd]
                try:
                    data, _ = s.recvfrom(65535)
                except BlockingIOError:
                    continue
                for line in data.decode(errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = int(rec.get("ts_ms", 0) or 0)
                    if rung.win_start_ms is None:
                        rung.win_start_ms = ts
                    rung.add(rec)
                    if ts - rung.win_start_ms >= win_ms:
                        write_records(rung.emit_all(ts, args.window_s))
                        rung.reset()
                        rung.win_start_ms = ts
    except KeyboardInterrupt:
        pass
    finally:
        if tally:
            print("# probe_log summary (cumulative PER by mcs):", file=sys.stderr)
            for mcs in sorted(tally):
                u, l = tally[mcs]
                acc = u + l
                per = (l / acc * 100) if acc else 0.0
                print(f"#   MCS{mcs}: {per:5.2f}% PER  ({l} lost / {acc} acct)",
                      file=sys.stderr)
            sys.stderr.flush()
        if args.out:
            out.close()


if __name__ == "__main__":
    main()
