#!/usr/bin/env python3
"""Controller-aligned link scorer: reproduce link_controller.c's decision logic.

The closed loop must flag the SAME events the real MCS/FEC controller acts on,
not an arbitrary per-record class. This module is a faithful, stdlib-only
re-implementation of `waybeam_wfb_ng/vehicle/link_controller.c`'s scorer +
selector (validated against the source, defaults from its set_defaults()):

  effective_rssi = EWMA(best-antenna avg RSSI, alpha=0.3) - loss_penalty
  loss_penalty   = 0.5 dB per 1% LOST (EWMA alpha=0.5), recovered = 0 dB, cap 20
  bucket FSM     = 3-bucket hysteresis (lo -70 / hi -50 / deadband 2) -> MCS 1/2/3
  reacts FAST DOWN (down_consecutive=1, cooldown 0.2s),
         SLOW UP  (up_consecutive=3, cooldown 3.0s, x3 backoff when oscillating)
  failsafe       = no rx_ant for 0.5s -> force bucket 0; recover after 3 good
                   samples (effective_rssi >= lo+deadband = -68)
  oscillation    = >=4 MCS changes in a 5s window

SNR is ignored entirely (the controller never reads it) -- matching our
"RSSI-primary" Tier-1. The per-record output (effective_rssi, bucket, the
loss penalty) is exactly the signal a controller-aligned Tier-1 should learn,
and the emitted DECISION POINTS (down / failsafe / oscillation / recovery) are
what the review queue should flag on.

    python3 telemetry/wfb_link_score.py --input telemetry/walk2_wfb.jsonl
    python3 telemetry/wfb_link_score.py --input ... --emit scored.jsonl   # per-record
    wfb_rx -Y ... | python3 telemetry/wfb_link_score.py --stdin
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass, field

from wfb_schema import best_chain, load_jsonl


# Defaults copied verbatim from link_controller.c set_defaults() (lines ~3956+).
@dataclass
class Config:
    rssi_thresh_low: float = -70.0
    rssi_thresh_high: float = -50.0
    rssi_deadband_db: float = 2.0
    rssi_ewma_alpha: float = 0.3
    loss_ewma_alpha: float = 0.5
    loss_lost_penalty_db_per_pct: float = 0.5
    loss_recovered_penalty_db_per_pct: float = 0.0
    loss_penalty_max_db: float = 20.0
    up_consecutive: int = 3
    down_consecutive: int = 1
    up_cooldown_s: float = 3.0
    down_cooldown_s: float = 0.2
    failsafe_timeout_s: float = 0.5
    failsafe_recovery_consecutive: int = 3
    oscillation_window_s: float = 5.0
    oscillation_threshold: int = 4
    oscillation_backoff: float = 3.0
    mcs_bucket: tuple = (1, 2, 3)   # bucket 0/1/2 -> MCS
    mcs_min: int = 0
    mcs_max: int = 11


class Scorer:
    """EWMA scorer -> effective_rssi (mirrors scorer_update)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.smoothed_rssi = 0.0
        self.smoothed_lost = 0.0
        self.smoothed_recov = 0.0
        self.have_rssi = False
        self.have_loss = False

    def update(self, record: dict):
        """Return a score dict, or None when no antenna reported (the no-data
        path that drives the failsafe watchdog)."""
        bc = best_chain(record)
        raw = bc["rssi"]            # AGG_BEST_AVG == max per-antenna avg == best_chain
        if raw is None:
            return None             # saw_any == false -> caller does tick_no_data

        cfg = self.cfg
        if not self.have_rssi:
            self.smoothed_rssi = raw
            self.have_rssi = True
        else:
            a = cfg.rssi_ewma_alpha
            self.smoothed_rssi = a * raw + (1.0 - a) * self.smoothed_rssi

        pkt = record.get("pkt", {})
        uniq = float(pkt.get("uniq", 0) or 0)
        lost = float(pkt.get("lost", 0) or 0)
        fec = float(pkt.get("fec_recovered", 0) or 0)
        denom = uniq if uniq > 0 else 1.0
        lost_r = min(1.0, lost / denom) if lost > 0 else 0.0
        recov_r = min(1.0, fec / denom) if fec > 0 else 0.0

        if not self.have_loss:
            self.smoothed_lost = lost_r
            self.smoothed_recov = recov_r
            self.have_loss = True
        else:
            a = cfg.loss_ewma_alpha
            self.smoothed_lost = a * lost_r + (1.0 - a) * self.smoothed_lost
            self.smoothed_recov = a * recov_r + (1.0 - a) * self.smoothed_recov

        penalty = (cfg.loss_lost_penalty_db_per_pct * 100.0 * self.smoothed_lost +
                   cfg.loss_recovered_penalty_db_per_pct * 100.0 * self.smoothed_recov)
        penalty = max(0.0, min(cfg.loss_penalty_max_db, penalty))
        return {
            "raw_rssi": raw,
            "smoothed_rssi": self.smoothed_rssi,
            "lost_ratio": lost_r,
            "smoothed_lost_ratio": self.smoothed_lost,
            "loss_penalty_db": penalty,
            "effective_rssi": self.smoothed_rssi - penalty,
        }


def bucket_from_rssi(rssi: float, current: int, cfg: Config) -> int:
    """3-bucket hysteresis FSM (mirrors bucket_from_rssi)."""
    lo, hi, db = cfg.rssi_thresh_low, cfg.rssi_thresh_high, cfg.rssi_deadband_db
    if current < 0:
        if rssi < lo:
            return 0
        if rssi < hi:
            return 1
        return 2
    if current == 0:
        if rssi >= lo + db:
            return 2 if rssi >= hi + db else 1
        return 0
    if current == 1:
        if rssi < lo - db:
            return 0
        if rssi >= hi + db:
            return 2
        return 1
    # current == 2
    if rssi < hi - db:
        return 0 if rssi < lo - db else 1
    return 2


def mcs_for_bucket(bucket: int, cfg: Config) -> int:
    mcs = cfg.mcs_bucket[bucket if bucket in (0, 1) else 2]
    return max(cfg.mcs_min, min(cfg.mcs_max, mcs))


class Selector:
    """MCS selector FSM with consecutive/cooldown/failsafe/oscillation
    (mirrors selector_update + selector_tick_no_data)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.current_bucket = -1
        self.current_mcs = -1
        self.last_change_us = 0
        self.last_datagram_us = 0
        self.pending_bucket = -1
        self.pending_streak = 0
        self.in_failsafe = False
        self.recovery_streak = 0
        self.changes = collections.deque()  # change timestamps (us), oscillation window

    def _expire_changes(self, now: int):
        w = int(self.cfg.oscillation_window_s * 1e6)
        if w == 0:
            self.changes.clear()
            return
        if now < w:
            return
        cutoff = now - w
        while self.changes and self.changes[0] < cutoff:
            self.changes.popleft()

    def _is_oscillating(self) -> bool:
        return len(self.changes) >= self.cfg.oscillation_threshold

    def _commit(self, bucket: int, now: int, reason: str):
        prev = self.current_mcs
        self.current_bucket = bucket
        self.current_mcs = mcs_for_bucket(bucket, self.cfg)
        if prev != self.current_mcs:
            self.last_change_us = now
            self.changes.append(now)
            self._expire_changes(now)
        self.pending_bucket = -1
        self.pending_streak = 0
        return reason

    def tick_no_data(self, now: int):
        """No antenna this interval -> failsafe watchdog."""
        self._expire_changes(now)
        if self.last_datagram_us == 0:
            return None
        gap = now - self.last_datagram_us
        if gap < int(self.cfg.failsafe_timeout_s * 1e6):
            return None
        if self.in_failsafe and self.current_bucket == 0:
            return None
        self.in_failsafe = True
        self.recovery_streak = 0
        return self._commit(0, now, "failsafe")

    def update(self, eff_rssi: float, now: int):
        self.last_datagram_us = now
        cfg = self.cfg
        candidate = bucket_from_rssi(eff_rssi, self.current_bucket, cfg)

        if self.current_bucket < 0:
            return self._commit(candidate, now, "init")

        recovered = False
        if self.in_failsafe:
            floor_eff = cfg.rssi_thresh_low + cfg.rssi_deadband_db
            if eff_rssi >= floor_eff:
                self.recovery_streak += 1
                if self.recovery_streak >= cfg.failsafe_recovery_consecutive:
                    self.in_failsafe = False
                    self.recovery_streak = 0
                    recovered = True
                else:
                    return None
            else:
                self.recovery_streak = 0
                return None

        if candidate == self.current_bucket:
            self.pending_bucket = -1
            self.pending_streak = 0
            return "recovered" if recovered else None

        if candidate != self.pending_bucket:
            self.pending_bucket = candidate
            self.pending_streak = 1
        else:
            self.pending_streak += 1

        going_down = candidate < self.current_bucket
        required = cfg.down_consecutive if going_down else cfg.up_consecutive
        if self.pending_streak < required:
            return "recovered" if recovered else None

        elapsed = (now - self.last_change_us) / 1e6
        if going_down:
            if elapsed < cfg.down_cooldown_s:
                return "recovered" if recovered else None
            return self._commit(candidate, now, "down")
        cooldown = cfg.up_cooldown_s
        if self._is_oscillating():
            cooldown *= cfg.oscillation_backoff
        if elapsed < cooldown:
            return "recovered" if recovered else None
        return self._commit(candidate, now, "up")


# Decision points the closed loop should FLAG on (per link-controller alignment):
# a down-shift, failsafe entry, oscillation, and recovery are the risk/transition
# events -- an 'up' or 'init' is not a risk and is not flagged.
FLAG_EVENTS = {"down", "failsafe", "recovered"}


class LinkScorer:
    """Combined scorer + selector. step(record) -> per-record dict."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.scorer = Scorer(self.cfg)
        self.selector = Selector(self.cfg)

    def step(self, record: dict) -> dict:
        now = int(float(record.get("ts_ms", 0)) * 1000)  # ms -> us
        score = self.scorer.update(record)
        if score is None:                       # no antenna -> watchdog path
            event = self.selector.tick_no_data(now)
            eff = None
        else:
            event = self.selector.update(score["effective_rssi"], now)
            eff = score["effective_rssi"]
        oscillating = self.selector._is_oscillating()
        flag = (event in FLAG_EVENTS) or (event in ("up", "down") and oscillating)
        out = {
            "ts_ms": record.get("ts_ms"),
            "effective_rssi": eff,
            "bucket": self.selector.current_bucket,
            "mcs": self.selector.current_mcs,
            "event": event,                     # init/up/down/failsafe/recovered/None
            "oscillating": oscillating,
            "in_failsafe": self.selector.in_failsafe,
            "flag": bool(flag),
        }
        if score is not None:
            out.update({k: round(score[k], 3) for k in
                        ("raw_rssi", "smoothed_rssi", "loss_penalty_db")})
        return out


def stream_records(args):
    if args.stdin:
        for line in sys.stdin:
            line = line.strip()
            if line:
                yield json.loads(line)
    else:
        yield from load_jsonl(args.input)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="telemetry/walk2_wfb.jsonl")
    ap.add_argument("--stdin", action="store_true", help="read JSONL from stdin (live)")
    ap.add_argument("--emit", help="write per-record scored JSONL here")
    ap.add_argument("--quiet", action="store_true", help="summary only, no event log")
    args = ap.parse_args()

    ls = LinkScorer()
    events = collections.Counter()
    bucket_dwell = collections.Counter()
    flags = 0
    n = 0
    emit_fh = open(args.emit, "w") if args.emit else None
    for rec in stream_records(args):
        out = ls.step(rec)
        n += 1
        bucket_dwell[out["bucket"]] += 1
        if out["event"]:
            events[out["event"]] += 1
            if not args.quiet:
                eff = out["effective_rssi"]
                eff_s = f"{eff:6.1f}" if eff is not None else "  n/a "
                tag = " *FLAG*" if out["flag"] else ""
                print(f"[{out['event']:9}] ts={out['ts_ms']} eff_rssi={eff_s} "
                      f"bucket={out['bucket']} mcs={out['mcs']}"
                      f"{' OSC' if out['oscillating'] else ''}{tag}", flush=True)
        if out["flag"]:
            flags += 1
        if emit_fh:
            emit_fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    if emit_fh:
        emit_fh.close()

    print(f"\n# {n} records | events: {dict(events) or '{}'} "
          f"| flagged decision points: {flags}", file=sys.stderr)
    dwell = {f"mcs{ls.cfg.mcs_bucket[b if b in (0,1) else 2]}(b{b})": c
             for b, c in sorted(bucket_dwell.items()) if b >= 0}
    print(f"# bucket dwell (records): {dwell}", file=sys.stderr)
    if args.emit:
        print(f"# wrote {args.emit}", file=sys.stderr)


if __name__ == "__main__":
    main()
