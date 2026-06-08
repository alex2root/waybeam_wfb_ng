#!/usr/bin/env python3
"""Reactive drop-detection latency on real video-link captures.

The slope/headroom model handles GRADUAL fades. This tool measures the OTHER
regime -- sudden drops (obstacle/null/maneuver) -- where there is no precursor and
the only levers are detection speed + reaction speed. For each drop event in a 10 Hz
rx_ant capture it reports, in milliseconds:

  onset        first sample where the link left "clean" on the way into a critical state
  -> bad       onset -> first ground-truth-critical sample (loss>=crit or dropout)
  -> react     onset -> first time wfb_link_score commits a DOWN-shift or FAILSAFE

A "drop" is a clean->critical transition preceded by >= min-healthy clean samples.
critical = loss_rate >= --crit-loss OR a dropout (no antenna reported / record gap).
"sudden" = onset->bad <= --sudden-ms. For total-loss RECORD GAPS the record-driven
replay can't fire the watchdog (no records arrive), so the live failsafe latency is
the configured failsafe_timeout -- reported analytically and flagged [gap].

Usage:
  python3 telemetry/drop_latency.py telemetry/walk2_wfb.jsonl [more.jsonl ...]
      [--crit-loss 0.5] [--clean-loss 0.1] [--min-healthy 5] [--sudden-ms 1000]
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys

import wfb_schema
from wfb_link_score import LinkScorer, Config


def load(path: str) -> list:
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "rx_ant" and "ts_ms" in r:
                recs.append(r)
    recs.sort(key=lambda r: r["ts_ms"])
    return recs


def annotate(recs: list) -> None:
    """Per-record: best-chain rssi (None=dropout), unrecovered loss_rate, gap."""
    prev_ts = None
    for r in recs:
        bc = wfb_schema.best_chain(r)
        pkt = r.get("pkt", {})
        uniq = float(pkt.get("uniq", 0) or 0)
        lost = float(pkt.get("lost", 0) or 0)
        acc = uniq + lost
        r["_rssi"] = bc["rssi"]
        r["_loss"] = (lost / acc) if acc else (None if bc["rssi"] is None else 0.0)
        ts = r["ts_ms"]
        r["_gap"] = (ts - prev_ts) if prev_ts is not None else 0
        prev_ts = ts


def segment(recs: list, crit_loss: float, clean_loss: float, min_healthy: int,
            interval: int) -> list:
    """Find clean->critical drop events. Returns list of event dicts."""
    def is_clean(r):
        return r["_rssi"] is not None and (r["_loss"] is not None and r["_loss"] < clean_loss)

    def is_crit(r):
        return r["_rssi"] is None or (r["_loss"] is not None and r["_loss"] >= crit_loss)

    events = []
    state = "healthy"
    healthy = min_healthy            # assume warmed up
    cur = None
    bad_run = 0
    for i, r in enumerate(recs):
        gap_drop = r["_gap"] > 1.5 * interval
        bad = is_crit(r) or gap_drop
        if state == "healthy":
            if is_clean(r) and not gap_drop:
                healthy += 1
            else:
                if healthy >= min_healthy:
                    cur = {"onset": i, "bad": (i if bad else None),
                           "min_rssi": r["_rssi"], "peak_loss": r["_loss"] or 0.0,
                           "had_gap": gap_drop, "max_gap": r["_gap"] if gap_drop else 0,
                           "bad_run_max": 0, "recover": 0}
                    bad_run = 1 if bad else 0
                    cur["bad_run_max"] = bad_run
                    state = "degraded"
                healthy = 0
        else:  # degraded
            bad_run = bad_run + 1 if bad else 0
            cur["bad_run_max"] = max(cur["bad_run_max"], bad_run)
            if bad and cur["bad"] is None:
                cur["bad"] = i
            if r["_rssi"] is not None:
                cur["min_rssi"] = (r["_rssi"] if cur["min_rssi"] is None
                                   else min(cur["min_rssi"], r["_rssi"]))
            cur["peak_loss"] = max(cur["peak_loss"], r["_loss"] or 0.0)
            if gap_drop:
                cur["had_gap"] = True
                cur["max_gap"] = max(cur["max_gap"], r["_gap"])
            if is_clean(r) and not gap_drop:
                cur["recover"] += 1
                if cur["recover"] >= min_healthy:
                    cur["recover_at"] = i - min_healthy + 1
                    if cur["bad"] is not None:
                        events.append(cur)
                    state, healthy, cur = "healthy", cur["recover"], None
            else:
                cur["recover"] = 0
    if cur and cur["bad"] is not None:
        cur["recover_at"] = len(recs) - 1
        events.append(cur)
    return events


def replay(recs: list, cfg: Config) -> list:
    """Per-record FSM step output, parallel to recs (for bucket-at-onset lookup)."""
    ls = LinkScorer(cfg)
    return [ls.step(r) for r in recs]


def run_sweep(captures: list, timeouts: list, baseline_s: float) -> None:
    """Replay at each failsafe_timeout; count warranted vs premature failsafes.

    For each failsafe trigger, the surrounding no-rx outage = (next reception -
    last reception). warranted = outage >= baseline (it would have tripped at the
    baseline timeout too); premature = a shorter blip that only trips because the
    timeout was lowered -> a needless drop to min MCS + slow-up recovery.
    """
    baseline_ms = int(baseline_s * 1000)
    caps = []
    for path in captures:
        recs = load(path)
        if not recs:
            continue
        annotate(recs)
        recv = [r["ts_ms"] for r in recs if r["_rssi"] is not None]
        caps.append((recs, recv, recs[-1]["ts_ms"]))
    print(f"failsafe_timeout sweep over {len(caps)} capture(s)  "
          f"(baseline {baseline_ms}ms; warranted = outage >= baseline)")
    print(f"{'timeout':>8} {'failsafes':>10} {'warranted':>10} {'premature':>10} "
          f"{'react':>7} {'gain':>6}")
    for T in timeouts:
        cfg = Config()
        cfg.failsafe_timeout_s = T
        n_fs = warr = prem = 0
        for recs, recv, end_ts in caps:
            ls = LinkScorer(cfg)
            for r in recs:
                if ls.step(r)["event"] == "failsafe":
                    ts = r["ts_ms"]
                    i = bisect.bisect_left(recv, ts)
                    last = recv[i - 1] if i else None
                    nxt = recv[i] if i < len(recv) else end_ts   # outage runs to capture end
                    span = (nxt - last) if last is not None else None
                    n_fs += 1
                    if span is not None and span >= baseline_ms:
                        warr += 1
                    else:
                        prem += 1
        Tms = int(T * 1000)
        print(f"{T:>7.2f}s {n_fs:>10} {warr:>10} {prem:>10} {Tms:>6}m "
              f"{baseline_ms - Tms:>5}m")
    print("  react = reaction latency for a warranted (sustained) outage (= the timeout)")
    print("  premature = failsafe on an outage shorter than baseline -> needless min-MCS")
    print("              drop + slow-up (3 s) recovery = throughput cost of reacting faster")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--crit-loss", type=float, default=0.5)
    ap.add_argument("--clean-loss", type=float, default=0.1)
    ap.add_argument("--min-healthy", type=int, default=5)
    ap.add_argument("--sudden-ms", type=int, default=1000)
    ap.add_argument("--min-bad", type=int, default=2,
                    help="min consecutive bad samples to count as a drop (filters glitches)")
    ap.add_argument("--sweep", metavar="T1,T2,...",
                    help="sweep failsafe_timeout values (s) and report warranted vs "
                         "premature failsafes, e.g. 0.5,0.4,0.3,0.2,0.15,0.1")
    args = ap.parse_args()

    cfg = Config()
    if args.sweep:
        run_sweep(args.captures, [float(x) for x in args.sweep.split(",")],
                  cfg.failsafe_timeout_s)
        return
    failsafe_ms = int(cfg.failsafe_timeout_s * 1000)
    lat_react, lat_bad = [], []
    n_total = n_sudden = n_floor = n_miss = n_gap = n_glitch = 0

    for path in args.captures:
        recs = load(path)
        if not recs:
            print(f"# {path}: no rx_ant records", file=sys.stderr)
            continue
        interval = int(recs[0].get("interval_ms", 100))
        annotate(recs)
        recv_ts = [r["ts_ms"] for r in recs if r["_rssi"] is not None]   # real receptions
        steps = replay(recs, cfg)
        events = segment(recs, args.crit_loss, args.clean_loss, args.min_healthy, interval)
        fsm = [(recs[i]["ts_ms"], s["event"]) for i, s in enumerate(steps)
               if s["event"] in ("down", "failsafe")]
        kept = [e for e in events if e["bad_run_max"] >= args.min_bad]
        n_glitch += len(events) - len(kept)
        dur = (recs[-1]["ts_ms"] - recs[0]["ts_ms"]) / 1000.0
        print(f"\n=== {path}  ({len(recs)} rec, {dur:.0f}s, {len(kept)} drops, "
              f"{len(events) - len(kept)} glitches filtered) ===")
        # 'dark' = ms from the LAST real reception to the FSM reaction = the true
        # reactive latency to sustained loss (anchoring on the fuzzy onset overstates it).
        print(f"{'onset_t':>8} {'rssi0':>6} {'minR':>5} {'loss':>5} {'bkt0':>4} "
              f"{'->bad':>6} {'dark':>6} {'via':>8} {'kind':>5} {'sud':>4}")
        for e in kept:
            o_ts = recs[e["onset"]]["ts_ms"]
            d_bad = recs[e["bad"]]["ts_ms"] - o_ts
            rec_ts = recs[e["recover_at"]]["ts_ms"]
            bkt0 = steps[e["onset"]]["bucket"]
            hit = next(((ts, ev) for ts, ev in fsm if o_ts <= ts <= rec_ts), None)
            if hit:
                react_ts, via, kind = hit[0], hit[1], "ok"
                i = bisect.bisect_left(recv_ts, react_ts)
                last_good = recv_ts[i - 1] if i else None
                dark = (react_ts - last_good) if last_good is not None else None
                if dark is not None:
                    lat_react.append(dark)
            elif bkt0 <= 0:
                dark, via, kind = None, "-", "floor"   # already min MCS, nothing to demote
                n_floor += 1
            elif e["had_gap"]:
                dark, via, kind = failsafe_ms, "fs*", "gap"  # live watchdog (no recs in replay)
                lat_react.append(failsafe_ms)
                n_gap += 1
            else:
                dark, via, kind = None, "-", "MISS"
                n_miss += 1
            sudden = d_bad <= args.sudden_ms
            n_total += 1
            n_sudden += sudden
            r0 = recs[e["onset"]]["_rssi"]
            print(f"{o_ts/1000:8.1f} {('drop' if r0 is None else f'{r0:.0f}'):>6} "
                  f"{(e['min_rssi'] if e['min_rssi'] is not None else 0):5.0f} "
                  f"{e['peak_loss']*100:4.0f}% {bkt0:4d} {d_bad:5d}m "
                  f"{(str(dark) + 'm') if dark is not None else '  --':>6} "
                  f"{via:>8} {kind:>5} {('S' if sudden else 'gr'):>4}")
            lat_bad.append(d_bad)

    def stats(xs):
        if not xs:
            return "n/a"
        xs = sorted(xs)
        return (f"median {xs[len(xs)//2]}ms  p90 {xs[min(len(xs)-1, int(len(xs)*0.9))]}ms  "
                f"max {xs[-1]}ms")

    print(f"\n=== summary: {n_total} drops ({n_sudden} sudden <= {args.sudden_ms}ms, "
          f"{n_glitch} glitches filtered) ===")
    print(f"  onset->ground-truth-bad      : {stats(lat_bad)}")
    print(f"  true-loss->FSM reaction (dark): {stats(lat_react)}  "
          f"(failsafe_timeout={failsafe_ms}ms)")
    print(f"  reacted (fast-down/failsafe)={len(lat_react)}  already-at-floor={n_floor}  "
          f"total-loss-gap(=failsafe {failsafe_ms}ms)={n_gap}  MISSED={n_miss}")


if __name__ == "__main__":
    main()
