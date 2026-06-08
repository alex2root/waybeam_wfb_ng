#!/usr/bin/env python3
"""Turn a probe-PER walk capture into per-window feature rows for the
headroom / predictive-critical model.

Input is the JSONL that `probe_log.py --by-mcs` writes (one record per received
MCS per ~1 s window): {type:probe, ts_ms, mcs, per, recv, lost, accounted,
rssi, snr, ...}. The swept TX visits cur, cur+1, cur+2 SEQUENTIALLY (~2 s dwell
each), so the three rungs are never measured at the same instant -- we bin
records into sweep-cycle windows (default 6 s = one full cur..cur+2 cycle) and,
within each bin, read each rung's measured PER. From that snapshot we derive the
measured link headroom.

Why this is the signal we want: the probe's RSSI/SNR are the video link's
RSSI/SNR (same drone->ground path, same PHY), and the per-rung PER is the GROUND
TRUTH of "can the link sustain MCS r right now". So a row pairs cheap,
video-derivable inputs (rssi, snr, rssi_slope) with a measured headroom label --
the predictive-critical target, on real loss-cliff data instead of one bench.

Each output row (one per bin):
  t          seconds from capture start (bin centre)
  rssi, snr  mean over the bin (best-antenna, controller-style)
  rssi_slope causal dB/s over the last `--slope-bins` bins (0 until enough history)
  per_cur, per_cur1, per_cur2   measured PER at cur / cur+1 / cur+2 (None if the
                                rung went totally dark in the bin -> treated as
                                unusable, since 0 rx = no seq ref = blackout)
  headroom_rungs  (highest usable rung) - cur; -1 if even cur is unusable.
                  "usable" = measured PER < --thresh.
  n_pkts     accounted packets in the bin (sanity / weight)
  phase      'out' (walking away, before the RSSI minimum) | 'back'

Usage:
  python3 build_probe_features.py walk.jsonl [--cur N] [--bin-s 6] [--thresh 0.10]
      [--slope-bins 2] [--out features.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def load(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "probe" and r.get("accounted"):
                rows.append(r)
    return rows


def build(rows: list, cur: int | None, bin_s: float, thresh: float,
          slope_bins: int) -> list:
    if not rows:
        return []
    rungs = sorted({r["mcs"] for r in rows})
    if cur is None:
        cur = rungs[0]
    # the three nominal rungs we report (cur, cur+1, cur+2), as available
    cols = [cur, cur + 1, cur + 2]

    t0 = min(r["ts_ms"] for r in rows)
    bin_ms = bin_s * 1000.0
    bins: dict[int, dict] = defaultdict(lambda: {
        "per": defaultdict(lambda: [0, 0]), "rssi": [], "snr": []})
    for r in rows:
        b = int((r["ts_ms"] - t0) // bin_ms)
        acc = bins[b]
        acc["per"][r["mcs"]][0] += r["recv"]
        acc["per"][r["mcs"]][1] += r["lost"]
        if r["rssi"] is not None:
            acc["rssi"].append(r["rssi"])
        if r["snr"] is not None:
            acc["snr"].append(r["snr"])

    out = []
    for b in sorted(bins):
        acc = bins[b]
        rssi = sum(acc["rssi"]) / len(acc["rssi"]) if acc["rssi"] else None
        snr = sum(acc["snr"]) / len(acc["snr"]) if acc["snr"] else None
        per = {}
        npk = 0
        for m in cols:
            u, l = acc["per"].get(m, [0, 0])
            tot = u + l
            npk += tot
            per[m] = (l / tot) if tot else None     # None = rung dark this bin
        usable = [m for m in cols if per[m] is not None and per[m] < thresh]
        headroom = (max(usable) - cur) if usable else -1
        out.append({
            "t": round((b + 0.5) * bin_s, 1),
            "rssi": round(rssi, 1) if rssi is not None else None,
            "snr": round(snr, 1) if snr is not None else None,
            "rssi_slope": None,                      # filled below (needs history)
            "per_cur": _r(per[cols[0]]),
            "per_cur1": _r(per[cols[1]]),
            "per_cur2": _r(per[cols[2]]),
            "headroom_rungs": headroom,
            "n_pkts": npk,
            "phase": None,                           # filled below
        })

    # causal RSSI slope (dB/s) over the last `slope_bins` bins
    for i, row in enumerate(out):
        j = i - slope_bins
        if j >= 0 and row["rssi"] is not None and out[j]["rssi"] is not None:
            dt = row["t"] - out[j]["t"]
            if dt > 0:
                row["rssi_slope"] = round((row["rssi"] - out[j]["rssi"]) / dt, 2)

    # phase: split at the RSSI minimum (walk out -> back)
    valid = [(i, r["rssi"]) for i, r in enumerate(out) if r["rssi"] is not None]
    if valid:
        imin = min(valid, key=lambda x: x[1])[0]
        for i, r in enumerate(out):
            r["phase"] = "out" if i <= imin else "back"
    return out


def _r(x):
    return round(x, 4) if x is not None else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walk", help="probe walk JSONL (probe_log --by-mcs output)")
    ap.add_argument("--cur", type=int, default=None,
                    help="current/base MCS (default: lowest rung seen)")
    ap.add_argument("--bin-s", type=float, default=6.0,
                    help="bin width seconds (default 6 = one cur..cur+2 sweep cycle)")
    ap.add_argument("--thresh", type=float, default=0.10,
                    help="PER below which a rung counts as usable (default 0.10)")
    ap.add_argument("--slope-bins", type=int, default=2,
                    help="bins back for the causal RSSI slope (default 2)")
    ap.add_argument("--out", help="write feature rows here as JSONL")
    ap.add_argument("--quiet", action="store_true", help="no table to stderr")
    args = ap.parse_args()

    rows = load(args.walk)
    feats = build(rows, args.cur, args.bin_s, args.thresh, args.slope_bins)
    if not feats:
        print("no usable probe records", file=sys.stderr)
        sys.exit(1)

    if args.out:
        with open(args.out, "w") as f:
            for r in feats:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

    if not args.quiet:
        cur = args.cur if args.cur is not None else sorted({r["mcs"] for r in rows})[0]
        print(f"# {len(feats)} bins  cur=MCS{cur}  bin={args.bin_s}s  "
              f"usable<{args.thresh:.0%}PER", file=sys.stderr)
        hd = f"{'t':>6} {'rssi':>5} {'slope':>6} {'snr':>4} | " \
             f"{'per'+str(cur):>7} {'per'+str(cur+1):>7} {'per'+str(cur+2):>7} | " \
             f"{'hdrm':>4} {'phase':>4} {'npk':>4}"
        print(hd, file=sys.stderr)
        for r in feats:
            sl = f"{r['rssi_slope']:+.2f}" if r['rssi_slope'] is not None else "   . "
            pc = lambda x: f"{x*100:6.1f}%" if x is not None else "   dark"
            print(f"{r['t']:6.0f} {r['rssi'] or 0:5.0f} {sl:>6} {r['snr'] or 0:4.0f} | "
                  f"{pc(r['per_cur'])} {pc(r['per_cur1'])} {pc(r['per_cur2'])} | "
                  f"{r['headroom_rungs']:>4} {r['phase']:>4} {r['n_pkts']:>4}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
