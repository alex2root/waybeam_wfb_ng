#!/usr/bin/env python3
"""Generate synthetic `wfb_rx_native -Y` (type:"rx_ant") JSONL for development.

This MATCHES the real wire schema captured from waybeam's wfb_rx_native -Y on a
2-adapter / 4-antenna ground station (ids 0,1 on adapter A; 100,101 on adapter
B), so a model trained here is feature-compatible with a real capture:

    {"ts_ms":..,"type":"rx_ant","ver":1,"seq":..,"interval_ms":100,
     "ant":[{"freq":5805,"mcs":3,"bw":20,"id":"0","pkts":..,
             "rssi":{"min":..,"avg":..,"max":..},
             "snr":{"min":..,"avg":..,"max":..}}, ...x4],
     "pkt":{"all":..,"bytes":..,"dec_err":..,"lost":..,"fec_recovered":..,
            "uniq":..,"diversity":..,"adapters":2, ...}}

Real captures have no `label`; this adds one for the supervised demo
(0=healthy 1=degraded 2=critical) because bench data is almost all healthy —
the synthetic degraded/critical records cover the bad-link space the trainer
needs. Drop `label` for inference; the trainer excludes it from features.

Replace with a real capture (still no code change needed downstream):
    tshark -i lo -f 'udp port 6600' -T fields -e data.data \
      | <hex-decode to one JSON obj per line> > telemetry/real_wfb.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

# physical antenna ids on this ground station: adapter A {0,1}, adapter B {100,101}
ANT_IDS = ["0", "1", "100", "101"]


def classify(rssi: float, snr: float, lost: int, fec_rec: int) -> int:
    if lost > 0 or snr < 8 or rssi < -82:
        return 2  # critical: uncorrectable loss / link on the edge
    if fec_rec > 6 or snr < 14 or rssi < -72:
        return 1  # degraded: FEC working hard, margin shrinking
    return 0      # healthy


def _band(center: float, spread: float, rng: random.Random) -> dict:
    """A {min,avg,max} object around an integer-rounded center (matches the
    real emitter, which reports integer dBm RSSI / dB SNR)."""
    avg = round(center)
    return {"min": avg - round(abs(rng.gauss(0, spread))),
            "avg": avg,
            "max": avg + round(abs(rng.gauss(0, spread)))}


def gen(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rssi, snr = -14.0, 36.0          # healthy bench baseline (vehicle close)
    fade = 0.0                       # slow range/obstruction component
    seq = 0
    interval_ms = 100                # 10 Hz stats
    # per-antenna fixed offsets so diversity looks realistic (B slightly weaker)
    ant_off = {"0": 0.0, "1": -4.0, "100": -10.0, "101": -8.0}
    out = []
    for i in range(n):
        if rng.random() < 0.05:
            fade = rng.uniform(20, 60)     # enter a fade/range event
        fade *= 0.95                        # decay back toward baseline (persists longer)
        cur_rssi = rssi - fade + rng.gauss(0, 1.5)
        cur_snr = max(0.0, snr - fade * 0.8 + rng.gauss(0, 1.0))

        # packet model: ~100 uniq/interval at 10 Hz; loss/FEC climb as SNR drops
        uniq = rng.randint(95, 115)
        margin = max(0.0, cur_snr)
        p_bad = max(0.0, min(0.5, (16 - margin) / 40))
        fec_rec = int(uniq * p_bad * rng.uniform(0.3, 0.7))
        lost = int(uniq * max(0.0, p_bad - 0.28) * rng.uniform(0.4, 1.2))
        diversity = uniq                   # pkts seen on >1 antenna (~all of them up close)
        all_pkts = uniq * 2 - rng.randint(0, 5)   # total across antennas (diversity ~2x)
        bytes_ = all_pkts * 1340
        label = classify(cur_rssi, cur_snr, lost, fec_rec)

        ant = []
        for aid in ANT_IDS:
            a_rssi = cur_rssi + ant_off[aid]
            a_snr = max(0.0, cur_snr + ant_off[aid] * 0.3)
            ant.append({
                "freq": 5805, "mcs": 3, "bw": 20, "id": aid,
                "pkts": uniq + rng.randint(-2, 2),
                "rssi": _band(a_rssi, 2.0, rng),
                "snr": _band(a_snr, 1.5, rng),
            })

        seq += 1
        out.append({
            "ts_ms": i * interval_ms,
            "type": "rx_ant",
            "ver": 1,
            "seq": seq,
            "interval_ms": interval_ms,
            "ant": ant,
            "pkt": {
                "all": all_pkts, "bytes": bytes_, "dec_err": 0, "session": 0,
                "data": all_pkts, "uniq": uniq, "fec_recovered": fec_rec,
                "lost": lost, "bad": int(lost * 0.2), "outgoing": uniq // 3,
                "outgoing_bytes": (uniq // 3) * 1340, "diversity": diversity,
                "mode_mismatch": 0, "adapters": 2,
            },
            "label": label,  # remove on a real capture; trainer drops it from features
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=2000, help="number of records")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("-o", "--out", default="telemetry/sample_wfb.jsonl")
    args = ap.parse_args()
    recs = gen(args.n, args.seed)
    with open(args.out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    counts = {0: 0, 1: 0, 2: 0}
    for r in recs:
        counts[r["label"]] += 1
    print(f"wrote {len(recs)} records -> {args.out}")
    print(f"labels: healthy={counts[0]} degraded={counts[1]} critical={counts[2]}")


if __name__ == "__main__":
    main()
