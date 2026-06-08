"""Schema-agnostic flattener for wfb_rx -Y JSON telemetry.

The exact field layout of `waybeam_wfb_ng wfb_rx -Y` can vary by build, so this
module deliberately does NOT hard-code a schema. It flattens any JSON object
into a flat {dot.path: float} feature dict, which means a real capture will work
without code changes. Drop your real records in and the trainer/probe/pipeline
pick up whatever numeric fields exist.

Rules:
  - nested dicts      -> dotted keys ("rx_ant_stats.5180:1:20.rssi_avg")
  - list of numbers   -> <key>.min/.avg/.max/.last/.n  (e.g. per-antenna rssi)
  - list of objects   -> keyed by each object's unique "id" field when present
                         ("ant.101.rssi.avg"), else by index ("ant.0.rssi.avg").
                         Keying by id is essential for wfb_rx: the per-antenna
                         "ant" array is emitted in map-iteration order, which is
                         NOT stable across records, so positional indexing would
                         silently swap one physical antenna's stats for another.
  - bool              -> 0.0 / 1.0
  - non-numeric       -> ignored (strings like "type":"rx", ids, etc.)

A stable feature *manifest* (sorted union of keys seen across a dataset) keeps
the vector ordering identical between training, the throughput probe, the live
pipeline, and the generated C model.
"""
from __future__ import annotations

import json
from typing import Any, Iterable


def _num(x: Any) -> float | None:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float)):
        return float(x)
    return None


def flatten(record: dict, prefix: str = "") -> dict[str, float]:
    """Flatten one wfb_rx -Y record into {feature_name: float}."""
    out: dict[str, float] = {}
    for key, val in record.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict):
            out.update(flatten(val, prefix=path + "."))
        elif isinstance(val, (list, tuple)):
            nums = [n for n in (_num(v) for v in val) if n is not None]
            if nums:
                out[f"{path}.min"] = min(nums)
                out[f"{path}.max"] = max(nums)
                out[f"{path}.avg"] = sum(nums) / len(nums)
                out[f"{path}.last"] = nums[-1]
                out[f"{path}.n"] = float(len(nums))
            # lists of objects -> key by a unique "id" field if every object
            # has one (order-independent), else fall back to positional index.
            dicts = [v for v in val if isinstance(v, dict)]
            use_id = bool(dicts) and all("id" in v for v in dicts) and \
                len({str(v["id"]) for v in dicts}) == len(dicts)
            for i, v in enumerate(val):
                if isinstance(v, dict):
                    tag = str(v["id"]) if use_id else str(i)
                    sub = {k: vv for k, vv in v.items() if not (use_id and k == "id")}
                    out.update(flatten(sub, prefix=f"{path}.{tag}."))
        else:
            n = _num(val)
            if n is not None:
                out[path] = n
    return out


def build_manifest(records: Iterable[dict], drop: set[str] | None = None) -> list[str]:
    """Sorted union of feature names across all records (the stable column order).

    `drop` removes label/bookkeeping columns (e.g. {"label","timestamp"}) so they
    never leak into the feature vector.
    """
    drop = drop or set()
    keys: set[str] = set()
    for r in records:
        keys.update(flatten(r).keys())
    keys -= drop
    return sorted(keys)


def vectorize(record: dict, manifest: list[str]) -> list[float]:
    """Turn one record into a fixed-length vector following the manifest order.
    Missing features are 0.0 (matches the C model's input contract)."""
    flat = flatten(record)
    return [flat.get(name, 0.0) for name in manifest]


def best_chain(record: dict) -> dict:
    """Reduce a record's antennas to the BEST diversity chain, RSSI-primary.

    RSSI is the robust signal that every driver reports and that the link
    controller acts on, so it is always the selector. SNR is used only when the
    driver provides it (some don't) -- it adds the implied noise floor
    (RSSI - SNR) and, over time, an interference signal (SNR dropping faster than
    RSSI = noise floor rising, vs a fade where both fall together; RSSI/SNR are
    ~0.8 correlated, so a large negative residual flags interference).

    Returns: {rssi, snr, noise_floor, snr_available, pkts}. rssi is None only
    when the record has no antennas at all (a dropout -- see signal_lost)."""
    ants = record.get("ant") or []
    rssi_ants = [a for a in ants if isinstance(a.get("rssi"), dict)
                 and a["rssi"].get("avg") is not None]
    if not rssi_ants:
        return {"rssi": None, "snr": None, "noise_floor": None,
                "snr_available": False, "pkts": record.get("pkt", {}).get("all")}
    best = max(rssi_ants, key=lambda a: a["rssi"]["avg"])
    rssi = float(best["rssi"]["avg"])
    snr = None
    if isinstance(best.get("snr"), dict) and best["snr"].get("avg") is not None:
        snr = float(best["snr"]["avg"])
    return {
        "rssi": rssi,
        "snr": snr,
        "noise_floor": (rssi - snr) if snr is not None else None,
        "snr_available": snr is not None,
        "pkts": best.get("pkts"),
    }


# --- Deployable reduced feature set ------------------------------------------
# The full flatten() manifest produces per-antenna columns keyed by the antenna
# ids that happen to appear in a session (ant.100.snr.max, ...). A tree trained
# on those overfits to one session's antenna layout AND leans on SNR, which the
# real link_controller ignores and not every driver reports. The deployable
# Tier-1 instead uses an antenna-id-independent, SNR-free reduction: the
# best-diversity-chain RSSI (what the controller acts on) plus the flat packet
# counters. Stable column order -> a fixed C input contract for the SSC338Q.
SIGNAL_LOST_RSSI = -128.0  # sentinel when no antenna reported (a dropout): the
# worst-case RSSI, so the tree unifies fade and total-loss as "very weak signal"
# and the controller failsafe (no rx -> lowest MCS) is mirrored numerically.
REDUCED_FEATURES = ["rssi", "pkt_all", "pkt_lost", "pkt_fec_recovered", "pkt_uniq"]


def reduced_features(record: dict) -> dict[str, float]:
    """Antenna-id-independent, SNR-free feature dict (see REDUCED_FEATURES).

    best_chain RSSI (sentinel -128 on a dropout) + received/lost/recovered/unique
    packet counters. This is the deployable Tier-1 contract."""
    bc = best_chain(record)
    rssi = bc["rssi"]
    pkt = record.get("pkt", {})

    def g(k: str) -> float:
        return float(pkt.get(k, 0) or 0)

    return {
        "rssi": SIGNAL_LOST_RSSI if rssi is None else float(rssi),
        "pkt_all": g("all"),
        "pkt_lost": g("lost"),
        "pkt_fec_recovered": g("fec_recovered"),
        "pkt_uniq": g("uniq"),
    }


def vectorize_reduced(record: dict) -> list[float]:
    """reduced_features in REDUCED_FEATURES order (the C model's input contract)."""
    f = reduced_features(record)
    return [f[name] for name in REDUCED_FEATURES]


# --- Causal trend features ---------------------------------------------------
# A single instantaneous record can't see the link DEGRADING: the deployable
# reduced model nails reactive criticals (signal_lost) but misses partial-loss
# and *predictive* (worsened-before-loss) criticals. Trend features add the
# missing time axis. They are strictly CAUSAL (trailing window, past + current
# only) so they compute online on the SSC338Q from a small ring buffer -- no
# lookahead, deployable as-is.
TREND_FEATURES = ["rssi_slope", "rssi_mean", "loss_rate", "fec_rate",
                  "time_since_loss", "dropout_frac"]
FULL_FEATURES = REDUCED_FEATURES + TREND_FEATURES


def _slope(ys: list[float]) -> float:
    """Least-squares slope of ys against x=0..n-1 (dB per record). 0 for n<2."""
    n = len(ys)
    if n < 2:
        return 0.0
    xbar = (n - 1) / 2.0
    ybar = sum(ys) / n
    num = sum((i - xbar) * (y - ybar) for i, y in enumerate(ys))
    den = sum((i - xbar) ** 2 for i in range(n))
    return num / den if den else 0.0


class TrendState:
    """Streaming, causal trend-feature extractor (one per link/session).

    Push records in order; each push() returns the full feature dict (the
    reduced base features + TREND_FEATURES) for THAT record, using only it and
    the preceding `window` records. Mirrors exactly what a C ring-buffer daemon
    would maintain on the air unit, so training and deploy stay bit-aligned."""

    def __init__(self, window: int = 20, since_cap: int = 200):
        import collections
        self.window = window
        self.since_cap = since_cap
        self.buf = collections.deque(maxlen=window)  # (rssi, lost, uniq, fec, dropout)
        self.since_loss = float(since_cap)            # records since last loss/dropout

    def push(self, record: dict) -> dict[str, float]:
        base = reduced_features(record)
        rssi, lost = base["rssi"], base["pkt_lost"]
        uniq, fec = base["pkt_uniq"], base["pkt_fec_recovered"]
        dropout = 1.0 if base["pkt_all"] == 0 else 0.0
        # time_since_loss is causal: 0 if THIS record lost/dropped, else +1 (capped)
        if lost > 0 or dropout:
            self.since_loss = 0.0
        else:
            self.since_loss = min(self.since_cap, self.since_loss + 1.0)
        self.buf.append((rssi, lost, uniq, fec, dropout))

        rssis = [b[0] for b in self.buf]
        lost_sum = sum(b[1] for b in self.buf)
        uniq_sum = sum(b[2] for b in self.buf)
        fec_sum = sum(b[3] for b in self.buf)
        drop_sum = sum(b[4] for b in self.buf)
        n = len(self.buf)
        trend = {
            "rssi_slope": _slope(rssis),
            "rssi_mean": sum(rssis) / n,
            "loss_rate": lost_sum / (lost_sum + uniq_sum) if (lost_sum + uniq_sum) else 0.0,
            "fec_rate": fec_sum / (uniq_sum + 1.0),
            "time_since_loss": self.since_loss,
            "dropout_frac": drop_sum / n,
        }
        return {**base, **trend}


def windowed_features(records: list[dict], window: int = 20) -> list[list[float]]:
    """Batch helper: per-record FULL_FEATURES vectors over a session (in order).

    Runs one TrendState across the records, so trend features are causal and
    identical to what the live extractor would produce. Do NOT mix sessions in
    one call -- trend would bleed across the boundary."""
    state = TrendState(window=window)
    out = []
    for r in records:
        f = state.push(r)
        out.append([f[name] for name in FULL_FEATURES])
    return out


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


if __name__ == "__main__":  # quick self-check — real wfb_rx_native -Y rx_ant shape
    sample = {
        "ts_ms": 2219190, "type": "rx_ant", "ver": 1, "seq": 27, "interval_ms": 100,
        "ant": [
            {"freq": 5805, "mcs": 3, "bw": 20, "id": "0", "pkts": 105,
             "rssi": {"min": -11, "avg": -10, "max": -10},
             "snr": {"min": 29, "avg": 36, "max": 45}},
            {"freq": 5805, "mcs": 3, "bw": 20, "id": "101", "pkts": 104,
             "rssi": {"min": -20, "avg": -20, "max": -20},
             "snr": {"min": 29, "avg": 34, "max": 38}},
        ],
        "pkt": {"all": 209, "lost": 0, "fec_recovered": 0, "dec_err": 0,
                "uniq": 105, "diversity": 104, "adapters": 2},
    }
    # note: ant entries are keyed by their stable "id" (ant.0.*, ant.101.*),
    # NOT by position, so reordering the array can't swap antennas' stats.
    flat = flatten(sample)
    print(json.dumps(flat, indent=2, sort_keys=True))
