#!/usr/bin/env python3
"""Hybrid Tier-1 + Tier-2 runtime: trained model handles the firehose,
Gemma explains only the exceptions.

  Tier 1 (cheap, every record): the trained tier1.pkl model -- or, if no model
          has been trained yet, a transparent threshold fallback -- classifies
          link health at full rate.
  Tier 2 (Gemma, rare): only when Tier-1 flags degraded/critical does it call
          the local LLM (think=false) for a human-readable diagnosis + decision,
          rate-limited so a sustained bad link can't swamp it.

This is the architecture you actually want for live telemetry: the LLM never
sees the firehose, so the SSC338Q-friendly tree carries 100% of the volume and
the LLM adds judgement on the <1% that matters.

    python3 telemetry/wfb_pipeline.py --input telemetry/sample_wfb.jsonl
    waybeam_wfb_ng wfb_rx -Y ... | python3 telemetry/wfb_pipeline.py --stdin
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import urllib.request

from wfb_schema import (flatten, load_jsonl, vectorize, vectorize_reduced,
                        FULL_FEATURES, TrendState)

OLLAMA = "http://localhost:11434/api/generate"
TIER2_SYSTEM = ("You are an FPV Wi-Fi link engineer. Given recent wfb_rx stats "
                "for a DEGRADED/CRITICAL link, reply ONLY compact JSON: "
                '{"cause":"<short>","action":"hold|lower_bitrate|switch_channel|rth"}.')
LABELS = {0: "ok", 1: "degraded", 2: "critical"}


def threshold_tier1(rec: dict) -> int:
    """Transparent fallback when no model is trained yet (mirrors gen_sample).

    Schema-tolerant: matches both the real wfb_rx_native -Y keys
    (ant.<id>.rssi.avg / ant.<id>.snr.avg / pkt.lost / pkt.fec_recovered) and
    the older synthetic keys (*.rssi_avg / *.lost.max / *.fec_rec.max). RSSI/SNR
    are reduced across antennas to the BEST chain (diversity = you only lose the
    link when every antenna is bad)."""
    f = flatten(rec)
    # total dropout: no packets received this interval (ant[] empty, all counters
    # zero). pkt.lost stays 0 because nothing arrived to count as lost, so the
    # outage must be read from the received-packet count -> unambiguously critical.
    pkt = rec.get("pkt", {})
    received = pkt.get("all", pkt.get("uniq", None))
    if received is not None and float(received or 0) == 0:
        return 2
    rssis = [v for k, v in f.items() if k.endswith("rssi.avg") or k.endswith("rssi_avg")]
    snrs = [v for k, v in f.items() if k.endswith("snr.avg") or k.endswith("snr_avg")]
    rssi = max(rssis) if rssis else -55.0
    snr = max(snrs) if snrs else 28.0
    lost = next((v for k, v in f.items()
                 if k.endswith("pkt.lost") or k.endswith("lost.last") or k.endswith("lost.max")), 0.0)
    fec = next((v for k, v in f.items()
                if k.endswith("fec_recovered") or k.endswith("fec_rec.last") or k.endswith("fec_rec.max")), 0.0)
    if lost > 0 or snr < 8 or rssi < -82:
        return 2
    if fec > 6 or snr < 14 or rssi < -72:
        return 1
    return 0


def load_tier1(path: str):
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError:
        return None


def model_vector(bundle, rec: dict) -> list[float]:
    """Stateless vectorize for the reduced or full-flatten bundles. NOTE: trend
    bundles are stateful -- use make_vectorizer() for those; this falls back to
    the reduced base (trend cols zeroed) if called on a trend bundle out of a
    stream context."""
    if bundle.get("trend"):
        f = TrendState(window=bundle.get("window", 10)).push(rec)  # cold state -> w=1
        return [f[k] for k in FULL_FEATURES]
    if bundle.get("reduced"):
        return vectorize_reduced(rec)
    return vectorize(rec, bundle["manifest"])


def make_vectorizer(bundle):
    """Return a per-record rec->vector callable matching how the bundle was
    trained. For a trend bundle this OWNS a TrendState, so call it once per
    record in stream order (exactly what a deployed ring buffer does)."""
    if bundle is None:
        return None
    if bundle.get("trend"):
        state = TrendState(window=bundle.get("window", 10))
        return lambda rec: [state.push(rec)[k] for k in FULL_FEATURES]
    if bundle.get("reduced"):
        return lambda rec: vectorize_reduced(rec)
    return lambda rec: vectorize(rec, bundle["manifest"])


def tier1_predict(bundle, rec: dict, vec=None) -> int:
    if bundle is None:
        return threshold_tier1(rec)
    x = [vec if vec is not None else model_vector(bundle, rec)]
    if bundle["kind"] == "iforest":
        return 2 if bundle["model"].predict(x)[0] == -1 else 0
    return int(bundle["model"].predict(x)[0])


def tier2_explain(model: str, window: list[dict]) -> str:
    rows = [flatten(r) for r in window]
    prompt = f"{TIER2_SYSTEM}\nstats={json.dumps(rows, separators=(',',':'))}\nverdict:"
    payload = {"model": model, "prompt": prompt, "think": False, "stream": False,
               "options": {"num_predict": 64, "temperature": 1.0}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except Exception as e:  # never let Tier-2 crash the loop
        return f"ERROR: tier2 unavailable ({e})"


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
    ap.add_argument("--input", default="telemetry/sample_wfb.jsonl")
    ap.add_argument("--stdin", action="store_true", help="read JSONL from stdin (live)")
    ap.add_argument("--model-path", default="telemetry/model/tier1.pkl")
    ap.add_argument("--llm", default="gemma4:26b-a4b-it-qat")
    ap.add_argument("--tier2-cooldown", type=float, default=5.0,
                    help="min seconds between LLM escalations")
    ap.add_argument("--context", type=int, default=10, help="records of history sent to Tier-2")
    ap.add_argument("--no-tier2", action="store_true", help="Tier-1 only (offline)")
    args = ap.parse_args()

    bundle = load_tier1(args.model_path)
    src = "trained " + bundle["kind"] if bundle else "threshold fallback (train for better)"
    if bundle and bundle.get("trend"):
        src += f" +trend(w={bundle.get('window', 10)})"
    print(f"# Tier-1: {src}   Tier-2: {'off' if args.no_tier2 else args.llm}", flush=True)

    vectorizer = make_vectorizer(bundle)
    history, last_t2, counts, escalations = [], 0.0, {0: 0, 1: 0, 2: 0}, 0
    for rec in stream_records(args):
        history.append(rec)
        history = history[-args.context:]
        vec = vectorizer(rec) if vectorizer else None
        state = tier1_predict(bundle, rec, vec=vec)
        counts[state] += 1
        if state == 0:
            continue
        now = time.monotonic()
        if not args.no_tier2 and (now - last_t2) >= args.tier2_cooldown:
            last_t2 = now
            escalations += 1
            verdict = tier2_explain(args.llm, history)
            print(f"[{LABELS[state].upper():8}] tier2 -> {verdict}", flush=True)
        else:
            print(f"[{LABELS[state].upper():8}] (tier1 only, cooldown)", flush=True)

    total = sum(counts.values())
    print(f"\n# {total} records | ok={counts[0]} degraded={counts[1]} critical={counts[2]} "
          f"| Tier-2 escalations={escalations} "
          f"({(escalations/total*100 if total else 0):.1f}% of volume hit the LLM)")


if __name__ == "__main__":
    main()
