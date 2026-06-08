#!/usr/bin/env python3
"""Closed-loop step 1: turn a wfb telemetry stream into a Tier-2 review queue.

This is the data-collection half of the closed training loop. It runs Tier-1
over every record (exactly like the live pipeline) but, instead of just acting,
it *logs the decision points* together with the objectively MEASURED outcome so
that a labeler (Gemma) and a supervisor (Tier-3) have something concrete to
verify against later.

For each record it computes:
  * the Tier-1 class AND its prediction confidence (tree predict_proba) -- the
    confidence lets us flag uncertain calls, not just bad ones (active learning:
    spend the labeler where the model is least sure).
  * a MEASURED outcome read from the *following* `--outcome-window` records
    (did packets actually drop? did FEC recover them? did the class return to
    ok?). This is ground truth from telemetry, NOT an LLM guess -- it is the
    anchor the whole loop is honest against.

A record is enqueued when ANY of:
    class != ok            (something Tier-1 thinks is wrong)
    confidence < --conf    (Tier-1 is unsure -- the most useful rows to label)
    boundary               (top-two class probs within --margin)

Output: one JSON object per flagged event into telemetry/loop/review_queue.jsonl,
carrying the context window, the Tier-1 guess, and the measured outcome. Feed it
to wfb_gemma_verify.py next.

    python3 telemetry/wfb_review_queue.py --input telemetry/sample_wfb.jsonl
    wfb_rx -Y ... | python3 telemetry/wfb_review_queue.py --stdin
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from wfb_schema import flatten, load_jsonl, best_chain
from wfb_pipeline import (load_tier1, threshold_tier1, LABELS, model_vector,
                          make_vectorizer)

# A signal_lost (no-packets) interval is ambiguous: the link could have faded out
# of range, OR the transmitter just went quiet at strong RSSI (interference, a
# reboot, vehicle-side experiments). These need DIFFERENT actions -- an RF fade
# wants lower MCS/bitrate; a TX dropout can't be fixed by the radio at all. We
# can't read RSSI during the silence, so we tag by the RSSI just BEFORE it went
# dark: strong -> tx_dropout, weak -> rf_fade.
TX_DROPOUT_RSSI = -65.0  # pre-outage RSSI at/above this = not a range problem


def tier1_with_proba(bundle, rec: dict, vec=None):
    """Return (class:int, confidence:float|None, proba:list|None).

    Trees give a real probability vector -> we can flag low-confidence / boundary
    calls. The threshold fallback and IsolationForest give a class only. Pass
    `vec` (precomputed) when the bundle is stateful (trend) and you own the
    stream-ordered TrendState; otherwise model_vector() is used statelessly."""
    if bundle is None:
        return threshold_tier1(rec), None, None
    x = [vec if vec is not None else model_vector(bundle, rec)]
    model = bundle["model"]
    if bundle["kind"] == "iforest":
        return (2 if model.predict(x)[0] == -1 else 0), None, None
    cls = int(model.predict(x)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        # map proba (ordered by model.classes_) onto class indices 0/1/2
        by_class = {int(c): float(p) for c, p in zip(model.classes_, proba)}
        vec = [by_class.get(i, 0.0) for i in range(3)]
        return cls, by_class.get(cls, max(vec)), vec
    return cls, None, None


def _pkt(rec: dict, field: str) -> float:
    return float(rec.get("pkt", {}).get(field, 0) or 0)


def signal_lost(rec: dict) -> bool:
    """A total-dropout interval: NO packets received at all. When the link fades
    out the `ant[]` array is empty (no RSSI to report) and every packet counter
    is zero -- and crucially pkt.lost stays 0 *because* nothing arrived to be
    counted lost. So the real outage signal is "received nothing", read from the
    packet counts, not from pkt.lost."""
    pkt = rec.get("pkt", {})
    received = pkt.get("all", pkt.get("uniq", None))
    if received is not None:
        return float(received or 0) == 0
    return not rec.get("ant")  # no pkt block -> fall back to empty antenna list


def signal_lost_kind(pre_rssi: float | None) -> str:
    """Classify a dropout by the last-known RSSI before it went dark. Strong ->
    tx_dropout (TX quiet / interference / reboot, NOT a range problem); weak ->
    rf_fade (faded out of range). The caller supplies pre_rssi via a forward-fill
    of the last record that still had a signal -- the flagged record is often
    already mid-outage (silent), so the RSSI must come from BEFORE it."""
    if pre_rssi is None:
        return "unknown"
    return "tx_dropout" if pre_rssi >= TX_DROPOUT_RSSI else "rf_fade"


def measured_outcome(after: list[dict], classes_after: list[int], cls_now: int,
                     horizon: int, pre_rssi: float | None = None) -> dict:
    """Objective outcome read from the records AFTER the flagged one.

    Pure telemetry arithmetic -- no model, no LLM. This is the ground truth the
    labeler's verdict is later checked against."""
    lost_sum = sum(_pkt(r, "lost") for r in after)
    fec_sum = sum(_pkt(r, "fec_recovered") for r in after)
    dec_err = sum(_pkt(r, "dec_err") for r in after)
    dropouts = sum(1 for r in after if signal_lost(r))
    worst = max(classes_after) if classes_after else cls_now
    ends_ok = bool(classes_after) and classes_after[-1] == 0
    out = {
        "horizon": horizon, "records_seen": len(after),
        "lost_sum": lost_sum, "fec_recovered_sum": fec_sum, "dec_err_sum": dec_err,
        "dropout_records": dropouts,
        "worst_class_after": worst, "ends_ok": ends_ok,
    }
    if not after:
        out["outcome"] = "unknown_no_lookahead"
    elif dropouts > 0:
        out["outcome"] = "signal_lost"          # received nothing -> true outage
        out["signal_lost_kind"] = signal_lost_kind(pre_rssi)  # tx_dropout|rf_fade|unknown
        out["pre_outage_rssi"] = pre_rssi
    elif lost_sum > 0:
        out["outcome"] = "loss_occurred"
    elif worst > cls_now:
        out["outcome"] = "worsened"
    elif cls_now != 0 and ends_ok:
        out["outcome"] = "recovered"
    elif cls_now != 0:
        out["outcome"] = "persisted"
    else:
        out["outcome"] = "stable_ok"
    return out


def flag(cls: int, conf, proba, conf_thr: float, margin: float):
    """Decide whether to enqueue, and why. Returns a reason str or None."""
    if cls != 0:
        return "tier1_class"
    if conf is not None and conf < conf_thr:
        return "low_confidence"
    if proba is not None:
        top2 = sorted(proba, reverse=True)[:2]
        if len(top2) == 2 and (top2[0] - top2[1]) < margin:
            return "boundary"
    return None


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
    ap.add_argument("--out", default="telemetry/loop/review_queue.jsonl")
    ap.add_argument("--context", type=int, default=5,
                    help="records of context kept on each side of the event")
    ap.add_argument("--outcome-window", type=int, default=20,
                    help="records AFTER the event used to measure the real outcome")
    ap.add_argument("--conf", type=float, default=0.85,
                    help="flag Tier-1 calls below this confidence (active learning)")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="flag when the top-two class probabilities are this close")
    args = ap.parse_args()

    bundle = load_tier1(args.model_path)
    src = "trained " + bundle["kind"] if bundle else "threshold fallback"
    proba_note = "" if (bundle and bundle["kind"] == "tree") else \
        "  (no predict_proba -> class-only flagging)"
    print(f"# Tier-1: {src}{proba_note}", file=sys.stderr, flush=True)

    # one pass to score everything (we need lookahead for measured outcomes);
    # held in memory -- fine for a session's worth of 10 Hz telemetry.
    vectorizer = make_vectorizer(bundle)
    scored = []
    for rec in stream_records(args):
        vec = vectorizer(rec) if vectorizer else None
        cls, conf, proba = tier1_with_proba(bundle, rec, vec=vec)
        # hard override: a no-packets interval is unambiguously critical, even if
        # the model maps the all-zero feature vector to 'ok' (the synthetic-
        # trained tree has never seen a real dropout). Don't let an outage slip
        # past the queue.
        if signal_lost(rec):
            cls, conf, proba = 2, 1.0, [0.0, 0.0, 1.0]
        scored.append({"rec": rec, "cls": cls, "conf": conf, "proba": proba})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_total = len(scored)
    classes = [s["cls"] for s in scored]
    # forward-fill the last record that still had a signal -> the RSSI just before
    # any dropout (the flagged record is often already silent, so we can't read it
    # from the record itself). Used to tag tx_dropout vs rf_fade.
    last_rssi = [None] * n_total
    seen = None
    for j in range(n_total):
        r = best_chain(scored[j]["rec"])["rssi"]
        if r is not None:
            seen = r
        last_rssi[j] = seen
    reasons: dict[str, int] = {}
    written = 0
    with open(args.out, "w") as fh:
        for i, s in enumerate(scored):
            reason = flag(s["cls"], s["conf"], s["proba"], args.conf, args.margin)
            if reason is None:
                continue
            lo = max(0, i - args.context)
            hi = min(n_total, i + args.context + 1)
            after = scored[i + 1: i + 1 + args.outcome_window]
            entry = {
                "seq": s["rec"].get("seq"),
                "ts_ms": s["rec"].get("ts_ms"),
                "tier1": {
                    "class": s["cls"], "label": LABELS[s["cls"]],
                    "confidence": s["conf"], "proba": s["proba"],
                    "flag_reason": reason,
                },
                "measured_outcome": measured_outcome(
                    [a["rec"] for a in after], [a["cls"] for a in after],
                    s["cls"], args.outcome_window, pre_rssi=last_rssi[i]),
                "window": [flatten(scored[j]["rec"]) for j in range(lo, hi)],
                "window_classes": classes[lo:hi],
            }
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
            written += 1
            reasons[reason] = reasons.get(reason, 0) + 1

    pct = (written / n_total * 100) if n_total else 0
    print(f"# {n_total} records -> {written} queued ({pct:.1f}%) by reason: "
          f"{reasons or '{}'}\n# wrote {args.out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
