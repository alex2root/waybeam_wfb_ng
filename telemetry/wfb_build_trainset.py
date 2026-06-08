#!/usr/bin/env python3
"""Closed-loop step 3: build an objectively-labeled training set from a capture.

The training label is derived from the MEASURED outcome of each record's t+K
future (derive_label / OUTCOME_TO_LABEL), NOT from the LLM -- so labels are
objective and reproducible. This is the "accumulate real labels -> retrain"
half of the loop: feed its output straight to train_tier1.py.

Every record gets a label (not just flagged ones), so the trainer sees the full
ok/degraded/critical distribution:
  * signal_lost / loss_occurred / worsened  -> critical (2)   [predictive policy]
  * persisted / recovered                   -> degraded (1)
  * stable_ok                               -> ok (0)
The last --outcome-window records have no lookahead and are dropped.

    python3 telemetry/wfb_build_trainset.py --input telemetry/walk2_wfb.jsonl \
        --out telemetry/loop/trainset_walk2.jsonl
    python3 telemetry/train_tier1.py --input telemetry/loop/trainset_walk2.jsonl ...
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

from wfb_schema import best_chain
from wfb_pipeline import load_tier1
from wfb_review_queue import tier1_with_proba, signal_lost, measured_outcome
from wfb_gemma_verify import derive_label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="capture JSONL (one or more)",
                    nargs="+")
    ap.add_argument("--out", default="telemetry/loop/trainset.jsonl")
    ap.add_argument("--model-path", default="telemetry/model/tier1.pkl",
                    help="model used only to compute the class trajectory for "
                         "worsened/recovered outcomes (loss/signal_lost are "
                         "model-independent)")
    ap.add_argument("--outcome-window", type=int, default=20)
    args = ap.parse_args()

    bundle = load_tier1(args.model_path)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    labels = collections.Counter()
    kept = dropped = 0
    with open(args.out, "w") as fh:
        for path in args.input:
            recs = [json.loads(l) for l in open(path) if l.strip()]
            # classify every record + forward-fill last-known RSSI (for the
            # signal_lost tx_dropout/rf_fade tag), exactly like the review queue.
            scored = []
            for r in recs:
                cls, _, _ = tier1_with_proba(bundle, r)
                if signal_lost(r):
                    cls = 2
                scored.append({"rec": r, "cls": cls})
            last_rssi, seen = [], None
            for s in scored:
                v = best_chain(s["rec"])["rssi"]
                if v is not None:
                    seen = v
                last_rssi.append(seen)

            n = len(scored)
            for i, s in enumerate(scored):
                after = scored[i + 1: i + 1 + args.outcome_window]
                mo = measured_outcome([a["rec"] for a in after],
                                      [a["cls"] for a in after], s["cls"],
                                      args.outcome_window, pre_rssi=last_rssi[i])
                label = derive_label(mo)
                if label is None:          # no lookahead (tail) -> can't label
                    dropped += 1
                    continue
                rec = dict(s["rec"])
                rec["label"] = label
                rec["_outcome"] = mo["outcome"]   # provenance (train_tier1 drops _*)
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
                labels[label] += 1
                kept += 1

    print(f"# {kept} labeled, {dropped} dropped (no lookahead)", file=sys.stderr)
    print(f"# label distribution: ok={labels[0]} degraded={labels[1]} "
          f"critical={labels[2]}", file=sys.stderr)
    print(f"# wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
