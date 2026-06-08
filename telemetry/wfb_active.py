#!/usr/bin/env python3
"""Active-learning loop over the data-session store (see DATASTORE.md, phase 5).

Closes the capture -> score -> label -> retrain loop, all against the SQLite
store, reusing the existing Tier-1 trainer and Tier-2 Gemma escalation:

  train   build a labeled dataset from the store (rule-bootstrap base, with human
          `state` labels overriding where present) and train Tier-1
          (train_tier1.train_and_export -> telemetry/model/).
  score   run a trained model (or the threshold fallback) over chosen sessions and
          write the results into `predictions` under a --model-ver, so the web UI
          overlay updates and you can diff model versions. Re-scoring the same
          version replaces it.
  gemma   for each contiguous degraded/critical span (per a model version), call
          Tier-2 Gemma (think=false) for a human-readable cause/action and write
          it onto the span's prediction as `tier2_note`; optionally suggest
          candidate `labels` (author=model:<ver>) for a human to confirm.

The human labels created in the web UI feed `train`; `score` repopulates the
overlay; `gemma` mines candidates — that is the active-learning cycle.

    python3 telemetry/wfb_active.py train  --sessions all --model-ver tier1-tree
    python3 telemetry/wfb_active.py score  --sessions all --model-ver tier1-tree
    python3 telemetry/wfb_active.py gemma  --session 1 --model-ver tier1-tree --dry-run
"""
from __future__ import annotations

import argparse
import json

import wfb_pipeline as pipeline
import wfb_store as store

# human `state` label values -> Tier-1 class
STATE_MAP = {"ok": 0, "healthy": 0, "0": 0,
             "degraded": 1, "degrade": 1, "1": 1,
             "critical": 2, "crit": 2, "2": 2}
CLASS_NAME = {0: "ok", 1: "degraded", 2: "critical"}


def parse_state(v) -> int | None:
    return STATE_MAP.get(str(v).strip().lower())


def resolve_sessions(conn, spec: str) -> list[int]:
    if spec == "all":
        return store.session_ids(conn)
    return [int(s) for s in spec.split(",") if s.strip()]


def state_spans(conn, sid: int) -> list[tuple[int, int, int]]:
    """Human `state` labels for a session as (t0_ms, t1_ms, class)."""
    out = []
    for l in store.list_labels(conn, sid):
        if l["kind"] == "state":
            v = parse_state(l["value"])
            if v is not None:
                out.append((l["t0_ms"], l["t1_ms"], v))
    return out


def labeled_records(conn, sids: list[int], base: str) -> tuple[list[dict], dict]:
    """Build training records from the store. Each record's `label` is the human
    `state` label covering its timestamp if any, else the rule-bootstrap base
    (base='rule') — or the record is skipped (base='none')."""
    out: list[dict] = []
    stats = {"human": 0, "base": 0, "skipped": 0}
    for sid in sids:
        spans = state_spans(conn, sid)
        for r in store.get_records(conn, sid):
            rec = json.loads(r["raw_json"])
            ts = r["ts_ms"]
            lbl = None
            for t0, t1, v in spans:           # later spans win on overlap
                if t0 <= ts <= t1:
                    lbl = v
            if lbl is not None:
                stats["human"] += 1
            elif base == "rule":
                lbl = pipeline.threshold_tier1(rec)
                stats["base"] += 1
            else:
                stats["skipped"] += 1
                continue
            rec["label"] = lbl
            out.append(rec)
    return out, stats


def contiguous_spans(rows) -> list[tuple[int, int, list]]:
    """Runs of records with tier1_state > 0 -> (start_idx, end_idx, rows[])."""
    spans = []
    i = 0
    while i < len(rows):
        if (rows[i]["tier1_state"] or 0) > 0:
            j = i
            while j + 1 < len(rows) and (rows[j + 1]["tier1_state"] or 0) > 0:
                j += 1
            spans.append((i, j, rows[i:j + 1]))
            i = j + 1
        else:
            i += 1
    return spans


# --- commands --------------------------------------------------------------

def cmd_train(conn, args):
    import train_tier1
    sids = resolve_sessions(conn, args.sessions)
    records, stats = labeled_records(conn, sids, args.base)
    print(f"# dataset: {len(records)} records from sessions {sids} "
          f"(human={stats['human']} base={stats['base']} skipped={stats['skipped']})")
    res = train_tier1.train_and_export(records, args.outdir, max_depth=args.max_depth,
                                       unsupervised=args.unsupervised)
    print(res["metrics"])
    print(f"# artifacts in {args.outdir}/  (now: wfb_active.py score --model-ver {args.model_ver})")


def cmd_score(conn, args):
    bundle = pipeline.load_tier1(args.model_path)
    src = ("trained " + bundle["kind"]) if bundle else "threshold fallback (no model trained)"
    print(f"# scoring with {src} -> predictions model_ver={args.model_ver!r}")
    sids = resolve_sessions(conn, args.sessions)
    total = {0: 0, 1: 0, 2: 0}
    for sid in sids:
        store.delete_predictions(conn, args.model_ver, sid)  # replace on re-score
        n = 0
        for r in store.get_records(conn, sid):
            state = pipeline.tier1_predict(bundle, json.loads(r["raw_json"]))
            store.add_prediction(conn, r["id"], args.model_ver, tier1_state=state)
            total[state] += 1
            n += 1
        conn.commit()
        print(f"  session {sid}: scored {n}")
    print(f"# done: ok={total[0]} degraded={total[1]} critical={total[2]} "
          f"(UI overlay now shows {args.model_ver!r})")


def cmd_gemma(conn, args):
    sids = resolve_sessions(conn, args.sessions) if args.sessions else [args.session]
    for sid in sids:
        rows = store.get_scored_records(conn, sid, args.model_ver)
        spans = contiguous_spans(rows)
        print(f"# session {sid}: {len(spans)} flagged span(s) under {args.model_ver!r}"
              + ("  [dry-run]" if args.dry_run else ""))
        for k, (i0, i1, span) in enumerate(spans):
            if args.max_spans and k >= args.max_spans:
                print(f"  … stopping at --max-spans={args.max_spans}")
                break
            worst = max(r["tier1_state"] or 0 for r in span)
            window = [json.loads(r["raw_json"]) for r in span][-args.context:]
            if args.dry_run:
                note = f"(dry-run) {len(span)} rec span, worst={CLASS_NAME[worst]}"
            else:
                note = pipeline.tier2_explain(args.llm, window)
            store.set_tier2_note(conn, span[0]["pred_id"], note)
            print(f"  span {k} [{span[0]['ts_ms']}–{span[-1]['ts_ms']}ms "
                  f"{CLASS_NAME[worst]}] -> {note}")
            if args.suggest_labels:
                store.add_label(conn, sid, span[0]["ts_ms"], span[-1]["ts_ms"],
                                "state", str(worst), f"model:{args.model_ver}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=store.DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train Tier-1 from store labels")
    tr.add_argument("--sessions", default="all", help="'all' or comma ids")
    tr.add_argument("--base", choices=["rule", "none"], default="rule",
                    help="label for records with no human label (rule-bootstrap or skip)")
    tr.add_argument("--outdir", default="telemetry/model")
    tr.add_argument("--max-depth", type=int, default=5)
    tr.add_argument("--unsupervised", action="store_true")
    tr.add_argument("--model-ver", default="tier1-tree", help="for the follow-up score step")

    sc = sub.add_parser("score", help="write predictions for sessions")
    sc.add_argument("--sessions", default="all", help="'all' or comma ids")
    sc.add_argument("--model-path", default="telemetry/model/tier1.pkl")
    sc.add_argument("--model-ver", default="tier1-tree")

    gm = sub.add_parser("gemma", help="Tier-2 explanations for flagged spans")
    gm.add_argument("--session", type=int, help="single session id")
    gm.add_argument("--sessions", default="", help="'all' or comma ids (overrides --session)")
    gm.add_argument("--model-ver", default="tier1-tree")
    gm.add_argument("--llm", default="gemma4:26b-a4b-it-qat")
    gm.add_argument("--context", type=int, default=10, help="records sent to Tier-2 per span")
    gm.add_argument("--max-spans", type=int, default=0, help="cap spans per session (0=all)")
    gm.add_argument("--dry-run", action="store_true", help="don't call Ollama; write placeholder notes")
    gm.add_argument("--suggest-labels", action="store_true",
                    help="also insert candidate state labels (author=model:<ver>) for review")

    args = ap.parse_args()
    conn = store.connect(args.db)
    store.init_db(conn)
    {"train": cmd_train, "score": cmd_score, "gemma": cmd_gemma}[args.cmd](conn, args)
    conn.close()


if __name__ == "__main__":
    main()
