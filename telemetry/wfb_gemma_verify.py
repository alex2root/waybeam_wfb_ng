#!/usr/bin/env python3
"""Closed-loop step 2: the local Gemma labels/verifies the review queue.

Tier-2 in the closed loop. Reads the review_queue.jsonl produced by
wfb_review_queue.py and, for each flagged event, asks the local Gemma to:
  * confirm or correct the Tier-1 class,
  * give a short human-readable cause,
  * say whether the Tier-1 guess is consistent with the MEASURED outcome
    (the objective telemetry fact already in the entry).

This runs OFFLINE, not in the control path. Gemma is the FAST LABELER here, never
the controller and NOT the deep reasoner -- that role belongs to the Tier-3 Opus
supervisor. It defaults to think=false: on real data think=true generated huge
hidden reasoning chains that overflowed num_predict (only ~14% of events returned
parseable JSON) and the sustained heavy decode overloaded the box. think=false is
~7x lighter (~3 s/event), reliable, and the concise "cause" field already carries
the justification. --think remains available for spot use but is heavy/fragile.

Output: labeled_real.jsonl = each queue entry + Gemma's verdict. That file is
what the Tier-3 supervisor (.claude/agents/wfb-loop-supervisor) audits and what
periodic retraining draws real labels from.

    python3 telemetry/wfb_gemma_verify.py            # whole queue, think=false
    python3 telemetry/wfb_gemma_verify.py --limit 5  # quick smoke
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

OLLAMA = "http://localhost:11434/api/generate"

SYSTEM = (
    "You are an FPV Wi-Fi (wfb-ng) link-telemetry analyst LABELING training data. "
    "You are given a short window of wfb_rx stats, a Tier-1 model's class guess, "
    "and the MEASURED outcome computed from the telemetry that FOLLOWED the event "
    "(this measured outcome is ground truth -- trust it over the Tier-1 guess). "
    "Decide the correct link class and explain the cause. "
    "The link controller acts on EFFECTIVE RSSI = RSSI minus a loss penalty (lost "
    "packets hurt; FEC-recovered packets do NOT) and on packet loss. It IGNORES "
    "SNR. Justify your cause with RSSI / effective-RSSI / loss / FEC terms, NOT SNR. "
    "Classes: ok=healthy; degraded=recoverable stress (FEC working, low/falling "
    "RSSI, no real loss); critical=actual packet loss, total signal loss (no "
    "packets received), OR a degraded state whose measured outcome is 'worsened' "
    "(trending into failure) -- label such pre-loss worsening CRITICAL so the "
    "controller can act early. Set outcome_consistent=false only if your label "
    "genuinely conflicts with the measured outcome, and if so reconsider the label. "
    "Reply with ONLY compact JSON, no markdown, no prose:\n"
    '{"label":"ok|degraded|critical","label_int":0|1|2,"cause":"<short>",'
    '"tier1_agrees":true|false,"outcome_consistent":true|false,'
    '"confidence":"low|med|high"}'
)


# Predictive labeling policy (user decision 2026-06-07): the TRAINING label is
# derived objectively from the MEASURED outcome, so it never depends on LLM mood.
# Gemma explains/cross-checks; this is the ground-truth label. Pre-loss worsening
# is CRITICAL (act early), matching the controller's fast react-down.
OUTCOME_TO_LABEL = {
    "signal_lost": 2, "loss_occurred": 2, "worsened": 2,  # critical
    "persisted": 1,                                        # degraded
    "recovered": 1, "stable_ok": 0,                        # eased / fine
    "unknown_no_lookahead": None,                          # can't label
}


def derive_label(measured_outcome: dict):
    """Outcome-derived training label (0/1/2) per the predictive policy, or None
    when the outcome window is too short to judge."""
    return OUTCOME_TO_LABEL.get(measured_outcome.get("outcome"))

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def compact_row(flat: dict) -> dict:
    """Reduce one flattened record (~54 keys) to the signals that decide link
    health, so the window prompt stays small enough to fit Gemma's context.

    RSSI is primary (always present); SNR is reported for the SAME best-RSSI chain
    when the driver provides it (mirrors wfb_schema.best_chain), plus the implied
    noise floor (RSSI-SNR). Picking rssi and snr from the same antenna matters --
    taking max(rssi) and max(snr) separately can mix two different chains."""
    ants: dict[str, dict] = {}
    for k, v in flat.items():
        if k.startswith("ant.") and (k.endswith(".rssi.avg") or k.endswith(".snr.avg")):
            parts = k.split(".")          # ant.<id>.rssi.avg
            ants.setdefault(parts[1], {})[parts[2]] = v
    rssi = snr = nf = None
    with_rssi = {a: d for a, d in ants.items() if "rssi" in d}
    if with_rssi:
        best = max(with_rssi.values(), key=lambda d: d["rssi"])
        rssi = best["rssi"]
        snr = best.get("snr")
        nf = (rssi - snr) if snr is not None else None
    return {
        "rssi": rssi, "snr": snr, "noise_floor": nf,
        "snr_available": snr is not None,
        "lost": flat.get("pkt.lost", 0),
        "fec": flat.get("pkt.fec_recovered", 0),
        "all": flat.get("pkt.all", 0),
    }


def extract_json(text: str) -> dict | None:
    """Gemma sometimes wraps JSON in ```json fences or adds stray prose -- pull
    out the first {...} blob and parse it."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def mem_available_mb() -> int:
    """Free RAM from /proc/meminfo (MemAvailable, in MiB)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1  # unknown -> don't block


def ask_gemma(model: str, entry: dict, think: bool, timeout: float,
              num_ctx: int, num_gpu: int) -> tuple[dict | None, str]:
    payload_view = {
        "tier1_guess": entry["tier1"]["label"],
        "tier1_confidence": entry["tier1"].get("confidence"),
        "flag_reason": entry["tier1"]["flag_reason"],
        "measured_outcome": entry["measured_outcome"],
        "window_classes": entry.get("window_classes"),
        "window": [compact_row(r) for r in entry["window"]],
    }
    prompt = (f"{SYSTEM}\nevent={json.dumps(payload_view, separators=(',',':'))}\n"
              "verdict:")
    # num_ctx must cover the (large) window prompt -- otherwise Ollama silently
    # truncates and Gemma emits one garbage token (same failure as the probe).
    # think=true emits reasoning tokens that ALSO count against num_predict, so
    # the budget must cover thinking + the JSON answer or `response` comes back
    # empty (done_reason=length with all tokens spent on `thinking`).
    num_predict = 4096 if think else 256  # think chains are long; spot use only
    # num_gpu=0 forces CPU-only: this box's Radeon 760M has just 3 GB VRAM, so
    # offloading the ~15 GB model spills it into GTT (shared system RAM) on top of
    # the CPU copy -> doubles the footprint past 27 GB RAM -> OOM hang. CPU-only
    # keeps one copy in RAM; per CLAUDE.md the workload is memory-bandwidth bound
    # so the iGPU barely helps anyway.
    payload = {"model": model, "prompt": prompt, "think": think, "stream": False,
               "keep_alive": "10m",
               "options": {"num_predict": num_predict, "temperature": 0.3,
                           "num_ctx": num_ctx, "num_gpu": num_gpu}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except Exception as e:
        return None, f"ERROR: {e}"
    raw = (body.get("response") or "").strip()
    # On a long reasoning chain the response can come back empty (thinking spent
    # the whole num_predict budget); the verdict JSON is often still inside the
    # thinking text, so fall back to scanning that before giving up.
    verdict = extract_json(raw)
    if verdict is None:
        thinking = (body.get("thinking") or "").strip()
        verdict = extract_json(thinking)
        if verdict is not None:
            raw = raw or thinking
    return verdict, raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default="telemetry/loop/review_queue.jsonl")
    ap.add_argument("--out", default="telemetry/loop/labeled_real.jsonl")
    ap.add_argument("--llm", default="gemma4:26b-a4b-it-qat")
    ap.add_argument("--limit", type=int, default=0, help="label only the first N (0=all)")
    ap.add_argument("--think", dest="think", action="store_true", default=False,
                    help="enable Ollama reasoning (HEAVY + fragile at scale; "
                         "overflows num_predict on most events -- spot use only)")
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="default: fast, light, reliable JSON labeling")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--num-ctx", type=int, default=2048,
                    help="context window; the compact event prompt is ~750 tok, "
                         "so 2048 fits with a small KV cache (lower RAM)")
    ap.add_argument("--num-gpu", type=int, default=0,
                    help="GPU layers; 0 = CPU-only (default, avoids iGPU GTT OOM "
                         "on this 3 GB-VRAM box). Set >0 only if you know it fits")
    ap.add_argument("--min-free-mb", type=int, default=2500,
                    help="abort before a call if MemAvailable drops below this "
                         "(OOM guard -- the box hard-hangs on out-of-memory)")
    args = ap.parse_args()

    entries = []
    with open(args.queue) as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if args.limit:
        entries = entries[: args.limit]

    mode = "CPU-only" if args.num_gpu == 0 else f"num_gpu={args.num_gpu}"
    print(f"# labeling {len(entries)} events with {args.llm} "
          f"(think={args.think}, {mode}, num_ctx={args.num_ctx})",
          file=sys.stderr, flush=True)

    ok, bad, agree, consistent = 0, 0, 0, 0
    with open(args.out, "w") as fh:
        for i, entry in enumerate(entries, 1):
            free = mem_available_mb()
            if 0 <= free < args.min_free_mb:
                print(f"# ABORT: only {free} MiB free (< {args.min_free_mb}); "
                      f"stopping before OOM. {i-1}/{len(entries)} labeled so far.",
                      file=sys.stderr, flush=True)
                break
            verdict, raw = ask_gemma(args.llm, entry, args.think, args.timeout,
                                     args.num_ctx, args.num_gpu)
            entry["gemma"] = verdict
            entry["gemma_raw"] = raw
            # objective training label from the measured outcome (predictive
            # policy); Gemma is the cross-check, not the label source.
            entry["label"] = derive_label(entry.get("measured_outcome", {}))
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
            fh.flush()
            if verdict is None:
                bad += 1
                tag = "PARSE-FAIL"
            else:
                ok += 1
                agree += bool(verdict.get("tier1_agrees"))
                consistent += bool(verdict.get("outcome_consistent"))
                tag = (f"{verdict.get('label','?'):8} "
                       f"agrees={verdict.get('tier1_agrees')} "
                       f"consistent={verdict.get('outcome_consistent')}")
            print(f"[{i}/{len(entries)}] seq={entry.get('seq')} "
                  f"tier1={entry['tier1']['label']:8} -> {tag}",
                  file=sys.stderr, flush=True)

    print(f"# done: {ok} labeled, {bad} parse-fail | "
          f"tier1_agrees={agree}/{ok} outcome_consistent={consistent}/{ok}\n"
          f"# wrote {args.out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
