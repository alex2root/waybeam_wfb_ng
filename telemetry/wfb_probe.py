#!/usr/bin/env python3
"""Throughput probe: how fast can local Gemma analyse wfb_rx -Y telemetry?

Answers the practical question "what telemetry data rate can the LLM keep up
with?" by measuring, on YOUR box and YOUR data format, the two numbers that
matter and that the repo docs don't yet have:

  * prefill rate  (tok/s) -- how fast it ingests the telemetry window
  * decode  rate  (tok/s) -- how fast it writes the verdict (~21.6 measured)

It sweeps the window size (records packed per prompt), runs N trials each at
think=false with a clamped tiny output, and reports sustained records/sec.

Ollama returns exact token counts and nanosecond timings on /api/generate
(prompt_eval_count/prompt_eval_duration, eval_count/eval_duration), so these are
real measurements, not estimates -- except in --dry-run, which estimates offline.

    python3 telemetry/wfb_probe.py --input telemetry/sample_wfb.jsonl \
        --windows 1,5,20,60 --trials 3
    python3 telemetry/wfb_probe.py --dry-run            # offline logic check
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

from wfb_schema import flatten, load_jsonl

OLLAMA = "http://localhost:11434/api/generate"
SYSTEM = ("You are a Wi-Fi link health classifier for an FPV video link. "
          "Given recent wfb_rx stats, reply with ONLY compact JSON: "
          '{"state":"ok|degraded|critical","action":"hold|lower_bitrate|switch_channel"}.')


def build_prompt(window: list[dict]) -> str:
    # compact, numeric-only view keeps input tokens down
    rows = [flatten(r) for r in window]
    body = json.dumps(rows, separators=(",", ":"))
    return f"{SYSTEM}\nstats={body}\nverdict:"


def call_ollama(model: str, prompt: str, num_predict: int, num_ctx: int) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "think": False,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 1.0,
                    "num_ctx": num_ctx},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
    wall = time.perf_counter() - t0
    pe_n = body.get("prompt_eval_count", 0)
    pe_d = body.get("prompt_eval_duration", 0) or 1
    ev_n = body.get("eval_count", 0)
    ev_d = body.get("eval_duration", 0) or 1
    # The prompt overflowed the context window: Ollama truncates it (in_tok pins
    # at num_ctx) and the model emits junk. Such a row is meaningless -- flag it
    # so we don't report a bogus decode rate from a 1-token, ~1ns "eval".
    truncated = pe_n >= num_ctx or ev_n < 2
    return {
        "wall": wall,
        "in_tok": pe_n,
        "out_tok": ev_n,
        "prefill_tps": pe_n / (pe_d / 1e9),
        "decode_tps": (ev_n / (ev_d / 1e9)) if ev_n >= 2 else float("nan"),
        "truncated": truncated,
    }


def dry_call(prompt: str, num_predict: int) -> dict:
    # offline estimate: ~4 chars/token; documented decode 21.6 tok/s;
    # assume prefill ~6x decode as a placeholder (REPLACE with a real run).
    in_tok = max(1, len(prompt) // 4)
    out_tok = num_predict
    decode_tps, prefill_tps = 21.6, 130.0
    wall = in_tok / prefill_tps + out_tok / decode_tps
    return {"wall": wall, "in_tok": in_tok, "out_tok": out_tok,
            "prefill_tps": prefill_tps, "decode_tps": decode_tps,
            "truncated": False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="telemetry/sample_wfb.jsonl")
    ap.add_argument("--model", default="gemma4:26b-a4b-it-qat")
    ap.add_argument("--windows", default="1,5,20,60", help="records per prompt, comma list")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=40, help="cap output tokens")
    ap.add_argument("--num-ctx", type=int, default=8192, help="model context window")
    ap.add_argument("--dry-run", action="store_true", help="no Ollama; offline estimate")
    args = ap.parse_args()

    records = load_jsonl(args.input)
    windows = [int(w) for w in args.windows.split(",")]
    call = (lambda p: dry_call(p, args.num_predict)) if args.dry_run \
        else (lambda p: call_ollama(args.model, p, args.num_predict, args.num_ctx))

    if args.dry_run:
        print("** DRY RUN -- numbers are ESTIMATES (prefill assumed 130 tok/s). "
              "Run for real against Ollama to measure. **")
    else:
        # warm up: first call pays model-load cost and would skew p95.
        call(build_prompt(records[:1]))
    print(f"\nmodel={args.model}  trials={args.trials}  "
          f"num_predict={args.num_predict}  num_ctx={args.num_ctx}\n")
    hdr = f"{'window':>6} {'in_tok':>7} {'prefill':>9} {'decode':>8} {'lat_p50':>8} {'lat_p95':>8} {'rec/s':>7}"
    print(hdr)
    print("-" * len(hdr))

    any_truncated = False
    for w in windows:
        if w > len(records):
            continue
        lats, ins, pfs, dcs = [], [], [], []
        trunc = False
        for t in range(args.trials):
            start = (t * w) % max(1, len(records) - w)
            window = records[start:start + w]
            m = call(build_prompt(window))
            lats.append(m["wall"]); ins.append(m["in_tok"])
            pfs.append(m["prefill_tps"]); dcs.append(m["decode_tps"])
            trunc = trunc or m["truncated"]
        p50 = statistics.median(lats)
        p95 = max(lats) if len(lats) < 3 else statistics.quantiles(lats, n=20)[-1]
        rec_s = w / p50 if p50 else 0
        decode = statistics.mean(dcs)
        flag = "  <- TRUNCATED (prompt > num_ctx)" if trunc else ""
        any_truncated = any_truncated or trunc
        print(f"{w:>6} {int(statistics.mean(ins)):>7} {statistics.mean(pfs):>9.1f} "
              f"{decode:>8.1f} {p50:>8.2f} {p95:>8.2f} {rec_s:>7.1f}{flag}")

    print("\nrec/s = records the LLM can sustain per second at that window "
          "(single stream, think=false). Compare to your real wfb_rx -Y rate.")
    if any_truncated:
        print("TRUNCATED rows overflowed the context window -- the prompt was cut "
              "and decode is unreliable; ignore them or raise --num-ctx / shrink the window.")


if __name__ == "__main__":
    main()
