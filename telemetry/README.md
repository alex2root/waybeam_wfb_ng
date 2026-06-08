# Live wfb telemetry analysis — Gemma throughput + Tier-1 trained model

Tooling to answer two questions for **`waybeam_wfb_ng wfb_rx -Y`** JSON telemetry:

1. **What data rate can local Gemma keep up with?** → measure it (`wfb_probe.py`).
2. **Am I better served by a trained inference model?** → almost certainly yes for
   the firehose; build/train/export one (`train_tier1.py`) and run the **hybrid**
   (`wfb_pipeline.py`). The trained model is exportable to dependency-free **C** for
   the **SigmaStar SSC338Q** (Cortex-A7).

> The wfb_rx `-Y` field layout varies by build, so `wfb_schema.py` is
> **schema-agnostic**: it flattens any JSON object to `{dot.path: float}`. Drop a
> real capture in and everything works unchanged — no field names are hard-coded.

### Local Gemma (Tier-2) dependency

This tooling moved here from the standalone **`gemma4-local`** repo (now a sibling
submodule of `waybeam-coordination`: the **Gemma inference box**). The coupling is a
loose **HTTP** one, not a code dependency:

- **Tier-2 only** — `wfb_probe.py`, `wfb_pipeline.py`, and `wfb_gemma_verify.py` POST to
  Ollama at **`http://localhost:11434/api/generate`** (model `gemma4:26b-a4b-it-qat`,
  `think=false`). Start that box (see the `gemma4-local` submodule) before running them.
- **Tier-1 has NO Gemma dependency.** The trained decision tree + its C export
  (`tier1_model.c` + `tier1_trend.h`) are pure, dependency-free, and are what ships to the
  SSC338Q and feeds the `gs_supervisor` / `link_controller` loop. Training (`train_tier1.py`)
  needs only sklearn/m2cgen — no Ollama.

### Real captured schema (waybeam `wfb_rx_native -Y`, validated on hardware)

A real per-interval datagram (2 adapters × 2 antennas → 4 chains, ids `0,1,100,101`):

```json
{"ts_ms":2219190,"type":"rx_ant","ver":1,"seq":27,"interval_ms":100,
 "ant":[{"freq":5805,"mcs":3,"bw":20,"id":"0","pkts":105,
         "rssi":{"min":-11,"avg":-10,"max":-10},"snr":{"min":29,"avg":36,"max":45}}, ...x4],
 "pkt":{"all":209,"bytes":287104,"dec_err":0,"session":0,"data":209,"uniq":105,
        "fec_recovered":0,"lost":0,"bad":0,"outgoing":69,"outgoing_bytes":92172,
        "diversity":104,"mode_mismatch":0,"adapters":2}}
```

The per-antenna `ant` array is emitted in **map-iteration order, which is NOT
stable** between records, so `flatten()` keys those objects by their `id`
(`ant.0.rssi.avg`, `ant.101.snr.avg`, …) rather than by position — otherwise one
physical antenna's stats silently swap for another's. `gen_sample.py` emits this
exact shape, so a model trained on synthetic data is feature-compatible with a
real capture (verified: identical 54-column manifest).

**Capture cleanly** (the supervisor already tees stats to `127.0.0.1:6600`):

```bash
sudo tshark -i lo -f 'udp port 6600' -T fields -e data.data \
  | python3 -c 'import sys,json; [print(json.dumps(json.loads(bytes.fromhex(l)))) \
      for l in sys.stdin if l.strip()]' > telemetry/real_wfb.jsonl
```

Use `tshark` (binary), **not** `tcpdump -A` text scraping — the latter can merge
adjacent datagrams (e.g. a bogus 8-antenna record from two 4-antenna packets).
For a non-disruptive live feed into the pipeline, insert `wfb_stats_tee.py`
(point the supervisor's `stats_out` at it → forwards to `:6600` unchanged + taps
a copy for analysis).

## TL;DR — the throughput reality

A single-stream LLM at ~21.6 tok/s **cannot** keep up with a 10 Hz stats stream
when each record is analysed individually. **Measured live** against Ollama
(`gemma4:26b-a4b-it-qat`, `think=false`, `num_ctx=8192`, dev box):

| window (records/prompt) | input tok | decode tok/s | LLM sustained |
|:--:|:--:|:--:|:--:|
| 1  | ~750  | ~21.6 | ~0.2–0.9 rec/s |
| 5  | ~3450 | ~21.3 | (prefill cache-sensitive) |
| 10 | ~6850 | ~20.9 | ~0.5 rec/s (cold) |
| 20 | 8191 (truncated) | — | overflows 8192-ctx |

The **decode rate is the cache-immune invariant: ~21.6 tok/s**, dead-on the
documented figure. A verdict JSON is ~15–25 output tokens, so writing the answer
alone costs ~0.7–1.2 s **before any input** → an absolute ceiling near **1
verdict/sec**, worse cold. `wfb_rx -Y` typically emits **~10 records/s**, so the
LLM is a minimum of **~10× too slow** (25–50× in the cold/large-window case), and
`think=true` makes it ~10× worse again.

> Each flattened record is ~700 tokens, so a window of ≳12 records overflows the
> 8192-token context and gets truncated — `wfb_probe.py` now flags such rows
> instead of reporting a bogus decode rate. Prefill / `rec/s` numbers are sensitive
> to Ollama's KV prefix cache (shared SYSTEM prompt + overlapping windows), so the
> decode rate and the decode-floor latency are the numbers that bound the design.

A trained decision tree classifies the **same record in microseconds** on a
Cortex-A7 — roughly **6 orders of magnitude** faster.

**Conclusion:** don't put the LLM in the data path. Use the tiered design below.

## The architecture: Tier-1 firehose + Tier-2 explainer

```
 wfb_rx -Y ──► Tier-1 trained model ──► state per record (ok/degraded/critical)
  (~10 Hz)     tree in C, µs/record        │
              runs at full rate            │ only on degraded/critical (<1% of volume)
                                           ▼
                                  Tier-2 Gemma (think=false)
                                  human-readable cause + decision, rate-limited
```

- **Tier 1** carries 100% of the volume, makes the real-time decision (hold /
  lower bitrate / switch channel / RTH), and is what you deploy to the SSC338Q.
- **Tier 2** (Gemma) only fires on the rare flagged window to explain *why* and
  propose an action a human can read — exactly the role this repo's
  `ask_local_gemma(think=false)` already plays.

## Files

| file | what it does | deps |
|------|--------------|------|
| `wfb_schema.py` | schema-agnostic JSON→features flattener + stable manifest | stdlib |
| `gen_sample.py` | synthetic `wfb_rx -Y`-style JSONL (placeholder data) | stdlib |
| `wfb_probe.py`  | measures LLM prefill/decode/latency vs window size | stdlib |
| `train_tier1.py`| trains the Tier-1 tree, exports `tier1_model.c` | sklearn, m2cgen |
| `wfb_pipeline.py`| hybrid Tier-1 + Tier-2 runtime (jsonl or `--stdin`) | stdlib |
| `wfb_stats_tee.py`| UDP tee: forward stats to the real consumer + tap a copy for analysis | stdlib |
| `wfb_store.py`  | SQLite data-session store: import/query/label (see `DATASTORE.md`) | stdlib |
| `wfb_ingest.py` | live UDP ingester (tee `--tap` → store) | stdlib |
| `wfb_active.py` | active-learning loop: train from store labels / score / Gemma-fill | sklearn (train) |
| `webui/webapp.py`| Flask + uPlot browse/label UI with ML-label overlay | Flask, uPlot |

## Quick start

```bash
# 0. (training only) deps on the dev box; the SSC338Q needs none
pip install -r telemetry/requirements-telemetry.txt

# 1. make placeholder data (or capture real: wfb_rx -Y ... > telemetry/real_wfb.jsonl)
python3 telemetry/gen_sample.py -n 2000

# 2. how fast can Gemma actually go on YOUR box? (drop --dry-run to hit Ollama)
#    keep windows <=10: each record ~700 tok, so >=12 overflows the 8192 context
python3 telemetry/wfb_probe.py --input telemetry/sample_wfb.jsonl --windows 1,5,10

# 3. train Tier-1 + emit portable C
python3 telemetry/train_tier1.py --input telemetry/sample_wfb.jsonl --max-depth 5

# 4. run the hybrid (Tier-1 every record, Gemma only on flags)
python3 telemetry/wfb_pipeline.py --input telemetry/sample_wfb.jsonl
#   live:  wfb_rx -Y ... | python3 telemetry/wfb_pipeline.py --stdin
```

---

## How to train the Tier-1 inference model

A trained model needs **features**, **labels**, and a **model class**.

### 1. Features
`wfb_schema.flatten()` already turns each `wfb_rx -Y` record into numeric features
(per-antenna RSSI/SNR min/avg/max, FEC recovered, lost/bad packet counts, etc.).
`build_manifest()` fixes a stable column order saved to `model/features.json` — this
is the **input contract** the C model expects (pass `0.0` for any absent field).

> Data-hygiene note: if antenna keys vary (e.g. `5180:1:20` vs `5805:1:20`) they
> become separate columns and absent ones are filled `0.0`. For production, consider
> normalising to fixed roles (`ant0_*`, `ant1_*`) in a small pre-map so a `0.0 dBm`
> RSSI isn't mistaken for a strong signal. Add a few engineered deltas (RSSI slope,
> loss-rate over the last N) — trends predict link collapse better than instants.

### 2. Labels — the part that needs thought
Supervised learning needs a target. Options, easiest → best:

- **Rule-bootstrap (start here):** label from thresholds you already trust
  (what `gen_sample.classify()` does: lost/SNR/RSSI bands → ok/degraded/critical).
  The tree then *compacts and generalises* your rules into fast C. Good day-one model.
- **Outcome labels (better):** label a window by what happened *next* — e.g. if
  uncorrectable loss occurred within the next 1 s, label the earlier window
  "should have acted". This trains a **predictive** model, not just a reactive one.
- **Imitation:** log a human/heuristic pilot's actions alongside stats and train to
  mimic the decision directly.
- **Unsupervised (no labels):** `train_tier1.py --unsupervised` fits an
  IsolationForest that flags anomalies — useful when you can't label, or to *find*
  events worth escalating to Tier-2.

### 3. Model class — why a decision tree (not a neural net)
For tabular link stats on a Cortex-A7, a small **DecisionTree / GBDT** beats a NN:
interpretable thresholds, **µs inference**, **a few KB of RAM**, and it exports to
**plain C with zero runtime**. `train_tier1.py` keeps depth small (default 5) so the
tree stays tiny and readable; on the sample it hits ~0.98 holdout accuracy with 12
leaves → ~2 KB of C.

---

## Deploying to the SigmaStar SSC338Q

The SSC338Q is a dual **Cortex-A7** (ARMv7, NEON) with tens of MB of RAM, typically
running OpenIPC/Buildroot. The generated `tier1_model.c` is dependency-free, so:

```bash
# cross-compile with your OpenIPC/Buildroot toolchain (musl example)
arm-openipc-linux-musleabihf-gcc -O2 -mcpu=cortex-a7 -mfpu=neon \
    -c telemetry/model/tier1_model.c -o tier1_model.o
```

Then build a tiny daemon that: reads `wfb_rx -Y` (UDP/stdin) → flattens fields into
`double input[N]` in **features.json order** → calls `score(input, output)` →
`argmax(output)` = state → acts (e.g. command a bitrate/MCS change over the uplink).
`score()` is the generated entry point; one call is a handful of comparisons.

> **Do you even need the NPU?** No. The SSC338Q's IPU/NPU is for CV and has a
> proprietary SDK; for tabular log decisions an if/else tree on the A7 is faster to
> ship, easier to debug, and already real-time. Reach for the NPU only if you later
> move to heavy CNN/LSTM models.

> **Topology check (FPV):** `wfb_rx` runs on the **ground station** (receiver), so
> Tier-1 most naturally runs there, sending decisions to the **air unit** over the
> uplink/MAVLink tunnel. If you want the model *on* the SSC338Q air unit, feed it the
> air-side stats (or relay ground stats up). The C model is portable either way —
> confirm where your decision loop lives so the I/O is wired to the right side.

## Retraining loop

1. Capture real traffic: `wfb_rx -Y ... > telemetry/real_wfb.jsonl`
2. Label (rule-bootstrap or outcomes), retrain:
   `python3 telemetry/train_tier1.py --input telemetry/real_wfb.jsonl`
3. Diff `model/metrics.txt`, redeploy `tier1_model.c`. Use Tier-2 Gemma offline to
   sanity-check disagreements and mine new labels (active learning).

### Or run the loop through the data-session store (recommended)

`DATASTORE.md` describes a SQLite store + Flask/uPlot UI that makes this loop
clickable. The committed `wfb.sqlite` ships a demo session so it works out of the
box:

```bash
pip install -r telemetry/requirements-webui.txt
python3 telemetry/webui/webapp.py --port 8080     # browse + label sessions
python3 telemetry/wfb_active.py train --sessions all --model-ver tier1-tree
python3 telemetry/wfb_active.py score --sessions all --model-ver tier1-tree   # UI overlay updates
python3 telemetry/wfb_active.py gemma --session 1 --model-ver tier1-tree --dry-run
```

Label in the UI → `train` (human `state` labels override the rule-bootstrap base)
→ `score` (writes a versioned prediction set; latest wins in the overlay) →
`gemma` (Tier-2 explanations + candidate labels) → relabel. Live captures land in
the same store via `wfb_ingest.py` on a `wfb_stats_tee.py --tap`.
