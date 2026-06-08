# Telemetry handoff — current state & next steps

For a local Claude Code CLI (running on the actual Gemma box / near the FPV gear)
to take over. Read `telemetry/README.md` first for the design and the SSC338Q
deploy guide; this file is the live to-do list. Two design specs sit alongside:
`telemetry/MCS_STRATEGY.md` (the per-MCS link-adaptation logic — RSSI→cliff
prediction with SNR as a privileged training signal, lookahead probe for
confirm+recalibration) and `telemetry/DATASTORE.md` (SQLite data-session store +
web UI for live/historical browsing, human labeling, and ML-label display).

## What's done (validated in CI container)
- Schema-agnostic flattener, throughput probe, Tier-1 trainer (+C export), and
  hybrid pipeline are written and run end-to-end.
- On synthetic data: DecisionTree → **0.98 holdout accuracy**, 12 leaves,
  `tier1_model.c` **compiles with gcc** (~3 KB object).
- **DONE — live throughput measured** (2026-06-07, on the Gemma box vs Ollama
  `gemma4:26b-a4b-it-qat`, `think=false`): decode rate **~21.6 tok/s** (cache-immune
  invariant; matches the documented figure), sustained **~0.2–0.9 rec/s** for small
  windows. A verdict is ~15–25 output tokens → ceiling ~1 verdict/sec, vs `wfb_rx -Y`
  ~10 Hz → ≥10× too slow. Tiered design confirmed by real measurement. Numbers are
  in `telemetry/README.md`. Probe bug fixed: windows ≳12 records overflow the
  8192-token context and were silently reporting a fake 1e6 tok/s decode — now
  flagged TRUNCATED. (Still TODO: re-run on a **real** `wfb_rx -Y` capture once we
  have one; synthetic records may differ in token size.)

- **DONE — real telemetry captured + whole chain validated on hardware**
  (2026-06-07). Real `wfb_rx_native -Y` schema is `type:"rx_ant"` with a per-antenna
  `ant[]` list (`rssi/snr {min,avg,max}`) + flat `pkt` counters — NOT the old
  `packets:[total,delta]` / `rx_ant_stats` dict the first synthetic generator
  assumed. Fixed across the toolchain: `flatten()` now keys `ant[]` by stable `id`
  (the array order is not stable); `gen_sample.py` emits the real shape (synthetic
  manifest == real manifest, 54 cols); `train_tier1` drops `ts_ms/seq/ver/interval_ms`;
  `wfb_pipeline` threshold reads real keys. Validated: probe on real data (~830
  tok/record, ~0.2 rec/s), trained tree 0.989 acc → `tier1_model.c` compiles with
  gcc, trained tree classifies real bench data 325/325 ok and synthetic across all
  3 classes, full Tier-2 Gemma escalation returns sane JSON. `wfb_stats_tee.py`
  added + loopback-validated (forwards verbatim + taps). Capture method: `tshark`
  (see README), NOT `tcpdump -A`.

## Deployable Tier-1 built (2026-06-07) — `train_tier1.py --reduced`

The full flatten() manifest overfits: a tree on all 114 columns hit 0.865 holdout
but leaned on per-antenna `ant.<id>.snr.*` keyed to one session's antenna ids and
on SNR (which the real `link_controller` ignores and not every driver reports).
**Do not deploy `model_real/`.**

The **deployable** model uses an antenna-id-independent, SNR-free reduction —
best-diversity-chain RSSI + flat packet counters (`wfb_schema.REDUCED_FEATURES`
= `rssi, pkt_all, pkt_lost, pkt_fec_recovered, pkt_uniq`; dropout → RSSI sentinel
−128). Rebuild it with:

```bash
.venv/bin/python train_tier1.py --input loop/trainset_real.jsonl \
    --outdir model_deploy --reduced --max-depth 5
gcc -O2 -I model_deploy -c model_deploy/tier1_model.c   # compiles clean (~6 KB C)
```

Result on the real walk2 trainset (ok=1899 deg=168 crit=1290): **holdout 0.810**,
29 leaves, importances `rssi 0.66, pkt_all 0.30, pkt_uniq 0.04` (SNR not present
at all). Per-class is the real story:
- **ok** prec/recall 0.92/0.94 — solid.
- **critical** prec 0.88 but recall **0.62** — and the split shows why:
  `signal_lost` recall **0.92** (reactive: RSSI −128 + pkt_all 0 is unambiguous),
  but `loss_occurred` **0.19** and predictive `worsened` **0.36**.
- **degraded** noisy (prec 0.28; mostly criticals down-graded — controller still
  reacts down on degraded, so safe-ish but imprecise).

⟹ Conclusion confirmed: a single instantaneous record cannot see the trend, so
partial-loss and *predictive* criticals are missed. The next gain is **trend
features** (item 2 below), NOT a bigger model. Artifacts (`model_deploy/`,
`model_real/`) are git-ignored — rebuild from the trainset.

## Trend features added (2026-06-07) — `train_tier1.py --trend`

CAUSAL trend features (trailing window, past+current only → online-computable on
the SSC338Q): `wfb_schema.TREND_FEATURES` = `rssi_slope, rssi_mean, loss_rate,
fec_rate, time_since_loss, dropout_frac`, on top of the 5 reduced base features
(11 total). Streaming extractor `wfb_schema.TrendState`; inference wired through
`wfb_pipeline.make_vectorizer()` (owns the per-link state) in both the pipeline
and the review queue. Rebuild:

```bash
.venv/bin/python train_tier1.py --input loop/trainset_real.jsonl \
    --outdir model_trend --trend --window 10
```

**Deploy contract is complete:** m2cgen only emits the tree (`score()`), so the
trend columns are computed in C by hand-written **`telemetry/tier1_trend.h`**
(a fixed-size ring buffer mirroring `TrendState`). Verified **bit-identical**:
C-vs-Python parity over 1623 real records = max feature diff 5e-7, **0 class
mismatches**. The full deploy unit (`tier1_model.c` + `tier1_trend.h`) compiles
clean with gcc.

**Do trend features help?** Honest answer: *yes, but one session can't prove it.*
- Like-for-like (random split): `loss_occurred` recall **0.19 → 0.54**, overall
  acc **0.810 → 0.836** (window=10 best). The tree leans on `rssi_mean` (0.59,
  the smoothed level) + `rssi_slope`/`loss_rate`/`time_since_loss`. Real signal.
- BUT a **temporal** split (honest, no near-duplicate-neighbor leakage) shows the
  predictive class barely moves (worsened recall ~0.1) AND has *zero*
  `loss_occurred` test cases — because walk2's partial-loss events are temporally
  clustered (by session quartile: **55 / 32 / 20 / 0**). One walk ≈ one
  degradation episode → can't train-then-test the predictive class on a tail.

⟹ **The binding constraint is now DATA DIVERSITY, not the model.** Trend features
are shipped + ready; honest validation needs MULTIPLE degradation episodes —
more walks across ranges, or the probe-PER cliff generator (item 6) which
manufactures the loss cliff repeatedly and bench-free. That is the next real
gain, ahead of any model tweak.

## What still needs a real environment (do these locally)

1. **Capture degraded/critical real telemetry.** DONE for one session (walk2:
   controller laddered MCS 3→2→1, 80 loss_occurred, cliff at MCS 1 ≈ −81 dBm).
   Now the bottleneck (see trend-features finding above): need MANY degradation
   episodes for an honest temporal split — more walks / wider range, or the
   probe-PER cliff generator (item 6). One session's events are too clustered.

2. **Feature engineering** (needs real data): add trend features the instantaneous
   record lacks — RSSI slope, rolling loss-rate over last N, time-since-last-loss.
   These predict link collapse far better than single-sample values. Add to
   `wfb_schema.py` (a windowed pass) and retrain.

3. **Labels.** Decide the labeling strategy (see README §"Labels"):
   rule-bootstrap → outcome-based (did uncorrectable loss occur within ~1 s?) →
   imitation. Replace the synthetic `label` with real ones; retrain.

4. **SSC338Q deploy.** Cross-compile `tier1_model.c` with the OpenIPC/Buildroot
   toolchain and wrap it in a daemon (read `wfb_rx -Y` → vectorize in
   `features.json` order → `score()` → act). Skeleton steps in README.

6. **MCS-ladder probe PER** (FUTURE, see `PROBE_PER_SPEC.md`). Measure real link
   headroom by probing candidate MCS rungs (drone->ground, FEC off) instead of
   inferring it from RSSI — both our −85 dBm finding and the gist agree only
   per-MCS PER reveals the cliff. Deferred until AFTER the current inference model
   is trained + concluded; it's a better headroom signal layered on later, and a
   bench-free loss-cliff data generator.

5. **Raise ground-side sampling to 20–50 Hz** (FOLLOW-UP). The 10 Hz (`wfb_rx
   -l 100`) was an *upstream* limit — what the air unit could handle. Now that
   parsing/Tier-1 runs ground-side, sample faster (`-l 50` = 20 Hz, `-l 20` =
   50 Hz) for finer event resolution (more samples inside the ~5 s decision
   window → cleaner EWMA, earlier transition detection), and emit only EVENTS
   upstream, not the firehose. Matches the tiered design. Re-check decode budget:
   higher rate = more records, but only flagged events hit Gemma so the LLM load
   is unchanged; Tier-1 (C) easily handles 50 Hz.

7. **MCS strategy (see `MCS_STRATEGY.md`).** Build the RSSI→cliff predictor:
   sweep RSSI/SNR across the full range at *each* MCS in capture; label cliffs
   from SNR; learn RSSI→cliff; add the lookahead-probe confirm/recalibration
   loop and the step-down(reactive)/step-up(predicted+probed) split. First code
   step: extend `gen_sample.py` to emit MCS-swept RSSI/SNR with realistic
   per-MCS cliffs and a way to drop SNR (simulate the RSSI-only inference path).

8. **Data-session store + web UI (see `DATASTORE.md`).** Store/ingest stdlib
   `sqlite3`; web layer locked to **Flask + uPlot** (`requirements-webui.txt`).
   Phased: ✅ schema+importer (`schema.sql` + `wfb_store.py`) → ✅ live ingester
   (`wfb_ingest.py`, UDP tap) → ✅ read-only Flask/uPlot UI (`webui/webapp.py`,
   session list + synced RSSI/SNR·PER/lost·MCS/Tier-1 charts with ML+human label
   overlays) → ✅ labeling tools (label-mode drag-select → write `labels`; labels
   manager + delete; session-metadata editor) → ✅ active-learning loop
   (`wfb_active.py train|score|gemma`: trains from store labels, writes versioned
   `predictions`, Gemma-fills `tier2_note`). uPlot vendored. The DB
   (`telemetry/wfb.sqlite`) is
   now **committed** (small + portable; only WAL/SHM sidecars ignored) and ships
   a tagged synthetic-demo session for out-of-the-box browsing.

## Closed training loop (built 2026-06-07, on branch claude/wfb-closed-loop)
- `wfb_review_queue.py` (Tier-1 + measured outcome), `wfb_gemma_verify.py`
  (Gemma fast labeler, CPU-only), `.claude/agents/wfb-loop-supervisor.md` (Tier-3
  Opus auditor/gate), `capture_walkaround.sh`. See git log + memory.
- **`wfb_link_score.py` — controller-aligned scorer BUILT (2026-06-07).** Faithful
  stdlib reproduction of `waybeam_wfb_ng/vehicle/link_controller.c` (defaults from
  its set_defaults(), logic verified against source): **effective_rssi =
  EWMA(best-avg RSSI, α0.3) − loss_penalty(0.5 dB/% lost, recovered=0, cap 20)**,
  SNR ignored, 3-bucket hysteresis FSM (lo −70/hi −50/db 2 → MCS 1/2/3), fast-down
  (consec 1 / cd 0.2 s) slow-up (consec 3 / cd 3 s, ×3 osc backoff), failsafe (no
  rx 0.5 s → bucket 0, recover after 3 samples ≥ −68), oscillation (≥4 changes/5 s).
  Validated: on walk2 it ladders **MCS 3→2→1** then failsafe→recover→climb, exactly
  the real controller's behavior that session; collapses the firehose to ~5–6
  flagged DECISION POINTS (down/failsafe/recovered) vs thousands of per-record
  classes. `--emit` writes per-record `{effective_rssi,bucket,mcs,event,flag,...}`.
  NEXT: feed its effective_rssi + decision-point flags into the review queue as the
  ALIGNED flagging signal (replace/augment the per-record Tier-1 class flag), and
  as a Tier-1 feature; drop the SNR thresholds in `gen_sample`/`threshold_tier1`.

## Open questions for the user (carry these forward)
- **Topology:** does the decision loop run on the **ground station** (where
  `wfb_rx` lives, commanding the air unit over the uplink) or **on the SSC338Q
  air unit** (needs air-side stats relayed/captured)? This sets where Tier-1 runs
  and how decisions are transported. — UNCONFIRMED.
- **Decision space:** confirm the actuator vocabulary (hold / lower_bitrate /
  switch_channel / RTH / MCS step) and how a decision is sent to the air unit
  (MAVLink? wfb tunnel? GPIO?). The Tier-1 output classes should match it.

## Guardrails
- Keep probe/pipeline stdlib-only; trainer may use sklearn/m2cgen.
- Don't commit `telemetry/sample_wfb.jsonl`, `telemetry/real_wfb.jsonl`, or
  `telemetry/model/` (already git-ignored).
- **Always `think=false` and CPU-only (`num_gpu:0`) for Gemma on this box.** The
  Radeon 760M has only 3 GB VRAM → offloading the 15 GB model spills into GTT
  (system RAM) and OOM-HANGS the whole box. think=true also overflows num_predict
  on most events. The labeler defaults handle this; don't override without reason.
