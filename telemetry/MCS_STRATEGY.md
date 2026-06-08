# MCS strategy — RSSI→cliff prediction + lookahead probe (wfb_ng)

Design spec for per-MCS link adaptation on the wfb_ng FPV link. This is the
**decision logic** that sits on top of the Tier-1/Tier-2 telemetry plumbing
(`README.md`): Tier-1 carries the firehose and makes the real-time MCS call;
Tier-2 Gemma explains the rare escalation.

## The wfb_ng constraint that frames everything

wfb_ng injects raw 802.11 frames at whatever MCS the **TX (air unit)** writes
into the radiotap header; the ground `wfb_rx` only receives. There is no free
"peek at MCS+1" — to actually observe PER at a higher MCS you must *transmit*
some frames there, and on a single-stream video link every probe frame at a
too-high MCS is potentially lost video. That cost rules out steady-state
exploration schemes (Minstrel-style samplers spend ~10% of airtime probing —
far too much here). The strategy must therefore be **predict-first, probe only
to confirm/recalibrate at a decision boundary**.

## Core model: RSSI → likely MCS cliff

The "cliff" for an MCS is the RSSI below which that MCS's PER explodes
(uncorrectable loss). With enough data sessions we learn, per MCS, where that
cliff sits, and at runtime pick the **highest MCS whose predicted PER is under
budget** at the current RSSI.

### Why RSSI, and the role of SNR (privileged training signal)

The PER cliff is fundamentally an **SNR** phenomenon — a modulation needs a
certain signal-*to-noise* ratio to decode. RSSI (signal power) is only a proxy,
valid **as long as the noise floor matches what we saw in training**.

- **Live:** SNR is not reliably reported, so the deployed predictor runs on
  **RSSI** (plus packet counters and trend features).
- **Training:** SNR *is* available. We use it as **privileged information** —
  train *with* it, infer *without* it (a LUPI-style setup):
  1. **Label the true cliff cleanly.** SNR gives a sharp, low-scatter cliff per
     MCS; learn the RSSI→cliff mapping against those clean SNR-defined labels.
  2. **Size the safety margin.** The residual scatter of RSSI around the
     SNR-defined cliff *is* the noise-floor uncertainty. That number sets how
     much RSSI headroom to demand before trusting an MCS.
  3. **Sanity-check drift.** Where live SNR *is* present, compare to the
     RSSI-predicted SNR to estimate the current noise-floor offset.

So: **SNR teaches, RSSI decides, the probe corrects.**

### The noise-floor blind spot (why the probe is mandatory, not optional)

RSSI cannot see the noise floor. When live interference raises it, the same
RSSI sits at a lower effective SNR and the cliff arrives *earlier* than the map
predicts. Under RSSI-only this is the dominant error source — and the only thing
that closes it is active probing. The probe is therefore load-bearing here, not
a nice-to-have.

## The lookahead probe: confirm + online recalibration

Before committing a **step-up**, transmit a brief burst at the candidate MCS and
measure its PER. The probe does double duty:

1. **Confirm** the predicted MCS survives *right now* before committing.
2. **Recalibrate.** Each probe yields a fresh `(RSSI, MCS, pass/fail)` sample
   that says where the cliff *actually* is in the current environment — slide
   the whole RSSI→cliff map up/down to track the invisible noise-floor drift
   between training and now.

Probe budget is cost-aware: a short burst only when near a step-up boundary,
never a continuous fraction of airtime.

## Step-down vs step-up asymmetry

These two decisions are not symmetric and must not share a path:

| direction | trigger | data needed | latency |
|-----------|---------|-------------|---------|
| **step DOWN** | live PER at current MCS spikes (`pkt.lost`, `dec_err`, `fec_recovered` climbing) | none extra — current traffic already measures it | **fast / reactive** — link is dying |
| **step UP** | RSSI prior says headroom exists | RSSI map + confirm probe | **lazy** — headroom isn't urgent |

RSSI-only uncertainty is tolerable for the lazy up-decision; it is *not* what you
bet a falling link on. The down-decision needs neither RSSI nor a probe — the
current MCS's own PER already tells you.

## RSSI-specific requirements

- **Combine the chains.** Stats arrive per antenna (`ant.0/1/100/101`). With
  diversity, effective performance tracks the *best/combined* chain, not any
  single antenna. Model input = combined/max RSSI across chains + the spread —
  never raw per-antenna values a `0.0` fill could poison.
- **Hysteresis / dwell.** RSSI is a noisier predictor than SNR, so widen the
  margin and require dwell time at a level before switching, to stop MCS
  flapping around the cliff.

## Decision flow

```
            ┌───────────────── every record (~10 Hz, Tier-1, C) ─────────────────┐
            │                                                                     │
  rx_ant ──►│  combine chains → rssi_comb, rssi_spread                            │
            │  live PER at current MCS = f(pkt.lost, dec_err, fec_recovered)      │
            │                                                                     │
            │   PER spiking? ──yes──► STEP DOWN now (reactive, no probe)          │
            │        │no                                                          │
            │        ▼                                                            │
            │   RSSI map says MCS+k has margin (≥ safety)? ──no──► HOLD           │
            │        │yes                                                         │
            │        ▼                                                            │
            │   lookahead PROBE at MCS+1 ──fail──► HOLD + recalibrate map down    │
            │        │pass                                                        │
            │        ▼                                                            │
            │   STEP UP + recalibrate map                                         │
            └─────────────────────────────────────────────────────────────────────┘
                         │ only on rare escalations (<1% volume)
                         ▼
              Tier-2 Gemma (think=false): human-readable "why we dropped 2 steps"
```

## Data requirements (what the capture campaign must produce)

The model is only as good as the coverage of the cliffs:

- **Sweep RSSI/SNR across the full range at *each* MCS** — not just healthy-vs-bad
  at one rate. You need records around the *knee* of every MCS's PER curve.
- Each record must carry: `mcs`, per-chain `rssi` **and** `snr` (train-time),
  packet counters for PER (`all/uniq/lost/fec_recovered/dec_err`), and enough
  context to compute trends (RSSI slope, rolling loss-rate, time-since-loss —
  HANDOFF #2).
- **Label cliffs from SNR**, then learn RSSI→cliff against those labels.
- Capture across environments (noise floors) so recalibration is exercised, not
  just one clean range walk.

`gen_sample.py` currently hard-codes `mcs:3`; to support this it must emit
MCS-swept RSSI/SNR with realistic per-MCS cliffs, and SNR must be droppable to
simulate the RSSI-only inference path. (Tracked in HANDOFF.)

## Open questions (confirm before building the actuator)

- **Per-packet / fast MCS switching:** can the air unit change MCS quickly and
  cheaply? If a switch is disruptive/slow, lean harder on prediction and switch
  rarely; if cheap, occasional step-up probes are viable.
- **Command transport:** how does the ground decision reach the air unit
  (MAVLink? wfb tunnel? other)? The Tier-1 output vocabulary must match.
- **FEC coupling:** does FEC (k/n) change with MCS? If so, the "PER budget" is
  really a *post-FEC* budget and the cliff labels must account for it.
- **Probe mechanics:** how is a probe burst injected at a different MCS without
  disrupting the video stream, and what burst size gives a usable PER estimate?
