#!/usr/bin/env python3
"""Train the Tier-1 "inference model" for wfb link decision-making.

Tier 1 is the cheap, always-on classifier that runs at full telemetry rate (on
the dev box now, on a SigmaStar SSC338Q later). It is intentionally a small
DECISION TREE, not a neural net, because:
  * it exports to dependency-free C (m2cgen) -> microsecond eval on a Cortex-A7
  * it is interpretable (you can read the thresholds it learned)
  * it needs no NPU / no runtime / a few KB of RAM

Two modes:
  supervised   (default, needs a `label` field): DecisionTreeClassifier
  unsupervised (--unsupervised): IsolationForest anomaly score (no labels needed)

Outputs (into telemetry/model/):
  features.json   stable feature order (the C model's input contract)
  tier1.pkl       sklearn model for the Python pipeline (Tier-2 gating)
  tier1_model.c   portable C -> cross-compile for the SSC338Q
  metrics.txt     accuracy / tree size

    pip install -r telemetry/requirements-telemetry.txt
    python3 telemetry/train_tier1.py --input telemetry/sample_wfb.jsonl --max-depth 5
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

from wfb_schema import (build_manifest, load_jsonl, vectorize,
                        REDUCED_FEATURES, vectorize_reduced,
                        FULL_FEATURES, windowed_features)

# never feed bookkeeping/label/monotonic-counter columns as features: ts_ms &
# seq increase forever (leakage/overfit), ver & interval_ms are constants.
DROP = {"timestamp", "label", "ts_ms", "seq", "ver", "interval_ms"}


def train_and_export(records: list[dict], outdir: str, max_depth: int = 5,
                     unsupervised: bool = False, reduced: bool = False,
                     trend: bool = False, window: int = 10) -> dict:
    """Train a Tier-1 model from in-memory records and write the artifacts.

    Shared by this CLI and the active-learning loop (wfb_active.py), which feeds
    records pulled from the data-session store. `records` need a `label` field
    for supervised mode. Returns {model, manifest, kind, metrics}.

    Feature set: default = full flatten() manifest; `reduced` = deployable
    SNR-free RSSI+packet-counter set; `trend` = reduced base + CAUSAL trend
    features (input MUST be one session in time order — trend is stateful).
    """
    try:
        import numpy as np
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise SystemExit("Install deps: pip install -r telemetry/requirements-telemetry.txt")

    if not records:
        raise SystemExit("no records to train on")
    if trend:
        manifest = list(FULL_FEATURES)
        X = np.array(windowed_features(records, window=window), dtype=float)
    elif reduced:
        manifest = list(REDUCED_FEATURES)
        X = np.array([vectorize_reduced(r) for r in records], dtype=float)
    else:
        manifest = build_manifest(records, drop=DROP)
        X = np.array([vectorize(r, manifest) for r in records], dtype=float)
    os.makedirs(outdir, exist_ok=True)

    if unsupervised:
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(n_estimators=50, contamination="auto", random_state=0)
        model.fit(X)
        kind = "iforest"
        metrics = (f"unsupervised IsolationForest, {len(manifest)} features, "
                   f"{model.n_estimators} trees\n"
                   "score < 0 => anomaly (link event worth escalating to Tier-2)\n")
    else:
        from sklearn.tree import DecisionTreeClassifier
        if "label" not in records[0]:
            raise SystemExit("No `label` field; use unsupervised or add labels.")
        y = np.array([r["label"] for r in records])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0,
                                              stratify=y)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=0,
                                       class_weight="balanced")
        model.fit(Xtr, ytr)
        kind = "tree"
        acc = model.score(Xte, yte)
        imp = sorted(zip(manifest, model.feature_importances_),
                     key=lambda t: -t[1])[:8]
        lines = [f"supervised DecisionTree depth<={max_depth}, "
                 f"{len(manifest)} features, {model.get_n_leaves()} leaves",
                 f"holdout accuracy: {acc:.3f}", "top features:"]
        lines += [f"  {n}: {v:.3f}" for n, v in imp if v > 0]
        metrics = "\n".join(lines) + "\n"

    # persist
    with open(os.path.join(outdir, "features.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(outdir, "tier1.pkl"), "wb") as fh:
        pickle.dump({"model": model, "manifest": manifest, "kind": kind,
                     "reduced": reduced or trend,
                     "trend": trend, "window": window}, fh)
    with open(os.path.join(outdir, "metrics.txt"), "w") as fh:
        fh.write(metrics)

    # export portable C for the SSC338Q
    try:
        import m2cgen as m2c
        c_code = m2c.export_to_c(model)
        trend_note = (
            " * TREND MODEL: inputs 5..10 are CAUSAL trend features that m2cgen does\n"
            " *   NOT compute -- feed each record through telemetry/tier1_trend.h\n"
            f" *   (t1_trend_push, TIER1_TREND_WINDOW={window}) to build input[].\n"
            if trend else "")
        header = (
            "/* Tier-1 wfb link model -- generated by train_tier1.py via m2cgen.\n"
            " * Dependency-free C: cross-compile for SigmaStar SSC338Q (Cortex-A7).\n"
            " * Input: double input[N] in the EXACT order of features.json:\n"
            " *   " + ", ".join(manifest) + "\n"
            + trend_note +
            " * Output: score(input, output). For the tree classifier, output[] holds\n"
            " *   per-class scores (argmax = predicted class: 0 ok,1 degraded,2 critical).\n"
            " * Missing/absent telemetry fields must be passed as 0.0 (see vectorize()).\n"
            " */\n"
        )
        with open(os.path.join(outdir, "tier1_model.c"), "w") as fh:
            fh.write(header + c_code)
        metrics += f"C export: {len(c_code)} bytes, {len(manifest)} inputs\n"
    except ImportError:
        metrics += "m2cgen not installed -> skipped C export (pip install m2cgen)\n"

    return {"model": model, "manifest": manifest, "kind": kind, "metrics": metrics}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="telemetry/sample_wfb.jsonl")
    ap.add_argument("--outdir", default="telemetry/model")
    ap.add_argument("--max-depth", type=int, default=5, help="keep small for SSC338Q")
    ap.add_argument("--unsupervised", action="store_true",
                    help="IsolationForest anomaly model (ignore labels)")
    ap.add_argument("--reduced", action="store_true",
                    help="deployable feature set: best-chain RSSI + packet counters "
                         "only (SNR-free, antenna-id-independent). The model to ship.")
    ap.add_argument("--trend", action="store_true",
                    help="reduced base + CAUSAL trend features (RSSI slope, rolling "
                         "loss-rate, time-since-loss, ...) for the predictive class. "
                         "Input MUST be one session in time order (trend is stateful).")
    ap.add_argument("--window", type=int, default=10,
                    help="trailing window (records) for --trend features")
    args = ap.parse_args()

    records = load_jsonl(args.input)
    res = train_and_export(records, args.outdir, max_depth=args.max_depth,
                           unsupervised=args.unsupervised, reduced=args.reduced,
                           trend=args.trend, window=args.window)
    print(res["metrics"])
    print(f"artifacts in {args.outdir}/  (features.json, tier1.pkl, tier1_model.c, metrics.txt)")


if __name__ == "__main__":
    main()
