#!/usr/bin/env python3
"""Train / evaluate the probe-headroom predictive-critical model from walk feature
files (build_probe_features.py output).

Scales across sessions: pass several --features files (one per walk/flight) and it
does LEAVE-ONE-SESSION-OUT cross-validation. With a single file it falls back to an
out-vs-back temporal split (train on the walk away, test on the way back), clearly
flagged as small-N / illustrative.

Inputs (cheap, video-derivable, C-exportable):  rssi, snr, rssi_slope
Target:  critical_within_H = headroom_rungs <= 0 at any bin in [t, t+horizon]
         -- "the link can't sustain the current rate within the horizon".
The PREDICTIVE cases are bins where critical_now=0 but critical_within=1: the cliff
is coming but isn't here yet. RSSI slope is what lets the model catch those early;
a plain RSSI threshold (the controller's current logic) only fires once RSSI is
already low. We compare rssi-only vs rssi+slope vs rssi+snr+slope against that
threshold baseline.

Usage:
  telemetry/.venv/bin/python telemetry/probe/train_headroom.py \
      --features telemetry/loop/walk_*_features.jsonl [--horizon 2] [--depth 2] \
      [--save telemetry/model_headroom/bundle.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import precision_recall_fscore_support

FEATURE_SETS = {
    "rssi": ["rssi"],
    "rssi+slope": ["rssi", "rssi_slope"],
    "rssi+snr+slope": ["rssi", "snr", "rssi_slope"],
}


def load_session(path: str, horizon: int) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            r = json.loads(line)
            if r.get("rssi") is None:
                continue
            r["rssi_slope"] = r["rssi_slope"] if r.get("rssi_slope") is not None else 0.0
            r["snr"] = r["snr"] if r.get("snr") is not None else 0.0
            rows.append(r)
    n = len(rows)
    keep = []
    for i, r in enumerate(rows):
        window = [rows[k]["headroom_rungs"] for k in range(i, min(i + horizon + 1, n))]
        r["critical_now"] = 1 if r["headroom_rungs"] <= 0 else 0
        r["critical_within"] = 1 if min(window) <= 0 else 0
        keep.append(r)
    return keep


def matrix(rows: list, feats: list):
    X = np.array([[r[f] for f in feats] for r in rows], dtype=float)
    y = np.array([r["critical_within"] for r in rows], dtype=int)
    return X, y


def evalset(name, Xtr, ytr, Xte, yte, depth):
    if len(set(ytr)) < 2:
        return None  # degenerate (one class) -- can't train
    clf = DecisionTreeClassifier(max_depth=depth, class_weight="balanced",
                                 random_state=0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, labels=[1],
                                                  zero_division=0, average="binary",
                                                  pos_label=1)
    acc = float((pred == yte).mean())
    return {"name": name, "acc": acc, "prec": p, "rec": r, "f1": f1, "clf": clf}


def rssi_threshold_baseline(Xtr_rssi, ytr, Xte_rssi, yte):
    """Best single RSSI cutoff on train (critical if rssi < T), scored on test."""
    cand = sorted(set(Xtr_rssi[:, 0]))
    best_T, best_f1 = None, -1.0
    for T in cand:
        pred = (Xtr_rssi[:, 0] < T).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(ytr, pred, labels=[1],
                                                      zero_division=0, average="binary")
        if f1 > best_f1:
            best_f1, best_T = f1, T
    pred = (Xte_rssi[:, 0] < best_T).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, labels=[1],
                                                  zero_division=0, average="binary")
    acc = float((pred == yte).mean())
    return {"name": f"RSSI<{best_T:.0f} (baseline)", "acc": acc, "prec": p,
            "rec": r, "f1": f1, "T": best_T}


def splits(sessions: list):
    """Yield (train_rows, test_rows, label). LOSO if >1 session else out/back."""
    if len(sessions) > 1:
        for i, (name, rows) in enumerate(sessions):
            train = [r for j, (_, rr) in enumerate(sessions) if j != i for r in rr]
            yield train, rows, f"hold-out {name}"
    else:
        name, rows = sessions[0]
        out = [r for r in rows if r.get("phase") == "out"]
        back = [r for r in rows if r.get("phase") == "back"]
        yield out, back, f"{name}: train=out test=back"


def fmt(m):
    if m is None:
        return "  (degenerate: single class in train)"
    return (f"  {m['name']:<22} acc={m['acc']:.2f}  "
            f"crit precision={m['prec']:.2f} recall={m['rec']:.2f} f1={m['f1']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="+", required=True,
                    help="walk feature JSONL file(s) -- one per session")
    ap.add_argument("--horizon", type=int, default=2,
                    help="bins ahead for critical_within (default 2 = ~12 s at 6 s bins)")
    ap.add_argument("--depth", type=int, default=2, help="tree max depth (default 2)")
    ap.add_argument("--save", help="write the trained rssi+slope bundle here (JSON)")
    args = ap.parse_args()

    sessions = [(os.path.basename(p), load_session(p, args.horizon))
                for p in args.features]
    allrows = [r for _, rows in sessions for r in rows]
    npos = sum(r["critical_within"] for r in allrows)
    npred = sum(1 for r in allrows if r["critical_now"] == 0 and r["critical_within"] == 1)
    print(f"sessions={len(sessions)}  bins={len(allrows)}  "
          f"critical_within={npos} ({100*npos/len(allrows):.0f}%)  "
          f"of which PREDICTIVE (not-yet-critical)={npred}")
    if len(sessions) == 1:
        print("NOTE: single session -> out/back split; metrics are illustrative, "
              "not a session-independent validation. Add flight sessions to harden.")

    for train, test, label in splits(sessions):
        print(f"\n== {label}  (train n={len(train)}, test n={len(test)}, "
              f"test criticals={sum(r['critical_within'] for r in test)}) ==")
        # baseline
        Xtr_r, ytr = matrix(train, ["rssi"])
        Xte_r, yte = matrix(test, ["rssi"])
        if len(set(ytr)) > 1:
            print(fmt(rssi_threshold_baseline(Xtr_r, ytr, Xte_r, yte)))
        # trees
        best = None
        for fname, feats in FEATURE_SETS.items():
            Xtr, ytr = matrix(train, feats)
            Xte, yte = matrix(test, feats)
            m = evalset(fname, Xtr, ytr, Xte, yte, args.depth)
            print(fmt(m))
            if m and (best is None or m["f1"] > best[0]["f1"]):
                best = (m, feats)
        if best and "slope" in "+".join(best[1]):
            clf = best[0]["clf"]
            print(f"  -- best tree rules ({best[0]['name']}):")
            for ln in export_text(clf, feature_names=best[1]).splitlines():
                print("     " + ln)

    # --- predictive-case separability (the honest single-session evidence) -------
    pred = [r for r in allrows if r["critical_now"] == 0 and r["critical_within"] == 1]
    clean = [r for r in allrows if r["critical_within"] == 0]
    if pred and clean:
        clean_rssi = sorted(r["rssi"] for r in clean)
        print(f"\n== predictive bins (not-yet-critical, cliff within {args.horizon} "
              f"bins) -- can RSSI alone catch them? ==")
        print(f"{'t':>6} {'rssi':>5} {'slope':>6}   ambiguity")
        for r in sorted(pred, key=lambda x: x["t"]):
            worse = sum(1 for cr in clean_rssi if cr <= r["rssi"])
            print(f"{r['t']:6.0f} {r['rssi']:5.0f} {r['rssi_slope']:+6.2f}   "
                  f"{worse} clean bins have equal/worse RSSI -> RSSI alone ambiguous")
        pr_lo, pr_hi = min(r["rssi"] for r in pred), max(r["rssi"] for r in pred)
        ovl = "OVERLAP -> RSSI insufficient, slope needed" if pr_hi > min(clean_rssi) \
              else "separable by RSSI"
        cm = sorted(r["rssi_slope"] for r in clean)[len(clean) // 2]
        print(f"  RSSI: clean {min(clean_rssi):.0f}..{max(clean_rssi):.0f} vs "
              f"predictive {pr_lo:.0f}..{pr_hi:.0f}  [{ovl}]")
        print(f"  slope: predictive {min(r['rssi_slope'] for r in pred):+.2f}.."
              f"{max(r['rssi_slope'] for r in pred):+.2f} dB/s vs clean median "
              f"{cm:+.2f} dB/s")

    if args.save:
        # train the rssi+slope model on ALL data for deployment/export
        X, y = matrix(allrows, FEATURE_SETS["rssi+slope"])
        if len(set(y)) > 1:
            clf = DecisionTreeClassifier(max_depth=args.depth, class_weight="balanced",
                                         random_state=0).fit(X, y)
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            bundle = {"features": FEATURE_SETS["rssi+slope"], "horizon": args.horizon,
                      "depth": args.depth, "n": len(allrows), "sessions": len(sessions),
                      "tree": export_text(clf, feature_names=FEATURE_SETS["rssi+slope"])}
            with open(args.save, "w") as f:
                json.dump(bundle, f, indent=2)
            print(f"\nsaved rssi+slope bundle -> {args.save}")


if __name__ == "__main__":
    main()
