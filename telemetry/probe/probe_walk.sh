#!/bin/bash
# probe_walk.sh -- one-command walk capture for the swept-TX probe-PER.
#
# Runs on the GROUND box. Auto-detects the current video MCS, pushes the swept
# drone script, starts the ground capture (RX-only adapter) into OUT, and starts
# the drone sweep DETACHED + HARD TIME-BOXED so it survives an ssh drop (the drone
# may leave wired-eth0 range mid-walk) yet ALWAYS self-terminates at the cap -- it
# can never leak a probe TX into production. Live per-MCS PER is printed while you
# walk; Ctrl-C (or stopping this task) tears both sides down and verifies clean.
#
# Usage:  ./probe_walk.sh [secs] [link_id] [cur_mcs_override]
#   env:  OUT=telemetry/loop/probe_walk.jsonl  DRONE=root@192.168.1.13  PPS=20
#
# Safety model (hard constraints, see gemma4 memory wfb-manual-probe-setup):
#   * NEVER touches production video (i=207/p0), uplink (i=208/p0) or 8000 ctrl.
#   * Probe is isolated on a fresh link_id; drone self-terminates at `secs`.
#   * Drone teardown targets the probe script by name -- NEVER `killall wfb_tx`.
set -u

SECS="${1:-600}"                 # hard cap (drone self-stops here even if ssh drops)
LINK_ID="${2:-50}"
CUR_OVERRIDE="${3:-}"
DRONE="${DRONE:-root@192.168.1.13}"
PPS="${PPS:-20}"
OUT="${OUT:-telemetry/loop/probe_walk.jsonl}"
SSH="ssh -o ConnectTimeout=6"
HERE="$(cd "$(dirname "$0")" && pwd)"
SWEPT_LOCAL="$HERE/probe_drone_swept.sh"
SWEPT_REMOTE="/tmp/probe_drone_swept.sh"
GROUND="$HERE/probe_ground_swept.sh"
SD="${SD:-/mnt/mmcblk0p1}"        # drone SD card (vfat ~58 GB): rootfs is RO, /tmp is RAM
DRONE_LOGDIR="$SD/probe"          # persist ANY drone-side data here (NOT rootfs/tmp)

# --- 1. resolve current video MCS (so the probe rungs track video) -------------
if [ -n "$CUR_OVERRIDE" ]; then
    CUR="$CUR_OVERRIDE"
else
    CUR="$($SSH "$DRONE" 'wfb_tx_cmd 8000 get_radio' 2>/dev/null \
           | sed -n 's/^mcs_index=//p' | tr -d '\r ')"
fi
case "$CUR" in
    ''|*[!0-9]*) echo "[walk] ERROR: could not read video MCS (pass it as arg 3)"; exit 1;;
esac
echo "[walk] video MCS=$CUR -> sweep $CUR..$((CUR+2)) (clamped 0..7)  link_id=$LINK_ID  cap=${SECS}s"
echo "[walk] out=$OUT  drone=$DRONE"
mkdir -p "$(dirname "$OUT")"

# --- 2. push the swept drone script -------------------------------------------
if ! cat "$SWEPT_LOCAL" | $SSH "$DRONE" "cat > $SWEPT_REMOTE && chmod +x $SWEPT_REMOTE"; then
    echo "[walk] ERROR: failed to push $SWEPT_LOCAL to $DRONE"; exit 1
fi
echo "[walk] pushed swept script -> $DRONE:$SWEPT_REMOTE"

# --- teardown (runs on Ctrl-C / task stop / cap reached) -----------------------
GROUND_PID=""
cleanup() {
    trap - EXIT INT TERM
    echo; echo "[walk] stopping & verifying..."
    # ground: it ran under sudo (root); INT triggers its trap (RX+log cleanup)
    [ -n "$GROUND_PID" ] && sudo kill -INT "$GROUND_PID" 2>/dev/null
    # drone: kill the probe SCRIPT by name -> its trap kills TX+feeder+socat.
    # production-safe (never killall wfb_tx); best-effort if the drone is reachable.
    $SSH "$DRONE" '
        for p in $(ps | grep "[p]robe_drone_swept" | awk "{print \$1}"); do kill "$p" 2>/dev/null; done
        killall -q socat 2>/dev/null
    ' 2>/dev/null
    sleep 1
    echo "[walk] drone state:"
    $SSH "$DRONE" 'ps | grep -E "[-]C 7050|[p]robe_drone_swept|[s]ocat" | grep -v grep \
        && echo "  ^ PROBE STILL ON DRONE (will self-stop at cap)" \
        || echo "  drone clean (no probe procs)"
        echo "  production: $(ps | grep -E "[-]i 207 |[-]i 208 " | grep -v grep | wc -l) tx/rx + link_controller" ' 2>/dev/null \
        || echo "  (drone unreachable -- probe is time-boxed, self-stops at cap)"
    echo "[walk] capture: $(wc -l < "$OUT" 2>/dev/null || echo 0) records in $OUT"
}
trap cleanup EXIT INT TERM

# --- 3. ground capture (time-boxed a bit past the drone, as a local backstop) ---
echo "[walk] starting ground capture (RX-only adapter)..."
sudo timeout --signal=INT "$((SECS + 15))" "$GROUND" "$CUR" "$LINK_ID" "$OUT" \
     >/tmp/probe_walk_ground.log 2>&1 &
GROUND_PID=$!
sleep 3                          # let RX + probe_log bind before traffic starts

# --- 4. drone sweep: DETACHED (nohup) + hard time-boxed (self-stops at SECS) ----
echo "[walk] starting drone sweep (detached, hard cap ${SECS}s)..."
echo "[walk] drone-side log -> $DRONE_LOGDIR (SD card; /tmp fallback)"
# Resolve the log path on the drone: SD card if mounted, else /tmp (RAM).
$SSH "$DRONE" "LOG=$DRONE_LOGDIR/probe_walk_drone.log; \
    [ -d $SD ] && mkdir -p $DRONE_LOGDIR 2>/dev/null || LOG=/tmp/probe_walk_drone.log; \
    nohup sh $SWEPT_REMOTE $CUR $LINK_ID $PPS $SECS >\$LOG 2>&1 </dev/null & \
    echo '[walk] drone probe pid' \$!"

echo
echo "================================================================"
echo "  WALK NOW.  Sweeping MCS $CUR..$((CUR+2)) for up to ${SECS}s."
echo "  Walk the drone out until the UPPER rungs' PER climbs = the cliff."
echo "  Stop early any time (Ctrl-C / tell the agent) -- both sides clean."
echo "================================================================"

# --- live per-MCS PER while walking -------------------------------------------
summ() {
    python3 -c '
import sys,json,collections
out=sys.argv[1]
try: rows=[json.loads(l) for l in open(out)]
except Exception: rows=[]
rows=[r for r in rows[-240:] if r.get("accounted")]
a=collections.defaultdict(lambda:[0,0])
for r in rows: a[r["mcs"]][0]+=r["recv"]; a[r["mcs"]][1]+=r["lost"]
parts=["MCS%s %.1f%%(%d)"%(m,100*l/(u+l),u+l) for m,(u,l) in sorted(a.items())]
print("  [%ss] recent PER: %s"%(sys.argv[2], " | ".join(parts) or "(no probe traffic yet)"))
' "$OUT" "$1"
}
t=0
while [ "$t" -lt "$SECS" ]; do
    sleep 15; t=$((t + 15))
    summ "$t"
done
echo "[walk] reached ${SECS}s cap."
# EXIT trap runs cleanup
