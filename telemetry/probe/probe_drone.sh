#!/bin/sh
# probe_drone.sh -- DRONE side of the probe-PER ladder (runs on the air unit).
#
# Launches a boundary MCS probe (default: current video MCS and current+1) as
# extra wfb_tx streams on a fresh link_id, alongside the live video link, plus a
# paced PRB feeder for each rung. Drone -> ground direction (video-headroom).
#
# Bench-validated rules baked in (see gemma4 memory wfb-manual-probe-setup):
#   * MIRROR THE VIDEO PHY (-B 20 -S 1 -L 1), vary ONLY -M. Without LDPC the high
#     rungs read a falsely pessimistic cliff (MCS7 went 99.6%->0.8% PER with -L 1).
#   * FEC OFF (-k 1 -n 1) so raw loss isn't masked; with 1/1 each lost pkt counts.
#   * PACE the feed (<=20 pps/rung): blasting overruns the wfb_tx UDP input and
#     drops packets silently (PER understated). 20 pps/rung is the spec rate.
#   * Fresh LINK_ID (channel_id = link_id<<8 | radio_port) keeps it off video
#     (i=207/p0) and uplink (i=208/p0). Use the RX-only adapter on the ground.
#   * AEAD off (-x); the ground probe RX must also run -x.
#
# Usage:  sh probe_drone.sh <cur_mcs> [link_id] [pps] [secs] ["rung list"]
#   e.g.  sh probe_drone.sh 2                 # probe MCS 2 (p50) and 3 (p51), 20pps
#         sh probe_drone.sh 5 50 20 0 "5 6 7" # 3 CONCURRENT fixed-MCS streams
#
# Each rung is its OWN continuous wfb_tx on a distinct radio_port (link_id+index),
# so the ground wfb_rx -Y buckets every rung by mcs EVERY -l window (no sweep
# latency): all rungs stay fresh at up to 10 Hz simultaneously. This is the
# "mixed/parallel" probe — preferred over the swept single TX when you need
# sub-second V+2 freshness (the swept feeder caps at ~20 pps total).
#
# Cleanup is via trap: ALL spawned PIDs are killed on exit/INT/TERM so no probe
# tx can leak into production (we hit exactly that leak by hand once).
set -u

CUR_MCS="${1:?usage: probe_drone.sh <cur_mcs> [link_id] [pps] [secs] [rungs]}"
LINK_ID="${2:-50}"
PPS="${3:-20}"
SECS="${4:-0}"                      # 0 = run until killed
RUNGS="${5:-}"                      # explicit MCS list; default = cur,cur+1
KEY="${KEY:-/etc/drone.key}"
IFACE="${IFACE:-wlan0}"
BW="${BW:-20}"; STBC="${STBC:-1}"; LDPC="${LDPC:-1}"   # MIRROR VIDEO PHY
PKT_BYTES="${PKT_BYTES:-1400}"
UDP_BASE="${UDP_BASE:-5750}"       # rung r reads udp UDP_BASE+r
WFB_TX="${WFB_TX:-wfb_tx}"

# rungs: default current and current+1 (boundary probe); clamp to the 1SS regime.
[ -z "$RUNGS" ] && RUNGS="$CUR_MCS $((CUR_MCS + 1))"
_clamped=""
for M in $RUNGS; do
    if [ "$M" -ge 0 ] && [ "$M" -le 7 ]; then _clamped="$_clamped $M"; fi
done
RUNGS="$_clamped"
[ -z "$RUNGS" ] && { echo "[probe] no valid rungs in 0..7" >&2; exit 1; }

PIDS=""
cleanup() {
    for p in $PIDS; do kill "$p" 2>/dev/null; done
    # The shell feeder spawns one short-lived socat per packet; a kill landing
    # mid-send can orphan it. socat is used ONLY by this probe on the drone, so
    # reap any stray as a backstop (the deploy-target C feeder won't need this).
    killall -q socat 2>/dev/null
}
# INT/TERM must cleanup AND exit (a bare trap returns and would leak the TXs).
trap 'cleanup; exit 0' INT TERM
trap cleanup EXIT

# Paced PRB feeder: ASCII "PRB<10-digit seq>" + 0xA5-ish padding to PKT_BYTES,
# at PPS packets/sec, to udp 127.0.0.1:$port. (C feeder is the deploy target;
# this shell feeder is the validated prototype -- see PROBE_PER_SPEC.md sec 8.)
feeder() {
    _port="$1"
    _pad=$(awk "BEGIN{for(i=0;i<$PKT_BYTES-14;i++)printf \"x\"}")
    _n=0; _c=0
    while :; do
        printf 'PRB%010d%s' "$_n" "$_pad" | socat -u - "UDP-SENDTO:127.0.0.1:$_port"
        _n=$((_n + 1)); _c=$((_c + 1))
        if [ "$_c" -ge "$PPS" ]; then _c=0; sleep 1; fi
    done
}

r=0
for M in $RUNGS; do
    P=$((LINK_ID + r))             # radio_port per rung (50,51,...)
    U=$((UDP_BASE + r))
    echo "[probe] rung MCS=$M radio_port=$P udp_in=$U (PHY: B$BW S$STBC L$LDPC, FEC 1/1)"
    "$WFB_TX" -K "$KEY" -i "$LINK_ID" -p "$P" -M "$M" -B "$BW" -S "$STBC" -L "$LDPC" \
              -k 1 -n 1 -x -u "$U" -l 1000 "$IFACE" >/dev/null 2>&1 &
    PIDS="$PIDS $!"
    sleep 1
    feeder "$U" &
    PIDS="$PIDS $!"
    r=$((r + 1))
done

echo "[probe] running (link_id=$LINK_ID, rungs: $RUNGS). PIDs:$PIDS"
if [ "$SECS" -gt 0 ]; then sleep "$SECS"; else while :; do sleep 3600; done; fi
