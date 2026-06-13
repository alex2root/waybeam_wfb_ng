#!/bin/sh
# probe_ground_parallel_soak.sh -- GS-side MIXED/PARALLEL probe loop for a soak.
#
# Pairs with: probe_drone.sh <cur> <link> <pps> <secs> "<rungs>" on the vehicle,
# which runs one continuous fixed-MCS wfb_tx per rung on ports link_id+index.
# Here we run ONE wfb_rx per rung port (forward-port-SAFE -u dead ports), feed
# them all into probe_log (one --rung each), and bridge the {"type":"probe"}
# records to the uplink udp_in (127.0.0.1:6600) so they ride to link_controller
# :5801. Every rung refreshes every -l window (10 Hz) -- no sweep latency.
#
# Assumes the normal GS video RX + uplink TX are already up (gs_supervisor).
# Run under sudo (pcap).
#
# Usage: sudo sh probe_ground_parallel_soak.sh "<rung list>" [link_id]
#   e.g. sudo sh probe_ground_parallel_soak.sh "5 6 7" 50
set -u

RUNGS="${1:?usage: probe_ground_parallel_soak.sh \"<rung list>\" [link_id]}"
LINK_ID="${2:-50}"
KEY="${KEY:-/etc/drone.key}"
ADAPTER="${ADAPTER:-wlx40a5ef2f229b}"
RX="${RX:-/home/snokvist/dev/waybeam-coordination/waybeam_wfb_ng/wfb-ng/build/wfb_rx_native}"
STATS_BASE="${STATS_BASE:-5850}"        # rung r -> -Y 127.0.0.1:STATS_BASE+r
CLIENT_BASE="${CLIENT_BASE:-5751}"      # rung r -> -u CLIENT_BASE+r (DEAD port, never 5600)
WINDOW_S="${WINDOW_S:-0.3}"             # short window: fresh PER, still enough packets at 20pps
UPLINK="${UPLINK:-127.0.0.1:6600}"      # GS uplink wfb_tx udp_in
HERE="$(cd "$(dirname "$0")" && pwd)"

PIDS=""
cleanup() { for p in $PIDS; do kill "$p" 2>/dev/null; done; }
trap 'cleanup; exit 0' INT TERM
trap cleanup EXIT

rung_args=""
r=0
for M in $RUNGS; do
    P=$((LINK_ID + r))
    SP=$((STATS_BASE + r))
    CP=$((CLIENT_BASE + r))
    echo "[soak-gs] rung MCS=$M radio_port=$P fwd=:$CP -Y :$SP"
    "$RX" -K "$KEY" -i "$LINK_ID" -p "$P" -c 127.0.0.1 -u "$CP" \
          -x -Y "127.0.0.1:$SP" -l 100 "$ADAPTER" >"/tmp/soak_rx_$P.log" 2>&1 &
    PIDS="$PIDS $!"
    rung_args="$rung_args --rung $P:$M:$SP"
    r=$((r + 1))
done

sleep 1
echo "[soak-gs] probe_log$rung_args (window=${WINDOW_S}s) | bridge -> $UPLINK"
python3 "$HERE/probe_log.py" $rung_args --window-s "$WINDOW_S" \
  | python3 "$HERE/probe_bridge.py" --to "$UPLINK" &
PIDS="$PIDS $!"

echo "[soak-gs] running. PIDs:$PIDS  (Ctrl-C cleans up)"
wait
