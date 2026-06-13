#!/bin/sh
# probe_ground_soak.sh -- GS-side full real probe loop for a mode=1 soak.
#
# Assumes the normal GS video RX + uplink TX are ALREADY running (gs_supervisor
# + host_x86.json: video link 207 -> :5600 viewer + stats; uplink link 208 reads
# udp_in 127.0.0.1:6600). This script adds ONLY the probe pieces:
#   1. probe wfb_rx (AEAD off, forward-port-SAFE -u <dead port>) -> -Y rx_ant
#   2. probe_log --by-mcs -> windowed {"type":"probe"} records
#   3. probe_bridge.py -> UDP to the uplink udp_in (127.0.0.1:6600), so probe PER
#      rides the SAME tunnel as video rx_ant up to link_controller :5801.
# Pairs with the vehicle's probe_drone_swept.sh. Run under sudo (pcap).
#
# Usage: sudo sh probe_ground_soak.sh <cur_mcs> [link_id]
#   e.g. sudo sh probe_ground_soak.sh 3 50
set -u

CUR_MCS="${1:?usage: probe_ground_soak.sh <cur_mcs> [link_id]}"
LINK_ID="${2:-50}"
KEY="${KEY:-/etc/drone.key}"
ADAPTER="${ADAPTER:-wlx40a5ef2f229b}"            # one video-diversity adapter (2nd pcap is fine)
RX="${RX:-/home/snokvist/dev/waybeam-coordination/waybeam_wfb_ng/wfb-ng/build/wfb_rx_native}"
RADIO_PORT="${RADIO_PORT:-50}"                   # must match probe_drone_swept.sh
STATS_PORT="${STATS_PORT:-5850}"
# CRITICAL: a probe wfb_rx with no -u inherits 127.0.0.1:5600 (the RTP video port)
# and flaps the video (root-caused 2026-06-10). Forward to a DEAD port; never 5600.
CLIENT_PORT="${CLIENT_PORT:-5751}"
UPLINK="${UPLINK:-127.0.0.1:6600}"               # GS uplink wfb_tx udp_in
HERE="$(cd "$(dirname "$0")" && pwd)"

PIDS=""
cleanup() { for p in $PIDS; do kill "$p" 2>/dev/null; done; }
# INT/TERM must cleanup AND exit (a bare trap returns and would leak the RX).
trap 'cleanup; exit 0' INT TERM
trap cleanup EXIT

echo "[soak-gs] probe RX i=$LINK_ID p=$RADIO_PORT fwd=:$CLIENT_PORT -Y :$STATS_PORT  bridge -> $UPLINK"
"$RX" -K "$KEY" -i "$LINK_ID" -p "$RADIO_PORT" -c 127.0.0.1 -u "$CLIENT_PORT" \
      -x -Y "127.0.0.1:$STATS_PORT" -l 100 "$ADAPTER" >/tmp/soak_rx.log 2>&1 &
PIDS="$PIDS $!"
sleep 1

echo "[soak-gs] probe_log --by-mcs (cur=$CUR_MCS) | probe_bridge -> $UPLINK"
python3 "$HERE/probe_log.py" --rung "$RADIO_PORT:$CUR_MCS:$STATS_PORT" --by-mcs --window-s 1.0 \
  | python3 "$HERE/probe_bridge.py" --to "$UPLINK" &
PIDS="$PIDS $!"

echo "[soak-gs] running. PIDs:$PIDS  (Ctrl-C cleans up)"
wait
