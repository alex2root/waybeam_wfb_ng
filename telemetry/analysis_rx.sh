#!/bin/bash
# Standalone HIGH-RATE analysis RX for capturing finer-grained event data.
#
# Why: the live gs_supervisor RX samples stats at 10 Hz (-l 100) because that
# rate goes up the uplink back-channel to the air unit. Now that Tier-1 parsing
# runs ground-side, we want 20-50 Hz for better event resolution WITHOUT loading
# the uplink. This launches a SECOND wfb_rx_native that reads the same monitor
# interfaces in parallel (read-only pcap -- does NOT disturb the supervisor's RX,
# the video, or the uplink) and emits -Y stats to a LOCAL port we capture directly.
# It is the prototype of the target design: ground samples fast, emits only
# events upstream.
#
#   sudo ./telemetry/analysis_rx.sh           # 50 Hz -> analysis_50hz_wfb.jsonl
#   sudo ./telemetry/analysis_rx.sh 20 walk2  # 20 Hz -> walk2_wfb.jsonl
# Ctrl-C to stop (tears down the analysis RX; the live RX is untouched).
#
# Params mirror the running supervisor RX (link_id 207, radio_port 0, -x, key,
# both adapters). Adjust here if host_x86.json changes.
set -euo pipefail
HZ="${1:-50}"
NAME="${2:-analysis_${HZ}hz}"
LOG_MS=$(( 1000 / HZ ))
KEY="/etc/drone.key"
LINK_ID=207
RADIO_PORT=0
IFACES=(wlx40a5ef2f229b wlx40a5ef2f2308)
STATS_PORT="${STATS_PORT:-6700}"     # our private analysis stats port
VIDEO_PORT="${VIDEO_PORT:-5699}"     # dump decoded video to an unused port
RXBIN="/home/snokvist/dev/waybeam-coordination/waybeam_wfb_ng/wfb-ng/build/wfb_rx_native"
OUT="$(dirname "$0")/${NAME}_wfb.jsonl"

[ -x "$RXBIN" ] || { echo "wfb_rx_native not found at $RXBIN"; exit 1; }
[ -r "$KEY" ]   || { echo "key $KEY not readable (need sudo?)"; exit 1; }
echo "# analysis RX @ ${HZ} Hz (-l ${LOG_MS}) on ${IFACES[*]}" >&2
echo "# stats -> 127.0.0.1:${STATS_PORT}  video -> :${VIDEO_PORT} (discarded)" >&2
echo "# writing $OUT   (Ctrl-C to stop)" >&2
: > "$OUT"

# capture listener (userspace UDP socket on OUR port -- no sudo, no tshark)
python3 -u -c "
import socket, json, sys
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('127.0.0.1', ${STATS_PORT}))
n = 0
out = open('${OUT}', 'a')
try:
    while True:
        data, _ = s.recvfrom(65535)
        line = data.decode('utf-8', 'replace').strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        out.write(json.dumps(rec) + '\n'); out.flush()
        n += 1
        if n % 50 == 0:
            sys.stderr.write(f'\r# {n} records'); sys.stderr.flush()
except KeyboardInterrupt:
    pass
" &
LISTENER=$!

# tear down both on exit
cleanup() { kill "$LISTENER" 2>/dev/null || true; kill "$RXPID" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

# the parallel analysis decoder
"$RXBIN" -K "$KEY" -i "$LINK_ID" -p "$RADIO_PORT" -x -l "$LOG_MS" \
    -c 127.0.0.1 -u "$VIDEO_PORT" -Y "127.0.0.1:${STATS_PORT}" "${IFACES[@]}" &
RXPID=$!
wait "$RXPID"
