#!/bin/bash
# Capture a wfb_rx -Y telemetry session (e.g. a move-away/walk-around run) into
# JSONL. The gs_supervisor already tees stats to 127.0.0.1:6600, so we sniff
# that loopback passively -- non-disruptive, doesn't touch the live link.
#
# Uses tshark (binary data.data), NOT tcpdump -A: text scraping can merge two
# adjacent UDP datagrams into one bogus record (we have 2 adapters x 2 antennas
# = 4 chains per record; a merged record would show 8 and corrupt training).
#
#   sudo ./telemetry/capture_walkaround.sh                 # -> walkaround_wfb.jsonl
#   sudo ./telemetry/capture_walkaround.sh far_edge        # -> far_edge_wfb.jsonl
# Ctrl-C to stop. Records are appended live; a counter prints to stderr.

set -euo pipefail
NAME="${1:-walkaround}"
OUT="$(dirname "$0")/${NAME}_wfb.jsonl"
PORT="${PORT:-6600}"

command -v tshark >/dev/null || { echo "tshark not installed (apt install tshark)"; exit 1; }
echo "# capturing udp port $PORT on lo -> $OUT   (Ctrl-C to stop)" >&2
: > "$OUT"

stdbuf -oL tshark -i lo -l -f "udp port $PORT" -T fields -e data.data 2>/dev/null \
  | python3 -u -c '
import sys, json
n = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        rec = json.loads(bytes.fromhex(line))
    except Exception:
        continue
    sys.stdout.write(json.dumps(rec) + "\n"); sys.stdout.flush()
    n += 1
    if n % 25 == 0:
        sys.stderr.write(f"\r# {n} records captured"); sys.stderr.flush()
' >> "$OUT"
