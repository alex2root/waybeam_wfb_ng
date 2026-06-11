#!/bin/sh
# walkout_logger — persist link_controller telemetry to SD for post-walkout
# analysis (RSSI noise amplitude, guard calibration, flap inspection).
#
# Started by S99wfb when wfbmode=1 (disable with: fw_setenv wfblog 0).
# REFUSES to run without the SD card mounted — logs must never land on the
# overlay (/root, /) per the platform rule; /tmp is tmpfs and lost on boot.
#
# Per boot session directory on the SD card:
#   walkout/<stamp>/lc.log        mirror of /tmp/wfb.log (every decision/
#                                 hb/probe line with lc-relative timestamps)
#   walkout/<stamp>/status.jsonl  1 Hz /status samples wrapped as
#                                 {"ts":<unix>,"up":<uptime_s>,"status":{...}}
#
# Keeps the newest 10 sessions; stops sampling if the session exceeds
# MAX_MB (tail keeps running — the decision log is the cheap part).
#
# Analysis on the host:
#   jq -r '[.up, .status.score.smoothed_rssi, .status.score.rssi_slope_db_s,
#           .status.mcs.current_mcs] | @tsv' status.jsonl

# Env-overridable for host testing only — the vehicle uses the defaults.
SD="${WALKOUT_SD:-/mnt/mmcblk0p1}"
API="${WALKOUT_API:-http://127.0.0.1:8765}"
SRC_LOG="${WALKOUT_SRC_LOG:-/tmp/wfb.log}"
KEEP_SESSIONS=10
MAX_MB=100

# SD card must be a real mount, not a leftover directory on the overlay.
grep -q " $SD " /proc/mounts || { echo "walkout: no SD at $SD — logging disabled"; exit 0; }

BASE="$SD/walkout"
mkdir -p "$BASE" || exit 0

# Prune: keep the newest KEEP_SESSIONS-1, then create ours.
n=$(ls -1d "$BASE"/*/ 2>/dev/null | wc -l)
while [ "$n" -ge "$KEEP_SESSIONS" ]; do
    old=$(ls -1d "$BASE"/*/ 2>/dev/null | sort | head -1)
    [ -n "$old" ] || break
    rm -rf "$old"
    n=$((n-1))
done

# No RTC: date may be 1970-based pre-NTP. $$ disambiguates same-second boots.
DIR="$BASE/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$DIR" || exit 0
echo "walkout: logging to $DIR"

# Mirror the wfb log (decision lines carry the precise commit timeline).
# Plain -f (not -F): /tmp/wfb.log exists before we start (S99wfb wrote to
# it) and never rotates within a boot; busybox tail -F is config-dependent.
tail -n +1 -f "$SRC_LOG" >> "$DIR/lc.log" 2>/dev/null &
TAIL_PID=$!

cleanup() { kill "$TAIL_PID" 2>/dev/null; exit 0; }
trap cleanup TERM INT

i=0
sampling=1
while :; do
    if [ "$sampling" = "1" ]; then
        TS=$(date +%s)
        UP=$(cut -d. -f1 /proc/uptime)
        S=$(wget -q -T 2 -O - "$API/status" 2>/dev/null)
        [ -n "$S" ] && echo "{\"ts\":$TS,\"up\":$UP,\"status\":$S}" >> "$DIR/status.jsonl"
        i=$((i+1))
        if [ $((i % 120)) -eq 0 ]; then
            SZ=$(du -sm "$DIR" 2>/dev/null | cut -f1)
            if [ -n "$SZ" ] && [ "$SZ" -ge "$MAX_MB" ]; then
                echo "walkout: $DIR hit ${MAX_MB}MB — sampling stopped (log mirror continues)"
                sampling=0
            fi
        fi
    fi
    sleep 1
done
