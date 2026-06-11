#!/bin/bash
# walkout_logger (GS side) — persist gs_supervisor status to disk for
# post-walkout analysis (probe tunnel health, wcmd emit counters, tunnel
# restarts) alongside the vehicle-side logger.
#
# Designed to be wired into the gs_supervisor config system hooks, which
# enforce a per-command deadline — so `start` daemonizes and returns
# immediately:
#   "system": { "up":   [ ".../walkout_logger.sh start" ],
#               "down": [ ".../walkout_logger.sh stop" ] }
#
# Session directory per start:
#   $BASE/<stamp>/gs_status.jsonl  1 Hz /api/v1/status samples
#   Keeps the newest 10 sessions.
#
# Analysis:
#   jq -r '[.ts, .status.tunnels[]?|select(.name=="probe_rx")|.state]' ...

API="${WALKOUT_API:-http://127.0.0.1:80}"
BASE="${WALKOUT_DIR:-/var/log/waybeam-walkout}"
PIDFILE="${WALKOUT_PID:-/var/run/gs_walkout_logger.pid}"
KEEP_SESSIONS=10

sample_loop() {
    local dir="$1"
    while :; do
        local ts s
        ts=$(date +%s)
        s=$(curl -s -m 2 "$API/api/v1/status" 2>/dev/null)
        [ -n "$s" ] && printf '{"ts":%s,"status":%s}\n' "$ts" "$s" >> "$dir/gs_status.jsonl"
        sleep 1
    done
}

case "${1:-start}" in
    start)
        # Already running?
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "walkout(gs): already running ($(cat "$PIDFILE"))"
            exit 0
        fi
        mkdir -p "$BASE" || exit 1
        # Prune old sessions.
        n=$(ls -1d "$BASE"/*/ 2>/dev/null | wc -l)
        while [ "$n" -ge "$KEEP_SESSIONS" ]; do
            old=$(ls -1d "$BASE"/*/ 2>/dev/null | sort | head -1)
            [ -n "$old" ] || break
            rm -rf "$old"
            n=$((n-1))
        done
        DIR="$BASE/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$DIR" || exit 1
        # Daemonize: system.up commands must return within their deadline.
        sample_loop "$DIR" </dev/null >>"$DIR/logger.log" 2>&1 &
        echo $! > "$PIDFILE"
        echo "walkout(gs): logging to $DIR (pid $(cat "$PIDFILE"))"
        ;;
    stop)
        if [ -f "$PIDFILE" ]; then
            kill "$(cat "$PIDFILE")" 2>/dev/null
            rm -f "$PIDFILE"
            echo "walkout(gs): stopped"
        fi
        ;;
    status)
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "walkout(gs): running ($(cat "$PIDFILE")), latest: $(ls -1d "$BASE"/*/ 2>/dev/null | sort | tail -1)"
        else
            echo "walkout(gs): not running"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"; exit 1 ;;
esac
exit 0
