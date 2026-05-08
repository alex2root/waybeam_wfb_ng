#!/usr/bin/env python3
"""CSA orchestrator (host side).

Sends identical csa_commit frames to two targets:
  - cpe510:5801 (over the wfb radio, reaches vehicle csa_agent on 5801)
  - cpe510:5802 (over LAN, reaches cpe510 csa_agent on 5802)

Both agents schedule `iw set channel` off their own monotonic clock, using
each frame's `dt_to_switch_ms` so all frames in a session resolve to the
same instant. No host/agent clock sync required.
"""
import argparse
import json
import socket
import sys
import time

# 5 GHz DFS channels (UNII-2 + UNII-2-extended). Hopping into these without
# CAC is a regulatory violation in most regions, so refuse by default.
DFS_CHANS = frozenset({
    52, 56, 60, 64,
    100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144,
})


def parse_allowlist(spec: str) -> list[tuple[int, str]]:
    """'149/HT20,153/HT20,161/HT40+' -> [(149,'HT20'), ...]."""
    out: list[tuple[int, str]] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "/" not in piece:
            raise ValueError(f"allowlist entry missing '/': {piece!r}")
        chan_s, ht = piece.split("/", 1)
        out.append((int(chan_s), ht))
    return out


def build_frame(sess: int, seq: int,
                target_chan: int, target_ht: str,
                prev_chan: int, prev_ht: str,
                dt_ms: int, t_revert_ms: int) -> bytes:
    frame = {
        "type": "csa_commit",
        "ver": 1,
        "sess": sess,
        "seq": seq,
        "src": "ground",
        "target_chan": target_chan,
        "target_ht": target_ht,
        "dt_to_switch_ms": dt_ms,
        "t_revert_ms": t_revert_ms,
        "prev_chan": prev_chan,
        "prev_ht": prev_ht,
    }
    return (json.dumps(frame, separators=(",", ":")) + "\n").encode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-chan", type=int, required=True)
    ap.add_argument("--target-ht", default="HT20",
                    choices=["HT20", "HT40+", "HT40-"])
    ap.add_argument("--prev-chan", type=int, default=161)
    ap.add_argument("--prev-ht", default="HT40+",
                    choices=["HT20", "HT40+", "HT40-"])
    ap.add_argument("--cpe510", default="192.168.2.2")
    ap.add_argument("--vehicle-via-wfb-port", type=int, default=5801,
                    help="cpe510 wfb_tx UDP injector port (over-air to vehicle)")
    ap.add_argument("--cpe510-csa-port", type=int, default=5802,
                    help="LAN port of csa_agent on cpe510")
    ap.add_argument("--lead-ms", type=int, default=1000,
                    help="ms from injection start to T_switch")
    ap.add_argument("--t-revert-ms", type=int, default=3000)
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--cadence-ms", type=int, default=20)
    ap.add_argument("--sess", type=int, default=int(time.time()))
    ap.add_argument("--allowlist", default="",
                    help="comma-separated CH/HT pairs (e.g. "
                         "149/HT20,153/HT20,161/HT40+); empty = permissive")
    ap.add_argument("--allow-dfs", action="store_true",
                    help="permit hops into 5GHz DFS channels (52..144)")
    args = ap.parse_args()

    if args.target_chan in DFS_CHANS and not args.allow_dfs:
        print(f"refusing DFS target ch{args.target_chan} "
              f"(pass --allow-dfs to override)", file=sys.stderr)
        return 2
    if args.allowlist:
        try:
            allow = parse_allowlist(args.allowlist)
        except ValueError as e:
            print(f"bad --allowlist: {e}", file=sys.stderr)
            return 2
        if (args.target_chan, args.target_ht) not in allow:
            print(f"target ch{args.target_chan} {args.target_ht} not in "
                  f"allowlist {allow}", file=sys.stderr)
            return 2

    print(f"CSA hop: ch{args.prev_chan} {args.prev_ht} -> "
          f"ch{args.target_chan} {args.target_ht}  "
          f"sess={args.sess} lead={args.lead_ms}ms",
          file=sys.stderr)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    addr_radio = (args.cpe510, args.vehicle_via_wfb_port)
    addr_lan = (args.cpe510, args.cpe510_csa_port)

    t_send0 = time.monotonic()
    cadence_s = args.cadence_ms / 1000.0
    for seq in range(args.n_frames):
        target_send = t_send0 + seq * cadence_s
        now = time.monotonic()
        if target_send > now:
            time.sleep(target_send - now)
        elapsed_ms = int((time.monotonic() - t_send0) * 1000)
        dt_ms = args.lead_ms - elapsed_ms
        body = build_frame(
            args.sess, seq,
            args.target_chan, args.target_ht,
            args.prev_chan, args.prev_ht,
            dt_ms, args.t_revert_ms,
        )
        sock.sendto(body, addr_radio)
        sock.sendto(body, addr_lan)
        print(f"  seq={seq} dt={dt_ms}ms ({len(body)}B) "
              f"-> radio:{addr_radio[1]} lan:{addr_lan[1]}",
              file=sys.stderr)

    elapsed = (time.monotonic() - t_send0) * 1000
    print(f"sent {args.n_frames} frames in {elapsed:.1f}ms; "
          f"T_switch is +{args.lead_ms}ms from t_send0", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
