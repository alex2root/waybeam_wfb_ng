# Ground station host setup

## NetworkManager: pin the wfb adapters unmanaged (required)

The dual-RTL88x2 ground station needs both monitor-mode adapters kept away from
NetworkManager and wpa_supplicant. Otherwise `gs_supervisor`'s monitor-mode
bring-up races NM and **intermittently** leaves one adapter in `managed` mode
(monitor reverts right after `ip link up`, `iw set channel` then fails with
`-16 busy`). With two identical adapters this failed ~50% of the time.

Install the drop-in (edit the MACs to match your adapters first):

```bash
sudo cp ground/setup/99-wfb-unmanaged.conf /etc/NetworkManager/conf.d/
sudo nmcli general reload
nmcli device status | grep wlx    # both should read: unmanaged
```

Once pinned unmanaged, the bring-up is deterministic and the config `system.up`
no longer needs any `nmcli ... managed no/yes` toggling — `host_x86.json` and
`host_x86_single.json` have had those lines removed accordingly.

Verify a clean bring-up:

```bash
sudo ./build/gs_supervisor.host --config ../config/host_x86.json
# log should show both ifaces reach: type monitor, channel 161
```
