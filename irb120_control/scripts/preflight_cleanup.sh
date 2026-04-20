#!/usr/bin/env bash
# Pre-flight cleanup — runs synchronously before the main bringup stack starts.
# Kills stale processes from a previous unclean shutdown, resets the RealSense
# USB device, and clears stale DDS shared memory.
set -euo pipefail

log() { echo "[preflight] $*"; }

# ── 1. Kill stale ROS2/RWS/EGM processes ─────────────────────────────────────
# Snapshot PIDs that exist RIGHT NOW (before this launch) so we don't kill
# anything the parent launch system already started.
log "Stopping stale processes from previous session..."

for pattern in ros2_control_node rws_client egm_handler servo_node netft_preprocessor net_ft_node; do
    while IFS= read -r pid; do
        [ -z "$pid" ] && continue
        log "  SIGTERM -> $pattern (pid $pid)"
        kill -SIGTERM "$pid" 2>/dev/null || true
    done < <(pgrep -f "$pattern" 2>/dev/null || true)
done

sleep 1.0

for pattern in ros2_control_node rws_client egm_handler servo_node netft_preprocessor net_ft_node; do
    while IFS= read -r pid; do
        [ -z "$pid" ] && continue
        log "  SIGKILL -> $pattern (pid $pid)"
        kill -SIGKILL "$pid" 2>/dev/null || true
    done < <(pgrep -f "$pattern" 2>/dev/null || true)
done

# ── 2. Stop and restart the ROS2 daemon cleanly ───────────────────────────────
log "Restarting ros2 daemon..."
ros2 daemon stop 2>/dev/null || true
sleep 0.5
ros2 daemon start 2>/dev/null || true

# ── 3. Clear stale DDS shared memory ─────────────────────────────────────────
log "Clearing stale DDS shared memory..."
rm -f /dev/shm/fastrtps_* /dev/shm/*fastdds* /dev/shm/*cyclone* 2>/dev/null || true

# ── 4. Reset RealSense USB connection ────────────────────────────────────────
log "Resetting RealSense USB device..."

RS_DEVNAME=""
for d in /sys/bus/usb/devices/*/; do
    [ -f "$d/idVendor" ] && [ -f "$d/idProduct" ] || continue
    v=$(cat "$d/idVendor" 2>/dev/null) || continue
    p=$(cat "$d/idProduct" 2>/dev/null) || continue
    case "$v:$p" in
        8086:0b07|8086:0b3a|8086:0b01|8086:0b5c|8086:0b64)
            RS_DEVNAME=$(basename "$d")
            break
            ;;
    esac
done

if [ -n "$RS_DEVNAME" ]; then
    log "Found RealSense at $RS_DEVNAME — triggering USB unbind/rebind"
    echo "$RS_DEVNAME" | sudo tee /sys/bus/usb/drivers/usb/unbind > /dev/null 2>&1 || true
    sleep 0.5
    echo "$RS_DEVNAME" | sudo tee /sys/bus/usb/drivers/usb/bind > /dev/null 2>&1 || true
    sleep 1.5
    log "RealSense USB reset complete"
else
    log "RealSense not found on USB — skipping reset"
fi

log "Pre-flight cleanup complete."
