#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# PAWS — Raspberry Pi power-saving setup for solar / powerbank 24-7 operation.
#
# Run on the DEPLOYMENT unit:   bash power_setup.sh
# Everything here is reversible and safe. Items are grouped by impact. The big
# wins are 1) running headless and 2) trimming idle peripherals — together they
# can cut idle draw by ~1.5-2.5 W on a Pi 4 (a large fraction of the 24-7 budget).
#
# It does NOT touch the detection code — that is already optimised (resident,
# idles the CPU in standby, streams only on demand).
# ──────────────────────────────────────────────────────────────────────────────
set -u
echo "=== PAWS power setup ==="

# ── 1. Turn the HDMI/display output off (saves ~0.25-0.5 W; no monitor in field) ──
if command -v vcgencmd >/dev/null 2>&1; then
    vcgencmd display_power 0 >/dev/null 2>&1 && echo "[ok] HDMI/display powered off"
fi

# ── 2. Turn off the onboard PWR/ACT LEDs (tiny, but free) ──
for led in /sys/class/leds/{ACT,PWR,led0,led1}; do
    [ -e "$led/brightness" ] && echo 0 | sudo tee "$led/brightness" >/dev/null 2>&1
done
echo "[ok] onboard LEDs off (until reboot)"

# ── 3. WiFi power-save ON (saves ~0.2-0.5 W). NOTE: can add a little latency to
#       the Blynk stream/commands. Comment out if the app feels sluggish. ──
if command -v iw >/dev/null 2>&1; then
    sudo iw dev wlan0 set power_save on >/dev/null 2>&1 && echo "[ok] WiFi power-save on"
fi

# ── 4. Disable Bluetooth if unused (saves ~0.1-0.2 W) ──
sudo systemctl disable --now hciuart        >/dev/null 2>&1
sudo systemctl disable --now bluetooth       >/dev/null 2>&1 && echo "[ok] Bluetooth disabled"

echo ""
echo "=== Applied (runtime). To make 2-4 persist across reboot, add the relevant"
echo "    lines to /etc/rc.local or a @reboot crontab. ==="
echo ""
echo "=== BIGGER WINS — do these manually on the field unit (need a reboot): ==="
cat <<'NOTES'
  • Boot to CONSOLE, not desktop (no GUI / Chromium running 24-7 — this is the
    single biggest idle saver on this Pi):
        sudo raspi-config nonint do_boot_behaviour B2   # console + autologin
    then start PAWS from /etc/rc.local or a systemd service.

  • config.txt (sudo nano /boot/firmware/config.txt) — add, then reboot:
        dtoverlay=disable-bt          # permanently free the BT radio
        # If your camera is USB you don't need the CSI stack; leave camera_auto_detect off.

  • FAN: removed — running heatsink-only (saves ~0.4-1 W). PAWS has a built-in
    thermal guard (config.py THERMAL_*) that pauses inference if the CPU reaches
    78°C, so a fanless unit won't hard-throttle. Watch the [HEALTH] log lines for
    CPU temp + under-voltage in the field.

  • Undervolt/limit clocks for less heat+power (optional, test stability):
        # in config.txt
        arm_freq=1200        # cap from 1800 MHz; inference still ~1 s/frame
        # over_voltage=-2     # only if stable on your board
NOTES
echo "Done."
