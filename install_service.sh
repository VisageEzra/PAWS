#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# PAWS autostart installer
# ──────────────────────────────────────────────────────────────────────────────
# Installs paws.service so the detection controller starts automatically when the
# Raspberry Pi powers on — no monitor, keyboard, or login required. Run once:
#
#     bash install_service.sh
#
# (It uses sudo for the system-level steps; you'll be asked for your password.)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_SRC="${PROJECT_DIR}/paws.service"
SERVICE_DST="/etc/systemd/system/paws.service"
VENV_PY="/home/paws/Desktop/PAWS_18_5/venv/bin/python"

echo "PAWS autostart installer"
echo "  project : ${PROJECT_DIR}"
echo "  service : ${SERVICE_DST}"
echo

# ── Sanity checks ────────────────────────────────────────────────────────────
[ -f "${SERVICE_SRC}" ] || { echo "ERROR: ${SERVICE_SRC} not found."; exit 1; }
[ -x "${VENV_PY}" ]     || { echo "ERROR: venv python ${VENV_PY} not found/executable."; exit 1; }
[ -f "${PROJECT_DIR}/.env" ] || echo "WARN: no .env in project dir — controller will fail validate_config() until you create one."

# ── Install + enable ─────────────────────────────────────────────────────────
sudo cp "${SERVICE_SRC}" "${SERVICE_DST}"
sudo systemctl daemon-reload
sudo systemctl enable paws.service
sudo systemctl restart paws.service

echo
echo "✓ paws.service installed and started (and enabled on boot)."
echo
echo "Useful commands:"
echo "  systemctl status paws        # is it running?"
echo "  journalctl -u paws -f        # live logs (Ctrl-C to exit)"
echo "  sudo systemctl restart paws  # restart after editing code/config"
echo "  sudo systemctl stop paws     # stop (sets system OFFLINE in Blynk)"
echo "  sudo systemctl disable paws  # stop auto-starting on boot"
echo
echo "──────────────────────────────────────────────────────────────────────────"
echo "RECOMMENDED for a battery/headless deployment (biggest power saving):"
echo "boot to console instead of the desktop GUI — the desktop wastes power and"
echo "you have no monitor anyway:"
echo
echo "    sudo systemctl set-default multi-user.target   # console-only boot"
echo "    # (revert with: sudo systemctl set-default graphical.target)"
echo
echo "Then reboot. PAWS keeps running headless; control it from the Blynk app."
echo "──────────────────────────────────────────────────────────────────────────"
