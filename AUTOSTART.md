# PAWS — Autostart, COCO & Powerbank Notes

This covers the three changes made for unattended, headless, powerbank operation.

---

## 1. Run automatically on boot (no monitor needed)

A systemd service (`paws.service`) starts `blynk_controller.py` as soon as the Pi
powers on and the network is up. The controller then waits for the **V1 switch**
in the Blynk app to start/stop detection — exactly as before, just no terminal or
desktop login required.

**Install (run once):**

```bash
cd ~/Desktop/PAWS_Rasberrypi
bash install_service.sh
```

**Day-to-day:**

```bash
systemctl status paws         # is it running?
journalctl -u paws -f         # live logs
sudo systemctl restart paws   # after editing code/config
sudo systemctl stop paws      # stop (pushes OFFLINE to Blynk first)
```

The service uses the `PAWS_18_5/venv` interpreter, runs as user `paws` (already in
the `video` + `gpio` groups, so camera and PIR/LED work), and restarts only on a
crash — a clean `systemctl stop` stays stopped.

**Recommended for battery/headless:** boot to console instead of the desktop GUI.
The desktop wastes power and you have no monitor anyway:

```bash
sudo systemctl set-default multi-user.target   # console-only boot (biggest power win)
# revert with: sudo systemctl set-default graphical.target
```

---

## 2. COCO species coverage broadened (still event-gated)

The COCO model (`yolov8n`) was already running as an event-gated *second opinion*
(person veto + a few large animals). It now recognises the **full COCO animal
set** the custom model has no class for:

> bird, horse, sheep, cow, elephant, bear, zebra, giraffe

**Battery impact: none in standby.** COCO still only runs when the custom 8-class
model already proposes a box — broadening the *class list* doesn't make it run any
more often, it just lets an already-firing detection be relabelled to more species.

Tuning lives in `config.py`:
- `COCO_EXTRA_CLASSES` — the COCO id→label map (add/remove species here).
- `COCO_EXTRA_CONF` / `COCO_EXTRA_CONF_BY_CLASS` — accept thresholds. `sheep` and
  `bird` carry a raised per-class bar: COCO mislabels real **tapirs** as "sheep",
  so the higher bar + the existing `COCO_WILDLIFE_MARGIN` protection keep a
  low-confidence "sheep" from masking a tapir.

> Note: because COCO is event-gated, an animal the **custom model misses entirely**
> still won't get a COCO label. If you later want COCO to catch animals the custom
> model never sees, it'd have to run every frame (≈2× inference cost) — we kept it
> gated to protect battery, per your choice.

---

## 3. Powerbank anti-shutoff keep-alive (Baseus H1 20000mAh)

Many USB powerbanks switch **off** when the connected device draws too little
current — they read a near-idle Pi as "finished / unplugged" and cut the rail,
killing the whole system overnight. All the power-saving in this project makes
deep standby draw low, which makes that *more* likely.

The always-on controller now emits a brief CPU load pulse on a fixed interval to
keep average draw above the bank's auto-off floor:

- `POWERBANK_KEEPALIVE = True`
- `KEEPALIVE_PULSE_MS = 120` — pulse length
- `KEEPALIVE_PULSE_SEC = 20` — interval

Cost is negligible (~0.6% duty on one core). **If the H1 still cuts off**, lower
`KEEPALIVE_PULSE_SEC` (pulse more often) and/or raise `KEEPALIVE_PULSE_MS`. If it
never cuts off, set `POWERBANK_KEEPALIVE = False` to reclaim that sliver of energy.

**How to test whether your H1 even needs this:** set `POWERBANK_KEEPALIVE = False`,
leave the system in SLEEP (switch OFF) on the powerbank for ~30–60 min, and see if
it stays alive. If it dies, turn the keep-alive back on.

### Rough runtime on the 20000mAh H1

A 20000mAh bank delivers ≈ 60–63 Wh at the 5V USB output (after cell-voltage and
conversion losses). Pi 4 + USB webcam draw, very roughly:

| State                                   | ~Power | ~Runtime  |
|-----------------------------------------|--------|-----------|
| Headless console, mostly SLEEP/standby  | ~3 W   | ~18–20 h  |
| Mixed (occasional PIR triggers)         | ~3.5 W | ~16–18 h  |
| Near-continuous inference (busy site)   | ~6 W   | ~10 h     |

These are estimates — measure your own draw to be sure. Biggest wins: console boot
(item 1), heatsink-only/no fan, and keeping `CAMERA_ACTIVE_SEC` short so noisy PIR
triggers don't pin the system in inference.

> ⚠ Earlier testing flagged firmware **under-voltage** on this Pi
> (`vcgencmd get_throttled` non-zero) — the powerbank's 5V sags under Pi+webcam
> load. Use a short, thick USB cable and, ideally, a bank that holds a solid 5V/3A.
> The keep-alive prevents auto-*shutoff*; it does not fix supply *sag*.
