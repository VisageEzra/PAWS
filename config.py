"""
PAWS — Shared Configuration
============================
Single source of truth for all API keys, pin mappings, thresholds, and paths.
Both `paws_detect.py` and `blynk_controller.py` import from here.

Auto-detects Raspberry Pi vs Windows and adjusts defaults accordingly.

Secrets are loaded in priority order:
  1. OS environment variables  (export BLYNK_AUTH_TOKEN=...)
  2. A `.env` file next to this script  (preferred — see .env.example)
  3. Empty string  (startup will fail at validate_config() with a clear message)
"""

import os
import platform


# ─────────────────── ENV FILE LOADER ───────────────────

def _load_env_file() -> None:
    """Read key=value pairs from .env into os.environ (no-op if file absent)."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


_load_env_file()


# ─────────────────── PLATFORM DETECTION ───────────────────
IS_PI = platform.machine().startswith("aarch64") or platform.machine().startswith("arm")

# ─────────────────── PATHS ───────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTIONS_DIR = os.path.join(SCRIPT_DIR, "detections")
HISTORY_FILE = os.path.join(SCRIPT_DIR, "terminal_history.json")
# Single custom 8-class wildlife model (NCNN, ARM-native — no SIGILL).
# Trained 2026-06-01; replaces the old best.pt + COCO yolov8s.pt pair.
# Classes: 0 monkey, 1 boar, 2 monitor_lizard, 3 tapir,
#          4 tiger, 5 dog, 6 cat, 7 black_panther
MODEL_CUSTOM  = os.path.join(SCRIPT_DIR, "Paws_custom_ncnn_model")

# Lightweight COCO model used ONLY as a "person veto": when the custom model
# proposes an animal, we check for an overlapping person and suppress it. This
# kills human-as-monkey false positives without raising thresholds (which would
# hurt real-animal recall). Runs only on detection events, not every frame.
MODEL_PERSON  = os.path.join(SCRIPT_DIR, "yolov8n.pt")

# ─────────────────── API KEYS ───────────────────
# Set in .env (copy .env.example → .env)
IMGBB_API_KEY: str = os.environ.get("IMGBB_API_KEY", "")
BLYNK_AUTH_TOKEN: str = os.environ.get("BLYNK_AUTH_TOKEN", "")


# ─────────────────── BLYNK VIRTUAL PINS ───────────────────
PIN_STATUS = "v0"         # Labeled Value — ONLINE / SLEEP / OFFLINE
PIN_SWITCH = "V1"         # Switch — ON/OFF control from the app
PIN_ALERT_GALLERY = "v2"  # Image Gallery — permanent evidence snapshots
PIN_STREAM = "v3"         # Video Streaming widget — live MJPEG via direct LAN URL
PIN_TERMINAL = "v4"       # Terminal — controller status logs
PIN_STATS = "v5"          # Value Display / SuperChart — total detections
PIN_EVENT_LOG = "v6"      # Terminal — detection event log

# ─────────────────── BLYNK PUSH NOTIFICATIONS ───────────────────
# Blynk sends a phone push notification by "logging an event". The event must be
# created first in the Blynk Console under your Template → Events, with the same
# event CODE as below and "Push Notifications" enabled on it. PAWS triggers it via
# the HTTP logEvent API once per confirmed animal crossing (same debounce as the
# event log, so you won't get spammed). Set NOTIFY_ENABLED = False to mute pushes
# without touching the console. Override the code in .env if you named it differently.
NOTIFY_ENABLED: bool = os.environ.get("NOTIFY_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
BLYNK_NOTIFY_EVENT: str = os.environ.get("BLYNK_NOTIFY_EVENT", "animal_detected")
# Re-send the push every NOTIFY_REPEAT_SEC seconds while an animal stays in view,
# so you keep getting alerts the whole time it's on camera (not just once). Set to
# 0 to fall back to a single push per crossing.
NOTIFY_REPEAT_SEC: float = float(os.environ.get("NOTIFY_REPEAT_SEC", "5"))

# ─────────────────── CAMERA ───────────────────
# Pi 4 can't handle 1280×720 inference — use 640×480 capture
CAM_WIDTH = 640 
CAM_HEIGHT = 480 
CAM_AUTO_EXPOSURE = 0.75  # 0.75 = auto, 0.25 = manual (OpenCV / DirectShow / V4L2)

# ─────────────────── DETECTION TUNING ───────────────────
DETECT_CONFIDENCE = 0.45  # global floor; per-class overrides below refine this

# Per-class confidence thresholds. Animals the model is very confident about can
# use a lower bar (more sensitive); classes that tend to false-positive get a
# higher bar. Falls back to DETECT_CONFIDENCE for any class not listed.
# Class IDs: 0 monkey 1 boar 2 monitor_lizard 3 tapir 4 tiger 5 dog 6 cat 7 black_panther
CLASS_CONFIDENCE: dict[int, float] = {
    0: 0.45,  # monkey
    1: 0.45,  # boar
    2: 0.50,  # monitor_lizard — slimmer profile, easier to confuse
    3: 0.45,  # tapir
    4: 0.45,  # tiger
    5: 0.55,  # dog   — domestic; raise bar to cut false alarms
    6: 0.55,  # cat   — domestic; raise bar to cut false alarms
    7: 0.45,  # black_panther
}

# ─────────────────── BOX-SIZE SANITY FILTER ───────────────────
# Reject any detection whose box covers more than this fraction of the frame.
# A near-full-frame box almost always means something is pressed against the
# lens (e.g. a person's head/hair up close — the main human false-positive),
# not a genuine animal at PIR-trigger distance. Raise toward 0.9 if you expect
# animals to legitimately fill the frame (very close camera placement).
MAX_BOX_FRAC = 0.80

# ─────────────────── PERSON VETO ───────────────────
# Suppress an animal detection if a person box overlaps it. Cheap because it
# only runs on frames where the custom model already proposed something.
PERSON_VETO_ENABLED = True
PERSON_IMGSZ        = 192    # yolov8n at 192 ≈ 0.2 s on Pi (person-only)
PERSON_CONF         = 0.35   # person detection threshold (kept low to catch partial humans)
# Fraction of the animal box that must lie inside a person box to veto it.
PERSON_VETO_OVERLAP = 0.45

# ─────────────────── COCO SECOND-OPINION (species correction) ───────────────────
# The person-veto model (yolov8n, COCO) already runs on detection events, so in
# the SAME pass we also ask it for animals the custom wildlife model has NO class
# for — it otherwise mislabels them as dog/tapir. Verified on real images: COCO
# detects cow at 0.85–0.94 even at PERSON_IMGSZ (192). This is the FULL COCO
# animal set (minus cat/dog, which the custom model already owns). It stays
# EVENT-GATED: COCO only runs when the custom model already proposed a box, so
# broadening these classes costs no extra power in standby — it just lets an
# already-firing detection be re-labelled to a wider range of species.
#
# ⚠ sheep/tapir caveat: COCO tends to call real tapirs "sheep". The custom model
# has a proper `tapir` class and is PROTECTED here — _merge_coco_extras only lets
# a COCO label override a wildlife label (tapir/tiger/…) if it out-scores it by
# COCO_WILDLIFE_MARGIN, and sheep additionally carries a raised per-class bar
# (COCO_EXTRA_CONF_BY_CLASS) so a low-confidence "sheep" can't mask a tapir.
# COCO IDs → label:
COCO_EXTRA_CLASSES: dict[int, str] = {
    14: "bird",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}
COCO_EXTRA_CONF      = 0.50   # default min confidence to accept a COCO extra-animal label
# Per-class confidence overrides for COCO extras (fall back to COCO_EXTRA_CONF).
# sheep is raised because COCO mislabels tapirs as sheep; bird is raised because
# small/distant birds are a common low-confidence false positive.
COCO_EXTRA_CONF_BY_CLASS: dict[int, float] = {
    18: 0.62,  # sheep — guard against tapir-as-sheep
    14: 0.60,  # bird  — cut small/distant false positives
}
COCO_OVERRIDE_IOU    = 0.45   # overlap (IoU or containment) to treat a COCO box == a custom box
# A confident COCO large-animal label replaces an overlapping custom "dog"/"cat"
# (the model's generic fallback) outright; it overrides a custom WILDLIFE label
# (tapir/tiger/boar/…) only if it beats that label's confidence by this margin.
COCO_WILDLIFE_MARGIN = 0.15

# ─────────────────── TEMPORAL CONFIRMATION ───────────────────
# Require a detection in this many CONSECUTIVE inference frames before firing an
# alert (photo/LED/log). Cuts single-frame flicker false positives. The bounding
# box still shows on the live stream immediately; only the alert waits.
# Set to 1 to fire on the first detection (no confirmation).
CONFIRM_FRAMES = 2

# ─────────────────── INFERENCE THREADS ───────────────────
# Pi 4 has 4 cores. Cap inference at 2 threads so the other 2 cores stay free
# for the camera-capture + JPEG-encode + Flask stream pipeline. Benchmarked:
# 2 vs 4 threads costs only ~0.15 s/inference but keeps the Blynk stream smooth.
INFER_THREADS = 2 if IS_PI else 4
SAVE_COOLDOWN_SEC = 5     # Kept for reference / legacy tests — not used in detection loop
ABSENT_RESET_SEC  = 3.0   # Seconds with no detection before a crossing is considered over

# ─────────────────── GPIO PINS ───────────────────
# Wiring confirmed from camera_motion_test.py:
PIR_PIN = 4   # BCM pin 4  — Physical pin 7  — PIR motion sensor input
LED_PIN = 17  # BCM pin 17 — Physical pin 11 — Alert LED output

# ─────────────────── DETECTION TIMING ───────────────────
# How long the camera keeps YOLO running after the PIR fires. The window EXTENDS
# on every PIR HIGH, so a present animal keeps it alive — this is just how long
# we keep inferring AFTER motion stops. Lower = less inference = less power.
# 30 s is a good solar/battery default (was 60).
CAMERA_ACTIVE_SEC = 30
# How long the LED blinks after an animal is confirmed by YOLO.
LED_BLINK_SEC     = 15
# LED blink rate (flashes per second) during alert.
LED_BLINK_HZ      = 5
# How long after the active window expires (with no new PIR trigger)
# before the detection engine exits and the system returns to SLEEP.
SLEEP_TIMEOUT_SEC = 60

# ─────────────────── THERMAL SAFETY (heatsink-only, no fan) ───────────────────
# Running fanless: if the CPU gets too hot (hot day + frequent PIR triggers),
# briefly PAUSE inference so it can cool, instead of letting the firmware hard-
# throttle (which slows everything down). Hysteresis avoids rapid on/off.
# The Pi soft-throttles at 80 °C and hard-throttles at ~85 °C, so we act earlier.
THERMAL_GUARD_ENABLED = True
THERMAL_PAUSE_C   = 78.0   # pause inference at/above this CPU temperature
THERMAL_RESUME_C  = 70.0   # resume once it cools back to this
THERMAL_LOG_SEC   = 300    # log CPU temp + under-voltage health at least this often

# ─────────────────── POWER MANAGEMENT (solar / powerbank, 24-7) ───────────────────
# Stay RESIDENT and idle the CPU when nothing is happening, instead of exiting on
# idle. Exiting forces blynk_controller to reload the models (~15-20 s CPU spike)
# every quiet cycle and trips its 5-restart give-up limit → system goes dead.
# Resident + idle is both lower power and far more reliable for unattended 24/7.
POWER_PERSIST_RESIDENT = True
# In deep standby (no PIR window, nobody viewing the stream) the capture loop just
# polls the PIR at this interval and lets the CPU sleep — no frame reads/encoding.
# 0.2 s = 5 Hz PIR polling, instant enough for motion while saving power.
STANDBY_POLL_SEC = 0.2
# Only encode + serve the MJPEG stream while a client is actually connected. 24/7
# nobody is usually watching, so this skips the continuous JPEG encode entirely.
STREAM_ON_DEMAND = True

# ─────────────────── POWERBANK ANTI-SHUTOFF KEEP-ALIVE ───────────────────
# Many USB powerbanks (the Baseus H1 20000mAh among them) auto-power-off when the
# connected device draws too little current — they read a near-idle Pi as "device
# unplugged / finished charging" and cut the rail, killing the whole system. All
# the power saving above makes deep standby draw LOW, which makes this MORE likely.
#
# Mitigation: emit a brief CPU load pulse on a fixed interval so the average draw
# never sits flat at the powerbank's auto-off floor. This runs in the ALWAYS-ON
# controller (blynk_controller.py), so it protects every state — switch OFF, SLEEP,
# and deep PIR standby alike. Cost is tiny: a ~120 ms pulse every 20 s is ~0.6%
# duty on one core ≈ negligible energy vs. the blackout it prevents.
#
# Tuning if your H1 STILL cuts off: lower KEEPALIVE_PULSE_SEC (pulse more often)
# and/or raise KEEPALIVE_PULSE_MS (bigger pulse). If it never cuts off, set
# POWERBANK_KEEPALIVE = False to reclaim that sliver of energy.
POWERBANK_KEEPALIVE  = True
KEEPALIVE_PULSE_SEC  = 20    # seconds between load pulses
KEEPALIVE_PULSE_MS   = 120   # duration of each CPU load pulse (milliseconds)

FLASK_PORT = 5000

# MJPEG stream resolution — Pi downscales before encoding (~40% faster encode,
# ~45% smaller payload). Desktop keeps native capture resolution.
STREAM_WIDTH = 480 if IS_PI else CAM_WIDTH
STREAM_HEIGHT = 360 if IS_PI else CAM_HEIGHT

# JPEG quality for the MJPEG stream — Pi sacrifices a bit of fidelity for speed.
STREAM_QUALITY = 55 if IS_PI else 85

# JPEG quality for the evidence snapshot pushed to the Blynk gallery. 85 is
# visually clean but ~half the bytes of the default 95, so the ImgBB upload (and
# therefore how fast the photo appears in the app) is noticeably quicker. This is
# the gallery image only — it has NO effect on detection accuracy.
GALLERY_JPEG_QUALITY = 85

# YOLO input resolution. The model was TRAINED at 320, so 320 gives the best
# accuracy. With only ONE model now (COCO dropped), the Pi runs 320 in ~0.7 s —
# still ~5× faster than the old two-model 160-px pipeline (~3.7 s/cycle).
INFER_IMGSZ = 320 if IS_PI else 640

# Run inference every Nth frame (1 = every frame, 3 = skip 2 frames between inferences)
# On Pi, skip frames so the camera stays responsive between slow inferences
INFER_EVERY_N = 3 if IS_PI else 1

# How many frames per second the capture thread is allowed to publish to the
# MJPEG stream. The Pi 4 spends ~5-10 ms per JPEG encode at 640×480 quality 65,
# so 30 FPS would burn a whole core on encoding alone. 15 FPS leaves headroom
# for YOLO while still feeling smooth in the Blynk app.
CAPTURE_FPS = 15 if IS_PI else 30

# Class IDs that count as "animal detected". All 8 trained classes are wildlife
# targets. To ignore a class (e.g. domestic dog/cat), remove its ID here.
CUSTOM_ANIMAL_IDS = frozenset({0, 1, 2, 3, 4, 5, 6, 7})

# The new model ships clean human-readable labels in metadata.yaml, so no
# overrides are needed. Add entries here only to relabel a class in the UI.
CUSTOM_NAME_OVERRIDES: dict[int, str] = {}


# ─────────────────── HELPERS ───────────────────

def pin_number(pin: str) -> int:
    """Convert 'v5' / 'V5' → 5."""
    return int(pin.lower().removeprefix("v"))


def validate_config() -> None:
    """Raise EnvironmentError if any required secret is missing.

    Call this at the start of main() — not at import time — so that tests
    can import config without needing real API keys.
    """
    missing = [name for name, val in [
        ("IMGBB_API_KEY", IMGBB_API_KEY),
        ("BLYNK_AUTH_TOKEN", BLYNK_AUTH_TOKEN),
    ] if not val]
    if missing:
        raise EnvironmentError(
            f"Missing required secrets: {', '.join(missing)}\n"
            "Copy .env.example → .env and fill in your API keys."
        )
