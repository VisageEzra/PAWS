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
MODEL_GENERAL = os.path.join(SCRIPT_DIR, "yolov8s.pt")
MODEL_CUSTOM = os.path.join(SCRIPT_DIR, "best.pt")

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

# ─────────────────── CAMERA ───────────────────
# Pi 4 can't handle 1280×720 inference — use 640×480 capture
CAM_WIDTH = 640 
CAM_HEIGHT = 480 
CAM_AUTO_EXPOSURE = 0.75  # 0.75 = auto, 0.25 = manual (OpenCV / DirectShow / V4L2)

# ─────────────────── DETECTION TUNING ───────────────────
DETECT_CONFIDENCE = 0.80  # YOLOv8 minimum confidence threshold
SAVE_COOLDOWN_SEC = 5     # Kept for reference / legacy tests — not used in detection loop
ABSENT_RESET_SEC  = 3.0   # Seconds with no detection before a crossing is considered over

# ─────────────────── MOTION SENSOR (PIR or simulator) ───────────────────
# An event fires only when YOLO finds an animal AND the motion sensor is active.
# "none"      → always inactive — safe default when no PIR is wired up (no events
#               fire at all; live stream still shows YOLO boxes for visualization)
# "framediff" → OpenCV pixel-diff stand-in — DO NOT trust for production; can't
#               distinguish a held-up photo from a real animal
# "gpio"      → real PIR on the Pi (uses gpiozero, defaults to BCM pin 17)
MOTION_SENSOR_BACKEND = "gpio" if IS_PI else "none"
MOTION_THRESHOLD      = 25     # pixel-diff threshold (higher = less sensitive)
MOTION_MIN_AREA       = 500    # min contour area in pixels (filters noise)
MOTION_HOLD_SEC       = 0.5    # report active this long after motion stops — short
                                # window so a stationary photo can't keep the gate open
MOTION_CONSECUTIVE    = 3      # frames of sustained motion needed to flip active —
                                # rejects single pulses like placing a photo in view
MOTION_PIR_PIN        = 17     # BCM pin for the real PIR

# How long without YOLO detections AND without motion before we mark the system idle.
SLEEP_TIMEOUT_SEC = 60

FLASK_PORT = 5000

# MJPEG stream resolution — Pi downscales before encoding (~40% faster encode,
# ~45% smaller payload). Desktop keeps native capture resolution.
STREAM_WIDTH = 480 if IS_PI else CAM_WIDTH
STREAM_HEIGHT = 360 if IS_PI else CAM_HEIGHT

# JPEG quality for the MJPEG stream — Pi sacrifices a bit of fidelity for speed.
STREAM_QUALITY = 55 if IS_PI else 85

# YOLO input resolution: 320 on Pi (fast), 640 on desktop (accurate)
INFER_IMGSZ = 320 if IS_PI else 640

# Run inference every Nth frame (1 = every frame, 3 = skip 2 frames between inferences)
# On Pi, skip frames so the camera stays responsive between slow inferences
INFER_EVERY_N = 3 if IS_PI else 1

# How many frames per second the capture thread is allowed to publish to the
# MJPEG stream. The Pi 4 spends ~5-10 ms per JPEG encode at 640×480 quality 65,
# so 30 FPS would burn a whole core on encoding alone. 15 FPS leaves headroom
# for YOLO while still feeling smooth in the Blynk app.
CAPTURE_FPS = 15 if IS_PI else 30

# Class IDs that count as "animal detected"
GENERAL_ANIMAL_IDS = frozenset({15, 16, 18, 19, 20, 21})
CUSTOM_ANIMAL_IDS = frozenset({0, 1, 3, 5, 6})

# Friendly name overrides for messy training labels
CUSTOM_NAME_OVERRIDES: dict[int, str] = {
    # Add entries like  2: "Wild Boar"  if the raw label is ugly
}


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
