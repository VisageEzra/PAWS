"""
PAWS Detection Engine
======================
Camera + YOLO inference subprocess launched by blynk_controller.py.

Communicates results to the controller via tagged stdout lines:
    [V6] DD/MM/YYYY hh:mm AM/PM - <event>   — detection event for the log
    [PIN_UPDATE] <pin> <value>               — virtual pin write
    [GALLERY] <pin> <url>                    — snapshot URL for image gallery
    [STREAM_URL] <pin> <url>                 — MJPEG stream URL for video widget

Hardware:
    PIR sensor  BCM 4  (Physical 7)  — triggers the 60 s detection window
    Alert LED   BCM 17 (Physical 11) — blinks fast for 15 s on animal detection

Behaviour:
    1. System starts in STANDBY — stream runs, YOLO is idle.
    2. PIR fires → 60 s ACTIVE window — YOLO runs on every captured frame.
    3. YOLO confirms animal → evidence photo saved, ImgBB upload, Blynk log,
       LED blinks at 5 Hz for 15 s.
    4. No new PIR trigger for CAMERA_ACTIVE_SEC + SLEEP_TIMEOUT_SEC → exit
       (blynk_controller sets status back to SLEEP).

Threading model:
    capture thread   — reads frames at camera speed; polls PIR; publishes MJPEG
    inference thread — runs YOLO only during the active window; fires events
    flask thread     — Werkzeug MJPEG server
"""

import base64
import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional

# Cap inference threads BEFORE cv2/ultralytics import (thread pools size at load
# time). On the Pi we use 2 of 4 cores for inference, leaving 2 cores free for
# the camera-capture + JPEG-encode + Flask stream pipeline so the live stream
# stays smooth. Mirrors config.INFER_THREADS (can't import config yet — it pulls
# in heavy modules — so re-derive the Pi check here cheaply).
import platform as _plat
_is_pi = _plat.machine().startswith(("aarch64", "arm"))
os.environ.setdefault("OMP_NUM_THREADS", "2" if _is_pi else "4")

import cv2
import requests
from flask import Flask, Response
from ultralytics import YOLO

from config import (
    IS_PI,
    IMGBB_API_KEY,
    CAM_WIDTH, CAM_HEIGHT, CAM_AUTO_EXPOSURE,
    DETECT_CONFIDENCE,
    ABSENT_RESET_SEC,
    NOTIFY_REPEAT_SEC,
    SLEEP_TIMEOUT_SEC,
    FLASK_PORT,
    STREAM_WIDTH, STREAM_HEIGHT, STREAM_QUALITY,
    INFER_IMGSZ, CAPTURE_FPS, GALLERY_JPEG_QUALITY,
    CUSTOM_ANIMAL_IDS, CLASS_CONFIDENCE, MAX_BOX_FRAC,
    CUSTOM_NAME_OVERRIDES,
    MODEL_CUSTOM, MODEL_PERSON,
    PERSON_VETO_ENABLED, PERSON_IMGSZ, PERSON_CONF, PERSON_VETO_OVERLAP,
    COCO_EXTRA_CLASSES, COCO_EXTRA_CONF, COCO_EXTRA_CONF_BY_CLASS,
    COCO_OVERRIDE_IOU, COCO_WILDLIFE_MARGIN,
    CONFIRM_FRAMES,
    DETECTIONS_DIR,
    PIR_PIN, LED_PIN,
    POWER_PERSIST_RESIDENT, STANDBY_POLL_SEC, STREAM_ON_DEMAND,
    THERMAL_GUARD_ENABLED, THERMAL_PAUSE_C, THERMAL_RESUME_C, THERMAL_LOG_SEC,
    CAMERA_ACTIVE_SEC, LED_BLINK_SEC, LED_BLINK_HZ,
    PIN_ALERT_GALLERY, PIN_STREAM, PIN_STATS,
)

# ─────────────────── GPIO SETUP ───────────────────
_GPIO = None
_gpio_ok = False

if IS_PI:
    try:
        import RPi.GPIO as _GPIO
        _GPIO.setmode(_GPIO.BCM)
        _GPIO.setup(PIR_PIN, _GPIO.IN)
        _GPIO.setup(LED_PIN, _GPIO.OUT)
        _GPIO.output(LED_PIN, _GPIO.LOW)
        _gpio_ok = True
    except Exception as _exc:
        print(f"LOG: [WARN] GPIO setup failed ({_exc}) — PIR/LED disabled; camera runs continuously", flush=True)

# ─────────────────── SHARED STATE ───────────────────
_stop = threading.Event()

# Raw frame: capture thread → inference thread
_raw_lock  = threading.Lock()
_latest_raw: Optional[cv2.typing.MatLike] = None
_raw_event  = threading.Event()

# Encoded JPEG: capture thread → Flask thread
_frame_lock  = threading.Lock()
_latest_jpeg: Optional[bytes] = None
_frame_event  = threading.Event()
# Number of connected MJPEG stream clients — when 0, we skip JPEG encoding to save
# power (on-demand streaming). Plain int; ±1 under _viewers_lock, read lock-free.
_viewers      = 0
_viewers_lock = threading.Lock()

# Inference results: inference thread → capture thread
_cache_lock   = threading.Lock()
_cached_boxes: list           = []
_cached_animal: Optional[str] = None

# PIR active-window state (written by capture thread, read by inference thread)
# _active_until: monotonic deadline — YOLO runs while now < _active_until
# _last_pir_trigger: monotonic time of the most recent PIR HIGH reading (None = never)
_active_until: float = 0.0
_last_pir_trigger: Optional[float] = None

# LED guard — prevents overlapping blink threads
_led_busy = threading.Event()


# ─────────────────── GPIO HELPERS ───────────────────

def _led_blink(duration: float = LED_BLINK_SEC) -> None:
    """Blink the alert LED at LED_BLINK_HZ for `duration` seconds.

    Non-reentrant: if a blink is already running, the call returns immediately.
    """
    if not _gpio_ok or _led_busy.is_set():
        return
    _led_busy.set()
    half = 0.5 / max(LED_BLINK_HZ, 1)
    end  = time.monotonic() + duration
    try:
        while time.monotonic() < end and not _stop.is_set():
            _GPIO.output(LED_PIN, _GPIO.HIGH)
            time.sleep(half)
            _GPIO.output(LED_PIN, _GPIO.LOW)
            time.sleep(half)
    finally:
        try:
            _GPIO.output(LED_PIN, _GPIO.LOW)
        except Exception:
            pass
        _led_busy.clear()


# ─────────────────── THERMAL / HEALTH HELPERS ───────────────────
def _cpu_temp_c() -> float:
    """CPU temperature in °C from the kernel thermal zone (-1.0 if unavailable)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as fh:
            return int(fh.read().strip()) / 1000.0
    except Exception:
        return -1.0


def _undervoltage() -> bool:
    """True if firmware reports under-voltage (now or since boot) — supply sag."""
    try:
        import subprocess
        out = subprocess.run(["vcgencmd", "get_throttled"],
                             capture_output=True, text=True, timeout=3).stdout
        val = int(out.strip().split("=")[1], 16)
        return bool(val & 0x1) or bool(val & 0x10000)
    except Exception:
        return False


# ─────────────────── ANNOTATION HELPERS ───────────────────
_BOX_COLOR = (0, 200, 0)
_BANNER_BG = (0, 0, 200)
_BANNER_FG = (255, 255, 255)
_BANNER_H  = 52


def _draw_boxes(frame: cv2.typing.MatLike, boxes: list) -> None:
    for x1, y1, x2, y2, label, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        cv2.putText(
            frame, f"{label} {conf:.2f}",
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _BOX_COLOR, 2,
        )


def _draw_alert_banner(frame: cv2.typing.MatLike, label_str: str) -> None:
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, _BANNER_H), _BANNER_BG, -1)
    cv2.putText(
        frame, "ANIMAL DETECTED",
        (8, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, _BANNER_FG, 2,
    )


def _extract_boxes(results, animal_ids: frozenset, names: dict, overrides: dict) -> list:
    """Keep only target classes that clear their per-class confidence threshold.

    (The box-size sanity filter runs later, after the COCO second opinion, so a
    legitimately close large animal can be corroborated before being dropped.)"""
    out = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in animal_ids:
                continue
            conf = float(box.conf[0])
            if conf < CLASS_CONFIDENCE.get(cls_id, DETECT_CONFIDENCE):
                continue
            xyxy  = box.xyxy[0].tolist()
            label = overrides.get(cls_id, names[cls_id])
            out.append((
                int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]),
                label, conf,
            ))
    return out


# ─────────────────── COCO SECOND OPINION (person veto + species correction) ──
# Custom-model labels that are the model's generic "I see a quadruped" fallback.
# A confident COCO large-animal label is allowed to overwrite these outright.
_GENERIC_LABELS  = {"dog", "cat"}
# Specific target species — protected from COCO unless COCO clearly out-scores them.
_WILDLIFE_LABELS = {"monkey", "boar", "monitor_lizard", "tapir", "tiger", "black_panther"}


def _coco_second_opinion(model_person: Optional[YOLO], frame) -> tuple:
    """One COCO pass → (persons, extras).

    persons : [(x1,y1,x2,y2), ...]                      — for the person veto
    extras  : [(x1,y1,x2,y2,label,conf), ...]           — COCO_EXTRA_CLASSES animals
    """
    if model_person is None:
        return [], []
    want = [0] + list(COCO_EXTRA_CLASSES)
    # Run the model at the LOWEST threshold any class needs, then apply the real
    # per-class bar below — otherwise a class with a sub-floor threshold would be
    # pre-filtered away by the model call before we ever see it.
    conf_floor = min([PERSON_CONF, COCO_EXTRA_CONF, *COCO_EXTRA_CONF_BY_CLASS.values()])
    try:
        results = model_person(
            frame, imgsz=PERSON_IMGSZ, conf=conf_floor,
            classes=want, verbose=False,
        )
    except Exception as exc:
        print(f"LOG: [COCO] Second-opinion model error (ignored): {exc}", flush=True)
        return [], []
    persons, extras = [], []
    for r in results:
        for box in r.boxes:
            cid  = int(box.cls[0])
            conf = float(box.conf[0])
            x    = box.xyxy[0].tolist()
            if cid == 0 and conf >= PERSON_CONF:
                persons.append((x[0], x[1], x[2], x[3]))
            elif cid in COCO_EXTRA_CLASSES and conf >= COCO_EXTRA_CONF_BY_CLASS.get(cid, COCO_EXTRA_CONF):
                extras.append((int(x[0]), int(x[1]), int(x[2]), int(x[3]),
                               COCO_EXTRA_CLASSES[cid], conf))
    return persons, extras


def _box_overlap(a: tuple, b: tuple) -> float:
    """Max of IoU and the containment fraction of the smaller box."""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    iou      = inter / max(1, area_a + area_b - inter)
    contain  = inter / max(1, min(area_a, area_b))
    return max(iou, contain)


def _suppress_dog_under_wildlife(boxes: list) -> list:
    """If the custom model fired both a wildlife class and 'dog'/'cat' on the same
    animal, keep the specific wildlife label and drop the generic one."""
    wild = [b for b in boxes if b[4] in _WILDLIFE_LABELS]
    if not wild:
        return boxes
    return [
        b for b in boxes
        if not (b[4] in _GENERIC_LABELS
                and any(_box_overlap(b, w) >= COCO_OVERRIDE_IOU for w in wild))
    ]


def _merge_coco_extras(custom_boxes: list, extras: list) -> list:
    """Fold COCO large-animal detections (cow/horse/…) into the custom boxes.

    A COCO extra overlapping a custom box overrides it when the custom label is a
    generic fallback (dog/cat), or when COCO beats a wildlife label by the margin.
    Non-overlapping extras are added (animals the custom model missed)."""
    result = list(custom_boxes)
    for ex in extras:
        elabel, econf = ex[4], ex[5]
        idx = next((i for i, cb in enumerate(result)
                    if _box_overlap(cb, ex) >= COCO_OVERRIDE_IOU), -1)
        if idx < 0:
            result.append(ex)
            continue
        clabel, cconf = result[idx][4], result[idx][5]
        if clabel in _GENERIC_LABELS or econf >= cconf + COCO_WILDLIFE_MARGIN:
            result[idx] = ex
    return result


# COCO large-animal labels — these are corroborated by the COCO model, so a
# near-full-frame box carrying one of them is a real close animal, not a lens FP.
_COCO_EXTRA_LABELS = set(COCO_EXTRA_CLASSES.values())


def _resolve_labels(labels: set) -> set:
    """For the event log: once COCO has supplied a specific large-animal label
    (cow/horse/…), drop the custom model's generic 'dog'/'cat' fallback so the
    log shows the CORRECTED species rather than the raw YOLO guess."""
    if labels & _COCO_EXTRA_LABELS:
        return (labels - _GENERIC_LABELS) or labels
    return labels


def _drop_oversized(boxes: list, frame) -> list:
    """Remove near-full-frame boxes (something pressed against the lens, e.g. a
    head) UNLESS a COCO large-animal label corroborates it as a real close animal."""
    h, w = frame.shape[:2]
    limit = MAX_BOX_FRAC * w * h
    kept = []
    for b in boxes:
        oversized = (b[2] - b[0]) * (b[3] - b[1]) >= limit
        if oversized and b[4] not in _COCO_EXTRA_LABELS:
            continue
        kept.append(b)
    return kept


def _overlaps_person(animal_box: tuple, persons: list) -> bool:
    """True if PERSON_VETO_OVERLAP of the animal box lies inside any person box.

    Containment (not IoU) is used because the false-positive box is usually a
    small patch of a human (hair / face) sitting well inside the full person box.
    """
    ax1, ay1, ax2, ay2 = animal_box[:4]
    area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    for px1, py1, px2, py2 in persons:
        ix1, iy1 = max(ax1, px1), max(ay1, py1)
        ix2, iy2 = min(ax2, px2), min(ay2, py2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter / area >= PERSON_VETO_OVERLAP:
            return True
    return False


# ─────────────────── FLASK MJPEG SERVER ───────────────────
_flask_app = Flask(__name__)


@_flask_app.route("/video_feed")
def _video_feed():
    def _gen():
        global _viewers
        with _viewers_lock:
            _viewers += 1                      # capture loop resumes encoding
        try:
            while not _stop.is_set():
                _frame_event.wait(timeout=0.5)
                _frame_event.clear()
                with _frame_lock:
                    jpeg = _latest_jpeg
                if jpeg is None:
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        finally:
            with _viewers_lock:
                _viewers = max(0, _viewers - 1)  # last viewer leaves → stop encoding
    return Response(_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _start_flask() -> None:
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    _flask_app.run(host="0.0.0.0", port=FLASK_PORT, threaded=True, use_reloader=False)


def _get_local_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ─────────────────── IMGBB UPLOAD ───────────────────
def _upload_imgbb(img_path: str) -> str:
    try:
        with open(img_path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("utf-8")
        resp = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": IMGBB_API_KEY, "image": encoded},
            timeout=15,
        )
        result = resp.json()
        if result.get("success"):
            return result["data"]["display_url"]
        print(f"LOG: [IMGBB] Upload rejected: {result.get('error', {}).get('message', '?')}", flush=True)
    except Exception as exc:
        print(f"LOG: [IMGBB] Upload failed: {exc}", flush=True)
    return ""


def _snapshot_and_upload(frame, boxes: list, label_str: str) -> None:
    """Annotate → encode → save → upload an evidence snapshot to the Blynk gallery.

    Runs in a background thread so the inference loop never blocks on the JPEG
    encode, disk write, or (slow) network upload. `frame` must be a private copy.
    """
    try:
        _draw_boxes(frame, boxes)
        _draw_alert_banner(frame, label_str)
        snap_path = os.path.join(DETECTIONS_DIR, f"det_{int(time.time() * 1000)}.jpg")
        cv2.imwrite(snap_path, frame, [cv2.IMWRITE_JPEG_QUALITY, GALLERY_JPEG_QUALITY])
        url = _upload_imgbb(snap_path)
        if url:
            print(f"[GALLERY] {PIN_ALERT_GALLERY} {url}", flush=True)
    except Exception as exc:
        print(f"LOG: [GALLERY] Snapshot/upload failed: {exc}", flush=True)


# ─────────────────── CAMERA SETUP ───────────────────
def _open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, CAM_AUTO_EXPOSURE)
    except Exception:
        pass
    return cap


# ─────────────────── CAPTURE THREAD ───────────────────
def _capture_loop(cap: cv2.VideoCapture) -> None:
    """Read camera frames, poll PIR to manage the active window, publish MJPEG."""
    global _latest_raw, _latest_jpeg, _active_until, _last_pir_trigger

    publish_interval = 1.0 / max(CAPTURE_FPS, 1)
    next_publish     = time.monotonic()
    fail_count       = 0
    _was_active      = False   # tracks active→standby transitions for logging

    while not _stop.is_set():
        now = time.monotonic()

        # ── PIR poll: extend the CAMERA_ACTIVE_SEC window on each HIGH reading ──
        if _gpio_ok:
            try:
                pir = bool(_GPIO.input(PIR_PIN))
            except Exception:
                pir = False

            if pir:
                _last_pir_trigger = now
                _active_until     = now + CAMERA_ACTIVE_SEC
                if not _was_active:
                    _was_active = True
                    print(
                        f"LOG: [PIR] Motion detected — YOLO active for {CAMERA_ACTIVE_SEC} s",
                        flush=True,
                    )
            elif _was_active and now >= _active_until:
                _was_active = False
                print("LOG: [PIR] Active window expired — standing by for next trigger", flush=True)

        # ── Power gate ──
        # active   : inside the PIR window → feed the inference thread
        # watching : a client is viewing the MJPEG stream → encode for it
        # Neither → deep standby: poll PIR only, let the CPU sleep (no read/encode).
        active   = (not _gpio_ok) or (now < _active_until)
        watching = (_viewers > 0) or not STREAM_ON_DEMAND
        if not active and not watching:
            time.sleep(STANDBY_POLL_SEC)
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            fail_count += 1
            if fail_count == 1 or fail_count % 50 == 0:
                print(f"LOG: [WARN] Camera read failed (#{fail_count})", flush=True)
            if fail_count > 300:
                print("LOG: [ERROR] Camera unresponsive — aborting.", flush=True)
                _stop.set()
                break
            time.sleep(0.05)
            continue
        fail_count = 0

        # Forward latest raw frame to the inference thread (cheap pointer swap)
        with _raw_lock:
            _latest_raw = frame
        _raw_event.set()

        # Encode for the MJPEG stream ONLY while a client is watching (on-demand).
        if not watching:
            continue

        # Rate-limited JPEG encoding for MJPEG stream
        if now < next_publish:
            continue
        next_publish = now + publish_interval

        with _cache_lock:
            boxes  = list(_cached_boxes)
            animal = _cached_animal

        if boxes or animal:
            display = frame.copy()
            if boxes:
                _draw_boxes(display, boxes)
            if animal:
                _draw_alert_banner(display, animal)
        else:
            display = frame

        stream = cv2.resize(display, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_AREA)
        ok2, buf = cv2.imencode(".jpg", stream, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
        if ok2:
            with _frame_lock:
                _latest_jpeg = buf.tobytes()
            _frame_event.set()


# ─────────────────── DETECTION STATE ───────────────────
class _DetectionState:
    def __init__(self) -> None:
        self.total:          int   = 0
        self.animal_present: bool  = False
        self.last_seen_ts:   float = 0.0
        self.last_activity_ts: float = time.monotonic()
        self.consec_hits:    int   = 0   # consecutive frames with a detection (for confirmation)
        self.last_notify_ts: float = 0.0  # when the last Blynk push fired (for repeat-while-present)
        self.crossing_labels: set[str] = set()  # all species seen during the current crossing
        self.reported_labels: set[str] = set()  # species already announced for this crossing


# ─────────────────── INFERENCE THREAD ───────────────────
def _inference_loop(
    model_custom:  YOLO,
    model_person:  Optional[YOLO],
    state:         _DetectionState,
) -> None:
    """Run YOLO within the PIR-triggered window; fire events and blink LED on detections.

    Outside the active window:
      - boxes are cleared (stream shows clean frames)
      - if no new PIR trigger for CAMERA_ACTIVE_SEC + SLEEP_TIMEOUT_SEC, subprocess exits
    """
    global _cached_boxes, _cached_animal

    processed_frame_id = -1
    thermal_paused     = False   # fanless safety: inference paused to let CPU cool
    last_health_log    = 0.0

    while not _stop.is_set():
        _raw_event.wait(timeout=0.5)
        _raw_event.clear()

        with _raw_lock:
            frame_id = id(_latest_raw)
            frame    = _latest_raw

        if frame is None or frame_id == processed_frame_id:
            continue
        processed_frame_id = frame_id

        now = time.monotonic()

        # ── Active-window gate ──
        # When GPIO is present, only run YOLO during the PIR-triggered window.
        # When GPIO is absent (desktop), always run.
        if _gpio_ok and now >= _active_until:
            with _cache_lock:
                _cached_boxes  = []
                _cached_animal = None

            # Sleep behaviour when the PIR window has been quiet for
            # CAMERA_ACTIVE_SEC + SLEEP_TIMEOUT_SEC.
            if (
                _last_pir_trigger is not None
                and now - _last_pir_trigger > CAMERA_ACTIVE_SEC + SLEEP_TIMEOUT_SEC
            ):
                if not POWER_PERSIST_RESIDENT:
                    print(
                        f"LOG: [PIR] No motion for {CAMERA_ACTIVE_SEC + SLEEP_TIMEOUT_SEC} s"
                        " — detection engine sleeping.",
                        flush=True,
                    )
                    _stop.set()
                    break
                # Resident mode: stay alive, just idle deeper to save power.
                time.sleep(STANDBY_POLL_SEC)
                continue

            time.sleep(0.1)  # yield CPU during standby
            continue

        # ── Thermal guard (fanless) + periodic health log ──
        if THERMAL_GUARD_ENABLED:
            temp = _cpu_temp_c()
            if (now - last_health_log) >= THERMAL_LOG_SEC:
                last_health_log = now
                uv = " ⚠UNDER-VOLTAGE" if _undervoltage() else ""
                print(f"LOG: [HEALTH] CPU {temp:.1f}°C{uv}", flush=True)

            if not thermal_paused and temp >= THERMAL_PAUSE_C:
                thermal_paused = True
                print(f"LOG: [THERMAL] CPU {temp:.1f}°C ≥ {THERMAL_PAUSE_C} — pausing inference to cool", flush=True)
            elif thermal_paused and 0 <= temp <= THERMAL_RESUME_C:
                thermal_paused = False
                print(f"LOG: [THERMAL] CPU {temp:.1f}°C — resuming inference", flush=True)

            if thermal_paused:
                with _cache_lock:
                    _cached_boxes  = []
                    _cached_animal = None
                time.sleep(0.5)  # cool-down; stream still served by capture thread
                continue

        # ── Run models ──
        all_boxes:       list     = []
        detected_labels: set[str] = set()

        try:
            c_results = model_custom(
                frame, imgsz=INFER_IMGSZ, conf=DETECT_CONFIDENCE, verbose=False,
            )
            c_boxes = _extract_boxes(
                c_results, CUSTOM_ANIMAL_IDS, model_custom.names, CUSTOM_NAME_OVERRIDES,
            )
            # Prefer a specific wildlife label over a generic dog/cat on the same animal.
            c_boxes = _suppress_dog_under_wildlife(c_boxes)

            # ── COCO second opinion (one pass, only when the custom model proposed
            #    something): gives both the person veto AND large-animal species
            #    correction (cow/horse/elephant/bear the custom model can't name).
            if c_boxes and model_person is not None:
                persons, extras = _coco_second_opinion(model_person, frame)
                if persons:
                    kept = [b for b in c_boxes if not _overlaps_person(b, persons)]
                    if len(kept) != len(c_boxes):
                        print(
                            f"LOG: [VETO] Suppressed {len(c_boxes) - len(kept)} detection(s)"
                            " overlapping a person",
                            flush=True,
                        )
                    c_boxes = kept
                if extras:
                    merged = _merge_coco_extras(c_boxes, extras)
                    new_labels = {b[4] for b in merged} - {b[4] for b in c_boxes}
                    if new_labels:
                        print(
                            f"LOG: [COCO] Species corrected/added: {', '.join(sorted(new_labels))}",
                            flush=True,
                        )
                    c_boxes = merged

            # Box-size sanity LAST: drop lens-pressed false positives, but keep
            # COCO-corroborated large animals that legitimately fill the frame.
            c_boxes = _drop_oversized(c_boxes, frame)

            all_boxes += c_boxes
            detected_labels.update(lbl for *_, lbl, _ in c_boxes)

        except Exception as exc:
            import traceback
            print(f"LOG: [ERROR] Inference error: {exc}", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            continue

        with _cache_lock:
            _cached_boxes  = all_boxes
            _cached_animal = ", ".join(sorted(detected_labels)) if detected_labels else None

        # ── Presence state machine: one event per crossing ──
        if detected_labels:
            state.last_activity_ts = now
            state.consec_hits     += 1
            # Accumulate every species seen this crossing so a later COCO
            # correction (dog → cow) is reflected even though the alert fires once.
            state.crossing_labels |= detected_labels

            if not state.animal_present and state.consec_hits >= CONFIRM_FRAMES:
                # ── Confirmed first detection of this crossing ──
                # (box already shows on the stream; alert waited for confirmation)
                state.animal_present  = True
                state.last_seen_ts    = now
                state.total          += 1

                report = _resolve_labels(state.crossing_labels)
                state.reported_labels = set(report)
                label_str = ", ".join(sorted(report))
                ts        = datetime.now().strftime("%d/%m/%Y %I:%M %p")
                print(f"[V6] {ts} - {label_str} detected",               flush=True)
                print(f"[PIN_UPDATE] {PIN_STATS} {state.total}",          flush=True)
                # Unique time suffix keeps each push distinct so Blynk's cloud
                # doesn't de-dupe repeats and swallow the alert.
                print(f"[NOTIFY] {label_str} detected at {datetime.now().strftime('%I:%M:%S %p')}", flush=True)
                state.last_notify_ts = now
                print(f"LOG: [DETECT] Event #{state.total}: {label_str}", flush=True)

                # Snapshot + upload entirely off-thread (encode/disk/network never
                # touch the inference loop — only the cheap frame.copy() does).
                threading.Thread(
                    target=_snapshot_and_upload,
                    args=(frame.copy(), list(all_boxes), label_str), daemon=True,
                ).start()

                # Blink LED for LED_BLINK_SEC seconds (background thread)
                threading.Thread(
                    target=_led_blink, args=(LED_BLINK_SEC,), daemon=True,
                ).start()

            elif state.animal_present:
                state.last_seen_ts = now   # animal still visible — extend window

                # ── Repeat the push while the animal stays in view ──
                # Keeps alerts coming the whole time something is on camera,
                # instead of a single notification per crossing.
                if NOTIFY_REPEAT_SEC > 0 and (now - state.last_notify_ts) >= NOTIFY_REPEAT_SEC:
                    label_str = ", ".join(sorted(state.reported_labels)) or "Animal"
                    print(f"[NOTIFY] {label_str} still detected at {datetime.now().strftime('%I:%M:%S %p')}", flush=True)
                    state.last_notify_ts = now

                # ── Species correction: COCO refined the label after the alert ──
                report = _resolve_labels(state.crossing_labels)
                if report - state.reported_labels:
                    state.reported_labels = set(report)
                    label_str = ", ".join(sorted(report))
                    ts        = datetime.now().strftime("%d/%m/%Y %I:%M %p")
                    print(f"[V6] {ts} - corrected: {label_str}",            flush=True)
                    print(f"LOG: [DETECT] Species corrected → {label_str}", flush=True)

                    # Re-upload a corrected evidence photo (gallery now matches the
                    # log). Same off-thread path → no detection slowdown.
                    threading.Thread(
                        target=_snapshot_and_upload,
                        args=(frame.copy(), list(all_boxes), label_str), daemon=True,
                    ).start()

        else:
            state.consec_hits = 0
            if state.animal_present:
                if (now - state.last_seen_ts) > ABSENT_RESET_SEC:
                    state.animal_present  = False
                    state.crossing_labels = set()
                    state.reported_labels = set()
                    print("LOG: [DETECT] Animal cleared — ready for next crossing.", flush=True)
            else:
                # Streak broke before confirmation — drop accumulated labels.
                state.crossing_labels = set()

        # ── Detection-idle exit (desktop / non-resident only) ──
        # In resident mode we never self-exit — the active-window gate above keeps
        # the engine idling at low power until the next PIR trigger.
        if (
            not POWER_PERSIST_RESIDENT
            and not _gpio_ok
            and state.total > 0
            and (now - state.last_activity_ts) > SLEEP_TIMEOUT_SEC
        ):
            print(f"LOG: [DETECT] No activity for {SLEEP_TIMEOUT_SEC} s — exiting.", flush=True)
            _stop.set()
            break


# ─────────────────── MAIN ───────────────────
def main() -> None:
    os.makedirs(DETECTIONS_DIR, exist_ok=True)

    # ── Open camera ──
    cap = _open_camera()
    if not cap.isOpened():
        print("LOG: [ERROR] Could not open camera — aborting.", flush=True)
        sys.exit(1)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"LOG: [DETECT] Camera opened: {actual_w}×{actual_h}", flush=True)

    # ── Load YOLO model (single NCNN custom model — task must be given since
    #    NCNN has no embedded task metadata) ──
    print("LOG: [DETECT] Loading YOLO model...", flush=True)
    model_custom = YOLO(MODEL_CUSTOM, task="detect")
    print(f"LOG: [DETECT] Model ready — classes: {model_custom.names}", flush=True)

    # ── Person-veto model (optional) ──
    model_person = None
    if PERSON_VETO_ENABLED and os.path.exists(MODEL_PERSON):
        try:
            model_person = YOLO(MODEL_PERSON)
            _coco_labels = ", ".join(sorted(COCO_EXTRA_CLASSES.values()))
            print(
                "LOG: [DETECT] COCO second-opinion model loaded "
                f"(person veto + species correction ON: {_coco_labels}).",
                flush=True,
            )
        except Exception as exc:
            print(f"LOG: [WARN] Person-veto model failed to load ({exc}) — veto disabled.", flush=True)
    elif PERSON_VETO_ENABLED:
        print(f"LOG: [WARN] Person-veto enabled but {MODEL_PERSON} missing — veto disabled.", flush=True)

    # ── Warmup: pay the first-inference JIT cost before detection starts ──
    print("LOG: [DETECT] Warming up models...", flush=True)
    ret, warm_frame = cap.read()
    if ret and warm_frame is not None:
        model_custom(warm_frame, imgsz=INFER_IMGSZ, verbose=False)
        if model_person is not None:
            model_person(warm_frame, imgsz=PERSON_IMGSZ,
                         classes=[0] + list(COCO_EXTRA_CLASSES), verbose=False)
        print("LOG: [DETECT] Warmup complete.", flush=True)
    else:
        print("LOG: [WARN] Could not read warmup frame — skipping warmup.", flush=True)

    state = _DetectionState()

    # ── Start threads ──
    threading.Thread(target=_capture_loop,
                     args=(cap,),                                  daemon=True).start()
    threading.Thread(target=_inference_loop,
                     args=(model_custom, model_person, state),     daemon=True).start()
    threading.Thread(target=_start_flask,                          daemon=True).start()

    time.sleep(0.5)   # give Flask a moment to bind the port

    stream_url = f"http://{_get_local_ip()}:{FLASK_PORT}/video_feed"
    print(f"[STREAM_URL] {PIN_STREAM} {stream_url}", flush=True)
    print(f"LOG: [DETECT] Live stream: {stream_url}", flush=True)

    if _gpio_ok:
        print(
            f"LOG: [DETECT] Detection engine ACTIVE"
            f" — PIR BCM {PIR_PIN} | LED BCM {LED_PIN}"
            f" | {CAMERA_ACTIVE_SEC} s window | {LED_BLINK_SEC} s blink",
            flush=True,
        )
        print("LOG: [PIR] Waiting for motion...", flush=True)
    else:
        print(
            "LOG: [DETECT] Detection engine ACTIVE"
            " (no GPIO — YOLO runs continuously)",
            flush=True,
        )

    try:
        _stop.wait()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        if _gpio_ok:
            try:
                _GPIO.output(LED_PIN, _GPIO.LOW)
                _GPIO.cleanup()
            except Exception:
                pass
        cap.release()
        print("LOG: [DETECT] Camera released. Detection engine stopped.", flush=True)


if __name__ == "__main__":
    main()
