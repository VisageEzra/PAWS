"""
PAWS Detection Engine
======================
Runs YOLOv8 inference on a live camera feed, draws alerts, saves evidence
snapshots, and communicates events to `blynk_controller.py` via stdout tags.

Stdout protocol (consumed by the controller):
    [V6] <msg>                → append to event log terminal
    [PIN_UPDATE] <pin> <v>    → write value to a virtual pin
    [GALLERY] <pin> <url>     → push image URL to a gallery widget
    [STREAM_URL] <pin> <url>  → set Video Streaming widget URL on *pin*
    LOG: ...                  → general log forwarded to V4 terminal

Threading model:
    main thread       — owns the cv2.imshow preview (desktop only) + lifecycle
    capture thread    — reads frames from the camera as fast as it can, overlays
                        cached box+banner annotations, publishes encoded JPEG
    inference thread  — runs YOLO on the most recent raw frame, alternating
                        between general and custom models, caches box coords
    flask thread      — Werkzeug dev server (MJPEG + snapshot endpoints)
"""

import base64
import logging
import os
import socket
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import requests
from flask import Flask, Response
from ultralytics import YOLO

from config import (
    ABSENT_RESET_SEC,
    CAM_AUTO_EXPOSURE,
    CAM_HEIGHT,
    CAM_WIDTH,
    CAPTURE_FPS,
    CUSTOM_ANIMAL_IDS,
    CUSTOM_NAME_OVERRIDES,
    DETECT_CONFIDENCE,
    DETECTIONS_DIR,
    FLASK_PORT,
    GENERAL_ANIMAL_IDS,
    IMGBB_API_KEY,
    INFER_IMGSZ,
    IS_PI,
    MODEL_CUSTOM,
    MODEL_GENERAL,
    MOTION_CONSECUTIVE,
    MOTION_HOLD_SEC,
    MOTION_MIN_AREA,
    MOTION_PIR_PIN,
    MOTION_SENSOR_BACKEND,
    MOTION_THRESHOLD,
    PIN_ALERT_GALLERY,
    PIN_STATS,
    PIN_STATUS,
    PIN_STREAM,
    SLEEP_TIMEOUT_SEC,
    STREAM_HEIGHT,
    STREAM_QUALITY,
    STREAM_WIDTH,
)
from motion_sensor import MotionSensor, create_motion_sensor

# ─────────────────── LOGGING ───────────────────
logging.getLogger("werkzeug").setLevel(logging.ERROR)
log = logging.getLogger("paws")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("LOG: %(message)s"))
log.addHandler(_handler)
log.propagate = False

# ─────────────────── GLOBALS ───────────────────
os.makedirs(DETECTIONS_DIR, exist_ok=True)

_stop = threading.Event()
_frame_event = threading.Event()        # signals a new JPEG is published
_raw_event = threading.Event()          # signals a new raw frame for inference

# Latest published frame (annotated) + its pre-encoded JPEG — read by Flask
_frame_lock = threading.Lock()
_latest_frame: Optional[cv2.typing.MatLike] = None
_latest_jpeg: Optional[bytes] = None

# Raw camera frame shared from capture thread → inference thread
_raw_lock = threading.Lock()
_latest_raw: Optional[cv2.typing.MatLike] = None

# Last inference result — drawn by the capture thread on every fresh frame
_cache_lock = threading.Lock()
_cached_boxes: list[tuple[int, int, int, int, str, float]] = []
_cached_animal: Optional[str] = None




# ─────────────────── FRAME PUBLISHING ───────────────────

def _publish_frame(annotated: cv2.typing.MatLike) -> None:
    """Encode the annotated frame to JPEG once and publish for streaming.

    On the Pi, the frame is downscaled to STREAM_WIDTH × STREAM_HEIGHT before
    encoding — about 40 % less encode time and 45 % smaller JPEG payload, which
    matters far more than the visible resolution drop on a phone screen.
    """
    global _latest_frame, _latest_jpeg

    if annotated.shape[1] > STREAM_WIDTH:
        encoded = cv2.resize(annotated, (STREAM_WIDTH, STREAM_HEIGHT),
                             interpolation=cv2.INTER_AREA)
    else:
        encoded = annotated

    ok, buf = cv2.imencode(".jpg", encoded, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
    if not ok:
        return
    jpeg = buf.tobytes()
    with _frame_lock:
        _latest_frame = encoded
        _latest_jpeg = jpeg
    _frame_event.set()


# ─────────────────── FLASK SERVER ───────────────────
_app = Flask(__name__)


def _generate_mjpeg():
    """Stream pre-encoded JPEG bytes — wait for new frames instead of busy-looping."""
    last = None
    while not _stop.is_set():
        _frame_event.wait(timeout=0.5)
        _frame_event.clear()

        jpeg = _latest_jpeg
        if jpeg is None or jpeg is last:
            continue
        last = jpeg

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )


@_app.route("/stream")
def _serve_stream():
    """Continuous MJPEG live stream — viewable in any browser on LAN."""
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@_app.route("/snapshot.jpg")
def _serve_snapshot():
    """Return the most recent annotated frame as a single JPEG."""
    jpeg = _latest_jpeg
    if jpeg is None:
        return Response("No frame yet", status=503)
    return Response(jpeg, mimetype="image/jpeg")


def _get_local_ip() -> str:
    """Discover this machine's LAN IP via a dummy UDP socket."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _start_flask() -> None:
    ip = _get_local_ip()
    log.info("[SERVER] Live stream (browser): http://%s:%d/stream", ip, FLASK_PORT)
    log.info("[SERVER] Snapshot endpoint:     http://%s:%d/snapshot.jpg", ip, FLASK_PORT)
    _app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False, threaded=True)


# ─────────────────── IPC HELPERS (stdout → controller) ───────────────────

def emit_gallery(pin: str, url: str, *, is_alert: bool = False) -> None:
    """Send an image URL to a Blynk Image Gallery widget via the controller."""
    print(f"[GALLERY] {pin} {url}", flush=True)
    if is_alert:
        log.info("[ALERT] Evidence pushed to pin %s", pin.upper())


def emit_pin(pin: str, value: str) -> None:
    """Ask the controller to write *value* to a virtual pin."""
    print(f"[PIN_UPDATE] {pin} {value}", flush=True)


def emit_event_log(message: str) -> None:
    """Append a line to the V6 event-log terminal via the controller."""
    print(f"[V6] {message}", flush=True)


def emit_stream_url(pin: str, url: str) -> None:
    """Tell the controller to set the Video Streaming widget URL on *pin*."""
    print(f"[STREAM_URL] {pin} {url}", flush=True)


# ─────────────────── IMGBB UPLOAD ───────────────────
_IMGBB_URL = "https://api.imgbb.com/1/upload"
_UPLOAD_RETRIES = 2


def upload_to_imgbb(image_path: str) -> Optional[str]:
    """Upload a JPEG to ImgBB with basic retry logic. Returns the URL or None."""
    for attempt in range(1, _UPLOAD_RETRIES + 1):
        try:
            with open(image_path, "rb") as f:
                payload = base64.b64encode(f.read()).decode("utf-8")
            resp = requests.post(
                _IMGBB_URL,
                data={"key": IMGBB_API_KEY, "image": payload},
                timeout=15,
            )
            if resp.status_code == 200:
                try:
                    return resp.json()["data"]["url"]
                except (KeyError, ValueError) as exc:
                    log.warning("[UPLOAD] Unexpected ImgBB response (attempt %d): %s", attempt, exc)
            else:
                log.warning("[UPLOAD] ImgBB returned HTTP %d (attempt %d)", resp.status_code, attempt)
        except requests.RequestException as exc:
            log.warning("[UPLOAD] ImgBB error (attempt %d): %s", attempt, exc)
        time.sleep(1)
    return None


# ─────────────────── AUTO STREAM URL ───────────────────

def _push_stream_url() -> None:
    """Wait for Flask to become ready, then push the local LAN stream
    URL to the Blynk Video Streaming widget so the app opens it
    automatically — no ngrok or external tunnel needed."""
    ip = _get_local_ip()
    stream_url = f"http://{ip}:{FLASK_PORT}/stream"

    # Wait for Flask to start accepting connections (up to 15 s)
    for _ in range(30):
        if _stop.is_set():
            return
        try:
            requests.get(f"http://{ip}:{FLASK_PORT}/snapshot.jpg", timeout=1)
            break
        except requests.RequestException:
            time.sleep(0.5)
    else:
        log.warning("[STREAM] Flask did not become ready in 15 s — URL not pushed")
        return

    emit_stream_url(PIN_STREAM, stream_url)
    log.info("[STREAM] Pushed LAN stream URL to Blynk: %s", stream_url)


# ─────────────────── ALERT DRAWING ───────────────────
_BANNER_H = 90
_BANNER_BG = (0, 0, 220)
_BANNER_FG = (255, 255, 255)
_BOX_COLOR = (0, 255, 0)


def draw_alert_banner(frame: cv2.typing.MatLike) -> None:
    """Overlay a red 'ANIMAL DETECTED' banner at the top of the frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, _BANNER_H), _BANNER_BG, -1)
    cv2.rectangle(frame, (0, 0), (w, _BANNER_H), _BANNER_FG, 2)
    cv2.putText(
        frame, "!!! ANIMAL DETECTED !!!",
        (int(w * 0.08), 62),
        cv2.FONT_HERSHEY_DUPLEX, 1.4, _BANNER_FG, 3,
    )


def _draw_boxes(
    frame: cv2.typing.MatLike,
    boxes: list[tuple[int, int, int, int, str, float]],
) -> None:
    """Draw cached YOLO boxes on *frame* with cheap cv2 primitives.

    Used by the capture thread to overlay the last inference result on every
    fresh camera frame, ~30× faster than ultralytics' r.plot().
    """
    for x1, y1, x2, y2, label, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        cv2.putText(
            frame, f"{label} {conf:.2f}",
            (x1, max(y1 - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _BOX_COLOR, 2,
        )


# ─────────────────── EVIDENCE CAPTURE ───────────────────

def save_and_upload(frame: cv2.typing.MatLike) -> None:
    """Save a timestamped JPEG and upload it to ImgBB → Blynk alert gallery."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(DETECTIONS_DIR, f"hazard_{ts}.jpg")
    cv2.imwrite(filepath, frame)
    log.info("[CAPTURE] Evidence saved: %s", filepath)

    url = upload_to_imgbb(filepath)
    if url:
        emit_gallery(PIN_ALERT_GALLERY, url, is_alert=True)
    else:
        log.warning("[CAPTURE] Upload to ImgBB failed — evidence saved locally: %s", filepath)


# ─────────────────── MODEL HELPERS ───────────────────

def patch_custom_names(results) -> None:
    """Apply friendly name overrides so r.plot() draws clean labels."""
    for r in results:
        for cls_id, raw_name in list(r.names.items()):
            if cls_id in CUSTOM_NAME_OVERRIDES:
                r.names[cls_id] = CUSTOM_NAME_OVERRIDES[cls_id]
            elif "boar" in raw_name.lower():
                r.names[cls_id] = "Boar"


def flush_stale_frames(cap: cv2.VideoCapture, count: int = 2) -> None:
    """Discard frames buffered during slow inference. Kept for compatibility —
    no longer called from the main loop because the capture/inference split
    makes it unnecessary (CAP_PROP_BUFFERSIZE=1 already drops stale frames)."""
    for _ in range(count):
        cap.grab()


def scan_results(results, animal_ids: frozenset[int]) -> Optional[str]:
    """Return the prettified name of the first animal detected, or None."""
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            if cls_id in animal_ids:
                return r.names[cls_id].title()
    return None


def should_run_inference(frame_idx: int, every_n: int) -> bool:
    """Return True if inference should run on this frame index.

    Kept for compatibility / tests — the split-thread design no longer needs
    a frame-skip counter because inference runs in its own thread.
    """
    return frame_idx % every_n == 0


def select_model(frame_idx: int, every_n: int) -> str:
    """Return 'general' or 'custom' based on which model's turn it is."""
    return "general" if (frame_idx // every_n) % 2 == 0 else "custom"


def _extract_boxes(
    results,
    animal_ids: frozenset[int],
) -> list[tuple[int, int, int, int, str, float]]:
    """Pull animal box coords + labels out of YOLO results.

    Returns a list of (x1, y1, x2, y2, label, confidence) tuples that
    _draw_boxes() can paint cheaply onto every captured frame.
    """
    out: list[tuple[int, int, int, int, str, float]] = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            if cls_id not in animal_ids:
                continue
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            out.append((
                int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]),
                r.names[cls_id].title(),
                conf,
            ))
    return out


# ─────────────────── CAMERA HELPERS ───────────────────

def open_camera(is_pi: bool) -> cv2.VideoCapture:
    """Open the camera with the correct backend for the platform."""
    backend = cv2.CAP_V4L2 if is_pi else cv2.CAP_DSHOW
    return cv2.VideoCapture(0, backend)


def configure_camera(cap: cv2.VideoCapture, is_pi: bool) -> None:
    """Apply resolution, exposure, and codec settings to the camera."""
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, CAM_AUTO_EXPOSURE)
    if is_pi:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


def _warm_up_models(cap: cv2.VideoCapture, *models: YOLO) -> None:
    """Run each model once on a real frame to pay the JIT cost up front."""
    ret, frame = cap.read()
    if not ret:
        return
    for m in models:
        m(frame, imgsz=INFER_IMGSZ, verbose=False)


def _tune_pi_threads() -> None:
    """Pin OpenCV and PyTorch to a sensible core split on the Pi 4 (4 cores).

    Defaults oversubscribe: OpenCV grabs all cores for imencode/resize, and
    PyTorch grabs all cores for YOLO. With both running concurrently they fight
    for CPU and both slow down. Explicit allocation: OpenCV gets 1 core for
    encoding, PyTorch gets 2 for inference, leaving 1 core for capture+Flask.
    """
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(2)
    except Exception:
        pass


# ─────────────────── DETECTION BOOKKEEPING ───────────────────

class _DetectionState:
    """Presence-based detection state — one event per crossing, not per second."""

    def __init__(self) -> None:
        self.total: int = 0
        self.animal_present: bool = False  # True while an animal is in the scene
        self.last_seen_ts: float = 0.0     # monotonic time of last positive detection
        self.last_activity_ts: float = time.monotonic()  # last YOLO hit or motion pulse
        self.is_idle: bool = False         # currently in sleep-status mode


def _handle_detection(annotated: cv2.typing.MatLike,
                      animal_name: str,
                      state: _DetectionState) -> None:
    """Log the first detection of a new crossing and queue evidence upload.

    Called only on the absent→present state transition, so no cooldown guard
    is needed here — the presence flag in the inference loop is the gate.
    """
    state.total += 1
    emit_pin(PIN_STATS, str(state.total))

    time_str = datetime.now().strftime("%d/%m/%Y %I:%M %p")
    emit_event_log(f"{time_str} - {animal_name} Detected")
    log.info("[ALERT] %s detected — total: %d", animal_name, state.total)

    threading.Thread(target=save_and_upload, args=(annotated,), daemon=True).start()


# ─────────────────── THREAD LOOPS ───────────────────

def _capture_loop(cap: cv2.VideoCapture, motion: MotionSensor) -> None:
    """Camera → MJPEG: read frames as fast as the camera allows, share each
    with the inference thread, but only encode/publish at CAPTURE_FPS.

    Pi 4 JPEG encoding takes ~5-10 ms per 640×480 frame at quality 65, so
    publishing at the camera's full rate would burn a whole core on encoding
    alone. Inference still gets fresh frames every iteration — only the
    expensive encode+publish path is gated.

    Also drives the motion sensor — the frame-diff backend needs every raw
    frame to compute motion; the GPIO backend ignores it.
    """
    global _latest_raw
    publish_interval = 1.0 / max(CAPTURE_FPS, 1)
    next_publish = time.monotonic()

    while not _stop.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        motion.update(frame)

        # Always feed the inference thread the latest raw frame (cheap).
        with _raw_lock:
            _latest_raw = frame
        _raw_event.set()       # wake inference thread immediately

        # Pace the encode+publish path at CAPTURE_FPS.
        now = time.monotonic()
        if now < next_publish:
            continue
        next_publish = now + publish_interval

        with _cache_lock:
            boxes = _cached_boxes
            animal = _cached_animal

        if boxes or animal:
            display = frame.copy()
            if boxes:
                _draw_boxes(display, boxes)
            if animal:
                draw_alert_banner(display)
        else:
            display = frame

        _publish_frame(display)


def _inference_loop(
    model_general: YOLO,
    model_custom: YOLO,
    motion: MotionSensor,
    state: _DetectionState,
) -> None:
    """Continuously run YOLO on the latest captured frame and cache results
    so the capture thread can overlay them on every fresh frame.

    Waits on `_raw_event` for new frames instead of polling — the capture
    thread sets the event after each cap.read(), so this loop runs only as
    often as needed and never burns idle CPU.

    An event (banner + log + upload) fires only when YOLO finds an animal
    AND the motion sensor is currently active. YOLO results are still cached
    every iteration so the live MJPEG keeps drawing boxes — only the event
    side is gated by motion.
    """
    global _cached_boxes, _cached_animal

    iteration = 0
    last_id: Optional[int] = None

    while not _stop.is_set():
        _raw_event.wait(timeout=0.5)
        _raw_event.clear()

        with _raw_lock:
            frame = _latest_raw

        if frame is None or id(frame) == last_id:
            continue
        last_id = id(frame)

        # Alternate models every iteration.
        if iteration % 2 == 0:
            model, ids, fixup = model_general, GENERAL_ANIMAL_IDS, None
        else:
            model, ids, fixup = model_custom, CUSTOM_ANIMAL_IDS, patch_custom_names

        results = model(
            frame, stream=False, conf=DETECT_CONFIDENCE,
            imgsz=INFER_IMGSZ, verbose=False,
        )
        if fixup is not None:
            fixup(results)

        boxes = _extract_boxes(results, ids)
        animal = scan_results(results, ids)

        now = time.monotonic()
        motion_active = motion.is_active()

        # Boxes always cached (live-view visualization). Banner only when the
        # full AND-gate passes — keeps the "ALERT" overlay aligned with the
        # actual event policy so the user isn't misled by YOLO-only detections.
        with _cache_lock:
            _cached_boxes = boxes
            _cached_animal = animal if (animal and motion_active) else None

        # Track activity for the sleep/wake status emitter below.
        if animal or motion_active:
            state.last_activity_ts = now
            if state.is_idle:
                state.is_idle = False
                emit_pin(PIN_STATUS, "ONLINE")
                log.info("[STATUS] Motion detected — system awake")

        # Event gating: YOLO + motion must AGREE. Fires once per crossing.
        if animal and motion_active:
            state.last_seen_ts = now
            if not state.animal_present:
                state.animal_present = True
                evidence = frame.copy()
                _draw_boxes(evidence, boxes)
                draw_alert_banner(evidence)
                _handle_detection(evidence, animal, state)
        elif state.animal_present and (now - state.last_seen_ts) > ABSENT_RESET_SEC:
            state.animal_present = False
            log.info("[ALERT] Animal cleared the scene — ready for next detection")

        # Sleep status: no animal AND no motion for SLEEP_TIMEOUT_SEC.
        if not state.is_idle and (now - state.last_activity_ts) > SLEEP_TIMEOUT_SEC:
            state.is_idle = True
            emit_pin(PIN_STATUS, "SLEEP")
            log.info("[STATUS] Idle for %ds — entering sleep mode", SLEEP_TIMEOUT_SEC)

        iteration += 1


# ─────────────────── MAIN ───────────────────

def main() -> None:
    log.info("[PAWS] Platform: %s | Loading models...", "Raspberry Pi" if IS_PI else "Windows")
    if IS_PI:
        _tune_pi_threads()
    model_general = YOLO(MODEL_GENERAL)
    model_custom = YOLO(MODEL_CUSTOM)

    cap = open_camera(IS_PI)
    if not cap.isOpened():
        log.error("[CAMERA] Failed to open camera — is it plugged in?")
        sys.exit(1)
    configure_camera(cap, IS_PI)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("[CAMERA] Resolution: %dx%d", actual_w, actual_h)
    log.info("[CONFIG] Inference size: %d | Stream quality: %d", INFER_IMGSZ, STREAM_QUALITY)

    _warm_up_models(cap, model_general, model_custom)

    if MOTION_SENSOR_BACKEND == "gpio":
        motion_kwargs = {"pin": MOTION_PIR_PIN}
    elif MOTION_SENSOR_BACKEND == "framediff":
        motion_kwargs = {"threshold": MOTION_THRESHOLD,
                         "min_area": MOTION_MIN_AREA,
                         "hold_sec": MOTION_HOLD_SEC,
                         "consecutive_required": MOTION_CONSECUTIVE}
    else:
        motion_kwargs = {}
    motion = create_motion_sensor(MOTION_SENSOR_BACKEND, **motion_kwargs)
    log.info("[MOTION] Sensor backend: %s", MOTION_SENSOR_BACKEND)
    if MOTION_SENSOR_BACKEND == "none":
        log.warning("[MOTION] No PIR connected — all detection events suppressed "
                    "(live stream still shows YOLO boxes for visualization)")

    state = _DetectionState()

    threading.Thread(target=_capture_loop, args=(cap, motion), daemon=True).start()
    threading.Thread(target=_inference_loop,
                     args=(model_general, model_custom, motion, state),
                     daemon=True).start()
    threading.Thread(target=_start_flask, daemon=True).start()
    threading.Thread(target=_push_stream_url, daemon=True).start()

    log.info("[PAWS] Animal Detection System Active")

    try:
        if IS_PI:
            # Headless — just wait for shutdown. timeout lets us notice _stop quickly.
            while not _stop.is_set():
                _stop.wait(timeout=1.0)
        else:
            # cv2.imshow must run on the main thread on some platforms.
            while not _stop.is_set():
                with _frame_lock:
                    frame = _latest_frame
                if frame is not None:
                    cv2.imshow("PAWS - Predictive Animal Warning System", frame)
                if cv2.waitKey(33) & 0xFF == ord("q"):
                    break
    finally:
        _stop.set()
        _frame_event.set()       # unblock MJPEG generators waiting on the event
        _raw_event.set()         # unblock inference thread waiting on raw frames

        cap.release()
        if not IS_PI:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
