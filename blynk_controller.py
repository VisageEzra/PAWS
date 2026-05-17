"""
PAWS Blynk Controller
======================
Acts as the central bridge between the Blynk mobile app and the PAWS
detection engine (`paws_detect.py`).

Responsibilities:
  • Connects to Blynk Cloud via a persistent TCP socket (BlynkLib)
  • Launches / terminates the detection subprocess on V1 switch events
  • Parses stdout tags from the subprocess and routes them to the
    correct Blynk widgets through a thread-safe queue
  • Persists detection event history to `terminal_history.json`
  • Gracefully sets the system OFFLINE on any shutdown path
"""

import ctypes
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import atexit
from collections import deque
from typing import Optional

import requests
import BlynkLib

from config import (
    BLYNK_AUTH_TOKEN,
    HISTORY_FILE,
    SCRIPT_DIR,
    PIN_STATUS,
    PIN_TERMINAL,
    pin_number,
    validate_config,
)

# ─────────────────── STATE ───────────────────
# heartbeat=45 → tells BlynkLib to allow 45s between server pings
# (default 10s is too aggressive — queue processing blocks run() and causes timeouts)
_blynk = BlynkLib.Blynk(BLYNK_AUTH_TOKEN, server="blynk.cloud", heartbeat=45)
_paws_process: Optional[subprocess.Popen] = None
_process_lock = threading.Lock()         # serialises all _paws_process state changes
_msg_queue: queue.Queue = queue.Queue()

_PAWS_SCRIPT = os.path.join(SCRIPT_DIR, "paws_detect.py")
_MAX_TERMINAL_LINES = 5
_HISTORY_SAVE_INTERVAL = 10  # seconds — debounce disk writes

_v4_history: deque[str] = deque(maxlen=_MAX_TERMINAL_LINES)  # controller log
_v6_history: deque[str] = deque(maxlen=_MAX_TERMINAL_LINES)  # detection event log
_history_db: dict[str, list[str]] = {}  # date → [events]
_history_dirty = False
_history_lock = threading.Lock()

# ─────────────────── BLYNK HTTP HELPERS ───────────────────

def _http_write(pin: str, value: str) -> None:
    """Write to a virtual pin via Blynk's HTTP REST API (fire-and-forget)."""
    try:
        requests.get(
            "https://blynk.cloud/external/api/update",
            params={"token": BLYNK_AUTH_TOKEN, pin: value},
            timeout=5,
        )
    except requests.RequestException:
        pass


def _http_set_property(pin: str, prop: str, value: str) -> None:
    """Set any widget property via Blynk's HTTP property API."""
    try:
        resp = requests.get(
            "https://blynk.cloud/external/api/update/property",
            params={"token": BLYNK_AUTH_TOKEN, "pin": pin, prop: value},
            timeout=5,
        )
        print(f"LOG: [HTTP] setProperty pin={pin} {prop}={value} -> {resp.status_code} {resp.text[:100]}")
    except requests.RequestException as exc:
        print(f"LOG: [HTTP] setProperty FAILED: {exc}")


def _http_set_color(pin: str, hex_color: str) -> None:
    _http_set_property(pin, "color", hex_color)


# ─────────────────── HISTORY PERSISTENCE ───────────────────

def _load_history() -> None:
    """Restore the detection history DB and rebuild the in-memory tail."""
    global _history_db
    try:
        if not os.path.exists(HISTORY_FILE):
            return
        with open(HISTORY_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Handle legacy flat-list format
        if "v6" in data and isinstance(data["v6"], list):
            for item in data["v6"]:
                add_to_db(_history_db, item)
        else:
            _history_db = data

        # Rebuild the rolling tail from the full DB.
        # deque(maxlen=_MAX_TERMINAL_LINES) keeps only the last N entries automatically.
        _v6_history.clear()
        for date_key in sorted(_history_db):
            for entry in _history_db[date_key]:
                _v6_history.append(f"{date_key} {entry}")
    except (json.JSONDecodeError, OSError) as exc:
        print(f"LOG: [WARN] Could not load history: {exc}")


def add_to_db(db: dict, full_msg: str) -> None:
    """Parse 'DD/MM/YYYY hh:mm AM - Event' and insert into the DB dict.

    Pure function — operates on the passed-in dict, no globals.
    """
    parts = full_msg.split(" ", 1)
    if len(parts) != 2:
        return
    date_str, time_and_event = parts
    bucket = db.setdefault(date_str, [])
    if time_and_event not in bucket:
        bucket.append(time_and_event)


def _save_history_now() -> None:
    """Flush the history DB to disk."""
    global _history_dirty
    with _history_lock:
        if not _history_dirty:
            return
        _history_dirty = False
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as fh:
            json.dump(_history_db, fh, indent=4)
    except OSError as exc:
        print(f"LOG: [ERROR] Failed to save history: {exc}")


def _history_flush_worker() -> None:
    """Background thread that periodically flushes dirty history to disk."""
    while True:
        time.sleep(_HISTORY_SAVE_INTERVAL)
        _save_history_now()


# ─────────────────── TERMINAL SYNC ───────────────────

def _sync_terminal(pin_num: int, history: deque, message: str) -> None:
    """Append *message* to a rolling history deque and queue a full redraw."""
    history.append(message)  # deque(maxlen=N) drops the oldest entry automatically

    if pin_num == 6:
        add_to_db(_history_db, message)
        with _history_lock:
            global _history_dirty
            _history_dirty = True

    _msg_queue.put(("terminal_sync", pin_num, list(history)))


def _log(msg: str, *, blocking: bool = False) -> None:
    """Print to local console and push to V4 terminal."""
    print(msg)
    if blocking:
        try:
            _http_write(PIN_TERMINAL, msg + "\n")
        except Exception:
            pass
    else:
        _sync_terminal(pin_number(PIN_TERMINAL), _v4_history, msg)


# ─────────────────── SUBPROCESS LIFECYCLE ───────────────────

def _spawn_paws() -> None:
    """Start paws_detect.py if not already running. Thread-safe."""
    global _paws_process
    with _process_lock:
        if _paws_process is not None:
            return
        proc = subprocess.Popen(
            [sys.executable, "-u", _PAWS_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _paws_process = proc
    threading.Thread(target=_read_subprocess, args=(proc,), daemon=True).start()


def _kill_paws() -> None:
    """Terminate paws_detect.py if running. Thread-safe."""
    global _paws_process
    with _process_lock:
        proc = _paws_process
        _paws_process = None
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _paws_is_running() -> bool:
    """Snapshot read of the subprocess state."""
    with _process_lock:
        return _paws_process is not None


# ─────────────────── SHUTDOWN ───────────────────

def _set_offline() -> None:
    """Push OFFLINE status to Blynk and kill the subprocess."""
    _log("LOG: [BLYNK] Controller shutting down — setting OFFLINE...", blocking=True)
    _kill_paws()
    _save_history_now()
    _http_write(PIN_STATUS, "OFFLINE")
    _http_set_color(PIN_STATUS, "#FF0000")


atexit.register(_set_offline)


def _handle_signal(signum, frame) -> None:
    _set_offline()
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, _handle_signal)

# ── Windows console-close handler (catches clicking the ✕ button) ──
if os.name == "nt":
    _HandlerType = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

    def _console_ctrl(ctrl_type: int) -> bool:
        if ctrl_type == 2:  # CTRL_CLOSE_EVENT
            _log("\nLOG: [BLYNK] Terminal closed — pushing OFFLINE...", blocking=True)
            _set_offline()
            return True
        return False

    # prevent GC of the C callback
    _console_handler_ref = _HandlerType(_console_ctrl)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_handler_ref, True)


# ─────────────────── BLYNK EVENTS ───────────────────

@_blynk.ON("connected")
def _on_connected():
    _log("LOG: [BLYNK] Connected to Blynk Cloud")
    _http_write(PIN_STATUS, "SLEEP")
    _http_set_color(PIN_STATUS, "#D3D3D3")


@_blynk.ON("V1")
def _on_switch(value):
    if value[0] == "1":
        _log("LOG: [BLYNK] Switch ON → waking PAWS...")
        _msg_queue.put(("vw", pin_number(PIN_STATUS), "ONLINE"))
        _http_set_color(PIN_STATUS, "#00FF00")
        _spawn_paws()
    else:
        _log("LOG: [BLYNK] Switch OFF → PAWS entering SLEEP")
        _msg_queue.put(("vw", pin_number(PIN_STATUS), "SLEEP"))
        _http_set_color(PIN_STATUS, "#FFA500")
        _kill_paws()


# ─────────────────── SUBPROCESS READER ───────────────────

def parse_subprocess_line(line: str) -> Optional[tuple]:
    """Parse a tagged stdout line from paws_detect into an action tuple.

    Pure function — returns a tuple describing the action, or None for
    lines that should be forwarded to the V4 terminal.

    Returns:
        ('v6', message)            — event log entry
        ('pin', pin_str, value)    — virtual pin write
        ('gallery', pin, url)      — image gallery push
        None                       — forward to V4 terminal as-is
    """
    if line.startswith("[V6] "):
        return ("v6", line[5:])
    elif line.startswith("[PIN_UPDATE] "):
        parts = line.split(" ", 2)
        if len(parts) == 3:
            return ("pin", parts[1], parts[2])
    elif line.startswith("[GALLERY] "):
        parts = line.split(" ", 2)
        if len(parts) == 3:
            return ("gallery", parts[1], parts[2])
    elif line.startswith("[STREAM_URL] "):
        parts = line.split(" ", 2)
        if len(parts) == 3:
            return ("stream_url", parts[1], parts[2])
    return None


def _read_subprocess(proc: subprocess.Popen) -> None:
    """Parse tagged lines from the detection subprocess and route them."""
    global _paws_process
    for raw_line in iter(proc.stdout.readline, ""):
        try:
            line = raw_line.strip()
            if not line:
                continue

            # Suppress noisy v3 live-feed gallery updates from local console
            if line.startswith("[GALLERY]") and "v3" in line:
                pass
            else:
                print(line)

            action = parse_subprocess_line(line)
            if action is None:
                _sync_terminal(pin_number(PIN_TERMINAL), _v4_history, line)
            elif action[0] == "v6":
                _sync_terminal(6, _v6_history, action[1])
            elif action[0] == "pin":
                _msg_queue.put(("vw", pin_number(action[1]), action[2]))
            elif action[0] == "gallery":
                _msg_queue.put(("gallery", action[1], action[2]))
            elif action[0] == "stream_url":
                # Video Streaming widget doesn't auto-refresh on property change.
                # "Clear then set" forces the player to stop → reinitialise.
                pin_upper = action[1].upper()
                _http_set_property(pin_upper, "url", "")
                time.sleep(1)
                _http_set_property(pin_upper, "url", action[2])
                _log(f"LOG: [STREAM] Live stream active: {action[2]}")
        except Exception as exc:
            print(f"LOG: [ERROR] Failed to process subprocess line: {exc}")

    # Subprocess stdout closed — process exited (crashed or finished normally).
    # Atomically clear _paws_process iff it still refers to *this* proc, so a
    # user-initiated _kill_paws() between the readline EOF and here is respected.
    with _process_lock:
        if _paws_process is proc:
            _paws_process = None
            became_idle = True
        else:
            became_idle = False
    if became_idle:
        print("LOG: [PAWS] Detection process exited — returning to SLEEP")
        _msg_queue.put(("vw", pin_number(PIN_STATUS), "SLEEP"))
        _http_set_color(PIN_STATUS, "#D3D3D3")


# ─────────────────── MAIN LOOP ───────────────────

# Blynk connection states (from BlynkLib source)
_BLK_DISCONNECTED = 0
_BLK_CONNECTED = 2

# Heartbeat: write to V0 every 30s to keep the TCP socket alive
_HEARTBEAT_SEC = 30
# How long to wait before reconnect attempts
_RECONNECT_DELAY = 3


def _force_reconnect() -> None:
    """Tear down the stale socket and establish a fresh connection."""
    try:
        _blynk.disconnect()
    except Exception:
        pass
    # Reset state so connect() doesn't bail with "if state != DISCONNECTED: return"
    _blynk.state = _BLK_DISCONNECTED
    try:
        _blynk.connect()
        print("LOG: [BLYNK] Reconnected successfully!")
    except Exception as exc:
        print(f"LOG: [BLYNK] Reconnect failed: {exc}")


def main() -> None:
    validate_config()
    _load_history()

    # Start background flusher so we don't write JSON on every single detection
    threading.Thread(target=_history_flush_worker, daemon=True).start()

    _log("LOG: [BLYNK] Controller Active — listening for app commands...")

    last_heartbeat = time.time()

    try:
        while True:
            try:
                # ── Only reconnect if truly DISCONNECTED (state=0) ──
                # State 1 (CONNECTING) means the login handshake is in progress — let run() finish it
                if _blynk.state == _BLK_DISCONNECTED:
                    print("LOG: [BLYNK] Connection lost — reconnecting...")
                    _force_reconnect()
                    time.sleep(_RECONNECT_DELAY)

                _blynk.run()

                # ── Heartbeat: ping Blynk every 30s to prevent silent timeout ──
                now = time.time()
                if now - last_heartbeat > _HEARTBEAT_SEC:
                    try:
                        # Write current status to V0 — keeps the TCP socket alive
                        if _paws_is_running():
                            _blynk.virtual_write(pin_number(PIN_STATUS), "ONLINE")
                        else:
                            _blynk.virtual_write(pin_number(PIN_STATUS), "SLEEP")
                        last_heartbeat = now
                    except Exception:
                        # Write failed → socket is dead, force reconnect next iteration
                        print("LOG: [BLYNK] Heartbeat write failed — connection dead")
                        _blynk.state = _BLK_DISCONNECTED
                        continue

                # ── Drain the message queue (max 5 items, then yield back to run()) ──
                items_processed = 0
                while not _msg_queue.empty() and items_processed < 5:
                    item = _msg_queue.get_nowait()
                    action = item[0]

                    try:
                        if action == "vw":
                            _blynk.virtual_write(item[1], item[2])

                        elif action == "terminal_sync":
                            pin, lines = item[1], item[2]
                            _blynk.virtual_write(pin, "clr")
                            _blynk.run()  # let Blynk answer pings between writes
                            time.sleep(0.05)
                            for text in lines:
                                _blynk.virtual_write(pin, text + "\n")
                                time.sleep(0.05)

                        elif action == "gallery":
                            target_pin, image_url = item[1], item[2]
                            try:
                                requests.get(
                                    "https://blynk.cloud/external/api/update/property",
                                    params={"token": BLYNK_AUTH_TOKEN, "pin": target_pin, "urls": image_url},
                                    timeout=5,
                                )
                                time.sleep(0.1)
                                requests.get(
                                    "https://blynk.cloud/external/api/update",
                                    params={"token": BLYNK_AUTH_TOKEN, target_pin: 1},
                                    timeout=5,
                                )
                            except requests.RequestException:
                                pass
                    except Exception:
                        # Any write failure means the socket died mid-queue
                        print("LOG: [BLYNK] Socket write failed during queue drain")
                        _blynk.state = _BLK_DISCONNECTED
                        break

                    items_processed += 1
                    # Keep the socket alive between queue items
                    _blynk.run()
                    time.sleep(0.05)

            except Exception as exc:
                print(f"LOG: [ERROR] Connection error: {exc}")
                _blynk.state = _BLK_DISCONNECTED
                time.sleep(_RECONNECT_DELAY)

    except KeyboardInterrupt:
        _log("\nLOG: [BLYNK] KeyboardInterrupt received.", blocking=True)
    finally:
        _set_offline()


if __name__ == "__main__":
    main()