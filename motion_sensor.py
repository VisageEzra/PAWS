"""
PAWS Motion Sensor
===================
Abstraction over the "is something moving?" signal that gates YOLO events.

In production this is a hardware PIR wired to a Pi GPIO pin. While the PIR
isn't connected yet, `FrameDiffMotion` is a software stand-in that runs the
same interface on top of OpenCV frame differencing — it triggers on any
visible movement in the camera feed.

Swap backends via `MOTION_SENSOR_BACKEND` in config.py.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

import cv2


class MotionSensor(ABC):
    """Returns True when something has moved recently."""

    @abstractmethod
    def is_active(self) -> bool: ...

    def update(self, frame: Optional[cv2.typing.MatLike] = None) -> None:
        """Feed a camera frame. No-op for backends that don't read pixels."""


class FrameDiffMotion(MotionSensor):
    """Software PIR built from frame-to-frame pixel differencing.

    Two-stage filter to reject false positives from holding a static photo:
      1. Requires `consecutive_required` frames of motion in a row to flip
         active — a single pulse (e.g. moving a photo into view) is ignored.
      2. After motion stops, `is_active()` stays True for only `hold_sec`
         seconds — short window so a stationary photo can't keep the gate
         open while YOLO sees the picture.
    """

    def __init__(self, threshold: int = 25, min_area: int = 500,
                 hold_sec: float = 0.5,
                 consecutive_required: int = 3) -> None:
        self._threshold = threshold
        self._min_area = min_area
        self._hold_sec = hold_sec
        self._required = max(1, consecutive_required)
        self._prev_gray: Optional[cv2.typing.MatLike] = None
        self._consecutive_motion: int = 0
        self._last_motion_ts: float = 0.0

    def update(self, frame: Optional[cv2.typing.MatLike] = None) -> None:
        if frame is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return

        delta = cv2.absdiff(self._prev_gray, gray)
        _, thresh = cv2.threshold(delta, self._threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        moved = any(cv2.contourArea(c) >= self._min_area for c in contours)

        if moved:
            self._consecutive_motion += 1
            if self._consecutive_motion >= self._required:
                self._last_motion_ts = time.monotonic()
        else:
            self._consecutive_motion = 0

        self._prev_gray = gray

    def is_active(self) -> bool:
        return (time.monotonic() - self._last_motion_ts) <= self._hold_sec


class NullMotion(MotionSensor):
    """Always inactive — use when no PIR is wired up.

    With this backend the AND-gate in paws_detect never passes, so YOLO can
    still draw boxes on the live stream for visualization, but no events,
    uploads, alerts, or log entries are produced. Safer than the frame-diff
    simulator when you want the system fully silent until real hardware arrives.
    """

    def is_active(self) -> bool:
        return False


class GPIOMotion(MotionSensor):
    """Real PIR on a Raspberry Pi GPIO pin. Requires `gpiozero` on the Pi."""

    def __init__(self, pin: int = 17) -> None:
        from gpiozero import MotionSensor as _GZMotion  # imported lazily for non-Pi hosts
        self._sensor = _GZMotion(pin)

    def is_active(self) -> bool:
        return bool(self._sensor.motion_detected)


def create_motion_sensor(backend: str, **kwargs) -> MotionSensor:
    """Factory — pick the configured backend without leaking GPIO imports off-Pi."""
    backend = backend.lower()
    if backend == "none":
        return NullMotion()
    if backend == "framediff":
        return FrameDiffMotion(**kwargs)
    if backend == "gpio":
        return GPIOMotion(**kwargs)
    raise ValueError(f"Unknown motion sensor backend: {backend!r}")
