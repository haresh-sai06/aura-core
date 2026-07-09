"""Distraction detector — YOLO watches the cabin for a phone (and other distracting objects)
and, the moment one appears, fires a proactive SAFETY EVENT to Aura Core so **Aoede interrupts
the driver and warns them** ("phone down, eyes up"). It also lands on the dashboard's Live
Monitor via the `distraction` bus message.

This runs alongside camera_monitor.py and is deliberately standalone: object detection is a
convenience/attention feature, so it must never block or slow the real-time safety pipeline.

Run (on the camera / System B laptop):
    # point it at Core (System A) over Wi-Fi if needed:
    setx AURA_CORE http://192.168.1.23:8765          # optional, once
    .venv\\Scripts\\python distraction_monitor.py

First run downloads yolov8n (~6 MB). Requires: ultralytics, opencv-python (installed).
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

import cv2  # type: ignore

CORE = os.environ.get("AURA_CORE", "http://127.0.0.1:8765").rstrip("/")
CAM_INDEX = int(os.environ.get("AURA_CAM", "0"))
CONF = float(os.environ.get("AURA_DISTRACT_CONF", "0.45"))
DEBOUNCE_S = float(os.environ.get("AURA_DISTRACT_DEBOUNCE", "9"))

# COCO class ids -> how Aoede should describe the distraction.
WATCH = {
    67: ("phone", "the driver just picked up their phone and is looking down at it"),
    73: ("book", "the driver is reading something instead of watching the road"),
    41: ("cup", "the driver is fumbling with a drink"),
    39: ("bottle", "the driver is reaching for a bottle"),
}


def post_event(text: str) -> None:
    try:
        data = json.dumps({"text": text}).encode()
        req = urllib.request.Request(f"{CORE}/aoede/event", data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
    except Exception as e:
        print(f"[distraction] could not reach Core: {e}")


def main() -> None:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        print("[distraction] ultralytics not installed — run: pip install ultralytics")
        return

    print(f"[distraction] loading YOLOv8n… (Core: {CORE})")
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[distraction] no camera at index {CAM_INDEX}")
        return
    print("[distraction] watching for phones / distracting objects. Ctrl+C to stop.")

    last_fired: dict[int, float] = {}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            result = model(frame, verbose=False, conf=CONF)[0]
            seen = {int(b.cls[0]) for b in result.boxes} if result.boxes is not None else set()
            now = time.time()
            for cid in seen:
                if cid in WATCH and (now - last_fired.get(cid, 0)) > DEBOUNCE_S:
                    last_fired[cid] = now
                    label, message = WATCH[cid]
                    print(f"[distraction] {label} detected -> warning driver")
                    post_event(message)
            time.sleep(0.12)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()
