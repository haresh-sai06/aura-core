"""Aura camera monitor — the real CV layer (interface #1).

Watches the webcam with MediaPipe Face Mesh, computes each eye's aspect ratio (EAR),
and feeds Aura Core over HTTP — exactly where the faked test events plugged in:

  - a face appears        -> POST /emit/identify   (dashboard says "Welcome")
  - eyes held closed      -> POST /emit/drowsy?eye_closure_s=<dur>   (adaptive alert; car pulls over)
  - eyes reopen / no face -> POST /emit/resume      (clear; car drives on)

The adaptive baseline still lives in Aura Core's policy — this just reports how long the
eyes have actually been closed, and Core decides whether that's drowsy *for this driver*.

Run order:
  1) python run.py                          # Aura Core
  2) pip install -r requirements-camera.txt # opencv-python, mediapipe
  3) python camera_monitor.py --show        # this; --show opens a preview window
"""
from __future__ import annotations

import argparse
import math
import time
import urllib.request

BASE = "http://127.0.0.1:8765"

# MediaPipe FaceMesh landmark indices for each eye: [corner, top1, top2, corner, bot2, bot1].
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(pts) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|). Low when the eye is closed.

    pts is six (x, y) points in image space, ordered [p1..p6] as in LEFT_EYE/RIGHT_EYE.
    Pure function — unit-testable without a camera.
    """
    p1, p2, p3, p4, p5, p6 = pts
    horizontal = _dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def post(path: str):
    try:
        req = urllib.request.Request(BASE + path, method="POST")
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.read().decode()
    except Exception as e:  # noqa: BLE001 - best-effort producer
        print(f"[camera] POST {path} failed: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Aura webcam drowsiness monitor")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--ear-threshold", type=float, default=0.21, help="smoothed EAR below this = eyes closed")
    ap.add_argument("--report-interval", type=float, default=0.4, help="seconds between drowsy reports while closed")
    ap.add_argument("--reopen-debounce", type=float, default=0.25,
                    help="eyes must read open continuously this long before the closure timer resets")
    ap.add_argument("--face-lost-grace", type=float, default=1.5,
                    help="ignore face-tracking dropouts shorter than this (seconds) so identify doesn't re-fire on blips")
    ap.add_argument("--state-interval", type=float, default=0.2,
                    help="seconds between live driver.state updates sent for the Live Monitor")
    ap.add_argument("--show", action="store_true", help="show the webcam window with EAR overlay")
    ap.add_argument("--debug", action="store_true", help="print EAR readings ~2x/sec for tuning")
    args = ap.parse_args()

    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        raise SystemExit("Missing deps. Run:  pip install -r requirements-camera.txt")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    face_present = False
    face_gone_since = None
    eyes_closed_since = None
    eyes_open_since = None
    ear_smooth = None
    last_report = 0.0
    last_state = 0.0
    alerted = False
    last_debug = 0.0

    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as fm:
        print("[camera] running — close your eyes to trigger a pull-over. Esc/Ctrl+C to stop.")
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = fm.process(rgb)
            now = time.time()

            if result.multi_face_landmarks:
                face_gone_since = None  # face is present this frame — cancel any pending "gone" timer
                if not face_present:
                    face_present = True
                    print("[camera] face detected -> identify")
                    post("/emit/identify")

                lm = result.multi_face_landmarks[0].landmark
                pts = lambda idx: [(lm[i].x * w, lm[i].y * h) for i in idx]  # noqa: E731
                raw_ear = (eye_aspect_ratio(pts(LEFT_EYE)) + eye_aspect_ratio(pts(RIGHT_EYE))) / 2.0
                # Exponential smoothing kills single-frame spikes from tracking noise.
                ear = raw_ear if ear_smooth is None else 0.6 * ear_smooth + 0.4 * raw_ear
                ear_smooth = ear
                closed = ear < args.ear_threshold

                if args.debug and now - last_debug >= 0.5:
                    last_debug = now
                    print(f"[camera] EAR={ear:.3f}  thr={args.ear_threshold}  "
                          f"{'CLOSED' if closed else 'open'}", flush=True)

                if closed:
                    eyes_open_since = None
                    if eyes_closed_since is None:
                        eyes_closed_since = now
                    duration = now - eyes_closed_since
                    if now - last_report >= args.report_interval:
                        last_report = now
                        resp = post(f"/emit/drowsy?eye_closure_s={duration:.2f}")
                        if resp and "safety.alert" in resp:
                            alerted = True
                            print(f"[camera] eyes closed {duration:.1f}s -> ALERT", flush=True)
                else:
                    # Debounce: a brief open blip during a closure shouldn't reset the timer.
                    if eyes_open_since is None:
                        eyes_open_since = now
                    if now - eyes_open_since >= args.reopen_debounce:
                        if alerted:
                            post("/emit/resume")
                            print("[camera] eyes reopened -> resume", flush=True)
                            alerted = False
                        eyes_closed_since = None

                # Throttled live state for the Live Monitor dashboard.
                if now - last_state >= args.state_interval:
                    last_state = now
                    closure = (now - eyes_closed_since) if eyes_closed_since is not None else 0.0
                    post(f"/emit/state?ear={ear:.3f}&eye_closure_s={closure:.2f}&face_present=true")

                if args.show:
                    color = (0, 0, 255) if closed else (0, 255, 0)
                    cv2.putText(frame, f"EAR {ear:.2f}", (12, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            else:
                # Debounce brief tracking dropouts — a 1-frame miss shouldn't re-fire identify.
                if face_gone_since is None:
                    face_gone_since = now
                if now - face_gone_since >= args.face_lost_grace:
                    if alerted:
                        post("/emit/resume")
                        alerted = False
                    face_present = False
                    eyes_closed_since = None
                    eyes_open_since = None
                    ear_smooth = None
                    if now - last_state >= args.state_interval:
                        last_state = now
                        post("/emit/state?ear=0&eye_closure_s=0&face_present=false")

            if args.show:
                cv2.imshow("Aura camera", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # Esc
                    break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
