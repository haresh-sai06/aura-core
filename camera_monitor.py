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
import base64
import json
import math
import os
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8765"

# MediaPipe FaceMesh landmark indices for each eye: [corner, top1, top2, corner, bot2, bot1].
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Face-ID: landmark pairs whose distance/face-width is person-distinctive. MUST match the
# dashboard's utils/faceSignature.ts so a face enrolled from either client is recognizable.
_SIG_W, _SIG_H = 640, 480
_SIG_PAIRS = [(33, 133), (362, 263), (133, 362), (10, 152), (168, 2), (61, 291),
              (13, 14), (105, 334), (172, 397), (159, 105), (2, 152), (133, 13)]


def face_signature(lm):
    """A pose-normalized geometric signature (numbers only, never an image) from FaceMesh."""
    if len(lm) < 468:
        return None
    def d(a, b):
        return math.hypot((lm[a].x - lm[b].x) * _SIG_W, (lm[a].y - lm[b].y) * _SIG_H)
    face_w = d(234, 454)
    if face_w < 1:
        return None
    return [round(d(a, b) / face_w, 4) for (a, b) in _SIG_PAIRS]


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


def post_json(path: str, obj, timeout: float = 3.0):
    try:
        data = json.dumps(obj).encode()
        req = urllib.request.Request(BASE + path, data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
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
    ap.add_argument("--recognize-interval", type=float, default=2.5, help="seconds between Face-ID recognition attempts")
    ap.add_argument("--vision-interval", type=float, default=12.0, help="seconds between vision-LLM scene reads")
    ap.add_argument("--no-face-id", action="store_true", help="disable on-device Face-ID recognition")
    ap.add_argument("--no-vision", action="store_true", help="disable vision-LLM scene understanding")
    ap.add_argument("--core", default=os.environ.get("AURA_CORE", "http://127.0.0.1:8765"),
                    help="Aura Core base URL. When the camera runs on System B, point it at System A's "
                         "LAN IP, e.g. http://192.168.1.23:8765 (or set the AURA_CORE env var).")
    args = ap.parse_args()
    args.face_id = not args.no_face_id
    args.vision = not args.no_vision

    global BASE
    BASE = args.core.rstrip("/")
    print(f"[camera] Aura Core -> {BASE}")

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

    # Face-ID + vision state.
    sig_ewma = None
    last_observe = 0.0
    last_recognize = 0.0
    last_vision = 0.0
    vision_busy = [False]  # list = mutable flag captured by the background vision thread

    def read_scene(jpg_b64: str) -> None:
        vision_busy[0] = True
        try:
            post_json("/vision/scene", {"image": jpg_b64, "kind": "cabin"}, timeout=90)
        finally:
            vision_busy[0] = False

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

                # --- Face-ID (geometry only) + vision-LLM (out of the safety loop) ---
                if args.face_id:
                    raw_sig = face_signature(lm)
                    if raw_sig:
                        sig_ewma = raw_sig if sig_ewma is None else [0.8 * a + 0.2 * b for a, b in zip(sig_ewma, raw_sig)]
                        rounded = [round(x, 4) for x in sig_ewma]
                        if now - last_observe >= 1.0:
                            last_observe = now
                            post_json("/faceid/observe", {"signature": rounded})
                        if now - last_recognize >= args.recognize_interval:
                            last_recognize = now
                            resp = post_json("/driver/recognize", {"signature": rounded})
                            try:
                                if resp and json.loads(resp).get("switched"):
                                    print("[camera] Face-ID recognized -> switched driver", flush=True)
                            except Exception:
                                pass
                # Vision runs on a background thread so llava's seconds don't stall the EAR loop.
                if args.vision and not vision_busy[0] and now - last_vision >= args.vision_interval:
                    last_vision = now
                    ok_enc, buf = cv2.imencode(".jpg", cv2.resize(frame, (512, 384)),
                                               [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    if ok_enc:
                        b64 = base64.b64encode(buf.tobytes()).decode()
                        threading.Thread(target=read_scene, args=(b64,), daemon=True).start()

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
                    if sig_ewma is not None:
                        sig_ewma = None
                        post_json("/faceid/observe", {"signature": None})  # clear live face
                    if now - last_state >= args.state_interval:
                        last_state = now
                        post("/emit/state?ear=0&eye_closure_s=0&face_present=false")

            if args.show:
                if args.face_id:
                    cv2.putText(frame, "E = enroll this driver's face", (12, h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv2.imshow("Aura camera", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Esc
                    break
                if key in (ord("e"), ord("E")):
                    r = post_json("/driver/enroll_current", {})
                    print(f"[camera] enroll current driver -> {r}", flush=True)

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
