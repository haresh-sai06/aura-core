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
import statistics
import threading
import time
import urllib.request
from collections import deque

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


def _pt(lm, i, w, h):
    return (lm[i].x * w, lm[i].y * h)


def mouth_aspect_ratio(lm, w, h) -> float:
    """MAR — vertical mouth opening / mouth width. Rises sharply during a yawn."""
    horiz = _dist(_pt(lm, 61, w, h), _pt(lm, 291, w, h))
    if horiz == 0:
        return 0.0
    vert = (_dist(_pt(lm, 13, w, h), _pt(lm, 14, w, h)) + _dist(_pt(lm, 0, w, h), _pt(lm, 17, w, h))) / 2.0
    return vert / horiz


def head_pose(lm, w, h):
    """Lightweight geometric head-pose proxy (degrees-ish): pitch = nod (down +), yaw = turn,
    roll = tilt. Robust and ~0 at rest — avoids solvePnP's calibration quirks for a live demo."""
    try:
        # roll: tilt of the eye line (real angle)
        roll = math.degrees(math.atan2(lm[263].y - lm[33].y, lm[263].x - lm[33].x))
        # yaw: nose horizontal offset from the eye-midpoint, normalized by face width
        eye_cx = (lm[33].x + lm[263].x) / 2.0
        face_w = abs(lm[454].x - lm[234].x) or 1e-6
        yaw = (lm[1].x - eye_cx) / face_w * 180.0
        # pitch: nose vertical position between eye-line and chin (level ~ 0; nodding down +)
        eye_cy = (lm[33].y + lm[263].y) / 2.0
        span = (lm[152].y - eye_cy) or 1e-6
        pitch = ((lm[1].y - eye_cy) / span - 0.30) * 160.0
        return (float(pitch), float(yaw), float(roll))
    except Exception:
        return (0.0, 0.0, 0.0)


def gaze_offsets(lm):
    """Mean horizontal + vertical iris offset from each eye's centre (~ -1..1). 0 = centred.
    Needs refine_landmarks (iris landmarks 468 / 473)."""
    try:
        def one(iris, outer, inner, top, bot):
            cx = (lm[outer].x + lm[inner].x) / 2.0
            cy = (lm[top].y + lm[bot].y) / 2.0
            wx = abs(lm[inner].x - lm[outer].x) or 1e-6
            wy = abs(lm[bot].y - lm[top].y) or 1e-6
            return (lm[iris].x - cx) / wx * 2.0, (lm[iris].y - cy) / wy * 2.0
        lx, ly = one(468, 33, 133, 159, 145)
        rx, ry = one(473, 263, 362, 386, 374)
        return (lx + rx) / 2.0, (ly + ry) / 2.0
    except Exception:
        return (0.0, 0.0)


def gaze_direction(gx, gy) -> str:
    if abs(gx) < 0.35 and abs(gy) < 0.55:
        return "center"
    if abs(gx) >= abs(gy):
        return "right" if gx > 0 else "left"
    return "down" if gy > 0 else "up"


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


def get_json(path: str):
    try:
        with urllib.request.urlopen(BASE + path, timeout=2) as r:
            return json.loads(r.read().decode())
    except Exception:
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

    # Rolling windows for the full drowsiness fusion (beyond EAR).
    perclos_win: deque = deque()   # (t, closed) over ~20s -> PERCLOS
    blink_times: deque = deque()   # (t, dur) blinks over ~60s -> blinks/min
    blink_open = [True]            # blink state machine (mutable so nested code can flip it)
    blink_started = [0.0]
    gaze_hist: deque = deque(maxlen=30)

    # Driver identity for the on-screen face box.
    driver_name = None            # current driver's display name (from Core)
    recognized = False            # True once Face-ID has matched this face
    last_health = 0.0

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
        max_num_faces=4,
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

                # Multiple people in frame? The DRIVER is the one closest to the camera =
                # the largest face (widest cheek-to-cheek span). Everyone else is ignored.
                faces = result.multi_face_landmarks
                primary = (max(faces, key=lambda f: abs(f.landmark[454].x - f.landmark[234].x))
                           if len(faces) > 1 else faces[0])
                lm = primary.landmark
                pts = lambda idx: [(lm[i].x * w, lm[i].y * h) for i in idx]  # noqa: E731
                raw_ear = (eye_aspect_ratio(pts(LEFT_EYE)) + eye_aspect_ratio(pts(RIGHT_EYE))) / 2.0
                # Exponential smoothing kills single-frame spikes from tracking noise.
                ear = raw_ear if ear_smooth is None else 0.6 * ear_smooth + 0.4 * raw_ear
                ear_smooth = ear
                closed = ear < args.ear_threshold

                # --- Full drowsiness fusion (previously EAR only) --------------------
                mar = mouth_aspect_ratio(lm, w, h)
                pitch, yaw, roll = head_pose(lm, w, h)
                gx, gy = gaze_offsets(lm)
                gaze_hist.append(gx)
                gaze_std = statistics.pstdev(gaze_hist) if len(gaze_hist) > 3 else 0.0
                gaze_stability = max(0.0, 1.0 - min(1.0, gaze_std * 5.0))
                gdir = gaze_direction(gx, gy)
                # PERCLOS = fraction of the last ~20s the eyes were closed.
                perclos_win.append((now, closed))
                while perclos_win and now - perclos_win[0][0] > 20.0:
                    perclos_win.popleft()
                perclos = sum(1 for _, c in perclos_win if c) / max(1, len(perclos_win))
                # Blink rate: count short closures (0.05–0.5s) over the last minute.
                if closed and blink_open[0]:
                    blink_open[0] = False
                    blink_started[0] = now
                elif (not closed) and not blink_open[0]:
                    blink_open[0] = True
                    bd = now - blink_started[0]
                    if 0.05 < bd < 0.5:
                        blink_times.append((now, bd))
                while blink_times and now - blink_times[0][0] > 60.0:
                    blink_times.popleft()
                blink_rate = len(blink_times)
                blink_dur = (sum(d for _, d in blink_times) / len(blink_times)) if blink_times else 0.0
                closure_now = (now - eyes_closed_since) if eyes_closed_since is not None else 0.0
                nod = max(0.0, min(1.0, (abs(pitch) - 12.0) / 25.0))
                # Fused 0–100 drowsiness score: PERCLOS + sustained closure + yawn + head-nod.
                fused_score = min(100.0,
                                  perclos * 55.0
                                  + min(closure_now, 2.5) / 2.5 * 30.0
                                  + max(0.0, min(1.0, (mar - 0.4) / 0.4)) * 18.0
                                  + nod * 12.0)

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
                                j = json.loads(resp) if resp else {}
                                if j.get("match"):
                                    driver_name = j.get("name") or driver_name
                                    recognized = True
                                    print(f"[camera] Face-ID recognized -> {driver_name}", flush=True)
                                else:
                                    recognized = False
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

                # Keep the current driver name fresh for the on-screen box (even without Face-ID).
                if now - last_health >= 3.0:
                    last_health = now
                    hj = get_json("/health")
                    if hj:
                        pdata = hj.get("persona") or {}
                        driver_name = pdata.get("name") or hj.get("driver") or driver_name

                # Throttled live state for the Live Monitor dashboard — the FULL signal set.
                if now - last_state >= args.state_interval:
                    last_state = now
                    post_json("/emit/state", {
                        "facePresent": True,
                        "ear": round(ear, 3), "mar": round(mar, 3), "perclos": round(perclos, 3),
                        "blinkRate": blink_rate, "blinkDuration": round(blink_dur, 3),
                        "headPitch": round(pitch, 1), "headYaw": round(yaw, 1), "headRoll": round(roll, 1),
                        "gazeStability": round(gaze_stability, 2), "gazeDirection": gdir,
                        "eyeClosureS": round(closure_now, 2), "score": round(fused_score, 1),
                    }, timeout=2.0)

                if args.show:
                    # Box around the DRIVER's face (the closest person), tagged with their name.
                    xs = [p.x * w for p in lm]
                    ys = [p.y * h for p in lm]
                    pad = int((max(xs) - min(xs)) * 0.16)
                    x1, y1 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
                    x2, y2 = min(w - 1, int(max(xs)) + pad), min(h - 1, int(max(ys)) + pad)
                    box_color = (0, 0, 255) if closed else ((80, 200, 80) if recognized else (0, 190, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                    tag = (driver_name or "Unknown") + ("  [Face-ID]" if recognized else "")
                    cv2.rectangle(frame, (x1, max(0, y1 - 32)), (x1 + 14 + len(tag) * 13, y1), box_color, -1)
                    cv2.putText(frame, tag, (x1 + 7, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
                    cv2.putText(frame, f"EAR {ear:.2f}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 0, 255) if closed else (0, 255, 0), 2)
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
