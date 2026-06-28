# Aura Core

The on-device **edge brain** for the Aura driver-persona prototype (TATA InnoVent — Edge AI for
Personalized & Connected Vehicles). Hub-and-spoke: this Python service owns the camera, the driver
persona, and the adaptive-safety policy, and broadcasts driver-state events over **WebSocket** to the
Unity game and the React dashboard. Everything runs on `localhost` — no cloud. That *is* the edge story.

```
 Camera ─▶ Aura Core ─┬─▶ Unity game   (reacts: alert / pull over)
 (CV)     (this repo) └─▶ React dash   (welcome / playlist / why)
```

## Run

```bash
# from this folder
uv venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # or: uv pip install -r requirements.txt
.venv\Scripts\python run.py
```

Server starts on `ws://127.0.0.1:8765/` (WebSocket) and `http://127.0.0.1:8765` (HTTP control).

## Fire test events (no Aura Core code change needed)

With the server running and Unity in Play mode:

```bash
python tools/send_test_event.py identify     # -> Unity shows "Welcome, Haresh"
python tools/send_test_event.py drowsy        # -> Unity shows "WAKE UP — pull_over"
python tools/send_test_event.py drowsy 1.0    # below Haresh's baseline -> no alert
```

Or hit the HTTP endpoints directly: `POST /emit/identify`, `POST /emit/drowsy`, `POST /emit/resume`, `GET /health`.

## Real camera (interface #1)

Instead of the faked events above, drive everything with your actual face:

```bash
pip install -r requirements-camera.txt     # opencv-python + mediapipe==0.10.14
python camera_monitor.py --show            # webcam window with live EAR overlay
```

[`camera_monitor.py`](camera_monitor.py) watches the webcam with MediaPipe Face Mesh, computes
eye-aspect-ratio (EAR), and feeds Core: a face appears → `identify`; eyes held closed → `drowsy`
(Core's policy decides if that's drowsy *for this driver*); eyes reopen → `resume`. Close your
eyes for ~2.5s and the car pulls over; open them and it drives on — no keyboard.

## Message contract

Every message is `{type, timestamp, payload}` JSON. Types live in [`aura_core/messages.py`](aura_core/messages.py)
and must stay in sync with the Unity `AuraClient` and the dashboard:

| type | direction | payload |
| --- | --- | --- |
| `driver.identified` | Core → clients | `{name, playlist}` |
| `safety.alert` | Core → clients | `{level, reason, action, modality, driver}` |
| `vehicle.telemetry` | Unity → Core | `{speed, position, ...}` |
| `explain` | Core → clients | `{decision, factors}` |

## Layout

```
aura_core/
  messages.py   message envelope + types (source of truth)
  persona.py    per-driver baselines + playlist (the personal safety threshold)
  policy.py     adaptive safety policy — warns vs YOUR baseline, not a fixed one
  server.py     FastAPI WebSocket hub + /emit control endpoints
run.py          start the hub
camera_monitor.py    real CV layer — webcam EAR -> /emit endpoints (interface #1)
requirements-camera.txt   opencv + mediapipe (pinned), install only where the camera runs
tools/
  send_test_event.py   CLI to inject test events
```

## Next

- Persist personas to SQLite; pick the persona via real face recognition (today the camera
  greets whoever appears as the current driver).
- Stream `vehicle.telemetry` from Unity → dashboard; add `explain` messages for the HUD.
- The adaptive eval study: false-alarm rate vs a generic fixed threshold.
