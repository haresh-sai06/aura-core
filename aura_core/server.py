"""Aura Core WebSocket hub.

- Unity (and later the React dashboard) connect over WebSocket at ws://127.0.0.1:8765/.
- HTTP POST endpoints let you (or the dashboard, or a test CLI) inject driver-state
  events that get broadcast to all connected clients. For the vertical slice the
  drowsiness signal is faked via /emit/drowsy; later it comes from the camera/CV layer.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .messages import MessageType, envelope
from .persona import PersonaStore
from .policy import AdaptiveSafetyPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [aura-core] %(message)s")
log = logging.getLogger("aura-core")

app = FastAPI(title="Aura Core", version="0.1.0")

# Allow the dashboard (served from localhost:5173) to POST signals from the in-browser camera.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

clients: Set[WebSocket] = set()
personas = PersonaStore()
policy = AdaptiveSafetyPolicy(personas)

# Which driver is "in the seat" for the demo. Later set by face recognition.
current_driver = "haresh"

# Dedupe: while an alert is active, repeated drowsy signals (e.g. from the camera
# streaming every frame) don't re-broadcast. Cleared on resume.
_alerted = False


async def broadcast(message: Dict[str, Any], quiet: bool = False) -> None:
    """Send one envelope to every connected client; drop any that have gone away.
    quiet=True suppresses the log line (used for high-frequency driver.state updates)."""
    text = json.dumps(message)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)
    if not quiet:
        log.info("broadcast %s -> %d client(s)", message.get("type"), len(clients))


@app.websocket("/")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    clients.add(ws)
    log.info("client connected (%d total)", len(clients))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # Inbound from Unity for now is telemetry/acks — just log it.
            log.info("recv %s %s", msg.get("type"), msg.get("payload"))
    except WebSocketDisconnect:
        clients.discard(ws)
        log.info("client disconnected (%d total)", len(clients))


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "clients": len(clients), "driver": current_driver}


@app.post("/emit/identify")
async def emit_identify() -> Dict[str, Any]:
    p = personas.get(current_driver)
    msg = envelope(MessageType.DRIVER_IDENTIFIED, {"name": p.display_name, "playlist": p.playlist})
    await broadcast(msg)
    return {"sent": msg}


@app.post("/emit/drowsy")
async def emit_drowsy(eye_closure_s: float = 3.2, score: Optional[float] = None) -> Dict[str, Any]:
    """A drowsiness signal. Prefer the fused `score` (browser 7-signal pipeline); fall back to
    `eye_closure_s` (Python camera). The policy applies THIS driver's personal threshold; dedupe
    avoids re-broadcasting while an alert is active."""
    global _alerted
    decision = policy.evaluate(
        current_driver,
        eye_closure_s=(None if score is not None else eye_closure_s),
        score=score,
    )
    if decision is None:
        return {"sent": None, "note": "below this driver's personal threshold — no alert"}
    if _alerted:
        return {"sent": None, "note": "alert already active (deduped)"}
    _alerted = True
    await broadcast(decision)
    return {"sent": decision}


@app.post("/emit/resume")
async def emit_resume() -> Dict[str, Any]:
    """Driver responded (eyes reopened) — clear the pull-over and let the car drive again."""
    global _alerted
    _alerted = False
    msg = envelope("safety.clear", {"driver": personas.get(current_driver).display_name})
    await broadcast(msg)
    return {"sent": msg}


@app.post("/emit/state")
async def emit_state(request: Request) -> Dict[str, Any]:
    """Continuous live driver signal for the Live Monitor (throttled). Accepts a rich JSON body
    (the browser's full 7-signal pipeline) or simple query params (the Python camera). Broadcast
    as driver.state; never gates an alert (that's /emit/drowsy)."""
    p = personas.get(current_driver)
    try:
        body = await request.json()
    except Exception:
        body = None

    if isinstance(body, dict) and body:
        payload = dict(body)
        payload.setdefault("driver", p.display_name)
        payload.setdefault("baseline", p.eye_closure_threshold_s)
        payload.setdefault("threshold", p.drowsiness_threshold)
    else:
        qp = request.query_params
        face = qp.get("face_present", "true") == "true"
        closure = float(qp.get("eye_closure_s", 0) or 0)
        ear = float(qp.get("ear", 0.3) or 0.3)
        score = 0.0 if not face else max(0.0, min(100.0, (closure / max(p.eye_closure_threshold_s, 0.1)) * 80.0))
        payload = {
            "facePresent": face, "ear": round(ear, 3), "eyeClosureS": round(closure, 2),
            "score": round(score, 1), "baseline": p.eye_closure_threshold_s,
            "threshold": p.drowsiness_threshold, "driver": p.display_name,
        }

    await broadcast(envelope(MessageType.DRIVER_STATE, payload), quiet=True)
    return {"ok": True}
