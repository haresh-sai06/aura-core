"""Aura Core WebSocket hub.

- Unity (and later the React dashboard) connect over WebSocket at ws://127.0.0.1:8765/.
- HTTP POST endpoints let you (or the dashboard, or a test CLI) inject driver-state
  events that get broadcast to all connected clients. For the vertical slice the
  drowsiness signal is faked via /emit/drowsy; later it comes from the camera/CV layer.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .messages import MessageType, envelope
from .persona import PersonaStore
from .policy import AdaptiveSafetyPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [aura-core] %(message)s")
log = logging.getLogger("aura-core")

app = FastAPI(title="Aura Core", version="0.1.0")

clients: Set[WebSocket] = set()
personas = PersonaStore()
policy = AdaptiveSafetyPolicy(personas)

# Which driver is "in the seat" for the demo. Later set by face recognition.
current_driver = "haresh"

# Dedupe: while an alert is active, repeated drowsy signals (e.g. from the camera
# streaming every frame) don't re-broadcast. Cleared on resume.
_alerted = False


async def broadcast(message: Dict[str, Any]) -> None:
    """Send one envelope to every connected client; drop any that have gone away."""
    text = json.dumps(message)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)
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
async def emit_drowsy(eye_closure_s: float = 3.2) -> Dict[str, Any]:
    """A drowsiness signal (faked by the test CLI, or real from the camera monitor).
    The policy decides if it warrants an alert for THIS driver; dedupe avoids re-broadcasting."""
    global _alerted
    decision = policy.evaluate(current_driver, eye_closure_s)
    if decision is None:
        return {"sent": None, "note": "below this driver's baseline — no alert (a generic DMS might false-alarm)"}
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
