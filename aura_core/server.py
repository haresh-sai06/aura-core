"""Aura Core WebSocket hub — the edge brain.

Unity, the React dashboard, and the browser camera all connect over one WebSocket at
ws://127.0.0.1:8765/. HTTP POST endpoints inject driver-state events that get broadcast
to every client. Everything runs on localhost — that *is* the edge story.

What's new over the first slice:
  • **Driver selection** — `/drivers` + `/driver/select` let the dashboard switch personas
    live, so the adaptive-threshold differentiator is demoable in seconds.
  • **Vehicle telemetry loop** — Unity streams `vehicle.telemetry` in; the hub rebroadcasts
    it so the dashboard shows the REAL car speed/steer/state instead of a faked number.
  • **Explainability** — every alert (and driver switch) is followed by an `explain` message
    answering "why did Aura act for ME?".
  • **Adaptive learning** — calm live-state samples tune each driver's threshold over time.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import llm, reasoning
from .copilot import Copilot
from .messages import MessageType, envelope
from .persona import PersonaStore
from .policy import AdaptiveSafetyPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [aura-core] %(message)s")
log = logging.getLogger("aura-core")

app = FastAPI(title="Aura Core", version="0.2.0")

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
copilot = Copilot()

# Last live driver-state seen (fed to the Reasoning Agent + Copilot for "right now" grounding).
latest_state: Dict[str, Any] = {}

# Which driver is "in the seat". Set by the dashboard selector or (later) face recognition.
current_driver = "haresh"

# Dedupe: while an alert is active, repeated drowsy signals don't re-broadcast. Cleared on resume.
_alerted = False

# Last vehicle telemetry seen from Unity (so a late-joining dashboard can be seeded via /health).
latest_telemetry: Dict[str, Any] = {}


async def broadcast(message: Dict[str, Any], quiet: bool = False, exclude: Optional[WebSocket] = None) -> None:
    """Send one envelope to every connected client; drop any that have gone away.
    quiet=True suppresses the log line (high-frequency updates); exclude skips the sender."""
    text = json.dumps(message)
    dead = []
    for ws in clients:
        if ws is exclude:
            continue
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)
    if not quiet:
        log.info("broadcast %s -> %d client(s)", message.get("type"), len(clients))


async def stream_reasoning(driver_id: str, state: Dict[str, Any], acted: bool = True) -> None:
    """Run the on-device Reasoning Agent and stream its natural-language 'why' onto the bus.

    Emits `reasoning` envelopes with phase start -> delta* -> done so the dashboard's World
    Model panel can 'type out' the explanation live. The blocking urllib generator is stepped
    via run_in_executor so the event loop (and every other client) stays responsive.
    """
    p = personas.get(driver_id)
    gen = reasoning.reason_stream(p, state, latest_telemetry or None, acted=acted)
    loop = asyncio.get_running_loop()

    def _next() -> Optional[str]:
        try:
            return next(gen)
        except StopIteration:
            return None

    await broadcast(envelope(MessageType.REASONING, {
        "phase": "start", "driver": p.display_name, "acted": acted, "text": "",
    }))
    full = ""
    while True:
        piece = await loop.run_in_executor(None, _next)
        if piece is None:
            break
        full += piece
        await broadcast(envelope(MessageType.REASONING, {
            "phase": "delta", "driver": p.display_name, "delta": piece, "text": full,
        }), quiet=True)
    await broadcast(envelope(MessageType.REASONING, {
        "phase": "done", "driver": p.display_name, "acted": acted, "text": full.strip(),
    }))
    log.info("reasoning streamed (%d chars) for %s", len(full), driver_id)


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
            mtype = msg.get("type")
            payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else {}

            # Unity streams the live car state — rebroadcast so the dashboard mirrors the
            # real vehicle instead of a hard-coded speed.
            if mtype == MessageType.VEHICLE_TELEMETRY.value:
                global latest_telemetry
                latest_telemetry = payload
                await broadcast(envelope(MessageType.VEHICLE_TELEMETRY, payload), quiet=True, exclude=ws)
            else:
                log.info("recv %s %s", mtype, payload)
    except WebSocketDisconnect:
        clients.discard(ws)
        log.info("client disconnected (%d total)", len(clients))


@app.on_event("startup")
async def _warmup() -> None:
    """Pre-warm the on-device stack so the first on-stage request isn't the slow one.
    Loads qwen2.5:7b into memory and embeds the KB — off the event loop, non-blocking."""
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        try:
            if await loop.run_in_executor(None, llm.available):
                await loop.run_in_executor(
                    None, lambda: llm.chat([{"role": "user", "content": "ok"}], num_predict=1)
                )
                await loop.run_in_executor(None, copilot.ensure_embedded)
                log.info("warmup: on-device LLM + copilot KB ready")
            else:
                log.info("warmup: Ollama not reachable — running in offline-fallback mode")
        except Exception as e:
            log.warning("warmup skipped (%s)", e)

    asyncio.create_task(_run())


@app.on_event("shutdown")
def _persist_on_shutdown() -> None:
    personas.save()


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "clients": len(clients),
        "driver": current_driver,
        "persona": personas.to_public(current_driver),
        "telemetry": latest_telemetry,
    }


# ── Driver / persona control ─────────────────────────────────────────

@app.get("/drivers")
async def drivers() -> Dict[str, Any]:
    """List every persona + who is currently in the seat (for the dashboard selector)."""
    out = personas.to_public()
    out["current"] = current_driver
    return out


@app.post("/driver/select")
async def select_driver(request: Request) -> Dict[str, Any]:
    """Switch the active driver. Accepts ?id=priya or JSON {"id": "priya"}. Broadcasts the
    new persona (welcome + playlist) and an explain so the whole demo re-personalizes at once."""
    global current_driver, _alerted
    did = request.query_params.get("id")
    if did is None:
        try:
            body = await request.json()
            did = body.get("id") if isinstance(body, dict) else None
        except Exception:
            did = None
    if not did or not personas.has(did):
        return {"ok": False, "error": f"unknown driver '{did}'", "drivers": personas.ids()}

    # Persist any learning for the outgoing driver before switching.
    personas.save()
    current_driver = did
    _alerted = False
    p = personas.get(did)
    await broadcast(envelope(MessageType.DRIVER_IDENTIFIED, {"name": p.display_name, "playlist": p.playlist}))
    await broadcast(policy.explain(did))
    log.info("driver switched -> %s", did)
    return {"ok": True, "driver": personas.to_public(did)}


# ── Event injection (camera / test CLI / dashboard) ──────────────────

@app.post("/emit/identify")
async def emit_identify() -> Dict[str, Any]:
    p = personas.get(current_driver)
    msg = envelope(MessageType.DRIVER_IDENTIFIED, {"name": p.display_name, "playlist": p.playlist})
    await broadcast(msg)
    await broadcast(policy.explain(current_driver))
    return {"sent": msg}


@app.post("/emit/drowsy")
async def emit_drowsy(eye_closure_s: float = 3.2, score: Optional[float] = None) -> Dict[str, Any]:
    """A drowsiness signal. Prefer the fused `score` (browser 7-signal pipeline); fall back to
    `eye_closure_s` (Python camera). The policy applies THIS driver's personal threshold; dedupe
    avoids re-broadcasting while an alert is active. Fires an `explain` alongside the alert."""
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
    await broadcast(policy.explain(current_driver, score=score, eye_closure_s=eye_closure_s))
    # Fire the natural-language Reasoning Agent AFTER the alert (never delay the safety signal
    # on the LLM). It streams onto the bus as `reasoning`. Scheduled, so this returns instantly.
    rstate = dict(latest_state)
    if score is not None:
        rstate["score"] = score              # fresh fused score IS the trigger
    else:
        rstate.pop("score", None)            # drop stale live score — trigger was eye-closure
        rstate["eyeClosureS"] = eye_closure_s
    asyncio.create_task(stream_reasoning(current_driver, rstate, acted=True))
    return {"sent": decision}


@app.post("/emit/reason")
async def emit_reason(request: Request) -> Dict[str, Any]:
    """Manually trigger the Reasoning Agent for the current driver + latest live state.
    Handy as a demo control ('explain what you're seeing now') independent of an alert."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    acted = bool(body.get("acted", False)) if isinstance(body, dict) else False
    state = dict(latest_state)
    if isinstance(body, dict) and isinstance(body.get("state"), dict):
        state.update(body["state"])
    asyncio.create_task(stream_reasoning(current_driver, state, acted=acted))
    return {"ok": True, "driver": current_driver, "acted": acted}


@app.post("/copilot/ask")
async def copilot_ask(request: Request) -> Dict[str, Any]:
    """Agentic RAG endpoint: answer a driver question grounded in the on-device knowledge base,
    enriched with the live driver/vehicle state. Also broadcasts the answer as `copilot.response`
    so every screen (and a future voice UI) can react."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    query = (body.get("query") or body.get("q") or "").strip() if isinstance(body, dict) else ""
    if not query:
        return {"ok": False, "error": "empty query"}

    p = personas.get(current_driver)
    ctx: Dict[str, Any] = {
        "driver": p.display_name,
        "threshold": round(p.drowsiness_threshold, 0),
        "score": latest_state.get("score"),
        "level": latest_state.get("level"),
        "speedKmh": (latest_telemetry or {}).get("speedKmh"),
        "scenario": (latest_telemetry or {}).get("scenario"),
    }
    if isinstance(body, dict) and isinstance(body.get("context"), dict):
        ctx.update(body["context"])

    # Retrieval + generation are blocking (urllib) — run off the event loop.
    result = await asyncio.get_running_loop().run_in_executor(
        None, lambda: copilot.answer(query, ctx)
    )
    payload = {"query": query, "driver": p.display_name, **result}
    await broadcast(envelope(MessageType.COPILOT_RESPONSE, payload))
    return {"ok": True, **payload}


@app.get("/llm/status")
async def llm_status() -> Dict[str, Any]:
    """Report the active LLM brain (cloud/on-device) + model so the UI can badge it honestly."""
    st = await asyncio.get_running_loop().run_in_executor(None, llm.status)
    st["kbChunks"] = len(copilot.chunks)
    st["chatModel"] = st.get("model")
    return st


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """Current LLM config (key redacted) — for a settings panel."""
    st = llm.status()
    return {
        "provider": st["provider"],
        "model": st["model"],
        "hasCloudKey": st["cloudKey"],
        "ollama": st["ollama"],
    }


@app.post("/config")
async def set_config(request: Request) -> Dict[str, Any]:
    """Set the LLM provider/key/model at runtime. Writes aura_config.json (gitignored) and
    hot-reloads so a freshly-pasted OpenRouter key takes effect without a restart."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return {"ok": False, "error": "expected JSON object"}

    import os as _os
    cfg_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "aura_config.json")
    existing: Dict[str, Any] = {}
    try:
        if _os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
    except Exception:
        existing = {}
    for k in ("provider", "openrouter_api_key", "openrouter_model", "nim_api_key", "nim_model"):
        if body.get(k) is not None:
            existing[k] = body[k]
    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        return {"ok": False, "error": f"could not write config: {e}"}

    llm.reload_config()
    # Re-warm the newly-selected brain in the background.
    asyncio.create_task(_warmup())
    return {"ok": True, **(await get_config())}


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
    (the browser's 7-signal pipeline) or simple query params (the Python camera). Broadcast as
    driver.state; also feeds adaptive learning when the driver is clearly awake."""
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

    # Remember the latest live signal so the Reasoning Agent + Copilot can ground on "now".
    global latest_state
    latest_state = dict(payload)

    # Adaptive learning: a clearly-awake sample (well under this driver's line) tunes the baseline.
    try:
        s = float(payload.get("score", 0) or 0)
        if payload.get("facePresent", True) and s < 0.6 * p.drowsiness_threshold:
            personas.record_calm(current_driver, s)
            payload["threshold"] = p.drowsiness_threshold  # reflect any learned change immediately
    except Exception:
        pass

    await broadcast(envelope(MessageType.DRIVER_STATE, payload), quiet=True)
    return {"ok": True}
