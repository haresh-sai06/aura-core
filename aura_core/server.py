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
import time
from datetime import datetime
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import llm, reasoning, vision
from .agents import Orchestrator
from .copilot import Copilot
from .forecast import Forecaster
from .messages import MessageType, envelope
from .persona import PersonaStore
from .policy import AdaptiveSafetyPolicy
from .conversation import router as conversation_router

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

# Buddy conversation + per-driver "Driver DNA" knowledge base (First Drive experience).
app.include_router(conversation_router)

clients: Set[WebSocket] = set()
personas = PersonaStore()
policy = AdaptiveSafetyPolicy(personas)
copilot = Copilot()
forecaster = Forecaster()
orchestrator = Orchestrator(personas, forecaster)

# Last live driver-state seen (fed to the Reasoning Agent + Copilot for "right now" grounding).
latest_state: Dict[str, Any] = {}

# Predictive world-model + multi-agent trace — kept so /state and the MCP server can read them.
latest_forecast: Dict[str, Any] = {}
latest_orchestration: Dict[str, Any] = {}
last_reasoning_text: str = ""

# Most recent live face signature observed by the camera process (for dashboard-driven enroll).
latest_signature: Optional[list] = None

# Trip context (for the Context agent) + throttles so the agent graph doesn't flood the bus.
_drive_start = time.monotonic()
_last_orch_ts = 0.0
_last_orch_level = -1
_last_cm_level = -1


def _trip_context() -> Dict[str, Any]:
    """Best-effort trip context for the Context agent: hours driven + time of day + scenario."""
    hours = max(0.0, (time.monotonic() - _drive_start) / 3600.0)
    hour = datetime.now().hour
    tod = ("late night" if hour < 5 else "early morning" if hour < 8 else "daytime"
           if hour < 17 else "evening" if hour < 21 else "night")
    return {"hoursDriven": round(hours, 2), "timeOfDay": tod, "scenario": (latest_telemetry or {}).get("scenario")}


async def run_orchestration(trigger: str = "state", force: bool = False, copilot_active: bool = False) -> Dict[str, Any]:
    """Run one multi-agent decision cycle and broadcast the trace + forecast + any countermeasures.
    Throttled to ~2/s unless the risk level changes (or force=True) so the graph stays live but calm."""
    global latest_orchestration, _last_orch_ts, _last_orch_level, _last_cm_level
    payload = orchestrator.run_cycle(
        current_driver, latest_state, latest_telemetry, _trip_context(),
        trigger=trigger, copilot_active=copilot_active,
    )
    latest_orchestration = payload
    level = payload["level"]
    now = time.monotonic()
    changed = level != _last_orch_level
    if force or changed or (now - _last_orch_ts) > 0.5:
        _last_orch_ts = now
        _last_orch_level = level
        await broadcast(envelope(MessageType.ORCHESTRATION, payload), quiet=not changed)
        # Broadcast the Wellness countermeasures once per level change so the HMI can react.
        if payload["actions"] and level != _last_cm_level and level not in (0, 4):
            _last_cm_level = level
            await broadcast(envelope(MessageType.COUNTERMEASURE, {
                "driver": payload["driver"], "level": level, "actions": payload["actions"],
            }))
        if level in (0,):
            _last_cm_level = -1
    return payload

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
    global last_reasoning_text
    last_reasoning_text = full.strip()
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
                # Pre-load the vision model too (a 1x1 pixel) so the first scene call isn't the
                # slow one. Guarded — if llava isn't pulled this is a quick no-op.
                if await loop.run_in_executor(None, vision.available):
                    ok = await loop.run_in_executor(None, vision.warm)
                    log.info("warmup: vision model (llava) %s", "resident" if ok else "load failed")
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


async def _switch_driver(did: str, source: str = "manual") -> None:
    """Make `did` the active driver and re-personalize the whole system: persist the outgoing
    driver's learning, reset trip context + forecaster, and broadcast welcome + explain. Shared by
    the manual selector and Face-ID recognition."""
    global current_driver, _alerted, _drive_start, _last_cm_level
    personas.save()                           # persist any learning for the outgoing driver
    current_driver = did
    _alerted = False
    _drive_start = time.monotonic()           # a new driver = a fresh trip for the Context agent
    _last_cm_level = -1
    forecaster.reset(did)                     # don't carry the old driver's trajectory over
    p = personas.get(did)
    await broadcast(envelope(MessageType.DRIVER_IDENTIFIED,
                             {"name": p.display_name, "playlist": p.playlist, "via": source}))
    await broadcast(policy.explain(did))
    log.info("driver switched -> %s (%s)", did, source)


@app.post("/driver/select")
async def select_driver(request: Request) -> Dict[str, Any]:
    """Switch the active driver. Accepts ?id=priya or JSON {"id": "priya"}. Broadcasts the
    new persona (welcome + playlist) and an explain so the whole demo re-personalizes at once."""
    did = request.query_params.get("id")
    if did is None:
        try:
            body = await request.json()
            did = body.get("id") if isinstance(body, dict) else None
        except Exception:
            did = None
    if not did or not personas.has(did):
        return {"ok": False, "error": f"unknown driver '{did}'", "drivers": personas.ids()}
    await _switch_driver(did, source="manual")
    return {"ok": True, "driver": personas.to_public(did)}


@app.post("/driver/enroll")
async def enroll_face(request: Request) -> Dict[str, Any]:
    """Enroll a geometric face signature (from the browser's MediaPipe landmarks) for a driver.
    Only the numeric signature is stored — never an image. Defaults to the current driver."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return {"ok": False, "error": "expected JSON object"}
    did = body.get("id") or current_driver
    sig = body.get("signature")
    if not personas.has(did) or not isinstance(sig, list) or not sig:
        return {"ok": False, "error": "need a valid driver id and non-empty signature"}
    ok = personas.enroll_face(did, sig)
    personas.save()
    return {"ok": ok, "driver": did, "enrolled": personas.enrolled_ids()}


@app.post("/driver/recognize")
async def recognize_face(request: Request) -> Dict[str, Any]:
    """Given a live face signature, find the closest ENROLLED driver and (if confident and not
    already active) switch to them — the 'sit down → Welcome, Haresh' moment. On-device."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    sig = body.get("signature") if isinstance(body, dict) else None
    if not isinstance(sig, list) or not sig:
        return {"ok": False, "error": "need a signature"}
    did, dist, conf = personas.recognize_face(sig)
    if not did:
        return {"ok": True, "match": None, "distance": dist, "enrolled": personas.enrolled_ids()}
    switched = did != current_driver
    if switched:
        await _switch_driver(did, source="face-id")
    return {"ok": True, "match": did, "name": personas.get(did).display_name,
            "distance": dist, "confidence": conf, "switched": switched}


@app.get("/faceid/status")
async def faceid_status() -> Dict[str, Any]:
    """Which drivers have a face enrolled + whether a live face is in view — for the panel."""
    return {"enrolled": personas.enrolled_ids(), "current": current_driver,
            "liveFace": latest_signature is not None}


@app.post("/faceid/observe")
async def faceid_observe(request: Request) -> Dict[str, Any]:
    """The camera process streams the current live face signature here (numbers only) so the
    dashboard can enroll the active driver with one click via /driver/enroll_current."""
    global latest_signature
    try:
        body = await request.json()
    except Exception:
        body = {}
    sig = body.get("signature") if isinstance(body, dict) else None
    latest_signature = sig if isinstance(sig, list) and sig else None
    return {"ok": True, "hasFace": latest_signature is not None}


@app.post("/driver/enroll_current")
async def enroll_current(request: Request) -> Dict[str, Any]:
    """Enroll the last-seen live signature for a driver (defaults to the active one). Lets the
    dashboard's 'Enroll this driver' button work without the browser owning the camera."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    did = (body.get("id") if isinstance(body, dict) else None) or current_driver
    if latest_signature is None:
        return {"ok": False, "error": "no live face in view — look at the camera and retry"}
    ok = personas.enroll_face(did, latest_signature)
    personas.save()
    return {"ok": ok, "driver": did, "enrolled": personas.enrolled_ids()}


@app.post("/vision/scene")
async def vision_scene(request: Request) -> Dict[str, Any]:
    """Vision-LLM scene understanding: describe a camera frame (base64) with llava. Runs OUT of
    the safety loop. Broadcasts `vision.scene` so any surface can show it."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    image = (body.get("image") or "") if isinstance(body, dict) else ""
    kind = (body.get("kind") or "cabin") if isinstance(body, dict) else "cabin"
    context = body.get("context") if isinstance(body, dict) else None
    if not image:
        return {"ok": False, "error": "no image"}
    result = await asyncio.get_running_loop().run_in_executor(
        None, lambda: vision.describe(image, kind=kind, context=context)
    )
    if result.get("ok"):
        await broadcast(envelope(MessageType.VISION_SCENE, {
            "description": result["description"], "kind": kind, "driver": personas.get(current_driver).display_name,
        }))
    return result


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
    # Force a fresh multi-agent cycle so the agent graph reflects the alert immediately.
    await run_orchestration(trigger="alert", force=True)
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

    # Light up the Copilot node in the agent graph while it thinks.
    await run_orchestration(trigger="copilot", force=True, copilot_active=True)
    # Retrieval + generation are blocking (urllib) — run off the event loop.
    result = await asyncio.get_running_loop().run_in_executor(
        None, lambda: copilot.answer(query, ctx)
    )
    payload = {"query": query, "driver": p.display_name, **result}
    await broadcast(envelope(MessageType.COPILOT_RESPONSE, payload))
    return {"ok": True, **payload}


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """A rich, read-only snapshot of the edge brain — the live driver state, the predictive
    world-model forecast, the latest multi-agent decision, and the last explanation. This is the
    single source the MCP server exposes to external AI clients, and it powers the agent graph on
    a late-joining dashboard."""
    p = personas.get(current_driver)
    return {
        "driver": {"id": current_driver, "name": p.display_name, "threshold": round(p.drowsiness_threshold, 1),
                   "genericThreshold": 50.0, "modality": p.preferred_modality, "playlist": p.playlist},
        "live": latest_state,
        "forecast": latest_forecast or forecaster.latest(current_driver, p.drowsiness_threshold),
        "orchestration": latest_orchestration,
        "lastExplanation": last_reasoning_text,
        "telemetry": latest_telemetry,
    }


@app.post("/agents/run")
async def agents_run() -> Dict[str, Any]:
    """Force one multi-agent cycle now (demo control / MCP)."""
    payload = await run_orchestration(trigger="manual", force=True)
    return {"ok": True, "orchestration": payload}


@app.post("/agents/countermeasure")
async def agents_countermeasure(request: Request) -> Dict[str, Any]:
    """Broadcast a Wellness countermeasure on demand. Lets an external MCP client (e.g. Claude)
    actually make the car do something — 'cool the cabin', 'play upbeat music', 'find a rest stop'."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    kind = (body.get("kind") or "").strip().lower() if isinstance(body, dict) else ""
    detail = body.get("detail") if isinstance(body, dict) else None
    presets = {
        "climate": {"type": "climate", "detail": detail or "Cool cabin to 19°C"},
        "music": {"type": "music", "detail": detail or "Switch to an upbeat playlist"},
        "windows": {"type": "windows", "detail": detail or "Crack windows for airflow"},
        "navigation": {"type": "navigation", "detail": detail or "Route to the nearest rest stop"},
        "rest": {"type": "navigation", "detail": detail or "Route to the nearest rest stop"},
        "break": {"type": "audio", "detail": detail or "Suggest taking a short break"},
    }
    action = presets.get(kind)
    if not action:
        return {"ok": False, "error": f"unknown countermeasure '{kind}'", "options": list(presets)}
    await broadcast(envelope(MessageType.COUNTERMEASURE, {
        "driver": personas.get(current_driver).display_name, "level": None, "actions": [action], "source": "mcp",
    }))
    return {"ok": True, "action": action}


@app.get("/llm/status")
async def llm_status() -> Dict[str, Any]:
    """Report the active LLM brain (cloud/on-device) + model so the UI can badge it honestly."""
    loop = asyncio.get_running_loop()
    st = await loop.run_in_executor(None, llm.status)
    st["kbChunks"] = len(copilot.chunks)
    st["chatModel"] = st.get("model")
    st["visionAvailable"] = await loop.run_in_executor(None, vision.available)
    st["visionModel"] = vision.VISION_MODEL
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
    global latest_state, latest_forecast
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

    # Predictive world model: update the forecast from the new score and broadcast it.
    try:
        s = float(payload.get("score", 0) or 0)
        latest_forecast = forecaster.update(current_driver, s, p.drowsiness_threshold)
        await broadcast(envelope(MessageType.FORECAST, latest_forecast), quiet=True)
    except Exception as e:
        log.warning("forecast update failed (%s)", e)

    # Multi-agent decision cycle (throttled inside run_orchestration).
    await run_orchestration(trigger="state")
    return {"ok": True}
