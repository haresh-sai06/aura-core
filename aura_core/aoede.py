"""Aoede — Aura's real-time voice + orchestrator, powered by the Gemini Live API.

Core hosts the live session and proxies audio to/from the browser head unit, so the Gemini
API key never leaves the server and Aoede sits right next to Core's live state + tools.

Phase 1: natural audio-to-audio buddy (Aoede voice, live transcripts).
Phase 2: TOOL-CALLING — Aoede can actually drive the car. UI tools (climate/music/nav) are
relayed to the browser to apply against the head-unit OS; state/memory tools run in Core.

Wire protocol (browser <-> Core WS at /ws/aoede), JSON text frames:
  browser -> core : {type:"init", name, driverId} | {type:"audio", data:<b64 pcm16 @16k>} |
                    {type:"text", data} | {type:"event", data} | {type:"end"}
  core -> browser : {type:"ready"} | {type:"audio", data:<b64 pcm16 @24k>} |
                    {type:"transcript", role, text} | {type:"tool", name, args} |
                    {type:"interrupted"} | {type:"turn_complete"} | {type:"error", error}
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from . import knowledge, llm

log = logging.getLogger("aura-core")
router = APIRouter()

# Active live sessions, so an external trigger (e.g. the distraction detector) can inject a
# proactive SAFETY EVENT into whatever Aoede session is currently talking to the driver.
_LIVE_SESSIONS: dict = {}

MODEL = "gemini-3.1-flash-live-preview"
VOICE = "Aoede"
SYSTEM = (
    "You are Aoede, the voice of Aura — a warm, witty in-car co-pilot and friend riding "
    "shotgun, not a formal assistant. Keep replies short, natural and personable. You can "
    "actually control the car with your tools: set the climate, play music, set a "
    "destination, remember what the driver likes, and check how they're doing. When the "
    "driver asks for something you can do, DO IT with the tool and then say so briefly. If "
    "you get a line starting 'SAFETY EVENT:', address it immediately — kindly but firmly, "
    "briefly. Never mention that you're an AI."
)

UI_TOOLS = {"set_climate", "play_music", "navigate_to"}

_TOOL_DECLS = [
    {
        "name": "set_climate",
        "description": "Set the cabin temperature (Celsius) and/or turn the air-conditioning on or off.",
        "parameters": {"type": "OBJECT", "properties": {
            "temp": {"type": "INTEGER", "description": "Target temperature 16-30 C"},
            "ac": {"type": "BOOLEAN", "description": "A/C on (true) or off (false)"},
        }},
    },
    {
        "name": "play_music",
        "description": "Start music, optionally matching a mood.",
        "parameters": {"type": "OBJECT", "properties": {
            "mood": {"type": "STRING", "description": "calm, upbeat, focus, or a genre"},
        }},
    },
    {
        "name": "navigate_to",
        "description": "Set the navigation destination.",
        "parameters": {"type": "OBJECT", "properties": {
            "place": {"type": "STRING", "description": "Destination, e.g. home, office, airport, or a place name"},
        }, "required": ["place"]},
    },
    {
        "name": "remember_preference",
        "description": "Remember a durable preference or fact about this driver for their Driver DNA.",
        "parameters": {"type": "OBJECT", "properties": {
            "text": {"type": "STRING", "description": "The preference/fact, e.g. 'prefers calm music'"},
            "category": {"type": "STRING", "description": "music, destination, climate, mood, driving, food, general"},
        }, "required": ["text"]},
    },
    {
        "name": "get_driver_state",
        "description": "Check the driver's current drowsiness score, vehicle speed and scenario.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
]


def gemini_key() -> str:
    return (llm._config().get("gemini_api_key") or "").strip()


def maps_key() -> str:
    return (llm._config().get("GOOGLE_MAPS_API_KEY") or llm._config().get("google_maps_api_key") or "").strip()


def _live_config():
    from google.genai import types
    kwargs = dict(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE))),
        system_instruction=types.Content(parts=[types.Part(text=SYSTEM)]),
        tools=[types.Tool(function_declarations=_TOOL_DECLS)],
    )
    try:
        kwargs["input_audio_transcription"] = types.AudioTranscriptionConfig()
        kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()
    except Exception:
        pass
    return types.LiveConnectConfig(**kwargs)


@router.get("/aoede/status")
def aoede_status() -> dict:
    return {"available": bool(gemini_key()), "model": MODEL, "voice": VOICE}


@router.get("/maps/key")
def get_maps_key() -> dict:
    """Expose the Maps JS key to the (localhost) head unit. Restrict it by HTTP referrer in
    the Google Cloud console — it's meant to be browser-visible."""
    return {"key": maps_key()}


# A plausible "current location" for the demo (Directions needs an origin).
DEMO_ORIGIN = "MG Road, Bengaluru"


def _directions(origin: str, destination: str) -> dict:
    import urllib.parse
    import urllib.request
    key = maps_key()
    url = ("https://maps.googleapis.com/maps/api/directions/json?"
           f"origin={urllib.parse.quote(origin)}&destination={urllib.parse.quote(destination)}"
           f"&mode=driving&key={key}")
    return json.loads(urllib.request.urlopen(url, timeout=12).read())


@router.get("/maps/route")
def maps_route(origin: str = "", destination: str = "") -> dict:
    """Real driving route (distance + ETA) via the Directions API — shown on the Nav card."""
    if not maps_key() or not destination:
        return {"ok": False, "error": "maps key or destination missing"}
    try:
        d = _directions(origin or DEMO_ORIGIN, destination)
        if d.get("status") != "OK" or not d.get("routes"):
            return {"ok": False, "status": d.get("status")}
        leg = d["routes"][0]["legs"][0]
        return {"ok": True, "distance": leg["distance"]["text"], "duration": leg["duration"]["text"],
                "start": leg["start_address"], "end": leg["end_address"]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:120]}


@router.get("/maps/static")
def maps_static(origin: str = "", destination: str = "", center: str = "", size: str = "640x280"):
    """A real Google map PNG — a drawn route when `destination` is given, else a plain map
    centred on `center`. Proxied so the key stays server-side, warm-dark styled for the cabin."""
    from fastapi import Response
    import urllib.parse
    import urllib.request

    def q(s: str) -> str:
        return urllib.parse.quote(s, safe="")

    key = maps_key()
    if not key:
        return Response(status_code=404)

    style = "".join("&style=" + q(s) for s in (
        "element:geometry|color:0x1b1712",
        "element:labels.text.fill|color:0xbcb2a2",
        "element:labels.text.stroke|color:0x100e0b",
        "feature:road|element:geometry|color:0x2a2018",
        "feature:water|element:geometry|color:0x0e1a1f",
    ))
    base = f"https://maps.googleapis.com/maps/api/staticmap?size={size}&scale=2"

    if destination:
        o = origin or DEMO_ORIGIN
        path = ""
        try:
            d = _directions(o, destination)
            if d.get("status") == "OK" and d.get("routes"):
                poly = d["routes"][0]["overview_polyline"]["points"]
                path = "&path=" + q(f"color:0xe8c99cff|weight:5|enc:{poly}")
        except Exception:
            pass
        url = (base + "&markers=" + q(f"color:0x4f8cff|label:A|{o}")
               + "&markers=" + q(f"color:0x7cc292|label:B|{destination}")
               + style + path + f"&key={key}")
    elif center:
        url = (base + "&zoom=13&center=" + q(center)
               + "&markers=" + q(f"color:0xcfa46a|{center}") + style + f"&key={key}")
    else:
        return Response(status_code=404)

    try:
        img = urllib.request.urlopen(url, timeout=12).read()
        return Response(content=img, media_type="image/png")
    except Exception:
        return Response(status_code=502)


@router.post("/aoede/event")
async def aoede_event(request: Request) -> dict:
    """Inject a proactive SAFETY EVENT into every active Aoede session — she'll interrupt and
    warn the driver. Used by the distraction detector and the drowsiness path. Body: {text}."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    text = str(body.get("text", "")).strip() if isinstance(body, dict) else ""
    if not text:
        return {"ok": False, "error": "text required"}
    sent = 0
    for session in list(_LIVE_SESSIONS.values()):
        try:
            await session.send_realtime_input(text=f"SAFETY EVENT: {text}")
            sent += 1
        except Exception:
            pass
    # Also surface it on the main bus so the dashboard (Live Monitor) can show it, and the
    # fallback head-unit voice can warn even when no live Aoede session is up.
    try:
        from . import server as srv
        from .messages import envelope
        await srv.broadcast(envelope("distraction", {"text": text, "sessions": sent}))
    except Exception:
        pass
    return {"ok": True, "sessions": sent}


async def _handle_tool(fc, ws: WebSocket, ctx: dict) -> dict:
    name = getattr(fc, "name", "")
    args = dict(getattr(fc, "args", None) or {})
    log.info("aoede tool-call: %s %s", name, args)
    if name in UI_TOOLS:
        # Relay to the head unit to apply against the OS; report success to Aoede.
        try:
            await ws.send_text(json.dumps({"type": "tool", "name": name, "args": args}))
        except Exception:
            pass
        return {"ok": True, "applied": args}
    if name == "remember_preference":
        knowledge.add_facts(ctx.get("driver_id") or "guest",
                            [{"text": args.get("text", ""), "category": args.get("category", "general")}])
        return {"ok": True}
    if name == "get_driver_state":
        try:
            from . import server as srv
            st = srv.latest_state or {}
            tel = srv.latest_telemetry or {}
            return {"driver": st.get("driver") or getattr(srv, "current_driver", None),
                    "drowsiness_score": st.get("score"), "speed_kmh": tel.get("speedKmh"),
                    "scenario": tel.get("scenario")}
        except Exception as e:
            return {"error": str(e)}
    return {"ok": False, "error": f"unknown tool {name}"}


@router.websocket("/ws/aoede")
async def aoede_ws(ws: WebSocket) -> None:
    await ws.accept()
    if not gemini_key():
        await ws.send_text(json.dumps({"type": "error", "error": "Gemini key not configured"}))
        await ws.close()
        return

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=gemini_key())
    ctx = {"driver_id": "guest"}

    try:
        async with client.aio.live.connect(model=MODEL, config=_live_config()) as session:
            await ws.send_text(json.dumps({"type": "ready", "voice": VOICE}))
            _LIVE_SESSIONS[id(ws)] = session

            async def browser_to_gemini() -> None:
                while True:
                    msg = json.loads(await ws.receive_text())
                    t = msg.get("type")
                    if t == "init":
                        ctx["driver_id"] = (msg.get("driverId") or knowledge.slug(msg.get("name", "guest")) or "guest")
                    elif t == "audio":
                        await session.send_realtime_input(audio=types.Blob(data=base64.b64decode(msg["data"]), mime_type="audio/pcm;rate=16000"))
                    elif t == "text" and msg.get("data"):
                        await session.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=msg["data"])]), turn_complete=True)
                    elif t == "event" and msg.get("data"):
                        await session.send_realtime_input(text=f"SAFETY EVENT: {msg['data']}")
                    elif t == "end":
                        await session.send_realtime_input(audio_stream_end=True)

            async def gemini_to_browser() -> None:
                async for resp in session.receive():
                    data = getattr(resp, "data", None)
                    if data:
                        await ws.send_text(json.dumps({"type": "audio", "data": base64.b64encode(data).decode()}))
                    tc = getattr(resp, "tool_call", None)
                    if tc and getattr(tc, "function_calls", None):
                        frs = []
                        for fc in tc.function_calls:
                            result = await _handle_tool(fc, ws, ctx)
                            frs.append(types.FunctionResponse(id=getattr(fc, "id", None), name=getattr(fc, "name", None), response=result))
                        await session.send_tool_response(function_responses=frs)
                    sc = getattr(resp, "server_content", None)
                    if not sc:
                        continue
                    it = getattr(sc, "input_transcription", None)
                    if it and getattr(it, "text", None):
                        await ws.send_text(json.dumps({"type": "transcript", "role": "user", "text": it.text}))
                    ot = getattr(sc, "output_transcription", None)
                    if ot and getattr(ot, "text", None):
                        await ws.send_text(json.dumps({"type": "transcript", "role": "aoede", "text": ot.text}))
                    if getattr(sc, "interrupted", False):
                        await ws.send_text(json.dumps({"type": "interrupted"}))
                    if getattr(sc, "turn_complete", False):
                        await ws.send_text(json.dumps({"type": "turn_complete"}))

            b2g = asyncio.create_task(browser_to_gemini())
            g2b = asyncio.create_task(gemini_to_browser())
            try:
                _, pending = await asyncio.wait({b2g, g2b}, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
            finally:
                _LIVE_SESSIONS.pop(id(ws), None)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning("aoede ws error: %s", e)
        try:
            await ws.send_text(json.dumps({"type": "error", "error": str(e)[:160]}))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
