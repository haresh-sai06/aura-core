"""Vision-Language scene understanding — the literal "Vision" in Vision→Language→Action.

MediaPipe stays Aura's fast, deterministic SAFETY sensor (the numbers that trigger a takeover).
This module adds a *semantic* layer on top: a local vision-language model (llava via Ollama)
looks at an actual camera frame and describes it in words — "driver rubbing their eyes, one hand
off the wheel, phone visible" or, for a road frame, "pedestrian near the crossing, wet road".

It is deliberately OUT of the safety loop: it runs occasionally, enriches the explanation, and
never gates a life-safety decision (a slow, hallucination-prone LLM must never do that). Fully
on-device; if llava isn't installed it reports unavailable and the rest of Aura is unaffected.
"""
from __future__ import annotations

import base64
import json
import logging
import struct
import urllib.request
import zlib
from typing import Any, Dict, Optional

log = logging.getLogger("aura-core")

OLLAMA_URL = "http://127.0.0.1:11434"
VISION_MODEL = "llava"
_TIMEOUT = 90

_CABIN_PROMPT = (
    "You are an in-cabin driver-monitoring camera. In 1-2 short sentences, describe only what is "
    "relevant to driving safety: the driver's eyes (open/closing), head pose, hands on/off the "
    "wheel, any phone or distraction, seatbelt, and visible signs of fatigue. Be factual and brief. "
    "If something isn't visible, don't guess."
)
_ROAD_PROMPT = (
    "You are a forward driving camera. In 1-2 short sentences, describe the road scene relevant to "
    "safety: road users (pedestrians, vehicles), lane, weather/road surface, and any hazard. Be "
    "factual and brief."
)


def available() -> bool:
    """Is the llava vision model present in the local Ollama? Lets the UI show an honest badge."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            tags = json.loads(r.read().decode("utf-8"))
        return any("llava" in (m.get("name") or "") for m in tags.get("models", []))
    except Exception:
        return False


def _strip_data_url(b64: str) -> str:
    """Accept either a raw base64 string or a full data: URL and return raw base64."""
    if b64.startswith("data:"):
        return b64.split(",", 1)[-1]
    return b64


def describe(image_b64: str, kind: str = "cabin", context: Optional[str] = None) -> Dict[str, Any]:
    """Describe one frame with llava. Returns {ok, description, model}. Best-effort — on any
    failure returns ok=False with a short reason so the caller degrades gracefully."""
    if not image_b64:
        return {"ok": False, "description": "", "error": "no image"}
    img = _strip_data_url(image_b64)
    prompt = _ROAD_PROMPT if kind == "road" else _CABIN_PROMPT
    if context:
        prompt += f"\nContext: {context}"
    body = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [img]}],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.2, "num_predict": 120},
    }
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        text = (out.get("message", {}) or {}).get("content", "").strip()
        return {"ok": bool(text), "description": text or "(no description)", "model": VISION_MODEL, "kind": kind}
    except Exception as e:
        log.warning("vision.describe failed (%s) — is llava pulled? `ollama pull llava`", e)
        return {"ok": False, "description": "", "error": str(e), "kind": kind}


def _tiny_png(size: int = 32, gray: int = 120) -> str:
    """Build a small VALID RGB PNG with only the stdlib (llava 400s on a 1x1), base64-encoded.
    Used to pre-load the vision model into memory so the first real call isn't the slow one."""
    row = b"\x00" + bytes([gray, gray, gray]) * size          # filter byte + RGB pixels
    raw = row * size
    def chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


def warm() -> bool:
    """Pre-load llava into memory with a valid tiny image. Returns True if it responded."""
    return describe(_tiny_png(), kind="cabin").get("ok", False)
