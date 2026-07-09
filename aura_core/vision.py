"""Vision-Language scene understanding — the literal "Vision" in Vision→Language→Action.

MediaPipe stays Aura's fast, deterministic SAFETY sensor (the numbers that trigger a takeover).
This module adds a *semantic* layer on top: a multimodal model looks at an actual camera frame and
describes it — "driver rubbing their eyes, one hand off the wheel, phone visible" or, for a road
frame, "pedestrian near the crossing, wet road".

Brain routing (mirrors [llm.py]): **Gemini multimodal** (fast, sharp) is primary when a Gemini key
is set; **local llava via Ollama** is the on-device fallback if the cloud is unavailable. Either
way it is deliberately OUT of the safety loop — it enriches the explanation and never gates a
life-safety decision.
"""
from __future__ import annotations

import base64
import json
import logging
import struct
import urllib.request
import zlib
from typing import Any, Dict, Optional

from . import llm

log = logging.getLogger("aura-core")

OLLAMA_URL = "http://127.0.0.1:11434"
LLAVA_MODEL = "llava"
VISION_MODEL = "llava"   # kept for back-compat; prefer active_model()
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


def _llava_present() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            tags = json.loads(r.read().decode("utf-8"))
        return any("llava" in (m.get("name") or "") for m in tags.get("models", []))
    except Exception:
        return False


def available() -> bool:
    """Can we describe a frame at all? True if a Gemini key is set OR local llava is present."""
    return bool(llm._gemini_key()) or _llava_present()


def active_model() -> str:
    """Which vision brain is in effect (for an honest status badge)."""
    return llm._gemini_model() if llm._gemini_key() else LLAVA_MODEL


def _strip_data_url(b64: str) -> str:
    """Return raw base64 whether given a raw string or a full data: URL."""
    return b64.split(",", 1)[-1] if b64.startswith("data:") else b64


# ── Gemini multimodal (primary) ──────────────────────────────────────
def _describe_gemini(data_url: str, prompt: str) -> Optional[str]:
    key = llm._gemini_key()
    if not key:
        return None
    body = {
        "model": llm._gemini_model(),
        "max_tokens": 150,
        "temperature": 0.2,
        "reasoning_effort": "none",   # gemini-2.5-flash thinks by default; keep the reply intact
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
    }
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            llm.GEMINI_URL, data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        return (out["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        log.warning("vision (gemini) failed (%s) — falling back to local llava", e)
        return None


# ── llava on Ollama (on-device fallback) ─────────────────────────────
def _describe_llava(img_raw: str, prompt: str) -> Optional[str]:
    body = {
        "model": LLAVA_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [img_raw]}],
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
        return (out.get("message", {}) or {}).get("content", "").strip()
    except Exception as e:
        log.warning("vision (llava) failed (%s) — is llava pulled? `ollama pull llava`", e)
        return None


def describe(image_b64: str, kind: str = "cabin", context: Optional[str] = None) -> Dict[str, Any]:
    """Describe one frame. Tries Gemini multimodal first, then local llava. Returns
    {ok, description, model, kind, via}; degrades gracefully on failure."""
    if not image_b64:
        return {"ok": False, "description": "", "error": "no image"}
    raw = _strip_data_url(image_b64)
    data_url = image_b64 if image_b64.startswith("data:") else f"data:image/jpeg;base64,{raw}"
    prompt = _ROAD_PROMPT if kind == "road" else _CABIN_PROMPT
    if context:
        prompt += f"\nContext: {context}"

    # 1) Gemini multimodal (primary).
    if llm._gemini_key():
        text = _describe_gemini(data_url, prompt)
        if text:
            return {"ok": True, "description": text, "model": llm._gemini_model(), "kind": kind, "via": "gemini"}

    # 2) On-device llava (fallback).
    text = _describe_llava(raw, prompt)
    if text:
        return {"ok": True, "description": text, "model": LLAVA_MODEL, "kind": kind, "via": "llava"}
    return {"ok": False, "description": "", "error": "no vision brain available", "kind": kind}


def _tiny_png(size: int = 32, gray: int = 120) -> str:
    """A small VALID RGB PNG built with only the stdlib — used to warm/validate the vision path."""
    row = b"\x00" + bytes([gray, gray, gray]) * size
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
    """Validate/pre-load the active vision path (Gemini has no cold-start; llava gets resident)."""
    return describe(f"data:image/png;base64,{_tiny_png()}", kind="cabin").get("ok", False)
