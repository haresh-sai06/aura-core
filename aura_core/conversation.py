"""Aura the buddy — a warm in-car conversation that quietly learns the driver.

Two LLM passes per turn (via the swappable brain in [llm.py]):
  1. REPLY      — Aura answers like a friendly co-pilot, grounded in what it already knows
                  about this driver, and nudges the chat forward (mood / destination / vibe).
  2. EXTRACT    — a second pass distils any durable preferences/facts from the driver's
                  message into structured records, stored per-driver in [knowledge.py].

Exposed as a FastAPI router that Aura Core includes, so it shares the same host/port as the
rest of the edge brain. Endpoints are sync `def` so the blocking LLM calls run in FastAPI's
threadpool instead of stalling the event loop.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from fastapi import APIRouter, Request

from . import knowledge, llm

log = logging.getLogger("aura-core")
router = APIRouter()

_BUDDY_SYSTEM = (
    "You are Aura, a warm, witty in-car co-pilot and friend — not a formal assistant. Talk to "
    "the driver like a buddy riding shotgun. Keep replies to 1-2 short sentences, natural and "
    "personable. If it fits, gently ask ONE follow-up about their mood, where they're headed, "
    "or what music / cabin vibe they'd like. Use what you already know about them to feel "
    "familiar. Never mention that you're an AI or that you're extracting data."
)

_EXTRACT_SYSTEM = (
    "You extract durable facts and preferences about a car driver from their message. "
    "Return ONLY a JSON array (no prose, no code fences) of objects like "
    '{"text": "...", "category": "..."}. Categories: music, destination, climate, mood, '
    "driving, food, general. Capture only concrete, lasting preferences or facts (e.g. 'prefers "
    "calm music', 'commutes to the office', 'likes the cabin cool', 'gets drowsy at night'). "
    "If there's nothing durable, return []."
)


def _known_block(profile: Dict[str, Any]) -> str:
    facts = profile.get("facts", [])
    if not facts:
        return "You don't know anything about this driver yet — this is a fresh introduction."
    lines = "; ".join(f"{f['text']}" for f in facts[-12:])
    return f"What you already know about {profile.get('name','them')}: {lines}."


def _extract(message: str) -> List[Dict[str, str]]:
    raw = llm.chat(
        [{"role": "system", "content": _EXTRACT_SYSTEM},
         {"role": "user", "content": f"Driver said: \"{message}\""}],
        temperature=0.1, num_predict=180,
    )
    if not raw:
        return []
    # tolerate code fences / stray prose around the JSON array
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    if isinstance(arr, list):
        for item in arr:
            if isinstance(item, dict) and item.get("text"):
                out.append({"text": str(item["text"]), "category": str(item.get("category", "general"))})
    return out


@router.post("/conversation")
def conversation(body: Dict[str, Any]) -> Dict[str, Any]:
    """One buddy turn: reply + learn. Body: {driver_id|name, message, history?[]}."""
    if not isinstance(body, dict):
        return {"reply": "Sorry, I didn't catch that.", "facts": [], "profile": {}}
    message = str(body.get("message", "")).strip()
    driver_id = str(body.get("driver_id") or knowledge.slug(str(body.get("name", "guest"))))
    if body.get("name"):
        knowledge.set_name(driver_id, str(body["name"]))
    if not message:
        return {"reply": "", "facts": [], "profile": knowledge.get(driver_id)}

    profile = knowledge.get(driver_id)
    history = body.get("history") if isinstance(body.get("history"), list) else []
    messages = [{"role": "system", "content": f"{_BUDDY_SYSTEM}\n\n{_known_block(profile)}"}]
    for turn in history[-8:]:
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": str(turn["content"])})
    messages.append({"role": "user", "content": message})

    reply = llm.chat(messages, temperature=0.7, num_predict=110) or "I'm with you."
    new_facts = knowledge.add_facts(driver_id, _extract(message))
    return {"reply": reply, "facts": new_facts, "profile": knowledge.get(driver_id)}


@router.post("/driver/name")
async def driver_name(request: Request) -> Dict[str, Any]:
    """Register / greet a driver by spoken name. Returns their profile + id."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    name = str(body.get("name", "")).strip() if isinstance(body, dict) else ""
    if not name:
        return {"ok": False, "error": "name required"}
    profile = knowledge.register(name)
    return {"ok": True, "profile": profile, "id": profile.get("id")}


@router.get("/driver/knowledge")
def driver_knowledge(id: str = "") -> Dict[str, Any]:
    """The Driver DNA profile for the Safety Monitor screen (polled)."""
    if not id:
        return {"profiles": knowledge.all_profiles()}
    return knowledge.get(id)
