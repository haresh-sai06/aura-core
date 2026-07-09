"""Per-driver knowledge base — the "Driver DNA" that Aura builds through conversation.

As the driver chats with Aura, the buddy agent (see [conversation.py]) distils durable
preferences and facts about them — music taste, usual destinations, climate comfort, mood
patterns, driving habits — and stores them here, per driver. The Safety Monitor (System B)
renders this as a living profile that visibly grows as the conversation builds.

Storage is a best-effort JSON file next to the personas; nothing here is safety-critical, so
every operation degrades quietly rather than raising.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

log = logging.getLogger("aura-core")

_STORE_PATH = os.path.join(os.path.dirname(__file__), "driver_knowledge.json")
_lock = threading.Lock()


def slug(name: str) -> str:
    """A stable driver id from a spoken name ('Ravi Kumar' -> 'ravi')."""
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s.split("-")[0] if s else "guest"


def _load() -> Dict[str, Any]:
    try:
        if os.path.exists(_STORE_PATH):
            with open(_STORE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.warning("knowledge: could not load store (%s)", e)
    return {}


def _save(data: Dict[str, Any]) -> None:
    try:
        with open(_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log.warning("knowledge: could not save store (%s)", e)


def _profile(data: Dict[str, Any], driver_id: str) -> Dict[str, Any]:
    p = data.get(driver_id)
    if not isinstance(p, dict):
        p = {"id": driver_id, "name": driver_id.title(), "facts": []}
        data[driver_id] = p
    p.setdefault("facts", [])
    return p


def set_name(driver_id: str, name: str) -> Dict[str, Any]:
    with _lock:
        data = _load()
        p = _profile(data, driver_id)
        p["name"] = name.strip() or p.get("name") or driver_id.title()
        p["id"] = driver_id
        _save(data)
        return dict(p)


def register(name: str) -> Dict[str, Any]:
    """Register (or greet-again) a driver by spoken name; returns their profile + id."""
    did = slug(name)
    return set_name(did, name)


def add_facts(driver_id: str, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add extracted preferences/facts, de-duplicated (case-insensitive). Returns newly added."""
    added: List[Dict[str, Any]] = []
    with _lock:
        data = _load()
        p = _profile(data, driver_id)
        existing = {str(f.get("text", "")).strip().lower() for f in p["facts"]}
        for f in facts:
            text = str(f.get("text", "")).strip()
            if not text or text.lower() in existing:
                continue
            rec = {
                "text": text,
                "category": str(f.get("category", "general")).strip().lower() or "general",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            p["facts"].append(rec)
            existing.add(text.lower())
            added.append(rec)
        if added:
            _save(data)
    return added


def get(driver_id: str) -> Dict[str, Any]:
    with _lock:
        data = _load()
        return dict(_profile(data, driver_id))


def all_profiles() -> List[Dict[str, Any]]:
    with _lock:
        return list(_load().values())
