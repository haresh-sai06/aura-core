"""eCall — Aura's automatic emergency escalation.

When AutoCare reaches a safe-takeover (Level 4) and the driver stays unresponsive, Aura can
alert the driver's emergency contacts with WHO is in trouble, WHERE they are (GPS + a map link),
and WHAT the cabin camera sees. This mirrors the EU-mandated eCall: an automatic emergency call
with location on a serious incident.

Pluggable outbound channels (only the single alert leaves the device; everything else is local):
  • WhatsApp via CallMeBot  — free personal WhatsApp messages (HTTP GET + per-contact api key).
  • SMS via Twilio          — REST Messages API.
  • Voice call via Twilio    — REST Calls API with inline TwiML <Say> (no server to host).

Config + contacts live in emergency.json (gitignored). Stdlib-only HTTP so there are no new deps.
Everything is best-effort and returns a per-contact result so the dashboard can show delivery.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

log = logging.getLogger("aura-core")

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "emergency.json")
_TIMEOUT = 20

CALLMEBOT_URL = "https://api.callmebot.com/whatsapp.php"
TWILIO_BASE = "https://api.twilio.com/2010-04-01/Accounts"


# ── config store ─────────────────────────────────────────────────────
def load_config() -> Dict[str, Any]:
    try:
        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.warning("ecall: could not read emergency.json (%s)", e)
    return {"contacts": [], "twilio": {}, "defaultLocation": {}}


def save_config(cfg: Dict[str, Any]) -> None:
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        log.warning("ecall: could not write emergency.json (%s)", e)


def public_config() -> Dict[str, Any]:
    """Config for the dashboard with secrets redacted."""
    cfg = load_config()
    tw = cfg.get("twilio", {}) or {}
    return {
        "contacts": [{"name": c.get("name"), "phone": c.get("phone"), "channel": c.get("channel"),
                      "hasKey": bool(c.get("callmebot_key"))} for c in cfg.get("contacts", [])],
        "twilioConfigured": bool(tw.get("sid") and tw.get("token") and tw.get("from")),
        "twilioFrom": tw.get("from"),
        "defaultLocation": cfg.get("defaultLocation", {}),
    }


# ── message building ─────────────────────────────────────────────────
def build_text(payload: Dict[str, Any]) -> str:
    driver = payload.get("driver") or "The driver"
    loc = payload.get("location") or {}
    lat, lng = loc.get("lat"), loc.get("lng")
    parts = [f"🚨 AURA EMERGENCY — {driver} may be in danger.",
             "The vehicle detected an unresponsive driver and is performing a safe stop."]
    if payload.get("reason"):
        parts.append(f"Reason: {payload['reason']}.")
    if lat is not None and lng is not None:
        parts.append(f"Location: {lat:.5f}, {lng:.5f}")
        parts.append(f"Map: https://maps.google.com/?q={lat},{lng}")
        if loc.get("label"):
            parts.append(f"Near: {loc['label']}")
    if payload.get("speedKmh") is not None:
        parts.append(f"Speed: {payload['speedKmh']} km/h")
    if payload.get("scene"):
        parts.append(f"Cabin camera: {payload['scene']}")
    parts.append("Please respond or call emergency services.")
    return "\n".join(parts)


def build_voice_script(payload: Dict[str, Any]) -> str:
    """A short spoken script for a Twilio voice call (kept calm and clear)."""
    driver = payload.get("driver") or "the driver"
    loc = payload.get("location") or {}
    lat, lng = loc.get("lat"), loc.get("lng")
    where = ""
    if lat is not None and lng is not None:
        where = f" Their location is latitude {lat:.4f}, longitude {lng:.4f}."
    return (f"This is an automated Aura emergency call. {driver} may be in danger. "
            f"Their vehicle detected an unresponsive driver and is stopping safely.{where} "
            f"Please check on them or call emergency services. Repeating: {driver} may be in danger.")


# ── channel senders ──────────────────────────────────────────────────
def _send_whatsapp(phone: str, apikey: str, text: str) -> Dict[str, Any]:
    if not apikey:
        return {"ok": False, "error": "no CallMeBot api key for this contact"}
    q = urllib.parse.urlencode({"phone": phone, "text": text, "apikey": apikey})
    try:
        with urllib.request.urlopen(f"{CALLMEBOT_URL}?{q}", timeout=_TIMEOUT) as r:
            body = r.read().decode("utf-8", "ignore")
        return {"ok": True, "detail": body[:120]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _twilio_post(sid: str, token: str, endpoint: str, form: Dict[str, str]) -> Dict[str, Any]:
    url = f"{TWILIO_BASE}/{sid}/{endpoint}"
    data = urllib.parse.urlencode(form).encode()
    auth = base64.b64encode(f"{sid}:{token}".encode()).decode()
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            out = json.loads(r.read().decode("utf-8"))
        return {"ok": True, "sid": out.get("sid"), "status": out.get("status")}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode('utf-8','ignore')[:160]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _e164(phone: str) -> str:
    """Twilio requires E.164 (+countrycode…). The dashboard stores digits only, so add the +."""
    p = (phone or "").strip()
    return p if p.startswith("+") else "+" + p


def _send_sms(tw: Dict[str, str], to: str, text: str) -> Dict[str, Any]:
    if not (tw.get("sid") and tw.get("token") and tw.get("from")):
        return {"ok": False, "error": "Twilio not configured"}
    return _twilio_post(tw["sid"], tw["token"], "Messages.json",
                        {"To": _e164(to), "From": tw["from"], "Body": text})


def _send_call(tw: Dict[str, str], to: str, script: str) -> Dict[str, Any]:
    if not (tw.get("sid") and tw.get("token") and tw.get("from")):
        return {"ok": False, "error": "Twilio not configured"}
    twiml = f"<Response><Say voice=\"alice\">{_xml_escape(script)}</Say></Response>"
    return _twilio_post(tw["sid"], tw["token"], "Calls.json",
                        {"To": _e164(to), "From": tw["from"], "Twiml": twiml})


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


# ── dispatch ─────────────────────────────────────────────────────────
def send_test(index: int) -> Dict[str, Any]:
    """Send a friendly TEST message to one contact so the user can validate their credentials
    before relying on it in a real emergency. Same channel path, harmless content."""
    cfg = load_config()
    contacts = cfg.get("contacts", []) or []
    tw = cfg.get("twilio", {}) or {}
    if index < 0 or index >= len(contacts):
        return {"ok": False, "error": "no such contact"}
    c = contacts[index]
    ch = (c.get("channel") or "whatsapp").lower()
    phone = c.get("phone") or ""
    text = ("✅ Aura test alert — your emergency-contact setup is working. "
            "This is only a test; no action needed.")
    voice = ("This is a test call from Aura. Your emergency contact setup is working correctly. "
             "No action is needed. Goodbye.")
    if ch == "whatsapp":
        res = _send_whatsapp(phone, c.get("callmebot_key") or "", text)
    elif ch == "sms":
        res = _send_sms(tw, phone, text)
    elif ch == "call":
        res = _send_call(tw, phone, voice)
    else:
        res = {"ok": False, "error": f"unknown channel '{ch}'"}
    log.info("ecall TEST -> %s via %s: %s", c.get("name"), ch, "ok" if res.get("ok") else res.get("error"))
    return {"name": c.get("name"), "channel": ch, **res}


def dispatch(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Send the alert to every configured contact over its channel. Returns per-contact results."""
    cfg = load_config()
    contacts = cfg.get("contacts", []) or []
    tw = cfg.get("twilio", {}) or {}
    text = build_text(payload)
    voice = build_voice_script(payload)
    results: List[Dict[str, Any]] = []
    if not contacts:
        return [{"name": None, "channel": None, "ok": False, "error": "no emergency contacts configured"}]
    for c in contacts:
        ch = (c.get("channel") or "whatsapp").lower()
        phone = c.get("phone") or ""
        if ch == "whatsapp":
            res = _send_whatsapp(phone, c.get("callmebot_key") or "", text)
        elif ch == "sms":
            res = _send_sms(tw, phone, text)
        elif ch == "call":
            res = _send_call(tw, phone, voice)
        else:
            res = {"ok": False, "error": f"unknown channel '{ch}'"}
        results.append({"name": c.get("name"), "phone": phone, "channel": ch, **res})
        log.info("ecall -> %s via %s: %s", c.get("name"), ch, "ok" if res.get("ok") else res.get("error"))
    return results
