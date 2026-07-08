"""Adaptive safety policy — the project's differentiator.

A generic DMS warns at one fixed threshold for everyone. Aura warns against the
driver's OWN baseline (which also *adapts* over time — see [persona.py]), and in the
modality that works for them. Same input signal, personalized decision. The policy also
emits an `explain` envelope so every alert can answer "why did Aura act for ME?".
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .messages import MessageType, envelope
from .persona import PersonaStore

# What a one-size-fits-all system would use — shown alongside the personal line so the
# dashboard can make the differentiator visible.
GENERIC_FIXED_THRESHOLD = 50.0


class AdaptiveSafetyPolicy:
    def __init__(self, personas: PersonaStore) -> None:
        self.personas = personas

    def evaluate(
        self,
        driver_id: str,
        eye_closure_s: Optional[float] = None,
        score: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return a safety.alert envelope if the signal exceeds THIS driver's personal
        threshold. Prefers the fused drowsiness `score`; falls back to raw `eye_closure_s`."""
        p = self.personas.get(driver_id)

        if score is not None:
            if score < p.drowsiness_threshold:
                return None  # not drowsy enough for THIS driver — a generic DMS might false-alarm
            reason = f"Drowsiness score {score:.0f} (your adaptive threshold {p.drowsiness_threshold:.0f})"
        elif eye_closure_s is not None:
            if eye_closure_s < p.eye_closure_threshold_s:
                return None
            reason = f"Eyes closed {eye_closure_s:.1f}s (your baseline {p.eye_closure_threshold_s:.1f}s)"
        else:
            return None

        payload = {
            "level": "critical",
            "reason": reason,
            "action": "pull_over",
            "modality": p.preferred_modality,
            "driver": p.display_name,
        }
        return envelope(MessageType.SAFETY_ALERT, payload)

    def explain(
        self,
        driver_id: str,
        score: Optional[float] = None,
        eye_closure_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build an `explain` envelope — the personalized 'why' behind a decision.

        Makes the differentiator concrete: shows the driver's personal (adaptive) line
        against the generic fixed line, and whether a one-size-fits-all system would have
        false-alarmed or missed this driver.
        """
        p = self.personas.get(driver_id)
        personal = p.drowsiness_threshold
        val = score if score is not None else 0.0

        factors = [
            {"name": "Drowsiness score", "value": round(val, 0)},
            {"name": "Your adaptive threshold", "value": round(personal, 0)},
            {"name": "Generic fixed threshold", "value": round(GENERIC_FIXED_THRESHOLD, 0)},
            {"name": "Tuned from baseline", "value": round(p.base_threshold, 0)},
        ]

        if score is not None:
            personal_fire = score >= personal
            generic_fire = score >= GENERIC_FIXED_THRESHOLD
            if personal_fire and not generic_fire:
                decision = f"Aura warned earlier than a generic system would for {p.display_name}."
            elif generic_fire and not personal_fire:
                decision = f"A generic system would have false-alarmed {p.display_name}; Aura held back."
            elif personal_fire and generic_fire:
                decision = f"Both would warn — Aura confirms with {p.display_name}'s own baseline."
            else:
                margin = personal - score
                decision = f"{p.display_name} is alert — {margin:.0f} points below their personal line."
        else:
            decision = f"Monitoring against {p.display_name}'s personal baseline."

        payload = {
            "driver": p.display_name,
            "decision": decision,
            "personalThreshold": round(personal, 1),
            "genericThreshold": GENERIC_FIXED_THRESHOLD,
            "modality": p.preferred_modality,
            "factors": factors,
        }
        return envelope(MessageType.EXPLAIN, payload)
