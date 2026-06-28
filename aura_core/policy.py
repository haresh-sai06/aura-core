"""Adaptive safety policy — the project's differentiator.

A generic DMS warns at one fixed threshold for everyone. Aura warns against the
driver's OWN baseline, and in the modality that works for them. Same input signal,
personalized decision. Right now the input (eye-closure duration) is faked; later it
comes from the camera/CV layer — but the policy is already the real, pitchable logic.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .messages import MessageType, envelope
from .persona import PersonaStore


class AdaptiveSafetyPolicy:
    def __init__(self, personas: PersonaStore) -> None:
        self.personas = personas

    def evaluate(self, driver_id: str, eye_closure_s: float) -> Optional[Dict[str, Any]]:
        """Return a safety.alert envelope if eye-closure exceeds THIS driver's baseline, else None."""
        p = self.personas.get(driver_id)
        if eye_closure_s < p.eye_closure_threshold_s:
            return None  # normal for this person — a generic system might false-alarm here

        payload = {
            "level": "critical",
            "reason": f"Eyes closed {eye_closure_s:.1f}s (your baseline {p.eye_closure_threshold_s:.1f}s)",
            "action": "pull_over",
            "modality": p.preferred_modality,
            "driver": p.display_name,
        }
        return envelope(MessageType.SAFETY_ALERT, payload)
