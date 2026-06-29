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

    def evaluate(
        self,
        driver_id: str,
        eye_closure_s: Optional[float] = None,
        score: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return a safety.alert envelope if the signal exceeds THIS driver's personal threshold.

        Prefers the fused drowsiness `score` (from the browser's 7-signal pipeline); falls back to
        raw `eye_closure_s` (from the simpler Python camera). Either way the threshold is personal —
        a sharp driver alerts later than a tired one. That's the differentiator.
        """
        p = self.personas.get(driver_id)

        if score is not None:
            if score < p.drowsiness_threshold:
                return None  # not drowsy enough for THIS driver — a generic DMS might false-alarm
            reason = f"Drowsiness score {score:.0f} (your threshold {p.drowsiness_threshold:.0f})"
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
