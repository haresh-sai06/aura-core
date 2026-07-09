"""Reasoning Agent — Aura's language layer over the perception world-model.

This is the Vision -> Language -> Action bridge. MediaPipe + the 7-signal fusion are the
*vision* (a real-time world-model of the DRIVER); the adaptive policy chooses the *action*;
this module is the *language* in between. It takes the structured driver state, the driver's
persona (their personal, adaptive threshold), and the live Unity telemetry, and asks the local
qwen2.5:7b to produce a concise, human-readable justification + a recommended action.

Why this matters: a fixed-threshold DMS can only say "score 62 > 50". Aura can say *why it
acted for THIS driver, right now* — turning a number into an explanation a human trusts. It
runs on-device (no cloud) and degrades to a deterministic template if the LLM is unavailable,
so the safety story never depends on the model being up.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional

from . import llm
from .persona import Persona

log = logging.getLogger("aura-core")

GENERIC_FIXED_THRESHOLD = 50.0

_SYSTEM = (
    "You are Aura, an on-device automotive safety reasoner. Given a driver's live drowsiness "
    "signals, their PERSONAL adaptive threshold, and the vehicle state, explain in 2-3 short "
    "sentences why the system is acting (or holding) FOR THIS DRIVER. Contrast with a generic "
    "fixed-threshold system when relevant. Be specific about the signals. Calm, precise, no "
    "hype, no invented numbers. End with one short recommended action in the form "
    "'Action: ...'."
)


def _facts(persona: Persona, state: Dict[str, Any], telemetry: Optional[Dict[str, Any]]) -> str:
    score = state.get("score")
    eye_closure = state.get("eyeClosureS")
    lines = [f"Driver: {persona.display_name}"]
    # The trigger can be the fused SCORE (browser 7-signal pipeline) or raw EYE-CLOSURE
    # seconds (Python/Unity camera path). Present whichever is the real trigger so the
    # reasoning never contradicts the alert.
    if score is not None:
        lines += [
            f"Fused drowsiness score (0-100): {score}",
            f"This driver's personal adaptive threshold: {persona.drowsiness_threshold:.0f}",
            f"Generic fixed threshold (for contrast): {GENERIC_FIXED_THRESHOLD:.0f}",
            f"Seed baseline before learning: {persona.base_threshold:.0f}",
        ]
    elif eye_closure is not None:
        lines += [
            f"Continuous eye-closure: {eye_closure} s (the trigger signal)",
            f"This driver's personal eye-closure baseline: {persona.eye_closure_threshold_s:.1f} s",
        ]
    else:
        lines += [f"This driver's personal adaptive threshold: {persona.drowsiness_threshold:.0f}"]
    lines.append(f"Preferred alert modality: {persona.preferred_modality}")
    # Per-signal detail when the browser pipeline provides it — makes the reasoning concrete.
    for key, label in (
        ("ear", "Eye Aspect Ratio (lower = eyes closing)"),
        ("mar", "Mouth Aspect Ratio (higher = yawning)"),
        ("perclos", "PERCLOS (fraction eyes closed)"),
        ("blinkRate", "Blink rate"),
        ("blinkDuration", "Blink duration (s)"),
        ("headPitch", "Head pitch (nodding)"),
        ("gazeStability", "Gaze stability"),
    ):
        if state.get(key) is not None:
            lines.append(f"{label}: {state[key]}")
    if telemetry:
        if telemetry.get("speedKmh") is not None:
            lines.append(f"Vehicle speed: {telemetry['speedKmh']} km/h")
        if telemetry.get("scenario"):
            lines.append(f"Driving scenario: {telemetry['scenario']}")
    return "\n".join(str(x) for x in lines)


def _fallback(persona: Persona, state: Dict[str, Any], acted: bool) -> str:
    """Deterministic explanation used when the LLM is offline — never leaves the driver
    without a 'why'."""
    score = state.get("score")
    personal = persona.drowsiness_threshold
    try:
        s = float(score)
    except (TypeError, ValueError):
        s = 0.0
    if acted:
        earlier = "earlier than" if personal < GENERIC_FIXED_THRESHOLD else "in line with"
        return (
            f"Fused score {s:.0f} crossed {persona.display_name}'s personal line of "
            f"{personal:.0f} — Aura acted {earlier} a generic system would. "
            f"Action: escalate via {persona.preferred_modality} and prepare a safe pull-over."
        )
    margin = personal - s
    return (
        f"Fused score {s:.0f} is {margin:.0f} points below {persona.display_name}'s personal "
        f"line of {personal:.0f}. Aura is holding — a generic fixed system might have "
        f"false-alarmed. Action: keep monitoring."
    )


def reason(
    persona: Persona,
    state: Dict[str, Any],
    telemetry: Optional[Dict[str, Any]] = None,
    acted: bool = True,
) -> str:
    """One-shot natural-language justification (blocking). Falls back to a template."""
    facts = _facts(persona, state, telemetry)
    prompt = f"{facts}\n\nThe system {'IS ACTING' if acted else 'is HOLDING (no alert)'}. Explain why for this driver."
    text = llm.chat(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.4,
        num_predict=180,
    )
    return text or _fallback(persona, state, acted)


def reason_stream(
    persona: Persona,
    state: Dict[str, Any],
    telemetry: Optional[Dict[str, Any]] = None,
    acted: bool = True,
) -> Iterator[str]:
    """Token stream for the live 'typing' World Model panel. Yields the fallback as a single
    chunk if the model is unavailable."""
    if not llm.available():
        yield _fallback(persona, state, acted)
        return
    facts = _facts(persona, state, telemetry)
    prompt = f"{facts}\n\nThe system {'IS ACTING' if acted else 'is HOLDING (no alert)'}. Explain why for this driver."
    produced = False
    for piece in llm.chat_stream(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.4,
        num_predict=180,
    ):
        produced = True
        yield piece
    if not produced:
        yield _fallback(persona, state, acted)
