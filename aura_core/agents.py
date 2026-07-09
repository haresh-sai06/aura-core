"""Multi-agent orchestration — Aura's brain as a coordinated crew, not a monolith.

Instead of one function deciding everything, Aura runs a small team of specialist agents on
the shared message bus, arbitrated by a Supervisor (Orchestrator). Each decision cycle produces
a *trace* — which agents fired, what they concluded, how information flowed, and the final plan —
which the dashboard renders as a live agent graph.

The crew:
  • Perception   — summarizes the fused 7-signal driver state (owns the raw signals).
  • World Model  — the predictive forecaster: where is fatigue heading? (see [forecast.py])
  • Context      — trip context: time of night, hours driven, scenario.
  • Safety Policy — applies THIS driver's personal adaptive threshold (see [policy.py]).
  • Critic       — before a takeover, independently verifies signal agreement to veto false alarms.
  • Wellness     — picks the best *countermeasure* for this driver (cabin, music, rest-stop).
  • Reasoner     — the LLM that explains the decision in plain language (see [reasoning.py]).
  • Copilot      — the grounded RAG assistant, engaged when the driver asks something.
  • Supervisor   — sequences the crew as risk rises and commits the plan.

The orchestrator is deliberately heuristic and fast (no LLM in this hot path) so the graph can
update every tick; the Reasoner's LLM call runs separately, off the safety path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .forecast import Forecaster
from .persona import PersonaStore
from .policy import GENERIC_FIXED_THRESHOLD

# Static graph topology: nodes grouped into lanes (sensing -> decision -> action) and the
# information-flow edges between them. The dashboard uses `group` for layout.
NODES = [
    {"id": "perception", "label": "Perception", "group": "sensing"},
    {"id": "forecast", "label": "World Model", "group": "sensing"},
    {"id": "context", "label": "Context", "group": "sensing"},
    {"id": "policy", "label": "Safety Policy", "group": "decision"},
    {"id": "critic", "label": "Critic", "group": "decision"},
    {"id": "orchestrator", "label": "Supervisor", "group": "decision"},
    {"id": "wellness", "label": "Wellness", "group": "action"},
    {"id": "reasoning", "label": "Reasoner", "group": "action"},
    {"id": "copilot", "label": "Copilot", "group": "action"},
]

EDGES = [
    ["perception", "forecast"],
    ["perception", "policy"],
    ["forecast", "orchestrator"],
    ["context", "orchestrator"],
    ["policy", "orchestrator"],
    ["orchestrator", "critic"],
    ["critic", "orchestrator"],
    ["orchestrator", "wellness"],
    ["orchestrator", "reasoning"],
    ["orchestrator", "copilot"],
]

_LEVEL_NAMES = {0: "MONITORING", 1: "GENTLE ALERT", 2: "ACTIVE ASSIST", 3: "PREPARE HANDOVER", 4: "SAFE TAKEOVER"}


def _abnormal_signals(state: Dict[str, Any], threshold: float) -> List[str]:
    """Which raw signals are out of range right now — the Critic's evidence for agreement."""
    out = []
    def num(k):
        try:
            return float(state.get(k))
        except (TypeError, ValueError):
            return None
    score = num("score")
    if score is not None and score >= threshold:
        out.append("score")
    ear = num("ear")
    if ear is not None and ear < 0.21:
        out.append("EAR")
    perclos = num("perclos")
    if perclos is not None and perclos > 0.30:
        out.append("PERCLOS")
    bd = num("blinkDuration")
    if bd is not None and bd > 0.40:
        out.append("blink")
    hp = num("headPitch")
    if hp is not None and abs(hp) > 12:
        out.append("head-nod")
    return out


class Orchestrator:
    def __init__(self, personas: PersonaStore, forecaster: Forecaster) -> None:
        self.personas = personas
        self.forecaster = forecaster
        self.cycle = 0

    def _level(self, score: Optional[float], threshold: float, forecast: Dict[str, Any]) -> int:
        """Map the current + predicted state to a 0-4 risk level. The predictive world model
        can PRE-EMPTIVELY raise the level before the score has actually crossed the line."""
        s = score if isinstance(score, (int, float)) else 0.0
        if s >= threshold + 22:
            return 4
        if s >= threshold + 10:
            return 3
        if s >= threshold:
            return 2
        base = 1 if s >= 0.6 * threshold else 0
        # Pre-emption: an imminent forecast lifts an otherwise-calm reading to active assist.
        if forecast.get("risk") == "imminent":
            base = max(base, 2)
        elif forecast.get("risk") == "elevated":
            base = max(base, 1)
        return base

    def _wellness_actions(self, level: int, persona) -> List[Dict[str, str]]:
        """The countermeasure the Wellness agent chooses for THIS driver at this level."""
        if level == 1:
            modality = {
                "audio": "Soft chime alert",
                "visual": "Gentle dashboard nudge",
                "haptic": "Light seat-vibration cue",
            }.get(persona.preferred_modality, "Gentle nudge")
            return [{"type": persona.preferred_modality, "detail": f"{modality} + suggest a short break"}]
        if level == 2:
            return [
                {"type": "climate", "detail": "Cool cabin to 19°C"},
                {"type": "music", "detail": "Switch to an upbeat playlist"},
            ]
        if level == 3:
            return [
                {"type": "hazards", "detail": "Hazard lights on"},
                {"type": "windows", "detail": "Crack windows for airflow"},
                {"type": "navigation", "detail": "Route to the nearest rest stop"},
            ]
        return []

    def run_cycle(
        self,
        driver_id: str,
        state: Dict[str, Any],
        telemetry: Optional[Dict[str, Any]],
        trip: Optional[Dict[str, Any]] = None,
        trigger: str = "state",
        copilot_active: bool = False,
    ) -> Dict[str, Any]:
        self.cycle += 1
        p = self.personas.get(driver_id)
        threshold = p.drowsiness_threshold
        score = state.get("score")
        s = float(score) if isinstance(score, (int, float)) else 0.0

        forecast = self.forecaster.latest(driver_id, threshold)
        level = self._level(score, threshold, forecast)
        telemetry = telemetry or {}
        trip = trip or {}

        # ── each agent reports status + a one-line note ──────────────────
        def node(id, label, group, status, note):
            return {"id": id, "label": label, "group": group, "status": status, "note": note}

        perc_status = "firing" if s >= 0.5 * threshold else "active"
        perception = node("perception", "Perception", "sensing", perc_status,
                          f"score {s:.0f} · EAR {state.get('ear', '—')} · PERCLOS {state.get('perclos', '—')}")

        fc_status = "firing" if forecast["risk"] in ("elevated", "imminent") else "active"
        forecast_node = node("forecast", "World Model", "sensing", fc_status, forecast["horizonText"])

        ctx_bits = []
        if trip.get("hoursDriven") is not None:
            ctx_bits.append(f"{trip['hoursDriven']:.1f}h driven")
        if trip.get("timeOfDay"):
            ctx_bits.append(str(trip["timeOfDay"]))
        if telemetry.get("scenario"):
            ctx_bits.append(str(telemetry["scenario"]))
        if telemetry.get("speedKmh") is not None:
            ctx_bits.append(f"{telemetry['speedKmh']} km/h")
        context = node("context", "Context", "sensing", "active", " · ".join(ctx_bits) or "no trip context")

        generic_fires = s >= GENERIC_FIXED_THRESHOLD
        personal_fires = s >= threshold
        if personal_fires and not generic_fires:
            pol_note = f"{s:.0f} ≥ your {threshold:.0f} (generic {GENERIC_FIXED_THRESHOLD:.0f} would miss)"
        elif generic_fires and not personal_fires:
            pol_note = f"{s:.0f} < your {threshold:.0f} — holding (generic would false-alarm)"
        else:
            pol_note = f"{s:.0f} vs your personal line {threshold:.0f}"
        policy = node("policy", "Safety Policy", "decision", "firing" if level >= 2 else ("active" if level >= 1 else "idle"), pol_note)

        # Critic: independent verification, only engaged when a takeover is on the table (L>=3).
        if level >= 3:
            signals = _abnormal_signals(state, threshold)
            if len(signals) >= 2:
                critic = node("critic", "Critic", "decision", "ok", f"Verified — {len(signals)} signals agree ({', '.join(signals)})")
                verified = True
            else:
                critic = node("critic", "Critic", "decision", "veto", f"Only {len(signals)} signal(s) — vetoing takeover to avoid false alarm")
                verified = False
        else:
            critic = node("critic", "Critic", "decision", "idle", "Standing by (engages before takeover)")
            verified = True

        # A vetoed takeover is capped at "prepare" until signals agree.
        effective_level = level if verified else min(level, 3)

        actions = self._wellness_actions(effective_level, p)
        if effective_level in (0, 4):
            well_status = "idle"
            well_note = "Standing by" if effective_level == 0 else "Standing down — Safety in control"
        else:
            well_status = "firing"
            well_note = "; ".join(a["detail"] for a in actions)
        wellness = node("wellness", "Wellness", "action", well_status, well_note)

        reasoning = node("reasoning", "Reasoner", "action",
                         "firing" if effective_level >= 2 else "idle",
                         "Generating plain-language explanation…" if effective_level >= 2 else "Idle")

        copilot = node("copilot", "Copilot", "action",
                       "firing" if copilot_active else "idle",
                       "Answering driver question…" if copilot_active else "Idle — ask a question")

        # ── Supervisor commits the plan ──────────────────────────────────
        if effective_level == 0:
            plan = "Monitoring against the driver's personal baseline."
        elif effective_level == 1:
            plan = "Early fatigue — gentle nudge, no intervention yet."
        elif effective_level == 2:
            plan = "Fatigue confirmed — running comfort countermeasures and explaining why."
        elif effective_level == 3:
            plan = "Driver unresponsive — preparing handover, Critic verifying before takeover."
        else:
            plan = "Takeover authorized — guiding the car to a safe stop."
        orchestrator = node("orchestrator", "Supervisor", "decision", "firing" if effective_level >= 1 else "active", plan)

        return {
            "cycle": self.cycle,
            "trigger": trigger,
            "level": effective_level,
            "levelName": _LEVEL_NAMES[effective_level],
            "requestedLevel": level,
            "vetoed": not verified,
            "driver": p.display_name,
            "score": round(s, 1),
            "threshold": round(threshold, 1),
            "forecast": forecast,
            "decision": plan,
            "actions": actions,
            "nodes": [perception, forecast_node, context, policy, critic, orchestrator, wellness, reasoning, copilot],
            "edges": EDGES,
        }
