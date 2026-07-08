"""Per-driver persona store — the heart of Aura's personalization.

Each driver carries their OWN safety baseline (how long *they* can have eyes closed,
and the fused drowsiness SCORE at which *they* should be warned) plus their in-car
preferences (playlist, how they like to be alerted). That per-driver threshold is the
differentiator: a generic DMS warns everyone at one fixed line and false-alarms the
calm drivers; Aura warns against you.

Two upgrades over the first prototype:
  • **Persistence** — tunable state is saved to `personas.json` so a driver's learned
    baseline survives restarts (best-effort; the demo still runs if the file is absent).
  • **Adaptive learning** — `record_calm()` watches each driver's normal "awake" scores
    and gently tunes their alert threshold, so Aura fits the driver over time instead of
    staying on a hand-picked number. See [policy.py] for how the threshold is applied.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger("aura-core")

_STORE_PATH = os.path.join(os.path.dirname(__file__), "personas.json")

# Bounds so adaptive learning can never drift into "never alerts" or "always alerts".
_MIN_THRESHOLD = 35.0
_MAX_THRESHOLD = 75.0
# How aggressively the learned baseline tracks recent calm behaviour (EWMA weight).
_LEARN_ALPHA = 0.05
# Head-room kept above a driver's normal awake score before Aura warns.
_LEARN_MARGIN = 32.0


@dataclass
class Persona:
    driver_id: str
    display_name: str
    playlist: str = "Default"
    # How long THIS driver can have eyes closed before it is abnormal for them.
    eye_closure_threshold_s: float = 2.0
    # The fused drowsiness SCORE (0-100) at which THIS driver should be warned. Adapts over time.
    drowsiness_threshold: float = 50.0
    # The hand-picked seed the driver started at (kept so the HUD can show "tuned from X").
    base_threshold: float = 50.0
    # How this driver is best warned (their reaction profile).
    preferred_modality: str = "audio"
    # Presentation + story fields (used by the dashboard driver selector).
    accent: str = "#3b82f6"
    note: str = ""
    # Runtime: EWMA of recent clearly-awake scores that drives adaptive learning.
    calm_ewma: float = field(default=0.0)
    samples: int = 0

    def learn(self, score: float) -> bool:
        """Fold a clearly-awake score into this driver's baseline. Returns True if the
        alert threshold moved (so the caller can persist / re-broadcast)."""
        self.samples += 1
        self.calm_ewma = score if self.samples == 1 else (
            (1 - _LEARN_ALPHA) * self.calm_ewma + _LEARN_ALPHA * score
        )
        target = min(_MAX_THRESHOLD, max(_MIN_THRESHOLD, self.calm_ewma + _LEARN_MARGIN))
        if abs(target - self.drowsiness_threshold) < 0.4:
            return False
        # Ease toward the target so the threshold never jumps mid-demo.
        self.drowsiness_threshold = round(self.drowsiness_threshold + 0.25 * (target - self.drowsiness_threshold), 1)
        return True


def _seed_personas() -> Dict[str, Persona]:
    return {
        "haresh": Persona(
            driver_id="haresh", display_name="Haresh", playlist="Focus Drive",
            eye_closure_threshold_s=2.4, drowsiness_threshold=55.0, base_threshold=55.0,
            preferred_modality="audio", accent="#3b82f6",
            note="Experienced night driver — a higher bar avoids nagging false alarms.",
        ),
        "priya": Persona(
            driver_id="priya", display_name="Priya", playlist="Calm Commute",
            eye_closure_threshold_s=1.7, drowsiness_threshold=44.0, base_threshold=44.0,
            preferred_modality="visual", accent="#ec4899",
            note="Prefers an earlier, gentler visual nudge — a lower, more cautious threshold.",
        ),
        "arjun": Persona(
            driver_id="arjun", display_name="Arjun", playlist="Highway Energy",
            eye_closure_threshold_s=2.1, drowsiness_threshold=50.0, base_threshold=50.0,
            preferred_modality="haptic", accent="#22c55e",
            note="Long-haul commuter — haptic seat alerts keep hands on the wheel.",
        ),
        "guest": Persona(
            driver_id="guest", display_name="Guest", playlist="Top Hits",
            eye_closure_threshold_s=1.6, drowsiness_threshold=42.0, base_threshold=42.0,
            preferred_modality="visual", accent="#a1a1aa",
            note="Unknown driver — the safest, most conservative baseline is used.",
        ),
    }


class PersonaStore:
    """In-memory personas with best-effort JSON persistence of the tunable fields."""

    # Fields safe to persist / restore (skip pure runtime derived values? keep calm_ewma so
    # learning resumes smoothly across restarts).
    _PERSIST = (
        "eye_closure_threshold_s", "drowsiness_threshold", "base_threshold",
        "preferred_modality", "playlist", "accent", "note", "calm_ewma", "samples",
    )

    def __init__(self, path: str = _STORE_PATH) -> None:
        self._path = path
        self._personas = _seed_personas()
        self._load()

    # ── access ────────────────────────────────────────────────────────
    def get(self, driver_id: str) -> Persona:
        return self._personas.get(driver_id, self._personas["guest"])

    def all(self) -> List[Persona]:
        return list(self._personas.values())

    def ids(self) -> List[str]:
        return list(self._personas.keys())

    def has(self, driver_id: str) -> bool:
        return driver_id in self._personas

    # ── learning + persistence ────────────────────────────────────────
    def record_calm(self, driver_id: str, score: float) -> bool:
        """Feed a clearly-awake score to the driver's adaptive learner. Returns True if the
        threshold moved. Persistence is handled by the caller on lifecycle events (driver
        switch / shutdown) so we never thrash the disk at the browser's tick rate."""
        p = self._personas.get(driver_id)
        if p is None:
            return False
        return p.learn(score)

    def _load(self) -> None:
        try:
            if not os.path.exists(self._path):
                return
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for did, saved in data.items():
                if did in self._personas and isinstance(saved, dict):
                    p = self._personas[did]
                    for k in self._PERSIST:
                        if k in saved:
                            setattr(p, k, saved[k])
            log.info("personas: loaded tuned baselines from %s", os.path.basename(self._path))
        except Exception as e:  # never let a bad file stop the demo
            log.warning("personas: could not load %s (%s) — using seed defaults", self._path, e)

    def save(self) -> None:
        try:
            out = {
                did: {k: v for k, v in asdict(p).items() if k in self._PERSIST}
                for did, p in self._personas.items()
            }
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            log.warning("personas: could not save %s (%s)", self._path, e)

    def to_public(self, driver_id: Optional[str] = None) -> dict:
        """A JSON-friendly view of a persona for the dashboard driver selector."""
        p = self.get(driver_id) if driver_id else None
        def view(x: Persona) -> dict:
            return {
                "id": x.driver_id, "name": x.display_name, "playlist": x.playlist,
                "threshold": round(x.drowsiness_threshold, 1), "baseThreshold": round(x.base_threshold, 1),
                "eyeClosureBaseline": x.eye_closure_threshold_s, "modality": x.preferred_modality,
                "accent": x.accent, "note": x.note, "samples": x.samples,
            }
        if p is not None:
            return view(p)
        return {"drivers": [view(x) for x in self.all()]}
