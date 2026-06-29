"""Per-driver persona store.

For the prototype this is an in-memory table. Later it becomes a local SQLite/JSON
store that the camera's face-recognition picks the key for. The important field for the
pitch is `eye_closure_threshold_s` — the PERSONAL safety baseline that makes Aura adaptive
instead of one-size-fits-all.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Persona:
    driver_id: str
    display_name: str
    playlist: str = "Default"
    # How long THIS driver can have eyes closed before it is abnormal for them.
    eye_closure_threshold_s: float = 2.0
    # The fused drowsiness SCORE (0-100) at which THIS driver should be warned. The differentiator:
    # a sharp driver gets a higher bar (fewer false alarms); a tired/elderly driver a lower one.
    drowsiness_threshold: float = 50.0
    # How this driver is best warned (their reaction profile).
    preferred_modality: str = "audio"


class PersonaStore:
    def __init__(self) -> None:
        self._personas = {
            "haresh": Persona(
                driver_id="haresh",
                display_name="Haresh",
                playlist="Focus Drive",
                eye_closure_threshold_s=2.4,
                drowsiness_threshold=55.0,
                preferred_modality="audio",
            ),
            "guest": Persona(
                driver_id="guest",
                display_name="Guest",
                playlist="Top Hits",
                eye_closure_threshold_s=1.6,
                drowsiness_threshold=42.0,
                preferred_modality="visual",
            ),
        }

    def get(self, driver_id: str) -> Persona:
        return self._personas.get(driver_id, self._personas["guest"])
