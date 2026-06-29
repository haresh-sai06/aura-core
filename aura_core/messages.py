"""Message contract shared by every Aura interface.

Every message on the wire is a JSON envelope: {type, timestamp, payload}.
Keep this file as the single source of truth for message types — the Unity
AuraClient and the React dashboard must agree with it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict


class MessageType(str, Enum):
    # Core -> clients
    DRIVER_IDENTIFIED = "driver.identified"   # face recognized -> welcome + persona
    DRIVER_STATE = "driver.state"             # continuous live signal: ear, closure, score
    SAFETY_ALERT = "safety.alert"             # adaptive drowsiness/distraction warning
    EXPLAIN = "explain"                        # "why did Aura warn you?" detail
    # Clients -> Core
    VEHICLE_TELEMETRY = "vehicle.telemetry"   # Unity -> speed/position/scenario


def envelope(msg_type: "MessageType | str", payload: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a payload in the standard {type, timestamp, payload} envelope."""
    type_str = msg_type.value if isinstance(msg_type, MessageType) else str(msg_type)
    return {
        "type": type_str,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
