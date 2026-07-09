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
    REASONING = "reasoning"                    # streamed natural-language "why" from the LLM
    COPILOT_RESPONSE = "copilot.response"      # grounded RAG answer to a driver question
    FORECAST = "forecast"                      # predictive world-model: time-to-microsleep, trend
    ORCHESTRATION = "orchestration"            # multi-agent decision cycle (the agent graph trace)
    COUNTERMEASURE = "countermeasure"          # a proactive action the Wellness agent chose
    VISION_SCENE = "vision.scene"              # vision-LLM description of the cabin/road frame
    ECALL = "ecall"                            # emergency escalation status (armed/dispatched/…)
    # Clients -> Core
    VEHICLE_TELEMETRY = "vehicle.telemetry"   # Unity -> speed/position/scenario
    COPILOT_QUERY = "copilot.query"           # driver asks the in-car assistant a question


def envelope(msg_type: "MessageType | str", payload: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a payload in the standard {type, timestamp, payload} envelope."""
    type_str = msg_type.value if isinstance(msg_type, MessageType) else str(msg_type)
    return {
        "type": type_str,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
