"""Aura MCP Server — exposes the edge brain as Model Context Protocol tools.

This makes Aura *speak MCP*: any MCP client (Claude Desktop, Claude Code, an OEM's fleet
assistant) can ask the live car about its driver, read the predictive world-model, see the
multi-agent decision trace, ask the grounded copilot, and even trigger a countermeasure — all
over a standard protocol. It's the "connected vehicles" story: Aura plugs into any AI ecosystem.

Design: this server is a thin MCP facade over the running Aura Core HTTP API on
http://127.0.0.1:8765, so it always reflects the LIVE demo state (whatever the dashboard and
Unity are doing right now). Aura Core must be running first (`python run.py`).

The MCP SDK lives in the aura-core venv, so launch with the VENV python (not a bare `python`):
    VENV = C:/Users/share/Documents/Tata Project/aura-core/.venv/Scripts/python.exe

Run (one-time install already done):  <VENV> mcp_server.py     # stdio transport

Connect from Claude Code:
    claude mcp add aura -- "C:/Users/share/Documents/Tata Project/aura-core/.venv/Scripts/python.exe" "C:/Users/share/Documents/Tata Project/aura-core/mcp_server.py"

Connect from Claude Desktop — add to claude_desktop_config.json:
    "mcpServers": { "aura": {
        "command": "C:/Users/share/Documents/Tata Project/aura-core/.venv/Scripts/python.exe",
        "args": ["C:/Users/share/Documents/Tata Project/aura-core/mcp_server.py"] } }

Aura Core (run.py) must be running first — the tools read its live HTTP API on port 8765.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "The MCP SDK is not installed. Run:  pip install \"mcp[cli]\"  in the aura-core venv."
    ) from e

CORE = "http://127.0.0.1:8765"
mcp = FastMCP("aura")


def _get(path: str) -> Dict[str, Any]:
    with urllib.request.urlopen(f"{CORE}{path}", timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(f"{CORE}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _core_down(e: Exception) -> Dict[str, Any]:
    return {"error": f"Aura Core is not reachable at {CORE} ({e}). Start it with `python run.py`."}


@mcp.tool()
def get_driver_state() -> Dict[str, Any]:
    """Get the current driver, their personal adaptive drowsiness threshold, and the latest live
    signals (fused score, EAR, PERCLOS, head pose). Use this to answer 'who is driving and how
    alert are they right now?'."""
    try:
        st = _get("/state")
        live = st.get("live") or {}
        return {
            "driver": st.get("driver"),
            "score": live.get("score"),
            "ear": live.get("ear"),
            "perclos": live.get("perclos"),
            "facePresent": live.get("facePresent"),
            "riskLevel": (st.get("orchestration") or {}).get("level"),
            "levelName": (st.get("orchestration") or {}).get("levelName"),
        }
    except Exception as e:
        return _core_down(e)


@mcp.tool()
def get_forecast() -> Dict[str, Any]:
    """Get Aura's predictive world-model forecast for the driver: whether fatigue is rising,
    and the projected time until it crosses this driver's personal threshold (time-to-microsleep).
    Use this to answer 'is the driver about to become unsafe?'."""
    try:
        return _get("/state").get("forecast", {})
    except Exception as e:
        return _core_down(e)


@mcp.tool()
def get_agent_trace() -> Dict[str, Any]:
    """Get the latest multi-agent decision cycle: each specialist agent (Perception, World Model,
    Context, Safety Policy, Critic, Wellness, Reasoner, Supervisor), its status and conclusion,
    plus the committed plan. Use this to explain HOW Aura reached its decision."""
    try:
        orch = _get("/state").get("orchestration") or {}
        return {
            "level": orch.get("level"),
            "levelName": orch.get("levelName"),
            "decision": orch.get("decision"),
            "vetoed": orch.get("vetoed"),
            "agents": [{"agent": n["label"], "status": n["status"], "note": n["note"]}
                       for n in orch.get("nodes", [])],
            "actions": orch.get("actions", []),
        }
    except Exception as e:
        return _core_down(e)


@mcp.tool()
def explain_last_decision() -> Dict[str, Any]:
    """Get the on-device Reasoner's plain-language explanation of Aura's most recent safety
    decision — the personalized 'why did Aura act for this driver?'."""
    try:
        st = _get("/state")
        return {"driver": (st.get("driver") or {}).get("name"),
                "explanation": st.get("lastExplanation") or "No decision has been explained yet."}
    except Exception as e:
        return _core_down(e)


@mcp.tool()
def list_drivers() -> Dict[str, Any]:
    """List the enrolled driver personas and their individual adaptive thresholds and preferences."""
    try:
        return _get("/drivers")
    except Exception as e:
        return _core_down(e)


@mcp.tool()
def select_driver(driver_id: str) -> Dict[str, Any]:
    """Switch the active driver in the seat (e.g. 'haresh', 'priya', 'arjun', 'guest'). The whole
    system re-personalizes to that driver's baseline."""
    try:
        return _post("/driver/select", {"id": driver_id})
    except Exception as e:
        return _core_down(e)


@mcp.tool()
def ask_owner_manual(question: str) -> Dict[str, Any]:
    """Ask the Aura Copilot a question grounded in the on-device knowledge base (owner's manual,
    SAE J3016 escalation protocol, safety policy). Returns a cited answer. Use for questions like
    'what does AutoCare Level 3 mean?' or 'what should the driver do during a takeover?'."""
    try:
        res = _post("/copilot/ask", {"query": question})
        return {"answer": res.get("answer"), "sources": res.get("sources", [])}
    except Exception as e:
        return _core_down(e)


@mcp.tool()
def trigger_countermeasure(kind: str) -> Dict[str, Any]:
    """Ask the car to run a wellness countermeasure now. kind is one of:
    'climate' (cool the cabin), 'music' (upbeat playlist), 'windows' (airflow),
    'navigation'/'rest' (route to a rest stop), 'break' (suggest a break). The action is
    broadcast to the vehicle HMI."""
    try:
        return _post("/agents/countermeasure", {"kind": kind})
    except Exception as e:
        return _core_down(e)


if __name__ == "__main__":
    mcp.run()
