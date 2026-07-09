"""Entry point: start the Aura Core hub on 0.0.0.0:8765 (LAN-reachable)."""
import uvicorn

if __name__ == "__main__":
    # Bind 0.0.0.0 so the second laptop (Safety Monitor / System B) can reach Core over Wi-Fi.
    # Head-unit laptop still uses http://127.0.0.1:8765; from System B point the dashboard at
    # this machine's LAN IP via ?core=<ip> (see aura-dashboard/src/config.ts).
    # ws_ping_interval/timeout disabled so the Unity client isn't dropped on idle stretches.
    uvicorn.run(
        "aura_core.server:app",
        host="0.0.0.0",
        port=8765,
        log_level="info",
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )
