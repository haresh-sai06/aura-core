"""Entry point: start the Aura Core hub on 127.0.0.1:8765."""
import uvicorn

if __name__ == "__main__":
    # ws_ping_interval/timeout disabled: this is a localhost demo and the Unity client
    # shouldn't be dropped for a missed keepalive pong during long idle stretches.
    uvicorn.run(
        "aura_core.server:app",
        host="127.0.0.1",
        port=8765,
        log_level="info",
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )
