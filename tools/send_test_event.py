"""Tiny CLI to fire test events at a running Aura Core (no extra deps).

Usage:
    python tools/send_test_event.py drowsy      # adaptive drowsiness alert
    python tools/send_test_event.py identify     # "Welcome, <driver>"
    python tools/send_test_event.py drowsy 1.0   # below baseline -> no alert
"""
import sys
import urllib.request

BASE = "http://127.0.0.1:8765"


def post(path: str) -> None:
    req = urllib.request.Request(BASE + path, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        print(r.read().decode())


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "drowsy"
    if cmd == "drowsy":
        secs = sys.argv[2] if len(sys.argv) > 2 else None
        post("/emit/drowsy" + (f"?eye_closure_s={secs}" if secs else ""))
    elif cmd == "identify":
        post("/emit/identify")
    elif cmd in ("resume", "clear", "wake"):
        post("/emit/resume")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
