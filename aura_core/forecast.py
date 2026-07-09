"""Predictive world-model — forecasts WHERE the driver's state is heading, not just where it is.

A fixed-threshold DMS is purely reactive: it fires the instant a score crosses a line. Aura's
forecaster watches the *trajectory* of the fused drowsiness score and projects it forward, so
the system can say "microsleep likely in ~40 s" and act BEFORE the line is crossed. This is the
"world model" the AutoCare panel refers to — here applied to the driver rather than the road.

Method (deliberately simple, robust, and on-device — no training needed for the demo):
  • Keep a short time-stamped history of the fused score per driver.
  • Estimate the current rate of change via an exponentially-weighted least-squares slope over a
    few seconds (smooths the noisy per-frame score without lagging badly).
  • If the score is rising toward the driver's PERSONAL threshold, project the time-to-cross.
Everything is bounded and best-effort; a flat or noisy signal simply reports "stable".
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional, Tuple

# How much history to keep per driver, and the window we actually fit the slope over.
_MAXLEN = 40
_FIT_WINDOW_S = 6.0
# Rising faster than this (points/sec) with a near threshold-crossing = "imminent".
_IMMINENT_S = 20.0
_ELEVATED_S = 60.0


class Forecaster:
    """Per-driver score-trajectory forecaster."""

    def __init__(self) -> None:
        self._hist: Dict[str, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=_MAXLEN))

    def reset(self, driver_id: Optional[str] = None) -> None:
        if driver_id is None:
            self._hist.clear()
        else:
            self._hist.pop(driver_id, None)

    def _slope(self, pts: list[Tuple[float, float]]) -> float:
        """Least-squares slope (points/second) over the recent window."""
        if len(pts) < 3:
            return 0.0
        t0 = pts[0][0]
        xs = [t - t0 for t, _ in pts]
        ys = [s for _, s in pts]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        if denom <= 1e-6:
            return 0.0
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return num / denom

    def update(self, driver_id: str, score: float, threshold: float, now: Optional[float] = None) -> Dict[str, Any]:
        """Record a new score sample and return the current forecast."""
        t = now if now is not None else time.monotonic()
        try:
            s = float(score)
        except (TypeError, ValueError):
            s = 0.0
        h = self._hist[driver_id]
        h.append((t, s))

        recent = [(tt, ss) for (tt, ss) in h if t - tt <= _FIT_WINDOW_S]
        slope = self._slope(recent)

        seconds_to_threshold: Optional[float] = None
        if slope > 0.3 and s < threshold:
            seconds_to_threshold = round((threshold - s) / slope, 1)

        # Classify near-term risk from the projection (and from already being over the line).
        if s >= threshold:
            risk = "imminent"
            horizon = "Over the personal line now — acting."
        elif seconds_to_threshold is not None and seconds_to_threshold <= _IMMINENT_S:
            risk = "imminent"
            horizon = f"Microsleep risk in ~{seconds_to_threshold:.0f}s — pre-empting."
        elif seconds_to_threshold is not None and seconds_to_threshold <= _ELEVATED_S:
            risk = "elevated"
            horizon = f"Fatigue trending up — threshold in ~{seconds_to_threshold:.0f}s."
        elif slope < -0.3:
            risk = "nominal"
            horizon = "Recovering — score trending down."
        else:
            risk = "nominal"
            horizon = "Stable — no near-term risk."

        trend = "rising" if slope > 0.3 else "falling" if slope < -0.3 else "stable"
        return {
            "driver": driver_id,
            "score": round(s, 1),
            "threshold": round(threshold, 1),
            "trend": trend,
            "slopePerSec": round(slope, 2),
            "secondsToThreshold": seconds_to_threshold,
            "risk": risk,
            "horizonText": horizon,
        }

    def latest(self, driver_id: str, threshold: float) -> Dict[str, Any]:
        """Forecast from the last known sample without adding a new one (for /state, MCP)."""
        h = self._hist.get(driver_id)
        if not h:
            return {
                "driver": driver_id, "score": 0.0, "threshold": round(threshold, 1),
                "trend": "stable", "slopePerSec": 0.0, "secondsToThreshold": None,
                "risk": "nominal", "horizonText": "No signal yet.",
            }
        last_score = h[-1][1]
        return self.update(driver_id, last_score, threshold, now=h[-1][0])
