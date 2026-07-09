# Aura Safety Policy — how and why Aura acts

## Personalized drowsiness threshold
Aura does not use a single fixed alert line for every driver. Each driver has a personal
drowsiness threshold (a fused score from 0–100) that reflects their own baseline. A generic
driver-monitoring system warns everyone at the same fixed line (score 50), which nags alert
drivers and reacts too late for tired ones. Aura warns against *your* line instead.

Example thresholds: Haresh 55 (experienced night driver, higher bar), Arjun 50 (long-haul
commuter), Priya 44 (prefers an earlier, gentler nudge), Guest 42 (unknown driver, safest
conservative baseline). At a fused score of 48, a generic system fires for everyone; Aura
fires for Priya but correctly holds for Haresh.

## Adaptive learning
Each driver's threshold self-tunes over time. When Aura observes clearly-awake driving well
below the current line, it gently eases the personal threshold toward that calm baseline plus
a safety margin. Learning is bounded between 35 and 75 so it can never drift into "never
alerts" or "always alerts", and it is eased gradually so the line never jumps mid-drive.

## The seven drowsiness signals
Aura fuses seven signals from the in-cabin camera, so no single noisy frame can trigger a
takeover: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR, yawning), PERCLOS (percent of
time eyes are closed), blink rate, blink duration, head pose (pitch/yaw/roll nodding), and
gaze stability. A TinyML model provides an independent second opinion.

## Why fusion and confidence gating
Escalation requires two or more abnormal signals held for several seconds, and higher
intervention levels require higher confidence (55% rising to 85%). This multi-signal
confirmation is what keeps false alarms low while still catching real microsleep early.

## Privacy — everything on the edge
All sensing, scoring, reasoning, and the language copilot run on-device. No camera frames,
no driver identity, and no personal baselines ever leave the vehicle. There is no cloud
dependency; Aura works fully offline.
