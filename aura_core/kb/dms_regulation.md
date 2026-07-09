# Driver Monitoring & ADAS — regulatory context (excerpts)

## Why driver monitoring is mandated
Driver drowsiness and distraction are among the leading contributors to road fatalities.
Regulators increasingly require Driver Monitoring Systems (DMS) and Advanced Driver
Assistance Systems (ADAS) that detect impairment and intervene.

## GSR / Euro NCAP direction
The EU General Safety Regulation (GSR) and Euro NCAP roadmaps push toward driver drowsiness
and attention warning (DDAW) and, progressively, direct camera-based driver monitoring.
Systems are expected to detect fatigue/distraction and warn — and, in more capable systems,
to bring the vehicle to a safe state if the driver is unresponsive (emergency stop / minimal
risk maneuver).

## SAE J3016 automation levels
J3016 defines automation Levels 0–5 and, crucially, the concept of a fallback and a
minimal-risk condition. When a driver fails to respond to a request to intervene, a capable
system should achieve a minimal-risk condition — typically slowing and pulling over safely.
Aura's AutoCare ladder is aligned to this: graded escalation ending in a safe pull-over.

## Functional safety (ISO 26262 direction)
Safety-critical vehicle functions are developed under functional-safety processes. A takeover
feature is designed with fail-safe states, watchdogs, driver-override priority, and an audit
trail of decisions. Aura's roadmap targets this direction for a production path.

## Personalization and false-alarm rate
A known failure mode of fixed-threshold DMS is nuisance alerts, which lead drivers to disable
the system. Personalizing the alert threshold to each driver's baseline reduces false alarms
while catching genuine fatigue earlier — improving both safety and driver acceptance.

## Privacy expectations
Camera-based DMS raises privacy concerns. Best practice, and Aura's design, is on-device
processing with no transmission or retention of biometric frames, explicit driver consent, and
a guarantee that personal data never leaves the vehicle.
