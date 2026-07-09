# Emergency eCall — credentials & setup

Aura's eCall alerts your emergency contacts (with GPS + a map link + what the cabin camera sees)
when the driver is unresponsive during a Level-4 safe-stop. **The core AI needs no keys** — only
the outbound alert channel does. Configure it all from the dashboard's **Emergency** page; secrets
are written to `emergency.json` (gitignored) and never committed.

---

## What you need

| Channel | Credential | Cost |
| --- | --- | --- |
| **WhatsApp** (CallMeBot) | one **API key** per receiving phone | Free, no signup |
| **SMS / Voice call** (Twilio) | **Account SID** + **Auth Token** + **From number** | Free trial (~$15 credit) |

You only need the channel(s) you actually want to demo. WhatsApp alone (free, ~1 min) is enough.

---

## 1. WhatsApp via CallMeBot (free)

On the phone that should **receive** the alert:

1. Add a contact for **+34 644 51 95 23** (CallMeBot).
2. Send it a WhatsApp message: **`I allow callmebot to send me messages`**
3. It replies: *"API Activated for your phone number. Your APIKEY is `1234567`"*
4. In the dashboard → **Emergency → add contact**:
   - Channel: **WhatsApp**
   - Phone: international format, **no `+`** (e.g. `919876543210`)
   - CallMeBot key: the `1234567` from step 3
5. Click **Test** on the contact to confirm the message arrives.

> CallMeBot's free tier sends to the number that authorized it, so each contact authorizes their
> own phone. For a solo demo, authorize your own phone and add yourself as the contact.

---

## 2. Twilio (SMS + Voice call)

1. Sign up at <https://www.twilio.com/try-twilio> (free trial).
2. From the Console home, copy your **Account SID** (`AC…`) and **Auth Token**.
3. Get your trial **phone number** (Console → Phone Numbers) — this is the **From** number.
4. **Trial limitation:** you can only text/call **verified** numbers. Console → *Verified Caller IDs*
   → add + verify each recipient (they get a code). (Upgrading with any funds removes this limit.)
5. In the dashboard → **Emergency → Twilio settings**: paste **SID**, **Auth Token**, **From**, Save.
6. Add contacts with channel **SMS** or **Voice call**, then click **Test**.

Approx. cost on paid accounts: SMS ~$0.008, voice ~$0.014/min (US); trial credit covers a demo easily.

---

## Notes

- **No credentials?** The whole UX still demos — countdown, cancel, payload preview, map link — and
  each contact simply reports "not configured" instead of sending. Great as a zero-setup fallback.
- **Location:** taken from the browser's Geolocation API at dispatch (allow the permission), with a
  configurable fallback coordinate for indoor demos.
- **Privacy:** only the single outbound alert leaves the device; all sensing/AI stays on-device.
