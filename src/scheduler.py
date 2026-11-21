# SPDX-License-Identifier: MPL-2.0
"""Show schedule: run cues at set times of day (installations, opening hours).

Saved with the show as {"enabled": bool, "events": [...]}; an event is
{"time": "HH:MM", "days": [0..6] (Monday = 0), "action": "cue" | "go" |
"blackout", "cue": "<cue number, position or name>" (for "cue")}.

Rules, chosen so a schedule never surprises the operator:
- Nothing runs unless the schedule is switched on (Show > Schedule...).
- An event runs in its minute only, once per day; events missed while
  VisualDeck was closed, or before the schedule was switched on, are not
  caught up.
- Times are this computer's local time.
No UI here; ControlInterface checks once a second.
"""
from __future__ import annotations

import datetime as _dt

ACTIONS = ("cue", "go", "blackout")
ACTION_LABELS = {"cue": "Go to cue", "go": "GO", "blackout": "Blackout"}
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
DAY_PRESETS = {"Every day": list(range(7)), "Weekdays": [0, 1, 2, 3, 4], "Weekends": [5, 6]}
DAY_PRESETS.update({name: [i] for i, name in enumerate(DAY_NAMES)})


def parse_time(text):
    """(hour, minute) from "HH:MM" (or "H:MM"), else None."""
    try:
        h, m = str(text).strip().split(":")
        h, m = int(h), int(m)
    except (ValueError, AttributeError):
        return None
    return (h, m) if 0 <= h <= 23 and 0 <= m <= 59 else None


def clean_event(raw):
    if not isinstance(raw, dict):
        return None
    t = parse_time(raw.get("time"))
    action = raw.get("action")
    if t is None or action not in ACTIONS:
        return None
    days = raw.get("days")
    if not isinstance(days, list):
        days = list(range(7))
    days = sorted({d for d in days if isinstance(d, int) and not isinstance(d, bool) and 0 <= d <= 6})
    if not days:
        return None
    ev = {"time": f"{t[0]:02d}:{t[1]:02d}", "days": days, "action": action}
    if action == "cue":
        cue = str(raw.get("cue", "")).strip()[:40]
        if not cue:
            return None
        ev["cue"] = cue
    return ev


def clean(raw):
    """{"enabled": bool, "events": [...]} with malformed events dropped, sorted by time."""
    if not isinstance(raw, dict):
        return {"enabled": False, "events": []}
    events = [e for e in (clean_event(x) for x in raw.get("events") or []) if e is not None]
    events.sort(key=lambda e: (e["time"], e["days"]))
    return {"enabled": bool(raw.get("enabled", False)), "events": events}


def days_text(days):
    for name, preset in DAY_PRESETS.items():
        if days == preset:
            return name
    return ", ".join(DAY_NAMES[d] for d in days)


def event_text(ev):
    what = f"cue {ev['cue']}" if ev["action"] == "cue" else ACTION_LABELS[ev["action"]]
    return f"{ev['time']}  {days_text(ev['days'])}  {what}"


def due(events, now, fired):
    """Events to run at `now` (a datetime): those whose minute it is, on one of
    their days, not yet run today. `fired` is a set of (date, index) and is
    updated."""
    out = []
    for i, ev in enumerate(events):
        h, m = parse_time(ev["time"])
        if now.weekday() in ev["days"] and (now.hour, now.minute) == (h, m) and (now.date(), i) not in fired:
            fired.add((now.date(), i))
            out.append(ev)
    return out


def next_event(events, now):
    """(datetime, event) of the next event after `now`, or None."""
    best = None
    for ev in events:
        h, m = parse_time(ev["time"])
        for add in range(8):
            day = now.date() + _dt.timedelta(days=add)
            when = _dt.datetime.combine(day, _dt.time(h, m), tzinfo=now.tzinfo)
            if day.weekday() in ev["days"] and when > now:
                if best is None or when < best[0]:
                    best = (when, ev)
                break
    return best
