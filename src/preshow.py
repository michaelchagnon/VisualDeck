# SPDX-License-Identifier: MPL-2.0
"""Preshow Check (Show > Preshow Check…): what an operator checks before the
house opens, in one list they can read in ten seconds.

checks(snap, show) turns Output Health's snapshot (ControlInterface.
_current_health) and a few facts about the show into rows
(level, title, detail): level is "fail" (fix it before the show), "warn"
(look at it) or "ok". Nothing here changes the show or the output.

No UI here; the window is ControlInterface._open_preshow.
"""
from __future__ import annotations

FAIL, WARN, OK = "fail", "warn", "ok"
LOW_DISK_WARN = 2 * 1024 ** 3       # bytes free where the show, autosaves and logs are written
LOW_DISK_FAIL = 300 * 1024 ** 2
GDI_WARN = 5000                     # Windows stops drawing for a program at 10,000
MAX_NAMES = 3                       # missing files named in the row (the Relink window lists them all)


def _gb(n):
    return f"{n / 1024 ** 3:.1f} GB"


def checks(snap, show):
    """[(level, title, detail)], the problems first.

    snap: Output Health's snapshot, or None if the output has not started.
    show: {'output': the Output choice ('None', 'Display 0', ...),
           'output_size': (w, h) or None, 'render_mode': 'opengl' | 'pygame_gpu' | 'cpu',
           'cues': cues with media, 'effects': effects used in the show (layers and displays),
           'blends': layers with Add/Multiply/Screen, 'dirty': bool, 'path': show file or None,
           'locked': bool, 'test_pattern': bool, 'disk_free': bytes or None,
           'presentation': bool, 'schedule': text or ''}."""
    rows = []
    add = lambda level, title, detail="": rows.append((level, title, detail))  # noqa: E731
    out = show.get("output")
    if not out or out == "None":
        add(FAIL, "No output", "Choose the display the audience sees: Output, above the preview.")
    else:
        size = show.get("output_size")
        add(OK, f"Output: {out}", f"{size[0]} × {size[1]}" if size else "")
    snap = snap or {}
    disp = snap.get("displays") or {}
    if disp.get("changed"):
        add(WARN, "Displays changed since the output started",
            "A display or projector was plugged in or out: Window › Output Health › Rescan Displays.")
    if not snap:
        add(WARN, "The output has not reported yet", "Wait a moment and press Check Again.")
    else:
        if not snap.get("render_thread_alive", True):
            add(FAIL, "The output has stopped", "Save, then restart VisualDeck.")
        rec = sum((snap.get("errors_recent") or {}).values())
        if rec:
            last = (snap.get("last_error") or {}).get("message", "")
            add(FAIL, f"{rec} output error{'s' if rec != 1 else ''} in the last minute",
                (last[:160] + " – " if last else "") + "see Output Health.")
        out_stats = snap.get("output") or {}
        fw = out_stats.get("frames_window") or 0
        late = out_stats.get("late_window", 0)
        if snap.get("active_content") and fw >= 30 and late / fw > 0.02:
            add(WARN, f"{late} late frames in the last 5 s",
                "The cue on the output stutters: see Output Health (decoders, render time).")
    missing = snap.get("missing_media", 0) if snap else 0
    if missing:
        names = list((snap.get("missing_media_names") or [])[:MAX_NAMES]) if snap else []
        more = missing - len(names)
        which = (", ".join(names) + (f" and {more} more" if more > 0 else "") + ". ") if names else ""
        add(FAIL, f"{missing} media file{'s' if missing != 1 else ''} missing",
            which + "File › Relink Missing Media… (Ctrl+Shift+R).")
    else:
        add(OK, "All media files found", f"{show.get('cues', 0)} cues with media")
    for d in snap.get("decoders") or []:
        st = d.get("state")
        if st in ("retrying", "error", "no video"):
            where = "on the output" if d.get("live") else "in the next cue"
            add(FAIL if d.get("live") or d.get("source") == "live" else WARN,
                f"{d.get('file', '?')}: {st}", (d.get("last_error") or "")[:160] + f" ({where})")
    fx = snap.get("effects") or {}
    for name, msg in sorted((fx.get("errors") or {}).items()):
        add(FAIL, f"Effect {name} cannot be drawn", str(msg)[:160])
    mode = show.get("render_mode")
    if show.get("effects") and mode != "opengl":
        add(WARN, f"{show['effects']} shader effect{'s' if show['effects'] != 1 else ''} not drawn",
            "Effects need Performance Mode (Preferences › Rendering Backend).")
    if show.get("blends") and mode != "opengl":
        add(WARN, "Blend modes drawn as Normal", "Add, Multiply and Screen need Performance Mode.")
    for key, label in (("remote", "OSC"), ("web", "Browser remote"), ("midi", "MIDI"),
                       ("dmx", "DMX"), ("ltc", "LTC")):
        r = snap.get(key) or {}
        if not r.get("enabled"):
            continue
        if r.get("listening"):
            add(OK, f"{label} listening", str(r.get("address") or r.get("port") or ""))
        else:
            add(FAIL, f"{label} is on but not listening", str(r.get("error", "?"))[:160])
    a = snap.get("audio") or {}
    if a.get("enabled"):
        if a.get("running", True) and not a.get("error"):
            add(OK, "Sound output open", str(a.get("device") or ""))
        else:
            add(FAIL, "Sound output not open", str(a.get("error", "?"))[:160])
    proc = snap.get("process") or {}
    if (proc.get("gdi") or 0) >= GDI_WARN:
        add(WARN, f"{proc['gdi']} GDI objects", "Windows stops drawing a program at 10,000: save and restart "
                                                "VisualDeck before the show.")
    if show.get("test_pattern"):
        add(WARN, "A test pattern is on the output", "Window › Test Pattern › Off puts the show back.")
    free = show.get("disk_free")
    if free is not None:
        if free < LOW_DISK_FAIL:
            add(FAIL, f"Only {_gb(free)} free on the show's drive", "Saves, autosaves and logs need room.")
        elif free < LOW_DISK_WARN:
            add(WARN, f"{_gb(free)} free on the show's drive", "Saves, autosaves and logs need room.")
    if not show.get("path"):
        add(WARN, "The show has not been saved", "File › Save As… before the house opens.")
    elif show.get("dirty"):
        add(WARN, "Unsaved changes", "File › Save (Ctrl+S) before the house opens.")
    else:
        add(OK, "Show saved", str(show["path"]))
    if not show.get("locked"):
        add(WARN, "Editing is not locked", "Show › Lock Editing (Ctrl+L) for the performance: nothing in "
                                           "the cue grid can be changed by accident; GO, BACK and remotes work.")
    else:
        add(OK, "Editing locked")
    if show.get("presentation"):
        add(OK, "Presentation Mode is on", "GO starts the automatic run through the cues.")
    if show.get("schedule"):
        add(OK, "Schedule on", show["schedule"])
    order = {FAIL: 0, WARN: 1, OK: 2}
    return sorted(rows, key=lambda r: order[r[0]])


def summary(rows):
    """'Ready for the show', or how many problems and warnings."""
    fails = sum(1 for r in rows if r[0] == FAIL)
    warns = sum(1 for r in rows if r[0] == WARN)
    if not fails and not warns:
        return OK, "Ready for the show"
    parts = []
    if fails:
        parts.append(f"{fails} problem{'s' if fails != 1 else ''} to fix")
    if warns:
        parts.append(f"{warns} thing{'s' if warns != 1 else ''} to look at")
    return (FAIL if fails else WARN), ", ".join(parts)


def as_text(rows):
    marks = {FAIL: "PROBLEM", WARN: "CHECK", OK: "ok"}
    return "\n".join(f"[{marks[l]}] {t}" + (f" – {d}" if d else "") for l, t, d in rows)
