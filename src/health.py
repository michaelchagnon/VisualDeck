# SPDX-License-Identifier: MPL-2.0
"""Runtime health metrics for the output.

The render thread records one sample per rendered frame (O(1)) and, every
half second, publishes an immutable snapshot dict that the UI and the log
read without locks. Nothing here touches Tk or OpenGL.

Definitions (shown in the Health window, keep in sync with docs):
- interval: time between two consecutive rendered frames.
- late frame: an interval longer than 1.5 x the render thread's target frame
  period while video or a fade is on screen, i.e. at least one output frame
  was missed. Gaps where the output was static are not counted.
- render cost: time the render thread spent inside the engine's render call
  for that frame (draw + upload + buffer swap; with VSync on it includes the
  wait for the display).
- dropped (decoder): frames a decoder produced that were replaced by a newer
  frame before the output used them. Some drops are normal when a clip's
  frame rate exceeds the output's; sustained drops at matching rates mean the
  output is not keeping up.
- decoder behind: a playing decoder producing frames slower than 90% of the
  clip's frame rate (the clip plays in slow motion / stutters).
"""
from __future__ import annotations

import os
import time
from collections import deque

LATE_FACTOR = 1.5
WINDOW_S = 5.0          # rolling window for rates and percentiles
ERROR_WINDOW_S = 60.0   # "recent" errors


def _pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = min(len(sorted_vals) - 1, max(0, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


class FrameStats:
    """Rolling per-frame timing, fed by the render thread."""

    def __init__(self, window_s=WINDOW_S, maxlen=2048):
        self.window_s = window_s
        self._samples = deque(maxlen=maxlen)  # (t_end, interval or None, render_s, late)
        self._last_t = None
        self._last_active = False
        self.frames_total = 0
        self.late_total = 0

    def record(self, t_end, render_s, active, target_fps):
        interval = None
        late = False
        if self._last_t is not None:
            interval = t_end - self._last_t
            if active and self._last_active and target_fps:
                late = interval > LATE_FACTOR / float(target_fps)
        self._last_t, self._last_active = t_end, active
        self.frames_total += 1
        self.late_total += late
        self._samples.append((t_end, interval, render_s, late))

    def snapshot(self, now):
        cutoff = now - self.window_s
        win = [s for s in self._samples if s[0] >= cutoff]
        intervals = sorted(s[1] for s in win if s[1] is not None)
        costs = sorted(s[2] for s in win)
        span = (win[-1][0] - win[0][0]) if len(win) > 1 else 0.0
        ms = lambda v: None if v is None else round(v * 1000.0, 2)  # noqa: E731
        return {
            "renders_per_s": round((len(win) - 1) / span, 1) if span > 0 else 0.0,
            "interval_ms": {"p50": ms(_pct(intervals, 50)), "p99": ms(_pct(intervals, 99)),
                            "max": ms(intervals[-1] if intervals else None)},
            "render_ms": {"p50": ms(_pct(costs, 50)), "p99": ms(_pct(costs, 99)),
                          "max": ms(costs[-1] if costs else None)},
            "late_window": sum(1 for s in win if s[3]),
            "frames_window": len(win),
            "late_total": self.late_total,
            "frames_total": self.frames_total,
            "last_frame_age_s": round(now - self._last_t, 3) if self._last_t is not None else None,
        }


class LoopStats:
    """Where the render loop's time goes, per iteration (render thread only):
    event pump, commands, render, preview and the pacing wait, plus ticks with
    moving content that did not render (a skipped tick is a late frame)."""

    PHASES = ("pump", "commands", "render", "preview", "wait")

    def __init__(self, window_s=WINDOW_S, maxlen=2048):
        self.window_s = window_s
        self._samples = deque(maxlen=maxlen)   # (t, {phase: s}, active, rendered)

    def record(self, t, phases, active, rendered):
        self._samples.append((t, phases, active, rendered))

    def snapshot(self, now):
        win = [x for x in self._samples if x[0] >= now - self.window_s]
        out = {}
        for ph in self.PHASES:
            vals = sorted(x[1].get(ph, 0.0) for x in win)
            out[ph] = {"p50": round((_pct(vals, 50) or 0) * 1000, 2), "p99": round((_pct(vals, 99) or 0) * 1000, 2),
                       "max": round((vals[-1] if vals else 0) * 1000, 2)}
        active = [x for x in win if x[2]]
        return {"phase_ms": out, "ticks_window": len(win), "active_ticks": len(active),
                "skipped_active_ticks": sum(1 for x in active if not x[3])}


class GcStats:
    """Pauses of Python's garbage collector (the cyclic one).

    While it runs, every Python thread waits, the render thread too, so a
    collection longer than a frame period is a late frame. A full collection
    (generation 2) scans every object, so it gets longer as objects pile up.
    Fed by gc.callbacks in whichever thread collects; it only assigns into a
    fixed list and adds to counters, so reading it from another thread (or
    a collection starting while a snapshot is made) cannot disturb it."""

    RING = 128

    def __init__(self, window_s=WINDOW_S, clock=None):
        self.window_s = window_s
        self.clock = clock or now
        self._start = None
        self._ring = [None] * self.RING     # (t_end, generation, seconds)
        self._i = 0
        self.counts = [0, 0, 0]
        self.total_s = [0.0, 0.0, 0.0]
        self.max_s = [0.0, 0.0, 0.0]
        self.last_full_s = None

    def callback(self, phase, info):
        if phase == "start":
            self._start = self.clock()
            return
        t0, self._start = self._start, None
        if t0 is None:
            return
        t = self.clock()
        g = min(2, max(0, int(info.get("generation", 0))))
        dt = t - t0
        self.counts[g] += 1
        self.total_s[g] += dt
        self.max_s[g] = max(self.max_s[g], dt)
        if g == 2:
            self.last_full_s = dt
        self._ring[self._i % self.RING] = (t, g, dt)
        self._i += 1

    def install(self):
        import gc
        if self.callback not in gc.callbacks:
            gc.callbacks.append(self.callback)
        return self

    def remove(self):
        import gc
        try:
            gc.callbacks.remove(self.callback)
        except ValueError:
            pass

    def snapshot(self, t_now):
        import gc
        import sys
        win = [e for e in list(self._ring) if e is not None and e[0] >= t_now - self.window_s]
        ms = lambda v: None if v is None else round(v * 1000.0, 2)  # noqa: E731
        return {
            "pause_max_ms_window": ms(max((e[2] for e in win), default=0.0)),
            "collections_window": len(win),
            "full_window": sum(1 for e in win if e[1] == 2),
            "full_total": self.counts[2],
            "full_pause_total_ms": ms(self.total_s[2]),
            "full_pause_max_ms": ms(self.max_s[2]),
            "full_pause_last_ms": ms(self.last_full_s),
            "collections_total": sum(self.counts),
            "blocks": sys.getallocatedblocks(),      # Python's own allocations: grows if objects pile up
            "frozen": gc.get_freeze_count(),
        }


class ErrorLog:
    """Timestamps of render-thread errors by phase."""

    def __init__(self, maxlen=500):
        self._events = deque(maxlen=maxlen)
        self.last = None  # (phase, message)

    def add(self, t, phase, exc):
        self._events.append((t, phase))
        self.last = (phase, f"{type(exc).__name__}: {exc}"[:300])

    def recent(self, now, window_s=ERROR_WINDOW_S):
        out = {}
        for t, phase in self._events:
            if t >= now - window_s:
                out[phase] = out.get(phase, 0) + 1
        return out


def player_status(key, info):
    """Status of one decoder from the engine's active_players entry."""
    p = info.get("player")
    q = info.get("queue")
    path = info.get("path") or (key.split("|", 1)[-1] if "|" in key else key)
    instance = key.split("|", 1)[0] if key.startswith("@") else ""
    st = {"key": key, "instance": instance, "file": os.path.basename(path), "path": path,
          "type": info.get("type")}
    if p is None:
        return st
    state = getattr(p, "state", None)
    if getattr(p, "held", False) and state in (None, "playing"):
        state = "held"
    if not getattr(p, "is_running", True) and state not in ("ended", "no video"):
        state = "stopped"
    pacer = getattr(p, "_pacer", None)
    st.update({
        "state": state or "starting",
        "loop": bool(getattr(p, "loop", False)),
        "frames_decoded": getattr(p, "frames_decoded", None),
        "source_fps": getattr(p, "source_fps", None),
        "speed": getattr(p, "speed", 1.0),
        "direction": getattr(p, "direction", "forward"),
        "decode_fps": round(pacer.actual_fps, 1) if pacer is not None and getattr(pacer, "actual_fps", 0) else None,
        "hw_requested": getattr(p, "hw_accel", None),
        "hw_engaged": getattr(p, "hw_engaged", None),
        "last_error": getattr(p, "last_error", None),
        "frames_shown": getattr(q, "taken", None),
        "frames_dropped": getattr(q, "dropped", None),
        "media": getattr(p, "media_info", None),          # codec, size, pixel format, decoder threads
        "scaled_to": getattr(p, "scaled_to", None),       # frames handed over smaller (drawn at that size)
    })
    if info.get("type") == "live":
        import live_source
        last = getattr(p, "last_frame_at", None)
        st.update({"file": live_source.label(path), "source": "live", "reconnects": getattr(p, "reconnects", 0),
                   "frame_age_s": round(time.monotonic() - last, 1) if last is not None else None,
                   "size": getattr(p, "size", None)})
    return st


FROZEN_WHY = {
    "no decoder": "no decoder is playing it; press GO or Update Output to start it again",
    "held": "its decoder was waiting on the first frame; released",
    "not taken": "the output is not taking its frames; press GO or Update Output to start it again",
}


def decoder_behind(st):
    return (st.get("state") == "playing" and st.get("source_fps") and st.get("decode_fps") is not None
            and (st.get("frames_decoded") or 0) > 30
            and st["decode_fps"] < 0.9 * st["source_fps"] * (st.get("speed") or 1.0))


def names_text(names, most=3):
    """": a.mov, b.png and 2 more" for file names, or "" without any."""
    names = [str(n) for n in (names or [])]
    if not names:
        return ""
    more = len(names) - most
    return ": " + ", ".join(names[:most]) + (f" and {more} more" if more > 0 else "")


def assess(snap):
    """('ok'|'warn'|'error', [reasons]) for the status indicator."""
    errors, warns = [], []
    if not snap.get("render_thread_alive", True):
        errors.append("render thread stopped")
    elif (snap.get("stale_s") or 0) > 3.0:
        errors.append(f"render thread not responding for {snap['stale_s']:.0f} s")
    rec = sum((snap.get("errors_recent") or {}).values())
    if rec:
        errors.append(f"{rec} render error{'s' if rec != 1 else ''} in the last minute")
    out = snap.get("output") or {}
    if snap.get("active_content") and (out.get("last_frame_age_s") or 0) > 1.0:
        errors.append("output not updating")
    bad_states = ("retrying", "error", "no video")
    for d in snap.get("decoders") or []:
        if d.get("state") in bad_states:
            # On screen: an error. Preloaded for the next cue: a warning (it will fail on GO).
            (errors if d.get("live") else warns).append(
                f"{d['file']}: {d['state']}" + ("" if d.get("live") else " (next cue)"))
    fw = out.get("frames_window") or 0
    if fw >= 30 and out.get("late_window", 0) / fw > 0.02:
        warns.append(f"{out['late_window']} late frames in the last {int(WINDOW_S)} s")
    tgt = snap.get("target_fps")
    if tgt and tgt < 60 and snap.get("active_content"):   # stills are drawn once: no target applies
        warns.append(f"render target lowered to {int(tgt)} fps under load")
    for d in snap.get("decoders") or []:
        if decoder_behind(d):
            warns.append(f"{d['file']}: decoding {d['decode_fps']:.0f} of "
                         f"{d['source_fps'] * (d.get('speed') or 1.0):.0f} fps")
    for f in snap.get("frozen") or []:
        warns.append(f"{f['file']}: picture not moving ({FROZEN_WHY.get(f['why'], f['why'])})")
    pre = snap.get("preroll") or {}
    if pre.get("timeouts_recent"):
        warns.append(f"{pre['timeouts_recent']} cut(s) waited the full preroll time")
    rem = snap.get("remote") or {}
    if rem.get("enabled") and not rem.get("listening"):
        warns.append(f"OSC remote control not listening ({rem.get('error', '?')})")
    web = snap.get("web") or {}
    if web.get("enabled") and not web.get("listening"):
        warns.append(f"browser remote not listening ({web.get('error', '?')})")
    midi = snap.get("midi") or {}
    if midi.get("enabled") and not midi.get("listening"):
        warns.append(f"MIDI remote control not listening ({midi.get('error', '?')})")
    lt = snap.get("ltc") or {}
    if lt.get("enabled") and not lt.get("listening"):
        warns.append(f"LTC input not listening ({lt.get('error', '?')})")
    dmx = snap.get("dmx") or {}
    if dmx.get("enabled") and not dmx.get("listening"):
        warns.append(f"DMX remote control not listening ({dmx.get('error', '?')})")
    disp = (snap.get("displays") or {}).get("changed")
    if disp:
        warns.append(f"displays changed: {len(disp['engine'])} when the output started, {len(disp['now'])} now "
                     f"(Output Health \u203a Rescan Displays)")
    fx = snap.get("effects") or {}
    for name, msg in sorted((fx.get("errors") or {}).items()):
        warns.append(f"effect {name} not drawn: {msg}"[:200])
    if fx.get("count") and fx.get("drawn") is False:
        warns.append("effects on the output need Performance Mode")
    if snap.get("missing_media"):
        warns.append(f"{snap['missing_media']} missing media file(s)" + names_text(snap.get("missing_media_names")))
    if errors:
        return "error", errors + warns
    return ("warn" if warns else "ok"), warns


def summary_line(snap):
    """One line for the periodic log entry."""
    out = snap.get("output") or {}
    decs = snap.get("decoders") or []
    states = {}
    for d in decs:
        states[d.get("state")] = states.get(d.get("state"), 0) + 1
    drops = sum(d.get("frames_dropped") or 0 for d in decs)
    level, reasons = assess(snap)
    iv = out.get("interval_ms") or {}
    rc = out.get("render_ms") or {}
    return (f"Health {level}: {out.get('renders_per_s', 0)} renders/s (target {snap.get('target_fps')}), "
            f"late {out.get('late_window', 0)}/{out.get('frames_window', 0)} in {int(WINDOW_S)} s "
            f"({out.get('late_total', 0)} total), interval p99 {iv.get('p99')} ms, render p99 {rc.get('p99')} ms, "
            f"errors {sum((snap.get('errors_recent') or {}).values())} recent/{sum((snap.get('errors_total') or {}).values())} total, "
            f"decoders {len(decs)} {dict(sorted((k or '?', v) for k, v in states.items()))}, dropped {drops}"
            + (f" | {'; '.join(reasons)}" if reasons else ""))


def now():
    return time.perf_counter()
