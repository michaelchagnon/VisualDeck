# SPDX-License-Identifier: MPL-2.0
"""Soak test: run a show for a long time and record whether anything drifts.

Help > Soak Test... plays the open show's cues on the output, GO every few
seconds (each cue with its own transition), for as long as asked. Every
SAMPLE_S it records memory, threads, decoders, GPU objects, render rate,
late frames, errors and sound underruns to a CSV file (written as it goes,
so a crash still leaves the data) and at the end a summary with plain-text
findings: memory that keeps growing, threads or decoders that pile up,
errors, frame rate drops. Nothing in the show is changed or saved.

The files go to ~/.visualdeck/soak/. No UI here (the window is in
control_interface.py).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time

from health import WINDOW_S

SAMPLE_S = 10.0
MIN_GROWTH_S = 600.0        # memory growth is judged from 10 minutes after warm-up on (earlier it is warm-up)
BAR_MB_PER_H = 10.0         # the acceptance bar for a long show (checklist I1)
LONG_RUN_S = 3600.0         # a note against that bar from runs of an hour on
FIELDS = ("t_s", "rss_mb", "threads", "decoders", "gl_textures", "renders_per_s", "late_window",
          "frames_window", "errors_total", "audio_underruns", "gos", "active_cue", "render_thread_alive",
          "active_content", "py_blocks", "gc_full_total", "gc_full_pause_total_ms", "gc_pause_max_ms",
          "gdi_objects")
GDI_LIMIT = 10_000          # Windows' default limit of GDI objects per process
FULL_WINDOW = 0.8           # a rate counts when its frames span this share of Output Health's window
_gdi = None
_mem = None


def _windows_memory_api():
    """GetProcessMemoryInfo with its structure, made once.

    Made on every call before (twice a second, for the status light), the
    structure class was never freed: ctypes keeps every pointer type it makes
    (POINTER(PMC) in argtypes), and with it the class. Measured on the
    owner's PC over an hour: Python's allocations grew by about 408 thousand
    blocks, memory by 90 MB, and each full garbage collection took longer
    (9 to 64 ms). The same code run here (not on Windows) kept 47 blocks,
    7.4 KB, per call: 338 thousand blocks and 51 MB an hour at two calls a
    second."""
    global _mem
    if _mem is None:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")   # own instances: touch nobody else
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        _mem = (PMC, psapi.GetProcessMemoryInfo, k32.GetCurrentProcess)
    return _mem


def rss_mb():
    """This process's resident memory in MB (None if it cannot be read)."""
    try:
        if sys.platform == "win32":
            import ctypes
            pmc_t, info, me = _windows_memory_api()
            pmc = pmc_t()
            pmc.cb = ctypes.sizeof(pmc_t)
            if info(me(), ctypes.byref(pmc), pmc.cb):
                return pmc.WorkingSetSize / 2 ** 20
            return None
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except Exception:  # noqa: BLE001
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return peak / 2 ** 20 if sys.platform == "darwin" else peak / 1024
        except Exception:  # noqa: BLE001
            return None


def gdi_objects():
    """GDI objects this process holds (Windows; None elsewhere). Drawing fails
    once a process reaches GDI_LIMIT, so a count that keeps rising is a leak
    that ends a long show (see displays.py for the one found)."""
    global _gdi
    if sys.platform != "win32":
        return None
    try:
        if _gdi is None:
            import ctypes
            from ctypes import wintypes
            user32, k32 = ctypes.WinDLL("user32"), ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            user32.GetGuiResources.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            user32.GetGuiResources.restype = wintypes.DWORD
            _gdi = (user32.GetGuiResources, k32.GetCurrentProcess)
        count, me = _gdi
        return int(count(me(), 0))                     # GR_GDIOBJECTS
    except Exception:  # noqa: BLE001
        return None


def sample(health, engine=None, gos=0, active_cue=None, t_s=0.0):
    """One row (dict of FIELDS) from a health snapshot (control_interface._current_health)."""
    snap = (health or {}).get("snapshot") or health or {}
    out = snap.get("output") or {}
    audio = snap.get("audio") or {}
    gcs = snap.get("gc") or {}
    textures = None
    if engine is not None:
        textures = len(getattr(engine, "textures", {}) or {}) + len(getattr(engine, "yuv_textures", {}) or {})
    return {
        "t_s": round(t_s, 1),
        "rss_mb": round(rss_mb() or 0.0, 1),
        "threads": threading.active_count(),
        "decoders": len(snap.get("decoders") or []),
        "gl_textures": textures,
        "renders_per_s": out.get("renders_per_s"),
        "late_window": out.get("late_window"),
        "frames_window": out.get("frames_window"),
        "errors_total": sum((snap.get("errors_total") or {}).values()),
        "audio_underruns": audio.get("underruns"),
        "gos": gos,
        "active_cue": active_cue,
        "render_thread_alive": snap.get("render_thread_alive"),
        # False while only stills are on the output: they are drawn once, not every frame
        "active_content": snap.get("active_content"),
        # Python's own allocations and its garbage collector (health.GcStats): objects that pile
        # up make each full collection longer, and everything waits while it runs
        "py_blocks": gcs.get("blocks"),
        "gc_full_total": gcs.get("full_total"),
        "gc_full_pause_total_ms": gcs.get("full_pause_total_ms"),
        "gc_pause_max_ms": gcs.get("pause_max_ms_window"),
        "gdi_objects": gdi_objects(),
    }


CENSUS_TOP = 10


def object_census():
    """{type name: count} of the objects Python's garbage collector tracks
    (those set aside by gc.freeze are not listed), after a full collection.
    Takes a few milliseconds after gc.freeze (tens without it), so the soak
    test takes it only twice: after warm-up and at the end."""
    import gc
    gc.collect()               # only live objects: garbage in a reference cycle waits for a full collection
    counts = {}
    for o in gc.get_objects():
        t = type(o)
        name = t.__qualname__ if t.__module__ in ("builtins", None) else f"{t.__module__}.{t.__qualname__}"
        counts[name] = counts.get(name, 0) + 1
    return counts


def census_growth(before, after, top=CENSUS_TOP):
    """[(type name, more objects)] of the types that grew most, largest first."""
    grew = [(k, after.get(k, 0) - before.get(k, 0)) for k in after]
    grew = [g for g in grew if g[1] > 0]
    grew.sort(key=lambda g: (-g[1], g[0]))
    return grew[:top]


def census_note(grew, total_growth, minimum=5000):
    """A line for the summary when the tracked objects grew noticeably, or None."""
    if total_growth < minimum or not grew:
        return None
    return ("Python objects that grew most after warm-up: "
            + ", ".join(f"{name} +{n:,}" for name, n in grew[:6]) + ".")


# Diagnostics: VISUALDECK_SOAK_TRACEMALLOC=1 makes the soak test trace Python's
# memory allocations after its warm-up and write the allocation sites that grew
# most to soak-<time>-tracemalloc.txt next to the summary. Tracing slows every
# allocation, so late frames in such a run are not comparable. (Owner's PC,
# 2026-11-15: Python's allocated blocks grew a steady 41,000 an hour over 8 h
# while the objects the garbage collector tracks did not.)
TRACE_ENV = "VISUALDECK_SOAK_TRACEMALLOC"
TRACE_FRAMES = 12
TRACE_TOP = 25


def trace_wanted():
    return os.environ.get(TRACE_ENV, "").strip() not in ("", "0")


def trace_start():
    """Start tracing (if not already). Only allocations made from now on are traced, so
    the baseline (trace_baseline) is taken later: by then the show's bounded buffers
    (Health's samples, frame queues, caches) hold traced allocations, and refilling
    them no longer looks like growth (the first reports, 2026-11-15, counted every
    refill since the start: thousands of Health samples)."""
    import tracemalloc
    if not tracemalloc.is_tracing():
        tracemalloc.start(TRACE_FRAMES)


def trace_stop():
    import tracemalloc
    if tracemalloc.is_tracing():
        tracemalloc.stop()


def trace_baseline():
    """The snapshot that trace_report compares the end with."""
    import tracemalloc
    return tracemalloc.take_snapshot()


def _trace_filters():
    import tracemalloc
    return [tracemalloc.Filter(False, tracemalloc.__file__), tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>")]


def trace_report(before, after=None, top=TRACE_TOP, extra_lines=()):
    """Text: the allocation sites (whole tracebacks) whose memory grew most
    between `before` and `after` (now), largest first. Stops tracing."""
    import tracemalloc
    if after is None:
        after = tracemalloc.take_snapshot()
    stats = after.filter_traces(_trace_filters()).compare_to(before.filter_traces(_trace_filters()), "traceback")
    grew = [st for st in stats if st.size_diff > 0 or st.count_diff > 0]
    total_kib = sum(st.size_diff for st in stats) / 1024
    total_blocks = sum(st.count_diff for st in stats)
    lines = [f"Python allocations from the baseline to the end: {total_kib:+.1f} KiB, {total_blocks:+,} blocks in all.",
             *extra_lines, f"The {min(top, len(grew))} allocation sites that grew most (newest call last):", ""]
    for st in grew[:top]:
        lines.append(f"{st.size_diff / 1024:+.1f} KiB, {st.count_diff:+,} blocks (now {st.count:,})")
        lines.extend("    " + ln for ln in st.traceback.format(limit=TRACE_FRAMES))
        lines.append("")
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    return "\n".join(lines)


def tk_command_count(root):
    """How many Python callbacks Tkinter has registered with Tcl, over all
    widgets (each is a name string; they pile up if code keeps binding)."""
    total, todo = 0, [root]
    while todo:
        w = todo.pop()
        total += len(getattr(w, "_tclCommands", None) or ())
        try:
            todo.extend(w.winfo_children())
        except Exception:  # noqa: BLE001
            pass
    return total


def _slope_per_hour(rows, key):
    """Least-squares slope of rows[key] against t_s, per hour (None if too few points)."""
    pts = [(r["t_s"], r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(t for t, _ in pts) / n
    my = sum(v for _, v in pts) / n
    den = sum((t - mx) ** 2 for t, _ in pts)
    if den <= 0:
        return None
    return sum((t - mx) * (v - my) for t, v in pts) / den * 3600.0


def _same_cue_growth(rows, key="rss_mb"):
    """(MB per hour, cue, first, last) from samples taken while the same cue was
    on the output, or None. Memory differs from cue to cue (a still versus a
    video, a clip with sound), so the first sample against the last can show
    growth that is only a heavier cue at the end. Each cue's own trend is
    weighted by its number of samples; cue, first and last are those of the
    cue with the most samples, for the text."""
    groups = {}
    for r in rows:
        if isinstance(r.get(key), (int, float)) and r.get("active_cue") is not None:
            groups.setdefault(str(r["active_cue"]), []).append(r)
    trends = []
    for cue, g in groups.items():
        s = _slope_per_hour(g, key)
        if s is not None and g[-1]["t_s"] - g[0]["t_s"] >= 2 * SAMPLE_S:
            trends.append((len(g), s, cue, g[0][key], g[-1][key]))
    if not trends:
        return None
    n = sum(t[0] for t in trends)
    slope = sum(t[0] * t[1] for t in trends) / n
    _, _, cue, a, b = max(trends, key=lambda t: t[0])
    return slope, cue, a, b


def _full_window(r):
    """Whether a sample's render rate covers most of Output Health's window. A window
    that starts or ends a transition onto a still picture holds only that transition's
    frames (one, when the sample falls on the GO itself), and its rate says nothing
    about the output: in the owner's 72-hour soak (2026-11-20) 22 such samples read
    0.0 renders/s, with every full window at 47.8 or more. Rows without the frame count
    are taken as they are."""
    fw, rate = r.get("frames_window"), r.get("renders_per_s")
    if not isinstance(fw, int) or not isinstance(rate, (int, float)):
        return True
    return fw >= 2 and rate > 0 and (fw - 1) / rate >= FULL_WINDOW * WINDOW_S


def _stalls(rows):
    """Samples where moving content drew fewer than two frames in the window, twice in a
    row: the output stopped drawing for a sample period or more. One such sample alone
    is a GO at the moment of sampling (the transition's first frame)."""
    def empty(r):
        return r.get("active_content") is True and isinstance(r.get("frames_window"), int) \
            and r["frames_window"] < 2
    return [a for a, b in zip(rows, rows[1:]) if empty(a) and empty(b)]


def summarize(rows, target_fps=None):
    """{'duration_s', 'gos', numbers..., 'findings': [text], 'ok': bool} for recorded rows.
    The first fifth of the run (at least one sample) is warm-up and not used
    for growth (caches and decoders fill up at the start)."""
    if not rows:
        return {"duration_s": 0, "gos": 0, "findings": ["No samples were recorded."], "ok": False}
    warm = max(1, len(rows) // 5)
    steady = rows[warm:] if len(rows) - warm >= 3 else rows
    first, last = steady[0], rows[-1]
    res = {"duration_s": last["t_s"], "gos": last.get("gos", 0), "samples": len(rows)}
    findings = []
    mem = [r["rss_mb"] for r in rows if r.get("rss_mb")]
    if mem:
        res.update(rss_start_mb=rows[0]["rss_mb"], rss_end_mb=last["rss_mb"], rss_peak_mb=max(mem))
    slope = _slope_per_hour(steady, "rss_mb")
    res["rss_growth_mb_per_h"] = round(slope, 1) if slope is not None else None
    same = _same_cue_growth(steady)
    steady_s = steady[-1]["t_s"] - steady[0]["t_s"]
    notes = []
    if steady_s < MIN_GROWTH_S:
        # In a few minutes the allocator and the caches are still filling: a trend
        # then is not a leak (a 4-minute run here extrapolated 32 MB to 3.6 GB/h).
        notes.append(f"Memory growth is judged from runs of {(MIN_GROWTH_S + 60) // 60:.0f} minutes or more; "
                     f"this one was too short to tell.")
    if same is not None:                          # cues differ in memory: compare each with itself
        slope, cue, a, b = same
        res["rss_same_cue_growth_mb_per_h"] = round(slope, 1)
        hours = steady_s / 3600.0
        if steady_s >= MIN_GROWTH_S and slope > 50 and slope * hours > 50:   # less noisy than first against last
            name = f"cue {int(cue) + 1}" if str(cue).isdigit() else f"cue {cue}"   # columns count from 0
            findings.append(f"Memory keeps growing: {slope:.0f} MB per hour after warm-up, comparing "
                            f"samples at the same cue ({name}: {a:.0f} to {b:.0f} MB).")
    else:
        grown = (last.get("rss_mb") or 0) - (first.get("rss_mb") or 0)
        if steady_s >= MIN_GROWTH_S and slope is not None and slope > 50 and grown > 100:
            findings.append(f"Memory keeps growing: {slope:.0f} MB per hour after warm-up "
                            f"({first['rss_mb']:.0f} to {last['rss_mb']:.0f} MB).")
    # The second half alone: memory that rises once (a cache or the allocator reaching
    # its size) and then stays level is not a leak, but it bends a straight line fitted
    # over the whole run. The owner's 8-hour soak of 2026-11-15 rose about 90 MB at
    # 3 h 9 min and then stayed level to the end: 15.5 MB an hour over the run, under
    # 1 MB an hour per cue over the last 4.7 hours.
    half = [r for r in rows if r["t_s"] >= rows[-1]["t_s"] / 2]
    if half and half[-1]["t_s"] - half[0]["t_s"] >= MIN_GROWTH_S:
        late = _same_cue_growth(half)
        if late is not None:
            res["rss_same_cue_growth_second_half_mb_per_h"] = round(late[0], 1)
    blocks = _same_cue_growth(steady, "py_blocks")
    if blocks is not None:
        b_slope, cue, a, b = blocks
        res["py_blocks_growth_per_h"] = round(b_slope)
        level = max(1.0, float(a))
        hours = (steady[-1]["t_s"] - steady[0]["t_s"]) / 3600.0
        if hours * 3600 >= MIN_GROWTH_S and b_slope > 50_000 and b_slope * hours > max(25_000, 0.05 * level):
            name = f"cue {int(cue) + 1}" if str(cue).isdigit() else f"cue {cue}"
            findings.append(f"Python objects keep growing: about {b_slope / 1000:.0f} thousand per hour at the "
                            f"same cue ({name}: {a / 1000:.0f} to {b / 1000:.0f} thousand). Each full garbage "
                            f"collection then takes longer.")
    full = [(r["gc_full_total"], r["gc_full_pause_total_ms"]) for r in steady
            if isinstance(r.get("gc_full_total"), int) and isinstance(r.get("gc_full_pause_total_ms"), (int, float))]
    if len(full) >= 2 and full[-1][0] > full[0][0]:
        n = full[-1][0] - full[0][0]
        res["gc_full_collections"] = n
        res["gc_full_pause_avg_ms"] = round((full[-1][1] - full[0][1]) / n, 2)
    pauses = [r["gc_pause_max_ms"] for r in rows[warm:] if isinstance(r.get("gc_pause_max_ms"), (int, float))]
    if pauses:
        res["gc_pause_max_ms"] = max(pauses)
        frame_ms = 1000.0 / target_fps if target_fps else None
        if frame_ms and max(pauses) > frame_ms:
            findings.append(f"Python's garbage collector held everything up for up to {max(pauses):.0f} ms "
                            f"(one frame is {frame_ms:.1f} ms): a late frame each time.")
    gdi = [(r["t_s"], r["gdi_objects"]) for r in steady if isinstance(r.get("gdi_objects"), int)]
    if len(gdi) >= 3:
        res["gdi_objects_start"], res["gdi_objects_end"] = gdi[0][1], gdi[-1][1]
        g_slope = _slope_per_hour(steady, "gdi_objects")
        res["gdi_growth_per_h"] = round(g_slope) if g_slope is not None else None
        if g_slope is not None and g_slope > 100 and gdi[-1][1] - gdi[0][1] > 50:     # a count: no warm-up
            hours = (GDI_LIMIT - gdi[-1][1]) / g_slope
            findings.append(f"GDI objects keep growing: {g_slope:.0f} per hour ({gdi[0][1]} to {gdi[-1][1]}); "
                            f"at this rate Windows' limit of {GDI_LIMIT:,} is reached in about {hours:.0f} "
                            f"hours, and then drawing fails.")
    for key, label, limit in (("threads", "threads", 8), ("decoders", "decoders", 4),
                              ("gl_textures", "GPU textures", 16)):
        vals = [r[key] for r in steady if isinstance(r.get(key), int)]
        if vals:
            res[f"{key}_start"], res[f"{key}_end"], res[f"{key}_max"] = vals[0], vals[-1], max(vals)
            if vals[-1] - min(vals) > limit and vals[-1] >= max(vals[:max(1, len(vals) // 2)]) + limit // 2:
                findings.append(f"The number of {label} grew from {min(vals)} to {vals[-1]}.")
    errs = (last.get("errors_total") or 0) - (rows[0].get("errors_total") or 0)
    res["errors"] = errs
    if errs:
        findings.append(f"{errs} render error(s) during the run (Output Health lists them).")
    moving = [r for r in rows[warm:] if r.get("active_content") is not False]   # stills are not redrawn
    rates = [r["renders_per_s"] for r in moving
             if isinstance(r.get("renders_per_s"), (int, float)) and _full_window(r)]
    if rates:
        res["renders_per_s_min"] = min(rates)
        res["renders_per_s_avg"] = round(sum(rates) / len(rates), 1)
        if target_fps and min(rates) < 0.9 * target_fps:
            findings.append(f"The output frame rate dropped to {min(rates):.1f} renders/s "
                            f"(target {target_fps:g}).")
    late = sum(r.get("late_window") or 0 for r in moving)
    frames = sum(r.get("frames_window") or 0 for r in moving)
    res["late_share"] = round(late / frames, 4) if frames else None
    if frames and late / frames > 0.01:
        findings.append(f"{late / frames:.1%} of frames were late.")
    und = [r["audio_underruns"] for r in rows if isinstance(r.get("audio_underruns"), int)]
    if und and und[-1] - und[0] > 0:
        findings.append(f"{und[-1] - und[0]} sound underrun(s).")
    stalls = _stalls(rows[warm:])
    if stalls:
        findings.append(f"The output stopped drawing moving content {len(stalls)} time(s), first at "
                        f"{stalls[0]['t_s'] / 60:.0f} min.")
    if any(r.get("render_thread_alive") is False for r in rows):
        findings.append("The render thread stopped during the run.")
    if res["gos"] == 0:
        findings.append("No cue was played (does the show have cues with media?).")
    # Below the findings' 50 MB an hour but above the acceptance bar for a long show
    # (the acceptance bar: under 10 MB an hour after the first hour):
    # said in a note, so an 8-hour run at 10 MB/h does not read as perfectly level.
    grow = res.get("rss_same_cue_growth_mb_per_h", res.get("rss_growth_mb_per_h"))
    if steady_s >= LONG_RUN_S and isinstance(grow, (int, float)) and BAR_MB_PER_H <= grow <= 50 \
            and not any(f.startswith("Memory keeps growing") for f in findings):
        second = res.get("rss_same_cue_growth_second_half_mb_per_h")
        if isinstance(second, (int, float)) and second < BAR_MB_PER_H:
            notes.append(f"Memory grew about {grow:.0f} MB an hour after warm-up by a straight line over the run, "
                         f"but {second:.1f} MB an hour over its second half: it rose earlier and then stayed "
                         f"level (the {BAR_MB_PER_H:.0f} MB an hour bar is exceeded by the first measure, "
                         f"not by the second).")
        else:
            notes.append(f"Memory grew about {grow:.0f} MB an hour after warm-up: within what this test calls "
                         f"drift (50), at or above the {BAR_MB_PER_H:.0f} MB an hour a long show should stay under.")
    level = ("threads, decoders and GPU textures stayed level" if notes
             else "memory, threads, decoders and GPU textures stayed level")
    res["findings"] = findings or [f"Nothing drifted: {level}, no errors, no frame rate drops."]
    res["notes"] = notes
    res["ok"] = not findings
    return res


class Recorder:
    """Rows to a CSV file as they come; summary JSON at the end."""

    def __init__(self, folder, stamp=None):
        os.makedirs(folder, exist_ok=True)
        stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
        self.csv_path = os.path.join(folder, f"soak-{stamp}.csv")
        self.summary_path = os.path.join(folder, f"soak-{stamp}.json")
        self.rows = []
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(FIELDS)

    def add(self, row):
        self.rows.append(row)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(k) for k in FIELDS])

    def finish(self, target_fps=None, extra=None):
        s = summarize(self.rows, target_fps)
        s.update(extra or {})
        self.rewrite(s)
        return s

    def rewrite(self, summary):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
