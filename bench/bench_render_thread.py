# SPDX-License-Identifier: MPL-2.0
"""Render-thread throughput benchmark (real RenderThread + OpenGLRenderEngine).

Simulates the common show setup: GUI on display 0, output on display 1, one
1080p H.264 layer playing, and the GUI asking for a preview every 33 ms (what
ControlInterface._gui_animation_loop does while video plays).

Two virtual 1920x1080 monitors are simulated on one 3840x1080 X screen by
patching screeninfo; run under Xvfb:

  xvfb-run -a -s "-screen 0 3840x1080x24" .venv/bin/python bench/bench_render_thread.py \
      --seconds 8 --label baseline

Results are environment-specific (CPU, GPU/driver, software vs hardware GL).
Only compare runs made on the same machine. Output: JSON on stdout and in
bench/results/<label>.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import statistics
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from screeninfo import Monitor  # noqa: E402
import render_engine as rem  # noqa: E402

MONITORS = [Monitor(x=0, y=0, width=1920, height=1080, name="sim0", is_primary=True),
            Monitor(x=1920, y=0, width=1920, height=1080, name="sim1", is_primary=False)]
rem.get_monitors = lambda: list(MONITORS)

import control_interface as ci_mod  # noqa: E402
from media_manager import MediaManager  # noqa: E402
from media_fixtures import testsrc_clip  # noqa: E402


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def run(seconds, preview, clip):
    eng = rem.OpenGLRenderEngine(MediaManager(), use_pbo=True)
    intervals, durations = [], []
    last = [None]
    orig_render = eng.render

    def timed_render(*a, **k):
        t0 = time.perf_counter()
        if last[0] is not None:
            intervals.append(t0 - last[0])
        last[0] = t0
        r = orig_render(*a, **k)
        durations.append(time.perf_counter() - t0)
        return r

    eng.render = timed_render
    cmd_q, prev_q, stop = queue.Queue(), queue.Queue(maxsize=1), threading.Event()
    rt = ci_mod.RenderThread(eng, cmd_q, prev_q, stop)
    rt.start()
    layers = {0: {"path": clip, "screens": [1], "order": 0, "loop": True,
                  "position": {"x": 0.5, "y": 0.5, "scale_x": 1.0, "scale_y": 1.0, "rotation": 0}}}
    cmd_q.put({"action": "render", "layers": layers, "target_monitor": 1,
               "transition_state": {"active": False}, "force_restart_paths": {clip},
               "gui_monitor": 0, "output_mode": "Display 1", "all_assigned_screens": set()})
    time.sleep(1.5)  # warm-up: window creation, decoder start
    intervals.clear()
    durations.clear()
    previews = 0
    t_end = time.perf_counter() + seconds
    next_req = time.perf_counter()
    while time.perf_counter() < t_end:
        now = time.perf_counter()
        if preview and now >= next_req:
            cmd_q.put({"action": "request_preview"})
            next_req = now + 0.033
        try:
            prev_q.get_nowait()
            previews += 1
        except queue.Empty:
            pass
        time.sleep(0.002)
    alive = rt.is_alive()
    stop.set()
    rt.join(3)
    n = len(intervals)
    return {
        "preview_requests": preview,
        "render_thread_alive_at_end": alive,
        "renders_per_sec": round(n / seconds, 1),
        "interval_ms_p50": round(1000 * pct(intervals, 50), 1) if n else None,
        "interval_ms_p95": round(1000 * pct(intervals, 95), 1) if n else None,
        "interval_ms_p99": round(1000 * pct(intervals, 99), 1) if n else None,
        "interval_ms_max": round(1000 * max(intervals), 1) if n else None,
        "render_call_ms_median": round(1000 * statistics.median(durations), 2) if durations else None,
        "previews_per_sec": round(previews / seconds, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--label", default="run")
    ap.add_argument("--mode", choices=["both", "preview", "nopreview"], default="both")
    args = ap.parse_args()
    clip = os.path.join(tempfile.gettempdir(), "vd_bench_testsrc_1080p.mp4")
    if not os.path.exists(clip):
        testsrc_clip(clip, 1920, 1080, n_frames=150)
    results = {"label": args.label, "python": platform.python_version(), "platform": platform.platform(),
               "cpu_count": os.cpu_count(), "clip": "1920x1080 H.264 30 fps (synthetic)", "runs": []}
    if args.mode in ("both", "nopreview"):
        results["runs"].append(run(args.seconds, False, clip))
    if args.mode in ("both", "preview"):
        results["runs"].append(run(args.seconds, True, clip))
    out_dir = os.path.join(ROOT, "bench", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.label}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
