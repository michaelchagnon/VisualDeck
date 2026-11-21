# SPDX-License-Identifier: MPL-2.0
"""Cue-trigger latency: time from a cut command to the first presented frame
that shows the new cue, through the real RenderThread + OpenGLRenderEngine.

Compares a cold cut (the new cue's decoders start on GO) with a preloaded cut
(the UI preloaded the cue with paused players, as ControlInterface does for
the next cue). Run under a display, e.g.:

  xvfb-run -a -s "-screen 0 1920x1080x24" .venv/bin/python bench/bench_cue_latency.py --label mymachine

Numbers depend heavily on the machine (decoder start-up cost, render loop
rate); only compare runs from the same machine. Results: bench/results/<label>-cue-latency.json
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
sys.path.insert(0, os.path.join(ROOT, "tests"))
import gl_harness as g  # noqa: E402  (patches pygame.display.flip to record presented frames)
import control_interface as ci_mod  # noqa: E402
from media_fixtures import solid_clip  # noqa: E402

TIMES = []
_flip = g.pygame.display.flip


def _timed_flip():
    _flip()
    TIMES.append(time.perf_counter())


g.pygame.display.flip = _timed_flip
g.LAST["center_only"] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", type=int, default=8, help="cuts per mode")
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    d = tempfile.mkdtemp(prefix="vd_latency_")
    white = solid_clip(os.path.join(d, "white.mp4"), 1280, 720, 235, 128, 128, n_frames=90)
    red = solid_clip(os.path.join(d, "red.mp4"), 1280, 720, 63, 102, 240, n_frames=90)

    eng = g.rem.OpenGLRenderEngine(g.MediaManager(), use_pbo=True)
    cmd_q, prev_q, stop = queue.Queue(), queue.Queue(maxsize=1), threading.Event()
    rt = ci_mod.RenderThread(eng, cmd_q, prev_q, stop)
    rt.start()
    act = [0]

    def lay(path, a, hold=False):
        scale = (eng.monitors[0].width / 1280.0) if eng.monitors else 1.5
        layer = dict(g.layer(path, 0, scale=scale), instance=f"@{a}:0:0|{path}")
        if hold:
            layer["hold"] = True
        return layer

    def cut(path, a):
        cmd_q.put({"action": "render", "preroll": True, "layers": {0: lay(path, a)}, "target_monitor": 0,
                   "transition_state": {"active": False}, "gui_monitor": 0, "output_mode": "Display 0",
                   "all_assigned_screens": set()})

    def is_red(c):
        return c[0] > 200 and c[1] < 60

    def measure_cut_to_red(preload):
        # live: white
        act[0] += 1
        cut(white, act[0])
        time.sleep(0.8)
        act[0] += 1
        red_act = act[0]
        if preload:
            cmd_q.put({"action": "update_layers", "layers": {"live": lay(white, red_act - 1),
                                                             "pre": lay(red, red_act, hold=True)}})
            time.sleep(0.8)  # preload settles: first frame decoded, player paused
        n0 = len(g.LAST["centers"])
        t0 = time.perf_counter()
        cut(red, red_act)
        deadline = t0 + 3.0
        while time.perf_counter() < deadline:
            centers = g.LAST["centers"][n0:]
            times = TIMES[n0:]
            for c, t in zip(centers, times):
                if is_red(c):
                    return (t - t0) * 1000.0
            time.sleep(0.002)
        return None

    results = {"cold_ms": [], "preloaded_ms": []}
    try:
        for i in range(args.cuts):
            for mode in (("cold_ms", False), ("preloaded_ms", True)):
                v = measure_cut_to_red(mode[1])
                if v is not None:
                    results[mode[0]].append(round(v, 1))
    finally:
        stop.set()
        rt.join(3)

    summary = {
        "label": args.label, "python": platform.python_version(), "platform": platform.platform(),
        "clip": "1280x720 H.264 solid colour (synthetic)",
        "cold_median_ms": round(statistics.median(results["cold_ms"]), 1) if results["cold_ms"] else None,
        "preloaded_median_ms": round(statistics.median(results["preloaded_ms"]), 1) if results["preloaded_ms"] else None,
        "raw": results,
        "render_errors": dict(getattr(rt, "error_counts", {})),
    }
    out = os.path.join(ROOT, "bench", "results", f"{args.label}-cue-latency.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
