# SPDX-License-Identifier: MPL-2.0
"""Render pacing: the frame pacer keeps its average rate when sleeps run over,
and the render thread renders on every tick while a video plays (owner's
report 2026-11-12: 45.6 renders/s and a third of frames late at a 60 fps
target with a 4 ms render). Timing tests on a shared machine: bounds are loose."""
import queue
import random
import threading
import time

import control_interface as ci_mod
import render_engine as rem
from test_render_thread import FlakyEngine, _video_layers


def test_pacer_makes_up_late_wakeups(monkeypatch):
    real_sleep = time.sleep
    rnd = random.Random(3)
    monkeypatch.setattr(rem.time, "sleep", lambda s: real_sleep(s + rnd.random() * 0.002))  # Windows-like overrun
    p = rem.PreciseFramePacer(60)
    t0 = time.perf_counter()
    n = 120
    for _ in range(n):
        p.wait_for_next_frame()
    rate = n / (time.perf_counter() - t0)
    assert 59.5 <= rate <= 60.5, rate


def test_pacer_does_not_rush_after_a_stall():
    p = rem.PreciseFramePacer(60)
    p.wait_for_next_frame()
    time.sleep(0.1)                       # a slow frame or a pause
    p.wait_for_next_frame()               # restarts the grid from now
    t = time.perf_counter()
    p.wait_for_next_frame()
    assert time.perf_counter() - t >= 0.9 / 60, "no burst of catch-up frames"


class SteadyEngine(FlakyEngine):
    def render(self, *a, **k):
        self.calls += 1
        t = time.perf_counter()
        while time.perf_counter() - t < 0.004:      # a 4 ms render, as measured on the owner's PC
            pass


def test_render_thread_renders_every_tick_with_a_video(fixtures):
    eng = SteadyEngine()
    cmd_q, prev_q, stop = queue.Queue(), queue.Queue(maxsize=1), threading.Event()
    rt = ci_mod.RenderThread(eng, cmd_q, prev_q, stop)
    rt.start()
    cmd_q.put({"action": "render", "layers": _video_layers(fixtures), "target_monitor": 0,
               "transition_state": {"active": False}, "all_assigned_screens": set()})
    time.sleep(2.5)
    snap = rt.loop_stats.snapshot(time.perf_counter())
    out = rt.frame_stats.snapshot(time.perf_counter())
    stop.set()
    rt.join(2)
    assert snap["active_ticks"] > 100
    assert snap["skipped_active_ticks"] <= snap["active_ticks"] * 0.02, snap
    assert out["renders_per_s"] >= 55 and out["late_window"] <= out["frames_window"] * 0.05, out


def test_cpu_preview_scales_before_converting():
    import numpy as np
    a = np.zeros((216, 384, 4), np.uint8)
    a[...] = (10, 20, 30, 200)                          # BGRA
    f = rem.BaseRenderEngine._video_frame_to_pil
    img = f((a.tobytes(), (384, 216)), 0.25, 0.25)
    assert img.size == (96, 54) and img.mode == "RGBA" and img.getpixel((10, 10)) == (30, 20, 10, 200)
    assert f(np.ascontiguousarray(a[..., :3]), 1.0, 1.0).getpixel((1, 1)) == (30, 20, 10, 255)
    assert f({"format": "rgba", "rgba": a}, 2.0, 2.0).size == (768, 432)
    assert f(a[..., 0].copy(), 0.5, 0.5).getpixel((0, 0)) == (10, 10, 10, 255)
    assert f((a.tobytes(), (384, 216)), 0.001, 0.25) is None


def test_overruns_count_late_frames_not_jitter(monkeypatch):
    real_sleep = time.sleep
    rnd = random.Random(5)
    monkeypatch.setattr(rem.time, "sleep", lambda s: real_sleep(s + rnd.random() * 0.002))
    p = rem.PreciseFramePacer(60)
    for _ in range(90):
        p.wait_for_next_frame()
    assert p.overrun_ratio() < 0.05, "1-2 ms late wake-ups that the grid makes up are not overruns"
    monkeypatch.setattr(rem.time, "sleep", real_sleep)
    p = rem.PreciseFramePacer(60)
    for _ in range(30):
        t = time.perf_counter()
        while time.perf_counter() - t < 0.026:      # 26 ms of work per frame at a 60 fps target
            pass
        p.wait_for_next_frame()
    assert p.overrun_ratio() > 0.8, "frames that cannot keep up are"
