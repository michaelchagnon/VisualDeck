# SPDX-License-Identifier: MPL-2.0
"""The render thread must keep producing output when one frame fails."""
import queue
import threading
import time

import control_interface as ci_mod


class FlakyEngine:
    """Minimal engine double: render() raises on selected calls."""
    def __init__(self, fail_on=(2,), preview_fails=False):
        self.window = None
        self.monitors = []
        self.calls = 0
        self.fail_on = set(fail_on)
        self.preview_fails = preview_fails
        self.cleared = False

    def render(self, layers, target_monitor=None, transition_state=None, all_assigned_screens=None):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("simulated GL/driver failure")

    def update_layers(self, layers, force_restart_paths=None):
        pass

    def render_layers_to_image(self, layers, dims):
        if self.preview_fails:
            raise ValueError("simulated preview failure")
        return None

    def clear(self):
        self.cleared = True


def _start(engine, layers):
    cmd_q, prev_q, stop = queue.Queue(), queue.Queue(maxsize=1), threading.Event()
    rt = ci_mod.RenderThread(engine, cmd_q, prev_q, stop)
    rt.start()
    cmd_q.put({"action": "render", "layers": layers, "target_monitor": 0,
               "transition_state": {"active": False}, "all_assigned_screens": set()})
    return rt, cmd_q, stop


def _video_layers(fixtures):
    # a real video path makes the render loop run continuously (~60 fps)
    return {0: {"path": fixtures["counting"], "screens": [0], "order": 0, "position": {}}}


def test_render_thread_survives_render_exception(fixtures):
    eng = FlakyEngine(fail_on={2})
    rt, cmd_q, stop = _start(eng, _video_layers(fixtures))
    time.sleep(0.6)
    alive = rt.is_alive()
    calls = eng.calls
    stop.set()
    rt.join(2)
    assert alive, "render thread died after a single failed frame"
    assert calls > 5


def test_render_thread_survives_preview_exception(fixtures):
    eng = FlakyEngine(fail_on=set(), preview_fails=True)
    rt, cmd_q, stop = _start(eng, _video_layers(fixtures))
    for _ in range(5):
        cmd_q.put({"action": "request_preview"})
        time.sleep(0.05)
    time.sleep(0.2)
    alive = rt.is_alive()
    stop.set()
    rt.join(2)
    assert alive, "render thread died after a failed preview"


def test_render_thread_stops_cleanly(fixtures):
    eng = FlakyEngine(fail_on=set())
    rt, cmd_q, stop = _start(eng, _video_layers(fixtures))
    time.sleep(0.1)
    cmd_q.put({"action": "stop"})
    rt.join(2)
    assert not rt.is_alive()
    assert eng.cleared


def test_a_still_picture_is_drawn_again_when_the_engine_asks():
    """New mesh maps built in the background change the picture of a still cue,
    which is otherwise only drawn when a command changes something."""
    eng = FlakyEngine(fail_on=set())
    eng.redraw_pending = False
    still = {0: {"path": "/no/such/still.png", "screens": [0], "order": 0, "position": {}}}
    rt, cmd_q, stop = _start(eng, still)
    try:
        time.sleep(0.4)
        idle = eng.calls
        time.sleep(0.3)
        assert eng.calls == idle, "a still cue is not redrawn on every tick"
        eng.redraw_pending = True
        time.sleep(0.3)
        assert eng.calls == idle + 1 and eng.redraw_pending is False
    finally:
        stop.set()
        rt.join(2)
