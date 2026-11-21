# SPDX-License-Identifier: MPL-2.0
"""Health metrics: frame timing, decoder status, assessment, render-thread snapshot."""
import os
import queue
import threading
import time

import pytest

import control_interface as ci_mod
import health
import render_engine as rem
from ring_buffer import FastRingBuffer
from test_player_instances import vlayer, wait_for
from test_render_thread import FlakyEngine


def feed(fs, start, n, period, active=True, target=60, cost=0.004):
    t = start
    for _ in range(n):
        t += period
        fs.record(t, cost, active, target)
    return t


def test_steady_output_has_no_late_frames():
    fs = health.FrameStats()
    t = feed(fs, 100.0, 300, 1 / 60)
    snap = fs.snapshot(t)
    assert snap["late_window"] == 0 and snap["late_total"] == 0
    assert 59.5 < snap["renders_per_s"] < 60.5
    assert abs(snap["interval_ms"]["p99"] - 16.67) < 0.1 and snap["render_ms"]["p50"] == 4.0


def test_a_stall_during_video_counts_as_late_but_a_static_gap_does_not():
    fs = health.FrameStats()
    t = feed(fs, 0.0, 60, 1 / 60)
    t += 0.1                               # 100 ms stall while video plays
    fs.record(t, 0.004, True, 60)
    t = feed(fs, t, 10, 1 / 60, active=False)
    t += 2.0                               # nothing on screen moving: renders on demand
    fs.record(t, 0.004, False, 60)
    t += 2.0                               # first frame after video starts again: gap not counted
    fs.record(t, 0.004, True, 60)
    snap = fs.snapshot(t)
    assert snap["late_total"] == 1 and snap["interval_ms"]["max"] == 2000.0


def test_late_threshold_follows_the_render_target():
    fs = health.FrameStats()
    t = feed(fs, 0.0, 30, 1 / 30, target=30)   # 33 ms frames at a 30 fps target: on time
    assert fs.snapshot(t)["late_total"] == 0
    feed(fs, t, 30, 1 / 30, target=60)         # same frames at a 60 fps target: all late
    assert fs.late_total == 30


def test_old_samples_leave_the_window():
    fs = health.FrameStats(window_s=5.0)
    t = feed(fs, 0.0, 60, 1 / 60)
    snap = fs.snapshot(t + 10.0)
    assert snap["frames_window"] == 0 and snap["frames_total"] == 60 and snap["last_frame_age_s"] == 10.0


def test_error_log_counts_recent_errors_by_phase():
    el = health.ErrorLog()
    el.add(0.0, "render", RuntimeError("old"))
    el.add(100.0, "render", RuntimeError("a"))
    el.add(101.0, "preview", ValueError("b"))
    assert el.recent(110.0) == {"render": 1, "preview": 1}
    assert el.last == ("preview", "ValueError: b")


def test_ring_buffer_counts_frames_taken_and_dropped():
    q = FastRingBuffer(2)
    for i in range(3):
        q.put_nowait_drop_old(i)      # third put overwrites the oldest
    assert q.dropped == 1
    assert q.get_latest() == 2 and q.dropped == 2 and q.taken == 1


def snap_with(**kw):
    base = {"render_thread_alive": True, "target_fps": 60, "active_content": True,
            "output": {"frames_window": 300, "late_window": 0, "last_frame_age_s": 0.01},
            "errors_recent": {}, "decoders": [], "preroll": {}, "missing_media": 0}
    base.update(kw)
    return base


def test_assessment_levels():
    assert health.assess(snap_with()) == ("ok", [])
    lvl, why = health.assess(snap_with(output={"frames_window": 300, "late_window": 9, "last_frame_age_s": 0.01}))
    assert lvl == "warn" and "9 late frames" in why[0]
    assert health.assess(snap_with(errors_recent={"render": 2}))[0] == "error"
    assert health.assess(snap_with(output={"frames_window": 0, "late_window": 0, "last_frame_age_s": 3.0}))[0] == "error"
    dec = {"file": "a.mp4", "state": "retrying"}
    assert health.assess(snap_with(decoders=[dict(dec, live=True)]))[0] == "error"
    lvl, why = health.assess(snap_with(decoders=[dict(dec, live=False)]))
    assert lvl == "warn" and "(next cue)" in why[0]
    slow = {"file": "4k.mov", "state": "playing", "source_fps": 60.0, "decode_fps": 41.0, "frames_decoded": 500}
    assert "decoding 41 of 60 fps" in health.assess(snap_with(decoders=[slow]))[1][0]
    assert health.assess(snap_with(target_fps=45))[0] == "warn"
    # Only stills on the output: the lowered target means nothing (owner's PC, 2026-11-14).
    assert health.assess(snap_with(target_fps=45, active_content=False)) == ("ok", [])
    assert health.assess(snap_with(missing_media=2))[1] == ["2 missing media file(s)"]
    # Named (owner's PC, 2026-11-14: the status line said only "1 missing media file(s)").
    named = snap_with(missing_media=4, missing_media_names=["a.mov", "b.png", "c.mp4", "d.mov"])
    assert health.assess(named)[1] == ["4 missing media file(s): a.mov, b.png, c.mp4 and 1 more"]


def test_summary_line_is_one_readable_line():
    line = health.summary_line(snap_with(output={"renders_per_s": 59.9, "frames_window": 300, "late_window": 1,
                                                 "late_total": 4, "interval_ms": {"p99": 17.1},
                                                 "render_ms": {"p99": 3.2}, "last_frame_age_s": 0.01},
                                         decoders=[{"file": "a.mp4", "state": "playing", "frames_dropped": 3}]))
    assert "\n" not in line and line.startswith("Health ok: 59.9 renders/s")
    assert "late 1/300" in line and "dropped 3" in line and "{'playing': 1}" in line


@pytest.fixture
def engine():
    eng = rem.BaseRenderEngine(media_manager=None, mode="opengl")
    yield eng
    eng.stop_players(list(eng.active_players))


def test_real_decoders_report_their_state(engine, fixtures, tmp_path):
    live = vlayer(fixtures["counting"], 1)
    pre = vlayer(fixtures["counting"], 2, row=1, hold=True)
    missing_but_present = str(tmp_path / "broken.mp4")
    open(missing_but_present, "wb").write(b"not a video")
    broken = vlayer(missing_but_present, 3, row=2)
    engine.update_layers({0: live, 1: pre, 2: broken})
    stats = lambda: {d["instance"]: d for d in engine.player_stats({live["instance"]})}  # noqa: E731
    got = wait_for(lambda: (s := stats()) and (s["@1:0:0"].get("frames_decoded") or 0) > 10
                   and s["@3:2:0"]["state"] == "retrying" and s)
    assert got, stats()
    a, b, c = got["@1:0:0"], got["@2:1:0"], got["@3:2:0"]
    assert a["live"] and a["state"] == "playing" and a["source_fps"] == 30.0 and a["hw_engaged"] is False
    assert b["state"] == "held" and b["frames_decoded"] == 1 and not b["live"]
    assert c["last_error"].startswith("cannot open")
    engine._get_latest_video_frame(live["instance"])
    assert engine.player_stats()[0]["frames_shown"] >= 1


def test_render_thread_publishes_snapshots(fixtures, monkeypatch):
    monkeypatch.setattr(ci_mod.RenderThread, "HEALTH_PERIOD", 0.1)
    eng = FlakyEngine(fail_on={3})
    eng.player_stats = lambda live: [{"file": "x.mp4", "state": "playing", "live": bool(live)}]
    eng.source_key = rem.BaseRenderEngine.source_key
    eng.render_layers_to_image = lambda layers, dims, reference_monitor=None: None
    cmd_q, prev_q, stop = queue.Queue(), queue.Queue(maxsize=1), threading.Event()
    rt = ci_mod.RenderThread(eng, cmd_q, prev_q, stop)
    rt.start()
    cmd_q.put({"action": "render", "target_monitor": 0, "transition_state": {"active": False},
               "all_assigned_screens": set(),
               "layers": {0: {"path": fixtures["counting"], "screens": [0], "order": 0, "position": {}}}})
    snap = wait_for(lambda: rt.health_snapshot and rt.health_snapshot["output"]["frames_window"] > 20
                    and rt.health_snapshot, timeout=3)
    stop.set()
    rt.join(2)
    assert snap, rt.health_snapshot
    assert snap["errors_recent"] == {"render": 1} and snap["last_error"]["phase"] == "render"
    assert snap["active_content"] and snap["decoders"][0]["live"]
    assert snap["output"]["renders_per_s"] > 20
    lvl, why = health.assess(snap)
    assert lvl == "error" and "render error" in why[0]


def test_a_failing_health_snapshot_never_stops_rendering(fixtures, monkeypatch):
    monkeypatch.setattr(ci_mod.RenderThread, "HEALTH_PERIOD", 0.05)
    eng = FlakyEngine(fail_on=())          # no source_key/player_stats: snapshot raises
    cmd_q, prev_q, stop = queue.Queue(), queue.Queue(maxsize=1), threading.Event()
    rt = ci_mod.RenderThread(eng, cmd_q, prev_q, stop)
    rt.start()
    cmd_q.put({"action": "render", "target_monitor": 0, "transition_state": {"active": False},
               "all_assigned_screens": set(),
               "layers": {0: {"path": fixtures["counting"], "screens": [0], "order": 0, "position": {}}}})
    time.sleep(0.5)
    calls, alive = eng.calls, rt.is_alive()
    stop.set()
    rt.join(2)
    assert alive and calls > 15 and rt.error_counts.get("health", 0) > 1


class FakeLabel:
    def __init__(self):
        self.kw = {}

    def config(self, **kw):
        self.kw.update(kw)


class FakeRT:
    def __init__(self, snap, alive=True):
        self.health_snapshot, self._alive = snap, alive

    def is_alive(self):
        return self._alive


@pytest.fixture
def ui(tmp_path, monkeypatch):
    from media_manager import MediaManager
    monkeypatch.setattr(ci_mod.ControlInterface, "CONFIG_DIR", str(tmp_path / ".visualdeck"))
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    obj = ci_mod.ControlInterface(MediaManager())
    obj.fps_label = FakeLabel()
    yield obj
    obj.proxy_executor.shutdown(wait=False)


def test_indicator_shows_level_rate_and_first_reason(ui):
    ui.render_thread = FakeRT(snap_with(time=time.time(), output={"renders_per_s": 59.6, "frames_window": 300,
                                                                   "late_window": 12, "last_frame_age_s": 0.01}))
    ui.grid_cells[(0, 0)] = "/gone/clip.mp4"
    ui._update_health_indicator()
    text, color = ui.fps_label.kw["text"], ui.fps_label.kw["fg"]
    assert text.startswith("● WARN   60 fps   12 late   0 errors")
    assert "12 late frames" in text and "(+1 more)" in text          # second reason: missing media
    assert color == ui.HEALTH_COLORS["warn"]


def test_indicator_says_still_picture_instead_of_0_fps(ui):
    still = snap_with(time=time.time(), active_content=False, output={"renders_per_s": 0.0, "frames_window": 0,
                                                                      "late_window": 0, "last_frame_age_s": 4.0})
    ui.render_thread = FakeRT(still)
    ui._update_health_indicator()
    assert ui.fps_label.kw["text"].startswith("● OK   still picture   0 late   0 errors")


def test_indicator_reports_a_stuck_or_dead_render_thread(ui):
    ui.render_thread = FakeRT(snap_with(time=time.time() - 10))
    ui._update_health_indicator()
    assert "not responding for 10 s" in ui.fps_label.kw["text"] and ui.fps_label.kw["fg"] == ui.HEALTH_COLORS["error"]
    ui.render_thread = FakeRT(snap_with(time=time.time()), alive=False)
    ui._update_health_indicator()
    assert "render thread stopped" in ui.fps_label.kw["text"]


def test_report_is_json_and_decoder_rows_are_readable(ui):
    import json
    ui.render_thread = FakeRT(snap_with(time=time.time()))
    json.dumps(ui._health_report())
    row = ui._decoder_row({"file": "a.mp4", "instance": "@3:0:1", "live": True, "state": "playing",
                           "hw_engaged": False, "hw_requested": "d3d11va", "source_fps": 30.0, "decode_fps": 29.9,
                           "frames_decoded": 10, "frames_shown": 9, "frames_dropped": 0})
    assert row[:5] == ("a.mp4", "@3:0:1", "On output", "playing", "SW (asked d3d11va)")
    assert ui._decoder_row({"state": "held", "hw_engaged": True, "hw_requested": "cuda"})[2:5] == \
        ("Next cue", "held", "HW cuda")


def test_gc_stats_time_each_collection_and_count_full_ones():
    t = [100.0]
    g = health.GcStats(clock=lambda: t[0])
    for gen, dur in ((0, 0.001), (2, 0.030), (1, 0.004), (2, 0.012)):
        g.callback("start", {"generation": gen})
        t[0] += dur
        g.callback("stop", {"generation": gen})
        t[0] += 2.0
    g.callback("stop", {"generation": 2})            # a stop without its start (installed mid-collection)
    s = g.snapshot(t[0])
    assert s["full_total"] == 2 and s["collections_total"] == 4
    assert s["full_pause_max_ms"] == 30.0 and s["full_pause_last_ms"] == 12.0 and s["full_pause_total_ms"] == 42.0
    assert s["pause_max_ms_window"] == 12.0 and s["collections_window"] == 2 and s["full_window"] == 1  # last 5 s
    assert s["blocks"] > 1000 and s["frozen"] >= 0


def test_gc_stats_see_real_collections_and_can_be_removed():
    import gc
    g = health.GcStats().install()
    try:
        assert g.callback in gc.callbacks
        gc.collect()
        s = g.snapshot(health.now())
        assert s["full_total"] >= 1 and s["full_pause_last_ms"] is not None
    finally:
        g.remove()
    assert g.callback not in gc.callbacks
    g.remove()                                        # twice is harmless


def test_render_thread_reports_the_collector_and_removes_its_callback(fixtures, monkeypatch):
    import gc
    monkeypatch.setattr(ci_mod.RenderThread, "HEALTH_PERIOD", 0.1)
    eng = FlakyEngine(fail_on=())
    eng.player_stats = lambda live: []
    eng.source_key = rem.BaseRenderEngine.source_key
    cmd_q, prev_q, stop = queue.Queue(), queue.Queue(maxsize=1), threading.Event()
    rt = ci_mod.RenderThread(eng, cmd_q, prev_q, stop)
    rt.start()
    snap = wait_for(lambda: rt.health_snapshot, timeout=3)
    gc.collect()
    snap = wait_for(lambda: rt.health_snapshot["gc"]["full_total"] >= 1 and rt.health_snapshot, timeout=3)
    assert rt.gc_stats.callback in gc.callbacks
    stop.set()
    rt.join(2)
    assert snap["gc"]["blocks"] > 1000
    assert rt.gc_stats.callback not in gc.callbacks
    text = ci_mod.ControlInterface._gc_text(snap["gc"])
    assert text.startswith(f"{snap['gc']['blocks'] / 1000:.0f}k objects   longest pause") and "full collections" in text
    assert ci_mod.ControlInterface._gc_text({}) == "?"



def test_long_lived_objects_are_set_aside_after_startup(ui, caplog):
    import gc
    try:
        with caplog.at_level("INFO"):
            ui._freeze_gc("startup")
        assert gc.get_freeze_count() > 10_000
        assert "objects set aside after startup" in caplog.text
        gc.collect()
        assert gc.get_freeze_count() > 10_000, "a later collection leaves them aside"
    finally:
        gc.unfreeze()


def _frozen_rt(tmp_path, players):
    from types import SimpleNamespace
    import control_interface as ci_mod
    rt = ci_mod.RenderThread.__new__(ci_mod.RenderThread)
    rt.re = SimpleNamespace(active_players=players)
    rt._pending_cut = None
    rt.released = []
    rt._release_held = lambda keys: rt.released.extend(keys)
    return rt


def test_frozen_video_watchdog(tmp_path):
    """A video on the output whose picture stopped (owner's report 2026-11-14:
    a video with an effect froze until a restart). Held with no cut waiting: released
    after FROZEN_AFTER_S; decoding but not taken by the output, or no decoder: reported."""
    from types import SimpleNamespace
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    key = f"@a1|{clip}"
    held = SimpleNamespace(held=True)
    rt = _frozen_rt(tmp_path, {key: {"player": held}})
    dec = [{"key": key, "state": "held", "frames_shown": 0, "frames_decoded": 1}]
    assert rt._frozen_videos(0.0, {key}, dec) == []                    # just now: a cut may be on its way
    assert rt._frozen_videos(1.0, {key}, dec) == []
    out = rt._frozen_videos(2.5, {key}, dec)
    assert out == [{"file": "clip.mp4", "why": "held"}] and rt.released == [key]
    level, reasons = health.assess({"frozen": out})
    assert level == "warn" and "clip.mp4: picture not moving" in reasons[0]

    # Playing, and the output takes its frames: fine. Then it stops taking them.
    rt = _frozen_rt(tmp_path, {key: {"player": SimpleNamespace(held=False)}})
    for t, shown, decoded in ((0.0, 10, 10), (0.5, 20, 20), (1.0, 30, 30)):
        assert rt._frozen_videos(t, {key}, [{"key": key, "state": "playing", "frames_shown": shown,
                                             "frames_decoded": decoded}]) == []
    for t in (1.5, 2.0, 3.0):
        out = rt._frozen_videos(t, {key}, [{"key": key, "state": "playing", "frames_shown": 30,
                                            "frames_decoded": 30 + int(t * 30)}])
    assert out == [{"file": "clip.mp4", "why": "not taken"}] and rt.released == []
    # Paused (decodes nothing new) is not frozen.
    rt = _frozen_rt(tmp_path, {key: {"player": SimpleNamespace(held=False)}})
    for t in (0.0, 1.0, 2.0, 3.0, 4.0):
        assert rt._frozen_videos(t, {key}, [{"key": key, "state": "playing", "frames_shown": 5,
                                             "frames_decoded": 5}]) == []

    # No decoder at all for a video the output shows; a missing file is not this.
    rt = _frozen_rt(tmp_path, {})
    assert rt._frozen_videos(0.0, {key}, []) == []
    assert rt._frozen_videos(2.1, {key}, []) == [{"file": "clip.mp4", "why": "no decoder"}]
    gone = f"@a2|{tmp_path / 'gone.mp4'}"
    assert rt._frozen_videos(5.0, {gone}, []) == []
    # A cut waiting for its videos: never reported.
    rt._pending_cut = {"videos": {key}}
    assert rt._frozen_videos(9.0, {key}, []) == []
