# SPDX-License-Identifier: MPL-2.0
"""Per-instance video players: one decoder per (cue activation, cell, file).

Replaces the old model where players were keyed by file path, which forced
fades between cues sharing a file to copy or hard-link the media file.
"""
import time

import pytest

import render_engine as rem


def vlayer(path, act, row=0, col=0, hold=False, loop=True):
    layer = {"path": path, "order": row, "position": {}, "loop": loop,
             "instance": f"@{act}:{row}:{col}|{path}"}
    if hold:
        layer["hold"] = True
    return layer


@pytest.fixture
def engine():
    eng = rem.BaseRenderEngine(media_manager=None, mode="opengl")
    yield eng
    eng.stop_players(list(eng.active_players))


def wait_for(pred, timeout=3.0, step=0.005):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = pred()
        if value:
            return value
        time.sleep(step)
    return None


def latest_ts(eng, key):
    frame = eng._get_latest_video_frame(key)
    return None if frame is None else frame["timestamp"]


def test_instance_key_keeps_file_extension(fixtures):
    key = rem.BaseRenderEngine.source_key(vlayer(fixtures["counting"], 3, 1, 2))
    assert key.startswith("@3:1:2|") and key.endswith(".mp4")
    assert rem.os.path.splitext(key)[1] == ".mp4"
    assert rem.BaseRenderEngine.source_key({"path": "/x/a.png"}) == "/x/a.png"


def test_same_file_in_two_instances_gets_two_independent_players(engine, fixtures):
    a, b = vlayer(fixtures["counting"], 1, row=0), vlayer(fixtures["counting"], 2, row=1)
    ka, kb = engine.source_key(a), engine.source_key(b)
    engine.update_layers({0: a, 1: b})
    assert set(engine.active_players) == {ka, kb}
    pa = engine.active_players[ka]["player"]
    assert pa is not engine.active_players[kb]["player"]

    time.sleep(0.6)
    engine.update_layers({0: a, 1: b}, force_restart_paths={kb})  # restart only b
    assert engine.active_players[ka]["player"] is pa, "restarting b must not touch a"
    assert wait_for(lambda: latest_ts(engine, kb) is not None)
    ts_a, ts_b = latest_ts(engine, ka), latest_ts(engine, kb)
    assert ts_a > 0.3 and ts_b < ts_a - 0.25, (ts_a, ts_b)


def test_held_player_waits_on_frame_zero_then_plays_from_frame_one(engine, fixtures):
    held = vlayer(fixtures["counting"], 5, hold=True)
    key = engine.source_key(held)
    engine.update_layers({0: held})
    player = engine.active_players[key]["player"]
    assert player.held
    assert wait_for(lambda: engine.has_frame(key))
    time.sleep(0.4)  # a running player would have produced ~12 more frames
    first = engine._get_latest_video_frame(key)
    assert first["timestamp"] == 0.0
    time.sleep(0.2)
    assert engine._get_latest_video_frame(key) is first, "held player produced more frames"

    engine.update_layers({0: vlayer(fixtures["counting"], 5)})  # same key, live -> release
    assert engine.active_players[key]["player"] is player and not player.held
    nxt = wait_for(lambda: (latest_ts(engine, key) or 0) > 0 and latest_ts(engine, key))
    assert nxt == pytest.approx(1 / 30, abs=1e-3), "playback must continue at frame 1 after release"
    time.sleep(0.4)
    assert 0.25 <= latest_ts(engine, key) <= 0.7, "timing must restart at release, not at preload"


def test_live_use_wins_over_preload_of_the_same_key(engine, fixtures):
    live, pre = vlayer(fixtures["counting"], 7), vlayer(fixtures["counting"], 7, hold=True)
    engine.update_layers({"pre": pre, "live": live})
    assert not engine.active_players[engine.source_key(live)]["player"].held


def test_loop_setting_reaches_running_player(engine, fixtures):
    layer = vlayer(fixtures["counting"], 8)
    engine.update_layers({0: layer})
    engine.update_layers({0: {**layer, "loop": False}})
    assert engine.active_players[engine.source_key(layer)]["player"].loop is False


def test_stopping_a_held_player_ends_its_thread(engine, fixtures):
    held = vlayer(fixtures["counting"], 9, hold=True)
    key = engine.source_key(held)
    engine.update_layers({0: held})
    player = engine.active_players[key]["player"]
    assert wait_for(lambda: engine.has_frame(key))
    engine.stop_players([key])
    player.join(2)
    assert not player.is_alive() and key not in engine.active_players


def test_cpu_mode_player_also_holds(fixtures):
    eng = rem.BaseRenderEngine(media_manager=None, mode="cpu")
    held = vlayer(fixtures["counting"], 1, hold=True)
    key = eng.source_key(held)
    try:
        eng.update_layers({0: held})
        assert wait_for(lambda: eng.has_frame(key))
        time.sleep(0.4)
        frame0 = eng._get_latest_video_frame(key)
        assert frame0.mean() < 8, "frame 0 of the counting clip is black"
        time.sleep(0.2)
        assert eng._get_latest_video_frame(key) is frame0
        eng.update_layers({0: vlayer(fixtures["counting"], 1)})
        time.sleep(0.5)
        assert eng._get_latest_video_frame(key).mean() > 15
    finally:
        eng.stop_players(list(eng.active_players))


def test_cpu_mode_keeps_a_clips_transparency(fixtures):
    """Owner's PC, 2026-11-14: HAP Alpha was drawn opaque in Compatibility Mode,
    whose OpenCV reader drops alpha. Clips with alpha are read with PyAV there."""
    assert rem.video_has_alpha(fixtures["alpha_video"]) is True
    assert rem.video_has_alpha(fixtures["counting"]) is False and rem.video_has_alpha("/no/such.mov") is False
    eng = rem.BaseRenderEngine(media_manager=None, mode="cpu")
    layer = vlayer(fixtures["alpha_video"], 1)
    key = eng.source_key(layer)
    try:
        eng.update_layers({0: layer})
        assert not isinstance(eng.active_players[key]["player"], rem.CPUVideoPlayerThread)
        assert wait_for(lambda: eng.has_frame(key))
        frame = eng._get_latest_video_frame(key)
        assert frame.ndim == 3 and frame.shape[2] == 4              # BGRA
        h, w = frame.shape[:2]
        assert frame[h // 2, w // 4, 3] == 255 and frame[h // 2, 3 * w // 4, 3] == 0   # left opaque, right clear
        assert frame[h // 2, w // 4, 2] > 200                      # red (BGRA order)
    finally:
        eng.stop_players(list(eng.active_players))
    plain = rem.BaseRenderEngine(media_manager=None, mode="cpu")
    layer = vlayer(fixtures["counting"], 1)
    try:
        plain.update_layers({0: layer})
        p = plain.active_players[plain.source_key(layer)]["player"]       # PyAV for every video (colors)
        assert isinstance(p, rem.GPUVideoPlayerThread) and p.output_format == "numpy"
    finally:
        plain.stop_players(list(plain.active_players))
