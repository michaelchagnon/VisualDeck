# SPDX-License-Identifier: MPL-2.0
"""Playback speed: decoder pacing, live changes, follow timing, health check."""
import time

import pytest

import health
import render_engine as rem
from test_player_instances import vlayer, wait_for
from test_follow import ci as follow_ci, last_follow  # noqa: F401
from test_blind_editing import ci as blind_ci  # noqa: F401  (follow_ci needs it)
from test_cue_activation import ci as _base_ci  # noqa: F401  (blind_ci needs it)


@pytest.fixture(params=["opengl", "cpu"])
def engine(request):
    eng = rem.BaseRenderEngine(media_manager=None, mode=request.param)
    yield eng
    eng.stop_players(list(eng.active_players))


def decoded_in(engine, key, seconds):
    p = engine.active_players[key]["player"]
    q = engine.active_players[key]["queue"]
    start, t0 = p.frames_decoded, time.monotonic()
    while time.monotonic() - t0 < seconds:
        q.get_latest()          # consume like the renderer, so the decoder is never blocked
        time.sleep(0.005)
    return p.frames_decoded - start


def test_speed_changes_the_decode_rate_and_can_change_live(engine, fixtures):
    fast = dict(vlayer(fixtures["counting"], 1), speed=2.0)
    normal = vlayer(fixtures["counting"], 2, row=1)
    engine.update_layers({0: fast, 1: normal})
    assert wait_for(lambda: all(engine.active_players[l["instance"]]["player"].frames_decoded > 3
                                for l in (fast, normal)))
    n_fast = decoded_in(engine, fast["instance"], 1.0)
    n_norm = decoded_in(engine, normal["instance"], 1.0)
    assert 48 <= n_fast <= 66 and 24 <= n_norm <= 34, (n_fast, n_norm)
    engine.update_layers({0: dict(fast, speed=0.5), 1: normal})          # live change, same player
    time.sleep(0.2)
    n_slow = decoded_in(engine, fast["instance"], 1.0)
    assert 11 <= n_slow <= 19, n_slow
    st = health.player_status(fast["instance"], engine.active_players[fast["instance"]])
    assert st["speed"] == 0.5 and not health.decoder_behind(dict(st, decode_fps=15.0))


@pytest.mark.parametrize("raw,expected", [(2, 2.0), ("x", 1.0), (0, 0.1), (100, 8.0), (float("nan"), 1.0)])
def test_speed_values_are_clamped(raw, expected, fixtures):
    p = rem.GPUVideoPlayerThread(fixtures["counting"], rem.FastRingBuffer(4), None, True)
    p.set_speed(raw)
    assert p.speed == expected


def test_follow_at_video_end_accounts_for_speed(follow_ci, fixtures):
    ci = follow_ci
    ci.cue_settings = {"0": {"follow": {"mode": "end"}}}
    ci.cell_positions[(0, 0)] = {"speed": 2.0}                           # 2 s clip at 2x = 1 s
    ci._cut_to_column(0)
    last_follow(ci)
    assert abs(ci._follow_due[0] - __import__("time").monotonic() - 1.0) <= 0.03   # the computed end
    assert ci._get_layers_for_col(0)[0]["speed"] == 2.0


def test_editor_speed_control_saves_and_omits_normal_speed(fixtures):
    import tkinter as tk
    from types import SimpleNamespace
    import editor
    import ui_theme
    root = tk.Tk()
    root.withdraw()
    ui_theme.apply(root)
    try:
        saved, live = [], []
        ed = editor.MediaEditor(SimpleNamespace(root=root), fixtures["counting"], None, SimpleNamespace(monitors=[]),
                                initial_pos={"x": 0.5, "y": 0.5, "scale_x": 1, "scale_y": 1, "rotation": 0},
                                callback=lambda p, pos: saved.append(dict(pos)),
                                realtime_callback=lambda pos: live.append(dict(pos)))
        ed.speed_var.set("2×")
        ed._on_speed_change()
        assert ed.current_pos["speed"] == 2.0
        ed.speed_var.set("1×")
        ed._on_speed_change()
        assert "speed" not in ed.current_pos
        ed.speed_var.set("0.5×")
        ed._on_speed_change()
        ed._save_and_close()
        assert saved[-1]["speed"] == 0.5
        bad = editor.MediaEditor(SimpleNamespace(root=root), fixtures["counting"], None, SimpleNamespace(monitors=[]),
                                 initial_pos={"speed": "fast"})
        assert bad.speed_var.get() == "1\u00d7"
        bad._cancel_and_close()
    finally:
        root.destroy()


def test_health_row_shows_speed():
    import control_interface as ci_mod
    row = ci_mod.ControlInterface._decoder_row({"file": "a.mp4", "source_fps": 30.0, "decode_fps": 59.8, "speed": 2.0})
    assert row[5] == "30.0 x2" and row[6] == "59.8"
    assert ci_mod.ControlInterface._decoder_row({"source_fps": 30.0, "speed": 1.0})[5] == "30.0"
