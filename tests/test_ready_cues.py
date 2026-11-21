# SPDX-License-Identifier: MPL-2.0
"""Cues kept ready besides the next and previous: Keep Ready, the next timecode cue, a scheduled cue."""
import datetime

import project_schema
from test_cue_activation import ci, commands  # noqa: F401


def ready_cols(ci):
    (cmd,) = commands(ci, "update_layers")
    return sorted({int(k.split(":")[1]) for k in cmd["layers"] if k.startswith("ready:")}), cmd


def add_cues(ci, fixtures, cols):
    ci.COLS = 16
    for c in cols:
        ci.grid_cells[(0, c)] = fixtures["counting"]


def test_keep_ready_cues_are_preloaded_held_and_claimed_on_a_jump(ci, fixtures):
    add_cues(ci, fixtures, (5, 6))
    ci.cue_settings = {"5": {"ready": True}, "6": {"ready": True}}
    ci._update_active_players()
    cols, cmd = ready_cols(ci)
    assert cols == [5, 6]
    held = [l for k, l in cmd["layers"].items() if k.startswith("ready:")]
    assert all(l["hold"] is True and "instance" in l for l in held)
    act5 = ci._reserved_activation[5]
    assert set(ci._reserved_activation) == {1, 5, 6}                  # next (1) as well
    ci._cut_to_column(5)                                              # a jump: the preloaded players go live
    assert ci._live_activation == {5: act5}
    (cmd,) = commands(ci, "render")
    assert all(l["instance"].startswith(f"@{act5}:") for l in cmd["layers"].values() if "instance" in l)


def test_at_most_max_ready_and_never_the_live_cue_or_an_empty_one(ci, fixtures):
    add_cues(ci, fixtures, range(3, 10))
    ci.cue_settings = {str(c): {"ready": True} for c in [0] + list(range(3, 10)) + [15]}   # 15 is empty
    assert ci._ready_cols() == [3, 4, 5, 6] and len(ci._ready_cols()) == ci.MAX_READY


def test_the_next_timecode_cue_is_kept_ready_and_moves_on(ci, fixtures):
    add_cues(ci, fixtures, (4, 7))
    ci.cue_settings = {"4": {"timecode": "01:00:10:00"}, "7": {"timecode": "01:00:20:00"},
                       "2": {"timecode": "01:00:05:00"}}
    ci.midi_timecode = True
    assert ci._ready_cols() == [2], "before any time: the first timecode cue"
    ci._ui_ready = True
    ci._update_active_players()
    commands(ci)
    ci._timecode_tick((1, 0, 6, 0, 1))                                # 01:00:06:00, 25 fps: past cue 2's time
    assert ci._ready_cols() == [4]
    cols, _ = ready_cols(ci)
    assert cols == [4], "preloaded again when the next timecode cue changed"
    ci._timecode_tick((1, 0, 6, 5, 1))
    assert commands(ci, "update_layers") == [], "no new preload while it stays the same"
    ci.midi_timecode = False
    assert ci._ready_cols() == []


def test_a_scheduled_cue_due_within_a_minute_is_kept_ready(ci, fixtures):
    add_cues(ci, fixtures, (8,))
    ci.schedule = {"enabled": True, "events": [{"time": "20:30", "days": list(range(7)), "action": "cue",
                                                 "cue": "9"}]}
    assert ci._upcoming_schedule_col(datetime.datetime(2026, 11, 13, 20, 29, 30)) == 8
    assert ci._upcoming_schedule_col(datetime.datetime(2026, 11, 13, 20, 28, 0)) is None
    ci.schedule["enabled"] = False
    assert ci._upcoming_schedule_col(datetime.datetime(2026, 11, 13, 20, 29, 30)) is None


def test_keep_ready_is_saved_and_cleaned():
    assert project_schema.clean_cue_settings({"3": {"ready": True}, "4": {"ready": "yes"}, "5": {"ready": False}}) == \
        {"3": {"ready": True}}
    import control_interface as ci_mod
    t = ["Show default", "None (cut)", "Fade"]
    f = ["Wait for GO", "after", "end"]
    assert ci_mod.ControlInterface._cue_settings_from_form(t[0], t, "", f[0], f, "5", "", ready=True) == \
        {"ready": True}


def test_keep_ready_from_cue_settings(ci, fixtures):
    import os
    import pytest
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import ui_theme
    add_cues(ci, fixtures, (5,))
    root = tk.Tk()
    root.withdraw()
    ui_theme.apply(root)
    ci.root, ci._ui_ready = root, True
    ci._highlight_live_column = lambda: None
    try:
        win = ci._open_cue_settings(5)
        win._vd["ready_var"].set(True)
        win._vd["ok"]()
        assert ci.cue_settings["5"] == {"ready": True}
        cols, _ = ready_cols(ci)
        assert cols == [5], "kept ready as soon as it is set"
    finally:
        root.destroy()
