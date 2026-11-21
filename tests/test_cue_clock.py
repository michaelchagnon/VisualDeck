# SPDX-License-Identifier: MPL-2.0
"""The cue clock: elapsed time of the live cue against its longest video."""
import pytest

import control_interface as ci_mod
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401

text = ci_mod.ControlInterface._clock_text


def test_text_for_stills_videos_and_loops():
    assert text(None) == ""
    assert text((0, 12.4, 0.0, False)) == "Cue 00:12"
    assert text((0, 12.4, 90.0, False)) == "Video 00:12 / 01:30  -01:18"
    assert text((0, 200.0, 90.0, False)) == "Video 01:30 / 01:30  -00:00"
    assert text((0, 100.0, 90.0, True)) == "Video 00:10 / 01:30  loop 2"
    assert text((0, 3725.0, 0.0, False)) == "Cue 1:02:05"


def test_clock_starts_when_a_cue_goes_live_and_follows_speed(blind_ci, monkeypatch):
    ci = blind_ci
    now = [1000.0]
    monkeypatch.setattr(ci_mod.time, "monotonic", lambda: now[0])
    ci.transition_var.set("None")
    ci.cell_positions[(0, 0)] = {"speed": 2.0, "loop": False}        # 2 s clip at 2x
    ci._cut_to_column(0)
    now[0] += 0.4
    col, elapsed, length, loops = ci._cue_clock()
    assert (col, round(elapsed, 3), length, loops) == (0, 0.4, 1.0, False)
    now[0] += 5
    ci._cut_to_column(1)                                             # new cue: clock restarts
    assert ci._cue_clock()[1] == 0.0 and ci._cue_clock()[3] is True  # looping by default
    assert ci._build_web_status()["clock"].startswith("Video 00:00 / 00:02")


def test_no_clock_while_black(blind_ci):
    ci = blind_ci
    ci.transition_var.set("None")
    ci._cut_to_column(0)
    assert ci._cue_clock() is not None
    ci._blackout()
    assert ci._cue_clock() is None and ci._build_web_status()["clock"] == ""


def test_a_missing_video_sets_no_length(blind_ci, monkeypatch, tmp_path):
    """Its remembered length (a 0.7 s clip) made the clock read "Video 00:00 / 00:00
    loop N" for a layer that is not drawn (owner's PC, 2026-11-14, checklist I3)."""
    ci = blind_ci
    monkeypatch.setattr(ci.mm, "get_media_info", lambda p: (1000, 9.0, 1920, 1080))   # a stale cache
    ci.grid_cells[(0, 3)] = str(tmp_path / "deleted.mov")
    assert ci._cue_video_length(3) == (0.0, False)
    there = tmp_path / "there.mov"
    there.write_bytes(b"x")
    assert ci._cue_video_duration(3) == 0.0                     # a follow "when its videos end" waits for GO
    ci.grid_cells[(1, 3)] = str(there)
    assert ci._cue_video_length(3) == (9.0, True) and ci._cue_video_duration(3) == 9.0
