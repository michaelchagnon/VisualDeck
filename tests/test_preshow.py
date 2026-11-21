# SPDX-License-Identifier: MPL-2.0
"""Preshow Check: the rows an operator reads before the house opens."""
import os

import pytest

import preshow
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401

GOOD = {"output": "Display 0", "output_size": (1920, 1080), "render_mode": "opengl", "cues": 3, "effects": 0,
        "blends": 0, "dirty": False, "path": "/shows/a.vjdeck", "locked": True, "test_pattern": False,
        "disk_free": 50 * 1024 ** 3, "presentation": False, "schedule": ""}
SNAP = {"render_thread_alive": True, "errors_recent": {}, "output": {"frames_window": 300, "late_window": 0},
        "missing_media": 0, "decoders": [], "effects": {"count": 0}, "displays": {"count": 1}}


def levels(rows):
    return {t: l for l, t, _d in rows}


def test_a_ready_show():
    rows = preshow.checks(SNAP, GOOD)
    assert preshow.summary(rows) == ("ok", "Ready for the show")
    t = levels(rows)
    assert t["Output: Display 0"] == "ok" and t["All media files found"] == "ok" and t["Editing locked"] == "ok"


def test_problems_come_first_and_are_counted():
    snap = dict(SNAP, missing_media=2, missing_media_names=["Intro.mov"], errors_recent={"render": 1}, last_error={"message": "GL error"},
                decoders=[{"file": "Camera video=LifeCam", "state": "retrying", "live": True, "source": "live",
                           "last_error": "cannot open: I/O error"}],
                remote={"enabled": True, "listening": False, "error": "port 7700 in use"},
                midi={"enabled": True, "listening": True, "port": "loopMIDI Port"},
                audio={"enabled": True, "running": False, "error": "no device"},
                effects={"count": 1, "errors": {"Bad.fs": "syntax error"}},
                displays={"count": 1, "changed": {"engine": [1, 2], "now": [1]}},
                active_content=True, output={"frames_window": 300, "late_window": 30}, process={"gdi": 6000})
    show = dict(GOOD, output="None", dirty=True, locked=False, test_pattern=True, effects=2, blends=1,
                render_mode="pygame_gpu", disk_free=100 * 1024 ** 2)
    rows = preshow.checks(snap, show)
    t = levels(rows)
    assert t["No output"] == "fail" and t["2 media files missing"] == "fail"
    detail = next(r for r in rows if r[1] == "2 media files missing")[2]
    assert detail.startswith("Intro.mov and 1 more. File › Relink Missing Media")    # owner's PC: name the file
    assert t["1 output error in the last minute"] == "fail" and t["Camera video=LifeCam: retrying"] == "fail"
    assert t["OSC is on but not listening"] == "fail" and t["MIDI listening"] == "ok"
    assert t["Sound output not open"] == "fail" and t["Effect Bad.fs cannot be drawn"] == "fail"
    assert t["Only 0.1 GB free on the show's drive"] == "fail"
    for title in ("Displays changed since the output started", "30 late frames in the last 5 s",
                  "2 shader effects not drawn", "Blend modes drawn as Normal", "A test pattern is on the output",
                  "Unsaved changes", "Editing is not locked", "6000 GDI objects"):
        assert t[title] == "warn", title
    lv = [l for l, _t, _d in rows]
    assert lv == sorted(lv, key={"fail": 0, "warn": 1, "ok": 2}.get)
    level, words = preshow.summary(rows)
    assert level == "fail" and words.startswith("8 problems to fix, 8 things to look at")
    assert "[PROBLEM] No output" in preshow.as_text(rows)


def test_untitled_show_and_no_snapshot():
    rows = preshow.checks(None, dict(GOOD, path=None))
    t = levels(rows)
    assert t["The show has not been saved"] == "warn" and t["The output has not reported yet"] == "warn"


def test_the_window_lists_the_checks(blind_ci, tmp_path):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import ui_theme
    root = tk.Tk()
    root.withdraw()
    ui_theme.apply(root)
    ci = blind_ci
    ci.root = root
    ci._current_health = lambda: dict(SNAP)
    ci.current_project_path = None
    ci.edit_locked = False
    try:
        win = ci._open_preshow()
        rows = win._vd_state["rows"]
        assert any(t == "The show has not been saved" for _l, t, _d in rows)
        assert any(t.startswith("Output: Display 0") for _l, t, _d in rows)
        ci.edit_locked = True
        win2 = ci._open_preshow()                     # a second open refreshes the same window
        assert win2 is win and any(t == "Editing locked" for _l, t, _d in win._vd_state["rows"])
    finally:
        root.destroy()
