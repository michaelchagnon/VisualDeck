# SPDX-License-Identifier: MPL-2.0
"""Output presets: cleaning, display mismatch warnings, the file (atomic, a
bad file moved aside, never overwritten), applying to the app, and the dialog."""
import json
import os
from types import SimpleNamespace

import pytest

import output_presets as op
from test_blind_editing import ci as blind_ci, renders  # noqa: F401
from test_cue_activation import ci as _base_ci, commands  # noqa: F401
from test_control_state import Var

PIN = [[0.05, 0.0], [1.0, 0.02], [0.97, 1.0], [0.0, 0.95]]
BLEND = {"left": 0.0, "right": 0.1, "top": 0.0, "bottom": 0.0, "curve": 2.0, "gamma": 2.2}


def mons(*sizes):
    return [SimpleNamespace(width=w, height=h) for w, h in sizes]


def test_clean_drops_bad_parts_and_rejects_non_presets():
    assert op.clean(None) is None and op.clean({"mode": "Display x"}) is None and op.clean({"mode": 3}) is None
    p = op.clean({"mode": "Multiple", "layer_screens": {"0": 1, "x": 2, "3": True, "4": -1, "5": "0"},
                  "warps": {"1": PIN, "2": [[0, 0], [1, 0], [0, 1], [1, 1]], "q": PIN},     # folded quad, bad key
                  "blends": {"0": BLEND, "1": {"left": 0}}, "displays": [[1920, 1080], ["a"], [0, 5]]})
    assert p["layer_screens"] == {"0": 1, "5": 0}
    assert list(p["warps"]) == ["1"] and list(p["blends"]) == ["0"] and p["displays"] == [[1920, 1080]]
    assert p["blends"]["0"] == BLEND
    assert op.clean(p) == p, "cleaning is stable"


def test_names_mismatches_and_summary():
    assert op.clean_name("  Venue   A  ") == "Venue A" and op.clean_name("   ") is None
    assert len(op.clean_name("x" * 99)) == op.MAX_NAME
    p = op.capture("Display 1", {}, {"1": PIN}, {"0": BLEND, "1": BLEND}, mons((1920, 1080), (1280, 800)))
    assert op.summary(p) == "Display 1 · pin on 1 · blend on 0, 1"
    assert op.mismatches(p, mons((1920, 1080), (1280, 800))) == []
    probs = op.mismatches(p, mons((1920, 1080)))
    assert any("Display 1, which this computer does not have (1 display)" in x for x in probs)
    assert any("saved with 2 displays, this computer has 1" in x for x in probs)
    probs = op.mismatches(p, mons((1920, 1080), (1920, 1200)))
    assert probs == ["Display 1 was 1280×800, it is 1920×1200 now (corner pins, blends and meshes are relative, "
                     "so they stretch)"]
    assert op.summary(op.clean({"mode": "Multiple"})) == "Multiple Displays"


def test_file_round_trip_and_bad_file_is_moved_aside(tmp_path):
    path = str(tmp_path / "vd" / op.FILE_NAME)
    assert op.load(path) == ({}, None, True)
    p = op.capture("Display 0", {}, {"0": PIN}, {}, mons((1920, 1080)))
    op.save(path, {"Main": p})
    assert op.load(path) == ({"Main": p}, None, True)
    assert json.load(open(path))["version"] == 1
    with open(path, "w") as f:
        f.write("{not json")
    presets, note, writable = op.load(path)
    assert presets == {} and writable and "moved to" in note
    assert open(path + ".bad").read() == "{not json" and not os.path.exists(path), "the bad file is kept"
    with open(path, "w") as f:
        json.dump({"version": 1, "presets": {"ok": p, "broken": {"mode": 7}, "": p}}, f)
    assert op.load(path)[0] == {"ok": p}


def test_unmovable_bad_file_blocks_saving(tmp_path, monkeypatch):
    path = str(tmp_path / op.FILE_NAME)
    with open(path, "w") as f:
        f.write("[]")
    monkeypatch.setattr(op.os, "replace", lambda *a: (_ for _ in ()).throw(PermissionError("locked")))
    presets, note, writable = op.load(path)
    assert presets == {} and not writable and "cannot be saved" in note


def _tracing(ci):
    """The test fixture's screen_var has no Tk trace; call the handler like the trace does."""
    class TVar(Var):
        def set(self, v):
            super().set(v)
            ci._on_output_change()
    ci.screen_var = TVar(ci.screen_var.get())
    ci.layer_setting_widgets = {}                 # the per-layer display buttons (not built here)


def test_apply_changes_show_and_output_settings(blind_ci):
    ci = blind_ci
    _tracing(ci)
    ci.re.monitors = mons((1920, 1080), (1280, 800))
    ci.output_warps = {"0": PIN}
    ci.output_blends = {}
    dirty = []
    ci._set_dirty_flag = lambda *a: dirty.append(1)
    preset = op.clean({"mode": "Multiple", "layer_screens": {"0": 1, "1": 0}, "warps": {},
                       "blends": {"1": BLEND}, "displays": [[1920, 1080], [1280, 800]]})
    commands(ci)
    assert ci._apply_output_preset(preset) == "Multiple"
    assert ci.screen_var.get() == "Multiple" and ci.selected_display is None
    assert ci.layer_screens == {"0": 1, "1": 0} and ci.output_warps == {} and ci.output_blends == {"1": BLEND}
    cmds = commands(ci)
    assert {"set_warps", "set_blends", "render"} <= {c["action"] for c in cmds}
    assert [c for c in cmds if c["action"] == "set_warps"][-1]["warps"] == {}
    assert dirty, "the show has changed"
    ci.layer_screens["0"] = 0
    assert preset["layer_screens"]["0"] == 1, "the show must not share dicts with the preset"
    # A preset for a display this computer lacks ends on no output (as choosing it would).
    ended = ci._apply_output_preset(op.clean({"mode": "Display 3"}))
    assert ended == "None" and ci.selected_display is None


def test_current_setup_is_captured(blind_ci):
    ci = blind_ci
    ci.re.monitors = mons((1920, 1080))
    ci.output_warps, ci.output_blends, ci.layer_screens = {"0": PIN}, {"0": BLEND}, {"2": 0}
    p = ci._current_output_preset()
    assert p == {"mode": "Display 0", "layer_screens": {"2": 0}, "warps": {"0": PIN}, "blends": {"0": BLEND},
                 "meshes": {}, "span": None, "displays": [[1920, 1080]]}


def test_dialog_save_apply_delete(blind_ci, monkeypatch):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import control_interface as ci_mod
    ci = blind_ci
    _tracing(ci)
    ci.re.monitors = mons((1920, 1080), (1280, 800))
    answers = []
    monkeypatch.setattr(ci_mod.messagebox, "askyesno", lambda *a, **k: answers.pop(0))
    root = tk.Tk()
    root.withdraw()
    try:
        ci.root = root
        ci.output_warps, ci.output_blends = {"0": PIN}, {}
        win = ci._open_output_presets()
        vd = win._vd
        vd["save"]()
        assert "Type a name" in vd["msg"].cget("text")
        vd["name_var"].set("Main room")
        vd["save"]()
        assert list(vd["st"]["presets"]) == ["Main room"] and vd["list"].size() == 1
        saved = json.load(open(ci._output_presets_path()))["presets"]["Main room"]
        assert saved["warps"] == {"0": PIN} and saved["displays"] == [[1920, 1080], [1280, 800]]
        # Saving under the same name asks first; No keeps the old one.
        ci.output_warps = {}
        answers.append(False)
        vd["save"]()
        assert vd["st"]["presets"]["Main room"]["warps"] == {"0": PIN}
        # Apply asks; No changes nothing, Yes applies.
        vd["list"].selection_set(0)
        vd["select"]()
        assert vd["name_var"].get() == "Main room"
        answers.append(False)
        vd["apply"]()
        assert ci.output_warps == {}
        answers.append(True)
        vd["apply"]()
        assert ci.output_warps == {"0": PIN} and "Applied" in vd["msg"].cget("text")
        answers.append(True)
        vd["delete"]()
        assert vd["st"]["presets"] == {} and json.load(open(ci._output_presets_path()))["presets"] == {}
        win.destroy()
    finally:
        root.destroy()
