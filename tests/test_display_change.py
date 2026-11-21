# SPDX-License-Identifier: MPL-2.0
"""Display hot-plug: detected and reported, never acted on without the operator."""
from types import SimpleNamespace

import control_interface as ci_mod
import health
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci, commands  # noqa: F401

A = SimpleNamespace(x=0, y=0, width=1920, height=1080)
B = SimpleNamespace(x=1920, y=0, width=1280, height=1024)


def test_change_is_reported_and_clears_when_restored(blind_ci, caplog):
    ci = blind_ci
    ci.re.monitors = [A, B]
    assert ci._check_displays([A, B]) is None
    commands(ci)                                                       # drain
    with caplog.at_level("WARNING"):
        ch = ci._check_displays([A])                                   # projector unplugged
    assert ch == {"engine": [(0, 0, 1920, 1080), (1920, 0, 1280, 1024)], "now": [(0, 0, 1920, 1080)]}
    assert "Displays changed since the output started: 2 -> 1" in caplog.text
    assert commands(ci) == [], "nothing is sent to the output"
    caplog.clear()
    ci._check_displays([A])
    assert "Displays changed" not in caplog.text, "logged once per change"
    level, reasons = health.assess({"output": {}, "displays": {"count": 2, "changed": ch}})
    assert level == "warn" and any("displays changed: 2 when the output started, 1 now" in r for r in reasons)
    assert ci._displays_text({"count": 2, "changed": ch}).startswith("CHANGED: 2 when")
    assert ci._check_displays([A, B]) is None and ci._displays_text({"count": 2}) == "2 as when the output started"


def test_rescan_only_after_the_operator_agrees(blind_ci, monkeypatch):
    ci = blind_ci
    calls = []
    ci._initialize_render_engine = lambda: calls.append("init")
    ci._trigger_column = lambda *a, **k: calls.append("trigger")
    ci._update_active_players = lambda *a, **k: calls.append("players")
    monkeypatch.setattr(ci_mod.messagebox, "askyesno", lambda *a, **k: False)
    assert not ci._rescan_displays() and calls == []
    monkeypatch.setattr(ci_mod.messagebox, "askyesno", lambda *a, **k: True)
    ci._display_change = {"engine": [], "now": []}
    assert ci._rescan_displays() and calls == ["init", "players", "trigger"] and ci._display_change is None


def test_enumeration_failure_keeps_the_last_state(blind_ci, monkeypatch):
    ci = blind_ci
    ci.re.monitors = [A]
    ci._display_change = None

    def broken():
        raise RuntimeError("no display server")
    monkeypatch.setattr(ci_mod, "get_monitors", broken)
    assert ci._check_displays() is None


def test_same_number_of_displays_in_other_units_is_not_a_change(blind_ci, caplog):
    ci = blind_ci
    ci.re.monitors = [A]
    scaled = SimpleNamespace(x=0, y=0, width=1280, height=720)        # e.g. 150 % scaling, DPI-unaware
    with caplog.at_level("INFO"):
        assert ci._check_displays([scaled]) is None
        ci._check_displays([scaled])
    assert caplog.text.count("Display geometry differs") == 1 and "Displays changed" not in caplog.text


def test_rebuilding_the_grid_does_not_mark_the_show_changed(_base_ci):
    """Opening a show from Recent Projects (and New Project) rebuilt the grid,
    which marked the show as changed: closing it then asked to save a show
    nobody had touched (found on the owner's PC, 2026-11-14)."""
    import inspect
    ci = _base_ci
    ci._ui_ready, ci._startup_inhibit_triggers = True, False
    ci._update_cell_preview = lambda r, c: None
    ci.layer_setting_widgets = {}
    ci.project_is_dirty = False
    ci._on_output_change(mark_dirty=False)
    assert ci.project_is_dirty is False
    ci._on_output_change()                       # the operator chose another output: a change
    assert ci.project_is_dirty is True
    assert "_on_output_change(mark_dirty=False)" in inspect.getsource(ci_mod.ControlInterface._build_grid)
