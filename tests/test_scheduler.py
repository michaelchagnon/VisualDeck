# SPDX-License-Identifier: MPL-2.0
"""Show schedule: validation, when events run (once, in their minute, never
caught up), the app wiring, save/load and the dialog."""
import datetime as dt
import json
import os

import pytest

import scheduler as sch
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401

THU = dt.datetime(2026, 11, 12, 17, 59, 30)          # a Thursday


@pytest.mark.parametrize("raw,ok", [
    ({"time": "18:00", "action": "go"}, True), ({"time": "7:05", "action": "blackout", "days": [5, 6]}, True),
    ({"time": "18:00", "action": "cue", "cue": "12.5"}, True), ({"time": "18:00", "action": "cue"}, False),
    ({"time": "24:00", "action": "go"}, False), ({"time": "xx", "action": "go"}, False),
    ({"time": "18:00", "action": "explode"}, False), ({"time": "18:00", "action": "go", "days": [9]}, False),
])
def test_events_are_validated(raw, ok):
    assert (sch.clean_event(raw) is not None) == ok
    if ok and raw["time"] == "7:05":
        assert sch.clean_event(raw)["time"] == "07:05"


def test_due_once_in_its_minute_and_never_caught_up():
    events = sch.clean({"events": [{"time": "18:00", "action": "go", "days": sch.DAY_PRESETS["Weekdays"]},
                                   {"time": "18:00", "action": "go", "days": [5]}]})["events"]
    fired = set()
    assert sch.due(events, THU, fired) == []
    at = THU.replace(hour=18, minute=0, second=1)
    assert len(sch.due(events, at, fired)) == 1, "only the weekday event"
    assert sch.due(events, at.replace(second=40), fired) == [], "once per day"
    assert sch.due(events, at.replace(minute=1), fired) == [], "a missed minute is not caught up"
    assert len(sch.due(events, at + dt.timedelta(days=1), fired)) == 1, "Friday again"
    assert len(sch.due(events, at + dt.timedelta(days=2), set())) == 1, "Saturday: the Saturday event"


def test_next_event():
    events = sch.clean({"events": [{"time": "18:00", "action": "go"},
                                   {"time": "09:00", "action": "cue", "cue": "3", "days": [0]}]})["events"]
    when, ev = sch.next_event(events, THU)
    assert when == THU.replace(hour=18, minute=0, second=0) and ev["action"] == "go"
    when, ev = sch.next_event(events, THU.replace(hour=18, minute=0, second=30))
    assert when == dt.datetime(2026, 11, 13, 18, 0), "the event of the current minute is not 'next'"
    assert sch.next_event([], THU) is None
    assert sch.event_text(events[0]) == "09:00  Mon  cue 3"


def test_app_runs_events_only_when_switched_on(blind_ci):
    ci = blind_ci
    calls = []
    for name in ("_trigger_go", "_blackout"):
        setattr(ci, name, (lambda n: lambda *a, **k: calls.append(n))(name))
    ci.schedule = sch.clean({"enabled": False, "events": [{"time": "18:00", "action": "go"},
                                                          {"time": "18:00", "action": "cue", "cue": "2"},
                                                          {"time": "18:05", "action": "blackout"}]})
    at = THU.replace(hour=18, minute=0, second=2)
    assert ci._schedule_check(at) == [] and calls == []
    ci.schedule["enabled"] = True
    assert len(ci._schedule_check(at)) == 2
    assert calls == ["_trigger_go"] and ci.transition_thread.started[-1][3] == 1
    assert ci._schedule_check(at.replace(second=50)) == []
    ci._schedule_check(at.replace(minute=5))
    assert calls == ["_trigger_go", "_blackout"]
    assert ci._schedule_text(THU) == "Schedule: next 18:00 GO (in 1 min)"
    assert ci._schedule_text(THU.replace(hour=16)).endswith("(in 1 h 01 min)")
    assert ci._schedule_text(THU.replace(hour=15, second=0)).endswith("(in 2 h 01 min)")


def test_saved_with_the_show(blind_ci, tmp_path):
    ci = blind_ci
    for name in ("_build_media_list", "_build_grid", "_initialize_render_engine", "_on_toggle_presentation_mode"):
        setattr(ci, name, lambda *a, **k: None)
    show = str(tmp_path / "s.vjdeck")
    ci._perform_save(show)
    assert "schedule" not in json.load(open(show)), "nothing written without a schedule"
    ci.schedule = sch.clean({"enabled": True, "events": [{"time": "10:00", "action": "go"}]})
    show2 = str(tmp_path / "s2.vjdeck")
    ci._perform_save(show2)
    data = json.load(open(show2))
    data["schedule"]["events"].append({"time": "nope", "action": "go"})
    ci._apply_project_data(data, show2)
    assert ci.schedule == {"enabled": True, "events": [{"time": "10:00", "days": list(range(7)), "action": "go"}]}
    assert "schedule" not in ci._unknown_project_keys


def test_dialog_adds_removes_and_saves(blind_ci):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    ci = blind_ci
    root = tk.Tk()
    root.withdraw()
    try:
        ci.root = root
        ci._set_dirty_flag = lambda *a: setattr(ci, "dirty", True)
        win = ci._open_schedule()
        vd = win._vd
        vd["time_var"].set("25:00")
        vd["add"]()
        assert "HH:MM" in vd["err"].cget("text") and vd["st"]["events"] == []
        vd["time_var"].set("18:30")
        vd["days_var"].set("Weekends")
        vd["cue_var"].set("4")
        vd["add"]()
        vd["time_var"].set("08:00")
        vd["action_var"].set("Blackout")
        vd["add"]()
        assert vd["list"].get(0, "end") == ("08:00  Weekends  Blackout", "18:30  Weekends  cue 4")
        vd["list"].selection_set(0)
        vd["remove"]()
        vd["enabled_var"].set(True)
        vd["save"]()
        assert ci.dirty and ci.schedule["enabled"]
        assert ci.schedule["events"] == [{"time": "18:30", "days": [5, 6], "action": "cue", "cue": "4"}]
    finally:
        root.destroy()
