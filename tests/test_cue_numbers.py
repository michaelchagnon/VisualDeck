# SPDX-License-Identifier: MPL-2.0
"""Cue numbers (Cue Settings): independent of column position, shown on the
header, used by OSC and MIDI; the browser remote's taps stay positional."""
import pytest

import project_schema as ps
import remote_control as rc
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401


@pytest.mark.parametrize("raw,clean", [("12", "12"), (" 12.5 ", "12.5"), ("A3", "A3"), ("", None), (None, None),
                                       ("1 2", None), ("x" * 17, None), (True, None), (7, "7"), ("12;", None)])
def test_numbers_are_validated(raw, clean):
    assert ps.clean_cue_number(raw) == clean
    kept = ps.clean_cue_settings({"0": {"number": raw}})
    assert kept == ({"0": {"number": clean}} if clean else {})


def test_remote_resolution_prefers_cue_numbers(blind_ci):
    ci = blind_ci
    ci.cue_settings = {"2": {"number": "1"}, "1": {"number": "12.5"}}
    ci.cue_names["0"] = "Opening"
    assert ci._cue_from_remote(1) == 2, "cue numbered 1, not the first column"
    assert ci._cue_from_remote(2) == 1 - 0 and ci._cue_from_remote("12.5") == 1
    assert ci._cue_from_remote("opening") == 0 and ci._cue_from_remote(99) is None
    assert rc.parse("/visualdeck/cue", [12.5]) == ("cue", "12.5")
    assert rc.parse("/visualdeck/cue", [-1.5]) is None and rc.parse("/visualdeck/cue", [float("nan")]) is None


def test_browser_taps_and_shortcuts_stay_positional(blind_ci):
    ci = blind_ci
    ci.cue_settings = {"2": {"number": "1"}}
    ci._handle_remote_command("cue_at", 2)
    assert ci.transition_thread.started[-1][3] == 1, "the second column, whatever the cue numbers"
    status = ci._build_web_status()
    assert status["cues"][2]["cue_number"] == "1" and status["cues"][2]["number"] == 3


def test_header_shows_the_number_and_numbers_move_with_cues(blind_ci):
    ci = blind_ci
    ci.cue_settings = {"0": {"number": "10"}}
    assert ci._cue_header_text(0).startswith("10  ")
    ci.cue_settings = ps.remap_cue_keys(ci.cue_settings, lambda c: {0: 2, 1: 0, 2: 1}.get(c, c))
    assert ci._cue_number(2) == "10" and ci._cue_number(0) is None


def test_form_and_duplicates(blind_ci):
    import control_interface as ci_mod
    tch = ["Show default", "None (cut)", "Fade", "Smooth Fade", "Dip to Black"]
    form = lambda n: ci_mod.ControlInterface._cue_settings_from_form(  # noqa: E731
        tch[0], tch, "", "Wait for GO", ["Wait for GO", "a", "b"], "5", "", n)
    assert form("5.1") == {"number": "5.1"} and form("") == {}
    with pytest.raises(ValueError):
        form("five point one")
    ci = blind_ci
    ci.cue_settings = {"1": {"number": "B2"}}
    assert ci._cue_number_taken("b2", 0) == 1 and ci._cue_number_taken("B2", 1) is None
