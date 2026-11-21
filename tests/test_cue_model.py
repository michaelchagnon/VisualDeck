# SPDX-License-Identifier: MPL-2.0
"""Cue model: show schema versions, per-cue settings and transitions, BACK,
Blackout, and adding/deleting cues without disturbing the output."""
import json

import pytest

import control_interface as ci_mod
import project_schema as ps
from test_blind_editing import ci as blind_ci, renders, keys  # noqa: F401
from test_cue_activation import ci as _base_ci, commands  # noqa: F401


# --- schema -----------------------------------------------------------------

def test_old_show_files_are_upgraded_in_memory():
    data, info = ps.migrate({"version": "2.8.3", "cues": {"0": "A"}})
    assert info == {"from": 1, "newer": False, "notes": [info["notes"][0]]}
    assert data["schema_version"] == ps.CURRENT and data["cue_settings"] == {} and data["media_info"] == {}


def test_newer_show_files_are_flagged_and_their_unknown_keys_kept():
    data, info = ps.migrate({"schema_version": 9, "timeline": [1, 2], "cues": {}})
    assert info["newer"] and data["schema_version"] == 9
    assert ps.unknown_keys(data) == {"timeline": [1, 2]}


def test_malformed_cue_settings_are_dropped_not_fatal():
    raw = {"0": {"transition": "Fade", "duration": 2, "notes": " hi ", "follow": {"mode": "after", "seconds": 3}},
           "1": {"transition": "Wipe", "duration": -1, "follow": {"mode": "after", "seconds": "x"}},
           "x": {"notes": "bad key"}, "2": "not a dict", "3": {"duration": True, "follow": {"mode": "end"}}}
    assert ps.clean_cue_settings(raw) == {
        "0": {"transition": "Fade", "duration": 2.0, "notes": " hi ", "follow": {"mode": "after", "seconds": 3.0}},
        "3": {"follow": {"mode": "end"}}}


def test_cue_settings_survive_save_and_load(blind_ci, tmp_path):
    ci = blind_ci
    ci.cue_settings = {"1": {"transition": "None", "notes": "Wait for the actor"}}
    ci._unknown_project_keys = {"future_thing": {"x": 1}}
    show = str(tmp_path / "s.vjdeck")
    for name in ("_build_media_list", "_build_grid", "_initialize_render_engine", "_on_toggle_presentation_mode"):
        setattr(ci, name, lambda *a, **k: None)
    ci._perform_save(show)
    saved = json.load(open(show))
    assert saved["schema_version"] == ps.CURRENT and saved["future_thing"] == {"x": 1}
    ci._apply_project_data(saved, show)
    assert ci.cue_settings == {"1": {"transition": "None", "notes": "Wait for the actor"}}
    assert ci._unknown_project_keys == {"future_thing": {"x": 1}}


# --- per-cue transitions ----------------------------------------------------

def test_a_cue_can_cut_while_the_show_default_is_a_fade(blind_ci):
    ci = blind_ci
    ci.cue_settings = {"1": {"transition": "None"}}
    ci._initiate_transition(0, 1)
    assert not ci.transition_thread.started and ci.active_col == 1
    assert renders(ci)[0].get("preroll") is True


def test_a_cue_can_have_its_own_fade_time(blind_ci):
    ci = blind_ci
    ci.cue_settings = {"1": {"duration": 2.5}}
    ci._initiate_transition(0, 1)
    assert ci.transition_thread.started[-1][2] == 2.5


def test_cue_settings_form_validation():
    t = ["default", "None (cut)", "Fade"]
    f = ["Wait for GO", "after", "end"]
    form = ci_mod.ControlInterface._cue_settings_from_form
    assert form("default", t, "", "Wait for GO", f, "5", "  ") == {}
    assert form("Fade", t, "1.5", "after", f, "12", " note ") == {
        "transition": "Fade", "duration": 1.5, "follow": {"mode": "after", "seconds": 12.0}, "notes": "note"}
    assert form("None (cut)", t, "", "end", f, "", "") == {"transition": "None", "follow": {"mode": "end"}}
    for bad in (("default", t, "abc", "Wait for GO", f, "5", ""), ("default", t, "-1", "Wait for GO", f, "5", ""),
                ("default", t, "", "after", f, "soon", "")):
        with pytest.raises(ValueError):
            form(*bad)


def test_header_marks_show_cue_settings(blind_ci):
    ci = blind_ci
    ci.cue_names["0"] = "Intro"
    ci.cue_settings = {"0": {"duration": 3.0, "follow": {"mode": "end"}, "notes": "x"}}
    assert ci._cue_header_text(0) == "Intro • → *"
    assert ci._cue_header_text(1) == "Cue 2"


# --- BACK and Blackout ------------------------------------------------------

def test_back_goes_to_the_previous_cue_with_media(blind_ci, fixtures):
    ci = blind_ci
    ci.grid_cells = {k: v for k, v in ci.grid_cells.items() if k[1] != 1}   # cue 2 empty
    ci._cut_to_column(2)
    commands(ci)
    ci._go_back()
    assert ci.transition_thread.started[-1][3] == 0
    ci.active_col = 0
    n = len(ci.transition_thread.started)
    ci._go_back()                         # at the first cue: nothing happens
    assert len(ci.transition_thread.started) == n


def test_blackout_fades_to_black_and_go_continues(blind_ci):
    ci = blind_ci
    on_screen = ci._program_layers()
    ci._blackout()
    frm, to, dur, target, _routing = ci.transition_thread.started[-1]
    assert keys(frm) == keys(on_screen) and to == {} and target is None and dur == 1.0
    ci._finalize_transition_from_thread(None)
    cmds = commands(ci)
    assert [c["layers"] for c in cmds if c["action"] == "render"][-1] == {}
    live = [k for c in cmds if c["action"] == "update_layers" for k in c["layers"] if k.startswith("live:")]
    assert live == [], "the blacked-out cue's players are released"
    assert ci._output_is_black() and ci.active_col == 0 and not ci._live_edits_pending()
    ci._blackout()                        # already black: no second fade
    assert len(ci.transition_thread.started) == 1
    ci._go_to_next_cue_manual()
    frm, to, *_ = ci.transition_thread.started[-1]
    assert frm == {} and ci.transition_thread.started[-1][3] == 1


def test_blackout_with_cut_transition_is_immediate(blind_ci):
    ci = blind_ci
    ci.transition_var.set("None")
    ci._blackout()
    assert not ci.transition_thread.started and renders(ci)[-1]["layers"] == {} and ci._output_is_black()


@pytest.mark.parametrize("mode", ["blind", "live"])
def test_edits_never_lift_a_blackout(blind_ci, mode):
    ci = blind_ci
    ci.live_edit_mode = mode
    ci.transition_var.set("None")
    ci._blackout()
    commands(ci)
    ci.cell_positions[(0, 0)] = {"x": 0.1}
    ci._trigger_column()
    assert all(c["layers"] == {} for c in renders(ci))


# --- adding and deleting cues -----------------------------------------------

def test_adding_a_cue_does_not_touch_the_output(blind_ci):
    ci = blind_ci
    ci._build_grid = lambda: None
    ci._add_cue()
    assert not renders(ci), "no re-cut, so the live videos keep playing"


def test_deleting_an_earlier_cue_keeps_the_live_videos_playing(blind_ci, fixtures):
    ci = blind_ci
    ci._cut_to_column(2)
    commands(ci)
    live_key = keys(ci._program_layers())[0]
    ci.cue_settings = {"2": {"notes": "live"}, "1": {"notes": "gone"}}
    ci.tk = None
    ci_mod.tk.messagebox.askyesno = lambda *a, **k: True
    ci._delete_cue(1)
    assert ci.active_col == 1 and ci.cue_settings == {"1": {"notes": "live"}}
    (cmd,) = renders(ci)
    assert keys(cmd["layers"])[0] == live_key and "preroll" not in cmd


def test_deleting_the_live_cue_keeps_it_on_the_output_until_go(blind_ci, fixtures):
    ci = blind_ci
    ci._cut_to_column(1)
    commands(ci)
    on_screen = ci._program_layers()
    ci_mod.tk.messagebox.askyesno = lambda *a, **k: True
    ci._delete_cue(1)
    (cmd,) = renders(ci)
    assert keys(cmd["layers"]) == keys(on_screen), "the output must not jump to another cue"
    assert ci.active_col == 0 and not ci._live_edits_pending()
    ci._go_to_next_cue_manual()           # GO: the cue that came after the deleted one
    assert ci.transition_thread.started[-1][3] == 1
    assert ci.grid_cells[(0, 1)] == fixtures["black_709"]


def test_null_or_bad_cell_positions_in_a_show_file_load_as_defaults(blind_ci, fixtures):
    ci = blind_ci
    for name in ("_build_media_list", "_build_grid", "_initialize_render_engine", "_on_toggle_presentation_mode"):
        setattr(ci, name, lambda *a, **k: None)
    ci._apply_project_data({"media_cells": {"0,0": {"path": fixtures["counting"], "pos": None},
                                            "1,0": {"path": fixtures["red_png"], "pos": "junk"},
                                            "bad": {"path": "x"}, "2,0": "not a dict"}}, None)
    assert ci.cell_positions[(0, 0)] == {} and ci.cell_positions[(1, 0)] == {}
    layers = ci._get_layers_for_col(0)
    assert layers[0]["loop"] is True and layers[1]["position"] == {}


def test_blackout_button_follows_the_output_state(blind_ci):
    ci = blind_ci

    class Btn:
        def config(self, **k):
            self.__dict__.update(k)
    ci.blackout_button = Btn()
    ci.transition_var.set("None")
    ci._blackout()
    assert ci.blackout_button.text == "OUTPUT BLACK"
    ci._go_to_next_cue_manual()           # a cut, because the show default is now "None"
    assert ci.blackout_button.text == "Blackout"
    ci.transition_var.set("Fade")
    ci._blackout()
    ci._finalize_transition_from_thread(None)
    ci._go_to_next_cue_manual()
    ci._finalize_transition_from_thread(ci.transition_thread.started[-1][3])
    assert ci.blackout_button.text == "Blackout"


def test_opening_a_show_keeps_this_computers_render_mode(blind_ci, tmp_path, caplog):
    """F20: the backend is a machine preference; a show file must not switch it."""
    ci = blind_ci
    for name in ("_build_media_list", "_build_grid", "_initialize_render_engine", "_on_toggle_presentation_mode"):
        setattr(ci, name, lambda *a, **k: None)
    ci.render_mode, ci.hw_accel = "pygame_gpu", "none"
    show = str(tmp_path / "s.vjdeck")
    ci._perform_save(show)
    data = json.load(open(show))
    assert data["settings"] == {"render_mode": "pygame_gpu", "hw_accel": "none"}   # still written for old versions
    data["settings"] = {"render_mode": "opengl", "hw_accel": "cuda"}
    with caplog.at_level("INFO"):
        ci._apply_project_data(data, show)
    assert (ci.render_mode, ci.hw_accel) == ("pygame_gpu", "none")
    assert "using this computer's render_mode=pygame_gpu" in caplog.text
    del data["settings"]                                   # very old shows: no settings at all
    ci._apply_project_data(data, show)
    assert ci.render_mode == "pygame_gpu"
