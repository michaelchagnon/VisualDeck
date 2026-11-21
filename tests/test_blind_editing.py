# SPDX-License-Identifier: MPL-2.0
"""Blind editing: the output shows the program; edits to the live cue reach it
only through Update Output (blind mode) or at once (live mode). Reorders never
restart live videos."""
import json

import pytest

import control_interface as ci_mod
from media_manager import MediaManager
from test_control_state import Var
from test_cue_activation import ci as _base_ci, commands  # noqa: F401  (fixture reuse)


@pytest.fixture
def ci(_base_ci):
    obj = _base_ci
    obj.re.invalidate_static_cache_for_column = lambda c: None
    obj._update_cell_preview = lambda r, c: None
    obj._build_grid = lambda: None
    obj._set_dirty_flag = lambda *a: None
    obj.live_edit_mode = "blind"
    obj._ui_ready = True
    obj._startup_inhibit_triggers = False   # cleared by the real start-up
    obj._cut_to_column(0)
    commands(obj)  # drain
    return obj


def renders(ci):
    return [c for c in commands(ci) if c["action"] == "render"]


def content(layers):
    return {r: (l["path"], l["position"].get("x", 0.5), l.get("loop", True)) for r, l in layers.items()}


def keys(layers):
    return {r: l.get("instance") for r, l in layers.items()}


def test_blind_edits_wait_for_update_output(ci, fixtures):
    before = ci._program_layers()
    ci._on_editor_save(0, 0, fixtures["counting"], {"x": 0.2, "y": 0.5, "scale_x": 1, "scale_y": 1, "rotation": 0})
    ci.grid_cells[(1, 0)] = fixtures["red_png"]
    ci._trigger_column()
    sent = renders(ci)
    assert sent and all(content(c["layers"]) == content(before) for c in sent), "an edit reached the output"
    assert ci._live_edits_pending()

    ci._apply_live_edits()
    (cmd,) = renders(ci)
    assert content(cmd["layers"]) == {0: (fixtures["counting"], 0.2, True), 1: (fixtures["red_png"], 0.5, True)}
    assert keys(cmd["layers"])[0] == keys(before)[0], "the unchanged video must not restart"
    assert "preroll" not in cmd, "no new video, nothing to preroll"
    assert not ci._live_edits_pending()


def test_update_output_prerolls_only_new_videos(ci, fixtures):
    old_key = keys(ci._program_layers())[0]
    ci.grid_cells[(2, 0)] = fixtures["black_709"]
    ci._apply_live_edits()
    (cmd,) = renders(ci)
    assert cmd.get("preroll") is True and keys(cmd["layers"])[0] == old_key
    assert keys(cmd["layers"])[2].startswith(old_key.split(":")[0] + ":2:0|")


def test_live_mode_applies_edits_at_once(ci, fixtures):
    ci.live_edit_mode = "live"
    ci._on_editor_update_realtime(0, 0, {"x": 0.7, "y": 0.5})
    (cmd,) = renders(ci)
    assert content(cmd["layers"])[0][1] == 0.7 and not ci._live_edits_pending()


def test_output_routing_changes_apply_without_leaking_edits(ci, fixtures):
    ci.cell_positions[(0, 0)] = {"x": 0.9}
    ci.layer_screens = {"0": 1}
    ci.screen_var.set(ci.MULTIPLE)
    ci._set_layer_output(0, 1)
    (cmd,) = renders(ci)
    assert cmd["layers"][0]["screens"] == [1] and cmd["output_mode"] == ci.MULTIPLE
    assert content(cmd["layers"])[0][1] == 0.5, "routing applied, edit still held back"
    assert ci._live_edits_pending()


def test_loop_toggle_on_the_live_cue_is_held_back_in_blind_mode(ci):
    ci.cell_positions[(0, 0)] = {"loop": True}
    ci._toggle_loop(0, 0, Var(False))
    live = [l for c in commands(ci) if c["action"] == "update_layers"
            for k, l in c["layers"].items() if k.startswith("live:")]
    assert all(l["loop"] is True for l in live)
    ci._apply_live_edits()
    live = [l for c in commands(ci) if c["action"] == "update_layers"
            for k, l in c["layers"].items() if k.startswith("live:")]
    assert live and all(l["loop"] is False for l in live)


def test_in_place_position_edits_do_not_change_the_program(ci):
    ci.cell_positions.setdefault((0, 0), {})["x"] = 0.5
    ci._cut_to_column(0)
    commands(ci)
    ci.cell_positions[(0, 0)]["x"] = 0.1          # mutate the dict in place
    assert ci._program_layers()[0]["position"]["x"] == 0.5
    assert ci._live_edits_pending()


@pytest.mark.parametrize("mode", ["blind", "live"])
def test_layer_reorder_never_restarts_live_videos(ci, fixtures, mode):
    ci.live_edit_mode = mode
    ci.grid_cells[(1, 0)] = fixtures["red_png"]
    ci._cut_to_column(0)
    commands(ci)
    video_key = keys(ci._program_layers())[0]
    ci._reorder_layers(0, 1)                      # video moves to row 1, image to row 0
    sent = renders(ci)
    if mode == "blind":
        assert all(keys(c["layers"]).get(0) == video_key for c in sent)  # output unchanged
        assert ci._live_edits_pending()
        ci._apply_live_edits()
        sent = renders(ci)
    (cmd,) = sent[-1:]
    assert keys(cmd["layers"])[1] == video_key and "preroll" not in cmd
    assert cmd["layers"][1]["order"] == 1 and cmd["layers"][0]["path"] == fixtures["red_png"]
    assert not ci._live_edits_pending()


def test_cue_reorder_moves_the_live_cue_without_restarting(ci):
    video_key = keys(ci._program_layers())[0]
    ci._reorder_cues(0, 3)
    assert ci.active_col == 3
    (cmd,) = renders(ci)
    assert keys(cmd["layers"])[0] == video_key and "preroll" not in cmd
    assert not ci._live_edits_pending()


def test_a_new_clip_in_a_moved_cells_old_place_gets_its_own_player(ci, fixtures):
    ci.live_edit_mode = "live"
    video_key = keys(ci._program_layers())[0]
    ci._reorder_layers(0, 1)
    commands(ci)
    ci.grid_cells[(0, 0)] = fixtures["counting"]     # same file where the moved cell used to be
    ci._trigger_column()
    (cmd,) = renders(ci)
    ks = keys(cmd["layers"])
    assert ks[1] == video_key and ks[0] != video_key and cmd.get("preroll") is True


def test_fade_from_an_edited_live_cue_fades_out_what_is_on_screen(ci):
    ci.cell_positions[(0, 0)] = {"x": 0.3}
    assert ci._live_edits_pending()
    ci._initiate_transition(0, 1)
    from_layers = ci.transition_thread.started[-1][0]
    assert content(from_layers)[0][1] == 0.5
    ci._finalize_transition_from_thread(1)
    assert ci.active_col == 1 and not ci._live_edits_pending()


def test_edits_during_a_fade_apply_at_its_end_in_live_mode(ci, fixtures):
    ci.live_edit_mode = "live"
    ci._initiate_transition(0, 1)
    commands(ci)
    ci.cell_positions[(0, 1)] = {"x": 0.25}
    ci._trigger_column()                               # during the fade: nothing sent
    assert not renders(ci)
    ci._finalize_transition_from_thread(1)
    sent = renders(ci)
    assert sent and content(sent[-1]["layers"])[0][1] == 0.25


def test_mode_is_saved_with_the_machine_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(ci_mod.ControlInterface, "CONFIG_DIR", str(tmp_path / ".vd"))
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    a = ci_mod.ControlInterface(MediaManager())
    assert a.live_edit_mode == "blind"
    a.live_edit_mode = "live"
    a._save_config()
    b = ci_mod.ControlInterface(MediaManager())
    b._load_config()                     # the app loads it in _start_gui
    assert b.live_edit_mode == "live"
    cfg = json.load(open(b.config_path))
    cfg["live_edit_mode"] = "nonsense"
    json.dump(cfg, open(b.config_path, "w"))
    c = ci_mod.ControlInterface(MediaManager())
    c._load_config()
    assert c.live_edit_mode == "blind"
    for o in (a, b, c):
        o.proxy_executor.shutdown(wait=False)
