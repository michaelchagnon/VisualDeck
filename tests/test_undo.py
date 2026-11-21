# SPDX-License-Identifier: MPL-2.0
"""Undo and redo: the stack (coalescing, limits, maps), and the app: cells,
cue settings, added / deleted / moved cues and layers, the live cue never
re-cut or restarted, output settings left alone, the Edit menu."""
import types

import pytest

import control_interface as ci_mod
import undo
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci, commands, instances  # noqa: F401


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_stack_records_coalesces_and_redoes():
    clk = Clock()
    s = undo.UndoStack(limit=3, coalesce_s=0.5, clock=clk)
    assert not s.record({"v": 0}) and s.baseline == {"v": 0}, "the first state is the baseline"
    assert not s.record({"v": 0}), "no change, no entry"
    s.record({"v": 1})
    clk.t = 0.3
    s.record({"v": 2})                                   # quick: same entry
    assert len(s.undo_entries) == 1 and s.undo_entries[0]["before"] == {"v": 0}
    clk.t = 1.0
    s.record({"v": 3})
    assert s.undo() == ({"v": 2}, None, None) and s.undo() == ({"v": 0}, None, None) and s.undo() is None
    assert s.redo() == ({"v": 2}, None, None)
    clk.t = 1.1
    s.record({"v": 9})                                   # a new change after a redo: no merge, redo cleared
    assert not s.can_redo() and [e["after"] for e in s.undo_entries] == [{"v": 2}, {"v": 9}]
    for i in range(10):
        clk.t += 1
        s.record({"v": 10 + i})
    assert len(s.undo_entries) == 3, "bounded"


def test_structural_entries_never_merge_and_invert_their_maps():
    clk = Clock()
    s = undo.UndoStack(clock=clk)
    s.reset({"v": 0})
    s.record({"v": 1}, "Move Cue", cols={0: 1, 1: 0, 2: 2})
    s.record({"v": 2})                                   # within the coalescing time, but after a move
    s.record({"v": 3}, "Delete Cue", cols={0: 0, 1: None, 2: 1})
    assert [e["label"] for e in s.undo_entries] == ["Move Cue", "Edit", "Delete Cue"]
    state, rows, cols = s.undo()
    assert state == {"v": 2} and rows is None and cols == {0: 0, 1: 2}
    s.undo()
    assert s.undo() == ({"v": 0}, None, {1: 0, 0: 1, 2: 2})
    assert s.redo()[2] == {0: 1, 1: 0, 2: 2}, "redo applies the forward map"
    assert s.undo_label() == "Move Cue" and s.redo_label() == "Edit"


@pytest.fixture
def ui(blind_ci, monkeypatch):
    ci = blind_ci
    ci._set_dirty_flag = types.MethodType(ci_mod.ControlInterface._set_dirty_flag, ci)
    ci._build_media_list = lambda: None
    ci._highlight_live_column = lambda: None
    # The real grid rebuild marks the show changed part-way through a cue or
    # layer change (via _on_output_change); undo must still record one step
    # with the change's index maps.
    ci._build_grid = lambda: ci._set_dirty_flag()
    monkeypatch.setattr(ci_mod.tk.messagebox, "askyesno", lambda *a, **k: True)
    ci._undo_reset()
    commands(ci)
    return ci


def live_keys(ci):
    return set(instances(ci._program["layers"]))


def renders(ci):
    return [c for c in commands(ci) if c["action"] == "render"]


def test_cell_edits_undo_and_redo(ui, fixtures):
    ci = ui
    ci._clear_cell(1, 1)
    assert ci.grid_cells[(1, 1)] is None
    ci._undo()
    assert ci.grid_cells[(1, 1)] == fixtures["red_png"] and ci.project_is_dirty
    ci._redo()
    assert ci.grid_cells[(1, 1)] is None
    ci._undo()
    ci._on_editor_save(0, 1, fixtures["counting"], {"x": 0.25, "y": 0.5, "mask": {"shape": "ellipse"}})
    ci._undo()
    assert (0, 1) not in ci.cell_positions or "mask" not in ci.cell_positions[(0, 1)]
    ci._redo()
    assert ci.cell_positions[(0, 1)]["mask"] == {"shape": "ellipse"}


def test_undoing_a_live_edit_with_blind_editing_waits_for_update(ui, fixtures):
    ci = ui
    keys = live_keys(ci)
    ci._on_editor_save(0, 0, fixtures["counting"], {"x": 0.2, "y": 0.5})
    assert ci._live_edits_pending()
    ci._apply_live_edits()
    commands(ci)
    ci._undo()                                           # back to x = 0.5, on the grid only
    sent = renders(ci)
    assert sent and all(l["position"].get("x") == 0.2 for c in sent for l in c["layers"].values())
    assert not any(c.get("preroll") for c in sent) and ci._live_edits_pending()
    assert live_keys(ci) == keys, "the live video keeps playing"


def test_moving_the_live_cue_undoes_without_restarting_it(ui):
    ci = ui
    keys = live_keys(ci)
    ci._reorder_cues(0, 2)
    assert ci.active_col == 2 and ci._program["col"] == 2
    assert ci._undo_stack().undo_label() == "Move Cue"
    commands(ci)
    ci._undo()
    assert ci.active_col == 0 and ci._program["col"] == 0 and live_keys(ci) == keys
    assert not any(c.get("preroll") for c in renders(ci)), "no cut"
    assert ci._get_layers_for_col(0)[0]["instance"] in keys, "the cell maps to the playing instance again"
    ci._redo()
    assert ci.active_col == 2 and ci._program["col"] == 2 and live_keys(ci) == keys


def test_deleted_cues_come_back_with_their_settings(ui, fixtures):
    ci = ui
    ci.cue_names["1"] = "Big reveal"
    ci.cue_settings["1"] = {"number": "12", "notes": "spot 3"}
    ci._set_dirty_flag()
    cols = ci.COLS
    ci._delete_cue(1)
    assert ci.COLS == cols - 1 and ci.grid_cells.get((0, 1)) == fixtures["black_709"]
    ci._undo()
    assert ci.COLS == cols and ci.cue_names["1"] == "Big reveal" and ci.cue_settings["1"]["number"] == "12"
    assert ci.grid_cells[(1, 1)] == fixtures["red_png"] and ci.grid_cells[(0, 2)] == fixtures["black_709"]
    # The live cue itself: deleting keeps it on the output; undoing brings the cue
    # back on the grid and still does not cut.
    keys = live_keys(ci)
    ci._delete_cue(0)
    assert ci._program["col"] is None
    commands(ci)
    ci._undo()
    assert ci.grid_cells[(0, 0)] == fixtures["counting"] and ci._program["col"] is None
    assert not any(c.get("preroll") for c in renders(ci)) and live_keys(ci) == keys


def test_layers_added_deleted_and_moved(ui, fixtures):
    ci = ui
    rows = ci.ROWS
    keys = live_keys(ci)
    ci._add_layer()
    ci._undo()
    assert ci.ROWS == rows and live_keys(ci) == keys
    ci.layer_names["1"] = "Logo"
    ci.layer_screens["1"] = 0
    ci._set_dirty_flag()
    ci._delete_layer(1)
    assert ci.grid_cells.get((1, 1)) is None and "1" not in ci.layer_screens
    ci._undo()
    assert ci.grid_cells[(1, 1)] == fixtures["red_png"] and ci.layer_names["1"] == "Logo"
    assert ci.layer_screens["1"] == 0 and ci.ROWS == rows
    ci._reorder_layers(0, 1)
    assert ci._program["rows"][0] == 1
    ci._undo()
    assert ci._program["rows"][0] == 0 and ci.grid_cells[(0, 0)] == fixtures["counting"] and live_keys(ci) == keys


def test_output_settings_and_unchanged_saves_make_no_entries(ui):
    ci = ui
    ci._set_output_warp(0, [[0.1, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    ci._set_dirty_flag()
    assert not ci._undo_stack().can_undo(), "output settings are not undone"


def test_edit_menu_follows_the_stack(ui):
    calls = {}

    class Menu:
        def entryconfigure(self, i, **kw):
            calls[i] = kw
    ci = ui
    ci.edit_menu = Menu()
    ci._refresh_undo_menu()
    assert calls[0]["state"] == "disabled" and calls[1]["state"] == "disabled"
    ci._add_cue()
    assert calls[0] == {"label": "Undo Add Cue", "state": "normal"} and calls[1]["state"] == "disabled"
    ci._undo()
    assert calls[0]["label"] == "Undo" and calls[1] == {"label": "Redo Add Cue", "state": "normal"}


def test_undo_waits_for_a_running_transition(ui):
    ci = ui
    ci._add_cue()
    cols = ci.COLS
    ci.transition_active = True
    ci._undo()
    assert ci.COLS == cols
    ci.transition_active = False
    ci._undo()
    assert ci.COLS == cols - 1


def test_duplicate_cue_copies_after_it_and_undoes(ui, fixtures):
    ci = ui
    ci.cue_names["1"] = "Reveal"
    ci.cue_settings["1"] = {"number": "12", "notes": "spot 3", "transition": "Dip to Black"}
    ci.cell_positions[(1, 1)] = {"x": 0.3, "y": 0.5, "mask": {"shape": "ellipse"}}
    ci._set_dirty_flag()
    ci._cut_to_column(2)                                  # the live cue is to the right: it moves
    keys = live_keys(ci)
    cols = ci.COLS
    assert ci._duplicate_cue(1) == 2
    assert ci.COLS == cols + 1 and ci.cue_names["2"] == "Reveal copy" and ci.cue_names["1"] == "Reveal"
    assert ci.grid_cells[(1, 2)] == fixtures["red_png"] and ci.cell_positions[(1, 2)]["mask"] == {"shape": "ellipse"}
    assert ci.cell_positions[(1, 2)] is not ci.cell_positions[(1, 1)], "a copy, not a shared dict"
    assert ci.cue_settings["2"] == {"notes": "spot 3", "transition": "Dip to Black"}, "no duplicate cue number"
    assert ci.active_col == 3 and ci._program["col"] == 3 and live_keys(ci) == keys
    assert ci.grid_cells[(0, 3)] == fixtures["black_709"]
    commands(ci)
    ci._undo()
    assert ci.COLS == cols and ci.active_col == 2 and ci._program["col"] == 2 and live_keys(ci) == keys
    assert ci.grid_cells[(0, 2)] == fixtures["black_709"] and "Reveal copy" not in ci.cue_names.values()
    assert not any(c.get("preroll") for c in renders(ci))
