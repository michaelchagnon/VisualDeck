# SPDX-License-Identifier: MPL-2.0
"""Adding or deleting a layer must not re-cut the live cue: its videos keep
playing (same player keys) and, with blind editing, pending edits stay off
the output."""
from test_blind_editing import ci as blind_ci, renders  # noqa: F401
from test_cue_activation import ci as _base_ci, commands, instances  # noqa: F401


def _live_keys(ci):
    return set(instances(ci._program['layers']))


def test_add_layer_keeps_the_live_cue(blind_ci, fixtures):
    ci = blind_ci
    before_keys = _live_keys(ci)
    ci.cell_positions[(0, 0)] = {"x": 0.2, "y": 0.5}             # a pending blind edit
    ci._set_dirty_flag()
    assert ci._live_edits_pending()
    commands(ci)
    ci._add_layer()
    sent = [c for c in commands(ci) if c["action"] == "render"]
    assert not any(c.get("preroll") for c in sent), "no cut"
    assert _live_keys(ci) == before_keys, "the live videos keep their players"
    assert all(l["position"].get("x", 0.5) == 0.5 for c in sent for l in c["layers"].values()), \
        "the pending edit must stay off the output"
    assert ci._live_edits_pending()


def test_delete_layer_keeps_the_live_cue(blind_ci, fixtures, monkeypatch):
    import control_interface as ci_mod
    ci = blind_ci
    monkeypatch.setattr(ci_mod.tk.messagebox, "askyesno", lambda *a, **k: True)
    ci.grid_cells[(2, 0)] = fixtures["red_png"]
    ci._cut_to_column(0)
    commands(ci)
    before_keys = _live_keys(ci)
    ci._delete_layer(1)                                          # an empty layer between two live ones
    sent = [c for c in commands(ci) if c["action"] == "render"]
    assert not any(c.get("preroll") for c in sent), "no cut"
    assert _live_keys(ci) == before_keys
    assert ci._program["rows"] == {0: 0, 2: 1}, "the program follows its layers"
