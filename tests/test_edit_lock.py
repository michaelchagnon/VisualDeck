# SPDX-License-Identifier: MPL-2.0
"""Show > Lock Editing: while on, the cue grid and the media list cannot be
changed (each attempt is refused with a notice); GO, BACK, cue selection,
Blackout and output settings still work."""
import control_interface as ci_mod
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci, commands  # noqa: F401


def snapshot(ci):
    return ci._undo_snapshot()


def test_locked_actions_change_nothing(blind_ci, fixtures, monkeypatch):
    ci = blind_ci
    monkeypatch.setattr(ci_mod.tk.messagebox, "askyesno", lambda *a, **k: True)
    ci._build_media_list = lambda: None
    ci._set_edit_lock(True)
    before = snapshot(ci)
    attempts = [lambda: ci._clear_cell(1, 1), lambda: ci._delete_cue(1), lambda: ci._delete_layer(1),
                lambda: ci._add_cue(), lambda: ci._add_layer(), lambda: ci._duplicate_cue(0),
                lambda: ci._reorder_cues(0, 2), lambda: ci._reorder_layers(0, 1), lambda: ci._toggle_loop(0, 0),
                lambda: ci._on_editor_save(0, 0, fixtures["counting"], {"x": 0.1, "y": 0.5}),
                lambda: ci._apply_cue_settings(1, {"notes": "x"}), lambda: ci._remove_media(fixtures["red_png"]),
                lambda: ci._clear_layer(0), lambda: ci._set_layer_output(0, 1), lambda: ci._undo()]
    for attempt in attempts:
        assert attempt() is None
    assert snapshot(ci) == before
    assert "Editing is locked" in ci._notice[0]
    ci._set_edit_lock(False)
    ci._clear_cell(1, 1)
    assert ci.grid_cells[(1, 1)] is None, "unlocked: works again"


def test_running_the_show_still_works_when_locked(blind_ci):
    ci = blind_ci
    ci._set_edit_lock(True)
    started = ci.transition_thread.started
    ci._trigger_go()
    assert started[-1][3] == 1, "GO"
    ci._finalize_transition_from_thread(1)
    ci._go_back()
    assert started[-1][3] == 0, "BACK"
    ci._finalize_transition_from_thread(0)
    ci._go_to_cue(2)
    assert started[-1][3] == 2, "a cue from the list or a remote"


def test_title_and_toggle(blind_ci):
    ci = blind_ci
    ci._toggle_edit_lock()
    assert ci.edit_locked and ci._title_text().endswith("[LOCKED]")
    ci._toggle_edit_lock()
    assert not ci.edit_locked and not ci._title_text().endswith("[LOCKED]")
