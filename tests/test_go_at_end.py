# SPDX-License-Identifier: MPL-2.0
"""Show option: GO after the last cue with media returns to the first cue
(default, as before) or stays (theater style)."""
import json

import pytest

from test_blind_editing import ci as blind_ci, commands  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401


class Label:
    mapped = False
    text = ""
    def config(self, **k): self.text = k.get("text", self.text)
    def winfo_ismapped(self): return self.mapped
    def pack(self, **k): self.mapped = True
    def pack_forget(self): self.mapped = False


def preloaded_cols(ci):
    cmds = [c for c in commands(ci) if c["action"] == "update_layers"]
    return {l["instance"].split(":")[2].split("|")[0] for k, l in cmds[-1]["layers"].items() if k.startswith("pre:")}


def test_default_wraps_like_before(blind_ci):
    ci = blind_ci
    assert ci.go_at_end == "wrap"
    ci.active_col = ci.COLS - 1
    assert ci._manual_next_col(ci.active_col) == 0
    assert ci._manual_next_col(2) == 3                         # empty columns are still GO targets


def test_stop_stays_on_the_last_cue_with_media(blind_ci):
    ci = blind_ci
    ci.go_at_end = "stop"
    ci.follow_label, ci._preview_top_bar = Label(), None
    ci.transition_var.set("None")
    ci._cut_to_column(2)                                       # cue 3 is the last cue with media
    assert ci._next_cue_col() is None
    n = len(ci.transition_thread.started)
    before = len(commands(ci))
    ci._go_to_next_cue_manual()
    assert ci.active_col == 2 and len(ci.transition_thread.started) == n
    assert not [c for c in commands(ci)[before:] if c["action"] == "render"], "the output must not change"
    assert ci.follow_label.mapped and ci.follow_label.text.startswith("End of show")
    ci._end_of_show_until = 0
    ci._refresh_follow_label()
    assert not ci.follow_label.mapped
    ci._cut_to_column(1)
    assert ci._next_cue_col() == 2
    ci._go_to_next_cue_manual()
    assert ci.active_col == 2


def test_toggle_updates_the_preload_and_marks_the_show_changed(blind_ci):
    ci = blind_ci

    class BoolVar:
        def __init__(self, v): self.v = v
        def get(self): return self.v
        def set(self, v): self.v = v
    ci._set_dirty_flag = lambda *a: setattr(ci, "dirty", True)
    ci._go_wraps_var = BoolVar(True)
    ci.COLS = 3                                                # cue 3 is the last column: wrap preloads cue 1
    ci.transition_var.set("None")
    ci._cut_to_column(2)
    ci._update_active_players()
    assert ci._next_cue_col() == 0 and preloaded_cols(ci) == {"0"}
    ci._go_wraps_var.set(False)
    ci._on_go_wraps_toggle()
    assert ci.go_at_end == "stop" and ci.dirty and ci._next_cue_col() is None
    assert preloaded_cols(ci) == set(), "nothing is preloaded after the last cue"


@pytest.mark.parametrize("stored,expected", [("stop", "stop"), ("wrap", "wrap"), (None, "wrap"), ("nonsense", "wrap")])
def test_option_is_saved_with_the_show(blind_ci, tmp_path, stored, expected):
    ci = blind_ci
    for name in ("_build_media_list", "_build_grid", "_initialize_render_engine", "_on_toggle_presentation_mode"):
        setattr(ci, name, lambda *a, **k: None)
    show = str(tmp_path / "s.vjdeck")
    ci.go_at_end = "stop"
    ci._perform_save(show)
    data = json.load(open(show))
    assert data["go_at_end"] == "stop"
    if stored is None:
        del data["go_at_end"]                                  # a show saved by an older version
    else:
        data["go_at_end"] = stored
    ci._apply_project_data(data, show)
    assert ci.go_at_end == expected
    assert "go_at_end" not in ci._unknown_project_keys
