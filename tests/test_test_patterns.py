# SPDX-License-Identifier: MPL-2.0
"""Output test patterns: images, and taking over / restoring the output safely."""
import os

import pytest

import control_interface as ci_mod
import test_patterns as tp
from test_blind_editing import ci as blind_ci, renders, keys  # noqa: F401
from test_cue_activation import ci as _base_ci, commands  # noqa: F401


def test_grid_marks_the_edges_and_is_the_output_size():
    img = tp.make("grid", 1280, 720, "Display 0")
    assert img.size == (1280, 720)
    for xy in ((0, 0), (1279, 0), (0, 719), (1279, 719), (640, 0), (0, 360)):
        assert min(img.getpixel(xy)) >= 250, xy          # the outermost pixels are white (cropping shows)
    assert img.getpixel((640, 360)) == (255, 255, 255)   # center cross


def test_bars_and_ramp():
    w, h = 800, 400
    img = tp.make("bars", w, h, "Display 1")
    for i, color in enumerate(tp.BARS_75):
        assert img.getpixel((i * w // 8 + w // 16, h // 10)) == color
    ramp = [img.getpixel((x, h - 2))[0] for x in range(0, w, 50)]
    assert ramp == sorted(ramp) and ramp[0] == 0 and img.getpixel((w - 1, h - 2))[0] == 255


def test_white_and_unknown_kinds():
    assert tp.make("white", 64, 36).getpixel((5, 5)) == (255, 255, 255)
    with pytest.raises(ValueError):
        tp.make("rainbow", 10, 10)


def test_files_are_written_once_per_size(tmp_path):
    a = tp.ensure_file("grid", 320, 180, "Display 0", str(tmp_path))
    t = os.path.getmtime(a)
    assert tp.ensure_file("grid", 320, 180, "Display 0", str(tmp_path)) == a and os.path.getmtime(a) == t
    assert tp.ensure_file("grid", 640, 360, "Display 0", str(tmp_path)) != a


class Mon:
    def __init__(self, w, h):
        self.width, self.height = w, h


@pytest.fixture
def ci(blind_ci, tmp_path, monkeypatch):
    blind_ci.re.monitors = [Mon(320, 180), Mon(640, 360)]
    blind_ci.CONFIG_DIR = str(tmp_path / "cfg")
    monkeypatch.setattr(ci_mod.messagebox, "askyesno", lambda *a, **k: True)
    monkeypatch.setattr(ci_mod.messagebox, "showinfo", lambda *a, **k: None)
    return blind_ci


def test_pattern_replaces_the_output_and_off_restores_it(ci):
    before = ci._program_layers()
    ci._show_test_pattern("grid")
    (cmd,) = renders(ci)
    (layer,) = cmd["layers"].values()
    assert layer["screens"] == [0] and "grid_320x180" in layer["path"] and ci._test_pattern["kind"] == "grid"
    ci._show_test_pattern("bars")                       # switching patterns keeps the original to restore
    ci._show_test_pattern("off")
    cmds = renders(ci)
    assert keys(cmds[-1]["layers"]) == keys(before) and cmds[-1].get("preroll") is True
    assert ci._test_pattern is None and ci.active_col == 0


def test_multiple_outputs_get_one_pattern_each_at_their_size(ci):
    ci.screen_var.set(ci.MULTIPLE)
    ci.layer_screens = {"0": 0, "1": 1}
    ci._show_test_pattern("white")
    cmd = renders(ci)[-1]
    assert sorted((l["screens"][0], os.path.basename(l["path"])[:15]) for l in cmd["layers"].values()) == \
        [(0, "white_320x180_D"), (1, "white_640x360_D")]
    assert cmd["all_assigned_screens"] == {0, 1}
    ci.layer_screens = {"0": 1}                         # routing edits must not move the pattern
    ci._trigger_column()
    assert sorted(l["screens"][0] for l in renders(ci)[-1]["layers"].values()) == [0, 1]


def test_declining_the_confirmation_changes_nothing(ci, monkeypatch):
    monkeypatch.setattr(ci_mod.messagebox, "askyesno", lambda *a, **k: False)
    ci._show_test_pattern("grid")
    assert not renders(ci) and ci._test_pattern is None


def test_go_replaces_the_pattern_with_the_next_cue(ci):
    ci.test_pattern_var = type("V", (), {"v": None, "set": lambda self, v: setattr(self, "v", v)})()
    ci._show_test_pattern("grid")
    ci._go_to_next_cue_manual()
    ci._finalize_transition_from_thread(ci.transition_thread.started[-1][3])
    assert ci._test_pattern is None and ci.test_pattern_var.v == "off" and ci.active_col == 1


def test_no_output_selected_explains_instead(ci):
    ci.selected_display = None
    ci._show_test_pattern("grid")
    assert not renders(ci) and ci._test_pattern is None
