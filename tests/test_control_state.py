# SPDX-License-Identifier: MPL-2.0
"""ControlInterface state handling that can be exercised without opening windows."""
import json
import os

import pytest

import control_interface as ci_mod
from media_manager import MediaManager


class Var:
    """Stand-in for a tkinter variable."""
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v

    def set(self, v):
        self.v = v


@pytest.fixture
def ci(tmp_path, monkeypatch):
    monkeypatch.setattr(ci_mod.ControlInterface, "CONFIG_DIR", str(tmp_path / ".visualdeck"))
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    monkeypatch.setattr(ci_mod.ControlInterface, "_update_title", lambda self: None)
    errors = []
    monkeypatch.setattr(ci_mod.messagebox, "showerror", lambda *a, **k: errors.append(a))
    obj = ci_mod.ControlInterface(MediaManager())
    obj._errors = errors
    obj.screen_var = Var("None")
    obj.transition_var = Var("Fade")
    obj.transition_timing_var = Var("1.0")
    obj.presentation_mode_var = Var(False)
    obj.presentation_timing_var = Var("5.0")
    obj.presentation_loop_var = Var(False)
    yield obj
    obj.proxy_executor.shutdown(wait=False)


def test_first_run_config_save_does_not_raise(ci):
    """Fresh install: no config.json exists yet."""
    assert not os.path.exists(ci.config_path)
    ci._load_config()
    ci.recent_files = ["/shows/demo.vjdeck"]
    ci._save_config()
    with open(ci.config_path) as f:
        assert json.load(f)["recent_files"] == ["/shows/demo.vjdeck"]


def test_save_roundtrip(ci, tmp_path):
    path = str(tmp_path / "show.vjdeck")
    ci.grid_cells[(0, 0)] = "/media/a.mp4"
    ci.cell_positions[(0, 0)] = {"x": 0.25, "y": 0.5, "scale_x": 1.5, "scale_y": 1.5, "rotation": 10, "loop": False}
    ci.cue_names["0"] = "Opening"
    ci._perform_save(path)
    with open(path) as f:
        data = json.load(f)
    assert data["media_cells"]["0,0"]["path"] == "/media/a.mp4"
    assert data["media_cells"]["0,0"]["pos"]["scale_x"] == 1.5
    assert data["cues"]["0"] == "Opening"
    assert not ci._errors


def test_failed_save_leaves_previous_show_file_intact(ci, tmp_path):
    """A failure while writing must never destroy the last good show file."""
    path = str(tmp_path / "show.vjdeck")
    ci._perform_save(path)
    with open(path) as f:
        good = f.read()
    ci.cue_names["0"] = object()  # not JSON serializable -> json.dump fails mid-write
    ci._perform_save(path)
    assert ci._errors, "user must be told the save failed"
    with open(path) as f:
        assert f.read() == good
