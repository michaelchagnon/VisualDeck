# SPDX-License-Identifier: MPL-2.0
"""Show-file portability and relinking as the controller does it (no windows opened)."""
import json
import os
import shutil

import pytest

import control_interface as ci_mod
from media_manager import MediaManager
from test_control_state import Var


def touch(path, size=16):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"v" * size)
    return path


class FakeRoot:
    def __init__(self):
        self.after_calls = []

    def after(self, ms, fn=None, *a):
        self.after_calls.append(fn)

    def title(self, *_):
        pass


@pytest.fixture
def ci(tmp_path, monkeypatch):
    monkeypatch.setattr(ci_mod.ControlInterface, "CONFIG_DIR", str(tmp_path / ".visualdeck"))
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    obj = ci_mod.ControlInterface(MediaManager())
    obj.root = FakeRoot()
    obj.screen_var = Var("None")
    obj.transition_var = Var("Fade")
    obj.transition_timing_var = Var("1.0")
    obj.presentation_mode_var = Var(False)
    obj.presentation_timing_var = Var("5.0")
    obj.presentation_loop_var = Var(False)
    obj.calls = []
    for name in ("_on_toggle_presentation_mode", "_initialize_render_engine", "_build_media_list",
                 "_build_grid", "_select_column", "_update_active_players"):
        monkeypatch.setattr(obj, name, (lambda n: lambda *a, **k: obj.calls.append(n))(name))
    monkeypatch.setattr(obj, "_update_cell_preview", lambda r, c: obj.calls.append(("preview", r, c)))
    yield obj
    obj.proxy_executor.shutdown(wait=False)


def load(ci, path):
    with open(path) as f:
        ci._apply_project_data(json.load(f), path)


def test_saved_show_records_relative_paths_and_sizes(ci, tmp_path):
    clip = touch(str(tmp_path / "show" / "media" / "clip.mp4"), size=33)
    ci.project_media_paths = [clip]
    ci.grid_cells[(0, 0)] = clip
    ci.cell_positions[(0, 0)] = {"x": 0.5, "y": 0.5}
    show = str(tmp_path / "show" / "show.vjdeck")
    ci._perform_save(show)
    data = json.load(open(show))
    assert data["media_cells"]["0,0"]["path"] == clip          # absolute path unchanged
    assert data["media_info"][clip] == {"rel": "media/clip.mp4", "size": 33}


def test_show_folder_moved_with_its_media_relinks_on_open(ci, tmp_path):
    clip = touch(str(tmp_path / "old" / "media" / "clip.mp4"))
    ci.project_media_paths = [clip]
    ci.grid_cells[(0, 1)] = clip
    ci.cell_positions[(0, 1)] = {"x": 0.5, "y": 0.5}
    ci._perform_save(str(tmp_path / "old" / "show.vjdeck"))
    shutil.move(str(tmp_path / "old"), str(tmp_path / "new"))

    load(ci, str(tmp_path / "new" / "show.vjdeck"))
    new_clip = os.path.normpath(str(tmp_path / "new" / "media" / "clip.mp4"))
    assert ci.grid_cells[(0, 1)] == new_clip and ci.project_media_paths == [new_clip]
    assert ci.project_is_dirty, "the file on disk still has the old paths"
    assert not [f for f in ci.root.after_calls if f], "nothing left missing, so no dialog"


def test_media_still_missing_after_open_schedules_the_relink_dialog(ci, tmp_path):
    show = str(tmp_path / "show.vjdeck")
    json.dump({"media_cells": {"0,0": {"path": "/gone/x.mp4", "pos": {}}}, "project_media": ["/gone/x.mp4"]},
              open(show, "w"))   # an older show file: no media_info at all
    opened = []
    ci._open_relink_dialog = lambda on_open=False: opened.append(on_open)
    load(ci, show)
    assert ci.grid_cells[(0, 0)] == "/gone/x.mp4" and not ci.project_is_dirty
    for fn in ci.root.after_calls:
        if fn:
            fn()
    assert opened == [True]


def test_relink_never_changes_the_live_cue_output(ci, tmp_path):
    new = touch(str(tmp_path / "found" / "x.mp4"))
    ci.grid_cells[(0, 0)] = "/gone/x.mp4"      # live cue 0
    ci.grid_cells[(0, 1)] = "/gone/y.mp4"
    ci.active_col = 0
    out = ci._apply_relink({"/gone/x.mp4": new})
    assert out == {"cells": 1, "live_affected": True}
    assert "_update_active_players" not in ci.calls and ("preview", 0, 0) in ci.calls
    assert ci.project_is_dirty

    ci.calls.clear()
    out = ci._apply_relink({"/gone/y.mp4": new})
    assert out["live_affected"] is False and "_update_active_players" in ci.calls  # preload refreshed


def test_missing_file_keeps_its_hints_across_save_as(ci, tmp_path):
    ci.current_project_path = str(tmp_path / "a" / "show.vjdeck")
    ci.grid_cells[(0, 0)] = "/gone/x.mp4"
    ci.media_info = {"/gone/x.mp4": {"rel": "media/x.mp4", "size": 5}}
    info = ci._media_info_for_save(str(tmp_path / "b" / "show.vjdeck"))
    assert info["/gone/x.mp4"] == {"rel": "../a/media/x.mp4", "size": 5}


def test_locate_relinks_siblings_in_the_same_folder(ci, tmp_path, monkeypatch):
    a = touch(str(tmp_path / "drive" / "a.mov"))
    b = touch(str(tmp_path / "drive" / "b.mov"))
    ci.grid_cells[(0, 3)] = "/old/a.mov"
    ci.grid_cells[(1, 3)] = "/old/b.mov"
    monkeypatch.setattr(ci_mod.filedialog, "askopenfilename", lambda **k: a)
    mapping = ci._locate_missing("/old/a.mov")
    assert mapping == {"/old/a.mov": os.path.normpath(a), "/old/b.mov": b}
    assert ci.grid_cells[(1, 3)] == b
