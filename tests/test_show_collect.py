# SPDX-License-Identifier: MPL-2.0
"""Collect Show: copies, safety rules, and a collected show that opens elsewhere."""
import json
import os
import shutil
from collections import namedtuple

import pytest

import media_relink
import show_collect as sc
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401


def touch(path, size=1000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(os.urandom(size))
    return path


def test_plan_keeps_names_unique_regardless_of_case(tmp_path):
    m = sc.plan(["/a/clip.mp4", "/b/Clip.mp4", "/c/clip.mp4", "D:\\shows\\logo.png", "/a/clip.mp4"], "/out/media")
    assert [os.path.basename(v) for v in m.values()] == ["clip.mp4", "Clip (2).mp4", "clip (3).mp4", "logo.png"]


def test_destination_must_be_empty_and_have_room(tmp_path, monkeypatch):
    busy = tmp_path / "busy"
    touch(str(busy / "x.txt"))
    with pytest.raises(sc.CollectError, match="not empty"):
        sc.check_destination(str(busy), 10)
    sc.check_destination(str(tmp_path / "new" / "deeper"), 10)          # does not exist yet: fine
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(sc.shutil, "disk_usage", lambda p: Usage(0, 0, 100 * 1024 * 1024))
    with pytest.raises(sc.CollectError, match="Not enough free space"):
        sc.check_destination(str(tmp_path / "big"), 80 * 1024 * 1024)
    assert sc.unique_folder(str(tmp_path), "busy").endswith("busy (2)")


def test_copy_reports_progress_and_skips_missing(tmp_path):
    a = touch(str(tmp_path / "src" / "a.mov"), 9 * 1024 * 1024)
    mapping = sc.plan([a, "/gone/b.mov"], str(tmp_path / "out" / "media"))
    seen = []
    copied, missing, canceled = sc.copy_files(mapping, progress=lambda d, t, n: seen.append((d, t)))
    assert copied == [a] and missing == ["/gone/b.mov"] and not canceled
    assert seen[-1] == (9 * 1024 * 1024, 9 * 1024 * 1024) and len(seen) == 3       # 4 MB chunks
    assert open(mapping[a], "rb").read() == open(a, "rb").read()


def test_cancel_leaves_only_a_clearly_named_partial_file(tmp_path):
    a = touch(str(tmp_path / "src" / "a.mov"), 9 * 1024 * 1024)
    mapping = sc.plan([a], str(tmp_path / "out" / "media"))
    calls = []
    copied, _, canceled = sc.copy_files(mapping, cancel=lambda: len(calls) > 1 or calls.append(1) and False)
    assert canceled and not copied and not os.path.exists(mapping[a])
    assert os.listdir(tmp_path / "out" / "media") == ["a.mov.partial"]


def test_never_overwrites(tmp_path):
    a = touch(str(tmp_path / "src" / "a.mov"))
    mapping = sc.plan([a], str(tmp_path / "out" / "media"))
    touch(mapping[a], 5)
    with pytest.raises(sc.CollectError):
        sc.copy_files(mapping)
    assert os.path.getsize(mapping[a]) == 5


def test_collected_show_opens_elsewhere_with_all_media(blind_ci, fixtures, tmp_path):
    ci = blind_ci
    for name in ("_build_media_list", "_build_grid", "_initialize_render_engine", "_on_toggle_presentation_mode"):
        setattr(ci, name, lambda *a, **k: None)
    ci.grid_cells[(2, 2)] = "/gone/missing.mov"
    ci.project_media_paths = [fixtures["counting"], fixtures["red_png"]]
    ci.current_project_path = str(tmp_path / "orig" / "My Show.vjdeck")
    ci.cue_settings = {"1": {"notes": "keep me"}}
    folder = sc.unique_folder(str(tmp_path / "usb"), "My Show")
    show = os.path.join(folder, "My Show.vjdeck")
    media = ci._all_media_paths()
    mapping = sc.plan(media, os.path.join(folder, "media"))
    result = sc.collect(ci._build_save_data(show), mapping, show)
    assert result["copied"] == 3 and result["missing"] == ["/gone/missing.mov"] and result["show"] == show
    assert ci.current_project_path.endswith(os.path.join("orig", "My Show.vjdeck")), "the open show is unchanged"

    moved = str(tmp_path / "other machine" / "My Show")
    shutil.move(folder, moved)                                     # the folder travels
    data = json.load(open(os.path.join(moved, "My Show.vjdeck")))
    assert data["cue_settings"] == {"1": {"notes": "keep me"}}
    ci._apply_project_data(data, os.path.join(moved, "My Show.vjdeck"))
    found = [p for p in ci._all_media_paths() if p != "/gone/missing.mov"]
    assert found and all(p.startswith(moved) for p in found)
    assert media_relink.find_missing(ci._all_media_paths()) == ["/gone/missing.mov"]


def test_a_shows_own_effects_travel_with_it(blind_ci, fixtures, tmp_path, monkeypatch):
    """Effects from the user's folder that the show uses are copied to
    <show>/effects; on another computer (without them) the show finds them
    there, and they win over a same-named effect of that computer's user."""
    import isf
    ci = blind_ci
    for name in ("_build_media_list", "_build_grid", "_initialize_render_engine", "_on_toggle_presentation_mode"):
        setattr(ci, name, lambda *a, **k: None)
    mine = tmp_path / "my effects"
    mine.mkdir()
    body = '/*{"DESCRIPTION": "%s", "INPUTS": [{"NAME": "inputImage", "TYPE": "image"}]}*/\nvoid main() {}\n'
    (mine / "Glow.fs").write_text(body % "designer's")
    (mine / "Unused.fs").write_text(body % "not in the show")
    monkeypatch.setattr(isf, "USER_DIR", str(mine))
    isf.set_show_folder(None)
    ci.cell_positions[(0, 0)] = {"x": 0.5, "y": 0.5, "effect": {"name": "Glow.fs", "inputs": {}}}
    ci.output_effects = {"0": {"name": "Invert.fs", "inputs": {}}}           # built in: not copied
    assert ci._effect_names() == {"Glow.fs", "Invert.fs"}
    folder = sc.unique_folder(str(tmp_path / "usb"), "Show")
    show = os.path.join(folder, "Show.vjdeck")
    mapping = sc.plan(ci._all_media_paths(), os.path.join(folder, "media"))
    effects = sc.effect_plan(isf.custom_files(ci._effect_names()).values(), os.path.join(folder, "effects"))
    assert list(effects) == [str(mine / "Glow.fs")]
    result = sc.collect(ci._build_save_data(show), mapping, show, effects=effects)
    assert result["effects"] == 1 and sorted(os.listdir(os.path.join(folder, "effects"))) == ["Glow.fs"]
    data = json.load(open(show))
    assert not any("Glow" in k for k in data["media_info"]), "effects are not media"

    other = tmp_path / "other user"
    other.mkdir()
    (other / "Glow.fs").write_text(body % "an older one")
    monkeypatch.setattr(isf, "USER_DIR", str(other))
    ci._apply_project_data(data, show)                                      # opened on the show computer
    assert isf.load("Glow.fs")["description"] == "designer's"
    ci._reset_project_state(new_project=True)
    assert isf.load("Glow.fs")["description"] == "an older one"            # no show: its folder is not searched
