# SPDX-License-Identifier: MPL-2.0
import os
import shutil

import media_relink as mr


def touch(path, size=10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x" * size)
    return path


def test_find_missing_is_unique_and_ordered(tmp_path):
    a = touch(str(tmp_path / "a.mp4"))
    gone = str(tmp_path / "gone.mp4")
    assert mr.find_missing([gone, a, gone, None, ""]) == [gone]


def test_relative_path_is_used_when_show_folder_moved(tmp_path):
    old_show = tmp_path / "old" / "show"
    media = touch(str(old_show / "media" / "clip.mp4"), size=42)
    project = str(old_show / "show.vjdeck")
    info = {media: mr.media_info_for(media, project)}
    assert info[media] == {"rel": "media/clip.mp4", "size": 42}

    shutil.move(str(tmp_path / "old"), str(tmp_path / "new"))   # whole show folder moved
    new_project = str(tmp_path / "new" / "show" / "show.vjdeck")
    found = mr.resolve_relative([media], info, new_project)
    assert found == {media: os.path.normpath(str(tmp_path / "new" / "show" / "media" / "clip.mp4"))}


def test_relative_match_rejected_when_size_differs(tmp_path):
    touch(str(tmp_path / "show" / "media" / "clip.mp4"), size=5)
    info = {"/elsewhere/clip.mp4": {"rel": "media/clip.mp4", "size": 999}}
    assert mr.resolve_relative(["/elsewhere/clip.mp4"], info, str(tmp_path / "show" / "s.vjdeck")) == {}


def test_search_matches_by_name_and_prefers_matching_parent_folders(tmp_path):
    root = tmp_path / "drive"
    right = touch(str(root / "shows" / "opera" / "act1" / "intro.mov"))
    touch(str(root / "backup" / "misc" / "intro.mov"))
    unique = touch(str(root / "x" / "Logo.PNG"))
    old_intro = "D:\\Media\\opera\\act1\\intro.mov"      # Windows path from another machine
    old_logo = "/Volumes/show/logo.png"                  # macOS path, different case
    res = mr.search_folder([old_intro, old_logo, "/nowhere/absent.mp4"], str(root))
    assert res.found == {old_intro: right, old_logo: unique}
    assert res.not_found == ["/nowhere/absent.mp4"] and not res.ambiguous


def test_search_reports_ties_as_ambiguous(tmp_path):
    a = touch(str(tmp_path / "a" / "loop.mp4"))
    b = touch(str(tmp_path / "b" / "loop.mp4"))
    res = mr.search_folder(["/old/loop.mp4"], str(tmp_path))
    assert not res.found and sorted(res.ambiguous["/old/loop.mp4"]) == sorted([a, b])


def test_search_uses_recorded_size_to_break_ties(tmp_path):
    touch(str(tmp_path / "a" / "loop.mp4"), size=10)
    b = touch(str(tmp_path / "b" / "loop.mp4"), size=77)
    res = mr.search_folder(["/old/loop.mp4"], str(tmp_path), media_info={"/old/loop.mp4": {"size": 77}})
    assert res.found == {"/old/loop.mp4": b}


def test_search_stops_at_file_limit(tmp_path):
    for i in range(30):
        touch(str(tmp_path / f"d{i}" / f"f{i}.bin"))
    res = mr.search_folder(["/x/f29.bin"], str(tmp_path), max_files=5)
    assert res.truncated and res.files_scanned < 30


def test_apply_mapping_rewrites_cells_media_list_and_info():
    cells = {(0, 0): "/old/a.mp4", (1, 0): "/old/b.png", (0, 1): "/old/a.mp4"}
    media = ["/old/a.mp4", "/new/a.mp4", "/old/b.png"]
    info = {"/old/a.mp4": {"size": 1}}
    n = mr.apply_mapping({"/old/a.mp4": "/new/a.mp4"}, cells, media, info)
    assert n == 2 and cells[(0, 0)] == cells[(0, 1)] == "/new/a.mp4"
    assert media == ["/new/a.mp4", "/old/b.png"]
    assert info == {"/new/a.mp4": {"size": 1}}


def test_rebase_rel_keeps_pointing_at_the_same_file(tmp_path):
    old_proj = str(tmp_path / "shows" / "a" / "show.vjdeck")
    new_proj = str(tmp_path / "shows" / "b" / "copy" / "show.vjdeck")
    rel = mr.rebase_rel("media/clip.mp4", old_proj, new_proj)
    assert rel == "../../a/media/clip.mp4"
    assert os.path.normpath(os.path.join(os.path.dirname(new_proj), rel)) == \
        os.path.normpath(str(tmp_path / "shows" / "a" / "media" / "clip.mp4"))


def test_siblings_found_next_to_a_hand_located_file(tmp_path):
    a = touch(str(tmp_path / "new" / "a.mp4"))
    b = touch(str(tmp_path / "new" / "B.mp4"), size=3)
    touch(str(tmp_path / "new" / "c.mp4"), size=1)
    missing = ["/old/a.mp4", "/old/b.mp4", "/old/c.mp4", "/old/d.mp4"]
    found = mr.siblings_mapping(a, missing, media_info={"/old/c.mp4": {"size": 999}})
    assert found == {"/old/a.mp4": a, "/old/b.mp4": b}   # c rejected by size, d absent


def test_summary_mentions_every_outcome():
    res = mr.SearchResult(found={"a": "x"}, ambiguous={"b": ["1", "2"]}, not_found=["c"],
                          files_scanned=1234, truncated=True)
    text = mr.summarize(res, still_missing=2)
    assert "Relinked 1 file." in text and "more than one" in text and "not found" in text and "1,234" in text
