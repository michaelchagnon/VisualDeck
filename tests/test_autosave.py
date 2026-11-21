# SPDX-License-Identifier: MPL-2.0
"""Autosave and recovery: unsaved changes are written to a separate file, never
to the show; saving or discarding removes it; after a crash the newer version
is offered when the show (or an untitled show) is opened again."""
import json
import os
import shutil
import time

import pytest

import control_interface as ci_mod
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401


REAL_LINK, REAL_COPY2 = os.link, shutil.copy2


@pytest.fixture
def ci(blind_ci, monkeypatch):
    # The base fixture forbids file copies (no media copies for transitions);
    # a second save of a show legitimately keeps the previous file as a backup.
    def only_show_files(real):
        def f(src, dst, *a, **k):
            if not str(src).endswith(".vjdeck"):
                raise AssertionError(f"unexpected copy of {src}")
            return real(src, dst, *a, **k)
        return f
    monkeypatch.setattr(os, "link", only_show_files(REAL_LINK))
    monkeypatch.setattr(shutil, "copy2", only_show_files(REAL_COPY2))
    for name in ("_build_media_list", "_build_grid", "_initialize_render_engine", "_on_toggle_presentation_mode",
                 "_update_cue_dropdown", "_trigger_column", "_add_to_recent", "_save_config"):
        setattr(blind_ci, name, lambda *a, **k: None)
    blind_ci.answers = []
    monkeypatch.setattr(ci_mod.messagebox, "askyesno", lambda *a, **k: blind_ci.answers.pop(0))
    return blind_ci


def saved_show(ci, tmp_path):
    show = str(tmp_path / "Gala Night.vjdeck")
    ci._perform_save(show)
    return show


def test_autosave_writes_a_separate_file_and_never_the_show(ci, tmp_path):
    show = saved_show(ci, tmp_path)
    before = open(show).read()
    assert ci._autosave_now() is None, "nothing to autosave while the show is saved"
    ci.cue_names["0"] = "Opening (changed)"
    ci.project_is_dirty = True
    p = ci._autosave_now()
    assert p.startswith(os.path.join(ci.CONFIG_DIR, "autosave")) and os.path.basename(p).startswith("Gala Night-")
    data = json.load(open(p))
    assert data["cues"]["0"] == "Opening (changed)" and data["autosave_of"] == show
    assert open(show).read() == before and ci.project_is_dirty, "the show file and the dirty flag are untouched"
    ci._perform_save(show)
    assert not os.path.exists(p), "saving removes the autosave"


def test_discarding_changes_removes_the_autosave(ci, tmp_path, monkeypatch):
    saved_show(ci, tmp_path)
    ci.project_is_dirty = True
    p = ci._autosave_now()
    monkeypatch.setattr(ci_mod.messagebox, "askyesnocancel", lambda *a, **k: False)   # "No"
    assert ci._check_before_closing_project() and not os.path.exists(p)


def test_after_a_crash_the_newer_version_is_offered(ci, tmp_path):
    show = saved_show(ci, tmp_path)
    ci.cue_names["0"] = "Recovered name"
    ci.project_is_dirty = True
    ci._autosave_now()
    ci.cue_names["0"] = "Cue 1"                                       # "crash": the app state is gone
    ci.project_is_dirty = False
    ci.answers = [True]
    ci._open_project(show, initial_load=True)
    assert ci.cue_names["0"] == "Recovered name" and ci.project_is_dirty and ci.current_project_path == show
    assert ci._unknown_project_keys == {}, "autosave bookkeeping must not end up in the show"
    ci._perform_save(show)
    assert "autosaved_at" not in json.load(open(show))


def test_declining_sets_the_autosave_aside(ci, tmp_path):
    show = saved_show(ci, tmp_path)
    ci.cue_names["0"] = "Unwanted"
    ci.project_is_dirty = True
    p = ci._autosave_now()
    ci.project_is_dirty = False
    ci.answers = [False]
    ci._open_project(show, initial_load=True)
    assert ci.cue_names["0"] != "Unwanted" and not ci.project_is_dirty
    assert not os.path.exists(p) and os.path.exists(p[:-7] + ".declined.vjdeck")
    ci._open_project(show, initial_load=True)                         # no second question
    assert ci.answers == []


def test_an_autosave_older_than_the_show_is_dropped_silently(ci, tmp_path):
    show = saved_show(ci, tmp_path)
    ci.project_is_dirty = True
    p = ci._autosave_now()
    later = time.time() + 5
    os.utime(show, (later, later))                                    # the show was saved after it
    ci._open_project(show, initial_load=True)                         # askyesno would raise IndexError
    assert not os.path.exists(p)


def test_an_untitled_show_is_offered_at_start(ci, tmp_path):
    ci.current_project_path = None
    ci.cue_names["1"] = "Never saved"
    ci.project_is_dirty = True
    p = ci._autosave_now()
    assert os.path.basename(p) == "Untitled-untitled.vjdeck"
    ci.cue_names["1"] = "Cue 2"
    ci.project_is_dirty = False
    ci.recent_files = []
    ci.answers = [True]
    ci._auto_load_last_project()
    assert ci.cue_names["1"] == "Never saved" and ci.current_project_path is None and ci.project_is_dirty
    show = str(tmp_path / "now saved.vjdeck")
    ci._perform_save(show)
    assert not os.path.exists(p), "saving the untitled show removes its autosave"


def test_autosave_failures_are_logged_not_raised(ci, tmp_path, monkeypatch, caplog):
    saved_show(ci, tmp_path)
    ci.project_is_dirty = True
    monkeypatch.setattr(ci_mod, "atomic_write_json", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert ci._autosave_now() is None and "Autosave" in caplog.text
    broken = ci._autosave_path(ci.current_project_path)
    os.makedirs(os.path.dirname(broken), exist_ok=True)
    open(broken, "w").write("{not json")
    assert ci._recoverable_autosave(ci.current_project_path) is None


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    def fail(*a, **k):
        raise AssertionError(f"unexpected dialog: {a}")
    for name in ("showerror", "showwarning", "showinfo", "askyesnocancel"):
        monkeypatch.setattr(ci_mod.messagebox, name, fail)


FULL_DISK_ENV = "VISUALDECK_TEST_FULL_DISK"


@pytest.mark.skipif(not os.environ.get(FULL_DISK_ENV), reason=f"needs {FULL_DISK_ENV}: a small empty file system")
def test_a_full_disk_leaves_the_show_unchanged(ci, monkeypatch, caplog):
    """Checklist I4 on a real file system that fills up (for example a 256 KiB
    tmpfs): Save fails with a message and the show file stays as it was, no
    temporary file is left behind; an autosave that fails is logged; nothing raises."""
    import errno
    disk = os.environ[FULL_DISK_ENV]
    show = os.path.join(disk, "Gala Night.vjdeck")
    ci._perform_save(show)
    before = open(show, "rb").read()
    filler = os.path.join(disk, "filler.bin")
    try:
        fd = os.open(filler, os.O_WRONLY | os.O_CREAT)    # fill what is left
        try:
            while True:
                os.write(fd, b"\0" * 512)
        except OSError as e:
            assert e.errno == errno.ENOSPC
        finally:
            os.close(fd)
        errors = []
        monkeypatch.setattr(ci_mod.messagebox, "showerror", lambda title, msg, **k: errors.append(msg))
        ci.cue_names["0"] = "Opening (changed) " + "x" * 2000
        ci.project_is_dirty = True
        assert ci._perform_save(show) is False
        assert errors and "was not modified" in errors[0]
        assert open(show, "rb").read() == before
        assert sorted(os.listdir(disk)) == ["Gala Night.vjdeck", "filler.bin"]     # no temporary file left
        monkeypatch.setattr(type(ci), "CONFIG_DIR", disk)                             # the autosave on it too
        assert ci._autosave_now() is None and "Autosave" in caplog.text
        assert open(show, "rb").read() == before
    finally:
        os.remove(filler)
        os.remove(show)
