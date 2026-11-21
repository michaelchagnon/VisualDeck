# SPDX-License-Identifier: MPL-2.0
"""UI cue logic: activation ids, preloading, fades and cuts (no windows opened)."""
import os
import queue

import pytest

import control_interface as ci_mod
from media_manager import MediaManager
from test_control_state import Var


class FakeEngine:
    monitors = [object()]
    source_key = staticmethod(ci_mod.OpenGLRenderEngine.source_key)


class FakeRoot:
    def __init__(self):
        self.calls = []

    def after(self, ms, fn=None, *args):
        self.calls.append(fn)
        return len(self.calls)


class FakeTransitionThread:
    def __init__(self):
        self.started = []

    def start_transition(self, *args, curve='linear'):
        self.started.append(args)
        self.curves = getattr(self, 'curves', []) + [curve]


@pytest.fixture
def ci(tmp_path, monkeypatch, fixtures):
    monkeypatch.setattr(ci_mod.ControlInterface, "CONFIG_DIR", str(tmp_path / ".visualdeck"))
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    monkeypatch.setattr(ci_mod.ControlInterface, "_update_title", lambda self: None)

    def no_copies(*a, **k):
        raise AssertionError("media files must never be linked or copied for a transition")
    monkeypatch.setattr(os, "link", no_copies)
    monkeypatch.setattr(ci_mod.shutil if hasattr(ci_mod, "shutil") else __import__("shutil"), "copy2", no_copies)

    obj = ci_mod.ControlInterface(MediaManager())
    obj.screen_var = Var("Display 0")
    obj.transition_var = Var("Fade")
    obj.transition_timing_var = Var("1.0")
    obj.presentation_mode_var = Var(False)
    obj.presentation_timing_var = Var("5.0")
    obj.presentation_loop_var = Var(False)
    obj.selected_display = 0
    obj.re = FakeEngine()
    obj.root = FakeRoot()
    obj.col_labels = {}
    obj.transition_thread = FakeTransitionThread()
    obj._update_cue_dropdown = lambda: None
    obj.set_preview_dirty = lambda: None
    # cue 0 and cue 1 both use the same video; cue 1 also has an image
    obj.grid_cells[(0, 0)] = fixtures["counting"]
    obj.grid_cells[(0, 1)] = fixtures["counting"]
    obj.grid_cells[(1, 1)] = fixtures["red_png"]
    obj.grid_cells[(0, 2)] = fixtures["black_709"]
    obj.active_col = 0
    yield obj
    obj.proxy_executor.shutdown(wait=False)


def commands(ci, action=None):
    out = []
    while not ci.render_command_queue.empty():
        c = ci.render_command_queue.get_nowait()
        if action is None or c.get("action") == action:
            out.append(c)
    return out


def instances(layers):
    return {l["instance"]: l for l in layers.values() if "instance" in l}


def test_live_cue_instances_are_stable_and_images_have_none(ci):
    a = ci._get_layers_for_col(1)
    b = ci._get_layers_for_col(1)
    assert a[0]["instance"] == b[0]["instance"]
    assert "instance" not in a[1], "images are shared by path"
    assert a[0]["instance"].endswith(a[0]["path"])


def test_next_cue_is_preloaded_paused_under_a_reserved_activation(ci):
    ci._update_active_players()
    (cmd,) = commands(ci, "update_layers")
    live = [l for k, l in cmd["layers"].items() if k.startswith("live:")]
    pre = [l for k, l in cmd["layers"].items() if k.startswith("pre:")]
    assert len(live) == 1 and not live[0].get("hold")
    assert len(pre) == 1 and pre[0]["hold"] is True and "instance" in pre[0]
    assert live[0]["path"] == pre[0]["path"] and live[0]["instance"] != pre[0]["instance"]
    assert set(ci._reserved_activation) == {1}


def test_fade_between_cues_sharing_a_file_uses_two_players_and_claims_the_preload(ci):
    ci._update_active_players()
    pre_key = next(l["instance"] for k, l in commands(ci, "update_layers")[0]["layers"].items() if k.startswith("pre:"))

    ci._initiate_transition(0, 1)
    (cmd,) = commands(ci, "update_layers")
    keys = instances(cmd["layers"])
    from_key = ci.transition_from_layers[0]["instance"]
    to_key = ci.transition_to_layers[0]["instance"]
    assert from_key != to_key, "outgoing and incoming cue must not share a decoder"
    assert to_key == pre_key, "GO must reuse the preloaded (paused) players"
    assert from_key in keys and to_key in keys and not keys[to_key].get("hold")
    assert ci.transition_thread.started, "fade thread started"

    ci._finalize_transition_from_thread(1)
    cmds = commands(ci)
    final_render = next(c for c in cmds if c["action"] == "render")
    assert final_render["layers"][0]["instance"] == to_key, "incoming players become the live ones (no restart)"
    after = instances(next(c for c in cmds if c["action"] == "update_layers")["layers"])
    assert to_key in after and from_key not in after, "outgoing cue's player is stopped"


def test_cut_to_the_live_cue_restarts_but_cut_to_preloaded_cue_reuses(ci):
    first = ci._get_layers_for_col(0)[0]["instance"]
    ci._cut_to_column(0)
    again = commands(ci, "render")[-1]["layers"][0]["instance"]
    assert again != first, "re-selecting the live cue restarts its video"

    ci._update_active_players()
    pre_key = next(l["instance"] for k, l in commands(ci, "update_layers")[0]["layers"].items() if k.startswith("pre:"))
    ci._cut_to_column(1)
    cut = commands(ci, "render")[-1]
    assert cut["preroll"] is True and cut["layers"][0]["instance"] == pre_key
    assert "force_restart_paths" not in cut


def test_presentation_preloads_the_next_cue_with_media(ci):
    ci.grid_cells.pop((0, 1)), ci.grid_cells.pop((1, 1))  # cue 1 empty
    ci.presentation_active = True
    ci._update_active_players()
    assert set(ci._reserved_activation) == {2}


def test_previous_cue_is_preloaded_for_back_and_back_claims_it(ci):
    ci.active_col = 1
    ci._cut_to_column(1)
    commands(ci)
    ci._update_active_players()
    (cmd,) = commands(ci, "update_layers")
    back = [l for k, l in cmd["layers"].items() if k.startswith("back:")]
    pre = [l for k, l in cmd["layers"].items() if k.startswith("pre:")]
    assert len(back) == 1 and back[0]["hold"] is True and back[0]["path"] == ci.grid_cells[(0, 0)]
    assert len(pre) == 1 and pre[0]["path"] == ci.grid_cells[(0, 2)]
    assert set(ci._reserved_activation) == {0, 2}
    ci._go_back()
    assert ci.transition_to_layers[0]["instance"] == back[0]["instance"], "BACK reuses the preloaded players"
