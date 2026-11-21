# SPDX-License-Identifier: MPL-2.0
"""Edited-cue preview: offscreen composite matches the engine's layout; UI shows
only the latest edit, and only while edits are pending."""
import os
import time

import numpy as np
import pytest
from PIL import Image

import preview_compose as pc
import render_engine as rem
from test_blind_editing import ci as blind_ci, renders  # noqa: F401  (fixture reuse)
from test_cue_activation import ci as _base_ci, commands  # noqa: F401


def over_black(img):
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    bg.alpha_composite(img)
    return np.asarray(bg.convert("RGB"), dtype=np.int16)


def png(path, size, rgba):
    Image.new("RGBA", size, rgba).save(path)
    return path


LAYOUTS = [
    {"x": 0.5, "y": 0.5, "scale_x": 1.0, "scale_y": 1.0, "rotation": 0},
    {"x": 0.2, "y": 0.8, "scale_x": 0.5, "scale_y": 2.0, "rotation": 0},
    {"x": 0.95, "y": 0.05, "scale_x": 3.0, "scale_y": 3.0, "rotation": 30},   # hangs off two edges
]


@pytest.mark.parametrize("pos", LAYOUTS)
def test_layout_matches_the_engines_preview_compositor(tmp_path, pos):
    red = png(str(tmp_path / "red.png"), (400, 300), (255, 0, 0, 255))
    half = png(str(tmp_path / "half.png"), (200, 200), (0, 255, 0, 128))
    layers = {0: {"path": red, "order": 0, "position": dict(pos)},
              1: {"path": half, "order": 1, "position": {"x": 0.6, "y": 0.4, "scale_x": 1.5, "scale_y": 1.5}}}
    eng = rem.BaseRenderEngine(media_manager=None, mode="opengl")
    eng.monitors = []          # the 1920x1080 reference, whatever display this machine has
    expected = eng.render_layers_to_image(layers, (480, 270))
    got = pc.compose(layers, (480, 270), (1920, 1080), pc.StillSource())
    assert np.abs(over_black(expected) - over_black(got)).max() == 0


def test_video_layers_show_a_still(fixtures):
    src = pc.StillSource()
    img = pc.compose({0: {"path": fixtures["red_709"], "order": 0, "position": {"scale_x": 3.0, "scale_y": 3.0}}},
                     (320, 180), (1920, 1080), src)
    r, g, b, _ = img.getpixel((160, 90))
    assert r > 200 and g < 60 and b < 60
    # The counting clip starts black (frame 0: luma 16), as a fade-in does: its still is
    # the frame at 1 s (frame 30: luma 16 + 3 * 30 = 106, gray 105), not the black one.
    still, _ = src.get(fixtures["counting"])
    assert all(95 < v < 115 for v in still.convert("RGB").getpixel((5, 5))), still.getpixel((5, 5))


def test_alpha_video_keeps_its_transparency(fixtures):
    img = pc.compose({0: {"path": fixtures["green_png"], "order": 0, "position": {"scale_x": 20.0, "scale_y": 20.0}},
                      1: {"path": fixtures["alpha_video"], "order": 1, "position": {"scale_x": 6.0, "scale_y": 6.0}}},
                     (320, 180), (1920, 1080), pc.StillSource())
    left, right = img.getpixel((80, 90)), img.getpixel((240, 90))
    assert left[0] > 200 and left[1] < 60, left            # opaque red half
    assert right[1] > 200 and right[0] < 60, right         # transparent half shows the green image


def test_stills_are_cached_until_the_file_changes(tmp_path):
    p = png(str(tmp_path / "a.png"), (10, 10), (255, 0, 0, 255))
    src = pc.StillSource()
    first = src.get(p)
    assert src.get(p) is first
    png(p, (10, 10), (0, 0, 255, 255))
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert src.get(p)[0].getpixel((0, 0))[2] == 255


def test_large_stills_are_reduced_without_changing_the_layout(tmp_path):
    big = png(str(tmp_path / "big.png"), (3000, 1000), (255, 255, 255, 255))
    layers = {0: {"path": big, "order": 0, "position": {"x": 0.5, "y": 0.5, "scale_x": 0.5, "scale_y": 0.5}}}
    eng = rem.BaseRenderEngine(media_manager=None, mode="opengl")
    eng.monitors = []
    expected = over_black(eng.render_layers_to_image(layers, (480, 270)))
    got = over_black(pc.compose(layers, (480, 270), (1920, 1080), pc.StillSource(max_side=1280)))
    assert (expected != got).any(axis=2).sum() <= 2 * 480   # at most a one-pixel edge difference


def test_unreadable_media_is_skipped(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    img = pc.compose({0: {"path": str(bad), "order": 0, "position": {}},
                      1: {"path": str(tmp_path / "gone.png"), "order": 1, "position": {}}},
                     (64, 36), (1920, 1080), pc.StillSource())
    assert img.size == (64, 36)


class SyncExecutor:
    def submit(self, fn):
        fn()


@pytest.fixture
def ci(blind_ci):
    blind_ci._edit_preview_executor = SyncExecutor()
    shown, hidden = [], []
    blind_ci._show_edit_preview = lambda img: shown.append(img)
    blind_ci._hide_edit_preview = lambda: hidden.append(True)
    blind_ci.shown, blind_ci.hidden = shown, hidden
    return blind_ci


def test_preview_appears_only_while_edits_are_pending(ci, fixtures):
    ci._update_edit_preview(ci._live_edits_pending())
    ci._poll_edit_preview()
    assert not ci.shown and ci.hidden
    ci.cell_positions[(0, 0)] = {"x": 0.25}
    ci._update_edit_preview(ci._live_edits_pending())
    ci._poll_edit_preview()
    assert len(ci.shown) == 1 and ci.shown[0].size == (480, 270)
    ci._update_edit_preview(ci._live_edits_pending())   # same edits: no new composite
    ci._poll_edit_preview()
    assert len(ci.shown) == 1
    ci._apply_live_edits()
    ci._update_edit_preview(ci._live_edits_pending())
    assert len(ci.hidden) == 2


def test_only_the_latest_edit_is_shown(ci, fixtures):
    ci.cell_positions[(0, 0)] = {"x": 0.25}
    ci._update_edit_preview(True)
    ci.cell_positions[(0, 0)] = {"x": 0.75}
    ci._update_edit_preview(True)          # two composites queued; only the second is current
    ci._poll_edit_preview()
    assert len(ci.shown) == 1
    ci.cell_positions[(0, 0)] = {"x": 0.1}
    ci._update_edit_preview(True)
    ci._update_edit_preview(False)         # edits applied/undone before the result arrived
    ci._poll_edit_preview()
    assert len(ci.shown) == 1, "a stale composite must not reappear"
