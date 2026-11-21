# SPDX-License-Identifier: MPL-2.0
"""Layer masks (rectangle / ellipse, soft edge, invert): values, the CPU math,
the GPU shader (read back from the output), Pygame compositing, previews, the
static-cue caches, the editor controls and the UI layers."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import layer_mask as lm
import preview_compose as pc
import render_engine as rem
from test_render_gl import run_gl_scenario
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401


@pytest.mark.parametrize("raw,expected", [
    (None, None), ({}, None), ({"shape": "star"}, None),
    ({"shape": "rect"}, None),                                              # changes nothing
    ({"shape": "rect", "invert": True}, ("rect", 0.0, 0.0, 1.0, 1.0, 0.0, True)),
    ({"shape": "ellipse"}, ("ellipse", 0.0, 0.0, 1.0, 1.0, 0.0, False)),
    ({"shape": "rect", "left": 0.7, "top": -1, "right": "x", "bottom": float("nan"), "feather": 9},
     ("rect", 0.49, 0.0, 1.0, 1.0, 0.5, False)),
])
def test_values_are_cleaned(raw, expected):
    assert lm.clean(raw) == expected


def test_to_dict_round_trips():
    raw = {"shape": "ellipse", "left": 0.1, "right": 0.2, "top": 0.05, "bottom": 0.3, "feather": 0.04, "invert": True}
    assert lm.to_dict(lm.clean(raw)) == raw
    assert lm.clean(lm.to_dict(lm.clean(raw))) == lm.clean(raw)
    assert "invert" not in lm.to_dict(lm.clean({"shape": "ellipse"})) and lm.to_dict(None) is None


def test_factor_rectangle_and_soft_edge():
    m = lm.clean({"shape": "rect", "left": 0.25})
    assert lm.factor(0.24, 0.5, m, 100, 100) == 0 and lm.factor(0.26, 0.5, m, 100, 100) == 1
    soft = lm.clean({"shape": "rect", "left": 0.25, "feather": 0.1})
    assert lm.factor(0.25, 0.5, soft, 100, 100) == 0
    assert lm.factor(0.30, 0.5, soft, 100, 100) == pytest.approx(0.5)      # halfway into the band
    assert lm.factor(0.36, 0.5, soft, 100, 100) == 1
    # The band is a fraction of the shorter side: on a 400x100 medium, 0.1 = 10 px
    # wide horizontally too, i.e. 0.025 in u.
    wide = lm.clean({"shape": "rect", "feather": 0.1, "invert": False, "left": 0.0001})
    assert lm.factor(0.0125, 0.5, wide, 400, 100) == pytest.approx(0.5, abs=0.02)
    assert lm.factor(0.5, 0.5, lm.clean({"shape": "rect", "left": 0.25, "invert": True}), 100, 100) == 0


def test_factor_ellipse_is_an_even_band_for_a_circle():
    m = lm.clean({"shape": "ellipse", "feather": 0.2})
    # Circle of radius 0.5 on a square: distance from the edge is exact.
    for r in (0.3, 0.35, 0.4, 0.45, 0.5):
        for ang in (0.0, 0.7, 2.0, 4.0):
            u, v = 0.5 + r * np.cos(ang), 0.5 + r * np.sin(ang)
            t = min(1.0, max(0.0, (0.5 - r) / 0.2))
            assert lm.factor(u, v, m, 50, 50) == pytest.approx(t * t * (3 - 2 * t), abs=1e-9)
    assert lm.factor(0.5, 0.5, m, 50, 50) == 1 and lm.factor(0.02, 0.02, m, 50, 50) == 0


def test_apply_multiplies_alpha_only_and_adds_alpha_to_bgr():
    m = lm.clean({"shape": "rect", "left": 0.5})
    bgr = np.full((4, 8, 3), (10, 20, 30), np.uint8)
    out = lm.apply(bgr, m)
    assert out.shape == (4, 8, 4) and (out[..., :3] == (10, 20, 30)).all()
    assert (out[:, :4, 3] == 0).all() and (out[:, 4:, 3] == 255).all()
    rgba = np.full((4, 8, 4), (10, 20, 30, 128), np.uint8)
    out = lm.apply(rgba, m)
    assert (out[:, 4:, 3] == 128).all() and (rgba[..., 3] == 128).all(), "the input must not change"
    assert lm.apply(rgba, None) is rgba
    assert not lm.alpha_mask(8, 4, m).flags.writeable


def test_gpu_shader_matches_the_cpu_math(fixtures, x_display):
    r = run_gl_scenario("layer_mask", fixtures, x_display)
    assert r["rect"] == {"left_hidden": 0, "top_hidden": 0, "center": 255, "right_edge_in": 255, "bottom_in": 255}, \
        "top of the media is v = 0, and the insets are per side"
    m = lm.clean({"shape": "ellipse", "feather": 0.1})
    want = [round(255 * float(lm.factor((i + 0.5) / 400, 100.5 / 200, m, 400, 200))) for i in range(0, 400, 4)]
    assert max(abs(g - w) for g, w in zip(r["ellipse_row"], want)) <= 3, (r["ellipse_row"], want)
    assert r["ellipse_row"][0] < 10 < r["ellipse_row"][2] < 250 and r["ellipse_corner"] == 0
    assert r["invert"] == [255, 0]
    assert r["static_cache_bypassed"]
    assert r["rotated"] == {"above": 255, "below": 0}, "the mask turns with the layer (left half -> bottom at 90)"
    assert r["video"]["corner"] == [0, 0, 0] and r["video"]["center"][0] > 240, r["video"]


def test_pygame_compositing_masks_and_caches_separately(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        p = str(tmp_path / "w.png")
        Image.new("RGBA", (40, 20), (255, 255, 255, 255)).save(p)
        eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
        base = {"path": p, "order": 0, "position": {}}
        masked = dict(base, mask={"shape": "rect", "left": 0.5})
        surf = eng._render_layers_to_surface({0: masked}, (40, 20))
        assert surf.get_at((5, 10))[3] == 0 and surf.get_at((35, 10))[3] == 255
        surf = eng._render_layers_to_surface({0: base}, (40, 20))
        assert surf.get_at((5, 10))[3] == 255, "an unmasked layer must not reuse the masked surface"
        # Static cue cache keys must tell masks (and colors) apart.
        k1 = eng.get_cue_cache_key({0: masked})
        assert k1 != eng.get_cue_cache_key({0: base}) != eng.get_cue_cache_key({0: dict(base, brightness=0.2)})
        # pygame_gpu video frames arrive as BGRA bytes: the same helper masks them.
        arr = np.full((20, 40, 4), 255, np.uint8)
        assert lm.apply(arr, lm.layer_mask(masked))[10, 5, 3] == 0
    finally:
        pygame.display.quit()


def test_previews_mask_before_rotating(tmp_path):
    p = str(tmp_path / "w.png")
    Image.new("RGBA", (40, 20), (255, 255, 255, 255)).save(p)
    layer = {"path": p, "order": 0, "position": {"x": 0.5, "y": 0.5, "rotation": 90, "scale_x": 10.0, "scale_y": 10.0},
             "mask": {"shape": "rect", "left": 0.5}}
    a = pc.compose({0: layer}, (192, 108), (1920, 1080), pc.StillSource())
    b = rem.BaseRenderEngine(media_manager=None, mode="opengl").render_layers_to_image({0: layer}, (192, 108))
    for img in (a, b):
        # 90 deg turns the media counter-clockwise: its (hidden) left half goes to the bottom.
        top, bottom = img.getpixel((96, 44)), img.getpixel((96, 64))
        assert top[0] > 200 and top[3] == 255 and (bottom[0] < 30 or bottom[3] < 30), (top, bottom)


def test_ui_layers_carry_the_mask(blind_ci):
    ci = blind_ci
    ci.cell_positions[(0, 0)] = {"x": 0.5, "y": 0.5, "mask": {"shape": "ellipse", "feather": 0.1}}
    layers = ci._get_layers_for_col(0)
    assert layers[0]["mask"] == {"shape": "ellipse", "feather": 0.1}
    layers[0]["mask"]["feather"] = 0.3
    assert ci.cell_positions[(0, 0)]["mask"]["feather"] == 0.1, "the layer is a copy"


def test_editor_controls(tmp_path):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    from editor import MediaEditor
    p = str(tmp_path / "w.png")
    Image.new("RGB", (400, 200), (255, 255, 255)).save(p)
    root = tk.Tk()
    root.withdraw()
    sent = []
    try:
        start = {"x": 0.5, "y": 0.5, "scale_x": 1.0, "scale_y": 1.0, "rotation": 0, "opacity": 0.5}
        ed = MediaEditor(SimpleNamespace(root=root), p, None, SimpleNamespace(monitors=[]), initial_pos=start,
                         realtime_callback=lambda pos: sent.append(dict(pos)))
        assert ed.mask_shape_var.get() == "None" and not ed.canvas.find_withtag("mask_outline")
        ed.mask_shape_var.set("Ellipse")
        ed.mask_vars["left"].set(20)
        ed.mask_vars["feather"].set(10)
        ed._on_mask_change("left")
        assert ed.current_pos["mask"] == {"shape": "ellipse", "left": 0.2, "right": 0.0, "top": 0.0, "bottom": 0.0,
                                          "feather": 0.1}
        assert ed.mask_labels["left"].cget("text") == "20%"
        assert len(ed.canvas.coords(ed.canvas.find_withtag("mask_outline")[0])) == 96
        ed._flush_realtime_callback()
        assert sent[-1]["mask"]["shape"] == "ellipse"
        ed._reset_position()
        assert ed.current_pos["mask"]["left"] == 0.2 and ed.current_pos["opacity"] == 0.5, \
            "Reset is for position, size and rotation only"
        ed.mask_shape_var.set("None")
        ed._on_mask_change()
        assert "mask" not in ed.current_pos and not ed.canvas.find_withtag("mask_outline")
        assert "mask" not in ed.original_pos, "Cancel must restore the layer as it was"
        ed.window.destroy()
        # An existing mask opens with its values.
        ed = MediaEditor(SimpleNamespace(root=root), p, None, SimpleNamespace(monitors=[]),
                         initial_pos={"mask": {"shape": "rect", "top": 0.3, "invert": True}})
        assert (ed.mask_shape_var.get(), ed.mask_vars["top"].get(), ed.mask_invert_var.get()) == ("Rectangle", 30, True)
        ed.window.destroy()
    finally:
        root.destroy()
