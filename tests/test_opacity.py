# SPDX-License-Identifier: MPL-2.0
"""Per-layer opacity outside the GPU: values, CPU previews, Pygame compositing, UI layers."""
import numpy as np
import pytest
from PIL import Image

import preview_compose as pc
import render_engine as rem
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401


@pytest.mark.parametrize("raw,expected", [(None, 1.0), ("x", 1.0), (0.25, 0.25), (-1, 0.0), (3, 1.0),
                                          (float("nan"), 1.0), ("0.5", 0.5)])
def test_opacity_values_are_clamped(raw, expected):
    layer = {} if raw is None else {"opacity": raw}
    assert rem.layer_opacity(layer) == expected


def test_cpu_previews_apply_opacity_like_the_output(tmp_path):
    red = str(tmp_path / "r.png")
    Image.new("RGBA", (400, 300), (255, 0, 0, 255)).save(red)
    layers = {0: {"path": red, "order": 0, "opacity": 0.5, "position": {"scale_x": 5.0, "scale_y": 5.0}}}
    eng = rem.BaseRenderEngine(media_manager=None, mode="opengl")
    a = eng.render_layers_to_image(layers, (160, 90))
    b = pc.compose(layers, (160, 90), (1920, 1080), pc.StillSource())
    assert a.getpixel((80, 45))[3] == 128
    assert abs(b.getpixel((80, 45))[0] - 128) <= 1      # composed over black


def test_pygame_compositing_applies_opacity_without_changing_cached_surfaces(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")   # scoped: GL tests in subprocesses need the real driver
    import pygame
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        red = str(tmp_path / "r.png")
        Image.new("RGBA", (40, 30), (255, 0, 0, 255)).save(red)
        eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
        half = {0: {"path": red, "order": 0, "opacity": 0.5, "position": {}}}
        full = {0: {"path": red, "order": 0, "position": {}}}
        surf = eng._render_layers_to_surface(half, (40, 30))
        assert abs(surf.get_at((20, 15))[3] - 128) <= 1
        surf = eng._render_layers_to_surface(full, (40, 30))
        assert surf.get_at((20, 15))[3] == 255, "the cached surface must not keep the 50% alpha"
    finally:
        pygame.display.quit()


def test_ui_layers_carry_opacity_and_it_counts_as_an_edit(blind_ci):
    ci = blind_ci
    assert ci._get_layers_for_col(0)[0]["opacity"] == 1.0
    ci.cell_positions[(0, 0)] = {"opacity": 0.4}
    assert ci._get_layers_for_col(0)[0]["opacity"] == 0.4 and ci._live_edits_pending()


@pytest.mark.parametrize("mode", ["normal", "add", "multiply", "screen"])
def test_cpu_previews_blend_like_the_gpu(tmp_path, mode):
    from test_render_gl import _expected_blend
    base, top = str(tmp_path / "b.png"), str(tmp_path / "t.png")
    Image.new("RGBA", (200, 150), (100, 150, 200, 255)).save(base)
    Image.new("RGBA", (200, 150), (200, 100, 50, 255)).save(top)
    big = {"scale_x": 12.0, "scale_y": 12.0}
    layers = {0: {"path": base, "order": 0, "position": big},
              1: {"path": top, "order": 1, "position": big, "blend": mode, "opacity": 0.5}}
    want = _expected_blend(mode, 0.5)
    a = rem.BaseRenderEngine(media_manager=None, mode="opengl").render_layers_to_image(layers, (160, 90))
    b = pc.compose(layers, (160, 90), (1920, 1080), pc.StillSource())
    for img in (a, b):
        got = img.getpixel((80, 45))[:3]
        assert all(abs(g - w) <= 2 for g, w in zip(got, want)), (mode, got, want)


def test_unknown_blend_is_normal():
    assert rem.layer_blend({"blend": "dissolve"}) == "normal" and rem.layer_blend({}) == "normal"
