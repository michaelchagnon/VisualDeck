# SPDX-License-Identifier: MPL-2.0
"""Balanced and Compatibility Mode: the preview is a small copy of the output window."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def pg(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.display.init()
    yield pygame
    pygame.display.quit()


def _quadrants(pygame, w, h):
    win = pygame.display.set_mode((w, h))
    win.fill((0, 0, 255))
    win.fill((255, 0, 0), pygame.Rect(0, 0, w // 2, h // 2))
    win.fill((0, 255, 0), pygame.Rect(w // 2, 0, w - w // 2, h // 2))
    return win


def test_the_preview_is_the_window_scaled_down(pg):
    import render_engine as rem
    eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
    assert eng.window_preview_image(160, 90) is None            # no output window
    eng.window = _quadrants(pg, 320, 180)
    img = eng.window_preview_image(160, 90)
    assert img.size == (160, 90) and img.mode == "RGB"
    assert img.getpixel((20, 20)) == (255, 0, 0) and img.getpixel((140, 20)) == (0, 255, 0)
    assert img.getpixel((80, 80)) == (0, 0, 255)
    assert not eng.window.get_locked()                          # the window can be drawn again
    eng.window.fill((0, 0, 0))
    assert eng.window_preview_image(640, 360).size == (320, 180)   # never larger than the window


def test_one_display_previews_the_window_and_other_layouts_compose(pg):
    """The render thread takes the window's picture for one display (it is what
    the output shows, pin and fades included); a span or several displays are
    still composed from the layers."""
    import control_interface as ci_mod
    import render_engine as rem
    eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
    eng.monitors = [SimpleNamespace(x=0, y=0, width=320, height=180)]
    eng.window = _quadrants(pg, 320, 180)
    composed = []
    eng.render_layers_to_image = lambda layers, dims: composed.append(dims)
    rt = SimpleNamespace(re=eng, _preview_cap_scale=1.0,
                         current_render_state={'layers': {}, 'output_mode': 'Display 0', 'target_monitor': 0})
    img = ci_mod.RenderThread._cpu_preview(rt)
    assert img.size == (320, 180) and img.getpixel((10, 10)) == (255, 0, 0) and composed == []
    rt.current_render_state = {'layers': {0: {'path': 'x'}}, 'output_mode': ci_mod.ControlInterface.MULTIPLE,
                               'target_monitor': None}
    ci_mod.RenderThread._cpu_preview(rt)
    assert composed, "several displays: composed from the layers"
    eng.window = None
    composed.clear()
    rt.current_render_state = {'layers': {0: {'path': 'x'}}, 'output_mode': 'Display 0', 'target_monitor': 0}
    ci_mod.RenderThread._cpu_preview(rt)
    assert composed, "no window: composed"


def test_the_first_layer_drawn_with_max_gives_the_same_pixels(pg):
    """The first layer on the empty canvas is drawn with BLEND_RGBA_MAX (less than
    half the cost of alpha blending); the picture must be exactly what alpha
    blending gave: every alpha value, a layer partly off the canvas, a second
    layer on top, and a first layer with opacity (blended as before)."""
    import numpy as np
    import render_engine as rem
    pg.display.set_mode((1, 1))
    rng = np.random.default_rng(3)

    def layer_surface(w, h):
        a = (rng.random((h, w, 4)) * 255).astype(np.uint8)
        a[:4, :4, 3] = 0                                    # fully transparent corner
        return pg.image.frombuffer(a.copy(), (w, h), "BGRA").copy()

    eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
    first, second = layer_surface(120, 70), layer_surface(60, 40)
    surfaces = {"a": first, "b": second}
    eng._to_surface = lambda m: surfaces[m["path"]]

    def reference(layers, dims):
        out = pg.Surface(dims, pg.SRCALPHA)
        out.fill((0, 0, 0, 0))
        for m in sorted(layers.values(), key=lambda x: x["order"]):
            sf = surfaces[m["path"]]
            p = m["position"]
            rect = sf.get_rect(center=(int(p["x"] * dims[0]), int(p["y"] * dims[1])))
            op = m.get("opacity", 1.0)
            if op < 1.0:
                sf = sf.copy()
                sf.set_alpha(int(op * 255 + 0.5))
            out.blit(sf, rect)
        return pg.image.tobytes(out, "RGBA")

    cases = [
        {0: {"path": "a", "order": 0, "position": {"x": 0.5, "y": 0.5}}},
        {0: {"path": "a", "order": 0, "position": {"x": 0.05, "y": 0.9}}},            # partly off the canvas
        {0: {"path": "a", "order": 0, "position": {"x": 0.5, "y": 0.5}},
         1: {"path": "b", "order": 1, "position": {"x": 0.6, "y": 0.4}}},
        {0: {"path": "a", "order": 0, "position": {"x": 0.5, "y": 0.5}, "opacity": 0.5},
         1: {"path": "b", "order": 1, "position": {"x": 0.6, "y": 0.4}}},
    ]
    for layers in cases:
        got = eng._render_layers_to_surface(layers, (160, 90))
        assert pg.image.tobytes(got, "RGBA") == reference(layers, (160, 90)), layers
