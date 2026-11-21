# SPDX-License-Identifier: MPL-2.0
"""Per-layer color correction (brightness, contrast, saturation): values, the
shared CPU math, the GPU shader, Pygame compositing, previews, UI layers."""
import numpy as np
import pytest
from PIL import Image

import preview_compose as pc
import render_engine as rem
from test_render_gl import run_gl_scenario
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401

BASE = (200, 100, 50)


def ref(rgb, b=0.0, c=1.0, s=1.0):
    """Straightforward float reference of the documented math."""
    x = np.array(rgb, dtype=np.float64) / 255.0
    x = np.clip((x - 0.5) * c + 0.5 + b, 0, 1)
    luma = x @ np.array([0.299, 0.587, 0.114])
    x = np.clip(luma + (x - luma) * s, 0, 1)
    return [round(v * 255) for v in x]


CASES = {"none": {}, "bright": {"b": 0.2}, "dark_contrast": {"b": -0.1, "c": 1.5}, "gray": {"s": 0.0},
         "vivid": {"s": 1.8}, "all": {"b": 0.1, "c": 0.6, "s": 1.4}}


@pytest.mark.parametrize("raw,expected", [({}, None), ({"brightness": 0}, None), ({"brightness": "x"}, None),
                                          ({"contrast": 5}, (0.0, 2.0, 1.0)), ({"saturation": -1}, (0.0, 1.0, 0.0)),
                                          ({"brightness": float("nan"), "saturation": 0.5}, (0.0, 1.0, 0.5))])
def test_color_values_are_clamped(raw, expected):
    assert rem.layer_color(raw) == expected


@pytest.mark.parametrize("name", list(CASES))
def test_cpu_math_matches_the_reference_in_both_channel_orders(name):
    k = CASES[name]
    adj = (k.get("b", 0.0), k.get("c", 1.0), k.get("s", 1.0))
    want = ref(BASE, **k)
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[...] = (*BASE, 77)
    out = rem.adjust_colors(rgba, None if adj == (0.0, 1.0, 1.0) else adj, order="rgb")
    assert all(abs(int(g) - w) <= 2 for g, w in zip(out[1, 1, :3], want)), (out[1, 1], want)
    assert out[1, 1, 3] == 77, "alpha must not change"
    bgr = np.ascontiguousarray(rgba[..., 2::-1])
    out = rem.adjust_colors(bgr, None if adj == (0.0, 1.0, 1.0) else adj, order="bgr")
    assert all(abs(int(g) - w) <= 2 for g, w in zip(out[1, 1, ::-1], want))


def test_gpu_shader_matches_the_cpu_math(fixtures, x_display):
    r = run_gl_scenario("color_adjust", fixtures, x_display)
    for name, k in CASES.items():
        want = ref(BASE, **k)
        assert all(abs(g - w) <= 3 for g, w in zip(r[name], want)), (name, r[name], want)
    semi = [round(v * 128 / 255) for v in ref(BASE, b=0.2)]           # straight alpha 50% over black
    assert all(abs(g - w) <= 3 for g, w in zip(r["semi_bright"], semi)), (r["semi_bright"], semi)
    assert r["static_cache_bypassed"]
    g = r["video_gray"]
    assert max(g) - min(g) <= 3 and 60 <= g[0] <= 90, g                 # red -> its luma, about 76


def test_pygame_and_previews_apply_color(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        p = str(tmp_path / "c.png")
        Image.new("RGBA", (40, 30), (*BASE, 255)).save(p)
        want = ref(BASE, b=0.1, c=0.6, s=1.4)
        adj = {"brightness": 0.1, "contrast": 0.6, "saturation": 1.4}
        eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
        surf = eng._render_layers_to_surface({0: {"path": p, "order": 0, "position": {}, **adj}}, (40, 30))
        got = list(surf.get_at((20, 15)))[:3]
        assert all(abs(g - w) <= 2 for g, w in zip(got, want)), (got, want)
        surf = eng._render_layers_to_surface({0: {"path": p, "order": 0, "position": {}}}, (40, 30))
        assert list(surf.get_at((20, 15)))[:3] == list(BASE), "unadjusted layer must not reuse the adjusted surface"
        # pygame_gpu video frames arrive as BGRA bytes
        arr = np.zeros((30, 40, 4), np.uint8)
        arr[...] = (BASE[2], BASE[1], BASE[0], 255)
        out = rem.adjust_colors(np.frombuffer(arr.tobytes(), np.uint8).reshape(30, 40, 4), (0.1, 0.6, 1.4))
        assert all(abs(int(g) - w) <= 2 for g, w in zip(out[5, 5, 2::-1], want))
    finally:
        pygame.display.quit()
    layers = {0: {"path": p, "order": 0, "position": {"scale_x": 50.0, "scale_y": 50.0}, **adj}}
    a = rem.BaseRenderEngine(media_manager=None, mode="opengl").render_layers_to_image(layers, (160, 90))
    b = pc.compose(layers, (160, 90), (1920, 1080), pc.StillSource())
    for img in (a, b):
        got = list(img.getpixel((80, 45)))[:3]
        assert all(abs(g - w) <= 2 for g, w in zip(got, want)), (got, want)


def test_ui_layers_carry_color_and_it_counts_as_an_edit(blind_ci):
    ci = blind_ci
    assert "brightness" not in ci._get_layers_for_col(0)[0]
    ci.cell_positions[(0, 0)] = {"saturation": 0.0}
    assert ci._get_layers_for_col(0)[0]["saturation"] == 0.0 and ci._live_edits_pending()


def test_editor_color_controls(tmp_path):
    import os
    import tkinter as tk
    from types import SimpleNamespace
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import editor
    import ui_theme
    p = str(tmp_path / "c.png")
    Image.new("RGBA", (40, 30), (*BASE, 255)).save(p)
    root = tk.Tk()
    root.withdraw()
    ui_theme.apply(root)
    try:
        live = []
        ed = editor.MediaEditor(SimpleNamespace(root=root), p, None, SimpleNamespace(monitors=[]),
                                initial_pos={"contrast": 1.5}, realtime_callback=lambda pos: live.append(dict(pos)))
        assert ed.color_vars["contrast"].get() == 150 and ed.color_vars["brightness"].get() == 0
        ed.color_vars["saturation"].set(0)
        ed._on_color_change("saturation")
        assert ed.current_pos["saturation"] == 0.0 and ed.color_labels["saturation"].cget("text") == "0"
        ed.color_vars["contrast"].set(100)
        ed._on_color_change("contrast")
        assert "contrast" not in ed.current_pos, "neutral values are not stored"
        ed._reset_color()
        assert not any(k in ed.current_pos for k in ("brightness", "contrast", "saturation"))
        ed._flush_realtime_callback()
        assert live and "saturation" not in live[-1]
        ed._cancel_and_close()
    finally:
        root.destroy()
