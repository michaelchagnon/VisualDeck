# SPDX-License-Identifier: MPL-2.0
"""Edge blending: the maths, the GPU pass, the Pygame path, show data and the dialog."""
import json
import os

import numpy as np
import pytest

import edge_blend as eb
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci, commands  # noqa: F401

BLEND = {"left": 0.25, "right": 0.1, "top": 0.0, "bottom": 0.2, "gamma": 2.2, "curve": 2.0}


def test_two_overlapping_ramps_add_up_to_full_light():
    t = np.linspace(0, 1, 101)
    for curve in (1.0, 2.0, 3.0):
        for gamma in (1.0, 2.2):
            light = eb.ramp(t, curve, gamma) ** gamma + eb.ramp(1 - t, curve, gamma) ** gamma
            assert np.allclose(light, 1.0), (curve, gamma)
    assert float(eb.ramp(0.0)) == 0.0 and float(eb.ramp(1.0)) == 1.0 and 0 < float(eb.ramp(0.5)) < 1


@pytest.mark.parametrize("raw,want", [
    ({}, None), ({"left": 0}, None), ("junk", None), ({"left": 0.1}, (0.1, 0, 0, 0, 2.0, 2.2)),
    ({"left": 9, "gamma": 0, "curve": 99}, (0.5, 0, 0, 0, 4.0, 1.0)), ({"left": "x", "right": 0.2}, (0, 0.2, 0, 0, 2.0, 2.2)),
])
def test_clean(raw, want):
    assert eb.clean(raw) == want


def test_cpu_mask_matches_the_factor_function():
    b = eb.clean(BLEND)
    img = np.full((40, 200, 4), 255, np.uint8)
    out = eb.apply(img, b)
    for x in (0, 10, 25, 49, 50, 120, 190, 199):
        want = eb.factor((x + 0.5) / 200, 0.5, b) * 255
        assert abs(int(out[20, x, 0]) - want) <= 1, (x, out[20, x, 0], want)
    assert (out[..., 3] == 255).all() and out[20, 100, 0] == 255


def test_gpu_edge_blend(fixtures, x_display):
    from test_render_gl import run_gl_scenario
    r = run_gl_scenario("edge_blend", fixtures, x_display)
    b = eb.clean(BLEND)
    for key, got in r["points"].items():
        x, y = map(float, key.split(","))
        want = eb.factor(x, y, b) * 255
        assert abs(got - want) <= 4, (key, got, want)
    assert r["pinned_outside"] == [0, 0, 0] and r["pinned_center"][0] >= 250
    assert r["targets"] == 1 and r["targets_after"] == 0


def test_pygame_applies_the_blend(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    import render_engine as rem
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
        surf = pygame.Surface((200, 100), pygame.SRCALPHA)
        surf.fill((255, 255, 255, 255))
        eng.set_output_blends({0: BLEND, 1: {"left": 0}})
        assert list(eng.output_blends) == [0]
        out = eng._warp_surface(surf, 0)
        assert out is not surf and tuple(surf.get_at((0, 50)))[:3] == (255, 255, 255)
        want = eb.factor(10.5 / 200, 50.5 / 100, eb.clean(BLEND)) * 255
        assert abs(out.get_at((10, 50))[0] - want) <= 1 and out.get_at((100, 30))[0] == 255
    finally:
        pygame.display.quit()


def test_pygame_blend_keeps_colors_and_the_fast_pixel_format(monkeypatch):
    """An edge blend (or pin) in Balanced Mode made a surface in RGBA byte order,
    whose blit to the window has no fast path in SDL: the 4K clip went from 50 to
    15 renders/s on the owner's PC (2026-11-15). The result must be in pygame's own
    order for a surface with alpha and keep red, green and blue apart."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    import render_engine as rem
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
        surf = pygame.Surface((200, 100), pygame.SRCALPHA)
        surf.fill((200, 100, 50, 255))
        surf.fill((10, 20, 230, 128), pygame.Rect(150, 0, 50, 100))
        eng.set_output_blends({0: {"right": 0.1}})
        out = eng._warp_surface(surf, 0)
        assert out.get_masks() == surf.get_masks(), (out.get_masks(), surf.get_masks())
        assert tuple(out.get_at((50, 50))) == (200, 100, 50, 255)        # outside the blend: unchanged
        r, g, b, a = out.get_at((195, 50))                                  # inside: dimmed, alpha kept
        assert a == 128 and r < 10 and g < 20 and 0 < b < 230 and b > 10 * g / 20
    finally:
        pygame.display.quit()


def test_show_data_commands_and_dialog(blind_ci, tmp_path):
    tk = pytest.importorskip("tkinter")
    from types import SimpleNamespace
    ci = blind_ci
    for name in ("_build_media_list", "_build_grid", "_initialize_render_engine", "_on_toggle_presentation_mode"):
        setattr(ci, name, lambda *a, **k: None)
    commands(ci)
    ci._set_output_blend(0, BLEND)
    sent = commands(ci, "set_blends")
    assert sent[-1]["blends"]["0"]["left"] == 0.25
    show = str(tmp_path / "s.vjdeck")
    ci._perform_save(show)
    data = json.load(open(show))
    assert data["outputs"]["blends"]["0"]["bottom"] == 0.2
    ci._apply_project_data(data, show)
    assert ci.output_blends["0"]["right"] == 0.1
    ci._set_output_blend(0, {"left": 0})
    assert ci.output_blends == {} and commands(ci, "set_blends")[-1]["blends"] == {}
    if not os.environ.get("DISPLAY") and os.name != "nt":
        return
    root = tk.Tk()
    root.withdraw()
    try:
        ci.root = root
        ci.re.monitors = [SimpleNamespace(x=0, y=0, width=1920, height=1080)]
        win = ci._open_corner_pin(0)
        v = win._vd["blend_vars"]
        v["left"].set("480")
        v["bottom"].set("10x")                                      # half-typed: ignored
        assert ci.output_blends["0"]["left"] == pytest.approx(0.25) and ci.output_blends["0"]["bottom"] == 0
        v["bottom"].set("108")
        assert ci.output_blends["0"]["bottom"] == pytest.approx(0.1)
        win.destroy()
    finally:
        root.destroy()


def test_first_frame_with_a_blend_after_one_display_draws_and_does_not_crash(fixtures, x_display):
    """Found while testing spans: switching from one display to two with an edge
    blend crashed in Performance Mode (segmentation fault in the first frame)."""
    from test_render_gl import run_gl_scenario
    r = run_gl_scenario("warp_first_frame", fixtures, x_display)
    assert r["single"] == [200, 100, 50]
    assert r["first_frame"] == [[200, 100, 50], [200, 100, 50]], r
