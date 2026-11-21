# SPDX-License-Identifier: MPL-2.0
"""Transition types: Fade (straight), Smooth Fade (ease in/out), Dip to Black."""
import pytest

import project_schema as ps
import render_engine as rem
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401


@pytest.mark.parametrize("curve,p,want", [
    ("linear", 0.25, (0.75, 0.25)), ("smooth", 0.25, (0.84375, 0.15625)), ("smooth", 0.5, (0.5, 0.5)),
    ("dip", 0.25, (0.5, 0.0)), ("dip", 0.5, (0.0, 0.0)), ("dip", 0.75, (0.0, 0.5)),
    ("linear", 1.0, (0.0, 1.0)), ("linear", 2.0, (0.0, 1.0)), ("nonsense", 0.4, (0.6, 0.4)),
])
def test_alphas(curve, p, want):
    assert rem.transition_alphas({"progress": p, "curve": curve}) == pytest.approx(want)


def expected(curve, p):
    fa, ta = rem.transition_alphas({"progress": p, "curve": curve})
    return [round(255 * fa), round(255 * ta), 0]                  # each cue in its own buffer, then mixed


def test_gpu_draws_each_curve(fixtures, x_display):
    from test_render_gl import run_gl_scenario
    r = run_gl_scenario("transition_curves", fixtures, x_display)
    assert all(abs(g - w) <= 3 for g, w in zip(r.pop("pinned_mix_center"), [128, 128, 0]))
    assert r.pop("pinned_mix_outside") == [0, 0, 0]
    for key, got in r.items():
        curve, p = key.split("@")
        want = expected(curve, float(p))
        assert all(abs(g - w) <= 3 for g, w in zip(got, want)), (key, got, want)


def test_cues_and_show_default_choose_the_curve(blind_ci):
    ci = blind_ci
    ci.transition_var.set("Smooth Fade")
    ci._initiate_transition(0, 1)
    assert ci.transition_thread.curves[-1] == "smooth"
    ci._finalize_transition_from_thread(1)
    ci.cue_settings = {"2": {"transition": "Dip to Black"}}
    ci._initiate_transition(1, 2)
    assert ci.transition_thread.curves[-1] == "dip"
    ci._finalize_transition_from_thread(2)
    ci.transition_var.set("Dip to Black")
    ci._blackout()                                   # to black: a dip would only use half the time
    assert ci.transition_thread.curves[-1] == "linear"


def test_cue_settings_form_and_schema_keep_the_new_types():
    import control_interface as ci_mod
    tchoices = ["Show default (Fade, 1 s)", "None (cut)", "Fade", "Smooth Fade", "Dip to Black"]
    out = ci_mod.ControlInterface._cue_settings_from_form("Dip to Black", tchoices, "", "Wait for GO",
                                                         ["Wait for GO", "a", "b"], "5", "")
    assert out["transition"] == "Dip to Black"
    assert ps.clean_cue_settings({"0": {"transition": "Smooth Fade"}, "1": {"transition": "Wipe"}}) == \
        {"0": {"transition": "Smooth Fade"}}


def test_pygame_crossfade_keeps_brightness(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
        red, green = pygame.Surface((40, 30), pygame.SRCALPHA), pygame.Surface((40, 30), pygame.SRCALPHA)
        red.fill((255, 0, 0, 255))
        green.fill((0, 255, 0, 255))
        for curve, p in (("linear", 0.5), ("smooth", 0.25), ("dip", 0.25), ("dip", 0.75)):
            fa, ta = rem.transition_alphas({"progress": p, "curve": curve})
            out = eng.get_transition_frame(red, green, p, (fa, ta))
            on_black = pygame.Surface((40, 30))
            on_black.fill((0, 0, 0))
            on_black.blit(out, (0, 0))                                # the output window is black underneath
            got = tuple(on_black.get_at((20, 15)))[:3]
            assert all(abs(g - w) <= 2 for g, w in zip(got, (255 * fa, 255 * ta, 0))), (curve, p, got)
        assert tuple(red.get_at((0, 0))) == (255, 0, 0, 255), "inputs are not changed"
    finally:
        pygame.display.quit()
