# SPDX-License-Identifier: MPL-2.0
"""Corner pin: the math, the GPU pass, the Pygame path, and the show data."""
import json

import numpy as np
import pytest

import output_warp as ow
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci, commands  # noqa: F401

WARP = ((0.1, 0.1), (0.9, 0.05), (1.0, 1.0), (0.0, 0.9))


def test_homography_maps_the_corners_and_inverts():
    h = ow.homography(WARP)
    for (u, v), (x, y) in zip(ow.IDENTITY, WARP):
        assert ow.apply(h, u, v) == pytest.approx((x, y), abs=1e-9)
    hi = ow.inverse(WARP)
    x, y = ow.apply(h, 0.3, 0.7)
    assert ow.apply(hi, x, y) == pytest.approx((0.3, 0.7), abs=1e-9)
    mid = ow.apply(h, 0.5, 0.5)                                       # straight lines stay straight:
    a, b = ow.apply(h, 0.0, 0.0), ow.apply(h, 1.0, 1.0)              # the center lies on the diagonal
    cross = (b[0] - a[0]) * (mid[1] - a[1]) - (b[1] - a[1]) * (mid[0] - a[0])
    assert abs(cross) < 1e-9


@pytest.mark.parametrize("corners,ok", [
    (WARP, True), (ow.IDENTITY, False), (((0, 0), (1, 0), (0, 1), (1, 1)), False),     # crossed (bow tie)
    (((0, 0), (1, 0), (0.6, 0.4), (0, 1)), False),                                    # folded in
    (((0, 0), (1, 0), (1, 1)), False), ("junk", False), (((0, 0), (9, 0), (1, 1), (0, 1)), False),
    (((-0.5, -0.5), (1.5, -0.5), (1.5, 1.5), (-0.5, 1.5)), True),                     # larger than the output
])
def test_clean_accepts_only_convex_non_identity_quads(corners, ok):
    assert (ow.clean(corners) is not None) == ok


def test_cpu_warp_places_the_picture_like_the_math():
    pic = np.zeros((100, 200, 4), np.uint8)
    pic[...] = (0, 0, 255, 255)
    pic[:50, :100] = (255, 0, 0, 255)
    out = ow.warp_image(pic, WARP)
    h = ow.homography(WARP)
    for (u, v), want in (((0.25, 0.25), (255, 0, 0)), ((0.75, 0.75), (0, 0, 255))):
        x, y = ow.apply(h, u, v)
        assert tuple(out[int(y * 100), int(x * 200), :3]) == want
    assert tuple(out[2, 2]) == (0, 0, 0, 255)                         # outside: opaque black


def test_gpu_corner_pin(fixtures, x_display):
    from test_render_gl import run_gl_scenario
    r = run_gl_scenario("corner_pin", fixtures, x_display)
    red, blue, black = [255, 0, 0], [0, 0, 255], [0, 0, 0]
    close = lambda a, b: all(abs(x - y) <= 8 for x, y in zip(a, b))  # noqa: E731
    assert close(r["plain"]["tl_quadrant"], red) and close(r["plain"]["outside_top_left"], red)
    p = r["pinned"]
    assert close(p["tl_quadrant"], red) and close(p["br_quadrant"], blue), p
    assert close(p["outside_top_left"], black) and close(p["outside_bottom_left"], black), p
    assert r["video_center"][0] > 200 and close(r["video_outside"], black), r
    assert r["targets"] == 1 and r["targets_after_reset"] == 0


def test_pygame_engine_pins_without_touching_cached_surfaces(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    import render_engine as rem
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
        surf = pygame.Surface((200, 100), pygame.SRCALPHA)
        surf.fill((0, 0, 255, 255))
        surf.fill((255, 0, 0, 255), pygame.Rect(0, 0, 100, 50))
        assert eng._warp_surface(surf, 0) is surf                     # no warp: the same surface
        eng.set_output_warps({0: WARP, 1: ow.IDENTITY, "x": WARP})
        assert list(eng.output_warps) == [0]
        out = eng._warp_surface(surf, 0)
        assert out is not surf and tuple(surf.get_at((2, 2)))[:3] == (255, 0, 0)
        assert tuple(out.get_at((2, 2)))[:3] == (0, 0, 0)
        x, y = ow.apply(ow.homography(WARP), 0.75, 0.75)
        assert tuple(out.get_at((int(x * 200), int(y * 100))))[:3] == (0, 0, 255)
    finally:
        pygame.display.quit()


def test_pygame_warp_reads_the_surface_without_converting_it(monkeypatch):
    """The warp reads an ARGB8888 surface (every composed frame) through a view of
    its memory instead of pygame.image.tobytes, which took 7.6 ms per 1080p frame,
    about half of a corner pin's cost in Balanced Mode on the test VM (2026-11-16).
    Other formats are converted; the source is left unlocked and unchanged; the
    result never shares the source's memory."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import numpy as np
    import pygame
    import render_engine as rem
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
        argb = pygame.Surface((200, 100), pygame.SRCALPHA)
        argb.fill((200, 100, 50, 255))
        argb.fill((10, 20, 230, 128), pygame.Rect(0, 0, 20, 10))
        plain = pygame.Surface((200, 100), 0, 24)
        plain.fill((200, 100, 50))
        part = pygame.Surface((240, 100), pygame.SRCALPHA)
        part.fill((200, 100, 50, 255))
        sub = part.subsurface(pygame.Rect(20, 0, 200, 100))            # its pitch is wider than its width
        for s in (argb, plain, sub):
            arr = eng._bgra_pixels(s)
            want = np.frombuffer(pygame.image.tobytes(s, "BGRA"), np.uint8).reshape(100, 200, 4)
            assert arr.shape == (100, 200, 4) and np.array_equal(arr, want)
            del arr
            assert not s.get_locked()
        eng.set_output_blends({0: {"right": 0.1}})
        for s in (argb, plain):
            before = pygame.image.tobytes(s, "BGRA")
            out = eng._warp_surface(s, 0)
            assert not s.get_locked() and pygame.image.tobytes(s, "BGRA") == before
            assert out.get_masks() == argb.get_masks() and tuple(out.get_at((50, 50))) == (200, 100, 50, 255)
        eng.set_output_blends({})
        eng.set_output_warps({0: WARP})
        out = eng._warp_surface(argb, 0)
        assert not argb.get_locked() and tuple(argb.get_at((2, 2))) == (10, 20, 230, 128)
        x, y = ow.apply(ow.homography(WARP), 0.5, 0.75)
        assert tuple(out.get_at((int(x * 200), int(y * 100))))[:3] == (200, 100, 50)
        argb.fill((0, 0, 0, 255))                                         # the result is its own copy
        assert tuple(out.get_at((int(x * 200), int(y * 100))))[:3] == (200, 100, 50)
        win = pygame.Surface((200, 100))
        win.blit(argb, (0, 0))                                            # the source can still be drawn
        win.blit(out, (0, 0))
    finally:
        pygame.display.quit()


def test_pin_tables_give_the_same_picture_as_warp_perspective():
    import cv2
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur((rng.random((90, 160, 4)) * 255).astype(np.uint8), (0, 0), 2)
    for corners in (WARP, ((0.45, 0.4), (0.55, 0.4), (1.0, 1.0), (0.0, 1.0))):    # the second: a horizon
        a = ow.warp_image(img, corners)
        b = ow.warp_with_maps(img, ow.inverse_maps(corners, 160, 90))
        d = np.abs(a.astype(int) - b.astype(int))
        assert d.mean() < 0.01 and (d > 2).sum() <= 0.001 * d.size, (d.max(), (d > 2).sum())


def test_pygame_pin_uses_tables_built_in_the_background(monkeypatch):
    """The pin is warped with cv2.warpPerspective until its remap tables are
    built in the background, then with the tables (half the cost); a moved
    corner is drawn at once, a removed pin drops its tables."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    import render_engine as rem
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
        surf = pygame.Surface((200, 100), pygame.SRCALPHA)
        surf.fill((0, 0, 255, 255))
        surf.fill((255, 0, 0, 255), pygame.Rect(0, 0, 100, 50))
        eng.set_output_warps({0: WARP})
        first = eng._warp_surface(surf, 0)                           # no tables yet: warpPerspective
        eng._pin_jobs[0].join(5)
        assert eng._pin_maps[0][0] == (eng.output_warps[0], 200, 100)
        second = eng._warp_surface(surf, 0)                          # with the tables
        x, y = ow.apply(ow.homography(WARP), 0.75, 0.75)
        for out in (first, second):
            assert tuple(out.get_at((2, 2)))[:3] == (0, 0, 0)
            assert tuple(out.get_at((int(x * 200), int(y * 100))))[:3] == (0, 0, 255)
        moved = ((0.3, 0.1), (0.9, 0.05), (1.0, 1.0), (0.0, 0.9))
        eng.set_output_warps({0: moved})
        out = eng._warp_surface(surf, 0)                             # the new shape at once
        x, y = ow.apply(ow.homography(moved), 0.02, 0.02)
        assert tuple(out.get_at((int(x * 200) + 1, int(y * 100) + 1)))[:3] == (255, 0, 0)
        eng._pin_jobs[0].join(5)
        assert eng._pin_maps[0][0][0] == eng.output_warps[0]
        eng.set_output_warps({})
        assert eng._pin_maps == {} and eng._warp_surface(surf, 0) is surf
    finally:
        pygame.display.quit()


def test_warps_are_saved_with_the_show_and_sent_to_the_renderer(blind_ci, tmp_path):
    ci = blind_ci
    for name in ("_build_media_list", "_build_grid", "_initialize_render_engine", "_on_toggle_presentation_mode"):
        setattr(ci, name, lambda *a, **k: None)
    show = str(tmp_path / "s.vjdeck")
    ci._perform_save(show)
    assert "warps" not in json.load(open(show))["outputs"], "nothing written when no output is pinned"
    commands(ci)
    ci._set_output_warp(0, WARP)
    sent = commands(ci, "set_warps")
    assert sent and sent[-1]["warps"] == {"0": [list(p) for p in WARP]}
    show = str(tmp_path / "s2.vjdeck")          # a new file: the base fixture forbids the backup copy
    ci._perform_save(show)
    data = json.load(open(show))
    assert data["outputs"]["warps"] == {"0": [list(p) for p in WARP]}
    data["outputs"]["warps"]["1"] = [[0, 0], [1, 0], [0, 1], [1, 1]]  # a crossed shape from a hand edit
    ci._apply_project_data(data, show)
    assert ci.output_warps == {"0": [list(p) for p in WARP]}
    ci._set_output_warp(0, None)
    assert ci.output_warps == {} and commands(ci, "set_warps")[-1]["warps"] == {}
    ci._set_output_warp(0, ((0, 0), (1, 0), (0, 1), (1, 1)))           # refused: would fold the picture
    assert ci.output_warps == {}


def test_corner_pin_dialog(blind_ci):
    import os
    from types import SimpleNamespace
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import ui_theme
    ci = blind_ci
    root = tk.Tk()
    root.withdraw()
    ui_theme.apply(root)
    ci.root = root
    ci.re.monitors = [SimpleNamespace(x=0, y=0, width=1920, height=1080),
                      SimpleNamespace(x=1920, y=0, width=1280, height=1024)]
    ci.selected_display = 0
    try:
        win = ci._open_corner_pin()
        vd = win._vd
        M, CW, h = vd["M"], vd["CW"], vd["ch"]()
        ev = lambda x, y, state=0: SimpleNamespace(x=x, y=y, state=state)  # noqa: E731
        vd["press"](ev(M + 3, M + 2))                                  # grab the top-left corner
        vd["motion"](ev(M + 0.1 * CW, M + 0.1 * h))
        assert ci.output_warps["0"][0] == pytest.approx([0.1, 0.1], abs=1e-3)
        assert commands(ci, "set_warps")[-1]["warps"]["0"][0] == pytest.approx([0.1, 0.1], abs=1e-3)
        vd["nudge"](1, 0, ev(0, 0, state=0x1))                         # Shift+Right: 10 px
        assert ci.output_warps["0"][0][0] == pytest.approx(0.1 + 10 / 1920, abs=1e-3)
        assert "top left" in vd["readout"].cget("text") and "x    202 px" in vd["readout"].cget("text")
        before = [list(p) for p in ci.output_warps["0"]]
        vd["motion"](ev(M + 1.3 * CW, M + 1.3 * h))                    # past the opposite corner: a fold
        assert ci.output_warps["0"] == before and "fold" in vd["note"].cget("text")
        vd["reset"]()
        assert ci.output_warps == {} and commands(ci, "set_warps")[-1]["warps"] == {}
        win.destroy()
    finally:
        root.destroy()


def test_corner_pin_arrow_keys_after_a_real_click(blind_ci):
    """A click on a corner leaves the keyboard on the canvas, so the arrow keys nudge it.

    The main window's click handler (it gives space-to-GO back after a click outside an
    entry) moved the focus to the main window after every click, also in this dialog,
    so the arrow keys did nothing (found on the owner's PC, 2026-11-16)."""
    import os
    from types import SimpleNamespace
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import ui_theme
    from control_interface import ControlInterface
    ci = blind_ci
    root = tk.Tk()
    ui_theme.apply(root)
    ci.root = root
    ci._is_input_like_widget = ControlInterface._is_input_like_widget.__get__(ci)
    root.bind_all("<Button-1>", ControlInterface._on_click_maybe_clear_focus.__get__(ci), add='+')
    ci.re.monitors = [SimpleNamespace(x=0, y=0, width=1920, height=1080)]
    ci.selected_display = 0
    try:
        win = ci._open_corner_pin()
        vd = win._vd
        canvas = vd["canvas"]
        root.update()
        win.focus_force()
        root.update()
        M = vd["M"]
        canvas.event_generate("<ButtonPress-1>", x=M + 2, y=M + 2)   # the top-left corner
        canvas.event_generate("<ButtonRelease-1>", x=M + 2, y=M + 2)
        root.update()
        root.after(50, lambda: None)
        root.update()
        assert root.focus_get() is canvas
        canvas.event_generate("<Shift-Right>")
        root.update()
        assert ci.output_warps["0"][0][0] == pytest.approx(10 / 1920, abs=1e-4)
        win.destroy()
    finally:
        root.destroy()
