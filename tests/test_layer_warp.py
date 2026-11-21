# SPDX-License-Identifier: MPL-2.0
"""Warp points on a layer (layer_warp.py): the surface the engines draw and the CPU bend."""
import numpy as np
import pytest

from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401

import layer_warp
import mesh_warp


def _pinned_right(dx):
    """2 x 2 warp with the right edge moved right by dx (fraction of the width)."""
    return layer_warp.clean({"cols": 2, "rows": 2, "points": [[0, 0], [1 + dx, 0], [0, 1], [1 + dx, 1]]})


def test_clean_and_saved_form():
    assert layer_warp.clean({"cols": 2, "rows": 2, "points": mesh_warp.identity(2, 2)}) is None   # nothing bent
    assert layer_warp.clean({"cols": 2, "rows": 2, "points": [[0, 0], [1, 0]]}) is None          # malformed
    m = _pinned_right(0.5)
    assert m is not None and layer_warp.to_dict(m)["points"][1] == [1.5, 0.0]
    assert layer_warp.layer_warp({"warp": layer_warp.to_dict(m)}) == m
    assert layer_warp.layer_warp({}) is None and layer_warp.layer_warp(None) is None


def test_gl_vertices_cover_the_layer_quad():
    m = _pinned_right(0.5)
    v = layer_warp.gl_vertices(m, segments=4)
    assert v.shape[1] == 5 and len(v) % 3 == 0
    xs, ys, us, vs = v[:, 0], v[:, 1], v[:, 3], v[:, 4]
    assert np.isclose(xs.min(), -0.5) and np.isclose(xs.max(), 1.0)       # the right edge at 1.5 - 0.5
    assert np.isclose(ys.min(), -0.5) and np.isclose(ys.max(), 0.5)       # y up, as the quad
    assert np.isclose(us.min(), 0) and np.isclose(us.max(), 1) and np.isclose(vs.min(), 0) and np.isclose(vs.max(), 1)
    # The top-left vertex (u, v) = (0, 0) sits at the quad's top-left (-0.5, +0.5).
    tl = v[(v[:, 3] == 0) & (v[:, 4] == 0)][0]
    assert np.allclose(tl[:2], [-0.5, 0.5])


def test_warp_array_bends_around_the_center():
    pic = np.zeros((40, 80, 4), np.uint8)
    pic[..., 2] = 255                                    # red in BGRA
    pic[..., 3] = 255
    out = layer_warp.warp_array(pic, _pinned_right(0.5))
    h, w = out.shape[:2]
    assert h >= 40 and w >= 80 + 2 * 40 - 2          # reaches 1.5 widths right of the left edge, centered
    cx, cy = w // 2, h // 2
    # The left edge of the picture stays at center - 40; it now reaches center + 80.
    assert out[cy, cx - 38, 3] == 255 and out[cy, cx + 78, 3] == 255
    assert out[cy, cx - 44, 3] == 0                   # left of the picture: transparent
    assert out[2, 2, 3] == 0 if h > 44 else True
    # A keystone: the top-right corner pulled in; the top edge's right end is transparent.
    m = layer_warp.clean({"cols": 2, "rows": 2, "points": [[0, 0], [0.7, 0.2], [0, 1], [1, 1]]})
    out = layer_warp.warp_array(pic, m)
    oh, ow = out.shape[:2]
    oy, ox = (oh - 40) // 2, (ow - 80) // 2
    assert out[oy + 1, ox + 78, 3] == 0 and out[oy + 38, ox + 70, 3] == 255
    with pytest.raises(ValueError):
        layer_warp.warp_array(pic[..., :3], m)


def test_maps_are_cached_and_big_pictures_use_half_size_maps(monkeypatch):
    calls = []
    real = mesh_warp.inverse_map

    def spy(mesh, corners, w, h, **kw):
        calls.append((w, h))
        return real(mesh, corners, w, h, **kw)
    monkeypatch.setattr(mesh_warp, "inverse_map", spy)
    layer_warp._maps.clear()
    m = _pinned_right(0.25)
    pic = np.full((300, 900, 4), 255, np.uint8)
    a = layer_warp.warp_array(pic, m)
    b = layer_warp.warp_array(pic, m)
    assert len(calls) == 1 and a.shape == b.shape                  # built once
    assert calls[0][0] < a.shape[1] * 0.6                           # half size, scaled up
    assert a[150, a.shape[1] // 2, 3] == 255


def test_opengl_draws_the_bent_layer(fixtures, x_display):
    from conftest import run_gl_scenario
    r = run_gl_scenario("layer_warp", fixtures, x_display)

    def close(a, b, tol=6):
        return all(abs(x - y) <= tol for x, y in zip(a, b))
    assert close(r["plain"]["inside"], [200, 100, 50]) and r["plain"]["right_of"] == [0, 0, 0]
    assert close(r["pulled"]["inside"], [200, 100, 50]) and close(r["pulled"]["right_of"], [200, 100, 50])
    assert r["pulled"]["beyond"] == [0, 0, 0] and r["static_cache_bypassed"]
    k = r["keystone"]
    assert close(k["top_left"], [200, 100, 50]) and k["top_right"] == [0, 0, 0]
    assert close(k["bottom_right"], [200, 100, 50])
    assert close(r["masked"]["left_part"], [200, 100, 50]) and r["masked"]["right_part"] == [0, 0, 0]
    assert r["video_right_of"][0] > 200 and r["video_right_of"][1] < 40
    assert r["geo_cached"] >= 1 and r["geo_after_close"] == 0


def test_balanced_and_compatibility_mode_bend_the_layer(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    import render_engine as rem
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        p = str(tmp_path / "w.png")
        Image.new("RGBA", (40, 20), (255, 255, 255, 255)).save(p)
        eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
        base = {"path": p, "order": 0, "position": {"x": 0.5, "y": 0.5}}
        bent = dict(base, warp={"cols": 2, "rows": 2, "points": [[0, 0], [1.5, 0], [0, 1], [1.5, 1]]})
        surf = eng._render_layers_to_surface({0: base}, (100, 40))
        assert surf.get_at((50, 20))[3] == 255 and surf.get_at((85, 20))[3] == 0
        surf = eng._render_layers_to_surface({0: bent}, (100, 40))
        assert surf.get_at((50, 20))[3] == 255 and surf.get_at((85, 20))[3] == 255     # reaches further right
        assert surf.get_at((95, 20))[3] == 0 and surf.get_at((25, 20))[3] == 0         # left edge unchanged
        surf = eng._render_layers_to_surface({0: base}, (100, 40))
        assert surf.get_at((85, 20))[3] == 0, "the plain layer must not reuse the bent surface"
        # The previews bend it too.
        prev = rem.BaseRenderEngine(media_manager=None, mode="opengl")
        big = {"x": 0.5, "y": 0.5, "scale_x": 24.0, "scale_y": 24.0}      # half of a 1920 px output
        cover = lambda L: sum(1 for p in prev.render_layers_to_image(L, (100, 40)).convert("RGB").getdata()
                              if p[0] > 128)
        assert cover({0: dict(bent, position=big)}) > cover({0: dict(base, position=big)}) * 1.3
    finally:
        pygame.display.quit()


def test_ui_layers_carry_the_warp(blind_ci):
    ci = blind_ci
    w = {"cols": 2, "rows": 2, "points": [[0, 0], [1.2, 0], [0, 1], [1, 1]]}
    ci.cell_positions[(0, 0)] = {"x": 0.5, "y": 0.5, "warp": w}
    ci.grid_cells[(0, 0)] = "/tmp/x.png"
    layer = ci._get_layers_for_col(0)[0]
    assert layer["warp"] == w and layer_warp.layer_warp(layer) is not None


def test_cpu_bend_with_fewer_triangles_matches_the_fine_one(monkeypatch):
    """MAP_SEGMENTS (16 per axis, for speed) against 64: a curved 3 x 3 warp of a
    gradient picture differs by at most a few levels inside the picture."""
    pic = np.zeros((120, 200, 4), np.uint8)
    pic[..., 0] = np.linspace(0, 255, 200, dtype=np.uint8)[None, :]
    pic[..., 1] = np.linspace(0, 255, 120, dtype=np.uint8)[:, None]
    pic[..., 3] = 255
    m = layer_warp.clean({"cols": 3, "rows": 3, "points": [[0, 0], [0.5, -0.1], [1, 0], [0.05, 0.5], [0.6, 0.55],
                                                           [1.1, 0.5], [0, 1], [0.5, 1.1], [1, 1]]})
    layer_warp._maps.clear()
    coarse = layer_warp.warp_array(pic, m).astype(int)
    layer_warp._maps.clear()
    monkeypatch.setattr(layer_warp, "MAP_SEGMENTS", 64)
    fine = layer_warp.warp_array(pic, m).astype(int)
    layer_warp._maps.clear()
    inside = (coarse[..., 3] == 255) & (fine[..., 3] == 255)
    assert inside.sum() > 0.5 * 120 * 200
    assert np.abs(coarse[inside][:, :3] - fine[inside][:, :3]).max() <= 6


def test_editor_draws_a_dragged_dot_at_most_once_per_tick(tmp_path, caplog):
    """Owner's report 2026-11-14: dragging a warp dot lagged. Moves are kept and
    drawn at most once per WARP_DRAG_MS, with the newest position; the mouse
    release draws the last one at once."""
    import os
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    from types import SimpleNamespace
    from PIL import Image
    from editor import MediaEditor
    p = str(tmp_path / "w.png")
    Image.new("RGB", (400, 200), (200, 100, 50)).save(p)
    root = tk.Tk()
    root.withdraw()
    try:
        ed = MediaEditor(SimpleNamespace(root=root), p, None, SimpleNamespace(monitors=[]),
                         initial_pos={"x": 0.5, "y": 0.5, "scale_x": 2.0, "scale_y": 2.0})
        ed.sections["warp"].set_open(True)
        moved = []
        real = ed._move_warp_point
        ed._move_warp_point = lambda *a: (moved.append(a), real(*a))[1]
        x, y = ed._media_to_canvas(1.0, 0.0)
        ed._on_mouse_down(SimpleNamespace(x=x, y=y, state=0))
        for i in range(1, 11):
            ed._on_mouse_drag(SimpleNamespace(x=x + 4 * i, y=y, state=0))
        assert moved == []                                   # nothing drawn yet: one draw is due
        with caplog.at_level("INFO", logger="visualdeck.editor"):
            ed._on_mouse_up(None)                            # the release draws the newest at once
        assert len(moved) == 1 and abs(moved[0][1] - ed._canvas_to_media(x + 40, y)[0]) < 1e-9
        # The log says how long the canvas took (to check a show PC by its app.log).
        assert any(r.getMessage().startswith("Warp point drag: 1 canvas updates, median ") for r in caplog.records)
        ed.window.update()
        assert len(moved) == 1                               # the due draw found nothing new
        assert ed.current_pos["warp"]["points"][1][0] > 1.0
        ed.window.destroy()
    finally:
        root.destroy()


def test_full_frame_corner_dots_are_drawn_inside_the_canvas_and_drag_without_a_jump(tmp_path):
    """Owner's PC, 2026-11-14: a picture that fills the output has its corners on
    the canvas's edge, where the warp dots were almost invisible. They are drawn
    pulled inside, and dragging one moves its point by the mouse's movement."""
    import os
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    from types import SimpleNamespace
    from PIL import Image
    from editor import MediaEditor
    p = str(tmp_path / "full.png")
    Image.new("RGB", (1920, 1080), (200, 100, 50)).save(p)
    root = tk.Tk()
    root.withdraw()
    try:
        ed = MediaEditor(SimpleNamespace(root=root), p, None, SimpleNamespace(monitors=[]))
        ed.sections["warp"].set_open(True)
        ed.window.update()
        w, h = ed.canvas_width, ed.canvas_height
        for item in ed.canvas.find_withtag("warp_point"):           # every dot whole on the canvas
            x0, y0, x1, y1 = ed.canvas.coords(item)
            assert x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h
        tr = ed._warp_dot_xy(1.0, 0.0)                               # the top right dot, pulled in
        assert ed._warp_point_at(*tr) == 1
        true_x, true_y = ed._media_to_canvas(1.0, 0.0)
        ed._on_mouse_down(SimpleNamespace(x=tr[0], y=tr[1], state=0))
        ed._on_mouse_drag(SimpleNamespace(x=tr[0] - 30, y=tr[1] + 20, state=0))
        ed._on_mouse_up(None)
        u, v = ed.current_pos["warp"]["points"][1]
        x, y = ed._media_to_canvas(u, v)
        assert abs(x - (true_x - 30)) < 0.5 and abs(y - (true_y + 20)) < 0.5   # moved by the mouse, no jump
        ed.window.destroy()
    finally:
        root.destroy()
