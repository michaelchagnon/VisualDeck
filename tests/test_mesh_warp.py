# SPDX-License-Identifier: MPL-2.0
"""Mesh warp: the surface, the checks, both engines' geometry, and the show data."""
import json

import numpy as np
import pytest

import mesh_warp as mw
import output_warp as ow
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci, commands  # noqa: F401

WARP = ((0.1, 0.1), (0.9, 0.05), (1.0, 1.0), (0.0, 0.9))
MESH = {"cols": 3, "rows": 3, "points": [[0, 0], [0.5, 0], [1, 0],
                                         [0.1, 0.5], [0.65, 0.6], [1, 0.5],
                                         [0, 1], [0.5, 1], [1, 1]]}


def bent(n, dx=0.05, dy=0.03):
    pts = mw.identity(n, n)
    c = (n * n) // 2
    pts[c] = [pts[c][0] + dx, pts[c][1] + dy]
    return {"cols": n, "rows": n, "points": pts}


def test_a_regular_grid_reproduces_the_picture_exactly():
    for cols, rows in ((2, 2), (3, 5), (9, 9)):
        m = mw.parse({"cols": cols, "rows": rows, "points": mw.identity(cols, rows)})
        us, vs, xy = mw.tessellate(m)
        assert np.abs(xy[..., 0] - np.asarray(us)[None, :]).max() < 1e-12
        assert np.abs(xy[..., 1] - np.asarray(vs)[:, None]).max() < 1e-12


def test_the_surface_passes_through_every_point_and_bends_smoothly():
    m = mw.clean(MESH)
    for j in range(3):
        for i in range(3):
            assert mw.evaluate(m, i / 2, j / 2) == pytest.approx(tuple(MESH["points"][j * 3 + i]), abs=1e-12)
    # No kink at the moved point: the slope just left of it equals the slope just right of it.
    e = 1e-4
    left = np.subtract(mw.evaluate(m, 0.5, 0.5), mw.evaluate(m, 0.5 - e, 0.5)) / e
    right = np.subtract(mw.evaluate(m, 0.5 + e, 0.5), mw.evaluate(m, 0.5, 0.5)) / e
    assert left == pytest.approx(right, abs=1e-2)
    # Moving the center point does not move the picture's corners.
    assert mw.evaluate(m, 0, 0) == pytest.approx((0, 0)) and mw.evaluate(m, 1, 1) == pytest.approx((1, 1))


def test_clean_rejects_identity_malformed_and_folded_meshes():
    assert mw.clean(None) is None and mw.clean([]) is None
    assert mw.clean({"cols": 3, "rows": 3, "points": mw.identity(3, 3)}) is None       # nothing to do
    assert mw.clean(bent(3)) is not None
    for bad in ({"cols": 1, "rows": 3, "points": [[0, 0]] * 3},
                {"cols": 10, "rows": 2, "points": [[0, 0]] * 20},
                {"cols": True, "rows": 2, "points": [[0, 0]] * 2},
                {"cols": 2.5, "rows": 2, "points": [[0, 0]] * 5},
                {"cols": 3, "rows": 3, "points": mw.identity(3, 3)[:8]},
                {"cols": 2, "rows": 2, "points": [[0, 0], [1, 0], [0, 1], ["x", 1]]},
                {"cols": 2, "rows": 2, "points": [[0, 0], [1, 0], [0, 1], [float("nan"), 1]]},
                {"cols": 2, "rows": 2, "points": [[0, 0], [1, 0], [0, 1], [3.5, 1]]},
                {"cols": 2, "rows": 2, "points": [[0, 0], [1, 0], [0, 1], [1, 1, 1]]}):
        assert mw.clean(bad) is None, bad
    folded = bent(3, dx=0.6)                    # the center pushed past the right edge's points
    assert mw.parse(folded) is not None and mw.folds(mw.parse(folded)) and mw.clean(folded) is None
    swapped = {"cols": 2, "rows": 2, "points": [[1, 0], [0, 0], [1, 1], [0, 1]]}      # mirrored
    assert mw.clean(swapped) is None


def test_resize_keeps_the_shape():
    m = mw.clean(MESH)
    five = mw.resize(m, 5, 5)
    m5 = mw.clean({"cols": 5, "rows": 5, "points": five})
    for j in range(3):                          # the 3x3 points are also 5x5 points
        for i in range(3):
            assert mw.evaluate(m5, i / 2, j / 2) == pytest.approx(mw.evaluate(m, i / 2, j / 2), abs=1e-9)
    assert mw.resize(None, 4, 2) == mw.identity(4, 2)


def test_gl_vertices_go_through_the_pin_in_homogeneous_form():
    m = mw.clean(MESH)
    plain = mw.gl_vertices(m)
    assert plain.dtype == np.float32 and plain.shape[1] == 5 and len(plain) % 3 == 0
    assert np.all(plain[:, 2] == 1.0)
    pinned = mw.gl_vertices(m, WARP)
    h = ow.homography(WARP)
    for row in pinned[::997]:
        x, y = mw.evaluate(m, float(row[3]), float(row[4]))
        assert (row[0] / row[2], row[1] / row[2]) == pytest.approx(ow.apply(h, x, y), abs=1e-5)
    assert mw.through_pin_ok(m, WARP) and mw.through_pin_ok(m, None)


def _reference_map(mesh, corners, width, height):
    """Slow, obvious version of inverse_map: every pixel center tested against
    every triangle with barycentric coordinates."""
    us, vs, xy = mw.tessellate(mesh, 16)
    hw = mw._homogeneous(xy, corners)
    w = hw[..., 2]
    sx, sy = hw[..., 0] / w * width - 0.5, hw[..., 1] / w * height - 0.5
    uu, vv = np.meshgrid(us, vs)
    mu = np.full((height, width), np.nan)
    mv = np.full((height, width), np.nan)
    gy, gx = np.mgrid[0:height, 0:width].astype(float)
    for j in range(len(vs) - 1):
        for i in range(len(us) - 1):
            q = ((j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i))
            for t in ((0, 1, 2), (0, 2, 3)):
                a, b, c = (q[k] for k in t)
                den = (sy[b] - sy[c]) * (sx[a] - sx[c]) + (sx[c] - sx[b]) * (sy[a] - sy[c])
                l0 = ((sy[b] - sy[c]) * (gx - sx[c]) + (sx[c] - sx[b]) * (gy - sy[c])) / den
                l1 = ((sy[c] - sy[a]) * (gx - sx[c]) + (sx[a] - sx[c]) * (gy - sy[c])) / den
                l2 = 1 - l0 - l1
                inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
                iw = l0 / w[a] + l1 / w[b] + l2 / w[c]
                mu[inside] = ((l0 * uu[a] / w[a] + l1 * uu[b] / w[b] + l2 * uu[c] / w[c]) / iw)[inside]
                mv[inside] = ((l0 * vv[a] / w[a] + l1 * vv[b] / w[b] + l2 * vv[c] / w[c]) / iw)[inside]
    return mu * width - 0.5, mv * height - 0.5


@pytest.mark.parametrize("corners", [None, WARP])
def test_cpu_map_matches_the_reference_rasterizer(corners):
    m = mw.clean(MESH)
    W, H = 96, 54
    mx, my = mw.inverse_map(m, corners, W, H, segments=16)
    rx, ry = _reference_map(m, corners, W, H)
    both = ~np.isnan(rx) & (mx > -1000)
    assert both.sum() > 0.6 * W * H
    err = np.maximum(np.abs(mx[both] - rx[both]), np.abs(my[both] - ry[both]))
    # Pixels on a triangle's edge may take the neighbor's plane: a small difference there only.
    assert err.max() < 0.25 and np.median(err) < 0.01, (err.max(), np.median(err))
    # Every pixel the reference covers is covered (the map also runs a little past the edge).
    assert not (~np.isnan(rx) & (mx <= -1000)).any()


def test_identity_mesh_map_is_the_identity_and_a_mesh_with_a_pin_matches_the_pin():
    ident = mw.parse({"cols": 2, "rows": 2, "points": mw.identity(2, 2)})
    mx, my = mw.inverse_map(ident, None, 64, 36)
    assert np.abs(mx - np.arange(64)[None, :]).max() < 1e-3 and np.abs(my - np.arange(36)[:, None]).max() < 1e-3
    img = np.zeros((90, 160, 4), np.uint8)
    img[..., 3] = 255
    img[:45, :80, 0] = 255
    img[45:, 80:, 2] = 255
    a = mw.warp_image(img, ident, WARP).astype(int)
    b = ow.warp_image(img, WARP).astype(int)
    assert (np.abs(a - b).max(axis=2) > 60).mean() < 0.01          # differences only on edge pixels


def test_warp_image_moves_the_picture_with_the_point():
    img = np.zeros((100, 200, 4), np.uint8)
    img[..., 3] = 255
    img[45:55, 95:105, :3] = 255                                   # a white spot at the picture's center
    m = mw.clean(bent(3, dx=0.1, dy=0.1))
    out = mw.warp_image(img, m)
    ys, xs = np.nonzero(out[..., 0] > 128)
    assert xs.mean() / 200 == pytest.approx(0.6, abs=0.02) and ys.mean() / 100 == pytest.approx(0.6, abs=0.02)
    assert tuple(out[2, 2]) == (0, 0, 0, 255)


def test_gpu_mesh_warp(fixtures, x_display):
    from test_render_gl import run_gl_scenario
    r = run_gl_scenario("mesh_warp", fixtures, x_display)
    red, green, blue, white, black = [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]
    close = lambda a, b: all(abs(x - y) <= 8 for x, y in zip(a, b))  # noqa: E731
    assert close(r["plain_moved_center"], white), r
    for name in ("mesh", "mesh_pin"):
        o = r[name]
        assert close(o["tl"], red) and close(o["tr"], green) and close(o["bl"], blue) and close(o["br"], white), o
        assert close(o["moved_center"], red), o                     # the red quadrant follows the point
        assert close(o["pulled_edge"], black), o                    # the left edge moved in
        assert o["mean_abs_diff"] < 2.0 and o["share_over_40"] < 0.01, o   # GPU and CPU paths agree
        assert o["geometry"] == 1
    assert r["geometry_after"] == 0 and r["targets_after"] == 0


def test_pygame_engine_uses_the_mesh_and_rebuilds_in_the_background(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import time
    import pygame
    import render_engine as rem
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
        surf = pygame.Surface((200, 100), pygame.SRCALPHA)
        surf.fill((255, 255, 255, 255))
        surf.fill((255, 0, 0, 255), pygame.Rect(0, 0, 100, 50))
        eng.set_output_meshes({0: MESH, 1: {"cols": 2, "rows": 2, "points": mw.identity(2, 2)}, "x": MESH})
        assert list(eng.output_meshes) == [0]
        out = eng._warp_surface(surf, 0)                            # first maps: built at once
        assert tuple(out.get_at((120, 55)))[:3] == (255, 0, 0)       # (0.6, 0.55): the moved red quadrant
        assert tuple(out.get_at((3, 50)))[:3] == (0, 0, 0)           # the pulled-in left edge
        assert tuple(surf.get_at((120, 55)))[:3] == (255, 255, 255)  # the cached surface is untouched
        moved = dict(MESH, points=[list(p) for p in MESH["points"]])
        moved["points"][4] = [0.35, 0.4]
        eng.set_output_meshes({0: moved})
        out = eng._warp_surface(surf, 0)                            # previous maps while the new build runs
        assert tuple(out.get_at((120, 55)))[:3] == (255, 0, 0)
        assert eng.redraw_pending is False
        eng._mesh_jobs[0].join(5)
        assert eng.redraw_pending is True                          # so a still cue is drawn again
        out = eng._warp_surface(surf, 0)
        assert tuple(out.get_at((120, 55)))[:3] == (255, 255, 255)
        assert tuple(out.get_at((60, 38)))[:3] == (255, 0, 0)          # (0.3, 0.38): inside the red area now
        eng.set_output_meshes({})
        assert eng._mesh_maps == {} and eng._warp_surface(surf, 0) is surf
    finally:
        pygame.display.quit()


def test_meshes_are_saved_with_the_show_and_sent_to_the_renderer(blind_ci, tmp_path):
    ci = blind_ci
    for name in ("_build_media_list", "_build_grid", "_initialize_render_engine", "_on_toggle_presentation_mode"):
        setattr(ci, name, lambda *a, **k: None)
    show = str(tmp_path / "s.vjdeck")
    ci._perform_save(show)
    assert "meshes" not in json.load(open(show))["outputs"], "nothing written when no display has a mesh"
    commands(ci)
    assert ci._set_output_mesh(0, MESH) is True
    sent = commands(ci, "set_meshes")
    assert sent and sent[-1]["meshes"]["0"]["points"][4] == [0.65, 0.6]
    show = str(tmp_path / "s2.vjdeck")
    ci._perform_save(show)
    data = json.load(open(show))
    assert data["outputs"]["meshes"]["0"] == {"cols": 3, "rows": 3, "points": MESH["points"]}
    data["outputs"]["meshes"]["1"] = bent(3, dx=0.6)                  # folded, from a hand edit
    data["outputs"]["meshes"]["2"] = {"cols": 3}
    ci._apply_project_data(data, show)
    assert list(ci.output_meshes) == ["0"]
    assert ci._set_output_mesh(0, bent(3, dx=0.6)) is False and ci.output_meshes["0"]["points"][4] == [0.65, 0.6]
    assert ci._set_output_mesh(0, {"cols": 3, "rows": 3, "points": mw.identity(3, 3)}) is True    # flat: none
    assert ci.output_meshes == {} and commands(ci, "set_meshes")[-1]["meshes"] == {}


def test_output_presets_carry_meshes():
    import output_presets as op
    from types import SimpleNamespace
    p = op.capture("Display 0", {}, {}, {}, [SimpleNamespace(width=1920, height=1080)],
                   meshes={"0": MESH, "1": bent(3, dx=0.6), "x": MESH})
    assert list(p["meshes"]) == ["0"] and p["meshes"]["0"]["points"] == MESH["points"]
    assert op.used_displays(p) == {0} and "mesh on 0" in op.summary(p)
    assert op.clean({"mode": "None"})["meshes"] == {}                 # presets saved before meshes


def test_corner_pin_dialog_edits_the_mesh(blind_ci):
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
    ci.re.monitors = [SimpleNamespace(x=0, y=0, width=1920, height=1080)]
    ci.selected_display = 0
    try:
        win = ci._open_corner_pin()
        vd = win._vd
        M, CW, h = vd["M"], vd["CW"], vd["ch"]()
        ev = lambda x, y, state=0: SimpleNamespace(x=x, y=y, state=state)  # noqa: E731
        vd["mesh_on"].set(True)
        assert "Mesh 3 × 3" in vd["readout"].cget("text")
        vd["press"](ev(M + 0.5 * CW + 2, M + 0.5 * h - 1))           # grab the center point
        assert vd["state"]["msel"] == 4
        vd["motion"](ev(M + 0.6 * CW, M + 0.55 * h))
        assert ci.output_meshes["0"]["points"][4] == pytest.approx([0.6, 0.55], abs=2e-3)
        assert commands(ci, "set_meshes")[-1]["meshes"]["0"]["points"][4] == pytest.approx([0.6, 0.55], abs=2e-3)
        vd["nudge"](1, 0, ev(0, 0, state=0x1))                         # Shift+Right: 10 px
        assert ci.output_meshes["0"]["points"][4][0] == pytest.approx(0.6 + 10 / 1920, abs=2e-3)
        before = ci.output_meshes["0"]["points"][4]
        vd["motion"](ev(M + 1.4 * CW, M + 0.55 * h))                   # past the right edge points: a fold
        assert ci.output_meshes["0"]["points"][4] == before and "fold" in vd["note"].cget("text")
        vd["cols_var"].set("5")                                        # more points, same shape
        m5 = mw.clean(ci.output_meshes["0"])
        assert m5[0] == 5 and mw.evaluate(m5, 0.5, 0.5) == pytest.approx(tuple(before), abs=1e-6)
        # A corner pin moves the mesh with it; mesh points are dragged through the pin.
        vd["mesh_on"].set(False)
        vd["press"](ev(M + 3, M + 2))
        vd["motion"](ev(M + 0.1 * CW, M + 0.1 * h))
        assert ci.output_warps["0"][0] == pytest.approx([0.1, 0.1], abs=1e-3)
        vd["mesh_on"].set(True)
        x, y = vd["to_out"](*vd["state"]["mpts"][vd["state"]["msel"]])
        vd["press"](ev(M + x * CW, M + y * h))
        vd["motion"](ev(M + (x + 0.02) * CW, M + y * h))
        x2, y2 = vd["to_out"](*vd["state"]["mpts"][vd["state"]["msel"]])
        assert (x2, y2) == pytest.approx((x + 0.02, y), abs=2e-3)
        vd["reset_mesh"]()
        assert ci.output_meshes == {} and commands(ci, "set_meshes")[-1]["meshes"] == {}
        assert ci.output_warps["0"][0] == pytest.approx([0.1, 0.1], abs=1e-3)   # the pin stays
        win.destroy()
    finally:
        root.destroy()
