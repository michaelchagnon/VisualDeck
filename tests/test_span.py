# SPDX-License-Identifier: MPL-2.0
"""Span: one picture across displays (layout, both engines, the app, presets, dialog)."""
import json
import os
from types import SimpleNamespace

import pytest

import span
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci, commands  # noqa: F401


def mons(*sizes):
    out, x = [], 0
    for w, h in sizes:
        out.append(SimpleNamespace(x=x, y=0, width=w, height=h, name=str(x)))
        x += w
    return out


def test_layout_two_and_three_displays():
    m = mons((1920, 1080), (1920, 1080))
    (cw, ch), parts = span.layout(m, span.clean({"displays": [0, 1], "overlap": 240}))
    assert (cw, ch) == (3600, 1080)
    assert parts[0]["rect"] == [0.0, 0.0, 1920 / 3600, 1.0] and parts[1]["offset"] == 1680
    assert parts[1]["rect"][2] == pytest.approx(1.0)
    m3 = mons((1280, 800), (1280, 800), (1280, 720))
    (cw, ch), parts = span.layout(m3, span.clean({"displays": [0, 1, 2], "overlap": 100}))
    assert (cw, ch) == (3640, 800) and [parts[i]["offset"] for i in range(3)] == [0, 1180, 2360]
    assert parts[2]["rect"][3] == pytest.approx(720 / 800)            # the shorter display: the top of the canvas
    assert span.describe(m3, span.clean({"displays": [0, 1, 2], "overlap": 100}))[0] == "Picture 3640 × 800"


def test_clean_and_fit():
    assert span.clean(None) is None and span.clean({"displays": [0]}) is None
    assert span.clean({"displays": [0, 0]}) is None and span.clean({"displays": [0, True]}) is None
    assert span.clean({"displays": [1, 0], "overlap": -5}) == {"displays": [1, 0], "overlap": 0}
    assert span.clean({"displays": [0, 2]}, n_monitors=2) is None
    m = mons((1920, 1080), (1920, 1080))
    assert span.layout(m, span.clean({"displays": [0, 1], "overlap": 961})) is None       # over half a display


def test_blends_on_the_sides_that_meet():
    m = mons((1920, 1080), (1920, 1080), (1920, 1080))
    sp = span.clean({"displays": [0, 1, 2], "overlap": 192})
    b = span.blends(sp, m, {"0": {"left": 0.05, "gamma": 2.4}})
    assert b["0"] == {"left": 0.05, "gamma": 2.4, "right": 0.1}
    assert b["1"] == {"left": 0.1, "right": 0.1} and b["2"] == {"left": 0.1}


def test_gpu_span(fixtures, x_display):
    from test_render_gl import run_gl_scenario
    r = run_gl_scenario("span", fixtures, x_display)
    # Display 0 (window x 0-959) shows canvas x 0-959; display 1 (960-1919) shows canvas 800-1759.
    want = {"0": 0, "100": 100, "959": 959, "960": 800, "961": 801, "1500": 1340, "1919": 1759}
    assert r["read"] == want, r
    assert r["composition"] == [[1760, 1080], [1760, 1080]] and r["targets"] == 2
    assert r["mix_read"] == {"100": 100, "1500": 1340}                 # a crossfade keeps the parts
    assert r["no_span_read"] == {"100": 500, "1500": 940}              # each display its own picture again
    assert r["targets_after"] == 0


def test_pygame_engine_shows_each_part(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import numpy as np
    import pygame
    import render_engine as rem
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
        eng.monitors = mons((96, 54), (96, 54))
        (cw, ch), parts = span.layout(eng.monitors, span.clean({"displays": [0, 1], "overlap": 16}))
        eng.set_output_regions(parts)
        assert eng.composition_size(0) == (176, 54) == eng.composition_size(1)
        canvas = pygame.Surface((cw, ch), pygame.SRCALPHA)
        for x in range(cw):
            canvas.fill((x, 0, 0, 255), pygame.Rect(x, 0, 1, ch))
        a, b = eng._warp_surface(canvas, 0), eng._warp_surface(canvas, 1)
        assert a.get_size() == b.get_size() == (96, 54)
        assert [a.get_at((x, 20))[0] for x in (0, 95)] == [0, 95]
        assert [b.get_at((x, 20))[0] for x in (0, 95)] == [80, 175]
        eng.set_output_regions({})
        assert eng._warp_surface(canvas, 0) is canvas and eng.composition_size(0) == (96, 54)
    finally:
        pygame.display.quit()


class TracedVar:
    """Like the app's tk StringVar with its trace: setting it runs _on_output_change."""
    def __init__(self, ci, value):
        self.ci, self.v = ci, value

    def get(self):
        return self.v

    def set(self, v):
        self.v = v
        self.ci._on_output_change()


def _app(ci, sizes=((1920, 1080), (1920, 1080))):
    ci.re.monitors = mons(*sizes)
    for name in ("_build_media_list", "_build_grid", "_initialize_render_engine", "_on_toggle_presentation_mode",
                 "_update_cell_preview"):
        setattr(ci, name, lambda *a, **k: None)
    ci.layer_setting_widgets = {}
    ci.screen_var = TracedVar(ci, "Display 0")
    return ci


def test_span_mode_routes_every_layer_to_every_display(blind_ci):
    ci = _app(blind_ci)
    commands(ci)
    assert ci._apply_span({"displays": [0, 1], "overlap": 240}) is True
    assert ci.screen_var.get() == ci.SPAN and ci.selected_display == 0
    regions = commands(ci, "set_regions")[-1]["regions"]
    assert regions["1"]["canvas"] == [3600, 1080] and regions["1"]["rect"][0] == pytest.approx(1680 / 3600)
    assert ci._screens_for_row(3) == [0, 1]
    routing = ci._output_routing()
    assert routing["target_monitor"] is None and routing["all_assigned_screens"] == {0, 1}
    assert ci._layout_size(0) == (3600, 1080) and ci._edit_preview_reference() == (3600, 1080)
    assert ci.output_blends["0"]["right"] == pytest.approx(240 / 1920)
    assert ci.output_blends["1"]["left"] == pytest.approx(240 / 1920)
    assert ci.SPAN in ci.output_screen_combobox["values"] if hasattr(ci, "output_screen_combobox") else True
    # Another output: no regions; back to Span: the regions again.
    ci.screen_var.set("Display 1")
    assert commands(ci, "set_regions")[-1]["regions"] == {} and ci._layout_size(1) == (1920, 1080)
    ci.screen_var.set(ci.SPAN)
    assert commands(ci, "set_regions")[-1]["regions"]["0"]["canvas"] == [3600, 1080]
    ci._remove_span()
    assert ci.output_span is None and ci.screen_var.get() == "Display 0"
    assert commands(ci, "set_regions")[-1]["regions"] == {}
    assert ci._apply_span({"displays": [0, 1], "overlap": 1000}) is False     # more than half a display


def test_span_without_its_displays_falls_back_to_no_output(blind_ci):
    ci = _app(blind_ci, sizes=((1920, 1080),))
    ci.output_span = {"displays": [0, 1], "overlap": 0}
    ci.screen_var.set(ci.SPAN)
    assert ci.screen_var.get() == ci.NONE and ci.selected_display is None


def test_span_is_saved_with_the_show_and_restored_before_the_mode(blind_ci, tmp_path):
    ci = _app(blind_ci)
    ci._apply_span({"displays": [1, 0], "overlap": 100}, blend=False)
    show = str(tmp_path / "span.vjdeck")
    ci._perform_save(show)
    data = json.load(open(show))
    assert data["outputs"]["span"] == {"displays": [1, 0], "overlap": 100} and data["outputs"]["mode"] == "Span"
    ci.output_span = None
    ci.screen_var.set(ci.NONE)
    ci._apply_project_data(data, show)
    assert ci.output_span == {"displays": [1, 0], "overlap": 100} and ci.screen_var.get() == ci.SPAN
    assert ci._span_displays() == [1, 0] and ci.selected_display == 1


def test_presets_carry_the_span():
    import output_presets as op
    m = mons((1920, 1080), (1920, 1080))
    p = op.capture("Span", {}, {}, {}, m, span={"displays": [0, 1], "overlap": 120})
    assert p["mode"] == "Span" and p["span"] == {"displays": [0, 1], "overlap": 120}
    assert op.used_displays(p) == {0, 1} and op.summary(p).startswith("Span of 0, 1 (120 px overlap)")
    assert op.clean({"mode": "Span"})["mode"] == "None"                # a span preset without its span
    assert op.clean({"mode": "Display 0"})["span"] is None


def test_test_pattern_on_a_span_is_one_picture(blind_ci, monkeypatch, tmp_path):
    ci = _app(blind_ci)
    ci._apply_span({"displays": [0, 1], "overlap": 240}, blend=False)
    made = []
    import test_patterns
    monkeypatch.setattr(test_patterns, "ensure_file",
                        lambda kind, w, h, label, out: made.append((kind, w, h, label)) or str(tmp_path / "p.png"))
    monkeypatch.setattr(ci, "_program_layers", lambda: {})
    ci._show_test_pattern("grid")
    assert made == [("grid", 3600, 1080, "Displays 0, 1 (span)")]
    sent = commands(ci, "render")[-1]
    assert ci._program["layers"][0]["screens"] == [0, 1] and sent["all_assigned_screens"] == {0, 1}


def test_span_dialog(blind_ci):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import ui_theme
    ci = _app(blind_ci)
    root = tk.Tk()
    root.withdraw()
    ui_theme.apply(root)
    ci.root = root
    try:
        win = ci._open_span()
        vd = win._vd
        assert "Picture 3840 × 1080" in vd["info"].cget("text")      # both ticked, no overlap
        vd["ov_var"].set("1200")
        assert "more than half" in vd["info"].cget("text") and vd["apply"]() is False
        vd["ov_var"].set("192")
        assert "Display 1: x 1728" in vd["info"].cget("text")
        assert vd["apply"]() is True and ci.output_span == {"displays": [0, 1], "overlap": 192}
        assert ci.screen_var.get() == ci.SPAN and ci.output_blends["1"]["left"] == pytest.approx(0.1)
        vd["use"][1].set(False)
        assert "two or more" in vd["info"].cget("text")
        vd["remove"]()
        assert ci.output_span is None and ci.screen_var.get() == "Display 0"
        win.destroy()
    finally:
        root.destroy()


def test_grid_layout_and_blends():
    m = [SimpleNamespace(x=x, y=y, width=1920, height=1080, name="") for y in (0, 1080) for x in (0, 1920)]
    sp = span.clean({"displays": [0, 1, 2, 3], "overlap": 200, "columns": 2, "overlap_y": 100})
    assert span.grid(sp) == (2, 2)
    (cw, ch), parts = span.layout(m, sp)
    assert (cw, ch) == (3640, 2060)
    assert (parts[3]["offset"], parts[3]["offset_y"]) == (1720, 980) and parts[2]["cell"] == (0, 1)
    b = span.blends(sp, m)
    assert b["0"] == {"right": 200 / 1920, "bottom": 100 / 1080}
    assert b["3"] == {"left": 200 / 1920, "top": 100 / 1080}
    assert span.describe(m, sp)[0] == "Picture 3640 × 2060 (2 × 2 displays)"
    assert span.clean({"displays": [0, 1, 2], "columns": 2}) is None              # rows must be full
    assert span.clean({"displays": [0, 1], "overlap": 10}) == {"displays": [0, 1], "overlap": 10}   # one row
    assert span.layout(m, span.clean({"displays": [0, 1, 2, 3], "columns": 2, "overlap_y": 600})) is None


def test_gpu_span_grid(fixtures, x_display):
    from test_render_gl import run_gl_scenario
    r = run_gl_scenario("span_grid", fixtures, x_display)
    assert r["canvas"] == [1760, 1020]
    # Display (column c, row r) at window (960 c, 540 r) shows canvas from (800 c, 480 r).
    want = [[x - (x >= 960) * 160, y - (y >= 540) * 60] for x, y in r["probes"]]
    assert r["read"] == want, (r["read"], want)


def test_grid_from_the_dialog(blind_ci):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import ui_theme
    ci = _app(blind_ci, sizes=())
    ci.re.monitors = [SimpleNamespace(x=x, y=y, width=1920, height=1080, name="") for y in (0, 1080)
                      for x in (0, 1920)]
    root = tk.Tk()
    root.withdraw()
    ui_theme.apply(root)
    ci.root = root
    try:
        vd = ci._open_span()._vd
        vd["cols_var"].set("3")
        assert "do not fill rows of 3" in vd["info"].cget("text")
        vd["cols_var"].set("2")
        vd["ovy_var"].set("100")
        assert "Picture 3840 × 2060 (2 × 2 displays)" in vd["info"].cget("text")
        assert vd["apply"]() is True
        assert ci.output_span == {"displays": [0, 1, 2, 3], "overlap": 0, "columns": 2, "overlap_y": 100}
        assert ci._layout_size(0) == (3840, 2060) and ci.output_blends["2"]["top"] == pytest.approx(100 / 1080)
    finally:
        root.destroy()


# --- slices: parts set by hand ---------------------------------------------------------------------
def test_slices_are_cleaned_and_laid_out_as_given():
    m = mons((1920, 1080), (1280, 720))
    sp = span.clean({"displays": [1, 0], "canvas": [3840, 1080],
                     "slices": {"0": [0, 0, 1920, 1080], "1": [2560, 360, 1280, 720], "9": [0, 0, 1, 1]}})
    assert sp == {"displays": [1, 0], "overlap": 0, "canvas": [3840, 1080],
                  "slices": {"1": [2560, 360, 1280, 720], "0": [0, 0, 1920, 1080]}}
    (cw, ch), parts = span.layout(m, sp)
    assert (cw, ch) == (3840, 1080) and parts[1]["rect"] == [2560 / 3840, 360 / 1080, 1.0, 1.0]
    assert parts[0]["offset"] == 0 and parts[1]["offset_y"] == 360 and parts[1]["cell"] is None
    one = span.clean({"displays": [1], "canvas": [1920, 1080], "slices": {"1": [480, 270, 960, 540]}})
    assert one is not None and span.layout(m, one)[1][1]["rect"] == [0.25, 0.25, 0.75, 0.75]   # one display: a zoom
    assert span.describe(m, one) == ["Picture 1920 × 1080, parts set by hand",
                                     "Display 1: x 480–1440, y 270–810, scaled to 1280 × 720"]
    b = span.blends(sp, m, {"0": {"left": 0.1}})
    assert b == {"1": {}, "0": {"left": 0.1}}, "parts set by hand keep the blends as they were"
    for bad in ({"canvas": [3840, 1080], "slices": {"0": [1920, 0, 1921, 1080]}},    # past the right edge
                {"canvas": [3840, 1080], "slices": {"0": [-1, 0, 100, 100]}},
                {"canvas": [3840, 1080], "slices": {"0": [0, 0, 0, 100]}},
                {"canvas": [3840, 1080], "slices": {}},                              # display 0 has no part
                {"canvas": [8, 1080], "slices": {"0": [0, 0, 8, 8]}},                # too small a picture
                {"canvas": [3840, 1080], "slices": {"0": [0.5, 0, 100, 100]}},
                {"canvas": [3840, 1080], "slices": {"0": [True, 0, 100, 100]}},
                {"canvas": "big", "slices": {"0": [0, 0, 100, 100]}}):
        assert span.clean(dict(bad, displays=[0])) is None, bad


def test_gpu_span_slices(fixtures, x_display):
    from test_render_gl import run_gl_scenario
    r = run_gl_scenario("span_slices", fixtures, x_display)
    assert r["canvas"] == [1600, 900] and r["composition"] == [[1600, 900], [1600, 900]]
    # display 0: window (x, y) shows canvas (x * 800/960, y * 900/1080); display 1: (1200 + (x-960) * 400/960,
    # 450 + y * 450/1080). Scaled with linear filtering: within 1 px.
    want = [[x * 800 / 960, y * 900 / 1080] if x < 960 else [1200 + (x - 960) * 400 / 960, 450 + y * 450 / 1080]
            for x, y in r["probes"]]
    for got, exp in zip(r["read"], want):
        assert abs(got[0] - exp[0]) <= 1.5 and abs(got[1] - exp[1]) <= 1.5, (r["read"], want)


def test_pygame_engine_scales_a_slice(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    import render_engine as rem
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
        eng.monitors = mons((96, 54))
        sp = span.clean({"displays": [0], "canvas": [192, 108], "slices": {"0": [96, 54, 48, 27]}})
        (cw, ch), parts = span.layout(eng.monitors, sp)
        eng.set_output_regions(parts)
        canvas = pygame.Surface((cw, ch), pygame.SRCALPHA)
        for x in range(cw):
            canvas.fill((x, 0, 0, 255), pygame.Rect(x, 0, 1, ch))
        a = eng._warp_surface(canvas, 0)
        assert a.get_size() == (96, 54)
        assert abs(a.get_at((0, 20))[0] - 96) <= 1 and abs(a.get_at((95, 20))[0] - 143) <= 1   # 48 px to 96
    finally:
        pygame.display.quit()


def test_slices_from_the_dialog(blind_ci):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import ui_theme
    ci = _app(blind_ci)
    root = tk.Tk()
    root.withdraw()
    ui_theme.apply(root)
    ci.root = root
    try:
        vd = ci._open_span()._vd
        vd["ov_var"].set("192")
        vd["slices_var"].set(True)                                   # starts from the row's layout
        assert [v.get() for v in vd["canvas_vars"]] == ["3648", "1080"]
        assert [v.get() for v in vd["part_vars"][1]] == ["1728", "0", "1920", "1080"]
        assert "parts set by hand" in vd["info"].cget("text")
        vd["part_vars"][1][2].set("2000")
        assert "inside the picture" in vd["info"].cget("text") and vd["apply"]() is False
        vd["part_vars"][1][2].set("x")
        assert "whole numbers" in vd["info"].cget("text")
        for v, val in zip(vd["part_vars"][1], ("1824", "540", "960", "540")):
            v.set(val)
        vd["use"][0].set(False)                                      # one display is enough with slices
        assert vd["apply"]() is True
        assert ci.output_span == {"displays": [1], "overlap": 0, "canvas": [3648, 1080],
                                  "slices": {"1": [1824, 540, 960, 540]}}
        assert ci.screen_var.get() == ci.SPAN and ci._layout_size(1) == (3648, 1080)
        assert ci._span_regions() == {"1": {"rect": [1824 / 3648, 0.5, 2784 / 3648, 1.0], "canvas": [3648, 1080]}}
    finally:
        root.destroy()
