# SPDX-License-Identifier: MPL-2.0
"""F23: the editor's corner handles sit on the rotated media's corners, and
dragging a handle of rotated media scales it along its own axes."""
import math
import os

import numpy as np
import pytest
from PIL import Image

tk = pytest.importorskip("tkinter")


@pytest.fixture
def make_editor(tmp_path):
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    from types import SimpleNamespace
    import editor
    root = tk.Tk()
    root.withdraw()
    made = []

    def make(rotation, w=200, h=100):
        p = str(tmp_path / f"m{w}x{h}.png")
        im = Image.new("RGBA", (w, h), (0, 0, 255, 255))
        im.paste((255, 0, 0, 255), (w - 10, 0, w, 10))          # red marks the top-right corner
        im.save(p)
        ed = editor.MediaEditor(SimpleNamespace(root=root), p, None, SimpleNamespace(monitors=[]),
                                initial_pos={"x": 0.5, "y": 0.5, "scale_x": 1.0, "scale_y": 1.0,
                                             "rotation": rotation})
        made.append(ed)
        return ed, im
    yield make
    for ed in made:
        try:
            ed.window.destroy()
        except tk.TclError:
            pass
    root.destroy()


@pytest.mark.parametrize("rotation", [0, 30, 90, -45, 135])
def test_handles_sit_on_the_rotated_corners(make_editor, rotation):
    ed, im = make_editor(rotation)
    shown = im.resize((int(im.width * ed.canvas_scale), int(im.height * ed.canvas_scale))).rotate(
        rotation, expand=True)                                      # what the canvas draws (and the output does)
    a = np.asarray(shown)
    ys, xs = np.nonzero((a[..., 0] > 200) & (a[..., 2] < 60) & (a[..., 3] > 200))
    red = (xs.mean() - shown.width / 2, ys.mean() - shown.height / 2)   # red mark relative to the center
    cx, cy = ed.current_pos["x"] * ed.canvas_width, ed.current_pos["y"] * ed.canvas_height
    ne = ed._get_handle_positions()["ne"]
    ne = (ne[0] - cx, ne[1] - cy)
    assert math.dist(ne, red) < 6, (ne, red)


def test_dragging_a_rotated_handle_scales_along_the_media_axes(make_editor):
    ed, _ = make_editor(90)
    ed.drag_start_pos = dict(ed.current_pos)
    ed.drag_start_handles = ed._get_handle_positions()
    ed.drag_item, ed.drag_handle = "resize", "se"
    ax, ay = ed.drag_start_handles["nw"]
    th = math.radians(90)
    u, v = 150, 50                                                  # wanted size in canvas px (media axes)
    sx, sy = u * math.cos(th) + v * math.sin(th), -u * math.sin(th) + v * math.cos(th)

    class Ev:
        x, y, state = ax + sx, ay + sy, 0
    ed._on_mouse_drag(Ev())
    assert ed.current_pos["scale_x"] == pytest.approx(150 / ed.canvas_scale / 200)
    assert ed.current_pos["scale_y"] == pytest.approx(50 / ed.canvas_scale / 100)
    assert ed.current_pos["x"] * ed.canvas_width == pytest.approx(ax + sx / 2)


def test_output_rotates_counter_clockwise_like_the_editor(fixtures, x_display):
    from test_render_gl import run_gl_scenario
    dx, dy = run_gl_scenario("rotation", fixtures, x_display)["red_rel"]
    assert dx < 0 and dy < 0, (dx, dy)                  # top-right corner went to the top-left
