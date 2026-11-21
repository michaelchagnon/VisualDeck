# SPDX-License-Identifier: MPL-2.0
"""Dragging a cue or layer header shows where it will land."""
import os
from types import SimpleNamespace

import pytest

from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401

tk = pytest.importorskip("tkinter")


@pytest.fixture
def grid(blind_ci):
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    root = tk.Tk()
    root.geometry("800x500+0+0")
    ci = blind_ci
    ci.root = root
    ci.grid_canvas = tk.Canvas(root, width=600, height=400)
    ci.grid_canvas.place(x=100, y=0)
    ci.col_labels = {c: tk.Label(root, text=f"C{c}") for c in range(3)}
    for c, lbl in ci.col_labels.items():
        lbl.place(x=150 + c * 154, y=0, width=150, height=24)
    ci.row_labels = {r: tk.Label(root, text=f"L{r}") for r in range(3)}
    for r, lbl in ci.row_labels.items():
        lbl.place(x=100, y=28 + r * 154, width=48, height=150)
    root.update()
    yield ci, root
    root.destroy()


def test_marker_sits_on_the_far_side_of_the_target(grid):
    ci, root = grid
    pad = ci.CELL_PAD // 2
    gw, gh = ci.grid_canvas.winfo_width(), ci.grid_canvas.winfo_height()
    assert ci._drop_marker_geometry("cue", 0, 2) == (150 + 2 * 154 + 150 + pad - 2, 0, 5, gh)    # after C2
    assert ci._drop_marker_geometry("cue", 2, 0) == (150 - pad - 2, 0, 5, gh)                    # before C0
    assert ci._drop_marker_geometry("layer", 2, 0) == (100, 28 - pad - 2, gw, 5)                  # above L0
    assert ci._drop_marker_geometry("cue", 1, 1) is None and ci._drop_marker_geometry("cue", 1, None) is None


def test_drag_shows_marker_and_label_and_cleans_up(grid):
    ci, root = grid
    ci.cue_names.update({"0": "Intro", "2": "Finale"})
    ci._start_cue_reorder_drag(0)
    lbl = ci.col_labels[2]
    ev = SimpleNamespace(x_root=lbl.winfo_rootx() + 10, y_root=lbl.winfo_rooty() + 10)
    ci._on_drag_motion(ev)
    root.update()
    assert ci._drop_marker.winfo_ismapped() and ci.drag_widget.cget("text") == "Intro  →  after Finale"
    own = ci.col_labels[0]
    ci._on_drag_motion(SimpleNamespace(x_root=own.winfo_rootx() + 5, y_root=own.winfo_rooty() + 5))
    root.update()
    assert not ci._drop_marker.winfo_ismapped() and ci.drag_widget.cget("text") == "Intro"
    ci._cleanup_drag()
    assert ci._drop_marker is None
