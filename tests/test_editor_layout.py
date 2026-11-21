# SPDX-License-Identifier: MPL-2.0
"""The position editor's layout (settings in collapsible sections on the right)
and its effect preview (requests to the render thread, results on the canvas)."""
import os
import queue
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

tk = pytest.importorskip("tkinter")


@pytest.fixture
def root():
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    r = tk.Tk()
    r.withdraw()
    yield r
    r.destroy()


def _editor(root, tmp_path, pos=None, engine=None, queue_=None, name="w.png", size=(400, 200)):
    from editor import MediaEditor
    p = str(tmp_path / name)
    im = Image.new("RGB", size, (0, 0, 255))
    im.paste((255, 0, 0), (0, 0, size[0], size[1] // 2))
    im.save(p)
    parent = SimpleNamespace(root=root)
    if queue_ is not None:
        parent.render_command_queue = queue_
    eng = engine or SimpleNamespace(monitors=[], draws_effects=True, effect_errors={})
    return MediaEditor(parent, p, None, eng, initial_pos=pos or {})


def test_sections_open_close_and_summaries(root, tmp_path):
    from editor import MediaEditor
    MediaEditor.open_sections = {'position'}
    ed = _editor(root, tmp_path, pos={"opacity": 0.5, "blend": "add", "mask": {"shape": "ellipse"},
                                      "effects": [{"name": "Invert.fs", "inputs": {}}]})
    try:
        secs = ed.sections
        assert list(secs) == ["position", "look", "color", "mask", "warp", "effects"]   # no Playback for an image
        assert secs["position"].opened and not secs["look"].opened
        assert secs["look"].hint.cget("text") == "50 %, Add"
        assert secs["mask"].hint.cget("text") == "Ellipse" and secs["effects"].hint.cget("text") == "Invert"
        assert secs["color"].hint.cget("text") == ""
        # Every control exists while its section is closed (its value is always known).
        assert ed.color_vars["brightness"].get() == 0 and ed.mask_shape_var.get() == "Ellipse"
        assert not secs["color"].body.winfo_ismapped()
        secs["color"].toggle()
        root.update()
        assert secs["color"].opened and secs["color"].body.winfo_ismapped()
        assert secs["color"].hint.cget("text") == ""                               # open: no summary
        ed.color_vars["contrast"].set(150)
        ed._on_color_change("contrast")
        secs["color"].toggle()
        assert secs["color"].hint.cget("text") == "changed"
        assert MediaEditor.open_sections == {"position"}                           # opened, then closed again
        secs["effects"].toggle()
        assert "effects" in MediaEditor.open_sections                               # remembered for the next layer
        ed.x_var.set(0.25)
        ed._on_slider_change(None)
        assert ed.x_label.cget("text") == "25%"
    finally:
        ed.window.destroy()
        MediaEditor.open_sections = {'position'}


def test_video_has_a_playback_section(root, tmp_path, fixtures):
    import shutil
    from editor import MediaEditor
    src = fixtures["red_709"] if isinstance(fixtures, dict) and "red_709" in fixtures else None
    if src is None:
        pytest.skip("no video fixture")
    dst = tmp_path / "clip.mp4"
    shutil.copy(src, dst)
    parent = SimpleNamespace(root=root)
    ed = MediaEditor(parent, str(dst), None, SimpleNamespace(monitors=[]), initial_pos={"speed": 2.0})
    try:
        assert "playback" in ed.sections and ed.sections["playback"].hint.cget("text") == "2×"
        assert ed.speed_var.get() == "2×"
    finally:
        ed.window.destroy()


def test_effect_preview_requests_and_shows_the_result(root, tmp_path):
    """The canvas asks the render thread for the picture with its effects and
    shows what comes back; the picture is sent once; closing ends it."""
    q = queue.Queue()
    ed = _editor(root, tmp_path, pos={"effects": [{"name": "Invert.fs", "inputs": {}}]}, queue_=q)
    try:
        req = q.get_nowait()
        assert req["action"] == "fx_preview" and req["effects"] == [{"name": "Invert.fs", "inputs": {}}]
        assert req["image"].shape == (200, 400, 4) and req["drawn"] == (400, 200)
        w, h = req["out"]
        green = np.zeros((h, w, 4), dtype=np.uint8)
        green[..., 1] = 255
        green[..., 3] = 255
        req["reply"].put((req["seq"], green, None))
        ed._fx_poll()
        ph = ed.photo._PhotoImage__photo
        assert tuple(ph.get(ph.width() // 2, ph.height() // 2)) == (0, 255, 0)     # the GPU's picture is shown
        assert ed.fx_note.cget("text") == "Effects shown as the output draws them."
        # A change asks again, without the picture (the render thread has it).
        ed.effect_ui.items[0].set_value("amount", 0.5)
        req2 = q.get_nowait()
        assert req2["image"] is None and req2["effects"][0]["inputs"] == {"amount": 0.5}
        # While one is on its way, a newer change waits for it (one at a time).
        ed.effect_ui.items[0].set_value("amount", 0.25)
        assert q.empty()
        req2["reply"].put((req2["seq"], None, "no output window"))
        ed._fx_poll()
        req3 = q.get_nowait()
        assert req3["effects"][0]["inputs"] == {"amount": 0.25}
        assert "no output window" in ed.fx_note.cget("text")
        req3["reply"].put((req3["seq"], None, "no picture"))            # the output started again
        ed._fx_poll()
        ed.effect_ui.items[0].set_value("amount", 0.75)
        assert q.get_nowait()["image"] is not None                     # so the picture is sent again
        ed._cancel_and_close()
        root.update()
        sent = []
        while not q.empty():
            sent.append(q.get_nowait()["action"])
        assert "fx_preview_end" in sent                                # the render thread frees its buffers
    finally:
        try:
            ed.window.destroy()
        except tk.TclError:
            pass


def test_effect_preview_not_in_other_render_modes(root, tmp_path):
    q = queue.Queue()
    ed = _editor(root, tmp_path, pos={"effects": [{"name": "Invert.fs", "inputs": {}}]}, queue_=q,
                 engine=SimpleNamespace(monitors=[], draws_effects=False, effect_errors={}))
    try:
        assert q.empty() and "Performance mode only" in ed.fx_note.cget("text")
    finally:
        ed.window.destroy()


def test_render_thread_answers_effect_previews():
    """RenderThread: a preview is drawn after the frame, one at a time; an
    engine without effects answers at once."""
    import control_interface as ci_mod
    rt = ci_mod.RenderThread.__new__(ci_mod.RenderThread)
    reply = queue.Queue()
    rt.re = SimpleNamespace()
    assert rt._apply_command({"action": "fx_preview", "seq": 1, "reply": reply}) is False
    assert reply.get_nowait() == (1, None, "Effects are drawn in Performance Mode only")
    started, done = [], []

    class Eng:
        busy = False
        def effect_preview_begin(self, req):
            started.append(req["seq"]); Eng.busy = True
        def effect_preview_busy(self):
            return Eng.busy
        def effect_preview_poll(self):
            if Eng.busy and done:
                Eng.busy = False
                return done.pop(), np.zeros((1, 1, 4), np.uint8)
            return None
    rt.re = Eng()
    rt._apply_command({"action": "fx_preview", "seq": 2, "image": "pic", "reply": reply})
    rt._apply_command({"action": "fx_preview", "seq": 3, "image": None, "reply": reply})   # replaces 2
    assert rt._fx_preview_job["seq"] == 3 and rt._fx_preview_job["image"] == "pic"       # keeps the picture
    rt._run_fx_preview()
    assert started == [3]
    rt._apply_command({"action": "fx_preview", "seq": 4, "image": None, "reply": reply})
    rt._run_fx_preview()
    assert started == [3]                                    # busy: waits
    done.append({"seq": 3, "reply": reply})
    rt._run_fx_preview()
    seq, arr, why = reply.get_nowait()
    assert seq == 3 and why is None and started == [3, 4]


def test_warp_points_on_the_canvas(root, tmp_path):
    """Warp points: open the section and the dots appear; dragging one bends the
    layer (saved as 'warp'); a fold is refused; more points keep the shape;
    arrow keys nudge; Reset straightens it."""
    import layer_warp
    import mesh_warp
    from editor import MediaEditor
    MediaEditor.open_sections = {'position'}
    ed = _editor(root, tmp_path, pos={"x": 0.5, "y": 0.5, "scale_x": 2.0, "scale_y": 2.0})
    try:
        assert ed.sections["warp"].hint.cget("text") == ""
        assert not ed.canvas.find_withtag("warp_point") and ed.canvas.find_withtag("handle")
        ed.sections["warp"].toggle()
        assert len(ed.canvas.find_withtag("warp_point")) == 4 and not ed.canvas.find_withtag("handle")
        # Drag the top-right dot to the right by 40 canvas px.
        x, y = ed._media_to_canvas(1.0, 0.0)
        ed._on_mouse_down(SimpleNamespace(x=x, y=y, state=0))
        assert ed.drag_item == "warp" and ed.warp_selected == 1
        ed._on_mouse_drag(SimpleNamespace(x=x + 40, y=y, state=0))
        ed._on_mouse_up(None)
        w = ed.current_pos["warp"]
        assert (w["cols"], w["rows"]) == (2, 2) and w["points"][1][0] > 1.05 and abs(w["points"][1][1]) < 1e-6
        assert "warp" not in ed.original_pos
        # Across the opposite corner: the picture would fold; refused.
        assert ed._move_warp_point(1, -0.5, 1.5) is False and "fold" in ed.warp_note.cget("text")
        assert ed.current_pos["warp"] == w
        # 3 x 3 keeps the shape: the new middle-right point lies between the right corners.
        ed.warp_cols_var.set("3"); ed.warp_rows_var.set("3")
        ed._on_warp_grid_change()
        m = layer_warp.layer_warp(ed.current_pos)
        assert m[0] == m[1] == 3 and len(m[2]) == 9
        assert abs(mesh_warp.evaluate(m, 1.0, 0.0)[0] - w["points"][1][0]) < 1e-6
        # Arrow keys: 1 output pixel at a time.
        ed.warp_selected = 4                                         # the middle point
        before = list(ed.warp_points[4])
        ed._nudge_warp(10, 0)
        after = ed.warp_points[4]
        assert abs((after[0] - before[0]) * ed.media_width * 2.0 - 10) < 0.01 and abs(after[1] - before[1]) < 1e-9
        ed.sections["warp"].toggle()
        assert ed.sections["warp"].hint.cget("text") == "3\u00d73 points, bent"
        ed._reset_warp()
        assert "warp" not in ed.current_pos
    finally:
        ed.window.destroy()
        MediaEditor.open_sections = {'position'}
