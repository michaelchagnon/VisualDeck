# SPDX-License-Identifier: MPL-2.0
"""Playback copies: short GOP, smaller, same frames and sound, never in place of the original."""
import os

import av
import numpy as np
import pytest

import playback_copy as pc
from test_reverse import clips  # noqa: F401
from test_start_from import peak_hz, tones  # noqa: F401


def lumas(path):
    with av.open(path) as c:
        return [float(f.to_ndarray(format="gray").mean()) for f in c.decode(video=0)]


def keyframe_gaps(path):
    with av.open(path) as c:
        v = c.streams.video[0]
        n, keys = 0, []
        for p in c.demux(v):
            if p.size:
                if p.is_keyframe:
                    keys.append(n)
                n += 1
    return keys, n


def test_same_frames_with_a_key_frame_every_half_second(clips, tmp_path):
    src = clips["one_gop"]                                 # 40 frames, one key frame
    dst = str(tmp_path / "copy.mp4")
    seen = []
    r = pc.make(src, dst, keyframe_s=0.5, progress=seen.append)
    a, b = lumas(src), lumas(dst)
    assert len(a) == len(b) == r["frames"]
    assert max(abs(x - y) for x, y in zip(a, b)) < 2.0, "the same pictures, in order"
    keys, n = keyframe_gaps(dst)
    assert keys == list(range(0, n, 15)), keys            # 30 fps: every 15 frames
    assert r["gop"] == 15 and seen and seen[-1] == 1.0
    assert not os.path.exists(dst + ".part")


def test_every_frame_a_key_frame_and_a_smaller_size(clips, tmp_path):
    dst = str(tmp_path / "intra.mp4")
    r = pc.make(clips["short_gop"], dst, max_size=(160, 90), keyframe_s=0)
    keys, n = keyframe_gaps(dst)
    assert keys == list(range(n)) and r["gop"] == 1
    with av.open(clips["short_gop"]) as c:
        sw, sh = c.streams.video[0].codec_context.width, c.streams.video[0].codec_context.height
    with av.open(dst) as c:
        cc = c.streams.video[0].codec_context
        assert (cc.width, cc.height) == pc.fit((sw, sh), (160, 90)) == r["size"]
    assert pc.fit((3840, 2160), (1920, 1080)) == (1920, 1080) and pc.fit((1280, 720), (1920, 1080)) == (1280, 720)
    assert pc.fit((1000, 1000), (1920, 1080)) == (1000, 1000) and pc.fit((3000, 1000), (1920, 1080)) == (1920, 640)


def test_the_sound_comes_along(tmp_path):
    rate = 48000
    wav = tones(tmp_path, rate)                            # 300 / 600 / 900 Hz, one second each
    src = str(tmp_path / "av.mp4")
    with av.open(wav) as a, av.open(src, "w") as out:     # a video with that sound
        v = out.add_stream("mpeg4", rate=25)
        v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
        s = out.add_stream("aac", rate=rate)
        for i in range(75):
            f = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i, np.uint8), format="rgb24")
            f.pts = i
            for p in v.encode(f):
                out.mux(p)
        for fr in a.decode(audio=0):
            fr.pts = None
            for p in s.encode(fr):
                out.mux(p)
        for st in (v, s):
            for p in st.encode():
                out.mux(p)
    dst = str(tmp_path / "copy.mp4")
    r = pc.make(src, dst)
    assert r["audio"] is True
    with av.open(dst) as c:
        x = np.concatenate([f.to_ndarray()[0] if f.format.is_planar else f.to_ndarray()[0, ::len(f.layout.channels)]
                            for f in c.decode(audio=0)])           # the first channel
    for second, hz in ((0, 300), (1, 600), (2, 900)):
        seg = x[int((second + 0.25) * rate): int((second + 0.75) * rate)]
        assert abs(peak_hz(seg, rate) - hz) < 10, (second, peak_hz(seg, rate))


def test_cancel_leaves_nothing_and_names_never_overwrite(clips, tmp_path):
    dst = str(tmp_path / "copy.mp4")
    with pytest.raises(pc.Canceled):
        pc.make(clips["one_gop"], dst, canceled=lambda: True)
    assert not os.path.exists(dst) and not os.path.exists(dst + ".part")
    src = str(tmp_path / "Show clip.mov")
    open(src, "wb").close()
    first = pc.target_path(src)
    assert first == str(tmp_path / "Show clip (playback).mp4")
    open(first, "wb").close()
    assert pc.target_path(src) == str(tmp_path / "Show clip (playback 2).mp4")
    assert pc.target_path(src, (1920, 1080)) == str(tmp_path / "Show clip (playback 1080p).mp4")


def test_the_window_makes_a_smaller_copy_and_uses_it_keeping_layer_sizes(tmp_path, monkeypatch, fixtures):
    import shutil
    import time
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import control_interface as ci_mod
    src = str(tmp_path / "big.mp4")
    with av.open(src, "w") as out:                          # 640 x 360, 1 s, a key frame at the start only
        v = out.add_stream("mpeg4", rate=25)
        v.width, v.height, v.pix_fmt = 640, 360, "yuv420p"
        v.codec_context.gop_size = 250
        for i in range(25):
            f = av.VideoFrame.from_ndarray(np.full((360, 640, 3), 8 * i, np.uint8), format="rgb24")
            f.pts = i
            for p in v.encode(f):
                out.mux(p)
        for p in v.encode():
            out.mux(p)
    monkeypatch.setattr(ci_mod.ControlInterface, "_build_media_list", lambda self: None)
    monkeypatch.setattr(ci_mod.ControlInterface, "_update_cell_preview", lambda self, r, c: None)
    monkeypatch.setattr(pc, "SIZES", (("Same size", None), ("320 at most", (320, 180))))
    obj = _make_ci(tmp_path, monkeypatch)
    obj.project_media_paths = [src]
    obj.grid_cells[(0, 3)] = src
    obj.cell_positions[(0, 3)] = {"x": 0.5, "y": 0.5, "scale_x": 0.5, "scale_y": 0.5, "rotation": 0}
    root = tk.Tk()
    root.withdraw()
    obj.root = root
    try:
        win = obj._open_playback_copy(src)
        vd = win._vd
        vd["size_var"].set("320 at most")
        dst = vd["make"]()
        t0 = time.monotonic()
        while vd["st"]["running"] and time.monotonic() - t0 < 30:
            root.update()
            time.sleep(0.05)
        assert vd["st"]["result"] and vd["st"]["result"]["size"] == (320, 180), vd["status"].cget("text")
        assert dst == str(tmp_path / "big (playback 180p).mp4") and os.path.exists(dst) and os.path.exists(src)
        assert obj.grid_cells[(0, 3)] == dst and obj.project_media_paths == [dst]
        pos = obj.cell_positions[(0, 3)]
        assert (pos["scale_x"], pos["scale_y"]) == (1.0, 1.0), "the same size on the output"
        assert "Used in 1 cell(s)" in vd["status"].cget("text")
        keys, n = keyframe_gaps(dst)
        assert keys == list(range(0, n, 12)) or keys == list(range(0, n, 13)), keys   # 25 fps: every half second
        win.destroy()
    finally:
        root.destroy()


def _make_ci(tmp_path, monkeypatch):
    import control_interface as ci_mod
    from media_manager import MediaManager
    from test_control_state import Var
    from test_cue_activation import FakeEngine, FakeRoot
    monkeypatch.setattr(ci_mod.ControlInterface, "CONFIG_DIR", str(tmp_path / ".visualdeck"))
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    monkeypatch.setattr(ci_mod.ControlInterface, "_update_title", lambda self: None)
    obj = ci_mod.ControlInterface(MediaManager())
    obj.screen_var = Var("Display 0")
    obj.transition_var = Var("Fade")
    obj.transition_timing_var = Var("1.0")
    obj.presentation_mode_var = Var(False)
    obj.re = FakeEngine()
    obj.root = FakeRoot()
    obj.col_labels = {}
    obj._update_cue_dropdown = lambda: None
    obj.set_preview_dirty = lambda: None
    obj.active_col = 0
    return obj
