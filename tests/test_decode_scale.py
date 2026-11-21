"""Decoding at the drawn size (Performance Mode): a clip larger than 1080p drawn at
# SPDX-License-Identifier: MPL-2.0
half its size or less is scaled down in its decoder thread, and drawn where the
full-size frames would be. Owner's PC, 2026-11-14: a 4K clip at Fit with Wave
was over the 1 % late-frame bar, a 1080p copy of it under."""
import av
import numpy as np
import pytest

import render_engine as rem
from conftest import run_gl_scenario


@pytest.mark.parametrize("sx, sy, warped, expected", [
    (0.5, 0.5, False, 0.5), (0.3, 0.4, False, 0.5), (0.25, 0.25, False, 0.25), (0.1, 0.2, False, 0.25),
    (0.51, 0.5, False, 1.0), (1.0, 1.0, False, 1.0), (-0.5, 0.5, False, 0.5),     # mirrored: its size counts
    (0.5, 0.5, True, 1.0), (0, 0, False, 1.0), ("x", None, False, 1.0),
])
def test_decode_scale_is_the_largest_step_at_least_the_drawn_size(sx, sy, warped, expected):
    assert rem.decode_scale_for(sx, sy, warped) == expected


def _yuv_frame(w, h, fmt="yuv420p", y=100, u=90, v=160, full=False):
    f = av.VideoFrame(w, h, fmt)
    if fmt == "nv12":
        f.planes[0].update(bytes([y]) * (f.planes[0].line_size * h))
        f.planes[1].update(bytes([u, v]) * (f.planes[1].line_size // 2 * ((h + 1) // 2)))
    else:
        for plane, val in zip(f.planes, (y, u, v)):
            plane.update(bytes([val]) * plane.buffer_size)
    if full:
        f.color_range = 2
    f.colorspace = 1
    return f


def test_a_4k_frame_at_half_size_keeps_its_values_and_reports_its_own_size():
    out = rem.convert_video_frame(_yuv_frame(3840, 2160, full=True), 0.5)
    assert (out["width"], out["height"], out["src_width"], out["src_height"]) == (1920, 1080, 3840, 2160)
    assert out["y"].shape == (1080, 1920) and out["u"].shape == (540, 960)
    assert int(out["y"].min()) == int(out["y"].max()) == 100           # scaled, not converted
    assert int(out["u"].mean()) == 90 and int(out["v"].mean()) == 160
    assert (out["range"], out["matrix"]) == ("full", "bt709")          # read before scaling


def test_quarter_size_and_hardware_nv12_frames():
    out = rem.convert_video_frame(_yuv_frame(3840, 2160, "nv12"), 0.25)
    assert out["format"] == "yuv420p" and (out["width"], out["height"]) == (960, 540)
    assert int(out["y"].mean()) == 100 and abs(int(out["u"].mean()) - 90) <= 1 and abs(int(out["v"].mean()) - 160) <= 1


def test_a_1080p_frame_is_never_scaled():
    out = rem.convert_video_frame(_yuv_frame(1920, 1080), 0.5)
    assert (out["width"], out["height"]) == (1920, 1080) and "src_width" not in out


def test_rgb_and_alpha_frames_at_half_size():
    rgba = np.zeros((2160, 3840, 4), np.uint8)
    rgba[..., 0], rgba[..., 3] = 200, 128
    out = rem.convert_video_frame(av.VideoFrame.from_ndarray(rgba, format="rgba"), 0.5)
    assert out["format"] == "rgba" and out["rgba"].shape == (1080, 1920, 4)
    assert int(out["rgba"][..., 0].mean()) == 200 and abs(int(out["rgba"][..., 3].mean()) - 128) <= 1   # alpha kept
    bgr0 = av.VideoFrame(3840, 2160, "bgr0")
    bgr0.planes[0].update(bytes([10, 20, 30, 0]) * (bgr0.planes[0].buffer_size // 4))
    out = rem.convert_video_frame(bgr0, 0.5)
    assert out["format"] == "rgba" and out["rgba"].shape == (1080, 1920, 4) and out["src_width"] == 3840
    assert out["rgba"][0, 0].tolist() == [30, 20, 10, 255]                # HAP-style RGB: opaque


def test_previews_of_a_scaled_frame_use_the_videos_own_size():
    small = rem.convert_video_frame(_yuv_frame(3840, 2160), 0.5)
    assert rem.BaseRenderEngine._video_frame_to_pil(small, 0.1, 0.1).size == (384, 216)


class _Player:
    """Stands in for a decoder thread in update_layers."""
    def __init__(self, path, q, *a, **kw):
        self.path, self.direction, self.trim, self.held = path, kw.get("direction", "forward"), kw.get("trim"), False
        self.decode_scale, self.loop = 1.0, True

    def set_speed(self, s):
        pass

    def start(self):
        pass

    def stop(self):
        pass


@pytest.mark.parametrize("engine_cls, mode, expected", [(rem.OpenGLRenderEngine, "opengl", 0.5),
                                                        (rem.PygameRenderEngine, "pygame_gpu", 0.5),
                                                        (rem.PygameRenderEngine, "cpu", 0.5)])
def test_the_engine_sets_each_decoders_size_from_its_layers(tmp_path, monkeypatch, engine_cls, mode, expected):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(rem, "GPUVideoPlayerThread", _Player)
    eng = engine_cls.__new__(engine_cls)
    eng.active_players, eng.video_frame_cache, eng.mode, eng.hw_accel = {}, {}, mode, None
    eng.release_texture = lambda key: None
    pos = {"x": 0.5, "y": 0.5, "scale_x": 0.5, "scale_y": 0.5, "rotation": 0}
    eng.update_layers({0: {"path": str(clip), "screens": [0], "order": 0, "position": pos}})
    p = next(iter(eng.active_players.values()))["player"]
    assert p.decode_scale == expected
    pos2 = dict(pos, scale_x=0.8, scale_y=0.8)                               # made larger live: full size again
    eng.update_layers({0: {"path": str(clip), "screens": [0], "order": 0, "position": pos2}})
    assert p.decode_scale == 1.0
    monkeypatch.setattr(rem, "DECODE_FULL_SIZE", True)                       # troubleshooting switch
    eng.update_layers({0: {"path": str(clip), "screens": [0], "order": 0, "position": pos}})
    assert p.decode_scale == 1.0


def test_a_4k_clip_at_half_size_is_drawn_where_it_was(fixtures, x_display):
    r = run_gl_scenario("decode_at_drawn_size", fixtures, x_display)
    W = r["output"][0]
    white, black = [255, 255, 255], [0, 0, 0]

    def near(a, b, tol=12):
        return all(abs(x - y) <= tol for x, y in zip(a, b))
    h = r["half"]
    if W == 1920:
        assert h["decode_scale"] == 0.5 and h["scaled_to"] == [1920, 1080], h
    assert h["tex_size"] == [3840, 2160]                                     # the geometry's size
    assert near(h["left_edge"], white) and near(h["top_left"], white), h     # fills the output
    assert near(h["left_of_middle"], white) and near(h["right_of_middle"], black), h
    for name in ("larger", "warped"):
        assert r[name]["decode_scale"] == 1.0 and r[name]["scaled_to"] is None, r[name]
    assert near(r["larger"]["left_of_middle"], white) and near(r["larger"]["right_of_middle"], black)
    inv = r["inverted"]
    assert r["fx_error"] is None and near(inv["left_of_middle"], black) and near(inv["right_of_middle"], white), inv


def test_balanced_mode_frames_at_the_drawn_size(monkeypatch, tmp_path):
    """Balanced Mode: the decoder hands over BGRA at the drawn size with the video's
    own size; the layer is drawn where the full-size frame was, with its colors."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    f = _yuv_frame(3840, 2160, y=63, u=102, v=240)                  # BT.709 red
    p = rem.GPUVideoPlayerThread("x.mp4", None, output_format="pygame")
    p.decode_scale = 0.5
    item = p._convert_frame(f)
    assert item[1] == (1920, 1080) and item[2] == (3840, 2160) and p.scaled_to == (1920, 1080)
    px = np.frombuffer(item[0], np.uint8).reshape(1080, 1920, 4)[5, 5]
    assert abs(int(px[2]) - 255) <= 3 and int(px[0]) <= 3 and int(px[1]) <= 3, px
    p.decode_scale = 1.0
    full = p._convert_frame(f)
    assert len(full) == 2 and full[1] == (3840, 2160) and p.scaled_to is None
    assert rem.BaseRenderEngine._video_frame_to_pil(item, 0.1, 0.1).size == (384, 216)
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        eng = rem.PygameRenderEngine(media_manager=None, mode="pygame_gpu")
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"x")
        layer = {"path": str(clip), "order": 0, "position": {"x": 0.5, "y": 0.5, "scale_x": 0.5, "scale_y": 0.5}}
        monkeypatch.setattr(eng, "_get_latest_video_frame", lambda key: item)
        sf = eng._to_surface(layer)
        assert sf.get_size() == (1920, 1080)                       # the video's own size x 0.5
        assert tuple(sf.get_at((10, 10)))[:3][0] >= 252
    finally:
        pygame.display.quit()


def test_compatibility_mode_frames_at_the_drawn_size(monkeypatch, tmp_path):
    """Compatibility Mode: the decoder hands over a BGR array at the drawn size that
    carries the video's own size; the layer is drawn at the size the full frame was."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    f = _yuv_frame(3840, 2160, y=63, u=102, v=240)                  # BT.709 red
    p = rem.GPUVideoPlayerThread("x.mp4", None, output_format="numpy")
    p.decode_scale = 0.5
    arr = p._convert_frame(f)
    assert arr.shape == (1080, 1920, 3) and arr.own_size == (3840, 2160) and p.scaled_to == (1920, 1080)
    assert abs(int(arr[5, 5, 2]) - 255) <= 3 and int(arr[5, 5, 0]) <= 3 and int(arr[5, 5, 1]) <= 3, arr[5, 5]
    assert getattr(arr[:, :10], "own_size", None) is None                 # a part of it is not the video
    assert rem.BaseRenderEngine._video_frame_to_pil(arr, 0.1, 0.1).size == (384, 216)
    p.decode_scale = 1.0
    full = p._convert_frame(f)
    assert full.shape == (2160, 3840, 3) and getattr(full, "own_size", None) is None and p.scaled_to is None
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    try:
        eng = rem.PygameRenderEngine(media_manager=None, mode="cpu")
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"x")
        layer = {"path": str(clip), "order": 0, "position": {"x": 0.5, "y": 0.5, "scale_x": 0.5, "scale_y": 0.5}}
        monkeypatch.setattr(eng, "_get_latest_video_frame", lambda key: arr)
        sf = eng._to_surface(layer)
        assert sf.get_size() == (1920, 1080)                       # the video's own size x 0.5
        assert tuple(sf.get_at((10, 10)))[:3][0] >= 252
        layer["position"].update(scale_x=0.25, scale_y=0.25)
        assert eng._to_surface(layer).get_size() == (960, 540)
    finally:
        pygame.display.quit()
