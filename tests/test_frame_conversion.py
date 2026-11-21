# SPDX-License-Identifier: MPL-2.0
"""YUV handling shared by the shader path and the CPU preview path."""
import av
import numpy as np
import pytest

import render_engine as rem


def test_matrix_maps_limited_black_and_white():
    m, off = rem.yuv_to_rgb_matrix("limited", "bt709")
    black = m @ (np.array([16, 128, 128]) / 255.0 - off)
    white = m @ (np.array([235, 128, 128]) / 255.0 - off)
    assert np.allclose(black, 0, atol=1e-3) and np.allclose(white, 1, atol=1e-3)


@pytest.mark.parametrize("matrix,yuv", [("bt709", (63, 102, 240)), ("bt601", (81, 90, 240))])
def test_matrix_red(matrix, yuv):
    m, off = rem.yuv_to_rgb_matrix("limited", matrix)
    rgb = m @ (np.array(yuv) / 255.0 - off)
    assert np.allclose(rgb, [1, 0, 0], atol=0.02), rgb


def test_full_range_black():
    m, off = rem.yuv_to_rgb_matrix("full", "bt709")
    assert np.allclose(m @ (np.array([0, 128, 128]) / 255.0 - off), 0, atol=1e-3)


def _frame(fmt, w, h, y=16, u=128, v=128):
    f = av.VideoFrame(w, h, fmt)
    f.planes[0].update(np.full(f.planes[0].buffer_size, y, np.uint8).tobytes())
    if fmt == "nv12":
        uv = np.empty(f.planes[1].buffer_size, np.uint8)
        uv[0::2], uv[1::2] = u, v
        f.planes[1].update(uv.tobytes())
    else:
        f.planes[1].update(np.full(f.planes[1].buffer_size, u, np.uint8).tobytes())
        f.planes[2].update(np.full(f.planes[2].buffer_size, v, np.uint8).tobytes())
    return f


@pytest.mark.parametrize("w,h", [(1920, 1080), (854, 480), (853, 479)])
def test_yuv420p_plane_extraction_handles_padding_and_odd_sizes(w, h):
    out = rem.convert_video_frame(_frame("yuv420p", w, h, y=50, u=60, v=70))
    assert out["format"] == "yuv420p"
    assert out["y"].shape == (h, w) and out["u"].shape == ((h + 1) // 2, (w + 1) // 2)
    assert out["y"].flags.c_contiguous and out["u"].flags.c_contiguous
    assert (out["y"] == 50).all() and (out["u"] == 60).all() and (out["v"] == 70).all()


def test_yuvj420p_is_full_range_and_not_rescaled():
    out = rem.convert_video_frame(_frame("yuvj420p", 64, 64, y=0))
    assert out["range"] == "full" and (out["y"] == 0).all()


def test_nv12_is_deinterleaved():
    out = rem.convert_video_frame(_frame("nv12", 64, 32, y=40, u=90, v=200))
    assert (out["u"] == 90).all() and (out["v"] == 200).all()


def test_matrix_defaults_from_resolution():
    assert rem.convert_video_frame(_frame("yuv420p", 1280, 720))["matrix"] == "bt709"
    assert rem.convert_video_frame(_frame("yuv420p", 720, 576))["matrix"] == "bt601"


def test_alpha_formats_become_rgba():
    f = av.VideoFrame.from_ndarray(np.zeros((8, 8, 4), np.uint8), format="rgba")
    out = rem.convert_video_frame(f)
    assert out["format"] == "rgba" and out["rgba"].shape == (8, 8, 4)


def test_cpu_conversion_colors_match_shader_math():
    for yuv, expect in [((16, 128, 128), (0, 0, 0)), ((235, 128, 128), (255, 255, 255)), ((63, 102, 240), (255, 0, 0))]:
        fd = rem.convert_video_frame(_frame("yuv420p", 64, 36, *yuv))
        fd["matrix"] = "bt709"
        rgb = rem.yuv_frame_to_rgb(fd, (32, 18))
        assert rgb.shape == (18, 32, 3)
        assert np.abs(rgb[9, 16].astype(int) - expect).max() <= 4, (yuv, rgb[9, 16])


def test_cpu_preview_does_not_consume_output_frames():
    eng = rem.BaseRenderEngine(media_manager=None)

    class Q:
        def __init__(self):
            self.taken = 0

        def get_latest(self):
            self.taken += 1
            return rem.convert_video_frame(_frame("yuv420p", 64, 36))

    q = Q()
    eng.active_players = {"/v.mp4": {"queue": q, "player": None, "type": "gpu_queue"}}
    layers = {0: {"path": "/v.mp4", "order": 0, "position": {}}}
    eng._get_latest_video_frame("/v.mp4")   # the output fetched a frame
    for _ in range(3):
        img = eng.render_layers_to_image(layers, (64, 36))
    assert q.taken == 1, "preview pulled frames from the output's queue"
    assert img.getpixel((32, 18))[:3] == (0, 0, 0)


@pytest.mark.parametrize("fmt", ["rgb0", "bgr0", "rgb24", "bgr24", "gbrp", "0rgb", "0bgr"])
@pytest.mark.parametrize("w,h", [(1920, 1080), (333, 101)])
def test_rgb_formats_become_rgba_at_full_color_resolution(fmt, w, h):
    """HAP decodes to rgb0, DXV and QuickTime Animation to RGB: they are
    uploaded as RGBA (alpha 255) instead of being turned into 4:2:0."""
    src = (np.random.default_rng(w).random((h, w, 3)) * 255).astype(np.uint8)
    frame = av.VideoFrame.from_ndarray(src, format="rgb24")
    if fmt != "rgb24":
        frame = frame.reformat(format=fmt)
    out = rem.convert_video_frame(frame)
    assert out["format"] == "rgba" and out["rgba"].shape == (h, w, 4) and out["rgba"].flags.c_contiguous
    assert np.array_equal(out["rgba"][..., :3], src) and (out["rgba"][..., 3] == 255).all()
    plane = np.frombuffer(frame.planes[0], np.uint8)
    assert not np.shares_memory(out["rgba"], plane), "the decoder reuses the frame's memory"
    before = bytes(frame.planes[0])
    rem.convert_video_frame(frame)
    assert bytes(frame.planes[0]) == before, "the frame itself is never written"
