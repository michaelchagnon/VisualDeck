# SPDX-License-Identifier: MPL-2.0
"""Colors outside Performance Mode's 4:2:0 shader path: PyAV's own YUV to RGB
conversion used BT.601 limited range for every frame (and OpenCV's reader the
same), so HD video (BT.709) had the wrong colors in Balanced and Compatibility
Mode, in videos with alpha (ProRes 4444), thumbnails and the editor."""
import os

import av
import numpy as np
import pytest

import media_fixtures as mf
import render_engine as rem


def near(a, b, tol=4):
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b))


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("colors")
    return {
        "red_709_hd": mf.solid_clip(str(d / "r709.mp4"), 1280, 720, 63, 102, 240),
        "red_601_sd": mf.solid_clip(str(d / "r601.mp4"), 640, 360, 81, 90, 240, colorspace=mf.CS_SMPTE170M),
        "gray_full": mf.solid_clip(str(d / "gfull.mp4"), 640, 360, 128, 128, 128, color_range=mf.RANGE_JPEG),
    }


def first(path):
    with av.open(path) as c:
        return next(c.decode(video=0))


def test_rgb_conversions_use_the_videos_matrix_and_range(clips):
    for name in ("red_709_hd", "red_601_sd", "red_709_hd"):        # twice: no state left behind
        f = first(clips[name])
        assert near(rem.to_rgb_array(f, "rgb24")[5, 5], (255, 0, 0)), name
        assert near(rem.to_rgb_array(f, "bgra")[5, 5], (0, 0, 255, 255)), name
    # PyAV's default gives the BT.601 result for the HD clip: what Balanced Mode showed before
    assert first(clips["red_709_hd"]).to_ndarray(format="rgb24")[5, 5, 0] < 240
    g = rem.to_rgb_array(first(clips["gray_full"]), "rgb24")[5, 5]
    assert near(g, (128, 128, 128), 2), g                            # full range: not stretched


def test_balanced_and_compatibility_frames_have_the_right_colors(clips):
    f = first(clips["red_709_hd"])
    for fmt, check in (("pygame", lambda o: np.frombuffer(o[0], np.uint8).reshape(720, 1280, 4)[5, 5]),
                       ("numpy", lambda o: o[5, 5])):
        p = rem.GPUVideoPlayerThread(clips["red_709_hd"], None, output_format=fmt)
        assert near(check(p._convert_frame(f))[:3], (0, 0, 255)), fmt        # BGR(A)


def _prores_4444_like(w, h, rgb=(220, 30, 30)):
    img = np.zeros((h, w, 4), np.uint8)
    img[..., :3], img[..., 3] = rgb, 255
    img[:, : w // 4, 3] = 0
    f = av.VideoFrame.from_ndarray(img, format="rgba").reformat(format="yuva444p", dst_colorspace="ITU709")
    f.colorspace = 2                                                  # untagged, as many files are
    return f


@pytest.mark.parametrize("w, h, scale", [(1920, 1080, 1.0), (3840, 2160, 0.5)])
def test_a_yuv_video_with_alpha_keeps_its_colors_and_transparency(w, h, scale):
    out = rem.convert_video_frame(_prores_4444_like(w, h), scale)
    rgba = out["rgba"]
    H, W = rgba.shape[:2]
    assert near(rgba[H // 2, W * 3 // 4], (220, 30, 30, 255)), rgba[H // 2, W * 3 // 4]
    assert rgba[H // 2, 5, 3] == 0


def test_stills_for_thumbnails_and_the_editor(clips):
    img = rem.video_still(clips["red_709_hd"])
    assert img.size == (1280, 720) and near(img.getpixel((5, 5)), (255, 0, 0))
    assert rem.video_still("/no/such/file.mp4") is None
    import preview_compose                                            # the EDITED CUE preview's stills
    still = preview_compose.StillSource._load(clips["red_709_hd"])
    assert near(still.getpixel((5, 5))[:3], (255, 0, 0)), still.getpixel((5, 5))


def test_compatibility_mode_decodes_with_pyav(clips):
    eng = rem.BaseRenderEngine(media_manager=None, mode="cpu")
    layer = {"path": clips["red_709_hd"], "screens": [0], "order": 0, "loop": True}
    try:
        eng.update_layers({0: layer})
        p = eng.active_players[eng.source_key(layer)]["player"]
        assert isinstance(p, rem.GPUVideoPlayerThread) and p.output_format == "numpy"
    finally:
        eng.stop_players(list(eng.active_players))
    assert os.path.exists(clips["red_709_hd"])
