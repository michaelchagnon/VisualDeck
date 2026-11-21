# SPDX-License-Identifier: MPL-2.0
"""Stills of videos (thumbnails, the position editor, the edited-cue preview).

A video that fades in from black has a black first frame; the owner saw no
picture for such a video in the media list, the editor or the edited-cue
preview (2026-11-15). render_engine.video_still now gives the first frame that
is not dark.
"""
import os
import time

import numpy as np
import pytest

from media_fixtures import _encode_yuv_frames, solid_yuv

W, H, FPS = 320, 180, 30
BLACK = (16, 128, 128)
RED = (63, 102, 240)          # BT.709 limited range
GRAY = (120, 128, 128)


def _clip(path, parts):
    """parts: [(n_frames, (y, u, v) or ndarray)]"""
    frames = []
    for n, what in parts:
        arr = what if isinstance(what, np.ndarray) else solid_yuv(W, H, *what)
        frames += [arr] * n
    return _encode_yuv_frames(str(path), frames, W, H, fps=FPS)


def _rgb_mean(img):
    return np.asarray(img.convert('RGB'), dtype=np.float64).reshape(-1, 3).mean(axis=0)


def _is_red(img):
    r, g, b = _rgb_mean(img)
    return r > 200 and g < 40 and b < 40


def test_a_video_that_fades_in_from_black_shows_a_picture(tmp_path):
    from render_engine import video_still
    path = _clip(tmp_path / "fade.mp4", [(30, BLACK), (60, RED)])     # black for 1 s
    img = video_still(path)
    assert img is not None and img.size == (W, H)
    assert _is_red(img), _rgb_mean(img)


def test_a_longer_black_start_uses_a_later_point(tmp_path):
    from render_engine import video_still
    path = _clip(tmp_path / "late.mp4", [(75, BLACK), (105, RED)])    # 6 s, black until 2.5 s
    assert _is_red(video_still(path))                                  # 1 s, 10 %, 25 % are black; 50 % is not


def test_a_first_frame_with_a_picture_is_kept(tmp_path):
    from render_engine import video_still
    path = _clip(tmp_path / "gray.mp4", [(1, GRAY), (40, RED)])
    r, g, b = _rgb_mean(video_still(path))
    assert abs(r - g) < 4 and abs(g - b) < 4 and 100 < r < 140          # the gray first frame, not the red


def test_a_title_on_black_is_kept(tmp_path):
    from render_engine import video_still
    title = solid_yuv(W, H, *BLACK)
    title[80:100, 120:200] = 235                                        # a small white title
    path = _clip(tmp_path / "title.mp4", [(1, title), (40, RED)])
    img = video_still(path)
    assert not _is_red(img) and np.asarray(img.convert('RGB')).max() > 200


def test_an_all_black_video_still_gives_its_picture(tmp_path):
    from render_engine import video_still
    path = _clip(tmp_path / "black.mp4", [(60, BLACK)])
    img = video_still(path)
    assert img is not None and img.size == (W, H) and _rgb_mean(img).max() < 8


def test_a_broken_file_gives_none(tmp_path):
    from render_engine import video_still
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video" * 100)
    assert video_still(str(bad)) is None


def test_the_edited_cue_preview_and_the_media_list_show_the_picture(tmp_path):
    import preview_compose
    from control_interface import ControlInterface
    path = _clip(tmp_path / "fade.mp4", [(30, BLACK), (60, RED)])
    img, _ = preview_compose.StillSource().get(path)
    assert img.mode == 'RGBA' and _is_red(img)
    still = ControlInterface._read_first_frame(None, path, False)        # the media list and cue thumbnails
    assert still.mode == 'RGBA' and _is_red(still)


def test_finding_the_picture_is_quick(tmp_path):
    from render_engine import video_still
    path = _clip(tmp_path / "fade.mp4", [(30, BLACK), (60, RED)])
    video_still(path)
    t0 = time.perf_counter()
    video_still(path)
    assert time.perf_counter() - t0 < 1.0


def test_a_key_frame_after_the_point_is_used_without_decoding_up_to_it(tmp_path):
    """With key frames every 12 frames, the still comes from the first key frame at
    or after 1 s (one frame decoded); the owner's 4K clip took 1.9 s when every
    frame up to 1 s was decoded."""
    from fractions import Fraction
    import av
    from render_engine import video_still
    path = str(tmp_path / "gop12.mp4")
    c = av.open(path, mode="w")
    s = c.add_stream("libx264", rate=FPS)
    s.width, s.height, s.pix_fmt = W, H, "yuv420p"
    s.options = {"preset": "ultrafast", "g": "12", "keyint_min": "12", "sc_threshold": "0", "bf": "0"}
    for i in range(90):         # black for 1 s, gray up to the key frame at 1.2 s, then red, then green from 2 s
        yuv = BLACK if i < 30 else GRAY if i < 36 else RED if i < 60 else (145, 54, 34)
        f = av.VideoFrame.from_ndarray(solid_yuv(W, H, *yuv), format="yuv420p")
        f.pts, f.time_base = i, Fraction(1, FPS)
        for pkt in s.encode(f):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()
    img = video_still(path)
    assert _is_red(img), _rgb_mean(img)    # the key frame at 1.2 s (frame 36): not decoded up to 1 s (gray)
