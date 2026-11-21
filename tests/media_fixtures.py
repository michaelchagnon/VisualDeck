# SPDX-License-Identifier: MPL-2.0
"""Synthetic media generation for VisualDeck tests and benchmarks.

Everything here is generated with PyAV (which bundles FFmpeg), so tests do
not depend on an ffmpeg CLI being installed or on any copyrighted footage.

All clips are tiny and deterministic. Solid-color clips are written with
exact Y/U/V code values so color-conversion tests can check exact results.
"""
from __future__ import annotations

import os
from fractions import Fraction

import av
import numpy as np
from PIL import Image

# AVColorSpace / AVColorRange values as exposed by PyAV
CS_BT709 = 1
CS_SMPTE170M = 6     # BT.601 (NTSC)
RANGE_MPEG = 1       # limited / "tv" range
RANGE_JPEG = 2       # full / "pc" range


def _encode_yuv_frames(path, frames_yuv, width, height, fps=30,
                       colorspace=CS_BT709, color_range=RANGE_MPEG, crf="1"):
    """Encode a list of yuv420p ndarrays (shape (h*3/2, w)) as H.264."""
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = width, height
    stream.pix_fmt = "yuv420p"
    stream.codec_context.colorspace = colorspace
    stream.codec_context.color_range = color_range
    stream.options = {"crf": crf, "preset": "ultrafast", "tune": "zerolatency"}
    for i, arr in enumerate(frames_yuv):
        frame = av.VideoFrame.from_ndarray(arr, format="yuv420p")
        frame.colorspace = colorspace
        frame.color_range = color_range
        frame.pts = i
        frame.time_base = Fraction(1, fps)
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return path


def solid_yuv(width, height, y, u, v):
    """A yuv420p ndarray filled with constant code values."""
    cw, ch = width // 2, height // 2
    arr = np.empty((height * 3 // 2, width), dtype=np.uint8)
    arr[:height] = y
    # U and V planes are stored consecutively after Y, each cw*ch bytes
    flat = arr[height:].reshape(-1)
    flat[: cw * ch] = u
    flat[cw * ch: 2 * cw * ch] = v
    return arr


def solid_clip(path, width, height, y, u, v, n_frames=15, **kw):
    arr = solid_yuv(width, height, y, u, v)
    return _encode_yuv_frames(path, [arr] * n_frames, width, height, **kw)


def split_clip(path, width, height, n_frames=15, **kw):
    """Left half white (Y=235), right half black (Y=16): detects row shear."""
    arr = solid_yuv(width, height, 16, 128, 128)
    arr[:height, : width // 2] = 235
    return _encode_yuv_frames(path, [arr] * n_frames, width, height, **kw)


def counting_clip(path, width=320, height=180, n_frames=60, fps=30, **kw):
    """Each frame's luma encodes its index (Y = 16 + 3*i) for ordering tests."""
    frames = [solid_yuv(width, height, min(235, 16 + 3 * i), 128, 128) for i in range(n_frames)]
    return _encode_yuv_frames(path, frames, width, height, fps=fps, **kw)


def testsrc_clip(path, width=1920, height=1080, n_frames=60, fps=30):
    """Moving gradient pattern, representative of real decode cost."""
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "23", "preset": "ultrafast"}
    xs = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    ys = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    for i in range(n_frames):
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = ((xs + i * 4) % 256).astype(np.uint8)
        rgb[..., 1] = ((ys + i * 2) % 256).astype(np.uint8)
        rgb[..., 2] = 128
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
        frame.pts = i
        frame.time_base = Fraction(1, fps)
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return path


def alpha_clip(path, width=320, height=180, n_frames=10, fps=30):
    """ProRes 4444 with alpha: left half opaque red, right half fully transparent."""
    container = av.open(path, mode="w")
    stream = container.add_stream("prores_ks", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuva444p10le"
    stream.options = {"profile": "4444"}
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[:, : width // 2] = (255, 0, 0, 255)
    for i in range(n_frames):
        frame = av.VideoFrame.from_ndarray(rgba, format="rgba").reformat(format="yuva444p10le")
        frame.pts = i
        frame.time_base = Fraction(1, fps)
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return path


def solid_png(path, size, rgba):
    Image.new("RGBA", size, rgba).save(path)
    return path


def build_fixture_set(directory):
    """Create the standard fixture set used by the test-suite. Returns a dict of paths."""
    os.makedirs(directory, exist_ok=True)
    p = lambda name: os.path.join(directory, name)
    out = {
        # BT.709 limited range: black, white and a saturated red
        "black_709": solid_clip(p("black_709.mp4"), 640, 360, 16, 128, 128),
        "white_709": solid_clip(p("white_709.mp4"), 640, 360, 235, 128, 128),
        # BT.709 limited-range code values for RGB (255, 0, 0)
        "red_709": solid_clip(p("red_709.mp4"), 640, 360, 63, 102, 240),
        # BT.601 limited-range code values for RGB (255, 0, 0)
        "red_601": solid_clip(p("red_601.mp4"), 640, 360, 81, 90, 240, colorspace=CS_SMPTE170M),
        # Full-range black
        "black_full": solid_clip(p("black_full.mp4"), 640, 360, 0, 128, 128, color_range=RANGE_JPEG),
        # 854 px wide: Y rows are not a multiple of 4 bytes
        "split_854": split_clip(p("split_854.mp4"), 854, 480),
        "counting": counting_clip(p("counting.mp4")),
        "short_loop": counting_clip(p("short_loop.mp4"), n_frames=8),
        "alpha_video": alpha_clip(p("alpha_4444.mov")),
        "red_png": solid_png(p("red.png"), (400, 300), (255, 0, 0, 255)),
        "green_png": solid_png(p("green.png"), (200, 150), (0, 255, 0, 255)),
    }
    return out
