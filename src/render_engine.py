# SPDX-License-Identifier: MPL-2.0
# =====================================================================
# src/render_engine.py
# =====================================================================
import collections
import math
import functools
import os
import sys
import ctypes
import pygame
import numpy as np
import cv2
import threading
import multiprocessing
import time
import av
import logging
import edge_blend
import isf
import layer_mask
import layer_warp
import live_source
import mesh_warp
import output_warp
import sdl_audio_safe
from ring_buffer import FastRingBuffer, SimpleFrameQueue
from queue import Queue, Empty, Full
from threading import Lock
from pygame.locals import *
from displays import monitors as get_monitors   # screeninfo without its leaks (displays.py)
from PIL import Image, ImageDraw
from collections import deque
from collections import OrderedDict

# --- PyOpenGL flags ---
# These are read when OpenGL.GL is first imported, so they must be set before
# the import below (setting them afterwards, as earlier versions did, has no
# effect). Error checking turns every GL error into a Python exception and adds
# a glGetError() after every call; keep it for debugging only.
import OpenGL
if os.environ.get('VISUALDECK_DEBUG') != '1':
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_LOGGING = False
    OpenGL.CONTEXT_CHECKING = False

# --- OpenGL specific imports ---
from OpenGL.GL import *
from OpenGL.GL.ARB.map_buffer_range import glMapBufferRange, GL_MAP_WRITE_BIT, GL_MAP_UNSYNCHRONIZED_BIT
from OpenGL.GL import shaders
from OpenGL.error import GLError
from OpenGL.GL.ARB.pixel_buffer_object import *

log = logging.getLogger("visualdeck.render")

# --- Constants ---
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')


def scale_bgra(arr, width, height):
    """Resize an image array: area averaging when shrinking (with a cheap
    first step for big reductions), bilinear when enlarging."""
    h, w = arr.shape[:2]
    if (width, height) == (w, h):
        return arr
    if width <= w and height <= h:
        if w > 3 * width and h > 3 * height:
            arr = cv2.resize(arr, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)


def is_streamed(path):
    """True for media that delivers frames from a player: video files and live sources."""
    return bool(path) and (live_source.is_live(path) or os.path.splitext(path)[1].lower() in VIDEO_EXTS)
FRAME_BUFFER_SIZE = 30  # Increased for GPU players
CPU_FRAME_BUFFER_SIZE = 10 # Added for CPU players

# ── PreciseFramePacer (drop-in replacement) ──────────────────────────────────
class PreciseFramePacer:
    """
    High-precision frame pacer with dynamic retargeting and basic overrun stats.
    - target_fps can be changed at runtime via set_target_fps()
    - actual_fps is a smoothed EMA
    - reports % of frames that overran their budget in the last N samples
    """
    def __init__(self, target_fps=60.0, ema_alpha=0.1, window=120):
        self._target_fps = max(1.0, float(target_fps))
        self._frame_period = 1.0 / self._target_fps
        self._last_ts = time.perf_counter()
        self.actual_fps = 0.0
        self._ema_alpha = float(ema_alpha)
        self._overruns = deque(maxlen=window)
        self._window = window

    def set_target_fps(self, fps: float):
        fps = max(1.0, float(fps))
        if abs(fps - self._target_fps) > 0.1:
            self._target_fps = fps
            self._frame_period = 1.0 / self._target_fps
            self._deadline = None

    def get_target_fps(self) -> float:
        return self._target_fps

    def reset(self):
        """Start timing afresh (e.g. after a pause) instead of rushing to catch up."""
        self._last_ts = time.perf_counter()
        self._deadline = None

    def wait_for_next_frame(self):
        """Wait for the next tick of a fixed grid (deadline += period), so a
        late wake-up (Windows sleeps can run 1-2 ms over) is made up on the next
        frame instead of stretching every period: the average rate stays at the
        target. Further behind than one period (a slow frame, a pause), the
        grid restarts from now rather than rushing to catch up."""
        deadline = getattr(self, '_deadline', None)
        if deadline is None:
            deadline = self._last_ts + self._frame_period
        now = time.perf_counter()
        behind = now - deadline              # > 0: this tick is already late (slow work)
        if behind > self._frame_period:
            deadline = now
        sleep_time = deadline - now

        # Hybrid sleep+spin
        if sleep_time > 0.002:
            time.sleep(sleep_time - 0.001)
        # Busy wait the final ~1ms
        while time.perf_counter() < deadline:
            pass

        end = time.perf_counter()
        frame_time = end - self._last_ts
        self._last_ts = end
        self._deadline = deadline + self._frame_period

        # Update stats
        if self.actual_fps == 0.0:
            self.actual_fps = 1.0 / frame_time if frame_time > 0 else self._target_fps
        else:
            inst = 1.0 / frame_time if frame_time > 0 else self._target_fps
            self.actual_fps = (self._ema_alpha * inst) + ((1 - self._ema_alpha) * self.actual_fps)

        # An overrun is a tick at least half a period after its slot (a frame
        # shown late), not a wake-up a millisecond late that the grid makes up
        # on the next frame. The old rule (any interval over 1.05 periods) let
        # timing jitter alone push the adaptive rate down to 45/30 fps; the
        # owner's log of 2026-11-13 shows it stepping down each time a 4K cue
        # started.
        lateness = max(behind, end - deadline)
        overran = lateness > 0.5 * self._frame_period
        self._overruns.append(1 if overran else 0)

        return frame_time  # caller can also inspect per-frame cost

    def overrun_ratio(self) -> float:
        if not self._overruns:
            return 0.0
        return sum(self._overruns) / len(self._overruns)
# ─────────────────────────────────────────────────────────────────────────────



# --- Video frame conversion ---------------------------------------------------
#
# Video frames travel from decoder threads to the renderer as plain dicts:
#   {'format': 'yuv420p', 'width', 'height', 'y', 'u', 'v' (contiguous uint8
#    arrays of shape (h, w) and (ch, cw)), 'range': 'limited'|'full',
#    'matrix': 'bt601'|'bt709'|'bt2020', 'timestamp'}
# or, for sources with an alpha channel (HAP Alpha, ProRes 4444, PNG/QT RLE…)
# and for RGB sources (HAP, DXV, QuickTime Animation, PNG, uncompressed RGB):
#   {'format': 'rgba', 'width', 'height', 'rgba': (h, w, 4) uint8, 'timestamp'}

# AVColorSpace values (libavutil/pixfmt.h)
_CS_MATRIX = {1: 'bt709', 5: 'bt601', 6: 'bt601', 9: 'bt2020', 10: 'bt2020'}
# Kr, Kb luma coefficients per matrix
_LUMA_COEFFS = {'bt601': (0.299, 0.114), 'bt709': (0.2126, 0.0722), 'bt2020': (0.2627, 0.0593)}
_ALPHA_FORMATS_PREFIXES = ('rgba', 'bgra', 'argb', 'abgr', 'yuva', 'gbrap', 'ya')
# RGB without alpha (FFmpeg decodes HAP to rgb0, QuickTime Animation to rgb24, ...):
# uploaded as RGBA with alpha 255. Turning them into 4:2:0 cost more (about 5 ms
# a 1080p frame here against 1-3 ms) and halved the color resolution.
_RGBX_ORDER = ('rgb0', 'bgr0')                                 # 4 bytes a pixel, the 4th unused
_RGB_FORMATS_PREFIXES = ('rgb24', 'bgr24', '0rgb', '0bgr', 'gbrp', 'rgb48', 'bgr48')


def layer_opacity(layer):
    """A layer's opacity (0..1) from its 'opacity' key; bad values count as 1."""
    try:
        v = float(layer.get('opacity', 1.0))
    except (TypeError, ValueError):
        return 1.0
    return 0.0 if v < 0 else (1.0 if v > 1 or v != v else v)


BLEND_MODES = ('normal', 'add', 'multiply', 'screen')
# (src factor, dst factor, premultiplied color). With the shader's color c and
# coverage a = texture alpha x opacity x fade:
#   add      dst + c*a                       multiply  dst*(1 - a + a*c)
#   screen   dst + a*c*(1 - dst)
_GL_BLEND = {'add': (GL_SRC_ALPHA, GL_ONE, False),
             'multiply': (GL_DST_COLOR, GL_ONE_MINUS_SRC_ALPHA, True),
             'screen': (GL_ONE, GL_ONE_MINUS_SRC_COLOR, True)}


def layer_blend(layer):
    m = layer.get('blend', 'normal')
    return m if m in BLEND_MODES else 'normal'


def blend_onto(target, pil_img, xy, mode):
    """Composite an RGBA PIL image onto RGBA `target` at xy with a blend mode
    (CPU previews; same math as the GL blend functions)."""
    if mode == 'normal':
        target.alpha_composite(pil_img, xy)
        return
    x0, y0 = max(0, xy[0]), max(0, xy[1])
    x1, y1 = min(target.width, xy[0] + pil_img.width), min(target.height, xy[1] + pil_img.height)
    if x1 <= x0 or y1 <= y0:
        return
    src = np.asarray(pil_img.crop((x0 - xy[0], y0 - xy[1], x1 - xy[0], y1 - xy[1])), dtype=np.float32) / 255.0
    region = target.crop((x0, y0, x1, y1))
    dst = np.asarray(region, dtype=np.float32) / 255.0
    c, a = src[..., :3], src[..., 3:4]
    d = dst[..., :3]
    if mode == 'add':
        out = d + c * a
    elif mode == 'multiply':
        out = d * (1.0 - a + a * c)
    else:  # screen
        out = d + a * c * (1.0 - d)
    res = np.empty_like(dst)
    res[..., :3] = np.clip(out, 0.0, 1.0)
    res[..., 3:4] = dst[..., 3:4]
    target.paste(Image.fromarray((res * 255.0 + 0.5).astype(np.uint8)), (x0, y0))


def transition_alphas(transition_state):
    """(outgoing, incoming) opacity for a transition at its progress.
    linear: straight fade; smooth: ease in and out (smoothstep); dip: the
    outgoing cue fades to black in the first half, the incoming one fades up
    in the second."""
    try:
        p = min(1.0, max(0.0, float(transition_state.get('progress', 1.0))))
    except (TypeError, ValueError):
        p = 1.0
    curve = transition_state.get('curve', 'linear')
    if curve == 'smooth':
        q = p * p * (3.0 - 2.0 * p)
        return 1.0 - q, q
    if curve == 'dip':
        return max(0.0, 1.0 - 2.0 * p), max(0.0, 2.0 * p - 1.0)
    return 1.0 - p, p


def layer_volume(layer):
    """A layer's audio volume, 0..2 (1 = as recorded); bad values count as 1."""
    try:
        v = float(layer.get('volume', 1.0))
    except (TypeError, ValueError):
        return 1.0
    return 1.0 if v != v else min(2.0, max(0.0, v))


def _normal_blend():
    """The usual blending: colors as glBlendFunc(SRC_ALPHA, ONE_MINUS_SRC_ALPHA),
    and alpha adds up to the picture's coverage (ONE, ONE_MINUS_SRC_ALPHA), which
    a display's effect needs to keep uncovered parts black. Nothing else reads it."""
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)


def layer_effects(layer):
    """A layer's shader effects in order ([{'name', 'inputs'}], isf.layer_stack)."""
    return isf.layer_stack(layer)


def layer_color(layer):
    """A layer's color correction as (brightness -1..1, contrast 0..2,
    saturation 0..2), or None when it has none. Bad values count as neutral."""
    def num(key, neutral, lo, hi):
        try:
            v = float(layer.get(key, neutral))
        except (TypeError, ValueError):
            return neutral
        return neutral if v != v else min(hi, max(lo, v))
    adj = (num('brightness', 0.0, -1.0, 1.0), num('contrast', 1.0, 0.0, 2.0), num('saturation', 1.0, 0.0, 2.0))
    return None if adj == (0.0, 1.0, 1.0) else adj


_LUMA_SAT = (0.299, 0.587, 0.114)   # weights for saturation; cv2's RGB->GRAY uses the same


@functools.lru_cache(maxsize=64)
def _bc_lut(brightness, contrast, channels):
    x = np.arange(256, dtype=np.float32) / 255.0
    v = (np.clip((x - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if channels == 1:
        return v
    lut = np.repeat(v[None, :, None], channels, axis=2)
    if channels == 4:
        lut[0, :, 3] = np.arange(256, dtype=np.uint8)     # alpha unchanged
    return np.ascontiguousarray(lut)


def adjust_colors(arr, adj, order='bgr'):
    """Color-correct a uint8 image array (HxW, HxWx3 or HxWx4; alpha kept)
    with the same math as the shader: brightness/contrast, clamp, then
    saturation around luma, clamp. Returns a new array."""
    if adj is None:
        return arr
    b, c, sat = adj
    arr = np.ascontiguousarray(arr)
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    out = cv2.LUT(arr, _bc_lut(b, c, ch)) if (b, c) != (0.0, 1.0) else arr.copy()
    if ch >= 3 and sat != 1.0:
        to_gray = {('bgr', 3): cv2.COLOR_BGR2GRAY, ('bgr', 4): cv2.COLOR_BGRA2GRAY,
                   ('rgb', 3): cv2.COLOR_RGB2GRAY, ('rgb', 4): cv2.COLOR_RGBA2GRAY}[(order, ch)]
        gray = cv2.cvtColor(out, to_gray)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA if ch == 4 else cv2.COLOR_GRAY2BGR)
        mixed = cv2.addWeighted(out, sat, gray, 1.0 - sat, 0.0)
        if ch == 4:
            mixed[..., 3] = out[..., 3]
        out = mixed
    return out


def adjust_pil(pil_img, adj):
    """adjust_colors for an RGBA PIL image (CPU previews)."""
    if adj is None:
        return pil_img
    return Image.fromarray(adjust_colors(np.asarray(pil_img.convert('RGBA')), adj, order='rgb'))


def apply_opacity(pil_img, opacity):
    """Scale a PIL RGBA image's alpha channel (returns a new image if needed)."""
    if opacity >= 1.0:
        return pil_img
    a = pil_img.getchannel('A').point(lambda v: int(v * opacity + 0.5))
    out = pil_img.copy()
    out.putalpha(a)
    return out


def frame_color_info(frame):
    """Return (range, matrix) for a decoded PyAV VideoFrame.

    Unspecified metadata follows common player practice: limited range, and
    BT.709 for HD (height >= 720) / BT.601 for SD.
    """
    fmt = frame.format.name
    rng = 'full' if (fmt.startswith('yuvj') or getattr(frame, 'color_range', 0) == 2) else 'limited'
    matrix = _CS_MATRIX.get(getattr(frame, 'colorspace', 2))
    if matrix is None:
        matrix = 'bt709' if frame.height >= 720 else 'bt601'
    return rng, matrix


# libswscale's matrix numbers (sws_getCoefficients). 6 is SMPTE 170M, the same
# matrix as BT.601; 5 would be "default", which PyAV does not pass on.
_SWS_MATRIX = {'bt709': 1, 'bt601': 6, 'bt2020': 9}


def to_rgb_array(frame, fmt, color=None, size=None):
    """frame.to_ndarray(format=fmt) for an RGB fmt ('rgba', 'bgra', 'bgr24', 'rgb24'),
    converting a YUV frame with its own matrix and range (frame_color_info, or
    `color` = (range, matrix) when the frame was scaled first).

    PyAV's own conversion uses BT.601 limited range for every YUV frame unless
    told otherwise (it passes a colorspace to FFmpeg only when the source and
    destination ones differ), so HD video, which is BT.709, came out with the
    wrong colors in Balanced and Compatibility Mode and in videos with alpha
    (a BT.709 red, 255 0 0, became 232 0 1). Performance Mode's 4:2:0 path
    converts in the shader with the right matrix and was not affected."""
    vf = frame.format
    sized = {} if size is None else {'width': size[0], 'height': size[1], 'interpolation': 'AREA'}
    if vf.is_rgb or vf.name.startswith(('gray', 'ya', 'pal', 'mono')):
        return frame.reformat(format=fmt, **sized).to_ndarray() if sized else frame.to_ndarray(format=fmt)
    rng, matrix = color or frame_color_info(frame)
    src = _SWS_MATRIX.get(matrix, 1)
    return frame.reformat(format=fmt, src_colorspace=src, dst_colorspace=1 if src != 1 else 6,
                          src_color_range='JPEG' if rng == 'full' else 'MPEG',
                          dst_color_range='JPEG', **sized).to_ndarray()


# A video that fades in from black has a black first frame, which made its
# thumbnail, its picture in the position editor and the edited-cue preview
# black (owner's report, 2026-11-15). A frame is "dark" when, reduced to 64x36,
# its mean is below STILL_DARK_MEAN and no pixel reaches STILL_DARK_MAX (a
# title on black is kept). A dark first frame is replaced by the first frame
# that is not dark at 1 s, 10 %, 25 % or 50 % of the video, else the brightest.
STILL_DARK_MEAN = 12.0
STILL_DARK_MAX = 48
STILL_SEEK_S = 1.0
STILL_FRACTIONS = (0.1, 0.25, 0.5)
STILL_MAX_DECODE = 120        # frames decoded after a seek to reach its time


def _still_level(frame):
    """(mean, max) of the frame reduced to 64x36 RGB, 0-255."""
    small = frame.reformat(width=64, height=36, format='rgb24').to_ndarray()
    return float(small.mean()), int(small.max())


def _still_times(container, stream):
    """Times (s) to try after a dark first frame, within the video."""
    dur = None
    if stream.duration and stream.time_base:
        dur = float(stream.duration * stream.time_base)
    elif container.duration:
        dur = container.duration / 1_000_000
    if not dur:
        return [STILL_SEEK_S, 3 * STILL_SEEK_S]
    out = []
    for t in [STILL_SEEK_S] + [f * dur for f in STILL_FRACTIONS]:
        if t < dur - 0.05 and all(abs(t - u) > 0.2 for u in out):
            out.append(t)
    return out


def _frame_at(container, stream, t):
    """A frame at or soon after `t` seconds, or None: the first key frame at or
    after `t` (one frame to decode), else the frame at `t` decoded from the key
    frame before it. Decoding up to `t` took about 1.9 s for the owner's 4K clip
    (2026-11-15), on the UI thread when a cue cell or the editor needs the still."""
    tb = stream.time_base
    start = stream.start_time or 0
    target = start + int(t / tb)
    try:
        container.seek(target, stream=stream, backward=False, any_frame=False)
        frame = next(container.decode(stream), None)
        if frame is not None and (frame.pts is None or frame.pts >= target - int(0.05 / tb)):
            return frame
    except (av.error.FFmpegError, OSError, StopIteration):
        pass                                      # no key frame after `t`: decode up to it
    container.seek(target, stream=stream, backward=True, any_frame=False)
    for i, frame in enumerate(container.decode(stream)):
        if frame.pts is None or frame.pts >= target or i >= STILL_MAX_DECODE:
            return frame
    return None


def _still_image(frame):
    if frame.format.name.startswith(_ALPHA_FORMATS_PREFIXES):
        return Image.fromarray(to_rgb_array(frame, 'rgba'))
    return Image.fromarray(to_rgb_array(frame, 'rgb24'))


def video_still(path):
    """A representative frame of a video as a PIL image (RGBA when it has alpha,
    else RGB) with the right colors (to_rgb_array), or None: the first frame,
    unless it is dark (see STILL_DARK_MEAN). For thumbnails, the position editor
    and the edited-cue preview; OpenCV's reader, used before, converts with
    BT.601 too."""
    t0 = time.perf_counter()
    name = os.path.basename(str(path))
    try:
        with av.open(path) as c:
            stream = c.streams.video[0]
            first = next(c.decode(stream), None)
            if first is None:
                return None
            mean, peak = _still_level(first)
            if mean >= STILL_DARK_MEAN or peak >= STILL_DARK_MAX:
                return _still_image(first)
            best, best_mean = first, mean
            for t in _still_times(c, stream):
                try:
                    frame = _frame_at(c, stream, t)
                except Exception as e:  # noqa: BLE001 - not seekable: keep what we have
                    log.debug(f"Still of {os.path.basename(str(path))}: no frame at {t:.1f} s: {e}")
                    break
                if frame is None:
                    continue
                mean, peak = _still_level(frame)
                if mean >= STILL_DARK_MEAN or peak >= STILL_DARK_MAX:
                    at = float(frame.pts * stream.time_base) if frame.pts is not None else t
                    log.info(f"Still of {name}: the first frame is dark; using the frame at {at:.2f} s "
                             f"({(time.perf_counter() - t0) * 1000:.0f} ms)")
                    return _still_image(frame)
                if mean > best_mean:
                    best, best_mean = frame, mean
            log.info(f"Still of {name}: dark throughout the frames tried; using the brightest "
                     f"({(time.perf_counter() - t0) * 1000:.0f} ms)")
            return _still_image(best)
    except Exception as e:  # noqa: BLE001 - a bad file: the caller shows its placeholder
        log.debug(f"No still from {name}: {e}")
    return None


def yuv_to_rgb_matrix(rng, matrix):
    """Return (3x3 matrix, offset vector) mapping normalized YUV in [0,1] to RGB.

    rgb = M @ (yuv - offset). Used by the shader and the CPU preview path so
    both produce identical colors.
    """
    kr, kb = _LUMA_COEFFS.get(matrix, _LUMA_COEFFS['bt709'])
    kg = 1.0 - kr - kb
    m = np.array([
        [1.0, 0.0, 2.0 * (1.0 - kr)],
        [1.0, -2.0 * kb * (1.0 - kb) / kg, -2.0 * kr * (1.0 - kr) / kg],
        [1.0, 2.0 * (1.0 - kb), 0.0],
    ], dtype=np.float64)
    if rng == 'full':
        scale = np.diag([1.0, 1.0, 1.0])
        offset = np.array([0.0, 128.0 / 255.0, 128.0 / 255.0])
    else:
        scale = np.diag([255.0 / 219.0, 255.0 / 224.0, 255.0 / 224.0])
        offset = np.array([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0])
    return (m @ scale).astype(np.float32), offset.astype(np.float32)


def _plane_array(plane, width, height):
    """Copy a PyAV plane into a contiguous (height, width) uint8 array, dropping row padding."""
    buf = np.frombuffer(plane, dtype=np.uint8)
    rows = buf[: plane.line_size * height].reshape(height, plane.line_size)
    return np.ascontiguousarray(rows[:, :width])


# Decoding at the drawn size (all three modes): a clip larger than this many
# pixels that is drawn at half its size or less is scaled down in its decoder
# thread (FFmpeg's scaler, which runs without Python's lock), so the render
# thread copies and uploads a quarter of the data. Owner's PC, 2026-11-14: a 4K
# clip at Fit with Wave was over the 1 % late-frame bar, its 1080p playback copy
# under it. Videos up to 1920 x 1080 are left alone (little to gain).
DECODE_SCALE_MIN_PIXELS = 1920 * 1080 + 1
# Troubleshooting: VISUALDECK_DECODE_FULL_SIZE=1 decodes every clip at its own size.
DECODE_FULL_SIZE = os.environ.get('VISUALDECK_DECODE_FULL_SIZE', '') not in ('', '0')


def decode_scale_for(scale_x, scale_y, warped=False):
    """Size a video is decoded at, as a fraction of its own (1, 0.5 or 0.25),
    for a layer drawn at (scale_x, scale_y) of it: the largest step that is
    still at least the drawn size. A layer warp can stretch parts of the
    picture, so a warped layer is always decoded at full size."""
    try:
        need = max(abs(float(scale_x)), abs(float(scale_y)))
    except (TypeError, ValueError):
        return 1.0
    if warped or not need or need > 0.5:
        return 1.0
    return 0.25 if need <= 0.25 else 0.5


def scaled_size(w, h, scale):
    """(w, h) at `scale`, rounded to even sizes (4:2:0 chroma), at least 2 x 2."""
    return max(2, int(round(w * scale / 2)) * 2), max(2, int(round(h * scale / 2)) * 2)


def convert_video_frame(frame, scale=1.0):
    """Convert a decoded PyAV VideoFrame to the renderer's frame dict (see above).

    scale < 1 (decode_scale_for): a frame larger than DECODE_SCALE_MIN_PIXELS is
    first scaled down; the dict's 'width'/'height' are then the scaled planes'
    and 'src_width'/'src_height' the video's own size, which the layer's
    geometry uses (a layer's scale is relative to the video's own size)."""
    if scale < 1.0 and frame.width * frame.height >= DECODE_SCALE_MIN_PIXELS:
        fmt = frame.format.name
        color = frame_color_info(frame)                # from the decoder's frame, before scaling
        if fmt in _RGBX_ORDER or fmt in ('yuv420p', 'yuvj420p') or fmt.startswith('yuva'):
            target = fmt                               # scaled only; colors converted after (to_rgb_array)
        elif fmt.startswith(_ALPHA_FORMATS_PREFIXES) or fmt.startswith(_RGB_FORMATS_PREFIXES):
            target = 'rgba'
        else:                                          # nv12, 10-bit, 4:2:2 ...: 8-bit 4:2:0, same range
            target = 'yuvj420p' if color[0] == 'full' else 'yuv420p'
        sw, sh = scaled_size(frame.width, frame.height, scale)
        try:
            small = frame.reformat(width=sw, height=sh, format=target, interpolation='AREA')
        except Exception as e:  # noqa: BLE001 - not scalable: the full frame
            log.debug(f"Scaling a {fmt} frame failed: {e}")
        else:
            out = _convert_video_frame(small, color)
            out['src_width'], out['src_height'] = frame.width, frame.height
            return out
    return _convert_video_frame(frame)


def _convert_video_frame(frame, color=None):
    fmt = frame.format.name
    ts = float(frame.pts * frame.time_base) if (frame.pts is not None and frame.time_base) else 0.0
    w, h = frame.width, frame.height
    if fmt.startswith(_ALPHA_FORMATS_PREFIXES):
        rgba = to_rgb_array(frame, 'rgba', color)              # ProRes 4444 etc. are YUV with alpha
        return {'format': 'rgba', 'width': w, 'height': h, 'rgba': np.ascontiguousarray(rgba), 'timestamp': ts}
    if fmt in _RGBX_ORDER:
        plane = frame.planes[0]
        buf = np.frombuffer(plane, dtype=np.uint8)
        px = buf[: plane.line_size * h].reshape(h, plane.line_size)[:, : w * 4].reshape(h, w, 4)
        if fmt == 'bgr0':                                             # B G R X -> R G B X (a new array)
            rgba = cv2.cvtColor(np.ascontiguousarray(px), cv2.COLOR_BGRA2RGBA)
            if np.shares_memory(rgba, px):                            # never write into the frame
                rgba = rgba.copy()
        else:
            rgba = px.copy()                                          # a copy: the frame's memory is reused
        rgba.view('<u4')[...] |= 0xFF000000                           # X -> 255 (byte 3 is the highest)
        return {'format': 'rgba', 'width': w, 'height': h, 'rgba': rgba, 'timestamp': ts}
    if fmt.startswith(_RGB_FORMATS_PREFIXES):
        rgba = frame.to_ndarray(format='rgba')
        return {'format': 'rgba', 'width': w, 'height': h, 'rgba': np.ascontiguousarray(rgba), 'timestamp': ts}
    rng, matrix = color or frame_color_info(frame)
    if fmt == 'nv12':
        cw, ch = (w + 1) // 2, (h + 1) // 2
        y = _plane_array(frame.planes[0], w, h)
        uv = _plane_array(frame.planes[1], cw * 2, ch)
        u, v = np.ascontiguousarray(uv[:, 0::2]), np.ascontiguousarray(uv[:, 1::2])
    else:
        if fmt not in ('yuv420p', 'yuvj420p'):
            # 10-bit, 4:2:2, 4:4:4 etc.: let swscale produce 8-bit 4:2:0
            # without changing range (the range read above still applies).
            frame = frame.reformat(format='yuvj420p' if rng == 'full' else 'yuv420p')
        cw, ch = frame.planes[1].width, frame.planes[1].height
        y = _plane_array(frame.planes[0], w, h)
        u = _plane_array(frame.planes[1], cw, ch)
        v = _plane_array(frame.planes[2], cw, ch)
    return {'format': 'yuv420p', 'width': w, 'height': h, 'y': y, 'u': u, 'v': v,
            'range': rng, 'matrix': matrix, 'timestamp': ts}


def yuv_frame_to_rgb(frame_dict, out_size=None):
    """CPU conversion of a yuv420p frame dict to an (h, w, 3) uint8 RGB array.

    When out_size=(w, h) is given the planes are downscaled first, so a preview
    costs a few milliseconds instead of converting the full-resolution frame.
    """
    y, u, v = frame_dict['y'], frame_dict['u'], frame_dict['v']
    if out_size is None:
        out_size = (frame_dict['width'], frame_dict['height'])
    ow, oh = max(1, int(out_size[0])), max(1, int(out_size[1]))
    interp_y = cv2.INTER_AREA if ow < y.shape[1] else cv2.INTER_LINEAR
    ys = cv2.resize(y, (ow, oh), interpolation=interp_y) if (ow, oh) != (y.shape[1], y.shape[0]) else y
    us = cv2.resize(u, (ow, oh), interpolation=cv2.INTER_LINEAR)
    vs = cv2.resize(v, (ow, oh), interpolation=cv2.INTER_LINEAR)
    m, off = yuv_to_rgb_matrix(frame_dict.get('range', 'limited'), frame_dict.get('matrix', 'bt709'))
    yuv = np.stack([ys, us, vs], axis=-1).astype(np.float32) * (1.0 / 255.0) - off
    rgb = yuv @ m.T
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


# --- Video Player Classes ---
class BaseVideoPlayer(threading.Thread):
    """Base class for video player threads.

    hold=True pre-rolls the clip: the player decodes and queues its first
    frame, then pauses until release() is called. Cues are preloaded this way
    so that GO starts them from frame 0 with the first frame already decoded.
    """
    def __init__(self, path, frame_queue, loop=True, hold=False):
        super().__init__(daemon=True, name=f"Decoder-{os.path.basename(path)[:40]}")
        self.path = path
        self.frame_queue = frame_queue
        self.loop = loop
        self.is_running = True
        self.lock = threading.Lock()
        self._released = threading.Event()
        if not hold:
            self._released.set()
        # Health (read by the render thread's health snapshot; plain attributes)
        self.state = 'starting'      # starting|playing|retrying|error|no video|ended|stopped
        self.last_error = None
        self.frames_decoded = 0
        self.source_fps = None
        self.hw_engaged = None       # True/False once the first frame is decoded
        self._pacer = None
        self.speed = 1.0             # playback speed (frames paced at clip rate x speed)
        self.loops = 0               # times the clip has started again (audio follows it)
        # When the picture's clock started (the first frame let through: at release
        # for a held player) and when the last loop began (perf_counter): the sound
        # catches up by the time it takes to hear of them (audio_engine).
        self.clock_start = None
        self.loop_at = None

    def set_speed(self, speed):
        try:
            v = float(speed)
        except (TypeError, ValueError):
            v = 1.0
        v = min(8.0, max(0.1, v)) if v == v else 1.0
        if v != self.speed:
            self.speed = v
            if self._pacer is not None and self.source_fps:
                self._pacer.set_target_fps(self.source_fps * v)

    @property
    def held(self):
        return not self._released.is_set()

    def release(self):
        """Start playback of a held (pre-rolled) player."""
        self._released.set()

    def _wait_while_held(self, pacer):
        """Called after the first frame is queued. Returns False if stopped while waiting."""
        if self._released.is_set():
            if self.clock_start is None:
                self.clock_start = time.perf_counter()
            return self.is_running
        while self.is_running and not self._released.wait(0.05):
            pass
        pacer.reset()
        self.clock_start = time.perf_counter()
        return self.is_running

    def _looped(self):
        self.loops += 1
        self.loop_at = time.perf_counter()

    def stop(self):
        with self.lock:
            self.is_running = False
        self._released.set()  # wake a held player so it can exit
            
    def _set_thread_priority(self):
        """Set thread priority on Windows for better performance."""
        if sys.platform == "win32":
            try:
                import win32api, win32process
                thread_handle = win32api.GetCurrentThread()
                # MODIFICATION: Changed to HIGHEST to keep up with the render thread
                win32process.SetThreadPriority(thread_handle, 
                    win32process.THREAD_PRIORITY_HIGHEST)
                log.debug(f"Set video decoder thread priority to HIGHEST for {os.path.basename(self.path)}")
            except:
                pass  # Silent fail if pywin32 not available


# MODIFICATION: Re-instated the smoother, thread-based player for CPU mode from the old codebase.
class CPUVideoPlayerThread(BaseVideoPlayer):
    """A dedicated thread for reading frames from a video file for CPU mode."""
    def __init__(self, path, frame_queue, loop=True, hold=False):
        super().__init__(path, frame_queue, loop, hold)
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_delay = 1.0 / self.fps if self.fps > 0 else 1/30.0
        self.source_fps = float(self.fps)
        self.hw_accel = None
        self.hw_engaged = False

    def run(self):
        log.info(f"CPUVideoPlayerThread starting for {os.path.basename(self.path)}")
        self._set_thread_priority()
        frame_pacer = PreciseFramePacer(target_fps=self.fps * self.speed)
        self._pacer = frame_pacer
        frame_pacer.set_target_fps(self.fps * self.speed)   # a set_speed() racing the line above
        consecutive_failures = 0

        while self.is_running:
            with self.lock:
                if not self.is_running:
                    break
                # cap.read() allocates a new array per call, so the queued frame
                # is never overwritten by the next decode. (Earlier versions
                # copied every frame into one shared buffer and queued that same
                # object repeatedly, so the renderer read a buffer that was being
                # overwritten.)
                ret, frame = self.cap.read()

            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    # Unreadable or missing file: back off instead of spinning.
                    if consecutive_failures == 3:
                        log.warning(f"CPU player cannot read {os.path.basename(self.path)}; retrying with backoff")
                        self.state = 'retrying'
                        self.last_error = 'cannot read file'
                    time.sleep(min(2.0, 0.1 * consecutive_failures))
                with self.lock:
                    if not self.is_running:
                        break
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if consecutive_failures == 1 and self.frames_decoded:
                    self._looped()                 # the end of the clip: it starts again (its sound too)
                continue
            consecutive_failures = 0

            try:
                if hasattr(self.frame_queue, 'put_nowait_drop_old'):
                    self.frame_queue.put_nowait_drop_old(frame)
                else:
                    self.frame_queue.put_nowait(frame)
            except Exception as e:
                log.debug(f"Failed to put frame in queue: {e}")
            self.frames_decoded += 1
            self.state = 'playing'

            if not self._wait_while_held(frame_pacer):
                break
            frame_pacer.wait_for_next_frame()

    def stop(self):
        super().stop()
        with self.lock:
            self.cap.release()

DIRECTIONS = ('forward', 'reverse', 'pingpong')
REVERSE_BUDGET_BYTES = 512 * 2 ** 20     # decoded frames held for reverse playback (two chunks); see KNOWN_ISSUES
REVERSE_MAX_FRAMES = 240                 # per chunk


def layer_trim(layer):
    """(in_s, out_s) of a video layer's in and out points in seconds from the
    clip's start (None: the start / the end), or (None, None)."""
    pos = layer or {}
    try:
        a = float(pos['in_s']) if pos.get('in_s') is not None else None
        b = float(pos['out_s']) if pos.get('out_s') is not None else None
    except (TypeError, ValueError):
        return None, None
    a = a if a is not None and a > 0 and a == a else None
    b = b if b is not None and b > 0 and b == b else None
    if a is not None and b is not None and b - a < 0.04:
        return None, None                       # less than a frame: ignore both
    return a, b


def layer_direction(layer):
    """'forward', 'reverse' or 'pingpong' from a layer's 'direction' (default forward)."""
    d = (layer or {}).get('direction', 'forward')
    return d if d in DIRECTIONS else 'forward'


def _item_bytes(item):
    """Approximate memory of one converted frame (any output format)."""
    if isinstance(item, dict):
        return sum(v.nbytes for v in item.values() if isinstance(v, np.ndarray))
    if isinstance(item, tuple) and item and isinstance(item[0], (bytes, bytearray)):
        return len(item[0])
    return getattr(item, 'nbytes', 0)


class _ReverseChunk:
    """Decoded frames just before `end` (a pts; None = the end of the clip),
    oldest first, at most `keep` of them, for playing backwards. Decoding
    starts at the keyframe before `end` and runs a few frames per step(), so
    the next chunk can be decoded while the current one is shown."""

    def __init__(self, player, container, stream, end, keep):
        self.player, self.stream, self.end = player, stream, end
        self.frames = collections.deque(maxlen=keep)
        self.done = False
        self.decoded = 0
        target = end - 1 if end is not None else player._duration_pts(container, stream)
        container.seek(max(0, int(target)), stream=stream, backward=True, any_frame=False)
        self._it = container.decode(stream)

    def step(self, n):
        for _ in range(n):
            if self.done:
                return
            try:
                frame = next(self._it)
            except StopIteration:
                self.done = True
                return
            pts = frame.pts if frame.pts is not None else 0
            if self.end is not None and pts >= self.end:
                self.done = True
                return
            if self.player.hw_engaged is None:
                self.player._report_decode_mode(self.stream, frame)
            self.decoded += 1
            self.frames.append((pts, self.player._convert_frame(frame)))

    def finish(self):
        while not self.done and self.player.is_running:
            self.step(8)


class GPUVideoPlayerThread(BaseVideoPlayer):
    def __init__(self, path, frame_queue, hw_accel=None, loop=True, output_format='numpy', force_multithread=True,
                 hold=False, direction='forward', trim=(None, None), start_at=None, start_back=False):
        super().__init__(path, frame_queue, loop, hold)
        self.hw_accel = hw_accel
        self.output_format = output_format
        self.force_multithread = force_multithread # Store the setting
        # Fixed for the player's life: a change of direction starts a new player (update_layers).
        self.direction = direction if direction in DIRECTIONS else 'forward'
        self.trim = tuple(trim) if trim else (None, None)   # (in_s, out_s), fixed like the direction
        self._in_pts = self._out_pts = None
        # Start From (rehearsal): the first pass starts at this clip time (seconds from the
        # file's start) instead of the in point; ping-pong with start_back starts on the way back.
        # Loops and later passes are as usual. Used once (not again after a decoder error).
        self.start_at = float(start_at) if start_at else None
        self.start_back = bool(start_back) and self.direction == 'pingpong'
        self._start_used = False
        # Performance Mode: the size frames are handed over at (decode_scale_for), set by the
        # engine from the layer's scale; 'scaled_to' is the size last used when smaller.
        self.decode_scale = 1.0
        self.scaled_to = None


    def run(self):
        """Decode loop. Never raises: a decoder failure must not take the app down."""
        self._set_thread_priority()
        name = os.path.basename(self.path)
        frame_pacer = None
        backoff = 0.5
        while self.is_running:
            try:
                container = self._open_container()
            except Exception as e:  # defensive: _open_container already catches
                log.error(f"Unexpected error opening {name}: {e}")
                container = None
            if container is None or not container.streams.video:
                if container is not None:
                    log.error(f"{name} has no video stream; player idle")
                    container.close()
                    self.state, self.last_error = 'no video', 'file has no video stream'
                else:
                    self.state = 'retrying'
                if not self._sleep_interruptible(backoff):
                    break
                backoff = min(backoff * 2, 10.0)
                continue
            backoff = 0.5
            try:
                stream = container.streams.video[0]
                self._set_threading(stream)
                if frame_pacer is None:
                    frame_pacer = PreciseFramePacer(target_fps=float(stream.average_rate or 30) * self.speed)
                    self._pacer = frame_pacer
                    self.source_fps = round(float(stream.average_rate or 30), 3)
                    frame_pacer.set_target_fps(self.source_fps * self.speed)   # set_speed() race
                reported_hw = False
                in_pts, out_pts = self._trim_pts(stream)
                lo = in_pts                               # the first frame of this pass
                if self.direction != 'forward':
                    self._run_directed(container, stream, frame_pacer)
                else:
                    lo = self._take_start(container, stream) or in_pts
                    if lo:
                        container.seek(lo, stream=stream)     # the key frame before the in (or start) point
                while self.is_running and self.direction == 'forward':
                    got_frame = False
                    for frame in container.decode(stream):
                        if not self.is_running:
                            break
                        if frame.pts is not None:
                            if lo and frame.pts < lo:
                                continue                  # decoded from the key frame, before the in point
                            if out_pts is not None and frame.pts >= out_pts:
                                break                     # the out point: end of this pass
                        got_frame = True
                        if not reported_hw:
                            self._report_decode_mode(stream, frame)
                            reported_hw = True
                        output_item = self._convert_frame(frame)
                        try:
                            if hasattr(self.frame_queue, 'put_nowait_drop_old'):
                                self.frame_queue.put_nowait_drop_old(output_item)
                            else:
                                self.frame_queue.put_nowait(output_item)
                        except Exception:
                            pass
                        self.frames_decoded += 1
                        self.state = 'playing'
                        if not self._wait_while_held(frame_pacer):
                            break
                        frame_pacer.wait_for_next_frame()
                    if not self.is_running or not self.loop or not got_frame:
                        break
                    # Loop by seeking instead of closing and reopening the file.
                    container.seek(in_pts or 0, stream=stream)
                    lo = in_pts
                    self._looped()
            except Exception as e:
                log.error(f"PyAV error during playback of {name}: {e}")
                self.state, self.last_error = 'error', f"{type(e).__name__}: {e}"[:300]
                if not self._sleep_interruptible(0.5):
                    break
            finally:
                try:
                    container.close()
                except Exception:
                    pass
            if not self.loop:
                if self.is_running and self.state == 'playing':
                    self.state = 'ended'
                break

    # --- Reverse and ping-pong ---------------------------------------------------------

    def _duration_pts(self, container, stream):
        """The clip's length in the stream's time base (a seek target past the last keyframe)."""
        tb = float(stream.time_base) if stream.time_base else 1 / 90000
        if stream.duration:
            return int(stream.duration) + int(stream.start_time or 0)
        if container.duration:
            return int(container.duration / 1e6 / tb)
        if stream.frames and stream.average_rate:
            return int(stream.frames / float(stream.average_rate) / tb)
        return int(3600 / tb)

    def _emit(self, item, pacer):
        """Queue one frame and wait for the next frame time. False when stopped."""
        try:
            if hasattr(self.frame_queue, 'put_nowait_drop_old'):
                self.frame_queue.put_nowait_drop_old(item)
            else:
                self.frame_queue.put_nowait(item)
        except Exception:
            pass
        self.frames_decoded += 1
        self.state = 'playing'
        if not self._wait_while_held(pacer):
            return False
        pacer.wait_for_next_frame()
        return self.is_running

    def _reverse_keep(self, stream):
        bpp = {'yuv': 1.5, 'pygame': 4, 'numpy': 3}.get(self.output_format, 4)
        per_frame = max(1, int((stream.width or 1920) * (stream.height or 1080) * bpp))
        return max(2, min(REVERSE_MAX_FRAMES, REVERSE_BUDGET_BYTES // (2 * per_frame)))

    def _trim_pts(self, stream):
        """(in pts, out pts) of self.trim in the stream's time base (None: no point)."""
        a, b = self.trim
        tb = float(stream.time_base) if stream.time_base else None
        if not tb:
            self._in_pts = self._out_pts = None
            return None, None
        st = int(stream.start_time or 0)
        self._in_pts = st + int(round(a / tb)) if a else None
        self._out_pts = st + int(round(b / tb)) if b else None
        return self._in_pts, self._out_pts

    def _take_start(self, container, stream):
        """The Start From point in the stream's time base, once (None: none, or used).
        Kept inside the in and out points."""
        if self.start_at is None or self._start_used:
            return None
        self._start_used = True
        tb = float(stream.time_base) if stream.time_base else None
        if not tb:
            return None
        st = int(stream.start_time or 0)
        pts = st + int(round(self.start_at / tb))
        lo = self._in_pts if self._in_pts is not None else st
        hi = self._out_pts if self._out_pts is not None else self._duration_pts(container, stream)
        frame = max(1, int(round(1.0 / (float(stream.average_rate or 30) * tb))))
        pts = max(lo, min(pts, hi - frame))             # at most the last frame before the out point
        return pts if pts > st else None

    def _first_frame_pts(self, container, stream):
        container.seek(self._in_pts or 0, stream=stream)
        for frame in container.decode(stream):
            pts = frame.pts if frame.pts is not None else 0
            if self._in_pts and pts < self._in_pts:
                continue
            return pts
        return None

    def _forward_pass(self, container, stream, pacer, skip_first, prefetch=None, start=None):
        """Play from the start (or `start`, a pts) to the end once, decoding
        `prefetch` (the first chunk of the way back, from another container) a
        little per frame. Returns the pts of the last frame shown."""
        lo = start or self._in_pts
        container.seek(lo or 0, stream=stream)
        last, first = None, True
        for frame in container.decode(stream):
            if not self.is_running:
                return last
            if frame.pts is not None:
                if lo and frame.pts < lo:
                    continue
                if self._out_pts is not None and frame.pts >= self._out_pts:
                    break
            if first and skip_first:              # shown as the last frame of the reverse pass
                first = False
                continue
            first = False
            if self.hw_engaged is None:
                self._report_decode_mode(stream, frame)
            if not self._emit(self._convert_frame(frame), pacer):
                return last
            last = frame.pts
            if prefetch is not None:
                prefetch.step(2)
        return last

    def _reverse_pass(self, container, stream, pacer, chunk, first_pts, wrap):
        """Play backwards from `chunk` (decoded) down to the first frame. The
        next chunk is decoded a few frames at a time while the current one is
        shown; with `wrap` that includes the clip's last chunk at the start,
        so a looping reverse clip does not stop at the turn. Returns (frames
        shown, the decoded chunk to start the next pass with, or None)."""
        keep = self._reverse_keep(stream)
        shown = 0
        while self.is_running:
            frames = sorted((f for f in chunk.frames if not self._in_pts or f[0] >= self._in_pts),
                            key=lambda f: f[0])
            if not frames:
                return shown, None
            at_start = first_pts is not None and frames[0][0] <= first_pts
            if at_start:
                nxt = _ReverseChunk(self, container, stream, self._out_pts, keep) if wrap else None
            else:
                nxt = _ReverseChunk(self, container, stream, frames[0][0], keep)
            per = max(1, -(-chunk.decoded // len(frames))) + 1     # decode steps per frame shown
            for _pts, item in reversed(frames):
                if not self._emit(item, pacer):
                    return shown, None
                shown += 1
                if nxt is not None:
                    nxt.step(per)
            if nxt is not None:
                nxt.finish()
            if at_start:
                return shown, nxt
            chunk = nxt
        return shown, None

    def _run_directed(self, container, stream, pacer):
        """Reverse or ping-pong (self.direction) until stopped, or one cycle
        when not looping. Ping-pong shows the first and last frames once per
        turn; the first chunk of the way back is decoded from a second
        container during the forward pass, so the turn does not stall.
        Long-GOP files (H.264, HEVC) decode a whole group of pictures for
        every chunk played backwards: intra-frame codecs (MJPEG, ProRes, HAP)
        or short GOPs play backwards far more cheaply."""
        pingpong = self.direction == 'pingpong'
        keep = self._reverse_keep(stream)
        first_pts = self._first_frame_pts(container, stream)
        aux = None
        if pingpong:
            aux = self._open_container()
            if aux is None or not aux.streams.video:
                self.state, self.last_error = 'error', 'cannot open the file twice for ping-pong'
                self._sleep_interruptible(2.0)
                return
            self._set_threading(aux.streams.video[0])
        try:
            skip_first = False
            ahead = None
            start = self._take_start(container, stream)
            if pingpong and start is not None and self.start_back:   # Start From on the way back
                chunk = _ReverseChunk(self, container, stream, start + 1, keep)
                chunk.finish()
                shown, _ = self._reverse_pass(container, stream, pacer, chunk, first_pts, wrap=False)
                if not shown or not self.loop or not self.is_running:
                    if self.is_running:
                        self.state = 'ended'
                    return
                self.loops += 1
                skip_first, start = True, None
            while self.is_running:
                if pingpong:
                    turn = _ReverseChunk(self, aux, aux.streams.video[0], self._out_pts, keep)
                    last = self._forward_pass(container, stream, pacer, skip_first, prefetch=turn, start=start)
                    start = None
                    if last is None or not self.is_running:
                        return
                    turn.finish()
                    turn.frames = collections.deque((f for f in turn.frames if f[0] < last), maxlen=keep)
                    chunk = turn                          # the last frame is not shown twice
                else:
                    chunk = ahead
                    if chunk is None:
                        end = start + 1 if start is not None else self._out_pts   # Start From: just after it
                        start = None
                        chunk = _ReverseChunk(self, container, stream, end, keep)
                        chunk.finish()
                shown, ahead = self._reverse_pass(container, stream, pacer, chunk, first_pts,
                                                  wrap=not pingpong and self.loop)
                if not shown:
                    if self.is_running:                   # nothing could be decoded backwards
                        self.state, self.last_error = 'error', 'cannot play this file backwards'
                        log.error(f"{os.path.basename(self.path)}: no frames decoded playing backwards")
                        self._sleep_interruptible(2.0)
                    return
                if not self.loop:
                    if self.is_running:
                        self.state = 'ended'
                    return
                self.loops += 1
                skip_first = True
        finally:
            if aux is not None:
                try:
                    aux.close()
                except Exception:  # noqa: BLE001
                    pass

    MAX_DECODE_THREADS = 8

    def _set_threading(self, stream):
        """Frame threading (several frames decoded in parallel) when multi-threaded
        decoding is on (Preferences, the default). PyAV's default is slice
        threading only, which for the usual one-slice-per-frame H.264/HEVC
        file decodes on a single core: a 4K clip then cannot reach 25 fps on
        some laptops (owner's report, 2026-11-12: 20 of 25 fps). Capped at
        MAX_DECODE_THREADS: each thread keeps a frame in flight (about 12 MB
        at 4K). Must run before the first decode opens the codec."""
        cc = stream.codec_context
        self.media_info = {'codec': cc.name, 'size': [stream.width, stream.height],
                           'pix_fmt': getattr(cc.format, 'name', None) if cc.format else None}
        if not self.force_multithread:
            self.media_info['threads'] = 'single'
            return
        try:
            cc.thread_count = min(self.MAX_DECODE_THREADS, os.cpu_count() or 2)
            cc.thread_type = 'AUTO'
            self.media_info['threads'] = f"{cc.thread_count} (frame+slice)"
        except Exception as e:  # noqa: BLE001 - keep decoding with the defaults
            log.debug(f"Decoder threading not changed for {os.path.basename(self.path)}: {e}")

    def _sleep_interruptible(self, seconds):
        """Sleep up to `seconds`, returning False early if the player was stopped."""
        deadline = time.monotonic() + seconds
        while self.is_running and time.monotonic() < deadline:
            time.sleep(0.05)
        return self.is_running

    def _report_decode_mode(self, stream, frame):
        self.hw_engaged = bool(getattr(stream.codec_context, 'is_hwaccel', False))
        if self.hw_accel and self.hw_accel not in ('auto', 'none'):
            engaged = self.hw_engaged
            log.info(f"Decode {os.path.basename(self.path)}: requested hwaccel={self.hw_accel}, "
                         f"engaged={engaged}, frame format={frame.format.name}")

    def _open_container(self):
        """Open the file, requesting hardware decoding when configured.

        Hardware decoding uses PyAV's hwaccel API with software fallback.
        (Earlier versions passed 'hwaccel' inside the demuxer options dict,
        which FFmpeg silently ignores, so decoding was always software.)
        """
        options = {'threads': 'auto'} if self.force_multithread else {}
        kwargs = {}
        if self.hw_accel and self.hw_accel not in ('auto', 'none'):
            try:
                from av.codec.hwaccel import HWAccel
                kwargs['hwaccel'] = HWAccel(device_type=self.hw_accel, allow_software_fallback=True)
            except Exception as e:
                log.warning(f"Hardware decoding '{self.hw_accel}' unavailable: {e}; using software decoding")
        try:
            return av.open(self.path, options=options, **kwargs)
        except Exception as e:
            if kwargs:
                log.warning(f"Open with hwaccel={self.hw_accel} failed for {os.path.basename(self.path)}: {e}; retrying in software")
                try:
                    return av.open(self.path, options=options)
                except Exception as e2:
                    e = e2
            log.error(f"PyAV could not open {os.path.basename(self.path)}: {e}")
            self.last_error = f"cannot open: {e}"[:300]
            return None

    def _convert_frame(self, frame):
        """Convert frame to the appropriate output format."""
        if self.output_format == 'pygame':
            # Balanced Mode: (BGRA bytes, (w, h)), or (bytes, (w, h), (video's own w, h)) when
            # decoded smaller than the video (decode_scale_for), scaled and converted in one pass.
            if self.decode_scale < 1.0 and frame.width * frame.height >= DECODE_SCALE_MIN_PIXELS:
                sw, sh = scaled_size(frame.width, frame.height, self.decode_scale)
                try:
                    arr = to_rgb_array(frame, 'bgra', size=(sw, sh))
                except Exception as e:  # noqa: BLE001 - not scalable: the full frame
                    log.debug(f"Scaling a {frame.format.name} frame failed: {e}")
                else:
                    self.scaled_to = (sw, sh)
                    return (arr.tobytes(), (sw, sh), (frame.width, frame.height))
            self.scaled_to = None
            arr = to_rgb_array(frame, 'bgra')
            h, w = arr.shape[:2]
            return (arr.tobytes(), (w, h))
        elif self.output_format == 'yuv':
            out = convert_video_frame(frame, self.decode_scale)
            self.scaled_to = (out['width'], out['height']) if 'src_width' in out else None
            return out
        else:
            # Compatibility Mode keeps the clip's transparency (BGRA), else BGR. Decoded
            # smaller than the video, the array carries the video's own size (SizedFrame).
            fmt = 'bgra' if frame.format.name.startswith(_ALPHA_FORMATS_PREFIXES) else 'bgr24'
            if self.decode_scale < 1.0 and frame.width * frame.height >= DECODE_SCALE_MIN_PIXELS:
                sw, sh = scaled_size(frame.width, frame.height, self.decode_scale)
                try:
                    arr = to_rgb_array(frame, fmt, size=(sw, sh))
                except Exception as e:  # noqa: BLE001 - not scalable: the full frame
                    log.debug(f"Scaling a {frame.format.name} frame failed: {e}")
                else:
                    self.scaled_to = (sw, sh)
                    return sized_frame(arr, (frame.width, frame.height))
            self.scaled_to = None
            return to_rgb_array(frame, fmt)


class SizedFrame(np.ndarray):
    """A Compatibility Mode frame decoded smaller than its video (decode_scale_for): an
    ordinary array that also carries the video's own (w, h) in own_size, so a layer's
    scale, which is relative to that size, draws it where the full-size frame would be.
    Arrays made from it (views, OpenCV results) have own_size None or are plain arrays."""
    own_size = None


def sized_frame(arr, own_size):
    out = arr.view(SizedFrame)
    out.own_size = tuple(own_size)
    return out


_alpha_cache = {}


def video_has_alpha(path):
    """True if the file's first video stream has an alpha channel (HAP Alpha,
    ProRes 4444, ...), from its pixel format; cached by path and change time.
    Compatibility Mode's OpenCV reader drops alpha, so such clips are read with
    PyAV there (owner's PC, 2026-11-14: HAP Alpha drawn opaque in that mode)."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return False
    hit = _alpha_cache.get(key)
    if hit is None:
        hit = False
        try:
            with av.open(path) as c:
                if c.streams.video:
                    hit = (c.streams.video[0].codec_context.pix_fmt or '').startswith(_ALPHA_FORMATS_PREFIXES)
        except Exception:  # noqa: BLE001 - not readable: the player reports it
            pass
        if len(_alpha_cache) > 512:
            _alpha_cache.clear()
        _alpha_cache[key] = hit
    return hit


# Capture formats whose connection is not closed after the device fails while
# streaming (see LiveSourcePlayer.run); the connections are kept here for good.
DEVICE_LOSS_KEEP_OPEN = ("dshow",)
_abandoned_devices = []


class LiveSourcePlayer(GPUVideoPlayerThread):
    """Player for a live source (live_source.py): connects, decodes as frames
    arrive (no pacing: the source sets the rate), keeps only the newest frame,
    and reconnects with a growing pause (0.5 s up to 5 s) when the signal
    stops or the source cannot be opened. Never raises."""

    def __init__(self, path, frame_queue, output_format='yuv'):
        super().__init__(path, frame_queue, hw_accel=None, loop=True, output_format=output_format, hold=False)
        self.name = f"Live-{live_source.label(path)[:40]}"
        self.state = 'connecting'
        self.reconnects = 0
        self.last_frame_at = None     # monotonic time of the newest frame
        self.size = None              # (width, height) of the newest frame
        self._audio_buf = None        # audio_engine.LiveAudioBuffer once someone wants the sound
        self.has_audio = None         # True/False once connected

    def audio_buffer(self):
        """The source's sound for the mixer (decoded from now on)."""
        if self._audio_buf is None:
            import audio_engine
            self._audio_buf = audio_engine.LiveAudioBuffer()
        return self._audio_buf

    def set_speed(self, speed):
        pass                          # a live source plays at its own rate

    def release(self):
        pass                          # never held: a preloaded cue shares the running source

    @property
    def held(self):
        return False

    def _decode_live(self, container, stream, astream):
        """Video frames, and the sound's frames while someone wants them."""
        streams = [stream] + ([astream] if astream is not None else [])
        for packet in container.demux(*streams):
            if packet.stream is astream and self._audio_buf is None:
                continue                          # nobody listens: do not decode the sound
            for frame in packet.decode():
                yield frame

    def run(self):
        url, fmt, opts = live_source.parse(self.path)
        name = live_source.label(self.path)
        backoff = 0.5
        while self.is_running:
            self.state = 'connecting' if self.frames_decoded == 0 else 'retrying'
            try:
                container = av.open(url, format=fmt, options=opts,
                                    timeout=(live_source.OPEN_TIMEOUT, live_source.READ_TIMEOUT))
            except Exception as e:  # noqa: BLE001
                self.state, self.last_error = 'retrying', f"cannot open: {e}"[:300]
                log.warning(f"Live source {name}: {self.last_error}; retrying in {backoff:.1f} s")
                if not self._sleep_interruptible(backoff):
                    break
                backoff = min(backoff * 2, 5.0)
                continue
            lost = False                      # the source failed or stopped while we were running
            try:
                if not container.streams.video:
                    self.state, self.last_error = 'no video', 'no video in this source'
                    log.error(f"Live source {name}: no video stream")
                else:
                    stream = container.streams.video[0]
                    astream = container.streams.audio[0] if container.streams.audio else None
                    self.has_audio = astream is not None
                    if stream.average_rate:
                        self.source_fps = round(float(stream.average_rate), 3)
                    log.info(f"Live source {name}: connected ({stream.codec_context.name}"
                             + (f", sound {astream.codec_context.name})" if astream is not None else ")"))
                    first = True
                    conv = None
                    for frame in self._decode_live(container, stream, astream):
                        if not self.is_running:
                            break
                        if isinstance(frame, av.AudioFrame):
                            buf = self._audio_buf
                            if buf is not None:
                                if conv is None:      # no batches: they would delay live sound
                                    import audio_engine
                                    conv = audio_engine.StereoConverter(48000, batch_s=0)
                                for arr in conv.convert(frame):
                                    buf.push(arr)
                            continue
                        if first:
                            self.last_error, first = None, False
                        item = self._convert_frame(frame)
                        try:
                            self.frame_queue.put_nowait_drop_old(item)
                        except Exception:  # noqa: BLE001
                            pass
                        self.frames_decoded += 1
                        self.size = (frame.width, frame.height)
                        self.last_frame_at = time.monotonic()
                        self.state = 'playing'
                        backoff = 0.5
                    if self.is_running:
                        self.last_error = 'the source stopped sending'
                        lost = True
            except Exception as e:  # noqa: BLE001 - a lost signal must never stop the player
                lost = True
                if self.is_running:
                    if type(e).__name__ == 'ExitError':          # FFmpeg's read timeout fired
                        self.last_error = f"no data for {live_source.READ_TIMEOUT:g} s"
                    else:
                        self.last_error = f"{type(e).__name__}: {e}"[:300]
            finally:
                if lost and fmt in DEVICE_LOSS_KEEP_OPEN:
                    # A capture device that failed while streaming (unplugged): closing
                    # FFmpeg's DirectShow graph for a device that is gone ended VisualDeck
                    # with an access violation on the owner's PC (2026-11-15). Its
                    # connection is kept, not closed: a small leak per unplug, not a crash.
                    _abandoned_devices.append(container)
                    log.warning(f"Live source {name}: the device stopped while sending; its old "
                                f"connection is left open (closing it can crash the capture driver)")
                else:
                    try:
                        container.close()
                    except Exception:  # noqa: BLE001
                        pass
            if not self.is_running:
                break
            self.reconnects += 1
            self.state = 'retrying'
            log.warning(f"Live source {name}: signal lost ({self.last_error}); reconnecting")
            if not self._sleep_interruptible(backoff):
                break
            backoff = min(backoff * 2, 5.0)
        if self.state != 'no video':
            self.state = 'stopped'


WARP_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 pos;
void main() { gl_Position = vec4(pos, 0.0, 1.0); }
"""

# Corner pin: each output pixel looks up where it comes from in the unwarped
# picture through the inverse homography (output_warp.inverse); outside the
# picture it is black.
WARP_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
uniform sampler2D picture;
uniform mat3 hinv;       // output (0..1, y down) -> picture (0..1, y down)
uniform vec4 viewport;   // x, y, width, height of this output in window pixels
uniform vec4 edges;      // edge blend widths: left, right, top, bottom (0 = none), see edge_blend.py
uniform vec2 blendShape; // curve, gamma
uniform vec4 region;     // the part of the picture this output shows: x, y, width, height (0..1, y down)
float ramp(float t) {
    t = clamp(t, 0.0, 1.0);
    float light = t < 0.5 ? 0.5 * pow(2.0 * t, blendShape.x) : 1.0 - 0.5 * pow(2.0 * (1.0 - t), blendShape.x);
    return pow(light, 1.0 / blendShape.y);
}
void main() {
    vec2 p = (gl_FragCoord.xy - viewport.xy) / viewport.zw;
    p.y = 1.0 - p.y;
    vec3 s = hinv * vec3(p, 1.0);
    vec2 uv = s.xy / s.z;
    if (s.z <= 0.0 || uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float k = 1.0;
    if (edges.x > 0.0 && p.x < edges.x) k *= ramp(p.x / edges.x);
    if (edges.y > 0.0 && 1.0 - p.x < edges.y) k *= ramp((1.0 - p.x) / edges.y);
    if (edges.z > 0.0 && p.y < edges.z) k *= ramp(p.y / edges.z);
    if (edges.w > 0.0 && 1.0 - p.y < edges.w) k *= ramp((1.0 - p.y) / edges.w);
    vec2 q = region.xy + uv * region.zw;
    FragColor = vec4(texture(picture, vec2(q.x, 1.0 - q.y)).rgb * k, 1.0);
}
"""

# Mesh warp (mesh_warp.py): the picture drawn on a tessellated grid. Each vertex
# carries its output position through the corner pin in homogeneous form
# (X, Y, W; 0..1 with y down after dividing by W), so the texture is
# interpolated with perspective like the corner-pin pass; the edge blend is in
# output coordinates as there.
MESH_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 posw;
layout (location = 1) in vec2 uvIn;
out vec2 uv;
void main() {
    uv = uvIn;
    gl_Position = vec4(2.0 * posw.x - posw.z, posw.z - 2.0 * posw.y, 0.0, posw.z);
}
"""

MESH_FRAGMENT_SHADER = """
#version 330 core
in vec2 uv;
out vec4 FragColor;
uniform sampler2D picture;
uniform vec4 viewport;
uniform vec4 edges;
uniform vec2 blendShape;
uniform vec4 region;
float ramp(float t) {
    t = clamp(t, 0.0, 1.0);
    float light = t < 0.5 ? 0.5 * pow(2.0 * t, blendShape.x) : 1.0 - 0.5 * pow(2.0 * (1.0 - t), blendShape.x);
    return pow(light, 1.0 / blendShape.y);
}
void main() {
    vec2 p = (gl_FragCoord.xy - viewport.xy) / viewport.zw;
    p.y = 1.0 - p.y;
    float k = 1.0;
    if (edges.x > 0.0 && p.x < edges.x) k *= ramp(p.x / edges.x);
    if (edges.y > 0.0 && 1.0 - p.x < edges.y) k *= ramp((1.0 - p.x) / edges.y);
    if (edges.z > 0.0 && p.y < edges.z) k *= ramp(p.y / edges.z);
    if (edges.w > 0.0 && 1.0 - p.y < edges.w) k *= ramp((1.0 - p.y) / edges.w);
    vec2 q = region.xy + uv * region.zw;
    FragColor = vec4(texture(picture, vec2(q.x, 1.0 - q.y)).rgb * k, 1.0);
}
"""


# Crossfade: the outgoing and the incoming cue are each drawn into their own
# buffer (on black), then mixed with their transition opacities, so a fade
# between two full-screen cues keeps its brightness (drawing one over the other
# with rising opacity dimmed the midpoint to about 75 %).
# A display's picture (premultiplied by its coverage) in plain colors, for its effects.
UNPREMULTIPLY_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 pos;
out vec2 uv;
void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }
"""
UNPREMULTIPLY_FRAGMENT_SHADER = """
#version 330 core
in vec2 uv;
out vec4 FragColor;
uniform sampler2D picture;
void main() {
    vec4 t = texture(picture, uv);
    FragColor = t.a > 0.0 ? vec4(t.rgb / t.a, t.a) : vec4(0.0);
}
"""

MIX_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
uniform sampler2D outgoing;
uniform sampler2D incoming;
uniform vec2 weights;    // transition_alphas()
uniform vec4 viewport;
void main() {
    vec2 uv = (gl_FragCoord.xy - viewport.xy) / viewport.zw;
    vec4 a = texture(outgoing, uv), b = texture(incoming, uv);
    // Alpha: the coverage (only a display with an effect clears its buffers to alpha 0).
    FragColor = vec4(a.rgb * weights.x + b.rgb * weights.y, clamp(a.a * weights.x + b.a * weights.y, 0.0, 1.0));
}
"""


class BaseRenderEngine:
    def __init__(self, media_manager, mode='cpu', hw_accel=None):
        self.mm = media_manager
        self.mode = mode
        self.hw_accel = hw_accel
        try:
            self.monitors = get_monitors()
        except Exception as e:  # e.g. no display server / enumerator available
            log.warning(f"Could not enumerate monitors ({e}); output disabled until displays are available")
            self.monitors = []
        self.window = None
        self.gl_info = None  # {vendor, renderer, version} once a GL context exists
        self.output_warps = {}  # monitor index -> corner pin (output_warp.clean()), only non-identity
        self.output_blends = {}  # monitor index -> edge blend (edge_blend.clean()), only when a side blends
        self.output_meshes = {}  # monitor index -> mesh warp (mesh_warp.clean()), only non-identity
        self.output_effects = {}  # monitor index -> shader effect (isf.clean()); drawn by the OpenGL engine
        # monitor index -> ((x0, y0, x1, y1), (canvas w, canvas h)): the part of a
        # shared canvas the display shows (span.py); layers are laid out on the canvas
        self.output_regions = {}
        # Set by a background job whose result changes the picture (the render
        # loop then draws even a still cue again); see PygameRenderEngine._mesh_job.
        self.redraw_pending = False
        self.active_players = {}
        self.video_frame_cache = {}
        self.static_cue_cache = {}  # {cache_key: pre_rendered_data}; render thread only
        self._static_cache_stale = False  # set by any thread, acted on by the render thread
        self.cue_is_static = {}  # {col_index: bool}
        self.static_cue_cache_hits = 0
        self.static_cue_cache_misses = 0

    def is_cue_static(self, layers):
        """Check if a cue contains only static media (no videos)."""
        if not layers:
            return True  # Empty cue is considered static
        
        for layer in layers.values():
            path = layer.get('path')
            if is_streamed(path):
                return False  # Found a video or live source, not static
        return True  # All media are static images

    def get_cue_cache_key(self, layers, target_monitor=None):
        """Generate a unique cache key for a cue configuration."""
        # Create a stable key based on the cue's content
        layer_info = []
        for layer_id, layer in sorted(layers.items()):
            if layer.get('path'):
                # Include all relevant layer properties in the key
                pos = layer.get('position', {})
                layer_info.append((
                    layer_id,
                    layer['path'],
                    pos.get('x', 0.5),
                    pos.get('y', 0.5),
                    pos.get('scale_x', 1.0),
                    pos.get('scale_y', 1.0),
                    pos.get('rotation', 0),
                    layer.get('order', 0),
                    tuple(layer.get('screens', [])),
                    layer_opacity(layer),
                    layer_blend(layer),
                    layer_color(layer),
                    layer_mask.layer_mask(layer),
                ))
        
        # Include monitor configuration in the key
        monitor_info = None
        if target_monitor is not None and self.monitors:
            if 0 <= target_monitor < len(self.monitors):
                mon = self.monitors[target_monitor]
                monitor_info = (mon.width, mon.height)
        
        return (tuple(layer_info), monitor_info)

    def invalidate_static_cache_for_column(self, col_index):
        """Forget the pre-rendered still cues (the keys do not say which column
        they came from, so all of them). Safe from any thread: it only sets a
        flag, and the render thread drops and frees the entries at its next
        frame (_drop_stale_static_cache). Until 2026-11 this deleted the
        entries from the UI thread, which raced with the render thread and, in
        Performance Mode, never freed their framebuffers (a monitor-sized
        texture each)."""
        self._static_cache_stale = True
        self.cue_is_static.pop(col_index, None)
        log.debug(f"Invalidated static cache for column {col_index}")

    def _drop_stale_static_cache(self):
        """Render thread: drop (and free) the cache if it was invalidated."""
        if not self._static_cache_stale:
            return
        self._static_cache_stale = False
        stale, self.static_cue_cache = self.static_cue_cache, {}
        for data in stale.values():
            try:
                self._release_static_entry(data)
            except Exception as e:  # noqa: BLE001 - never fail a frame over cleanup
                log.debug(f"Could not free a pre-rendered cue: {e}")

    def _release_static_entry(self, data):
        """Free one pre-rendered cue (GL objects in the OpenGL engine; surfaces
        are freed by Python)."""

    def clear_static_cache(self):
        """Clear all static cue caches (any thread; see invalidate_static_cache_for_column)."""
        self._static_cache_stale = True
        self.cue_is_static.clear()
        self.static_cue_cache_hits = 0
        self.static_cue_cache_misses = 0
        log.debug("Cleared all static cue caches")

    def _init_pygame_and_gl(self):
        """Base initialization - can be overridden by subclasses"""
        sdl_audio_safe.init_pygame()       # without pygame.mixer (sound has its own output)
        pygame.display.init()

    @staticmethod
    def source_key(layer):
        """Identifies the decoder, frame cache and texture used by a layer.

        The UI gives every video layer an 'instance' (one per cue activation,
        cell and file), so the same file can play independently in several
        layers or in the outgoing and incoming cue of a fade. Instance keys end
        with the file path, so extension checks on a key still work. Layers
        without an instance (images, older callers) are keyed by path.
        """
        return layer.get('instance') or layer.get('path')

    def set_output_warps(self, warps):
        """{monitor index: corners}; identity or invalid corners mean no warp."""
        cleaned = {}
        for idx, corners in (warps or {}).items():
            c = output_warp.clean(corners)
            try:
                if c is not None:
                    cleaned[int(idx)] = c
            except (TypeError, ValueError):
                continue
        self.output_warps = cleaned
        self._on_warps_changed()

    def _on_warps_changed(self):
        pass

    def set_output_regions(self, regions):
        """{monitor index: {'rect': [x0, y0, x1, y1], 'canvas': [w, h]}} (span.layout)."""
        cleaned = {}
        for idx, raw in (regions or {}).items():
            try:
                x0, y0, x1, y1 = (float(v) for v in raw['rect'])
                cw, ch = (int(v) for v in raw['canvas'])
                i = int(idx)
            except (TypeError, ValueError, KeyError):
                continue
            if 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 and 0 < cw <= 32768 and 0 < ch <= 32768:
                cleaned[i] = ((x0, y0, x1, y1), (cw, ch))
        self.output_regions = cleaned
        self._on_warps_changed()

    def composition_size(self, idx):
        """Pixel size layers are laid out in for display idx: its canvas in a
        span, else the display itself."""
        r = self.output_regions.get(idx)
        if r is not None:
            return r[1]
        m = self.monitors[idx]
        return m.width, m.height

    @staticmethod
    def _region_uniform(region):
        if region is None:
            return 0.0, 0.0, 1.0, 1.0
        x0, y0, x1, y1 = region[0]
        return x0, y0, x1 - x0, y1 - y0

    def set_output_meshes(self, meshes):
        """{monitor index: {'cols', 'rows', 'points'}}; identity, folded or
        malformed meshes mean no mesh."""
        cleaned = {}
        for idx, raw in (meshes or {}).items():
            m = mesh_warp.clean(raw)
            try:
                if m is not None:
                    cleaned[int(idx)] = m
            except (TypeError, ValueError):
                continue
        self.output_meshes = cleaned
        self._on_warps_changed()

    def set_output_effects(self, effects):
        """{monitor index: [{'name', 'inputs'}, ...] (or one)}: shader effects on
        the display's whole picture (isf.py), after its layers and before its
        corner pin."""
        cleaned = {}
        for idx, raw in (effects or {}).items():
            fx = isf.stack(raw)
            try:
                if fx:
                    cleaned[int(idx)] = fx
            except (TypeError, ValueError):
                continue
        self.output_effects = cleaned
        self._on_warps_changed()

    @property
    def output_effects_moving(self):
        """True if a display's effect changes by itself (the output is redrawn every frame)."""
        return any(isf.animated(fx['name']) for st in self.output_effects.values() for fx in st)

    def set_output_blends(self, blends):
        """{monitor index: {'left', 'right', 'top', 'bottom', 'curve', 'gamma'}}."""
        cleaned = {}
        for idx, raw in (blends or {}).items():
            b = edge_blend.clean(raw)
            try:
                if b is not None:
                    cleaned[int(idx)] = b
            except (TypeError, ValueError):
                continue
        self.output_blends = cleaned
        self._on_warps_changed()

    def update_layers(self, layers, force_restart_paths=None):
        """Start/stop decoder threads so that exactly the videos in `layers`
        (plus keys in `self.preroll_keep`) have a player.

        A layer with hold=True gets a pre-rolled player that waits on its first
        frame; the same key appearing without hold releases it. Restarted keys
        keep their texture, so a layer shows its last frame until the new
        decoder delivers instead of going black. Players are stopped without
        joining, so a slow decoder cannot stall the render thread.
        `force_restart_paths` may contain keys or file paths.
        """
        force = set(force_restart_paths or ())
        keep = getattr(self, 'preroll_keep', None) or set()

        wanted = {}
        scales = {}                                # key -> size to decode at (the largest any layer needs)
        for layer in layers.values():
            path = layer.get('path')
            if not is_streamed(path):
                continue
            key = self.source_key(layer)
            if key not in wanted or not layer.get('hold'):  # a live use wins over a preload
                wanted[key] = layer
            pos = layer.get('position') or {}
            scales[key] = max(scales.get(key, 0.0), decode_scale_for(
                pos.get('scale_x', 1.0), pos.get('scale_y', 1.0), layer_warp.layer_warp(layer) is not None))

        if force:
            for key, info in list(self.active_players.items()):
                if key in force or info.get('path') in force:
                    log.info(f"Restarting player for: {os.path.basename(info.get('path', key))}")
                    self.active_players.pop(key)['player'].stop()
                    self.video_frame_cache.pop(key, None)

        for key in list(self.active_players):
            if key not in wanted and key not in keep:
                self.active_players.pop(key)['player'].stop()
                self.video_frame_cache.pop(key, None)
                if hasattr(self, 'release_texture'):
                    self.release_texture(key)

        for key, layer in wanted.items():
            hold = bool(layer.get('hold'))
            loop = layer.get('loop', True)
            info = self.active_players.get(key)
            direction = layer_direction(layer)
            trim = layer_trim(layer)
            if info is not None and info.get('type') != 'live' and (
                    getattr(info['player'], 'direction', 'forward') != direction
                    or tuple(getattr(info['player'], 'trim', (None, None))) != trim):
                info['player'].stop()                 # another direction or trim: a new player (from its start);
                self.active_players.pop(key, None)    # its last frame stays up until the new one has one
                info = None
            if info is not None:
                player = info['player']
                player.loop = loop
                player.set_speed(layer.get('speed', 1.0))
                self._set_decode_scale(info, scales.get(key, 1.0))
                if not hold and player.held:
                    player.release()
                continue
            path = layer['path']
            if live_source.is_live(path):
                # One player per source, shared by every cue that uses it (no
                # 'instance' key), never paused or restarted per cue.
                q = FastRingBuffer(FRAME_BUFFER_SIZE)
                player = LiveSourcePlayer(path, q, output_format={'opengl': 'yuv', 'cpu': 'numpy'}.get(
                    self.mode, 'pygame'))
                self.active_players[key] = {'player': player, 'type': 'live', 'queue': q, 'path': path}
                player.start()
                continue
            if not os.path.exists(path):
                continue
            start = {'start_at': layer.get('start_at'), 'start_back': bool(layer.get('start_back'))}
            if self.mode == 'cpu':
                # PyAV, like the other modes. OpenCV's reader (CPUVideoPlayerThread), used here
                # before, drops alpha and converts every video with the BT.601 matrix, so HD
                # video (BT.709) had the wrong colors (to_rgb_array).
                q = SimpleFrameQueue(maxsize=CPU_FRAME_BUFFER_SIZE)
                player = GPUVideoPlayerThread(path, q, None, loop, output_format='numpy',
                                              force_multithread=getattr(self, 'force_multithread', True),
                                              hold=hold, direction=direction, trim=trim, **start)
                kind = 'cpu_thread'
            else:
                q = FastRingBuffer(FRAME_BUFFER_SIZE)
                player = GPUVideoPlayerThread(
                    path, q, self.hw_accel, loop,
                    output_format='yuv' if self.mode == 'opengl' else 'pygame',
                    force_multithread=getattr(self, 'force_multithread', True),
                    hold=hold, direction=direction, trim=trim, **start,
                )
                kind = 'gpu_queue'
            player.set_speed(layer.get('speed', 1.0))
            self.active_players[key] = {'player': player, 'type': kind, 'queue': q, 'path': path}
            self._set_decode_scale(self.active_players[key], scales.get(key, 1.0))
            player.start()

    # Performance Mode decodes a large clip drawn at half size or less at that size
    # (decode_scale_for); the other engines size frames from the decoded picture.
    decodes_at_drawn_size = False

    def _set_decode_scale(self, info, scale):
        if info.get('type') == 'live' or not self.decodes_at_drawn_size or DECODE_FULL_SIZE:
            scale = 1.0
        player = info['player']
        if getattr(player, 'decode_scale', 1.0) != scale:
            if hasattr(player, 'decode_scale'):
                log.info(f"Decoding {os.path.basename(info.get('path') or '')} at "
                         f"{'its own size' if scale >= 1 else f'{scale:g} of its size (drawn at that size or smaller)'}")
            player.decode_scale = scale

    def player_stats(self, live_keys=()):
        """Health status of every decoder (see health.player_status)."""
        import health
        out = []
        for key, info in list(self.active_players.items()):
            try:
                st = health.player_status(key, info)
            except Exception as e:  # noqa: BLE001 - diagnostics must not break rendering
                st = {'key': key, 'file': os.path.basename(str(info.get('path', key))), 'state': f'unknown ({e})'}
            st['live'] = key in live_keys
            out.append(st)
        out.sort(key=lambda d: (not d.get('live'), d.get('instance', ''), d.get('file', '')))
        return out

    def stop_players(self, keys):
        """Stop and forget the players for `keys` (missing keys are ignored)."""
        for key in keys:
            info = self.active_players.pop(key, None)
            if info is None:
                continue
            info['player'].stop()
            self.video_frame_cache.pop(key, None)
            if hasattr(self, 'release_texture'):
                self.release_texture(key)

    def has_frame(self, path):
        """True once the player for `path` has produced a frame we can show."""
        if path in self.video_frame_cache:
            return True
        p_info = self.active_players.get(path)
        if p_info is None:
            return False
        try:
            return p_info['queue'].qsize() > 0
        except Exception:
            return False

    def _get_latest_video_frame(self, path):
        """Newest decoded frame for `path` (consumes the player's queue).

        Never blocks: if the decoder has not produced a frame yet, the last
        cached frame (or None) is returned and the layer simply appears a
        frame or two later. Earlier versions waited up to 100 ms here, on the
        render thread, for every newly started video.
        """
        player_info = self.active_players.get(path)
        if player_info is None:
            return None
        q = player_info['queue']
        frame = None
        try:
            if hasattr(q, 'get_latest'):
                frame = q.get_latest()
            elif not q.empty():
                frame = q.get_nowait()
        except Exception:
            frame = None
        if frame is not None:
            self.video_frame_cache[path] = frame
        return self.video_frame_cache.get(path)

    def _peek_video_frame(self, path):
        """Frame for previews: reuse what the output last showed so previews do
        not steal frames from the output; fetch only if nothing is cached yet."""
        cached = self.video_frame_cache.get(path)
        if cached is not None:
            return cached
        return self._get_latest_video_frame(path)

    def render_layers_to_image(self, layers, dimensions, reference_monitor=None):
        """CPU composite of `layers` into an RGBA PIL image of size `dimensions`.

        Used for the preview monitor when no GPU readback is available. Video
        frames are converted at their on-screen preview size (not full
        resolution) and are peeked, not consumed, so the output never loses
        frames to the preview.
        """
        w, h = dimensions
        if not layers:
            return None
        ref_w, ref_h = 1920, 1080
        if self.monitors:
            idx = reference_monitor if (reference_monitor is not None and 0 <= reference_monitor < len(self.monitors)) else 0
            ref_w, ref_h = self.monitors[idx].width, self.monitors[idx].height
        preview_scale = min(w / ref_w, h / ref_h, 1.0)

        target_image = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        for m in sorted(layers.values(), key=lambda l: l.get('order', 0)):
            path = m.get('path')
            if not path:
                continue
            ext = os.path.splitext(path)[1].lower()
            pos = m.get('position', {}) or {}
            scale_x = pos.get('scale_x', 1.0) * preview_scale
            scale_y = pos.get('scale_y', 1.0) * preview_scale
            try:
                if is_streamed(path):
                    frame = self._peek_video_frame(self.source_key(m))
                    pil_img = self._video_frame_to_pil(frame, scale_x, scale_y)
                else:
                    pil_img = self._image_to_pil(path, scale_x, scale_y)
            except Exception as e:
                log.warning(f"Could not convert layer for CPU preview ({os.path.basename(path)}): {e}")
                continue
            if pil_img is None:
                continue
            pil_img = layer_mask.apply_pil(adjust_pil(pil_img, layer_color(m)), layer_mask.layer_mask(m))
            warp = layer_warp.layer_warp(m)
            if warp is not None:                  # bent around the same center
                pil_img = layer_warp.warp_pil(pil_img, warp)
            rotation = pos.get('rotation', 0)
            if rotation:
                pil_img = pil_img.rotate(rotation, expand=True, resample=Image.BICUBIC)
            pil_img = apply_opacity(pil_img, layer_opacity(m))
            paste_x = int(pos.get('x', 0.5) * w - pil_img.width / 2)
            paste_y = int(pos.get('y', 0.5) * h - pil_img.height / 2)
            blend_onto(target_image, pil_img, (paste_x, paste_y), layer_blend(m))
        return target_image

    @staticmethod
    def _video_frame_to_pil(frame, scale_x, scale_y):
        if frame is None:
            return None
        if isinstance(frame, dict) and frame.get('format') == 'yuv420p':
            out = (int(frame.get('src_width', frame['width']) * scale_x),
                   int(frame.get('src_height', frame['height']) * scale_y))
            if out[0] <= 0 or out[1] <= 0:
                return None
            return Image.fromarray(yuv_frame_to_rgb(frame, out)).convert('RGBA')
        # Scale first, then convert colors: a preview is small, and converting
        # a whole 4K frame first cost about 65 ms per preview on the test VM
        # (5 ms this way), on the render thread in Balanced/Compatibility Mode.
        if isinstance(frame, dict) and frame.get('format') == 'rgba':
            arr, order = frame['rgba'], 'rgba'
            if 'src_width' in frame:                  # decoded smaller: scale from the video's own size
                out = (int(frame['src_width'] * scale_x), int(frame['src_height'] * scale_y))
                if out[0] <= 0 or out[1] <= 0:
                    return None
                return Image.fromarray(np.ascontiguousarray(
                    cv2.resize(arr, out, interpolation=cv2.INTER_AREA if out[0] < arr.shape[1] else cv2.INTER_LINEAR)))
        elif isinstance(frame, tuple):  # Pygame-mode (bytes, (w, h)[, video's own (w, h)]) BGRA
            data, (fw, fh) = frame[0], frame[1]
            arr, order = np.frombuffer(data, np.uint8).reshape(fh, fw, 4), 'bgra'
            if len(frame) > 2:                         # decoded smaller: scale from the video's own size
                out = (int(frame[2][0] * scale_x), int(frame[2][1] * scale_y))
                if out[0] <= 0 or out[1] <= 0:
                    return None
                if out != (fw, fh):
                    arr = cv2.resize(arr, out, interpolation=cv2.INTER_AREA if out[0] < fw else cv2.INTER_LINEAR)
                return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA))
        else:  # BGR(A) or gray ndarray (Compatibility Mode: own_size when decoded smaller)
            arr = frame
            order = 'gray' if arr.ndim == 2 else ('bgra' if arr.shape[2] == 4 else 'bgr')
        h, w = arr.shape[:2]
        bw, bh = getattr(frame, 'own_size', None) or (w, h)
        out = (int(bw * scale_x), int(bh * scale_y))
        if out[0] <= 0 or out[1] <= 0:
            return None
        if out != (w, h):
            arr = cv2.resize(arr, out, interpolation=cv2.INTER_AREA if out[0] < w else cv2.INTER_LINEAR)
        if order == 'rgba':
            return Image.fromarray(np.ascontiguousarray(arr))
        code = {'bgra': cv2.COLOR_BGRA2RGBA, 'bgr': cv2.COLOR_BGR2RGBA, 'gray': cv2.COLOR_GRAY2RGBA}[order]
        return Image.fromarray(cv2.cvtColor(arr, code))

    def _image_to_pil(self, path, scale_x, scale_y):
        img = self._preview_image_cache_get(path)
        if img is None:
            return None
        out = (int(img.width * scale_x), int(img.height * scale_y))
        if out[0] <= 0 or out[1] <= 0:
            return None
        return img.resize(out, Image.BILINEAR)

    def _preview_image_cache_get(self, path):
        """Decoded still images for previews, cached by (path, mtime)."""
        cache = self.__dict__.setdefault('_preview_image_cache', OrderedDict())
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            return None
        img = cache.get(key)
        if img is None:
            img = Image.open(path).convert('RGBA')
            cache[key] = img
            while len(cache) > 32:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        return img

    def close_window(self):
        if self.window:
            pygame.display.quit()
            self.window = None

    def clear(self):
        self.close_window()
        for path, p_info in list(self.active_players.items()):
            p_info['player'].stop()
            if hasattr(p_info['player'], 'join'):
                p_info['player'].join(0.5)
        self.active_players.clear()
        self.video_frame_cache.clear()

VERTEX_SHADER = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aTexCoord;
out vec2 TexCoord;
uniform mat4 model;
uniform mat4 projection;
void main() {
    gl_Position = projection * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D ourTexture;
uniform sampler2D yTexture;
uniform sampler2D uTexture;
uniform sampler2D vTexture;
uniform mat3 yuvMat;   // range scaling + color matrix, see yuv_to_rgb_matrix()
uniform vec3 yuvOff;   // black level / chroma offset
uniform float alpha;
uniform bool isBGR;
uniform bool isYUV;
uniform bool premultiply;  // multiply/screen blend modes need premultiplied color
uniform bool adjust;       // color correction (layer_color / adjust_colors use the same math)
uniform vec3 colorAdj;     // brightness, contrast, saturation
uniform bool masked;       // layer mask (layer_mask.py uses the same math)
uniform int maskShape;     // 0 rectangle, 1 ellipse
uniform vec4 maskRect;     // left, top, right, bottom edges in texture coordinates
uniform vec2 maskAspect;   // media width and height in units of its shorter side
uniform float maskFeather; // soft edge width, in shorter sides
uniform bool maskInvert;

float maskValue(vec2 uv) {
    vec2 lo = maskRect.xy;
    vec2 hi = maskRect.zw;
    float d;               // distance inside the outline, in shorter sides
    if (maskShape == 0) {
        vec2 dd = min(uv - lo, hi - uv) * maskAspect;
        d = min(dd.x, dd.y);
    } else {
        vec2 r = max((hi - lo) * 0.5 * maskAspect, vec2(1e-6));
        vec2 p = (uv - (lo + hi) * 0.5) * maskAspect;
        float k = length(p / r);
        float g = length(p / (r * r));
        d = g > 1e-9 ? (1.0 - k) * k / g : 1e9;
    }
    float t = maskFeather > 0.0 ? clamp(d / maskFeather, 0.0, 1.0) : (d >= 0.0 ? 1.0 : 0.0);
    t = t * t * (3.0 - 2.0 * t);
    return maskInvert ? 1.0 - t : t;
}

void main() {
    vec4 texColor;
    if (isYUV) {
        vec3 yuv = vec3(texture(yTexture, TexCoord).r,
                        texture(uTexture, TexCoord).r,
                        texture(vTexture, TexCoord).r);
        texColor = vec4(clamp(yuvMat * (yuv - yuvOff), 0.0, 1.0), 1.0);
    } else {
        texColor = texture(ourTexture, TexCoord);
        if (isBGR) {
            texColor.rgb = texColor.bgr;
        }
    }
    if (texColor.a < 0.1) discard;
    if (adjust) {
        vec3 c = clamp((texColor.rgb - 0.5) * colorAdj.y + 0.5 + colorAdj.x, 0.0, 1.0);
        float l = dot(c, vec3(0.299, 0.587, 0.114));
        texColor.rgb = clamp(mix(vec3(l), c, colorAdj.z), 0.0, 1.0);
    }
    float a = texColor.a * alpha;
    if (masked) a *= maskValue(TexCoord);
    FragColor = premultiply ? vec4(texColor.rgb * a, a) : vec4(texColor.rgb, a);
}
"""

# Note: an "instanced" batch renderer existed here until 2026-11. It was removed
# because (a) its fragment shader indexed a sampler array with a per-instance
# value, which GLSL 3.30 forbids (Mesa rejects it; other drivers may misrender),
# and (b) it drew all image layers before any video layer, breaking layer order.
# With at most a few dozen layers, one draw call per layer is not a bottleneck.

PREVIEW_MAX_SIZE = (640, 360)


class OpenGLRenderEngine(BaseRenderEngine):
    def __init__(self, media_manager, mode='opengl', hw_accel=None, use_pbo=True, use_vsync=False,
                 preview_bridge=None, **_ignored):
        super().__init__(media_manager, mode, hw_accel)
        log.info(f"OpenGLRenderEngine init '{mode}', accel={hw_accel}, PBO: {use_pbo}, VSync: {use_vsync}")
        self.shader = self.vao = self.vbo = self.ebo = None
        self.textures = {}
        self.texture_formats = {}
        self.u_locs = {}
        self.spanning_window_details = None
        self.preview_bridge = preview_bridge

        # Store the advanced settings
        self.use_pbo = use_pbo
        self.use_vsync = use_vsync

        # PBO management
        self.pbo_pool = {}
        self.pbo_index = {}
        self.PBO_POOL_SIZE = 3

        # YUV texture management: path -> {'y','u','v' texture ids, sizes}
        self.yuv_textures = {}
        self.yuv_pbo_pool = {}
        self.yuv_pbo_index = {}

        # Preview capture (GPU downscale of the back buffer before the swap)
        self._preview_request = None      # monitor index to capture on next render
        self._preview_fbo = None
        self._preview_tex = None
        self._preview_fbo_size = None
        self.last_preview_image = None
        # Corner pin: per monitor (fbo, texture, width, height); one shader + VAO
        self._warp_targets = {}
        self._warp_prog = None
        self._warp_vao = self._warp_vbo = None
        self._warp_locs = {}
        self._fx_reset()
        # Mesh warp: per monitor (key, (vao, vbo, vertex count) or None when it cannot be drawn)
        self._mesh_geo = {}
        self._mesh_prog = None
        self._mesh_locs = {}
        self._mix_targets = {}            # (monitor, 'from'|'to') -> (fbo, texture, width, height)
        self._mix_prog = None
        self._mix_locs = {}

    def _init_pygame_and_gl(self):
        import os, sys, pygame
        sdl_audio_safe.init_pygame()       # without pygame.mixer (sound has its own output)
        pygame.display.init()

        # Ensure the output window uses the same icon as the main GUI (.ico file)
        try:
            from PIL import Image
            base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ico_path = os.path.join(base_dir, 'src', 'visualdeck.ico')
            if os.path.exists(ico_path):
                im = Image.open(ico_path)
                # Pick largest frame in the .ico for best fidelity
                if getattr(im, "n_frames", 1) > 1:
                    best = None; best_area = -1
                    for i in range(im.n_frames):
                        im.seek(i)
                        area = im.size[0] * im.size[1]
                        if area > best_area:
                            best = im.copy()
                            best_area = area
                    im = best
                if im.mode != "RGBA":
                    im = im.convert("RGBA")
                surf = pygame.image.frombuffer(im.tobytes(), im.size, "RGBA")
                pygame.display.set_icon(surf)
        except Exception:
            # Best-effort; never fail initialization over an icon
            pass

        if sys.platform == "win32":
            pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 1)

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

        vsync_val = 1 if self.use_vsync else 0
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, vsync_val)

    def pre_render_static_cue(self, layers, target_monitor=None):
        """Pre-render a static cue to a framebuffer object (FBO)."""
        if not layers or not self.window:
            return None
        
        # Determine dimensions based on target monitor
        if target_monitor is not None and 0 <= target_monitor < len(self.monitors):
            mon = self.monitors[target_monitor]
            width, height = mon.width, mon.height
        else:
            width, height = 1920, 1080  # Default dimensions
        
        # Create FBO for off-screen rendering
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        
        # Create texture to render to
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Attach texture to FBO
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
        
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            log.error("Failed to create framebuffer for static cue pre-rendering")
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [texture])
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return None
        
        # Set viewport for FBO rendering
        glViewport(0, 0, width, height)
        
        # Clear and render
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        # Store premultiplied color and correct coverage in the FBO; drawing it
        # back uses (ONE, ONE_MINUS_SRC_ALPHA). With the normal blend function
        # semi-transparent pixels had their alpha applied twice (a 50% pixel
        # showed at 25%).
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set up projection for FBO
        self._set_projection(self._ortho(0, width, 0, height, -1, 1))
        
        # Render all layers to the FBO
        for layer in sorted(layers.values(), key=lambda x: x.get('order', 0)):
            path = layer.get('path')
            if not path or not os.path.exists(path):
                continue
            
            # Update/get texture for this media
            texture_id = self._update_texture_pbo(path)
            if not texture_id:
                continue
            
            tex_info = self.textures[path]
            
            # Bind texture and set uniforms
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glUniform1i(self.u_locs['ourTexture'], 0)
            glUniform1i(self.u_locs['isYUV'], 0)
            
            is_bgr = self.texture_formats.get(path, False)
            glUniform1i(self.u_locs['isBGR'], int(is_bgr))
            
            # Set layer transform
            model_matrix = self._model_matrix(
                layer.get('position', {}),
                tex_info['width'],
                tex_info['height'],
                (width, height)
            )
            
            glUniformMatrix4fv(self.u_locs['model'], 1, GL_TRUE, model_matrix)
            glUniform1f(self.u_locs['alpha'], layer.get('alpha', 1.0) * layer_opacity(layer))
            
            # Draw
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        _normal_blend()
        # Unbind FBO
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Store the pre-rendered texture and FBO for later use
        pre_rendered_data = {
            'texture': texture,
            'fbo': fbo,
            'width': width,
            'height': height
        }
        
        return pre_rendered_data

    def render_pre_rendered_cue(self, pre_rendered_data, viewport_dims, alpha=1.0):
        """Render a pre-rendered static cue texture to the current viewport."""
        if not pre_rendered_data:
            return
        
        texture = pre_rendered_data['texture']
        width = pre_rendered_data['width']
        height = pre_rendered_data['height']
        
        # Bind the pre-rendered texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        glUniform1i(self.u_locs['ourTexture'], 0)
        glUniform1i(self.u_locs['isYUV'], 0)
        glUniform1i(self.u_locs['isBGR'], 0)
        
        # FIX: Create a flipped model matrix to account for FBO coordinate system
        # We need to flip the Y-axis to correct the upside-down rendering
        model_matrix = self._model_matrix(
            {'x': 0.5, 'y': 0.5, 'scale_x': 1.0, 'scale_y': -1.0, 'rotation': 0},  # Note: scale_y is negative
            width,
            height,
            viewport_dims
        )
        
        glUniformMatrix4fv(self.u_locs['model'], 1, GL_TRUE, model_matrix)
        glUniform1f(self.u_locs['alpha'], alpha)
        
        # Draw the quad (its color is premultiplied, see pre_render_static_cue)
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        _normal_blend()

    def _release_static_entry(self, data):
        self.cleanup_pre_rendered_cue(data)

    def cleanup_pre_rendered_cue(self, pre_rendered_data):
        """Clean up resources used by a pre-rendered cue."""
        if pre_rendered_data:
            if 'fbo' in pre_rendered_data:
                glDeleteFramebuffers(1, [pre_rendered_data['fbo']])
            if 'texture' in pre_rendered_data:
                glDeleteTextures(1, [pre_rendered_data['texture']])

    def _create_window(self, geom, is_spanning=False):
        """Creates a single window, potentially spanning multiple monitors."""
        x, y, w, h = geom
        
        # Check if a window with these exact dimensions already exists
        if self.window and self.window.get_size() == (w, h) and self.spanning_window_details == geom:
            return True
            
        self.close_window()
        # Reset per-context GL caches – the old IDs are invalid after we recreate the window
        self.yuv_textures.clear()
        self.yuv_pbo_pool.clear()
        self.yuv_pbo_index.clear()

        # Prevent implicit centering and request explicit position
        os.environ['SDL_VIDEO_CENTERED'] = '0'
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"

        flags = NOFRAME | DOUBLEBUF | OPENGL
        try:
            # pygame 2 applies vsync to OpenGL windows via SDL_GL_SetSwapInterval.
            self.window = pygame.display.set_mode((w, h), flags, vsync=1 if self.use_vsync else 0)
        except pygame.error as e:
            log.warning(f"Could not set vsync={self.use_vsync} ({e}); creating window without it")
            self.window = pygame.display.set_mode((w, h), flags)
        pygame.display.set_caption("VisualDeck Output (OpenGL)")
        self.spanning_window_details = geom if is_spanning else None

        # Force the position after creation (some drivers ignore WINDOW_POS on first create)
        try:
            import pygame._sdl2 as sdl2
            win = sdl2.Window.from_display_module()
            win.position = (int(x), int(y))
        except Exception:
            pass
        
        glEnable(GL_BLEND)
        _normal_blend()
        # Video planes and BGR frames are tightly packed; the GL default
        # alignment of 4 bytes misreads any row whose byte length is not a
        # multiple of 4 (e.g. 854-pixel-wide video), and with PBOs it makes
        # the upload fail with GL_INVALID_OPERATION.
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)

        # Re-initialize GL resources only if they don't exist
        if not self.shader:
            # Once per GL context: which GPU/driver is actually rendering. On
            # hybrid-graphics laptops this may be the integrated GPU even when
            # a discrete one is present (Windows picks per executable).
            try:
                vendor, renderer, version = (glGetString(e).decode(errors='replace')
                                             for e in (GL_VENDOR, GL_RENDERER, GL_VERSION))
                self.gl_info = {'vendor': vendor, 'renderer': renderer, 'version': version}
                log.info("OpenGL context: vendor=%s | renderer=%s | version=%s", vendor, renderer, version)
            except Exception as e:
                log.warning(f"Could not query OpenGL renderer: {e}")
            self.shader = shaders.compileProgram(
                shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            )
            glUseProgram(self.shader)
            for name in ('projection', 'model', 'alpha', 'isBGR', 'isYUV', 'yuvMat', 'yuvOff',
                         'ourTexture', 'yTexture', 'uTexture', 'vTexture', 'premultiply', 'adjust', 'colorAdj',
                         'masked', 'maskShape', 'maskRect', 'maskAspect', 'maskFeather', 'maskInvert'):
                self.u_locs[name] = glGetUniformLocation(self.shader, name)

            # Geometry data
            quad = np.array([
                 0.5,  0.5, 0.0, 1.0, 0.0,
                 0.5, -0.5, 0.0, 1.0, 1.0,
                -0.5, -0.5, 0.0, 0.0, 1.0,
                -0.5,  0.5, 0.0, 0.0, 0.0
            ], dtype=np.float32)
            indices = np.array([0, 1, 3, 1, 2, 3], dtype=np.uint32)

            self.vao = glGenVertexArrays(1)
            self.vbo = glGenBuffers(1)
            self.ebo = glGenBuffers(1)
            glBindVertexArray(self.vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
            glEnableVertexAttribArray(1)

            # Texture units: RGB(A) on 0; Y/U/V planes on 0/1/2
            glUniform1i(self.u_locs['ourTexture'], 0)
            glUniform1i(self.u_locs['yTexture'], 0)
            glUniform1i(self.u_locs['uTexture'], 1)
            glUniform1i(self.u_locs['vTexture'], 2)
            glBindVertexArray(self.vao)

        return True

    def _ortho(self, l, r, b, t, n, f):
        return np.array([
            [2/(r-l), 0, 0, -(r+l)/(r-l)],
            [0, 2/(t-b), 0, -(t+b)/(t-b)],
            [0, 0, -2/(f-n), -(f+n)/(f-n)],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def _model_matrix(self, pos_data, width, height, viewport_dims):
        win_w, win_h = viewport_dims
        S = np.eye(4, dtype=np.float32)
        S[0,0] = width * pos_data.get('scale_x', 1.0)
        S[1,1] = height * pos_data.get('scale_y', 1.0)
        angle_rad = np.radians(pos_data.get('rotation', 0))
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s, 0, 0], [s,  c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[0,3] = pos_data.get('x', 0.5) * win_w
        T[1,3] = (1.0 - pos_data.get('y', 0.5)) * win_h
        return T @ R @ S

    def _update_texture_pbo(self, path):
        """Updates texture using PBO for asynchronous transfer with BGR support."""
        # Quick return if texture exists and is static (not a video)
        if path in self.textures and os.path.splitext(path)[1].lower() not in VIDEO_EXTS:
            return self.textures[path].get('id')
            
        is_video = os.path.splitext(path)[1].lower() in VIDEO_EXTS
        frame = self._get_latest_video_frame(path) if is_video else cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if frame is None:
            return self.textures.get(path, {}).get('id')

        # Determine format and prepare frame
        h, w = frame.shape[:2]
        use_bgr = is_video  # Use BGR upload for videos, RGBA for images
        
        if is_video:
            # Keep BGR format for videos
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            # If already 3 channels, assume it's BGR from the video decoder
            gl_format = GL_BGR
            gl_internal_format = GL_RGB  # Internal storage is still RGB
            channels = 3
        else:
            # Convert images to RGBA as before
            if frame.ndim == 2: 
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
            elif frame.shape[2] == 3: 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            else: 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            gl_format = GL_RGBA
            gl_internal_format = GL_RGBA
            channels = 4
        
        frame = np.ascontiguousarray(frame)
        frame_size = frame.nbytes
        
        # Store format information - we don't need shader swizzling when using GL_BGR
        self.texture_formats[path] = False  # Shader doesn't need to swizzle

        # Check if this is a restart (texture exists but PBO index was reset)
        is_restart = path in self.textures and path not in self.pbo_pool

        if path not in self.textures:
            # First time setup for this media
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            # Allocate texture memory with appropriate format
            glTexImage2D(GL_TEXTURE_2D, 0, gl_internal_format, w, h, 0, gl_format, GL_UNSIGNED_BYTE, frame)
            
            self.textures[path] = {'id': texture_id, 'width': w, 'height': h, 'channels': channels}
            
            # Create PBO pool for video files
            if is_video:
                self.pbo_index[path] = 0
                pbos = glGenBuffers(self.PBO_POOL_SIZE)
                # Convert to list if it's a numpy array
                if isinstance(pbos, np.ndarray):
                    self.pbo_pool[path] = pbos.tolist()
                elif isinstance(pbos, int):
                    self.pbo_pool[path] = [pbos]
                else:
                    self.pbo_pool[path] = list(pbos)
                
                # Initialize ALL PBOs with the first frame data
                for i, pbo in enumerate(self.pbo_pool[path]):
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, int(pbo))
                    glBufferData(GL_PIXEL_UNPACK_BUFFER, frame_size, frame, GL_STREAM_DRAW)
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        else:
            # Update existing texture
            tex_info = self.textures[path]
            
            # Handle size or format changes
            if tex_info['width'] != w or tex_info['height'] != h or tex_info.get('channels', 4) != channels:
                glBindTexture(GL_TEXTURE_2D, tex_info['id'])
                glTexImage2D(GL_TEXTURE_2D, 0, gl_internal_format, w, h, 0, gl_format, GL_UNSIGNED_BYTE, frame)
                tex_info['width'], tex_info['height'], tex_info['channels'] = w, h, channels
                
                # Recreate PBOs with new size if needed
                if path in self.pbo_pool:
                    glDeleteBuffers(len(self.pbo_pool[path]), self.pbo_pool[path])
                
                if is_video:
                    pbos = glGenBuffers(self.PBO_POOL_SIZE)
                    # Convert to list if it's a numpy array
                    if isinstance(pbos, np.ndarray):
                        self.pbo_pool[path] = pbos.tolist()
                    elif isinstance(pbos, int):
                        self.pbo_pool[path] = [pbos]
                    else:
                        self.pbo_pool[path] = list(pbos)
                    self.pbo_index[path] = 0
                    
                    # Initialize ALL PBOs with the first frame
                    for pbo in self.pbo_pool[path]:
                        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, int(pbo))
                        glBufferData(GL_PIXEL_UNPACK_BUFFER, frame_size, frame, GL_STREAM_DRAW)
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
                
                return tex_info['id']
            
            # Handle restart case - recreate PBOs
            if is_restart and is_video:
                self.pbo_index[path] = 0
                pbos = glGenBuffers(self.PBO_POOL_SIZE)
                # Convert to list if it's a numpy array
                if isinstance(pbos, np.ndarray):
                    self.pbo_pool[path] = pbos.tolist()
                elif isinstance(pbos, int):
                    self.pbo_pool[path] = [pbos]
                else:
                    self.pbo_pool[path] = list(pbos)
                
                # Initialize ALL PBOs with the first frame
                for pbo in self.pbo_pool[path]:
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, int(pbo))
                    glBufferData(GL_PIXEL_UNPACK_BUFFER, frame_size, frame, GL_STREAM_DRAW)
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        
        # Use PBO for video updates, direct update for images
        if is_video and path in self.pbo_pool and self.use_pbo:
            # For the first few frames after restart, update texture directly
            current_index = self.pbo_index[path]
            
            # Check if we need to prime the PBO pipeline
            if current_index == 0 and is_restart:
                # Direct update for the first frame after restart
                glBindTexture(GL_TEXTURE_2D, self.textures[path]['id'])
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, gl_format, GL_UNSIGNED_BYTE, frame)
                # Also fill the first PBO
                pbo = int(self.pbo_pool[path][0])
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
                glBufferData(GL_PIXEL_UNPACK_BUFFER, frame_size, frame, GL_STREAM_DRAW)
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
                self.pbo_index[path] = 1
            else:
                # Normal PBO update path
                next_index = (current_index + 1) % self.PBO_POOL_SIZE
                
                # Upload frame data to the NEXT PBO
                next_pbo = int(self.pbo_pool[path][next_index])
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, next_pbo)
                
                # Orphan the buffer to avoid sync issues
                glBufferData(GL_PIXEL_UNPACK_BUFFER, frame_size, None, GL_STREAM_DRAW)
                
                # Map buffer and copy data
                try:
                    ptr = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, frame_size, 
                                         GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT)
                    if ptr:
                        ctypes.memmove(ptr, frame.ctypes.data, frame_size)
                        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER)
                except GLError as e:
                    log.warning(f"PBO mapping failed, falling back to glBufferSubData: {e}")
                    # Fallback to glBufferSubData if mapping fails
                    glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, frame_size, frame)
                
                # Update texture from CURRENT PBO (data from previous frame)
                current_pbo = int(self.pbo_pool[path][current_index])
                glBindTexture(GL_TEXTURE_2D, self.textures[path]['id'])
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, current_pbo)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, gl_format, GL_UNSIGNED_BYTE, None)
                
                # Advance index for next frame
                self.pbo_index[path] = next_index
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        else:
            # Direct update for images or if PBOs are disabled
            glBindTexture(GL_TEXTURE_2D, self.textures[path]['id'])
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, gl_format, GL_UNSIGNED_BYTE, frame)
                
        return self.textures[path]['id']

    # --- Video textures -------------------------------------------------------

    def _upload_plane(self, tex, pbo, width, height, data, gl_internal, gl_format):
        """Upload one tightly packed plane. Uses an orphaned PBO when enabled."""
        glBindTexture(GL_TEXTURE_2D, tex)
        if pbo is not None:
            nbytes = data.nbytes
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, nbytes, None, GL_STREAM_DRAW)  # orphan
            glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, nbytes, data)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, gl_format, GL_UNSIGNED_BYTE, None)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, gl_format, GL_UNSIGNED_BYTE, data)

    @staticmethod
    def _new_texture(width, height, gl_internal, gl_format):
        tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, gl_internal, width, height, 0, gl_format, GL_UNSIGNED_BYTE, None)
        return tex

    def _delete_video_textures(self, path):
        self._fx_forget(path)
        info = self.yuv_textures.pop(path, None)
        self.textures.pop(path, None)
        self.texture_formats.pop(path, None)
        if not info:
            return
        if self.window:
            try:
                glDeleteTextures(len(info['textures']), info['textures'])
                if info.get('pbos'):
                    glDeleteBuffers(len(info['pbos']), info['pbos'])
            except Exception as e:
                log.debug(f"Texture cleanup for {os.path.basename(path)} failed: {e}")

    def _update_texture_yuv(self, path):
        """Upload the newest decoded frame of a video and return its texture id.

        Handles 'yuv420p' frames (three R8 planes, converted in the shader) and
        'rgba' frames (videos with alpha). A frame that has already been
        uploaded is not uploaded again.
        """
        frame = self._get_latest_video_frame(path)
        info = self.yuv_textures.get(path)
        if frame is None:
            return self.textures.get(path, {}).get('id')
        if not isinstance(frame, dict):
            return self._update_texture_pbo(path)  # legacy BGR ndarray (CPU decoder)
        if info is not None and info.get('frame') is frame:
            return self.textures[path]['id']

        w, h = frame['width'], frame['height']
        if frame['format'] == 'rgba':
            layout = ('rgba', w, h)
            planes = [(frame['rgba'], w, h, GL_RGBA8, GL_RGBA)]
        else:
            u = frame['u']
            layout = ('yuv', w, h, u.shape[1], u.shape[0])
            planes = [(frame['y'], w, h, GL_R8, GL_RED),
                      (frame['u'], u.shape[1], u.shape[0], GL_R8, GL_RED),
                      (frame['v'], u.shape[1], u.shape[0], GL_R8, GL_RED)]

        if info is None or info['layout'] != layout:
            self._delete_video_textures(path)
            texs = [self._new_texture(pw, ph, gi, gf) for (_, pw, ph, gi, gf) in planes]
            pbos = None
            if self.use_pbo:
                ids = glGenBuffers(len(planes))
                pbos = [int(i) for i in (np.atleast_1d(ids).tolist())]
            info = {'layout': layout, 'textures': texs, 'pbos': pbos, 'frame': None}
            self.yuv_textures[path] = info
            # The layer's geometry uses the video's own size (its scale is relative to it),
            # also when the frames are decoded smaller (decode_scale_for).
            self.textures[path] = {'id': texs[0], 'width': frame.get('src_width', w),
                                   'height': frame.get('src_height', h),
                                   'channels': 4 if layout[0] == 'rgba' else 3,
                                   'is_yuv': layout[0] == 'yuv'}
            self.texture_formats[path] = False

        for i, (data, pw, ph, gi, gf) in enumerate(planes):
            pbo = info['pbos'][i] if info['pbos'] else None
            self._upload_plane(info['textures'][i], pbo, pw, ph, data, gi, gf)
        info['frame'] = frame
        if layout[0] == 'yuv':
            info['color'] = (frame.get('range', 'limited'), frame.get('matrix', 'bt709'))
        return info['textures'][0]

    def _set_yuv_uniforms(self, color):
        cache = self.__dict__.setdefault('_yuv_matrix_cache', {})
        if color not in cache:
            cache[color] = yuv_to_rgb_matrix(*color)
        m, off = cache[color]
        glUniformMatrix3fv(self.u_locs['yuvMat'], 1, GL_TRUE, m)
        glUniform3f(self.u_locs['yuvOff'], float(off[0]), float(off[1]), float(off[2]))

    # --- Shader effects (isf.py) -------------------------------------------------
    draws_effects = True              # the other engines draw layers without their effect
    decodes_at_drawn_size = True      # textures keep the video's own size for the geometry
    FX_MAX_SIDE = 4096

    def _set_projection(self, matrix):
        self._projection = matrix         # kept for the effect passes, which change it
        glUniformMatrix4fv(self.u_locs['projection'], 1, GL_TRUE, matrix)

    def _fx_reset(self):
        self.__dict__.pop('_warp_geo', None)     # layer warp surfaces belong to the old context
        self._projection = None
        self._fx_programs = {}        # effect name -> (program, effect, uniform locations) or None
        self._fx_targets = {}         # (layer key, 'a' | 'b') -> (fbo, texture, w, h)
        self._fx_clock = {}           # layer key -> [first time, last time, frames]
        self.effect_errors = {}       # effect name -> why it does not run
        self._fx_stale = set()        # names whose files changed: compiled again when next drawn
        self._fx_buffers = {}         # (stage key, effect name, TARGET) -> pass buffer (_fx_buffer)
        self._unpremult_prog = None
        self._unpremult_loc = -1
        self._fx_preview = {'src': None, 'id': None, 'read_fbo': None, 'pbo': None, 'pending': None}

    def reload_effects(self, names):
        """UI thread: these effect files changed; the render thread compiles them again."""
        self._fx_stale.update(names)

    def _delete_fx(self):
        for fbo, tex, _w, _h in self._fx_targets.values():
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [tex])
        for buf in self._fx_buffers.values():
            self._delete_fx_buffer(buf)
        if self._unpremult_prog:
            glDeleteProgram(self._unpremult_prog)
        self.effect_preview_end()
        for entry in self._fx_programs.values():
            if entry:
                glDeleteProgram(entry[0])

    def _fx_forget(self, key):
        """A layer's (or display's) effect buffers, every stage, when it goes."""
        def ours(k):
            return k == key or (isinstance(k, tuple) and len(k) == 2 and k[0] == key and isinstance(k[1], int))
        for k in [k for k in self._fx_targets if ours(k[0])]:
            t = self._fx_targets.pop(k)
            if self.window:
                try:
                    glDeleteFramebuffers(1, [t[0]])
                    glDeleteTextures(1, [t[1]])
                except Exception:  # noqa: BLE001
                    pass
        for k in [k for k in self._fx_clock if ours(k)]:
            self._fx_clock.pop(k, None)
        for k in [k for k in self._fx_buffers if ours(k[0])]:
            buf = self._fx_buffers.pop(k)
            if self.window:
                self._delete_fx_buffer(buf)

    def _effect_program(self, name):
        if self._fx_stale and name in self._fx_stale:
            self._fx_stale.discard(name)
            old = self._fx_programs.pop(name, None)
            if old:
                glDeleteProgram(old[0])
        if name in self._fx_programs:
            return self._fx_programs[name]
        entry = None
        try:
            eff = isf.load(name)
            prog = shaders.compileProgram(shaders.compileShader(isf.VERTEX_SHADER, GL_VERTEX_SHADER),
                                          shaders.compileShader(isf.fragment_source(eff), GL_FRAGMENT_SHADER))
            names = ['inputImage', 'RENDERSIZE', 'TIME', 'TIMEDELTA', 'FRAMEINDEX', 'PASSINDEX', 'DATE',
                     'vd_bottomUp', 'vd_premultiplied', 'vd_finalPass']
            locs = {n: glGetUniformLocation(prog, n)
                    for n in names + [i['NAME'] for i in eff['inputs']] + list(eff['targets'])}
            entry = (prog, eff, locs)
            self.effect_errors.pop(name, None)
        except Exception as e:  # noqa: BLE001 - a bad effect file must not stop the output
            self.effect_errors[name] = isf.compile_message(e)
            log.error(f"Effect {name}: {self.effect_errors[name]}; the layer is drawn without it")
        self._fx_programs[name] = entry
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)
        return entry

    def queue_effect_warmup(self, names):
        """Compile these effects' shaders before their first draw (warm_one_effect,
        one per frame). A first draw compiled on the spot stalled the output: 152 ms
        on the owner's AMD driver for one effect at a GO (acceptance run, 2026-11-14)."""
        pend = self.__dict__.setdefault('_fx_warm', [])
        for n in names or ():
            if n and n not in pend and n not in getattr(self, '_fx_programs', {}):
                pend.append(n)

    def warm_one_effect(self):
        """Compile one queued effect between frames (render thread, with the output's
        GL context). Returns the name compiled, or None."""
        pend = getattr(self, '_fx_warm', None)
        while pend and self.window is not None:
            name = pend.pop(0)
            if name in self._fx_programs:
                continue
            t0 = time.perf_counter()
            entry = self._effect_program(name)
            ms = (time.perf_counter() - t0) * 1000.0
            log.info(f"Effect {name} {'compiled' if entry else 'failed'} ahead of its first use ({ms:.0f} ms)")
            return name
        return None

    def _free_unused_fx(self):
        """Free effect buffers this frame did not use: an effect taken off a
        layer, a layer no longer drawn (its persistent buffers start again
        when it is shown again), a shorter stack."""
        seen = getattr(self, '_fx_seen', None)
        if seen is None:
            return
        keep = lambda k: k in seen or self._is_preview_key(k[0])   # noqa: E731 - the editor's, freed by it
        for k in [k for k in self._fx_targets if not keep(k)]:
            fbo, tex, _w, _h = self._fx_targets.pop(k)
            try:
                glDeleteFramebuffers(1, [fbo])
                glDeleteTextures(1, [tex])
            except Exception:  # noqa: BLE001
                pass
        for k in [k for k in self._fx_buffers if not keep(k)]:
            self._delete_fx_buffer(self._fx_buffers.pop(k))

    # --- The position editor's effect preview ---------------------------------
    # The editor shows a layer with its effects as the output draws them: the
    # render thread draws the effects on the editor's picture here, scales the
    # result to the editor's canvas and reads it back through a pixel buffer, so
    # the output never waits for the read (effect_preview_poll collects it once
    # the GPU has finished, a frame or two later).
    PREVIEW_KEY = '@editor-preview'

    @classmethod
    def _is_preview_key(cls, k):
        if isinstance(k, tuple) and len(k) == 2 and isinstance(k[1], int):
            k = k[0]                                   # a later effect of a stack: (key, i)
        return isinstance(k, str) and k.startswith(cls.PREVIEW_KEY)

    def effect_preview_busy(self):
        return self._fx_preview['pending'] is not None

    def effect_preview_begin(self, req):
        """Start drawing req['effects'] on the editor's picture. req: 'image'
        (RGBA uint8, top row first; None = the one sent before), 'image_id',
        'drawn' (the layer's size on the output, pixels) and 'out' (the size
        wanted). Returns None when started, else why it cannot be drawn."""
        if not self.window:
            return "no output window"
        if not self._ensure_warp_program():
            return "the effect shaders are not available"
        pv = self._fx_preview
        img = req.get('image')
        if img is not None and pv['id'] != req.get('image_id'):
            h, w = img.shape[:2]
            if pv['src'] is not None:
                glDeleteTextures(1, [pv['src']['id']])
            tex = self._new_texture(w, h, GL_RGBA8, GL_RGBA)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE,
                            np.ascontiguousarray(img, dtype=np.uint8))
            pv['src'], pv['id'] = {'id': tex, 'width': w, 'height': h}, req.get('image_id')
            self._fx_forget(self.PREVIEW_KEY)             # Trails and the like start again on a new picture
        if pv['src'] is None:
            return "no picture"
        fx_tex = self._apply_effects(self.PREVIEW_KEY, req.get('effects') or [], pv['src'], None,
                                     drawn=req.get('drawn'))
        tex = fx_tex if fx_tex is not None else pv['src']['id']
        glBindTexture(GL_TEXTURE_2D, tex)
        sw = int(glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH))
        sh = int(glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT))
        ow, oh = (max(1, int(v)) for v in req['out'])
        dst = self._fx_target(self.PREVIEW_KEY + '/out', 'b', ow, oh)
        if dst is None:
            return "no buffer for the preview"
        prev_fb = int(glGetIntegerv(GL_FRAMEBUFFER_BINDING))
        if pv['read_fbo'] is None:
            pv['read_fbo'] = int(glGenFramebuffers(1))
            pv['pbo'] = int(np.atleast_1d(glGenBuffers(1))[0])
        glBindFramebuffer(GL_READ_FRAMEBUFFER, pv['read_fbo'])
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst[0])
        glBlitFramebuffer(0, 0, sw, sh, 0, 0, ow, oh, GL_COLOR_BUFFER_BIT, GL_LINEAR)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, dst[0])
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pv['pbo'])
        glBufferData(GL_PIXEL_PACK_BUFFER, ow * oh * 4, None, GL_STREAM_READ)
        glPixelStorei(GL_PACK_ALIGNMENT, 4)
        glReadPixels(0, 0, ow, oh, GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fb)
        pv['pending'] = (fence, ow, oh, req)
        return None

    def effect_preview_poll(self, wait=False):
        """(req, RGBA array top row first) once the GPU has finished the preview
        begun last, else None. wait: block until it has (tests)."""
        pv = self._fx_preview
        if pv['pending'] is None:
            return None
        fence, ow, oh, req = pv['pending']
        r = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 2_000_000_000 if wait else 0)
        if r not in (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED):
            return None
        glDeleteSync(fence)
        pv['pending'] = None
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pv['pbo'])
        data = glGetBufferSubData(GL_PIXEL_PACK_BUFFER, 0, ow * oh * 4)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        arr = np.frombuffer(bytes(data) if not isinstance(data, np.ndarray) else data.tobytes(),
                            dtype=np.uint8).reshape(oh, ow, 4)
        return req, arr

    def effect_preview_end(self):
        """The editor closed: free its picture and the effects' buffers."""
        pv = getattr(self, '_fx_preview', None)
        if not pv:
            return
        try:
            if pv['pending'] is not None:
                glDeleteSync(pv['pending'][0])
            if pv['src'] is not None:
                glDeleteTextures(1, [pv['src']['id']])
            if pv['read_fbo'] is not None:
                glDeleteFramebuffers(1, [pv['read_fbo']])
                glDeleteBuffers(1, [pv['pbo']])
        except Exception:  # noqa: BLE001 - the context may already be gone
            pass
        self._fx_preview = {'src': None, 'id': None, 'read_fbo': None, 'pbo': None, 'pending': None}
        for k in (self.PREVIEW_KEY, self.PREVIEW_KEY + '/out'):
            try:
                self._fx_forget(k)
            except Exception:  # noqa: BLE001
                pass

    def _fx_target(self, key, slot, w, h):
        seen = getattr(self, '_fx_seen', None)
        if seen is not None:
            seen.add((key, slot))
        t = self._fx_targets.get((key, slot))
        if t is not None and t[2:] == (w, h):
            return t
        if t is not None:
            glDeleteFramebuffers(1, [t[0]])
            glDeleteTextures(1, [t[1]])
        tex = self._new_texture(w, h, GL_RGBA8, GL_RGBA)
        fbo = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [tex])
            self._fx_targets.pop((key, slot), None)
            return None
        self._fx_targets[(key, slot)] = t = (fbo, tex, w, h)
        return t

    @classmethod
    def effect_scale(cls, src_w, src_h, drawn=None):
        """The fraction of a picture's size its effects run at: no more than
        FX_MAX_SIDE, and no more than the size it is drawn at on the output
        (`drawn`, pixels): a 4K clip shown at 1920 wide runs its effects at
        1920, a quarter of the work, for the same picture (measured on the
        owner's PC, 2026-11-14: Wave on a 4K video took late frames from 12 to
        about 60 in 5 s). Rounded up to eighths so a moving or edited scale
        does not resize the effect's buffers (and restart Trails) every frame."""
        s = min(1.0, cls.FX_MAX_SIDE / max(src_w, src_h, 1))
        if drawn is not None:
            need = max(abs(drawn[0]) / max(src_w, 1), abs(drawn[1]) / max(src_h, 1))
            s = min(s, max(0.125, math.ceil(need * 8 - 1e-9) / 8))
        return s

    def _apply_effects(self, key, effects, tex_info, yuv, display=False, drawn=None):
        """Run a stack of effects in order (each on the result of the one
        before); the last result's texture id, or None if none was drawn. An
        effect that cannot run is left out of the chain. drawn: the picture's
        size on the output in pixels (see effect_scale)."""
        tex = None
        for i, fx in enumerate(effects):
            stage = key if i == 0 else (key, i)
            info = tex_info if tex is None else {'id': tex, 'width': tex_info['width'], 'height': tex_info['height']}
            out = self._apply_effect(stage, fx, info, yuv if tex is None else None, display, drawn)
            if out is not None:
                tex = out
        return tex

    def _apply_effect(self, key, fx, tex_info, yuv, display=False, drawn=None):
        """Run one effect on a picture (all its passes); the result's texture
        id, or None (no effect drawn: unknown, refused or failed, logged once).
        display: the picture is a display's (bottom row first, premultiplied by
        its coverage); the effect gets plain colors and its result is
        premultiplied again."""
        entry = self._effect_program(fx['name'])
        if entry is None or not self._ensure_warp_program():
            return None
        prog, eff, locs = entry
        w, h = tex_info['width'], tex_info['height']
        s = self.effect_scale(w, h, drawn)
        w, h = max(1, int(round(w * s))), max(1, int(round(h * s)))
        prev_fb = int(glGetIntegerv(GL_FRAMEBUFFER_BINDING))
        prev_vp = [int(v) for v in glGetIntegerv(GL_VIEWPORT)]
        blending = bool(glIsEnabled(GL_BLEND))
        glDisable(GL_BLEND)

        def done(result):
            glBindFramebuffer(GL_FRAMEBUFFER, prev_fb)
            glViewport(*prev_vp)
            glUseProgram(self.shader)
            glBindVertexArray(self.vao)
            if blending:
                glEnable(GL_BLEND)
            return result

        bgr = bool(self.texture_formats.get(key, False))
        src = tex_info['id']
        if yuv is not None or bgr:
            # The picture as RGBA, top row first, drawn by the layer shader (it converts YUV and BGR).
            a = self._fx_target(key, 'a', w, h)
            if a is None:
                return done(None)
            glBindFramebuffer(GL_FRAMEBUFFER, a[0])
            glViewport(0, 0, w, h)
            glClearColor(0, 0, 0, 0)
            glClear(GL_COLOR_BUFFER_BIT)
            if yuv is not None:
                for unit, tex in enumerate(yuv['textures']):
                    glActiveTexture(GL_TEXTURE0 + unit)
                    glBindTexture(GL_TEXTURE_2D, tex)
                glActiveTexture(GL_TEXTURE0)
                glUniform1i(self.u_locs['isYUV'], 1)
                self._set_yuv_uniforms(yuv.get('color', ('limited', 'bt709')))
            else:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, src)
                glUniform1i(self.u_locs['isYUV'], 0)
                glUniform1i(self.u_locs['isBGR'], 1)
            glUniformMatrix4fv(self.u_locs['projection'], 1, GL_TRUE, np.eye(4, dtype=np.float32))
            glUniformMatrix4fv(self.u_locs['model'], 1, GL_TRUE, np.diag([2.0, -2.0, 1.0, 1.0]).astype(np.float32))
            glUniform1f(self.u_locs['alpha'], 1.0)
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
            if self._projection is not None:
                glUniformMatrix4fv(self.u_locs['projection'], 1, GL_TRUE, self._projection)
            src = a[1]
        elif display:
            # A display's picture in plain colors (alpha keeps the coverage).
            a = self._fx_target(key, 'a', w, h)
            if a is None or not self._ensure_unpremultiply_program():
                return done(None)
            glBindFramebuffer(GL_FRAMEBUFFER, a[0])
            glViewport(0, 0, w, h)
            glUseProgram(self._unpremult_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, src)
            glUniform1i(self._unpremult_loc, 0)
            glBindVertexArray(self._warp_vao)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            src = a[1]
        b = self._fx_target(key, 'b', w, h)
        if b is None:
            return done(None)
        now = time.perf_counter()
        clock = self._fx_clock.setdefault(key, [now, now, 0])
        dt, clock[1] = now - clock[1], now
        glUseProgram(prog)
        glUniform1f(locs['TIME'], float(now - clock[0]))
        glUniform1f(locs['TIMEDELTA'], float(dt))
        glUniform1i(locs['FRAMEINDEX'], int(clock[2]))
        glUniform1i(locs['vd_bottomUp'], int(bool(display)))
        glUniform1i(locs['vd_premultiplied'], int(bool(display)))
        lt = time.localtime()
        glUniform4f(locs['DATE'], float(lt.tm_year), float(lt.tm_mon), float(lt.tm_mday),
                    float(lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec))
        clock[2] += 1
        values = {}
        for inp in eff['inputs']:
            v = isf.clean_value(inp, fx['inputs'].get(inp['NAME'], inp['DEFAULT']))
            values[inp['NAME']] = v
            loc = locs.get(inp['NAME'], -1)
            if loc is None or loc < 0:
                continue
            t = inp['TYPE']
            if t == 'float':
                glUniform1f(loc, float(v))
            elif t in ('bool', 'long'):
                glUniform1i(loc, int(v))
            elif t == 'color':
                glUniform4f(loc, *[float(c) for c in v])
            elif t == 'point2D':
                glUniform2f(loc, float(v[0]), float(v[1]))
        # Buffers of the passes (TARGET), each at the size its first pass gives it.
        buffers = {}
        for p in eff['passes']:
            name = p['target']
            if name and name not in buffers:
                pw, ph = self._pass_size(p, w, h, values)
                buf = self._fx_buffer((key, fx['name'], name), pw, ph, p['float'])
                if buf is None:
                    return done(None)
                buffers[name] = buf
        result = b[1]
        glBindVertexArray(self._warp_vao)
        last = len(eff['passes']) - 1
        for i, p in enumerate(eff['passes']):
            buf = buffers.get(p['target'])
            if buf is not None:
                pw, ph = buf['size']
                glBindFramebuffer(GL_FRAMEBUFFER, buf['fbo'][1])        # draw into the back texture
            else:
                pw, ph = w, h
                glBindFramebuffer(GL_FRAMEBUFFER, b[0])
                glClearColor(0, 0, 0, 0)
                glClear(GL_COLOR_BUFFER_BIT)
            glViewport(0, 0, pw, ph)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, src)
            glUniform1i(locs['inputImage'], 0)
            for unit, name in enumerate(eff['targets'], start=1):    # every buffer as it is now (front)
                glActiveTexture(GL_TEXTURE0 + unit)
                glBindTexture(GL_TEXTURE_2D, buffers[name]['tex'][0])
                loc = locs.get(name, -1)
                if loc is not None and loc >= 0:
                    glUniform1i(loc, unit)
            glActiveTexture(GL_TEXTURE0)
            glUniform2f(locs['RENDERSIZE'], float(pw), float(ph))
            glUniform1i(locs['PASSINDEX'], i)
            glUniform1i(locs['vd_finalPass'], int(i == last))
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            if buf is not None:                                      # what was drawn is now the buffer
                buf['tex'].reverse()
                buf['fbo'].reverse()
                if i == last:
                    result = buf['tex'][0]
        return done(result)

    def _pass_size(self, p, w, h, values):
        """(width, height) of a pass's buffer: its WIDTH/HEIGHT expressions, or the picture's size."""
        out = []
        for expr, full in ((p.get('width'), w), (p.get('height'), h)):
            v = full
            if expr is not None:
                try:
                    v = isf.size_expr(expr, w, h, values)
                except isf.ISFError:
                    v = full
            out.append(max(1, min(self.FX_MAX_SIDE, int(round(v)))))
        return tuple(out)

    def _fx_buffer(self, bkey, w, h, flt):
        """A pass buffer: two textures with framebuffers (the one read, the one
        drawn), cleared to transparent black when made or resized."""
        seen = getattr(self, '_fx_seen', None)
        if seen is not None:
            seen.add(bkey)
        buf = self._fx_buffers.get(bkey)
        if buf is not None and buf['size'] == (w, h) and buf['float'] == flt:
            return buf
        if buf is not None:
            self._delete_fx_buffer(buf)
        fmt = GL_RGBA32F if flt else GL_RGBA8
        tex = [self._new_texture(w, h, fmt, GL_RGBA) for _ in range(2)]
        fbo = []
        for t in tex:
            f = int(glGenFramebuffers(1))
            glBindFramebuffer(GL_FRAMEBUFFER, f)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0)
            fbo.append(f)
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                self._delete_fx_buffer({'tex': tex, 'fbo': fbo})
                self._fx_buffers.pop(bkey, None)
                return None
            glViewport(0, 0, w, h)
            glClearColor(0, 0, 0, 0)
            glClear(GL_COLOR_BUFFER_BIT)
        self._fx_buffers[bkey] = buf = {'tex': tex, 'fbo': fbo, 'size': (w, h), 'float': flt}
        return buf

    @staticmethod
    def _delete_fx_buffer(buf):
        try:
            glDeleteFramebuffers(len(buf['fbo']), buf['fbo'])
            glDeleteTextures(len(buf['tex']), buf['tex'])
        except Exception:  # noqa: BLE001
            pass

    def _ensure_unpremultiply_program(self):
        if self._unpremult_prog is not None:
            return True
        try:
            self._unpremult_prog = shaders.compileProgram(
                shaders.compileShader(UNPREMULTIPLY_VERTEX_SHADER, GL_VERTEX_SHADER),
                shaders.compileShader(UNPREMULTIPLY_FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
        except Exception:  # noqa: BLE001
            log.exception("Display effect: the color conversion shader failed to compile")
            return False
        self._unpremult_loc = glGetUniformLocation(self._unpremult_prog, 'picture')
        return True

    def _draw_layer(self, path, layer, viewport_dims):
        tex_info = self.textures.get(path)
        if not tex_info:
            return
        yuv = self.yuv_textures.get(path) if tex_info.get('is_yuv') else None
        effects = layer_effects(layer)
        fx_tex = None
        if effects:
            pos = layer.get('position', {}) or {}
            drawn = (tex_info['width'] * float(pos.get('scale_x', 1.0) or 0.0),
                     tex_info['height'] * float(pos.get('scale_y', 1.0) or 0.0))
            fx_tex = self._apply_effects(path, effects, tex_info, yuv, drawn=drawn)
        if fx_tex is not None:                    # the effect's picture, as a plain RGBA texture
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, fx_tex)
            glUniform1i(self.u_locs['isYUV'], 0)
            glUniform1i(self.u_locs['isBGR'], 0)
        elif yuv is not None:
            for unit, tex in enumerate(yuv['textures']):
                glActiveTexture(GL_TEXTURE0 + unit)
                glBindTexture(GL_TEXTURE_2D, tex)
            glActiveTexture(GL_TEXTURE0)
            glUniform1i(self.u_locs['isYUV'], 1)
            self._set_yuv_uniforms(yuv.get('color', ('limited', 'bt709')))
        else:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, tex_info['id'])
            glUniform1i(self.u_locs['isYUV'], 0)
            glUniform1i(self.u_locs['isBGR'], int(self.texture_formats.get(path, False)))
        model_matrix = self._model_matrix(layer.get('position', {}), tex_info['width'], tex_info['height'], viewport_dims)
        glUniformMatrix4fv(self.u_locs['model'], 1, GL_TRUE, model_matrix)
        glUniform1f(self.u_locs['alpha'], layer.get('alpha', 1.0))
        warp = self._layer_warp_geometry(layer_warp.layer_warp(layer))
        mode = layer_blend(layer)
        if mode != 'normal':
            src, dst, premul = _GL_BLEND[mode]
            glBlendFunc(src, dst)
            glUniform1i(self.u_locs['premultiply'], int(premul))
        adj = layer_color(layer)
        if adj is not None:
            glUniform1i(self.u_locs['adjust'], 1)
            glUniform3f(self.u_locs['colorAdj'], *adj)
        mask = layer_mask.layer_mask(layer)
        if mask is not None:
            shape, lo_u, lo_v, hi_u, hi_v, feather, invert = mask
            glUniform1i(self.u_locs['masked'], 1)
            glUniform1i(self.u_locs['maskShape'], layer_mask.SHAPES.index(shape))
            glUniform4f(self.u_locs['maskRect'], lo_u, lo_v, hi_u, hi_v)
            glUniform2f(self.u_locs['maskAspect'], *layer_mask.aspect(tex_info['width'], tex_info['height']))
            glUniform1f(self.u_locs['maskFeather'], feather)
            glUniform1i(self.u_locs['maskInvert'], int(invert))
        if warp is not None:                      # the bent picture (layer_warp): the same shader, more triangles
            glBindVertexArray(warp[0])
            glDrawArrays(GL_TRIANGLES, 0, warp[2])
            glBindVertexArray(self.vao)
        else:
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        if adj is not None:
            glUniform1i(self.u_locs['adjust'], 0)
        if mask is not None:
            glUniform1i(self.u_locs['masked'], 0)
        if mode != 'normal':
            _normal_blend()
            glUniform1i(self.u_locs['premultiply'], 0)

    LAYER_WARP_CACHE = 16

    def _layer_warp_geometry(self, mesh):
        """(vao, vbo, vertex count) of a layer warp's surface, kept per shape
        (a few at a time), or None for no warp."""
        if mesh is None:
            return None
        cache = self.__dict__.setdefault('_warp_geo', collections.OrderedDict())
        geo = cache.get(mesh)
        if geo is not None:
            cache.move_to_end(mesh)
            return geo
        data = layer_warp.gl_vertices(mesh)
        vao, vbo = int(glGenVertexArrays(1)), int(glGenBuffers(1))
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
        glBindVertexArray(self.vao)
        cache[mesh] = geo = (vao, vbo, int(len(data)))
        while len(cache) > self.LAYER_WARP_CACHE:
            self._delete_warp_geo(cache.popitem(last=False)[1])
        return geo

    @staticmethod
    def _delete_warp_geo(geo):
        try:
            glDeleteVertexArrays(1, [geo[0]])
            glDeleteBuffers(1, [geo[1]])
        except Exception:  # noqa: BLE001 - the context may be gone
            pass

    def _clear_warp_geo(self):
        for geo in self.__dict__.pop('_warp_geo', {}).values():
            self._delete_warp_geo(geo)

    # --- Frame rendering ------------------------------------------------------

    def render(self, layers, target_monitor=None, transition_state=None, all_assigned_screens=None):
        self._drop_stale_static_cache()
        self._fx_seen = set()                 # effect buffers used by this frame (_free_unused_fx)
        transitioning = bool(transition_state and transition_state.get('active'))
        layers_to_draw = []
        mix_weights = None
        if transitioning:
            fa, ta = transition_alphas(transition_state)
            # Both cues have content: draw each into its own buffer at full
            # strength and mix (constant brightness). Fading from or to black:
            # one group, drawn with its fade alpha as before.
            if transition_state['from'] and transition_state['to'] and self._ensure_mix_program():
                mix_weights = (fa, ta)
            for group, source, a in (('from', transition_state['from'], fa),
                                     ('to', transition_state['to'], ta)):
                for layer in source.values():
                    layer_copy = layer.copy()
                    layer_copy['alpha'] = (1.0 if mix_weights else a) * layer_opacity(layer)
                    layer_copy['transition_group'] = group
                    layers_to_draw.append(layer_copy)
        elif layers:
            for layer in layers.values():
                layer_copy = layer.copy()
                layer_copy['alpha'] = layer_opacity(layer)
                layers_to_draw.append(layer_copy)

        n_mon = len(self.monitors)
        valid = lambda s: s is not None and 0 <= s < n_mon
        # Group layers by target screen. Screen indices that do not exist on
        # this machine (e.g. a show file from a venue with more outputs) are
        # ignored instead of raising IndexError every frame.
        layers_by_screen = {}
        all_screens = {s for s in (all_assigned_screens or ()) if valid(s)}
        for layer in layers_to_draw:
            screens = [target_monitor] if target_monitor is not None else layer.get('screens', [])
            for screen_idx in screens:
                if not valid(screen_idx):
                    continue
                layers_by_screen.setdefault(screen_idx, []).append(layer)
                all_screens.add(screen_idx)
        if target_monitor is not None and valid(target_monitor):
            all_screens.add(target_monitor)
        if not all_screens:
            self.close_window()
            return

        min_x = min(self.monitors[s].x for s in all_screens)
        min_y = min(self.monitors[s].y for s in all_screens)
        max_r = max(self.monitors[s].x + self.monitors[s].width for s in all_screens)
        max_b = max(self.monitors[s].y + self.monitors[s].height for s in all_screens)
        total_w, total_h = max_r - min_x, max_b - min_y
        if not self._create_window((min_x, min_y, total_w, total_h), is_spanning=True):
            return

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClearColor(0, 0, 0, 1)
        glViewport(0, 0, total_w, total_h)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

        # Upload each texture once per frame
        texture_ids = {}
        for screen_idx in all_screens:
            for layer in layers_by_screen.get(screen_idx, []):
                if not layer.get('path'):
                    continue
                key = self.source_key(layer)
                if key in texture_ids:
                    continue
                if key in self.active_players:
                    texture_ids[key] = self._update_texture_yuv(key)
                elif not live_source.is_live(layer['path']):
                    texture_ids[key] = self._update_texture_pbo(key)

        # Image-only cues in single-output mode are composited once into an FBO
        pre_rendered_data = None
        if (not transitioning and target_monitor is not None and target_monitor not in self.output_regions
                and self.is_cue_static(layers)
                and all(layer_opacity(l) >= 1.0 and layer_blend(l) == 'normal' and layer_color(l) is None
                        and layer_mask.layer_mask(l) is None and not layer_effects(l)
                        and layer_warp.layer_warp(l) is None for l in layers.values())):
            key = self.get_cue_cache_key(layers, target_monitor)
            pre_rendered_data = self.static_cue_cache.get(key)
            if pre_rendered_data is not None:
                self.static_cue_cache_hits += 1
            else:
                pre_rendered_data = self.pre_render_static_cue(layers, target_monitor)
                if pre_rendered_data:
                    self.static_cue_cache[key] = pre_rendered_data
                    self.static_cue_cache_misses += 1
                    if len(self.static_cue_cache) > 10:
                        oldest = next(iter(self.static_cue_cache))
                        self.cleanup_pre_rendered_cue(self.static_cue_cache.pop(oldest))
                glUseProgram(self.shader)
                glBindVertexArray(self.vao)

        for screen_idx in sorted(all_screens):
            glUseProgram(self.shader)             # each display starts from the layer program and quad
            glBindVertexArray(self.vao)
            mon = self.monitors[screen_idx]
            vp_x = mon.x - min_x
            vp_y = total_h - (mon.y - min_y) - mon.height  # OpenGL Y is bottom-up
            corners = self.output_warps.get(screen_idx)
            blend = self.output_blends.get(screen_idx)
            mesh = self.output_meshes.get(screen_idx)
            region = self.output_regions.get(screen_idx)
            cw, ch = region[1] if region is not None else (mon.width, mon.height)   # layout size
            out_fx = self.output_effects.get(screen_idx)
            pin = corners
            if corners is None and (blend is not None or mesh is not None or region is not None
                                    or out_fx is not None):
                corners = output_warp.IDENTITY
            target = None
            if corners is not None and self._ensure_warp_program():
                target = self._warp_target(screen_idx, cw, ch)
            # With a display effect the picture keeps its coverage in alpha (0 where nothing is drawn).
            clear_a = 0.0 if out_fx is not None else 1.0
            if target is not None:            # draw the picture off screen, then pin it
                glBindFramebuffer(GL_FRAMEBUFFER, target[0])
                glViewport(0, 0, cw, ch)
                glClearColor(0, 0, 0, clear_a)
                glClear(GL_COLOR_BUFFER_BIT)
            else:
                glViewport(vp_x, vp_y, mon.width, mon.height)
            self._set_projection(self._ortho(0, cw, 0, ch, -1, 1))
            mixed = False
            if mix_weights is not None:
                a = self._mix_target((screen_idx, 'from'), cw, ch)
                b = self._mix_target((screen_idx, 'to'), cw, ch)
                if a is not None and b is not None:
                    final_fb = target[0] if target is not None else 0
                    final_vp = (0, 0) if target is not None else (vp_x, vp_y)
                    for group, buf in (('from', a), ('to', b)):
                        glBindFramebuffer(GL_FRAMEBUFFER, buf[0])
                        glViewport(0, 0, cw, ch)
                        glClearColor(0, 0, 0, clear_a)
                        glClear(GL_COLOR_BUFFER_BIT)
                        for layer in sorted((l for l in layers_by_screen.get(screen_idx, [])
                                             if l.get('transition_group') == group), key=lambda l: l.get('order', 0)):
                            key = self.source_key(layer) if layer.get('path') else None
                            if key and texture_ids.get(key):
                                self._draw_layer(key, layer, (cw, ch))
                    glBindFramebuffer(GL_FRAMEBUFFER, final_fb)
                    glViewport(final_vp[0], final_vp[1], cw, ch)
                    self._draw_mix(a[1], b[1], mix_weights, (final_vp[0], final_vp[1], cw, ch))
                    mixed = True
            if mixed:
                pass
            elif pre_rendered_data and target_monitor == screen_idx:
                self.render_pre_rendered_cue(pre_rendered_data, (mon.width, mon.height))
            else:
                # Strictly in layer order: lower 'order' is drawn first (underneath).
                for layer in sorted(layers_by_screen.get(screen_idx, []), key=lambda l: l.get('order', 0)):
                    key = self.source_key(layer) if layer.get('path') else None
                    if key and texture_ids.get(key):
                        if mix_weights is not None:          # mixing buffers unavailable: fade alphas instead
                            w = mix_weights[0] if layer.get('transition_group') == 'from' else mix_weights[1]
                            layer = dict(layer, alpha=layer.get('alpha', 1.0) * w)
                        self._draw_layer(key, layer, (cw, ch))
            if target is not None:
                picture = target[1]
                if out_fx is not None:            # the display's effect on its whole picture
                    fx_tex = self._apply_effects(('display', screen_idx), out_fx,
                                                 {'id': target[1], 'width': cw, 'height': ch}, None, display=True)
                    if fx_tex is not None:
                        picture = fx_tex
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                glViewport(vp_x, vp_y, mon.width, mon.height)
                geo = self._mesh_geometry(screen_idx, mesh, pin) if mesh is not None else None
                if geo is not None:
                    self._draw_meshed(picture, geo, (vp_x, vp_y, mon.width, mon.height), blend, region)
                else:
                    self._draw_warped(picture, corners, (vp_x, vp_y, mon.width, mon.height), blend, region)

        if self._preview_request is not None:
            self._capture_preview(self._preview_request, all_screens, min_x, min_y, total_h)
        self._free_unused_fx()
        pygame.display.flip()

    # --- Preview capture --------------------------------------------------------

    def request_preview_capture(self, monitor_idx):
        """Ask the next render() to capture `monitor_idx` for the preview monitor."""
        self._preview_request = monitor_idx
        self.last_preview_image = None

    def take_preview_image(self):
        img, self.last_preview_image = self.last_preview_image, None
        self._preview_request = None
        return img

    def _capture_preview(self, monitor_idx, all_screens, min_x, min_y, total_h):
        """Downscale the monitor's region of the back buffer on the GPU and read
        back the small image, before the swap (front-buffer contents after a
        swap are undefined, which is why earlier GL_FRONT readbacks were black)."""
        self._preview_request = None
        if monitor_idx not in all_screens:
            return
        mon = self.monitors[monitor_idx]
        scale = min(PREVIEW_MAX_SIZE[0] / mon.width, PREVIEW_MAX_SIZE[1] / mon.height, 1.0)
        pw, ph = max(1, int(mon.width * scale)), max(1, int(mon.height * scale))
        if self._preview_fbo is None or self._preview_fbo_size != (pw, ph):
            self._delete_preview_target()
            self._preview_tex = self._new_texture(pw, ph, GL_RGBA8, GL_RGBA)
            self._preview_fbo = int(glGenFramebuffers(1))
            glBindFramebuffer(GL_FRAMEBUFFER, self._preview_fbo)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._preview_tex, 0)
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                self._delete_preview_target()
                log.warning("Preview FBO incomplete; GPU preview disabled for this frame")
                return
            self._preview_fbo_size = (pw, ph)
        vp_x = mon.x - min_x
        vp_y = total_h - (mon.y - min_y) - mon.height
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        glReadBuffer(GL_BACK)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._preview_fbo)
        glBlitFramebuffer(vp_x, vp_y, vp_x + mon.width, vp_y + mon.height, 0, 0, pw, ph,
                          GL_COLOR_BUFFER_BIT, GL_LINEAR)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._preview_fbo)
        pixels = glReadPixels(0, 0, pw, ph, GL_RGBA, GL_UNSIGNED_BYTE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        img = Image.frombytes("RGBA", (pw, ph), pixels).transpose(Image.FLIP_TOP_BOTTOM)
        self.last_preview_image = img.convert('RGB').convert('RGBA')

    def _delete_preview_target(self):
        if self.window:
            try:
                if self._preview_fbo:
                    glDeleteFramebuffers(1, [self._preview_fbo])
                if self._preview_tex:
                    glDeleteTextures(1, [self._preview_tex])
            except Exception:
                pass
        self._preview_fbo = self._preview_tex = self._preview_fbo_size = None

    # --- Resource lifetime ------------------------------------------------------

    def release_texture(self, path):
        """Cleanup texture and associated PBOs."""
        if path in self.yuv_textures:
            self._delete_video_textures(path)
            return
        if path in self.textures:
            self._fx_forget(path)
            tex_info = self.textures.pop(path)
            if self.window:
                try:
                    glDeleteTextures(1, [tex_info['id']])
                    if path in self.pbo_pool:
                        pbos = self.pbo_pool.pop(path)
                        glDeleteBuffers(len(pbos), pbos)
                except Exception as e:
                    log.debug(f"release_texture({os.path.basename(path)}): {e}")
            self.pbo_pool.pop(path, None)
            self.pbo_index.pop(path, None)

    def _release_gl_resources(self):
        """Delete all GL objects of the current context and forget their ids."""
        try:
            for data in self.static_cue_cache.values():
                self.cleanup_pre_rendered_cue(data)
            for path in list(self.yuv_textures):
                self._delete_video_textures(path)
            ids = [t['id'] for t in self.textures.values()]
            if ids:
                glDeleteTextures(len(ids), ids)
            for pbos in self.pbo_pool.values():
                glDeleteBuffers(len(pbos), pbos)
            self._delete_preview_target()
            self._delete_warp_targets()
            self._delete_mesh_geometry()
            self._delete_mix_targets()
            self._delete_fx()
            self._clear_warp_geo()
            if self._mesh_prog:
                glDeleteProgram(self._mesh_prog)
            if self._mix_prog:
                glDeleteProgram(self._mix_prog)
            if self._warp_prog:
                glDeleteProgram(self._warp_prog)
            if self._warp_vao:
                glDeleteVertexArrays(1, [self._warp_vao])
            if self._warp_vbo:
                glDeleteBuffers(1, [self._warp_vbo])
            if self.shader:
                glDeleteProgram(self.shader)
            if self.vao:
                glDeleteVertexArrays(1, [self.vao])
            if self.vbo:
                glDeleteBuffers(1, [self.vbo])
            if self.ebo:
                glDeleteBuffers(1, [self.ebo])
        except Exception as e:
            log.debug(f"GL resource release incomplete: {e}")
        # Ids belong to the context being destroyed; never reuse them.
        self.shader = self.vao = self.vbo = self.ebo = None
        self.static_cue_cache.clear()
        self.textures.clear()
        self.yuv_textures.clear()
        self.u_locs.clear()
        self.pbo_pool.clear()
        self.pbo_index.clear()
        self.texture_formats.clear()
        self._preview_fbo = self._preview_tex = self._preview_fbo_size = None
        self._warp_targets = {}
        self._warp_prog = self._warp_vao = self._warp_vbo = None
        self._warp_locs = {}
        self._mesh_geo = {}
        self._mesh_prog = None
        self._mesh_locs = {}
        self._mix_targets = {}
        self._mix_prog = None
        self._mix_locs = {}
        self._fx_reset()

    # --- Corner pin -----------------------------------------------------------------

    def _on_warps_changed(self):
        if self.window:                                   # called on the render thread
            for idx in [i for i in self._warp_targets if i not in self.output_warps and i not in self.output_blends
                        and i not in self.output_meshes and i not in self.output_regions
                        and i not in self.output_effects]:
                self._delete_warp_targets(idx)
            shown = {('display', i) for i in self.output_effects}
            for key in {k for k, _slot in self._fx_targets if isinstance(k, tuple) and k[0] == 'display'}:
                if key not in shown:
                    self._fx_forget(key)
            for idx in [i for i in self._mesh_geo if i not in self.output_meshes]:
                self._delete_mesh_geometry(idx)

    def _delete_warp_targets(self, only=None):
        for idx in ([only] if only is not None else list(self._warp_targets)):
            t = self._warp_targets.pop(idx, None)
            if t is None:
                continue
            try:
                glDeleteFramebuffers(1, [t[0]])
                glDeleteTextures(1, [t[1]])
            except Exception:  # noqa: BLE001
                pass

    def _delete_mix_targets(self):
        for t in self._mix_targets.values():
            try:
                glDeleteFramebuffers(1, [t[0]])
                glDeleteTextures(1, [t[1]])
            except Exception:  # noqa: BLE001
                pass
        self._mix_targets = {}

    def _mix_target(self, key, w, h):
        t = self._mix_targets.get(key)
        if t is not None and t[2:] == (w, h):
            return t
        if t is not None:
            try:
                glDeleteFramebuffers(1, [t[0]])
                glDeleteTextures(1, [t[1]])
            except Exception:  # noqa: BLE001
                pass
            del self._mix_targets[key]
        tex = self._new_texture(w, h, GL_RGBA8, GL_RGBA)
        fbo = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if not ok:
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [tex])
            return None
        self._mix_targets[key] = t = (fbo, tex, w, h)
        return t

    def _ensure_mix_program(self):
        if self._mix_prog is not None:
            return True
        if not self._ensure_warp_program():          # provides the full-viewport quad
            return False
        try:
            self._mix_prog = shaders.compileProgram(
                shaders.compileShader(WARP_VERTEX_SHADER, GL_VERTEX_SHADER),
                shaders.compileShader(MIX_FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
        except Exception:  # noqa: BLE001
            log.exception("Crossfade shader failed to compile; fades draw one cue over the other")
            self._mix_prog = None
            return False
        self._mix_locs = {n: glGetUniformLocation(self._mix_prog, n)
                          for n in ('outgoing', 'incoming', 'weights', 'viewport')}
        return True

    def _draw_mix(self, tex_from, tex_to, weights, viewport):
        glUseProgram(self._mix_prog)
        glDisable(GL_BLEND)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_from)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex_to)
        glUniform1i(self._mix_locs['outgoing'], 0)
        glUniform1i(self._mix_locs['incoming'], 1)
        glUniform2f(self._mix_locs['weights'], float(weights[0]), float(weights[1]))
        glUniform4f(self._mix_locs['viewport'], *[float(v) for v in viewport])
        glBindVertexArray(self._warp_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_BLEND)
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

    def _warp_target(self, idx, w, h):
        t = self._warp_targets.get(idx)
        if t is not None and t[2:] == (w, h):
            return t
        self._delete_warp_targets(idx)
        tex = self._new_texture(w, h, GL_RGBA8, GL_RGBA)
        fbo = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if not ok:
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [tex])
            log.error(f"Corner pin: framebuffer for display {idx} incomplete; drawing it unwarped")
            return None
        self._warp_targets[idx] = t = (fbo, tex, w, h)
        return t

    def _ensure_warp_program(self):
        if self._warp_prog is not None:
            return True
        try:
            self._warp_prog = shaders.compileProgram(
                shaders.compileShader(WARP_VERTEX_SHADER, GL_VERTEX_SHADER),
                shaders.compileShader(WARP_FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
        except Exception:  # noqa: BLE001
            log.exception("Corner pin shader failed to compile; outputs are drawn unwarped")
            self._warp_prog = None
            return False
        self._warp_locs = {n: glGetUniformLocation(self._warp_prog, n)
                           for n in ('picture', 'hinv', 'viewport', 'edges', 'blendShape', 'region')}
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self._warp_vao = int(glGenVertexArrays(1))
        self._warp_vbo = int(glGenBuffers(1))
        glBindVertexArray(self._warp_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._warp_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        # Back to the layer quad: this runs in the middle of a frame (the first
        # time a display has a pin, blend, mesh or span) and the layers are drawn
        # next. Leaving VAO 0 bound made that draw read indices from address 0:
        # a crash here (Mesa) after the output had been on one display.
        glBindVertexArray(self.vao if self.vao else 0)
        return True

    def _draw_warped(self, tex, corners, viewport, blend=None, region=None):
        """Draw the monitor's picture (tex) into the bound default framebuffer's
        viewport through the corner pin and the edge blend."""
        glUseProgram(self._warp_prog)
        b = blend or (0.0, 0.0, 0.0, 0.0, edge_blend.DEFAULT_CURVE, edge_blend.DEFAULT_GAMMA)
        glUniform4f(self._warp_locs['edges'], *[float(v) for v in b[:4]])
        glUniform2f(self._warp_locs['blendShape'], float(b[4]), float(b[5]))
        glDisable(GL_BLEND)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glUniform1i(self._warp_locs['picture'], 0)
        glUniformMatrix3fv(self._warp_locs['hinv'], 1, GL_TRUE, output_warp.inverse(corners).astype(np.float32))
        glUniform4f(self._warp_locs['viewport'], *[float(v) for v in viewport])
        glUniform4f(self._warp_locs['region'], *[float(v) for v in self._region_uniform(region)])
        glBindVertexArray(self._warp_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glEnable(GL_BLEND)
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

    # --- Mesh warp ------------------------------------------------------------------

    def _ensure_mesh_program(self):
        if self._mesh_prog is not None:
            return True
        try:
            self._mesh_prog = shaders.compileProgram(
                shaders.compileShader(MESH_VERTEX_SHADER, GL_VERTEX_SHADER),
                shaders.compileShader(MESH_FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
        except Exception:  # noqa: BLE001
            log.exception("Mesh warp shader failed to compile; outputs are drawn without their mesh")
            self._mesh_prog = None
            return False
        self._mesh_locs = {n: glGetUniformLocation(self._mesh_prog, n)
                           for n in ('picture', 'viewport', 'edges', 'blendShape', 'region')}
        return True

    def _delete_mesh_geometry(self, only=None):
        for idx in ([only] if only is not None else list(self._mesh_geo)):
            g = self._mesh_geo.pop(idx, None)
            if g is None or g[1] is None:
                continue
            try:
                glDeleteVertexArrays(1, [g[1][0]])
                glDeleteBuffers(1, [g[1][1]])
            except Exception as e:  # noqa: BLE001
                log.debug(f"Mesh geometry release incomplete: {e}")

    def _mesh_geometry(self, idx, mesh, pin):
        """(vao, vbo, vertex count) of display idx's mesh through its pin, built
        when either changes; None when it cannot be drawn (the pin alone is)."""
        key = (mesh, pin)
        g = self._mesh_geo.get(idx)
        if g is not None and g[0] == key:
            return g[1]
        self._delete_mesh_geometry(idx)
        geo = None
        if not self._ensure_mesh_program():
            pass
        elif not mesh_warp.through_pin_ok(mesh, pin):
            log.warning(f"Mesh warp on display {idx} does not fit through its corner pin; drawing the pin alone")
        else:
            data = mesh_warp.gl_vertices(mesh, pin)
            vao = int(glGenVertexArrays(1))
            vbo = int(glGenBuffers(1))
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
            glBindVertexArray(self.vao)
            geo = (vao, vbo, int(len(data)))
        self._mesh_geo[idx] = (key, geo)
        return geo

    def _draw_meshed(self, tex, geo, viewport, blend=None, region=None):
        """Draw the monitor's picture (tex) into the bound default framebuffer's
        viewport on its mesh (already through the pin), with the edge blend.
        The window was cleared to black, so the area outside the mesh is black."""
        glUseProgram(self._mesh_prog)
        b = blend or (0.0, 0.0, 0.0, 0.0, edge_blend.DEFAULT_CURVE, edge_blend.DEFAULT_GAMMA)
        glUniform4f(self._mesh_locs['edges'], *[float(v) for v in b[:4]])
        glUniform2f(self._mesh_locs['blendShape'], float(b[4]), float(b[5]))
        glDisable(GL_BLEND)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glUniform1i(self._mesh_locs['picture'], 0)
        glUniform4f(self._mesh_locs['viewport'], *[float(v) for v in viewport])
        glUniform4f(self._mesh_locs['region'], *[float(v) for v in self._region_uniform(region)])
        glBindVertexArray(geo[0])
        glDrawArrays(GL_TRIANGLES, 0, geo[2])
        glEnable(GL_BLEND)
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

    def close_window(self):
        if self.window:
            self._release_gl_resources()
            self.spanning_window_details = None
        super().close_window()

    def clear(self):
        super().clear()
        if self.window:
            self.close_window()


class PygameRenderEngine(BaseRenderEngine):
    # Balanced Mode's decoders hand over BGRA bytes at the drawn size with the video's own
    # size; Compatibility Mode's hand over arrays that carry it (SizedFrame).
    decodes_at_drawn_size = True

    def __init__(self, media_manager, mode='cpu', hw_accel=None):
        super().__init__(media_manager, mode, hw_accel)
        log.info(f"PygameRenderEngine initialized in '{mode}' mode.")
        if self.mode == 'pygame_gpu':
            log.info(f"-> GPU Accel: {hw_accel}")
        self.spanning_window_details = None
        
        # Add transition surface caching with separate keys for from/to
        self.transition_cache = {
            'from_surface': None,
            'to_surface': None,
            'from_cache_key': None,
            'to_cache_key': None
        }
        
        # Surface cache for transformed media
        self.surface_cache = OrderedDict()  # Cache for transformed surfaces
        self.surface_cache_size = 50  # Limit cache size
        # Mesh warp: per monitor (key, (map_x, map_y) or None) and the thread
        # building the next maps (see _mesh_maps_for)
        self._mesh_maps = {}
        self._mesh_jobs = {}
        # Corner pin alone: per monitor (key, fixed-point remap tables), built in
        # the background (see _pin_maps_for)
        self._pin_maps = {}
        self._pin_jobs = {}
        self.cache_hits = 0  # For debugging
        self.cache_misses = 0  # For debugging

    def _init_pygame_and_gl(self):
        import os, sys, pygame
        sdl_audio_safe.init_pygame()       # without pygame.mixer (sound has its own output)
        pygame.display.init()

        # Ensure the output window uses the same icon as the main GUI (.ico file)
        try:
            from PIL import Image
            base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ico_path = os.path.join(base_dir, 'src', 'visualdeck.ico')
            if os.path.exists(ico_path):
                im = Image.open(ico_path)
                # Pick largest frame in the .ico for best fidelity
                if getattr(im, "n_frames", 1) > 1:
                    best = None; best_area = -1
                    for i in range(im.n_frames):
                        im.seek(i)
                        area = im.size[0] * im.size[1]
                        if area > best_area:
                            best = im.copy()
                            best_area = area
                    im = best
                if im.mode != "RGBA":
                    im = im.convert("RGBA")
                surf = pygame.image.frombuffer(im.tobytes(), im.size, "RGBA")
                pygame.display.set_icon(surf)
        except Exception:
            # Best-effort; never fail initialization over an icon
            pass


    def pre_render_static_cue(self, layers, dims):
        """Pre-render a static cue to a surface."""
        if not layers:
            return None
        
        # Use the existing _render_layers_to_surface method
        surface = self._render_layers_to_surface(layers, dims)
        
        if surface:
            # Convert the surface for optimal blitting
            surface = surface.convert_alpha()
        
        return surface
        
    def clear(self):
        """Clear all resources including transition cache and surface cache."""
        # Clear transition cache
        self.transition_cache = {
            'from_surface': None,
            'to_surface': None,
            'from_cache_key': None,
            'to_cache_key': None
        }
        # Clear surface cache
        self.surface_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        # Call parent class clear method
        super().clear()

    def _create_window(self, geom):
        x, y, w, h = geom
        if self.window and self.window.get_size() == (w, h) and self.spanning_window_details == geom:
            return True

        # Clear transition cache on window change
        self.transition_cache = {
            'from_surface': None,
            'to_surface': None,
            'from_cache_key': None,
            'to_cache_key': None
        }

        self.close_window()
        
        # Prevent implicit centering and request explicit position
        os.environ['SDL_VIDEO_CENTERED'] = '0'
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
        
        # Use only safe, standard flags
        flags = NOFRAME | DOUBLEBUF
        
        # Try to use a hardware surface, but fall back safely if not supported
        try:
            self.window = pygame.display.set_mode((w, h), flags | HWSURFACE)
        except pygame.error:
            self.window = pygame.display.set_mode((w, h), flags)

        pygame.display.set_caption("VisualDeck Output (Pygame)")
        self.spanning_window_details = geom

        # Force the position after creation (some drivers ignore WINDOW_POS on first create)
        try:
            import pygame._sdl2 as sdl2
            win = sdl2.Window.from_display_module()
            win.position = (int(x), int(y))
        except Exception:
            pass

        return True

    def _get_transition_cache_key(self, layers, dims):
        """Generate a cache key for transition surfaces based on layers and dimensions."""
        # Create a unique key based on layer paths and dimensions
        layer_info = []
        for layer_id, layer in sorted(layers.items()):
            if layer.get('path'):
                layer_info.append((layer_id, layer['path'], layer.get('order', 0)))
        return (tuple(layer_info), dims)

    def _render_layers_to_surface(self, layers, dims):
        w, h = dims
        
        if not layers:
            log.debug(f"No layers to render")
            return None

        # Ensure layers is a dictionary before calling .values()
        if isinstance(layers, list):
            # This is a quick fix for the error.
            # A better solution would be to ensure
            # the data structure is consistent throughout the application.
            # Assuming the list contains layer dictionaries.
            layers_dict = {i: layer for i, layer in enumerate(layers)}
        else:
            layers_dict = layers

        surf = pygame.Surface((w, h), SRCALPHA)
        surf.fill((0, 0, 0, 0))
        
        drawn_anything = False
        layers_rendered = 0
        for m in sorted(layers_dict.values(), key=lambda x: x.get('order', 0)):
            sf = self._to_surface(m)
            if not sf: 
                log.debug(f"No surface returned for layer with path: {m.get('path', 'Unknown')}")
                continue
            
            drawn_anything = True
            layers_rendered += 1
            pos = m.get('position', {})
            x = int(pos.get('x', 0.5) * w)
            y = int(pos.get('y', 0.5) * h)
            rect = sf.get_rect(center=(x, y))
            op = layer_opacity(m)
            if op <= 0.0:
                continue
            if op < 1.0:
                sf = sf.copy()            # surfaces may be cached: never change the cached one
                sf.set_alpha(int(op * 255 + 0.5))
                surf.blit(sf, rect)
            elif layers_rendered == 1 and sf.get_flags() & SRCALPHA:
                # The first layer on the empty (all-zero) canvas: the per-channel
                # maximum gives the same pixels as alpha blending onto it (checked
                # bit for bit, every alpha value) at less than half the cost: 1.9
                # against 4.5 ms for a 1080p layer on the test VM (2026-11-17).
                surf.blit(sf, rect, special_flags=pygame.BLEND_RGBA_MAX)
            else:
                surf.blit(sf, rect)
            log.debug(f"Blitted surface at ({x}, {y}), size: {sf.get_size()}")

        log.debug(f"Rendered {layers_rendered} layers, drawn_anything: {drawn_anything}")
        return surf if drawn_anything else None

    def window_preview_image(self, max_w, max_h):
        """A small copy of what the output window shows now (an RGB PIL image
        fitting max_w x max_h), or None when there is no window or its pixels
        cannot be read. It is a resize of the window's own memory, about 0.5 ms
        for 1080p on the test VM, where composing the layers again at the
        preview's size took 4-17 ms a preview on the owner's PC (Output Health,
        Balanced Mode, 2026-11-16) and pushed the render loop past its frame
        budget. It shows exactly what the output shows (corner pin, mesh,
        blend, a fade, Blackout, a test pattern), as Performance Mode's GPU
        preview does."""
        win = self.window
        if win is None:
            return None
        buf = view = None
        try:
            w, h = win.get_size()
            if w < 2 or h < 2 or win.get_bitsize() != 32 or tuple(win.get_masks())[:3] != (0xFF0000, 0xFF00, 0xFF):
                return None
            scale = min(max_w / w, max_h / h, 1.0)
            pw, ph = max(1, int(w * scale)), max(1, int(h * scale))
            buf = np.frombuffer(win.get_buffer(), dtype=np.uint8)
            view = buf.reshape(h, win.get_pitch())[:, :w * 4].reshape(h, w, 4)
            # Halve with an exact 2x box filter while the preview is at most half as big
            # (OpenCV's fast path), then the last step bilinear: 0.5 ms for 1080p to
            # 640x360 on the test VM, against 2.8 ms for one INTER_AREA resize.
            small, cw, ch = view, w, h
            while pw * 2 <= cw and ph * 2 <= ch:
                cw, ch = cw // 2, ch // 2
                small = cv2.resize(small, (cw, ch), interpolation=cv2.INTER_AREA)
            if (cw, ch) != (pw, ph):
                small = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_LINEAR)
            return Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGRA2RGB))   # a new array
        except Exception:  # noqa: BLE001 - the caller composes the preview instead
            log.debug("Window preview not available; composing it", exc_info=True)
            return None
        finally:
            del view, buf                     # unlocks the window surface

    def render_layers_to_image(self, layers, dimensions):
        # FIX: Ensure a display mode is set before performing surface operations.
        needs_display_init = not pygame.display.get_active()
        if needs_display_init:
            # If no display is active (e.g., generating a CPU preview in a thread),
            # create a temporary 1x1 pixel display to provide the necessary context.
            pygame.display.set_mode((1, 1), NOFRAME)

        pygame_surface = self._render_layers_to_surface(layers, dimensions)
        
        pil_image = None
        if pygame_surface:
            try:
                image_data = pygame.image.tostring(pygame_surface, "RGBA", False)
                pil_image = Image.frombytes("RGBA", dimensions, image_data)
            except pygame.error as e:
                log.error(f"Failed to convert Pygame surface to Image for preview: {e}")
                pil_image = None

        # We don't quit the display here, as the main application thread will handle it.
        # The check for get_active() prevents this from running unnecessarily.

        return pil_image

    def _to_surface(self, media):
        path = media.get('path')
        live = live_source.is_live(path)
        if not path or not (live or os.path.exists(path)):
            return None

        ext = '.live' if live else os.path.splitext(path)[1].lower()
        streamed = live or ext in VIDEO_EXTS

        # Generate cache key including all transformations
        pos = media.get('position', {})
        key = self.source_key(media)
        cache_key = (
            key,
            pos.get('scale_x', 1.0),
            pos.get('scale_y', 1.0),
            pos.get('rotation', 0),
            id(self.video_frame_cache.get(key)) if streamed else None,
            layer_color(media),
            layer_mask.layer_mask(media),
            layer_warp.layer_warp(media),
        )

        # Only cache non-video surfaces
        if not streamed and cache_key in self.surface_cache:
            self.surface_cache.move_to_end(cache_key)
            self.cache_hits += 1
            return self.surface_cache[cache_key]

        self.cache_misses += 1

        # Original surface creation logic
        frame_data = self._get_latest_video_frame(key) if streamed else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if frame_data is None: 
            return None

        own = None                                # the video's own size when decoded smaller
        if streamed and isinstance(frame_data, tuple):
            # pygame_gpu mode: (BGRA bytes, (w, h)[, video's own (w, h)]) from the decoder
            frame_bytes, dims = frame_data[0], frame_data[1]
            arr = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(dims[1], dims[0], 4)
            own = frame_data[2] if len(frame_data) > 2 else None
        else:
            # CPU mode video frames and still images
            arr = frame_data
            own = getattr(frame_data, 'own_size', None)       # decoded smaller (SizedFrame)
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
            elif arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
        h0, w0 = arr.shape[:2]
        scale_x = pos.get('scale_x', 1.0)
        scale_y = pos.get('scale_y', 1.0)
        bw, bh = own or (w0, h0)                  # a layer's scale is relative to the video's own size
        new_w, new_h = int(bw * scale_x), int(bh * scale_y)
        if new_w <= 0 or new_h <= 0:
            return None
        adj, mask = layer_color(media), layer_mask.layer_mask(media)
        # Scale with OpenCV (about 9x faster than pygame's smoothscale for a 4K
        # frame on the test VM), and do color and mask at the smaller of the
        # two sizes: a 4K clip shown at half size costs a quarter.
        shrink = new_w * new_h < w0 * h0
        if shrink:
            arr = scale_bgra(arr, new_w, new_h)
        arr = layer_mask.apply(adjust_colors(arr, adj), mask)
        if not shrink and (new_w, new_h) != (w0, h0):
            arr = scale_bgra(arr, new_w, new_h)
        warp = layer_warp.layer_warp(media)
        if warp is not None:                      # bent around the same center (layer_warp)
            arr = layer_warp.warp_array(np.ascontiguousarray(arr), warp)
            new_h, new_w = arr.shape[:2]
        if isinstance(frame_data, tuple) and (new_w, new_h) == (w0, h0) and adj is None and mask is None \
                and warp is None:
            sf = pygame.image.frombuffer(frame_data[0], (w0, h0), "BGRA")      # untouched decoder frame: no copy
        else:
            # The array itself, not a bytes copy of it (0.6 ms per 1080p frame): the
            # surface keeps a reference to it.
            sf = pygame.image.frombuffer(np.ascontiguousarray(arr), (new_w, new_h), "BGRA")
        if not streamed:
            sf = sf.convert_alpha()

        if pos.get('rotation', 0):
            sf = pygame.transform.rotate(sf, pos.get('rotation', 0))

        # LRU cache for non-video surfaces
        if not streamed:
            self.surface_cache[cache_key] = sf
            self.surface_cache.move_to_end(cache_key)
            if len(self.surface_cache) > self.surface_cache_size:
                self.surface_cache.popitem(last=False)

        # Log cache performance occasionally
        if (self.cache_hits + self.cache_misses) % 100 == 0 and self.cache_hits > 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100
            log.debug(f"Surface cache hit rate: {hit_rate:.1f}%")

        return sf


    def get_transition_frame(self, from_surf, to_surf, progress, alphas=None):
        """Blend two cue frames for a transition (`alphas` = (outgoing,
        incoming) opacity, see transition_alphas; default a straight fade).

        When the two opacities add up to 1 (Fade, Smooth Fade) the incoming cue
        is drawn with its opacity over the outgoing one at full strength:
        out = from x (1 - a) + to x a, so a crossfade between full-screen cues
        keeps its brightness (drawing both faded dimmed the midpoint to about
        75 %). Where the incoming cue is transparent the outgoing one stays
        until the end. Dip to Black has one cue at a time."""
        if not from_surf and not to_surf:
            return None
        size = (from_surf or to_surf).get_size()
        fa, ta = alphas if alphas is not None else (1.0 - progress, progress)
        base = pygame.Surface(size, SRCALPHA)
        base.fill((0, 0, 0, 0))
        crossfade = from_surf is not None and to_surf is not None and fa > 0.0 and ta > 0.0 \
            and abs(fa + ta - 1.0) < 1e-6
        if from_surf and fa > 0.0:
            if crossfade or fa >= 1.0:
                base.blit(from_surf, (0, 0))
            else:
                faded = from_surf.copy()
                faded.set_alpha(int(fa * 255))
                base.blit(faded, (0, 0))
        if to_surf and ta > 0.0:
            if ta >= 1.0:
                base.blit(to_surf, (0, 0))
            else:
                faded = to_surf.copy()
                faded.set_alpha(int(ta * 255))
                base.blit(faded, (0, 0))
        return base
        
    def _on_warps_changed(self):
        for idx in [i for i in self._mesh_maps if i not in self.output_meshes]:
            self._mesh_maps.pop(idx, None)
        for idx in [i for i in self._pin_maps if i not in self.output_warps]:
            self._pin_maps.pop(idx, None)
        if self.output_regions != getattr(self, '_regions_seen', {}):
            self._regions_seen = dict(self.output_regions)
            self.static_cue_cache.clear()           # cached cues were laid out for the old sizes

    @staticmethod
    def _build_mesh_maps(mesh, corners, w, h):
        try:
            if not mesh_warp.through_pin_ok(mesh, corners):
                log.warning("Mesh warp does not fit through the corner pin; drawing the pin alone")
                return None
            return mesh_warp.inverse_map(mesh, corners, w, h)
        except Exception:  # noqa: BLE001
            log.exception("Mesh warp map failed; drawing the pin alone")
            return None

    def _mesh_job(self, idx, key):
        maps = self._build_mesh_maps(*key)
        if idx in self.output_meshes:
            self._mesh_maps[idx] = (key, maps)
            self.redraw_pending = True                   # a still cue is not drawn again otherwise

    def _mesh_maps_for(self, idx, mesh, corners, w, h):
        """The remap tables for display idx (None: draw the pin alone). The
        first ones are built here (about 0.1 s for 1080p, 0.45 s for 4K on the
        test VM); after a change the previous ones stay on the output while a
        background thread builds the new ones, so dragging a point does not
        stall playback."""
        key = (mesh, corners, w, h)
        have = self._mesh_maps.get(idx)
        if have is not None and have[0] == key:
            return have[1]
        if have is None:
            maps = self._build_mesh_maps(*key)
            self._mesh_maps[idx] = (key, maps)
            return maps
        job = self._mesh_jobs.get(idx)
        if job is None or not job.is_alive():
            job = threading.Thread(target=self._mesh_job, args=(idx, key), daemon=True, name=f"mesh-map-{idx}")
            self._mesh_jobs[idx] = job
            job.start()
        return have[1]

    def _pin_job(self, idx, key):
        corners, w, h = key
        try:
            maps = output_warp.inverse_maps(corners, w, h)
        except Exception:  # noqa: BLE001 - warpPerspective goes on drawing it
            log.exception("Corner pin tables failed; warping each frame instead")
            return
        if self.output_warps.get(idx) == corners:
            self._pin_maps[idx] = (key, maps)

    def _pin_maps_for(self, idx, corners, w, h):
        """Remap tables for display idx's corner pin, or None while they are being
        built: the frame is then warped with cv2.warpPerspective, which gives the
        same picture at about twice the cost (11.3 against 6.8 ms per 1080p frame
        on the test VM; building the tables takes about 60 ms there). So the
        output follows a dragged corner at once, and settles on the tables when
        the corner rests."""
        key = (corners, w, h)
        have = self._pin_maps.get(idx)
        if have is not None and have[0] == key:
            return have[1]
        job = self._pin_jobs.get(idx)
        if job is None or not job.is_alive():
            job = threading.Thread(target=self._pin_job, args=(idx, key), daemon=True, name=f"pin-map-{idx}")
            self._pin_jobs[idx] = job
            job.start()
        return None

    def _region_surface(self, surface, monitor_idx):
        """The display's part of a span canvas (a surface of the display's size);
        the surface itself when the display is not in a span."""
        region = self.output_regions.get(monitor_idx)
        if region is None or not (0 <= monitor_idx < len(self.monitors)):
            return surface
        (x0, y0, x1, y1), _ = region
        mon = self.monitors[monitor_idx]
        sw, sh = surface.get_size()
        rect = pygame.Rect(int(round(x0 * sw)), int(round(y0 * sh)), 0, 0)
        rect.width = max(1, min(sw - rect.x, int(round((x1 - x0) * sw))))
        rect.height = max(1, min(sh - rect.y, int(round((y1 - y0) * sh))))
        try:
            part = surface.subsurface(rect)
        except ValueError:
            return surface
        if part.get_size() != (mon.width, mon.height):
            return pygame.transform.smoothscale(part.convert_alpha() if part.get_bitsize() != 32 else part,
                                                (mon.width, mon.height))
        return part.copy()

    @staticmethod
    def _bgra_pixels(surface):
        """The surface's pixels as an (h, w, 4) uint8 array in BGRA order. For a
        surface that is already ARGB8888 (every composed frame: SRCALPHA) this is
        a view of the surface's own memory, with no conversion: pygame.image.tobytes
        took 7.6 ms per 1080p frame on the test VM, about half of a corner pin's
        cost. The view locks the surface until the array is released. Other
        formats are converted (a copy)."""
        w, h = surface.get_size()
        if surface.get_bitsize() == 32 and tuple(surface.get_masks()) == (0xFF0000, 0xFF00, 0xFF, 0xFF000000):
            pitch = surface.get_pitch()
            buf = np.frombuffer(surface.get_buffer(), dtype=np.uint8)
            return buf.reshape(h, pitch)[:, :w * 4].reshape(h, w, 4)
        return np.frombuffer(pygame.image.tobytes(surface, "BGRA"), dtype=np.uint8).reshape(h, w, 4)

    def _warp_surface(self, surface, monitor_idx):
        """The surface through the monitor's mesh, corner pin and edge blend (a
        new surface; cached surfaces are never changed). CPU cost per 1080p
        frame on the test VM (2026-11-16): a pin about 6 ms with its tables (9 ms
        while they are built), a blend about 2 ms."""
        corners = self.output_warps.get(monitor_idx)
        blend = self.output_blends.get(monitor_idx)
        mesh = self.output_meshes.get(monitor_idx)
        if surface is not None:
            surface = self._region_surface(surface, monitor_idx)
        if (corners is None and blend is None and mesh is None) or surface is None:
            return surface
        try:
            # BGRA: pygame's own byte order for a surface with alpha (ARGB8888), so the
            # result blits to the window on SDL's fast path. "RGBA" made a surface in
            # another order whose blit took about 25 ms per 1080p frame on the test VM
            # (2.7 ms for BGRA), and an edge blend in Balanced Mode took the 4K clip
            # from 50 to 15 renders/s on the owner's PC (2026-11-15). The warps and the
            # blend treat the three color channels alike, so the order does not matter.
            w, h = surface.get_size()
            arr = self._bgra_pixels(surface)
            maps = self._mesh_maps_for(monitor_idx, mesh, corners, w, h) if mesh is not None else None
            if maps is not None:
                out = mesh_warp.warp_image(arr, mesh, corners, maps)
            elif corners is not None:
                pin = self._pin_maps_for(monitor_idx, corners, w, h)
                out = output_warp.warp_with_maps(arr, pin) if pin is not None else output_warp.warp_image(arr, corners)
            else:
                out = arr
            if blend is not None:
                out = edge_blend.apply(np.ascontiguousarray(out), blend)
            if out is arr:
                out = arr.copy()      # never hand out the source surface's own memory
            del arr                   # unlocks the source surface (a view of it locks it)
            # frombuffer keeps a reference to the array, so it lives as long as the surface.
            return pygame.image.frombuffer(np.ascontiguousarray(out), (w, h), "BGRA")
        except Exception:  # noqa: BLE001 - never lose the output over a warp
            log.exception("Corner pin failed; drawing this frame unwarped")
            return surface

    def _layers_contain_video(self, layers):
        """Check if any layer contains a video file."""
        for layer in layers.values():
            path = layer.get('path')
            if is_streamed(path) and (live_source.is_live(path) or os.path.exists(path)):
                return True
        return False

    def render(self, layers, target_monitor=None, transition_state=None, all_assigned_screens=None):
        self._drop_stale_static_cache()
        layers_to_draw = []
        if transition_state and transition_state.get('active'):
            fa, ta = transition_alphas(transition_state)
            for layer in transition_state['from'].values():
                layer['alpha'] = fa
                layers_to_draw.append(layer)
            for layer in transition_state['to'].values():
                layer['alpha'] = ta
                layers_to_draw.append(layer)
        elif layers:
            layers_to_draw.extend(layers.values())

        if target_monitor is not None:
            # FIX: Validate the target_monitor index
            if not (self.monitors and 0 <= target_monitor < len(self.monitors)):
                log.warning(f"Invalid target_monitor index ({target_monitor}) provided. Closing output window.")
                self.close_window()
                return

            if not self._create_window((self.monitors[target_monitor].x, self.monitors[target_monitor].y, 
                                       self.monitors[target_monitor].width, self.monitors[target_monitor].height)): 
                log.debug("Failed to create window")
                return
            
            final_surface = None
            dims = self.window.get_size()
            if target_monitor in self.output_regions:
                dims = self.composition_size(target_monitor)   # laid out on the span's canvas
            
            # Check for static cue caching opportunity (only when not in transition)
            use_static_cache = False
            static_cache_key = None
            
            if not (transition_state and transition_state.get('active')):
                # Not in transition, check if this is a static cue
                if self.is_cue_static(layers):
                    static_cache_key = self.get_cue_cache_key(layers, target_monitor)
                    
                    if static_cache_key in self.static_cue_cache:
                        # Use cached pre-rendered cue
                        final_surface = self.static_cue_cache[static_cache_key]
                        use_static_cache = True
                        self.static_cue_cache_hits += 1
                        log.debug(f"Static cue cache hit (total hits: {self.static_cue_cache_hits})")
                    else:
                        # Pre-render the static cue
                        final_surface = self.pre_render_static_cue(layers, dims)
                        if final_surface:
                            # Make a copy for caching to avoid surface modification issues
                            cached_surface = final_surface.copy()
                            self.static_cue_cache[static_cache_key] = cached_surface
                            use_static_cache = True
                            self.static_cue_cache_misses += 1
                            log.debug(f"Pre-rendered static cue (total misses: {self.static_cue_cache_misses})")
                            
                            # Limit cache size
                            if len(self.static_cue_cache) > 10:
                                # Remove oldest entry
                                oldest_key = next(iter(self.static_cue_cache))
                                self.static_cue_cache.pop(oldest_key)
                                log.debug("Removed oldest static cue from cache")
            
            # If we didn't use static cache, proceed with normal rendering
            if not use_static_cache:
                if transition_state and transition_state.get('active'):
                    # Handle transition rendering (existing code)
                    from_has_video = self._layers_contain_video(transition_state['from'])
                    to_has_video = self._layers_contain_video(transition_state['to'])
                    
                    # Use transition cache for static content
                    if from_has_video:
                        from_surface = self._render_layers_to_surface(transition_state['from'], dims)
                    else:
                        from_cache_key = self._get_transition_cache_key(transition_state['from'], dims)
                        if self.transition_cache.get('from_cache_key') != from_cache_key:
                            self.transition_cache['from_surface'] = self._render_layers_to_surface(transition_state['from'], dims)
                            self.transition_cache['from_cache_key'] = from_cache_key
                        from_surface = self.transition_cache['from_surface']
                    
                    if to_has_video:
                        to_surface = self._render_layers_to_surface(transition_state['to'], dims)
                    else:
                        to_cache_key = self._get_transition_cache_key(transition_state['to'], dims)
                        if self.transition_cache.get('to_cache_key') != to_cache_key:
                            self.transition_cache['to_surface'] = self._render_layers_to_surface(transition_state['to'], dims)
                            self.transition_cache['to_cache_key'] = to_cache_key
                        to_surface = self.transition_cache['to_surface']
                    
                    final_surface = self.get_transition_frame(from_surface, to_surface, transition_state['progress'],
                                                              transition_alphas(transition_state))
                else:
                    # Clear transition cache if not transitioning
                    if self.transition_cache.get('from_cache_key') is not None or self.transition_cache.get('to_cache_key') is not None:
                        self.transition_cache = {
                            'from_surface': None,
                            'to_surface': None,
                            'from_cache_key': None,
                            'to_cache_key': None
                        }
                    # Render current layers normally
                    final_surface = self._render_layers_to_surface(layers, dims)

            # Blit the final surface to the window
            self.window.fill((0, 0, 0))
            if final_surface:
                self.window.blit(self._warp_surface(final_surface, target_monitor), (0, 0))
            pygame.display.flip()
            return

        # --- MULTI-MONITOR MODE ---
        layers_by_screen = {}
        all_screens = set(all_assigned_screens) if all_assigned_screens is not None else set()
        
        for layer in layers_to_draw:
            for screen_idx in layer.get('screens', []):
                if screen_idx is None: continue
                if screen_idx not in layers_by_screen:
                    layers_by_screen[screen_idx] = []
                layers_by_screen[screen_idx].append(layer)
                all_screens.add(screen_idx)

        if not all_screens:
            self.close_window()
            return

        min_x = min(self.monitors[s].x for s in all_screens)
        min_y = min(self.monitors[s].y for s in all_screens)
        max_r = max(self.monitors[s].x + self.monitors[s].width for s in all_screens)
        max_b = max(self.monitors[s].y + self.monitors[s].height for s in all_screens)
        total_w, total_h = max_r - min_x, max_b - min_y
        
        if not self._create_window((min_x, min_y, total_w, total_h)): 
            return

        self.window.fill((0, 0, 0))

        # Multi-monitor mode with static cue caching per screen
        for screen_idx in all_screens:
            mon = self.monitors[screen_idx]
            dims = self.composition_size(screen_idx)       # the display, or its span's canvas
            blit_pos = (mon.x - min_x, mon.y - min_y)
            
            final_surface = None
            
            if transition_state and transition_state.get('active'):
                # Handle transitions in multi-monitor mode
                layers_from = {k: v for k, v in transition_state['from'].items() if screen_idx in v.get('screens', [])}
                layers_to = {k: v for k, v in transition_state['to'].items() if screen_idx in v.get('screens', [])}
                
                from_surf = self._render_layers_to_surface(layers_from, dims)
                to_surf = self._render_layers_to_surface(layers_to, dims)
                final_surface = self.get_transition_frame(from_surf, to_surf, transition_state['progress'],
                                                          transition_alphas(transition_state))
            else:
                # Check for static cue caching per screen
                screen_layers = layers_by_screen.get(screen_idx, [])
                screen_layers_dict = {i: layer for i, layer in enumerate(screen_layers)}
                
                if self.is_cue_static(screen_layers_dict):
                    # Generate cache key for this screen's layers
                    static_cache_key = self.get_cue_cache_key(screen_layers_dict, screen_idx)
                    
                    if static_cache_key in self.static_cue_cache:
                        # Use cached surface for this screen
                        final_surface = self.static_cue_cache[static_cache_key]
                        self.static_cue_cache_hits += 1
                        log.debug(f"Static cue cache hit for screen {screen_idx}")
                    else:
                        # Pre-render and cache this screen's static content
                        final_surface = self._render_layers_to_surface(screen_layers, dims)
                        if final_surface:
                            cached_surface = final_surface.copy()
                            self.static_cue_cache[static_cache_key] = cached_surface
                            self.static_cue_cache_misses += 1
                            log.debug(f"Pre-rendered static cue for screen {screen_idx}")
                            
                            # Limit cache size
                            if len(self.static_cue_cache) > 20:  # Allow more entries for multi-monitor
                                oldest_key = next(iter(self.static_cue_cache))
                                self.static_cue_cache.pop(oldest_key)
                else:
                    # Render normally for dynamic content
                    final_surface = self._render_layers_to_surface(screen_layers, dims)
            
            if final_surface:
                self.window.blit(self._warp_surface(final_surface, screen_idx), blit_pos)

        pygame.display.flip()
