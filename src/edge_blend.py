# SPDX-License-Identifier: MPL-2.0
"""Edge blending (soft edges) for overlapping projectors.

Where two projectors overlap, each one's picture is darkened towards its edge
so the overlap is not twice as bright. Per display, a blend width for each
side (as a fraction of the output's width or height) plus two shape controls:

- curve (p, default 2): how quickly the ramp rises (Paul Bourke's blend
  function: 0.5 * (2t)^p for the first half, 1 - 0.5 * (2(1-t))^p for the
  second, so two overlapping ramps add up to 1);
- gamma (default 2.2): the projector's gamma. The ramp is computed in light
  and converted to pixel values with ^(1/gamma), so the light adds up to a
  constant across the overlap.

The same math is used by the OpenGL shader (in the corner-pin pass) and the
Pygame path (a cached multiply mask). No GL or UI here.
"""
from __future__ import annotations

import functools

import numpy as np

SIDES = ("left", "right", "top", "bottom")
MAX_WIDTH = 0.5
DEFAULT_CURVE = 2.0
DEFAULT_GAMMA = 2.2


def clean(raw):
    """(left, right, top, bottom, curve, gamma) or None when no side blends.
    Widths are fractions 0..0.5; curve 1..4; gamma 1..3."""
    if not isinstance(raw, dict):
        return None

    def num(key, default, lo, hi):
        try:
            v = float(raw.get(key, default))
        except (TypeError, ValueError):
            return default
        return default if v != v else min(hi, max(lo, v))
    widths = tuple(num(s, 0.0, 0.0, MAX_WIDTH) for s in SIDES)
    if not any(w > 0 for w in widths):
        return None
    return widths + (num("curve", DEFAULT_CURVE, 1.0, 4.0), num("gamma", DEFAULT_GAMMA, 1.0, 3.0))


def ramp(t, curve=DEFAULT_CURVE, gamma=DEFAULT_GAMMA):
    """Pixel multiplier at position t across the blend (0 = outer edge, 1 = inside)."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    light = np.where(t < 0.5, 0.5 * (2.0 * t) ** curve, 1.0 - 0.5 * (2.0 * (1.0 - t)) ** curve)
    return light ** (1.0 / gamma)


def axis_factor(n, near, far, curve, gamma):
    """Multipliers for n pixel centers along one axis (near side width, far side width)."""
    x = (np.arange(n, dtype=np.float64) + 0.5) / n
    f = np.ones(n)
    if near > 0:
        f *= np.where(x < near, ramp(x / near, curve, gamma), 1.0)
    if far > 0:
        f *= np.where(x > 1.0 - far, ramp((1.0 - x) / far, curve, gamma), 1.0)
    return f


def factor(x, y, blend):
    """Multiplier at output point (x, y), normalized with y down."""
    left, right, top, bottom, curve, gamma = blend
    k = 1.0
    for pos, width in ((x, left), (1.0 - x, right), (y, top), (1.0 - y, bottom)):
        if width > 0 and pos < width:
            k *= float(ramp(pos / width, curve, gamma))
    return k


@functools.lru_cache(maxsize=8)
def mask(width, height, blend, channels=4):
    """uint8 multiply mask (255 = unchanged) for an image of width x height;
    alpha (4th channel) stays 255."""
    left, right, top, bottom, curve, gamma = blend
    fx = axis_factor(width, left, right, curve, gamma)
    fy = axis_factor(height, top, bottom, curve, gamma)
    m = np.clip(np.outer(fy, fx) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out = np.repeat(m[:, :, None], channels, axis=2)
    if channels == 4:
        out[..., 3] = 255
    out.setflags(write=False)
    return out


def apply(arr, blend):
    """Edge-blend an HxWxC uint8 image (returns a new array)."""
    import cv2
    h, w = arr.shape[:2]
    m = mask(w, h, blend, arr.shape[2] if arr.ndim == 3 else 1)
    return cv2.multiply(arr, m, scale=1.0 / 255.0)
