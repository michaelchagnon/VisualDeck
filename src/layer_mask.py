# SPDX-License-Identifier: MPL-2.0
"""Layer masks: show only part of a layer (a crop or a vignette), with a soft edge.

Saved with the layer as "mask": {"shape": "rect" | "ellipse", "left", "right",
"top", "bottom" (insets from each side of the media, fractions 0..0.49),
"feather" (soft edge width, a fraction of the media's shorter side, 0..0.5),
"invert" (show only what is outside the shape)}.

The mask is in the media's own coordinates, so it moves, scales and rotates
with the layer. The soft edge lies inside the shape: the shape's outline is
where the layer is fully hidden, and it is fully visible `feather` further in
(with a smoothstep). An ellipse is the one that fits the inset rectangle; its
edge distance is a first-order estimate (exact for a circle), so the soft edge
keeps an even width.

The OpenGL fragment shader (render_engine.FRAGMENT_SHADER, maskValue) and the
CPU paths (factor/apply here) use the same math. No GL or UI here.
"""
from __future__ import annotations

import functools

import numpy as np

SHAPES = ("rect", "ellipse")
SHAPE_LABELS = {"rect": "Rectangle", "ellipse": "Ellipse"}
SIDES = ("left", "right", "top", "bottom")
MAX_INSET = 0.49
MAX_FEATHER = 0.5


def clean(raw):
    """(shape, left, top, right, bottom, feather, invert) with `right` and
    `bottom` as edges (1 - inset), or None for no mask (missing, malformed, or
    a rectangle that changes nothing)."""
    if not isinstance(raw, dict) or raw.get("shape") not in SHAPES:
        return None

    def num(key, hi):
        try:
            v = float(raw.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if v != v else min(hi, max(0.0, v))
    left, right, top, bottom = (num(s, MAX_INSET) for s in SIDES)
    feather = num("feather", MAX_FEATHER)
    invert = bool(raw.get("invert", False))
    shape = raw["shape"]
    if shape == "rect" and not invert and left == right == top == bottom == feather == 0.0:
        return None
    return (shape, left, top, 1.0 - right, 1.0 - bottom, feather, invert)


def to_dict(m):
    """The show-file form of a cleaned mask (None stays None)."""
    if m is None:
        return None
    shape, left, top, right_edge, bottom_edge, feather, invert = m
    out = {"shape": shape, "left": round(left, 4), "right": round(1.0 - right_edge, 4),
           "top": round(top, 4), "bottom": round(1.0 - bottom_edge, 4), "feather": round(feather, 4)}
    if invert:
        out["invert"] = True
    return out


def layer_mask(layer):
    """The cleaned mask of a render layer (its 'mask' key), or None."""
    return clean(layer.get("mask")) if isinstance(layer, dict) else None


def aspect(width, height):
    """Media width and height in units of its shorter side."""
    s = float(max(1, min(width, height)))
    return width / s, height / s


def edge_distance(u, v, m, width, height):
    """Distance inside the mask's outline (negative outside), in shorter
    sides, at texture coordinates u, v (arrays or floats, 0..1, v down)."""
    shape, lo_u, lo_v, hi_u, hi_v, _feather, _invert = m
    ax, ay = aspect(width, height)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if shape == "rect":
        dx = np.minimum(u - lo_u, hi_u - u) * ax
        dy = np.minimum(v - lo_v, hi_v - v) * ay
        return np.minimum(dx, dy)
    cu, cv = (lo_u + hi_u) * 0.5, (lo_v + hi_v) * 0.5
    rx = max((hi_u - lo_u) * 0.5 * ax, 1e-6)
    ry = max((hi_v - lo_v) * 0.5 * ay, 1e-6)
    px, py = (u - cu) * ax, (v - cv) * ay
    k = np.hypot(px / rx, py / ry)
    g = np.hypot(px / (rx * rx), py / (ry * ry))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(g > 1e-9, (1.0 - k) * k / np.where(g > 1e-9, g, 1.0), 1e9)
    return d


def factor(u, v, m, width, height):
    """Visibility 0..1 at texture coordinates (u, v) of a width x height medium."""
    d = edge_distance(u, v, m, width, height)
    feather, invert = m[5], m[6]
    if feather > 0:
        t = np.clip(d / feather, 0.0, 1.0)
    else:
        t = (d >= 0).astype(np.float64)
    t = t * t * (3.0 - 2.0 * t)
    return 1.0 - t if invert else t


@functools.lru_cache(maxsize=16)
def alpha_mask(width, height, m):
    """uint8 HxW visibility (255 = fully visible) at pixel centers; read-only, cached."""
    u = (np.arange(width, dtype=np.float64) + 0.5) / width
    v = (np.arange(height, dtype=np.float64) + 0.5) / height
    uu, vv = np.meshgrid(u, v)
    out = np.clip(factor(uu, vv, m, width, height) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out.setflags(write=False)
    return out


def apply(arr, m, alpha_channel=3):
    """Mask an HxWx4 uint8 image by multiplying its alpha (returns a new array;
    a 3-channel image gets an alpha channel, BGR(A) or RGB(A) alike)."""
    if m is None:
        return arr
    import cv2
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
    elif arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)      # channel order is kept, alpha 255
    else:
        arr = arr.copy()
    h, w = arr.shape[:2]
    a = np.ascontiguousarray(arr[..., alpha_channel])
    arr[..., alpha_channel] = cv2.multiply(a, alpha_mask(w, h, m), scale=1.0 / 255.0)
    return arr


def apply_pil(pil_img, m):
    """apply() for a PIL image (CPU previews); returns RGBA."""
    if m is None:
        return pil_img
    from PIL import Image
    return Image.fromarray(apply(np.asarray(pil_img.convert("RGBA")), m))
