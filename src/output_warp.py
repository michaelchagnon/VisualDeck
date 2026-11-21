# SPDX-License-Identifier: MPL-2.0
"""Corner pin (four-corner keystone) for an output.

A warp is four corners, top-left, top-right, bottom-right, bottom-left, in the
output's normalized coordinates (0..1, y down). The unwarped output is
IDENTITY. The picture is mapped onto that quadrilateral with a projective
transform (homography), which is what a projector's keystone correction does,
so straight lines stay straight.

No GL or UI here: the OpenGL engine samples with the inverse homography in a
shader, the Pygame engine uses cv2.warpPerspective, both from these matrices.
"""
from __future__ import annotations

import numpy as np

IDENTITY = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
CORNER_NAMES = ("top left", "top right", "bottom right", "bottom left")
LIMIT = 2.0          # corners may sit up to this far outside the output (normalized)


def clean(corners):
    """Four (x, y) float pairs within +-LIMIT forming a convex quadrilateral,
    or None (identity, malformed, or a shape that would fold the picture)."""
    try:
        pts = tuple((float(x), float(y)) for x, y in corners)
    except (TypeError, ValueError):
        return None
    if len(pts) != 4 or any(not (-LIMIT <= v <= 1 + LIMIT) or v != v for p in pts for v in p):
        return None
    if is_identity(pts) or not convex(pts):
        return None
    return pts


def is_identity(corners, tol=1e-6):
    return all(abs(a - b) <= tol for p, q in zip(corners, IDENTITY) for a, b in zip(p, q))


def convex(corners):
    """True if the corners, in order, make a convex quadrilateral (no fold, no
    crossing, no three corners on a line)."""
    signs = []
    for i in range(4):
        (x0, y0), (x1, y1), (x2, y2) = corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]
        cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
        if abs(cross) < 1e-9:
            return False
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def homography(corners):
    """3x3 matrix H with H @ (u, v, 1) ~ (x, y, 1): the unit square (picture
    coordinates) onto the corners (output coordinates), both y down."""
    src = np.array(IDENTITY, dtype=np.float64)
    dst = np.array(corners, dtype=np.float64)
    a = []
    b = []
    for (u, v), (x, y) in zip(src, dst):
        a.append([u, v, 1, 0, 0, 0, -u * x, -v * x])
        b.append(x)
        a.append([0, 0, 0, u, v, 1, -u * y, -v * y])
        b.append(y)
    h = np.linalg.solve(np.array(a), np.array(b))
    return np.append(h, 1.0).reshape(3, 3)


def inverse(corners):
    """Output coordinates -> picture coordinates (what a shader samples with)."""
    return np.linalg.inv(homography(corners))


def apply(h, x, y):
    p = h @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def pixel_matrix(corners, width, height):
    """H in pixel units for an image of width x height (for cv2.warpPerspective)."""
    s = np.diag([width, height, 1.0])
    return s @ homography(corners) @ np.linalg.inv(s)


def warp_image(arr, corners, out_size=None):
    """Warp an HxWxC uint8 array; the area outside the quadrilateral is black
    (alpha 255 for 4 channels, so it reads as black on the output)."""
    import cv2
    h, w = arr.shape[:2]
    ow, oh = out_size or (w, h)
    m = pixel_matrix(corners, ow, oh) @ np.diag([ow / w, oh / h, 1.0])
    border = (0, 0, 0, 255) if arr.ndim == 3 and arr.shape[2] == 4 else 0
    return cv2.warpPerspective(arr, m, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=border)


def inverse_maps(corners, width, height):
    """Remap tables for a width x height picture through the pin, in OpenCV's
    fixed-point form (cv2.convertMaps, CV_16SC2): what warp_image computes on
    every call, made once. cv2.remap with them takes about 60 % of the time of
    cv2.warpPerspective on the test VM (6.8 against 11.3 ms per 1080p frame),
    and the result is the same picture (warpPerspective samples in the same
    fixed point). Pixels beyond the pin's horizon (where the homography turns
    over) are outside, so nothing is drawn there."""
    import cv2
    hi = np.linalg.inv(pixel_matrix(corners, width, height)).astype(np.float64)
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)[:, None]
    den = hi[2, 0] * xs + hi[2, 1] * ys + hi[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        mx = (hi[0, 0] * xs + hi[0, 1] * ys + hi[0, 2]) / den
        my = (hi[1, 0] * xs + hi[1, 1] * ys + hi[1, 2]) / den
    bad = (den <= 1e-12) | ~np.isfinite(mx) | ~np.isfinite(my)
    mx = np.where(bad, -1e4, np.clip(mx, -1e4, 3e4)).astype(np.float32)
    my = np.where(bad, -1e4, np.clip(my, -1e4, 3e4)).astype(np.float32)
    return cv2.convertMaps(mx, my, cv2.CV_16SC2)


def warp_with_maps(arr, maps):
    """warp_image with tables from inverse_maps() for the same size."""
    import cv2
    border = (0, 0, 0, 255) if arr.ndim == 3 and arr.shape[2] == 4 else 0
    return cv2.remap(arr, maps[0], maps[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=border)
