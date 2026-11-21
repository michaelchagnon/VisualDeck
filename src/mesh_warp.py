# SPDX-License-Identifier: MPL-2.0
"""Mesh warp (grid warp) for an output.

A mesh is a grid of control points, `cols` x `rows` (2..9 each), listed row by
row from the top left. Point (i, j) is where the picture's point
(i / (cols - 1), j / (rows - 1)) goes, in the output's normalized coordinates
(0..1, y down). The unwarped mesh is a regular grid (identity). Between the
points the picture follows a smooth surface through every point
(Catmull-Rom spline along each axis, extended linearly past the outer row and
column), so a regular grid reproduces the picture exactly and moving one point
bends the picture around it without kinks.

The mesh works inside the corner pin: the corner pin moves the whole mesh, the
mesh shapes the picture within it (final position = pin(mesh(u, v))). Without
a pin, the pin is the identity.

Both engines draw a tessellation of that surface (small triangles):
- OpenGL: the triangles, with the pin applied per vertex in homogeneous
  coordinates so the texture is interpolated with perspective, as the
  corner-pin shader does.
- Pygame: an inverse map (output pixel -> picture pixel) rasterized once per
  change from the same triangles, then cv2.remap every frame.

No GL or UI here.
"""
from __future__ import annotations

import functools
import math

import numpy as np

import output_warp

MIN_POINTS = 2
MAX_POINTS = 9
LIMIT = output_warp.LIMIT       # points may sit this far outside the output (normalized)
SEGMENTS = 64                   # tessellation: about this many triangle strips per axis


def identity(cols, rows):
    """The regular grid: [[x, y], ...], row by row."""
    return [[i / (cols - 1), j / (rows - 1)] for j in range(rows) for i in range(cols)]


def is_identity(cols, rows, points, tol=1e-6):
    return all(abs(a - b) <= tol for p, q in zip(points, identity(cols, rows)) for a, b in zip(p, q))


def _size(v):
    if isinstance(v, bool):
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if MIN_POINTS <= n <= MAX_POINTS and n == v else None


def parse(raw):
    """(cols, rows, points) from a saved mesh, whatever it holds, or None when
    it is malformed. Identity and folded meshes are returned too (see clean)."""
    if not isinstance(raw, dict):
        return None
    cols, rows = _size(raw.get("cols")), _size(raw.get("rows"))
    pts = raw.get("points")
    if cols is None or rows is None or not isinstance(pts, (list, tuple)) or len(pts) != cols * rows:
        return None
    try:
        points = tuple((float(p[0]), float(p[1])) for p in pts)
    except (TypeError, ValueError, IndexError):
        return None
    if any(len(p) != 2 for p in pts) or any(v != v or not (-LIMIT <= v <= 1 + LIMIT) for p in points for v in p):
        return None
    return cols, rows, points


def clean(raw):
    """(cols, rows, points) for a mesh that changes the picture, or None
    (missing, malformed, identity, or a shape that folds the picture over
    itself)."""
    m = parse(raw)
    if m is None or is_identity(*m) or folds(m):
        return None
    return m


def as_dict(mesh):
    cols, rows, points = mesh
    return {"cols": cols, "rows": rows, "points": [[round(x, 6), round(y, 6)] for x, y in points]}


# --- The surface ------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def _weights(n, samples):
    """(len(samples), n) matrix: the Catmull-Rom weights of the n control
    points for each parameter t in `samples` (0..1 along that axis). The
    ghost points before the first and after the last are 2*P0 - P1 and
    2*Pn-1 - Pn-2 (linear extension), so evenly spaced points reproduce a
    straight line exactly."""
    t = np.asarray(samples, dtype=np.float64) * (n - 1)
    seg = np.clip(np.floor(t), 0, n - 2).astype(int)
    f = t - seg
    f2, f3 = f * f, f * f * f
    basis = np.stack([(-f3 + 2 * f2 - f) / 2, (3 * f3 - 5 * f2 + 2) / 2,
                      (-3 * f3 + 4 * f2 + f) / 2, (f3 - f2) / 2], axis=1)
    w = np.zeros((len(t), n + 2))                    # padded: index 0 is the ghost before point 0
    for k in range(4):
        np.add.at(w, (np.arange(len(t)), seg + k), basis[:, k])
    out = w[:, 1:n + 1].copy()
    out[:, 0] += 2 * w[:, 0]
    out[:, 1] -= w[:, 0]
    out[:, n - 1] += 2 * w[:, n + 1]
    out[:, n - 2] -= w[:, n + 1]
    out.setflags(write=False)
    return out


def _grid_params(cols, rows, segments=SEGMENTS):
    """Picture coordinates of the tessellation: (us, vs), each 0..1, including
    every control point's own position."""
    def axis(n):
        sub = max(1, math.ceil(segments / (n - 1)))
        return tuple(np.linspace(0.0, 1.0, (n - 1) * sub + 1))
    return axis(cols), axis(rows)


def surface(mesh, us, vs):
    """Output positions of the picture points (us x vs): array (len(vs), len(us), 2)."""
    cols, rows, points = mesh
    p = np.asarray(points, dtype=np.float64).reshape(rows, cols, 2)
    wu = _weights(cols, tuple(float(u) for u in us))
    wv = _weights(rows, tuple(float(v) for v in vs))
    return np.einsum("aj,jik,bi->abk", wv, p, wu)


def evaluate(mesh, u, v):
    """Output position of one picture point (u, v)."""
    s = surface(mesh, (u,), (v,))
    return float(s[0, 0, 0]), float(s[0, 0, 1])


def tessellate(mesh, segments=SEGMENTS):
    """(us, vs, xy): the picture grid and where each of its points goes
    (xy has shape (len(vs), len(us), 2))."""
    us, vs = _grid_params(mesh[0], mesh[1], segments)
    return us, vs, surface(mesh, us, vs)


def _signed_areas(xy):
    """Twice the signed area of the two triangles of every tessellation cell
    (y down, so the unwarped picture is positive)."""
    a, b = xy[:-1, :-1], xy[:-1, 1:]
    c, d = xy[1:, 1:], xy[1:, :-1]

    def cross(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    return cross(a, b, c), cross(a, c, d)


def folds(mesh, segments=SEGMENTS):
    """True if the surface turns over anywhere (a triangle of the tessellation
    is flipped or flat), which would show part of the picture mirrored."""
    _, _, xy = tessellate(mesh, segments)
    t1, t2 = _signed_areas(xy)
    return bool((t1 <= 1e-12).any() or (t2 <= 1e-12).any())


def resize(mesh_or_none, cols, rows):
    """A cols x rows mesh with the same shape (the surface sampled at the new
    grid), or the identity when there was no mesh."""
    if mesh_or_none is None:
        return identity(cols, rows)
    us = [i / (cols - 1) for i in range(cols)]
    vs = [j / (rows - 1) for j in range(rows)]
    s = surface(mesh_or_none, us, vs)
    return [[float(s[j, i, 0]), float(s[j, i, 1])] for j in range(rows) for i in range(cols)]


# --- What the engines draw -----------------------------------------------------------

def _homogeneous(xy, corners):
    """(X, Y, W) of each point through the corner pin (W = 1 without one)."""
    pts = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], axis=-1)
    if corners is None:
        return pts
    h = output_warp.homography(corners)
    return pts @ h.T


def through_pin_ok(mesh, corners, segments=SEGMENTS):
    """False if the pin would put part of the mesh behind the projector
    (W <= 0) or fold it; then the engines draw the pin alone."""
    if corners is None:
        return True
    _, _, xy = tessellate(mesh, segments)
    hw = _homogeneous(xy, corners)
    if (hw[..., 2] <= 1e-9).any():
        return False
    out = hw[..., :2] / hw[..., 2:3]
    t1, t2 = _signed_areas(out)
    return bool((t1 > 0).all() and (t2 > 0).all()) or bool((t1 < 0).all() and (t2 < 0).all())


def gl_vertices(mesh, corners=None, segments=SEGMENTS):
    """float32 array of triangles for GL_TRIANGLES, 5 floats per vertex:
    X, Y, W (output position through the pin, homogeneous, 0..1 y down after
    dividing by W) and u, v (picture coordinates, 0..1 y down)."""
    us, vs, xy = tessellate(mesh, segments)
    hw = _homogeneous(xy, corners)
    uu, vv = np.meshgrid(np.asarray(us), np.asarray(vs))
    vert = np.concatenate([hw, uu[..., None], vv[..., None]], axis=-1)    # (R, C, 5)
    a, b = vert[:-1, :-1], vert[:-1, 1:]
    c, d = vert[1:, 1:], vert[1:, :-1]
    tris = np.stack([a, b, c, a, c, d], axis=2)                           # (R-1, C-1, 6, 5)
    return np.ascontiguousarray(tris.reshape(-1, 5), dtype=np.float32)


def _triangles(a):
    """(N, K, 3) per-vertex values of a (R, C, K) grid -> (2 * cells, K, 3):
    the two triangles of every cell, components in rows, vertices in columns."""
    p, q, r, s = a[:-1, :-1], a[:-1, 1:], a[1:, 1:], a[1:, :-1]
    k = a.shape[-1]
    return np.concatenate([np.stack([p, q, r], -1).reshape(-1, k, 3), np.stack([p, r, s], -1).reshape(-1, k, 3)])


def inverse_map(mesh, corners, width, height, src_size=None, segments=SEGMENTS, edge_px=2):
    """(map_x, map_y) float32 arrays of shape (height, width) for cv2.remap:
    for every output pixel, the picture pixel to sample. The picture is
    src_size (w, h), the output's size by default. Texture coordinates are
    interpolated with perspective, as the GL path does.

    Each triangle is filled with its number (cv2.fillConvexPoly), then every
    pixel evaluates its triangle's plane equations. The map is carried
    edge_px pixels past the picture's edge, so cv2.remap's constant border
    (black) gives a smooth edge; pixels further out get -1e4 (black)."""
    import cv2
    sw, sh = src_size or (width, height)
    us, vs, xy = tessellate(mesh, segments)
    hw = _homogeneous(xy, corners)
    wgt = hw[..., 2]
    one = np.ones_like(wgt)
    pos = np.stack([hw[..., 0] / wgt * width - 0.5, hw[..., 1] / wgt * height - 0.5, one], -1)
    uu, vv = np.meshgrid(np.asarray(us), np.asarray(vs))
    # Attributes divided by W vary linearly on screen (perspective-correct).
    attr = np.stack([uu / wgt, vv / wgt, one / wgt], -1)
    P, T = _triangles(pos), _triangles(attr)
    det = np.linalg.det(P)
    xs, ys = P[:, 0, :], P[:, 1, :]
    visible = ((np.abs(det) > 1e-9) & (xs.max(1) >= -edge_px - 1) & (xs.min(1) <= width + edge_px)
               & (ys.max(1) >= -edge_px - 1) & (ys.min(1) <= height + edge_px))
    coef = np.zeros_like(P)
    coef[visible] = T[visible] @ np.linalg.inv(P[visible])     # (x, y, 1) -> (u/W, v/W, 1/W)
    shift = 4
    verts = np.round(np.stack([xs, ys], -1) * (1 << shift)).astype(np.int32)
    ids = np.full((height, width), -1.0, dtype=np.float32)
    for k in np.nonzero(visible)[0]:
        cv2.fillConvexPoly(ids, verts[k], float(k), lineType=cv2.LINE_8, shift=shift)
    if edge_px > 0:
        grown = cv2.dilate(ids, np.ones((2 * edge_px + 1, 2 * edge_px + 1), np.uint8))
        ids = np.where(ids >= 0, ids, grown)
    idx = ids.astype(np.int32)
    miss = idx < 0
    idx[miss] = 0
    gy, gx = np.mgrid[0:height, 0:width].astype(np.float32)
    cf = coef.astype(np.float32)
    a = [cf[:, r, 0][idx] * gx + cf[:, r, 1][idx] * gy + cf[:, r, 2][idx] for r in range(3)]
    with np.errstate(divide="ignore", invalid="ignore"):
        map_x = a[0] / a[2] * sw - 0.5
        map_y = a[1] / a[2] * sh - 0.5
    miss |= ~np.isfinite(map_x) | ~np.isfinite(map_y)
    map_x[miss] = -1e4
    map_y[miss] = -1e4
    return map_x.astype(np.float32), map_y.astype(np.float32)


def warp_image(arr, mesh, corners=None, maps=None):
    """An HxWxC uint8 array through the mesh (and pin); outside the picture is
    black (alpha 255 for 4 channels). `maps` from inverse_map() saves building
    them again."""
    import cv2
    h, w = arr.shape[:2]
    mx, my = maps if maps is not None else inverse_map(mesh, corners, w, h)
    border = (0, 0, 0, 255) if arr.ndim == 3 and arr.shape[2] == 4 else 0
    return cv2.remap(arr, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=border)
