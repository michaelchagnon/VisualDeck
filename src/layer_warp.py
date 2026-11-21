# SPDX-License-Identifier: MPL-2.0
"""Warp points on a layer: bend one picture onto a piece of scenery.

A layer's warp is a mesh (mesh_warp's format: `cols` x `rows` control points,
2..9 each, row by row from the top left) in the layer's own coordinates:
point (i, j) is where the picture's point (i / (cols - 1), j / (rows - 1))
goes, as a fraction of the layer's unwarped rectangle (0..1, y down; points
may go up to mesh_warp.LIMIT outside it). The layer's position, size and
rotation then place the bent picture on the output, so moving or resizing the
layer keeps its warp. Between the points the picture follows the same smooth
surface as a display's mesh warp.

Saved in the cell's settings as 'warp' ({'cols', 'rows', 'points'}); a regular
grid (nothing bent) is not saved.

- OpenGL draws the layer as the tessellated surface (gl_vertices) with the
  layer's own shader, so opacity, color, masks (in the picture's coordinates),
  blend modes and effects all work on the bent picture.
- Balanced and Compatibility Mode, and the editor's canvas, bend the picture
  with an inverse map and cv2.remap (warp_array), cached per shape and size.

No GL or UI here.
"""
from __future__ import annotations

import collections
import threading

import numpy as np

import mesh_warp

MAP_CACHE = 8                 # inverse maps kept (each is two float32 arrays the size of the layer)
# Triangles per axis for the CPU bend. The surface is smooth between the points, and
# 16 per axis keeps the bent picture within a fraction of a pixel of 64 (the GPU's)
# for the shapes warp points make, at a quarter of the cost: a new shape at the
# editor's canvas size took 50 ms with 64 (owner's report: dragging a dot lagged).
MAP_SEGMENTS = 16
HALF_RES_PIXELS = 400_000     # bigger pictures: maps built at half size and scaled up (4x faster)
_maps = collections.OrderedDict()
_lock = threading.Lock()


def clean(raw):
    """(cols, rows, points) for a warp that bends the picture, or None."""
    return mesh_warp.clean(raw)


def layer_warp(layer):
    """The cleaned warp of a render layer (its 'warp' key), or None."""
    return clean(layer.get("warp")) if isinstance(layer, dict) else None


def to_dict(mesh):
    return mesh_warp.as_dict(mesh) if mesh is not None else None


def gl_vertices(mesh, segments=mesh_warp.SEGMENTS):
    """float32 triangles for GL_TRIANGLES in the layer quad's space, 5 floats
    per vertex: x, y (-0.5..0.5, y up, as the layer's quad), 0, u, v (picture
    coordinates, v down)."""
    us, vs, xy = mesh_warp.tessellate(mesh, segments)
    uu, vv = np.meshgrid(np.asarray(us), np.asarray(vs))
    vert = np.stack([xy[..., 0] - 0.5, 0.5 - xy[..., 1], np.zeros_like(uu), uu, vv], axis=-1)
    a, b = vert[:-1, :-1], vert[:-1, 1:]
    c, d = vert[1:, 1:], vert[1:, :-1]
    tris = np.stack([a, b, c, a, c, d], axis=2)
    return np.ascontiguousarray(tris.reshape(-1, 5), dtype=np.float32)


def extent(mesh, segments=16):
    """How far the bent picture reaches from the layer's center, as fractions of
    its width and height: (half width, half height), at least (0.5, 0.5)."""
    _, _, xy = mesh_warp.tessellate(mesh, segments)
    hx = float(np.abs(xy[..., 0] - 0.5).max())
    hy = float(np.abs(xy[..., 1] - 0.5).max())
    return max(0.5, hx), max(0.5, hy)


def _maps_for(mesh, w, h):
    """(map_x, map_y, out_w, out_h): the inverse map from the bent picture
    (out_w x out_h, centered on the layer's center) to the picture (w x h)."""
    key = (mesh, w, h)
    with _lock:
        hit = _maps.get(key)
        if hit is not None:
            _maps.move_to_end(key)
            return hit
    import cv2
    hx, hy = extent(mesh)
    out_w, out_h = int(np.ceil(2 * hx * w)) + 2, int(np.ceil(2 * hy * h)) + 2
    out_w, out_h = min(out_w, 4 * w + 2), min(out_h, 4 * h + 2)       # mesh_warp.LIMIT keeps it bounded
    cols, rows, points = mesh
    # The layer's coordinates (0..1 of w x h, centered at 0.5) as fractions of the output box.
    ox, oy = (out_w - w) / 2.0, (out_h - h) / 2.0
    pts = tuple(((x * w + ox) / out_w, (y * h + oy) / out_h) for x, y in points)
    box = (cols, rows, pts)
    half = out_w * out_h > HALF_RES_PIXELS
    bw, bh = (max(2, out_w // 2), max(2, out_h // 2)) if half else (out_w, out_h)
    mx, my = mesh_warp.inverse_map(box, None, bw, bh, src_size=(w, h), segments=MAP_SEGMENTS)
    if half:
        mx = cv2.resize(mx, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        my = cv2.resize(my, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    hit = (mx, my, out_w, out_h)
    with _lock:
        _maps[key] = hit
        while len(_maps) > MAP_CACHE:
            _maps.popitem(last=False)
    return hit


def warp_array(arr, mesh):
    """An HxWx4 uint8 picture (any channel order) bent by the warp: a new array
    centered on the same point (it can be bigger than the picture); what the
    picture does not cover is transparent."""
    import cv2
    h, w = arr.shape[:2]
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("warp_array needs a 4-channel picture")
    mx, my, _ow, _oh = _maps_for(mesh, w, h)
    return cv2.remap(arr, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(0, 0, 0, 0))


def warp_pil(img, mesh):
    """A PIL image bent by the warp (RGBA, centered on the same point)."""
    from PIL import Image
    arr = np.asarray(img.convert("RGBA"), dtype=np.uint8)
    return Image.fromarray(warp_array(arr, mesh))
