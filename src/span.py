# SPDX-License-Identifier: MPL-2.0
"""Span: one picture across several displays side by side (projectors with
overlapping edges, or LED processors fed by two outputs).

The displays in a span, left to right, share one canvas: its width is the sum
of their widths less the overlaps, its height the tallest display's. Each
display shows its own part of the canvas, at the canvas's pixel size (no
scaling), so a layer laid out on the canvas continues from one display to the
next. Where two projectors overlap, both show the same strip of the canvas;
the edge blend (edge_blend.py) on those sides makes the overlap even.

In the show file: outputs.span = {"displays": [1, 2], "overlap": 200}
(overlap in pixels, the same between each pair); a grid (a video wall, four
projectors) adds "columns" and "overlap_y": {"displays": [0, 1, 2, 3],
"overlap": 0, "columns": 2, "overlap_y": 0}, displays row by row. The output menu's "Span"
then sends every layer to every display in the span.

Slices (parts set by hand) replace the computed layout: a canvas size and,
per display, the rectangle of the canvas it shows, in canvas pixels. A part
of another size than its display is scaled to fill the display (a zoom, or
an LED processor's input). One display is enough: {"displays": [1],
"overlap": 0, "canvas": [3840, 1080], "slices": {"1": [960, 0, 1920, 1080]}}.
Parts may overlap or leave gaps; no blends are set for them.

No GL or UI here.
"""
from __future__ import annotations

MAX_OVERLAP_SHARE = 0.5          # an overlap may cover at most half of the narrower display
MAX_CANVAS = 16384               # pixels per side (the largest texture most GPUs take)


def _int(v):
    if isinstance(v, bool):
        raise ValueError(v)
    f = float(v)
    if f != f or f != int(f):
        raise ValueError(v)
    return int(f)


def _clean_slices(raw, disp):
    """(canvas [w, h], {str(display): [x, y, w, h]}) or None: every display has a part
    of at least 1 x 1 px inside the canvas."""
    try:
        cw, ch = (_int(v) for v in raw.get("canvas"))
        if not (16 <= cw <= MAX_CANVAS and 16 <= ch <= MAX_CANVAS):
            return None
        parts = {}
        for i in disp:
            x, y, w, h = (_int(v) for v in (raw.get("slices") or {})[str(i)])
            if x < 0 or y < 0 or w < 1 or h < 1 or x + w > cw or y + h > ch:
                return None
            parts[str(i)] = [x, y, w, h]
    except (TypeError, ValueError, KeyError, AttributeError):
        return None
    return [cw, ch], parts


def clean(raw, n_monitors=None):
    """{'displays': [unique indices, at least two], 'overlap': px >= 0,
    'columns': displays per row, 'overlap_y': px >= 0} or None. Saved spans
    without columns are one row."""
    if not isinstance(raw, dict):
        return None
    disp = []
    for d in raw.get("displays") or []:
        if isinstance(d, bool):
            return None
        try:
            i = int(d)
        except (TypeError, ValueError):
            return None
        if i != d or i < 0 or i > 63 or i in disp:
            return None
        if n_monitors is not None and i >= n_monitors:
            return None
        disp.append(i)
    if raw.get("slices") is not None:            # parts set by hand: one display is enough
        got = _clean_slices(raw, disp) if disp else None
        if got is None:
            return None
        return {"displays": disp, "overlap": 0, "canvas": got[0], "slices": got[1]}
    if len(disp) < 2:
        return None
    try:
        ov = int(raw.get("overlap", 0))
        ov_y = int(raw.get("overlap_y", 0))
        cols = int(raw.get("columns", len(disp)))
    except (TypeError, ValueError):
        return None
    cols = max(1, min(len(disp), cols))
    if len(disp) % cols:
        return None                                   # rows must be full
    out = {"displays": disp, "overlap": max(0, ov)}
    if cols != len(disp):                             # a grid: saved only when there is more than one row
        out.update(columns=cols, overlap_y=max(0, ov_y))
    return out


def has_slices(span):
    return bool(span and span.get("slices"))


def grid(span):
    """(columns, rows) of a cleaned span (slices: one row of its displays)."""
    cols = span.get("columns", len(span["displays"]))
    return cols, len(span["displays"]) // cols


def layout(monitors, span):
    """(canvas (w, h), {display: {'rect': [x0, y0, x1, y1] (normalized of the
    canvas), 'canvas': [w, h], 'offset': x px, 'offset_y': y px, 'cell': (column,
    row)}}) for a cleaned span, or None if it does not fit these monitors
    (missing display, an overlap over half a display). Displays fill the grid
    row by row; each column is as wide as its widest display, each row as tall
    as its tallest."""
    if span is None:
        return None
    try:
        mons = [monitors[i] for i in span["displays"]]
    except (IndexError, TypeError):
        return None
    if has_slices(span):
        cw, ch = span["canvas"]
        out = {}
        for i in span["displays"]:
            x, y, w, h = span["slices"][str(i)]
            out[i] = {"rect": [x / cw, y / ch, (x + w) / cw, (y + h) / ch], "canvas": [cw, ch],
                      "offset": x, "offset_y": y, "cell": None}
        return (cw, ch), out
    cols, rows = grid(span)
    ov, ov_y = int(span["overlap"]), int(span.get("overlap_y", 0))
    if (cols > 1 and ov > MAX_OVERLAP_SHARE * min(m.width for m in mons)) or \
            (rows > 1 and ov_y > MAX_OVERLAP_SHARE * min(m.height for m in mons)):
        return None
    col_w = [max(mons[r * cols + c].width for r in range(rows)) for c in range(cols)]
    row_h = [max(mons[r * cols + c].height for c in range(cols)) for r in range(rows)]
    cw = sum(col_w) - ov * (cols - 1)
    ch = sum(row_h) - ov_y * (rows - 1)
    out = {}
    for k, (i, m) in enumerate(zip(span["displays"], mons)):
        c, r = k % cols, k // cols
        x = sum(col_w[:c]) - ov * c
        y = sum(row_h[:r]) - ov_y * r
        out[i] = {"rect": [x / cw, y / ch, (x + m.width) / cw, (y + m.height) / ch], "canvas": [cw, ch],
                  "offset": x, "offset_y": y, "cell": (c, r)}
    return (cw, ch), out


def blends(span, monitors, existing=None):
    """Edge blends for the overlaps: {display: blend dict}. The sides that
    meet a neighbor get the overlap; the outer sides, gamma and curve stay as
    they were in `existing` (edge_blend dicts by display)."""
    existing = existing or {}
    if has_slices(span):                          # parts set by hand: blends are set by hand too
        return {str(i): dict(existing.get(str(i)) or {}) for i in span["displays"]}
    out = {}
    cols, rows = grid(span)
    ov, ov_y = int(span["overlap"]), int(span.get("overlap_y", 0))
    for k, i in enumerate(span["displays"]):
        c, r = k % cols, k // cols
        b = dict(existing.get(str(i)) or {})
        w, h = monitors[i].width, monitors[i].height
        if c > 0:
            b["left"] = ov / w
        if c < cols - 1:
            b["right"] = ov / w
        if r > 0:
            b["top"] = ov_y / h
        if r < rows - 1:
            b["bottom"] = ov_y / h
        out[str(i)] = b
    return out


def describe(monitors, span):
    """One line per display: what part of the canvas it shows."""
    lay = layout(monitors, span)
    if lay is None:
        return []
    (cw, ch), parts = lay
    if has_slices(span):
        lines = [f"Picture {cw} \u00d7 {ch}, parts set by hand"]
        for i in span["displays"]:
            x, y, w, h = span["slices"][str(i)]
            m = monitors[i]
            scaled = "" if (w, h) == (m.width, m.height) else f", scaled to {m.width} \u00d7 {m.height}"
            lines.append(f"Display {i}: x {x}\u2013{x + w}, y {y}\u2013{y + h}{scaled}")
        return lines
    cols, rows = grid(span)
    lines = [f"Picture {cw} \u00d7 {ch}" + (f" ({cols} \u00d7 {rows} displays)" if rows > 1 else "")]
    for i in span["displays"]:
        p = parts[i]
        line = f"Display {i}: x {p['offset']}\u2013{p['offset'] + monitors[i].width}"
        if rows > 1:
            line += f", y {p['offset_y']}\u2013{p['offset_y'] + monitors[i].height}"
        lines.append(line)
    return lines
