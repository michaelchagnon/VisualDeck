# SPDX-License-Identifier: MPL-2.0
"""Output test patterns (Window > Test Pattern), drawn at each output's exact size.

- grid:  edge border (cropping/overscan), 10 x 10 grid, center cross, a circle
         as tall as the output (aspect ratio), corner marks, display label.
- bars:  eight 75 % color bars over a 0-255 gray ramp (levels, banding).
- white: full white (brightness and uniformity), small label.

Patterns are written as PNG files and shown as ordinary still-image layers,
so they take exactly the same path to the output as show content.
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

KINDS = ("grid", "bars", "white")
VERSION = 1   # bump when the drawing changes, so cached files are redrawn
LABELS = {"grid": "Grid", "bars": "Color bars", "white": "White"}
BARS_75 = [(191, 191, 191), (191, 191, 0), (0, 191, 191), (0, 191, 0),
           (191, 0, 191), (191, 0, 0), (0, 0, 191), (0, 0, 0)]


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:   # Pillow < 10.1: fixed-size bitmap font
        return ImageFont.load_default()


def _label(draw, w, h, text, fill, y_frac=0.5, bg=None):
    font = _font(max(12, h // 30))
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    x, y = (w - tw) // 2, int(h * y_frac - th / 2)
    if bg is not None:
        pad = max(4, th // 3)
        draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=bg)
    draw.text((x - box[0], y - box[1]), text, fill=fill, font=font)


def make(kind, width, height, label=""):
    if kind not in KINDS:
        raise ValueError(f"unknown test pattern {kind!r}")
    w, h = int(width), int(height)
    text = f"VisualDeck test pattern  -  {label}  -  {w} x {h}" if label else f"{w} x {h}"
    if kind == "white":
        img = Image.new("RGB", (w, h), (255, 255, 255))
        _label(ImageDraw.Draw(img), w, h, text, (128, 128, 128), y_frac=0.95)
        return img
    if kind == "bars":
        img = Image.new("RGB", (w, h), (0, 0, 0))
        d = ImageDraw.Draw(img)
        bar_h = int(h * 0.75)
        for i, color in enumerate(BARS_75):
            d.rectangle([i * w // 8, 0, (i + 1) * w // 8 - 1, bar_h - 1], fill=color)
        for x in range(w):                       # gray ramp, 0 at the left to 255 at the right
            v = round(255 * x / max(1, w - 1))
            d.line([(x, bar_h), (x, h - 1)], fill=(v, v, v))
        _label(d, w, h, text, (255, 255, 255), y_frac=0.68, bg=(0, 0, 0))
        return img
    img = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(img)
    line = max(1, min(w, h) // 540)
    for i in range(1, 10):
        x, y = i * w // 10, i * h // 10
        d.line([(x, 0), (x, h - 1)], fill=(90, 90, 90), width=line)
        d.line([(0, y), (w - 1, y)], fill=(90, 90, 90), width=line)
    r = h // 2 - 1
    d.ellipse([w // 2 - r, h // 2 - r, w // 2 + r, h // 2 + r], outline=(0, 200, 255), width=2 * line)
    c = min(w, h) // 20
    d.line([(w // 2 - c, h // 2), (w // 2 + c, h // 2)], fill=(255, 255, 255), width=2 * line)
    d.line([(w // 2, h // 2 - c), (w // 2, h // 2 + c)], fill=(255, 255, 255), width=2 * line)
    m = min(w, h) // 12
    for (cx, cy, sx, sy) in ((0, 0, 1, 1), (w - 1, 0, -1, 1), (0, h - 1, 1, -1), (w - 1, h - 1, -1, -1)):
        d.line([(cx, cy), (cx + sx * m, cy)], fill=(255, 64, 64), width=4 * line)
        d.line([(cx, cy), (cx, cy + sy * m)], fill=(255, 64, 64), width=4 * line)
    d.rectangle([0, 0, w - 1, h - 1], outline=(255, 255, 255), width=2 * line)   # the very edge pixels
    _label(d, w, h, text, (255, 255, 255), y_frac=0.40, bg=(0, 0, 0))
    return img


def ensure_file(kind, width, height, label, out_dir):
    """Write the pattern (once per kind/size/label) and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    safe = "".join(ch if ch.isalnum() else "_" for ch in label)[:40]
    path = os.path.join(out_dir, f"{kind}_{int(width)}x{int(height)}_{safe}_v{VERSION}.png")
    if not os.path.exists(path):
        tmp = path + ".tmp.png"
        make(kind, width, height, label).save(tmp)
        os.replace(tmp, path)
    return path
