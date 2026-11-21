# SPDX-License-Identifier: MPL-2.0
"""Offscreen composite of a cue, for the edited-cue preview (blind editing).

The output preview shows what is on the output (the program). When the live
cue has edits that are not applied yet, this module composites the edited cue
so it can be checked before pressing Update Output.

It runs on a worker thread and never touches the render engine: videos are
shown by a representative frame (render_engine.video_still: the first frame
unless it is black, as in a fade-in), stills are loaded here.
The layout follows BaseRenderEngine.render_layers_to_image exactly (tested
against it), so what you see matches the output's placement.
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict

from PIL import Image

log = logging.getLogger("visualdeck.preview")

VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.mpg', '.mpeg', '.m4v', '.ts')


class StillSource:
    """Stills of media files (a representative frame of a video), cached by
    (path, mtime, size).
    Use from one thread only (the preview worker)."""

    def __init__(self, max_items=48, max_side=1280, live_still=None):
        self._cache = OrderedDict()
        self.max_items = max_items
        self.max_side = max_side   # stills are reduced to this; previews are small
        self.live_still = live_still   # path -> RGBA PIL image for a live source (it is never opened here)

    def get(self, path):
        import live_source
        if live_source.is_live(path):
            img = self.live_still(path) if self.live_still else None
            if img is None:
                return None, 1.0
            factor = max(1.0, max(img.size) / float(self.max_side))
            if factor > 1.0:
                img = img.resize((max(1, round(img.width / factor)), max(1, round(img.height / factor))),
                                 Image.BILINEAR)
            return img, factor
        try:
            st = os.stat(path)
        except OSError:
            return None, 1.0
        key = (path, st.st_mtime, st.st_size)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        img = self._load(path)
        factor = 1.0
        if img is not None and max(img.size) > self.max_side:
            factor = max(img.size) / float(self.max_side)
            img = img.resize((max(1, round(img.width / factor)), max(1, round(img.height / factor))), Image.BILINEAR)
        hit = (img, factor)   # factor: how much the still was reduced (layout uses original size)
        self._cache[key] = hit
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return hit

    @staticmethod
    def _load(path):
        if os.path.splitext(path)[1].lower() in VIDEO_EXTS:
            from render_engine import video_still     # the video's own color matrix; not a black fade-in
            still = video_still(path)
            if still is None:
                log.warning(f"Edited-cue preview: no picture for {os.path.basename(path)}")
                return None
            return still if still.mode == 'RGBA' else still.convert('RGBA')
        try:
            with Image.open(path) as im:
                return im.convert('RGBA')
        except Exception as e:  # noqa: BLE001
            log.warning(f"Edited-cue preview: cannot load {os.path.basename(path)}: {e}")
            return None


def compose(layers, dimensions, reference_size, source):
    """RGBA image of `layers` at `dimensions`, laid out as on an output of
    `reference_size` (the output monitor's pixel size)."""
    w, h = dimensions
    ref_w, ref_h = reference_size
    preview_scale = min(w / ref_w, h / ref_h, 1.0)
    target = Image.new('RGBA', (w, h), (0, 0, 0, 255))
    for layer in sorted(layers.values(), key=lambda l: l.get('order', 0)):
        path = layer.get('path')
        if not path:
            continue
        img, reduced = source.get(path)
        if img is None:
            continue
        pos = layer.get('position') or {}
        sx = pos.get('scale_x', 1.0) * preview_scale * reduced
        sy = pos.get('scale_y', 1.0) * preview_scale * reduced
        size = (int(img.width * sx), int(img.height * sy))
        if size[0] <= 0 or size[1] <= 0:
            continue
        pil = img.resize(size, Image.BILINEAR)
        import layer_mask
        from render_engine import adjust_pil, apply_opacity, blend_onto, layer_blend, layer_color, layer_opacity
        # Color and mask before rotation: the mask is in the media's own coordinates.
        pil = layer_mask.apply_pil(adjust_pil(pil, layer_color(layer)), layer_mask.layer_mask(layer))
        import layer_warp
        warp = layer_warp.layer_warp(layer)
        if warp is not None:                      # bent around the same center
            pil = layer_warp.warp_pil(pil, warp)
        rotation = pos.get('rotation', 0)
        if rotation:
            pil = pil.rotate(rotation, expand=True, resample=Image.BICUBIC)
        pil = apply_opacity(pil, layer_opacity(layer))
        x = int(pos.get('x', 0.5) * w - pil.width / 2)
        y = int(pos.get('y', 0.5) * h - pil.height / 2)
        blend_onto(target, pil, (x, y), layer_blend(layer))
    return target
