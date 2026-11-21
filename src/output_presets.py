# SPDX-License-Identifier: MPL-2.0
"""Output presets: named output setups kept on this computer.

A preset holds what the output controls set: the output (one display, or
Multiple Displays with each layer's display), and each display's corner pin,
edge blend and mesh warp. It also records the displays it was made with (their sizes,
in order), so applying it to a different set of displays warns first.

Presets are a machine setting, like Preferences: ~/.visualdeck/
output_presets.json, {"version": 1, "presets": {name: preset}}. Applying one
copies its settings into the open show (where they are saved as before) and
changes the output at once, after a confirmation. A file that cannot be read
is moved aside (output_presets.json.bad), never overwritten.

No UI here.
"""
from __future__ import annotations

import json
import logging
import os

import edge_blend
import mesh_warp
import span as span_mod
import output_warp
from project_io import atomic_write_json

log = logging.getLogger("visualdeck.output_presets")

FILE_NAME = "output_presets.json"
MAX_NAME = 40
MAX_PRESETS = 50


def clean_name(name):
    """A preset name: stripped, at most MAX_NAME characters, or None if empty."""
    name = " ".join(str(name or "").split())[:MAX_NAME]
    return name or None


def _display_index(v):
    if isinstance(v, bool):
        return None
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    return i if 0 <= i <= 63 else None


def capture(mode, layer_screens, warps, blends, monitors, meshes=None, span=None):
    """A preset from the app's current output state. `monitors` are objects
    with width and height (screeninfo monitors)."""
    return clean({
        "mode": mode,
        "layer_screens": dict(layer_screens or {}),
        "warps": dict(warps or {}),
        "blends": dict(blends or {}),
        "meshes": dict(meshes or {}),
        "span": span,
        "displays": [[int(m.width), int(m.height)] for m in monitors or []],
    })


def clean(raw):
    """A valid preset (malformed parts dropped), or None if `raw` is not one."""
    if not isinstance(raw, dict) or not isinstance(raw.get("mode"), str):
        return None
    mode = raw["mode"]
    if mode not in ("None", "Multiple", "Span") and (not mode.startswith("Display ") or _display_index(mode[8:]) is None):
        return None
    layer_screens = {}
    for row, scr in (raw.get("layer_screens") or {}).items() if isinstance(raw.get("layer_screens"), dict) else ():
        r, s = _display_index(row), _display_index(scr)
        if r is not None and s is not None:
            layer_screens[str(r)] = s
    warps = {}
    for disp, corners in (raw.get("warps") or {}).items() if isinstance(raw.get("warps"), dict) else ():
        c = output_warp.clean(corners) if _display_index(disp) is not None else None
        if c is not None:
            warps[str(_display_index(disp))] = [list(p) for p in c]
    blends = {}
    for disp, b in (raw.get("blends") or {}).items() if isinstance(raw.get("blends"), dict) else ():
        cb = edge_blend.clean(b) if _display_index(disp) is not None else None
        if cb is not None:
            blends[str(_display_index(disp))] = dict(zip(edge_blend.SIDES + ("curve", "gamma"), cb))
    meshes = {}
    for disp, m in (raw.get("meshes") or {}).items() if isinstance(raw.get("meshes"), dict) else ():
        cm = mesh_warp.clean(m) if _display_index(disp) is not None else None
        if cm is not None:
            meshes[str(_display_index(disp))] = mesh_warp.as_dict(cm)
    displays = []
    for d in raw.get("displays") or [] if isinstance(raw.get("displays"), list) else []:
        try:
            w, h = int(d[0]), int(d[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if 0 < w <= 32768 and 0 < h <= 32768:
            displays.append([w, h])
    sp = span_mod.clean(raw.get("span"))
    if mode == "Span" and sp is None:
        mode = "None"                                  # a span preset without its span: no output
    return {"mode": mode, "layer_screens": layer_screens, "warps": warps, "blends": blends, "meshes": meshes,
            "span": sp, "displays": displays}


def used_displays(preset):
    """Display indices the preset refers to."""
    used = set()
    if preset["mode"].startswith("Display "):
        used.add(int(preset["mode"][8:]))
    if preset["mode"] == "Multiple":
        used.update(preset["layer_screens"].values())
    used.update(int(k) for k in preset["warps"])
    used.update(int(k) for k in preset["blends"])
    used.update(int(k) for k in preset.get("meshes", {}))
    if preset.get("span"):
        used.update(preset["span"]["displays"])
    return used


def mismatches(preset, monitors):
    """Reasons the preset may not fit this computer's displays (empty: it fits)."""
    here = [[int(m.width), int(m.height)] for m in monitors or []]
    out = []
    missing = sorted(i for i in used_displays(preset) if i >= len(here))
    if missing:
        out.append("it uses " + ", ".join(f"Display {i}" for i in missing) +
                   f", which this computer does not have ({len(here)} display{'s' if len(here) != 1 else ''})")
    saved = preset.get("displays") or []
    if saved and len(saved) != len(here):
        out.append(f"it was saved with {len(saved)} display{'s' if len(saved) != 1 else ''}, "
                   f"this computer has {len(here)}")
    for i, (a, b) in enumerate(zip(saved, here)):
        if a != b and i in used_displays(preset):
            out.append(f"Display {i} was {a[0]}×{a[1]}, it is {b[0]}×{b[1]} now "
                       "(corner pins, blends and meshes are relative, so they stretch)")
    return out


def summary(preset):
    """One line for the preset list, e.g. "Display 1 · pin on 1 · blend on 0, 1"."""
    if preset["mode"] == "Span" and preset.get("span"):
        sp = preset["span"]
        first = f"Span of {', '.join(map(str, sp['displays']))} ({sp['overlap']} px overlap)"
        if sp.get("columns"):
            first += f", {sp['columns']} per row"
    else:
        first = None
    parts = [first or ("Multiple Displays" if preset["mode"] == "Multiple" else
                       ("No output" if preset["mode"] == "None" else preset["mode"]))]
    if preset["warps"]:
        parts.append("pin on " + ", ".join(sorted(preset["warps"], key=int)))
    if preset["blends"]:
        parts.append("blend on " + ", ".join(sorted(preset["blends"], key=int)))
    if preset.get("meshes"):
        parts.append("mesh on " + ", ".join(sorted(preset["meshes"], key=int)))
    return " · ".join(parts)


def load(path):
    """(presets, note, writable): {name: preset} from `path`. A missing file
    is empty. An unreadable one is moved aside to <path>.bad and reads as
    empty; if it cannot be moved, `writable` is False and the caller must not
    save (that would overwrite it). `note` says what happened, or is None."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        presets = data.get("presets") if isinstance(data, dict) else None
        if not isinstance(presets, dict):
            raise ValueError("no 'presets' object")
    except FileNotFoundError:
        return {}, None, True
    except (OSError, ValueError) as e:
        bad = path + ".bad"
        try:
            os.replace(path, bad)
        except OSError as e2:
            log.error(f"Output presets: {path} could not be read ({e}) or moved aside ({e2})")
            return {}, f"{path} could not be read ({e}); presets cannot be saved until it is fixed or removed", False
        log.warning(f"Output presets: {path} could not be read ({e}); moved to {bad}")
        return {}, f"{path} could not be read ({e}); it was moved to {bad}", True
    out = {}
    for name, raw in presets.items():
        n, p = clean_name(name), clean(raw)
        if n and p and len(out) < MAX_PRESETS:
            out[n] = p
        else:
            log.warning(f"Output presets: skipped an unreadable preset {name!r}")
    return out, None, True


def save(path, presets):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    atomic_write_json(path, {"version": 1, "presets": presets}, keep_backup=False, indent=2, sort_keys=True)
