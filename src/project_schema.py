# SPDX-License-Identifier: MPL-2.0
"""Show-file schema versions and migration.

Version 1: every show file written before schema versioning existed (no
           'schema_version' key; app 'version' 2.8.x).
Version 2: adds 'schema_version', 'media_info' (relink hints) and
           'cue_settings' (per-cue transition, notes, follow).

Rules:
- Loading never fails because of the schema: older files are upgraded in
  memory; files from a newer VisualDeck are loaded as far as understood and
  the operator is told.
- Top-level keys this version does not know are kept and written back on
  save, so opening and saving a newer show here does not silently drop data.
"""
from __future__ import annotations

import copy

CURRENT = 2
KNOWN_KEYS = {
    "version", "schema_version", "settings", "grid", "cues", "layers", "project_media", "media_info",
    "outputs", "transitions", "presentation", "media_cells", "cue_settings", "go_at_end", "schedule",
    "autosave_of", "autosaved_at",     # only in autosave files; never kept in a saved show
}
GO_AT_END = ("wrap", "stop")   # GO on the last cue with media: back to the first cue, or stay
TRANSITION_TYPES = ("None", "Fade", "Smooth Fade", "Dip to Black")
# How each fade type moves: see render_engine.transition_alphas
TRANSITION_CURVES = {"Fade": "linear", "Smooth Fade": "smooth", "Dip to Black": "dip"}
FOLLOW_MODES = ("after", "end")


def migrate(data: dict):
    """Return (upgraded copy of data, info). info: {'from': n, 'newer': bool, 'notes': [...]}."""
    data = copy.deepcopy(data) if isinstance(data, dict) else {}
    try:
        found = int(data.get("schema_version", 1))
    except (TypeError, ValueError):
        found = 1
    info = {"from": found, "newer": found > CURRENT, "notes": []}
    if found < 2:
        data.setdefault("media_info", {})
        data.setdefault("cue_settings", {})
        info["notes"].append("upgraded from schema 1 (no relink hints or cue settings yet)")
    data["cue_settings"] = clean_cue_settings(data.get("cue_settings"))
    data["schema_version"] = max(found, CURRENT)
    return data, info


def unknown_keys(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in KNOWN_KEYS}


def clean_cue_settings(raw) -> dict:
    """Validate per-cue settings; drop anything malformed instead of failing."""
    out = {}
    if not isinstance(raw, dict):
        return out
    for col, s in raw.items():
        if not isinstance(s, dict):
            continue
        try:
            int(col)
        except (TypeError, ValueError):
            continue
        clean = {}
        if s.get("transition") in TRANSITION_TYPES:
            clean["transition"] = s["transition"]
        dur = s.get("duration")
        if isinstance(dur, (int, float)) and not isinstance(dur, bool) and 0 <= dur <= 600:
            clean["duration"] = float(dur)
        notes = s.get("notes")
        if isinstance(notes, str) and notes.strip():
            clean["notes"] = notes[:5000]
        f = s.get("follow")
        if isinstance(f, dict) and f.get("mode") in FOLLOW_MODES:
            follow = {"mode": f["mode"]}
            if f["mode"] == "after":
                secs = f.get("seconds")
                if isinstance(secs, (int, float)) and not isinstance(secs, bool) and 0 <= secs <= 86400:
                    follow["seconds"] = float(secs)
                else:
                    follow = None
            if follow:
                clean["follow"] = follow
        num = clean_cue_number(s.get("number"))
        if num:
            clean["number"] = num
        import timecode
        tc = timecode.clean(s.get("timecode")) if isinstance(s.get("timecode"), str) else None
        if tc:
            clean["timecode"] = tc
        if s.get("ready") is True:
            clean["ready"] = True                          # Keep Ready: always preloaded
        if clean:
            out[str(int(col))] = clean
    return out


def clean_cue_number(raw):
    """A cue number as the operator types it ("12", "12.5", "A3"), or None.
    Letters, digits, '.', '-' and '_' only, at most 16 characters."""
    if raw is None or isinstance(raw, bool):
        return None
    s = str(raw).strip()
    if not s or len(s) > 16 or not all(ch.isalnum() or ch in "._-" for ch in s):
        return None
    return s


def remap_cue_keys(settings: dict, col_map) -> dict:
    """Move per-cue entries with their cues (col_map returns None for a deleted cue)."""
    out = {}
    for col, s in settings.items():
        new = col_map(int(col))
        if new is not None:
            out[str(new)] = s
    return out
