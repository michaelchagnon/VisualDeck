# SPDX-License-Identifier: MPL-2.0
"""Collect Show: copy a show and all its media into one new folder.

  <destination>/<show name>/<show name>.vjdeck
  <destination>/<show name>/media/<files>
  <destination>/<show name>/effects/<shader effects that are not built in>

The collected show refers to the copies (absolute paths plus show-relative
'rel' hints), so it opens on another machine once the folder is copied there.

Safety rules: never writes into a folder that already has files in it, never
overwrites or deletes anything, checks free space first, writes the show file
last (only if every available file was copied). Missing media is reported and
keeps its old path in the collected show.
"""
from __future__ import annotations

import copy
import os
import shutil

import media_relink

CHUNK = 4 * 1024 * 1024


class CollectError(Exception):
    pass


def human_size(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def plan(media_paths, media_dir):
    """{source: destination} with unique, case-insensitively distinct names."""
    out, taken = {}, set()
    for src in media_paths:
        if src in out:
            continue
        base, ext = os.path.splitext(media_relink.basename(src))
        name, n = base + ext, 2
        while name.lower() in taken:
            name = f"{base} ({n}){ext}"
            n += 1
        taken.add(name.lower())
        out[src] = os.path.join(media_dir, name)
    return out


def check_destination(folder, needed_bytes):
    """Raise CollectError if `folder` cannot receive the collection."""
    if os.path.exists(folder) and (not os.path.isdir(folder) or os.listdir(folder)):
        raise CollectError(f"{folder} already exists and is not empty; choose another place or name.")
    probe = folder
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    free = shutil.disk_usage(probe).free
    if needed_bytes + 50 * 1024 * 1024 > free:
        raise CollectError(f"Not enough free space: {human_size(needed_bytes)} needed, {human_size(free)} free.")


def copy_files(mapping, progress=None, cancel=None):
    """Copy every existing source to its destination in chunks.
    progress(done_bytes, total_bytes, current_name); cancel() -> True stops.
    Returns (copied, missing, canceled). Destinations are never overwritten."""
    present = {s: d for s, d in mapping.items() if os.path.isfile(s)}
    missing = [s for s in mapping if s not in present]
    total = sum(os.path.getsize(s) for s in present)
    done, copied = 0, []
    for src, dst in present.items():
        if cancel and cancel():
            return copied, missing, True
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            raise CollectError(f"{dst} appeared during collection; nothing was overwritten.")
        tmp = dst + ".partial"
        with open(src, "rb") as fi, open(tmp, "xb") as fo:
            while True:
                if cancel and cancel():
                    fo.close()
                    return copied, missing, True      # the .partial file is left, clearly named
                buf = fi.read(CHUNK)
                if not buf:
                    break
                fo.write(buf)
                done += len(buf)
                if progress:
                    progress(done, total, os.path.basename(src))
        shutil.copystat(src, tmp)
        os.replace(tmp, dst)
        copied.append(src)
    return copied, missing, False


def rewrite(data, mapping, show_path):
    """Copy of save data pointing at the collected media."""
    data = copy.deepcopy(data)
    m = lambda p: mapping.get(p, p)  # noqa: E731
    data["project_media"] = [m(p) for p in data.get("project_media", [])]
    for cell in data.get("media_cells", {}).values():
        if isinstance(cell, dict) and cell.get("path"):
            cell["path"] = m(cell["path"])
    info = {}
    for old, entry in (data.get("media_info") or {}).items():
        new = m(old)
        info[new] = media_relink.media_info_for(new, show_path) if os.path.exists(new) else entry
    for new in mapping.values():
        if os.path.exists(new) and new not in info:
            info[new] = media_relink.media_info_for(new, show_path)
    data["media_info"] = info
    return data


def effect_plan(effect_paths, effects_dir):
    """{source: destination} for the show's own shader effects (by file name: names are unique)."""
    return {p: os.path.join(effects_dir, os.path.basename(p)) for p in effect_paths}


def collect(data, mapping, show_path, progress=None, cancel=None, effects=None):
    """Copy the media (and `effects`, {source: destination}: effect files are
    found by name, so the show data does not change for them), then write the
    collected show (worker-thread safe: no UI).
    Returns {'canceled', 'copied', 'missing', 'bytes', 'show', 'effects'}."""
    from project_io import atomic_write_json
    effects = {s: d for s, d in (effects or {}).items() if s not in mapping}
    copied, missing, canceled = copy_files({**mapping, **effects}, progress, cancel)
    nbytes = sum(os.path.getsize((mapping.get(s) or effects[s])) for s in copied)
    media_copied = [s for s in copied if s in mapping]
    result = {"canceled": canceled, "copied": len(media_copied), "missing": missing, "bytes": nbytes,
              "show": None, "effects": len(copied) - len(media_copied)}
    if canceled:
        return result
    out = rewrite(data, {s: mapping[s] for s in media_copied}, show_path)
    atomic_write_json(show_path, out, keep_backup=False, indent=4)
    result["show"] = show_path
    return result


def unique_folder(parent, name):
    """parent/name, or parent/name (2), (3)... : the first that does not exist yet."""
    cand, n = os.path.join(parent, name), 2
    while os.path.exists(cand):
        cand = os.path.join(parent, f"{name} ({n})")
        n += 1
    return cand
