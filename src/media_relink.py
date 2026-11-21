# SPDX-License-Identifier: MPL-2.0
"""Missing-media detection and relinking for show files.

Show files store absolute media paths. Since 2026-11 they also store, per
media file, its path relative to the show file and its size in bytes
(`media_info`). That allows two repairs:

1. Automatic: if the absolute path is gone but the relative path (resolved
   against the show file's current folder) exists, use it. This covers a show
   folder copied to another machine or drive together with its media.
2. Search: the operator picks a folder; files are matched by name. When several
   files have the same name, the one whose parent folders best match the
   original path wins; a recorded size must match. Ties are reported as
   ambiguous instead of guessed.

Nothing here touches the UI, so it can run on a worker thread.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

log = logging.getLogger("visualdeck.relink")


def _parts(path: str) -> list[str]:
    """Path components, lower-cased, splitting on both / and \\ (paths in a
    show file may come from Windows or from macOS/Linux)."""
    return [p.lower() for p in path.replace("\\", "/").split("/") if p and not p.endswith(":")]


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


basename = _basename  # file name of a path written on any OS


def media_info_for(path: str, project_path: str | None) -> dict:
    """Portable metadata stored next to an absolute media path in the show file."""
    info = {}
    if project_path:
        try:
            rel = os.path.relpath(path, os.path.dirname(os.path.abspath(project_path)))
            info["rel"] = rel.replace(os.sep, "/")
        except ValueError:
            pass  # different drive on Windows: no relative path
    try:
        info["size"] = os.path.getsize(path)
    except OSError:
        pass
    return info


def find_missing(paths) -> list[str]:
    """Unique paths (in first-seen order) that do not exist on disk."""
    seen, missing = set(), []
    for p in paths:
        if not p or p in seen:
            continue
        seen.add(p)
        if not os.path.exists(p) and not _is_live(p):
            missing.append(p)
    return missing


def _is_live(p):
    import live_source
    return live_source.is_live(p)


def resolve_relative(missing, media_info: dict, project_path: str | None) -> dict:
    """{old_path: new_path} for missing files found at their stored relative
    location next to the show file (and, if recorded, with the same size)."""
    if not project_path:
        return {}
    base = os.path.dirname(os.path.abspath(project_path))
    found = {}
    for old in missing:
        info = media_info.get(old) or {}
        rel = info.get("rel")
        if not rel:
            continue
        candidate = os.path.normpath(os.path.join(base, *rel.split("/")))
        if os.path.isfile(candidate) and _size_ok(candidate, info.get("size")):
            found[old] = candidate
    return found


def _size_ok(path: str, size) -> bool:
    if size is None:
        return True
    try:
        return os.path.getsize(path) == size
    except OSError:
        return False


@dataclass
class SearchResult:
    found: dict = field(default_factory=dict)        # old -> new
    ambiguous: dict = field(default_factory=dict)    # old -> [candidates]
    not_found: list = field(default_factory=list)
    files_scanned: int = 0
    truncated: bool = False                          # hit the file or time limit


def _suffix_score(old: str, candidate: str) -> int:
    """How many trailing parent folders the candidate shares with the original."""
    a, b = _parts(old)[:-1], _parts(candidate)[:-1]
    n = 0
    while n < len(a) and n < len(b) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def search_folder(missing, folder: str, media_info: dict | None = None,
                  max_files: int = 500_000, time_limit: float = 60.0, cancel=None) -> SearchResult:
    """Look for each missing file under `folder` by file name."""
    media_info = media_info or {}
    wanted = {}
    for old in missing:
        wanted.setdefault(_basename(old).lower(), []).append(old)
    candidates = {name: [] for name in wanted}
    res = SearchResult()
    deadline = time.monotonic() + time_limit
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]  # skip hidden/system folders
        for fn in filenames:
            res.files_scanned += 1
            key = fn.lower()
            if key in candidates:
                candidates[key].append(os.path.join(dirpath, fn))
        if res.files_scanned >= max_files or time.monotonic() > deadline or (cancel and cancel()):
            res.truncated = True
            break
    for name, olds in wanted.items():
        for old in olds:
            size = (media_info.get(old) or {}).get("size")
            cands = [c for c in candidates.get(name, []) if _size_ok(c, size)]
            if not cands:
                res.not_found.append(old)
                continue
            scored = sorted(((_suffix_score(old, c), c) for c in cands), reverse=True)
            if len(scored) > 1 and scored[0][0] == scored[1][0]:
                res.ambiguous[old] = [c for s, c in scored if s == scored[0][0]]
            else:
                res.found[old] = scored[0][1]
    return res


def apply_mapping(mapping: dict, grid_cells: dict, project_media: list, media_info: dict):
    """Rewrite paths in place: grid cells, the project media list and media_info.
    Returns the number of grid cells changed."""
    changed = 0
    for key, path in list(grid_cells.items()):
        if path in mapping:
            grid_cells[key] = mapping[path]
            changed += 1
    for i, path in enumerate(project_media):
        if path in mapping:
            project_media[i] = mapping[path]
    # de-duplicate the media list (two missing entries may map to one file)
    seen, dedup = set(), []
    for p in project_media:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    project_media[:] = dedup
    for old, new in mapping.items():
        if old in media_info:
            media_info.setdefault(new, media_info.pop(old))
    return changed


def rebase_rel(rel: str, old_project: str, new_project: str) -> str | None:
    """Re-express a stored relative path when the show is saved somewhere else
    (Save As) while the file itself is missing. None if not expressible."""
    old_dir = os.path.dirname(os.path.abspath(old_project))
    target = os.path.normpath(os.path.join(old_dir, *rel.split("/")))
    try:
        return os.path.relpath(target, os.path.dirname(os.path.abspath(new_project))).replace(os.sep, "/")
    except ValueError:
        return None


def siblings_mapping(chosen_new: str, missing, media_info: dict | None = None) -> dict:
    """After the operator locates one file by hand, look for the other missing
    files (same name, same size if recorded) in that file's folder."""
    media_info = media_info or {}
    folder = os.path.dirname(chosen_new)
    try:
        names = {fn.lower(): fn for fn in os.listdir(folder)}
    except OSError:
        return {}
    found = {}
    for old in missing:
        fn = names.get(_basename(old).lower())
        if not fn:
            continue
        cand = os.path.join(folder, fn)
        if os.path.isfile(cand) and _size_ok(cand, (media_info.get(old) or {}).get("size")):
            found[old] = cand
    return found


def summarize(res: SearchResult, still_missing: int) -> str:
    parts = [f"Relinked {len(res.found)} file{'s' if len(res.found) != 1 else ''}."]
    if res.ambiguous:
        parts.append(f"{len(res.ambiguous)} matched more than one file; select and use Locate to choose.")
    if res.not_found:
        parts.append(f"{len(res.not_found)} not found in that folder.")
    if res.truncated:
        parts.append(f"Search stopped early after {res.files_scanned:,} files; try a smaller folder.")
    if still_missing == 0:
        parts.append("No missing media left.")
    return " ".join(parts)
