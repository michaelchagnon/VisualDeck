# SPDX-License-Identifier: MPL-2.0
"""Crash-safe file writing for show files and settings.

A show file is the operator's work; losing it during a save is not acceptable.
Writes here never truncate the existing file:

1. the content is serialized completely in memory first (so serialization
   errors happen before the disk is touched),
2. written to a temporary file in the same directory and flushed + fsynced,
3. the previous version is kept as ``<name>.bak``,
4. the temporary file atomically replaces the target (``os.replace``).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile

log = logging.getLogger("visualdeck.io")


def atomic_write_text(path: str, text: str, keep_backup: bool = True, encoding: str = "utf-8") -> None:
    """Atomically replace `path` with `text`. Raises OSError on failure; the
    existing file is left untouched in that case."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".vd_save_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if keep_backup and os.path.exists(path):
            try:
                shutil.copy2(path, path + ".bak")
            except OSError as e:  # a missing backup must not block the save itself
                log.warning("Could not write backup %s.bak: %s", path, e)
        os.replace(tmp, path)
        tmp = None
        _fsync_directory(directory)
    finally:
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass


def atomic_write_json(path: str, data, keep_backup: bool = True, **dump_kwargs) -> None:
    """Serialize `data` fully, then atomically write it. Raises TypeError /
    ValueError for unserializable data before any file is touched."""
    dump_kwargs.setdefault("indent", 4)
    text = json.dumps(data, **dump_kwargs)
    atomic_write_text(path, text, keep_backup=keep_backup)


def _fsync_directory(directory: str) -> None:
    """Persist the rename on POSIX filesystems (no-op where unsupported)."""
    if os.name != "posix":
        return
    try:
        dfd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)
