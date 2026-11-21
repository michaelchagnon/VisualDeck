# SPDX-License-Identifier: MPL-2.0
"""Undo and redo for show edits.

An entry is the show's grid document before and after one change (see
ControlInterface._undo_snapshot), plus, for changes that move cues or layers
(add, delete, reorder), the maps from old to new column and row indices. The
maps let undo move the live cue's player keys with its cells the same way the
original change did, so undoing never restarts or replaces what the output is
showing.

Changes in quick succession (within COALESCE_S, e.g. dragging a slider that
saves on every step) become one entry, unless either one moved cues or layers.
No UI here.
"""
from __future__ import annotations

import time

LIMIT = 100
COALESCE_S = 0.6


def inverse_map(forward):
    """{new: old} from {old: new} (entries mapped to None are dropped)."""
    return {new: old for old, new in (forward or {}).items() if new is not None}


def map_fn(mapping):
    """A function for ControlInterface._remap_program: index -> mapped index (None if gone)."""
    if mapping is None:
        return None
    return lambda i: mapping.get(i)


class UndoStack:
    def __init__(self, limit=LIMIT, coalesce_s=COALESCE_S, clock=time.monotonic):
        self.limit = limit
        self.coalesce_s = coalesce_s
        self.clock = clock
        self.undo_entries = []
        self.redo_entries = []
        self.baseline = None        # the document as it is now (after the newest entry)

    def reset(self, state):
        self.undo_entries.clear()
        self.redo_entries.clear()
        self.baseline = state

    def record(self, state, label="Edit", rows=None, cols=None):
        """Note that the document is now `state`. Returns True if an entry was
        added or extended. `rows`/`cols`: {old index: new index or None} for a
        change that moved layers/cues."""
        if self.baseline is None:
            self.baseline = state
            return False
        if state == self.baseline:
            return False
        now = self.clock()
        structural = rows is not None or cols is not None
        last = self.undo_entries[-1] if self.undo_entries else None
        if (not structural and last is not None and not last["structural"] and last["label"] == label
                and now - last["t"] <= self.coalesce_s and not self.redo_entries):
            last["after"], last["t"] = state, now
        else:
            self.undo_entries.append({"before": self.baseline, "after": state, "rows": rows, "cols": cols,
                                      "structural": structural, "label": label, "t": now})
            del self.undo_entries[:-self.limit]
        self.redo_entries.clear()
        self.baseline = state
        return True

    def can_undo(self):
        return bool(self.undo_entries)

    def can_redo(self):
        return bool(self.redo_entries)

    def undo_label(self):
        return self.undo_entries[-1]["label"] if self.undo_entries else None

    def redo_label(self):
        return self.redo_entries[-1]["label"] if self.redo_entries else None

    def undo(self):
        """(state to restore, rows map, cols map) or None. The maps take the
        current indices to the restored ones."""
        if not self.undo_entries:
            return None
        e = self.undo_entries.pop()
        self.redo_entries.append(e)
        self.baseline = e["before"]
        return e["before"], (inverse_map(e["rows"]) if e["rows"] is not None else None), \
            (inverse_map(e["cols"]) if e["cols"] is not None else None)

    def redo(self):
        if not self.redo_entries:
            return None
        e = self.redo_entries.pop()
        e["t"] = float("-inf")          # a redone entry never absorbs the next change
        self.undo_entries.append(e)
        self.baseline = e["after"]
        return e["after"], e["rows"], e["cols"]
