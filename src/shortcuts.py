# SPDX-License-Identifier: MPL-2.0
"""Keyboard shortcuts: one table drives the key bindings, the menu labels and
the Help page, so they cannot drift apart.

Every shortcut except F1 is ignored while the keyboard focus is in a text
field (cue names, notes, numbers), where Tk uses several Ctrl keys for editing
(Ctrl+B cursor left, Ctrl+H delete, Ctrl+E end of line, Ctrl+T swap letters).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shortcut:
    action: str          # ControlInterface handler key (see ControlInterface._shortcut_actions)
    sequences: tuple     # Tk event sequences
    label: str           # as shown in menus and Help
    description: str
    group: str
    in_text_fields: bool = False   # True only for keys that never mean anything while typing


def _ctrl(letter):
    return (f"<Control-{letter.lower()}>", f"<Control-{letter.upper()}>")


def _ctrl_shift(letter):
    return (f"<Control-Shift-{letter.upper()}>", f"<Control-Shift-{letter.lower()}>")


SHORTCUTS = (
    Shortcut("go", ("<space>",), "Space", "GO: next cue (in presentation mode: start/stop)", "Show"),
    Shortcut("back", ("<Shift-space>",), "Shift+Space", "BACK: previous cue with media", "Show"),
    Shortcut("blackout", _ctrl("b"), "Ctrl+B", "Blackout: fade the output to black", "Show"),
    Shortcut("update", _ctrl("u"), "Ctrl+U", "Update Output: apply edits to the live cue", "Show"),
    *(Shortcut(f"goto_{n}", (f"<Control-Key-{n}>",), f"Ctrl+{n}", f"Go to cue {n}", "Show") for n in range(1, 10)),
    Shortcut("cue_settings", _ctrl("e"), "Ctrl+E", "Cue Settings of the live cue", "Show"),
    Shortcut("presentation", _ctrl("p"), "Ctrl+P", "Presentation mode on/off", "Show"),
    Shortcut("test_pattern", _ctrl("t"), "Ctrl+T", "Test pattern (grid) on/off", "Output"),
    Shortcut("health", _ctrl("h"), "Ctrl+H", "Output Health window", "Output"),
    Shortcut("new", _ctrl("n"), "Ctrl+N", "New show", "File"),
    Shortcut("open", _ctrl("o"), "Ctrl+O", "Open show", "File"),
    Shortcut("save", _ctrl("s"), "Ctrl+S", "Save show", "File"),
    Shortcut("save_as", _ctrl_shift("s"), "Ctrl+Shift+S", "Save show as…", "File"),
    Shortcut("relink", _ctrl_shift("r"), "Ctrl+Shift+R", "Relink missing media", "File"),
    Shortcut("preferences", ("<Control-comma>",), "Ctrl+,", "Preferences", "File"),
    Shortcut("undo", _ctrl("z"), "Ctrl+Z", "Undo the last change to the show", "Edit"),
    Shortcut("redo", _ctrl_shift("z") + _ctrl("y"), "Ctrl+Shift+Z", "Redo (also Ctrl+Y)", "Edit"),
    Shortcut("add_media", _ctrl("i"), "Ctrl+I", "Add media to the show", "Insert"),
    Shortcut("add_cue", _ctrl_shift("c"), "Ctrl+Shift+C", "Add a cue (column)", "Insert"),
    Shortcut("add_layer", _ctrl_shift("l"), "Ctrl+Shift+L", "Add a layer (row)", "Insert"),
    Shortcut("lock", _ctrl("l"), "Ctrl+L", "Lock or unlock editing (GO, BACK and Blackout still work)", "Show"),
    Shortcut("help", ("<F1>",), "F1", "Help", "Help", in_text_fields=True),
)

BY_ACTION = {s.action: s for s in SHORTCUTS}


def label(action):
    """Menu accelerator text for an action ('' if it has none)."""
    s = BY_ACTION.get(action)
    return s.label if s else ""
