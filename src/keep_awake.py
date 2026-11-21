# SPDX-License-Identifier: MPL-2.0
"""Keeps the computer awake, with its display on, while VisualDeck is open (Windows).

A show machine must not go to sleep or blank its screens during a long cue, a
rehearsal break or an overnight soak test. Windows' SetThreadExecutionState
asks for that without changing any power setting: the request belongs to the
thread that made it and ends when VisualDeck closes (or release() is called),
after which the computer's own settings apply again. Made from the Tk main
thread, which lives as long as the program. Other systems: not done here
(pygame 2.6 keeps the screen saver off while its display is open, by default).

No UI here.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("visualdeck.keep_awake")

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_USER_PRESENT = 0x00000004
WANTED = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
SYSTEM_EXECUTION_STATE = 16          # CallNtPowerInformation: the computer's current requests

_held = False


def _kernel32():
    if os.name != "nt":
        return None
    import ctypes
    k = ctypes.windll.kernel32
    k.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    k.SetThreadExecutionState.restype = ctypes.c_uint32
    return k


def system_state():
    """What Windows reports it is being asked for (all programs together), as a
    set of 'system' / 'display' / 'user present', or None if it cannot tell."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        out = ctypes.c_ulong(0)
        rc = ctypes.windll.powrprof.CallNtPowerInformation(SYSTEM_EXECUTION_STATE, None, 0,
                                                           ctypes.byref(out), ctypes.sizeof(out))
        if rc != 0:
            return None
        return state_names(out.value)
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


def state_names(flags):
    return {n for bit, n in ((ES_SYSTEM_REQUIRED, "system"), (ES_DISPLAY_REQUIRED, "display"),
                             (ES_USER_PRESENT, "user present")) if flags & bit}


def hold():
    """Ask Windows to stay awake with the display on. True if Windows took the
    request; False on other systems or if it refused."""
    global _held
    try:
        k = _kernel32()
        if k is None:
            return False
        if k.SetThreadExecutionState(WANTED) == 0:
            log.warning("Windows did not take the request to keep the computer awake")
            return False
    except Exception as e:  # noqa: BLE001 - never stops start-up
        log.warning(f"Could not ask Windows to keep the computer awake: {e}")
        return False
    _held = True
    now = system_state()
    log.info("Keeping the computer awake with its display on while VisualDeck is open"
             + (f" (Windows reports requests for: {', '.join(sorted(now)) or 'none'})" if now is not None else ""))
    return True


def release():
    """End the request (the computer's own sleep and display settings apply again)."""
    global _held
    if not _held:
        return
    try:
        k = _kernel32()
        if k is not None:
            k.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:  # noqa: BLE001
        pass
    _held = False


def is_held():
    return _held
