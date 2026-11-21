# SPDX-License-Identifier: MPL-2.0
"""The displays (monitors) of this computer, asked for again and again.

The app asks which displays there are every few seconds (the display check:
a projector unplugged during a show) and when its window moves. The library
it used for that, screeninfo (0.8.1), leaks a little on every call:

- Windows: each call takes a device context of the whole desktop (GetDC)
  and calls ReleaseDC with one argument instead of two, so the context is
  never given back: one more GDI object per call (screeninfo issue #73,
  measured there in Task Manager's "GDI objects" column). A process may
  hold 10,000 GDI objects; at one check every 3 seconds that is about 8
  hours, and after that drawing fails (the reporter's windows stopped
  repainting).
- Linux (xrandr): each call defines its ctypes structures anew, and ctypes
  keeps every pointer type made from them, so they are never freed (here:
  about 22 KB per call, 6.7 MB after 300 calls).

monitors() gives the same screeninfo Monitor objects without that:

- Windows: Windows is asked directly (EnumDisplayMonitors, GetMonitorInfoW,
  GetDeviceCaps), with the structures defined once and the device context
  released. The first answer is compared with screeninfo's; if they differ
  in anything, screeninfo is used from then on (and the difference logged).
- Linux and others: screeninfo, then the pointer types it made are dropped
  from ctypes' cache, which frees its structures.

No UI here.
"""
from __future__ import annotations

import ctypes
import logging
import sys

import screeninfo
from screeninfo.common import Monitor

log = logging.getLogger("visualdeck.displays")

HORZSIZE, VERTSIZE = 4, 6                 # GetDeviceCaps: the size in mm
PROCESS_PER_MONITOR_DPI_AWARE = 2         # as screeninfo sets it, so the units stay the same

_api = None
_own = sys.platform == "win32"            # ask Windows directly (until the cross-check disagrees)
_checked = False


def signature(mons):
    """What is compared: everything screeninfo reports."""
    return [(m.x, m.y, m.width, m.height, m.width_mm, m.height_mm, m.name, m.is_primary) for m in mons or ()]


def _windows_api():
    """user32/gdi32 functions with their argument types, and the structures, made once."""
    global _api
    if _api is not None:
        return _api
    from ctypes import wintypes

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                    ("dwFlags", wintypes.DWORD), ("szDevice", wintypes.WCHAR * 32)]

    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                              wintypes.LPARAM)
    user32 = ctypes.WinDLL("user32")      # own instances: the argument types set here touch nobody else
    gdi32 = ctypes.WinDLL("gdi32")
    user32.GetDC.argtypes, user32.GetDC.restype = [wintypes.HWND], wintypes.HDC
    user32.ReleaseDC.argtypes, user32.ReleaseDC.restype = [wintypes.HWND, wintypes.HDC], ctypes.c_int
    user32.EnumDisplayMonitors.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT), proc, wintypes.LPARAM]
    user32.EnumDisplayMonitors.restype = wintypes.BOOL
    user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MONITORINFOEXW)]
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    gdi32.GetDeviceCaps.argtypes, gdi32.GetDeviceCaps.restype = [wintypes.HDC, ctypes.c_int], ctypes.c_int
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)  # set already: no effect
    except (OSError, AttributeError):
        pass
    _api = {"user32": user32, "gdi32": gdi32, "proc": proc, "info": MONITORINFOEXW}
    return _api


def enumerate_windows(api):
    """Monitor objects from Windows (api: _windows_api(), or a stand-in in tests).
    The same values screeninfo's Windows enumerator reports."""
    user32, gdi32, info_t = api["user32"], api["gdi32"], api["info"]
    found = []

    def callback(hmon, hdc, rect, _data):
        info = info_t()
        info.cbSize = ctypes.sizeof(info_t)
        name = info.szDevice if user32.GetMonitorInfoW(hmon, ctypes.byref(info)) else None
        r = rect.contents
        found.append(Monitor(x=r.left, y=r.top, width=r.right - r.left, height=r.bottom - r.top,
                             width_mm=gdi32.GetDeviceCaps(hdc, HORZSIZE),
                             height_mm=gdi32.GetDeviceCaps(hdc, VERTSIZE),
                             name=name, is_primary=(r.left == 0 and r.top == 0)))
        return 1

    cb = api["proc"](callback)            # kept alive for the call
    dc = user32.GetDC(None)               # the desktop: each monitor's own context is passed to the callback
    try:
        user32.EnumDisplayMonitors(dc, None, cb, 0)
    finally:
        if dc:
            user32.ReleaseDC(None, dc)
    return found


def _drop_pointer_types():
    """Free the ctypes structures screeninfo made in this call (see the top)."""
    cache = getattr(ctypes, "_pointer_type_cache", None)
    if not isinstance(cache, dict):
        return
    for k in list(cache):
        if isinstance(k, type) and str(getattr(k, "__module__", "")).startswith("screeninfo."):
            cache.pop(k, None)


def _screeninfo():
    try:
        return list(screeninfo.get_monitors())      # looked up now: tests stand in for it
    finally:
        _drop_pointer_types()


def monitors():
    """The displays now, as screeninfo.get_monitors() reports them (raises as it does)."""
    global _own, _checked
    if _own:
        try:
            own = enumerate_windows(_windows_api())
        except Exception as e:  # noqa: BLE001
            log.info(f"Asking Windows for the displays failed ({e}); using screeninfo")
            _own = False
        else:
            if own and _checked:
                return own
            ref = _screeninfo()
            if own and signature(own) != signature(ref):
                log.info(f"Displays from Windows {signature(own)} differ from screeninfo's {signature(ref)}; "
                         f"using screeninfo")
                _own = False
            _checked = _checked or bool(own)
            return ref
    return _screeninfo()
