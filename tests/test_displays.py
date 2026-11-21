# SPDX-License-Identifier: MPL-2.0
"""Asking for the displays again and again without leaking (displays.py)."""
import collections
import ctypes
import gc
import os
from ctypes import wintypes

import pytest
from screeninfo.common import Monitor

import displays


class FakeUser32:
    """Stands in for user32: two monitors, each with its own device context."""

    def __init__(self, rects, names=("\\\\.\\DISPLAY1", "\\\\.\\DISPLAY2")):
        self.rects, self.names = rects, names
        self.got, self.released = [], []

    def GetDC(self, hwnd):
        assert hwnd is None
        self.got.append(1000 + len(self.got))
        return self.got[-1]

    def ReleaseDC(self, hwnd, hdc):
        self.released.append((hwnd, hdc))
        return 1

    def EnumDisplayMonitors(self, hdc, clip, proc, data):
        assert hdc in self.got and clip is None
        for i, (l, t, r, b) in enumerate(self.rects):
            rect = wintypes.RECT(l, t, r, b)
            proc(i + 1, 50 + i, ctypes.pointer(rect), data)
        return 1

    def GetMonitorInfoW(self, hmon, ref):
        info = ref._obj
        assert info.cbSize == ctypes.sizeof(type(info))
        info.szDevice = self.names[hmon - 1]
        return 1


class FakeGdi32:
    @staticmethod
    def GetDeviceCaps(hdc, what):
        return {(50, 4): 600, (50, 6): 340, (51, 4): 520, (51, 6): 320}[(hdc, what)]


def fake_api(rects):
    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                    ("dwFlags", wintypes.DWORD), ("szDevice", wintypes.WCHAR * 32)]
    proc = ctypes.CFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                            wintypes.LPARAM)
    return {"user32": FakeUser32(rects), "gdi32": FakeGdi32(), "proc": proc, "info": MONITORINFOEXW}


RECTS = [(0, 0, 1920, 1080), (1920, 0, 3200, 1024)]
EXPECTED = [Monitor(x=0, y=0, width=1920, height=1080, width_mm=600, height_mm=340,
                    name="\\\\.\\DISPLAY1", is_primary=True),
            Monitor(x=1920, y=0, width=1280, height=1024, width_mm=520, height_mm=320,
                    name="\\\\.\\DISPLAY2", is_primary=False)]


def test_windows_enumeration_reports_what_screeninfo_does_and_gives_the_context_back():
    api = fake_api(RECTS)
    for _ in range(5):
        assert displays.enumerate_windows(api) == EXPECTED
    u = api["user32"]
    assert u.released == [(None, dc) for dc in u.got], "every desktop context is released, with both arguments"


@pytest.fixture
def windows(monkeypatch):
    calls = []

    def reference():
        calls.append(1)
        return list(EXPECTED)
    monkeypatch.setattr(displays.screeninfo, "get_monitors", reference)
    monkeypatch.setattr(displays, "_own", True)
    monkeypatch.setattr(displays, "_checked", False)
    monkeypatch.setattr(displays, "_api", fake_api(RECTS))
    return calls


def test_on_windows_screeninfo_is_asked_once_to_check(windows):
    assert displays.monitors() == EXPECTED and windows == [1]
    for _ in range(10):
        assert displays.monitors() == EXPECTED
    assert windows == [1], "after the first answer agrees, only Windows itself is asked"


def test_on_windows_a_different_answer_falls_back_to_screeninfo(windows, monkeypatch, caplog):
    monkeypatch.setattr(displays, "_api", fake_api([(0, 0, 1280, 720), (1280, 0, 2560, 1024)]))  # other units
    with caplog.at_level("INFO"):
        assert displays.monitors() == EXPECTED
    assert "differ from screeninfo's" in caplog.text and displays._own is False
    displays.monitors()
    assert windows == [1, 1]


def test_on_windows_a_failure_falls_back_to_screeninfo(windows, monkeypatch):
    def broken(api):
        raise OSError("no user32")
    monkeypatch.setattr(displays, "enumerate_windows", broken)
    assert displays.monitors() == EXPECTED and displays._own is False


def test_the_app_asks_through_this_module():
    import control_interface as ci_mod
    import render_engine as rem
    assert ci_mod.get_monitors is displays.monitors and rem.get_monitors is displays.monitors


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs an X display")
def test_on_linux_repeated_calls_free_screeninfos_structures(monkeypatch):
    monkeypatch.setattr(displays, "_own", False)

    def fields():
        gc.collect()
        return collections.Counter(type(o).__name__ for o in gc.get_objects())["CField"]
    try:
        displays.monitors()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no monitors here: {e}")
    before = fields()
    for _ in range(100):
        displays.monitors()
    assert fields() - before < 10, "screeninfo's per-call ctypes structures are freed"
