"""Decoding test (Preferences > GPU Hardware Acceleration > Test with a Video...):
# SPDX-License-Identifier: MPL-2.0
the player's own code path in software and with each hardware decoder, and
the recommendation (the fastest that was really used)."""
import os
import time

import pytest

import decode_test as dt


def test_software_decoding_is_measured(fixtures):
    r = dt.run_one(fixtures["counting"], None, seconds=0.5)
    assert r["ok"] and r["fps"] and r["fps"] > 0 and r["device"] == "software" and r["engaged"] is False
    assert r["codec"] == "h264" and r["size"] and r["pix_fmt"]


def test_a_missing_hardware_decoder_is_reported_not_recommended(fixtures, monkeypatch):
    # A decoder this computer's FFmpeg does not have (d3d11va works on Windows, so
    # it cannot be the missing one there; videotoolbox is macOS only).
    missing = next((d for d in ("videotoolbox", "d3d11va", "vaapi") if d not in dt.available_devices()), None)
    if missing is None:
        pytest.skip("every candidate decoder is in this FFmpeg")
    monkeypatch.setattr(dt, "available_devices", lambda: [missing])
    rs = dt.run_all(fixtures["counting"], seconds=0.3)
    assert [r["device"] for r in rs] == ["software", missing]
    assert rs[1]["engaged"] is not True
    lines, rec = dt.summary(rs, 25)
    assert rec == "software" and len(lines) >= 2


def test_summary_recommends_the_fastest_used_and_warns_when_too_slow():
    rs = [{"device": "software", "ok": True, "fps": 19.9, "engaged": False, "error": None},
          {"device": "d3d11va", "ok": True, "fps": 120.0, "engaged": True, "error": None},
          {"device": "cuda", "ok": True, "fps": 300.0, "engaged": False, "error": "not used"},
          {"device": "dxva2", "ok": False, "fps": None, "engaged": None, "error": "cannot open"}]
    lines, rec = dt.summary(rs, 25)
    assert rec == "d3d11va" and any("(not used)" in x for x in lines) and any("cannot open" in x for x in lines)
    lines, rec = dt.summary(rs[:1], 25)
    assert rec == "software" and "below the clip's 25 fps" in lines[-1]


def test_dialog_runs_and_offers_the_result(fixtures, monkeypatch):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import control_interface as ci_mod
    from media_manager import MediaManager
    monkeypatch.setattr(dt, "available_devices", lambda: [])
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    ci = ci_mod.ControlInterface(MediaManager())
    root = tk.Tk()
    root.withdraw()
    try:
        ci.root = root
        var = tk.StringVar(value="d3d11va")
        win = ci._open_decode_test(root, var, path=fixtures["counting"])
        t0 = time.monotonic()
        while "recommended" not in win._vd and time.monotonic() - t0 < 30:
            root.update()
            time.sleep(0.05)
        assert win._vd["recommended"] == "auto" and "software" in win._vd["out"].cget("text")
        win._vd["use"].invoke()
        assert var.get() == "auto"
    finally:
        ci.proxy_executor.shutdown(wait=False)
        root.destroy()
