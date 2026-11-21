# SPDX-License-Identifier: MPL-2.0
"""DMX (Art-Net / sACN) cue control: packet parsing, the three-channel decoder,
the UDP listener on loopback, the app wiring, settings and the setup dialog.
No lighting console was available: packets are built here to the published
layouts (Art-Net 4 ArtDmx, ANSI E1.31 data packet)."""
import json
import os
import socket
import time

import pytest

import dmx_remote as dr
import health
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401


def send(port, packet):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(packet, ("127.0.0.1", port))
    s.close()


def wait(cond, timeout=2.0):
    t = time.monotonic()
    while time.monotonic() - t < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_artnet_and_sacn_round_trip_and_rejects():
    data = bytes(range(1, 21))
    assert dr.parse_artnet(dr.build_artnet(0, data)) == (0, data)
    assert dr.parse_artnet(dr.build_artnet(300, data))[0] == 300                 # net/sub-net/universe
    assert dr.parse_sacn(dr.build_sacn(7, data)) == (7, data)
    assert dr.parse_sacn(dr.build_sacn(7, data, preview=True)) is None           # preview data
    pkt = bytearray(dr.build_sacn(7, data))
    pkt[125] = 0xDD                                                               # not a DMX start code
    assert dr.parse_sacn(bytes(pkt)) is None
    poll = bytearray(dr.build_artnet(0, data))
    poll[8:10] = b"\x00\x20"                                                      # ArtPoll, not ArtDmx
    for junk in (b"", b"Art-Net\x00", bytes(poll), b"x" * 200, dr.build_artnet(0, data)[:20]):
        assert dr.parse_artnet(junk) is None and dr.parse_sacn(junk) is None


def test_decoder_fires_on_changes_only_after_a_baseline():
    d = dr.Decoder(channel=10)
    frame = lambda cue, go, black: bytes(9) + bytes([cue, go, black]) + bytes(20)  # noqa: E731
    assert d.feed(frame(5, 255, 0)) == [], "what the console already sends does nothing"
    assert d.feed(frame(5, 255, 0)) == []
    assert d.feed(frame(7, 255, 0)) == [("cue", 7)]
    assert d.feed(frame(7, 0, 0)) == [] and d.feed(frame(7, 200, 0)) == [("go", None)]
    assert d.feed(frame(0, 200, 130)) == [("blackout", None)], "cue value 0 does nothing"
    assert d.feed(bytes(5)) == [], "a short frame reads as zeros"
    assert dr.Decoder(channel=999).channel == 510


def test_listener_on_loopback(monkeypatch):
    got = []
    m = dr.DMXRemote(lambda c, a: got.append((c, a)), "artnet", universe=2, channel=1, network=False)
    assert m.start(port=0) and m.listening
    for cue in (0, 3, 3, 4):
        send(m.port, dr.build_artnet(2, bytes([cue, 0, 0])))
    send(m.port, dr.build_artnet(1, bytes([9, 0, 0])))                           # other universe
    send(m.port, dr.build_artnet(2, bytes([4, 255, 0])))
    assert wait(lambda: len(got) == 3), got
    assert got == [("cue", 3), ("cue", 4), ("go", None)] and m.packets == 5
    busy = dr.DMXRemote(lambda *a: None, "artnet", network=False)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    try:
        if os.name != "nt":                      # Windows lets SO_REUSEADDR sockets share a port
            assert not busy.start(port=s.getsockname()[1]) and "cannot listen" in busy.error
    finally:
        s.close()
        busy.stop()
    m.stop()
    assert not m.listening
    sacn = dr.DMXRemote(lambda c, a: got.append((c, a)), "sacn", universe=1, channel=5, network=False)
    assert sacn.start(port=0)
    send(sacn.port, dr.build_sacn(1, bytes(10)))
    send(sacn.port, dr.build_sacn(1, bytes(4) + bytes([12])))
    assert wait(lambda: got[-1] == ("cue", 12))
    sacn.stop()


def test_app_wiring_health_and_settings(blind_ci, monkeypatch, tmp_path):
    ci = blind_ci
    monkeypatch.setattr(dr, "ARTNET_PORT", 0)
    ci.dmx_enabled, ci.dmx_network, ci.dmx_universe, ci.dmx_channel = True, False, 0, 1
    ci._restart_dmx()
    st = ci._dmx_status()
    assert st["listening"] and st["values"] is None and "no data yet" in ci._dmx_text(st)
    port = ci._dmx.port
    send(port, dr.build_artnet(0, bytes([0, 0, 0])))
    send(port, dr.build_artnet(0, bytes([2, 0, 0])))                              # cue select -> second cue
    assert wait(lambda: not ci._remote_queue.empty())
    ci._poll_remote()
    assert ci.transition_thread.started[-1][3] == 1
    assert "2 / 0 / 0" in ci._dmx_text(ci._dmx_status()) and ci._dmx_status()["values"] == [2, 0, 0]
    ci._stop_dmx()
    level, reasons = health.assess({"output": {}, "dmx": {"enabled": True, "listening": False, "error": "port in use"}})
    assert any("DMX remote control not listening (port in use)" in r for r in reasons)
    assert ci._dmx_text({"enabled": False}) == "DMX off (Preferences)"


def test_settings_persist_and_bad_values_fall_back(tmp_path, monkeypatch):
    import control_interface as ci_mod
    from media_manager import MediaManager
    monkeypatch.setattr(ci_mod.ControlInterface, "CONFIG_DIR", str(tmp_path / ".vd"))
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    a = ci_mod.ControlInterface(MediaManager())
    assert (a.dmx_enabled, a.dmx_protocol, a.dmx_universe, a.dmx_channel) == (False, "artnet", 0, 1)
    a.dmx_enabled, a.dmx_protocol, a.dmx_universe, a.dmx_channel, a.dmx_network = True, "sacn", 5, 100, False
    a._save_config()
    b = ci_mod.ControlInterface(MediaManager())
    b._load_config()
    assert (b.dmx_enabled, b.dmx_protocol, b.dmx_universe, b.dmx_channel, b.dmx_network) == (True, "sacn", 5, 100, False)
    cfg = json.load(open(b.config_path))
    cfg.update(dmx_protocol="dmx512", dmx_channel=600, dmx_universe="x")
    json.dump(cfg, open(b.config_path, "w"))
    c = ci_mod.ControlInterface(MediaManager())
    c._load_config()
    assert (c.dmx_protocol, c.dmx_universe, c.dmx_channel) == ("artnet", 0, 1)
    for o in (a, b, c):
        o.proxy_executor.shutdown(wait=False)


def test_setup_dialog_validates_and_saves(blind_ci, monkeypatch):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    monkeypatch.setattr(dr, "SACN_PORT", 0)
    ci = blind_ci
    root = tk.Tk()
    root.withdraw()
    try:
        ci.root = root
        ci._save_config = lambda: setattr(ci, "saved", True)
        win = ci._open_dmx_setup(root)
        vd = win._vd
        vd["enabled_var"].set(True)
        vd["proto_var"].set("sACN (E1.31)")
        vd["uni_var"].set("0")                                     # sACN has no universe 0
        vd["net_var"].set(False)
        vd["save"]()
        assert "Universe" in vd["err"].cget("text") and not getattr(ci, "saved", False)
        vd["uni_var"].set("3")
        vd["ch_var"].set("20")
        vd["save"]()
        assert ci.saved and (ci.dmx_protocol, ci.dmx_universe, ci.dmx_channel) == ("sacn", 3, 20)
        assert ci._dmx is not None and ci._dmx.listening
    finally:
        ci._stop_dmx()
        root.destroy()
