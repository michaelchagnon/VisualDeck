# SPDX-License-Identifier: MPL-2.0
"""DMX out: the live cue for a lighting console (Art-Net and sACN)."""
import socket

import pytest

import dmx_remote as dm
from test_cue_activation import ci, commands  # noqa: F401


class Sock:
    def __init__(self):
        self.sent = []

    def sendto(self, pkt, addr):
        self.sent.append((pkt, addr))

    def close(self):
        pass


def test_frames_on_change_and_keepalive():
    s = Sock()
    out = dm.DMXOutput("artnet", universe=3, channel=10, target="10.0.0.5", sock=s)
    assert out.update(4, 1.0, now=0.0) and out.update(4, 1.0, now=0.5) is False
    assert out.update(4, 1.0, now=1.2)                          # keepalive
    assert out.update(5, 1.0, now=1.3)                          # change
    pkt, addr = s.sent[-1]
    assert addr == ("10.0.0.5", dm.ARTNET_PORT)
    universe, data = dm.parse_artnet(pkt)
    assert universe == 3 and data[9] == 5 and data[10] == 255 and len(data) == 512


def test_sacn_goes_to_its_multicast_group():
    s = Sock()
    out = dm.DMXOutput("sacn", universe=258, channel=1, sock=s)
    out.update(7, 0.5, now=0.0)
    pkt, addr = s.sent[-1]
    assert addr == ("239.255.1.2", dm.SACN_PORT)
    universe, data = dm.parse_sacn(pkt)
    assert universe == 258 and data[0] == 7 and data[1] == 128


def test_a_real_udp_packet_arrives():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)
    out = dm.DMXOutput("artnet", universe=0, channel=1, target="127.0.0.1")
    out.address = lambda: ("127.0.0.1", rx.getsockname()[1])
    out.update(12, 1.0, now=0.0)
    universe, data = dm.parse_artnet(rx.recv(1024))
    out.stop()
    rx.close()
    assert data[0] == 12 and data[1] == 255 and out.sent == 1 and out.error is None


def test_app_sends_the_live_cue(ci):
    s = Sock()
    ci._dmx_out = dm.DMXOutput("artnet", universe=0, channel=1, sock=s)
    ci.active_col = None
    ci._dmx_out_tick(now=0.0)
    assert dm.parse_artnet(s.sent[-1][0])[1][:2] == bytes([0, 0])            # nothing on the output
    ci.cue_settings = {"2": {"number": "7"}}
    ci.active_col = 2
    ci._output_is_black = lambda: False
    ci._dmx_out_tick(now=0.1)
    assert dm.parse_artnet(s.sent[-1][0])[1][:2] == bytes([7, 255])          # its cue number
    ci.active_col = 3
    ci._dmx_out_tick(now=0.2)
    assert dm.parse_artnet(s.sent[-1][0])[1][0] == 4                         # no number: its position
    ci._output_is_black = lambda: True
    ci._dmx_out_tick(now=0.3)
    assert dm.parse_artnet(s.sent[-1][0])[1][:2] == bytes([0, 0])            # Blackout


def test_dmx_setup_saves_the_sending_settings(ci, monkeypatch):
    import os
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import ui_theme
    root = tk.Tk()
    root.withdraw()
    ui_theme.apply(root)
    ci.root = root
    monkeypatch.setattr(ci, "_save_config", lambda: None)
    monkeypatch.setattr(ci, "_restart_dmx", lambda: None)
    try:
        vd = ci._open_dmx_setup()._vd
        vd["out_var"].set(True)
        vd["out_ch"].set("600")
        vd["save"]()
        assert "channel 1-511" in vd["err"].cget("text") and not ci.dmx_out_enabled
        vd["out_ch"].set("20")
        vd["out_uni"].set("2")
        vd["out_to"].set("192.168.1.40")
        vd["save"]()
        assert (ci.dmx_out_enabled, ci.dmx_out_universe, ci.dmx_out_channel, ci.dmx_out_target) == \
            (True, 2, 20, "192.168.1.40")
        assert ci._dmx_out is not None and ci._dmx_out.address() == ("192.168.1.40", dm.ARTNET_PORT)
    finally:
        if ci._dmx_out is not None:
            ci._dmx_out.stop()
        root.destroy()
