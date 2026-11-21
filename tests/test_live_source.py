# SPDX-License-Identifier: MPL-2.0
"""Live sources (network streams, capture devices): URLs, labels, the player
in every render mode against a local UDP MPEG-TS stream (lost signal, held
last frame, reconnect), probing, and health. No camera or remote network was
used: the stream comes from tests/stream_sender.py on this computer."""
import os
import time

import pytest

import health
import live_source as ls
import render_engine as rem
from stream_sender import UdpSender, free_udp_port
from test_render_gl import run_gl_scenario
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401


@pytest.mark.parametrize("text,ok", [
    ("udp://0.0.0.0:5000", True), ("  rtsp://cam.local/stream1 ", True), ("SRT://host:9000", True),
    ("dshow://video=Logitech BRIO", True), ("v4l2:///dev/video0", True), ("https://x.test/live.m3u8", True),
    ("C:\\\\clips\\\\a.mp4", False), ("/home/a.mp4", False), ("ftp://x/y", False), ("udp://", False),
    ("udp:///", False), ("udp://a\nb", False), ("", False), (None, False),
])
def test_urls_are_recognized_and_cleaned(text, ok):
    assert (ls.clean(text) is not None) == ok


def test_parse_and_labels():
    assert ls.parse("dshow://video=Cam Link 4K") == ("video=Cam Link 4K", "dshow", {})
    assert ls.parse("v4l2:///dev/video0")[:2] == ("/dev/video0", "v4l2")
    url, fmt, opts = ls.parse("rtsp://10.0.0.5/live")
    assert fmt is None and opts["rtsp_transport"] == "tcp" and opts["fflags"] == "nobuffer"
    assert ls.parse("udp://0.0.0.0:5000")[2]["overrun_nonfatal"] == "1"
    assert ls.label("rtsp://admin:secret@10.0.0.5:554/live?x=1") == "RTSP 10.0.0.5:554/live"
    assert ls.label("dshow://video=BRIO") == "Camera video=BRIO"
    assert rem.is_streamed("udp://0.0.0.0:5000") and rem.is_streamed("/a/b.mp4") and not rem.is_streamed("/a/b.png")
    assert set(ls.available()) == {"dshow", "v4l2", "avfoundation"}


def test_gpu_output_follows_the_stream_through_a_lost_signal(fixtures, x_display):
    r = run_gl_scenario("live_source", fixtures, x_display, timeout=120)
    assert r["players"] == 1, "two layers with one source share one player"
    assert r["red"][0] > 240 and r["first_frame_s"] < 8
    assert r["after_stop_state"] == "retrying" and r["after_stop_s"] < 6
    assert r["held_frame"][0] > 240, "the last frame stays on the output while the signal is lost"
    assert r["green"][1] > 240 and r["reconnects"] >= 1 and r["state_end"] == "playing"
    assert r["health"]["file"].startswith("UDP 127.0.0.1:") and r["health"]["source"] == "live"
    assert r["players_after"] == 0


@pytest.mark.parametrize("mode", ["pygame_gpu", "cpu"])
def test_pygame_modes_show_the_stream(mode, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    port = free_udp_port()
    url = f"udp://127.0.0.1:{port}"
    eng = rem.PygameRenderEngine(media_manager=None, mode=mode)
    layers = {0: {"path": url, "order": 0, "position": {}}}
    snd = UdpSender(port, (0, 0, 255)).start()
    try:
        eng.update_layers(layers)
        assert eng.active_players[url]["type"] == "live"
        got = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10:
            surf = eng._render_layers_to_surface(layers, (320, 240))
            if surf is not None:
                got = list(surf.get_at((160, 120)))[:3]
                if got[2] > 240:
                    break
            time.sleep(0.05)
        assert got is not None and got[2] > 240 and got[0] < 15, got
        assert not eng.is_cue_static(layers), "a live source must never be cached as a still"
        img = eng.render_layers_to_image(layers, (160, 120))
        assert img.getpixel((80, 60))[2] > 240, "previews show it too"
    finally:
        snd.stop()
        eng.update_layers({})
        pygame.display.quit()


def test_probe_reports_a_stream_and_a_dead_one():
    port = free_udp_port()
    snd = UdpSender(port, (10, 200, 10)).start()
    try:
        r = ls.probe(f"udp://127.0.0.1:{port}")
    finally:
        snd.stop()
    assert r["ok"] and (r["width"], r["height"]) == (320, 240) and r["codec"] == "mpeg2video", r
    assert r["audio"] is None and r["preview"].size == (240, 180)          # kept to 320 x 180
    px = r["preview"].convert("RGB").getpixel((120, 90))
    assert px[1] > 150 and px[0] < 60 and px[2] < 60, px                     # the green it sends
    port = free_udp_port()
    snd = UdpSender(port, (10, 10, 200), tone_hz=440).start()
    try:
        r = ls.probe(f"udp://127.0.0.1:{port}")
    finally:
        snd.stop()
    assert r["ok"] and r["audio"] == "mp2", r
    t0 = time.monotonic()
    r = ls.probe(f"udp://127.0.0.1:{free_udp_port()}", timeout=1.0)
    assert not r["ok"] and time.monotonic() - t0 < 5, r


def test_health_flags_a_lost_live_source_on_screen():
    snap = {"output": {}, "decoders": [{"file": "UDP 0.0.0.0:5000", "state": "retrying", "live": True,
                                        "source": "live"}]}
    level, reasons = health.assess(snap)
    assert level == "error" and any("UDP 0.0.0.0:5000: retrying" in r for r in reasons)


def test_live_sources_are_never_missing_media(fixtures):
    import media_relink
    url = "udp://0.0.0.0:5000"
    assert media_relink.find_missing([url, fixtures["red_png"], "/nope.png"]) == ["/nope.png"]


def test_live_cells_share_one_player_and_get_a_card(blind_ci, fixtures):
    ci = blind_ci
    url = "udp://0.0.0.0:5000"
    ci.grid_cells[(2, 0)] = url
    ci.grid_cells[(2, 1)] = url
    a, b = ci._get_layers_for_col(0), ci._get_layers_for_col(1)
    assert "instance" not in a[2] and "instance" not in b[2], "keyed by source: one player for both cues"
    assert ci._live_size(url) == (1920, 1080)
    card = ci._load_frame(url, for_thumbnail=True)
    assert card.mode == "RGBA" and max(card.size) <= max(ci.THUMB_SOURCE_MAX)
    big = ci._live_card(url, (640, 360))
    assert big.size == (640, 360) and big.getpixel((320, 150))[0] > 150, "red LIVE badge"


def test_add_dialog_tests_and_adds(blind_ci, monkeypatch):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    ci = blind_ci
    ci._build_media_list = lambda: None
    root = tk.Tk()
    root.withdraw()
    port = free_udp_port()
    url = f"udp://127.0.0.1:{port}"
    snd = UdpSender(port, (200, 0, 0)).start()
    try:
        ci.root = root
        win = ci._open_live_source_dialog()
        vd = win._vd
        vd["url_var"].set("C:\\\\not a url")
        vd["test"]()
        assert "Type an address" in vd["result"].cget("text")
        vd["url_var"].set(url)
        vd["test"]()
        t0 = time.monotonic()
        while vd["st"]["testing"] and time.monotonic() - t0 < 15:
            root.update()
            time.sleep(0.05)
        assert vd["result"].cget("text").startswith("Receiving: 320\u00d7240"), vd["result"].cget("text")
        assert ", no sound" in vd["result"].cget("text")
        assert vd["picture"].winfo_ismapped() or vd["picture"].winfo_manager() == "pack"
        assert vd["st"]["photo"].width() == 240, "the picture it sends is shown"
        assert ci._live_size(url) == (320, 240), "the tested size is remembered"
        assert vd["add"]() == url and url in ci.project_media_paths
        win2 = ci._open_live_source_dialog()
        win2._vd["url_var"].set(url)
        assert win2._vd["add"]() is None and "already" in win2._vd["result"].cget("text")
        win2.destroy()
    finally:
        snd.stop()
        root.destroy()


def test_the_next_cues_live_source_is_connected_before_go(blind_ci):
    from test_cue_activation import commands
    ci = blind_ci
    url = "udp://0.0.0.0:5000"
    ci.grid_cells[(3, 1)] = url                      # cue 2 is next after the live cue 1
    commands(ci)
    ci._update_active_players()
    keep = [c for c in commands(ci) if c["action"] == "update_layers"][-1]["layers"]
    assert any(l["path"] == url for l in keep.values())


# --- Find Devices --------------------------------------------------------------------------------
DSHOW_NEW = """[dshow @ 000001eb780baf00] "USB Capture HDMI" (video)
[dshow @ 000001eb780baf00]   Alternative name "@device_pnp_\\\\?\\usb#vid_2935&pid_0006&mi_00#8"
[dshow @ 000001eb780baf00] "OBS Virtual Camera" (none)
[dshow @ 000001eb780baf00]   Alternative name "@device_sw_{860BB310-5D01-11D0-BD3B-00A0C911CE86}"
[dshow @ 000001eb780baf00] "IN 3-4 (2- BEHRINGER UMC 404HD 192k)" (audio)
dummy: Immediate exit requested
"""
DSHOW_OLD = """[dshow @ 000000000050a400] DirectShow video devices (some may be both video and audio devices)
[dshow @ 000000000050a400]  "HP HD Webcam [Fixed]"
[dshow @ 000000000050a400]     Alternative name "@device_pnp_x"
[dshow @ 000000000050a400] DirectShow audio devices
[dshow @ 000000000050a400]  "Internal Microphone Array (IDT "
[dshow @ 000000000050a400]     Alternative name "@device_cm_y"
"""
AVF = """[AVFoundation indev @ 0x7f8] AVFoundation video devices:
[AVFoundation indev @ 0x7f8] [0] FaceTime HD Camera
[AVFoundation indev @ 0x7f8] [1] Capture screen 0
[AVFoundation indev @ 0x7f8] AVFoundation audio devices:
[AVFoundation indev @ 0x7f8] [0] MacBook Pro Microphone
"""


def test_device_lists_in_both_dshow_layouts_and_avfoundation():
    assert ls.parse_device_list(DSHOW_NEW, "dshow") == [
        ("USB Capture HDMI", "video"), ("OBS Virtual Camera", "none"),
        ("IN 3-4 (2- BEHRINGER UMC 404HD 192k)", "audio")]
    assert ls.parse_device_list(DSHOW_OLD, "dshow") == [
        ("HP HD Webcam [Fixed]", "video"), ("Internal Microphone Array (IDT ", "audio")]
    assert ls.parse_device_list(AVF, "avfoundation") == [
        ("0", "video", "FaceTime HD Camera"), ("1", "video", "Capture screen 0"),
        ("0", "audio", "MacBook Pro Microphone")]
    assert ls.parse_device_list("", "dshow") == []


def test_find_devices_offers_video_devices_as_addresses(monkeypatch):
    monkeypatch.setattr(ls, "available", lambda: {"dshow": True, "v4l2": False, "avfoundation": False})
    monkeypatch.setattr(ls, "_ffmpeg_device_list", lambda fmt, url: DSHOW_NEW)
    r = ls.find_devices()
    assert r["devices"] == [("dshow://video=USB Capture HDMI", "USB Capture HDMI"),
                            ("dshow://video=OBS Virtual Camera", "OBS Virtual Camera (type not reported)")]
    assert all(ls.clean(u) == u for u, _ in r["devices"])
    monkeypatch.setattr(ls, "_ffmpeg_device_list", lambda fmt, url: "")
    assert ls.find_devices() == {"devices": [], "note": "No capture devices found."}
    monkeypatch.setattr(ls, "available", lambda: {"dshow": False, "v4l2": False, "avfoundation": False})
    assert "no camera support" in ls.find_devices()["note"]


def test_ffmpeg_listing_reads_ffmpegs_own_messages():
    """The listing goes through PyAV's log capture; here an error FFmpeg logs
    shows that the capture and the level switch work (no dshow on Linux)."""
    import av.logging
    level = av.logging.get_level()
    text = ls._ffmpeg_device_list("lavfi", "nosuchfilter_xyz")
    assert "No such filter: 'nosuchfilter_xyz'" in text
    assert av.logging.get_level() == level, "the log level is put back"


def test_find_devices_in_the_dialog(blind_ci, monkeypatch):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    monkeypatch.setattr(ls, "find_devices", lambda: {
        "devices": [("dshow://video=USB Capture HDMI", "USB Capture HDMI")], "note": None})
    ci = blind_ci
    root = tk.Tk()
    root.withdraw()
    ci.root = root
    try:
        vd = ci._open_live_source_dialog()._vd
        vd["find_devices"]()
        t0 = time.monotonic()
        while vd["st"]["finding"] and time.monotonic() - t0 < 5:
            root.update()
            time.sleep(0.02)
        assert "1 device(s) found" in vd["result"].cget("text")
        vd["dev_list"].selection_set(0)
        vd["dev_list"].event_generate("<<ListboxSelect>>")
        root.update()
        assert vd["url_var"].get() == "dshow://video=USB Capture HDMI"
    finally:
        root.destroy()


def test_cameras_used_anywhere_in_the_show_stay_connected(blind_ci):
    """On the owner's PC a cut to a cue with a webcam (not the next cue) waited
    the full preroll and went on without it: opening a camera is slow."""
    from test_cue_activation import commands
    ci = blind_ci
    cam = "dshow://video=Microsoft® LifeCam Cinema(TM)"
    ci.grid_cells[(2, 5)] = cam                      # cue 6: neither live, next nor previous
    ci.grid_cells[(3, 6)] = cam                      # and cue 7: one connection for both
    ci.grid_cells[(2, 4)] = "udp://0.0.0.0:5004"     # a network stream there is not kept
    commands(ci)
    ci._update_active_players()
    keep = [c for c in commands(ci) if c["action"] == "update_layers"][-1]["layers"]
    cams = [k for k, l in keep.items() if l["path"] == cam]
    assert cams == [f"dev:{cam}"]
    assert not any(l["path"] == "udp://0.0.0.0:5004" for l in keep.values())
    assert ls.is_device(cam) and not ls.is_device("udp://0.0.0.0:5004") and not ls.is_device("/x/a.mp4")
    ci.grid_cells[(2, 5)] = ci.grid_cells[(3, 6)] = None
    ci._update_active_players()
    keep = [c for c in commands(ci) if c["action"] == "update_layers"][-1]["layers"]
    assert not any(l["path"] == cam for l in keep.values()), "taken out of the show: let go"


class _LostDevice:
    """A connection whose reading fails like an unplugged camera (I/O error)."""
    closed = 0

    def __init__(self):
        import types
        self.streams = types.SimpleNamespace(video=[types.SimpleNamespace(
            average_rate=30, codec_context=types.SimpleNamespace(name="rawvideo"))], audio=[])

    def demux(self, *streams):
        raise OSError(5, "I/O error")
        yield  # noqa: unreachable - a generator

    def close(self):
        type(self).closed += 1


@pytest.mark.parametrize("url, kept", [("dshow://video=Cam", True), ("udp://127.0.0.1:5999", False)])
def test_a_camera_lost_while_sending_is_not_closed(monkeypatch, url, kept):
    """Closing FFmpeg's DirectShow connection of a camera that was unplugged while
    sending ended VisualDeck with an access violation (owner's PC, 2026-11-15);
    such a connection is kept instead. Other sources are closed as before."""
    import render_engine as rem
    from ring_buffer import FastRingBuffer
    conns = []

    def fake_open(*a, **k):
        c = _LostDevice()
        conns.append(c)
        if len(conns) >= 2:
            p.is_running = False                  # one reconnect, then stop
        return c
    monkeypatch.setattr(rem.av, "open", fake_open)
    _LostDevice.closed = 0
    before = len(rem._abandoned_devices)
    p = rem.LiveSourcePlayer(url, FastRingBuffer(4))
    p._sleep_interruptible = lambda s: True
    p.run()
    assert len(conns) == 2 and "I/O error" in p.last_error
    if kept:
        assert _LostDevice.closed == 0 and rem._abandoned_devices[before:] == conns
    else:
        assert _LostDevice.closed == 2 and len(rem._abandoned_devices) == before
    del rem._abandoned_devices[before:]
