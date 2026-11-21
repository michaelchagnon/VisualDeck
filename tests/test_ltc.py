# SPDX-License-Identifier: MPL-2.0
"""SMPTE LTC: the frame layout, decoding audio, the input, and following it."""
import numpy as np
import pytest

import ltc
import timecode


def decode_all(audio, rate=48000, chunk=None):
    d = ltc.LTCDecoder(rate)
    out = []
    step = chunk or audio.size
    for i in range(0, audio.size, step):
        out += d.feed(audio[i:i + step])
    return out, d


def test_frame_layout_round_trips_and_has_the_sync_word():
    for tc in ((0, 0, 0, 0), (23, 59, 59, 24), (12, 34, 56, 7), (1, 2, 3, 29)):
        bits = ltc.frame_bits(tc, fps=30)
        assert len(bits) == 80 and tuple(bits[64:]) == ltc.SYNC
        assert bits.count(0) % 2 == 0                          # polarity correction bit
        assert ltc.decode_bits(bits) == tuple(tc) + (False,)
    assert ltc.decode_bits(ltc.frame_bits((1, 0, 0, 0), drop=True))[4] is True
    bad = ltc.frame_bits((1, 0, 0, 0))
    bad[0:4] = [1, 1, 1, 1]                                    # frame units 15: not BCD
    assert ltc.decode_bits(bad) is None


@pytest.mark.parametrize("fps,rate,code", [(24, 48000, 0), (25, 48000, 1), (30, 48000, 3), (29.97, 48000, 2),
                                           (25, 44100, 1), (30, 96000, 3)])
def test_decodes_every_frame_at_each_rate(fps, rate, code):
    audio = ltc.encode((10, 59, 58, 0), 80, fps=fps, rate=rate, drop=(fps == 29.97), rise_samples=3)
    out, d = decode_all(audio, rate, chunk=331)
    nominal = 30 if fps == 29.97 else fps
    # Every frame but the last (its final bit ends with the audio), each one
    # frame later than the frame it came in (its time is known at its end).
    want = [timecode.add_frames((10, 59, 58, 0), i + 1, nominal) + (code,) for i in range(79)]
    assert out == want and d.errors == 0 and d.fps == nominal


def test_survives_level_noise_dc_inverted_polarity_and_varispeed():
    rng = np.random.default_rng(1)
    start = (1, 0, 0, 20)
    base = ltc.encode(start, 60, fps=25, rise_samples=4, level=0.05)
    for name, audio in (("quiet with noise and DC", base + 0.004 * rng.standard_normal(base.size) + 0.02),
                        ("inverted", -base),
                        ("8 % fast", ltc.encode(start, 60, fps=25, speed=1.08, rise_samples=2)),
                        ("8 % slow", ltc.encode(start, 60, fps=25, speed=0.92, rise_samples=2))):
        out, d = decode_all(audio.astype(np.float32), chunk=512)
        assert len(out) == 59, (name, len(out), d.errors)
        # From the first second boundary (frame 24 -> 0) the rate is counted, not guessed from the bit
        # rate (8 % slow 25 fps has the bit rate of 23 fps).
        assert out[4:] == [timecode.add_frames(start, i + 1, 25) + (1,) for i in range(4, 59)], name


def test_starts_in_the_middle_and_recovers_after_a_dropout():
    a = ltc.encode((2, 0, 0, 0), 40, fps=25)
    b = ltc.encode((2, 0, 1, 15), 40, fps=25)              # 40 frames later the tape is at 2:00:01:15
    spf = 48000 // 25
    audio = np.concatenate([a[spf // 3:], np.zeros(spf * 2, np.float32), b])
    out, d = decode_all(audio, chunk=480)
    times = [timecode.seconds(t, 25) for t in out]
    assert out[0][:4] == (2, 0, 0, 2)                      # the first whole frame (the second one sent)
    assert any(t[:4] == (2, 0, 1, 16) for t in out)        # locked again within a frame after the gap
    assert times == sorted(times)


def test_backwards_and_silence_give_no_times():
    a = ltc.encode((3, 0, 0, 0), 30, fps=25)
    out, d = decode_all(a[::-1].copy())
    assert out == [] and d.backwards >= 25
    out, d = decode_all(np.zeros(48000, np.float32))
    assert out == [] and d.errors == 0
    out, d = decode_all((0.003 * np.random.default_rng(2).standard_normal(48000)).astype(np.float32))
    assert out == []                                       # hiss below the input threshold


def test_input_reads_the_chosen_channel():
    sig = ltc.encode((4, 0, 0, 0), 20, fps=25)
    stereo = np.stack([np.zeros_like(sig), sig], axis=1)
    for channel, want in (("Right", True), ("Mix", True), ("Left", False)):
        inp = ltc.LTCInput(channel=channel, backend="offline")
        assert inp.start() and inp.listening
        for i in range(0, len(stereo), 512):
            inp.feed_stereo(stereo[i:i + 512])
        assert (inp.timecode is not None) is want, channel
    st = inp.status()
    assert st["listening"] and st["channel"] == "Left" and "timecode" not in st
    inp = ltc.LTCInput(channel="Right", backend="offline")
    inp.start()
    inp.feed_stereo(stereo)
    first = inp.timecode
    st = inp.status()
    assert st["timecode"] == "04:00:00:19" and st["rate"] == "25" and st["frames"] == 19
    inp.feed_stereo(np.zeros((480, 2), np.float32))
    assert inp.timecode is first                           # no new frame: the same object


# --- In the app ---------------------------------------------------------------------

from test_blind_editing import ci as blind_ci  # noqa: E402,F401
from test_cue_activation import ci as _base_ci  # noqa: E402,F401


class StandIn:
    """The app reads only .timecode from its inputs."""
    def __init__(self):
        self.timecode = None


def follow(ci, source, start, frames, fps=25, rate=1):
    fired = []
    t = start
    for _ in range(frames):
        source.timecode = t + (rate, 0.0)                 # a new tuple per frame, as LTCInput does
        r = ci._timecode_tick()
        if r is not None:
            fired.append((timecode.fmt(t), r))
        t = timecode.add_frames(t, 1, fps)
    return fired


def test_app_follows_ltc_and_ignores_midi_timecode_meanwhile(blind_ci):
    ci = blind_ci
    ci.cue_settings["1"] = {"timecode": "10:00:00:00"}
    ci.cue_settings["2"] = {"timecode": "10:00:01:00"}
    ci._ltc, ci._midi = StandIn(), StandIn()
    ci.ltc_enabled, ci.midi_timecode = True, True
    assert follow(ci, ci._ltc, (9, 59, 59, 20), 10) == [("10:00:00:00", 1)]
    assert follow(ci, ci._midi, (10, 0, 0, 20), 10) == []             # MIDI timecode: not followed now
    # Off: nothing fires; MIDI timecode is followed again (its first time is a start, not a crossing).
    ci.ltc_enabled = False
    assert follow(ci, ci._ltc, (10, 0, 0, 20), 10) == []
    assert follow(ci, ci._midi, (10, 0, 0, 20), 10) == [("10:00:01:00", 2)]


def test_switching_to_ltc_does_not_fire_on_the_first_time(blind_ci):
    ci = blind_ci
    ci.cue_settings["1"] = {"timecode": "10:00:00:10"}
    ci._ltc, ci._midi = StandIn(), StandIn()
    ci.midi_timecode = True
    follow(ci, ci._midi, (10, 0, 0, 0), 5)                          # MIDI at 10:00:00:04
    ci.ltc_enabled = True
    ci._ltc.timecode = (10, 0, 0, 12, 1, 0.0)                       # LTC already past the cue time
    assert ci._timecode_tick() is None


def test_ltc_status_text_and_health_warning(blind_ci):
    import health
    ci = blind_ci
    assert ci._ltc_text(ci._ltc_status()) == "LTC off (Preferences)"
    ci.ltc_enabled = True
    ci._ltc = ltc.LTCInput("Line In", "Right", backend="offline")
    ci._ltc.start()
    text = ci._ltc_text(ci._ltc_status())
    assert text.startswith("LTC on 'offline' (Right)") and "no timecode now" in text
    sig = ltc.encode((10, 0, 0, 0), 10, fps=30)
    ci._ltc.feed_stereo(np.stack([sig, sig], axis=1))
    assert "10:00:00:09 (30 fps)" in ci._ltc_text(ci._ltc_status())
    ci._ltc = ltc.LTCInput(backend="offline")                     # not started: an error
    ci._ltc.error = "cannot open audio input: no audio inputs"
    st = ci._ltc_status()
    assert "LTC not listening: cannot open audio input" in ci._ltc_text(st)
    level, reasons = health.assess({"ltc": st, "render_thread_alive": True, "time": 0, "stale_s": 0})[:2]
    assert any("LTC input not listening" in r for r in reasons), reasons


def test_ltc_settings_persist(tmp_path, monkeypatch):
    import json
    import control_interface as ci_mod
    from media_manager import MediaManager
    monkeypatch.setattr(ci_mod.ControlInterface, "CONFIG_DIR", str(tmp_path / ".vd"))
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    a = ci_mod.ControlInterface(MediaManager())
    assert (a.ltc_enabled, a.ltc_device, a.ltc_channel) == (False, "", "Left")
    a.ltc_enabled, a.ltc_device, a.ltc_channel = True, "Line In (Focusrite)", "Right"
    a._save_config()
    saved = json.load(open(a.config_path))
    assert (saved["ltc_enabled"], saved["ltc_device"], saved["ltc_channel"]) == (True, "Line In (Focusrite)", "Right")
    saved["ltc_channel"] = "Center"                                 # a hand edit: back to Left
    json.dump(saved, open(a.config_path, "w"))
    b = ci_mod.ControlInterface(MediaManager())
    b._load_config()
    assert (b.ltc_enabled, b.ltc_device, b.ltc_channel) == (True, "Line In (Focusrite)", "Left")
    for o in (a, b):
        o.proxy_executor.shutdown(wait=False)


def test_ltc_setup_window_saves_and_opens_the_input(blind_ci, monkeypatch):
    import os
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import ui_theme
    ci = blind_ci
    opened = []

    class Offline(ltc.LTCInput):
        def __init__(self, device=None, channel="Left", backend="offline"):
            super().__init__(device, channel, backend="offline")
            opened.append((self.device_name, self.channel))
    monkeypatch.setattr(ltc, "LTCInput", Offline)
    monkeypatch.setattr(ltc, "capture_device_names", lambda: (["Line In (Focusrite)", "Microphone"], None))
    saved = []
    monkeypatch.setattr(ci, "_save_config", lambda: saved.append(True))
    root = tk.Tk()
    root.withdraw()
    ui_theme.apply(root)
    ci.root = root
    try:
        win = ci._open_ltc_setup()
        vd = win._vd
        assert vd["choices"] == [ltc.DEFAULT_DEVICE, "Line In (Focusrite)", "Microphone"]
        vd["enabled_var"].set(True)
        vd["dev_var"].set("Line In (Focusrite)")
        vd["ch_var"].set("Right")
        vd["save"]()
        assert saved and (ci.ltc_enabled, ci.ltc_device, ci.ltc_channel) == (True, "Line In (Focusrite)", "Right")
        assert opened == [("Line In (Focusrite)", "Right")] and ci._ltc.listening
        win = ci._open_ltc_setup()
        win._vd["enabled_var"].set(False)
        win._vd["save"]()
        assert ci._ltc is None and not ci.ltc_enabled
    finally:
        ci._stop_ltc()
        root.destroy()
