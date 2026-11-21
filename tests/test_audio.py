"""Audio of videos: decoding a clip's sound, the mixer (gains, ramps, clipping),
# SPDX-License-Identifier: MPL-2.0
looping with the picture, which videos the render thread asks for, settings,
and the real SDL output through SDL's file-writing "disk" driver (no sound
card here: what reaches a speaker was not heard)."""
import json
import os
import subprocess
import sys
import time

import av
import numpy as np
import pytest

import audio_engine as ae
import control_interface as ci_mod

HERE = os.path.dirname(os.path.abspath(__file__))


def tone_clip(path, freq, seconds=2.0, rate=44100, fps=25, layout="mono"):
    """A small video with a sine tone (mono AAC, like a camera clip; stereo: the
    right side at half the level)."""
    out = av.open(path, "w")
    v = out.add_stream("libx264", rate=fps)
    v.width, v.height, v.pix_fmt = 160, 120, "yuv420p"
    a = out.add_stream("aac", rate=rate)
    a.layout = layout
    img = np.zeros((120, 160, 3), np.uint8)
    for i in range(int(seconds * fps)):
        fr = av.VideoFrame.from_ndarray(img, format="rgb24")
        fr.pts = i
        for p in v.encode(fr):
            out.mux(p)
    t = np.arange(int(seconds * rate)) / rate
    samples = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    for i in range(0, len(samples), 1024):
        part = samples[i:i + 1024]
        if layout == "stereo":
            af = av.AudioFrame.from_ndarray(np.stack([part, part * 0.5], axis=1).reshape(1, -1).copy(),
                                            format="flt", layout="stereo")
        else:
            af = av.AudioFrame.from_ndarray(part[None, :], format="flt", layout="mono")
        af.sample_rate = rate
        af.pts = i
        for p in a.encode(af):
            out.mux(p)
    for s in (v, a):
        for p in s.encode(None):
            out.mux(p)
    out.close()
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("audio")
    return {"1k": tone_clip(str(d / "tone1k.mp4"), 1000), "500": tone_clip(str(d / "tone500.mp4"), 500),
            "silent": str(d / "silent.mp4")}


def dominant(x, rate=ae.RATE):
    x = np.asarray(x, np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / rate)[np.argmax(spec)]


def band(x, f, rate=ae.RATE):
    spec = np.abs(np.fft.rfft(np.asarray(x, np.float64) * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    return spec[(freqs > f - 30) & (freqs < f + 30)].sum()


def wait(cond, t=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < t:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_stream_decodes_resamples_and_loops(clips):
    s = ae.AudioStream(clips["1k"], loop=True)
    s.start()
    try:
        assert wait(lambda: s._buffered >= ae.RATE * 0.4)
        a = s.read(ae.RATE // 2)
        assert a.shape == (ae.RATE // 2, 2) and a.dtype == np.float32
        assert abs(dominant(a[:, 0]) - 1000) < 5 and np.allclose(a[:, 0], a[:, 1])     # mono -> both sides
        rms = float(np.sqrt((a[2000:, 0] ** 2).mean()))
        assert 0.22 < rms < 0.28, rms        # 0.5 peak sine = 0.354 RMS; mono to stereo at -3 dB per side: 0.25
        s.restart()                                                                   # the video looped
        assert wait(lambda: s._buffered >= ae.RATE * 0.2)
        assert s.underruns == 0
    finally:
        s.stop()


def test_a_48k_stereo_stream_keeps_its_sides_and_can_be_joined(tmp_path):
    """48 kHz stereo is converted without FFmpeg's resampler (StereoConverter);
    the stream thread can be joined (its stop flag no longer hides Thread._stop)."""
    s = ae.AudioStream(tone_clip(str(tmp_path / "st48.mp4"), 1000, rate=48000, layout="stereo"), loop=True)
    s.start()
    try:
        assert wait(lambda: s._buffered >= ae.RATE * 0.4)
        a = s.read(ae.RATE // 2)
        assert abs(dominant(a[:, 0]) - 1000) < 5 and abs(dominant(a[:, 1]) - 1000) < 5
        left, right = (float(np.sqrt((a[2000:, c] ** 2).mean())) for c in (0, 1))
        assert 0.33 < left < 0.38 and abs(right / left - 0.5) < 0.02, (left, right)   # 0.5 peak sine: 0.354
        s.restart()
        assert wait(lambda: s._buffered >= ae.RATE * 0.2)
        assert s.underruns == 0
    finally:
        s.stop()
    s.join(2)
    assert not s.is_alive()


def test_mixer_gains_ramps_and_clipping(clips):
    eng = ae.AudioEngine(backend="offline")
    eng.set_streams({"a": {"path": clips["1k"], "gain": 1.0, "playing": True},
                     "b": {"path": clips["500"], "gain": 0.25, "playing": True},
                     "c": {"path": clips["1k"], "gain": 1.0, "playing": False}})
    st = {k: e["stream"] for k, e in eng._streams.items()}
    assert wait(lambda: all(s._buffered > ae.RATE * 0.3 for s in st.values()))
    eng.mix(4800)                                 # the first buffer ramps the gains up from 0
    out = eng.mix(ae.RATE // 4)[:, 0]
    ratio = band(out, 1000) / band(out, 500)
    assert 3.3 < ratio < 4.8, ratio               # gains 1 : 0.25
    assert st["c"].frames_out == 0, "a paused (next cue) stream is not consumed"
    assert np.abs(np.diff(eng.mix(1024)[:, 0])).max() < 0.2, "no clicks"
    eng.set_streams({"a": {"path": clips["1k"], "gain": 6.0, "playing": True}})   # far too loud
    assert wait(lambda: st["b"]._stop_event.is_set()), "streams no longer wanted stop"
    eng.mix(1024)
    assert np.abs(eng.mix(4096)).max() <= 1.0
    eng.stop()


def test_render_thread_asks_for_the_output_and_the_next_cue(fixtures):
    class P:
        def __init__(self, held=False, frames=5, loops=0):
            self.held, self.frames_decoded, self.loops, self.loop = held, frames, loops, True

    class Eng:
        def __init__(self):
            self.active_players = {}

        @staticmethod
        def source_key(layer):
            return layer.get("instance") or layer.get("path")

    class Audio:
        def set_streams(self, w):
            self.w = w

    rt = ci_mod.RenderThread.__new__(ci_mod.RenderThread)
    rt.re, rt.audio = Eng(), Audio()
    v = fixtures["counting"]
    rt.re.active_players = {"@1:0:0|" + v: {"player": P(loops=2), "path": v, "type": "gpu_queue"},
                            "@2:0:1|" + v: {"player": P(held=True, frames=1), "path": v, "type": "gpu_queue"},
                            "@3:0:0|" + v: {"player": P(), "path": v, "type": "gpu_queue"},
                            "udp://x:1": {"player": P(held=False), "path": "udp://x:1", "type": "live"}}
    layer = lambda inst, **k: {"path": v, "instance": inst + "|" + v, **k}  # noqa: E731
    rt._update_audio({"layers": {0: layer("@1:0:0", volume=0.5), 1: {"path": fixtures["red_png"]},
                                 2: {"path": "udp://x:1"}}, "transition_state": {"active": False}})
    w = rt.audio.w
    assert w["@1:0:0|" + v] == {"path": v, "gain": 0.5, "playing": True, "loop": True, "loops": 2,
                                "trim": (None, None), "clock_start": None, "loop_at": None}
    assert w["@2:0:1|" + v]["playing"] is False and w["@2:0:1|" + v]["gain"] == 0.0, "next cue: decoded, paused"
    assert "udp://x:1" not in w and fixtures["red_png"] not in w
    ts = {"active": True, "progress": 0.25, "curve": "linear",
          "from": {0: layer("@1:0:0")}, "to": {0: layer("@3:0:0", speed=2.0)}}
    rt._update_audio({"layers": {}, "transition_state": ts})
    w = rt.audio.w
    assert w["@1:0:0|" + v]["gain"] == pytest.approx(0.75) and w["@3:0:0|" + v]["gain"] == 0.0, \
        "fades follow the transition; a sped-up video is silent"


def test_setting_persists_and_health_text(tmp_path, monkeypatch):
    from media_manager import MediaManager
    monkeypatch.setattr(ci_mod.ControlInterface, "CONFIG_DIR", str(tmp_path / ".vd"))
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    a = ci_mod.ControlInterface(MediaManager())
    assert a.audio_device == "" and a._audio_text(a._audio_status()) == "Audio off (Preferences)"
    a.audio_device = "Speakers (Realtek)"
    a._save_config()
    assert json.load(open(a.config_path))["audio_device"] == "Speakers (Realtek)"
    b = ci_mod.ControlInterface(MediaManager())
    b._load_config()
    assert b.audio_device == "Speakers (Realtek)"
    assert "not playing" in b._audio_text(b._audio_status())
    for o in (a, b):
        o.proxy_executor.shutdown(wait=False)


SDL_SCRIPT = r"""
import os, sys, time, numpy as np
sys.path.insert(0, sys.argv[1])
import audio_engine as ae
eng = ae.AudioEngine("")
assert eng.start(), eng.error
eng.set_streams({"k": {"path": sys.argv[2], "gain": 1.0, "playing": True}})
time.sleep(1.2)
st = eng.status()
eng.stop()
print("STATUS", st["running"], st["callbacks"] if "callbacks" in st else eng.callbacks, st["underruns"])
"""


def test_sdl_output_writes_the_tone(clips, tmp_path):
    raw = str(tmp_path / "out.raw")
    env = dict(os.environ, SDL_AUDIODRIVER="disk", SDL_DISKAUDIOFILE=raw, PYGAME_HIDE_SUPPORT_PROMPT="1")
    r = subprocess.run([sys.executable, "-c", SDL_SCRIPT, os.path.join(os.path.dirname(HERE), "src"), clips["1k"]],
                       env=env, capture_output=True, text=True, timeout=60)
    assert "STATUS True" in r.stdout, r.stdout + r.stderr
    data = np.fromfile(raw, np.float32).reshape(-1, 2)
    assert len(data) > ae.RATE * 0.8
    start = int(np.argmax(np.abs(data[:, 0]) > 0.01))
    seg = data[start:start + ae.RATE // 2, 0]
    assert start > 0 or abs(data[0, 0]) > 0.01
    assert len(seg) == ae.RATE // 2 and np.sqrt((seg ** 2).mean()) > 0.2, "sound reached the device"
    assert abs(dominant(seg) - 1000) < 5


def test_late_start_keeps_sync(clips):
    """The picture started before the sound was decoded: the missing part is
    skipped, so the sound lines up with the picture instead of lagging."""
    s = ae.AudioStream(clips["1k"])
    for _ in range(3):
        assert not s.read(4800).any()            # 0.3 s asked for before anything is decoded
    s.start()
    try:
        assert wait(lambda: s._buffered > ae.RATE * 0.45)
        s.read(1000)
        assert s._owed == 0, "the 0.3 s owed was skipped"
        assert s.frames_out == 1000
    finally:
        s.stop()


def test_audio_delay_plays_silence_first_and_again_after_a_loop(clips):
    s = ae.AudioStream(clips["1k"], delay_s=0.1)
    s.start()
    try:
        assert wait(lambda: s._buffered > ae.RATE * 0.3)
        a = s.read(ae.RATE // 5)                         # 200 ms: 100 ms silence, then the tone
        first = int(np.argmax(np.abs(a[:, 0]) > 0.01))
        assert ae.RATE // 10 <= first <= ae.RATE // 10 + 5, first
        assert s._owed == 0
        s.restart()
        assert wait(lambda: s._silence == s.delay_frames and s._buffered > ae.RATE * 0.2)
        assert not s.read(ae.RATE // 20).any(), "the delay applies again from the loop point"
    finally:
        s.stop()
    eng = ae.AudioEngine(backend="offline", delay_ms=5000)
    assert eng.delay_s == 1.0, "capped at 1 s"


RESTART_SCRIPT = r"""
import os, sys, time
sys.path.insert(0, sys.argv[1])
import pygame, audio_engine as ae, sdl_audio_safe
eng = ae.AudioEngine("")
assert eng.start(), eng.error
time.sleep(0.3)
# What the render thread does when the engine restarts (Preferences > Save, opening a show):
sdl_audio_safe.init_pygame(); pygame.display.init(); pygame.display.set_mode((64, 64)); pygame.display.quit()
c0 = eng.callbacks; time.sleep(0.5); c1 = eng.callbacks
pygame.quit()                     # what it used to do
c2 = eng.callbacks; time.sleep(0.5); c3 = eng.callbacks
print("CALLBACKS", c1 - c0, c3 - c2)
"""


def test_sound_output_survives_a_render_engine_restart(tmp_path):
    """Owner's report 2026-11-13: no sound. The render thread ended with
    pygame.quit(), which also shuts SDL's audio, and Preferences > Save restarts
    the render thread right after opening the sound output."""
    env = dict(os.environ, SDL_AUDIODRIVER="disk", SDL_DISKAUDIOFILE=str(tmp_path / "o.raw"),
               SDL_VIDEODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    r = subprocess.run([sys.executable, "-c", RESTART_SCRIPT, os.path.join(os.path.dirname(HERE), "src")],
                       env=env, capture_output=True, text=True, timeout=60)
    line = next(x for x in r.stdout.splitlines() if x.startswith("CALLBACKS"))
    after_display_quit, after_pygame_quit = map(int, line.split()[1:])
    assert after_display_quit > 10, "closing the display must leave the sound running"
    assert after_pygame_quit == 0, "pygame.quit() stops it (why the render thread no longer calls it)"


def test_render_thread_shutdown_leaves_sdl_audio_alone(monkeypatch):
    import queue
    import threading
    import pygame
    from test_render_thread import FlakyEngine
    calls = []
    monkeypatch.setattr(pygame, "quit", lambda: calls.append("quit"))
    monkeypatch.setattr(pygame.display, "quit", lambda: calls.append("display.quit"))
    stop = threading.Event()
    rt = ci_mod.RenderThread(FlakyEngine(), queue.Queue(), queue.Queue(maxsize=1), stop)
    rt.start()
    time.sleep(0.2)
    stop.set()
    rt.join(2)
    assert calls == ["display.quit"]


def test_watchdog_reopens_a_stalled_output(monkeypatch):
    from media_manager import MediaManager
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    ci = ci_mod.ControlInterface(MediaManager())
    try:
        eng = ae.AudioEngine(backend="offline")
        eng.backend = "sdl"                       # pretend: a device whose callbacks stopped
        ci._audio, ci.audio_device = eng, "System default"
        reopened = []
        ci._restart_audio = lambda: reopened.append(1)
        assert not ci._audio_watchdog(now=100.0)
        assert not ci._audio_watchdog(now=101.0)
        eng.callbacks += 5
        assert not ci._audio_watchdog(now=101.5), "callbacks moving: fine"
        assert not ci._audio_watchdog(now=103.0)
        assert ci._audio_watchdog(now=103.6) and reopened == [1], "2 s without a callback: reopen"
        assert ci._audio_status()["restarts"] == 1
    finally:
        ci.proxy_executor.shutdown(wait=False)


def test_test_tone_mixes_in_and_ends():
    eng = ae.AudioEngine(backend="offline")
    tone = eng.play_test_tone(freq=1000, seconds=0.25)
    eng.set_streams({})                                  # the render thread's update must not drop it
    out = eng.mix(ae.RATE // 4)[:, 0]
    assert abs(dominant(out[500:-500]) - 1000) < 5 and 0.2 < np.abs(out).max() <= 0.25
    assert abs(out[0]) < 0.01, "faded in (no click)"
    assert tone.ended
    eng.set_streams({})
    assert "__test_tone__" not in eng._streams


TONE_SCRIPT = r"""
import os, sys, time, numpy as np
sys.path.insert(0, sys.argv[1])
import audio_engine as ae
eng = ae.AudioEngine("")
assert eng.start(), eng.error
eng.play_test_tone(freq=440, seconds=0.5)
time.sleep(0.8)
eng.stop()
print("DONE")
"""


def test_test_tone_reaches_the_sdl_output(tmp_path):
    raw = str(tmp_path / "tone.raw")
    env = dict(os.environ, SDL_AUDIODRIVER="disk", SDL_DISKAUDIOFILE=raw, PYGAME_HIDE_SUPPORT_PROMPT="1")
    r = subprocess.run([sys.executable, "-c", TONE_SCRIPT, os.path.join(os.path.dirname(HERE), "src")],
                       env=env, capture_output=True, text=True, timeout=60)
    assert "DONE" in r.stdout, r.stderr
    d = np.fromfile(raw, np.float32).reshape(-1, 2)[:, 0]
    start = int(np.argmax(np.abs(d) > 0.01))
    seg = d[start + 1000:start + 1000 + ae.RATE // 5]
    assert len(seg) == ae.RATE // 5 and abs(dominant(seg) - 440) < 5


def test_stopping_an_audio_device_does_not_deadlock_with_its_callback(tmp_path):
    """pygame's AudioDevice.pause(1) holds Python's lock while SDL waits for the
    callback, which needs that lock: with busy Python threads the first pause
    hung. sdl_audio_safe calls SDL through ctypes (lock released)."""
    import subprocess
    import sys
    import textwrap
    script = tmp_path / "stress.py"
    script.write_text(textwrap.dedent(f'''
        import sys, threading, time, faulthandler
        faulthandler.dump_traceback_later(40, exit=True)
        sys.path.insert(0, {str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src")!r})
        import pygame, pygame._sdl2.audio as a
        import sdl_audio_safe
        sdl_audio_safe.init_pygame()        # as the app: pygame's mixer would hold SDL's output too
        name = a.get_audio_device_names(False)[0]
        def cb(dev, mem):
            mem[:] = bytes(len(mem))
        def busy():
            while True:
                sum(range(2000))
        for _ in range(3):
            threading.Thread(target=busy, daemon=True).start()
        assert sdl_audio_safe.sdl() is not None, "pygame's SDL not found through ctypes"
        for i in range(60):
            d = a.AudioDevice(devicename=name, iscapture=False, frequency=48000, audioformat=a.AUDIO_F32,
                              numchannels=2, chunksize=256, allowed_changes=0, callback=cb)
            d.pause(0)
            time.sleep(0.01)
            sdl_audio_safe.pause_and_close(d)
        print("STRESS_OK")
    '''))
    env = dict(__import__("os").environ, SDL_AUDIODRIVER="disk", SDL_DISKAUDIOFILE=str(tmp_path / "out.raw"))
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60, env=env)
    assert "STRESS_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


MIXER_SCRIPT = r"""
import sys
sys.path.insert(0, sys.argv[1])
import pygame, pygame._sdl2.audio as a, sdl_audio_safe as s
s.init_pygame()
s.init_pygame()                      # again, as when the render thread restarts
names = a.get_audio_device_names(False)
d = a.AudioDevice(devicename=names[0], iscapture=False, frequency=48000, audioformat=a.AUDIO_F32,
                  numchannels=2, chunksize=256, allowed_changes=0, callback=lambda dev, mem: None)
s.pause_and_close(d)
print("MIXER", pygame.mixer.get_init(), "AUDIO", bool(s.sdl().SDL_WasInit(s.SDL_INIT_AUDIO)), "OPENED", True)
"""


def test_pygame_init_leaves_no_second_sound_output(tmp_path):
    """pygame.init() also opens pygame.mixer on the default output; VisualDeck
    does not use it. Found on the owner's PC: SDL's file output then refused
    VisualDeck's own open. The mixer is closed again and SDL's audio stays up."""
    env = dict(os.environ, SDL_AUDIODRIVER="disk", SDL_DISKAUDIOFILE=str(tmp_path / "m.raw"),
               PYGAME_HIDE_SUPPORT_PROMPT="1")
    r = subprocess.run([sys.executable, "-c", MIXER_SCRIPT, os.path.join(os.path.dirname(HERE), "src")],
                       env=env, capture_output=True, text=True, timeout=60)
    assert "MIXER None AUDIO True OPENED True" in r.stdout, r.stdout + r.stderr


def test_each_files_sound_format_is_logged_once(clips, caplog):
    import logging
    ae._formats_logged.discard(clips["500"])
    with caplog.at_level(logging.INFO, logger="visualdeck.audio"):
        for _ in range(2):
            s = ae.AudioStream(clips["500"])
            s.start()
            assert wait(lambda: s._buffered > 0)
            s.stop()
            s.join(2)
    lines = [r.getMessage() for r in caplog.records if "Hz" in r.getMessage()]
    assert lines == ["Audio of tone500.mp4: aac 44100 Hz mono fltp, resampled to 48000 Hz stereo"], lines


def test_a_file_without_sound_is_logged_once(tmp_path, caplog):
    import logging
    import media_fixtures as mf
    clip = str(tmp_path / "silent_video.mp4")
    mf.counting_clip(clip, n_frames=10)
    with caplog.at_level(logging.INFO, logger="visualdeck.audio"):
        for _ in range(3):
            s = ae.AudioStream(clip)
            s.start()
            assert wait(lambda: s.state == "no audio")
            s.join(2)
    assert [r.getMessage() for r in caplog.records if "no sound track" in r.getMessage()] == \
        ["Audio of silent_video.mp4: the file has no sound track"]
