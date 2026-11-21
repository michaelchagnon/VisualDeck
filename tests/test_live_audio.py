# SPDX-License-Identifier: MPL-2.0
"""Sound of live sources: the buffer, the player decoding it, the mixer and the render thread."""
import time

import numpy as np

import audio_engine as ae


def tone(n, hz=440.0, rate=48000, level=0.3):
    t = np.arange(n) / rate
    x = (level * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    return np.stack([x, x], axis=1)


def test_buffer_waits_a_little_then_plays_and_keeps_the_delay_short():
    b = ae.LiveAudioBuffer()
    b.push(tone(1000))
    assert not b.read(480).any()                            # less than START_S: silence, nothing consumed
    b.push(tone(4000))
    out = b.read(6000)
    assert np.abs(out[:5000]).max() > 0.2 and not out[5000:].any() and b.underruns == 1  # then ran dry
    assert not b.read(480).any()                            # builds up again before playing on
    for _ in range(40):                                     # the source sends much more than is read
        b.push(tone(1024))
    assert b._buffered <= int(b.MAX_S * b.rate) and b.dropped > 0
    b.clear()
    assert b._buffered == 0 and not b.read(480).any()


def test_mixer_plays_a_live_buffer_at_its_gain():
    eng = ae.AudioEngine(backend="offline")
    b = ae.LiveAudioBuffer()
    b.push(tone(48000))
    eng.set_streams({"udp://x": {"path": "udp://x", "gain": 0.5, "playing": True, "live": b}})
    assert eng._streams["udp://x"]["stream"] is b and b._buffered == 0     # cleared: sound from now on
    b.push(tone(48000))
    eng.mix(4800)                                           # the gain ramps up
    out = eng.mix(4800)[:, 0]
    assert abs(np.abs(out).max() - 0.15) < 0.01
    eng.set_streams({})                                     # off the output: the player keeps its buffer
    assert "udp://x" not in eng._streams


def test_live_player_decodes_the_sound_only_when_asked():
    import render_engine as rem
    from ring_buffer import FastRingBuffer
    from stream_sender import UdpSender, free_udp_port
    port = free_udp_port()
    snd = UdpSender(port, tone_hz=440).start()
    p = rem.LiveSourcePlayer(f"udp://127.0.0.1:{port}", FastRingBuffer(2), output_format="pygame")
    p.start()
    try:
        deadline = time.monotonic() + 10
        while p.frames_decoded < 10 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert p.has_audio is True and p._audio_buf is None   # connected; nobody listens yet
        buf = p.audio_buffer()
        time.sleep(1.0)
        chunks = [buf.read(2400) for _ in range(5)]
        x = np.concatenate(chunks)[:, 0]
        assert np.abs(x).max() > 0.2, "the tone arrives"
        spec = np.abs(np.fft.rfft(x))
        assert abs(np.argmax(spec) * 48000 / len(x) - 440) < 15
        assert p.frames_decoded > 10 and p.state == "playing"  # the picture keeps coming
    finally:
        p.stop()
        p.join(5)
        snd.stop()


def test_render_thread_sends_a_live_sources_sound():
    import control_interface as ci_mod

    class Audio:
        def set_streams(self, w):
            self.w = w

    class LivePlayer:
        held, has_audio = False, True

        def __init__(self):
            self.buf = ae.LiveAudioBuffer()

        def audio_buffer(self):
            return self.buf

    class Silent(LivePlayer):
        has_audio = False

    class Eng:
        @staticmethod
        def source_key(layer):
            return layer["path"]
    rt = ci_mod.RenderThread.__new__(ci_mod.RenderThread)
    rt.audio, rt.re = Audio(), Eng()
    lp = LivePlayer()
    rt.re.active_players = {"udp://a:1": {"player": lp, "path": "udp://a:1", "type": "live"},
                            "udp://b:2": {"player": Silent(), "path": "udp://b:2", "type": "live"}}
    rt._update_audio({"layers": {0: {"path": "udp://a:1", "volume": 0.5}, 1: {"path": "udp://b:2"}},
                      "transition_state": {"active": False}})
    assert rt.audio.w == {"udp://a:1": {"path": "udp://a:1", "gain": 0.5, "playing": True, "live": lp.buf}}
