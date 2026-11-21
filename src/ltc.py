# SPDX-License-Identifier: MPL-2.0
"""SMPTE linear timecode (LTC) from an audio input.

LTC is timecode recorded as audio: 80 bits per frame, biphase mark coded (a
level change at the start of every bit, and one more in the middle of a 1).
A frame holds the time in BCD, a drop-frame flag and user bits, and ends with
the sync word 0011 1111 1111 1101, which also gives the direction.

- LTCDecoder: feed it samples (mono float); it returns each frame's time,
  advanced by one frame (a frame's time is known when its last bit has
  arrived, which is when the next frame starts). The bit rate is tracked as
  it goes (varispeed and tape wow are fine), the frame rate is taken from it
  (24, 25 or 30; 30 with the drop-frame flag is 29.97 drop, counted as 30
  frames per timecode second like MTC, see timecode.py). Played backwards,
  frames are recognized but not used: following timecode only moves forward.
- encode(): LTC audio for a run of frames (tests and loopback checks).
- LTCInput: an audio input device (SDL capture) feeding a decoder on SDL's
  audio thread; `timecode` is the newest time, a new tuple for every frame,
  like MidiRemote.timecode.

No UI here.
"""
from __future__ import annotations

import collections
import logging
import threading
import time

import numpy as np

import sdl_audio_safe
import timecode

log = logging.getLogger("visualdeck.ltc")

SYNC = (0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1)        # bits 64..79, in the order sent
SYNC_BACKWARDS = tuple(reversed(SYNC))
RATE_CODES = {24: 0, 25: 1, 30: 3}                             # timecode.RATES codes; 2 = 29.97 drop
CHANNELS = ("Left", "Right", "Mix")
DEFAULT_DEVICE = "System default"


def _bcd(bits, start, n):
    return sum(bits[start + i] << i for i in range(n))


def frame_bits(tc, fps=25, drop=False, user=0):
    """The 80 bits of one LTC frame, in the order sent."""
    h, m, s, f = tc[:4]
    bits = [0] * 80

    def put(start, n, value):
        for i in range(n):
            bits[start + i] = (value >> i) & 1
    put(0, 4, f % 10)
    put(8, 2, f // 10)
    bits[10] = 1 if drop else 0
    put(16, 4, s % 10)
    put(24, 3, s // 10)
    put(32, 4, m % 10)
    put(40, 3, m // 10)
    put(48, 4, h % 10)
    put(56, 2, h // 10)
    for g in range(8):                                         # user bits: 4 per group
        put((4, 12, 20, 28, 36, 44, 52, 60)[g], 4, (user >> (4 * g)) & 0x0F)
    bits[64:80] = SYNC
    # Polarity correction bit: an even number of zeros per frame, so every
    # frame starts with the same level (bit 27 at 30 fps, bit 59 at 24/25).
    pc = 27 if fps == 30 else 59
    if bits.count(0) % 2:
        bits[pc] ^= 1
    return bits


def decode_bits(bits):
    """(h, m, s, f, drop) from the first 64 bits of a frame, or None."""
    f = _bcd(bits, 0, 4) + 10 * _bcd(bits, 8, 2)
    s = _bcd(bits, 16, 4) + 10 * _bcd(bits, 24, 3)
    m = _bcd(bits, 32, 4) + 10 * _bcd(bits, 40, 3)
    h = _bcd(bits, 48, 4) + 10 * _bcd(bits, 56, 2)
    if _bcd(bits, 0, 4) > 9 or _bcd(bits, 16, 4) > 9 or _bcd(bits, 32, 4) > 9 or _bcd(bits, 48, 4) > 9:
        return None
    if timecode.parse("%02d:%02d:%02d:%02d" % (h, m, s, f)) is None:
        return None
    return h, m, s, f, bool(bits[10])


def encode(start_tc, frames, fps=25, rate=48000, level=0.5, drop=False, rise_samples=0, speed=1.0):
    """float32 mono LTC audio for `frames` frames from start_tc (fps 24, 25,
    29.97 or 30). rise_samples > 0 softens the edges (a moving average), as a
    real cable and converter do; speed != 1 plays it faster or slower."""
    nominal = 30 if fps in (29.97, 30) else int(fps)
    bits = []
    tc = tuple(start_tc[:4])
    for _ in range(frames):
        bits += frame_bits(tc, nominal, drop)
        tc = timecode.add_frames(tc, 1, nominal)
    spb = rate / (80.0 * fps * speed)                          # samples per bit
    n = int(round(len(bits) * spb))
    edges = []                                                 # sample positions where the level changes
    for i, b in enumerate(bits):
        edges.append(i * spb)
        if b:
            edges.append((i + 0.5) * spb)
    toggles = np.zeros(n + 1, dtype=np.int64)
    for e in edges:
        k = int(round(e))
        if k <= n:
            toggles[k] += 1
    sign = np.where(np.cumsum(toggles[:n]) % 2 == 1, 1.0, -1.0)
    out = sign * level
    if rise_samples > 0:
        k = np.ones(int(rise_samples)) / int(rise_samples)
        out = np.convolve(out, k, mode="same")
    return out.astype(np.float32)


class LTCDecoder:
    """Streaming LTC decoder. feed(samples) -> [(h, m, s, f, rate_code), ...]."""

    MIN_LEVEL = 0.01                     # half the range below this: silence
    ENVELOPE_S = 0.02                    # the range is measured over this much signal

    def __init__(self, rate=48000):
        self.rate = rate
        self.bit = rate / 2000.0         # samples per bit; 25 fps to start, then tracked
        self._state = 0                  # input level: +1, -1, or 0 (unknown: start, after silence)
        self._last_edge = None           # absolute sample index of the last level change
        self._pos = 0                    # absolute index of the next sample fed
        self._half = False               # the first half of a 1 has been seen
        self._bits = []
        self._env = collections.deque()  # (samples, max, min) of recent blocks
        self.ENVELOPE = int(rate * self.ENVELOPE_S)
        self._prev = None                # last decoded (h, m, s, f)
        self._counted_fps = None         # frames per second seen at a second boundary
        self.frames = 0                  # frames decoded (forward)
        self.backwards = 0               # frames recognized running backwards
        self.errors = 0                  # bit timing errors (noise, dropouts)
        self.level = 0.0                 # peak input level of the last block
        self.fps = None

    def feed(self, samples):
        x = np.asarray(samples, dtype=np.float64)
        if x.size == 0:
            return []
        start = self._pos
        self._pos += x.size
        # The signal's range over the last ENVELOPE samples (at least): level
        # changes are found with hysteresis around its middle, so a DC offset
        # or any level works and silence is recognized.
        self._env.append((x.size, float(x.max()), float(x.min())))
        total = sum(n for n, _, _ in self._env)
        while len(self._env) > 1 and total - self._env[0][0] >= self.ENVELOPE:
            total -= self._env.popleft()[0]
        hi = max(h for _, h, _ in self._env)
        lo = min(m for _, _, m in self._env)
        amp, mid = (hi - lo) / 2.0, (hi + lo) / 2.0
        self.level = amp
        if amp < self.MIN_LEVEL:
            self._last_edge = None
            self._half = False
            self._bits = []              # a frame cut off by the silence is lost
            self._state = 0              # the first level after silence starts a bit
            return []
        th = 0.3 * amp
        s = np.where(x > mid + th, 1, np.where(x < mid - th, -1, 0))
        idx = np.where(s != 0, np.arange(s.size), -1)
        idx = np.maximum.accumulate(idx)
        filled = np.where(idx >= 0, s[np.maximum(idx, 0)], self._state)
        prev = np.concatenate([[self._state], filled[:-1]])
        edges = np.nonzero((filled != prev) & (filled != 0))[0]
        self._state = int(filled[-1])
        out = []
        for e in edges:
            pos = start + int(e)
            if self._last_edge is not None:
                r = self._interval(pos - self._last_edge)
                if r is not None:
                    out.append(r)
            self._last_edge = pos
        return out

    def _interval(self, d):
        bit = self.bit
        if d > 1.6 * bit or d < 0.3 * bit:     # a dropout or a glitch: start over
            self.errors += 1
            self._half = False
            self._bits = []
            if 0.8 * self.rate / 2400.0 <= d <= 1.25 * self.rate / 1920.0:
                self.bit = float(d)               # maybe the rate changed: try this as a bit
            return None
        if d > 0.75 * bit:                        # a whole bit: 0
            if self._half:                        # a 1 that never got its second half
                self.errors += 1
                self._half = False
            self.bit += 0.1 * (d - self.bit)
            return self._push(0)
        if not self._half:                        # first half of a 1
            self._half = True
            return None
        self._half = False
        self.bit += 0.1 * (2 * d - self.bit)
        return self._push(1)

    def _frame_rate(self, h, m, s, f):
        """24, 25 or 30. Counted at a second boundary (the frame before frame 0)
        once seen, so varispeed cannot make 25 look like 24; until then from
        the highest frame number and the bit rate."""
        prev, self._prev = self._prev, (h, m, s, f)
        if f == 0 and prev is not None and prev[3] + 1 in (24, 25, 30):
            if timecode.add_frames(prev, 1, prev[3] + 1) == (h, m, s, 0):
                self._counted_fps = prev[3] + 1
        if self._counted_fps is not None and f < self._counted_fps:
            return self._counted_fps
        if f >= 25:
            return 30
        guess = min((24, 25, 30), key=lambda r: abs(self.rate / (80.0 * self.bit) - r))
        return max(guess, 25) if f == 24 else guess

    def _push(self, b):
        bits = self._bits
        bits.append(b)
        if len(bits) > 160:
            del bits[:-80]
        if len(bits) < 80:
            return None
        tail = tuple(bits[-16:])
        if tail == SYNC:
            frame = bits[-80:]
            self._bits = []
            got = decode_bits(frame)
            if got is None:
                self.errors += 1
                return None
            h, m, s, f, drop = got
            fps = self._frame_rate(h, m, s, f)
            if f >= fps:
                self.errors += 1
                return None
            self.fps = fps
            code = 2 if (fps == 30 and drop) else RATE_CODES[fps]
            self.frames += 1
            return timecode.add_frames((h, m, s, f), 1, fps) + (code,)
        if tail == SYNC_BACKWARDS:
            self.backwards += 1
            self._bits = []
        return None


def capture_device_names():
    """(names, error) of the audio inputs SDL can open here."""
    try:
        import pygame
        import pygame._sdl2.audio as sdl_audio
        if not pygame.get_init():
            sdl_audio_safe.init_pygame()       # without pygame.mixer
        return list(sdl_audio.get_audio_device_names(True)), None
    except Exception as e:  # noqa: BLE001
        return [], f"{type(e).__name__}: {e}"[:200]


class LTCInput:
    """Reads LTC from an audio input. `timecode` is the newest time (a new
    tuple per frame: h, m, s, f, rate code, time.monotonic() when it came, as
    MidiRemote.timecode). backend="offline" opens nothing (tests feed()
    samples themselves)."""

    RATE = 48000
    CHUNK = 512

    def __init__(self, device=None, channel="Left", backend="sdl"):
        self.device_name = device if device and device != DEFAULT_DEVICE else None
        self.channel = channel if channel in CHANNELS else "Left"
        self.backend = backend
        self.decoder = LTCDecoder(self.RATE)
        self.timecode = None
        self.last_time = None
        self.opened = None
        self.error = None
        self.callbacks = 0
        self._dev = None
        self._lock = threading.Lock()

    @property
    def listening(self):
        return self.opened is not None

    def start(self):
        if self.backend == "offline":
            self.opened = "offline"
            return True
        try:
            import pygame
            import pygame._sdl2.audio as sdl_audio
            if not pygame.get_init():
                sdl_audio_safe.init_pygame()       # without pygame.mixer
            names = list(sdl_audio.get_audio_device_names(True))
            name = self.device_name if self.device_name in names else (names[0] if names else None)
            if name is None:
                raise RuntimeError("no audio inputs")
            if self.device_name and self.device_name not in names:
                log.warning(f"LTC input '{self.device_name}' not found; using '{name}'")
            self._dev = sdl_audio.AudioDevice(devicename=name, iscapture=True, frequency=self.RATE,
                                              audioformat=sdl_audio.AUDIO_F32, numchannels=2,
                                              chunksize=self.CHUNK, allowed_changes=0, callback=self._callback)
            self._dev.pause(0)
            self.opened = name
            log.info(f"LTC input '{name}' ({self.channel}) at {self.RATE} Hz")
            return True
        except Exception as e:  # noqa: BLE001
            self.error = f"cannot open audio input: {e}"[:200]
            log.error(f"LTC: {self.error}")
            self._dev = None
            return False

    def stop(self):
        dev, self._dev = self._dev, None
        self.opened = None
        if dev is not None:
            try:
                import sdl_audio_safe          # pygame's pause/close can deadlock with the callback
                sdl_audio_safe.pause_and_close(dev)
            except Exception:  # noqa: BLE001
                pass

    def _callback(self, _dev, mem):
        try:
            self.callbacks += 1
            x = np.frombuffer(bytes(mem), dtype=np.float32).reshape(-1, 2)
            self.feed_stereo(x)
        except Exception:  # noqa: BLE001 - never raise into SDL's audio thread
            log.exception("LTC input callback failed")

    def feed_stereo(self, x):
        if self.channel == "Left":
            mono = x[:, 0]
        elif self.channel == "Right":
            mono = x[:, 1]
        else:
            mono = x.mean(axis=1)
        self.feed(mono)

    def feed(self, mono):
        with self._lock:
            got = self.decoder.feed(mono)
        if got:
            self.last_time = time.monotonic()
            self.timecode = tuple(got[-1]) + (self.last_time,)

    def status(self):
        d = self.decoder
        st = {"listening": self.listening, "device": self.opened, "channel": self.channel,
              "frames": d.frames, "errors": d.errors, "level": round(d.level, 3)}
        if self.timecode is not None:
            st["timecode"] = timecode.fmt(self.timecode)
            st["rate"] = timecode.RATE_LABELS.get(self.timecode[4], "?")
            st["age_s"] = round(time.monotonic() - self.last_time, 2)
        if d.backwards:
            st["backwards"] = d.backwards
        if self.error:
            st["error"] = self.error
        return st
