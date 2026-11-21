# SPDX-License-Identifier: MPL-2.0
"""Audio from videos: decode each playing video's sound and mix it to one output.

Off until an output device is chosen (Preferences > Audio). Each video layer
on the output gets an AudioStream (its own decoder of the file's first audio
track, resampled to the device rate, stereo float32), started with the
picture and restarted when the video loops. The mixer runs in SDL's audio
callback: the sum of the streams, each multiplied by its gain (the layer's
Volume times the cue's fade), with the gain ramped over each buffer so changes
never click, then clipped to -1..1.

Kept simple on purpose (first version):
- Audio and picture start together and re-align on every loop; within one
  loop the picture is paced by the computer's clock and the sound by the
  audio device's, so they can drift apart on long clips (typically well under
  0.1 % of the clip length; not measured on real hardware).
- A video at a speed other than 1x plays silent.
- The next cue's sound is decoded ahead (paused), so it starts with the picture.

The render thread calls AudioEngine.set_streams() every frame; nothing here
touches the UI or OpenGL. `pygame._sdl2.audio` is the device backend (SDL2,
already used for the windows); tests use an offline backend.
"""
from __future__ import annotations

import collections
import logging
import threading
import time

import numpy as np

import sdl_audio_safe

log = logging.getLogger("visualdeck.audio")

DEFAULT_DEVICE = "System default"
RATE = 48000
CHUNK = 1024               # frames per device callback (about 21 ms)
AHEAD_S = 0.5              # decoded audio kept ready per stream


def device_names():
    """(names, error): the system's audio outputs."""
    try:
        import pygame
        import pygame._sdl2.audio as sdl_audio
        if not pygame.get_init():
            sdl_audio_safe.init_pygame()       # without pygame.mixer
        return list(sdl_audio.get_audio_device_names(False)), None
    except Exception as e:  # noqa: BLE001
        return [], f"no audio system: {e}"


def has_audio(path):
    """True if the file has an audio track (cheap: opens the header)."""
    try:
        import av
        with av.open(path) as c:
            return bool(c.streams.audio)
    except Exception:  # noqa: BLE001
        return False


class LiveAudioBuffer:
    """Sound of a live source (render_engine.LiveSourcePlayer pushes it as it
    arrives). The mixer reads it like an AudioStream. It keeps at most MAX_S
    of sound: when the source sends faster than the sound card plays (their
    clocks differ), the oldest is dropped, so the delay stays short. When
    nothing has arrived, silence."""

    MAX_S = 0.25
    START_S = 0.06                  # wait for this much before playing, then keep about that much ahead

    def __init__(self, rate=RATE):
        self.rate = rate
        self._buf = collections.deque()
        self._buffered = 0
        self._lock = threading.Lock()
        self._started = False
        self.state = "starting"
        self.ended = False
        self.underruns = 0
        self.dropped = 0
        self.frames_out = 0
        self.loop, self.trim, self.error = True, (None, None), None

    def push(self, arr):
        with self._lock:
            self._buf.append(arr)
            self._buffered += len(arr)
            limit = int(self.MAX_S * self.rate)
            while self._buffered > limit and len(self._buf) > 1:
                old = self._buf.popleft()
                self._buffered -= len(old)
                self.dropped += len(old)
        self.state = "playing"

    def read(self, n):
        out = np.zeros((n, 2), np.float32)
        with self._lock:
            if not self._started:
                if self._buffered < int(self.START_S * self.rate):
                    return out
                self._started = True
            got = 0
            while got < n and self._buf:
                a = self._buf[0]
                take = min(n - got, len(a))
                out[got:got + take] = a[:take]
                if take == len(a):
                    self._buf.popleft()
                else:
                    self._buf[0] = a[take:]
                got += take
            self._buffered -= got
            if got < n:
                self.underruns += 1
                self._started = False        # build up a little again before playing on
        self.frames_out += got
        return out

    def restart(self):
        pass                                 # a live source does not loop

    def stop(self):
        pass                                 # owned by the live source's player

    def clear(self):
        with self._lock:
            self._buf.clear()
            self._buffered = 0
            self._started = False


BATCH_S = 0.25             # decoded audio resampled per call when the resampler is needed

# Integer sample formats as FFmpeg converts them to float (x / 2^(bits-1)).
_INT_SCALE = {"s16": 1.0 / 32768.0, "s32": 1.0 / 2147483648.0}
_FAST_FORMATS = {"flt", "fltp", "dbl", "dblp", "s16", "s16p", "s32", "s32p"}
_MONO_GAIN = np.float32(np.sqrt(0.5))     # FFmpeg's mono to stereo: the same value on each side


class StereoConverter:
    """Decoded audio frames to interleaved float32 stereo at `rate`, (n, 2) arrays: what
    av.AudioResampler(format="flt", layout="stereo", rate=rate) gives, called less.

    PyAV 15.0.0 on Python 3.12 keeps one Python string (41 bytes plus the message)
    for every FFmpeg error it raises with a system error code, and its resampler
    raises one (EAGAIN) in every call: one call per decoded frame grew a show with
    sound by about 12 MB an hour per playing 48 kHz stream (owner's PC, soak traced
    2026-11-15; tests/test_pyav_memory.py). So stereo and mono frames already at the
    rate are converted here with numpy (the same values as the resampler, tested), and other
    frames go to the resampler in batches of `batch_s` (0: one call per frame, for
    live sources, where a batch would add its length to the delay)."""

    def __init__(self, rate=RATE, batch_s=BATCH_S):
        self.rate, self.batch_s = rate, batch_s
        self._resampler = None
        self._batch, self._batched = [], 0
        self.resample_calls = 0

    def convert(self, frame):
        """Arrays ready now, in order (none while a batch fills)."""
        fast = self._fast(frame)
        if fast is not None:
            return self._drain() + [fast]
        self._batch.append(frame)
        self._batched += frame.samples
        if self._batched >= self.batch_s * (frame.sample_rate or self.rate):
            return self._drain()
        return []

    def flush(self):
        """At the end of the audio: the batch and what the resampler still holds."""
        out = self._drain()
        if self._resampler is not None:
            self.resample_calls += 1
            out += [self._array(rf) for rf in self._resampler.resample(None)]
            self._resampler = None
        return out

    def _fast(self, frame):
        fmt, layout = frame.format.name, frame.layout.name
        if frame.sample_rate != self.rate or fmt not in _FAST_FORMATS or layout not in ("stereo", "mono"):
            return None
        arr = frame.to_ndarray()
        if layout == "mono":
            out = arr.reshape(-1).astype(np.float32)
        else:
            out = (arr.T if frame.format.is_planar else arr.reshape(-1, 2)).astype(np.float32)
        scale = _INT_SCALE.get(fmt.rstrip("p"))
        if scale is not None:
            out *= np.float32(scale)
        if layout == "mono":                  # to both sides at -3 dB, as FFmpeg's default matrix
            out *= _MONO_GAIN
            out = np.stack([out, out], axis=1)
        return np.ascontiguousarray(out)

    @staticmethod
    def _array(rf):
        return np.ascontiguousarray(rf.to_ndarray().reshape(-1, 2), dtype=np.float32)

    def _drain(self):
        if not self._batch:
            return []
        import av
        frames, self._batch, self._batched = self._batch, [], 0
        if self._resampler is None:
            self._resampler = av.AudioResampler(format="flt", layout="stereo", rate=self.rate)
        merged = self._merge(frames)
        out = []
        for f in ([merged] if merged is not None else frames):
            self.resample_calls += 1
            out += [self._array(rf) for rf in self._resampler.resample(f)]
        return out

    @staticmethod
    def _merge(frames):
        """One frame holding the samples of `frames` (same format and layout), or None."""
        if len(frames) == 1:
            return frames[0]
        import av
        f0 = frames[0]
        try:
            if any(f.format.name != f0.format.name or f.layout.name != f0.layout.name
                   or f.sample_rate != f0.sample_rate for f in frames):
                return None
            out = av.AudioFrame.from_ndarray(np.concatenate([f.to_ndarray() for f in frames], axis=1),
                                             format=f0.format.name, layout=f0.layout.name)
            out.sample_rate = f0.sample_rate
            if f0.time_base is not None:
                out.pts, out.time_base = f0.pts, f0.time_base
            return out
        except Exception as e:  # noqa: BLE001 - not mergeable (format): one call per frame
            log.debug(f"Audio frames not merged ({f0.format.name}, {f0.layout.name}): {e}")
            return None


_formats_logged = set()


def _log_format_once(path, name, cc, rate):
    """One log line per file and session: its sound format and whether it is
    converted directly (StereoConverter) or resampled in batches. (A file without
    sound is also logged once: 62 times in one morning's tests on the owner's PC.)"""
    if path in _formats_logged:
        return
    _formats_logged.add(path)
    try:
        fmt, layout, sr = cc.format.name, cc.layout.name, cc.sample_rate
        direct = sr == rate and fmt in _FAST_FORMATS and layout in ("stereo", "mono")
        log.info(f"Audio of {name}: {cc.name} {sr} Hz {layout} {fmt}, "
                 f"{'converted directly' if direct else f'resampled to {rate} Hz stereo'}")
    except Exception:  # noqa: BLE001 - diagnostics only
        pass


class AudioStream(threading.Thread):
    """Decodes one file's audio ahead of the mixer. read() never blocks."""

    def __init__(self, path, rate=RATE, loop=True, delay_s=0.0, trim=(None, None), start_at=None):
        super().__init__(daemon=True, name="Audio-" + path.replace("\\", "/").rsplit("/", 1)[-1][:40])
        self.path, self.rate, self.loop = path, rate, loop
        self.trim = tuple(trim) if trim else (None, None)   # (start_s, end_s): the video's in and out points
        self.start_at = float(start_at) if start_at else None   # Start From: the first pass starts here
        self._buf = collections.deque()          # float32 (n, 2) arrays
        self._buffered = 0                       # frames in _buf
        self._lock = threading.Lock()
        self._stop_event = threading.Event()     # not _stop: Thread.join() and is_alive() call self._stop()
        self._restart = threading.Event()
        self.state = "starting"                  # starting|playing|ended|no audio|error|stopped
        self.error = None
        self.underruns = 0
        self.frames_out = 0
        self.ended = False
        self._owed = 0                           # frames the mixer asked for but did not get
        self.delay_frames = max(0, int(round(delay_s * rate)))   # Audio delay (Preferences)
        self._silence = self.delay_frames        # silence still to play before the sound starts

    # -- consumer side (audio callback) --------------------------------------------
    def read(self, n):
        """n frames (float32, (n, 2)); silence where nothing is decoded yet. Frames
        asked for but missing are skipped when they arrive, so the sound stays in
        time with the picture (a late start or a slow decoder loses sound, not sync)."""
        out = np.zeros((n, 2), np.float32)
        got = quiet = 0
        with self._lock:
            if self._silence:                    # the delay: silence first, nothing consumed
                quiet = min(self._silence, n)
                self._silence -= quiet
                if quiet == n:
                    self.frames_out += n
                    return out
                got = quiet
            while self._owed and self._buf:
                a = self._buf[0]
                drop = min(self._owed, len(a))
                if drop == len(a):
                    self._buf.popleft()
                else:
                    self._buf[0] = a[drop:]
                self._buffered -= drop
                self._owed -= drop
            while got < n and self._buf:
                a = self._buf[0]
                take = min(n - got, len(a))
                out[got:got + take] = a[:take]
                if take == len(a):
                    self._buf.popleft()
                else:
                    self._buf[0] = a[take:]
                got += take
            self._buffered -= got - quiet
            if got < n and not self.ended:
                self._owed += n - got
        if got < n and not self.ended and self.state == "playing":
            self.underruns += 1
        self.frames_out += got
        return out

    MAX_CATCH_UP_S = 1.0         # a longer lag means something else is wrong: not skipped

    def restart(self, since=None):
        """Start again from the beginning (the video looped, at perf_counter `since`:
        the sound skips what it missed since then, so it stays with the picture)."""
        self._restart_since = since
        self._restart.set()

    def skip(self, seconds):
        """Drop `seconds` of sound from where it is now (it started that late)."""
        if 0 < seconds < self.MAX_CATCH_UP_S:
            with self._lock:
                self._owed += int(round(seconds * self.rate))

    def stop(self):
        self._stop_event.set()

    # -- decoder side --------------------------------------------------------------
    def _push(self, arr):
        with self._lock:
            self._buf.append(arr)
            self._buffered += len(arr)

    def _clear(self):
        with self._lock:
            self._buf.clear()
            self._buffered = 0
            self._owed = 0
            self._silence = self.delay_frames

    def run(self):
        import av
        try:
            container = av.open(self.path)
        except Exception as e:  # noqa: BLE001
            self.state, self.error = "error", f"cannot open: {e}"[:200]
            log.warning(f"Audio of {self.name[6:]}: {self.error}")
            return
        try:
            if not container.streams.audio:
                self.state = "no audio"
                if self.path not in _formats_logged:  # once per file (it was every preload)
                    _formats_logged.add(self.path)
                    log.info(f"Audio of {self.name[6:]}: the file has no sound track")
                return
            stream = container.streams.audio[0]
            _log_format_once(self.path, self.name[6:], stream.codec_context, self.rate)
            start_s, end_s = self.trim
            tb = float(stream.time_base) if stream.time_base else None
            s0 = float(stream.start_time or 0) * tb if tb else 0.0
            lo = max(self.start_at or 0.0, start_s or 0.0) or None   # where this pass starts
            if lo and end_s is not None:
                lo = min(lo, end_s)
            if lo and tb:
                container.seek(int(round(lo / tb)) + int(stream.start_time or 0), stream=stream)
            while not self._stop_event.is_set():
                conv = StereoConverter(self.rate)
                self.ended = False
                restarted = False
                cursor = None                        # clip time of the next sample pushed (trimmed clips)

                def take(arrays):
                    nonlocal cursor
                    for arr in arrays:
                        if cursor is not None and (lo or end_s is not None):
                            a = cursor
                            cursor += len(arr) / self.rate
                            if lo and a < lo:                        # before the in (or start) point
                                arr = arr[min(len(arr), int(round((lo - a) * self.rate))):]
                            if end_s is not None and cursor > end_s:  # past the out point
                                arr = arr[:max(0, len(arr) - int(round((cursor - end_s) * self.rate)))]
                        if len(arr):
                            self._push(arr)
                            self.state = "playing"           # sound is there (not before: a frame
                                                             # before the in point pushes nothing)

                for frame in container.decode(stream):
                    if self._stop_event.is_set():
                        return
                    if self._restart.is_set():
                        restarted = True
                        break
                    t = (float(frame.pts) * tb - s0) if (frame.pts is not None and tb) else None
                    if end_s is not None and t is not None and t >= end_s:
                        break                        # the out point
                    if cursor is None and t is not None:
                        cursor = t
                    take(conv.convert(frame))
                    while self._buffered > self.rate * AHEAD_S and not self._stop_event.is_set() \
                            and not self._restart.is_set():
                        time.sleep(0.01)
                if not restarted:
                    take(conv.flush())                # the last batch and the resampler's tail
                    self.ended = True
                    if not self.loop:
                        self.state = "ended"
                        while not self._stop_event.is_set() and not self._restart.is_set():
                            time.sleep(0.02)
                        if self._stop_event.is_set():
                            return
                    else:
                        # Loop by itself only if the video does not tell us (restart());
                        # wait for the video's loop signal up to a short grace time.
                        self._restart.wait(0.25)
                if self._restart.is_set():
                    self._restart.clear()
                    self._clear()
                    since = getattr(self, "_restart_since", None)
                    if since is not None:
                        self.skip(time.perf_counter() - since)   # the picture looped this long ago
                lo = start_s                         # later passes start at the in point
                if start_s and tb:
                    container.seek(int(round(start_s / tb)) + int(stream.start_time or 0), stream=stream)
                else:
                    container.seek(0, stream=stream)
        except Exception as e:  # noqa: BLE001 - a broken track must not take anything down
            self.state, self.error = "error", f"{type(e).__name__}: {e}"[:200]
            log.warning(f"Audio of {self.name[6:]}: {self.error}")
        finally:
            try:
                container.close()
            except Exception:  # noqa: BLE001
                pass
            if self.state not in ("error", "no audio"):
                self.state = "stopped"


class ToneStream:
    """A test tone with AudioStream's read() interface (Preferences > Test Sound)."""

    def __init__(self, freq=440.0, seconds=1.0, level=0.25, rate=RATE):
        self.freq, self.level, self.rate = float(freq), float(level), rate
        self.total = int(seconds * rate)
        self.pos = 0
        self.state, self.error, self.underruns, self.frames_out, self.ended = "playing", None, 0, 0, False

    def read(self, n):
        out = np.zeros((n, 2), np.float32)
        k = max(0, min(n, self.total - self.pos))
        if k:
            t = (np.arange(self.pos, self.pos + k) / self.rate)
            fade = np.minimum(1.0, np.minimum(np.arange(self.pos, self.pos + k),
                                              self.total - np.arange(self.pos, self.pos + k)) / (0.01 * self.rate))
            out[:k] = (self.level * fade * np.sin(2 * np.pi * self.freq * t)).astype(np.float32)[:, None]
            self.pos += k
            self.frames_out += k
        if self.pos >= self.total:
            self.ended, self.state = True, "ended"
        return out

    def restart(self):
        self.pos = 0

    def stop(self):
        self.pos = self.total


class AudioEngine:
    """Mixes the streams of the videos on the output to one audio device."""

    def __init__(self, device_name="", rate=RATE, chunk=CHUNK, backend="sdl", delay_ms=0):
        self.device_name, self.rate, self.chunk = device_name, rate, chunk
        self.delay_s = max(0.0, min(1.0, float(delay_ms or 0) / 1000.0))
        self.backend = backend
        self._streams = {}          # key -> {'stream', 'gain', 'target', 'playing', 'loops', 'path'}
        self._lock = threading.Lock()
        self._dev = None
        self.error = None
        self.callbacks = 0
        self.peak = 0.0             # last buffer's peak level (meter)

    @property
    def running(self):
        return self._dev is not None or self.backend == "offline"

    def start(self):
        if self.backend == "offline":
            return True
        try:
            import pygame
            import pygame._sdl2.audio as sdl_audio
            if not pygame.get_init():
                sdl_audio_safe.init_pygame()       # without pygame.mixer
            names = list(sdl_audio.get_audio_device_names(False))
            name = self.device_name if self.device_name in names else (names[0] if names else None)
            if name is None:
                raise RuntimeError("no audio output devices")
            if self.device_name and self.device_name not in names:
                log.warning(f"Audio output '{self.device_name}' not found; using '{name}'")
            self._dev = sdl_audio.AudioDevice(devicename=name, iscapture=False, frequency=self.rate,
                                              audioformat=sdl_audio.AUDIO_F32, numchannels=2,
                                              chunksize=self.chunk, allowed_changes=0, callback=self._callback)
            self._dev.pause(0)
            self.opened = name
            log.info(f"Audio output '{name}' at {self.rate} Hz")
            return True
        except Exception as e:  # noqa: BLE001
            self.error = f"cannot open audio output: {e}"[:200]
            log.error(f"Audio: {self.error}")
            self._dev = None
            return False

    def stop(self):
        dev, self._dev = self._dev, None
        if dev is not None:
            try:
                import sdl_audio_safe          # pygame's pause/close can deadlock with the callback
                sdl_audio_safe.pause_and_close(dev)
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            streams, self._streams = self._streams, {}
        for s in streams.values():
            s["stream"].stop()

    def set_streams(self, wanted):
        """wanted: {key: {'path', 'gain' (0..1+), 'playing' (bool), 'loop' (bool),
        'loops' (the video's loop count), 'trim' ((in_s, out_s), optional),
        'start_at' (Start From: the first pass's start, used when the stream is made),
        'clock_start' / 'loop_at' (perf_counter of the picture's start and last loop:
        the sound catches up to them)}}.
        Missing streams start, others stop."""
        with self._lock:
            cur = dict(self._streams)
        new = {}
        for key, w in wanted.items():
            e = cur.pop(key, None)
            if w.get("live") is not None:            # a live source's sound: its player's buffer
                if e is None or e["stream"] is not w["live"]:
                    w["live"].clear()                # sound from now on, not what queued meanwhile
                    e = {"stream": w["live"], "gain": 0.0, "path": w["path"], "loops": 0}
                e["target"] = max(0.0, float(w.get("gain", 1.0)))
                e["playing"] = bool(w.get("playing", True))
                new[key] = e
                continue
            trim = tuple(w.get("trim") or (None, None))
            if e is not None and tuple(e["stream"].trim) != trim:
                e["stream"].stop()                  # new in/out points: the video restarts, so does its sound
                e = None
            if e is None:
                st = AudioStream(w["path"], self.rate, loop=w.get("loop", True), delay_s=self.delay_s, trim=trim,
                                 start_at=w.get("start_at"))
                st.start()
                e = {"stream": st, "gain": 0.0, "path": w["path"], "loops": w.get("loops", 0)}
            elif w.get("loops", 0) != e["loops"]:
                e["stream"].restart(w.get("loop_at"))   # the video looped: start the sound again with it
                e["loops"] = w.get("loops", 0)
            if w.get("clock_start") is not None:
                e["clock_start"] = w["clock_start"]     # when the picture started (see mix)
            e["stream"].loop = w.get("loop", True)
            e["target"] = max(0.0, float(w.get("gain", 1.0)))
            e["playing"] = bool(w.get("playing", True))
            new[key] = e
        for key, e in list(cur.items()):            # a test tone stays until it has played
            if e.get("keep") and not e["stream"].ended:
                new[key] = cur.pop(key)
        with self._lock:
            self._streams = new
        for e in cur.values():                      # no longer on the output
            e["stream"].stop()

    def play_test_tone(self, freq=440.0, seconds=1.0, level=0.25):
        """A short tone (with 10 ms fades) mixed in with whatever is playing."""
        tone = ToneStream(freq, seconds, level, self.rate)
        with self._lock:
            self._streams = dict(self._streams)
            self._streams["__test_tone__"] = {"stream": tone, "gain": 1.0, "target": 1.0, "playing": True,
                                              "path": "test tone", "loops": 0, "keep": True}
        return tone

    def mix(self, n):
        """n frames of the mix, float32 (n, 2). Called by the device callback."""
        with self._lock:
            streams = list(self._streams.values())
        out = np.zeros((n, 2), np.float32)
        for e in streams:
            if not e.get("playing"):
                continue
            if not e.get("started"):
                e["started"] = True
                cs = e.get("clock_start")
                if cs is not None and hasattr(e["stream"], "skip"):
                    # Sound starts at the first callback after the render thread said
                    # "playing", a render loop or more after the picture's clock began.
                    e["stream"].skip(time.perf_counter() - cs)
            g0, g1 = e["gain"], e.get("target", 0.0)
            if g0 == 0.0 and g1 == 0.0:
                e["stream"].read(n)                 # keep it moving with the picture
                continue
            a = e["stream"].read(n)
            if g0 == g1:
                out += a * g1
            else:
                out += a * np.linspace(g0, g1, n, endpoint=False, dtype=np.float32)[:, None]
            e["gain"] = g1
        np.clip(out, -1.0, 1.0, out=out)
        self.peak = float(np.abs(out).max()) if n else 0.0
        return out

    def _callback(self, _dev, mem):
        try:
            self.callbacks += 1
            n = len(mem) // 8
            mem[:] = self.mix(n).tobytes()
        except Exception:  # noqa: BLE001 - silence rather than a dead audio thread
            mem[:] = bytes(len(mem))

    def status(self):
        with self._lock:
            streams = list(self._streams.values())
        return {"running": self.running, "device": getattr(self, "opened", None), "error": self.error,
                "streams": len(streams), "playing": sum(1 for e in streams if e.get("playing")),
                "underruns": sum(e["stream"].underruns for e in streams), "peak": round(self.peak, 3),
                "callbacks": self.callbacks,
                "stream_states": [{"file": e["path"].replace("\\", "/").rsplit("/", 1)[-1], "state": e["stream"].state,
                                   "error": e["stream"].error, "playing": bool(e.get("playing")),
                                   "gain": round(e.get("target", 0.0), 3)} for e in streams]}
