"""Audio decoding must not keep memory per frame.
# SPDX-License-Identifier: MPL-2.0

PyAV 15.0.0 on Python 3.12 keeps the message of every FFmpeg error it raises with
a system error code (EAGAIN, EIO, ENOENT): one Python string (41 bytes plus the
message) that is never freed. Its audio resampler raises EAGAIN inside every
call, so resampling each decoded frame grew a show with sound by about 12 MB an
hour per playing 48 kHz stream: the owner's 8-hour soak (2026-11-14/15) grew by
about 41,000 Python blocks an hour, and a traced soak on that PC named
`resampler.resample` in audio_engine.py, with blocks of exactly 73 bytes
("Resource temporarily unavailable"). Measured here: the same PyAV on Python 3.13,
and PyAV 17 and later (FFmpeg 8) on Python 3.12, keep nothing; PyAV 15.1.0 and
16.1.0 on Python 3.12 keep the strings too. audio_engine.StereoConverter avoids
the resampler for stereo and mono audio at the output rate and batches the rest.
"""
import gc
from fractions import Fraction
import sys

import av
import numpy as np
import pytest
from av.error import FFmpegError


def _blocks_per_call(fn, n=3000):
    for _ in range(100):
        fn()
    gc.collect()
    before = sys.getallocatedblocks()
    for _ in range(n):
        fn()
    gc.collect()
    return (sys.getallocatedblocks() - before) / n


def _pyav_keeps_error_messages():
    def raise_and_catch():
        try:
            av.error.err_check(-11)
        except FFmpegError:
            pass
    return _blocks_per_call(raise_and_catch, n=500) > 0.5


def _tone_frames(rate=44100, n_frames=40, size=1024, fmt="fltp", layout="stereo", timed=True):
    """Frames as a decoder hands them over (AAC: planar float, 1,024 samples)."""
    out, t = [], 0
    ch = 1 if layout == "mono" else 2
    for _ in range(n_frames):
        x = np.sin(2 * np.pi * 440 * np.arange(t, t + size) / rate).astype(np.float32)
        planar = np.stack([x * (0.9 - 0.3 * c) for c in range(ch)])
        if fmt.startswith("s16"):
            planar = (planar * 32000).astype(np.int16)
        elif fmt.startswith("s32"):
            planar = (planar * 2.1e9).astype(np.int32)
        elif fmt.startswith("dbl"):
            planar = planar.astype(np.float64)
        arr = planar if fmt.endswith("p") else planar.T.reshape(1, -1)
        f = av.AudioFrame.from_ndarray(np.ascontiguousarray(arr), format=fmt, layout=layout)
        f.sample_rate, f.pts = rate, t
        if timed:
            f.time_base = Fraction(1, rate)
        out.append(f)
        t += size
    return out


@pytest.mark.parametrize("rate", [48000, 44100])
def test_converting_audio_keeps_no_memory_per_frame(rate):
    import audio_engine
    frames = _tone_frames(rate=rate)
    conv = audio_engine.StereoConverter(48000)
    i = [0]

    def convert_one():
        conv.convert(frames[i[0] % len(frames)])
        i[0] += 1
    per_frame = _blocks_per_call(convert_one)
    assert per_frame < (0.01 if rate == 48000 else 0.15), per_frame     # 44.1 kHz: one call per ~11 frames
    if rate == 48000:
        assert conv.resample_calls == 0


def test_the_leak_this_works_around_is_measured_here():
    """Documents PyAV's behavior in this environment (not a requirement)."""
    leaks = _pyav_keeps_error_messages()
    print(f"PyAV {av.__version__}, Python {sys.version.split()[0]}: "
          f"{'keeps' if leaks else 'does not keep'} FFmpeg error messages")


@pytest.mark.parametrize("fmt", ["fltp", "flt", "s16", "s16p", "s32", "s32p", "dbl", "dblp"])
def test_mono_at_the_output_rate_matches_the_resampler(fmt):
    import audio_engine
    frames = _tone_frames(rate=48000, n_frames=6, fmt=fmt, layout="mono")
    conv = audio_engine.StereoConverter(48000)
    ours = np.concatenate([a for f in frames for a in conv.convert(f)] + conv.flush())
    r = av.AudioResampler(format="flt", layout="stereo", rate=48000)
    ref = np.concatenate([audio_engine.StereoConverter._array(rf) for f in frames for rf in r.resample(f)]
                         + [audio_engine.StereoConverter._array(rf) for rf in r.resample(None)])
    assert conv.resample_calls == 0 and ours.dtype == np.float32 and ours.shape == ref.shape == (6 * 1024, 2)
    assert np.allclose(ours, ref, rtol=0, atol=1e-7), float(np.abs(ours - ref).max())   # equal but for doubles
    if not fmt.startswith("dbl"):
        assert np.array_equal(ours, ref)


@pytest.mark.parametrize("fmt", ["fltp", "flt", "s16", "s16p", "s32", "s32p", "dbl", "dblp"])
def test_stereo_at_the_output_rate_matches_the_resampler(fmt):
    import audio_engine
    frames = _tone_frames(rate=48000, n_frames=6, fmt=fmt)
    conv = audio_engine.StereoConverter(48000)
    ours = np.concatenate([a for f in frames for a in conv.convert(f)] + conv.flush())
    r = av.AudioResampler(format="flt", layout="stereo", rate=48000)
    ref = np.concatenate([audio_engine.StereoConverter._array(rf) for f in frames for rf in r.resample(f)]
                         + [audio_engine.StereoConverter._array(rf) for rf in r.resample(None)])
    assert conv.resample_calls == 0 and ours.dtype == np.float32 and ours.shape == ref.shape == (6 * 1024, 2)
    assert np.array_equal(ours, ref), float(np.abs(ours - ref).max())
    assert float(ours[:, 0].max()) > 0.85 and abs(float(ours[:, 1].max() / ours[:, 0].max()) - 2 / 3) < 0.01  # L, R kept


@pytest.mark.parametrize("rate, layout, timed", [(44100, "stereo", True), (44100, "mono", True),
                                                 (22050, "mono", True), (44100, "stereo", False)])
def test_other_audio_is_resampled_in_batches_with_the_same_samples(rate, layout, timed):
    import audio_engine
    frames = _tone_frames(rate=rate, n_frames=60, layout=layout, timed=timed)
    conv = audio_engine.StereoConverter(48000)
    ours = np.concatenate([a for f in frames for a in conv.convert(f)] + conv.flush())
    r = av.AudioResampler(format="flt", layout="stereo", rate=48000)
    ref = np.concatenate([audio_engine.StereoConverter._array(rf) for f in frames for rf in r.resample(f)]
                         + [audio_engine.StereoConverter._array(rf) for rf in r.resample(None)])
    assert abs(len(ours) - len(ref)) <= 64, (len(ours), len(ref))
    n = min(len(ours), len(ref))
    assert np.allclose(ours[:n], ref[:n], atol=2e-3), float(np.abs(ours[:n] - ref[:n]).max())
    assert conv.resample_calls <= 60 * 1024 / (0.25 * rate) + 2          # batches, not one call per frame


def test_live_sources_convert_without_batches():
    import audio_engine
    frames = _tone_frames(rate=44100, n_frames=3)
    conv = audio_engine.StereoConverter(48000, batch_s=0)
    assert all(conv.convert(f) for f in frames[1:2] + frames[2:])  # sound comes out for every frame
