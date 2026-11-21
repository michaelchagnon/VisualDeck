# SPDX-License-Identifier: MPL-2.0
"""Playback copies: a video re-encoded so it plays easily in a show.

Camera and download files are often long-GOP H.264 or H.265 (a key frame
every few seconds) at a size larger than the output. Every way of starting
such a video somewhere other than its start (Start Cue From, in points,
reverse, ping-pong) decodes from the key frame before that point, and a 4K
file costs a laptop's processor most of its time. A playback copy has a key
frame every half second (or every frame) and can be made smaller, so those
starts are quick and decoding is lighter.

The copy is a new file next to the original ("<name> (playback).mp4"); the
original is never changed. Made through PyAV (FFmpeg) with the first H.264
encoder this FFmpeg has (libx264, then OpenH264), else MPEG-2; sound is
re-encoded to AAC. No UI here (the window is in control_interface.py).
"""
from __future__ import annotations

import os
from fractions import Fraction

SIZES = (("Same size", None), ("1920 × 1080 at most", (1920, 1080)), ("1280 × 720 at most", (1280, 720)))
KEYFRAMES = (("Every half second", 0.5), ("Every frame (largest file, quickest anywhere)", 0.0))
CRF = 18                    # visually lossless for most footage
PRESET = "veryfast"         # several times quicker than "medium" at a slightly larger file
AUDIO_BITRATE = 192_000


def encoder():
    """(codec name, kind) of the video encoder to use, or (None, None)."""
    import av
    for name, kind in (("libx264", "x264"), ("libopenh264", "openh264"), ("mpeg2video", "mpeg2")):
        try:
            av.codec.Codec(name, "w")
            return name, kind
        except Exception:  # noqa: BLE001
            continue
    return None, None


def fit(size, box):
    """(w, h) of `size` scaled down (never up) to fit `box`, even numbers."""
    w, h = size
    if box and (w > box[0] or h > box[1]):
        s = min(box[0] / w, box[1] / h)
        w, h = w * s, h * s
    return max(2, int(round(w / 2)) * 2), max(2, int(round(h / 2)) * 2)


def target_path(src, size=None):
    """A new file next to `src` that does not exist yet."""
    folder, name = os.path.split(src)
    stem = os.path.splitext(name)[0]
    tag = "playback" if not size else f"playback {size[1]}p"
    path = os.path.join(folder, f"{stem} ({tag}).mp4")
    n = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem} ({tag} {n}).mp4")
        n += 1
    return path


class Canceled(Exception):
    pass


def make(src, dst, max_size=None, keyframe_s=0.5, progress=None, canceled=None):
    """Write the playback copy of `src` to `dst` (through dst + ".part", so a
    half-written file is never left under the final name). progress(fraction)
    is called now and then; canceled() returning True stops it (Canceled).
    Returns {'size': (w, h), 'fps', 'gop' (frames), 'codec', 'audio' (bool),
    'frames'}."""
    import av
    from av.video.frame import PictureType
    name, kind = encoder()
    if name is None:
        raise RuntimeError("this FFmpeg has no video encoder for playback copies")
    part = dst + ".part"
    inp = av.open(src)
    out = None
    try:
        vin = inp.streams.video[0]
        ain = inp.streams.audio[0] if inp.streams.audio else None
        vin.thread_type = "AUTO"
        rate = vin.average_rate or vin.guessed_rate or Fraction(25)
        rate = Fraction(rate).limit_denominator(1001)
        w, h = fit((vin.codec_context.width, vin.codec_context.height), max_size)
        gop = 1 if keyframe_s <= 0 else max(1, int(round(float(rate) * keyframe_s)))
        duration = float(inp.duration / 1_000_000) if inp.duration else (
            float(vin.duration * vin.time_base) if vin.duration and vin.time_base else None)

        out = av.open(part, mode="w", format="mp4")
        vout = out.add_stream(name, rate=rate)
        vout.width, vout.height, vout.pix_fmt = w, h, "yuv420p"
        vout.codec_context.time_base = 1 / rate
        vout.codec_context.gop_size = gop
        if kind == "x264":
            vout.options = {"crf": str(CRF), "preset": PRESET, "keyint_min": str(gop), "sc_threshold": "0",
                            "bf": "0" if gop == 1 else "2"}
        else:
            vout.codec_context.bit_rate = int(w * h * float(rate) * 0.25)   # about 25 Mb/s at 1080p25
        for attr in ("color_range", "colorspace", "color_primaries", "color_trc"):
            try:                                     # keep the source's color description where PyAV can
                setattr(vout.codec_context, attr, getattr(vin.codec_context, attr))
            except Exception:  # noqa: BLE001
                pass
        aout = None
        if ain is not None:
            aout = out.add_stream("aac", rate=ain.codec_context.sample_rate or 48000)
            aout.codec_context.bit_rate = AUDIO_BITRATE
            try:
                aout.layout = ain.codec_context.layout.name if ain.codec_context.layout else "stereo"
            except Exception:  # noqa: BLE001
                pass

        frames = 0
        streams = [vin] + ([ain] if ain is not None else [])
        for packet in inp.demux(*streams):
            if canceled is not None and canceled():
                raise Canceled()
            for frame in packet.decode():
                if packet.stream is vin:
                    t = frame.time
                    frame = frame.reformat(width=w, height=h, format="yuv420p")
                    frame.pts, frame.time_base = frames, 1 / rate
                    # A decoded frame keeps its picture type, and the encoder makes every
                    # frame that was a key frame in the source one again: let it choose.
                    frame.pict_type = PictureType.NONE
                    for p in vout.encode(frame):
                        out.mux(p)
                    frames += 1
                    if progress is not None and duration and t is not None and frames % 10 == 0:
                        progress(min(1.0, max(0.0, t / duration)))
                elif aout is not None:
                    frame.pts = None
                    for p in aout.encode(frame):
                        out.mux(p)
        for p in vout.encode():
            out.mux(p)
        if aout is not None:
            for p in aout.encode():
                out.mux(p)
        out.close()
        out = None
        inp.close()
        os.replace(part, dst)
        if progress is not None:
            progress(1.0)
        return {"size": (w, h), "fps": float(rate), "gop": gop, "codec": name, "audio": aout is not None,
                "frames": frames}
    except BaseException:
        for c in (out, inp):
            try:
                if c is not None:
                    c.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            os.remove(part)
        except OSError:
            pass
        raise
