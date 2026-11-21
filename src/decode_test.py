"""Decoding test: how fast can this computer decode a given video, in software
# SPDX-License-Identifier: MPL-2.0
and with each hardware decoder FFmpeg offers here?

Preferences > GPU Hardware Acceleration > Test with a Video... runs it on a
background thread. Each try uses the player's own code path (open with the
requested hardware decoder and software fallback, the same threading, the
same frame conversion for the output), decodes for a few seconds as fast as
it can, and reports frames per second and whether the hardware decoder was
really used. Nothing is shown on the output. No UI here.
"""
from __future__ import annotations

import os
import time

CANDIDATES = ("d3d11va", "dxva2", "cuda", "qsv", "videotoolbox", "vaapi")


def available_devices():
    """Hardware decoder types in the installed FFmpeg that VisualDeck can request."""
    try:
        from av.codec.hwaccel import hwdevices_available
        have = set(hwdevices_available())
    except Exception:  # noqa: BLE001
        have = set()
    return [d for d in CANDIDATES if d in have]


def run_one(path, device=None, seconds=3.0, max_frames=300, output_format="yuv", multithread=True):
    """{'device', 'ok', 'fps', 'frames', 'engaged', 'pix_fmt', 'size', 'codec', 'error'}."""
    from render_engine import GPUVideoPlayerThread
    from ring_buffer import FastRingBuffer
    p = GPUVideoPlayerThread(path, FastRingBuffer(2), hw_accel=device, loop=False, output_format=output_format,
                             force_multithread=multithread)
    res = {"device": device or "software", "ok": False, "fps": None, "frames": 0, "engaged": None,
           "pix_fmt": None, "size": None, "codec": None, "error": None}
    container = p._open_container()
    if container is None:
        res["error"] = p.last_error or "cannot open"
        return res
    try:
        if not container.streams.video:
            res["error"] = "no video stream"
            return res
        stream = container.streams.video[0]
        p._set_threading(stream)
        info = getattr(p, "media_info", {}) or {}
        res.update(codec=info.get("codec"), size=info.get("size"))
        t0 = time.perf_counter()
        for frame in container.decode(stream):
            if res["frames"] == 0:
                res["engaged"] = bool(getattr(stream.codec_context, "is_hwaccel", False))
                res["pix_fmt"] = frame.format.name
                t0 = time.perf_counter()          # the first frame includes opening: not counted
            p._convert_frame(frame)
            res["frames"] += 1
            if res["frames"] >= max_frames or time.perf_counter() - t0 >= seconds:
                break
        dt = time.perf_counter() - t0
        if res["frames"] > 1 and dt > 0:
            res["fps"] = round((res["frames"] - 1) / dt, 1)
        res["ok"] = res["frames"] > 0
        if device and not res["engaged"]:
            res["error"] = "hardware decoder not used (FFmpeg fell back to software)"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"[:200]
    finally:
        try:
            container.close()
        except Exception:  # noqa: BLE001
            pass
    return res


def run_all(path, output_format="yuv", seconds=3.0, progress=None):
    """Software, then each available hardware decoder. progress(result) after each."""
    out = []
    for dev in [None] + available_devices():
        r = run_one(path, dev, seconds=seconds, output_format=output_format)
        out.append(r)
        if progress:
            progress(r)
    return out


def summary(results, source_fps=None):
    """One line per try, and the recommendation (the fastest that was really used)."""
    lines = []
    for r in results:
        if r["ok"] and r["fps"]:
            note = "" if r["device"] == "software" or r["engaged"] else "  (not used)"
            lines.append(f"{r['device']:<12} {r['fps']:>6.1f} frames/s{note}")
        else:
            lines.append(f"{r['device']:<12} {'—':>6}           {r['error'] or 'failed'}")
    good = [r for r in results if r["ok"] and r["fps"] and (r["device"] == "software" or r["engaged"])]
    best = max(good, key=lambda r: r["fps"]) if good else None
    rec = None
    if best is not None:
        rec = best["device"]
        if source_fps and best["fps"] < source_fps:
            lines.append(f"Even the fastest ({best['fps']:.0f}) is below the clip's {source_fps:g} fps: "
                         "use a lighter file (lower resolution, H.264) or a proxy.")
    return lines, rec


def clip_fps(path):
    try:
        import av
        with av.open(path) as c:
            s = c.streams.video[0]
            return round(float(s.average_rate), 3) if s.average_rate else None
    except Exception:  # noqa: BLE001
        return None


def label(path):
    return os.path.basename(path)
