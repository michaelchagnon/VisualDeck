# SPDX-License-Identifier: MPL-2.0
"""Live video sources: network streams and capture devices, used like media files.

A live source is stored where a media path would be, as a URL:

  udp://0.0.0.0:5000          MPEG-TS over UDP (listen on port 5000; a
                              multicast address joins the group)
  rtp://... rtsp://... srt://... rtmp://... http(s)://... (HLS) tcp://...
  dshow://video=<name>        a capture device on Windows (DirectShow)
  v4l2:///dev/video0          a capture device on Linux
  avfoundation://<index>      a capture device on macOS

Everything is opened through FFmpeg (PyAV); which protocols and devices work
depends on the FFmpeg in the installed PyAV (`available()` tells).

Unlike a file, a live source never ends, is not re-started per cue and is
shared by every cue that uses it: one connection, so a fade between two cues
with the same camera works, and a UDP port is only opened once. When the
signal stops, the last frame stays on the output and the source reconnects
in the background (state "retrying"); Output Health shows it.

No UI here; the player is render_engine.LiveSourcePlayer.
"""
from __future__ import annotations

import re
import time

NETWORK_SCHEMES = ("udp", "rtp", "rtsp", "rtsps", "srt", "rtmp", "rtmps", "http", "https", "tcp")
DEVICE_FORMATS = {"dshow": "dshow", "v4l2": "v4l2", "avfoundation": "avfoundation"}
SCHEMES = NETWORK_SCHEMES + tuple(DEVICE_FORMATS)
OPEN_TIMEOUT = 5.0           # seconds to connect / get the first packets
READ_TIMEOUT = 3.0           # seconds without data before reconnecting
MAX_LEN = 500

_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)://")

# Low latency: start decoding from little data and never queue frames.
_LOW_LATENCY = {"fflags": "nobuffer", "flags": "low_delay", "probesize": "500000", "analyzeduration": "1000000"}


def scheme(path):
    m = _SCHEME_RE.match(str(path or ""))
    return m.group(1).lower() if m else None


def is_live(path):
    """True if `path` is a live source URL rather than a file."""
    return scheme(path) in SCHEMES


def is_device(path):
    """True for a camera or capture card (dshow://, v4l2://, avfoundation://)."""
    return scheme(path) in DEVICE_FORMATS


def clean(text):
    """A live source URL from what the operator typed, or None."""
    t = str(text or "").strip()
    if not t or len(t) > MAX_LEN or any(ord(ch) < 32 for ch in t) or not is_live(t):
        return None
    rest = t.split("://", 1)[1]
    return t if rest.strip("/") else None


def parse(path):
    """(url, format or None, options) for av.open."""
    sch = scheme(path)
    rest = str(path).split("://", 1)[1]
    if sch in DEVICE_FORMATS:
        return rest, DEVICE_FORMATS[sch], {}
    opts = dict(_LOW_LATENCY)
    if sch == "udp" or sch == "rtp":
        opts.update({"overrun_nonfatal": "1", "fifo_size": "1000000"})
    elif sch in ("rtsp", "rtsps"):
        opts["rtsp_transport"] = "tcp"         # works through firewalls and NAT; cameras accept it
    return str(path), None, opts


def label(path):
    """Short name for lists and headers, e.g. "UDP 0.0.0.0:5000", "Camera video=BRIO"."""
    sch = scheme(path)
    rest = str(path).split("://", 1)[1] if sch else str(path)
    if sch in DEVICE_FORMATS:
        return f"Camera {rest.lstrip('/')}"
    rest = rest.split("?", 1)[0]
    if "@" in rest.split("/", 1)[0]:            # never show a password from user:pass@host
        rest = rest.split("@", 1)[1]
    return f"{sch.upper()} {rest}" if sch else rest


def available():
    """Which device formats the installed FFmpeg (PyAV) has: {"dshow": bool, ...}."""
    try:
        import av
        fmts = av.formats_available
    except Exception:  # noqa: BLE001
        fmts = set()
    return {name: name in fmts for name in DEVICE_FORMATS}


_LOG_PREFIX = re.compile(r"^\[[^\]]*@ [0-9a-fA-Fx]+\]\s?")
_DSHOW_NAME = re.compile(r'^"(?P<name>[^"]+)"\s*(?:\((?P<kind>video|audio|none)\))?$')
_AVF_ITEM = re.compile(r"^\[(?P<index>\d+)\]\s+(?P<name>.+)$")


def parse_device_list(text, fmt):
    """[(name or index, kind)] from FFmpeg's list_devices output for `fmt`
    ("dshow" or "avfoundation"). dshow writes either "name" (video) per line
    (FFmpeg 5 and later) or a "DirectShow video devices" heading with names
    under it (older); "Alternative name" lines are skipped. avfoundation
    writes headings and "[0] FaceTime HD Camera" lines."""
    out, section = [], None
    for raw in str(text or "").splitlines():
        line = _LOG_PREFIX.sub("", raw).strip()
        low = line.lower()
        if not line or low.startswith("alternative name"):
            continue
        if "video devices" in low:
            section = "video"
            continue
        if "audio devices" in low:
            section = "audio"
            continue
        if fmt == "dshow":
            m = _DSHOW_NAME.match(line)
            if m:
                out.append((m.group("name"), m.group("kind") or section))
        elif fmt == "avfoundation":
            m = _AVF_ITEM.match(line)
            if m and section:
                out.append((m.group("index"), section, m.group("name")))
    return out


def _ffmpeg_device_list(fmt, url):
    """FFmpeg's own device listing (list_devices) as text, through PyAV's log."""
    import av
    import av.logging as avlog
    old = avlog.get_level()
    avlog.set_level(avlog.INFO)
    try:
        with avlog.Capture() as logs:
            try:
                av.open(url, format=fmt, options={"list_devices": "true"}).close()
            except Exception:  # noqa: BLE001 - listing always ends with "cannot open dummy"
                pass
    finally:
        avlog.set_level(old)
    return "".join(str(m[2]) for m in logs)


def find_devices():
    """Capture devices this computer's FFmpeg can open: {"devices": [(url,
    description)], "note": text or None}. Blocks for a moment (FFmpeg asks
    the system): call it off the UI thread."""
    import glob
    import os
    have = available()
    found, notes = [], []
    if have.get("dshow"):
        try:
            for name, kind in parse_device_list(_ffmpeg_device_list("dshow", "dummy"), "dshow"):
                if kind in ("video", "none", None):
                    found.append((f"dshow://video={name}", name + (" (type not reported)" if kind != "video" else "")))
        except Exception as e:  # noqa: BLE001
            notes.append(f"DirectShow list failed: {e}")
    if have.get("avfoundation"):
        try:
            for index, kind, name in parse_device_list(_ffmpeg_device_list("avfoundation", ""), "avfoundation"):
                if kind == "video":
                    found.append((f"avfoundation://{index}", name))
        except Exception as e:  # noqa: BLE001
            notes.append(f"AVFoundation list failed: {e}")
    if have.get("v4l2"):
        for dev in sorted(glob.glob("/dev/video*")):
            name = os.path.basename(dev)
            try:
                with open(f"/sys/class/video4linux/{name}/name") as f:
                    name = f.read().strip() or name
            except OSError:
                pass
            found.append((f"v4l2://{dev}", f"{name} ({dev})"))
    if not any(have.values()):
        notes.append("The FFmpeg in this VisualDeck has no camera support (network streams work).")
    elif not found and not notes:
        notes.append("No capture devices found.")
    return {"devices": found, "note": "; ".join(notes) or None}


PREVIEW_SIZE = (320, 180)
PREVIEW_SETTLE_S = 0.5       # a camera's first frames are often dark while its exposure settles


def probe(path, timeout=OPEN_TIMEOUT):
    """Open the source and decode a frame. Returns {"ok": True, "width",
    "height", "fps", "codec", "seconds", "audio" (codec name or None),
    "preview" (a PIL image at most PREVIEW_SIZE, from the frame PREVIEW_SETTLE_S
    after the first, or None)} or {"ok": False, "error"}. Blocks up to about
    2 x timeout: call it off the UI thread."""
    import av
    t0 = time.monotonic()
    url, fmt, opts = parse(path)
    try:
        container = av.open(url, format=fmt, options=opts, timeout=(timeout, timeout))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"cannot open: {e}"[:300]}
    try:
        if not container.streams.video:
            return {"ok": False, "error": "no video in this source"}
        stream = container.streams.video[0]
        audio = container.streams.audio[0].codec_context.name if container.streams.audio else None
        first, last, t_first = None, None, None
        for frame in container.decode(stream):
            if first is None:
                first, t_first = frame, time.monotonic()
            last = frame
            if time.monotonic() - t_first >= PREVIEW_SETTLE_S:
                break
        if first is None:
            return {"ok": False, "error": "the source ended before a frame arrived"}
        preview = None
        try:
            from PIL import Image
            from render_engine import to_rgb_array        # the source's own color matrix
            preview = Image.fromarray(to_rgb_array(last, 'rgb24'))
            preview.thumbnail(PREVIEW_SIZE)
        except Exception:  # noqa: BLE001 - the test result stands without a picture
            preview = None
        return {"ok": True, "width": first.width, "height": first.height,
                "fps": round(float(stream.average_rate), 2) if stream.average_rate else None,
                "codec": stream.codec_context.name, "seconds": round(t_first - t0, 2), "audio": audio,
                "preview": preview}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    finally:
        try:
            container.close()
        except Exception:  # noqa: BLE001
            pass
