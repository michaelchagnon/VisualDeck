# SPDX-License-Identifier: MPL-2.0
"""VisualDeck --self-test [report.json]: checks that a build has what it needs.

Run by tools/windows/build_windows.ps1 on the program it has just built (the
Windows build runs without Python or the development launcher), and usable on
any installation. It opens no show and no output window and changes nothing
outside a temporary folder: it imports every part of VisualDeck, decodes a
clip it writes itself, finds the built-in effects, the icon, the sound and
MIDI systems, starts Tk and SDL, and opens a hidden OpenGL 3.3 window to
compile the output's shaders. Each check is "ok" or the error; the exit code
is 0 only if every check passed.

No UI here.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import time
import traceback

MODULES = (
    "audio_engine", "control_interface", "decode_test", "displays", "dmx_remote", "edge_blend", "editor",
    "effect_ui", "health", "help_content", "isf", "layer_mask", "layer_warp", "live_source", "logging_setup",
    "ltc", "media_manager", "media_relink", "mesh_warp", "midi_remote", "output_presets", "output_warp",
    "playback_copy", "preshow", "preview_compose", "project_io", "project_schema", "remote_control",
    "render_engine", "ring_buffer", "scheduler", "sdl_audio_safe", "shortcuts", "show_collect", "soak", "span",
    "test_patterns", "timecode", "ui_theme", "undo", "web_remote", "benchmark", "keep_awake", "ffmpeg_license",
)


def _check(results, name, fn):
    t0 = time.perf_counter()
    try:
        detail = fn()
        results[name] = {"ok": True, "detail": detail, "ms": round((time.perf_counter() - t0) * 1000)}
    except Exception as e:  # noqa: BLE001 - each check reports its own failure
        results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300],
                         "trace": traceback.format_exc()[-1500:]}


def _imports():
    import importlib
    for m in MODULES:
        importlib.import_module(m)
    return f"{len(MODULES)} modules"


def _decode():
    import av
    import numpy as np
    path = os.path.join(tempfile.mkdtemp(prefix="vd-selftest-"), "clip.mp4")
    with av.open(path, "w") as c:
        s = c.add_stream("mpeg4", rate=25)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(5):
            img = np.full((48, 64, 3), (i * 40) % 255, np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    with av.open(path) as c:
        n = sum(1 for _ in c.decode(video=0))
    if n < 5:
        raise RuntimeError(f"decoded {n} of 5 frames")
    from render_engine import video_has_alpha
    video_has_alpha(path)
    return f"PyAV {av.__version__}: wrote and decoded {n} frames"


def _ffmpeg_license():
    """Never fails: says whether this build's FFmpeg is LGPL or GPL (release notes need it)."""
    import ffmpeg_license
    return ffmpeg_license.summary(ffmpeg_license.report())


def _opencv():
    import cv2
    import numpy as np
    out = cv2.resize(np.zeros((8, 8, 4), np.uint8), (4, 4))
    return f"OpenCV {cv2.__version__}, {out.shape}"


def _effects():
    import isf
    names = isf.available()
    if len(names) < 9:
        raise RuntimeError(f"only {len(names)} built-in effects in {isf.BUILTIN_DIR}")
    for n in names:
        isf.fragment_source(isf.load(n))
    return f"{len(names)} effects parsed from {isf.BUILTIN_DIR}"


def _assets():
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    found = [f for f in ("visualdeck.ico", "splash.png") if os.path.exists(os.path.join(base, "src", f))]
    if len(found) < 2:
        raise RuntimeError(f"found only {found} in {os.path.join(base, 'src')}")
    return ", ".join(found)


def _tk():
    import tkinter as tk
    r = tk.Tk()
    r.withdraw()
    v = r.tk.call("info", "patchlevel")
    r.destroy()
    return f"Tk {v}"


def _sound():
    import pygame
    import sdl_audio_safe
    import audio_engine
    sdl_audio_safe.init_pygame()
    names, err = audio_engine.device_names()
    if err:
        raise RuntimeError(err)
    return f"pygame {pygame.version.ver}, SDL {'.'.join(map(str, pygame.get_sdl_version()))}, " \
           f"{len(names)} sound output(s), mixer closed: {pygame.mixer.get_init() is None}"


def _midi_osc():
    import rtmidi
    import pythonosc  # noqa: F401
    ports = rtmidi.MidiIn().get_ports()
    return f"python-rtmidi {getattr(rtmidi, '__version__', '?')}, {len(ports)} MIDI input(s); python-osc ok"


def _opengl():
    """A hidden 3.3 core window and the output's layer shader compiled."""
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "-2000,-2000")
    import pygame
    from OpenGL.GL import GL_RENDERER, GL_VERSION, glGetString
    import render_engine as rem
    pygame.display.init()
    try:
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
        from OpenGL.GL import GL_FRAGMENT_SHADER, GL_VERTEX_SHADER, shaders
        shaders.compileProgram(shaders.compileShader(rem.VERTEX_SHADER, GL_VERTEX_SHADER),
                               shaders.compileShader(rem.FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
        return f"{glGetString(GL_RENDERER).decode(errors='replace')} | {glGetString(GL_VERSION).decode(errors='replace')}"
    finally:
        pygame.display.quit()


def _awake():
    """Windows: the request to stay awake is taken and reported back, then ended."""
    import keep_awake
    if os.name != "nt":
        return "not used on this system"
    if not keep_awake.hold():
        raise RuntimeError("Windows did not take the request to stay awake")
    try:
        state = keep_awake.system_state()
        if state is None:
            return "request taken; Windows did not report its state"
        if not {"system", "display"} <= state:
            raise RuntimeError(f"Windows reports only {sorted(state)} after the request")
        return f"Windows reports requests for {', '.join(sorted(state))}"
    finally:
        keep_awake.release()


CHECKS = (("imports", _imports), ("decode", _decode), ("opencv", _opencv), ("effects", _effects),
          ("assets", _assets), ("tk", _tk), ("sound", _sound), ("midi_osc", _midi_osc), ("opengl", _opengl),
          ("awake", _awake), ("ffmpeg", _ffmpeg_license))


def run(report_path=None):
    """Run every check; write the report (JSON) if a path is given. Returns the exit code."""
    results = {}
    for name, fn in CHECKS:
        _check(results, name, fn)
    try:
        from control_interface import SPLASH_VERSION_TEXT as ver
    except Exception:  # noqa: BLE001
        ver = "?"
    report = {"visualdeck": ver, "frozen": bool(getattr(sys, "frozen", False)), "python": platform.python_version(),
              "platform": platform.platform(), "executable": sys.executable,
              "passed": all(r["ok"] for r in results.values()), "checks": results}
    text = json.dumps(report, indent=2)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(text)
    if sys.stdout is not None:
        try:
            print(text)
        except Exception:  # noqa: BLE001 - a windowed build may have no usable console
            pass
    return 0 if report["passed"] else 1
