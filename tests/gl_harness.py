# SPDX-License-Identifier: MPL-2.0
"""Run one OpenGL render scenario in an isolated process and print JSON results.

Used by tests/test_render_gl.py and bench scripts. Requires an X display
(Xvfb is fine; Mesa llvmpipe provides OpenGL 4.5 core in software).

The harness captures the BACK buffer immediately before each buffer swap, so
results reflect exactly what the engine was about to present.

Usage: python tests/gl_harness.py <scenario> <fixtures.json>
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402
from PIL import Image  # noqa: E402

# render_engine must be imported before OpenGL.GL: it sets PyOpenGL's flags,
# which only take effect if set before the first OpenGL.GL import.
import render_engine as rem  # noqa: E402
from OpenGL.GL import (glFinish, glReadBuffer, glReadPixels, GL_BACK, GL_RGBA,  # noqa: E402
                       GL_UNSIGNED_BYTE, glGetString, GL_RENDERER, GL_VERSION)

FORCE_INSTANCED = os.environ.get("VD_FORCE_INSTANCED") == "1"
if FORCE_INSTANCED and hasattr(rem, "INSTANCED_FRAGMENT_SHADER"):
    # Mesa rejects the original GLSL 3.30 shader (non-constant sampler-array
    # index). Compiling it as GLSL 4.00 lets the original instanced code path
    # run on Mesa so its draw-order behavior can be observed.
    rem.INSTANCED_VERTEX_SHADER = rem.INSTANCED_VERTEX_SHADER.replace("#version 330 core", "#version 400 core")
    rem.INSTANCED_FRAGMENT_SHADER = rem.INSTANCED_FRAGMENT_SHADER.replace("#version 330 core", "#version 400 core")

from media_manager import MediaManager  # noqa: E402

LAST = {"img": None, "swaps": 0, "centers": [], "center_only": False}
_orig_flip = pygame.display.flip


def _capturing_flip():
    surf = pygame.display.get_surface()
    w, h = surf.get_size()
    glFinish()
    glReadBuffer(GL_BACK)
    if LAST["center_only"]:
        c = glReadPixels(w // 2, h // 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE)
        LAST["centers"].append(list(bytes(c))[:3])
    else:
        px = glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE)
        LAST["img"] = Image.frombytes("RGBA", (w, h), px).transpose(Image.FLIP_TOP_BOTTOM)
    LAST["swaps"] += 1
    _orig_flip()


pygame.display.flip = _capturing_flip


def layer(path, order, scale=1.0, x=0.5, y=0.5, screens=(0,)):
    return {"path": path, "screens": list(screens), "order": order, "loop": True,
            "position": {"x": x, "y": y, "scale_x": scale, "scale_y": scale, "rotation": 0}}


def px(img, x, y):
    return list(img.getpixel((int(x), int(y))))[:3]


def make_engine(**kw):
    kw.setdefault("use_pbo", True)
    eng = rem.OpenGLRenderEngine(MediaManager(), **kw)
    eng._init_pygame_and_gl()
    return eng


def settle(eng, layers, n=25, dt=0.02):
    for _ in range(n):
        eng.render(layers, target_monitor=0, transition_state={"active": False})
        time.sleep(dt)
    return LAST["img"]


def scenario_colors(fx):
    eng = make_engine()
    out = {}
    mon = eng.monitors[0]
    for key in ("black_709", "white_709", "red_709", "red_601", "black_full"):
        # 640x360 clip scaled 3x covers a 1920x1080 monitor
        L = {0: layer(fx[key], 0, scale=mon.width / 640.0)}
        eng.update_layers(L)
        img = settle(eng, L)
        out[key] = px(img, img.width / 2, img.height / 2)
        eng.update_layers({})
    out["renderer"] = (glGetString(GL_RENDERER) or b"?").decode()
    out["gl"] = (glGetString(GL_VERSION) or b"?").decode()
    eng.clear()
    return out


def scenario_shear(fx):
    eng = make_engine()
    L = {0: layer(fx["split_854"], 0, scale=1.0)}
    eng.update_layers(L)
    img = settle(eng, L)
    w, h = img.size
    # clip is 854x480 centered on screen; find the white/black boundary column on
    # the clip's first and last rows. Without shear both rows agree.
    x0, y0 = (w - 854) // 2, (h - 480) // 2

    def boundary(row):
        for x in range(x0, x0 + 854):
            if px(img, x, row)[0] < 128:
                return x - x0
        return None

    res = {"top_boundary": boundary(y0 + 2), "bottom_boundary": boundary(y0 + 477)}
    eng.clear()
    return res


def scenario_layer_order(fx):
    eng = make_engine()  # baseline default is use_instancing=True
    mon = eng.monitors[0]
    L = {0: layer(fx["black_709"], 0, scale=mon.width / 640.0),
         1: layer(fx["red_png"], 1),
         2: layer(fx["green_png"], 2, x=0.25)}
    eng.update_layers(L)
    img = settle(eng, L)
    res = {"instanced_active": bool(getattr(eng, "instanced_shader", None)),
           "center": px(img, img.width / 2, img.height / 2),
           "left_quarter": px(img, img.width / 4, img.height / 2)}
    eng.clear()
    return res


def _cut_through_render_thread(fx, first_key, second_key):
    """Drive the real RenderThread with the commands the UI sends for a cut and
    record the center pixel of every presented frame."""
    import queue as _q
    import threading as _t
    import control_interface as ci_mod
    eng = rem.OpenGLRenderEngine(MediaManager(), use_pbo=True)
    cmd_q, prev_q, stop = _q.Queue(), _q.Queue(maxsize=1), _t.Event()
    rt = ci_mod.RenderThread(eng, cmd_q, prev_q, stop)
    LAST["center_only"] = True
    rt.start()

    activation = [0]

    def cut(key):
        # Same protocol as ControlInterface._cut_to_column: a new activation
        # (instance key) per cut, no forced restarts.
        path = fx[key]
        activation[0] += 1
        scale = (eng.monitors[0].width / 640.0) if eng.monitors else 3.0
        lay = dict(layer(path, 0, scale=scale), instance=f"@{activation[0]}:0:0|{path}")
        cmd_q.put({"action": "render", "preroll": True, "layers": {0: lay},
                   "target_monitor": 0, "transition_state": {"active": False},
                   "gui_monitor": 0, "output_mode": "Display 0",
                   "all_assigned_screens": set()})

    cut(first_key)
    time.sleep(1.2)
    start = len(LAST["centers"])
    cut(second_key)
    time.sleep(1.2)
    players = sorted(eng.active_players)  # before shutdown, which clears all players
    stop.set()
    rt.join(3)
    frames = LAST["centers"][start:]
    black = [c for c in frames if max(c) < 30]
    return {"frames": len(frames), "black_frames": len(black), "first": frames[:1], "last": frames[-1:],
            "errors": dict(getattr(rt, "error_counts", {})),
            "players_left": players}


def scenario_cut_different_files(fx):
    return _cut_through_render_thread(fx, "white_709", "red_709")


def scenario_cut_same_file(fx):
    return _cut_through_render_thread(fx, "white_709", "white_709")


def scenario_cut_gap(fx):
    """Reproduce the UI's hard-cut sequence and report the frame presented in between."""
    eng = make_engine()
    mon = eng.monitors[0]
    A = {0: layer(fx["white_709"], 0, scale=mon.width / 640.0)}
    eng.update_layers(A)
    before = settle(eng, A)
    # _initiate_transition (transition 'None') sends update_layers({}) first,
    # and the render thread renders its current (old) layers before the new cue arrives.
    eng.update_layers({})
    eng.render(A, target_monitor=0, transition_state={"active": False})
    gap = LAST["img"]
    res = {"before": px(before, 50, 50), "gap": px(gap, 50, 50)}
    eng.clear()
    return res


def scenario_invalid_screen(fx):
    """Layers routed to a display index that does not exist must not break output."""
    eng = make_engine()
    L = {0: layer(fx["red_png"], 0, screens=(7,)), 1: layer(fx["green_png"], 1, screens=(0,))}
    out = {"errors": []}
    for _ in range(5):
        try:
            eng.render(L, target_monitor=None, transition_state={"active": False}, all_assigned_screens={0, 7})
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"{type(e).__name__}: {e}")
            break
    img = LAST["img"]
    out["center"] = px(img, img.width / 2, img.height / 2) if img is not None else None
    eng.clear()
    return out


def scenario_alpha_video(fx):
    """Video with alpha (left half opaque red, right half transparent) over a green image."""
    eng = make_engine()
    mon = eng.monitors[0]
    L = {0: layer(fx["green_png"], 0, scale=mon.width / 200.0),
         1: layer(fx["alpha_video"], 1, scale=mon.width / 320.0)}
    eng.update_layers(L)
    img = settle(eng, L)
    res = {"left": px(img, img.width * 0.25, img.height / 2), "right": px(img, img.width * 0.75, img.height / 2)}
    eng.clear()
    return res


def scenario_preview_capture(fx):
    """GPU preview capture returns the output frame of the requested monitor."""
    eng = make_engine()
    mon = eng.monitors[0]
    L = {0: layer(fx["red_709"], 0, scale=mon.width / 640.0), 1: layer(fx["green_png"], 1)}
    eng.update_layers(L)
    settle(eng, L)
    eng.request_preview_capture(0)
    eng.render(L, target_monitor=0, transition_state={"active": False})
    img = eng.take_preview_image()
    res = {"size": list(img.size) if img else None,
           "center": px(img, img.width / 2, img.height / 2) if img else None,
           "corner": px(img, 5, 5) if img else None}
    eng.clear()
    return res


def scenario_instances_same_file(fx):
    """Same file on two layers as two instances: left one playing, right one
    held on frame 0. Per-instance textures must show different frames."""
    eng = make_engine()
    path = fx["counting"]  # 320x180, frame i has luma 16 + 3*i
    # loop=False: the 2 s clip must not wrap back to its black first frames
    left = dict(layer(path, 0, scale=2.0, x=0.25), instance=f"@1:0:0|{path}", loop=False)
    right = dict(layer(path, 1, scale=2.0, x=0.75), instance=f"@2:1:0|{path}", hold=True)
    L = {0: left, 1: right}
    eng.update_layers(L)
    img = settle(eng, L, n=50, dt=0.02)  # ~1 s: left reaches roughly frame 20-30
    res = {"left": px(img, img.width * 0.25, img.height / 2),
           "right": px(img, img.width * 0.75, img.height / 2),
           "players": len(eng.active_players),
           "textures": len(eng.yuv_textures)}
    eng.clear()
    return res


def scenario_opacity(fx):
    """Per-layer opacity: red at 50% over full-screen green; then the same red
    layer fading in (progress 0.5) over black, where the fade multiplies it."""
    eng = make_engine()
    mon = eng.monitors[0]
    green = layer(fx["green_png"], 0, scale=mon.width / 200.0)
    red = dict(layer(fx["red_png"], 1, scale=mon.width / 200.0), opacity=0.5)
    L = {0: green, 1: red}
    eng.update_layers(L)
    img = settle(eng, L, n=5)
    res = {"half_over_green": px(img, img.width / 2, img.height / 2)}
    alone = {1: red}                                   # still-image cue: goes through the static FBO cache
    img = settle(eng, alone, n=5)
    res["half_alone"] = px(img, img.width / 2, img.height / 2)
    import tempfile
    semi = os.path.join(tempfile.mkdtemp(), "semi.png")         # a PNG whose pixels are 50% transparent
    Image.new("RGBA", (200, 150), (255, 0, 0, 128)).save(semi)
    S = {0: layer(semi, 0, scale=mon.width / 200.0)}
    eng.update_layers(S)
    img = settle(eng, S, n=5)
    res["semi_png_static"] = px(img, img.width / 2, img.height / 2)
    faint = os.path.join(os.path.dirname(semi), "faint.png")     # 20% alpha: dropped if alpha were applied twice
    Image.new("RGBA", (200, 150), (255, 0, 0, 51)).save(faint)
    F = {0: layer(faint, 0, scale=mon.width / 200.0)}
    eng.update_layers(F)
    img = settle(eng, F, n=5)
    res["faint_png_static"] = px(img, img.width / 2, img.height / 2)
    res["static_cache_used"] = eng.static_cue_cache_misses >= 2
    ts = {"active": True, "progress": 0.5, "from": {}, "to": {1: red}}
    for _ in range(3):
        eng.render({}, target_monitor=0, transition_state=ts)
    res["half_in_half_fade"] = px(LAST["img"], LAST["img"].width / 2, LAST["img"].height / 2)
    eng.clear()
    return res


def scenario_blend_modes(fx):
    """Full-screen top image (200,100,50) over full-screen base (100,150,200) per blend mode."""
    import tempfile
    d = tempfile.mkdtemp()
    base_p, top_p = os.path.join(d, "base.png"), os.path.join(d, "top.png")
    Image.new("RGBA", (200, 150), (100, 150, 200, 255)).save(base_p)
    Image.new("RGBA", (200, 150), (200, 100, 50, 255)).save(top_p)
    eng = make_engine()
    mon = eng.monitors[0]
    sc = mon.width / 200.0 * 1.1
    res = {}
    for mode, opacity in (("normal", 1.0), ("add", 1.0), ("multiply", 1.0), ("screen", 1.0),
                          ("multiply", 0.5), ("screen", 0.5), ("add", 0.5)):
        L = {0: layer(base_p, 0, scale=sc), 1: dict(layer(top_p, 1, scale=sc), blend=mode, opacity=opacity)}
        eng.update_layers(L)
        img = settle(eng, L, n=4)
        res[f"{mode}@{opacity}"] = px(img, img.width / 2, img.height / 2)
    eng.clear()
    return res


def scenario_color_adjust(fx):
    """Color correction: a full-screen (200,100,50) image with several
    adjustments; a half-transparent PNG (alpha must be kept); a red video
    (YUV path) desaturated to gray; static cues must not use the cache."""
    import tempfile
    d = tempfile.mkdtemp()
    img_p, semi_p = os.path.join(d, "c.png"), os.path.join(d, "semi.png")
    Image.new("RGBA", (200, 150), (200, 100, 50, 255)).save(img_p)
    Image.new("RGBA", (200, 150), (200, 100, 50, 128)).save(semi_p)
    eng = make_engine()
    mon = eng.monitors[0]
    sc = mon.width / 200.0 * 1.1
    res = {}
    for name, adj in (("none", {}), ("bright", {"brightness": 0.2}), ("dark_contrast", {"brightness": -0.1, "contrast": 1.5}),
                      ("gray", {"saturation": 0.0}), ("vivid", {"saturation": 1.8}), ("all", {"brightness": 0.1, "contrast": 0.6, "saturation": 1.4})):
        L = {0: dict(layer(img_p, 0, scale=sc), **adj)}
        eng.update_layers(L)
        img = settle(eng, L, n=4)
        res[name] = px(img, img.width / 2, img.height / 2)
    misses = eng.static_cue_cache_misses
    L = {0: dict(layer(semi_p, 0, scale=sc), brightness=0.2)}
    eng.update_layers(L)
    img = settle(eng, L, n=4)
    res["semi_bright"] = px(img, img.width / 2, img.height / 2)
    res["static_cache_bypassed"] = eng.static_cue_cache_misses == misses
    V = {0: dict(layer(fx["red_709"], 0, scale=mon.width / 640.0), saturation=0.0)}
    eng.update_layers(V)
    img = settle(eng, V)
    res["video_gray"] = px(img, img.width / 2, img.height / 2)
    eng.update_layers({})
    eng.clear()
    return res


def scenario_rotation(fx):
    """Where the top-right corner of a 2:1 image ends up when rotated by +90:
    counter-clockwise rotation puts it at the top-left of the rotated image."""
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "mark.png")
    im = Image.new("RGBA", (200, 100), (0, 0, 255, 255))
    im.paste((255, 0, 0, 255), (180, 0, 200, 20))
    im.save(p)
    eng = make_engine()
    mon = eng.monitors[0]
    sc = mon.height / 200.0 * 0.9                      # rotated, the 200 px side spans 90 % of the height
    L = {0: dict(layer(p, 0, scale=sc), position={"x": 0.5, "y": 0.5, "scale_x": sc, "scale_y": sc, "rotation": 90})}
    eng.update_layers(L)
    img = settle(eng, L, n=4).convert("RGB")
    import numpy as np
    a = np.asarray(img)
    ys, xs = np.nonzero((a[..., 0] > 200) & (a[..., 2] < 60))
    eng.clear()
    return {"red_rel": [float(xs.mean() - img.width / 2), float(ys.mean() - img.height / 2)]}


WARP = ((0.1, 0.1), (0.9, 0.05), (1.0, 1.0), (0.0, 0.9))


def scenario_corner_pin(fx):
    """A picture with a red top-left quadrant and blue elsewhere, pinned to
    WARP; sample where output_warp says picture points land, and outside."""
    import tempfile
    import output_warp
    d = tempfile.mkdtemp()
    p = os.path.join(d, "quad.png")
    im = Image.new("RGBA", (200, 100), (0, 0, 255, 255))
    im.paste((255, 0, 0, 255), (0, 0, 100, 50))
    im.save(p)
    eng = make_engine()
    mon = eng.monitors[0]
    L = {0: dict(layer(p, 0), position={"x": 0.5, "y": 0.5, "scale_x": mon.width / 200.0,
                                         "scale_y": mon.height / 100.0, "rotation": 0})}
    res = {}
    for name, warps in (("plain", {}), ("pinned", {0: WARP})):
        eng.set_output_warps(warps)
        eng.update_layers(L)
        img = settle(eng, L, n=4).convert("RGB")
        h = output_warp.homography(WARP)
        pts = {"tl_quadrant": (0.25, 0.25), "br_quadrant": (0.75, 0.75)}
        out = {}
        for k, (u, v) in pts.items():
            x, y = output_warp.apply(h, u, v) if warps else (u, v)
            out[k] = px(img, x * img.width, y * img.height)
        out["outside_top_left"] = px(img, 5, 5)
        out["outside_bottom_left"] = px(img, 5, img.height - 5)
        res[name] = out
    V = {0: dict(layer(fx["red_709"], 0, scale=mon.width / 640.0))}
    eng.set_output_warps({0: WARP})
    eng.update_layers(V)
    img = settle(eng, V).convert("RGB")
    res["video_center"] = px(img, img.width / 2, img.height / 2)
    res["video_outside"] = px(img, 5, 5)
    res["targets"] = len(eng._warp_targets)
    eng.set_output_warps({})
    res["targets_after_reset"] = len(eng._warp_targets)
    eng.update_layers({})
    eng.clear()
    return res


def scenario_transition_curves(fx):
    """Full-screen red fading to full-screen green: centre pixel per curve and progress."""
    import tempfile
    d = tempfile.mkdtemp()
    red, green = os.path.join(d, "r.png"), os.path.join(d, "g.png")
    Image.new("RGBA", (200, 150), (255, 0, 0, 255)).save(red)
    Image.new("RGBA", (200, 150), (0, 255, 0, 255)).save(green)
    eng = make_engine()
    mon = eng.monitors[0]
    sc = mon.width / 200.0 * 1.1
    F, T = {0: layer(red, 0, scale=sc)}, {0: layer(green, 0, scale=sc)}
    eng.update_layers({0: F[0], 1: T[0]})
    res = {}
    for curve in ("linear", "smooth", "dip"):
        for p in (0.25, 0.5, 0.75):
            ts = {"active": True, "progress": p, "curve": curve, "from": F, "to": T}
            for _ in range(2):
                eng.render({}, target_monitor=0, transition_state=ts)
            img = LAST["img"]
            res[f"{curve}@{p}"] = px(img, img.width / 2, img.height / 2)
    eng.set_output_warps({0: WARP})                    # a crossfade on a corner-pinned display
    ts = {"active": True, "progress": 0.5, "curve": "linear", "from": F, "to": T}
    for _ in range(2):
        eng.render({}, target_monitor=0, transition_state=ts)
    img = LAST["img"]
    res["pinned_mix_center"] = px(img, img.width / 2, img.height / 2)
    res["pinned_mix_outside"] = px(img, 5, 5)
    eng.clear()
    return res


BLEND = {"left": 0.25, "right": 0.1, "top": 0.0, "bottom": 0.2, "gamma": 2.2, "curve": 2.0}


def scenario_edge_blend(fx):
    """Full-screen white with edge blends: pixel values across the left, right and
    bottom blends; then with a corner pin as well."""
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "white.png")
    Image.new("RGBA", (200, 150), (255, 255, 255, 255)).save(p)
    eng = make_engine()
    mon = eng.monitors[0]
    L = {0: layer(p, 0, scale=mon.width / 200.0 * 1.1)}
    eng.set_output_blends({0: BLEND})
    eng.update_layers(L)
    img = settle(eng, L, n=4)
    res = {"points": {}}
    for x, y in ((0.05, 0.5), (0.125, 0.5), (0.2, 0.5), (0.5, 0.5), (0.95, 0.5), (0.5, 0.9), (0.02, 0.95)):
        res["points"][f"{x},{y}"] = px(img, x * img.width, y * img.height)[0]
    eng.set_output_warps({0: WARP})
    img = settle(eng, L, n=4)
    res["pinned_outside"] = px(img, 5, 5)
    res["pinned_center"] = px(img, img.width / 2, img.height / 2)
    res["targets"] = len(eng._warp_targets)
    eng.set_output_warps({})
    eng.set_output_blends({})
    res["targets_after"] = len(eng._warp_targets)
    eng.clear()
    return res


MESH = {"cols": 3, "rows": 3, "points": [[0, 0], [0.5, 0], [1, 0],
                                         [0.1, 0.5], [0.65, 0.6], [1, 0.5],
                                         [0, 1], [0.5, 1], [1, 1]]}


def scenario_mesh_warp(fx):
    """A picture of four colored quadrants on a 3x3 mesh (center point moved,
    left edge pulled in), without and with a corner pin: colors where
    mesh_warp says the quadrant centers land, the moved center, the pulled-in
    edge, and the whole frame against the CPU path (mesh_warp.warp_image of
    the unwarped frame)."""
    import tempfile
    import numpy as np
    import mesh_warp
    import output_warp
    d = tempfile.mkdtemp()
    p = os.path.join(d, "quads.png")
    im = Image.new("RGBA", (200, 100), (255, 255, 255, 255))
    im.paste((255, 0, 0, 255), (0, 0, 100, 50))
    im.paste((0, 255, 0, 255), (100, 0, 200, 50))
    im.paste((0, 0, 255, 255), (0, 50, 100, 100))
    im.save(p)
    eng = make_engine()
    mon = eng.monitors[0]
    L = {0: dict(layer(p, 0), position={"x": 0.5, "y": 0.5, "scale_x": mon.width / 200.0,
                                         "scale_y": mon.height / 100.0, "rotation": 0})}
    eng.update_layers(L)
    plain = np.asarray(settle(eng, L, n=4).convert("RGBA")).copy()
    mesh = mesh_warp.clean(MESH)
    res = {"plain_moved_center": px(Image.fromarray(plain), 0.6 * mon.width, 0.55 * mon.height)}
    for name, pin in (("mesh", None), ("mesh_pin", WARP)):
        eng.set_output_warps({0: pin} if pin else {})
        eng.set_output_meshes({0: MESH})
        img = settle(eng, L, n=4).convert("RGBA")
        out = {}
        h = output_warp.homography(pin) if pin else None
        for k, (u, v) in {"tl": (0.25, 0.25), "tr": (0.75, 0.25), "bl": (0.25, 0.75), "br": (0.75, 0.75)}.items():
            x, y = mesh_warp.evaluate(mesh, u, v)
            if h is not None:
                x, y = output_warp.apply(h, x, y)
            out[k] = px(img, x * img.width, y * img.height)
        x, y = 0.6, 0.55
        if h is not None:
            x, y = output_warp.apply(h, x, y)
        out["moved_center"] = px(img, x * img.width, y * img.height)
        x, y = (0.03, 0.5) if h is None else output_warp.apply(h, 0.03, 0.5)
        out["pulled_edge"] = px(img, x * img.width, y * img.height)
        gl = np.asarray(img).astype(np.int16)
        cpu = mesh_warp.warp_image(plain, mesh, pin).astype(np.int16)
        diff = np.abs(gl[..., :3] - cpu[..., :3]).max(axis=2)
        out["mean_abs_diff"] = float(diff.mean())
        out["share_over_40"] = float((diff > 40).mean())
        out["geometry"] = len(eng._mesh_geo)
        res[name] = out
    eng.set_output_meshes({})
    eng.set_output_warps({})
    settle(eng, L, n=2)
    res["geometry_after"] = len(eng._mesh_geo)
    res["targets_after"] = len(eng._warp_targets)
    eng.update_layers({})
    eng.clear()
    return res


def _span_setup():
    """Two 960x1080 displays side by side on the 1920x1080 screen, spanned with
    a 160 px overlap: a 1760x1080 canvas. Picture: canvas x encoded as
    R = x % 256, G = x // 256."""
    import tempfile
    import numpy as np
    from types import SimpleNamespace
    import span
    mons = [SimpleNamespace(x=0, y=0, width=960, height=1080, name="A"),
            SimpleNamespace(x=960, y=0, width=960, height=1080, name="B")]
    sp = span.clean({"displays": [0, 1], "overlap": 160})
    (cw, ch), parts = span.layout(mons, sp)
    xs = np.arange(cw)
    arr = np.zeros((ch, cw, 4), np.uint8)
    arr[..., 0] = (xs % 256)[None, :]
    arr[..., 1] = (xs // 256)[None, :]
    arr[..., 3] = 255
    p = os.path.join(tempfile.mkdtemp(), "canvas.png")
    Image.fromarray(arr, "RGBA").save(p)
    lay = {0: dict(layer(p, 0, screens=(0, 1)), position={"x": 0.5, "y": 0.5, "scale_x": 1.0,
                                                           "scale_y": 1.0, "rotation": 0})}
    return mons, parts, (cw, ch), lay


def _canvas_x(img, x, y=540):
    r, g, _ = px(img, x, y)
    return g * 256 + r


def scenario_span(fx):
    """Each display shows its own part of the canvas at 1:1: canvas x read back
    at points of both displays, before and after a corner pin, and in a
    crossfade."""
    mons, parts, canvas, L = _span_setup()
    eng = make_engine()
    eng.monitors = mons
    eng.set_output_regions(parts)
    eng.update_layers(L)
    for _ in range(4):
        eng.render(L, target_monitor=None, transition_state={"active": False}, all_assigned_screens={0, 1})
    img = LAST["img"].convert("RGB")
    probes = (0, 100, 959, 960, 961, 1500, 1919)
    res = {"canvas": list(canvas), "size": list(img.size), "read": {str(x): _canvas_x(img, x) for x in probes},
           "composition": [list(eng.composition_size(0)), list(eng.composition_size(1))],
           "targets": len(eng._warp_targets)}
    # Crossfade into the same picture on the span: still each display's part
    ts = {"active": True, "progress": 0.5, "curve": "linear", "from": L, "to": L}
    for _ in range(2):
        eng.render({}, target_monitor=None, transition_state=ts, all_assigned_screens={0, 1})
    img = LAST["img"].convert("RGB")
    res["mix_read"] = {str(x): _canvas_x(img, x) for x in (100, 1500)}
    eng.set_output_regions({})
    for _ in range(2):
        eng.render(L, target_monitor=None, transition_state={"active": False}, all_assigned_screens={0, 1})
    img = LAST["img"].convert("RGB")
    res["no_span_read"] = {str(x): _canvas_x(img, x) for x in (100, 1500)}
    res["targets_after"] = len(eng._warp_targets)
    eng.update_layers({})
    eng.clear()
    return res


def scenario_warp_first_frame(fx):
    """The first frame after a display gets an edge blend draws the picture.
    (The warp program is compiled in that frame; it left no layer quad bound,
    and after the output had been on one display that draw crashed.)"""
    import tempfile
    from types import SimpleNamespace
    p = os.path.join(tempfile.mkdtemp(), "w.png")
    Image.new("RGBA", (960, 1080), (200, 100, 50, 255)).save(p)
    eng = make_engine()
    eng.monitors = [SimpleNamespace(x=0, y=0, width=960, height=1080, name="A"),
                    SimpleNamespace(x=960, y=0, width=960, height=1080, name="B")]
    single = {0: layer(p, 0, screens=(0,))}
    both = {0: layer(p, 0, screens=(0, 1))}
    eng.update_layers(single)
    for _ in range(3):
        eng.render(single, target_monitor=0, transition_state={"active": False})
    res = {"single": px(LAST["img"].convert("RGB"), 480, 540)}
    eng.set_output_blends({1: {"left": 0.1}})
    eng.render(both, target_monitor=None, transition_state={"active": False}, all_assigned_screens={0, 1})
    img = LAST["img"].convert("RGB")
    res["first_frame"] = [px(img, 480, 540), px(img, 1440, 540)]
    eng.clear()
    return res


def scenario_span_grid(fx):
    """A 2 x 2 grid of 960x540 displays (the four quarters of the screen),
    overlaps 160 px across and 60 px down: a 1760 x 1020 canvas. Picture:
    canvas x and y encoded in R, G, B. Reads canvas positions back at points
    of all four displays."""
    import tempfile
    import numpy as np
    from types import SimpleNamespace
    import span
    mons = [SimpleNamespace(x=x, y=y, width=960, height=540, name=f"{x},{y}")
            for y in (0, 540) for x in (0, 960)]
    sp = span.clean({"displays": [0, 1, 2, 3], "overlap": 160, "columns": 2, "overlap_y": 60})
    (cw, ch), parts = span.layout(mons, sp)
    yy, xx = np.mgrid[0:ch, 0:cw]
    arr = np.zeros((ch, cw, 4), np.uint8)
    arr[..., 0] = xx % 256
    arr[..., 1] = (xx // 256) | ((yy // 256) << 4)
    arr[..., 2] = yy % 256
    arr[..., 3] = 255
    p = os.path.join(tempfile.mkdtemp(), "grid.png")
    Image.fromarray(arr, "RGBA").save(p)
    L = {0: dict(layer(p, 0, screens=(0, 1, 2, 3)), position={"x": 0.5, "y": 0.5, "scale_x": 1.0,
                                                                 "scale_y": 1.0, "rotation": 0})}
    eng = make_engine()
    eng.monitors = mons
    eng.set_output_regions(parts)
    eng.update_layers(L)
    for _ in range(4):
        eng.render(L, target_monitor=None, transition_state={"active": False}, all_assigned_screens={0, 1, 2, 3})
    img = LAST["img"].convert("RGB")

    def read(x, y):
        r, g, b = px(img, x, y)
        return [(g & 15) * 256 + r, (g >> 4) * 256 + b]
    probes = [(100, 100), (900, 500), (1060, 100), (1800, 500), (100, 600), (900, 1000), (1060, 600), (1900, 1070)]
    res = {"canvas": [cw, ch], "read": [read(x, y) for x, y in probes], "probes": probes}
    eng.update_layers({})
    eng.clear()
    return res


def scenario_span_slices(fx):
    """Slices set by hand on two 960x1080 displays (the two halves of the
    screen), canvas 1600 x 900: display 0 shows the part x 0-800, y 0-900
    scaled up to 960 x 1080; display 1 the part x 1200-1600, y 450-900
    (a zoom: 400 x 450 scaled to 960 x 1080). Picture: canvas x and y
    encoded in R, G, B."""
    import tempfile
    import numpy as np
    from types import SimpleNamespace
    import span
    mons = [SimpleNamespace(x=x, y=0, width=960, height=1080, name=str(x)) for x in (0, 960)]
    sp = span.clean({"displays": [0, 1], "canvas": [1600, 900],
                     "slices": {"0": [0, 0, 800, 900], "1": [1200, 450, 400, 450]}})
    (cw, ch), parts = span.layout(mons, sp)
    yy, xx = np.mgrid[0:ch, 0:cw]
    arr = np.zeros((ch, cw, 4), np.uint8)
    arr[..., 0] = xx % 256
    arr[..., 1] = (xx // 256) | ((yy // 256) << 4)
    arr[..., 2] = yy % 256
    arr[..., 3] = 255
    p = os.path.join(tempfile.mkdtemp(), "slices.png")
    Image.fromarray(arr, "RGBA").save(p)
    L = {0: dict(layer(p, 0, screens=(0, 1)), position={"x": 0.5, "y": 0.5, "scale_x": 1.0,
                                                         "scale_y": 1.0, "rotation": 0})}
    eng = make_engine()
    eng.monitors = mons
    eng.set_output_regions(parts)
    eng.update_layers(L)
    for _ in range(4):
        eng.render(L, target_monitor=None, transition_state={"active": False}, all_assigned_screens={0, 1})
    img = LAST["img"].convert("RGB")

    def read(x, y):
        r, g, b = px(img, x, y)
        return [(g & 15) * 256 + r, (g >> 4) * 256 + b]
    probes = [(120, 120), (840, 960), (1080, 120), (1800, 960)]
    res = {"canvas": [cw, ch], "read": [read(x, y) for x, y in probes], "probes": probes,
           "composition": [list(eng.composition_size(i)) for i in (0, 1)]}
    eng.update_layers({})
    eng.clear()
    return res


def scenario_layer_mask(fx):
    """Layer masks on a 400x200 white image at 1:1 in the middle of the output:
    a rectangle (orientation), a soft ellipse (values across the soft edge, to
    compare with layer_mask.factor), an inverted mask, a rotated layer (the
    mask turns with it), a video (YUV path) and the static-cue cache bypass."""
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "white.png")
    Image.new("RGBA", (400, 200), (255, 255, 255, 255)).save(p)
    eng = make_engine()
    res = {}

    def shot(mask, rotation=0, path=p, scale=1.0):
        L = {0: dict(layer(path, 0, scale=scale), mask=mask)}
        L[0]["position"]["rotation"] = rotation
        eng.update_layers(L)
        img = settle(eng, L, n=4)
        return img, (img.width - 400) // 2, (img.height - 200) // 2

    img, x0, y0 = shot({"shape": "rect", "left": 0.25, "top": 0.1})
    res["rect"] = {k: px(img, x0 + u * 400, y0 + v * 200)[0] for k, (u, v) in
                   {"left_hidden": (0.1, 0.5), "top_hidden": (0.5, 0.05), "center": (0.5, 0.5),
                    "right_edge_in": (0.98, 0.5), "bottom_in": (0.5, 0.97)}.items()}
    img, x0, y0 = shot({"shape": "ellipse", "feather": 0.1})
    res["ellipse_row"] = [px(img, x0 + i, y0 + 100)[0] for i in range(0, 400, 4)]
    res["ellipse_corner"] = px(img, x0 + 8, y0 + 8)[0]
    misses = eng.static_cue_cache_misses
    img, x0, y0 = shot({"shape": "rect", "left": 0.5, "invert": True})
    res["invert"] = [px(img, x0 + 100, y0 + 100)[0], px(img, x0 + 300, y0 + 100)[0]]
    res["static_cache_bypassed"] = eng.static_cue_cache_misses == misses
    img, _, _ = shot({"shape": "rect", "left": 0.5}, rotation=90)
    cx, cy = img.width // 2, img.height // 2
    res["rotated"] = {"above": px(img, cx, cy - 100)[0], "below": px(img, cx, cy + 100)[0]}
    mon = eng.monitors[0]
    img, _, _ = shot({"shape": "ellipse"}, path=fx["red_709"], scale=mon.width / 640.0)
    res["video"] = {"center": px(img, img.width / 2, img.height / 2), "corner": px(img, 20, 20)}
    eng.update_layers({})
    eng.clear()
    return res


def scenario_static_cache_release(fx):
    """A still cue is pre-rendered into an FBO; a UI-side invalidation must
    free that FBO and texture (on the render thread, at the next frame), not
    just forget them."""
    from OpenGL.GL import glIsTexture, glIsFramebuffer
    eng = make_engine()
    L = {0: layer(fx["red_png"], 0)}
    eng.update_layers(L)
    settle(eng, L, n=3)
    data = next(iter(eng.static_cue_cache.values()), None)
    res = {"cached": len(eng.static_cue_cache), "is_texture_before": bool(data and glIsTexture(data["texture"]))}
    eng.invalidate_static_cache_for_column(0)          # what the UI thread does after an edit
    res["entries_after_invalidate"] = len(eng.static_cue_cache)
    eng.update_layers({})
    settle(eng, {}, n=2)                                 # next frames, nothing new to pre-render
    res["is_texture_after"] = bool(glIsTexture(data["texture"])) if data else None
    res["is_fbo_after"] = bool(glIsFramebuffer(data["fbo"])) if data else None
    res["entries_after_frames"] = len(eng.static_cue_cache)
    eng.clear()
    return res


def scenario_live_source(fx):
    """A UDP MPEG-TS stream as a layer: frames arrive and show; two layers with
    the same source share one player; when the sender stops, the last frame
    stays and the player retries; when it comes back (now green), the output
    follows without anything being re-cut."""
    sys.path.insert(0, HERE)
    from stream_sender import UdpSender, free_udp_port
    port = free_udp_port()
    url = f"udp://127.0.0.1:{port}"
    eng = make_engine()
    mon = eng.monitors[0]
    sc = mon.width / 320.0 * 1.05
    L = {0: {**layer(url, 0, scale=sc), "path": url}, 1: {**layer(url, 1, scale=0.2, x=0.1, y=0.1), "path": url}}
    eng.update_layers(L)
    snd = UdpSender(port, (255, 0, 0)).start()
    res = {"players": len(eng.active_players)}
    t0 = time.monotonic()
    img = None
    while time.monotonic() - t0 < 10:
        img = settle(eng, L, n=2)
        if px(img, img.width / 2, img.height / 2)[0] > 200:
            break
    res["first_frame_s"] = round(time.monotonic() - t0, 2)
    res["red"] = px(img, img.width / 2, img.height / 2)
    p = next(iter(eng.active_players.values()))["player"]
    res["state_playing"] = p.state
    snd.stop()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10 and p.state == "playing":
        img = settle(eng, L, n=2)
    res["after_stop_state"] = p.state
    res["after_stop_s"] = round(time.monotonic() - t0, 2)
    img = settle(eng, L, n=2)
    res["held_frame"] = px(img, img.width / 2, img.height / 2)
    snd = UdpSender(port, (0, 255, 0)).start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 15:
        img = settle(eng, L, n=2)
        if px(img, img.width / 2, img.height / 2)[1] > 200:
            break
    res["recovered_s"] = round(time.monotonic() - t0, 2)
    res["green"] = px(img, img.width / 2, img.height / 2)
    res["reconnects"] = p.reconnects
    res["state_end"] = p.state
    snd.stop()
    import health
    res["health"] = health.player_status(url, next(iter(eng.active_players.values())))
    eng.update_layers({})
    res["players_after"] = len(eng.active_players)
    eng.clear()
    return res


def scenario_effects(fx):
    """Shader effects (isf.py): every built-in effect compiles and draws;
    Invert on a solid image, at full and half opacity; orientation (Mirror
    keeping the top or the bottom half of a red-over-blue image, and a test
    effect that draws isf_FragNormCoord.y); a video (YUV path); a broken
    effect (the layer is drawn without it and the error is kept); the
    static-cue cache bypass; the frame counter of a moving effect."""
    import tempfile
    import isf
    d = tempfile.mkdtemp()
    solid, split = os.path.join(d, "solid.png"), os.path.join(d, "split.png")
    Image.new("RGBA", (200, 150), (200, 100, 50, 255)).save(solid)
    im = Image.new("RGBA", (200, 150), (0, 0, 255, 255))
    im.paste((255, 0, 0, 255), (0, 0, 200, 75))
    im.save(split)
    coord = os.path.join(d, "coord.fs")
    with open(coord, "w") as f:
        f.write('/*{"INPUTS": [{"NAME": "inputImage", "TYPE": "image"}]}*/\n'
                'void main() { gl_FragColor = vec4(isf_FragNormCoord.y, isf_FragNormCoord.x, 0.0, 1.0); }\n')
    broken = os.path.join(d, "broken.fs")
    with open(broken, "w") as f:
        f.write('/*{"INPUTS": [{"NAME": "inputImage", "TYPE": "image"}]}*/\n'
                'void main() { gl_FragColor = not_a_function(1.0); }\n')
    eng = make_engine()
    mon = eng.monitors[0]
    sc = mon.width / 200.0 * 1.1
    res = {}

    def shot(path, effect, scale=sc, n=4, **extra):
        L = {0: dict(layer(path, 0, scale=scale), effect=effect, **extra)}
        eng.update_layers(L)
        return settle(eng, L, n=n)

    def mid(img):
        return px(img, img.width / 2, img.height / 2)

    misses = eng.static_cue_cache_misses
    res["invert"] = mid(shot(solid, {"name": "Invert.fs", "inputs": {}}))
    L = {0: dict(layer(solid, 0, scale=sc), effect={"name": "Invert.fs", "inputs": {}})}
    eng.request_preview_capture(0)
    eng.render(L, target_monitor=0, transition_state={"active": False})
    prev = eng.take_preview_image()
    res["preview_invert"] = px(prev, prev.width / 2, prev.height / 2) if prev else None
    res["static_cache_bypassed"] = eng.static_cue_cache_misses == misses
    res["invert_half_amount"] = mid(shot(solid, {"name": "Invert.fs", "inputs": {"amount": 0.5}}))
    res["invert_half_opacity"] = mid(shot(solid, {"name": "Invert.fs", "inputs": {}}, opacity=0.5))
    for direction, key in ((2, "mirror_top"), (3, "mirror_bottom")):
        img = shot(split, {"name": "Mirror.fs", "inputs": {"direction": direction}})
        res[key] = [px(img, img.width / 2, img.height * 0.25), px(img, img.width / 2, img.height * 0.75)]
    img = shot(split, {"name": "Mirror.fs", "inputs": {"direction": 0}})
    res["mirror_left_keeps_rows"] = [px(img, img.width / 2, img.height * 0.25), px(img, img.width / 2, img.height * 0.75)]
    # 1:1 in the middle: the layer covers 200x150 pixels; y near its top must be near 1.
    img = shot(split, {"name": coord, "inputs": {}}, scale=1.0)
    x0, y0 = (img.width - 200) // 2, (img.height - 150) // 2
    res["coord_top_left"] = px(img, x0 + 2, y0 + 2)
    res["coord_bottom_right"] = px(img, x0 + 197, y0 + 147)
    with open(coord, "w") as f:                 # the file changes: compiled again once told
        f.write('/*{"INPUTS": [{"NAME": "inputImage", "TYPE": "image"}]}*/\n'
                'void main() { gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0); }\n')
    res["coord_unchanged_until_told"] = px(shot(split, {"name": coord, "inputs": {}}, scale=1.0), x0 + 2, y0 + 2)
    eng.reload_effects({coord})
    res["coord_reloaded"] = px(shot(split, {"name": coord, "inputs": {}}, scale=1.0), x0 + 2, y0 + 2)
    res["builtins"] = {}
    for name in sorted(isf.available()):
        img = shot(solid, {"name": name, "inputs": {}})
        res["builtins"][name] = {"error": eng.effect_errors.get(name), "center": mid(img)}
    img = shot(solid, {"name": broken, "inputs": {}})
    res["broken"] = {"center": mid(img), "error": eng.effect_errors.get(broken)}
    # Blur (two passes) across the red/blue edge of a 1:1 picture in the middle of the output
    img = shot(split, {"name": "Blur.fs", "inputs": {"radius": 20}}, scale=1.0)
    x0, y0 = (img.width - 200) // 2, (img.height - 150) // 2
    res["blur_edge"] = [px(img, x0 + 100, y0 + 75 + dy) for dy in (-60, -10, 0, 10, 60)]
    # Stacks: each effect works on the result of the one before; one that cannot run is left out.
    def stack(path, effects, **kw):
        L = {0: dict(layer(path, 0, scale=kw.get("scale", sc)), effects=effects)}
        eng.update_layers(L)
        return settle(eng, L, n=4)
    inv = {"name": "Invert.fs", "inputs": {}}
    res["stack_invert_twice"] = mid(stack(solid, [inv, inv]))
    img = stack(split, [{"name": "Mirror.fs", "inputs": {"direction": 2}}, inv])
    res["stack_mirror_then_invert"] = [px(img, img.width / 2, img.height * 0.25), px(img, img.width / 2, img.height * 0.75)]
    res["stack_with_broken"] = mid(stack(solid, [inv, {"name": broken, "inputs": {}}]))
    V = {0: dict(layer(fx["red_709"], 0, scale=mon.width / 640.0), effects=[inv, inv, inv])}
    eng.update_layers(V)
    res["stack_video_three"] = mid(settle(eng, V))
    eng.update_layers({})
    wave = shot(solid, {"name": "Wave.fs", "inputs": {}}, n=6)
    res["wave_frames"] = eng._fx_clock.get(solid, [0, 0, 0])[2]
    V = {0: dict(layer(fx["red_709"], 0, scale=mon.width / 640.0), effect={"name": "Invert.fs", "inputs": {}})}
    eng.update_layers(V)
    res["video_invert"] = mid(settle(eng, V))
    # Drawn smaller than it is: the effect runs at the drawn size (a quarter of
    # the work at half size), and the picture is still the effect's.
    V = {0: dict(layer(fx["red_709"], 0, scale=0.5), effect={"name": "Invert.fs", "inputs": {}})}
    eng.update_layers(V)
    img = settle(eng, V)
    vkey = eng.source_key(V[0])
    tw, th = eng.textures[vkey]["width"], eng.textures[vkey]["height"]
    res["half_scale"] = {"source": [tw, th], "target": list(eng._fx_targets[(vkey, "b")][2:]), "center": mid(img)}
    res["fx_targets_before_clear"] = len(eng._fx_targets)
    eng.update_layers({})
    settle(eng, {}, n=3)
    res["fx_targets_after_clear"] = len(eng._fx_targets)
    eng.clear()
    return res


def scenario_output_effects(fx):
    """A display's effect (set_output_effects) on its whole picture: Invert on
    a full-screen image; orientation (Mirror keeping the top half of a
    red-top/blue-bottom picture: red everywhere); the corner pin applied
    after it; the still-cue cache path; buffers freed when it is removed."""
    import tempfile
    import output_warp
    d = tempfile.mkdtemp()
    solid, split = os.path.join(d, "solid.png"), os.path.join(d, "split.png")
    Image.new("RGBA", (192, 108), (200, 100, 50, 255)).save(solid)
    im = Image.new("RGBA", (192, 108), (0, 0, 255, 255))
    im.paste((255, 0, 0, 255), (0, 0, 192, 54))
    im.save(split)
    eng = make_engine()
    mon = eng.monitors[0]
    full = {"x": 0.5, "y": 0.5, "scale_x": mon.width / 192.0, "scale_y": mon.height / 108.0, "rotation": 0}
    res = {}

    def shot(path, n=4):
        L = {0: dict(layer(path, 0), position=dict(full))}
        eng.update_layers(L)
        return settle(eng, L, n=n).convert("RGB")

    img = shot(solid)
    res["plain"] = px(img, img.width / 2, img.height / 2)
    eng.set_output_effects({0: {"name": "Invert.fs", "inputs": {}}})
    img = shot(solid)
    res["invert"] = px(img, img.width / 2, img.height / 2)
    res["invert_corner"] = px(img, 3, 3)
    L = {0: dict(layer(solid, 0), position=dict(full))}
    eng.request_preview_capture(0)
    eng.render(L, target_monitor=0, transition_state={"active": False})
    prev = eng.take_preview_image()
    res["preview"] = px(prev, prev.width / 2, prev.height / 2) if prev else None
    # What no layer covers stays black (Invert would make black white): a small
    # image in the middle, the same at half opacity, and nothing at all.
    L = {0: layer(solid, 0)}
    eng.update_layers(L)
    img = settle(eng, L, n=4).convert("RGB")
    res["small_inside"], res["small_outside"] = px(img, img.width / 2, img.height / 2), px(img, 20, 20)
    L = {0: dict(layer(solid, 0), position=dict(full), opacity=0.5)}
    eng.update_layers(L)
    img = settle(eng, L, n=4).convert("RGB")
    res["half_opacity"] = px(img, img.width / 2, img.height / 2)
    eng.update_layers({})
    img = settle(eng, {}, n=3).convert("RGB")
    res["nothing"] = px(img, img.width / 2, img.height / 2)
    eng.set_output_effects({"0": {"name": "Mirror.fs", "inputs": {"direction": 2}}})   # keep the top half
    img = shot(split)
    res["mirror_top"] = [px(img, img.width / 2, img.height * 0.25), px(img, img.width / 2, img.height * 0.75)]
    eng.set_output_effects({0: {"name": "Mirror.fs", "inputs": {"direction": 0}}})     # keep the left half
    img = shot(split)
    res["mirror_left_rows"] = [px(img, img.width / 2, img.height * 0.25), px(img, img.width / 2, img.height * 0.75)]
    eng.set_output_effects({0: [{"name": "Mirror.fs", "inputs": {"direction": 3}},       # keep the bottom half,
                                {"name": "Invert.fs", "inputs": {}}]})                  # then invert
    img = shot(split)
    res["stack"] = [px(img, img.width / 2, img.height * 0.25), px(img, img.width / 2, img.height * 0.75)]
    res["stack_targets"] = len([k for k in eng._fx_targets if k[0] == ("display", 0) or k[0] == (("display", 0), 1)])
    # Corner pin after the effect: inverted picture inside the pin, black outside.
    eng.set_output_effects({0: {"name": "Invert.fs", "inputs": {}}})
    eng.set_output_warps({0: WARP})
    img = shot(solid)
    h = output_warp.homography(WARP)
    x, y = output_warp.apply(h, 0.5, 0.5)
    res["pinned_center"] = px(img, x * img.width, y * img.height)
    res["pinned_outside"] = px(img, 5, 5)
    eng.set_output_warps({})
    # A crossfade between two small images: the mixed picture keeps its coverage too.
    F, T = {0: layer(solid, 0)}, {0: layer(split, 0, x=0.25)}
    eng.update_layers({0: F[0], 1: T[0]})
    for linear_p in (0.5,):
        ts = {"active": True, "progress": linear_p, "curve": "linear", "from": F, "to": T}
        for _ in range(3):
            eng.render({}, target_monitor=0, transition_state=ts)
    img = LAST["img"].convert("RGB")
    res["mix_center"], res["mix_empty"] = px(img, img.width / 2, img.height / 2), px(img, img.width - 20, 20)
    res["moving"] = eng.output_effects_moving
    eng.set_output_effects({0: {"name": "Wave.fs", "inputs": {}}})
    res["moving_wave"] = eng.output_effects_moving
    res["targets_with"] = [len(eng._warp_targets), len([k for k in eng._fx_targets if k[0][0] == "display"])]
    eng.set_output_effects({})
    img = shot(solid)
    res["removed"] = px(img, img.width / 2, img.height / 2)
    res["targets_without"] = [len(eng._warp_targets), len([k for k in eng._fx_targets if k[0][0] == "display"])]
    eng.update_layers({})
    eng.clear()
    return res


def scenario_multipass(fx):
    """ISF passes: a smaller TARGET read by the last pass; a PERSISTENT FLOAT
    buffer that adds up from frame to frame (read and drawn in the same
    pass); orientation through a buffer, on a layer and on a display (whose
    uncovered parts stay black)."""
    import tempfile
    d = tempfile.mkdtemp()
    solid, split = os.path.join(d, "solid.png"), os.path.join(d, "split.png")
    Image.new("RGBA", (200, 150), (200, 100, 50, 255)).save(solid)
    im = Image.new("RGBA", (200, 150), (0, 0, 255, 255))
    im.paste((255, 0, 0, 255), (0, 0, 200, 75))
    im.save(split)
    files = {
        "small": '/*{"INPUTS": [{"NAME": "inputImage", "TYPE": "image"}],'
                 ' "PASSES": [{"TARGET": "small", "WIDTH": "$WIDTH/4", "HEIGHT": "floor($HEIGHT/4)"}, {}]}*/\n'
                 'void main() {\n'
                 '  if (PASSINDEX == 0) gl_FragColor = vec4(1.0 - IMG_THIS_NORM_PIXEL(inputImage).rgb, 1.0);\n'
                 '  else gl_FragColor = IMG_NORM_PIXEL(small, isf_FragNormCoord);\n}\n',
        "count": '/*{"INPUTS": [{"NAME": "inputImage", "TYPE": "image"}, {"NAME": "step", "TYPE": "float",'
                 ' "DEFAULT": 0.05, "MIN": 0, "MAX": 1}],'
                 ' "PASSES": [{"TARGET": "acc", "PERSISTENT": true, "FLOAT": true}, {}]}*/\n'
                 'void main() {\n'
                 '  if (PASSINDEX == 0) gl_FragColor = vec4(IMG_THIS_NORM_PIXEL(acc).r + step, 0.0, 0.0, 1.0);\n'
                 '  else gl_FragColor = vec4(min(IMG_THIS_NORM_PIXEL(acc).r, 1.0), 0.0, 0.0, 1.0);\n}\n',
        "copy": '/*{"INPUTS": [{"NAME": "inputImage", "TYPE": "image"}], "PASSES": [{"TARGET": "c"}, {}]}*/\n'
                'void main() {\n'
                '  if (PASSINDEX == 0) gl_FragColor = IMG_THIS_NORM_PIXEL(inputImage);\n'
                '  else gl_FragColor = vec4(1.0 - IMG_THIS_NORM_PIXEL(c).rgb, IMG_THIS_NORM_PIXEL(c).a);\n}\n',
    }
    paths = {}
    for k, text in files.items():
        paths[k] = os.path.join(d, k + ".fs")
        with open(paths[k], "w") as f:
            f.write(text)
    eng = make_engine()
    mon = eng.monitors[0]
    sc = mon.width / 200.0 * 1.1
    res = {}

    def shot(path, effects, n=4, scale=sc):
        L = {0: dict(layer(path, 0, scale=scale), effects=effects)}
        eng.update_layers(L)
        return settle(eng, L, n=n)

    def mid(img):
        return px(img, img.width / 2, img.height / 2)

    res["small"] = mid(shot(solid, [{"name": paths["small"], "inputs": {}}]))
    res["small_size"] = [list(b["size"]) for k, b in eng._fx_buffers.items() if k[2] == "small"]
    res["errors"] = dict(eng.effect_errors)
    count = [{"name": paths["count"], "inputs": {}}]
    a = mid(shot(solid, count, n=4))[0]
    b = mid(shot(solid, count, n=6))[0]
    res["count"] = [a, b, eng._fx_clock.get(solid, [0, 0, 0])[2]]
    img = shot(split, [{"name": paths["copy"], "inputs": {}}])
    res["copy_layer"] = [px(img, img.width / 2, img.height * 0.25), px(img, img.width / 2, img.height * 0.75)]
    eng.update_layers({})
    full = {"x": 0.5, "y": 0.5, "scale_x": mon.width / 200.0, "scale_y": mon.height / 150.0 / 2, "rotation": 0}
    eng.set_output_effects({0: [{"name": paths["copy"], "inputs": {}}]})
    L = {0: dict(layer(split, 0), position=full)}                 # covers the middle half of the height
    eng.update_layers(L)
    img = settle(eng, L, n=4)
    res["copy_display"] = [px(img, img.width / 2, img.height * 0.375), px(img, img.width / 2, img.height * 0.625),
                           px(img, img.width / 2, 10)]
    eng.set_output_effects({})
    eng.update_layers({})
    settle(eng, {}, n=2)
    res["buffers_left"] = len(eng._fx_buffers)
    eng.clear()
    return res


def scenario_effect_preview(fx):
    """The position editor's effect preview (effect_preview_begin/poll/end):
    the editor's picture with its effects, scaled to the canvas size, read
    back through a pixel buffer; orientation (top row first), a moving effect
    keeps its buffers across output frames, and closing frees them."""
    import tempfile
    import numpy as np
    d = tempfile.mkdtemp()
    solid = os.path.join(d, "solid.png")
    Image.new("RGBA", (64, 48), (200, 100, 50, 255)).save(solid)
    eng = make_engine()
    L = {0: layer(solid, 0)}
    eng.update_layers(L)
    settle(eng, L, n=2)                                  # the output window exists
    split = np.zeros((300, 400, 4), dtype=np.uint8)
    split[..., 3] = 255
    split[:150, :, 0] = 255                              # red top half
    split[150:, :, 2] = 255                              # blue bottom half
    res = {}

    def run(effects, image=None, out=(200, 150), drawn=(400, 300)):
        why = eng.effect_preview_begin({"image": image, "image_id": "pic1", "effects": effects,
                                        "drawn": drawn, "out": out, "seq": 1})
        if why is not None:
            return {"why": why}
        got = eng.effect_preview_poll(wait=True)
        arr = got[1]
        h, w = arr.shape[:2]
        return {"shape": list(arr.shape), "top": arr[h * 2 // 15, w // 2].tolist(),
                "bottom": arr[h * 13 // 15, w // 2].tolist()}

    res["plain"] = run([], image=split)
    res["mirror_bottom"] = run([{"name": "Mirror.fs", "inputs": {"direction": 3}}])   # keep bottom: all blue
    res["invert"] = run([{"name": "Invert.fs", "inputs": {}}])
    res["small_out"] = run([{"name": "Invert.fs", "inputs": {}}], out=(40, 30))
    # A moving effect with a buffer (Trails): its buffers survive the output's frames.
    run([{"name": "Trails.fs", "inputs": {}}])
    before = len(eng._fx_buffers)
    settle(eng, L, n=3)
    res["trails_kept"] = before > 0 and len(eng._fx_buffers) >= before
    res["busy_after"] = eng.effect_preview_busy()
    eng.effect_preview_end()
    res["after_end"] = {"targets": sum(1 for k in eng._fx_targets if eng._is_preview_key(k[0])),
                        "buffers": sum(1 for k in eng._fx_buffers if eng._is_preview_key(k[0]))}
    res["no_picture"] = eng.effect_preview_begin({"image": None, "image_id": "pic1", "effects": [],
                                                  "drawn": (1, 1), "out": (10, 10), "seq": 2})
    eng.clear()
    return res


def scenario_layer_warp(fx):
    """Warp points on a layer in the OpenGL engine: the right edge pulled out
    (the picture now covers what was black), a keystone (a corner pulled in
    leaves black), the mask following the bent picture, a video, the still-cue
    cache not used, and the surfaces freed with the context."""
    import tempfile
    d = tempfile.mkdtemp()
    solid = os.path.join(d, "solid.png")
    Image.new("RGBA", (200, 100), (200, 100, 50, 255)).save(solid)
    eng = make_engine()
    mon = eng.monitors[0]
    s = mon.width / 2 / 200.0                          # half the output's width, in the middle
    res = {}

    def shot(warp=None, path=solid, **extra):
        L = {0: dict(layer(path, 0, scale=s), **extra)}
        if warp:
            L[0]["warp"] = warp
        eng.update_layers(L)
        return settle(eng, L, n=4)
    W, H = mon.width, mon.height
    right_out = {"cols": 2, "rows": 2, "points": [[0, 0], [1.4, 0], [0, 1], [1.4, 1]]}
    img = shot()
    res["plain"] = {"inside": px(img, W * 0.5, H * 0.5), "right_of": px(img, W * 0.8, H * 0.5)}
    misses = eng.static_cue_cache_misses
    img = shot(right_out)
    res["pulled"] = {"inside": px(img, W * 0.5, H * 0.5), "right_of": px(img, W * 0.8, H * 0.5),
                     "beyond": px(img, W * 0.95, H * 0.5)}
    res["static_cache_bypassed"] = eng.static_cue_cache_misses == misses
    keystone = {"cols": 2, "rows": 2, "points": [[0, 0], [0.6, 0.3], [0, 1], [1, 1]]}
    img = shot(keystone)
    # The layer spans x 0.25..0.75 W; its top right corner moved to 0.55 W, 30 % down.
    y_top = H * 0.5 - 100 * s / 2 + 100 * s * 0.05     # 5 % down: below the sloping top edge near the left
    res["keystone"] = {"top_left": px(img, W * 0.27, y_top), "top_right": px(img, W * 0.73, y_top),
                       "bottom_right": px(img, W * 0.73, H * 0.5 + 100 * s / 2 - 3)}
    # A mask in the picture's coordinates (its right half hidden) follows the bend.
    img = shot(right_out, mask={"shape": "rect", "right": 0.5})
    res["masked"] = {"left_part": px(img, W * 0.35, H * 0.5), "right_part": px(img, W * 0.8, H * 0.5)}
    V = {0: dict(layer(fx["red_709"], 0, scale=W / 2 / 640.0), warp=right_out)}
    eng.update_layers(V)
    img = settle(eng, V)
    res["video_right_of"] = px(img, W * 0.8, H * 0.5)
    res["geo_cached"] = len(getattr(eng, "_warp_geo", {}))
    eng.close_window()
    res["geo_after_close"] = len(getattr(eng, "_warp_geo", {}))
    eng.clear()
    return res


def scenario_effect_warmup(fx):
    """Effects queued for warm-up are compiled one per call, between frames,
    before their first draw; a broken one records its error; nothing twice."""
    import tempfile
    d = tempfile.mkdtemp()
    solid = os.path.join(d, "solid.png")
    Image.new("RGBA", (64, 48), (200, 100, 50, 255)).save(solid)
    eng = make_engine()
    L = {0: layer(solid, 0)}
    eng.update_layers(L)
    settle(eng, L, n=2)                                  # the output window exists
    res = {"before": sorted(eng._fx_programs)}
    eng.queue_effect_warmup(["Wave.fs", "Invert.fs", "Wave.fs", "NoSuch.fs"])
    res["queued"] = list(eng._fx_warm)
    res["first"] = eng.warm_one_effect()
    res["after_one"] = sorted(eng._fx_programs)
    res["rest"] = [eng.warm_one_effect(), eng.warm_one_effect(), eng.warm_one_effect()]
    res["programs"] = {k: v is not None for k, v in eng._fx_programs.items()}
    res["error"] = eng.effect_errors.get("NoSuch.fs")
    eng.queue_effect_warmup(["Wave.fs"])                 # compiled already: not queued again
    res["requeued"] = list(eng._fx_warm)
    L2 = {0: dict(layer(solid, 0), effects=[{"name": "Wave.fs", "inputs": {}}])}
    eng.update_layers(L2)
    settle(eng, L2, n=2)                                 # drawn with the compiled program
    res["drawn_ok"] = "Wave.fs" not in eng.effect_errors
    eng.clear()
    return res


def scenario_decode_at_drawn_size(fx):
    """A 4K clip (white left half, black right half) drawn at half size is decoded
    at 1920 x 1080 and drawn exactly where the full-size frames were: it fills
    the output, the edge between the halves stays in the middle. Drawn larger,
    or with warp points, it is decoded at its own size. An effect still works."""
    import tempfile
    import media_fixtures as mf
    d = tempfile.mkdtemp()
    clip = mf.split_clip(os.path.join(d, "split_4k.mp4"), 3840, 2160, n_frames=12, fps=25)
    eng = make_engine()
    mon = eng.monitors[0]
    W, H = mon.width, mon.height
    half = W / 3840.0
    res = {"output": [W, H]}

    def shot(name, **extra):
        L = {0: dict(layer(clip, 0, scale=extra.pop("scale", half)), **extra)}
        eng.update_layers(L)
        img = settle(eng, L, n=20)
        info = eng.active_players[eng.source_key(L[0])]
        tex = eng.textures.get(eng.source_key(L[0])) or {}
        res[name] = {"decode_scale": info["player"].decode_scale, "scaled_to": info["player"].scaled_to,
                     "tex_size": [tex.get("width"), tex.get("height")],
                     "left_edge": px(img, 2, H / 2), "left_of_middle": px(img, W / 2 - 6, H / 2),
                     "right_of_middle": px(img, W / 2 + 6, H / 2), "top_left": px(img, 2, 2)}
    shot("half")
    shot("larger", scale=half * 1.2)
    shot("warped", warp={"cols": 2, "rows": 2, "points": [[0, 0], [1.1, 0], [0, 1], [1.1, 1]]})
    shot("inverted", effects=[{"name": "Invert.fs", "inputs": {}}])
    res["fx_error"] = eng.effect_errors.get("Invert.fs")
    eng.clear()
    return res


def scenario_all_builtin_effects(fx):
    """Every built-in effect compiles and draws on this GPU (checklist D1 on the owner's
    PC runs this on its AMD driver): no error, and the picture it draws is not black."""
    import tempfile
    import isf
    d = tempfile.mkdtemp()
    img = os.path.join(d, "quads.png")
    q = Image.new("RGBA", (320, 180), (220, 30, 30, 255))
    q.paste((30, 200, 30, 255), (160, 0, 320, 90))
    q.paste((30, 30, 220, 255), (0, 90, 160, 180))
    q.paste((235, 235, 235, 255), (160, 90, 320, 180))
    q.save(img)
    eng = make_engine()
    mon = eng.monitors[0]
    sc = mon.width / 320.0
    res = {}
    for name in isf.available():
        L = {0: dict(layer(img, 0, scale=sc), effects=[{"name": name, "inputs": {}}])}
        eng.update_layers(L)
        shot = settle(eng, L, n=4)
        pts = [px(shot, mon.width * fx_, mon.height * fy_) for fx_ in (0.3, 0.7) for fy_ in (0.3, 0.7)]
        res[name] = {"error": eng.effect_errors.get(name), "lit": sum(1 for p in pts if max(p) > 20)}
    eng.clear()
    return res


SCENARIOS = {
    "all_builtin_effects": scenario_all_builtin_effects,
    "decode_at_drawn_size": scenario_decode_at_drawn_size,
    "effect_warmup": scenario_effect_warmup,
    "layer_warp": scenario_layer_warp,
    "effect_preview": scenario_effect_preview,
    "multipass": scenario_multipass,
    "output_effects": scenario_output_effects,
    "effects": scenario_effects,
    "span_grid": scenario_span_grid,
    "span_slices": scenario_span_slices,
    "warp_first_frame": scenario_warp_first_frame,
    "span": scenario_span,
    "mesh_warp": scenario_mesh_warp,
    "live_source": scenario_live_source,
    "static_cache_release": scenario_static_cache_release,
    "layer_mask": scenario_layer_mask,
    "edge_blend": scenario_edge_blend,
    "transition_curves": scenario_transition_curves,
    "corner_pin": scenario_corner_pin,
    "rotation": scenario_rotation,
    "color_adjust": scenario_color_adjust,
    "blend_modes": scenario_blend_modes,
    "opacity": scenario_opacity,
    "colors": scenario_colors,
    "shear": scenario_shear,
    "layer_order": scenario_layer_order,
    "cut_gap": scenario_cut_gap,
    "cut_different_files": scenario_cut_different_files,
    "cut_same_file": scenario_cut_same_file,
    "instances_same_file": scenario_instances_same_file,
    "invalid_screen": scenario_invalid_screen,
    "alpha_video": scenario_alpha_video,
    "preview_capture": scenario_preview_capture,
}

if __name__ == "__main__":
    name, fixtures_json = sys.argv[1], sys.argv[2]
    with open(fixtures_json) as f:
        fixtures = json.load(f)
    result = SCENARIOS[name](fixtures)
    print("RESULT_JSON:" + json.dumps(result))
