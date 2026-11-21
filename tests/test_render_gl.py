# SPDX-License-Identifier: MPL-2.0
"""OpenGL output correctness, checked by reading back the frame about to be presented.

These run the real OpenGLRenderEngine against Mesa (llvmpipe) under Xvfb.
They prove behavior of the engine's shader/upload/draw logic; they are not
a substitute for validation on the Windows GPU drivers VisualDeck targets.
"""
import re

import pytest

from conftest import run_gl_scenario
import render_engine as rem


def close(a, b, tol):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


@pytest.fixture(scope="module")
def colors(fixtures, x_display):
    return run_gl_scenario("colors", fixtures, x_display)


def test_limited_range_black_is_black(colors):
    # Y=16 is black in limited ("TV") range, which almost all H.264/H.265 uses.
    assert close(colors["black_709"], [0, 0, 0], 3), colors


def test_limited_range_white_is_white(colors):
    assert close(colors["white_709"], [255, 255, 255], 3), colors


def test_bt709_red(colors):
    assert close(colors["red_709"], [255, 0, 0], 8), colors


def test_bt601_red_uses_601_matrix(colors):
    assert close(colors["red_601"], [255, 0, 0], 8), colors


def test_full_range_black(colors):
    assert close(colors["black_full"], [0, 0, 0], 3), colors


def test_video_width_not_multiple_of_four_is_not_sheared(fixtures, x_display):
    r = run_gl_scenario("shear", fixtures, x_display)
    assert r["top_boundary"] == r["bottom_boundary"] == 427, r


def test_layer_order_respected_with_instancing_path(fixtures, x_display):
    """Images on layers above a video must be drawn on top of it."""
    r = run_gl_scenario("layer_order", fixtures, x_display, extra_env={"VD_FORCE_INSTANCED": "1"})
    assert close(r["center"], [255, 0, 0], 3), r
    assert close(r["left_quarter"], [0, 255, 0], 3), r


def test_cut_between_cues_never_presents_a_black_frame(fixtures, x_display):
    """White cue -> red cue via the real RenderThread: every presented frame is
    white or red. (Before the preroll change the UI stopped all players first
    and the output showed black frames; see scenario 'cut_gap'.)"""
    r = run_gl_scenario("cut_different_files", fixtures, x_display)
    assert r["frames"] > 10 and not r["errors"], r
    assert r["black_frames"] == 0, r
    assert close(r["last"][0], [255, 0, 0], 8), r


def test_cut_to_cue_restarting_the_same_file_never_presents_black(fixtures, x_display):
    r = run_gl_scenario("cut_same_file", fixtures, x_display)
    assert r["frames"] > 10 and r["black_frames"] == 0, r
    # the outgoing activation's player is stopped once the cut commits
    assert len(r["players_left"]) == 1 and r["players_left"][0].startswith("@2:"), r


def test_layers_routed_to_missing_display_do_not_break_output(fixtures, x_display):
    r = run_gl_scenario("invalid_screen", fixtures, x_display)
    assert not r["errors"], r
    assert close(r["center"], [0, 255, 0], 3), r


def test_video_alpha_channel_is_composited(fixtures, x_display):
    r = run_gl_scenario("alpha_video", fixtures, x_display)
    assert close(r["left"], [255, 0, 0], 6), r     # opaque half of the video
    assert close(r["right"], [0, 255, 0], 6), r    # transparent half shows the layer below


def test_gpu_preview_capture_matches_output(fixtures, x_display):
    r = run_gl_scenario("preview_capture", fixtures, x_display)
    assert r["size"] == [640, 360], r
    assert close(r["center"], [0, 255, 0], 8), r   # green image on top in the center
    assert close(r["corner"], [255, 0, 0], 8), r   # red video elsewhere


def test_same_file_on_two_layers_renders_independent_frames(fixtures, x_display):
    """One file, two instances: the playing one has advanced, the held one still
    shows frame 0 (black). With one player per file both would be identical."""
    r = run_gl_scenario("instances_same_file", fixtures, x_display)
    assert r["players"] == 2 and r["textures"] == 2, r
    assert r["right"] == [0, 0, 0] or max(r["right"]) <= 3, r
    assert min(r["left"]) >= 30, r


def test_layer_opacity_blends_and_multiplies_with_fades(fixtures, x_display):
    r = run_gl_scenario("opacity", fixtures, x_display)
    red, green, blue = r["half_over_green"]
    assert abs(red - 128) <= 4 and abs(green - 128) <= 4 and blue <= 4, r      # 50% red over green
    red, green, blue = r["half_alone"]
    assert abs(red - 128) <= 4 and green <= 4 and blue <= 4, r                 # 50% red over the black output
    red, green, blue = r["semi_png_static"]
    assert r["static_cache_used"] and abs(red - 128) <= 4 and green <= 4, r     # cached still cue, alpha once
    red, green, blue = r["faint_png_static"]
    assert abs(red - 51) <= 4 and green <= 4, r                                # 20% pixels are not dropped
    red, green, blue = r["half_in_half_fade"]
    assert abs(red - 64) <= 4 and green <= 4 and blue <= 4, r                  # 0.5 opacity x 0.5 fade


def _expected_blend(mode, op, b=(100, 150, 200), c=(200, 100, 50)):
    out = []
    for bv, cv in zip(b, c):
        d, s = bv / 255.0, cv / 255.0
        if mode == "normal":
            v = d * (1 - op) + s * op
        elif mode == "add":
            v = min(1.0, d + s * op)
        elif mode == "multiply":
            v = d * (1 - op + op * s)
        else:
            v = d + op * s * (1 - d)
        out.append(round(v * 255))
    return out


def test_blend_modes_on_the_gpu(fixtures, x_display):
    r = run_gl_scenario("blend_modes", fixtures, x_display)
    for key, got in r.items():
        mode, op = key.split("@")
        want = _expected_blend(mode, float(op))
        assert all(abs(g - w) <= 3 for g, w in zip(got, want)), (key, got, want)


def test_invalidated_still_cues_free_their_framebuffers(fixtures, x_display):
    """An edit invalidates the pre-rendered still cues from the UI thread; the
    render thread must free their FBO and texture (each the size of the
    output) at its next frame. Before 2026-11-12 they were only forgotten."""
    r = run_gl_scenario("static_cache_release", fixtures, x_display)
    assert r["cached"] == 1 and r["is_texture_before"]
    assert r["entries_after_invalidate"] == 1, "the UI thread must not touch the render thread's cache"
    assert r["entries_after_frames"] == 0 and r["is_texture_after"] is False and r["is_fbo_after"] is False, r


def test_invalidation_only_flags_and_the_render_thread_drops():
    eng = rem.BaseRenderEngine(media_manager=None, mode="opengl")
    freed = []
    eng._release_static_entry = freed.append
    eng.static_cue_cache = {"a": 1, "b": 2}
    eng.cue_is_static = {3: True, 4: True}
    eng.invalidate_static_cache_for_column(3)
    assert eng.static_cue_cache == {"a": 1, "b": 2} and eng.cue_is_static == {4: True}
    eng._drop_stale_static_cache()
    assert eng.static_cue_cache == {} and sorted(freed) == [1, 2]
    eng._drop_stale_static_cache()                       # nothing pending: nothing happens
    assert sorted(freed) == [1, 2]
    eng.static_cue_cache = {"c": 3}
    eng.clear_static_cache()
    assert eng.static_cue_cache == {"c": 3} and eng.cue_is_static == {}
    eng._drop_stale_static_cache()
    assert freed[-1] == 3 and eng.static_cue_cache == {}


def test_shader_effects(fixtures, x_display):
    """isf.py effects in the OpenGL engine (scenario 'effects')."""
    r = run_gl_scenario("effects", fixtures, x_display)
    assert close(r["invert"], [55, 155, 205], 3)
    assert r["static_cache_bypassed"]                     # an effect layer is drawn every time
    assert close(r["preview_invert"], [55, 155, 205], 3)  # the output preview shows the effect
    assert close(r["invert_half_amount"], [128, 128, 128], 3)
    assert close(r["invert_half_opacity"], [28, 78, 103], 3)   # opacity applies to the effect's picture
    # Orientation: Mirror keeping the top half shows red (the image's top) everywhere.
    assert all(close(c, [255, 0, 0], 3) for c in r["mirror_top"])
    assert all(close(c, [0, 0, 255], 3) for c in r["mirror_bottom"])
    assert close(r["mirror_left_keeps_rows"][0], [255, 0, 0], 3) and close(r["mirror_left_keeps_rows"][1], [0, 0, 255], 3)
    # isf_FragNormCoord: y is 1 at the top, x is 0 at the left (ISF's convention).
    assert r["coord_top_left"][0] > 240 and r["coord_top_left"][1] < 15
    assert r["coord_bottom_right"][0] < 15 and r["coord_bottom_right"][1] > 240
    assert r["coord_unchanged_until_told"][0] > 240 and close(r["coord_reloaded"], [0, 255, 0], 3)
    for name, b in r["builtins"].items():
        assert b["error"] is None, (name, b["error"])
    assert close(r["builtins"]["Invert.fs"]["center"], [55, 155, 205], 3)
    for name in ("Blur.fs", "Trails.fs", "Mirror.fs"):                       # a plain color stays as it is
        assert close(r["builtins"][name]["center"], [200, 100, 50], 3), name
    far_red, near_red, edge, near_blue, far_blue = r["blur_edge"]
    assert close(far_red, [255, 0, 0], 3) and close(far_blue, [0, 0, 255], 3)
    assert abs(edge[0] - edge[2]) <= 30 and 90 <= edge[0] <= 165                # about half and half at the edge
    assert near_red[0] > edge[0] > near_blue[0] and near_red[2] < edge[2] < near_blue[2]
    # A broken effect: the layer is drawn as it is, and the reason is kept.
    assert close(r["broken"]["center"], [200, 100, 50], 3)
    assert "not_a_function" in r["broken"]["error"] and re.match(r"0\s*[:(]\s*2\b", r["broken"]["error"])
    assert r["wave_frames"] >= 5
    assert close(r["stack_invert_twice"], [200, 100, 50], 3)
    assert all(close(c, [0, 255, 255], 3) for c in r["stack_mirror_then_invert"])
    assert close(r["stack_with_broken"], [55, 155, 205], 3)
    assert close(r["stack_video_three"], [0, 255, 255], 6)
    assert close(r["video_invert"], [0, 255, 255], 6)       # YUV video through the effect
    assert r["fx_targets_after_clear"] < r["fx_targets_before_clear"]   # the video's buffers went with it
    (sw, sh), (tw, th) = r["half_scale"]["source"], r["half_scale"]["target"]
    assert (tw, th) == (round(sw / 2), round(sh / 2)), r["half_scale"]   # run at the size it is drawn
    assert close(r["half_scale"]["center"], [0, 255, 255], 6)


def test_display_effects(fixtures, x_display):
    """A display's effect on its whole picture (scenario 'output_effects')."""
    r = run_gl_scenario("output_effects", fixtures, x_display)
    assert close(r["plain"], [200, 100, 50], 3)
    assert close(r["invert"], [55, 155, 205], 3) and close(r["invert_corner"], [55, 155, 205], 3)
    assert close(r["preview"], [55, 155, 205], 3)                  # the output preview shows it
    # What no layer covers stays black, though Invert turns black white.
    assert close(r["small_inside"], [55, 155, 205], 3) and r["small_outside"] == [0, 0, 0]
    assert close(r["half_opacity"], [28, 78, 103], 3)      # the effect, then the opacity
    assert r["nothing"] == [0, 0, 0]
    assert close(r["mix_center"], [28, 78, 103], 3) and r["mix_empty"] == [0, 0, 0]   # in a crossfade too
    assert all(close(c, [255, 0, 0], 3) for c in r["mirror_top"])                     # not upside down
    assert close(r["mirror_left_rows"][0], [255, 0, 0], 3) and close(r["mirror_left_rows"][1], [0, 0, 255], 3)
    assert close(r["pinned_center"], [55, 155, 205], 3) and r["pinned_outside"] == [0, 0, 0]  # pin after effect
    assert all(close(c, [255, 255, 0], 3) for c in r["stack"])                    # blue kept, inverted
    assert r["stack_targets"] == 4                  # per effect: the picture in plain colors, and the result
    assert r["moving"] is False and r["moving_wave"] is True
    assert r["targets_with"] == [1, 2] and r["targets_without"] == [0, 0]
    assert close(r["removed"], [200, 100, 50], 3)


def test_multipass_effects(fixtures, x_display):
    """ISF passes with buffers (scenario 'multipass')."""
    r = run_gl_scenario("multipass", fixtures, x_display)
    assert r["errors"] == {}
    assert close(r["small"], [55, 155, 205], 3) and r["small_size"] == [[50, 37]]   # $WIDTH/4, floor($HEIGHT/4)
    a, b, _frames = r["count"]
    assert abs(a - 0.20 * 255) <= 2 and abs(b - 0.50 * 255) <= 2, r["count"]      # kept from frame to frame
    assert close(r["copy_layer"][0], [0, 255, 255], 3) and close(r["copy_layer"][1], [255, 255, 0], 3)
    top, bottom, uncovered = r["copy_display"]
    assert close(top, [0, 255, 255], 3) and close(bottom, [255, 255, 0], 3) and uncovered == [0, 0, 0]
    assert r["buffers_left"] == 0                  # nothing drawn with them any more: freed


def test_editor_effect_preview(fixtures, x_display):
    """The editor's canvas shows effects drawn by the output's engine (scenario 'effect_preview')."""
    r = run_gl_scenario("effect_preview", fixtures, x_display)
    assert r["plain"]["shape"] == [150, 200, 4]
    assert close(r["plain"]["top"], [255, 0, 0, 255], 3) and close(r["plain"]["bottom"], [0, 0, 255, 255], 3)
    assert close(r["mirror_bottom"]["top"], [0, 0, 255, 255], 3)            # top row first, not upside down
    assert close(r["invert"]["top"], [0, 255, 255, 255], 3) and close(r["invert"]["bottom"], [255, 255, 0, 255], 3)
    assert r["small_out"]["shape"] == [30, 40, 4]
    assert r["trails_kept"] and r["busy_after"] is False
    assert r["after_end"] == {"targets": 0, "buffers": 0}
    assert r["no_picture"] == "no picture"


def test_effects_are_compiled_ahead_of_their_first_draw(fixtures, x_display):
    """Owner's PC, 2026-11-14: a GO to a cue with an effect not drawn before stalled the
    output while the shader compiled (152 ms). The show's effects are compiled ahead,
    one per render loop (scenario 'effect_warmup')."""
    r = run_gl_scenario("effect_warmup", fixtures, x_display)
    assert r["before"] == [] and r["queued"] == ["Wave.fs", "Invert.fs", "NoSuch.fs"]
    assert r["first"] == "Wave.fs" and r["after_one"] == ["Wave.fs"]
    assert r["rest"] == ["Invert.fs", "NoSuch.fs", None]
    assert r["programs"] == {"Wave.fs": True, "Invert.fs": True, "NoSuch.fs": False}
    assert r["error"] and r["requeued"] == [] and r["drawn_ok"]


def test_every_builtin_effect_compiles_and_draws(fixtures, x_display):
    """Checklist D1 at the GPU level: all built-in effects, on whatever GPU runs the
    tests (the owner's AMD Radeon in the PC runs)."""
    import isf
    r = run_gl_scenario("all_builtin_effects", fixtures, x_display)
    assert set(r) == set(isf.available()) and len(r) >= 9
    bad = {k: v for k, v in r.items() if v["error"] or v["lit"] < 2}
    assert not bad, bad
