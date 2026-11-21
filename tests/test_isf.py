# SPDX-License-Identifier: MPL-2.0
"""Shader effects in the ISF format (isf.py): parsing, values, the GLSL that
is compiled, and the effects that come with VisualDeck. Drawing them is
tested in test_render_gl.py (scenario 'effects')."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import isf  # noqa: E402

IMAGE = {"NAME": "inputImage", "TYPE": "image"}


def src(inputs=(), body="void main() { gl_FragColor = IMG_THIS_PIXEL(inputImage); }", **meta):
    import json
    return "/*" + json.dumps({"INPUTS": [IMAGE, *inputs], **meta}) + "*/\n" + body


def test_parse_reads_inputs_and_body():
    e = isf.parse(src([{"NAME": "amount", "TYPE": "float", "DEFAULT": 0.5, "MIN": 0, "MAX": 1}],
                      DESCRIPTION="Test", CATEGORIES=["Color"]), "T.fs")
    assert e["name"] == "T.fs" and e["description"] == "Test" and e["categories"] == ["Color"]
    assert [i["NAME"] for i in e["inputs"]] == ["amount"]       # the image input is not a control
    assert e["inputs"][0]["DEFAULT"] == 0.5
    assert "void main" in e["body"] and e["body_line"] == 1


@pytest.mark.parametrize("text, why", [
    ("void main() {}", "no ISF header"),
    ("/*{not json}*/ void main() {}", "not valid JSON"),
    (src(PASSES=[{"TARGET": "bad name"}, {}]), "not a GLSL name"),
    (src([{"NAME": "amount", "TYPE": "float"}], PASSES=[{"TARGET": "amount"}, {}]), "name of an input"),
    (src(PASSES=[{"TARGET": "TIME"}, {}]), "name of an input or a built-in"),
    (src(PASSES=[{"TARGET": "b"}, {"TARGET": "b", "PERSISTENT": True}]), "declared twice"),
    (src(PASSES=[{"TARGET": "b", "WIDTH": "$WIDTH/"}, {}]), "pass WIDTH"),
    (src(PASSES=[{"TARGET": "b", "HEIGHT": "__import__('os')"}, {}]), "pass HEIGHT"),
    (src(PASSES={"TARGET": "b"}), "not a list"),
    (src(IMPORTED={"noise": {"PATH": "n.png"}}), "imported images"),
    (src([{"NAME": "other", "TYPE": "image"}]), "type 'image'"),
    (src([{"NAME": "sound", "TYPE": "audio"}]), "type 'audio'"),
    ("/*" + '{"INPUTS": [{"NAME": "a", "TYPE": "float"}]}' + "*/ void main() {}", "not a filter"),
    (src([{"NAME": "bad name", "TYPE": "float"}]), "not a GLSL name"),
])
def test_parse_refuses_what_cannot_run(text, why):
    with pytest.raises(isf.ISFError, match=why):
        isf.parse(text)


def test_a_single_pass_entry_is_accepted():
    e = isf.parse(src(PASSES=[{}]))
    assert e["inputs"] == [] and e["targets"] == [] and len(e["passes"]) == 1


def test_passes_targets_and_sizes():
    e = isf.parse(src([{"NAME": "blur", "TYPE": "float", "DEFAULT": 2, "MIN": 1, "MAX": 8}],
                      PASSES=[{"TARGET": "half", "WIDTH": "$WIDTH/$blur", "HEIGHT": 100},
                              {"TARGET": "acc", "PERSISTENT": True, "FLOAT": 1}, {"TARGET": "acc", "PERSISTENT": 1,
                                                                                 "FLOAT": True}, {}]))
    assert e["targets"] == ["half", "acc"]
    assert [(p["target"], p["persistent"], p["float"]) for p in e["passes"]] == [
        ("half", False, False), ("acc", True, True), ("acc", True, True), (None, False, False)]
    assert isf.size_expr(e["passes"][0]["width"], 1920, 1080, {"blur": 4.0}) == 480
    assert isf.size_expr(100, 1920, 1080, {}) == 100
    glsl = isf.fragment_source(e)
    assert "uniform sampler2D half;" in glsl and "uniform sampler2D acc;" in glsl


@pytest.mark.parametrize("expr, value", [
    ("$WIDTH", 640), ("$HEIGHT / 2", 240), ("floor($WIDTH / 3)", 213), ("max($WIDTH, $HEIGHT) * 0.5", 320),
    ("($WIDTH + 10) % 100", 50), ("2 ** 3", 8), ("-$x + 10", 7), ("round(sqrt(16))", 4), ("$on * 5", 5)])
def test_size_expressions(expr, value):
    assert isf.size_expr(expr, 640, 480, {"x": 3, "on": True, "c": [1, 2]}) == value


@pytest.mark.parametrize("expr", ["$c", "$nope", "1/0", "open('x')", "a.b", "2 ** 100", "[1]", "lambda: 1",
                                  "$WIDTH if 1 else 2", "sqrt(-1)"])
def test_size_expressions_refused(expr):
    with pytest.raises(isf.ISFError):
        isf.size_expr(expr, 640, 480, {"c": [1, 2]})


@pytest.mark.parametrize("inp, value, expected", [
    ({"TYPE": "float", "MIN": 0, "MAX": 2, "DEFAULT": 1}, 5, 2),
    ({"TYPE": "float", "MIN": 0, "MAX": 2, "DEFAULT": 1}, "x", 1),
    ({"TYPE": "float", "MIN": 0, "MAX": 2, "DEFAULT": 1}, float("nan"), 1),
    ({"TYPE": "float", "MIN": 2, "MAX": 0, "DEFAULT": 1}, -1, 0),         # MIN and MAX the wrong way round
    ({"TYPE": "bool", "DEFAULT": True}, None, True),
    ({"TYPE": "bool", "DEFAULT": True}, 0, False),
    ({"TYPE": "long", "VALUES": [0, 1, 2], "DEFAULT": 1}, 2, 2),
    ({"TYPE": "long", "VALUES": [0, 1, 2], "DEFAULT": 1}, 7, 1),
    ({"TYPE": "long", "MIN": 0, "MAX": 10, "DEFAULT": 3}, 12, 10),
    ({"TYPE": "color", "DEFAULT": [1, 0, 0, 1]}, [0.5, 2, -1], [0.5, 1.0, 0.0, 1.0]),
    ({"TYPE": "color", "DEFAULT": [1, 0, 0, 1]}, "red", [1.0, 0.0, 0.0, 1.0]),
    ({"TYPE": "point2D", "DEFAULT": [0.5, 0.5], "MIN": [0, 0], "MAX": [1, 1]}, [2, -1], [1.0, 0.0]),
    ({"TYPE": "point2D"}, None, [0.5, 0.5]),
])
def test_clean_value(inp, value, expected):
    assert isf.clean_value(inp, value) == expected


def test_fragment_source_declares_inputs_and_keeps_line_numbers():
    e = isf.parse("/*{\n" + '"INPUTS": [{"NAME": "inputImage", "TYPE": "image"},'
                  ' {"NAME": "k", "TYPE": "long"}, {"NAME": "c", "TYPE": "color"},'
                  ' {"NAME": "p", "TYPE": "point2D"}, {"NAME": "on", "TYPE": "bool"}]\n}*/\n'
                  "#version 120\nvoid main() { gl_FragColor = texture2D(inputImage, vv_FragNormCoord); }\n")
    glsl = isf.fragment_source(e)
    assert glsl.startswith("#version 330 core") and glsl.count("#version") == 1
    for decl in ("uniform int k;", "uniform vec4 c;", "uniform vec2 p;", "uniform bool on;"):
        assert decl in glsl
    # The header ends on line 3: the body's first line is numbered 3, so main() is line 5 as in the file.
    after = glsl.split("#line 3\n", 1)[1].split("\n")
    assert after[2].startswith("void main")


def test_compile_message_takes_the_compilers_error_line():
    class ShaderError(Exception):
        pass
    e = ShaderError("Shader compile failure (0): b\"0:5(12): error: syntax error, unexpected ';'\\n\"",
                    [b"#version 330 core ..."], 35632)
    assert isf.compile_message(e) == "0:5(12): error: syntax error, unexpected ';'"


@pytest.mark.parametrize("log,expected", [
    # The owner's AMD driver (2026-11-14) starts with this header line; the rest as recalled for AMD.
    ("Fragment shader failed to compile with the following errors:\n"
     "ERROR: 0:2: error(#202) No matching overloaded function found: not_a_function\n"
     "ERROR: error(#273) 1 compilation errors.  No code generated\n",
     "0:2: error(#202) No matching overloaded function found: not_a_function"),
    ("0(2) : error C1008: undefined variable \"not_a_function\"\n", "0(2) : error C1008: undefined variable \"not_a_function\""),
    ("Fragment shader failed to compile with the following errors:\nERROR: error(#273) 1 compilation errors.\n",
     "Fragment shader failed to compile with the following errors:"),
])
def test_compile_message_names_the_line_with_any_drivers_wording(log, expected):
    class ShaderError(Exception):
        pass
    e = ShaderError(f"Shader compile failure (0): {log.encode()!r}", [b"#version 330 core ..."], 35632)
    assert isf.compile_message(e) == expected


def test_clean_saved_value():
    assert isf.clean(None) is None and isf.clean({"name": " "}) is None and isf.clean({"name": 3}) is None
    assert isf.clean({"name": "Invert.fs", "inputs": {"amount": 1}}) == {"name": "Invert.fs", "inputs": {"amount": 1}}
    assert isf.clean({"name": "Invert.fs", "inputs": "x"}) == {"name": "Invert.fs", "inputs": {}}


def test_builtin_effects_parse_and_have_titles():
    found = isf.available()
    assert {"Invert.fs", "Pixelate.fs", "Mirror.fs", "RGB Split.fs", "Kaleidoscope.fs", "Wave.fs", "Tint.fs"} <= set(found)
    for name in found:
        e = isf.load(name)
        assert e["description"], name
        assert isf.title(name) == name[:-3]
        for inp in e["inputs"]:
            assert isf.clean_value(inp, inp["DEFAULT"]) == inp["DEFAULT"], (name, inp["NAME"])


def test_user_folder_overrides_builtin(tmp_path, monkeypatch):
    monkeypatch.setattr(isf, "USER_DIR", str(tmp_path))
    (tmp_path / "Invert.fs").write_text(src(DESCRIPTION="mine"))
    (tmp_path / "notes.txt").write_text("not an effect")
    found = isf.available()
    assert found["Invert.fs"] == str(tmp_path / "Invert.fs") and "notes.txt" not in found
    assert isf.load("Invert.fs")["description"] == "mine"


def test_animated_effects():
    isf.forget()
    assert isf.animated("Wave.fs") and not isf.animated("Invert.fs")
    assert not isf.animated("No Such Effect.fs")


def test_layer_effects_read_the_saved_values():
    import render_engine
    assert render_engine.layer_effects({}) == []
    assert render_engine.layer_effects({"effect": {"name": "Wave.fs"}}) == [{"name": "Wave.fs", "inputs": {}}]
    stack = [{"name": "A.fs"}, {"name": " "}, {"name": "B.fs", "inputs": {"x": 1}}]
    assert render_engine.layer_effects({"effects": stack, "effect": {"name": "Old.fs"}}) == \
        [{"name": "A.fs", "inputs": {}}, {"name": "B.fs", "inputs": {"x": 1}}]       # the list wins
    assert render_engine.layer_effects({"effects": [], "effect": {"name": "Old.fs"}}) == []
    assert len(isf.stack([{"name": f"{i}.fs"} for i in range(9)])) == isf.MAX_STACK


def test_pick_an_input_in_a_stack():
    st = [{"name": "Invert.fs", "inputs": {}}, {"name": "Tint.fs", "inputs": {}}, {"name": "Invert.fs", "inputs": {}}]
    assert isf.pick(st, "amount")[0] == 0                      # the first effect that has it
    assert isf.pick(st, "tint")[0] == 1
    assert isf.pick(st, "3:amount")[0] == 2 and isf.pick(st, "2:amount")[0] == 1
    assert isf.pick(st, "Tint:Amount")[0] == 1 and isf.pick(st, "invert.fs:amount")[0] == 0
    assert isf.pick(st, "4:amount") == (None, None, None) and isf.pick(st, "Wave:amount") == (None, None, None)


def test_effects_reach_the_output_layers():
    """_get_layers_for_col copies the layer's effect from the show data."""
    import inspect
    import control_interface
    code = inspect.getsource(control_interface.ControlInterface._get_layers_for_col)
    assert "'effect'" in code and "'effects'" in code


def test_editor_effect_controls(tmp_path):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    from types import SimpleNamespace
    from PIL import Image
    from editor import MediaEditor
    p = str(tmp_path / "w.png")
    Image.new("RGB", (400, 200), (255, 255, 255)).save(p)
    root = tk.Tk()
    root.withdraw()
    sent = []
    eng = SimpleNamespace(monitors=[], draws_effects=True, effect_errors={})
    try:
        ed = MediaEditor(SimpleNamespace(root=root), p, None, eng, initial_pos={},
                         realtime_callback=lambda pos: sent.append(dict(pos)))
        st = ed.effect_ui
        assert st.items == [] and "effects" not in ed.current_pos
        assert {"Invert", "Kaleidoscope", "Mirror"} <= set(st.add_box.cget("values"))
        st.add_var.set("Kaleidoscope")
        st._add_chosen()
        assert ed.current_pos["effects"] == [{"name": "Kaleidoscope.fs",
                                              "inputs": {"sides": 6.0, "turn": 0.0, "center": [0.5, 0.5]}}]
        fx = st.items[0]
        assert len(fx.inputs_frame.winfo_children()) == 3        # sides, turn, center
        fx.set_value("sides", 8.0)
        ed._flush_realtime_callback()
        assert sent[-1]["effects"][0]["inputs"]["sides"] == 8.0
        assert "effects" not in ed.original_pos, "Cancel must restore the layer as it was"
        st.add_var.set("Mirror")                                  # a second effect, after the first
        st._add_chosen()
        assert [f["name"] for f in ed.current_pos["effects"]] == ["Kaleidoscope.fs", "Mirror.fs"]
        assert ed.current_pos["effects"][0]["inputs"]["sides"] == 8.0
        boxes = [w for row in st.items[1].inputs_frame.winfo_children() for w in row.winfo_children()
                 if w.winfo_class() == "TCombobox"]
        assert boxes and "Top half" in boxes[0].cget("values")    # a long input with labels: a list
        st.move(1, -1)
        assert [f["name"] for f in ed.current_pos["effects"]] == ["Mirror.fs", "Kaleidoscope.fs"]
        # The output could not compile it: the editor says so.
        eng.effect_errors["Mirror.fs"] = "0:3(1): error: syntax error"
        st.items[0].check_error("Mirror.fs")
        assert "syntax error" in st.items[0].note.cget("text")
        st.items[0].var.set("None")                               # None in an effect's list takes it out
        st.items[0].choose()
        assert [f["name"] for f in ed.current_pos["effects"]] == ["Kaleidoscope.fs"]
        st.remove(0)
        assert "effects" not in ed.current_pos and "effect" not in ed.current_pos
        for name in ("Invert", "Tint", "Wave", "Pixelate"):
            st.add_var.set(name)
            st._add_chosen()
        assert len(ed.current_pos["effects"]) == 4 and str(st.add_box.cget("state")) == "disabled"
        ed.window.destroy()
        # An older show's single effect opens as a list of one; saved values come back.
        ed = MediaEditor(SimpleNamespace(root=root), p, None, SimpleNamespace(monitors=[]),
                         initial_pos={"effect": {"name": "Invert.fs", "inputs": {"amount": 0.25}}})
        assert ed.effect_ui.items[0].var.get() == "Invert" and ed.effect_ui.items[0].values == {"amount": 0.25}
        assert "need Performance mode" in ed.effect_ui.empty_note.cget("text")   # this engine draws none
        ed.effect_ui.items[0].set_value("amount", 0.5)
        assert ed.current_pos["effects"] == [{"name": "Invert.fs", "inputs": {"amount": 0.5}}]
        assert "effect" not in ed.current_pos
        ed.window.destroy()
        ed = MediaEditor(SimpleNamespace(root=root), p, None, eng,
                         initial_pos={"effects": [{"name": "Gone.fs", "inputs": {"x": 1}}]})
        item = ed.effect_ui.items[0]
        assert item.var.get() == "Gone (missing)" and "not found" in item.note.cget("text")
        assert ed.effect_ui.value() == [{"name": "Gone.fs", "inputs": {"x": 1}}], "kept until changed"
        ed.window.destroy()
    finally:
        root.destroy()


def test_changed_reports_written_effect_files(tmp_path, monkeypatch):
    monkeypatch.setattr(isf, "USER_DIR", str(tmp_path))
    monkeypatch.setattr(isf, "_mtimes", {})
    monkeypatch.setattr(isf.changed, "_called", False, raising=False)
    assert isf.changed() == set()                          # the first call only takes note
    f = tmp_path / "Mine.fs"
    f.write_text(src())
    assert isf.changed() == {"Mine.fs"}
    assert isf.changed() == set()
    os.utime(f, (1, 1))
    assert isf.changed() == {"Mine.fs"}
    f.unlink()
    assert isf.changed() == {"Mine.fs"}


from test_blind_editing import ci as blind_ci  # noqa: E402,F401
from test_cue_activation import ci as _base_ci, commands  # noqa: E402,F401  (fixture reuse)


def test_display_effects_show_data_commands_and_window(blind_ci, tmp_path):
    """Window > Output Effects: saved with the show, sent to the render
    thread at once (also with blind editing), and the window's controls."""
    import json
    tk = pytest.importorskip("tkinter")
    from types import SimpleNamespace
    ci = blind_ci
    for name in ("_build_media_list", "_build_grid", "_initialize_render_engine", "_on_toggle_presentation_mode"):
        setattr(ci, name, lambda *a, **k: None)
    commands(ci)
    ci._set_output_effect(1, {"name": "Tint.fs", "inputs": {"amount": 0.5}})
    sent = commands(ci, "set_output_effects")
    assert sent[-1]["effects"] == {"1": [{"name": "Tint.fs", "inputs": {"amount": 0.5}}]}
    show = str(tmp_path / "s.vjdeck")
    ci._perform_save(show)
    data = json.load(open(show))
    assert data["outputs"]["effects"]["1"][0]["name"] == "Tint.fs"
    data["outputs"]["effects"]["2"] = {"name": " "}                     # junk is dropped on load
    ci._apply_project_data(data, show)
    assert ci.output_effects == {"1": [{"name": "Tint.fs", "inputs": {"amount": 0.5}}]}
    ci._set_output_effect(1, None)
    assert ci.output_effects == {} and commands(ci, "set_output_effects")[-1]["effects"] == {}
    if not os.environ.get("DISPLAY") and os.name != "nt":
        return
    root = tk.Tk()
    root.withdraw()
    try:
        ci.root = root
        ci.re.monitors = [SimpleNamespace(x=0, y=0, width=1920, height=1080),
                          SimpleNamespace(x=1920, y=0, width=1280, height=720)]
        win = ci._open_output_effects(1)
        st = win._vd["state"]["controls"]
        st.add_var.set("Invert")
        st._add_chosen()
        assert ci.output_effects == {"1": [{"name": "Invert.fs", "inputs": {"amount": 1.0}}]}
        st.items[0].set_value("amount", 0.25)
        assert ci.output_effects["1"][0]["inputs"]["amount"] == 0.25
        assert commands(ci, "set_output_effects")[-1]["effects"]["1"][0]["inputs"]["amount"] == 0.25
        win._vd["show"](0)                                              # the other display: none yet
        assert win._vd["state"]["controls"].items == []
        win._vd["show"](1)
        assert win._vd["state"]["controls"].items[0].var.get() == "Invert"
        win.destroy()
    finally:
        root.destroy()


def test_render_thread_redraws_for_a_moving_display_effect():
    """set_output_effects reaches the engine; a moving one (Wave) counts like a video."""
    import render_engine as rem
    eng = rem.BaseRenderEngine(media_manager=None, mode="opengl")
    eng.set_output_effects({"0": {"name": "Invert.fs"}, "x": {"name": "Wave.fs"}, 1: None})
    assert eng.output_effects == {0: [{"name": "Invert.fs", "inputs": {}}]} and not eng.output_effects_moving
    eng.set_output_effects({0: {"name": "Wave.fs"}})
    assert eng.output_effects_moving


def test_editor_canvas_fits_the_screen():
    """The position editor's canvas is up to 60 % of the output's size, or
    smaller so the window (canvas and the settings on the right) fits on the screen."""
    from types import SimpleNamespace
    from editor import MediaEditor
    ed = MediaEditor.__new__(MediaEditor)

    def scale(screen, output):
        ed.screen_width, ed.screen_height = output
        root = SimpleNamespace(winfo_screenwidth=lambda: screen[0], winfo_screenheight=lambda: screen[1])
        return ed._fit_canvas_scale(SimpleNamespace(root=root))
    assert scale((1920, 1080), (1920, 1080)) == MediaEditor.MAX_CANVAS_SCALE == 0.6  # settings on the right only
    assert 1920 * 0.6 + MediaEditor.CONTROLS_WIDTH <= 1920 and 1080 * 0.6 + MediaEditor.CHROME_HEIGHT <= 1080
    s = scale((1920, 1080), (3840, 2160))                                 # a 4K output on a 1080p screen
    assert 3840 * s + MediaEditor.CONTROLS_WIDTH <= 1920 and 2160 * s + MediaEditor.CHROME_HEIGHT <= 1080
    s = scale((1366, 768), (1920, 1080))
    assert 1920 * s + MediaEditor.CONTROLS_WIDTH <= 1366 and s < 0.5
    assert ed._fit_canvas_scale(SimpleNamespace()) == 0.5                 # no screen size known


def test_output_health_reports_effects(blind_ci):
    """Output Health: how many effects are on the output, and a warning for one
    the output could not compile (e.g. a graphics driver that refuses it)."""
    import health
    from types import SimpleNamespace
    import control_interface as ci_mod
    rt = ci_mod.RenderThread.__new__(ci_mod.RenderThread)
    rt.re = SimpleNamespace(output_effects={0: [{"name": "Wave.fs", "inputs": {}}]}, draws_effects=True,
                            effect_errors={"Bad.fs": "0:3(1): error: syntax error", "Other.fs": "not used"})
    rt.current_render_state = {"layers": {0: {"effects": [{"name": "Invert.fs"}, {"name": "Bad.fs"}]}}}
    st = rt._effects_status()
    assert st == {"count": 3, "drawn": True, "errors": {"Bad.fs": "0:3(1): error: syntax error"}}
    level, reasons = health.assess({"effects": st})
    assert level == "warn" and any("Bad.fs not drawn" in r for r in reasons)
    assert "Bad not drawn: 0:3(1)" in ci_mod.ControlInterface._effects_text(st)
    rt.re.draws_effects = False
    st = rt._effects_status()
    assert "need Performance Mode" in " ".join(health.assess({"effects": st})[1])
    rt.current_render_state, rt.re.output_effects = {"layers": {}}, {}
    assert rt._effects_status() == {"count": 0} and health.assess({"effects": {"count": 0}})[0] == "ok"
    assert ci_mod.ControlInterface._effects_text({}) == "none on the output"


def test_kaleidoscope_mirrors_past_the_edge():
    """The far corners of a wide picture sample outside it: mirrored, not the
    edge row smeared out (pale corners on the owner's test image, 2026-11-14)."""
    src = open(os.path.join(os.path.dirname(isf.__file__), "effects", "Kaleidoscope.fs")).read()
    assert "clamp(center + q" not in src and "mod(s, 2.0)" in src


def test_the_shows_effects_are_sent_for_warm_up(blind_ci):
    """Opening a show or starting the output asks the render thread to compile
    every effect the show uses (layers and displays) before its first draw."""
    ci = blind_ci
    ci.cell_positions[(0, 0)] = {"effects": [{"name": "Wave.fs", "inputs": {}}, {"name": "Invert.fs"}]}
    ci.cell_positions[(1, 2)] = {"effect": {"name": "Wave.fs", "inputs": {}}}        # an older single effect
    ci.output_effects = {"0": [{"name": "Tint.fs", "inputs": {}}]}
    assert ci._show_effect_names() == ["Wave.fs", "Invert.fs", "Tint.fs"]
    ci._warm_effects()
    sent = [c for c in commands(ci) if c["action"] == "fx_warm"]
    assert sent == [{"action": "fx_warm", "names": ["Wave.fs", "Invert.fs", "Tint.fs"]}]
