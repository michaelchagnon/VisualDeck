# SPDX-License-Identifier: MPL-2.0
"""Shader effects in the ISF format (Interactive Shader Format), per layer and per display.

An ISF file is a GLSL fragment shader that starts with a JSON comment
describing it: {"DESCRIPTION", "CATEGORIES", "INPUTS": [{"NAME", "TYPE",
"DEFAULT", "MIN", "MAX", "LABEL", "VALUES", "LABELS"}], ...}. VisualDeck runs
filters: the shader gets the layer's picture as an image input named
"inputImage" and draws a new picture of the same size, which is then drawn
as the layer (with its position, opacity, mask and blend mode).

Supported: inputs of type float, bool, long, color, point2D (and the image
input "inputImage"); the uniforms TIME, TIMEDELTA, FRAMEINDEX, RENDERSIZE,
DATE, PASSINDEX; isf_FragNormCoord (and the older vv_FragNormCoord); the
functions IMG_THIS_PIXEL, IMG_THIS_NORM_PIXEL, IMG_NORM_PIXEL, IMG_PIXEL,
IMG_SIZE; gl_FragColor and texture2D; several passes ("PASSES") with
TARGET buffers, PERSISTENT ones (kept from frame to frame, starting
transparent black), FLOAT ones (32-bit), and WIDTH / HEIGHT expressions
($WIDTH, $HEIGHT, $<input>; size_expr). The last pass is the result. A pass
may read the buffer it draws into (it reads the previous contents). Not
supported (the effect is refused with a reason): extra image or audio
inputs, imported images, a separate vertex shader.
In a layer's effect gl_FragCoord is in the picture's pixels with y going
down, which differs from ISF hosts (in a display's effect it goes up, as
there); isf_FragNormCoord is as ISF says (0, 0 at the bottom left).

Effects are looked up in the effects folder that comes with VisualDeck
(src/effects), in ~/.visualdeck/effects and in the open show's "effects"
folder (Collect Show copies the show's own effects there). Saved with the layer as
"effect": {"name": file name, "inputs": {NAME: value}}.

Only Performance Mode (OpenGL) draws effects. No GL here (render_engine.py
compiles and runs them).
"""
from __future__ import annotations

import json
import os
import re

BUILTIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "effects")
USER_DIR = os.path.join(os.path.expanduser("~"), ".visualdeck", "effects")
INPUT_TYPES = ("float", "bool", "long", "color", "point2D")
_HEADER = re.compile(r"^\s*/\*(.*?)\*/", re.S)


class ISFError(ValueError):
    pass


def parse(text, name="effect"):
    """{'name', 'description', 'categories', 'inputs': [input dicts with 'NAME',
    'TYPE', 'DEFAULT', 'MIN', 'MAX', 'LABEL', ('VALUES', 'LABELS')], 'body': the
    GLSL after the header}; raises ISFError for what VisualDeck cannot run."""
    m = _HEADER.match(text or "")
    if not m:
        raise ISFError("no ISF header (a /*{ ... }*/ JSON comment at the top)")
    try:
        meta = json.loads(m.group(1))
    except ValueError as e:
        raise ISFError(f"the header is not valid JSON: {e}") from None
    if not isinstance(meta, dict):
        raise ISFError("the header is not a JSON object")
    if meta.get("IMPORTED"):
        raise ISFError("imported images are not supported")
    inputs, has_image = [], False
    for raw in meta.get("INPUTS") or []:
        if not isinstance(raw, dict) or not isinstance(raw.get("NAME"), str):
            raise ISFError("an input without a NAME")
        t = raw.get("TYPE")
        if t == "image" and raw["NAME"] == "inputImage":
            has_image = True
            continue
        if t not in INPUT_TYPES:
            raise ISFError(f"input {raw['NAME']}: type {t!r} is not supported")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw["NAME"]):
            raise ISFError(f"input name {raw['NAME']!r} is not a GLSL name")
        inp = dict(raw)
        inp["DEFAULT"] = clean_value(inp, raw.get("DEFAULT"))
        inputs.append(inp)
    if not has_image:
        raise ISFError("not a filter: it has no image input named inputImage")
    passes = _passes(meta.get("PASSES"), {i["NAME"] for i in inputs})
    return {"name": name, "description": str(meta.get("DESCRIPTION") or ""),
            "categories": [str(c) for c in meta.get("CATEGORIES") or []], "inputs": inputs,
            "passes": passes, "targets": list(dict.fromkeys(p["target"] for p in passes if p["target"])),
            "body": text[m.end():], "body_line": text.count("\n", 0, m.end()) + 1}


_RESERVED = {"inputImage", "RENDERSIZE", "TIME", "TIMEDELTA", "FRAMEINDEX", "PASSINDEX", "DATE"}


def _passes(raw, input_names):
    """[{'target': name or None, 'persistent', 'float', 'width', 'height'}]: the
    rendering passes in order (one without a target if the file has none).
    Each pass renders the whole shader with PASSINDEX set; a pass with a
    TARGET draws into that buffer, which later passes (and, if PERSISTENT, the
    next frame) read by its name. The last pass is the effect's result."""
    if raw in (None, [], {}):
        return [{"target": None, "persistent": False, "float": False, "width": None, "height": None}]
    if not isinstance(raw, list) or not all(isinstance(p, dict) for p in raw):
        raise ISFError("PASSES is not a list of passes")
    out, kinds = [], {}
    for p in raw:
        target = p.get("TARGET")
        if target in ("", None):
            target = None
        elif not isinstance(target, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target):
            raise ISFError(f"pass target {target!r} is not a GLSL name")
        elif target in _RESERVED or target in input_names:
            raise ISFError(f"pass target {target!r} has the name of an input or a built-in")
        persistent, flt = bool(p.get("PERSISTENT")), bool(p.get("FLOAT"))
        if target is not None:
            if target in kinds and kinds[target] != (persistent, flt):
                raise ISFError(f"pass target {target!r} is declared twice differently")
            kinds[target] = (persistent, flt)
        size = []
        for key in ("WIDTH", "HEIGHT"):
            v = p.get(key)
            if v is not None and not isinstance(v, (str, int, float)) or isinstance(v, bool):
                raise ISFError(f"pass {key} {v!r} is not a number or an expression")
            if isinstance(v, str):
                try:
                    size_expr(v, 640, 480, {})                  # a syntax check; inputs are checked when used
                except ISFError:
                    try:
                        size_expr(v, 640, 480, {n: 1.0 for n in input_names})
                    except ISFError as e:
                        raise ISFError(f"pass {key} {v!r}: {e}") from None
            size.append(v)
        out.append({"target": target, "persistent": persistent, "float": flt, "width": size[0], "height": size[1]})
    return out


def size_expr(expr, width, height, values):
    """A pass WIDTH or HEIGHT: a number, or an expression with $WIDTH,
    $HEIGHT, $<input name> (a number input's value), + - * / % ** and
    parentheses, and floor, ceil, round, abs, min, max, sqrt, pow, log, exp.
    Raises ISFError for anything else."""
    import ast
    import math
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return float(expr)
    names = {"WIDTH": float(width), "HEIGHT": float(height)}
    for k, v in (values or {}).items():
        if isinstance(v, bool):
            names[k] = float(v)
        elif isinstance(v, (int, float)):
            names[k] = float(v)
    funcs = {"floor": math.floor, "ceil": math.ceil, "round": round, "abs": abs, "min": min, "max": max,
             "sqrt": math.sqrt, "pow": pow, "log": math.log, "exp": math.exp}
    text = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"vd_\1", str(expr))
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ISFError("not an expression") from None
    ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.Mod: lambda a, b: a % b, ast.Pow: lambda a, b: a ** b}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id.startswith("vd_"):
            if n.id[3:] not in names:
                raise ISFError(f"${n.id[3:]} is not $WIDTH, $HEIGHT or a number input")
            return names[n.id[3:]]
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 64:
                raise ISFError("too large a power")
            return ops[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs and not n.keywords:
            return float(funcs[n.func.id](*[ev(a) for a in n.args]))
        raise ISFError("only numbers, $WIDTH, $HEIGHT, $inputs, + - * / % ** and simple functions")
    try:
        v = ev(tree)
    except ISFError:
        raise
    except (ZeroDivisionError, ValueError, OverflowError, TypeError):
        raise ISFError("cannot be worked out") from None
    if v != v or v in (float("inf"), float("-inf")):
        raise ISFError("cannot be worked out")
    return v


def _num(v, default=0.0):
    try:
        f = float(v)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def clean_value(inp, value):
    """A value for this input, within MIN/MAX, of the right shape (the
    input's DEFAULT, or a neutral value, where it is missing or wrong)."""
    t = inp.get("TYPE")
    if t == "float":
        lo, hi = _num(inp.get("MIN"), 0.0), _num(inp.get("MAX"), 1.0)
        if hi < lo:
            lo, hi = hi, lo
        d = _num(inp.get("DEFAULT"), lo)
        return min(hi, max(lo, _num(value, d)))
    if t == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return bool(inp.get("DEFAULT", False)) if not isinstance(inp.get("DEFAULT"), str) else False
    if t == "long":
        values = inp.get("VALUES")
        d = inp.get("DEFAULT", values[0] if values else 0)
        try:
            v = int(value)
        except (TypeError, ValueError):
            v = int(_num(d, 0))
        if values:
            return v if v in values else (int(_num(d, values[0])) if _num(d, values[0]) in values else values[0])
        lo, hi = int(_num(inp.get("MIN"), 0)), int(_num(inp.get("MAX"), 100))
        return min(hi, max(lo, v))
    if t == "color":
        d = inp.get("DEFAULT") if isinstance(inp.get("DEFAULT"), list) else [1.0, 1.0, 1.0, 1.0]
        v = value if isinstance(value, (list, tuple)) and len(value) in (3, 4) else d
        v = [min(1.0, max(0.0, _num(c, 1.0))) for c in v] + [1.0] * (4 - len(v))
        return v[:4]
    if t == "point2D":
        d = inp.get("DEFAULT") if isinstance(inp.get("DEFAULT"), list) and len(inp["DEFAULT"]) == 2 else [0.5, 0.5]
        v = value if isinstance(value, (list, tuple)) and len(value) == 2 else d
        lo = inp.get("MIN") if isinstance(inp.get("MIN"), list) and len(inp["MIN"]) == 2 else None
        hi = inp.get("MAX") if isinstance(inp.get("MAX"), list) and len(inp["MAX"]) == 2 else None
        out = [_num(c, dc) for c, dc in zip(v, d)]
        if lo and hi:
            out = [min(_num(h, o), max(_num(l, o), o)) for o, l, h in zip(out, lo, hi)]
        return out
    return value


_UNIFORM_TYPE = {"float": "float", "bool": "bool", "long": "int", "color": "vec4", "point2D": "vec2"}

FRAGMENT_PRELUDE = """#version 330 core
uniform sampler2D inputImage;
uniform vec2 RENDERSIZE;
uniform float TIME;
uniform float TIMEDELTA;
uniform int FRAMEINDEX;
uniform int PASSINDEX;
uniform vec4 DATE;
uniform bool vd_bottomUp;
uniform bool vd_premultiplied;
uniform bool vd_finalPass;
in vec2 isf_FragNormCoord;
out vec4 vd_FragColor;
#define gl_FragColor vd_FragColor
#define vv_FragNormCoord isf_FragNormCoord
#define texture2D vd_texture
#define texture2DRect vd_texture
#define varying in
// A layer's texture has its top row first (an output's picture, drawn by GL,
// its bottom row: vd_bottomUp); ISF images have (0, 0) at the bottom left.
// Every image here has plain (not premultiplied) colors; a display's picture
// is turned into plain colors before the effect and its result multiplied
// back by its coverage after the last pass (vd_premultiplied, vd_finalPass),
// so what no layer covers stays black whatever the effect does to black.
vec4 vd_texture(sampler2D s, vec2 c) { return texture(s, vec2(c.x, vd_bottomUp ? c.y : 1.0 - c.y)); }
#define IMG_NORM_PIXEL(img, c) vd_texture(img, c)
#define IMG_THIS_NORM_PIXEL(img) vd_texture(img, isf_FragNormCoord)
#define IMG_THIS_PIXEL(img) vd_texture(img, isf_FragNormCoord)
#define IMG_PIXEL(img, c) vd_texture(img, (c) / RENDERSIZE)
#define IMG_SIZE(img) vec2(textureSize(img, 0))
"""

FRAGMENT_EPILOGUE = """
#undef main
void main() {
    vd_isf_main();
    if (vd_premultiplied && vd_finalPass) vd_FragColor.rgb *= clamp(vd_FragColor.a, 0.0, 1.0);
}
"""

VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec2 pos;
uniform bool vd_bottomUp;
out vec2 isf_FragNormCoord;
void main() {
    // Row 0 of the result is the top of the picture, as in the layer textures
    // (or its bottom, for an output's picture: vd_bottomUp).
    isf_FragNormCoord = vec2(pos.x * 0.5 + 0.5, vd_bottomUp ? pos.y * 0.5 + 0.5 : 0.5 - pos.y * 0.5);
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""


def fragment_source(effect):
    """GLSL 3.30 source for an effect from parse()."""
    lines = [FRAGMENT_PRELUDE]
    for inp in effect["inputs"]:
        lines.append(f"uniform {_UNIFORM_TYPE[inp['TYPE']]} {inp['NAME']};")
    for target in effect.get("targets", []):
        lines.append(f"uniform sampler2D {target};")
    # Ours comes first; a #version line of the file's becomes an empty line (the numbering stays).
    body = re.sub(r"^[ \t]*#version[^\n]*", "", effect["body"], flags=re.M)
    lines.append("#define main vd_isf_main")
    lines.append(f"#line {effect.get('body_line', 1)}")   # compiler messages give the file's line numbers
    lines.append(body)
    lines.append(FRAGMENT_EPILOGUE)
    return "\n".join(lines)


def compile_message(error):
    """The compiler's first error line from a PyOpenGL shader exception (its
    text otherwise holds the whole source): '0:12(5): error: ...'."""
    text = error.args[0] if getattr(error, "args", None) and isinstance(error.args[0], str) else str(error)
    m = re.search(r"b(['\"])(.*?)\1(?:,|$)", text, re.S)
    if m:
        try:
            text = m.group(2).encode("latin-1", "backslashreplace").decode("unicode_escape")
        except UnicodeError:
            text = m.group(2)
    else:
        text = re.sub(r"^Shader compile failure \(\d+\):\s*", "", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:                       # a line that says where: "0:12(5): error", "ERROR: 0:2: ...", "0(2) : error"
        m = _LOCATED.match(ln)
        if m and "error" in ln.lower():
            return m.group(1)[:300]
    for ln in lines:                       # AMD starts with "Fragment shader failed to compile with the following errors:"
        low = ln.lower()
        if "error" in low and "following error" not in low and "compilation error" not in low:
            return ln[:300]
    return (lines[0] if lines else type(error).__name__)[:300]


_LOCATED = re.compile(r"^(?:ERROR:\s*)?(\d+\s*[:(]\s*\d+.*)$", re.I)


SHOW_DIR = None        # the open show's "effects" folder (Collect Show copies effects there)


def set_show_folder(folder):
    """Also look for effects in <folder>/effects (the open show's folder), or not (None)."""
    global SHOW_DIR
    new = os.path.join(folder, "effects") if folder else None
    if new != SHOW_DIR:
        SHOW_DIR = new
        forget()


def search_dirs():
    """Later folders win: built-in, then the user's, then the show's own (a
    collected show plays with the effects it was made with)."""
    return [BUILTIN_DIR, USER_DIR] + ([SHOW_DIR] if SHOW_DIR else [])


def custom_files(names):
    """{name: path} of the effects in `names` that are not built-in (found in
    the user's or the show's folder): what Collect Show copies."""
    found = available()
    out = {}
    for n in names:
        p = found.get(n)
        if p and os.path.dirname(os.path.abspath(p)) != os.path.abspath(BUILTIN_DIR):
            out[n] = p
    return out


def available():
    """{file name: path} of the .fs files found (search_dirs: later folders override earlier ones)."""
    out = {}
    for d in search_dirs():
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for n in names:
            if n.lower().endswith((".fs", ".frag")):
                out[n] = os.path.join(d, n)
    return out


def load(name):
    """parse() of an effect by file name (or path); raises ISFError / OSError."""
    path = name if os.path.isabs(name) else available().get(name)
    if not path:
        raise ISFError(f"effect {name!r} not found")
    with open(path, encoding="utf-8") as f:
        return parse(f.read(), os.path.basename(path))


_ANIMATED = re.compile(r"\b(TIME|TIMEDELTA|FRAMEINDEX|DATE)\b")
_animated_cache = {}


def animated(name):
    """True if the effect changes from frame to frame by itself (it uses
    TIME, TIMEDELTA, FRAMEINDEX or DATE, or keeps a PERSISTENT buffer), so a
    layer with it must be redrawn every frame even when its picture is still.
    Cached per file name; an effect that cannot be loaded counts as still."""
    if name not in _animated_cache:
        try:
            e = load(name)
            _animated_cache[name] = bool(_ANIMATED.search(e["body"])) or any(p["persistent"] for p in e["passes"])
        except (ISFError, OSError):
            _animated_cache[name] = False
    return _animated_cache[name]


def forget(names=None):
    """Drop cached facts about effect files (all, or these names)."""
    if names is None:
        _animated_cache.clear()
        _inputs_cache.clear()
    else:
        for n in names:
            _animated_cache.pop(n, None)
            _inputs_cache.pop(n, None)


_inputs_cache = {}


def inputs_of(effect_name):
    """The input dicts of an effect (cached like find_input); [] if it cannot be loaded."""
    find_input(effect_name, "")
    return list(_inputs_cache.get(effect_name, []))


def find_input(effect_name, input_name):
    """The input dict of an effect's input, by NAME or LABEL (any case); None
    if the effect cannot be loaded or has no such input. Cached per effect."""
    if effect_name not in _inputs_cache:
        try:
            _inputs_cache[effect_name] = load(effect_name)["inputs"]
        except (ISFError, OSError):
            _inputs_cache[effect_name] = []
    want = str(input_name).strip().lower()
    for inp in _inputs_cache[effect_name]:
        if inp["NAME"].lower() == want:
            return inp
    for inp in _inputs_cache[effect_name]:
        if str(inp.get("LABEL") or "").strip().lower() == want:
            return inp
    return None


def fader_value(inp, v):
    """The value for `inp` at a fader's position v (0-127, as MIDI Control
    Change sends): across MIN..MAX for a slider, one of the values for a
    list, on from 64 for on/off. None for a color or a point (one fader
    cannot set them)."""
    t = inp.get("TYPE")
    f = min(127, max(0, int(v))) / 127.0
    if t == "bool":
        return int(v) >= 64
    if t == "long" and inp.get("VALUES"):
        vals = inp["VALUES"]
        return vals[min(len(vals) - 1, int(f * len(vals)))]
    if t in ("float", "long"):
        lo, hi = _num(inp.get("MIN"), 0.0), _num(inp.get("MAX"), 1.0 if t == "float" else 100)
        x = lo + (hi - lo) * f
        return clean_value(inp, int(round(x)) if t == "long" else x)
    return None


def remote_value(inp, values):
    """A value for `inp` from the numbers a remote control sent (OSC or the
    browser API): one number for float, long and bool (not 0 is on), three or
    four for color (0-1), two for point2D. Cleaned like any value (within
    MIN/MAX); None if the count does not fit."""
    t = inp.get("TYPE")
    vals = list(values or [])
    if t in ("float", "long", "bool"):
        if len(vals) != 1:
            return None
        v = vals[0]
        v = bool(v) if t == "bool" else (int(round(float(v))) if t == "long" else float(v))
    elif t == "color":
        if len(vals) not in (3, 4):
            return None
        v = [float(x) for x in vals]
    elif t == "point2D":
        if len(vals) != 2:
            return None
        v = [float(x) for x in vals]
    else:
        return None
    if t == "long" and inp.get("VALUES") and v not in inp["VALUES"]:
        return None                          # a list: only its values
    return clean_value(inp, v)


_mtimes = {}


def changed():
    """Names of effect files added, changed or removed since the last call
    (the first call only takes note). The editor calls it when it opens, so
    an effect being written is picked up without restarting."""
    now = {}
    for name, path in available().items():
        try:
            now[name] = (path, os.path.getmtime(path))
        except OSError:
            continue
    first = not _mtimes and not getattr(changed, "_called", False)
    changed._called = True
    diff = set() if first else {n for n in set(now) | set(_mtimes) if now.get(n) != _mtimes.get(n)}
    _mtimes.clear()
    _mtimes.update(now)
    return diff


MAX_STACK = 4          # effects on one layer or display, applied in order


def stack(raw):
    """[clean() values] from a saved list of effects, or from one saved effect
    (shows from before effects could be stacked); at most MAX_STACK."""
    items = raw if isinstance(raw, list) else [raw]
    return [c for c in (clean(x) for x in items) if c is not None][:MAX_STACK]


def layer_stack(pos):
    """A layer's effects in order: its 'effects' list when it has one (an empty
    list means none), else its older single 'effect'."""
    if not isinstance(pos, dict):
        return []
    return stack(pos['effects']) if 'effects' in pos else stack(pos.get('effect'))


def pick(effects, selector):
    """(index, effect, input) for a remote control's input name in a stack:
    "amount" is the first effect that has that input; "2:amount" the second
    effect; "Invert:amount" the first Invert. (None, None, None) if none fits."""
    sel = str(selector).strip()
    which, name = (sel.split(":", 1) + [None])[:2] if ":" in sel else (None, sel)
    for i, fx in enumerate(effects):
        if which is not None:
            w = which.strip()
            if w.isdigit():
                if int(w) != i + 1:
                    continue
            elif w.lower() not in (title(fx).lower(), fx["name"].lower()):
                continue
        inp = find_input(fx["name"], name)
        if inp is not None:
            return i, fx, inp
    return None, None, None


def clean(raw):
    """The saved layer value: {'name': str, 'inputs': {NAME: value}} or None."""
    if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not raw["name"].strip():
        return None
    inputs = raw.get("inputs") if isinstance(raw.get("inputs"), dict) else {}
    return {"name": raw["name"].strip(), "inputs": {str(k): v for k, v in inputs.items()}}


def title(effect_or_name):
    """'Pixelate' from 'Pixelate.fs'."""
    name = effect_or_name["name"] if isinstance(effect_or_name, dict) else str(effect_or_name)
    return os.path.splitext(name)[0]
