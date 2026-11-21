# SPDX-License-Identifier: MPL-2.0
"""Controls for one shader effect (isf.py): a list of the effects found and a
control for each input of the chosen one (a slider, a check box, a list, a
color, a point). Used by the position editor (a layer's effect) and by
Window > Output Effects (a display's effect).

The value is what is saved: None, or {'name': file name, 'inputs': {NAME:
value}}. Every change is reported to on_change with a new dict (callers keep
the old one for Cancel).
"""
import tkinter as tk
from tkinter import ttk

import isf
import ui_theme as T

NONE = 'None'


class EffectControls:
    def __init__(self, parent, value, on_change, render_engine=None, bg=None, wraplength=200,
                 on_layout=None, preview_hint=True, mode_hint=True):
        self.on_change = on_change
        self.preview_hint = preview_hint        # the window has a preview that does not show it
        self.mode_hint = mode_hint              # say where it shows (a stack says it once, below)
        self.render_engine = render_engine
        self.on_layout = on_layout              # called after the controls are rebuilt (window sizes)
        self.bg = bg or T.BG
        self.frame = tk.Frame(parent, bg=self.bg)
        stale = isf.changed()                   # effect files written since the last look: use them anew
        if stale:
            isf.forget(stale)
            if hasattr(render_engine, 'reload_effects'):
                render_engine.reload_effects(stale)
        self.effects = isf.available()
        saved = isf.clean(value)
        names = sorted(self.effects, key=lambda n: isf.title(n).lower())
        self.titles = {NONE: None, **{isf.title(n): n for n in names}}
        current = NONE
        if saved:
            current = isf.title(saved['name'])
            if saved['name'] not in self.effects:
                current = f"{current} (missing)"
                self.titles[current] = saved['name']
        self.var = tk.StringVar(value=current)
        self.box = ttk.Combobox(self.frame, textvariable=self.var, state='readonly', width=18,
                                values=list(self.titles))
        self.box.pack()
        self.box.bind('<<ComboboxSelected>>', lambda e: self.choose())
        self.inputs_frame = tk.Frame(self.frame, bg=self.bg)
        self.inputs_frame.pack(fill='x')
        self.note = tk.Label(self.frame, fg=T.MUTED, bg=self.bg, font=T.font(9), wraplength=wraplength,
                             justify='center')
        self.note.pack()
        self.effect = None          # the parsed effect (isf.parse) whose controls are shown
        self.values = {}
        self._missing = saved if saved and saved['name'] not in self.effects else None
        self.show(saved['name'] if saved else None, saved['inputs'] if saved else {})

    # --- value ------------------------------------------------------------------

    def value(self):
        if self._missing is not None:            # kept as it was until another effect is chosen
            return {'name': self._missing['name'], 'inputs': dict(self._missing['inputs'])}
        if self.effect is None:
            return None
        return {'name': self.effect['name'], 'inputs': dict(self.values)}

    def choose(self):
        """The list's choice: that effect with its default values."""
        self._missing = None
        self.show(self.titles.get(self.var.get()), {})
        self._changed()

    def set_value(self, key, value):
        self.values[key] = value
        self._changed()

    def _changed(self):
        v = self.value()
        self.on_change(v)
        if v is not None:
            try:
                self.frame.after(800, lambda: self.check_error(v['name']))
            except tk.TclError:
                pass

    def check_error(self, name):
        """The output compiles an effect when it first draws it: show its error."""
        try:
            if not self.frame.winfo_exists():
                return
        except tk.TclError:
            return
        err = (getattr(self.render_engine, 'effect_errors', None) or {}).get(name)
        if err and self.effect is not None and self.effect['name'] == name:
            self.note.config(text=self.note_text(name, err))

    # --- controls ---------------------------------------------------------------

    def note_text(self, name, error=None):
        if error:
            return f"Not drawn: {error}"
        if name is None:
            return "Effects need Performance mode"
        if self.effect is None:
            return "Not drawn: not found in the effects folders"
        desc = self.effect.get('description') or ''
        if not self.mode_hint:
            return desc
        if not getattr(self.render_engine, 'draws_effects', False):
            return (desc + "\n" if desc else '') + "Not drawn in this render mode: effects need Performance mode"
        return (desc + "\n" if desc else '') + ("Shown on the output (not in this preview)" if self.preview_hint
                                                 else "Shown on the output and in the output preview")

    def show(self, name, values):
        """Controls for effect `name` with saved `values` (missing ones: the defaults)."""
        for w in self.inputs_frame.winfo_children():
            w.destroy()
        self.effect, error = None, None
        if name is not None:
            try:
                self.effect = isf.load(name)
            except (isf.ISFError, OSError) as e:
                error = None if name not in self.effects else str(e)
        self.values = {}
        for inp in (self.effect or {}).get('inputs', []):
            v = isf.clean_value(inp, values.get(inp['NAME'], inp['DEFAULT']))
            self.values[inp['NAME']] = v
            self._control(inp, v)
        self.note.config(text=self.note_text(name, error))
        if self.on_layout is not None:
            self.on_layout()

    def _control(self, inp, value):
        name, t = inp['NAME'], inp['TYPE']
        label = str(inp.get('LABEL') or name)
        bg = self.bg
        row = tk.Frame(self.inputs_frame, bg=bg)
        row.pack(fill='x', pady=1)
        if t == 'bool':
            var = tk.BooleanVar(value=bool(value))
            ttk.Checkbutton(row, text=label, variable=var,
                            command=lambda: self.set_value(name, var.get())).pack()
            return
        tk.Label(row, text=label, fg='white', bg=bg, width=9, anchor='w').pack(side='left')
        if t == 'long' and inp.get('VALUES'):
            labels = [str(x) for x in (inp.get('LABELS') or inp['VALUES'])]
            if len(labels) != len(inp['VALUES']):
                labels = [str(x) for x in inp['VALUES']]
            by_label = dict(zip(labels, inp['VALUES']))
            var = tk.StringVar(value=labels[inp['VALUES'].index(value)])
            box = ttk.Combobox(row, textvariable=var, state='readonly', width=12, values=labels)
            box.pack(side='left', fill='x', expand=True)
            box.bind('<<ComboboxSelected>>', lambda e: self.set_value(name, by_label[var.get()]))
            return
        if t == 'color':
            swatch = tk.Button(row, width=6, relief='flat', bg=_hex(value), activebackground=_hex(value))
            swatch.config(command=lambda: self._pick_color(name, swatch))
            swatch.pack(side='left')
            return
        if t == 'point2D':
            lo = inp.get('MIN') if isinstance(inp.get('MIN'), list) and len(inp['MIN']) == 2 else [0.0, 0.0]
            hi = inp.get('MAX') if isinstance(inp.get('MAX'), list) and len(inp['MAX']) == 2 else [1.0, 1.0]
            cell = tk.Frame(row, bg=bg)
            cell.pack(side='left', fill='x', expand=True)
            pvars = []
            for i, axis in enumerate(('X', 'Y')):
                sub = tk.Frame(cell, bg=bg)
                sub.pack(fill='x')
                tk.Label(sub, text=axis, fg=T.MUTED, bg=bg, width=2).pack(side='left')
                var = tk.DoubleVar(value=value[i])
                pvars.append(var)
                ttk.Scale(sub, from_=lo[i], to=hi[i], variable=var,
                          command=lambda _v: self.set_value(name, [round(pvars[0].get(), 3),
                                                                   round(pvars[1].get(), 3)])
                          ).pack(side='left', fill='x', expand=True)
            return
        # float, or long without VALUES: a slider
        lo, hi = isf._num(inp.get('MIN'), 0.0), isf._num(inp.get('MAX'), 1.0 if t == 'float' else 100)
        lo, hi = min(lo, hi), max(lo, hi)
        whole = t == 'long'
        var = tk.DoubleVar(value=value)
        shown = tk.Label(row, text=_fmt(value, whole, hi - lo), fg='white', bg=bg, width=4, anchor='e')

        def changed(_v):
            v = int(round(var.get())) if whole else round(var.get(), 4)
            shown.config(text=_fmt(v, whole, hi - lo))
            self.set_value(name, v)
        ttk.Scale(row, from_=lo, to=hi, variable=var, command=changed).pack(side='left', fill='x', expand=True)
        shown.pack(side='left', padx=(3, 0))

    def _pick_color(self, name, swatch):
        from tkinter import colorchooser
        old = self.values.get(name, [1.0, 1.0, 1.0, 1.0])
        picked = colorchooser.askcolor(color=_hex(old), parent=self.frame.winfo_toplevel(), title="Effect color")
        if not picked or not picked[0]:
            return
        rgb = [round(c / 255.0, 4) for c in picked[0]]
        swatch.config(bg=_hex(rgb), activebackground=_hex(rgb))
        self.set_value(name, rgb + [old[3] if len(old) > 3 else 1.0])


def _fmt(v, whole, span):
    if whole:
        return str(int(v))
    return f"{v:.0f}" if span >= 50 else f"{v:.1f}" if span >= 5 else f"{v:.2f}"


def _hex(rgba):
    return '#%02x%02x%02x' % tuple(int(round(max(0.0, min(1.0, c)) * 255)) for c in rgba[:3])


class EffectStack:
    """Several effects applied in order (at most isf.MAX_STACK): each with its
    controls under a small header (its number, move up, move down, remove),
    and a list to add one more. The value is a list of saved effects."""

    def __init__(self, parent, value, on_change, render_engine=None, bg=None, wraplength=200,
                 on_layout=None, preview_hint=True, shown_text=None):
        self.on_change = on_change
        self.render_engine = render_engine
        self.on_layout = on_layout
        self.preview_hint = preview_hint
        self.shown_text = shown_text          # where the effects show, if the window says it itself
        self.wraplength = wraplength
        self.bg = bg or T.BG
        self.frame = tk.Frame(parent, bg=self.bg)
        self.items_frame = tk.Frame(self.frame, bg=self.bg)
        self.items_frame.pack(fill='x')
        self.add_var = tk.StringVar(value="Add effect…")
        self.add_box = ttk.Combobox(self.frame, textvariable=self.add_var, state='readonly', width=18)
        self.add_box.pack(pady=(4, 0))
        self.add_box.bind('<<ComboboxSelected>>', lambda e: self._add_chosen())
        self.empty_note = tk.Label(self.frame, fg=T.MUTED, bg=self.bg, font=T.font(9), wraplength=wraplength,
                                   justify='center')
        self.items = []                       # EffectControls, in order
        self._values = isf.stack(value)
        self._rebuild()

    def value(self):
        return [dict(v, inputs=dict(v['inputs'])) for v in self._values]

    def _changed(self):
        self.on_change(self.value())

    def _rebuild(self):
        for w in self.items_frame.winfo_children():
            w.destroy()
        self.items = []
        n = len(self._values)
        for i, v in enumerate(self._values):
            box = tk.Frame(self.items_frame, bg=self.bg)
            box.pack(fill='x', pady=(0 if i == 0 else 8, 0))
            if n > 1:
                head = tk.Frame(box, bg=self.bg)
                head.pack(fill='x')
                tk.Label(head, text=f"{i + 1}.", fg=T.MUTED, bg=self.bg, font=T.font(9, 'bold')).pack(side='left')
                for text, cmd, ok in (("✕", lambda i=i: self.remove(i), True),
                                      ("▼", lambda i=i: self.move(i, 1), i < n - 1),
                                      ("▲", lambda i=i: self.move(i, -1), i > 0)):
                    b = tk.Button(head, text=text, command=cmd, relief='flat', bg=T.RAISED, fg='white',
                                  font=T.font(8), padx=4, pady=0, state='normal' if ok else 'disabled')
                    b.pack(side='right', padx=(2, 0))
            ctl = EffectControls(box, v, lambda val, i=i: self._item_changed(i, val), self.render_engine,
                                 bg=self.bg, wraplength=self.wraplength, mode_hint=False)
            ctl.frame.pack(fill='x')
            self.items.append(ctl)
        names = sorted(isf.available(), key=lambda nm: isf.title(nm).lower())
        self._add_titles = {isf.title(nm): nm for nm in names}
        self.add_box.config(values=list(self._add_titles),
                            state='readonly' if n < isf.MAX_STACK else 'disabled')
        self.add_var.set("Add effect…" if n else "Add an effect…")
        draws = getattr(self.render_engine, 'draws_effects', False)
        if not n:
            text = "No effect." + ("" if draws else " Effects need Performance mode.")
        elif not draws:
            text = "Not drawn in this render mode: effects need Performance mode"
        else:
            shown = self.shown_text or ("Shown on the output (not in this preview)" if self.preview_hint
                                        else "Shown on the output and in the output preview")
            text = ("Applied in this order, top first. " + shown) if n > 1 else shown
        self.empty_note.config(text=text)
        self.empty_note.pack()
        if self.on_layout is not None:
            self.on_layout()

    def _item_changed(self, i, val):
        if val is None:                       # "None" chosen in its list: take it out
            self.remove(i)
            return
        self._values[i] = val
        self._changed()

    def _add_chosen(self):
        name = self._add_titles.get(self.add_var.get())
        if name is None or len(self._values) >= isf.MAX_STACK:
            return
        self._values.append({'name': name, 'inputs': {}})
        self._rebuild()
        # the new controls hold the defaults: report those
        self._values[-1] = self.items[-1].value() or self._values[-1]
        self._changed()
        ctl = self.items[-1]
        try:                                   # the output compiles it when it first draws it
            ctl.frame.after(800, lambda: ctl.check_error(name))
        except tk.TclError:
            pass

    def remove(self, i):
        if 0 <= i < len(self._values):
            del self._values[i]
            self._rebuild()
            self._changed()

    def move(self, i, d):
        j = i + d
        if 0 <= i < len(self._values) and 0 <= j < len(self._values):
            self._values[i], self._values[j] = self._values[j], self._values[i]
            self._rebuild()
            self._changed()
