# SPDX-License-Identifier: MPL-2.0
"""VisualDeck's look: one dark palette, the platform's UI font, flat controls.

apply(root) must run right after the Tk root is created. Code that sets
colors explicitly uses the constants below (never hex literals), so the whole
interface changes from here.
"""
from __future__ import annotations

import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

# --- palette -------------------------------------------------------------------
BG = '#1e1f22'          # window, panels
SURFACE = '#2b2d31'     # dialogs, grouped areas
RAISED = '#383a40'      # buttons, headers, grid cells
HOVER = '#45484f'
PRESSED = '#2f3136'
INPUT = '#141517'       # text fields, lists, tables
BORDER = '#4a4d55'
TEXT = '#e9eaee'
MUTED = '#a3a7ae'
DISABLED = '#6b6f76'
ACCENT = '#4f8ff7'      # live cue, focus, selection
GO = '#21a35a'
STOP = '#d64545'
WARN = '#e6b422'
PLAYHEAD = '#5a6470'    # selected cue when nothing from the grid is live

FAMILY = 'Segoe UI'     # replaced in apply() on other platforms
MONO = 'Consolas'
BASE_SIZE = 10


def font(size=None, *styles):
    return (FAMILY, size or BASE_SIZE, *styles)


def mono(size=None, *styles):
    return (MONO, size or BASE_SIZE, *styles)


def _pick(families, wanted, fallback):
    for name in wanted:
        if name in families:
            return name
    return fallback


def _shade(color, amount):
    """Lighten (amount > 0) or darken (< 0) a #rrggbb / #rgb color."""
    c = color.lstrip('#')
    if len(c) == 3:
        c = ''.join(ch * 2 for ch in c)
    try:
        r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return color
    if amount >= 0:
        r, g, b = (round(v + (255 - v) * amount) for v in (r, g, b))
    else:
        r, g, b = (round(v * (1 + amount)) for v in (r, g, b))
    return f'#{r:02x}{g:02x}{b:02x}'


def hover_color(color):
    return _shade(color, 0.12)


_IMAGES = []   # keep PhotoImages alive


def _indicator_images(size):
    """Anti-aliased checkbox/radio images: (unchecked, checked, unchecked_hover, disabled) x 2 kinds."""
    from PIL import Image, ImageDraw, ImageTk
    k = 4                                    # supersampling
    S = size * k

    def rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)

    def make(kind, state):
        im = Image.new('RGBA', (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        on = state in ('on', 'on_disabled')
        dis = state.endswith('disabled')
        edge = rgb(DISABLED if dis else (ACCENT if on or state == 'hover' else BORDER))
        fill = rgb(SURFACE if dis else (ACCENT if on else INPUT))
        w = max(k, round(S / 9))
        box = [w // 2, w // 2, S - 1 - w // 2, S - 1 - w // 2]
        if kind == 'check':
            d.rounded_rectangle(box, radius=S // 5, fill=fill, outline=edge, width=w)
            if on:
                pts = [(S * 0.25, S * 0.52), (S * 0.43, S * 0.70), (S * 0.76, S * 0.33)]
                d.line(pts, fill=rgb('#ffffff' if not dis else MUTED), width=round(S / 7.5), joint='curve')
        else:
            d.ellipse(box, fill=rgb(INPUT if not dis else SURFACE), outline=edge, width=w)
            if on:
                r = S * 0.22
                d.ellipse([S / 2 - r, S / 2 - r, S / 2 + r, S / 2 + r], fill=rgb(ACCENT if not dis else DISABLED))
        small = im.resize((size, size), Image.LANCZOS)
        padded = Image.new('RGBA', (size + max(5, size // 3), size), (0, 0, 0, 0))   # gap before the label
        padded.paste(small, (0, 0))
        return ImageTk.PhotoImage(padded)

    return {kind: {st: make(kind, st) for st in ('off', 'on', 'hover', 'off_disabled', 'on_disabled')}
            for kind in ('check', 'radio')}


def _image_indicators(root, st):
    size = max(14, round(tkfont.nametofont('TkDefaultFont', root).metrics('linespace') * 0.85))
    imgs = _indicator_images(size)
    _IMAGES.append(imgs)
    for kind, cls in (('check', 'TCheckbutton'), ('radio', 'TRadiobutton')):
        i = imgs[kind]
        name = f'VD.{kind}.indicator'
        if name not in st.element_names():
            st.element_create(name, 'image', i['off'],
                              ('disabled', 'selected', i['on_disabled']), ('disabled', i['off_disabled']),
                              ('selected', i['on']), ('active', i['hover']), sticky='')
        st.layout(cls, [('Checkbutton.padding', {'sticky': 'nswe', 'children': [
            (name, {'side': 'left', 'sticky': ''}),
            ('Checkbutton.focus', {'side': 'left', 'sticky': 'w', 'children': [
                ('Checkbutton.label', {'sticky': 'nswe'})]})]})])


def apply(root):
    """Fonts, widget defaults and ttk styles for the whole application."""
    global FAMILY, MONO
    fams = set(tkfont.families(root))
    if sys.platform == 'win32':
        FAMILY = _pick(fams, ('Segoe UI Variable Text', 'Segoe UI'), 'Segoe UI')
        MONO = _pick(fams, ('Cascadia Mono', 'Consolas'), 'Courier New')
    elif sys.platform == 'darwin':
        FAMILY = tkfont.nametofont('TkDefaultFont', root).actual('family')
        MONO = _pick(fams, ('SF Mono', 'Menlo'), 'Courier')
    else:
        FAMILY = _pick(fams, ('Inter', 'Noto Sans', 'Ubuntu', 'Cantarell', 'DejaVu Sans', 'Liberation Sans'),
                       tkfont.nametofont('TkDefaultFont', root).actual('family'))
        MONO = _pick(fams, ('JetBrains Mono', 'DejaVu Sans Mono', 'Liberation Mono', 'Noto Sans Mono'), 'Courier')

    for name, size, weight in (('TkDefaultFont', BASE_SIZE, 'normal'), ('TkTextFont', BASE_SIZE, 'normal'),
                               ('TkMenuFont', BASE_SIZE, 'normal'), ('TkHeadingFont', BASE_SIZE, 'bold'),
                               ('TkCaptionFont', BASE_SIZE + 1, 'bold'), ('TkSmallCaptionFont', BASE_SIZE - 1, 'normal'),
                               ('TkTooltipFont', BASE_SIZE - 1, 'normal'), ('TkIconFont', BASE_SIZE, 'normal')):
        try:
            tkfont.nametofont(name, root).configure(family=FAMILY, size=size, weight=weight)
        except tk.TclError:
            pass
    try:
        tkfont.nametofont('TkFixedFont', root).configure(family=MONO, size=BASE_SIZE)
    except tk.TclError:
        pass

    o = root.option_add
    for pattern, value in (
        ('*Background', BG), ('*Foreground', TEXT), ('*Font', 'TkDefaultFont'),
        ('*highlightBackground', BG), ('*selectBackground', ACCENT), ('*selectForeground', '#ffffff'),
        ('*insertBackground', TEXT), ('*disabledForeground', DISABLED),
        ('*Button.relief', 'flat'), ('*Button.borderWidth', 0), ('*Button.highlightThickness', 0),
        ('*Button.background', RAISED), ('*Button.activeBackground', HOVER), ('*Button.activeForeground', TEXT),
        ('*Button.padX', 12), ('*Button.padY', 4), ('*Button.cursor', 'hand2'),
        ('*Entry.relief', 'flat'), ('*Entry.borderWidth', 4), ('*Entry.background', INPUT),
        ('*Entry.highlightThickness', 1), ('*Entry.highlightColor', ACCENT), ('*Entry.highlightBackground', BORDER),
        ('*Text.relief', 'flat'), ('*Text.borderWidth', 6), ('*Text.background', INPUT),
        ('*Text.highlightThickness', 1), ('*Text.highlightColor', ACCENT), ('*Text.highlightBackground', BORDER),
        ('*Listbox.relief', 'flat'), ('*Listbox.borderWidth', 0), ('*Listbox.background', INPUT),
        ('*Listbox.highlightThickness', 1), ('*Listbox.highlightColor', BORDER), ('*Listbox.activeStyle', 'none'),
        ('*Checkbutton.selectColor', INPUT), ('*Checkbutton.activeBackground', BG),
        ('*Checkbutton.activeForeground', TEXT), ('*Checkbutton.highlightThickness', 0),
        ('*Radiobutton.selectColor', INPUT), ('*Radiobutton.activeBackground', BG),
        ('*Radiobutton.activeForeground', TEXT), ('*Radiobutton.highlightThickness', 0),
        ('*Menu.background', SURFACE), ('*Menu.foreground', TEXT), ('*Menu.activeBackground', ACCENT),
        ('*Menu.activeForeground', '#ffffff'), ('*Menu.relief', 'flat'), ('*Menu.borderWidth', 1),
        ('*Menu.activeBorderWidth', 0),
        # The check mark and radio dot of menu entries. Tk's default on Windows is the
        # system menu text color (black), invisible on SURFACE until the entry is
        # hovered (owner's report, 2026-11-15: "GO After the Last Cue..." in Show).
        ('*Menu.selectColor', TEXT),
        ('*Canvas.highlightThickness', 0), ('*Scale.troughColor', INPUT), ('*Scale.highlightThickness', 0),
        ('*TCombobox*Listbox.background', INPUT), ('*TCombobox*Listbox.foreground', TEXT),
        ('*TCombobox*Listbox.selectBackground', ACCENT), ('*TCombobox*Listbox.selectForeground', '#ffffff'),
        ('*TCombobox*Listbox.font', 'TkDefaultFont'),
    ):
        o(pattern, value)

    st = ttk.Style(root)
    st.theme_use('clam')
    st.configure('.', background=BG, foreground=TEXT, fieldbackground=INPUT, bordercolor=BORDER,
                 lightcolor=BG, darkcolor=BG, troughcolor=INPUT, selectbackground=ACCENT,
                 selectforeground='#ffffff', insertcolor=TEXT, focuscolor=ACCENT, font='TkDefaultFont')
    st.map('.', foreground=[('disabled', DISABLED)])
    st.configure('TButton', background=RAISED, padding=(12, 5), relief='flat', borderwidth=0,
                 lightcolor=RAISED, darkcolor=RAISED, focusthickness=0)
    st.map('TButton', background=[('disabled', SURFACE), ('pressed', PRESSED), ('active', HOVER)],
           lightcolor=[('pressed', PRESSED), ('active', HOVER)], darkcolor=[('pressed', PRESSED), ('active', HOVER)])
    st.configure('Accent.TButton', background=ACCENT, foreground='#ffffff', lightcolor=ACCENT, darkcolor=ACCENT)
    st.map('Accent.TButton', background=[('active', hover_color(ACCENT))])
    for w in ('TEntry', 'TCombobox', 'TSpinbox'):
        st.configure(w, fieldbackground=INPUT, background=INPUT, foreground=TEXT, bordercolor=BORDER,
                     lightcolor=INPUT, darkcolor=INPUT, arrowcolor=MUTED, arrowsize=14, padding=4,
                     insertcolor=TEXT)
        st.map(w, bordercolor=[('focus', ACCENT)], lightcolor=[('focus', ACCENT)],
               fieldbackground=[('readonly', INPUT), ('disabled', SURFACE)],
               foreground=[('disabled', DISABLED), ('readonly', TEXT)],
               selectbackground=[('readonly', INPUT)], selectforeground=[('readonly', TEXT)],
               background=[('active', HOVER), ('pressed', PRESSED)])
    _image_indicators(root, st)
    for w in ('TCheckbutton', 'TRadiobutton'):
        st.configure(w, background=BG, foreground=TEXT, padding=(2, 3), focusthickness=0)
        st.map(w, background=[('active', BG)])
    st.configure('TLabel', background=BG, foreground=TEXT)
    st.configure('TFrame', background=BG)
    st.configure('TLabelframe', background=BG, bordercolor=BORDER, lightcolor=BG, darkcolor=BG, relief='solid',
                 borderwidth=1, padding=8)
    st.configure('TLabelframe.Label', background=BG, foreground=MUTED, font=font(BASE_SIZE - 1, 'bold'))
    st.configure('Horizontal.TScale', background=ACCENT, troughcolor=INPUT, bordercolor=BORDER,
                 lightcolor=ACCENT, darkcolor=ACCENT, sliderthickness=14)
    st.configure('Treeview', background=INPUT, fieldbackground=INPUT, foreground=TEXT, rowheight=24,
                 bordercolor=BORDER, lightcolor=INPUT, darkcolor=INPUT)
    st.map('Treeview', background=[('selected', ACCENT)], foreground=[('selected', '#ffffff')])
    st.configure('Treeview.Heading', background=RAISED, foreground=TEXT, relief='flat', bordercolor=BORDER,
                 lightcolor=RAISED, darkcolor=RAISED, padding=(6, 4), font=font(BASE_SIZE - 1, 'bold'))
    st.map('Treeview.Heading', background=[('active', HOVER)])
    st.configure('TProgressbar', background=ACCENT, troughcolor=INPUT, bordercolor=BORDER,
                 lightcolor=ACCENT, darkcolor=ACCENT, thickness=10)
    for w in ('Vertical.TScrollbar', 'Horizontal.TScrollbar'):
        st.configure(w, background=RAISED, troughcolor=BG, bordercolor=BG, lightcolor=RAISED, darkcolor=RAISED,
                     arrowcolor=MUTED, gripcount=0)
        st.map(w, background=[('active', HOVER), ('pressed', ACCENT)])
    st.configure('TSeparator', background=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    st.configure('TNotebook', background=BG, bordercolor=BORDER)
    st.configure('TNotebook.Tab', background=RAISED, foreground=MUTED, padding=(12, 5))
    st.map('TNotebook.Tab', background=[('selected', BG)], foreground=[('selected', TEXT)])

    # Hover feedback for classic Tk buttons (ttk ones use the style maps above).
    def enter(e):
        w = e.widget
        try:
            if str(w.cget('state')) == 'disabled':
                return
            bg = w.cget('bg')
            hv = hover_color(bg)
            w._vd_hover = (bg, hv)
            w.configure(bg=hv, activebackground=hv)
        except (tk.TclError, AttributeError):
            pass

    def leave(e):
        w = e.widget
        try:
            bg, hv = getattr(w, '_vd_hover', (None, None))
            if bg is not None and w.cget('bg') == hv:     # untouched while hovered: restore
                w.configure(bg=bg)
            w._vd_hover = (None, None)
        except (tk.TclError, AttributeError):
            pass
    root.bind_class('Button', '<Enter>', enter, add='+')
    root.bind_class('Button', '<Leave>', leave, add='+')
    try:
        root.configure(bg=BG)
    except tk.TclError:
        pass
