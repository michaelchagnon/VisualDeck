# SPDX-License-Identifier: MPL-2.0
# =====================================================================
# src/editor.py
# =====================================================================
import tkinter as tk
import ui_theme as T
from tkinter import ttk, messagebox
import os
from PIL import Image, ImageTk
import cv2
import json
import math
import numpy as np
import queue
import time
import uuid
import logging

log = logging.getLogger("visualdeck.editor")

class MediaEditor:
    def __init__(self, parent, media_path, screen_idx, render_engine,
             initial_pos=None, callback=None, realtime_callback=None,
             preview_bridge=None):
        # preview_bridge is accepted for compatibility and not used: nothing ever
        # fed it, and its poll started on a worker thread that touched Tk (F19).
        self.parent = parent
        self.media_path = media_path
        self.screen_idx = screen_idx
        self.render_engine = render_engine
        self.callback = callback
        self.realtime_callback = realtime_callback
        if screen_idx is not None and hasattr(parent, '_layout_size'):
            # The display's size, or its span's whole canvas (span.py)
            self.screen_width, self.screen_height = parent._layout_size(screen_idx)
        elif screen_idx is not None and screen_idx < len(render_engine.monitors):
            mon = render_engine.monitors[screen_idx]
            self.screen_width, self.screen_height = mon.width, mon.height
        else:
            self.screen_width, self.screen_height = 1920, 1080
        self.canvas_scale = self._fit_canvas_scale(parent)
        self.canvas_width = int(self.screen_width * self.canvas_scale)
        self.canvas_height = int(self.screen_height * self.canvas_scale)
        default_pos = {'x': 0.5, 'y': 0.5, 'scale_x': 1.0, 'scale_y': 1.0, 'rotation': 0}
        self.original_pos = {**default_pos, **initial_pos} if isinstance(initial_pos, dict) else default_pos.copy()
        self.current_pos = self.original_pos.copy()
        self.drag_item = None
        self.drag_handle = None
        self.drag_start_pos = None
        self.drag_start_mouse = None
        self.drag_start_handles = None
        self.last_update_time = 0
        self._create_window()
        
    PANEL_WIDTH = 330                 # the settings on the right
    CONTROLS_WIDTH = PANEL_WIDTH + 70 # the panel with its scroll bar and padding
    CHROME_HEIGHT = 220               # title bar, the button row, the taskbar
    MAX_CANVAS_SCALE = 0.6

    # Which sections are open, kept while VisualDeck runs (the next layer opens the same way).
    open_sections = {'position'}

    def _fit_canvas_scale(self, parent):
        """Up to 60 % of the output's size, less so the window fits on this screen
        (a 4K output's canvas would otherwise be wider than a 1080p screen)."""
        try:
            sw, sh = parent.root.winfo_screenwidth(), parent.root.winfo_screenheight()
        except Exception:  # noqa: BLE001 - no screen size: half size
            return 0.5
        avail_w = max(320, sw - self.CONTROLS_WIDTH - 40)
        avail_h = max(180, sh - self.CHROME_HEIGHT)
        return max(0.1, min(self.MAX_CANVAS_SCALE, avail_w / max(1, self.screen_width),
                            avail_h / max(1, self.screen_height)))

    def _create_window(self):
        self.window = tk.Toplevel(self.parent.root)
        self.window.title(f"Edit: {os.path.basename(self.media_path)}")
        self.window.configure(bg=T.BG)
        self.window.transient(self.parent.root)
        self.window.grab_set()
        self.window.bind('<MouseWheel>', lambda e: "break")

        # Dialog buttons in a row along the bottom of the whole window, right
        # aligned (Cancel, then the default action, Save), like the app's other
        # dialogs. Packed first so they keep their space on short screens.
        button_bar = tk.Frame(self.window, bg=T.BG)
        button_bar.pack(side='bottom', fill='x', padx=10, pady=(0, 10))
        ttk.Separator(self.window, orient='horizontal').pack(side='bottom', fill='x', padx=10, pady=(0, 10))
        self.save_button = tk.Button(button_bar, text="Save", command=self._save_and_close, bg=T.GO, fg='white',
                                     activeforeground='white', width=10, font=T.font(10, 'bold'))
        self.save_button.pack(side='right')
        self.cancel_button = tk.Button(button_bar, text="Cancel", command=self._cancel_and_close, width=10)
        self.cancel_button.pack(side='right', padx=(0, 8))
        self.canvas_hint = tk.Label(button_bar, text="", fg=T.MUTED, bg=T.BG, font=T.font(9))
        self.canvas_hint.pack(side='left')
        self.window.bind('<Return>', lambda e: self._save_and_close())
        self.window.bind('<Escape>', lambda e: self._cancel_and_close())

        main_frame = tk.Frame(self.window, bg=T.BG)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Settings on the right: one collapsible section per kind of setting.
        self._build_panel(main_frame)

        canvas_frame = tk.Frame(main_frame, bg=T.BG)
        canvas_frame.pack(side='left', fill='both', expand=True)
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, bg=T.INPUT,
                                highlightthickness=0)
        self.canvas.pack(padx=5, pady=5, anchor='n')
        self.canvas.create_rectangle(2, 2, self.canvas_width - 1, self.canvas_height - 1, outline=T.BORDER,
                                     dash=(5, 5), width=2)
        # Under the canvas: what it shows of the effects (see _fx_preview_*).
        self.fx_note = tk.Label(canvas_frame, text="", fg=T.MUTED, bg=T.BG, font=T.font(9), anchor='w',
                                justify='left', wraplength=max(200, self.canvas_width))
        self.fx_note.pack(fill='x', padx=5)

        self._fx_preview_init()
        self._load_media()
        self._update_canvas()
        self._update_canvas_hint()

        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        for key, (dx, dy) in (('Left', (-1, 0)), ('Right', (1, 0)), ('Up', (0, -1)), ('Down', (0, 1))):
            self.canvas.bind(f'<{key}>', lambda e, d=(dx, dy): self._nudge_warp(*d))
            self.canvas.bind(f'<Shift-{key}>', lambda e, d=(dx, dy): self._nudge_warp(d[0] * 10, d[1] * 10))

    # --- the settings panel ------------------------------------------------------

    def _build_panel(self, parent):
        """A scrolling column of collapsible sections: a header with an arrow
        (› closed, ˅ open), the name and, while closed, a short summary of
        what is set; click the header to open or close it."""
        outer = tk.Frame(parent, bg=T.BG, width=self.PANEL_WIDTH + 18)
        outer.pack(side='right', fill='y', padx=(10, 0))
        outer.pack_propagate(False)
        self.panel_canvas = tk.Canvas(outer, bg=T.BG, highlightthickness=0, width=self.PANEL_WIDTH)
        sb = ttk.Scrollbar(outer, orient='vertical', command=self.panel_canvas.yview)
        self.panel_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.panel_canvas.pack(side='left', fill='both', expand=True)
        self.panel = tk.Frame(self.panel_canvas, bg=T.BG, width=self.PANEL_WIDTH)
        self.panel_canvas.create_window(0, 0, window=self.panel, anchor='nw', width=self.PANEL_WIDTH)
        self.panel.bind('<Configure>', lambda e: self.panel_canvas.configure(
            scrollregion=(0, 0, self.PANEL_WIDTH, self.panel.winfo_reqheight())))

        def wheel(e):
            if self.panel.winfo_reqheight() > self.panel_canvas.winfo_height():
                self.panel_canvas.yview_scroll(-1 if (getattr(e, 'delta', 0) > 0 or e.num == 4) else 1, 'units')
            return "break"
        for seq in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            outer.bind_all(seq, lambda e: wheel(e) if self._over_panel(e) else None, add='+')

        is_video = os.path.splitext(self.media_path)[1].lower() in ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v')
        self.sections = {}
        for key, title, build, summary in (
                ('position', "Position and size", self._build_position_controls, self._position_summary),
                ('look', "Opacity and blend", self._build_look_controls, self._look_summary),
                ('color', "Color", self._build_color_controls, self._color_summary),
                ('playback', "Playback", self._build_playback_controls if is_video else None,
                 self._playback_summary),
                ('mask', "Mask", self._build_mask_controls, self._mask_summary),
                ('warp', "Warp points", self._build_warp_controls, self._warp_summary),
                ('effects', "Effects", self._build_effect_controls, self._effects_summary)):
            if build is None:
                continue
            self.sections[key] = _Section(self.panel, key, title, key in self.open_sections, build, summary,
                                          on_toggle=self._on_section_toggle)

    def _over_panel(self, e):
        try:
            w = self.window.winfo_containing(e.x_root, e.y_root)
        except (tk.TclError, KeyError):
            return False
        while w is not None:
            if w is self.panel_canvas:
                return True
            w = getattr(w, 'master', None)
        return False

    def _on_section_toggle(self, key, opened):
        (MediaEditor.open_sections.add if opened else MediaEditor.open_sections.discard)(key)
        if key == 'warp' and hasattr(self, 'canvas'):
            self._update_canvas()
            self._update_canvas_hint()

    def _update_canvas_hint(self):
        hint = getattr(self, 'canvas_hint', None)
        if hint is not None:
            hint.config(text="Drag a dot to bend the picture, the picture to move it; arrow keys move the chosen dot."
                        if self._warp_editing() else
                        "Drag the picture to move it, a corner to resize (Ctrl keeps its shape).")

    def _refresh_summaries(self):
        for sec in getattr(self, 'sections', {}).values():
            sec.refresh()

    # Summaries shown on a closed section's header ("" when nothing is set).
    def _position_summary(self):
        p = self.current_pos
        return f"{int(round(p.get('scale_x', 1.0) * 100))} %" + (
            f", {int(round(p.get('rotation', 0)))}°" if round(p.get('rotation', 0)) else "")

    def _look_summary(self):
        p = self.current_pos
        parts = []
        if p.get('opacity', 1.0) < 1.0:
            parts.append(f"{int(round(p['opacity'] * 100))} %")
        if p.get('blend', 'normal') != 'normal':
            parts.append(str(p['blend']).capitalize())
        return ", ".join(parts)

    def _color_summary(self):
        return "changed" if any(k in self.current_pos for k in self._COLOR_NEUTRAL) else ""

    def _playback_summary(self):
        p = self.current_pos
        parts = []
        if p.get('speed', 1.0) != 1.0:
            parts.append(f"{p['speed']:g}×")
        if p.get('direction', 'forward') != 'forward':
            parts.append(self._DIRECTION_LABELS.get(p['direction'], ''))
        if p.get('in_s') is not None or p.get('out_s') is not None:
            parts.append("in/out")
        if p.get('volume', 1.0) != 1.0:
            parts.append(f"vol {int(round(p['volume'] * 100))} %")
        return ", ".join(x for x in parts if x)

    def _mask_summary(self):
        m = self.current_pos.get('mask')
        return {'rect': "Rectangle", 'ellipse': "Ellipse"}.get((m or {}).get('shape'), "") if m else ""

    def _warp_summary(self):
        m = self._warp_mesh()
        return f"{m[0]}\u00d7{m[1]} points, bent" if m is not None else ""

    def _effects_summary(self):
        import isf
        return ", ".join(isf.title(fx['name']) for fx in isf.layer_stack(self.current_pos))

    # --- section contents ----------------------------------------------------------

    def _slider_row(self, parent, label, var, lo, hi, command, value_text=None):
        row = tk.Frame(parent, bg=T.BG)
        row.pack(fill='x', pady=1)
        tk.Label(row, text=label, fg='white', bg=T.BG, width=10, anchor='w').pack(side='left')
        ttk.Scale(row, from_=lo, to=hi, variable=var, command=command).pack(side='left', fill='x', expand=True)
        shown = None
        if value_text is not None:
            shown = tk.Label(row, text=value_text, fg='white', bg=T.BG, width=5, anchor='e')
            shown.pack(side='left', padx=(3, 0))
        return shown

    def _build_position_controls(self, frame):
        self.x_var = tk.DoubleVar(value=self.current_pos['x'])
        self.x_label = self._slider_row(frame, "X", self.x_var, 0, 1, self._on_slider_change, "")
        self.y_var = tk.DoubleVar(value=self.current_pos['y'])
        self.y_label = self._slider_row(frame, "Y", self.y_var, 0, 1, self._on_slider_change, "")
        self._refresh_xy_labels()
        self.scale_var = tk.DoubleVar(value=self.current_pos.get('scale_x', 1.0))
        self.scale_label = self._slider_row(frame, "Size", self.scale_var, 0.01, 5.0, self._on_scale_slider_change,
                                            f"{int(self.scale_var.get() * 100)}%")
        self.rotation_var = tk.DoubleVar(value=self.current_pos['rotation'])
        self.rotation_label = self._slider_row(frame, "Rotation", self.rotation_var, -180, 180,
                                               self._on_slider_change, f"{int(self.current_pos['rotation'])}°")
        preset_frame = tk.Frame(frame, bg=T.BG)
        preset_frame.pack(pady=(6, 0))
        for text, cmd in (("Center", self._center_media), ("Fit", self._fit_to_screen),
                          ("Fill", self._fill_screen), ("Reset", self._reset_position)):
            tk.Button(preset_frame, text=text, command=cmd, bg=T.RAISED, fg='white', width=6,
                      padx=4).pack(side='left', padx=2)

    def _refresh_xy_labels(self):
        """X and Y as a percentage of the output (50 % = the middle)."""
        for lab, key in ((getattr(self, 'x_label', None), 'x'), (getattr(self, 'y_label', None), 'y')):
            if lab is not None:
                lab.config(text=f"{int(round(self.current_pos[key] * 100))}%")

    def _build_look_controls(self, frame):
        self.opacity_var = tk.DoubleVar(value=round(float(self.current_pos.get('opacity', 1.0)) * 100))
        self.opacity_label = self._slider_row(frame, "Opacity", self.opacity_var, 0, 100, self._on_opacity_change,
                                              f"{int(self.opacity_var.get())}%")
        row = tk.Frame(frame, bg=T.BG)
        row.pack(fill='x', pady=(4, 1))
        tk.Label(row, text="Blend", fg='white', bg=T.BG, width=10, anchor='w').pack(side='left')
        self.blend_var = tk.StringVar(value=str(self.current_pos.get('blend', 'normal')).capitalize())
        blend_box = ttk.Combobox(row, textvariable=self.blend_var, state='readonly', width=12,
                                 values=['Normal', 'Add', 'Multiply', 'Screen'])
        blend_box.pack(side='left')
        blend_box.bind('<<ComboboxSelected>>', self._on_blend_change)
        tk.Label(frame, text="Add, Multiply and Screen need Performance mode", fg=T.MUTED, bg=T.BG,
                 font=T.font(9), anchor='w').pack(fill='x')

    def _build_color_controls(self, frame):
        from render_engine import layer_color
        b, c, sat = layer_color(self.current_pos) or (0.0, 1.0, 1.0)
        self.color_vars, self.color_labels = {}, {}
        for key, label, lo, hi, val in (('brightness', 'Brightness', -100, 100, b * 100),
                                         ('contrast', 'Contrast', 0, 200, c * 100),
                                         ('saturation', 'Saturation', 0, 200, sat * 100)):
            var = tk.DoubleVar(value=round(val))
            lab = self._slider_row(frame, label, var, lo, hi, lambda _v, k=key: self._on_color_change(k),
                                   f"{int(round(val))}")
            self.color_vars[key], self.color_labels[key] = var, lab
        ttk.Button(frame, text="Reset color", command=self._reset_color).pack(pady=(4, 0))

    def _build_playback_controls(self, frame):
        def labeled(text):
            row = tk.Frame(frame, bg=T.BG)
            row.pack(fill='x', pady=(2, 1))
            tk.Label(row, text=text, fg='white', bg=T.BG, width=10, anchor='w').pack(side='left')
            return row
        try:
            speed = float(self.current_pos.get('speed', 1.0))
        except (TypeError, ValueError):
            speed = 1.0
        self.speed_var = tk.StringVar(value=f"{speed:g}×")
        speed_box = ttk.Combobox(labeled("Speed"), textvariable=self.speed_var, state='readonly', width=8,
                                 values=[f"{v:g}×" for v in (0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4)])
        speed_box.pack(side='left')
        speed_box.bind('<<ComboboxSelected>>', self._on_speed_change)
        d = self.current_pos.get('direction', 'forward')
        self.direction_var = tk.StringVar(value=self._DIRECTION_LABELS.get(d, 'Forward'))
        dir_box = ttk.Combobox(labeled("Direction"), textvariable=self.direction_var, state='readonly', width=10,
                               values=list(self._DIRECTION_LABELS.values()))
        dir_box.pack(side='left')
        dir_box.bind('<<ComboboxSelected>>', self._on_direction_change)
        trim_row = labeled("In – out (s)")
        self.clip_seconds = self._clip_length()
        self.in_var = tk.StringVar(value=self._fmt_s(self.current_pos.get('in_s')))
        self.out_var = tk.StringVar(value=self._fmt_s(self.current_pos.get('out_s')))
        for i, var in enumerate((self.in_var, self.out_var)):
            if i:
                tk.Label(trim_row, text="to", fg='white', bg=T.BG).pack(side='left', padx=4)
            ent = ttk.Entry(trim_row, textvariable=var, width=7)
            ent.pack(side='left')
            ent.bind('<Return>', lambda e: self._on_trim_change())
            ent.bind('<FocusOut>', lambda e: self._on_trim_change())
        self.trim_note = tk.Label(frame, fg=T.MUTED, bg=T.BG, text=self._trim_text(), wraplength=self.PANEL_WIDTH - 20,
                                  justify='left', anchor='w', font=T.font(9))
        self.trim_note.pack(fill='x')
        from render_engine import layer_volume
        self.volume_var = tk.DoubleVar(value=round(layer_volume(self.current_pos) * 100))
        self.volume_label = self._slider_row(frame, "Volume", self.volume_var, 0, 200,
                                             lambda _v: self._on_volume_change(), f"{int(self.volume_var.get())}%")

    # --- Warp points (layer_warp.py) ------------------------------------------------

    def _build_warp_controls(self, frame):
        """A grid of points on the picture: drag them on the canvas (shown while
        this section is open) to bend the picture onto a surface."""
        import layer_warp
        import mesh_warp
        m = layer_warp.layer_warp(self.current_pos)
        self.warp_cols, self.warp_rows = (m[0], m[1]) if m else (2, 2)
        self.warp_points = [list(p) for p in m[2]] if m else mesh_warp.identity(2, 2)
        self.warp_selected = None
        tk.Label(frame, text="Drag the dots on the picture to bend it onto a surface: the corners for a "
                             "keystone, more points for a curve. Click a dot and use the arrow keys for 1 px "
                             "(Shift: 10 px).", fg=T.MUTED, bg=T.BG, font=T.font(9), justify='left', anchor='w',
                 wraplength=self.PANEL_WIDTH - 20).pack(fill='x', pady=(0, 4))
        row = tk.Frame(frame, bg=T.BG)
        row.pack(fill='x', pady=1)
        tk.Label(row, text="Points", fg='white', bg=T.BG, width=10, anchor='w').pack(side='left')
        self.warp_cols_var = tk.StringVar(value=str(self.warp_cols))
        self.warp_rows_var = tk.StringVar(value=str(self.warp_rows))
        for i, var in enumerate((self.warp_cols_var, self.warp_rows_var)):
            if i:
                tk.Label(row, text="\u00d7", fg='white', bg=T.BG).pack(side='left', padx=3)
            sp = ttk.Spinbox(row, from_=mesh_warp.MIN_POINTS, to=mesh_warp.MAX_POINTS, width=3, textvariable=var,
                             state='readonly', command=self._on_warp_grid_change)
            sp.pack(side='left')
        tk.Label(row, text="across \u00d7 down", fg=T.MUTED, bg=T.BG, font=T.font(9)).pack(side='left', padx=(6, 0))
        btns = tk.Frame(frame, bg=T.BG)
        btns.pack(fill='x', pady=(4, 0))
        ttk.Button(btns, text="Reset warp", command=self._reset_warp).pack(side='left')
        self.warp_note = tk.Label(frame, text="", fg=T.WARN, bg=T.BG, font=T.font(9), justify='left', anchor='w',
                                  wraplength=self.PANEL_WIDTH - 20)
        self.warp_note.pack(fill='x')

    def _warp_mesh(self):
        """The warp as edited (cols, rows, points), or None while nothing is bent."""
        import mesh_warp
        if not hasattr(self, 'warp_points'):
            return None
        m = (self.warp_cols, self.warp_rows, tuple(tuple(p) for p in self.warp_points))
        return None if mesh_warp.is_identity(*m) else m

    def _warp_editing(self):
        sec = getattr(self, 'sections', {}).get('warp')
        return sec is not None and sec.opened

    def _store_warp(self):
        import layer_warp
        m = self._warp_mesh()
        if m is None:
            self.current_pos.pop('warp', None)
        else:
            self.current_pos['warp'] = layer_warp.to_dict(m)    # a new dict: original_pos must not change
        self._update_canvas()
        self._trigger_realtime_callback()

    def _on_warp_grid_change(self):
        import mesh_warp
        try:
            cols, rows = int(self.warp_cols_var.get()), int(self.warp_rows_var.get())
        except ValueError:
            return
        if (cols, rows) == (self.warp_cols, self.warp_rows):
            return
        old = (self.warp_cols, self.warp_rows, tuple(tuple(p) for p in self.warp_points))
        self.warp_points = mesh_warp.resize(None if mesh_warp.is_identity(*old) else old, cols, rows)
        self.warp_cols, self.warp_rows = cols, rows
        self.warp_selected = None
        self._store_warp()

    def _reset_warp(self):
        import mesh_warp
        self.warp_points = mesh_warp.identity(self.warp_cols, self.warp_rows)
        self.warp_selected = None
        self.warp_note.config(text="")
        self._store_warp()

    def _canvas_to_media(self, x, y):
        """Picture coordinates (u, v; 0..1 across the unwarped layer) of a canvas point."""
        cx = self.current_pos['x'] * self.canvas_width
        cy = self.current_pos['y'] * self.canvas_height
        w = self.media_width * self.current_pos.get('scale_x', 1.0) * self.canvas_scale
        h = self.media_height * self.current_pos.get('scale_y', 1.0) * self.canvas_scale
        rx, ry = x - cx, y - cy
        rad = math.radians(self.current_pos.get('rotation', 0))
        dx = rx * math.cos(rad) - ry * math.sin(rad)
        dy = rx * math.sin(rad) + ry * math.cos(rad)
        return (dx / w + 0.5 if w else 0.5), (dy / h + 0.5 if h else 0.5)

    def _move_warp_point(self, i, u, v):
        """Put point i at (u, v); refused (and said) if the picture would fold over itself."""
        import mesh_warp
        lim = mesh_warp.LIMIT
        u, v = max(-lim, min(1 + lim, u)), max(-lim, min(1 + lim, v))
        pts = [list(p) for p in self.warp_points]
        pts[i] = [round(u, 5), round(v, 5)]
        if mesh_warp.folds((self.warp_cols, self.warp_rows, tuple(tuple(p) for p in pts))):
            self.warp_note.config(text="Not there: the picture would fold over itself.")
            return False
        self.warp_note.config(text="")
        self.warp_points = pts
        self._store_warp()
        return True

    WARP_DOT_INSET = 7      # a dot on the canvas's edge is drawn this far inside it (radius 5 + outline)

    def _warp_dot_xy(self, u, v):
        """Where point (u, v)'s dot is drawn: at the point, or pulled inside the
        canvas when the point is on or past its edge. A picture that fills the
        output has its corners on the edge, where the dots were almost invisible
        (owner's PC, 2026-11-14). Dragging moves the point by the mouse's
        movement, so a pulled-in dot does not make the point jump."""
        x, y = self._media_to_canvas(u, v)
        m = self.WARP_DOT_INSET
        return (min(max(x, m), self.canvas_width - m), min(max(y, m), self.canvas_height - m))

    def _warp_point_at(self, x, y, radius=9):
        best, best_d = None, radius * radius
        for i, (u, v) in enumerate(self.warp_points):
            px, py = self._warp_dot_xy(u, v)
            d = (px - x) ** 2 + (py - y) ** 2
            if d <= best_d:
                best, best_d = i, d
        return best

    def _nudge_warp(self, dx, dy):
        if self.warp_selected is None or not self._warp_editing():
            return
        u, v = self.warp_points[self.warp_selected]
        px, py = self._media_to_canvas(u, v)
        # dx, dy in output pixels: the canvas shows the output at canvas_scale
        nu, nv = self._canvas_to_media(px + dx * self.canvas_scale, py + dy * self.canvas_scale)
        self._move_warp_point(self.warp_selected, nu, nv)
        return "break"

    def _redraw_warp_points(self):
        self.canvas.delete('warp_point')
        self.canvas.delete('warp_line')
        self._draw_warp_points()

    def _draw_warp_points(self):
        if not self._warp_editing():
            return
        pts = [self._media_to_canvas(u, v) for u, v in self.warp_points]
        cols, rows = self.warp_cols, self.warp_rows
        for j in range(rows):                      # the grid's lines, between the points
            for i in range(cols):
                x, y = pts[j * cols + i]
                if i + 1 < cols:
                    self.canvas.create_line(x, y, *pts[j * cols + i + 1], fill=T.ACCENT, dash=(3, 3),
                                            tags=('media', 'warp_line'))
                if j + 1 < rows:
                    self.canvas.create_line(x, y, *pts[(j + 1) * cols + i], fill=T.ACCENT, dash=(3, 3),
                                            tags=('media', 'warp_line'))
        for i, (u, v) in enumerate(self.warp_points):
            x, y = self._warp_dot_xy(u, v)
            sel = i == self.warp_selected
            r = 7 if sel else 5
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=T.WARN if sel else T.ACCENT, outline='white',
                                    width=2 if sel else 1, tags=('media', 'warp_point', f'wp{i}'))

    _MASK_SHAPES = {'None': None, 'Rectangle': 'rect', 'Ellipse': 'ellipse'}

    def _build_mask_controls(self, frame):
        """Mask: shape, an inset per side, soft edge, invert (see layer_mask)."""
        import layer_mask
        m = layer_mask.to_dict(layer_mask.layer_mask(self.current_pos)) or {}
        labels = {v: k for k, v in self._MASK_SHAPES.items()}
        self.mask_shape_var = tk.StringVar(value=labels.get(m.get('shape'), 'None'))
        shape_row = tk.Frame(frame, bg=T.BG)
        shape_row.pack(fill='x', pady=(0, 2))
        tk.Label(shape_row, text="Shape", fg='white', bg=T.BG, width=10, anchor='w').pack(side='left')
        self.mask_shape_box = ttk.Combobox(shape_row, textvariable=self.mask_shape_var, state='readonly', width=12,
                                           values=list(self._MASK_SHAPES))
        self.mask_shape_box.pack(side='left')
        self.mask_shape_box.bind('<<ComboboxSelected>>', lambda e: self._on_mask_change())
        self.mask_vars, self.mask_labels = {}, {}
        for key, label, hi in (('left', 'Left', 49), ('right', 'Right', 49), ('top', 'Top', 49),
                               ('bottom', 'Bottom', 49), ('feather', 'Soft edge', 50)):
            row = tk.Frame(frame, bg=T.BG)
            row.pack(fill='x', pady=1)
            tk.Label(row, text=label, fg='white', bg=T.BG, width=10, anchor='w').pack(side='left')
            val = round(float(m.get(key, 0.0)) * 100)
            var = tk.DoubleVar(value=val)
            ttk.Scale(row, from_=0, to=hi, variable=var,
                      command=lambda _v, k=key: self._on_mask_change(k)).pack(side='left', fill='x', expand=True)
            lab = tk.Label(row, text=f"{val}%", fg='white', bg=T.BG, width=4, anchor='e')
            lab.pack(side='left', padx=(3, 0))
            self.mask_vars[key], self.mask_labels[key] = var, lab
        self.mask_invert_var = tk.BooleanVar(value=bool(m.get('invert')))
        ttk.Checkbutton(frame, text="Invert (show outside)", variable=self.mask_invert_var,
                        command=self._on_mask_change).pack(anchor='w', pady=(2, 0))

    def _on_mask_change(self, key=None):
        import layer_mask
        if key is not None:
            self.mask_labels[key].config(text=f"{int(round(self.mask_vars[key].get()))}%")
        shape = self._MASK_SHAPES.get(self.mask_shape_var.get())
        raw = {'shape': shape, 'invert': self.mask_invert_var.get(),
               **{k: round(v.get()) / 100.0 for k, v in self.mask_vars.items()}}
        mask = layer_mask.to_dict(layer_mask.clean(raw)) if shape else None
        if mask:
            self.current_pos['mask'] = mask          # a new dict: original_pos must not change
        else:
            self.current_pos.pop('mask', None)
        self._update_canvas()
        self._trigger_realtime_callback()

    # --- Shader effect (effect_ui.py) ---------------------------------------------

    def _build_effect_controls(self, frame):
        """Effects: shader effects applied in order, each with a control per input."""
        import isf
        from effect_ui import EffectStack
        self.effect_ui = EffectStack(frame, isf.layer_stack(self.current_pos), self._on_effect_value,
                                     self.render_engine, bg=T.BG, wraplength=self.PANEL_WIDTH - 20,
                                     shown_text="Shown on the canvas and on the output")
        self.effect_ui.frame.pack(fill='x')

    def _on_effect_value(self, effects):
        self.current_pos.pop('effect', None)   # the older single effect becomes the list
        if effects:                            # a new list: original_pos must not change
            self.current_pos['effects'] = effects
        else:
            self.current_pos.pop('effects', None)
        self._update_canvas()
        self._trigger_realtime_callback()

    def _media_to_canvas(self, u, v):
        """Canvas point of media texture coordinates (u, v), with the layer's
        position, size and rotation (as _get_handle_positions)."""
        cx = self.current_pos['x'] * self.canvas_width
        cy = self.current_pos['y'] * self.canvas_height
        w = self.media_width * self.current_pos.get('scale_x', 1.0) * self.canvas_scale
        h = self.media_height * self.current_pos.get('scale_y', 1.0) * self.canvas_scale
        dx, dy = (u - 0.5) * w, (v - 0.5) * h
        rad = math.radians(self.current_pos.get('rotation', 0))
        return cx + dx * math.cos(rad) + dy * math.sin(rad), cy - dx * math.sin(rad) + dy * math.cos(rad)

    def _mask_outline(self):
        """Canvas polygon points of the mask's outline, or None."""
        import layer_mask
        m = layer_mask.layer_mask(self.current_pos)
        if m is None:
            return None
        shape, lo_u, lo_v, hi_u, hi_v = m[:5]
        if shape == 'rect':
            uv = [(lo_u, lo_v), (hi_u, lo_v), (hi_u, hi_v), (lo_u, hi_v)]
        else:
            cu, cv, ru, rv = (lo_u + hi_u) / 2, (lo_v + hi_v) / 2, (hi_u - lo_u) / 2, (hi_v - lo_v) / 2
            uv = [(cu + ru * math.cos(t), cv + rv * math.sin(t)) for t in (i * math.pi / 24 for i in range(48))]
        return [c for p in uv for c in self._media_to_canvas(*p)]

    def _load_media(self):
        import live_source
        if live_source.is_live(self.media_path):
            # The newest picture if the source is running, else a card of its size.
            frame = getattr(self.render_engine, 'video_frame_cache', {}).get(self.media_path)
            img = None
            if frame is not None:
                try:
                    img = self.render_engine._video_frame_to_pil(frame, 1.0, 1.0)
                except Exception:  # noqa: BLE001
                    img = None
            if img is None:
                size = getattr(self.parent, '_live_size', lambda p: (1920, 1080))(self.media_path)
                card = getattr(self.parent, '_live_card', None)
                img = card(self.media_path, size) if card else Image.new('RGB', size, (28, 30, 36))
            self.original_image = img
            self.media_width, self.media_height = img.size
            return
        try:
            ext = os.path.splitext(self.media_path)[1].lower()
            if ext in ('.mp4', '.avi', '.mov', '.mkv'):
                from render_engine import video_still     # the video's own color matrix (BT.709 for HD)
                still = video_still(self.media_path)
                if still is not None:
                    self.original_image = still
                else:
                    cap = cv2.VideoCapture(self.media_path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret: self.original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    else: self.original_image = Image.new('RGB', (100, 100), color='red')
            else: self.original_image = Image.open(self.media_path)
            self.media_width, self.media_height = self.original_image.size
        except Exception as e:
            self.original_image = Image.new('RGB', (100, 100), color='red')
            self.media_width, self.media_height = 100, 100
            messagebox.showerror("Error", f"Failed to load media: {e}")
            log.exception(f"Failed to load media for editor: {self.media_path}")

    def _base_image(self, w, h):
        """The picture at the canvas size (kept while the size stays: moving it
        does not resize a 4K frame again)."""
        cache = getattr(self, '_base_cache', None)
        if cache is None or cache[0] != (w, h) or cache[1] is not self.original_image:
            self._base_cache = cache = ((w, h), self.original_image,
                                        self.original_image.resize((w, h), Image.LANCZOS))
        return cache[2]

    # --- effect preview ----------------------------------------------------------------
    # The canvas shows the layer's effects as the output draws them: the render
    # thread runs them on this picture with the output's own shaders (in
    # Performance mode, while the output window is open) and sends the result
    # back (render_engine.effect_preview_begin). Moving effects (Wave, Trails)
    # are drawn again about ten times a second.
    FX_SOURCE_MAX = 2048          # the picture sent to the GPU, at most this wide or high
    FX_ANIMATE_S = 0.1

    def _fx_preview_init(self):
        self._fx_q = queue.Queue()
        self._fx_seq = 0
        self._fx_sigs = {}
        self._fx_inflight = None          # (seq, time sent)
        self._fx_again = False
        self._fx_result = None            # (signature, PIL image)
        self._fx_image_id = uuid.uuid4().hex
        self._fx_sent_id = None
        self._fx_why = None
        self._fx_tick, self._fx_tick_at = 0, time.monotonic()
        self.window.bind('<Destroy>', self._fx_on_destroy, add='+')
        self.window.after(40, self._fx_poll)

    def _fx_queue(self):
        return getattr(self.parent, 'render_command_queue', None)

    def _fx_effects(self):
        import isf
        return isf.layer_stack(self.current_pos)

    def _fx_animated(self, effects):
        import isf
        return any(isf.animated(fx['name']) for fx in effects)

    def _set_fx_note(self, text):
        note = getattr(self, 'fx_note', None)
        if note is not None and note.cget('text') != text:
            note.config(text=text)

    def _fx_preview_image(self, w, h):
        """The picture with its effects at w x h, or None (no effects, or not
        available: the note under the canvas says why)."""
        effects = self._fx_effects()
        if not effects:
            self._set_fx_note("")
            return None
        if not getattr(self.render_engine, 'draws_effects', False):
            self._set_fx_note("Effects are drawn in Performance mode only; this render mode shows the layer "
                              "without them.")
            return None
        if self._fx_queue() is None:
            self._set_fx_note("The effects are not drawn here; they show on the output.")
            return None
        animated = self._fx_animated(effects)
        drawn = (round(self.media_width * abs(self.current_pos.get('scale_x', 1.0))),
                 round(self.media_height * abs(self.current_pos.get('scale_y', 1.0))))
        sig = (json.dumps(effects, sort_keys=True), w, h, drawn, self._fx_tick if animated else 0)
        if self._fx_result is not None and self._fx_result[0] == sig:
            return self._fx_result[1]
        self._fx_request(sig, effects, drawn, (w, h))
        if self._fx_result is not None and self._fx_result[1].size == (w, h) and self._fx_result[0][0] == sig[0]:
            return self._fx_result[1]             # the same effects a moment ago, until the new one comes
        return None

    def _fx_request(self, sig, effects, drawn, out):
        if self._fx_inflight is not None:
            self._fx_again = True                 # one at a time; the newest is asked for when it returns
            return
        img = None
        if self._fx_sent_id != self._fx_image_id:
            src = self.original_image.convert('RGBA')
            if max(src.size) > self.FX_SOURCE_MAX:
                src = src.copy()
                src.thumbnail((self.FX_SOURCE_MAX, self.FX_SOURCE_MAX), Image.LANCZOS)
            img = np.asarray(src, dtype=np.uint8)
        self._fx_seq += 1
        self._fx_sigs = {self._fx_seq: sig}
        self._fx_queue().put({'action': 'fx_preview', 'image': img, 'image_id': self._fx_image_id,
                              'effects': effects, 'drawn': drawn, 'out': out, 'seq': self._fx_seq,
                              'reply': self._fx_q})
        if img is not None:
            self._fx_sent_id = self._fx_image_id
        self._fx_inflight = (self._fx_seq, time.monotonic())

    def _fx_poll(self):
        try:
            if not self.window.winfo_exists():
                return
        except tk.TclError:
            return
        redraw = False
        while True:
            try:
                seq, arr, why = self._fx_q.get_nowait()
            except queue.Empty:
                break
            if self._fx_inflight is not None and seq == self._fx_inflight[0]:
                self._fx_inflight = None
            sig = self._fx_sigs.get(seq)
            if arr is not None and sig is not None:
                self._fx_result = (sig, Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8)))
                self._fx_why = None
                redraw = True
            elif why:
                if why == "no picture":
                    self._fx_sent_id = None       # the output started again: send the picture again
                self._fx_why = why
                redraw = True
        if self._fx_inflight is not None and time.monotonic() - self._fx_inflight[1] > 2.0:
            self._fx_inflight = None              # no answer (the output is busy or closed)
            self._fx_why = self._fx_why or "no answer from the output"
            redraw = True
        effects = self._fx_effects()
        if effects and self._fx_animated(effects) and time.monotonic() - self._fx_tick_at >= self.FX_ANIMATE_S:
            self._fx_tick, self._fx_tick_at = self._fx_tick + 1, time.monotonic()
            redraw = True
        if self._fx_again and self._fx_inflight is None:
            self._fx_again = False
            redraw = True
        if effects and getattr(self.render_engine, 'draws_effects', False) and self._fx_queue() is not None:
            if self._fx_why:
                self._set_fx_note("Effects are not shown here (" + str(self._fx_why) + "); they show on the "
                                  "output when the cue is on it. The canvas shows them while the output is open "
                                  "in Performance mode.")
            elif self._fx_result is not None:
                self._set_fx_note("Effects shown as the output draws them.")
        if redraw:
            self._update_canvas()
        self.window.after(40, self._fx_poll)

    def _fx_on_destroy(self, event):
        if event.widget is not self.window:
            return
        q = self._fx_queue()
        if q is not None and self._fx_sent_id is not None:
            q.put({'action': 'fx_preview_end'})
            self._fx_sent_id = None

    def _trigger_realtime_callback(self):
        """Send the position to the app, at most every 30 ms. An update that
        falls inside that window is sent shortly afterwards (trailing edge),
        so the last change of a drag or a quick second change is never lost."""
        self._refresh_summaries()
        now = time.perf_counter()
        if now - self.last_update_time > 0.03:
            if self.realtime_callback: self.realtime_callback(self.current_pos)
            self.last_update_time = now
        elif not getattr(self, '_trailing_pending', False):
            self._trailing_pending = True
            self.window.after(35, self._flush_realtime_callback)

    def _flush_realtime_callback(self):
        self._trailing_pending = False
        if self.realtime_callback:
            self.realtime_callback(self.current_pos)
        self.last_update_time = time.perf_counter()

    def _update_canvas(self):
        self.canvas.delete('media')
        scaled_width = self.media_width * self.current_pos.get('scale_x', 1.0)
        scaled_height = self.media_height * self.current_pos.get('scale_y', 1.0)
        canvas_x = self.current_pos['x'] * self.canvas_width
        canvas_y = self.current_pos['y'] * self.canvas_height
        display_width = int(scaled_width * self.canvas_scale)
        display_height = int(scaled_height * self.canvas_scale)
        if display_width > 0 and display_height > 0:
            resized = self._base_image(display_width, display_height)
            fx = self._fx_preview_image(display_width, display_height)
            if fx is not None:
                resized = fx                      # the layer with its effects, drawn by the output's engine
            import layer_mask
            from render_engine import adjust_pil, layer_color
            adj = layer_color(self.current_pos)
            if adj is not None:
                resized = adjust_pil(resized, adj)
            # The mask is in the media's own coordinates: before rotation.
            resized = layer_mask.apply_pil(resized, layer_mask.layer_mask(self.current_pos))
            warp = self._warp_mesh()
            if warp is not None:                  # bent around the same center (layer_warp)
                import layer_warp
                resized = layer_warp.warp_pil(resized, warp)
            if self.current_pos.get('rotation', 0) != 0:
                resized = resized.rotate(self.current_pos['rotation'], expand=True, resample=Image.BICUBIC)
            op = self.current_pos.get('opacity', 1.0)
            if op < 1.0:
                from render_engine import apply_opacity
                resized = apply_opacity(resized.convert('RGBA'), max(0.0, op))
            self.photo = ImageTk.PhotoImage(resized)
            self.canvas.create_image(canvas_x, canvas_y, image=self.photo, anchor='center', tags=('media', 'draggable'))
            outline = self._mask_outline()
            if outline:
                self.canvas.create_polygon(*outline, outline=T.WARN, fill='', dash=(4, 3), width=1,
                                           tags=('media', 'draggable', 'mask_outline'))
            if self._warp_editing():
                self._draw_warp_points()           # the warp's dots instead of the size handles
            else:
                handle_positions = self._get_handle_positions()
                for handle_id, pos in handle_positions.items():
                    x, y = pos
                    self.canvas.create_rectangle(x-4, y-4, x+4, y+4, fill=T.GO, outline='white', width=1, tags=('media', 'handle', handle_id))
    
    def _get_handle_positions(self):
        center_x = self.current_pos['x'] * self.canvas_width
        center_y = self.current_pos['y'] * self.canvas_height
        half_w = (self.media_width * self.current_pos.get('scale_x', 1.0) * self.canvas_scale) / 2
        half_h = (self.media_height * self.current_pos.get('scale_y', 1.0) * self.canvas_scale) / 2
        rad = math.radians(self.current_pos['rotation'])
        cos_rad = math.cos(rad)
        sin_rad = math.sin(rad)
        corners = {'nw': (-half_w, -half_h), 'ne': (half_w, -half_h), 'sw': (-half_w, half_h), 'se': (half_w, half_h)}
        # Positive rotation turns the media counter-clockwise on screen (PIL,
        # pygame and the GL model matrix all agree); the canvas y axis points
        # down, so the corners turn by -angle in canvas coordinates (F23).
        rotated_corners = {}
        for key, (dx, dy) in corners.items():
            rx = dx * cos_rad + dy * sin_rad
            ry = -dx * sin_rad + dy * cos_rad
            rotated_corners[key] = (center_x + rx, center_y + ry)
        return rotated_corners
    
    def _on_slider_change(self, value):
        self.current_pos['x'] = self.x_var.get()
        self.current_pos['y'] = self.y_var.get()
        self.current_pos['rotation'] = self.rotation_var.get()
        self.rotation_label.config(text=f"{int(self.current_pos['rotation'])}°")
        self._refresh_xy_labels()
        self._update_canvas()
        self._trigger_realtime_callback()
    
    def _on_opacity_change(self, value):
        pct = int(round(self.opacity_var.get()))
        self.current_pos['opacity'] = pct / 100.0
        self.opacity_label.config(text=f"{pct}%")
        self._update_canvas()
        self._trigger_realtime_callback()

    _COLOR_NEUTRAL = {'brightness': 0.0, 'contrast': 1.0, 'saturation': 1.0}

    def _on_color_change(self, key):
        pct = int(round(self.color_vars[key].get()))
        self.color_labels[key].config(text=str(pct))
        v = pct / 100.0
        if v == self._COLOR_NEUTRAL[key]:
            self.current_pos.pop(key, None)
        else:
            self.current_pos[key] = v
        self._update_canvas()
        self._trigger_realtime_callback()

    def _reset_color(self):
        for key, neutral in self._COLOR_NEUTRAL.items():
            self.color_vars[key].set(round(neutral * 100))
            self.color_labels[key].config(text=str(int(round(neutral * 100))))
            self.current_pos.pop(key, None)
        self._update_canvas()
        self._trigger_realtime_callback()

    def _on_speed_change(self, event=None):
        v = float(self.speed_var.get().rstrip('\u00d7'))
        if v == 1.0:
            self.current_pos.pop('speed', None)
        else:
            self.current_pos['speed'] = v
        self._trigger_realtime_callback()

    _DIRECTION_LABELS = {'forward': 'Forward', 'reverse': 'Reverse', 'pingpong': 'Ping-pong'}

    @staticmethod
    def _fmt_s(v):
        return "" if v is None else f"{float(v):.2f}".rstrip('0').rstrip('.')

    def _clip_length(self):
        try:
            return float(self.parent.mm.get_media_info(self.media_path)[1] or 0) or None
        except Exception:  # noqa: BLE001
            return None

    def _trim_text(self):
        a, b = self.current_pos.get('in_s'), self.current_pos.get('out_s')
        clip = f"clip {self.clip_seconds:.2f} s" if self.clip_seconds else "clip length unknown"
        if a is None and b is None:
            return f"Empty: the whole clip ({clip})"
        end = b if b is not None else self.clip_seconds
        played = f"plays {end - (a or 0):.2f} s of the {clip}" if end else clip
        return played

    @staticmethod
    def _parse_s(text):
        """Seconds from "12.5" or "1:02.5" (minutes:seconds); None for empty; ValueError otherwise."""
        t = (text or "").strip()
        if not t:
            return None
        if ':' in t:
            m, sec = t.split(':', 1)
            v = int(m) * 60 + float(sec)
        else:
            v = float(t)
        if v != v or v < 0:
            raise ValueError(t)
        return v

    def _on_trim_change(self):
        """In and out points from the two fields: seconds from the clip's start;
        empty means the start / the end."""
        try:
            a, b = self._parse_s(self.in_var.get()), self._parse_s(self.out_var.get())
        except ValueError:
            self.trim_note.config(fg=T.WARN, text="Use seconds, e.g. 12.5 or 1:02.5")
            return False
        if a == 0:
            a = None
        if b is not None and self.clip_seconds and b >= self.clip_seconds - 1e-3:
            b = None                                  # at or past the end: the end
        if a is not None and self.clip_seconds and a >= self.clip_seconds:
            self.trim_note.config(fg=T.WARN, text=f"In must be before the clip's end ({self.clip_seconds:.2f} s)")
            return False
        if a is not None and b is not None and b - a < 0.04:
            self.trim_note.config(fg=T.WARN, text="Out must be after In")
            return False
        changed = (self.current_pos.get('in_s'), self.current_pos.get('out_s')) != (a, b)
        for key, v in (('in_s', a), ('out_s', b)):
            if v is None:
                self.current_pos.pop(key, None)
            else:
                self.current_pos[key] = round(v, 3)
        self.in_var.set(self._fmt_s(a))
        self.out_var.set(self._fmt_s(b))
        self.trim_note.config(fg=T.MUTED, text=self._trim_text())
        if changed:
            self._trigger_realtime_callback()
        return True

    def _on_direction_change(self, event=None):
        label = self.direction_var.get()
        d = next((k for k, v in self._DIRECTION_LABELS.items() if v == label), 'forward')
        if d == 'forward':
            self.current_pos.pop('direction', None)
        else:
            self.current_pos['direction'] = d
        self._trigger_realtime_callback()

    def _on_volume_change(self):
        pct = int(round(self.volume_var.get()))
        self.volume_label.config(text=f"{pct}%")
        if pct == 100:
            self.current_pos.pop('volume', None)
        else:
            self.current_pos['volume'] = pct / 100.0
        self._trigger_realtime_callback()

    def _on_blend_change(self, event=None):
        mode = self.blend_var.get().lower()
        if mode == 'normal':
            self.current_pos.pop('blend', None)
        else:
            self.current_pos['blend'] = mode
        self._trigger_realtime_callback()

    def _on_scale_slider_change(self, value):
        val = self.scale_var.get()
        self.current_pos['scale_x'] = val
        self.current_pos['scale_y'] = val
        self.scale_label.config(text=f"{int(val*100)}%")
        self._update_canvas()
        self._trigger_realtime_callback()

    def _on_mouse_down(self, event):
        self.drag_start_pos = self.current_pos.copy()
        self.drag_start_handles = self._get_handle_positions()
        self.drag_start_mouse = (event.x, event.y)
        if self._warp_editing():
            self.canvas.focus_set()                # arrow keys nudge the chosen dot
            i = self._warp_point_at(event.x, event.y)
            if i is not None:
                self.warp_selected = i
                px, py = self._media_to_canvas(*self.warp_points[i])
                self._warp_grab = (px - event.x, py - event.y)   # the dot may be drawn pulled in from the point
                self.drag_item = 'warp'
                self._redraw_warp_points()         # only the dots: the picture has not changed
                return
        clicked = self.canvas.find_closest(event.x, event.y)
        if not clicked: return
        tags = self.canvas.gettags(clicked[0])
        if 'handle' in tags:
            self.drag_item = 'resize'
            self.drag_handle = [t for t in tags if t in ['nw', 'ne', 'sw', 'se']][0]
        elif 'draggable' in tags:
            self.drag_item = 'move'
    
    def _on_mouse_drag(self, event):
        if not self.drag_item: return
        if self.drag_item == 'warp':
            # The mouse sends more moves than the picture can be bent and drawn: keep the
            # newest and draw at most once per WARP_DRAG_MS (owner's report 2026-11-14:
            # dragging a dot lagged behind the mouse; each move was drawn, ~60 ms each).
            gx, gy = getattr(self, '_warp_grab', (0.0, 0.0))
            self._warp_drag_to = (self.warp_selected, *self._canvas_to_media(event.x + gx, event.y + gy))
            if not getattr(self, '_warp_drag_due', False):
                self._warp_drag_due = True
                self.window.after(self.WARP_DRAG_MS, self._apply_warp_drag)
            return
        if self.drag_item == 'move':
            dx = event.x - self.drag_start_mouse[0]
            dy = event.y - self.drag_start_mouse[1]
            new_x = self.drag_start_pos['x'] + (dx / self.canvas_width)
            new_y = self.drag_start_pos['y'] + (dy / self.canvas_height)
            self.current_pos['x'] = new_x
            self.current_pos['y'] = new_y
            self.x_var.set(self.current_pos['x'])
            self.y_var.set(self.current_pos['y'])
        elif self.drag_item == 'resize':
            anchor_map = {'nw': 'se', 'ne': 'sw', 'sw': 'ne', 'se': 'nw'}
            anchor_handle = anchor_map[self.drag_handle]
            anchor_x, anchor_y = self.drag_start_handles[anchor_handle]
            # Size along the media's own (rotated) axes, not the screen's.
            rad = math.radians(self.current_pos.get('rotation', 0))
            ddx, ddy = event.x - anchor_x, event.y - anchor_y
            new_w = abs(ddx * math.cos(rad) - ddy * math.sin(rad))
            new_h = abs(ddx * math.sin(rad) + ddy * math.cos(rad))
            
            lock_aspect = (event.state & 4) != 0
            
            if lock_aspect and self.media_width > 0 and self.media_height > 0:
                original_aspect = self.media_height / self.media_width
                if new_w * original_aspect > new_h:
                    new_h = new_w * original_aspect
                else:
                    new_w = new_h / original_aspect

            new_scale_x = (new_w / self.canvas_scale) / self.media_width if self.media_width > 0 else 0
            new_scale_y = (new_h / self.canvas_scale) / self.media_height if self.media_height > 0 else 0
            
            self.current_pos['scale_x'] = max(0.01, min(5.0, new_scale_x))
            self.current_pos['scale_y'] = max(0.01, min(5.0, new_scale_y))
            
            new_center_x = anchor_x + (event.x - anchor_x) / 2
            new_center_y = anchor_y + (event.y - anchor_y) / 2
            
            self.current_pos['x'] = new_center_x / self.canvas_width
            self.current_pos['y'] = new_center_y / self.canvas_height
            
            self.x_var.set(self.current_pos['x'])
            self.y_var.set(self.current_pos['y'])
            
            if lock_aspect:
                self.scale_var.set(self.current_pos['scale_x'])
                self.scale_label.config(text=f"{int(self.scale_var.get()*100)}%")

        self._refresh_xy_labels()
        self._update_canvas()
        self._trigger_realtime_callback()

    WARP_DRAG_MS = 15

    def _apply_warp_drag(self):
        self._warp_drag_due = False
        target, self._warp_drag_to = getattr(self, '_warp_drag_to', None), None
        if target is not None:
            t0 = time.perf_counter()
            self._move_warp_point(*target)
            self.__dict__.setdefault('_warp_drag_ms', []).append((time.perf_counter() - t0) * 1000.0)

    def _on_mouse_up(self, event):
        if self.drag_item == 'warp' and getattr(self, '_warp_drag_to', None) is not None:
            self._apply_warp_drag()                # the last position, at once
        times = self.__dict__.pop('_warp_drag_ms', None)
        if times:                                  # for checking "the canvas follows the mouse" on a show PC
            times.sort()
            log.info(f"Warp point drag: {len(times)} canvas updates, median {times[len(times) // 2]:.0f} ms, "
                     f"slowest {times[-1]:.0f} ms")
        self.drag_item = None
        self.drag_handle = None
        self.drag_start_pos = None
        self.drag_start_mouse = None
        self.drag_start_handles = None

    def _update_ui_from_pos(self):
        self.x_var.set(self.current_pos['x'])
        self.y_var.set(self.current_pos['y'])
        self.scale_var.set(self.current_pos['scale_x'])
        self.rotation_var.set(self.current_pos['rotation'])
        self.scale_label.config(text=f"{int(self.current_pos['scale_x']*100)}%")
        self.rotation_label.config(text=f"{int(self.current_pos['rotation'])}°")
        self._refresh_xy_labels()
        self._update_canvas()
        self._trigger_realtime_callback()

    def _center_media(self):
        self.current_pos['x'] = 0.5
        self.current_pos['y'] = 0.5
        self._update_ui_from_pos()

    def _fit_to_screen(self):
        if self.media_height == 0: return
        screen_aspect = self.screen_width / self.screen_height
        media_aspect = self.media_width / self.media_height
        scale = (self.screen_width / self.media_width) if media_aspect > screen_aspect else (self.screen_height / self.media_height)
        self.current_pos['scale_x'] = scale
        self.current_pos['scale_y'] = scale
        self._center_media()

    def _fill_screen(self):
        if self.media_height == 0: return
        screen_aspect = self.screen_width / self.screen_height
        media_aspect = self.media_width / self.media_height
        scale = (self.screen_height / self.media_height) if media_aspect > screen_aspect else (self.screen_width / self.media_width)
        self.current_pos['scale_x'] = scale
        self.current_pos['scale_y'] = scale
        self._center_media()

    def _reset_position(self):
        # Position, size and rotation only: opacity, blend, color, speed and
        # the mask keep their values (and their controls stay true).
        self.current_pos.update({'x': 0.5, 'y': 0.5, 'scale_x': 1.0, 'scale_y': 1.0, 'rotation': 0})
        self._update_ui_from_pos()

    def _save_and_close(self):
        if hasattr(self, 'in_var') and self._on_trim_change() is False:
            return                                    # an In/Out typed but not valid: say so, stay open
        if self.callback: self.callback(self.media_path, self.current_pos)
        self.window.destroy()

    def _cancel_and_close(self):
        if self.realtime_callback: self.realtime_callback(self.original_pos)
        self.window.destroy()


class _Section:
    """A collapsible part of the editor's settings: a header (arrow, name and,
    while closed, a short summary of what is set) and the controls under it,
    built at once (every control exists, so its value is always known) and
    shown only while the section is open."""

    def __init__(self, parent, key, title, opened, build, summary, on_toggle=None):
        self.key, self.summary, self.on_toggle = key, summary, on_toggle
        self.frame = tk.Frame(parent, bg=T.BG)
        self.frame.pack(fill='x', pady=(0, 2))
        self.header = tk.Frame(self.frame, bg=T.SURFACE, cursor='hand2')
        self.header.pack(fill='x')
        self.arrow = tk.Canvas(self.header, width=14, height=14, bg=T.SURFACE, highlightthickness=0)
        self.arrow.pack(side='left', padx=(8, 4), pady=7)
        self.title = tk.Label(self.header, text=title, fg=T.TEXT, bg=T.SURFACE, font=T.font(10, 'bold'))
        self.title.pack(side='left')
        self.hint = tk.Label(self.header, text="", fg=T.MUTED, bg=T.SURFACE, font=T.font(9), anchor='e')
        self.hint.pack(side='right', padx=(4, 10))
        self.body = tk.Frame(self.frame, bg=T.BG)
        build(self.body)
        for w in (self.header, self.arrow, self.title, self.hint):
            w.bind('<Button-1>', lambda e: self.toggle())
        self.opened = None
        self.set_open(opened)

    def set_open(self, opened):
        if opened == self.opened:
            return
        self.opened = opened
        self.arrow.delete('all')
        pts = (4, 2, 10, 7, 4, 12) if not opened else (2, 4, 7, 10, 12, 4)     # › closed, ˅ open
        self.arrow.create_line(*pts, fill=T.ACCENT if opened else T.MUTED, width=2, capstyle='round',
                               joinstyle='round')
        if opened:
            self.body.pack(fill='x', padx=(10, 4), pady=(6, 10))
        else:
            self.body.pack_forget()
        self.refresh()

    def toggle(self):
        self.set_open(not self.opened)
        if self.on_toggle is not None:
            self.on_toggle(self.key, self.opened)

    def refresh(self):
        try:
            text = "" if self.opened else (self.summary() or "")
        except Exception:  # noqa: BLE001 - a summary must never break the editor
            text = ""
        if self.hint.cget('text') != text:
            self.hint.config(text=text, fg=T.ACCENT if text else T.MUTED)
