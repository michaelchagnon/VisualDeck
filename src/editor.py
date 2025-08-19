# =====================================================================
# src/editor.py
# =====================================================================
import tkinter as tk
from tkinter import ttk, messagebox
import os
from PIL import Image, ImageTk
import cv2
import math
import time
import logging

class MediaEditor:
    def __init__(self, parent, media_path, screen_idx, render_engine,
             initial_pos=None, callback=None, realtime_callback=None,
             preview_bridge=None):
        self.parent = parent
        self.media_path = media_path
        self.screen_idx = screen_idx
        self.render_engine = render_engine
        self.callback = callback
        self.realtime_callback = realtime_callback
        if screen_idx is not None and screen_idx < len(render_engine.monitors):
            mon = render_engine.monitors[screen_idx]
            self.screen_width, self.screen_height = mon.width, mon.height
        else:
            self.screen_width, self.screen_height = 1920, 1080
        self.canvas_scale = 0.5
        self.canvas_width = int(self.screen_width * self.canvas_scale)
        self.canvas_height = int(self.screen_height * self.canvas_scale)
        default_pos = {'x': 0.5, 'y': 0.5, 'scale_x': 1.0, 'scale_y': 1.0, 'rotation': 0}
        self.original_pos = initial_pos.copy() if initial_pos else default_pos.copy()
        self.current_pos = self.original_pos.copy()
        self.drag_item = None
        self.drag_handle = None
        self.drag_start_pos = None
        self.drag_start_mouse = None
        self.drag_start_handles = None
        self.last_update_time = 0
        self._create_window()
        if preview_bridge is not None:
            self._init_preview_async(preview_bridge)
 
    def _init_preview_async(self, preview_bridge):
        import threading, queue
        from PIL import Image, ImageTk

        self._preview_bridge = preview_bridge
        self._preview_imgtk = None
        self._preview_label = self.preview_label  # or canvas if used
        self._preview_poll_stop = False

        def _poll():
            if self._preview_poll_stop:
                return
            try:
                item = self._preview_bridge.pop_latest()
                if item:
                    rgba, (w, h), ts = item
                    img = Image.fromarray(rgba)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self._preview_imgtk = imgtk
                    self._preview_label.config(image=imgtk)
            except Exception:
                pass
            self._preview_label.after(33, _poll)  # ~30 FPS

        threading.Thread(target=_poll, daemon=True).start()
        
    def _create_window(self):
        self.window = tk.Toplevel(self.parent.root)
        self.window.title(f"Edit: {os.path.basename(self.media_path)}")
        self.window.configure(bg='#2e2e2e')
        self.window.transient(self.parent.root)
        self.window.grab_set()
        self.window.bind('<MouseWheel>', lambda e: "break")
        
        main_frame = tk.Frame(self.window, bg='#2e2e2e')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        canvas_frame = tk.Frame(main_frame, bg='#2e2e2e')
        canvas_frame.pack(side='left', fill='both', expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(padx=5, pady=5)
        self.canvas.create_rectangle(2, 2, self.canvas_width - 1, self.canvas_height - 1, outline='#666', dash=(5, 5), width=2)
        
        self.preview_label = tk.Label(canvas_frame, bg='#1a1a1a')
        self.preview_label.pack(padx=5, pady=5)
        
        controls_frame = tk.Frame(main_frame, bg='#2e2e2e', width=180)
        controls_frame.pack(side='right', fill='y', padx=(10, 0))
        controls_frame.pack_propagate(False)
        
        tk.Label(controls_frame, text="Position", fg='white', bg='#2e2e2e', font=('Arial', 10, 'bold')).pack(pady=(0, 5))
        x_frame = tk.Frame(controls_frame, bg='#2e2e2e')
        x_frame.pack(fill='x', pady=2)
        tk.Label(x_frame, text="X:", fg='white', bg='#2e2e2e', width=8).pack(side='left')
        self.x_var = tk.DoubleVar(value=self.current_pos['x'])
        self.x_slider = ttk.Scale(x_frame, from_=0, to=1, variable=self.x_var, command=self._on_slider_change)
        self.x_slider.pack(side='left', fill='x', expand=True)
        y_frame = tk.Frame(controls_frame, bg='#2e2e2e')
        y_frame.pack(fill='x', pady=2)
        tk.Label(y_frame, text="Y:", fg='white', bg='#2e2e2e', width=8).pack(side='left')
        self.y_var = tk.DoubleVar(value=self.current_pos['y'])
        self.y_slider = ttk.Scale(y_frame, from_=0, to=1, variable=self.y_var, command=self._on_slider_change)
        self.y_slider.pack(side='left', fill='x', expand=True)
        
        tk.Label(controls_frame, text="Scale", fg='white', bg='#2e2e2e', font=('Arial', 10, 'bold')).pack(pady=(15, 5))
        scale_frame = tk.Frame(controls_frame, bg='#2e2e2e')
        scale_frame.pack(fill='x', pady=2)
        tk.Label(scale_frame, text="Size:", fg='white', bg='#2e2e2e', width=8).pack(side='left')
        self.scale_var = tk.DoubleVar(value=self.current_pos.get('scale_x', 1.0))
        self.scale_slider = ttk.Scale(scale_frame, from_=0.01, to=5.0, variable=self.scale_var, command=self._on_scale_slider_change)
        self.scale_slider.pack(side='left', fill='x', expand=True)
        self.scale_label = tk.Label(controls_frame, text=f"{int(self.scale_var.get()*100)}%", fg='white', bg='#2e2e2e')
        self.scale_label.pack()

        tk.Label(controls_frame, text="Rotation", fg='white', bg='#2e2e2e', font=('Arial', 10, 'bold')).pack(pady=(15, 5))
        rot_frame = tk.Frame(controls_frame, bg='#2e2e2e')
        rot_frame.pack(fill='x', pady=2)
        tk.Label(rot_frame, text="Angle:", fg='white', bg='#2e2e2e', width=8).pack(side='left')
        self.rotation_var = tk.DoubleVar(value=self.current_pos['rotation'])
        self.rotation_slider = ttk.Scale(rot_frame, from_=-180, to=180, variable=self.rotation_var, command=self._on_slider_change)
        self.rotation_slider.pack(side='left', fill='x', expand=True)
        self.rotation_label = tk.Label(controls_frame, text=f"{int(self.current_pos['rotation'])}°", fg='white', bg='#2e2e2e')
        self.rotation_label.pack()
        
        tk.Label(controls_frame, text="Presets", fg='white', bg='#2e2e2e', font=('Arial', 10, 'bold')).pack(pady=(15, 5))
        preset_frame = tk.Frame(controls_frame, bg='#2e2e2e')
        preset_frame.pack(anchor='center')
        tk.Button(preset_frame, text="Center", command=self._center_media, bg='#444', fg='white', width=8).pack(side='left', padx=2)
        tk.Button(preset_frame, text="Fit", command=self._fit_to_screen, bg='#444', fg='white', width=8).pack(side='left', padx=2)
        preset_frame2 = tk.Frame(controls_frame, bg='#2e2e2e')
        preset_frame2.pack(anchor='center', pady=(5, 0))
        tk.Button(preset_frame2, text="Fill", command=self._fill_screen, bg='#444', fg='white', width=8).pack(side='left', padx=2)
        tk.Button(preset_frame2, text="Reset", command=self._reset_position, bg='#444', fg='white', width=8).pack(side='left', padx=2)

        button_frame = tk.Frame(controls_frame, bg='#2e2e2e')
        button_frame.pack(side='bottom', fill='x', pady=(20, 0))
        tk.Button(button_frame, text="Save", command=self._save_and_close, bg='#5a5', fg='black', width=8, height=2).pack(side='left', padx=2)
        tk.Button(button_frame, text="Cancel", command=self._cancel_and_close, bg='#a55', fg='white', width=8, height=2).pack(side='right', padx=2)
        
        self._load_media()
        self._update_canvas()
        
        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        
    def _load_media(self):
        try:
            ext = os.path.splitext(self.media_path)[1].lower()
            if ext in ('.mp4', '.avi', '.mov', '.mkv'):
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
            logging.exception(f"Failed to load media for editor: {self.media_path}")

    def _trigger_realtime_callback(self):
        now = time.perf_counter()
        if now - self.last_update_time > 0.03:
            if self.realtime_callback: self.realtime_callback(self.current_pos)
            self.last_update_time = now

    def _update_canvas(self):
        self.canvas.delete('media')
        scaled_width = self.media_width * self.current_pos.get('scale_x', 1.0)
        scaled_height = self.media_height * self.current_pos.get('scale_y', 1.0)
        canvas_x = self.current_pos['x'] * self.canvas_width
        canvas_y = self.current_pos['y'] * self.canvas_height
        display_width = int(scaled_width * self.canvas_scale)
        display_height = int(scaled_height * self.canvas_scale)
        if display_width > 0 and display_height > 0:
            resized = self.original_image.resize((display_width, display_height), Image.LANCZOS)
            if self.current_pos.get('rotation', 0) != 0:
                resized = resized.rotate(self.current_pos['rotation'], expand=True, resample=Image.BICUBIC)
            self.photo = ImageTk.PhotoImage(resized)
            self.canvas.create_image(canvas_x, canvas_y, image=self.photo, anchor='center', tags=('media', 'draggable'))
            handle_positions = self._get_handle_positions()
            for handle_id, pos in handle_positions.items():
                x, y = pos
                self.canvas.create_rectangle(x-4, y-4, x+4, y+4, fill='#5a5', outline='white', width=1, tags=('media', 'handle', handle_id))
    
    def _get_handle_positions(self):
        center_x = self.current_pos['x'] * self.canvas_width
        center_y = self.current_pos['y'] * self.canvas_height
        half_w = (self.media_width * self.current_pos.get('scale_x', 1.0) * self.canvas_scale) / 2
        half_h = (self.media_height * self.current_pos.get('scale_y', 1.0) * self.canvas_scale) / 2
        rad = math.radians(self.current_pos['rotation'])
        cos_rad = math.cos(rad)
        sin_rad = math.sin(rad)
        corners = {'nw': (-half_w, -half_h), 'ne': (half_w, -half_h), 'sw': (-half_w, half_h), 'se': (half_w, half_h)}
        rotated_corners = {}
        for key, (dx, dy) in corners.items():
            rx = dx * cos_rad - dy * sin_rad
            ry = dx * sin_rad + dy * cos_rad
            rotated_corners[key] = (center_x + rx, center_y + ry)
        return rotated_corners
    
    def _on_slider_change(self, value):
        self.current_pos['x'] = self.x_var.get()
        self.current_pos['y'] = self.y_var.get()
        self.current_pos['rotation'] = self.rotation_var.get()
        self.rotation_label.config(text=f"{int(self.current_pos['rotation'])}°")
        self._update_canvas()
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
            new_w = abs(event.x - anchor_x)
            new_h = abs(event.y - anchor_y)
            
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

        self._update_canvas()
        self._trigger_realtime_callback()

    def _on_mouse_up(self, event):
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
        self.current_pos = {'x': 0.5, 'y': 0.5, 'scale_x': 1.0, 'scale_y': 1.0, 'rotation': 0}
        self._update_ui_from_pos()

    def _save_and_close(self):
        if self.callback: self.callback(self.media_path, self.current_pos)
        self.window.destroy()

    def _cancel_and_close(self):
        if self.realtime_callback: self.realtime_callback(self.original_pos)
        self.window.destroy()
