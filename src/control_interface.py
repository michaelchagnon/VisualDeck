# =====================================================================
# src/control_interface.py
# =====================================================================
import os
import cv2
import time
import threading
import queue
import json
import re
import logging
import pygame
import multiprocessing
import sys
import platform
import subprocess
import tkinter as tk
from tkinter import (
    ttk, filedialog, Menu, PanedWindow, Frame, Label,
    Button, Canvas, Scrollbar, simpledialog, messagebox
)
from PIL import Image, ImageTk, ImageDraw
from utils.file_watcher import FileWatcher
from utils.tooltips import CreateToolTip
from utils.ffmpeg_utils import FFMpegOptimizer, FFMpegProxyGenerator
from utils.preview_bridge import PreviewBridge
from editor import MediaEditor
from render_engine import PygameRenderEngine, OpenGLRenderEngine
from pygame.locals import NOFRAME, DOUBLEBUF, OPENGL
from ring_buffer import SimpleFrameQueue
from ring_buffer import SimpleFrameQueue
from render_engine import PreciseFramePacer
from OpenGL.GL import glDeleteTextures
from concurrent.futures import ThreadPoolExecutor
from screeninfo import get_monitors

# Windows-specific imports
if sys.platform == "win32":
    try:
        import win32api
        import win32process
        import win32con
        WINDOWS_IMPORTS_AVAILABLE = True
    except ImportError:
        WINDOWS_IMPORTS_AVAILABLE = False
        logging.warning("pywin32 not installed. Windows optimizations will be limited.")
else:
    WINDOWS_IMPORTS_AVAILABLE = False

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
SUPPORTED_EXTS = VIDEO_EXTS + IMAGE_EXTS

SPLASH_VERSION_TEXT = "v1.0b"     # Edit to change the version label; "" disables.
SPLASH_VERSION_FONT_PT = 14        # Point size (Word-equivalent).
SPLASH_VERSION_FONT_PATH = None    # Optional full path to .ttf/.otf
SPLASH_VERSION_MARGIN_X = 16       # Horizontal margin from the right edge (px)
SPLASH_VERSION_MARGIN_Y = 16       # Vertical margin from the bottom edge (px)
VD_SHOW_VIDEO_OPTIMIZE_MENU = False      # hide "Optimize" on video right-click menu
VD_SHOW_DELETE_FROM_DISK_MENU = False    # hide "Remove and DELETE FROM DISK"

class RenderThread(threading.Thread):
    """A dedicated thread for handling all rendering operations."""
    def __init__(self, render_engine, command_queue, preview_queue, stop_event):
        super().__init__(daemon=True)
        self.re = render_engine
        self.command_queue = command_queue
        self.preview_queue = preview_queue
        self.stop_event = stop_event
        # ── Adaptive performance state ──
        self._adaptive_levels = [60, 45, 30]   # target FPS choices
        self._level_idx = 0                    # start at 60
        self._stable_ok_frames = 0
        self._stress_frames = 0
        self._raise_after = 240                # ~4s at 60fps of stability
        self._lower_after = 60                 # ~1s of stress to step down
        self._preview_cap_scale = 1.0          # 1.0 → 0.75 → 0.5 under stress
        self.current_render_state = {
            'layers': {},
            'target_monitor': None,
            'transition_state': {'active': False},
            'request_preview': True,
            'gui_monitor': 0,
            'output_mode': 'None',
            'all_assigned_screens': set(),
        }

    def run(self):
        if hasattr(self.re, '_init_pygame_and_gl'):
            self.re._init_pygame_and_gl()

        # Keep Windows priority boost as in your code
        if sys.platform == "win32" and WINDOWS_IMPORTS_AVAILABLE:
            try:
                thread_handle = win32api.GetCurrentThread()
                win32process.SetThreadPriority(thread_handle, win32process.THREAD_PRIORITY_TIME_CRITICAL)
                logging.info("Set render thread priority to TIME_CRITICAL")
            except Exception as e:
                logging.warning(f"Could not set render thread priority: {e}")

        # IMPORTANT: start clean so we don't render a stale frame before window placement is correct
        state_dirty = False
        last_render_time = 0
        min_render_interval = 1.0 / 120.0  # Cap at 120 FPS max

        # Track if we have active videos
        has_active_videos = False

        # Preview resolution caps (adaptive system can change the scale later)
        MAX_PREVIEW_WIDTH = int(640 * self._preview_cap_scale)
        MAX_PREVIEW_HEIGHT = int(360 * self._preview_cap_scale)

        while not self.stop_event.is_set():
            frame_start_time = time.perf_counter()

            if self.re.window:
                pygame.event.pump()

            # Process pending commands quickly
            commands_processed = 0
            command_deadline = frame_start_time + 0.002  # ~2ms
            try:
                while commands_processed < 10 and time.perf_counter() < command_deadline:
                    command = self.command_queue.get_nowait()
                    action = command.get('action')

                    if action == 'render':
                        if 'force_restart_paths' in command:
                            self.re.update_layers(
                                command.get('layers', {}),
                                force_restart_paths=command.get('force_restart_paths')
                            )
                        self.current_render_state.update(command)
                        state_dirty = True

                    elif action == 'update_layers':
                        self.re.update_layers(
                            command.get('layers', {}),
                            force_restart_paths=command.get('force_restart_paths')
                        )
                        state_dirty = True

                    elif action == 'request_preview':
                        self.current_render_state['request_preview'] = True
                        # Do not mark state_dirty for preview-only requests

                    elif action == 'stop':
                        self.stop_event.set()
                        break

                    commands_processed += 1
            except queue.Empty:
                pass

            if self.stop_event.is_set():
                break

            # Is any video active?
            current_layers = self.current_render_state.get('layers', {})
            has_active_videos = any(
                layer.get('path')
                and os.path.exists(layer['path'])
                and os.path.splitext(layer['path'])[1].lower() in ('.mp4', '.avi', '.mov', '.mkv')
                for layer in current_layers.values()
            )

            # Is a transition active?
            transition_active = self.current_render_state.get('transition_state', {}).get('active', False)

            # Decide whether to render this frame
            time_since_last_render = frame_start_time - last_render_time
            should_render = False

            if state_dirty:
                should_render = True
            elif has_active_videos or transition_active:
                if time_since_last_render >= (1.0 / 60.0):  # ~60 FPS for motion
                    should_render = True
            elif self.current_render_state.get('request_preview') and self.re.window:
                # Only render for preview if the output window actually exists
                should_render = True

            if should_render and time_since_last_render >= min_render_interval:
                self.re.render(
                    self.current_render_state['layers'],
                    self.current_render_state['target_monitor'],
                    transition_state=self.current_render_state['transition_state'],
                    all_assigned_screens=self.current_render_state['all_assigned_screens']
                )
                last_render_time = frame_start_time
                state_dirty = False

            # Handle preview generation (allow even if no output window exists)
            if self.current_render_state.get('request_preview'):
                preview_img = None
                output_mode = self.current_render_state.get('output_mode')
                output_monitor = self.current_render_state.get('target_monitor')
                gui_monitor = self.current_render_state.get('gui_monitor')

                # On the same monitor as the GUI? Prefer CPU path to avoid stealing the swap chain.
                force_cpu_preview = (output_monitor is not None and output_monitor == gui_monitor)

                # Try GPU readback first when available and safe
                if self.re.window and not force_cpu_preview:
                    if isinstance(self.re, OpenGLRenderEngine):
                        preview_img = self.re.read_pixels_for_preview()
                    elif isinstance(self.re, PygameRenderEngine):
                        # No fast path; fall through to CPU render
                        pass

                if preview_img is None:
                    layers = self.current_render_state.get('layers', {})
                    ts = self.current_render_state.get('transition_state')

                    preview_target_monitor = 0 if output_mode == 'Multiple' else output_monitor

                    # Recompute caps (adaptive scale may have changed)
                    MAX_PREVIEW_WIDTH = int(640 * self._preview_cap_scale)
                    MAX_PREVIEW_HEIGHT = int(360 * self._preview_cap_scale)

                    if preview_target_monitor is not None and self.re.monitors:
                        mon = self.re.monitors[preview_target_monitor]
                        original_width, original_height = mon.width, mon.height
                        scale_factor = min(
                            MAX_PREVIEW_WIDTH / original_width,
                            MAX_PREVIEW_HEIGHT / original_height,
                            1.0
                        )
                        dims = (int(original_width * scale_factor), int(original_height * scale_factor))

                        if isinstance(self.re, PygameRenderEngine):
                            def scale_layer_for_preview(layer):
                                layer_copy = layer.copy()
                                if 'position' in layer_copy:
                                    pos = layer_copy['position'].copy()
                                    pos['scale_x'] = pos.get('scale_x', 1.0) * scale_factor
                                    pos['scale_y'] = pos.get('scale_y', 1.0) * scale_factor
                                    layer_copy['position'] = pos
                                return layer_copy

                            if output_mode == 'Multiple':
                                layers_for_preview = {
                                    k: scale_layer_for_preview(v) for k, v in layers.items()
                                    if preview_target_monitor in v.get('screens', [])
                                }
                                from_layers_for_preview = {
                                    k: scale_layer_for_preview(v) for k, v in ts.get('from', {}).items()
                                    if preview_target_monitor in v.get('screens', [])
                                }
                                to_layers_for_preview = {
                                    k: scale_layer_for_preview(v) for k, v in ts.get('to', {}).items()
                                    if preview_target_monitor in v.get('screens', [])
                                }
                            else:
                                layers_for_preview = {k: scale_layer_for_preview(v) for k, v in layers.items()}
                                from_layers_for_preview = {k: scale_layer_for_preview(v) for k, v in ts.get('from', {}).items()}
                                to_layers_for_preview = {k: scale_layer_for_preview(v) for k, v in ts.get('to', {}).items()}
                        else:
                            # OpenGL path; don't pre-scale layer positions
                            if output_mode == 'Multiple':
                                layers_for_preview = {k: v for k, v in layers.items()
                                                      if preview_target_monitor in v.get('screens', [])}
                                from_layers_for_preview = {k: v for k, v in ts.get('from', {}).items()
                                                           if preview_target_monitor in v.get('screens', [])}
                                to_layers_for_preview = {k: v for k, v in ts.get('to', {}).items()
                                                         if preview_target_monitor in v.get('screens', [])}
                            else:
                                layers_for_preview = layers
                                from_layers_for_preview = ts.get('from', {})
                                to_layers_for_preview = ts.get('to', {})
                    else:
                        # Fallback dims if monitor info isn’t available yet
                        scale_factor = min(MAX_PREVIEW_WIDTH / 1920, MAX_PREVIEW_HEIGHT / 1080, 1.0)
                        dims = (int(1920 * scale_factor), int(1080 * scale_factor))

                        if isinstance(self.re, PygameRenderEngine):
                            def scale_layer_for_preview(layer):
                                layer_copy = layer.copy()
                                if 'position' in layer_copy:
                                    pos = layer_copy['position'].copy()
                                    pos['scale_x'] = pos.get('scale_x', 1.0) * scale_factor
                                    pos['scale_y'] = pos.get('scale_y', 1.0) * scale_factor
                                    layer_copy['position'] = pos
                                return layer_copy

                            layers_for_preview = {k: scale_layer_for_preview(v) for k, v in layers.items()}
                            from_layers_for_preview = {k: scale_layer_for_preview(v) for k, v in ts.get('from', {}).items()}
                            to_layers_for_preview = {k: scale_layer_for_preview(v) for k, v in ts.get('to', {}).items()}
                        else:
                            layers_for_preview = layers
                            from_layers_for_preview = ts.get('from', {})
                            to_layers_for_preview = ts.get('to', {})

                    if ts and ts.get('active'):
                        from_img = self.re.render_layers_to_image(from_layers_for_preview, dims)
                        to_img = self.re.render_layers_to_image(to_layers_for_preview, dims)
                        if from_img and to_img:
                            preview_img = Image.blend(from_img.convert('RGBA'), to_img.convert('RGBA'), ts['progress'])
                        elif to_img:
                            preview_img = to_img
                        else:
                            preview_img = from_img
                    else:
                        preview_img = self.re.render_layers_to_image(layers_for_preview, dims)

                try:
                    if not self.preview_queue.empty():
                        self.preview_queue.get_nowait()
                    self.preview_queue.put_nowait(preview_img)
                except queue.Full:
                    pass

                self.current_render_state['request_preview'] = False

            # Frame pacing + adaptive tuning
            if not hasattr(self, 'frame_pacer'):
                from render_engine import PreciseFramePacer
                self.frame_pacer = PreciseFramePacer(target_fps=self._adaptive_levels[self._level_idx])
                self.static_frame_pacer = PreciseFramePacer(target_fps=30)

            if has_active_videos or transition_active:
                ft = self.frame_pacer.wait_for_next_frame()
                self._adaptive_update(ft, active_content=True)
            else:
                ft = self.static_frame_pacer.wait_for_next_frame()
                self._adaptive_update(ft, active_content=False)

        logging.info("Render thread shutting down.")
        self.re.clear()
        pygame.quit()


    def _adaptive_update(self, last_frame_time: float, active_content: bool):
        """
        Adjust target FPS and preview cap scale based on load.
        - If we overrun too often, step down (60→45→30) and also shrink preview cap.
        - If we’re stable for a while, step back up and restore preview.
        """
        # Decide which pacer we’re using this frame
        pacer = self.frame_pacer if active_content else self.static_frame_pacer

        # Overrun signal from pacer
        overrun_ratio = pacer.overrun_ratio()
        target_fps = pacer.get_target_fps()

        # Basic stability / stress counters
        if overrun_ratio > 0.25:   # >25% frames missing their budget
            self._stress_frames += 1
            self._stable_ok_frames = 0
        else:
            self._stable_ok_frames += 1
            self._stress_frames = 0

        # Step down if sustained stress
        if self._stress_frames >= self._lower_after and self._level_idx < (len(self._adaptive_levels) - 1):
            self._level_idx += 1
            new_fps = self._adaptive_levels[self._level_idx]
            self.frame_pacer.set_target_fps(new_fps)
            # tighten preview caps when we step down (up to 0.5x)
            if self._preview_cap_scale > 0.5:
                self._preview_cap_scale = 0.75 if self._preview_cap_scale == 1.0 else 0.5
            logging.info(f"[Adaptive] Load high -> lowering target FPS to {new_fps}, preview_scale={self._preview_cap_scale}")
            self._stress_frames = 0  # reset

        # Step up if sustained stability (only when active content)
        elif active_content and self._stable_ok_frames >= self._raise_after and self._level_idx > 0:
            self._level_idx -= 1
            new_fps = self._adaptive_levels[self._level_idx]
            self.frame_pacer.set_target_fps(new_fps)
            # relax preview caps as we step up
            if self._preview_cap_scale < 1.0:
                self._preview_cap_scale = 1.0 if self._preview_cap_scale == 0.75 else 0.75
            logging.info(f"[Adaptive] Stable -> raising target FPS to {new_fps}, preview_scale={self._preview_cap_scale}")
            self._stable_ok_frames = 0  # reset

class TransitionThread(threading.Thread):
    """Dedicated thread for handling transition timing and updates."""
    def __init__(self, control_interface):
        super().__init__(daemon=True)
        self.ci = control_interface
        self.stop_event = threading.Event()
        self.transition_event = threading.Event()
        self.lock = threading.Lock()
        
        # Transition parameters (set when starting a transition)
        self.from_layers = {}
        self.to_layers = {}
        self.duration = 1.0
        self.target_col = 0
        self.start_time = 0
        
    def start_transition(self, from_layers, to_layers, duration, target_col):
        """Start a new transition with the given parameters."""
        with self.lock:
            self.from_layers = from_layers
            self.to_layers = to_layers
            self.duration = duration
            self.target_col = target_col
            self.start_time = time.perf_counter()
            self.transition_event.set()
    
    def stop_transition(self):
        """Stop the current transition."""
        self.transition_event.clear()
        
    def shutdown(self):
        """Shutdown the thread."""
        self.stop_event.set()
        self.transition_event.set()  # Wake up if sleeping
        
    def run(self):
        """Main thread loop for handling transitions."""
        try:
            frame_pacer = PreciseFramePacer(target_fps=60)  # 60 FPS for smooth transitions
            
            while not self.stop_event.is_set():
                # Wait for a transition to start
                self.transition_event.wait(timeout=0.1)
                
                if self.stop_event.is_set():
                    break
                    
                if not self.transition_event.is_set():
                    continue
                
                # Transition is active
                with self.lock:
                    start_time = self.start_time
                    duration = self.duration
                    from_layers = self.from_layers.copy()
                    to_layers = self.to_layers.copy()
                    target_col = self.target_col
                
                # Run transition loop
                last_progress = -1
                while self.transition_event.is_set() and not self.stop_event.is_set():
                    elapsed = time.perf_counter() - start_time
                    progress = min(elapsed / duration, 1.0)
                    
                    # Only send update if progress changed meaningfully
                    if abs(progress - last_progress) >= 0.01 or progress >= 1.0:
                        try:
                            # Send transition state to render thread
                            transition_state = {
                                'active': True,
                                'progress': progress,
                                'from': from_layers,
                                'to': to_layers
                            }
                            
                            output_mode = self.ci.screen_var.get()
                            target_monitor = self.ci.selected_display if output_mode != self.ci.MULTIPLE else None
                            
                            all_screens = set()
                            if output_mode == self.ci.MULTIPLE:
                                all_screens = {s for s in self.ci.layer_screens.values() if s is not None}
                            
                            self.ci.render_command_queue.put({
                                'action': 'render',
                                'layers': {},
                                'target_monitor': target_monitor,
                                'transition_state': transition_state,
                                'gui_monitor': self.ci.gui_monitor_index,
                                'output_mode': output_mode,
                                'all_assigned_screens': all_screens,
                            })
                            
                            last_progress = progress
                        except Exception as e:
                            logging.error(f"Error sending transition update: {e}")
                    
                    # Check if transition is complete
                    if progress >= 1.0:
                        # Transition complete - notify main thread
                        try:
                            self.ci.root.after_idle(lambda: self.ci._finalize_transition_from_thread(target_col))
                        except Exception as e:
                            logging.error(f"Error finalizing transition: {e}")
                        self.transition_event.clear()
                        break
                    
                    # Precise frame timing
                    frame_pacer.wait_for_next_frame()
                    
        except Exception as e:
            logging.error(f"Transition thread error: {e}", exc_info=True)

class ControlInterface:
    NONE = 'None'
    MULTIPLE = 'Multiple'
    DEFAULT_ROWS = 5
    DEFAULT_COLS = 7
    CELL_SIZE = 150
    CELL_PAD = 4
    HIGHLIGHT_BG = '#555'
    DEFAULT_BG = '#333'
    CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.visualdeck')
    CONFIG_FILENAME = 'config.json'

    def __init__(self, media_manager):
        self.mm = media_manager
        self.re = None
        self.render_mode = 'cpu'
        self.hw_accel = 'auto'
        self.use_pbo = True
        self.use_vsync = False # Default to off for lower latency
        self.force_multithread = True
        self.proxy_executor = ThreadPoolExecutor(max_workers=2)
        
        # --- UI readiness gating for splash ---
        self._ui_ready = False
        self._deferred_trigger = False
        self._startup_inhibit_triggers = True
        self._startup_render_sent = False
        
        self.render_thread = None
        self.render_command_queue = queue.Queue()
        self.render_preview_queue = queue.Queue(maxsize=1)
        self.render_stop_event = threading.Event()
        self.transition_thread = None

        self.frac1, self.frac2 = 0.25, 0.75
        self.drag_type = None
        self.drag_data = None
        self.drag_widget = None
        self._hover_label = None
        self.selected_widget = None
        self.frames_by_position = {}
        self._cell_frames = {}
        self.drag_motion_detected = False # MODIFICATION: Add flag to track drag state.
        self._resize_timer = None
        
        # --- Project State ---
        self.current_project_path = None
        self.project_is_dirty = False
        self.recent_files = []
        os.makedirs(self.CONFIG_DIR, exist_ok=True)
        self.config_path = os.path.join(self.CONFIG_DIR, self.CONFIG_FILENAME)
        self._reset_project_state() # Initialize with default state
        
        self.is_video_playing = False
        self.preview_dirty = True

        self.thumbnail_cache = {}
        self.thumbnail_queue = queue.Queue()
        self.media_list_labels = {}
        self.thumbnail_thread_lock = threading.Lock()
        self.thumbnail_thread = None
        
        # Presentation Mode State
        self.presentation_active = False
        self.presentation_timer = None

        # FileWatcher is now started/stopped dynamically when projects are loaded
        self.file_watcher_thread = None
        self.file_watcher_observer = None

        self._start_gui()

    def _start_cue_reorder_drag(self, col: int) -> None:
        """Begin a cue header drag; plain click will select in _on_cue_reorder_drop."""
        self.drag_motion_detected = False
        self.drag_type = 'cue_reorder'
        self.drag_data = {'col': col}
        name = self.cue_names.get(str(col), f'Cue {col+1}')
        # lightweight ghost follows mouse via _on_drag_motion
        self.drag_widget = Label(self.root, text=name, bg='#444', fg='white', bd=1, relief='solid')
        self.root.config(cursor='exchange')
        self.root.bind('<B1-Motion>', self._on_drag_motion)
        self.root.bind('<ButtonRelease-1>', self._on_cue_reorder_drop)

    def _on_cue_reorder_drop(self, event) -> None:
            src = self.drag_data.get('col')
            # No movement: preserve original single-click selection behavior.
            if not self.drag_motion_detected:
                try:
                    self._select_column(src)
                    # Immediately reflect selection in the cue header highlight (matches GO behavior)
                    for c, lbl in self.col_labels.items():
                        try:
                            lbl.config(bg=('#ADD8E6' if c == src else '#444'))
                        except tk.TclError:
                            pass
                finally:
                    self._cleanup_drag()
                return
            dst = self._get_col_under(event.x_root, event.y_root)
            self._cleanup_drag()
            if dst is None or dst == src:
                return
            self._reorder_cues(src, dst)


    def _reorder_cues(self, src: int, dst: int) -> None:
        """Stable remap of cue columns and associated data structures."""
        if not (0 <= src < self.COLS and 0 <= dst < self.COLS) or src == dst:
            return
        new_cells, new_positions = {}, {}

        # Remap cells by column
        for (r, c), media in list(self.grid_cells.items()):
            if c == src:
                nc = dst
            elif dst > src and src < c <= dst:
                nc = c - 1
            elif dst < src and dst <= c < src:
                nc = c + 1
            else:
                nc = c
            new_cells[(r, nc)] = media
            if (r, c) in self.cell_positions:
                pos = self.cell_positions[(r, c)]
                new_positions[(r, nc)] = pos.copy() if isinstance(pos, dict) else pos

        # Remap cue names
        new_names = {}
        for c_str, nm in self.cue_names.items():
            c = int(c_str)
            if c == src:
                nc = dst
            elif dst > src and src < c <= dst:
                nc = c - 1
            elif dst < src and dst <= c < src:
                nc = c + 1
            else:
                nc = c
            new_names[str(nc)] = nm

        # Active column
        ac = self.active_col
        if ac == src:
            ac = dst
        elif dst > src and src < ac <= dst:
            ac = ac - 1
        elif dst < src and dst <= ac < src:
            ac = ac + 1

        # Commit
        self.grid_cells = new_cells
        self.cell_positions = new_positions
        self.cue_names = new_names
        self.active_col = ac

        # Invalidate caches only across affected span
        if self.re:
            lo, hi = sorted((src, dst))
            for c in range(lo, hi + 1):
                try:
                    self.re.invalidate_static_cache_for_column(c)
                except Exception:
                    pass

        # Rebuild visuals once and update state
        self._build_grid()
        self._select_column(self.active_col)
        self._update_cue_dropdown()
        self._trigger_column()
        self._set_dirty_flag()


    def _get_col_under(self, x_root: int, y_root: int):
        """Hit-test cue headers to find target column."""
        for c, lbl in self.col_labels.items():
            try:
                lx, ly = lbl.winfo_rootx(), lbl.winfo_rooty()
                lw, lh = lbl.winfo_width(), lbl.winfo_height()
                if lx <= x_root <= lx + lw and ly <= y_root <= ly + lh:
                    return c
            except tk.TclError:
                pass
        return None

    def _start_layer_reorder_drag(self, row: int) -> None:
        """Begin a layer header drag (left click)."""
        self.drag_motion_detected = False
        self.drag_type = 'layer_reorder'
        self.drag_data = {'row': row}
        name = self.layer_names.get(str(row), f'Layer {row+1}')
        self.drag_widget = Label(self.root, text=name, bg='#444', fg='white', bd=1, relief='solid')
        self.root.config(cursor='exchange')
        self.root.bind('<B1-Motion>', self._on_drag_motion)
        self.root.bind('<ButtonRelease-1>', self._on_layer_reorder_drop)

    def _on_layer_reorder_drop(self, event) -> None:
        src = self.drag_data.get('row')
        if not self.drag_motion_detected:
            self._cleanup_drag()
            return
        dst = self._get_row_under(event.x_root, event.y_root)
        self._cleanup_drag()
        if dst is None or dst == src:
            return
        self._reorder_layers(src, dst)

    def _reorder_layers(self, src: int, dst: int) -> None:
        """Stable remap of layer rows and associated data structures."""
        if not (0 <= src < self.ROWS and 0 <= dst < self.ROWS) or src == dst:
            return
        new_cells, new_positions = {}, {}

        for (r, c), media in list(self.grid_cells.items()):
            if r == src:
                nr = dst
            elif dst > src and src < r <= dst:
                nr = r - 1
            elif dst < src and dst <= r < src:
                nr = r + 1
            else:
                nr = r
            new_cells[(nr, c)] = media
            if (r, c) in self.cell_positions:
                # IMPORTANT: copy to avoid shared references that can corrupt transforms/scaling
                pos = self.cell_positions[(r, c)]
                new_positions[(nr, c)] = pos.copy() if isinstance(pos, dict) else pos

        # Remap layer names
        new_layer_names = {}
        for r_str, nm in self.layer_names.items():
            r = int(r_str)
            if r == src:
                nr = dst
            elif dst > src and src < r <= dst:
                nr = r - 1
            elif dst < src and dst <= r < src:
                nr = r + 1
            else:
                nr = r
            new_layer_names[str(nr)] = nm

        # Remap layer->screen assignments
        new_layer_screens = {}
        for r_str, scr in self.layer_screens.items():
            r = int(r_str)
            if r == src:
                nr = dst
            elif dst > src and src < r <= dst:
                nr = r - 1
            elif dst < src and dst <= r < src:
                nr = r + 1
            else:
                nr = r
            new_layer_screens[str(nr)] = scr

        # Commit
        self.grid_cells = new_cells
        self.cell_positions = new_positions
        self.layer_names = new_layer_names
        self.layer_screens = new_layer_screens

        # Invalidate all columns because layer reordering changes per-column z-order and cached transforms
        if self.re:
            for c in range(self.COLS):
                try:
                    self.re.invalidate_static_cache_for_column(c)
                except Exception:
                    pass  # keep compatibility if renderer lacks column caches

        # Rebuild visuals and re-trigger active column to refresh preview/output
        self._build_grid()
        self._select_column(self.active_col)
        self._trigger_column()
        self._set_dirty_flag()


    def _get_row_under(self, x_root: int, y_root: int):
        """Hit-test layer labels to find target row."""
        for r, canvas in self.row_labels.items():
            try:
                lx, ly = canvas.winfo_rootx(), canvas.winfo_rooty()
                lw, lh = canvas.winfo_width(), canvas.winfo_height()
                if lx <= x_root <= lx + lw and ly <= y_root <= ly + lh:
                    return r
            except tk.TclError:
                pass
        return None

    def _apply_windows_optimizations(self):
        """Apply Windows-specific performance optimizations."""
        if sys.platform != "win32":
            return
            
        if WINDOWS_IMPORTS_AVAILABLE:
            try:
                # Get current process handle
                handle = win32api.GetCurrentProcess()
                
                # Set process priority to HIGH (one step below REALTIME)
                # REALTIME can cause system instability, so we use HIGH
                win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)
                logging.info("Set process priority to HIGH_PRIORITY_CLASS")
                
                # Enable multi-threaded performance optimizations
                # This tells Windows we're a multimedia application
                win32process.SetProcessPriorityBoost(handle, False)  # Disable priority boost throttling
                
            except Exception as e:
                logging.warning(f"Could not set Windows process priority: {e}")
        
        # Set timer resolution for more accurate sleep (improves frame timing)
        try:
            import ctypes
            winmm = ctypes.WinDLL('winmm')
            winmm.timeBeginPeriod(1)  # 1ms timer resolution
            logging.info("Set Windows timer resolution to 1ms")
        except Exception as e:
            logging.warning(f"Could not set timer resolution: {e}")

    def _cleanup_transition_links(self, keep_paths=frozenset()):
        """
        Remove any temporary .vd_link files we created for transitions,
        except those explicitly listed in keep_paths. If a file can't be
        deleted (e.g., still open), schedule a retry. Also evict entries
        from the reuse index for any deleted paths. Registers atexit cleanup once.
        Additionally, sweep the app-local vd_temp directory for stray .vd_link_* files.
        """
        import os, atexit

        leftovers = []
        keep = set(keep_paths or ())

        # Register final cleanup once
        if not getattr(self, "_vd_atexit_registered", False):
            try:
                atexit.register(lambda: self._cleanup_transition_links_blocking())
                self._vd_atexit_registered = True
            except Exception:
                pass

        # Remove tracked links not in keep
        for attr in ('_active_transition_links', '_temp_transition_links'):
            links = list(getattr(self, attr, []) or [])
            retained = []
            for p in links:
                if not p or p in keep:
                    retained.append(p)
                    continue
                try:
                    if os.path.exists(p):
                        os.remove(p)
                    # If it doesn't exist, treat as cleaned.
                except Exception:
                    leftovers.append(p)  # Could be in use; try again later.
                    retained.append(p)
            setattr(self, attr, retained)

        # Evict from deterministic index if not kept
        idx = getattr(self, "_vd_link_index", {}) or {}
        if idx:
            to_del_keys = []
            for k, p in list(idx.items()):
                if not p:
                    to_del_keys.append(k)
                    continue
                if p in keep:
                    continue
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    leftovers.append(p)
                    continue
                to_del_keys.append(k)
            for k in to_del_keys:
                idx.pop(k, None)

        # Sweep the app-local temp dir for any stray .vd_link_* not in keep
        try:
            temp_dir = self._get_vd_temp_dir()
            for name in os.listdir(temp_dir):
                if not name.startswith(".vd_link_"):
                    continue
                p = os.path.join(temp_dir, name)
                if p in keep:
                    continue
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    leftovers.append(p)
        except Exception:
            pass

        # Retry later if any were in use
        if leftovers:
            try:
                self.root.after(2000, lambda: self._cleanup_transition_links(keep_paths))
            except Exception:
                pass

    def _get_vd_temp_dir(self):
        """
        Return the path to the app-local temp directory for transition links,
        creating it if necessary. Lives alongside the application executable.
        """
        import os, sys
        # Cache the app dir to avoid recomputing
        app_dir = getattr(self, "_app_dir", None)
        if not app_dir:
            try:
                # If bundled (e.g., PyInstaller), _MEIPASS may exist
                base = getattr(sys, "_MEIPASS", None)
                if base:
                    app_dir = base
                else:
                    app_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            except Exception:
                app_dir = os.getcwd()
            self._app_dir = app_dir

        temp_dir = os.path.join(app_dir, "vd_temp")
        try:
            os.makedirs(temp_dir, exist_ok=True)
        except Exception:
            # Last resort: fall back to CWD if app dir is not writable
            temp_dir = os.path.join(os.getcwd(), "vd_temp")
            os.makedirs(temp_dir, exist_ok=True)
        return temp_dir


    def _get_or_create_vd_link(self, orig_path, lid):
        """
        Create or reuse a deterministic .vd_link for (orig_path, lid) inside the app-local temp dir.
        Returns link_path or None on failure.
        """
        import os, hashlib, shutil, logging, sys, stat

        if not orig_path:
            return None

        temp_dir = self._get_vd_temp_dir()

        # index: {(orig_path, lid): link_path}
        idx = getattr(self, "_vd_link_index", None)
        if idx is None:
            idx = self._vd_link_index = {}

        key = (orig_path, str(lid))
        root, ext = os.path.splitext(os.path.basename(orig_path))
        digest = hashlib.md5(orig_path.encode("utf-8")).hexdigest()[:8]
        link_name = f".vd_link_{lid}_{digest}{ext}"
        link_path = os.path.join(temp_dir, link_name)

        # Reuse if exists
        if os.path.exists(link_path):
            idx[key] = link_path
            return link_path

        # Attempt hardlink if on same device/drive, else fall back to copy
        can_hardlink = True
        try:
            if os.name == "nt":
                # Compare drive letters on Windows
                src_drive = os.path.splitdrive(os.path.abspath(orig_path))[0].lower()
                dst_drive = os.path.splitdrive(os.path.abspath(link_path))[0].lower()
                if src_drive != dst_drive:
                    can_hardlink = False
            else:
                # Compare st_dev on POSIX
                src_dev = os.stat(orig_path).st_dev
                dst_dev = os.stat(temp_dir).st_dev
                if src_dev != dst_dev:
                    can_hardlink = False
        except Exception:
            # If we can't determine, try hardlink and fall back on failure
            can_hardlink = True

        if can_hardlink:
            try:
                os.link(orig_path, link_path)
            except Exception as e:
                logging.warning(f"Hardlink failed for '{orig_path}' → '{link_path}': {e}; falling back to copy2.")
                try:
                    shutil.copy2(orig_path, link_path)
                except Exception as e2:
                    logging.error(f"Copy fallback failed for '{orig_path}' → '{link_path}': {e2}")
                    return None
        else:
            try:
                shutil.copy2(orig_path, link_path)
            except Exception as e2:
                logging.error(f"Copy failed for '{orig_path}' → '{link_path}': {e2}")
                return None

        idx[key] = link_path
        return link_path

    def _cleanup_transition_links_blocking(self, retries=6, delay=0.25):
        """
        Best-effort final cleanup for any remaining .vd_link files.
        Runs on interpreter exit via atexit and can be called explicitly.
        """
        import os, time

        # Gather known temp links
        seen = set()
        to_try = []

        for attr in ('_active_transition_links', '_temp_transition_links'):
            for p in list(getattr(self, attr, []) or []):
                if p and p not in seen:
                    seen.add(p)
                    to_try.append(p)

        # Also include any .vd_link_* in the app-local temp directory
        try:
            temp_dir = self._get_vd_temp_dir()
            for name in os.listdir(temp_dir):
                if name.startswith(".vd_link_"):
                    p = os.path.join(temp_dir, name)
                    if p not in seen:
                        seen.add(p)
                        to_try.append(p)
        except Exception:
            pass

        # Try multiple times in case other threads close handles slightly later
        for _ in range(max(1, int(retries))):
            remaining = []
            for p in to_try:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    remaining.append(p)
            if not remaining:
                break
            to_try = remaining
            try:
                time.sleep(max(0.05, float(delay)))
            except Exception:
                break

        # Clear tracked lists to avoid reattempts next run
        try:
            self._active_transition_links = []
            self._temp_transition_links = []
        except Exception:
            pass

    def _reset_project_state(self, new_project=True):
        """Resets all project-specific variables to their defaults."""
        self.ROWS = self.DEFAULT_ROWS
        self.COLS = self.DEFAULT_COLS
        self.project_media_paths = []
        self.grid_cells = {(r, c): None for r in range(self.ROWS) for c in range(self.COLS)}
        self.cell_positions = {}
        self.active_col = 0
        self.selected_display = None
        self.header_height = 0
        self.cue_names = {str(i): f'Cue {i+1}' for i in range(self.COLS)}
        self.layer_names = {str(i): f'Layer {i+1}' for i in range(self.ROWS)}
        self.gui_monitor_index = 0
        self.layer_screens = {}
        
        if new_project:
            self.current_project_path = None
            self.project_is_dirty = False
            
        self.transition_active = False
        self.transition_start_time = 0
        self.transition_duration = 1.0
        self.transition_from_layers = {}
        self.transition_to_layers = {}
        self.transition_target_col = 0

        def _start_gui(self):
            multiprocessing.freeze_support()

            self.root = tk.Tk()
            self.root.configure(bg='#2e2e2e')
            self.root.protocol("WM_DELETE_WINDOW", self._on_exit)
            self._update_title()

            # Hide main UI until ready
            self.root.withdraw()

            # Apply Windows optimizations before mainloop
            self._apply_windows_optimizations()

            # Set application icon
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            icon_path = os.path.join(base_dir, 'src', 'visualdeck.ico')
            if os.path.exists(icon_path):
                try:
                    self.root.iconbitmap(icon_path)
                except tk.TclError:
                    logging.warning(f"Could not load icon at '{icon_path}'.")
            else:
                logging.warning(f"Icon file not found at '{icon_path}'")

            # --- Splash screen (PNG) ---
            self._splash_win = None
            self._splash_img = None
            splash_path = os.path.join(base_dir, 'src', 'splash.png')
            try:
                if os.path.exists(splash_path):
                    self._splash_win = tk.Toplevel(self.root)
                    self._splash_win.overrideredirect(True)
                    try:
                        self._splash_win.attributes('-topmost', True)
                    except Exception:
                        pass
                    # Load and (optionally) annotate the splash image with version text
                    img = Image.open(splash_path).convert("RGBA")
                    try:
                        if isinstance(SPLASH_VERSION_TEXT, str) and SPLASH_VERSION_TEXT.strip():
                            from PIL import ImageDraw, ImageFont
                            draw = ImageDraw.Draw(img)
                            font = ImageFont.load_default()
                            text = SPLASH_VERSION_TEXT.strip()
                            # Measure text
                            try:
                                bbox = draw.textbbox((0, 0), text, font=font)
                                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                            except Exception:
                                tw, th = draw.textsize(text, font=font)
                            # Bottom-right with small margin
                            margin_x, margin_y = 12, 8
                            x = max(0, img.width - margin_x - tw)
                            y = max(0, img.height - margin_y - th)
                            draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
                    except Exception:
                        logging.exception("Failed to render version text on splash image.")

                    self._splash_img = ImageTk.PhotoImage(img)
                    lbl = Label(self._splash_win, image=self._splash_img, borderwidth=0, highlightthickness=0, bg='#000')
                    lbl.pack()
                    sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
                    w, h = self._splash_img.width(), self._splash_img.height()
                    x, y = max(0, (sw - w)//2), max(0, (sh - h)//2)
                    self._splash_win.geometry(f"{w}x{h}+{x}+{y}")
                else:
                    logging.warning(f"Splash image not found at '{splash_path}'")
            except Exception:
                logging.exception("Failed to show splash screen.")
                try:
                    if self._splash_win:
                        self._splash_win.destroy()
                except Exception:
                    pass
                self._splash_win = None
            self._splash_started = time.perf_counter()

            # ---- Build UI (still hidden) ----
            self.root.geometry('1200x700')
            self._load_config()
            self._build_menu()

            self.pan = PanedWindow(self.root, orient=tk.HORIZONTAL)
            self.pan.pack(fill='both', expand=True)
            self.media_frame = Frame(self.pan, bg='#2e2e2e')
            self.grid_frame = Frame(self.pan, bg='#2e2e2e')
            self.preview_frame = Frame(self.pan, bg='#2e2e2e')
            self.pan.add(self.media_frame)
            self.pan.add(self.grid_frame)
            self.pan.add(self.preview_frame)

            self._build_media_list()
            self._build_grid()
            self._build_preview_panel()

            self.pan.bind('<ButtonRelease-1>', self._update_fracs)
            self.root.bind('<Configure>', self._on_gui_configure)
            self.root.update_idletasks()
            self._apply_fracs()

            self.root.bind_all('<MouseWheel>', self._on_mousewheel, add='+')
            self.root.bind_all('<Shift-MouseWheel>', self._on_shift_mousewheel, add='+')
    
    def _start_gui(self):
        multiprocessing.freeze_support()

        self.root = tk.Tk()
        self.root.configure(bg='#2e2e2e')
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)
        self._update_title()

        # Hide main UI until ready
        self.root.withdraw()

        # Apply Windows optimizations before mainloop
        self._apply_windows_optimizations()

        # Set application icon
        # CHANGED: resolve base_dir via sys._MEIPASS when bundled
        base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        icon_path = os.path.join(base_dir, 'src', 'visualdeck.ico')
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except tk.TclError:
                logging.warning(f"Could not load icon at '{icon_path}'.")
        else:
            logging.warning(f"Icon file not found at '{icon_path}'")

        # --- Splash screen (PNG) ---
        self._splash_win = None
        self._splash_img = None
        splash_path = os.path.join(base_dir, 'src', 'splash.png')
        try:
            if os.path.exists(splash_path):
                self._splash_win = tk.Toplevel(self.root)
                self._splash_win.overrideredirect(True)
                try:
                    self._splash_win.attributes('-topmost', True)
                except Exception:
                    pass
                # NEW: give splash the same icon (taskbar/tray consistency)
                try:
                    if os.path.exists(icon_path):
                        self._splash_win.iconbitmap(icon_path)
                except Exception:
                    pass

                img = Image.open(splash_path).convert("RGBA")

                # Render version text (bottom-right, white) if configured
                try:
                    if isinstance(SPLASH_VERSION_TEXT, str) and SPLASH_VERSION_TEXT.strip():
                        from PIL import ImageDraw, ImageFont
                        draw = ImageDraw.Draw(img)

                        # Resolve font path
                        font_path = None
                        if SPLASH_VERSION_FONT_PATH and os.path.isfile(SPLASH_VERSION_FONT_PATH):
                            font_path = SPLASH_VERSION_FONT_PATH
                        else:
                            candidates = []
                            if sys.platform == "win32":
                                candidates += [
                                    r"C:\Windows\Fonts\segoeui.ttf",
                                    r"C:\Windows\Fonts\arial.ttf",
                                    r"C:\Windows\Fonts\tahoma.ttf",
                                    r"C:\Windows\Fonts\calibri.ttf",
                                ]
                            else:
                                candidates += [
                                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                                    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                                    "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
                                    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
                                ]
                            for p in candidates:
                                if os.path.isfile(p):
                                    font_path = p
                                    break

                        # Convert pt -> px assuming 96 DPI
                        font_px = max(19, int(round((SPLASH_VERSION_FONT_PT or 14) * 96 / 72.0)))

                        if font_path:
                            font = ImageFont.truetype(font_path, font_px)
                        else:
                            font = ImageFont.load_default()

                        text = SPLASH_VERSION_TEXT.strip()

                        # Measure text
                        try:
                            bbox = draw.textbbox((0, 0), text, font=font)
                            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        except Exception:
                            tw, th = draw.textsize(text, font=font)

                        # Bottom-right with configurable margins
                        margin_x = int(SPLASH_VERSION_MARGIN_X or 0)
                        margin_y = int(SPLASH_VERSION_MARGIN_Y or 0)
                        x = max(0, img.width - margin_x - tw)
                        y = max(0, img.height - margin_y - th)

                        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
                except Exception:
                    logging.exception("Failed to render version text on splash image.")

                self._splash_img = ImageTk.PhotoImage(img)
                lbl = Label(self._splash_win, image=self._splash_img, borderwidth=0, highlightthickness=0, bg='#000')
                lbl.pack()

                sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
                w, h = self._splash_img.width(), self._splash_img.height()
                x, y = max(0, (sw - w)//2), max(0, (sh - h)//2)
                self._splash_win.geometry(f"{w}x{h}+{x}+{y}")
            else:
                logging.warning(f"Splash image not found at '{splash_path}'")
        except Exception:
            logging.exception("Failed to show splash screen.")
            try:
                if self._splash_win:
                    self._splash_win.destroy()
            except Exception:
                pass
            self._splash_win = None
        self._splash_started = time.perf_counter()

        # ---- Build UI (still hidden) ----
        self.root.geometry('1200x700')
        self._load_config()
        self._build_menu()

        self.pan = PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.pan.pack(fill='both', expand=True)
        self.media_frame = Frame(self.pan, bg='#2e2e2e')
        self.grid_frame = Frame(self.pan, bg='#2e2e2e')
        self.preview_frame = Frame(self.pan, bg='#2e2e2e')
        self.pan.add(self.media_frame)
        self.pan.add(self.grid_frame)
        self.pan.add(self.preview_frame)

        self._build_media_list()
        self._build_grid()
        self._build_preview_panel()

        self.pan.bind('<ButtonRelease-1>', self._update_fracs)
        self.root.bind('<Configure>', self._on_gui_configure)
        self.root.update_idletasks()
        self._apply_fracs()

        self.root.bind_all('<MouseWheel>', self._on_mousewheel, add='+')
        self.root.bind_all('<Shift-MouseWheel>', self._on_shift_mousewheel, add='+')

        # Defer render engine init until splash completion
        self._select_column(0)

        # Queues/loops (safe while hidden)
        self._process_thumbnail_queue()
        self._process_preview_queue()
        self._gui_animation_loop()

        # Auto-load last project (safe; triggers are gated)
        self.root.after(100, self._auto_load_last_project)

        # --- Finish splash after ≥1.5s; init render; wait for thread & monitors; single first render; then show UI ---
        self._re_init_started = False

        def _maybe_finish_startup():
            min_elapsed = (time.perf_counter() - getattr(self, "_splash_started", 0.0)) >= 1.5

            if min_elapsed and not self._re_init_started and (self.render_thread is None or not self.render_thread.is_alive()):
                self._re_init_started = True
                try:
                    self._initialize_render_engine()
                except Exception:
                    logging.exception("Render engine init failed during startup.")
                    self._re_init_started = False

            rt_ok = (self.render_thread is not None and self.render_thread.is_alive())
            mon_ready = bool(self.re and getattr(self.re, "monitors", None) and len(self.re.monitors) > 0)

            if not (min_elapsed and rt_ok and mon_ready):
                self.root.after(50, _maybe_finish_startup)
                return

            try:
                cur = self.screen_var.get()
                if cur not in (self.NONE, self.MULTIPLE):
                    self.screen_var.set(self.NONE)
                    self._on_output_change()
                    self.screen_var.set(cur)
                    self._on_output_change()
                else:
                    self._on_output_change()
            except Exception:
                logging.exception("Startup output pre-toggle failed.")

            self._ui_ready = True
            self._startup_inhibit_triggers = False
            try:
                self._trigger_column()
                self._startup_render_sent = True
            except Exception:
                logging.exception("Deferred render trigger failed.")
            self._startup_inhibit_triggers = True

            start_wait = time.perf_counter()
            expect_window = (self.screen_var.get() != self.NONE)

            def _finish_show():
                if expect_window:
                    try:
                        win_ok = bool(self.re and getattr(self.re, 'window', None))
                    except Exception:
                        win_ok = False
                    if not win_ok and (time.perf_counter() - start_wait) < 2.0:
                        self.root.after(16, _finish_show)
                        return

                try:
                    if self._splash_win:
                        self._splash_win.destroy()
                finally:
                    self._splash_win = None

                self.root.deiconify()
                self.root.update_idletasks()

                def _lift_inhibit():
                    self._startup_inhibit_triggers = False
                self.root.after(200, _lift_inhibit)

            self.root.after(10, _finish_show)

        self.root.after(10, _maybe_finish_startup)
        self.root.mainloop()

    def _build_menu(self):
        menubar = Menu(self.root)
        
        # File Menu
        fm = Menu(menubar, tearoff=0)
        fm.add_command(label='New Project', command=self._new_project, accelerator="Ctrl+N")
        fm.add_command(label='Open Project', command=self._open_project, accelerator="Ctrl+O")
        
        self.recent_menu = Menu(fm, tearoff=0)
        fm.add_cascade(label='Recent Projects', menu=self.recent_menu)
        self._populate_recent_menu()

        fm.add_separator()
        fm.add_command(label='Save', command=self._save_project, accelerator="Ctrl+S")
        fm.add_command(label='Save As', command=self._save_project_as, accelerator="Ctrl+Shift+S")
        fm.add_separator()
        fm.add_command(label='Preferences', command=self._open_preferences)
        fm.add_separator()
        fm.add_command(label='Exit', command=self._on_exit)
        menubar.add_cascade(label='File', menu=fm)

        # Insert Menu
        im = Menu(menubar, tearoff=0)
        im.add_command(label='Add Media to Project.', command=self._menu_add)
        im.add_separator()
        im.add_command(label='Cue', command=self._add_cue)
        im.add_command(label='Layer', command=self._add_layer)
        menubar.add_cascade(label='Insert', menu=im)
        
        self.root.config(menu=menubar)

        # Bind shortcuts
        self.root.bind_all("<Control-n>", lambda e: self._new_project())
        self.root.bind_all("<Control-o>", lambda e: self._open_project())
        self.root.bind_all("<Control-s>", lambda e: self._save_project())
        self.root.bind_all("<Control-Shift-S>", lambda e: self._save_project_as())

        # Space bar triggers GO (global, non-destructive)
        self.root.bind_all("<space>", self._on_space_go, add='+')

        # Install high-priority global space handler via bindtags
        self._install_global_space_go()

        # NEW: clear focus when clicking outside inputs, so space resumes triggering cues
        self.root.bind_all("<Button-1>", self._on_click_maybe_clear_focus, add='+')

    def _on_click_maybe_clear_focus(self, event):
        """
        De-select inputs when clicking outside them so the spacebar 'GO' works again,
        without breaking dropdown selection or menus. Hardened against transient
        popdown widget names that may disappear before resolution.
        """
        xr, yr = event.x_root, event.y_root

        def _deferred_clear(xr=xr, yr=yr):
            # winfo_containing may fail if the click occurred on a transient popdown
            # that has already been destroyed by the time this runs.
            try:
                w = self.root.winfo_containing(xr, yr)
            except Exception:
                return  # e.g., KeyError('popdown') / TclError – safely ignore

            if not w:
                return  # pointer outside our toplevel; do nothing

            # Do NOT clear if click is within an input, a combobox popdown, listbox, or a menu.
            try:
                if self._is_input_like_widget(w):
                    return
            except Exception:
                return  # be conservative; if we can't inspect, don't clear

            # Safe to clear focus so global <space> binding can fire again.
            try:
                self.root.focus_set()
            except Exception:
                pass  # non-fatal

        # Defer until after Tk completes widget-specific handling (prevents closing popdowns prematurely)
        self.root.after_idle(_deferred_clear)


    def _is_input_like_widget(self, w):
        """
        Return True if 'w' behaves like an input or is part of a combobox popdown/menu.
        This guards against clearing focus while the user is interacting with dropdowns.
        """
        # Direct isinstance checks (Tk classic + ttk)
        try:
            import tkinter as tk
            from tkinter import ttk
            if isinstance(
                w,
                (
                    tk.Entry, tk.Text, tk.Listbox, tk.Spinbox,
                    ttk.Entry, ttk.Combobox, ttk.Spinbox,
                ),
            ):
                return True
        except Exception:
            # If imports/types vary, fall through to class-name checks
            pass

        # Class-name checks cover themed widgets and platform-specific popdowns
        try:
            cls = w.winfo_class()
            if cls in {
                "Entry", "Text", "Listbox", "Spinbox",
                "TEntry", "TCombobox", "TSpinbox", "Treeview", "Menu"
            }:
                return True
        except Exception:
            pass

        # If the click is inside a Combobox popdown toplevel, skip clearing focus.
        try:
            tw = w.winfo_toplevel()
            top_cls = tw.winfo_class()
            # Many Tk builds name the popdown toplevel 'TComboboxPopdown'
            # The widget path often contains 'popdown' as well.
            if ("Combobox" in top_cls) or ("popdown" in str(tw)):
                return True
        except Exception:
            pass

        return False

    def _on_space_go(self, event):
        """Global spacebar handler -> trigger GO unless user is typing/selecting."""
        w = event.widget
        try:
            # Allow default behavior in inputs/editable selectors
            if isinstance(
                w,
                (
                    tk.Entry, tk.Text, tk.Listbox, tk.Spinbox,
                    ttk.Entry, ttk.Combobox, ttk.Spinbox,
                    tk.Checkbutton, tk.Radiobutton,
                    ttk.Checkbutton, ttk.Radiobutton,
                ),
            ):
                return  # do not intercept space in inputs
        except Exception:
            pass

        # Debounce key-repeat (single GO per tap)
        now = time.perf_counter()
        last = getattr(self, '_last_space_ts', 0.0)
        if (now - last) < 0.20:
            return "break"
        self._last_space_ts = now

        self._trigger_go()
        return "break"
        
    def _install_global_space_go(self):
        """
        Ensure our space handler runs at high priority for most widgets by
        inserting a custom bindtag right after the widget tag. Newly created
        widgets get tagged on FocusIn.
        """
        self._SPACE_TAG = 'VD_GlobalSpaceGo'
        try:
            # Bind our handler to the custom class/tag
            self.root.bind_class(self._SPACE_TAG, self._on_space_go, add='+')
        except Exception:
            pass

        # Apply to existing widgets
        try:
            self._add_space_tag_recursive(self.root)
        except Exception:
            pass

        # Ensure future/focused widgets also get the tag
        try:
            self.root.bind_all('<FocusIn>', self._on_focus_in_space_tag, add='+')
        except Exception:
            pass

    def _add_space_tag_recursive(self, widget):
        # Insert our tag so it runs before class/toplevel/'all'
        try:
            tags = list(widget.bindtags())
            if getattr(self, '_SPACE_TAG', None) and self._SPACE_TAG not in tags:
                tags.insert(1, self._SPACE_TAG)  # after the widget-specific tag
                widget.bindtags(tuple(tags))
        except Exception:
            pass
        # Recurse into children
        try:
            for child in widget.winfo_children():
                self._add_space_tag_recursive(child)
        except Exception:
            pass

    def _on_focus_in_space_tag(self, event):
        # Tag the focused widget (e.g., popdown Listbox of a Combobox, new dialogs)
        w = getattr(event, 'widget', None)
        if w is not None:
            self._add_space_tag_recursive(w)

    def _on_gui_configure(self, event):
        if event.widget is self.root:
            if hasattr(self, "_resize_timer") and isinstance(self._resize_timer, str):
                self.root.after_cancel(self._resize_timer)
            self._resize_timer = self.root.after(250, self._apply_resize_logic)

    def _apply_resize_logic(self):
        self._apply_fracs()
        self._update_gui_monitor_index()
        
    def _update_gui_monitor_index(self):
        self.root.update_idletasks()
        gui_x = self.root.winfo_x()
        gui_y = self.root.winfo_y()
        for i, monitor in enumerate(get_monitors()):
            if monitor.x <= gui_x < monitor.x + monitor.width and \
               monitor.y <= gui_y < monitor.y + monitor.height:
                if self.gui_monitor_index != i:
                    self.gui_monitor_index = i
                return
        self.gui_monitor_index = 0

    def _on_exit(self):
        if not self._check_before_closing_project():
            return
        if self.presentation_timer:
            self.root.after_cancel(self.presentation_timer)
        self.proxy_executor.shutdown(wait=False)
        
        # ADD THIS: Stop transition thread
        if self.transition_thread and self.transition_thread.is_alive():
            logging.info("Stopping transition thread...")
            self.transition_thread.shutdown()
            self.transition_thread.join(timeout=1.0)
        
        if self.render_thread and self.render_thread.is_alive():
            logging.info("Requesting render thread to stop...")
            self.render_command_queue.put({'action': 'stop'})
            self.render_thread.join(timeout=2.0)
        self.root.destroy()

    def _initialize_render_engine(self):
        # Clear any existing static cache before switching engines
        if self.re:
            self.re.clear_static_cache()
            
        if self.render_thread and self.render_thread.is_alive():
            self.render_command_queue.put({'action': 'stop'})
            self.render_thread.join(timeout=1.0)

        self.render_command_queue = queue.Queue()
        self.preview_bridge = PreviewBridge()
        self.render_preview_queue = queue.Queue(maxsize=1)
        self.render_stop_event = threading.Event()

        if self.render_mode == 'cpu' or self.render_mode == 'pygame_gpu':
            self.re = PygameRenderEngine(self.mm, mode=self.render_mode, hw_accel=self.hw_accel)
        elif self.render_mode == 'opengl':
            # Pass the instancing setting to the engine
            self.re = OpenGLRenderEngine(
                self.mm, 
                mode=self.render_mode, 
                hw_accel=self.hw_accel,
                use_pbo=self.use_pbo,
                use_vsync=self.use_vsync,
                use_instancing=getattr(self, 'use_instancing', True),
                preview_bridge=self.preview_bridge
            )
        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")

        self.render_thread = RenderThread(
            self.re, self.render_command_queue,
            self.render_preview_queue, self.render_stop_event
        )
        self.render_thread.start()

        if self.transition_thread is None:
            self.transition_thread = TransitionThread(self)
            self.transition_thread.start()

        opts = [self.NONE, self.MULTIPLE] + [f'Display {i}' for i in range(len(self.re.monitors))]
        self.output_screen_combobox['values'] = opts
        if not self.re.monitors:
            self.screen_var.set(self.NONE)
        
        self._update_gui_monitor_index()
        self.set_preview_dirty()
        
    def _run_optimization_test(self, render_mode_var):
        """Sets up and starts the non-blocking performance test using a separate process."""
        test_win = tk.Toplevel(self.root)
        test_win.title("Performance Test")
        test_win.configure(bg='#3e3e3e')
        test_win.geometry("350x200")
        test_win.transient(self.root)
        test_win.grab_set()
        test_win.protocol("WM_DELETE_WINDOW", lambda: None)

        main_frame = Frame(test_win, bg='#3e3e3e', padx=15, pady=15)
        main_frame.pack(fill='both', expand=True)

        status_label = tk.Label(main_frame, text="Preparing test...", fg='white', bg='#3e3e3e', justify='left')
        status_label.pack(anchor='w', pady=5)

        progress = ttk.Progressbar(main_frame, orient='horizontal', length=300, mode='determinate')
        progress.pack(fill='x', pady=5)

        benchmark_script_path = os.path.join(os.path.dirname(__file__), 'benchmark.py')

        if not os.path.exists(benchmark_script_path):
            status_label.config(text="Error: benchmark.py not found!")
            messagebox.showerror("Error", "Benchmark script is missing. Cannot run test.", parent=test_win)
            self.root.after(100, test_win.destroy)
            return

        self._test_state = {
            'window': test_win,
            'status_label': status_label,
            'progress_bar': progress,
            'tests_to_run': ['opengl', 'pygame_gpu'],
            'results': {},
            'process': None,
            'script_path': benchmark_script_path,
            'render_mode_var': render_mode_var
        }

        self.root.after(200, self._launch_next_process)

    def _launch_next_process(self):
        """Launches the benchmark script in a subprocess for the next test."""
        state = self._test_state

        if not state['tests_to_run']:
            self._analyze_test_results()
            return

        mode_to_test = state['tests_to_run'][0]
        
        mode_names = {'opengl': 'Performance Mode (OpenGL)', 'pygame_gpu': 'Balanced Mode (Pygame)'}
        progress_values = {'opengl': 10, 'pygame_gpu': 60}
        state['status_label'].config(text=f"Testing: {mode_names.get(mode_to_test, 'Unknown')}...")
        state['progress_bar'].config(value=progress_values.get(mode_to_test, 0))
        self.root.update_idletasks()

        command = [sys.executable, state['script_path'], mode_to_test]
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0

        try:
            state['process'] = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                creationflags=creation_flags
            )
            self.root.after(100, self._check_process_status)
        except Exception as e:
            logging.error(f"Failed to launch benchmark process for {mode_to_test}: {e}")
            state['results'][mode_to_test] = float('inf')
            state['tests_to_run'].pop(0)
            self.root.after(100, self._launch_next_process)

    def _check_process_status(self):
        """Checks if the subprocess has finished and reads its result."""
        state = self._test_state
        process = state.get('process')

        if not process or process.poll() is None:
            self.root.after(100, self._check_process_status)
            return

        mode_tested = state['tests_to_run'].pop(0)
        stdout, stderr = process.communicate()
        
        if stderr:
            logging.warning(f"Benchmark for {mode_tested} produced an error: {stderr.strip()}")

        try:
            last_line = stdout.strip().split('\n')[-1]
            result_time = float(last_line)
            state['results'][mode_tested] = result_time
        except (ValueError, TypeError, IndexError):
            logging.error(f"Could not parse benchmark result for {mode_tested}. Output: '{stdout}'")
            state['results'][mode_tested] = float('inf')
        
        self.root.after(100, self._launch_next_process)

    def _analyze_test_results(self):
        """Analyzes results, asks the user to apply them, and updates live if they agree."""
        state = self._test_state
        state['status_label'].config(text="Test complete. Analyzing results...")
        state['progress_bar'].config(value=100)
        
        best_mode, best_time = 'cpu', float('inf')
        if state['results'].get('opengl', float('inf')) < 10.0:
            best_mode, best_time = 'opengl', state['results']['opengl']
        if state['results'].get('pygame_gpu', float('inf')) < best_time:
            best_mode = 'pygame_gpu'

        mode_names = {'opengl': 'Performance Mode', 'pygame_gpu': 'Balanced Mode', 'cpu': 'Compatibility Mode'}
        
        question = f"The recommended setting for your system is:\n\n**{mode_names[best_mode]}**\n\nWould you like to switch to this setting now?"
        
        if messagebox.askyesno("Optimization Complete", question, parent=state['window']):
            self.render_mode = best_mode
            state['render_mode_var'].set(best_mode)
            
            self._initialize_render_engine()
            self._update_active_players()
            self._trigger_column()
            
            messagebox.showinfo("Setting Applied", "The new setting has been applied. Click 'Save and Apply' to make it permanent.", parent=state['window'])
        
        state['window'].destroy()
        del self._test_state

    def _open_preferences(self):
        pref_win = tk.Toplevel(self.root)
        pref_win.title("Preferences")
        pref_win.configure(bg='#3e3e3e')
        pref_win.transient(self.root)
        pref_win.grab_set()

        # --- Variables for settings ---
        render_mode_var = tk.StringVar(value=self.render_mode)
        hw_accel_var = tk.StringVar(value=self.hw_accel)
        # Tkinter variables for the options
        use_pbo_var = tk.BooleanVar(value=self.use_pbo)
        use_vsync_var = tk.BooleanVar(value=self.use_vsync)
        force_multithread_var = tk.BooleanVar(value=self.force_multithread)
        # Add instancing variable
        use_instancing_var = tk.BooleanVar(value=getattr(self, 'use_instancing', True))

        main_frame = tk.Frame(pref_win, bg='#3e3e3e', padx=15, pady=15)
        main_frame.pack(fill='both', expand=True)

        # --- Rendering Backend Section ---
        mode_lf = ttk.LabelFrame(main_frame, text="Rendering Backend", padding=(10, 5))
        mode_lf.pack(fill='x', expand=True, pady=(0, 10))

        opengl_rb = ttk.Radiobutton(mode_lf, text="Performance Mode (Requires Good GPU)", variable=render_mode_var, value='opengl')
        opengl_rb.pack(anchor='w', padx=5, pady=2)
        pygame_gpu_rb = ttk.Radiobutton(mode_lf, text="Balanced Mode (Recommended)", variable=render_mode_var, value='pygame_gpu')
        pygame_gpu_rb.pack(anchor='w', padx=5, pady=2)
        cpu_rb = ttk.Radiobutton(mode_lf, text="Compatibility Mode (Safe Mode)", variable=render_mode_var, value='cpu')
        cpu_rb.pack(anchor='w', padx=5, pady=2)

        # --- Advanced OpenGL Options Section ---
        advanced_lf = ttk.LabelFrame(main_frame, text="Advanced Performance Options", padding=(10, 5))

        pbo_check = ttk.Checkbutton(advanced_lf, text="Use Asynchronous PBO Uploads", variable=use_pbo_var)
        pbo_check.pack(anchor='w', padx=5, pady=2)
        CreateToolTip(pbo_check, "Enables faster texture uploads on most dedicated GPUs, but can cause issues on some integrated graphics.")

        vsync_check = ttk.Checkbutton(advanced_lf, text="Enable VSync", variable=use_vsync_var)
        vsync_check.pack(anchor='w', padx=5, pady=2)
        CreateToolTip(vsync_check, "Synchronizes framerate to your monitor's refresh rate to prevent screen tearing. May introduce input latency.")
        
        multithread_check = ttk.Checkbutton(advanced_lf, text="Force Multi-threaded Decoding", variable=force_multithread_var)
        multithread_check.pack(anchor='w', padx=5, pady=2)
        CreateToolTip(multithread_check, "Tells the video decoder to use multiple CPU threads. Can improve performance for high-resolution video, but may be unstable.")
        
        # Add instancing checkbox
        instancing_check = ttk.Checkbutton(advanced_lf, text="Use Instanced Rendering (Faster for 2+ layers)", 
                                           variable=use_instancing_var)
        instancing_check.pack(anchor='w', padx=5, pady=2)
        CreateToolTip(instancing_check, "Renders multiple layers in a single GPU draw call. Dramatically improves performance with multiple layers. Disable if you experience rendering issues.")

        # --- GPU Hardware Acceleration Section ---
        gpu_lf = ttk.LabelFrame(main_frame, text="GPU Hardware Acceleration", padding=(10, 5))
        gpu_lf.pack(fill='x', expand=True, pady=(0, 10))
        
        accel_options = ['auto', 'cuda', 'dxva2', 'd3d11va', 'videotoolbox']
        for accel in accel_options:
            rb = ttk.Radiobutton(gpu_lf, text=accel.upper(), variable=hw_accel_var, value=accel)
            rb.pack(anchor='w', padx=5, pady=2)
            if accel == 'auto': 
                CreateToolTip(rb, "Let the system automatically choose the best method. (Recommended)")

        # --- Logic to show/hide options ---
        def on_mode_change(*args):
            selected_mode = render_mode_var.get()
            
            # Show/hide GPU acceleration options
            gpu_state = 'normal' if selected_mode != 'cpu' else 'disabled'
            for child in gpu_lf.winfo_children():
                child.configure(state=gpu_state)
                
            # Show/hide Advanced OpenGL options
            if selected_mode == 'opengl':
                advanced_lf.pack(fill='x', expand=True, pady=(0, 10), before=gpu_lf)
            else:
                advanced_lf.pack_forget()

        render_mode_var.trace_add('write', on_mode_change)
        on_mode_change() # Call once to set initial state

        # --- Buttons Section ---
        ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=(15, 10))
        optimize_btn = tk.Button(main_frame, text="⚡ Find Best Setting For Me", bg='#3a7ecf', fg='white', 
                                command=lambda: self._run_optimization_test(render_mode_var))
        optimize_btn.pack(fill='x')
        
        btn_frame = tk.Frame(main_frame, bg='#3e3e3e')
        btn_frame.pack(fill='x', pady=(15, 0))

        def save_prefs():
            # Save all settings
            self.render_mode = render_mode_var.get()
            self.hw_accel = hw_accel_var.get()
            self.use_pbo = use_pbo_var.get()
            self.use_vsync = use_vsync_var.get()
            self.force_multithread = force_multithread_var.get()
            self.use_instancing = use_instancing_var.get()  # Now properly gets the boolean value
            
            logging.info(f"Preferences saved. Mode: {self.render_mode}, PBO: {self.use_pbo}, VSync: {self.use_vsync}, Instancing: {self.use_instancing}")
            self._set_dirty_flag()
            self._save_config()
            self._initialize_render_engine()
            self._update_active_players()
            self._trigger_column()
            pref_win.destroy()

        save_btn = tk.Button(btn_frame, text="Save and Apply", command=save_prefs, bg='#5a5', fg='black')
        save_btn.pack(side='right', padx=5)
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=pref_win.destroy, bg='#a55', fg='white')
        cancel_btn.pack(side='right')

    def _open_optimize_window(self, path):
        optimizer = FFMpegOptimizer(self.root, path)
        optimizer.show()
        self.root.wait_window(optimizer.window)
        self._build_media_list()

    def _gui_animation_loop(self):
        import os, time, gc
        from queue import Empty

        layers = self._get_layers_for_col(self.active_col)
        self.is_video_playing = any(
            l['path'] and os.path.exists(l['path']) and os.path.splitext(l['path'])[1].lower() in VIDEO_EXTS
            for l in layers.values()
        )
        if self.is_video_playing:
            self.set_preview_dirty()

        # Update FPS display if render thread has a frame pacer (unchanged behavior)
        if hasattr(self, 'render_thread') and self.render_thread and hasattr(self.render_thread, 'frame_pacer'):
            fps = self.render_thread.frame_pacer.actual_fps
            self.fps_label.config(text=f"FPS: {fps:.1f}")

        # Throttle preview updates to about 30 FPS outside transitions (unchanged)
        current_time = time.perf_counter()
        last_preview_time = getattr(self, '_last_preview_time', 0)

        # Transition state
        transitioning = bool(getattr(self, 'transition_active', False))

        # Pause GC only during transitions to prevent GC pauses mid-fade
        gc_paused = getattr(self, '_gc_paused', False)
        if transitioning and not gc_paused:
            try:
                gc.disable()
            finally:
                self._gc_paused = True
        elif (not transitioning) and gc_paused:
            try:
                gc.enable()
                gc.collect(0)
            finally:
                self._gc_paused = False

        # While transitioning, drain pending preview requests to prevent backlog bursts
        if transitioning and hasattr(self, 'render_command_queue') and self.render_command_queue is not None:
            retained = []
            try:
                while True:
                    msg = self.render_command_queue.get_nowait()
                    if not (isinstance(msg, dict) and msg.get('action') == 'request_preview'):
                        retained.append(msg)
            except Empty:
                pass
            for m in retained:
                self.render_command_queue.put(m)

        # Only request a new preview if not transitioning, timing allows, AND NOT Multiple
        if (not transitioning) and (not self.is_video_playing or (current_time - last_preview_time > 0.033)):
            if getattr(self, 'preview_dirty', False):
                if self.screen_var.get() == self.MULTIPLE:
                    # NEW: disable preview generation and show placeholder
                    try:
                        self._update_preview_canvas_with_image(None)
                    except Exception:
                        pass
                    self.preview_dirty = False
                    self._last_preview_time = current_time
                else:
                    self.render_command_queue.put({'action': 'request_preview'})
                    self.preview_dirty = False
                    self._last_preview_time = current_time

        self.root.after(16, self._gui_animation_loop)

    def _thumbnail_generator_thread(self, paths_to_generate):
        for path in paths_to_generate:
            try:
                if path in self.thumbnail_cache:
                    continue
                pil_img = self._load_frame(path, for_thumbnail=True)
                if pil_img:
                    pil_img.thumbnail((80, 45))
                    self.thumbnail_queue.put((path, pil_img))
            except Exception:
                logging.exception(f"Error generating thumbnail for {path}")
        self.thumbnail_thread_lock.release()

    def _process_thumbnail_queue(self):
        try:
            while not self.thumbnail_queue.empty():
                path, pil_image = self.thumbnail_queue.get_nowait()
                imgtk = ImageTk.PhotoImage(pil_image)
                self.thumbnail_cache[path] = imgtk
                if path in self.media_list_labels:
                    lbl_img = self.media_list_labels[path]
                    lbl_img.config(image=imgtk)
                    lbl_img.image = imgtk
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._process_thumbnail_queue)

    def _process_preview_queue(self):
        # NEW: if Multiple outputs are selected, keep the preview blank and ignore queue frames
        if self.screen_var.get() == self.MULTIPLE:
            try:
                self._update_preview_canvas_with_image(None)
            except Exception:
                pass
            finally:
                self.root.after(33, self._process_preview_queue)
            return

        try:
            final_image = self.render_preview_queue.get_nowait()
            self._update_preview_canvas_with_image(final_image)
        except queue.Empty:
            pass
        finally:
            self.root.after(33, self._process_preview_queue)

    def _build_media_list(self):
        for w in self.media_frame.winfo_children(): w.destroy()
        canvas = Canvas(self.media_frame, bg='#2e2e2e', highlightthickness=0)
        vsb = Scrollbar(self.media_frame, orient='vertical', command=canvas.yview)
        container = Frame(canvas, bg='#2e2e2e')
        container.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0,0), window=container, anchor='nw')
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')

        self.media_list_labels.clear()
        paths_to_generate = []
        placeholder = ImageTk.PhotoImage(Image.new('RGB', (80, 45), '#444'))

        # The media list is now built from the project's specific list of paths
        for path in self.project_media_paths:
            is_missing = not os.path.exists(path)
            ext = os.path.splitext(path)[1].lower()
            if not is_missing and ext in VIDEO_EXTS:
                proxy_path = FFMpegProxyGenerator.get_proxy_path(path)
                if not os.path.exists(proxy_path):
                    self.proxy_executor.submit(FFMpegProxyGenerator(path).generate)

            if ext not in SUPPORTED_EXTS: continue

            frm = Frame(container, bg=self.DEFAULT_BG, pady=5)
            frm.pack(fill='x', padx=5, pady=2)
            frm.bind('<Button-1>', lambda e,p=path,w=frm: self._start_media_drag(p,w))
            frm.bind('<Button-3>', lambda e,p=path: self._show_media_menu(e,p))

            imgtk = None
            if not is_missing:
                if path in self.thumbnail_cache:
                    imgtk = self.thumbnail_cache[path]
                else:
                    imgtk = placeholder
                    paths_to_generate.append(path)

            lbl_img = Label(frm, image=imgtk, bg=self.DEFAULT_BG)
            if imgtk: lbl_img.image = imgtk
            lbl_img.pack(side='left', padx=5)
            self.media_list_labels[path] = lbl_img
            
            display_name = os.path.basename(path)
            text_color = 'white'
            if is_missing:
                display_name = f"[MISSING] {display_name}"
                text_color = '#ff8888' # Light red

            lbl_txt = Label(frm, text=display_name, fg=text_color, bg=self.DEFAULT_BG, justify='left')
            lbl_txt.pack(side='left', padx=5)
            for child in frm.winfo_children():
                child.bind('<Button-1>', lambda e,p=path,w=frm: self._start_media_drag(p,w))
                child.bind('<Button-3>', lambda e,p=path: self._show_media_menu(e,p))

        if paths_to_generate:
            if self.thumbnail_thread_lock.acquire(blocking=False):
                self.thumbnail_thread = threading.Thread(
                    target=self._thumbnail_generator_thread,
                    args=(paths_to_generate,),
                    daemon=True
                )
                self.thumbnail_thread.start()
            else:
                logging.warning("Thumbnail generator is already running.")

    def _build_grid(self):
        for w in self.grid_frame.winfo_children(): w.destroy()
        self.grid_frame.grid_propagate(False)
        self.grid_frame.rowconfigure(0, weight=1)
        self.grid_frame.columnconfigure(0, weight=1)
        canvas = Canvas(self.grid_frame, bg='#2e2e2e', highlightthickness=0)
        vsb = Scrollbar(self.grid_frame, orient='vertical', command=canvas.yview)
        hsb = Scrollbar(self.grid_frame, orient='horizontal', command=canvas.xview)
        canvas.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        inner = Frame(canvas, bg='#2e2e2e')
        canvas.create_window((0,0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        Label(inner, bg='#2e2e2e').grid(row=0, column=0)

        self.col_labels = {}
        for c in range(self.COLS):
            lbl = Label(inner, text=self.cue_names.get(str(c), f'Cue {c+1}'), bg='#444', fg='white', height=1)
            lbl.grid(row=0, column=c+1, sticky='ew', padx=self.CELL_PAD//2)
            # Changed: start a cue reorder drag; plain click (no motion) still selects in drop handler
            lbl.bind('<Button-1>', lambda e, col=c: self._start_cue_reorder_drag(col))
            lbl.bind('<Button-3>', lambda e, col=c: self._show_cue_menu(e, col))
            self.col_labels[c] = lbl

        inner.columnconfigure(0, minsize=50)
        for c in range(1, self.COLS+1): inner.columnconfigure(c, minsize=self.CELL_SIZE)
        for r in range(1, self.ROWS+1): inner.rowconfigure(r, minsize=self.CELL_SIZE)

        self.row_labels = {}
        self.layer_setting_widgets = {}
        for r in range(self.ROWS):
            # Container for layer name and settings button
            label_container = Frame(inner, bg='#444')
            label_container.grid(row=r+1, column=0, sticky='ns', pady=self.CELL_PAD//2)

            canvas_height = self.CELL_SIZE - 30
            label_canvas = Canvas(label_container, bg='#444', width=50, height=canvas_height, highlightthickness=0)
            text_id = label_canvas.create_text(
                25, canvas_height//2,
                text=self.layer_names.get(str(r), f'Layer {r+1}'),
                fill='white', angle=90, anchor='center'
            )
            label_canvas.text_id = text_id
            # New: start a layer reorder drag on left click
            label_canvas.bind('<Button-1>', lambda e, row=r: self._start_layer_reorder_drag(row))
            label_canvas.bind('<Button-3>', lambda e, row=r: self._show_layer_menu(e, row))
            label_canvas.pack(side='top', fill='x')
            self.row_labels[r] = label_canvas

            # Settings button for multi-output
            settings_btn = Button(
                label_container, text='⚙️', font=('Arial', 12), relief='flat', bg='#444', fg='white',
                command=lambda row=r: self._show_layer_output_menu(row)
            )
            settings_btn.pack(side='bottom', anchor='center', pady=0)
            self.layer_setting_widgets[r] = settings_btn

            for c in range(self.COLS):
                frm = Frame(
                    inner, bg='#444', width=self.CELL_SIZE, height=self.CELL_SIZE,
                    highlightthickness=1, highlightbackground='black'
                )
                frm.grid(row=r+1, column=c+1, padx=self.CELL_PAD//2, pady=self.CELL_PAD//2)
                frm.propagate(False)
                lbl = Label(frm, bg='#444', fg='#ff8888')  # Set default color for missing text
                lbl.pack(expand=True, fill='both')
                self.frames_by_position[(r, c)] = frm
                self._cell_frames[frm] = (r, c)
                for widget in [frm, lbl]:
                    widget.bind('<Button-1>', lambda e, r=r, c=c: self._start_cell_drag(r, c))
                    widget.bind('<Button-3>', lambda e, row=r, col=c: self._show_cell_menu(e, row, col))
                    widget.bind('<Double-Button-1>', lambda e, row=r, col=c: self._edit_media(row, col))
                if self.grid_cells.get((r, c)):
                    self._update_cell_preview(r, c)

        self.grid_canvas = canvas
        self._on_output_change()  # Ensure settings icons are correctly hidden/shown on build

    def _build_preview_panel(self):
        for w in self.preview_frame.winfo_children(): w.destroy()
        
        top = Frame(self.preview_frame, bg='#2e2e2e')
        top.pack(fill='x', pady=5)
        Label(top, text='Output:', fg='white', bg='#2e2e2e').pack(side='left', padx=5)

        self.screen_var = tk.StringVar(value=self.NONE)
        self.output_screen_combobox = ttk.Combobox(top, values=[self.NONE], textvariable=self.screen_var, state='readonly', width=12)
        self.output_screen_combobox.pack(side='left', padx=5)
        self.screen_var.trace_add('write', self._on_output_change)

        Label(top, text='Cue:', fg='white', bg='#2e2e2e').pack(side='left', padx=5)
        self.cue_name_var = tk.StringVar()
        self.cue_select_combobox = ttk.Combobox(top, textvariable=self.cue_name_var, state='readonly', width=10)
        self.cue_select_combobox.pack(side='left', padx=5)
        self.cue_select_combobox.bind('<<ComboboxSelected>>', self._on_cue_select)
        self._update_cue_dropdown()

        self.preview_canvas = Canvas(self.preview_frame, bg='black', highlightthickness=0)
        self.preview_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        # FPS display for monitoring frame pacing
        self.fps_label = Label(self.preview_frame, text="FPS: --", fg='#888', bg='#2e2e2e', font=('Arial', 8))
        self.fps_label.pack(anchor='w', padx=5)
        self.preview_canvas.bind('<Configure>', lambda e: self.set_preview_dirty())

        controls_container = Frame(self.preview_frame, bg='#2e2e2e')
        controls_container.pack(fill='x', padx=5, pady=(5,0))
        controls_container.columnconfigure(1, weight=1)
        Label(controls_container, text="Transition:", fg='white', bg='#2e2e2e').grid(row=0, column=0, sticky='w', pady=2)
        self.transition_var = tk.StringVar(value='Fade')
        self.transition_dropdown = ttk.Combobox(controls_container, textvariable=self.transition_var, values=[self.NONE, 'Fade'], state='readonly', width=15)
        self.transition_dropdown.grid(row=0, column=1, sticky='ew', padx=5)
        self.transition_dropdown.bind('<<ComboboxSelected>>', self._on_transition_select)
        Label(controls_container, text="Timing:", fg='white', bg='#2e2e2e').grid(row=1, column=0, sticky='w', pady=2)
        timing_widget_frame = Frame(controls_container, bg='#2e2e2e')
        timing_widget_frame.grid(row=1, column=1, sticky='ew', padx=5)
        vcmd = (self.root.register(self._validate_numeric), '%P')
        self.transition_timing_var = tk.StringVar(value='1.0')
        self.timing_entry = tk.Entry(timing_widget_frame, textvariable=self.transition_timing_var, width=6, validate='key', validatecommand=vcmd)
        self.timing_entry.pack(side='left')
        self.timing_entry.bind('<KeyRelease>', self._set_dirty_flag)
        Label(timing_widget_frame, text="seconds", fg='white', bg='#2e2e2e').pack(side='left', padx=4)
        self._on_transition_select()
        
        ttk.Separator(self.preview_frame, orient='horizontal').pack(fill='x', padx=5, pady=10)

        # --- Presentation Mode UI ---
        pres_frame = Frame(self.preview_frame, bg='#2e2e2e')
        pres_frame.pack(fill='x', padx=5, pady=5)
        Label(pres_frame, text="Presentation Mode", fg='white', bg='#2e2e2e', font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.presentation_mode_var = tk.BooleanVar(value=False)
        self.presentation_mode_check = tk.Checkbutton(pres_frame, text="Enable", variable=self.presentation_mode_var, command=self._on_toggle_presentation_mode, bg='#2e2e2e', fg='white', selectcolor='#444', activebackground='#2e2e2e', activeforeground='white')
        self.presentation_mode_check.pack(anchor='w')
        self.presentation_mode_check.bind('<Button-1>', self._set_dirty_flag)

        timing_pres_frame = Frame(pres_frame, bg='#2e2e2e')
        timing_pres_frame.pack(fill='x')
        Label(timing_pres_frame, text="Seconds:", fg='white', bg='#2e2e2e').pack(side='left', padx=(20,5))
        self.presentation_timing_var = tk.StringVar(value="5.0")
        self.presentation_timing_entry = tk.Entry(timing_pres_frame, textvariable=self.presentation_timing_var, width=6, validate='key', validatecommand=vcmd, state='disabled')
        self.presentation_timing_entry.pack(side='left')
        self.presentation_timing_entry.bind('<KeyRelease>', self._set_dirty_flag)

        self.presentation_loop_var = tk.BooleanVar(value=False)
        self.presentation_loop_check = tk.Checkbutton(pres_frame, text="Loop", variable=self.presentation_loop_var, state='disabled', bg='#2e2e2e', fg='white', selectcolor='#444', activebackground='#2e2e2e', activeforeground='white')
        self.presentation_loop_check.pack(anchor='w', padx=(0,5))
        self.presentation_loop_check.bind('<Button-1>', self._set_dirty_flag)

        self.go_button = Button(self.preview_frame, text='GO', bg='#5a5', fg='black', width=10, height=2, command=self._trigger_go)
        self.go_button.pack(pady=5)

    def set_preview_dirty(self):
        self.preview_dirty = True

    def _update_preview_canvas_with_image(self, pil_image):
        self.preview_canvas.delete('all')
        canvas_w, canvas_h = self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            self.preview_canvas.image = None
            return

        w_res, h_res = (1920, 1080)
        if pil_image:
            w_res, h_res = pil_image.size

        ratio = w_res / h_res if h_res > 0 else 1
        pw, ph = (canvas_w, int(canvas_w/ratio)) if canvas_w/ratio <= canvas_h else (int(canvas_h*ratio), canvas_h)
        off_x, off_y = (canvas_w - pw)//2, (canvas_h - ph)//2

        self.preview_canvas.create_rectangle(off_x, off_y, off_x+pw, off_y+ph, outline='#666', dash=(5, 5), width=2)

        if not pil_image:
            self.preview_canvas.create_text(off_x + pw/2, off_y + ph/2, text="Preview", fill="white", anchor='center')
            self.preview_canvas.image = None
            return

        resized_pil = pil_image.resize((pw, ph), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(resized_pil)

        self.preview_canvas.create_image(off_x, off_y, image=imgtk, anchor='nw')
        self.preview_canvas.image = imgtk

    def _update_active_players(self, force_restart_paths=None):
        """
        Maintain the set of active media players on the render side.
        During a fade transition, ensure that outgoing ('from') VIDEO players
        remain alive even if their layer IDs collide with 'current' or 'to' layers,
        by duplicating them under synthetic numeric keys. Also preload 'to' videos
        under synthetic numeric keys to avoid collisions.
        """
        if not self.re:
            return

        if force_restart_paths is None:
            force_restart_paths = set()

        layers_to_keep_alive = {}

        # 1) Always include the currently visible column's layers (these are the "from" side during a fade).
        current_layers = self._get_layers_for_col(self.active_col)
        layers_to_keep_alive.update(current_layers)

        if getattr(self, "transition_active", False):
            # --- Transitioning: keep 'from' videos alive + preload 'to' videos ---

            # a) Preserve any extra 'from' layers not already in the current column.
            from_layers = (getattr(self, "transition_from_layers", {}) or {})
            to_layers   = (getattr(self, "transition_to_layers",   {}) or {})

            for lid, layer in from_layers.items():
                if lid not in layers_to_keep_alive:
                    layers_to_keep_alive[lid] = layer

            # b) If a 'from' VIDEO shares an ID with an entry already in keep_alive,
            #    ensure the outgoing VIDEO's path stays present by duplicating it
            #    under a synthetic numeric key so the render thread doesn't stop it.
            base_from_keep = 30_000_000  # large offset to avoid any collision with real layer IDs
            dup_idx = 0
            for lid, layer in from_layers.items():
                path = layer.get('path')
                if not path:
                    continue
                ext = os.path.splitext(path)[1].lower()
                if ext in VIDEO_EXTS:
                    # If this ID exists but references a DIFFERENT path (or a non-video), duplicate the video.
                    keep_entry = layers_to_keep_alive.get(lid)
                    if keep_entry is None or keep_entry.get('path') != path:
                        layers_to_keep_alive[base_from_keep + dup_idx] = layer
                        dup_idx += 1

            # c) Preload ONLY videos from the 'to' side under synthetic numeric keys.
            base_to_preload = 10_000_000
            preload_idx = 0
            for _, layer in to_layers.items():
                path = layer.get('path')
                if not path:
                    continue
                ext = os.path.splitext(path)[1].lower()
                if ext in VIDEO_EXTS:
                    layers_to_keep_alive[base_to_preload + preload_idx] = layer
                    preload_idx += 1

        else:
            # --- Not transitioning: preload NEXT column videos under synthetic numeric keys (existing behavior) ---
            next_col = (self.active_col + 1) % self.COLS
            if next_col != self.active_col:
                next_layers = self._get_layers_for_col(next_col)
                base_next_preload = 20_000_000
                preload_idx = 0
                for _, layer_data in next_layers.items():
                    path = layer_data.get('path')
                    if not path:
                        continue
                    ext = os.path.splitext(path)[1].lower()
                    if ext in VIDEO_EXTS:
                        layers_to_keep_alive[base_next_preload + preload_idx] = layer_data
                        preload_idx += 1

        # Send to renderer (preserves 'from' video players during fades; preloads 'to' videos safely).
        command = {
            'action': 'update_layers',
            'layers': layers_to_keep_alive,
            'force_restart_paths': force_restart_paths
        }
        self.render_command_queue.put(command)

        # Keep a quick status flag
        self.is_video_playing = any(
            (l.get('path') and os.path.splitext(l['path'])[1].lower() in VIDEO_EXTS)
            for l in current_layers.values()
        )

    def _update_transition(self):
        elapsed = time.perf_counter() - self.transition_start_time
        progress = min(elapsed / self.transition_duration, 1.0)

        # Only send a render command if progress has changed by a meaningful amount
        last_progress = getattr(self, '_last_transition_progress', -1)
        if abs(progress - last_progress) < 0.01 and progress < 1.0:
            return  # Skip update if the change is tiny

        # Store the progress for the next check
        self._last_transition_progress = progress

        transition_state = {
            'active': True,
            'progress': progress,
            'from': self.transition_from_layers,
            'to': self.transition_to_layers
        }
        
        output_mode = self.screen_var.get()
        target_monitor = self.selected_display if output_mode != self.MULTIPLE else None
        
        all_screens = set()
        if output_mode == self.MULTIPLE:
            all_screens = {s for s in self.layer_screens.values() if s is not None}

        self.render_command_queue.put({
            'action': 'render',
            'layers': {},
            'target_monitor': target_monitor,
            'transition_state': transition_state,
            'gui_monitor': self.gui_monitor_index,
            'output_mode': output_mode,
            'all_assigned_screens': all_screens,
        })

        if progress >= 1.0:
            # Clean up the progress tracker after the transition is done
            if hasattr(self, '_last_transition_progress'):
                del self._last_transition_progress
            self._finalize_transition()

    def _finalize_transition(self):
        """Cleans up after a transition and sets the new column as active."""
        import os

        # End transition; keep 'to' layers so the active player (temp link) continues.
        self.transition_active = False
        self.transition_from_layers = {}

        # Activate the target column
        target_col = getattr(self, "transition_target_col", None)
        if target_col is not None:
            self.active_col = target_col

        # Update UI to reflect the new active column
        if hasattr(self, "col_labels"):
            for c, lbl in self.col_labels.items():
                try:
                    lbl.config(bg=('#ADD8E6' if c == self.active_col else '#444'))
                except Exception:
                    pass
        if hasattr(self, "_update_cue_dropdown"):
            self._update_cue_dropdown()

        # Do NOT refresh players here; that would snap playback back to the original path/player.
        # Instead, promote temp links to "active" so they persist while cue 2 is live.
        new_active = set(getattr(self, '_temp_transition_links', []))
        old_active = set(getattr(self, '_active_transition_links', []))

        # Remove any old active links not needed anymore
        for _p in (old_active - new_active):
            try:
                if _p and os.path.exists(_p):
                    os.remove(_p)
            except Exception:
                pass

        self._active_transition_links = list(new_active)
        self._temp_transition_links = []

        # Issue a final render/update for the now-active column
        self._trigger_column()



    def _validate_numeric(self, value_if_allowed):
        if value_if_allowed == "": return True
        try:
            float(value_if_allowed)
            return True
        except ValueError:
            return False

    def _on_transition_select(self, event=None):
        if self.transition_var.get() == self.NONE:
            self.timing_entry.config(state='disabled')
        else:
            self.timing_entry.config(state='normal')
        self._set_dirty_flag()

    def _start_media_drag(self, path, widget):
        self.drag_motion_detected = False # MODIFICATION: Reset flag on new drag attempt.
        if self.selected_widget:
            try:
                self.selected_widget.config(bg=self.DEFAULT_BG)
                for child in self.selected_widget.winfo_children(): child.config(bg=self.DEFAULT_BG)
            except tk.TclError: self.selected_widget = None
        widget.config(bg=self.HIGHLIGHT_BG)
        for child in widget.winfo_children(): child.config(bg=self.HIGHLIGHT_BG)
        self.selected_widget = widget
        self.drag_type = 'new_media'
        self.drag_data = path

        pil_img = self._load_frame(path, for_thumbnail=True)
        if pil_img:
            thumb_img = pil_img.copy()
            thumb_img.thumbnail((40, 30))
            imgtk = ImageTk.PhotoImage(thumb_img)
        else:
            imgtk = ImageTk.PhotoImage(Image.new('RGB', (40,30), 'red'))

        self.drag_widget = Label(self.root, image=imgtk, bd=0)
        self.drag_widget.image = imgtk
        self.root.config(cursor='plus')
        self.root.bind('<B1-Motion>', self._on_drag_motion)
        self.root.bind('<ButtonRelease-1>', self._on_media_drop)

    def _start_cell_drag(self, row, col):
        if not self.grid_cells.get((row, col)): return
        self.drag_motion_detected = False # MODIFICATION: Reset flag on new drag attempt.
        self.drag_type = 'cell'
        self.drag_data = {'row': row, 'col': col}
        self.drag_widget = Frame(self.root, width=self.CELL_SIZE, height=self.CELL_SIZE, bg='#2e2e2e', highlightbackground='aqua', highlightthickness=2)
        self.root.config(cursor='exchange')
        self.root.bind('<B1-Motion>', self._on_drag_motion)
        self.root.bind('<ButtonRelease-1>', self._on_cell_drop)

    def _on_drag_motion(self, event):
        self.drag_motion_detected = True # MODIFICATION: Set flag to true only when mouse moves.
        if not self.drag_widget: return
        x = self.root.winfo_pointerx() - self.root.winfo_rootx() - self.drag_widget.winfo_width() // 2
        y = self.root.winfo_pointery() - self.root.winfo_rooty() - self.drag_widget.winfo_height() // 2
        self.drag_widget.place(x=x, y=y)
        cell = self._get_cell_under(event.x_root, event.y_root)
        label = cell.winfo_children()[0] if cell and cell.winfo_children() else None
        if label is not self._hover_label:
            if self._hover_label:
                try: self._hover_label.config(bg='#444')
                except tk.TclError: pass
            if label: label.config(bg='#666')
            self._hover_label = label

    def _on_media_drop(self, event):
        dest_cell_frame = self._get_cell_under(event.x_root, event.y_root)
        if dest_cell_frame and self.drag_motion_detected:
            r, c = self._cell_frames[dest_cell_frame]
            self.grid_cells[(r, c)] = self.drag_data
            self.cell_positions[(r, c)] = {'x': 0.5, 'y': 0.5, 'scale_x': 1.0, 'scale_y': 1.0, 'rotation': 0, 'loop': True}
            self._update_cell_preview(r, c)
            
            # ADD THIS - Invalidate static cache for this column
            if self.re:
                self.re.invalidate_static_cache_for_column(c)
            
            if c == self.active_col:
                self._update_active_players()
                self._trigger_column()
            self._set_dirty_flag()
        self._cleanup_drag()

    def _on_cell_drop(self, event):
        dest_cell_frame = self._get_cell_under(event.x_root, event.y_root)
        if dest_cell_frame and self.drag_motion_detected:  # MODIFICATION: Check flag before executing drop.
            dest_r, dest_c = self._cell_frames[dest_cell_frame]
            src_r, src_c = self.drag_data['row'], self.drag_data['col']
            if (dest_r, dest_c) != (src_r, src_c):
                # Detect Ctrl held during drop (copy mode)
                ctrl_pressed = bool(getattr(event, 'state', 0) & 0x0004) or getattr(self, '_ctrl_down', False)

                if ctrl_pressed:
                    # COPY: duplicate media & position from source into destination; leave source unchanged
                    self.grid_cells[(dest_r, dest_c)] = self.grid_cells.get((src_r, src_c))
                    self.cell_positions[(dest_r, dest_c)] = self.cell_positions.get((src_r, src_c), {}).copy()
                    self._update_cell_preview(dest_r, dest_c)
                    if self.active_col in [src_c, dest_c]:
                        self._trigger_column()
                    self._set_dirty_flag()
                else:
                    # Existing behavior: SWAP the two cells
                    self.grid_cells[(dest_r, dest_c)], self.grid_cells[(src_r, src_c)] = \
                        self.grid_cells.get((src_r, src_c)), self.grid_cells.get((dest_r, dest_c))
                    self.cell_positions[(dest_r, dest_c)], self.cell_positions[(src_r, src_c)] = \
                        self.cell_positions.get((src_r, src_c), {}).copy(), self.cell_positions.get((dest_r, dest_c), {}).copy()

                    self._update_cell_preview(src_r, src_c)
                    self._update_cell_preview(dest_r, dest_c)
                    if self.active_col in [src_c, dest_c]:
                        self._trigger_column()
                    self._set_dirty_flag()
        self._cleanup_drag()

    def _cleanup_drag(self):
        if self.drag_widget: self.drag_widget.destroy()
        if self._hover_label:
            try: self._hover_label.config(bg='#444')
            except tk.TclError: pass
        self.root.config(cursor='')
        self.root.unbind('<B1-Motion>')
        self.root.unbind('<ButtonRelease-1>')
        self.drag_type = None
        self.drag_data = None
        self.drag_widget = None
        self._hover_label = None

    def _get_cell_under(self, x_root, y_root):
        cx = self.grid_canvas.canvasx(x_root - self.grid_canvas.winfo_rootx())
        cy = self.grid_canvas.canvasy(y_root - self.grid_canvas.winfo_rooty())
        header_h = self.col_labels[0].winfo_height() if self.col_labels else 30
        if cy < header_h: return None
        step = self.CELL_SIZE + self.CELL_PAD
        col = int((cx - 50) / step) # Adjusted for wider layer column
        row = int((cy - header_h) / step)
        if 0 <= row < self.ROWS and 0 <= col < self.COLS:
            return self.frames_by_position.get((row, col))
        return None

    def _update_cell_preview(self, row, col):
            frame = self.frames_by_position.get((row, col))
            path = self.grid_cells.get((row, col))
            if not frame: return
            label = frame.winfo_children()[0]

            if not path or not os.path.exists(path):
                label.config(image='', bg='#444', text='[Missing]' if path else '')
                label.image = None
                return
            
            pos_data = self.cell_positions.get((row, col))
            if not pos_data: return
            
            pil_media = self._load_frame(path, for_thumbnail=True)
            if not pil_media: return
            
            _, _, ow, oh = self.mm.get_media_info(path)
            if ow == 0 or oh == 0: return

            w_res, h_res = (1920, 1080)
            output_mode = self.screen_var.get()
            if output_mode == self.MULTIPLE:
                screen_idx = self.layer_screens.get(str(row))
                # FIX: Validate the screen index
                if screen_idx is not None and self.re and self.re.monitors and 0 <= screen_idx < len(self.re.monitors):
                    w_res, h_res = self.re.monitors[screen_idx].width, self.re.monitors[screen_idx].height
            # FIX: Validate the selected display
            elif self.selected_display is not None and self.re and self.re.monitors and 0 <= self.selected_display < len(self.re.monitors):
                 w_res, h_res = self.re.monitors[self.selected_display].width, self.re.monitors[self.selected_display].height
            
            display_aspect = w_res / h_res if h_res > 0 else 1.0
            preview_w, preview_h = (self.CELL_SIZE, int(self.CELL_SIZE/display_aspect)) if display_aspect >= 1 else (int(self.CELL_SIZE*display_aspect), self.CELL_SIZE)
            layer_comp = Image.new('RGBA', (preview_w, preview_h), (0,0,0,0))
            scale_factor = preview_w / w_res
            
            preview_layer_w = int(ow * pos_data.get('scale_x', 1.0) * scale_factor)
            preview_layer_h = int(oh * pos_data.get('scale_y', 1.0) * scale_factor)
            
            if preview_layer_w > 0 and preview_layer_h > 0:
                transformed_pil = pil_media.resize((preview_layer_w, preview_layer_h), Image.LANCZOS)
                if pos_data.get('rotation', 0) != 0: transformed_pil = transformed_pil.rotate(pos_data['rotation'], expand=True, resample=Image.BICUBIC)
                transformed_pil = transformed_pil.convert('RGBA')
                center_x, center_y = pos_data['x']*preview_w, pos_data['y']*preview_h
                paste_x, paste_y = int(center_x - transformed_pil.width/2), int(center_y - transformed_pil.height/2)
                layer_comp.alpha_composite(transformed_pil, (paste_x, paste_y))
            
            cell_bg = Image.new('RGB', (self.CELL_SIZE, self.CELL_SIZE), (0,0,0))
            bg_offset_x, bg_offset_y = (self.CELL_SIZE - preview_w)//2, (self.CELL_SIZE - preview_h)//2
            draw = ImageDraw.Draw(cell_bg)
            border_rect = (bg_offset_x, bg_offset_y, bg_offset_x + preview_w -1, bg_offset_y + preview_h-1)
            draw.rectangle(border_rect, outline='#666', width=1)
            cell_bg.paste(layer_comp, (bg_offset_x, bg_offset_y), layer_comp)
            imgtk = ImageTk.PhotoImage(cell_bg)
            label.config(image=imgtk, text='')
            label.image = imgtk

    def _load_frame(self, path, for_thumbnail=False):
        if not path or not os.path.exists(path):
            return None
        ext = os.path.splitext(path)[1].lower()
        load_path = path

        if for_thumbnail and ext in VIDEO_EXTS:
            proxy_path = FFMpegProxyGenerator.get_proxy_path(path)
            if os.path.exists(proxy_path):
                load_path = proxy_path
            
        try:
            if ext in VIDEO_EXTS:
                cap = cv2.VideoCapture(load_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
            elif ext in IMAGE_EXTS:
                return Image.open(path).convert('RGBA')
        except Exception:
            logging.exception(f"Error loading frame from {load_path}")
            return None
        
        return None

    def _select_column(self, col):
        if self.transition_active: 
            return
        if not self.re: 
            return
        if col >= self.COLS: 
            col = self.COLS - 1
        if col < 0: 
            col = 0

        # Preserve existing selection semantics
        self.active_col = col

        # NEW: ensure the selected cue header is visually highlighted on selection/load
        # Use the same color used elsewhere for selected cue headers.
        try:
            for c, lbl in self.col_labels.items():
                lbl.config(bg=('#ADD8E6' if c == col else '#444'))
        except Exception:
            # Never let UI hiccups break selection or rendering
            pass

        # Build layers for the active column (unchanged)
        current_layers = self._get_layers_for_col(self.active_col)

        # Determine which video players to restart (unchanged)
        import os
        paths_to_restart = {
            layer['path'] for layer in current_layers.values()
            if layer['path'] and os.path.splitext(layer['path'])[1].lower() in VIDEO_EXTS
        }

        # Compute output/monitor routing (unchanged)
        output_mode = self.screen_var.get()
        target_monitor = self.selected_display if output_mode != self.MULTIPLE else None
        all_screens = {s for s in self.layer_screens.values() if s is not None} if output_mode == self.MULTIPLE else set()

        # Send render command (keep fields consistent with _trigger_column) 
        # including gui_monitor/output_mode/all_assigned_screens.
        self.render_command_queue.put({
            'action': 'render',
            'layers': current_layers,
            'target_monitor': target_monitor,
            'transition_state': {'active': False},
            'force_restart_paths': paths_to_restart,
            'gui_monitor': self.gui_monitor_index,
            'output_mode': output_mode,
            'all_assigned_screens': all_screens,
        })

        # >>> CHANGE: ensure a preview is requested after selecting a cue
        self.set_preview_dirty()

    def _initiate_transition(self, from_col, to_col):
        """Initiate a transition between the current active column and the target column."""
        import os, time, logging, shutil

        # Clean up any leftover links from prior transitions before creating/using links
        self._cleanup_transition_links(keep_paths=frozenset())

        transition_type = self.transition_var.get()
        try:
            duration = float(self.transition_timing_var.get())
        except Exception:
            duration = 0

        # Immediate cut path (unchanged)
        if transition_type == self.NONE or duration <= 0:
            if self.render_mode in ['opengl', 'pygame_gpu']:
                self.render_command_queue.put({'action': 'update_layers', 'layers': {}})
                self.root.after(50, lambda: self._select_column_immediate_cut(to_col))
            else:
                self._select_column(to_col)
            return

        # Build from/to layer maps
        from_layers = self._get_layers_for_col(from_col)
        to_layers = self._get_layers_for_col(to_col)

        # Ensure “to” videos start at 0 without disturbing “from” when the same file is used
        VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')
        from_video_paths = {
            l.get('path')
            for l in from_layers.values()
            if l and l.get('path') and os.path.splitext(l['path'])[1].lower() in VIDEO_EXTS
        }

        temp_links = []
        keep_now = set()
        for lid, layer in list(to_layers.items()):
            path = layer.get('path')
            if not path:
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext in VIDEO_EXTS and path in from_video_paths:
                link_path = self._get_or_create_vd_link(path, lid)
                if link_path and os.path.exists(link_path):
                    to_layers[lid] = {**layer, 'path': link_path}
                    temp_links.append(link_path)
                    keep_now.add(link_path)

        # Commit possibly-updated to_layers back to transition state
        self.transition_to_layers = to_layers
        self.transition_from_layers = from_layers
        self._temp_transition_links = temp_links

        # Remove any stale links not needed for this transition
        self._cleanup_transition_links(keep_paths=frozenset(keep_now))

        # Paths to restart = all "to" video paths (after linking), so they begin at 0
        paths_to_restart = {
            layer['path'] for layer in self.transition_to_layers.values()
            if layer.get('path') and os.path.splitext(layer['path'])[1].lower() in VIDEO_EXTS
        }

        # Set transition state and start timing
        self.transition_active = True
        self.transition_duration = duration
        self.transition_target_col = to_col
        self.transition_start_time = time.perf_counter()
        self._last_transition_progress = -1  # reset throttling state if present

        # Preload/keep-alive players and restart only the "to" paths
        self._update_active_players(force_restart_paths=paths_to_restart)

        # Start transition on the dedicated thread so GO actually animates
        if getattr(self, "transition_thread", None):
            self.transition_thread.start_transition(
                self.transition_from_layers,
                self.transition_to_layers,
                duration,
                to_col
            )

    def _finalize_transition_from_thread(self, target_col):
        """Called from transition thread when transition completes. Runs on main GUI thread."""
        import os

        # Mark transition done but KEEP transition_to_layers so temp-linked paths remain in use.
        self.transition_active = False
        self.active_col = target_col

        # Update UI
        for c, lbl in self.col_labels.items():
            lbl.config(bg=('#ADD8E6' if c == target_col else '#444'))
        self._update_cue_dropdown()

        # Promote temp links to active; then clean anything not needed anymore.
        new_active = set(getattr(self, '_temp_transition_links', []))
        self._active_transition_links = list(new_active)
        # Remove any temp links that weren’t promoted
        if hasattr(self, '_temp_transition_links'):
            for _p in self._temp_transition_links:
                if _p not in new_active:
                    try:
                        if _p and os.path.exists(_p):
                            os.remove(_p)
                    except Exception:
                        pass
        self._temp_transition_links = []

        # Render new active column WITHOUT restarting players
        output_mode = self.screen_var.get()
        target_monitor = self.selected_display if output_mode != self.MULTIPLE else None
        all_screens = {s for s in self.layer_screens.values() if s is not None} if output_mode == self.MULTIPLE else set()
        layers = self._get_layers_for_col(self.active_col)

        self.render_command_queue.put({
            'action': 'render',
            'layers': layers,                     # uses temp-linked paths via _get_layers_for_col override
            'target_monitor': target_monitor,
            'transition_state': {'active': False},
            'gui_monitor': self.gui_monitor_index,
            'output_mode': output_mode,
            'all_assigned_screens': all_screens,
        })

        # IMPORTANT: Now prune/refresh the active players to match the new cue (post-fade).
        # This removes the outgoing players and ensures only the new cue's videos remain.
        try:
            self.root.after(500, self._update_active_players)
        except Exception:
            pass



    def _select_column_immediate_cut(self, col):
        if col >= self.COLS: col = self.COLS - 1
        if col < 0: col = 0

        self.active_col = col

        for c, lbl in self.col_labels.items():
            lbl.config(bg=('#ADD8E6' if c == col else '#444'))

        current_layers = self._get_layers_for_col(self.active_col)
        
        # MODIFICATION: Determine which video players to restart
        paths_to_restart = {
            layer['path'] for layer in current_layers.values()
            if layer['path'] and os.path.splitext(layer['path'])[1].lower() in VIDEO_EXTS
        }
        self.render_command_queue.put({
            'action': 'update_layers',
            'layers': current_layers,
            'force_restart_paths': paths_to_restart
        })
        
        self._update_cue_dropdown()
        self._trigger_column()
        
        self.root.after(100, self._update_active_players)

    def _trigger_go(self):
        if self.presentation_active:
            self._stop_presentation()
        elif self.presentation_mode_var.get():
            self._start_presentation()
        else:
            self._go_to_next_cue_manual()

    def _go_to_next_cue_manual(self):
        if self.transition_active or not self.re: return
        from_col = self.active_col
        to_col = (from_col + 1) % self.COLS
        
        self._initiate_transition(from_col, to_col)

    def _get_layers_for_col(self, col):
        layers = {}
        output_mode = self.screen_var.get()

        for (r, c), p in self.grid_cells.items():
            if c == col and p:
                pos_data = self.cell_positions.get((r, c), {})

                screens = []
                if output_mode == self.MULTIPLE:
                    # JSON keys are strings
                    screen_idx = self.layer_screens.get(str(r))
                    if screen_idx is not None:
                        screens = [screen_idx]
                elif self.selected_display is not None:
                    screens = [self.selected_display]

                layers[r] = {
                    'path': p,
                    'screens': screens,
                    'order': r,
                    'position': pos_data,
                    'loop': pos_data.get('loop', True),
                }

        # If this column is the transition target, keep using temp-linked paths so playback
        # does NOT snap back to the original file/player. Do NOT apply this override to other columns.
        try:
            import os
            if col == getattr(self, "transition_target_col", None):
                to_layers = getattr(self, "transition_to_layers", {}) or {}
                for lid, t_layer in to_layers.items():
                    t_path = t_layer.get('path')
                    if t_path and lid in layers and os.path.exists(t_path):
                        layers[lid] = {**layers[lid], 'path': t_path}
        except Exception:
            # Never fail layer construction due to override logic.
            pass

        return layers

    def _on_cue_select(self, event):
        if self.transition_active: return
        selected_name = self.cue_name_var.get()
        for idx_str, name in self.cue_names.items():
            if name == selected_name:
                self._select_column(int(idx_str))
                break

    def _on_output_change(self, *args):
        if not self.re:
            return

        val = self.screen_var.get()
        is_multiple = (val == self.MULTIPLE)

        for r in range(self.ROWS):
            btn = self.layer_setting_widgets.get(r)
            if btn:
                if is_multiple:
                    btn.pack(side='bottom', anchor='center', pady=0)
                else:
                    btn.pack_forget()

        if is_multiple:
            self.selected_display = None
        elif val == self.NONE:
            self.selected_display = None
        else:
            try:
                idx = int(val.split()[-1])
                if self.re.monitors and 0 <= idx < len(self.re.monitors):
                    self.selected_display = idx
                else:
                    self.selected_display = None
                    self.screen_var.set(self.NONE)
            except (ValueError, IndexError):
                self.selected_display = None

        # Gate actual render trigger until startup finishes
        if getattr(self, "_startup_inhibit_triggers", False):
            return

        if getattr(self, "_ui_ready", True):
            self._trigger_column()
        else:
            self._deferred_trigger = True

        # Update cell preview borders and mark preview dirty
        for r in range(self.ROWS):
            for c in range(self.COLS):
                self._update_cell_preview(r, c)

        # NEW: if Multiple, immediately blank the preview panel
        if is_multiple:
            try:
                self._update_preview_canvas_with_image(None)
            except Exception:
                pass

        self.set_preview_dirty()
        self._set_dirty_flag()
        
    def _trigger_column(self):
        """Sends the current render state to the render thread."""
        if not self.re:
            return
        if getattr(self, "_startup_inhibit_triggers", False):
            return
        if not getattr(self, "_ui_ready", True):
            self._deferred_trigger = True
            return

        layers = self._get_layers_for_col(self.active_col)
        output_mode = self.screen_var.get()

        target_monitor = self.selected_display if output_mode != self.MULTIPLE else None

        all_screens = set()
        if output_mode == self.MULTIPLE:
            all_screens = {s for s in self.layer_screens.values() if s is not None}

        command = {
            'action': 'render',
            'layers': layers,
            'target_monitor': target_monitor,
            'transition_state': {'active': False},
            'gui_monitor': self.gui_monitor_index,
            'output_mode': output_mode,
            'all_assigned_screens': all_screens,
        }
        self.render_command_queue.put(command)
        self.set_preview_dirty()

    def _show_cell_menu(self, event, row, col):
        menu = Menu(self.root, tearoff=0)
        path = self.grid_cells.get((row, col))
        if path and os.path.exists(path):
            menu.add_command(label='Edit', command=lambda: self._edit_media(row, col))

            ext = os.path.splitext(path)[1].lower()
            if ext in VIDEO_EXTS:
                loop_var = tk.BooleanVar()
                loop_var.set(self.cell_positions.get((row, col), {}).get('loop', True))
                menu.add_checkbutton(label="Loop", variable=loop_var,
                                     command=lambda r=row, c=col, v=loop_var: self._toggle_loop(r, c, v))

            menu.add_separator()
        menu.add_command(label='Clear', command=lambda: self._clear_cell(row, col))
        menu.tk_popup(event.x_root, event.y_root)
        
    def _show_layer_output_menu(self, row):
        """Shows a popup menu to select the output screen for a layer."""
        menu = Menu(self.root, tearoff=0)
        current_screen = self.layer_screens.get(str(row)) # JSON keys are strings

        # Add 'None' option
        none_label = "✓ None" if current_screen is None else "  None"
        menu.add_command(label=none_label, command=lambda r=row: self._set_layer_output(r, None))
        menu.add_separator()
        
        # Add each available display
        if self.re:
            for i in range(len(self.re.monitors)):
                display_label = f"✓ Display {i}" if current_screen == i else f"  Display {i}"
                menu.add_command(label=display_label, command=lambda r=row, idx=i: self._set_layer_output(r, idx))
        
        # Position and show the menu
        btn = self.layer_setting_widgets[row]
        x = btn.winfo_rootx() + btn.winfo_width()
        y = btn.winfo_rooty()
        menu.tk_popup(x, y)

    def _set_layer_output(self, row, screen_index):
        """Callback to set a layer's target screen and update rendering."""
        row_str = str(row)
        if screen_index is None:
            if row_str in self.layer_screens:
                del self.layer_screens[row_str]
        else:
            self.layer_screens[row_str] = screen_index
        
        logging.info(f"Set Layer {row+1} output to: {'Display ' + str(screen_index) if screen_index is not None else 'None'}")
        
        # Update cell previews for the entire row
        for c in range(self.COLS):
            self._update_cell_preview(row, c)
            
        self._trigger_column()
        self._set_dirty_flag()

    def _toggle_loop(self, row, col, var):
        if (row, col) in self.cell_positions:
            self.cell_positions[(row, col)]['loop'] = var.get()
            self._update_active_players()
            self._set_dirty_flag()

    def _clear_cell(self, row, col):
        if self.grid_cells.get((row, col)) is not None:
            self._set_dirty_flag()
        self.grid_cells[(row, col)] = None
        if (row, col) in self.cell_positions: 
            del self.cell_positions[(row, col)]
        self._update_cell_preview(row, col)
        
        # ADD THIS - Invalidate static cache for this column
        if self.re:
            self.re.invalidate_static_cache_for_column(col)
        
        if col == self.active_col:
            self._trigger_column()

    def _show_cue_menu(self, event, col):
        menu = Menu(self.root, tearoff=0)
        menu.add_command(label='Rename', command=lambda: self._rename_cue(col))
        menu.add_separator()
        menu.add_command(label='Clear Cue', command=lambda: self._clear_cue(col))
        menu.add_command(label='Delete Cue', command=lambda: self._delete_cue(col))
        menu.tk_popup(event.x_root, event.y_root)

    def _clear_cue(self, col):
        for r in range(self.ROWS): self._clear_cell(r, col)

    def _show_layer_menu(self, event, row):
        menu = Menu(self.root, tearoff=0)
        menu.add_command(label='Rename', command=lambda: self._rename_layer(row))
        menu.add_separator()
        menu.add_command(label='Clear Layer', command=lambda: self._clear_layer(row))
        menu.add_command(label='Delete Layer', command=lambda: self._delete_layer(row))
        menu.tk_popup(event.x_root, event.y_root)

    def _clear_layer(self, row):
        for c in range(self.COLS): self._clear_cell(row, c)

    def _show_media_menu(self, event, path):
        menu = Menu(self.root, tearoff=0)
        if os.path.exists(path):
            ext = os.path.splitext(path)[1].lower()
            # Only show Optimize for videos if explicitly allowed
            if ext in VIDEO_EXTS and VD_SHOW_VIDEO_OPTIMIZE_MENU:
                menu.add_command(label='Optimize', command=lambda: self._open_optimize_window(path))
                menu.add_separator()
        menu.add_command(label='Remove from Project', command=lambda: self._remove_media(path, delete_from_disk=False))
        # Hide the destructive option unless explicitly allowed
        if VD_SHOW_DELETE_FROM_DISK_MENU:
            menu.add_command(label='Remove and DELETE FROM DISK',
                             command=lambda: self._remove_media(path, delete_from_disk=True))
        menu.tk_popup(event.x_root, event.y_root)


    def _rename_cue(self, col):
        col_str = str(col)
        current_name = self.cue_names.get(col_str, f'Cue {col+1}')
        new_name = simpledialog.askstring('Rename Cue', 'Enter new name:', initialvalue=current_name)
        if new_name:
            self.cue_names[col_str] = new_name
            self.col_labels[col].config(text=new_name)
            self._update_cue_dropdown()
            self._set_dirty_flag()

    def _rename_layer(self, row):
        row_str = str(row)
        current_name = self.layer_names.get(row_str, f'Layer {row+1}')
        new_name = simpledialog.askstring('Rename Layer', 'Enter new name:', initialvalue=current_name)
        if new_name:
            self.layer_names[row_str] = new_name
            canvas = self.row_labels[row]
            canvas.itemconfig(canvas.text_id, text=new_name)
            self._set_dirty_flag()

    def _delete_cue(self, col):
        if self.COLS <= 1: return
        if not tk.messagebox.askyesno('Delete Cue', f'Delete "{self.cue_names.get(str(col), f"Cue {col+1}")}"?'): return
        new_cells, new_positions, new_names = {}, {}, {}
        for (r, c), media in self.grid_cells.items():
            if c < col:
                new_cells[(r, c)] = media
                if (r, c) in self.cell_positions: new_positions[(r, c)] = self.cell_positions[(r, c)]
            elif c > col:
                new_cells[(r, c-1)] = media
                if (r, c) in self.cell_positions: new_positions[(r, c-1)] = self.cell_positions[(r, c)]
        for c_str, name in self.cue_names.items():
            c_int = int(c_str)
            if c_int < col: new_names[c_str] = name
            elif c_int > col: new_names[str(c_int-1)] = name
        self.grid_cells, self.cell_positions, self.cue_names = new_cells, new_positions, new_names
        self.COLS -= 1
        if self.active_col >= self.COLS: self.active_col = self.COLS - 1
        elif self.active_col > col: self.active_col -= 1
        self._build_grid()
        self._select_column(self.active_col)
        self._set_dirty_flag()

    def _delete_layer(self, row):
        if self.ROWS <= 1: return
        if not tk.messagebox.askyesno('Delete Layer', f'Delete "{self.layer_names.get(str(row), f"Layer {row+1}")}"?'): return
        
        new_cells, new_positions, new_names = {}, {}, {}
        new_layer_screens = {}

        for (r, c), media in self.grid_cells.items():
            if r < row:
                new_cells[(r, c)] = media
                if (r, c) in self.cell_positions: new_positions[(r, c)] = self.cell_positions[(r, c)]
            elif r > row:
                new_cells[(r-1, c)] = media
                if (r, c) in self.cell_positions: new_positions[(r-1, c)] = self.cell_positions[(r, c)]
        
        for r_str, name in self.layer_names.items():
            r_int = int(r_str)
            if r_int < row: new_names[r_str] = name
            elif r_int > row: new_names[str(r_int-1)] = name
            
        for r_str, screen in self.layer_screens.items():
            r_int = int(r_str)
            if r_int < row: new_layer_screens[r_str] = screen
            elif r_int > row: new_layer_screens[str(r_int-1)] = screen

        self.grid_cells = new_cells
        self.cell_positions = new_positions
        self.layer_names = new_names
        self.layer_screens = new_layer_screens
        self.ROWS -= 1
        
        self._build_grid()
        self._select_column(self.active_col)
        self._set_dirty_flag()

    def _remove_media(self, path, delete_from_disk=False):
        if delete_from_disk:
            prompt = f"This will attempt to PERMANENTLY DELETE the following file from your computer. This action cannot be undone.\n\n{path}\n\nAre you absolutely sure?"
            if not messagebox.askyesno("Confirm Permanent Deletion", prompt):
                return
            try:
                if os.path.exists(path):
                    os.remove(path)
                # Also remove proxy if it exists
                proxy_path = FFMpegProxyGenerator.get_proxy_path(path)
                if os.path.exists(proxy_path):
                    os.remove(proxy_path)
            except OSError as e:
                messagebox.showerror("Error", f"Could not delete file:\n{e}")
        
        # Remove from project's media list
        if path in self.project_media_paths:
            self.project_media_paths.remove(path)
            self._set_dirty_flag()

        # Clear any cells using this media
        for (r, c), media_path in list(self.grid_cells.items()):
            if media_path == path: self._clear_cell(r, c)
        
        self._build_media_list()

    def _menu_add(self):
        filetypes = [('Media Files', ' '.join(f'*{ext}' for ext in SUPPORTED_EXTS)), ('All files', '*.*')]
        paths = filedialog.askopenfilenames(title="Add Media to Project", filetypes=filetypes)
        
        added_any = False
        for p in paths:
            if p not in self.project_media_paths:
                self.project_media_paths.append(p)
                added_any = True
        
        if added_any:
            self._build_media_list()
            self._set_dirty_flag()

    def _add_cue(self):
        self.COLS += 1
        for r in range(self.ROWS): self.grid_cells[(r, self.COLS-1)] = None
        self.cue_names[str(self.COLS-1)] = f'Cue {self.COLS}'
        self._build_grid()
        self._select_column(self.active_col)
        self._set_dirty_flag()

    def _add_layer(self):
        self.ROWS += 1
        for c in range(self.COLS): self.grid_cells[(self.ROWS-1, c)] = None
        self.layer_names[str(self.ROWS-1)] = f'Layer {self.ROWS}'
        self._build_grid()
        self._select_column(self.active_col)
        self._set_dirty_flag()

    def _update_cue_dropdown(self):
        cue_list = sorted(self.cue_names.values(), key=lambda name: [int(s) if s.isdigit() else s for s in re.split('([0-9]+)', name)])
        self.cue_select_combobox['values'] = cue_list
        if self.COLS > 0 and self.active_col < self.COLS:
            self.cue_name_var.set(self.cue_names.get(str(self.active_col), ''))

    def _edit_media(self, row, col):
        if not self.re: return
        media_path = self.grid_cells.get((row, col))
        if not media_path or not os.path.exists(media_path):
            messagebox.showinfo("Media Missing", "Cannot edit, the media file is missing.")
            return
        
        output_mode = self.screen_var.get()
        if output_mode == self.MULTIPLE:
            screen_idx = self.layer_screens.get(str(row))
        else:
            screen_idx = self.selected_display

        initial_position = self.cell_positions.get((row, col))
        editor = MediaEditor(
            self, media_path, screen_idx, self.re, initial_pos=initial_position,
            callback=lambda path, pos: self._on_editor_save(row, col, path, pos),
            realtime_callback=lambda pos: self._on_editor_update_realtime(row, col, pos),
            preview_bridge=self.preview_bridge)

    def _on_editor_save(self, row, col, media_path, position_data):
        self.cell_positions[(row, col)] = position_data
        self._update_cell_preview(row, col)
        
        # ADD THIS - Invalidate static cache for this column
        if self.re:
            self.re.invalidate_static_cache_for_column(col)
        
        if col == self.active_col:
            self._trigger_column()
        self._set_dirty_flag()

    def _on_editor_update_realtime(self, row, col, position_data):
        self.cell_positions[(row, col)] = position_data
        if col == self.active_col:
            self._trigger_column()

    def _update_fracs(self, event=None):
        self.pan.update_idletasks()
        w = max(self.pan.winfo_width(), 1)
        try:
            x1, _ = self.pan.sash_coord(0)
            x2, _ = self.pan.sash_coord(1)
            self.frac1, self.frac2 = x1/w, x2/w
        except: pass

    def _apply_fracs(self):
        self.pan.update_idletasks()
        W, H = self.pan.winfo_width(), self.pan.winfo_height()
        x1, x2 = int(W*self.frac1), int(W*self.frac2)
        y = H//2
        try:
            self.pan.sash_place(0, x1, y)
            self.pan.sash_place(1, x2, y)
        except: pass

    def _on_mousewheel(self, event):
        if isinstance(event.widget.winfo_toplevel(), tk.Toplevel): return
        x, y = self.root.winfo_pointerx(), self.root.winfo_pointery()
        gx, gy = self.grid_canvas.winfo_rootx(), self.grid_canvas.winfo_rooty()
        gw, gh = self.grid_canvas.winfo_width(), self.grid_canvas.winfo_height()
        if gx <= x <= gx+gw and gy <= y <= gy+gh:
            delta = -1 if getattr(event, 'num', None) == 5 else int(-event.delta/120)
            top_frac, _ = self.grid_canvas.yview()
            if top_frac <= 0.0 and delta < 0:
                self.grid_canvas.yview_moveto(0)
                return
            self.grid_canvas.yview_scroll(delta, 'units')

    def _on_shift_mousewheel(self, event):
        if isinstance(event.widget.winfo_toplevel(), tk.Toplevel): return
        x, y = self.root.winfo_pointerx(), self.root.winfo_pointery()
        gx, gy = self.grid_canvas.winfo_rootx(), self.grid_canvas.winfo_rooty()
        gw, gh = self.grid_canvas.winfo_width(), self.grid_canvas.winfo_height()
        if gx <= x <= gx+gw and gy <= y <= gy+gh:
            delta = int(-event.delta/120)
            self.grid_canvas.xview_scroll(delta, 'units')
            
    # --- Presentation Mode Methods ---

    def _on_toggle_presentation_mode(self, *args):
        is_enabled = self.presentation_mode_var.get()
        new_state = 'normal' if is_enabled else 'disabled'
        self.presentation_timing_entry.config(state=new_state)
        self.presentation_loop_check.config(state=new_state)
        
        if not is_enabled and self.presentation_active:
            self._stop_presentation()
        
        self.go_button.config(text="GO", bg='#5a5')
        self._set_dirty_flag()


    def _start_presentation(self):
        if self.presentation_active: return
        logging.info("Starting presentation...")
        self.presentation_active = True
        self.go_button.config(text="STOP", bg='#a55')
        self._advance_presentation()

    def _stop_presentation(self):
        if not self.presentation_active: return
        logging.info("Stopping presentation...")
        if self.presentation_timer:
            self.root.after_cancel(self.presentation_timer)
            self.presentation_timer = None
        self.presentation_active = False
        self.go_button.config(text="GO", bg='#5a5')

    def _advance_presentation(self):
        if not self.presentation_active: return

        from_col = self.active_col
        next_col = self._find_next_cue_with_media(from_col)

        if next_col is None:
            logging.info("End of presentation.")
            self._stop_presentation()
            return

        self._initiate_transition(from_col, next_col)

        try:
            delay_sec = float(self.presentation_timing_var.get())
        except ValueError:
            delay_sec = 5.0
        
        delay_ms = int(delay_sec * 1000)
        self.presentation_timer = self.root.after(delay_ms, self._advance_presentation)

    def _generate_thumbnail(self, path, size=(80, 45)):
        """Generate a thumbnail for the given media path."""
        pil_img = self._load_frame(path, for_thumbnail=True)
        if pil_img:
            pil_img.thumbnail(size)
            return ImageTk.PhotoImage(pil_img)
        return None

    def _find_next_cue_with_media(self, start_col):
        """Finds the next column with media, skipping empty ones."""
        # First, search from the next column to the end
        for c in range(start_col + 1, self.COLS):
            if any(self.grid_cells.get((r, c)) for r in range(self.ROWS)):
                return c

        # If loop is disabled and we reached the end, stop.
        if not self.presentation_loop_var.get():
            return None

        # If loop is enabled, search from the beginning up to the start column
        for c in range(start_col + 1): # Search from 0 up to and including start_col
            if any(self.grid_cells.get((r, c)) for r in range(self.ROWS)):
                return c
        
        # If still nothing is found (e.g., only one cue with media), return None
        return None

    # --- Project Management Methods ---

    def _set_dirty_flag(self, event=None):
        if not self.project_is_dirty:
            self.project_is_dirty = True
            self._update_title()

    def _update_title(self):
        name = os.path.basename(self.current_project_path) if self.current_project_path else "Untitled Project"
        dirty_marker = "*" if self.project_is_dirty else ""
        self.root.title(f"VisualDeck - {name}{dirty_marker}")

    def _check_before_closing_project(self):
        if not self.project_is_dirty:
            return True
        
        response = messagebox.askyesnocancel("Save Changes?", "You have unsaved changes. Would you like to save them?")
        
        if response is True:  # Yes
            return self._save_project()
        elif response is False:  # No
            return True
        else:  # Cancel
            return False

    def _new_project(self, initial_load=False):
        if not initial_load:
            if not self._check_before_closing_project():
                return
        
        self._reset_project_state(new_project=True)
        
        self._build_media_list()
        self._build_grid()
        self._update_cue_dropdown()
        self._update_title()
        self._trigger_column()

    def _auto_load_last_project(self):
        # This is the very first action after startup, so we don't need to check for unsaved changes.
        if self.recent_files:
            last_project = self.recent_files[0]
            if os.path.exists(last_project):
                logging.info(f"Auto-loading last project: {last_project}")
                # The `initial_load=True` flag will bypass the "save changes?" check
                self._open_project(path=last_project, initial_load=True)
            else:
                logging.warning("Last project not found, starting new project.")
                self._new_project(initial_load=True)
        else:
            self._new_project(initial_load=True)

    def _open_project(self, path=None, initial_load=False):
        if not initial_load:
            if not self._check_before_closing_project():
                return
        
        if path is None:
            path = filedialog.askopenfilename(
                title="Open Project",
                filetypes=[("VisualDeck Project", "*.vjdeck"), ("All files", "*.*")]
            )
        
        if not path: return

        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self._apply_project_data(data, path)
            self._add_to_recent(path)
            self._save_config()
            logging.info(f"Successfully opened project: {path}")

        except Exception:
            logging.exception(f"Could not open or parse project file: {path}")
            messagebox.showerror("Error Opening Project", f"Could not open or parse project file.")
            return

    def _apply_project_data(self, data, path):
        """Applies loaded project data to the application state."""
        self._reset_project_state(new_project=False)
        
        # Load settings
        settings = data.get('settings', {})
        self.render_mode = settings.get('render_mode', 'opengl')
        self.hw_accel = settings.get('hw_accel', 'auto')

        # Load grid dimensions and names
        grid = data.get('grid', {})
        self.ROWS = grid.get('rows', self.DEFAULT_ROWS)
        self.COLS = grid.get('cols', self.DEFAULT_COLS)
        active_col = grid.get('active_cue', 0)
        self.cue_names = data.get('cues', {str(i): f'Cue {i+1}' for i in range(self.COLS)})
        self.layer_names = data.get('layers', {str(i): f'Layer {i+1}' for i in range(self.ROWS)})
        
        # Load project media list
        self.project_media_paths = data.get('project_media', [])

        # Load media cells
        self.grid_cells = {}
        self.cell_positions = {}
        for key, val in data.get('media_cells', {}).items():
            try:
                r_s, c_s = key.split(',')
                r, c = int(r_s), int(c_s)
                self.grid_cells[(r, c)] = val['path'] # Paths are stored as absolute
                self.cell_positions[(r, c)] = val['pos']
            except (ValueError, KeyError):
                logging.warning(f"Could not parse media cell data for key '{key}'")

        # Load output settings
        outputs = data.get('outputs', {})
        self.screen_var.set(outputs.get('mode', self.NONE))
        self.layer_screens = outputs.get('layer_screens', {})

        # Load transition and presentation settings
        trans = data.get('transitions', {})
        self.transition_var.set(trans.get('type', 'Fade'))
        self.transition_timing_var.set(trans.get('duration', '1.0'))

        pres = data.get('presentation', {})
        self.presentation_mode_var.set(pres.get('enabled', False))
        self.presentation_timing_var.set(pres.get('duration', '5.0'))
        self.presentation_loop_var.set(pres.get('loop', False))
        self._on_toggle_presentation_mode()

        # Update state and UI
        self.current_project_path = path
        self.project_is_dirty = False
        self._update_title()
        self._initialize_render_engine()
        self._build_media_list()
        self._build_grid()
        for r in range(self.ROWS):
            for c in range(self.COLS):
                self._update_cell_preview(r, c)
        self._select_column(active_col)

    def _save_project(self):
        if not self.current_project_path:
            return self._save_project_as()
        else:
            self._perform_save(self.current_project_path)
            return True

    def _save_project_as(self):
        path = filedialog.asksaveasfilename(
            title="Save Project As",
            initialfile="Untitled.vjdeck",
            defaultextension=".vjdeck",
            filetypes=[("VisualDeck Project", "*.vjdeck"), ("All files", "*.*")]
        )
        if path:
            self._perform_save(path)
            self._add_to_recent(path)
            self._save_config()
            return True
        return False

    def _perform_save(self, path):
        """Gathers all state and writes it to a JSON file."""
        data = {}
        data['version'] = '2.8.3'
        data['settings'] = {'render_mode': self.render_mode, 'hw_accel': self.hw_accel}
        data['grid'] = {'rows': self.ROWS, 'cols': self.COLS, 'active_cue': self.active_col}
        data['cues'] = self.cue_names
        data['layers'] = self.layer_names
        data['project_media'] = self.project_media_paths
        data['outputs'] = {'mode': self.screen_var.get(), 'layer_screens': self.layer_screens}
        data['transitions'] = {'type': self.transition_var.get(), 'duration': self.transition_timing_var.get()}
        data['presentation'] = {
            'enabled': self.presentation_mode_var.get(),
            'duration': self.presentation_timing_var.get(),
            'loop': self.presentation_loop_var.get()
        }

        media_cells = {}
        for (r, c), media_path in self.grid_cells.items():
            if media_path:
                media_cells[f"{r},{c}"] = {
                    'path': media_path, # Store absolute path
                    'pos': self.cell_positions.get((r, c))
                }
        data['media_cells'] = media_cells

        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            
            self.current_project_path = path
            self.project_is_dirty = False
            self._update_title()
            logging.info(f"Project saved to {path}")
        except Exception:
            logging.exception(f"Could not save project file: {path}")
            messagebox.showerror("Save Error", f"Could not save project file.")

    # --- Recent Files Methods ---
    def _load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.recent_files = [p for p in config.get('recent_files', []) if os.path.exists(os.path.dirname(p))]
                    self.render_mode = config.get('render_mode', 'cpu')
                    self.hw_accel = config.get('hw_accel', 'auto')
                    self.use_pbo = config.get('use_pbo', True)
                    self.use_vsync = config.get('use_vsync', False)
                    self.force_multithread = config.get('force_multithread', True)
                    self.use_instancing = config.get('use_instancing', True)  # ADD THIS LINE
        except (json.JSONDecodeError, IOError):
            self.recent_files = []
            self.use_instancing = True  # ADD THIS DEFAULT

    def _save_config(self):
        try:
            with open(self.config_path, 'w') as f:
                config_data = {
                    'recent_files': self.recent_files,
                    'render_mode': self.render_mode,
                    'hw_accel': self.hw_accel,
                    'use_pbo': self.use_pbo,
                    'use_vsync': self.use_vsync,
                    'force_multithread': self.force_multithread,
                    'use_instancing': self.use_instancing  # ADD THIS LINE
                }
                json.dump(config_data, f)
        except IOError as e:
            logging.warning(f"Could not save config file: {e}")

    def _add_to_recent(self, path):
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        self.recent_files = self.recent_files[:10] # Keep max 10 recent files
        self._populate_recent_menu()

    def _populate_recent_menu(self):
        self.recent_menu.delete(0, 'end')
        if not self.recent_files:
            self.recent_menu.add_command(label="(No recent projects)", state='disabled')
        else:
            for path in self.recent_files:
                self.recent_menu.add_command(label=path, command=lambda p=path: self._open_project(p))
