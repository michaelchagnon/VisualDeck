# SPDX-License-Identifier: MPL-2.0
# =====================================================================
# src/control_interface.py
# =====================================================================
import os
import itertools
import collections
import copy
import cv2
import gc
import datetime
import hashlib
import math
import numpy as np
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
import ui_theme as T
import version
from tkinter import (
    ttk, filedialog, Menu, PanedWindow, Frame, Label,
    Button, Canvas, Scrollbar, simpledialog, messagebox
)
from PIL import Image, ImageTk, ImageDraw
from utils.file_watcher import FileWatcher
from utils.tooltips import CreateToolTip
from utils.ffmpeg_utils import FFMpegOptimizer, FFMpegProxyGenerator
from utils.preview_bridge import PreviewBridge
from project_io import atomic_write_json
import media_relink
import health
import preview_compose
import project_schema
import dmx_remote
import edge_blend
import midi_remote
import scheduler
import ltc
import timecode
import mesh_warp
import span as span_mod
import soak
import output_warp
import output_presets
import remote_control
import undo
import web_remote
import test_patterns
import show_collect
import shortcuts
from editor import MediaEditor
from render_engine import PygameRenderEngine, OpenGLRenderEngine, transition_alphas, is_streamed, layer_volume, \
    layer_trim
import audio_engine
import live_source
import playback_copy
import isf
from pygame.locals import NOFRAME, DOUBLEBUF, OPENGL
from ring_buffer import SimpleFrameQueue
from ring_buffer import SimpleFrameQueue
from render_engine import PreciseFramePacer, video_still
from OpenGL.GL import glDeleteTextures
from concurrent.futures import ThreadPoolExecutor
from displays import monitors as get_monitors   # screeninfo without its leaks (displays.py)

log = logging.getLogger("visualdeck.control")

# Windows-specific imports
if sys.platform == "win32":
    try:
        import win32api
        import win32process
        import win32con
        WINDOWS_IMPORTS_AVAILABLE = True
    except ImportError:
        WINDOWS_IMPORTS_AVAILABLE = False
        log.warning("pywin32 not installed. Windows optimizations will be limited.")
else:
    WINDOWS_IMPORTS_AVAILABLE = False

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
SUPPORTED_EXTS = VIDEO_EXTS + IMAGE_EXTS

SPLASH_VERSION_TEXT = f"v{version.LABEL}"   # "" disables the version label.
SPLASH_VERSION_FONT_PT = 14        # Point size (Word-equivalent).
SPLASH_VERSION_FONT_PATH = None    # Optional full path to .ttf/.otf
SPLASH_VERSION_MARGIN_X = 16       # Horizontal margin from the right edge (px)
SPLASH_VERSION_MARGIN_Y = 16       # Vertical margin from the bottom edge (px)
VD_SHOW_VIDEO_OPTIMIZE_MENU = False      # hide "Optimize" on video right-click menu
VD_SHOW_DELETE_FROM_DISK_MENU = False    # hide "Remove and DELETE FROM DISK"

def _moving_effect(layer):
    """True if the layer has a shader effect that changes by itself (isf.animated)."""
    return any(isf.animated(fx['name']) for fx in isf.layer_stack(layer))


class RenderThread(threading.Thread):
    """A dedicated thread for handling all rendering operations."""
    def __init__(self, render_engine, command_queue, preview_queue, stop_event):
        super().__init__(daemon=True, name="RenderThread")
        self.re = render_engine
        self.command_queue = command_queue
        self.preview_queue = preview_queue
        self.stop_event = stop_event
        self.audio = None          # audio_engine.AudioEngine, set by the UI when audio is on
        # ── Adaptive performance state ──
        self._adaptive_levels = [60, 45, 30]   # target FPS choices
        self._level_idx = 0                    # start at 60
        self._stable_ok_frames = 0
        self._stress_frames = 0
        self._raise_after = 240                # ~4s at 60fps of stability
        self._lower_after = 60                 # ~1s of stress to step down
        self._preview_cap_scale = 1.0          # 1.0 → 0.75 → 0.5 under stress
        self._pending_cut = None   # see _begin_preroll
        # Health: per-frame timing, error history, cut preroll waits; the
        # snapshot is replaced (never mutated) every HEALTH_PERIOD seconds.
        self.error_counts = {}
        self.frame_stats = health.FrameStats()
        self.loop_stats = health.LoopStats()
        self.error_log = health.ErrorLog()
        self.gc_stats = health.GcStats()      # installed while the thread runs
        self.preroll_events = collections.deque(maxlen=200)  # (t, wait_ms, timed_out)
        self.health_snapshot = None
        self.current_render_state = {
            'layers': {},
            'target_monitor': None,
            'transition_state': {'active': False},
            'request_preview': True,
            'gui_monitor': 0,
            'output_mode': 'None',
            'all_assigned_screens': set(),
        }

    # -- fault isolation ----------------------------------------------------
    def _report_error(self, phase, exc):
        """Log a failure in one phase of the loop without stopping the thread.

        Full tracebacks for the first few occurrences, then every 100th, so a
        persistent fault cannot flood the log at 60 Hz.
        """
        n = self.error_counts.get(phase, 0) + 1
        self.error_counts[phase] = n
        try:
            self.error_log.add(time.perf_counter(), phase, exc)
        except Exception:  # noqa: BLE001
            pass
        if n <= 5 or n % 100 == 0:
            log.error(f"Render thread: {phase} failed (occurrence {n}); output continues", exc_info=exc)

    def run(self):
        try:
            if hasattr(self.re, '_init_pygame_and_gl'):
                self.re._init_pygame_and_gl()
        except Exception as e:
            self._report_error('init', e)

        if sys.platform == "win32" and WINDOWS_IMPORTS_AVAILABLE:
            try:
                thread_handle = win32api.GetCurrentThread()
                win32process.SetThreadPriority(thread_handle, win32process.THREAD_PRIORITY_TIME_CRITICAL)
                log.info("Set render thread priority to TIME_CRITICAL")
            except Exception as e:
                log.warning(f"Could not set render thread priority: {e}")

        self.gc_stats.install()
        from render_engine import PreciseFramePacer
        self.frame_pacer = PreciseFramePacer(target_fps=self._adaptive_levels[self._level_idx])
        self.static_frame_pacer = PreciseFramePacer(target_fps=30)
        state_dirty = False
        last_render_time = 0.0
        min_render_interval = 1.0 / 120.0  # Cap at 120 FPS max
        last_health = 0.0
        last_health_log = time.perf_counter()  # first summary after one period

        while not self.stop_event.is_set():
            frame_start_time = time.perf_counter()
            phases = {}
            rendered = False

            if self.re.window:
                try:
                    pygame.event.pump()
                except Exception as e:
                    self._report_error('event pump', e)
            t_cmd = time.perf_counter()
            phases['pump'] = t_cmd - frame_start_time

            # 1) commands (each isolated: one bad command must not drop the rest)
            command_deadline = frame_start_time + 0.002  # ~2ms
            commands_processed = 0
            while commands_processed < 10 and time.perf_counter() < command_deadline:
                try:
                    command = self.command_queue.get_nowait()
                except queue.Empty:
                    break
                commands_processed += 1
                try:
                    state_dirty |= self._apply_command(command)
                except Exception as e:
                    self._report_error(f"command '{command.get('action')}'", e)
            if self.stop_event.is_set():
                break
            if self._pending_cut is not None:
                try:
                    if self._preroll_ready():
                        self._commit_preroll()
                        state_dirty = True
                except Exception as e:
                    self._pending_cut = None
                    self.re.preroll_keep = set()
                    self._report_error('cue preroll', e)

            state = self.current_render_state
            current_layers = state.get('layers', {}) or {}
            # A layer with a moving effect (it uses TIME) is redrawn every frame like a video.
            fx = getattr(self.re, 'draws_effects', False)
            has_active_videos = any(is_streamed(layer.get('path')) or (fx and _moving_effect(layer))
                                    for layer in current_layers.values()) \
                or (fx and bool(current_layers) and self.re.output_effects_moving)
            t_after_cmd = time.perf_counter()
            phases['commands'] = t_after_cmd - t_cmd
            transition_active = state.get('transition_state', {}).get('active', False)

            # 2) decide whether to render this iteration
            time_since_last_render = frame_start_time - last_render_time
            should_render = state_dirty
            if not should_render and (has_active_videos or transition_active):
                # Every pacer tick renders. (The old test, ">= 1/60 s since the
                # last render", failed whenever a tick came a fraction of a
                # millisecond early, so about a third of frames were skipped and
                # shown late at a 60 fps target; owner's report 2026-11-12.)
                should_render = time_since_last_render >= 0.5 / self.frame_pacer.get_target_fps()
            if not should_render and getattr(self.re, 'redraw_pending', False):
                self.re.redraw_pending = False           # e.g. new mesh maps for a still picture
                should_render = True

            preview_wanted = bool(state.get('request_preview'))
            gpu_preview_monitor = self._gpu_preview_monitor() if preview_wanted else None
            if gpu_preview_monitor is not None:
                self.re.request_preview_capture(gpu_preview_monitor)
                should_render = True

            # 3) render
            if should_render and time_since_last_render >= min_render_interval:
                t_render = time.perf_counter()
                try:
                    self.re.render(
                        state['layers'],
                        state['target_monitor'],
                        transition_state=state['transition_state'],
                        all_assigned_screens=state['all_assigned_screens']
                    )
                except Exception as e:
                    self._report_error('render', e)
                t_done = time.perf_counter()
                self.frame_stats.record(t_done, t_done - t_render, has_active_videos or transition_active,
                                        self.frame_pacer.get_target_fps())
                phases['render'] = t_done - t_render
                rendered = True
                last_render_time = frame_start_time
                state_dirty = False
            t_preview = time.perf_counter()

            # 4) preview: GPU capture from this frame, else CPU composite
            if preview_wanted:
                preview_img = None
                try:
                    if gpu_preview_monitor is not None:
                        preview_img = self.re.take_preview_image()
                    if preview_img is None:
                        preview_img = self._cpu_preview()
                except Exception as e:
                    self._report_error('preview', e)
                try:
                    if not self.preview_queue.empty():
                        self.preview_queue.get_nowait()
                    self.preview_queue.put_nowait(preview_img)
                except (queue.Full, queue.Empty):
                    pass
                state['request_preview'] = False

            if self.audio is not None:
                try:
                    self._update_audio(state)
                except Exception as e:
                    self._report_error('audio', e)
            if getattr(self, '_fx_preview_job', None) is not None or getattr(self.re, 'effect_preview_busy', bool)():
                try:
                    self._run_fx_preview()
                except Exception as e:
                    job, self._fx_preview_job = getattr(self, '_fx_preview_job', None), None
                    if job is not None:
                        job['reply'].put((job.get('seq'), None, f"preview failed: {e}"))
                    self._report_error('effect preview', e)

            # Effects of the show compiled ahead of their first draw, one per loop, never
            # during a transition (a compile can take a frame or more on some drivers).
            if not transition_active and getattr(self.re, '_fx_warm', None):
                try:
                    self.re.warm_one_effect()
                except Exception as e:
                    self._report_error('effect warm-up', e)

            # 5) pacing + adaptive tuning
            t_wait = time.perf_counter()
            phases['preview'] = t_wait - t_preview
            try:
                if has_active_videos or transition_active:
                    ft = self.frame_pacer.wait_for_next_frame()
                    self._adaptive_update(ft, active_content=True)
                else:
                    ft = self.static_frame_pacer.wait_for_next_frame()
                    self._adaptive_update(ft, active_content=False)
            except Exception as e:
                self._report_error('pacing', e)
                time.sleep(0.005)
            phases['wait'] = time.perf_counter() - t_wait
            self.loop_stats.record(t_wait, phases, has_active_videos or transition_active, rendered)

            # 6) health snapshot (cheap: ~0.1 ms every HEALTH_PERIOD)
            now = time.perf_counter()
            if now - last_health >= self.HEALTH_PERIOD:
                last_health = now
                try:
                    self.health_snapshot = self._build_health_snapshot(now, has_active_videos or transition_active)
                    if now - last_health_log >= self.HEALTH_LOG_PERIOD:
                        last_health_log = now
                        log.info(health.summary_line(self.health_snapshot),
                                 extra={'data': {'health': self.health_snapshot}})
                except Exception as e:
                    self._report_error('health', e)

        log.info("Render thread shutting down.")
        self.gc_stats.remove()
        try:
            self.re.clear()
        except Exception as e:
            self._report_error('shutdown', e)
        try:
            # Only the display: pygame.quit() would also shut SDL's audio down,
            # silencing the sound output (owner's report 2026-11-13: no sound
            # after Preferences > Save, which restarts the render engine).
            pygame.display.quit()
        except Exception:
            pass

    def _update_audio(self, state):
        """Tell the audio engine which videos sound and how loud: the layers on
        the output (during a fade both cues, at their fade levels), and the next
        cue's videos decoded ahead but paused."""
        ts = state.get('transition_state') or {}
        if ts.get('active'):
            fa, ta = transition_alphas(ts)
            groups = [(ts.get('from') or {}, fa), (ts.get('to') or {}, ta)]
        else:
            groups = [(state.get('layers') or {}, 1.0)]
        players = self.re.active_players
        wanted = {}
        for layers, alpha in groups:
            for layer in layers.values():
                path = layer.get('path')
                if path and live_source.is_live(path):     # a live source's sound, from its player
                    key = self.re.source_key(layer)
                    info = players.get(key)
                    get_buf = getattr(info['player'], 'audio_buffer', None) if info else None
                    if get_buf is not None and getattr(info['player'], 'has_audio', None) is not False:
                        gain = layer_volume(layer) * alpha
                        wanted[key] = {'path': path, 'gain': max(gain, wanted.get(key, {}).get('gain', 0.0)),
                                       'playing': True, 'live': get_buf()}
                    continue
                if not path or os.path.splitext(path)[1].lower() not in VIDEO_EXTS:
                    continue                      # files and live sources only
                key = self.re.source_key(layer)
                info = players.get(key)
                p = info['player'] if info else None
                try:
                    speed = float(layer.get('speed', 1.0))
                except (TypeError, ValueError):
                    speed = 1.0
                forward = layer.get('direction', 'forward') == 'forward'      # backwards: silent
                gain = layer_volume(layer) * alpha if abs(speed - 1.0) < 1e-6 and forward else 0.0
                wanted[key] = {'path': path, 'gain': max(gain, wanted.get(key, {}).get('gain', 0.0)),
                               'playing': p is not None and not p.held and p.frames_decoded > 0,
                               'loop': layer.get('loop', True), 'loops': getattr(p, 'loops', 0),
                               'trim': layer_trim(layer),
                               'clock_start': getattr(p, 'clock_start', None) if p is not None and not p.held else None,
                               'loop_at': getattr(p, 'loop_at', None)}
                if layer.get('start_at'):
                    wanted[key]['start_at'] = layer['start_at']      # Start From (rehearsal)
        for key, info in list(players.items()):
            p = info['player']
            if key not in wanted and info.get('type') != 'live' and getattr(p, 'held', False):
                wanted[key] = {'path': info['path'], 'gain': 0.0, 'playing': False,
                               'loop': getattr(p, 'loop', True), 'loops': getattr(p, 'loops', 0),
                               'trim': tuple(getattr(p, 'trim', (None, None)))}
                if getattr(p, 'start_at', None):
                    wanted[key]['start_at'] = p.start_at       # a held Start From player (preroll)
        self.audio.set_streams(wanted)

    PREROLL_TIMEOUT = 0.35  # seconds to wait for a new cue's first video frames
    HEALTH_PERIOD = 0.5     # seconds between health snapshots
    HEALTH_LOG_PERIOD = 60.0

    def _build_health_snapshot(self, now, active_content):
        live = self._video_keys(self.current_render_state)
        decoders = self.re.player_stats(live) if hasattr(self.re, 'player_stats') else []
        recent_cuts = [e for e in self.preroll_events if e[0] >= now - health.ERROR_WINDOW_S]
        waits = [e[1] for e in self.preroll_events]
        last_err = self.error_log.last
        return {
            'time': time.time(),
            'render_thread_alive': True,
            'engine': type(self.re).__name__,
            'gl': getattr(self.re, 'gl_info', None),
            'active_content': bool(active_content),
            'target_fps': self.frame_pacer.get_target_fps(),
            'loop_fps': round(self.frame_pacer.actual_fps, 1),
            'output': self.frame_stats.snapshot(now),
            'loop': self.loop_stats.snapshot(now),
            'gc': self.gc_stats.snapshot(now),
            'errors_total': dict(self.error_counts),
            'errors_recent': self.error_log.recent(now),
            'last_error': {'phase': last_err[0], 'message': last_err[1]} if last_err else None,
            'decoders': decoders,
            'preroll': {'cuts_total': len(self.preroll_events),
                        'last_wait_ms': waits[-1] if waits else None,
                        'max_wait_ms': max(waits) if waits else None,
                        'timeouts_recent': sum(1 for e in recent_cuts if e[2])},
            'effects': self._effects_status(),
            'frozen': self._frozen_videos(now, live if active_content else set(), decoders),
        }

    FROZEN_AFTER_S = 2.0    # a video on the output that has shown no new frame for this long

    def _frozen_videos(self, now, live, decoders):
        """Videos on the output whose picture has stopped while it should move
        (owner's report 2026-11-14: a video with an effect froze until the app
        was restarted; that session's log had a playing decoder whose every
        frame was dropped). Three cases, each logged once with the keys involved:
        - no decoder for a video the output shows (its last picture stays up);
        - its decoder is held on its first frame although no cut is waiting for
          it: released here, as the cut would have done;
        - its decoder decodes but the output takes none of its frames.
        Returns [{'file', 'why'}] for Output Health."""
        if self._pending_cut is not None or not live:
            self._frozen_seen = {}
            return []
        seen = getattr(self, '_frozen_seen', {})
        stats = {d.get('key'): d for d in decoders}
        out, still = [], {}
        for key in live:
            d = stats.get(key)
            path = key.split('|', 1)[-1]
            if d is None:
                if not os.path.exists(path):
                    continue                      # missing media: reported as such
                why = 'no decoder'
            elif d.get('type') == 'live' or d.get('state') in ('ended', 'no video', 'error', 'retrying', 'stopped'):
                continue
            elif d.get('state') == 'held':
                why = 'held'
            else:
                shown, decoded = d.get('frames_shown'), d.get('frames_decoded')
                prev = seen.get(key)
                if (prev is None or prev[0] != 'watch' or shown is None or decoded is None
                        or shown != prev[2] or decoded == prev[3]):
                    still[key] = ('watch', now, shown, decoded)   # taking frames, or not decoding (paused)
                    continue
                still[key] = prev                 # decoding, but the output took none since prev[1]
                if now - prev[1] < self.FROZEN_AFTER_S:
                    continue
                why = 'not taken'
            if why != 'not taken':
                prev = seen.get(key)
                still[key] = first = prev if prev is not None and prev[0] == why else (why, now)
                if now - first[1] < self.FROZEN_AFTER_S:
                    continue
            logged = getattr(self, '_frozen_logged', set())
            if (key, why) not in logged:
                logged.add((key, why))
                self._frozen_logged = logged
                players = {k: bool(getattr(i.get('player'), 'held', False)) for k, i in self.re.active_players.items()}
                log.warning(f"Frozen video on the output: {os.path.basename(path)} ({why}); output keys "
                            f"{sorted(live)}, decoders (key: held) {players}")
            if why == 'held':
                self._release_held([key])
                log.warning(f"Frozen video on the output: released {os.path.basename(path)}, "
                            f"held with no cut waiting for it")
            out.append({'file': os.path.basename(path), 'why': why})
        self._frozen_seen = still
        return out

    def _effects_status(self):
        """Shader effects on the output: how many, and why any of them is not drawn."""
        layers = (self.current_render_state.get('layers') or {}).values()
        names = [fx['name'] for l in layers for fx in isf.layer_stack(l)]
        names += [fx['name'] for st in (getattr(self.re, 'output_effects', None) or {}).values() for fx in st]
        if not names:
            return {'count': 0}
        st = {'count': len(names), 'drawn': bool(getattr(self.re, 'draws_effects', False))}
        errs = getattr(self.re, 'effect_errors', None) or {}
        failed = {n: errs[n] for n in set(names) if n in errs}
        if failed:
            st['errors'] = failed
        return st

    def _video_keys(self, state):
        """Player keys of every video a render state (layers + fade) shows."""
        layers = dict(state.get('layers', {}) or {})
        ts = state.get('transition_state') or {}
        if ts.get('active'):
            layers.update({f'f{k}': v for k, v in (ts.get('from') or {}).items()})
            layers.update({f't{k}': v for k, v in (ts.get('to') or {}).items()})
        return {self.re.source_key(l) for l in layers.values() if is_streamed(l.get('path'))}

    def _begin_preroll(self, command):
        """Start the new cue's decoders but keep showing the current cue until
        every new video has a frame (or PREROLL_TIMEOUT expires). This makes a
        cut a cut: previously the old players were stopped first and the
        output showed black until the new ones started."""
        videos = lambda layers: self._video_keys({'layers': layers})
        self.re.preroll_keep = self._video_keys(self.current_render_state)
        # The new cue's video files wait on their first frame until the cut, as a
        # preloaded cue's do: a decoder started cold (a cue that was not preloaded,
        # Start From) ran on during the wait, so its picture was ahead of its sound,
        # which starts at the cut (measured: 130 ms with Start From). Released in
        # _commit_preroll; a player that is already playing is not held again.
        held = {k: (dict(l, hold=True) if is_streamed(l.get('path')) and not live_source.is_live(l.get('path'))
                    else l) for k, l in (command.get('layers') or {}).items()}
        self.re.update_layers(held, force_restart_paths=command.get('force_restart_paths'))
        # Only wait for videos that actually have a decoder (missing files have none).
        wanted = {k for k in videos(command.get('layers')) if k in self.re.active_players}
        t0 = time.perf_counter()
        self._pending_cut = {'command': command, 'videos': wanted, 't0': t0,
                             'deadline': t0 + self.PREROLL_TIMEOUT}

    def _preroll_ready(self):
        pending = self._pending_cut
        missing = [p for p in pending['videos'] if not self.re.has_frame(p)]
        if not missing:
            return True
        if time.perf_counter() >= pending['deadline']:
            log.warning("Cue preroll timed out waiting for: " + ", ".join(os.path.basename(p) for p in missing))
            pending['timed_out'] = True
            return True
        return False

    def _commit_preroll(self):
        pending = self._pending_cut
        command = pending['command']
        self._pending_cut = None
        now = time.perf_counter()
        self.preroll_events.append((now, round((now - pending.get('t0', now)) * 1000.0, 1),
                                    bool(pending.get('timed_out'))))
        outgoing = self.re.preroll_keep
        self.re.preroll_keep = set()
        self._release_held(pending['videos'])
        state = {k: v for k, v in command.items() if k not in ('preroll', 'force_restart_paths')}
        self.current_render_state.update(state)
        # Stop only the outgoing cue's players; preloads the UI started while
        # this cut was prerolling must survive.
        self.re.stop_players(outgoing - self._video_keys(self.current_render_state))

    def _run_fx_preview(self):
        """Collect a finished effect preview, then start the next one (one at a
        time, after the output's frame, read back without waiting)."""
        done = self.re.effect_preview_poll()
        if done is not None:
            req, arr = done
            req['reply'].put((req.get('seq'), arr, None))
        job = getattr(self, '_fx_preview_job', None)
        if job is not None and not self.re.effect_preview_busy():
            self._fx_preview_job = None
            why = self.re.effect_preview_begin(job)
            if why is not None:
                job['reply'].put((job.get('seq'), None, why))

    def _release_held(self, keys):
        """Let the players of a cut run (held on their first frame during preroll)."""
        for k in keys:
            info = self.re.active_players.get(k)
            p = info.get('player') if info else None
            if p is not None and getattr(p, 'held', False) and hasattr(p, 'release'):
                p.release()

    def _apply_command(self, command):
        """Apply one command; return True if the output needs re-rendering."""
        action = command.get('action')
        if action == 'render' and command.get('preroll'):
            self._begin_preroll(command)
            return False
        if action == 'render' and self._pending_cut is not None:
            # A newer explicit state supersedes a cut that is still prerolling;
            # stop players that only the abandoned preroll needed.
            abandoned = self.re.preroll_keep | self._pending_cut['videos']
            self._release_held(self._pending_cut['videos'] & self._video_keys(command))
            self._pending_cut = None
            self.re.preroll_keep = set()
            self.re.stop_players(abandoned - self._video_keys(command))
        if action == 'render':
            if 'force_restart_paths' in command:
                self.re.update_layers(command.get('layers', {}),
                                      force_restart_paths=command.get('force_restart_paths'))
            self.current_render_state.update(command)
            return True
        if action == 'update_layers':
            self.re.update_layers(command.get('layers', {}),
                                  force_restart_paths=command.get('force_restart_paths'))
            return True
        if action == 'request_preview':
            self.current_render_state['request_preview'] = True
            return False
        if action == 'fx_preview':
            # The position editor's effect preview: drawn after this loop's frame
            # (_run_fx_preview); a newer request replaces one not started yet.
            if not hasattr(self.re, 'effect_preview_begin'):
                command['reply'].put((command.get('seq'), None, "Effects are drawn in Performance Mode only"))
                return False
            old = getattr(self, '_fx_preview_job', None)
            if old is not None and command.get('image') is None and old.get('image') is not None:
                command = dict(command, image=old['image'])      # keep the picture a skipped request carried
            self._fx_preview_job = command
            return False
        if action == 'fx_warm':
            if hasattr(self.re, 'queue_effect_warmup'):
                self.re.queue_effect_warmup(command.get('names') or [])
            return False
        if action == 'fx_preview_end':
            self._fx_preview_job = None
            if hasattr(self.re, 'effect_preview_end'):
                self.re.effect_preview_end()
            return False
        if action == 'set_warps':
            self.re.set_output_warps(command.get('warps') or {})
            return True
        if action == 'set_blends':
            self.re.set_output_blends(command.get('blends') or {})
            return True
        if action == 'set_meshes':
            self.re.set_output_meshes(command.get('meshes') or {})
            return True
        if action == 'set_output_effects':
            self.re.set_output_effects(command.get('effects') or {})
            return True
        if action == 'set_regions':
            self.re.set_output_regions(command.get('regions') or {})
            return True
        if action == 'stop':
            self.stop_event.set()
        return False

    def _gpu_preview_monitor(self):
        """Monitor index to capture on the GPU for the preview, or None for the CPU path."""
        state = self.current_render_state
        if not isinstance(self.re, OpenGLRenderEngine) or not self.re.window:
            return None
        if state.get('output_mode') == ControlInterface.MULTIPLE:
            return None
        return state.get('target_monitor')

    def _cpu_preview(self):
        """Composite the preview on the CPU (no output window, or Pygame engine)."""
        state = self.current_render_state
        layers = state.get('layers', {}) or {}
        ts = state.get('transition_state') or {}
        output_mode = state.get('output_mode')
        output_monitor = state.get('target_monitor')
        preview_target_monitor = 0 if output_mode == ControlInterface.MULTIPLE else output_monitor
        max_w = int(640 * self._preview_cap_scale)
        max_h = int(360 * self._preview_cap_scale)
        monitors = getattr(self.re, 'monitors', None) or []
        regions = getattr(self.re, 'output_regions', None) or {}
        if output_mode == ControlInterface.SPAN and regions:
            preview_target_monitor = None                 # the whole canvas of the span
            ref_w, ref_h = next(iter(regions.values()))[1]
        elif preview_target_monitor is not None and 0 <= preview_target_monitor < len(monitors):
            ref_w, ref_h = monitors[preview_target_monitor].width, monitors[preview_target_monitor].height
        else:
            preview_target_monitor = None
            ref_w, ref_h = 1920, 1080
        scale_factor = min(max_w / ref_w, max_h / ref_h, 1.0)
        dims = (max(1, int(ref_w * scale_factor)), max(1, int(ref_h * scale_factor)))
        # One display in Balanced or Compatibility Mode: a small copy of the output
        # window itself (cheaper than composing the layers again, and the same
        # picture, warp and fades included); composed only without a window.
        if isinstance(self.re, PygameRenderEngine) and preview_target_monitor is not None \
                and output_mode not in (ControlInterface.MULTIPLE, ControlInterface.SPAN):
            img = self.re.window_preview_image(max_w, max_h)
            if img is not None:
                return img

        def pick(src):
            if output_mode == ControlInterface.MULTIPLE:
                src = {k: v for k, v in src.items() if preview_target_monitor in v.get('screens', [])}
            if isinstance(self.re, PygameRenderEngine):
                out = {}
                for k, v in src.items():
                    v = v.copy()
                    pos = dict(v.get('position') or {})
                    pos['scale_x'] = pos.get('scale_x', 1.0) * scale_factor
                    pos['scale_y'] = pos.get('scale_y', 1.0) * scale_factor
                    v['position'] = pos
                    out[k] = v
                return out
            return src

        def composite(src):
            if isinstance(self.re, PygameRenderEngine):
                return self.re.render_layers_to_image(pick(src), dims)
            return self.re.render_layers_to_image(pick(src), dims, reference_monitor=preview_target_monitor)

        if ts.get('active'):
            from_img = composite(ts.get('from', {}))
            to_img = composite(ts.get('to', {}))
            fa, ta = transition_alphas(ts)
            parts = [(img, a) for img, a in ((from_img, fa), (to_img, ta)) if img is not None]
            if not parts:
                return None
            size = parts[0][0].size
            out = np.zeros((size[1], size[0], 4), dtype=np.float32)
            for img, a in parts:
                arr = np.asarray(img.convert('RGBA').resize(size), dtype=np.float32)
                out[..., :3] += arr[..., :3] * (arr[..., 3:4] / 255.0) * a
            out[..., 3] = 255.0
            return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))
        return composite(layers)

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
            log.info(f"[Adaptive] Load high -> lowering target FPS to {new_fps}, preview_scale={self._preview_cap_scale}")
            self._stress_frames = 0  # reset

        # Step up if sustained stability (only when active content)
        elif active_content and self._stable_ok_frames >= self._raise_after and self._level_idx > 0:
            self._level_idx -= 1
            new_fps = self._adaptive_levels[self._level_idx]
            self.frame_pacer.set_target_fps(new_fps)
            # relax preview caps as we step up
            if self._preview_cap_scale < 1.0:
                self._preview_cap_scale = 1.0 if self._preview_cap_scale == 0.75 else 0.75
            log.info(f"[Adaptive] Stable -> raising target FPS to {new_fps}, preview_scale={self._preview_cap_scale}")
            self._stable_ok_frames = 0  # reset

class TransitionThread(threading.Thread):
    """Dedicated thread for handling transition timing and updates."""
    def __init__(self, control_interface):
        super().__init__(daemon=True, name="TransitionThread")
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
        self.routing = {}
        # Completed target columns, consumed by the Tk thread
        # (ControlInterface._gui_animation_loop). This thread never calls Tk.
        self.completed = queue.Queue()

    def start_transition(self, from_layers, to_layers, duration, target_col, routing=None, curve='linear'):
        """Start a new transition. Call from the Tk thread; `routing` carries the
        output settings so this thread never has to read Tk variables."""
        with self.lock:
            self.curve = curve
            self.from_layers = from_layers
            self.to_layers = to_layers
            self.duration = max(float(duration), 1e-3)
            self.target_col = target_col
            self.routing = dict(routing or {})
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
                    routing = dict(self.routing)
                    curve = getattr(self, 'curve', 'linear')
                
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
                                'curve': curve,
                                'from': from_layers,
                                'to': to_layers
                            }
                            
                            self.ci.render_command_queue.put({
                                'action': 'render',
                                'layers': {},
                                'target_monitor': routing.get('target_monitor'),
                                'transition_state': transition_state,
                                'gui_monitor': routing.get('gui_monitor', 0),
                                'output_mode': routing.get('output_mode'),
                                'all_assigned_screens': set(routing.get('all_assigned_screens', ())),
                            })
                            
                            last_progress = progress
                        except Exception as e:
                            log.error(f"Error sending transition update: {e}")
                    
                    # Check if transition is complete
                    if progress >= 1.0:
                        # Transition complete: the Tk thread finalizes it
                        self.completed.put(target_col)
                        self.transition_event.clear()
                        break
                    
                    # Precise frame timing
                    frame_pacer.wait_for_next_frame()
                    
        except Exception as e:
            log.error(f"Transition thread error: {e}", exc_info=True)


LOCK_NOTICE_S = 4.0


def _locked_out(fn):
    """For actions that change the cue grid or the media list: refused, with a
    short notice, while Show > Lock Editing is on (GO, BACK, cue selection,
    Blackout and output settings stay available)."""
    import functools

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if getattr(self, 'edit_locked', False):
            self._locked_notice(fn.__name__)
            return None
        return fn(self, *args, **kwargs)
    return wrapper


def _undo_step_op(fn):
    """For changes that add, delete or move cues or layers: record one undo
    step when the change is complete, with the index maps the change set in
    self._undo_pending. Intermediate _set_dirty_flag calls (the grid rebuild
    makes some) record nothing."""
    import functools

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if getattr(self, '_undo_in_op', False):
            return fn(self, *args, **kwargs)
        self._undo_in_op = True
        try:
            return fn(self, *args, **kwargs)
        finally:
            self._undo_in_op = False
            self._undo_note()
    return wrapper

class ControlInterface:
    NONE = 'None'
    MULTIPLE = 'Multiple'
    SPAN = 'Span'                    # one picture across the displays of output_span (span.py)
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
        # 'blind': edits to the live cue wait for Update Output; 'live': apply at once
        self.live_edit_mode = 'blind'
        # OSC remote control (machine settings): off by default, this computer only.
        self.osc_enabled = False
        self.osc_port = 7700
        self.osc_network = False
        self._remote = None
        self._remote_queue = queue.Queue()
        # MIDI remote (machine settings): off by default.
        self.midi_enabled = False
        self.midi_port = ''              # input port name ('' = the first port)
        self.midi_mapping = {}           # action -> [kind, channel, number], see midi_remote
        self.midi_faders = []            # Control Changes for effect inputs (midi_remote.clean_faders)
        self.midi_program_change = True  # Program Change n selects cue n
        self.midi_timecode = False       # follow MIDI timecode (cue Timecode fields), timecode.py
        self.midi_msc = False            # MIDI Show Control (midi_remote.parse_msc)
        self.midi_msc_device = 0         # its device ID, 0-111
        self.ltc_enabled = False         # follow SMPTE LTC from an audio input instead (ltc.py)
        self.ltc_device = ''             # audio input ('' = the first one)
        self.ltc_channel = 'Left'        # which input channel carries the LTC
        self._ltc = None
        self.edit_locked = False         # Show > Lock Editing (this session only)
        self.audio_device = ''           # audio output for video sound ('' = off), audio_engine.py
        self.audio_delay_ms = 0          # delay the sound to match a projector's picture delay
        self._audio = None
        self._midi = None
        # DMX (Art-Net / sACN) from a lighting console (machine settings): off by default.
        self.dmx_enabled = False
        self.dmx_protocol = 'artnet'
        self.dmx_universe = 0
        self.dmx_channel = 1
        self.dmx_network = True          # a console is another computer
        self.dmx_out_enabled = False     # send the live cue on DMX (dmx_remote.DMXOutput)
        self.dmx_out_protocol = 'artnet'
        self.dmx_out_universe = 0
        self.dmx_out_channel = 1
        self.dmx_out_target = '255.255.255.255'
        self._dmx_out = None
        self._dmx = None
        # Browser remote (machine settings): off by default, this computer only.
        self.web_enabled = False
        self.web_port = 7701
        self.web_network = False
        self._web = None
        self._web_status = {}
        self.proxy_executor = ThreadPoolExecutor(max_workers=2)
        # Edited-cue preview (blind editing): one worker, latest request wins.
        self._edit_preview_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="EditPreview")
        self._edit_preview_source = preview_compose.StillSource(live_still=lambda p: self._live_card(p))  # worker only
        self._edit_preview_results = queue.Queue()
        self._edit_preview_seq = 0
        self._edit_preview_sig = None
        
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
        self._drop_marker = None         # accent bar showing where a dragged cue/layer will land
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
        # First frames for thumbnails, reduced to THUMB_SOURCE_MAX, keyed by
        # (path, mtime, size): grid rebuilds (show load, output change) used to
        # open every video and full-size image on the UI thread (F21).
        self._thumb_sources = collections.OrderedDict()
        self._thumb_sources_lock = threading.Lock()
        
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
        self.drag_widget = Label(self.root, text=name, bg=T.RAISED, fg='white', bd=1, relief='solid')
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
                            lbl.config(bg=(T.ACCENT if c == src else T.RAISED))
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


    @_locked_out
    @_undo_step_op
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

        def cmap(c):
            if c == src:
                return dst
            if dst > src and src < c <= dst:
                return c - 1
            if dst < src and dst <= c < src:
                return c + 1
            return c

        # Commit
        self.grid_cells = new_cells
        self.cell_positions = new_positions
        self.cue_names = new_names
        self.cue_settings = project_schema.remap_cue_keys(self.cue_settings, cmap)
        self.active_col = ac
        # The live cue keeps playing: its activation and player keys move with it.
        self._remap_program(col_map=cmap)

        # Invalidate caches only across affected span
        if self.re:
            lo, hi = sorted((src, dst))
            for c in range(lo, hi + 1):
                try:
                    self.re.invalidate_static_cache_for_column(c)
                except Exception:
                    pass

        # Rebuild visuals once and update state (no re-cut: live videos keep playing)
        self._build_grid()
        self._highlight_live_column()
        self._update_cue_dropdown()
        self._trigger_column()
        self._undo_pending = ('Move Cue', None, {c: cmap(c) for c in range(self.COLS)})
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
        self.drag_widget = Label(self.root, text=name, bg=T.RAISED, fg='white', bd=1, relief='solid')
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

    @_locked_out
    @_undo_step_op
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

        def rmap(r):
            if r == src:
                return dst
            if dst > src and src < r <= dst:
                return r - 1
            if dst < src and dst <= r < src:
                return r + 1
            return r

        # Commit
        self.grid_cells = new_cells
        self.cell_positions = new_positions
        self.layer_names = new_layer_names
        self.layer_screens = new_layer_screens
        self._remap_program(row_map=rmap)

        # Invalidate all columns because layer reordering changes per-column z-order and cached transforms
        if self.re:
            for c in range(self.COLS):
                try:
                    self.re.invalidate_static_cache_for_column(c)
                except Exception:
                    pass  # keep compatibility if renderer lacks column caches

        # Rebuild visuals; the output changes only as live/blind editing allows
        # (no re-cut, so live videos do not restart).
        self._build_grid()
        self._highlight_live_column()
        self._trigger_column()
        self._undo_pending = ('Move Layer', {r: rmap(r) for r in range(self.ROWS)}, None)
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
                log.info("Set process priority to HIGH_PRIORITY_CLASS")
                
                # Enable multi-threaded performance optimizations
                # This tells Windows we're a multimedia application
                win32process.SetProcessPriorityBoost(handle, False)  # Disable priority boost throttling
                
            except Exception as e:
                log.warning(f"Could not set Windows process priority: {e}")
        
        # Set timer resolution for more accurate sleep (improves frame timing)
        try:
            import ctypes
            winmm = ctypes.WinDLL('winmm')
            winmm.timeBeginPeriod(1)  # 1ms timer resolution
            log.info("Set Windows timer resolution to 1ms")
        except Exception as e:
            log.warning(f"Could not set timer resolution: {e}")

    def _reset_project_state(self, new_project=True):
        """Resets all project-specific variables to their defaults."""
        self.ROWS = self.DEFAULT_ROWS
        self.COLS = self.DEFAULT_COLS
        self.project_media_paths = []
        self.media_info = {}  # abs path -> {'rel': path relative to the show file, 'size': bytes}
        self.grid_cells = {(r, c): None for r in range(self.ROWS) for c in range(self.COLS)}
        self.cell_positions = {}
        self.active_col = 0
        self.selected_display = None
        self.header_height = 0
        self.cue_names = {str(i): f'Cue {i+1}' for i in range(self.COLS)}
        self.cue_settings = {}           # str(col) -> {'transition','duration','notes','follow'} (non-defaults only)
        if getattr(self, '_follow_timer', None) is not None:
            try:
                self.root.after_cancel(self._follow_timer)
            except Exception:  # noqa: BLE001
                pass
        self._follow_timer = None
        self._follow_token = 0
        self._follow_due = None          # (monotonic due time, from col, to col) of a pending follow
        self._unknown_project_keys = {}  # top-level keys from a newer VisualDeck, written back on save
        self.go_at_end = 'wrap'          # GO after the last cue with media: 'wrap' to the first or 'stop'
        self.schedule = {'enabled': False, 'events': []}    # scheduler.clean(), saved with the show
        self._schedule_fired = set()     # (date, event index) already run today
        self._end_of_show_until = 0.0    # monotonic time until which "end of show" is shown
        if getattr(self, '_go_wraps_var', None) is not None:
            self._go_wraps_var.set(True)
        self.layer_names = {str(i): f'Layer {i+1}' for i in range(self.ROWS)}
        self.gui_monitor_index = 0
        self.layer_screens = {}
        self.output_warps = {}           # str(display index) -> 4 corners (output_warp), saved with the show
        self.output_blends = {}          # str(display index) -> edge blend dict (edge_blend), saved with the show
        self.output_meshes = {}          # str(display index) -> mesh warp dict (mesh_warp), saved with the show
        self.output_effects = {}         # str(display index) -> shader effect (isf.clean), saved with the show
        self.output_span = None          # {'displays': [...], 'overlap': px} for Output: Span (span.py), saved
        self._regions_sent = {}
        
        if new_project:
            self.current_project_path = None
            self.project_is_dirty = False
            isf.set_show_folder(None)          # no show folder: no show effects

        self.transition_active = False
        self.transition_start_time = 0
        self.transition_duration = 1.0
        self.transition_from_layers = {}
        self.transition_to_layers = {}
        self.transition_target_col = 0

        # Video players are keyed per cue *activation* (see _get_layers_for_col):
        # each time a cue goes live it gets a new id, so the outgoing and the
        # incoming cue of a fade never share a decoder even when they use the
        # same file. The next cue is preloaded under a reserved id with paused
        # players; GO claims that id so those players start from frame 0.
        self._activation_seq = itertools.count(1)
        self._live_activation = {}       # col -> activation id currently on screen
        self._reserved_activation = {}   # col -> activation id of the preloaded next cue
        self._transition_activation = None  # (col, id) of the cue a fade is bringing in
        self._instance_alias = {}        # (activation, row, col, path) -> instance key of a moved cell
        # The program: what the output shows for the live cue, as sent. Show
        # data edits do not change it until they are applied (blind editing).
        # 'rows' maps a program layer's row to the row that layer occupies now
        # (layer reorders), so per-layer output routing still finds it.
        self._program = None             # {'col': int, 'layers': {...}, 'rows': {int: int}}
        self._test_pattern = None        # {'kind': str, 'saved': program or None} while a pattern is shown

    def _start_gui(self):
        multiprocessing.freeze_support()

        self.root = tk.Tk()
        # An error in a Tk callback goes to the log (Tk printed it only to the console, where
        # nobody sees it: owner's PC, 2026-11-14, a grid refresh error found in the launcher window).
        self.root.report_callback_exception = self._tk_callback_error
        T.apply(self.root)   # fonts, palette, flat controls (ui_theme.py)
        self.root.configure(bg=T.BG)
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
                log.warning(f"Could not load icon at '{icon_path}'.")
        else:
            log.warning(f"Icon file not found at '{icon_path}'")

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
                    log.exception("Failed to render version text on splash image.")

                self._splash_img = ImageTk.PhotoImage(img)
                lbl = Label(self._splash_win, image=self._splash_img, borderwidth=0, highlightthickness=0, bg='#000')
                lbl.pack()

                sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
                w, h = self._splash_img.width(), self._splash_img.height()
                x, y = max(0, (sw - w)//2), max(0, (sh - h)//2)
                self._splash_win.geometry(f"{w}x{h}+{x}+{y}")
            else:
                log.warning(f"Splash image not found at '{splash_path}'")
        except Exception:
            log.exception("Failed to show splash screen.")
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
        self.media_frame = Frame(self.pan, bg=T.BG)
        self.grid_frame = Frame(self.pan, bg=T.BG)
        self.preview_frame = Frame(self.pan, bg=T.BG)
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
                    log.exception("Render engine init failed during startup.")
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
                log.exception("Startup output pre-toggle failed.")

            self._ui_ready = True
            self._startup_inhibit_triggers = False
            self._restart_remote()
            self._restart_web()
            self._restart_midi()
            self._restart_ltc()
            self._restart_dmx()
            self._restart_dmx_out()
            self._restart_audio()
            self.root.after(self.AUTOSAVE_PERIOD_MS, self._autosave_tick)
            self.root.after(self.DISPLAY_POLL_MS, self._display_poll_tick)
            self.root.after(1000, self._schedule_tick)
            self.root.after(self.GC_FREEZE_DELAY_MS, lambda: self._freeze_gc("startup"))
            try:
                self._trigger_column()
                self._startup_render_sent = True
            except Exception:
                log.exception("Deferred render trigger failed.")
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
        acc = shortcuts.label
        fm.add_command(label='New Project', command=self._new_project, accelerator=acc('new'))
        fm.add_command(label='Open Project…', command=self._open_project, accelerator=acc('open'))
        
        self.recent_menu = Menu(fm, tearoff=0)
        fm.add_cascade(label='Recent Projects', menu=self.recent_menu)
        self._populate_recent_menu()

        fm.add_separator()
        fm.add_command(label='Save', command=self._save_project, accelerator=acc('save'))
        fm.add_command(label='Save As…', command=self._save_project_as, accelerator=acc('save_as'))
        fm.add_command(label='Relink Missing Media…', command=self._open_relink_dialog, accelerator=acc('relink'))
        fm.add_command(label='Collect Show…', command=self._collect_show)
        fm.add_separator()
        fm.add_command(label='Preferences…', command=self._open_preferences, accelerator=acc('preferences'))
        fm.add_separator()
        fm.add_command(label='Exit', command=self._on_exit)
        menubar.add_cascade(label='File', menu=fm)

        # Edit Menu (labels and states follow the undo stack: _refresh_undo_menu)
        em = Menu(menubar, tearoff=0)
        em.add_command(label='Undo', command=self._undo, accelerator=acc('undo'), state='disabled')
        em.add_command(label='Redo', command=self._redo, accelerator=acc('redo'), state='disabled')
        menubar.add_cascade(label='Edit', menu=em)
        self.edit_menu = em

        # Insert Menu
        im = Menu(menubar, tearoff=0)
        im.add_command(label='Add Media to Project…', command=self._menu_add, accelerator=acc('add_media'))
        im.add_command(label='Add Live Source…', command=self._open_live_source_dialog)
        im.add_separator()
        im.add_command(label='Cue', command=self._add_cue, accelerator=acc('add_cue'))
        im.add_command(label='Layer', command=self._add_layer, accelerator=acc('add_layer'))
        menubar.add_cascade(label='Insert', menu=im)

        # Show menu: every show action with its key, so shortcuts can be discovered
        sm = Menu(menubar, tearoff=0)
        sm.add_command(label='GO', command=self._trigger_go, accelerator=acc('go'))
        sm.add_command(label='BACK', command=self._go_back, accelerator=acc('back'))
        sm.add_command(label='Blackout', command=self._blackout, accelerator=acc('blackout'))
        sm.add_command(label='Update Output', command=self._apply_live_edits, accelerator=acc('update'))
        sm.add_separator()
        sm.add_command(label='Cue Settings (live cue)…', command=lambda: self._open_cue_settings(self.active_col),
                       accelerator=acc('cue_settings'))
        sm.add_command(label='Go to Cue 1 … 9', state='disabled', accelerator='Ctrl+1 … 9')
        sm.add_command(label='Start Cue From…', command=self._open_start_from)
        sm.add_command(label='Presentation Mode On/Off', command=self._toggle_presentation_shortcut,
                       accelerator=acc('presentation'))
        sm.add_separator()
        self._go_wraps_var = tk.BooleanVar(value=self.go_at_end == 'wrap')
        sm.add_checkbutton(label='GO After the Last Cue Returns to the First', variable=self._go_wraps_var,
                           command=self._on_go_wraps_toggle)
        sm.add_command(label='Schedule…', command=self._open_schedule)
        sm.add_command(label='Preshow Check…', command=self._open_preshow)
        self._lock_var = tk.BooleanVar(value=getattr(self, 'edit_locked', False))
        sm.add_checkbutton(label='Lock Editing', variable=self._lock_var, accelerator=acc('lock'),
                           command=lambda: self._set_edit_lock(self._lock_var.get()))
        menubar.add_cascade(label='Show', menu=sm)

        wm = Menu(menubar, tearoff=0)
        wm.add_command(label='Output Health…', command=self._open_health_window, accelerator=acc('health'))
        self.test_pattern_var = tk.StringVar(value='off')
        tpm = Menu(wm, tearoff=0)
        tpm.add_radiobutton(label='Off', value='off', variable=self.test_pattern_var,
                            command=lambda: self._show_test_pattern('off'))
        for kind in test_patterns.KINDS:
            tpm.add_radiobutton(label=test_patterns.LABELS[kind], value=kind, variable=self.test_pattern_var,
                                command=lambda k=kind: self._show_test_pattern(k))
        wm.add_cascade(label='Test Pattern', menu=tpm)
        wm.add_command(label='Corner Pin, Mesh and Edge Blend…', command=self._open_corner_pin)
        wm.add_command(label='Output Effects…', command=self._open_output_effects)
        wm.add_command(label='Span Displays…', command=self._open_span)
        wm.add_command(label='Output Presets…', command=self._open_output_presets)
        menubar.add_cascade(label='Window', menu=wm)

        hm = Menu(menubar, tearoff=0)
        hm.add_command(label='VisualDeck Help', command=self._open_help, accelerator=acc('help'))
        hm.add_command(label='Keyboard Shortcuts', command=lambda: self._open_help('Keyboard shortcuts'))
        hm.add_separator()
        hm.add_command(label='Open Log Folder', command=self._open_log_folder)
        hm.add_command(label='Soak Test…', command=self._open_soak)
        hm.add_command(label='About VisualDeck', command=lambda: self._open_help('About VisualDeck'))
        menubar.add_cascade(label='Help', menu=hm)
        
        self.root.config(menu=menubar)
        self._bind_shortcuts()

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

            # Only clicks in the main window: a dialog keeps its own focus (the corner pin
            # canvas takes the arrow keys after a click on a corner; moving the focus to
            # the main window made the nudges do nothing).
            try:
                if str(w.winfo_toplevel()) != str(self.root):
                    return
            except Exception:
                return

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

    # -- keyboard shortcuts (table in shortcuts.py) -------------------------------
    _TEXT_INPUTS = (tk.Entry, tk.Text, tk.Spinbox, ttk.Entry, ttk.Combobox, ttk.Spinbox)

    def _shortcut_actions(self):
        acts = {
            'back': self._go_back, 'blackout': self._blackout, 'update': self._apply_live_edits,
            'cue_settings': lambda: self._open_cue_settings(self.active_col),
            'presentation': self._toggle_presentation_shortcut,
            'test_pattern': lambda: self._show_test_pattern('off' if self._test_pattern else 'grid'),
            'health': self._open_health_window, 'new': self._new_project, 'open': self._open_project,
            'save': self._save_project, 'save_as': self._save_project_as, 'relink': self._open_relink_dialog,
            'preferences': self._open_preferences, 'add_media': self._menu_add, 'add_cue': self._add_cue,
            'add_layer': self._add_layer, 'help': self._open_help, 'undo': self._undo, 'redo': self._redo,
            'lock': self._toggle_edit_lock,
        }
        for n in range(1, 10):
            acts[f'goto_{n}'] = (lambda n=n: self._go_to_cue(n - 1))
        return acts

    def _bind_shortcuts(self):
        acts = self._shortcut_actions()
        for sc in shortcuts.SHORTCUTS:
            if sc.action == 'go':
                continue            # Space keeps its own handler (_on_space_go)
            for seq in sc.sequences:
                self.root.bind_all(seq, lambda e, a=sc.action: self._on_shortcut(a, e))
        self._shortcut_table = acts

    def _on_shortcut(self, action, event=None):
        sc = shortcuts.BY_ACTION[action]
        widget = getattr(event, 'widget', None)
        if not sc.in_text_fields and isinstance(widget, self._TEXT_INPUTS):
            return None                      # let the text field have its own editing keys
        fn = getattr(self, '_shortcut_table', None) or self._shortcut_actions()
        try:
            fn[action]()
        except Exception:
            log.exception(f"Shortcut {sc.label} ({action}) failed")
        return "break"

    def _toggle_presentation_shortcut(self):
        self.presentation_mode_var.set(not self.presentation_mode_var.get())
        self._on_toggle_presentation_mode()

    def _go_to_cue(self, col):
        """Go to a cue by index with its transition (re-triggers the live cue)."""
        if not self.re or not (0 <= col < self.COLS):
            return
        if self.transition_active:
            log.info(f"Go to cue {col + 1}: ignored during a transition")
            return
        if self.presentation_active:
            self._stop_presentation()
        if col == self.active_col and not self._output_is_black():
            self._cut_to_column(col)
        else:
            self._initiate_transition(self.active_col, col)

    # -- Help window -------------------------------------------------------------
    def _help_link(self, parent, topic, bg=T.BG):
        """A small "?" that opens Help at `topic` (contextual help next to a control)."""
        import help_content
        assert topic in help_content.topic_titles(), topic
        opts = dict(bg=bg) if bg else {}
        link = Label(parent, text="?", fg=T.ACCENT, cursor='hand2', font=T.font(9, 'bold'), **opts)
        link.bind('<Button-1>', lambda e: self._open_help(topic))
        CreateToolTip(link, f"Help: {topic}")
        return link

    def _open_help(self, topic=None):
        """Help > VisualDeck Help (F1): topics on the left, text on the right, search."""
        import help_content
        win = getattr(self, '_help_win', None)
        if win is not None and win.winfo_exists():
            win.lift()
            if topic:
                self._help_show(topic)
            return
        win = tk.Toplevel(self.root)
        self._help_win = win
        win.title("VisualDeck Help")
        win.configure(bg=T.BG)
        win.geometry("940x640")
        win.minsize(640, 400)
        side = Frame(win, bg=T.SURFACE, width=250)
        side.pack(side='left', fill='y')
        side.pack_propagate(False)
        Label(side, text="Help", bg=T.SURFACE, fg=T.TEXT, font=T.font(14, 'bold'), anchor='w').pack(
            fill='x', padx=14, pady=(14, 6))
        query = tk.StringVar()
        search = ttk.Entry(side, textvariable=query)
        search.pack(fill='x', padx=12, pady=(0, 8))
        topics = tk.Listbox(side, bg=T.SURFACE, fg=T.TEXT, relief='flat', borderwidth=0, highlightthickness=0,
                            activestyle='none', selectbackground=T.ACCENT, selectforeground='#ffffff',
                            font=T.font(10), exportselection=False)
        topics.pack(fill='both', expand=True, padx=6, pady=(0, 10))
        body = Frame(win, bg=T.BG)
        body.pack(side='left', fill='both', expand=True)
        text = tk.Text(body, wrap='word', bg=T.BG, fg=T.TEXT, relief='flat', borderwidth=0, highlightthickness=0,
                       padx=28, pady=20, cursor='arrow', font=T.font(10), spacing1=2, spacing3=2)
        sb = ttk.Scrollbar(body, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        text.pack(side='left', fill='both', expand=True)
        text.tag_configure('h1', font=T.font(17, 'bold'), spacing3=12)
        text.tag_configure('p', spacing3=10)
        text.tag_configure('li', lmargin1=8, lmargin2=26, spacing3=6)
        text.tag_configure('bullet', foreground=T.ACCENT)
        text.tag_configure('group', font=T.font(11, 'bold'), foreground=T.MUTED, spacing1=10, spacing3=4)
        text.tag_configure('key', font=T.mono(10), foreground=T.ACCENT, lmargin1=8)
        text.tag_configure('row', tabs=('190',), spacing3=3)
        text.tag_configure('term', font=T.mono(10, 'bold'), foreground=T.ACCENT, lmargin1=8, spacing1=6)
        text.tag_configure('def', lmargin1=26, lmargin2=26, spacing3=6)
        text.tag_configure('hit', background=T.WARN, foreground='#000000')
        self._help_widgets = (topics, text, query)

        def filtered():
            q = query.get().strip().lower()
            out = []
            for title, blocks in help_content.TOPICS:
                blob = title + ' ' + help_content.topic_text(blocks)
                if not q or q in blob.lower():
                    out.append(title)
            return out

        def refill(*_):
            names = filtered()
            topics.delete(0, 'end')
            for n in names:
                topics.insert('end', '  ' + n)
            if names:
                cur = getattr(self, '_help_current', None)
                self._help_show(cur if cur in names else names[0])
        query.trace_add('write', refill)
        topics.bind('<<ListboxSelect>>', lambda e: topics.curselection() and self._help_show(
            topics.get(topics.curselection()[0]).strip()))
        win.bind('<Escape>', lambda e: win.destroy())
        refill()
        self._help_show(topic or help_content.TOPICS[0][0])
        search.focus_set()

    def _help_show(self, title):
        import help_content
        topics, text, query = self._help_widgets
        blocks = dict(help_content.TOPICS).get(title)
        if blocks is None:
            return
        self._help_current = title
        names = [topics.get(i).strip() for i in range(topics.size())]
        if title in names:
            topics.selection_clear(0, 'end')
            topics.selection_set(names.index(title))
            topics.see(names.index(title))
        text.configure(state='normal')
        text.delete('1.0', 'end')
        text.insert('end', title + '\n', 'h1')
        for kind, val in blocks:
            if kind == 'p':
                text.insert('end', val + '\n', 'p')
            elif kind == 'list':
                for item in val:
                    text.insert('end', '\u2022  ', ('li', 'bullet'))
                    text.insert('end', item + '\n', 'li')
                text.insert('end', '\n')
            elif kind == 'keys':
                text.insert('end', val + '\n', 'group')
                for k, d in help_content._rows_for(val):
                    text.insert('end', k, ('row', 'key'))
                    text.insert('end', '\t' + d + '\n', 'row')
            elif kind == 'defs':
                for term, desc in help_content.defs_rows(val):
                    text.insert('end', term + '\n', 'term')
                    text.insert('end', desc + '\n', 'def')
        q = query.get().strip()
        if q:
            start = '1.0'
            while True:
                pos = text.search(q, start, stopindex='end', nocase=True)
                if not pos:
                    break
                end = f"{pos}+{len(q)}c"
                text.tag_add('hit', pos, end)
                start = end
        text.configure(state='disabled')

    def _open_log_folder(self):
        try:
            import logging_setup
            d = logging_setup.log_dir()
            if sys.platform == 'win32':
                os.startfile(d)  # noqa: S606
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', d])
            else:
                subprocess.Popen(['xdg-open', d])
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Open Log Folder", f"Could not open the log folder: {e}", parent=self.root)

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
        self._soak_stop("app closed")                # a running soak test keeps its summary
        if self.presentation_timer:
            self.root.after_cancel(self.presentation_timer)
        self.proxy_executor.shutdown(wait=False)
        self._edit_preview_executor.shutdown(wait=False, cancel_futures=True)
        self._stop_remote()
        self._stop_web()
        self._stop_midi()
        self._stop_ltc()
        self._stop_dmx()
        if self._dmx_out is not None:
            self._dmx_out.stop()
            self._dmx_out = None
        self._stop_audio()
        
        # ADD THIS: Stop transition thread
        if self.transition_thread and self.transition_thread.is_alive():
            log.info("Stopping transition thread...")
            self.transition_thread.shutdown()
            self.transition_thread.join(timeout=1.0)
        
        if self.render_thread and self.render_thread.is_alive():
            log.info("Requesting render thread to stop...")
            self.render_command_queue.put({'action': 'stop'})
            self.render_thread.join(timeout=2.0)
        self.root.destroy()

    def _tk_callback_error(self, exc, val, tb):
        log.error(f"Error in the window code (VisualDeck keeps running): {exc.__name__}: {val}",
                  exc_info=(exc, val, tb))

    def _show_effect_names(self):
        """Every shader effect the show uses: on layers and on displays."""
        names = []
        for pos in list((getattr(self, 'cell_positions', None) or {}).values()):
            names += [fx['name'] for fx in isf.layer_stack(pos)]
        for raw in (getattr(self, 'output_effects', None) or {}).values():
            names += [fx['name'] for fx in isf.stack(raw)]
        return list(dict.fromkeys(names))

    def _warm_effects(self):
        """Ask the render thread to compile the show's effects before they are needed."""
        names = self._show_effect_names()
        if names and getattr(self, 'render_command_queue', None) is not None:
            self.render_command_queue.put({'action': 'fx_warm', 'names': names})

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
                preview_bridge=self.preview_bridge
            )
        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")

        self.re.force_multithread = self.force_multithread
        self.re.set_output_warps(self.output_warps)    # before the render thread uses the engine
        self.re.set_output_blends(self.output_blends)
        self.re.set_output_meshes(self.output_meshes)
        self.re.set_output_effects(self.output_effects)
        self._regions_sent = self._span_regions()
        self.re.set_output_regions(self._regions_sent)
        self.render_thread = RenderThread(
            self.re, self.render_command_queue,
            self.render_preview_queue, self.render_stop_event
        )
        self.render_thread.audio = getattr(self, '_audio', None)
        self.render_thread.start()
        self._warm_effects()

        if self.transition_thread is None:
            self.transition_thread = TransitionThread(self)
            self.transition_thread.start()

        self._refresh_output_options()
        if not self.re.monitors:
            self.screen_var.set(self.NONE)
        
        self._update_gui_monitor_index()
        self.set_preview_dirty()
        
    def _open_decode_test(self, parent, hw_accel_var=None, path=None):
        """Decode a chosen video in software and with each hardware decoder
        (decode_test.py) and offer the fastest that really worked."""
        import decode_test
        if path is None:
            path = filedialog.askopenfilename(parent=parent, title="Test decoding with which video?",
                                              filetypes=[('Videos', ' '.join(f'*{e}' for e in VIDEO_EXTS)),
                                                         ('All files', '*.*')])
            if not path:
                return None
        win = tk.Toplevel(parent)
        win.title("Decoding Test")
        win.configure(bg=T.SURFACE)
        win.transient(parent)
        body = tk.Frame(win, bg=T.SURFACE, padx=14, pady=14)
        body.pack(fill='both', expand=True)
        fps = decode_test.clip_fps(path)
        tk.Label(body, bg=T.SURFACE, fg=T.TEXT, anchor='w', justify='left', wraplength=460,
                 text=f"{decode_test.label(path)}" + (f"  ({fps:g} fps)" if fps else "")).pack(fill='x')
        out = tk.Label(body, bg=T.SURFACE, fg=T.TEXT, anchor='w', justify='left', font=T.mono(10),
                       text="Decoding in software…")
        out.pack(fill='x', pady=(8, 0))
        rec_lbl = tk.Label(body, bg=T.SURFACE, fg=T.MUTED, anchor='w', justify='left', wraplength=460)
        rec_lbl.pack(fill='x', pady=(8, 0))
        btns = tk.Frame(body, bg=T.SURFACE)
        btns.pack(fill='x', pady=(12, 0))
        use_btn = ttk.Button(btns, text="Use it", state='disabled')
        use_btn.pack(side='left')
        tk.Button(btns, text="Close", command=win.destroy, width=9).pack(side='right')
        results, q = [], queue.Queue()
        mode = {'opengl': 'yuv', 'cpu': 'numpy'}.get(self.render_mode, 'pygame')

        def work():
            try:
                decode_test.run_all(path, output_format=mode, progress=q.put)
            except Exception as e:  # noqa: BLE001
                log.exception("Decoding test failed")
                q.put({'device': '?', 'ok': False, 'error': str(e), 'fps': None, 'engaged': None})
            q.put(None)

        def poll():
            done = False
            try:
                while True:
                    r = q.get_nowait()
                    if r is None:
                        done = True
                        break
                    results.append(r)
            except queue.Empty:
                pass
            if not win.winfo_exists():
                return
            lines, rec = decode_test.summary(results, fps)
            out.config(text="\n".join(lines + ([] if done else ["testing…"])))
            if not done:
                win.after(150, poll)
                return
            log.info(f"Decoding test {decode_test.label(path)}: " + " | ".join(lines))
            if rec is None:
                rec_lbl.config(text="Nothing could decode this video.")
                return
            value = 'auto' if rec == 'software' else rec
            rec_lbl.config(text=f"Fastest here: {rec}." + (" Choose it, then Save and Apply." if hw_accel_var else ""))
            if hw_accel_var is not None and value in ('auto', 'cuda', 'dxva2', 'd3d11va', 'videotoolbox'):
                use_btn.config(state='normal', text=f"Use {rec}", command=lambda: (hw_accel_var.set(value),
                                                                                   win.destroy()))
            win._vd['recommended'] = value

        threading.Thread(target=work, daemon=True, name="DecodeTest").start()
        win.after(150, poll)
        win._vd = {'results': results, 'out': out, 'rec': rec_lbl, 'use': use_btn, 'poll': poll}   # for tests
        return win

    def _run_optimization_test(self, render_mode_var):
        """Sets up and starts the non-blocking performance test using a separate process."""
        test_win = tk.Toplevel(self.root)
        test_win.title("Performance Test")
        test_win.configure(bg=T.SURFACE)
        test_win.geometry("350x200")
        test_win.transient(self.root)
        test_win.grab_set()
        test_win.protocol("WM_DELETE_WINDOW", lambda: None)

        main_frame = Frame(test_win, bg=T.SURFACE, padx=15, pady=15)
        main_frame.pack(fill='both', expand=True)

        status_label = tk.Label(main_frame, text="Preparing test...", fg='white', bg=T.SURFACE, justify='left')
        status_label.pack(anchor='w', pady=5)

        progress = ttk.Progressbar(main_frame, orient='horizontal', length=300, mode='determinate')
        progress.pack(fill='x', pady=5)

        benchmark_script_path = os.path.join(os.path.dirname(__file__), 'benchmark.py')

        if not getattr(sys, 'frozen', False) and not os.path.exists(benchmark_script_path):
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

        import benchmark
        command = benchmark.command(mode_to_test)      # the Windows build runs itself with --benchmark
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
            log.error(f"Failed to launch benchmark process for {mode_to_test}: {e}")
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
            log.warning(f"Benchmark for {mode_tested} produced an error: {stderr.strip()}")

        try:
            last_line = stdout.strip().split('\n')[-1]
            result_time = float(last_line)
            state['results'][mode_tested] = result_time
        except (ValueError, TypeError, IndexError):
            log.error(f"Could not parse benchmark result for {mode_tested}. Output: '{stdout}'")
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
        
        question = (f"The recommended setting for your system is:\n\n{mode_names[best_mode]}\n\n"
                    "Would you like to switch to this setting now?")
        
        if messagebox.askyesno("Optimization Complete", question, parent=state['window']):
            self.render_mode = best_mode
            state['render_mode_var'].set(best_mode)
            
            self._initialize_render_engine()
            self._update_active_players()
            self._trigger_column()
            
            messagebox.showinfo("Setting Applied", "The new setting has been applied. Click 'Save and Apply' to make it permanent.", parent=state['window'])
        
        state['window'].destroy()
        del self._test_state

    # Preferences pages (the list on the left) and the line under each page's title.
    PREFS_PAGES = ("Editing the live cue", "Remote control", "Audio", "Rendering", "Video decoding")
    PREFS_NOTES = {
        "Editing the live cue": "What happens when you change the cue that is on the output.",
        "Remote control": "Control VisualDeck from other programs, consoles, phones and timecode. "
                          "All are off by default.",
        "Audio": "Where the sound of videos plays, and how much it waits for the picture.",
        "Rendering": "How the output is drawn. Changing it restarts the output (the live cue starts "
                     "again) when you press Save and Apply.",
        "Video decoding": "How videos are decoded. Changing it restarts the output (the live cue starts "
                          "again) when you press Save and Apply.",
    }

    def _open_preferences(self):
        pref_win = tk.Toplevel(self.root)
        pref_win.title("Preferences")
        pref_win.configure(bg=T.SURFACE)
        pref_win.transient(self.root)
        pref_win.grab_set()

        # --- Variables for settings ---
        render_mode_var = tk.StringVar(value=self.render_mode)
        hw_accel_var = tk.StringVar(value=self.hw_accel)
        # Tkinter variables for the options
        use_pbo_var = tk.BooleanVar(value=self.use_pbo)
        use_vsync_var = tk.BooleanVar(value=self.use_vsync)
        force_multithread_var = tk.BooleanVar(value=self.force_multithread)
        live_edit_var = tk.StringVar(value=self.live_edit_mode)

        # Pages listed on the left, the chosen page's settings on the right (owner's
        # request, 2026-11-15). Every page is built now, so Save and Apply saves all.
        main_frame = tk.Frame(pref_win, bg=T.SURFACE)
        main_frame.pack(fill='both', expand=True)
        body = tk.Frame(main_frame, bg=T.SURFACE)
        body.pack(fill='both', expand=True)
        st = ttk.Style(pref_win)
        st.configure('Nav.Treeview', background=T.SURFACE, fieldbackground=T.SURFACE, foreground=T.TEXT,
                     rowheight=34, borderwidth=0, lightcolor=T.SURFACE, darkcolor=T.SURFACE,
                     bordercolor=T.SURFACE, font=T.font(T.BASE_SIZE + 1))
        st.map('Nav.Treeview', background=[('selected', T.ACCENT)], foreground=[('selected', '#ffffff')])
        st.layout('Nav.Treeview', [('Treeview.treearea', {'sticky': 'nswe'})])
        nav = ttk.Treeview(body, show='tree', selectmode='browse', height=len(self.PREFS_PAGES),
                           style='Nav.Treeview', takefocus=True)
        nav.column('#0', width=190, stretch=False)
        for name in self.PREFS_PAGES:
            nav.insert('', 'end', iid=name, text='  ' + name)
        nav.grid(row=0, column=0, sticky='ns', padx=(8, 0), pady=8)
        content = tk.Frame(body, bg=T.BG)
        content.grid(row=0, column=1, sticky='nsew', padx=(8, 0))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        page_area = tk.Frame(content, bg=T.BG)
        page_area.pack(fill='both', expand=True, padx=18, pady=(14, 16))
        page_area.columnconfigure(0, weight=1)
        pages = {}
        for name in self.PREFS_PAGES:
            pg = tk.Frame(page_area, bg=T.BG)
            pg.grid(row=0, column=0, sticky='nsew')
            ttk.Label(pg, text=name, font=T.font(T.BASE_SIZE + 4, 'bold')).pack(anchor='w')
            ttk.Label(pg, text=self.PREFS_NOTES[name], foreground=T.MUTED, wraplength=440,
                      justify='left').pack(anchor='w', pady=(2, 12))
            pages[name] = pg

        # --- Editing the live cue ---
        show_lf = ttk.Frame(pages["Editing the live cue"])
        show_lf.pack(fill='x')
        blind_rb = ttk.Radiobutton(show_lf, text="Wait for Update Output (Ctrl+U) — recommended for shows",
                                   variable=live_edit_var, value='blind')
        blind_rb.pack(anchor='w', padx=5, pady=2)
        CreateToolTip(blind_rb, "Changes to the cue that is on the output are saved in the show but do not "
                                "reach the output until you press Update Output or trigger the cue again.")
        live_rb = ttk.Radiobutton(show_lf, text="Apply immediately (live editing)", variable=live_edit_var, value='live')
        live_rb.pack(anchor='w', padx=5, pady=2)

        # --- Remote control ---
        osc_enabled_var = tk.BooleanVar(value=self.osc_enabled)
        osc_port_var = tk.StringVar(value=str(self.osc_port))
        osc_network_var = tk.BooleanVar(value=self.osc_network)
        osc_lf = ttk.Frame(pages["Remote control"])
        osc_lf.pack(fill='x')
        osc_cb = ttk.Checkbutton(osc_lf, text="Listen for OSC commands on UDP port", variable=osc_enabled_var)
        osc_cb.grid(row=0, column=0, sticky='w', padx=5, pady=2)
        CreateToolTip(osc_cb, "Show control (QLab, Companion, TouchOSC, a console) sends /visualdeck/go, "
                              "/visualdeck/cue 3 and more. Help \u203a OSC commands lists them all.")
        ttk.Entry(osc_lf, textvariable=osc_port_var, width=7).grid(row=0, column=1, sticky='w')
        self._help_link(osc_lf, "OSC commands", bg=None).grid(row=0, column=2, sticky='w', padx=(6, 0))
        osc_net_cb = ttk.Checkbutton(osc_lf, text="Accept commands from other computers on the network",
                                     variable=osc_network_var)
        osc_net_cb.grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        CreateToolTip(osc_net_cb, "Off: only programs on this computer can send commands. On: any device that "
                                  "can reach this computer can trigger cues.")
        web_enabled_var = tk.BooleanVar(value=self.web_enabled)
        web_port_var = tk.StringVar(value=str(self.web_port))
        web_network_var = tk.BooleanVar(value=self.web_network)
        web_cb = ttk.Checkbutton(osc_lf, text="Serve a browser remote page on port", variable=web_enabled_var)
        web_cb.grid(row=2, column=0, sticky='w', padx=5, pady=(8, 2))
        CreateToolTip(web_cb, "Open http://<this computer>:<port>/ on a phone or tablet: GO, BACK, Blackout, "
                              "Update Output and the cue list.")
        ttk.Entry(osc_lf, textvariable=web_port_var, width=7).grid(row=2, column=1, sticky='w', pady=(8, 2))
        self._help_link(osc_lf, "Browser remote and HTTP API", bg=None).grid(row=2, column=2, sticky='w',
                                                                             padx=(6, 0), pady=(8, 2))
        ttk.Checkbutton(osc_lf, text="Allow phones and other computers on the network to open it",
                        variable=web_network_var).grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        midi_row = ttk.Frame(osc_lf)
        midi_row.grid(row=4, column=0, columnspan=2, sticky='w', padx=5, pady=(8, 2))
        midi_state = ttk.Label(midi_row, foreground=T.MUTED,
                               text=self._midi_text(self._midi_status()).replace(" (Preferences)", ""))
        ttk.Button(midi_row, text="MIDI Setup…",
                   command=lambda: self._open_midi_setup(pref_win, midi_state)).pack(side='left')
        midi_state.pack(side='left', padx=(8, 0))
        ltc_row = ttk.Frame(osc_lf)
        ltc_row.grid(row=5, column=0, columnspan=2, sticky='w', padx=5, pady=(4, 2))
        ltc_state = ttk.Label(ltc_row, foreground=T.MUTED,
                              text=self._ltc_text(self._ltc_status()).replace(" (Preferences)", ""))
        ttk.Button(ltc_row, text="LTC Setup…",
                   command=lambda: self._open_ltc_setup(pref_win, ltc_state)).pack(side='left')
        ltc_state.pack(side='left', padx=(8, 0))
        dmx_row = ttk.Frame(osc_lf)
        dmx_row.grid(row=6, column=0, columnspan=2, sticky='w', padx=5, pady=(4, 2))
        dmx_state = ttk.Label(dmx_row, foreground=T.MUTED,
                              text=self._dmx_text(self._dmx_status()).replace(" (Preferences)", ""))
        ttk.Button(dmx_row, text="DMX Setup…",
                   command=lambda: self._open_dmx_setup(pref_win, dmx_state)).pack(side='left')
        dmx_state.pack(side='left', padx=(8, 0))

        # --- Audio (sound of videos) ---
        audio_lf = ttk.Frame(pages["Audio"])
        audio_lf.pack(fill='x')
        names, audio_err = audio_engine.device_names()
        audio_choices = ["Off", audio_engine.DEFAULT_DEVICE] + names
        cur = self.audio_device or "Off"
        if cur not in audio_choices:
            audio_choices.append(cur)                 # a saved device that is not connected now
        audio_var = tk.StringVar(value=cur)
        arow = ttk.Frame(audio_lf)
        arow.pack(fill='x')
        ttk.Label(arow, text="Play the sound of videos on").pack(side='left')
        ttk.Combobox(arow, textvariable=audio_var, values=audio_choices, state='readonly', width=26).pack(
            side='left', padx=(6, 0))
        drow = ttk.Frame(audio_lf)
        drow.pack(fill='x', pady=(4, 0))
        ttk.Label(drow, text="Delay the sound by").pack(side='left')
        audio_delay_var = tk.StringVar(value=str(self.audio_delay_ms))
        delay_entry = ttk.Entry(drow, textvariable=audio_delay_var, width=6)
        delay_entry.pack(side='left', padx=(6, 4))
        ttk.Label(drow, text="ms (0\u20131000)").pack(side='left')
        test_lbl = ttk.Label(audio_lf, foreground=T.MUTED)

        def test_sound():
            choice = audio_var.get()
            ok, text = self._play_test_sound('' if choice == "Off" else choice)
            test_lbl.config(text=text, foreground=T.GO if ok else '#ff8888')
        ttk.Button(drow, text="Test Sound", command=test_sound).pack(side='right')
        test_lbl.pack(anchor='w', pady=(2, 0))
        CreateToolTip(delay_entry, "Projectors and TVs show the picture a little late (often 30\u2013100 ms). Delay "
                                   "the sound by the same amount so they match: film a clap or a flash test clip.")
        ttk.Label(audio_lf, foreground=T.MUTED, wraplength=360, justify='left',
                  text=(audio_err or "Each video's Volume is in the position editor. Videos at a speed other "
                                     "than 1\u00d7 play silent.")).pack(anchor='w', pady=(4, 0))

        # --- Rendering Backend Section ---
        render_page = pages["Rendering"]
        mode_lf = ttk.LabelFrame(render_page, text="Rendering Backend", padding=(10, 5))
        mode_lf.pack(fill='x', pady=(0, 10))

        opengl_rb = ttk.Radiobutton(mode_lf, text="Performance Mode (Requires Good GPU)", variable=render_mode_var, value='opengl')
        opengl_rb.pack(anchor='w', padx=5, pady=2)
        pygame_gpu_rb = ttk.Radiobutton(mode_lf, text="Balanced Mode (Recommended)", variable=render_mode_var, value='pygame_gpu')
        pygame_gpu_rb.pack(anchor='w', padx=5, pady=2)
        cpu_rb = ttk.Radiobutton(mode_lf, text="Compatibility Mode (Safe Mode)", variable=render_mode_var, value='cpu')
        cpu_rb.pack(anchor='w', padx=5, pady=2)

        # --- Advanced OpenGL Options Section ---
        advanced_lf = ttk.LabelFrame(render_page, text="Advanced Performance Options (Performance Mode)",
                                     padding=(10, 5))

        pbo_check = ttk.Checkbutton(advanced_lf, text="Use Asynchronous PBO Uploads", variable=use_pbo_var)
        pbo_check.pack(anchor='w', padx=5, pady=2)
        CreateToolTip(pbo_check, "Enables faster texture uploads on most dedicated GPUs, but can cause issues on some integrated graphics.")

        vsync_check = ttk.Checkbutton(advanced_lf, text="Enable VSync", variable=use_vsync_var)
        vsync_check.pack(anchor='w', padx=5, pady=2)
        CreateToolTip(vsync_check, "Synchronizes framerate to your monitor's refresh rate to prevent screen tearing. May introduce input latency.")
        
        multithread_check = ttk.Checkbutton(advanced_lf, text="Force Multi-threaded Decoding", variable=force_multithread_var)
        multithread_check.pack(anchor='w', padx=5, pady=2)
        CreateToolTip(multithread_check, "Tells the video decoder to use multiple CPU threads. Can improve performance for high-resolution video, but may be unstable.")

        # --- GPU Hardware Acceleration Section ---
        gpu_lf = ttk.LabelFrame(pages["Video decoding"], text="GPU Hardware Acceleration", padding=(10, 5))
        gpu_lf.pack(fill='x', pady=(0, 10))
        
        accel_options = ['auto', 'cuda', 'dxva2', 'd3d11va', 'videotoolbox']
        for accel in accel_options:
            label = 'Software decoding (default)' if accel == 'auto' else accel.upper()
            rb = ttk.Radiobutton(gpu_lf, text=label, variable=hw_accel_var, value=accel)
            rb.pack(anchor='w', padx=5, pady=2)
            if accel == 'auto':
                CreateToolTip(rb, "Software decoding (default). Choose a specific API to request hardware decoding; "
                                  "VisualDeck falls back to software if the device is unavailable and logs which was used.")
        cpu_decode_note = ttk.Label(pages["Video decoding"], foreground=T.MUTED, wraplength=440, justify='left',
                                    text="Compatibility Mode always decodes in software. Hardware decoding needs "
                                         "Performance or Balanced Mode (the Rendering page).")
        ttk.Button(gpu_lf, text="Test with a Video…",
                   command=lambda: self._open_decode_test(pref_win, hw_accel_var)).pack(anchor='w', padx=5, pady=(6, 2))

        # --- Logic to show/hide options ---
        def on_mode_change(*args):
            selected_mode = render_mode_var.get()
            
            # Show/hide GPU acceleration options
            gpu_state = 'normal' if selected_mode != 'cpu' else 'disabled'
            for child in gpu_lf.winfo_children():
                child.configure(state=gpu_state)
            if selected_mode == 'cpu':
                cpu_decode_note.pack(anchor='w', after=gpu_lf)
            else:
                cpu_decode_note.pack_forget()
                
            # Show/hide Advanced OpenGL options
            if selected_mode == 'opengl':
                advanced_lf.pack(fill='x', pady=(0, 10), after=mode_lf)
            else:
                advanced_lf.pack_forget()

        optimize_btn = tk.Button(render_page, text="\u26a1 Find Best Setting For Me", bg=T.ACCENT, fg='white',
                                 command=lambda: self._run_optimization_test(render_mode_var))
        optimize_btn.pack(fill='x', pady=(4, 0))

        # The page area keeps the size of the largest page (measured with the
        # Performance Mode options shown), so switching pages does not resize it.
        advanced_lf.pack(fill='x', pady=(0, 10), after=mode_lf)
        cpu_decode_note.pack(anchor='w', after=gpu_lf)
        pref_win.update_idletasks()
        page_area.configure(width=max(pg.winfo_reqwidth() for pg in pages.values()),
                            height=max(pg.winfo_reqheight() for pg in pages.values()))
        page_area.grid_propagate(False)
        render_mode_var.trace_add('write', on_mode_change)
        on_mode_change() # Call once to set initial state

        def show_page(name):
            for n, pg in pages.items():
                if n == name:
                    pg.grid()
                else:
                    pg.grid_remove()
            self._prefs_page = name
        nav.bind('<<TreeviewSelect>>', lambda e: nav.selection() and show_page(nav.selection()[0]))
        first = getattr(self, '_prefs_page', None)
        first = first if first in pages else self.PREFS_PAGES[0]
        nav.selection_set(first)
        nav.focus(first)
        show_page(first)
        nav.focus_set()

        # --- Buttons Section ---
        ttk.Separator(main_frame, orient='horizontal').pack(fill='x')
        btn_frame = tk.Frame(main_frame, bg=T.SURFACE, padx=15, pady=12)
        btn_frame.pack(fill='x')

        def save_prefs():
            # Save all settings. Only the render settings restart the render engine (a
            # restart redraws the output and re-triggers the live cue), and only the two
            # the show file stores (render mode, hardware decoding) mark the show changed:
            # turning on a remote during a show must not touch the output or the show.
            render_before = (self.render_mode, self.hw_accel, self.use_pbo, self.use_vsync,
                             self.force_multithread)
            self.render_mode = render_mode_var.get()
            self.hw_accel = hw_accel_var.get()
            self.use_pbo = use_pbo_var.get()
            self.use_vsync = use_vsync_var.get()
            self.force_multithread = force_multithread_var.get()
            self.live_edit_mode = live_edit_var.get()
            try:
                port = int(osc_port_var.get())
                if not 1 <= port <= 65535:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Remote control", "The OSC port must be a number from 1 to 65535.",
                                     parent=pref_win)
                return
            osc_changed = (self.osc_enabled, self.osc_port, self.osc_network) != \
                (osc_enabled_var.get(), port, osc_network_var.get())
            try:
                wport = int(web_port_var.get())
                if not 1 <= wport <= 65535 or (osc_enabled_var.get() and web_enabled_var.get() and wport == port):
                    raise ValueError
            except ValueError:
                messagebox.showerror("Remote control", "The browser remote port must be a number from 1 to 65535 "
                                     "and not the OSC port.", parent=pref_win)
                return
            self.osc_enabled, self.osc_port, self.osc_network = osc_enabled_var.get(), port, osc_network_var.get()
            if osc_changed:
                self._restart_remote()
            try:
                delay = int(audio_delay_var.get().strip() or 0)
                if not 0 <= delay <= 1000:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Audio", "The sound delay must be a whole number of milliseconds, 0 to 1000.",
                                     parent=pref_win)
                return
            new_audio = '' if audio_var.get() == "Off" else audio_var.get()
            if (new_audio, delay) != (self.audio_device, self.audio_delay_ms):
                self.audio_device, self.audio_delay_ms = new_audio, delay
                self._restart_audio()
            web_changed = (self.web_enabled, self.web_port, self.web_network) != \
                (web_enabled_var.get(), wport, web_network_var.get())
            self.web_enabled, self.web_port, self.web_network = web_enabled_var.get(), wport, web_network_var.get()
            if web_changed:
                self._restart_web()

            log.info(f"Preferences saved. Live cue edits: {self.live_edit_mode}.")
            log.info(f"Preferences saved. Mode: {self.render_mode}, PBO: {self.use_pbo}, VSync: {self.use_vsync}")
            render_after = (self.render_mode, self.hw_accel, self.use_pbo, self.use_vsync, self.force_multithread)
            if render_after[:2] != render_before[:2]:
                self._set_dirty_flag()               # the show stores these two
            self._save_config()
            if render_after != render_before:
                self._initialize_render_engine()
                self._update_active_players()
                self._trigger_column()
            pref_win.destroy()

        save_btn = tk.Button(btn_frame, text="Save and Apply", command=save_prefs, bg=T.GO, fg='black')
        save_btn.pack(side='right', padx=5)
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=pref_win.destroy, bg=T.STOP, fg='white')
        cancel_btn.pack(side='right')
        pref_win.bind('<Escape>', lambda e: pref_win.destroy())
        pref_win.resizable(False, False)

    def _open_optimize_window(self, path):
        optimizer = FFMpegOptimizer(self.root, path)
        optimizer.show()
        self.root.wait_window(optimizer.window)
        self._build_media_list()

    PREVIEW_FADE_INTERVAL = 1 / 15      # seconds between preview requests during a fade

    def _preview_due(self, transitioning, now, last):
        """Whether to request a preview now: when something changed, at most
        about 30 per second while video plays, and about 15 per second during a
        fade so the preview follows the fade (it used to freeze for the whole
        fade, and a still cue reached after a fade was never shown)."""
        if transitioning:
            self.preview_dirty = True
        gap = self.PREVIEW_FADE_INTERVAL if transitioning else (0.033 if self.is_video_playing else 0.0)
        return bool(getattr(self, 'preview_dirty', False)) and now - last > gap

    def _gui_animation_loop(self):
        import os, time, gc
        from queue import Empty

        # Finalize fades completed by the transition thread (on this, the Tk, thread)
        tt = getattr(self, 'transition_thread', None)
        if tt is not None:
            while True:
                try:
                    done_col = tt.completed.get_nowait()
                except Empty:
                    break
                try:
                    self._finalize_transition_from_thread(done_col)
                except Exception:
                    log.exception("Finalizing transition failed")

        layers = self._get_layers_for_col(self.active_col)
        self.is_video_playing = any(
            is_streamed(l['path']) and (live_source.is_live(l['path']) or os.path.exists(l['path']))
            for l in layers.values()
        )
        if self.is_video_playing:
            self.set_preview_dirty()

        # Output health indicator (replaces the plain FPS label), twice a second
        now_t = time.perf_counter()
        if now_t - getattr(self, '_last_clock_ui', 0.0) >= 0.25:
            self._last_clock_ui = now_t
            try:
                self._refresh_cue_clock()
            except Exception:
                log.exception("Cue clock update failed")
        if now_t - getattr(self, '_last_health_ui', 0.0) >= 0.5:
            self._last_health_ui = now_t
            try:
                self._update_health_indicator()
            except Exception:
                log.exception("Health indicator update failed")
            self._refresh_live_edit_ui()
            try:
                self._refresh_follow_label()
            except Exception:
                log.exception("Follow countdown update failed")
            if self._web is not None:
                try:
                    self._web_status = self._build_web_status()
                except Exception:
                    log.exception("Web remote status update failed")

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

        if self._preview_due(transitioning, current_time, last_preview_time):
            if True:
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
                log.exception(f"Error generating thumbnail for {path}")
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
        try:
            self._poll_remote()
        except Exception:
            log.exception("Remote command handling failed")
        try:
            self._poll_edit_preview()
        except Exception:
            log.exception("Edited-cue preview update failed")
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
        canvas = Canvas(self.media_frame, bg=T.BG, highlightthickness=0)
        vsb = ttk.Scrollbar(self.media_frame, orient='vertical', command=canvas.yview)
        container = Frame(canvas, bg=T.BG)
        container.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0,0), window=container, anchor='nw')
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')

        self.media_list_labels.clear()
        paths_to_generate = []
        placeholder = ImageTk.PhotoImage(Image.new('RGB', (80, 45), T.RAISED))

        # The media list is now built from the project's specific list of paths
        for path in self.project_media_paths:
            live = live_source.is_live(path)
            is_missing = not live and not os.path.exists(path)
            ext = os.path.splitext(path)[1].lower()
            if not live and not is_missing and ext in VIDEO_EXTS:
                proxy_path = FFMpegProxyGenerator.get_proxy_path(path)
                if not os.path.exists(proxy_path):
                    self.proxy_executor.submit(FFMpegProxyGenerator(path).generate)

            if not live and ext not in SUPPORTED_EXTS: continue

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
            
            display_name = live_source.label(path) if live else os.path.basename(path)
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
                log.warning("Thumbnail generator is already running.")

    def _build_grid(self):
        # Keep the scroll position: Undo, Redo and other edits rebuild the grid, which
        # jumped back to cue 1 (owner's PC, acceptance run 2026-11-14).
        old = getattr(self, 'grid_canvas', None)
        keep = None
        try:
            if old is not None and old.winfo_exists():
                keep = (old.xview()[0], old.yview()[0])
        except tk.TclError:
            keep = None
        for w in self.grid_frame.winfo_children(): w.destroy()
        # The old cells are gone: a show with fewer cues or layers than the last one left
        # destroyed cells here, and a refresh that reached one failed with "bad window path
        # name" (owner's PC, 2026-11-14).
        self.frames_by_position = {}
        self._cell_frames = {}
        self.grid_frame.grid_propagate(False)
        self.grid_frame.rowconfigure(0, weight=1)
        self.grid_frame.columnconfigure(0, weight=1)
        canvas = Canvas(self.grid_frame, bg=T.BG, highlightthickness=0)
        vsb = ttk.Scrollbar(self.grid_frame, orient='vertical', command=canvas.yview)
        hsb = ttk.Scrollbar(self.grid_frame, orient='horizontal', command=canvas.xview)
        canvas.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        inner = Frame(canvas, bg=T.BG)
        canvas.create_window((0,0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        Label(inner, bg=T.BG).grid(row=0, column=0)

        self.col_labels = {}
        for c in range(self.COLS):
            lbl = Label(inner, text=self._cue_header_text(c), bg=T.RAISED, fg='white', height=1)
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
            label_container = Frame(inner, bg=T.RAISED)
            label_container.grid(row=r+1, column=0, sticky='ns', pady=self.CELL_PAD//2)

            canvas_height = self.CELL_SIZE - 30
            label_canvas = Canvas(label_container, bg=T.RAISED, width=50, height=canvas_height, highlightthickness=0)
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
                label_container, text='⚙️', font=T.font(12), relief='flat', bg=T.RAISED, fg='white',
                command=lambda row=r: self._show_layer_output_menu(row)
            )
            settings_btn.pack(side='bottom', anchor='center', pady=0)
            self.layer_setting_widgets[r] = settings_btn

            for c in range(self.COLS):
                frm = Frame(
                    inner, bg=T.RAISED, width=self.CELL_SIZE, height=self.CELL_SIZE,
                    highlightthickness=1, highlightbackground='black'
                )
                frm.grid(row=r+1, column=c+1, padx=self.CELL_PAD//2, pady=self.CELL_PAD//2)
                frm.propagate(False)
                lbl = Label(frm, bg=T.RAISED, fg='#ff8888')  # Set default color for missing text
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
        if keep is not None and keep != (0.0, 0.0):
            def restore(canvas=canvas, keep=keep):
                try:
                    canvas.update_idletasks()
                    canvas.configure(scrollregion=canvas.bbox('all'))
                    canvas.xview_moveto(keep[0])
                    canvas.yview_moveto(keep[1])
                except tk.TclError:
                    pass                       # the grid was rebuilt again meanwhile
            self.root.after_idle(restore)
        # Settings icons shown or hidden for the output. Rebuilding the grid (opening a show, New
        # Project) changes nothing in the show: the show was marked as changed by it (a show just
        # opened from Recent Projects asked to be saved).
        self._on_output_change(mark_dirty=False)

    def _build_preview_panel(self):
        for w in self.preview_frame.winfo_children(): w.destroy()
        
        # Header: output/cue selectors, then the cue clock on its own line (the
        # panel can be narrow). Status lines (notes, follow, test pattern) are
        # packed after the whole header.
        header = Frame(self.preview_frame, bg=T.BG)
        header.pack(fill='x', pady=5)
        top = Frame(header, bg=T.BG)
        top.pack(fill='x')
        self._preview_top_bar = header
        # Notes of the live cue (Cue Settings); shown only when it has notes.
        self.cue_notes_label = Label(self.preview_frame, bg=T.BG, fg='#9fd3ff', anchor='w', justify='left',
                                     font=T.font(9, 'italic'), wraplength=360)
        self.test_pattern_label = Label(self.preview_frame, bg=T.BG, fg='#ff6b6b', anchor='w',
                                        font=T.font(9, 'bold'),
                                        text="TEST PATTERN on the output \u2013 Window \u203a Test Pattern \u203a Off restores it")
        # Pending follow (auto-continue) countdown.
        self.follow_label = Label(self.preview_frame, bg=T.BG, fg=T.WARN, anchor='w',
                                  font=T.font(9, 'bold'))
        Label(top, text='Output:', fg='white', bg=T.BG).pack(side='left', padx=5)

        self.screen_var = tk.StringVar(value=self.NONE)
        self.output_screen_combobox = ttk.Combobox(top, values=[self.NONE], textvariable=self.screen_var, state='readonly', width=12)
        self.output_screen_combobox.pack(side='left', padx=5)
        self.screen_var.trace_add('write', self._on_output_change)

        Label(top, text='Cue:', fg='white', bg=T.BG).pack(side='left', padx=5)
        self.cue_name_var = tk.StringVar()
        self.cue_select_combobox = ttk.Combobox(top, textvariable=self.cue_name_var, state='readonly', width=10)
        self.cue_select_combobox.pack(side='left', padx=5)
        self.cue_select_combobox.bind('<<ComboboxSelected>>', self._on_cue_select)
        self._update_cue_dropdown()
        self.cue_clock_label = Label(header, text='', fg=T.MUTED, bg=T.BG, font=T.mono(9), anchor='w')
        self.cue_clock_label.pack(fill='x', padx=5, pady=(3, 0))

        # Small requested height: the preview grows into spare space and gives it
        # back first when the window is short (GO must stay visible).
        self.preview_canvas = Canvas(self.preview_frame, bg='black', highlightthickness=0, height=120)
        self.preview_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        # Edited-cue preview: shown only while the live cue has unapplied edits.
        self.edit_preview_frame = Frame(self.preview_frame, bg=T.BG)
        Label(self.edit_preview_frame, bg=T.BG, fg=T.WARN, anchor='w', font=T.font(9, 'bold'),
              text="EDITED CUE \u2013 not on the output yet").pack(fill='x')
        self.edit_preview_caption = Label(self.edit_preview_frame, bg=T.BG, fg='#bbbbbb', anchor='w',
                                          font=T.font(9), justify='left', text="")
        self.edit_preview_caption.pack(fill='x')
        self.edit_preview_frame.bind('<Configure>', lambda e: self.edit_preview_caption.config(
            wraplength=max(100, e.width - 4)))
        self.edit_preview_canvas = Canvas(self.edit_preview_frame, bg='black', height=130,
                                          highlightthickness=1, highlightbackground=T.WARN)
        self.edit_preview_canvas.pack(fill='x')
        self.edit_preview_canvas.bind('<Configure>', lambda e: self._redraw_edit_preview())
        self._edit_preview_image = None
        # Output health: status, render rate, late frames, errors. Click for details.
        self.fps_label = Label(self.preview_frame, text="Output: starting…", fg='#888', bg=T.BG,
                               font=T.font(9), cursor='hand2', anchor='w', justify='left')
        self.fps_label.pack(anchor='w', fill='x', padx=5)
        self.fps_label.bind('<Button-1>', lambda e: self._open_health_window())
        self.preview_canvas.bind('<Configure>', lambda e: self.set_preview_dirty())

        controls_container = Frame(self.preview_frame, bg=T.BG)
        controls_container.pack(fill='x', padx=5, pady=(5,0))
        controls_container.columnconfigure(1, weight=1)
        Label(controls_container, text="Transition:", fg='white', bg=T.BG).grid(row=0, column=0, sticky='w', pady=2)
        self.transition_var = tk.StringVar(value='Fade')
        self.transition_dropdown = ttk.Combobox(controls_container, textvariable=self.transition_var, values=[self.NONE, 'Fade', 'Smooth Fade', 'Dip to Black'], state='readonly', width=15)
        self.transition_dropdown.grid(row=0, column=1, sticky='ew', padx=5)
        self.transition_dropdown.bind('<<ComboboxSelected>>', self._on_transition_select)
        Label(controls_container, text="Timing:", fg='white', bg=T.BG).grid(row=1, column=0, sticky='w', pady=2)
        timing_widget_frame = Frame(controls_container, bg=T.BG)
        timing_widget_frame.grid(row=1, column=1, sticky='ew', padx=5)
        vcmd = (self.root.register(self._validate_numeric), '%P')
        self.transition_timing_var = tk.StringVar(value='1.0')
        self.timing_entry = ttk.Entry(timing_widget_frame, textvariable=self.transition_timing_var, width=6, validate='key', validatecommand=vcmd)
        self.timing_entry.pack(side='left')
        self.timing_entry.bind('<KeyRelease>', self._set_dirty_flag)
        Label(timing_widget_frame, text="seconds", fg='white', bg=T.BG).pack(side='left', padx=4)
        self._help_link(timing_widget_frame, "Running the show").pack(side='left', padx=(6, 0))
        CreateToolTip(self.transition_dropdown, "The show's transition for GO: None cuts, Fade crossfades, Smooth "
                                                "Fade eases in and out, Dip to Black goes through black. A cue can "
                                                "have its own (Cue Settings).")
        self._on_transition_select()
        
        ttk.Separator(self.preview_frame, orient='horizontal').pack(fill='x', padx=5, pady=10)

        # --- Presentation Mode UI ---
        pres_frame = Frame(self.preview_frame, bg=T.BG)
        pres_frame.pack(fill='x', padx=5, pady=5)
        pres_head = Frame(pres_frame, bg=T.BG)
        pres_head.pack(fill='x')
        Label(pres_head, text="Presentation Mode", fg='white', bg=T.BG, font=T.font(10, 'bold')).pack(side='left')
        self._help_link(pres_head, "Presentation Mode").pack(side='left', padx=(6, 0))
        Label(pres_frame, text="Runs the cues by itself: GO starts it, then it goes to the next cue every "
                               "Seconds. GO (STOP) ends it.", fg=T.MUTED, bg=T.BG, font=T.font(8),
              justify='left', anchor='w', wraplength=300).pack(fill='x')

        self.presentation_mode_var = tk.BooleanVar(value=False)
        self.presentation_mode_check = ttk.Checkbutton(pres_frame, text="Enable", variable=self.presentation_mode_var, command=self._on_toggle_presentation_mode)
        self.presentation_mode_check.pack(anchor='w')
        CreateToolTip(self.presentation_mode_check, "On: the GO button starts an automatic run through the cues "
                                                    "(it reads STOP while it runs). Off: GO goes to the next cue "
                                                    "only. Ctrl+P. Saved with the show.")
        self.presentation_mode_check.bind('<Button-1>', self._set_dirty_flag)

        timing_pres_frame = Frame(pres_frame, bg=T.BG)
        timing_pres_frame.pack(fill='x')
        Label(timing_pres_frame, text="Seconds:", fg='white', bg=T.BG).pack(side='left', padx=(20,5))
        self.presentation_timing_var = tk.StringVar(value="5.0")
        self.presentation_timing_entry = ttk.Entry(timing_pres_frame, textvariable=self.presentation_timing_var, width=6, validate='key', validatecommand=vcmd, state='disabled')
        self.presentation_timing_entry.pack(side='left')
        self.presentation_timing_entry.bind('<KeyRelease>', self._set_dirty_flag)
        CreateToolTip(self.presentation_timing_entry, "How long each cue stays, counted from the start of its "
                                                      "transition (so it includes the fade).")

        self.presentation_loop_var = tk.BooleanVar(value=False)
        self.presentation_loop_check = ttk.Checkbutton(pres_frame, text="Loop", variable=self.presentation_loop_var, state='disabled')
        self.presentation_loop_check.pack(anchor='w', padx=(0,5))
        self.presentation_loop_check.bind('<Button-1>', self._set_dirty_flag)
        CreateToolTip(self.presentation_loop_check, "After the last cue, start again from the first one. "
                                                    "Off: stop on the last cue.")

        # BACK / Blackout row under GO; both rows are packed first (at the bottom)
        # so they are never squeezed out by the previews or settings above.
        nav_row = Frame(self.preview_frame, bg=T.BG)
        nav_row.pack(side='bottom', pady=(0, 5), before=self.preview_frame.pack_slaves()[0])
        back = Button(nav_row, text='\u25C0 BACK', width=9, bg=T.RAISED, fg='#ddd', command=self._go_back)
        back.pack(side='left')
        CreateToolTip(back, "BACK (Shift+Space): the previous cue that has media.")
        self.blackout_button = Button(nav_row, text='Blackout', width=12, bg=T.RAISED, fg='#ddd', command=self._blackout)
        self.blackout_button.pack(side='left', padx=(8, 0))
        CreateToolTip(self.blackout_button, "Fade the output to black with the show's default transition. "
                                            "GO then continues with the next cue.")
        go_row = Frame(self.preview_frame, bg=T.BG)
        go_row.pack(side='bottom', pady=5, before=nav_row)
        self.go_button = Button(go_row, text='GO', bg=T.GO, fg='white', activeforeground='white', width=9, height=2,
                                font=T.font(14, 'bold'), command=self._trigger_go)
        self.go_button.pack(side='left')
        CreateToolTip(self.go_button, "GO (Space): the next cue, with its transition. With Presentation Mode "
                                      "enabled: start or stop the automatic run.")
        # Blind editing: edits to the live cue reach the output only from here (Ctrl+U).
        self.update_output_button = Button(go_row, text='Update Output', state='disabled', height=2,
                                           wraplength=150, bg=T.RAISED, fg='#888', command=self._apply_live_edits)
        self.update_output_button.pack(side='left', padx=(8, 0))
        CreateToolTip(self.update_output_button, "Blind editing: edits to the cue on the output wait here. "
                                                 "Update Output (Ctrl+U) puts them on the output; unchanged "
                                                 "videos keep playing.")
        self._refresh_live_edit_ui()

    def set_preview_dirty(self):
        self.preview_dirty = True

    def _update_preview_canvas_with_image(self, pil_image):
        self._draw_on_canvas(self.preview_canvas, pil_image)

    @staticmethod
    def _draw_on_canvas(canvas, pil_image, placeholder="Preview"):
        """Fit `pil_image` into `canvas` (letterboxed), or draw a placeholder."""
        canvas.delete('all')
        canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            canvas.image = None
            return

        w_res, h_res = (1920, 1080)
        if pil_image:
            w_res, h_res = pil_image.size

        ratio = w_res / h_res if h_res > 0 else 1
        pw, ph = (canvas_w, int(canvas_w/ratio)) if canvas_w/ratio <= canvas_h else (int(canvas_h*ratio), canvas_h)
        off_x, off_y = (canvas_w - pw)//2, (canvas_h - ph)//2

        canvas.create_rectangle(off_x, off_y, off_x+pw, off_y+ph, outline=T.BORDER, dash=(5, 5), width=2)

        if not pil_image:
            canvas.create_text(off_x + pw/2, off_y + ph/2, text=placeholder, fill="white", anchor='center')
            canvas.image = None
            return

        resized_pil = pil_image.resize((pw, ph), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(resized_pil)

        canvas.create_image(off_x, off_y, image=imgtk, anchor='nw')
        canvas.image = imgtk

    def _update_active_players(self):
        """Tell the renderer which video players must exist.

        - the live cue's videos;
        - during a fade, the outgoing and incoming cue's videos (separate
          activations, so a file used by both plays twice, independently);
        - otherwise the next cue's (GO) and the previous cue's (BACK) videos,
          preloaded and paused on frame 0, and their live sources, connected
          (and shared with any cue that uses them).
        Everything else is stopped by the renderer.
        """
        if not self.re:
            return
        keep = {}
        current_layers = self._program_layers()  # what is on the output, not unapplied edits
        keep.update(self._device_layers())
        keep.update({f"live:{k}": v for k, v in current_layers.items()})
        if getattr(self, "transition_active", False):
            keep.update({f"from:{k}": v for k, v in (self.transition_from_layers or {}).items()})
            keep.update({f"to:{k}": v for k, v in (self.transition_to_layers or {}).items()})
        else:
            reserved = {}
            # The next cue (GO) and the previous one (BACK), paused on frame 0;
            # then the cues the cue list says will come (see _ready_cols).
            wanted = [("pre", self._next_cue_col()), ("back", self._prev_cue_col())]
            wanted += [("ready", c) for c in self._ready_cols()]
            for tag, col in wanted:
                if col is None or col == self.active_col or col in reserved:
                    continue
                act = self._reserved_activation.get(col)
                if act is None:
                    act = next(self._activation_seq)
                reserved[col] = act
                for k, layer in self._get_layers_for_col(col, activation=act, hold=True).items():
                    if 'instance' in layer or live_source.is_live(layer.get('path')):
                        keep[f"{tag}:{col}:{k}"] = layer  # live sources: connected before GO
            self._reserved_activation = reserved
        self.render_command_queue.put({'action': 'update_layers', 'layers': keep})
        self.is_video_playing = any(is_streamed(l.get('path')) for l in current_layers.values())

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
        self.drag_widget = Frame(self.root, width=self.CELL_SIZE, height=self.CELL_SIZE, bg=T.BG, highlightbackground='aqua', highlightthickness=2)
        self.root.config(cursor='exchange')
        self.root.bind('<B1-Motion>', self._on_drag_motion)
        self.root.bind('<ButtonRelease-1>', self._on_cell_drop)

    def _on_drag_motion(self, event):
        self.drag_motion_detected = True # MODIFICATION: Set flag to true only when mouse moves.
        if not self.drag_widget: return
        x = self.root.winfo_pointerx() - self.root.winfo_rootx() - self.drag_widget.winfo_width() // 2
        y = self.root.winfo_pointery() - self.root.winfo_rooty() - self.drag_widget.winfo_height() // 2
        self.drag_widget.place(x=x, y=y)
        if self.drag_type in ('cue_reorder', 'layer_reorder'):
            self._update_drop_marker(event.x_root, event.y_root)
            return
        cell = self._get_cell_under(event.x_root, event.y_root)
        label = cell.winfo_children()[0] if cell and cell.winfo_children() else None
        if label is not self._hover_label:
            if self._hover_label:
                try: self._hover_label.config(bg=T.RAISED)
                except tk.TclError: pass
            if label: label.config(bg=T.BORDER)
            self._hover_label = label

    @_locked_out
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

    @_locked_out
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

    # -- where a dragged cue / layer will land ----------------------------------------
    def _drop_target(self, x_root, y_root):
        """(kind, src, dst) for a header drag over a header, else None."""
        if self.drag_type == 'cue_reorder':
            return 'cue', self.drag_data.get('col'), self._get_col_under(x_root, y_root)
        if self.drag_type == 'layer_reorder':
            return 'layer', self.drag_data.get('row'), self._get_row_under(x_root, y_root)
        return None

    def _drop_marker_geometry(self, kind, src, dst):
        """(x, y, width, height) of the marker in root coordinates, or None. The
        moved cue/layer takes dst's place, so the marker is on the side of dst
        facing away from src: after dst when moving right/down, before it when
        moving left/up."""
        if dst is None or src is None or dst == src:
            return None
        lbl = (self.col_labels if kind == 'cue' else self.row_labels).get(dst)
        canvas = getattr(self, 'grid_canvas', None)
        if lbl is None or canvas is None:
            return None
        try:
            rx, ry = self.root.winfo_rootx(), self.root.winfo_rooty()
            lx, ly, lw, lh = lbl.winfo_rootx() - rx, lbl.winfo_rooty() - ry, lbl.winfo_width(), lbl.winfo_height()
            gx, gy = canvas.winfo_rootx() - rx, canvas.winfo_rooty() - ry
            gw, gh = canvas.winfo_width(), canvas.winfo_height()
        except tk.TclError:
            return None
        bar = 5
        after = dst > src
        if kind == 'cue':
            edge = lx + lw + self.CELL_PAD // 2 if after else lx - self.CELL_PAD // 2
            x = int(edge - bar // 2)
            return (x, gy, bar, gh) if gx - bar <= x <= gx + gw else None
        edge = ly + lh + self.CELL_PAD // 2 if after else ly - self.CELL_PAD // 2
        y = int(edge - bar // 2)
        return (gx, y, gw, bar) if gy - bar <= y <= gy + gh else None

    def _update_drop_marker(self, x_root, y_root):
        target = self._drop_target(x_root, y_root)
        geo = self._drop_marker_geometry(*target) if target else None
        if geo is None:
            if self._drop_marker is not None:
                self._drop_marker.place_forget()
            if self.drag_widget is not None and target:
                names = self.cue_names if target[0] == 'cue' else self.layer_names
                self.drag_widget.config(text=names.get(str(target[1]), ''))
            return
        if self._drop_marker is None:
            # Amber, not the accent blue the live cue's header uses.
            self._drop_marker = tk.Frame(self.root, bg=T.WARN, highlightthickness=0, bd=0)
        x, y, w, h = geo
        self._drop_marker.place(x=x, y=y, width=w, height=h)
        self._drop_marker.lift()
        kind, src, dst = target
        names, default = (self.cue_names, 'Cue') if kind == 'cue' else (self.layer_names, 'Layer')
        name = names.get(str(src), f'{default} {src + 1}')
        other = names.get(str(dst), f'{default} {dst + 1}')
        self.drag_widget.config(text=f"{name}  \u2192  {'after' if dst > src else 'before'} {other}")
        self.drag_widget.lift()

    def _cleanup_drag(self):
        if self._drop_marker is not None:
            try:
                self._drop_marker.destroy()
            except tk.TclError:
                pass
            self._drop_marker = None
        if self.drag_widget: self.drag_widget.destroy()
        if self._hover_label:
            try: self._hover_label.config(bg=T.RAISED)
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
            if not frame or not frame.winfo_exists(): return
            kids = frame.winfo_children()
            if not kids: return
            label = kids[0]

            live = live_source.is_live(path)
            if not path or not (live or os.path.exists(path)):
                label.config(image='', bg=T.RAISED, text='[Missing]' if path else '')
                label.image = None
                return
            
            pos_data = self.cell_positions.get((row, col))
            if not pos_data: return
            
            pil_media = self._load_frame(path, for_thumbnail=True)
            if not pil_media: return
            
            _, _, ow, oh = (0, 0) + self._live_size(path) if live else self.mm.get_media_info(path)
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
                 w_res, h_res = self._layout_size(self.selected_display)
            
            display_aspect = w_res / h_res if h_res > 0 else 1.0
            preview_w, preview_h = (self.CELL_SIZE, int(self.CELL_SIZE/display_aspect)) if display_aspect >= 1 else (int(self.CELL_SIZE*display_aspect), self.CELL_SIZE)
            layer_comp = Image.new('RGBA', (preview_w, preview_h), (0,0,0,0))
            scale_factor = preview_w / w_res
            
            preview_layer_w = int(ow * pos_data.get('scale_x', 1.0) * scale_factor)
            preview_layer_h = int(oh * pos_data.get('scale_y', 1.0) * scale_factor)
            
            if preview_layer_w > 0 and preview_layer_h > 0:
                transformed_pil = pil_media.resize((preview_layer_w, preview_layer_h), Image.LANCZOS)
                if pos_data.get('rotation', 0) != 0: transformed_pil = transformed_pil.rotate(pos_data.get('rotation', 0), expand=True, resample=Image.BICUBIC)
                transformed_pil = transformed_pil.convert('RGBA')
                center_x, center_y = pos_data.get('x', 0.5) * preview_w, pos_data.get('y', 0.5) * preview_h
                paste_x, paste_y = int(center_x - transformed_pil.width/2), int(center_y - transformed_pil.height/2)
                layer_comp.alpha_composite(transformed_pil, (paste_x, paste_y))
            
            cell_bg = Image.new('RGB', (self.CELL_SIZE, self.CELL_SIZE), (0,0,0))
            bg_offset_x, bg_offset_y = (self.CELL_SIZE - preview_w)//2, (self.CELL_SIZE - preview_h)//2
            draw = ImageDraw.Draw(cell_bg)
            border_rect = (bg_offset_x, bg_offset_y, bg_offset_x + preview_w -1, bg_offset_y + preview_h-1)
            draw.rectangle(border_rect, outline=T.BORDER, width=1)
            cell_bg.paste(layer_comp, (bg_offset_x, bg_offset_y), layer_comp)
            imgtk = ImageTk.PhotoImage(cell_bg)
            label.config(image=imgtk, text='')
            label.image = imgtk

    THUMB_SOURCE_MAX = (480, 270)
    THUMB_SOURCE_ENTRIES = 64

    def _load_frame(self, path, for_thumbnail=False):
        """First frame (or the image) as RGBA. For thumbnails: a reduced copy
        from a cache that is invalidated when the file changes. A live source
        gives a card with its name (it is never opened here)."""
        if live_source.is_live(path):
            card = self._live_card(path)
            if for_thumbnail:
                card.thumbnail(self.THUMB_SOURCE_MAX)
            return card
        if not for_thumbnail:
            return self._read_first_frame(path, False)
        try:
            st = os.stat(path)
        except (OSError, TypeError):
            return None
        key = (path, st.st_mtime_ns, st.st_size)
        with self._thumb_sources_lock:
            hit = self._thumb_sources.get(key)
            if hit is not None:
                self._thumb_sources.move_to_end(key)
                return hit.copy()
        img = self._read_first_frame(path, True)
        if img is None:
            return None
        img.thumbnail(self.THUMB_SOURCE_MAX, Image.BILINEAR)
        with self._thumb_sources_lock:
            for old in [k for k in self._thumb_sources if k[0] == path]:
                del self._thumb_sources[old]                  # an older version of the file
            self._thumb_sources[key] = img
            while len(self._thumb_sources) > self.THUMB_SOURCE_ENTRIES:
                self._thumb_sources.popitem(last=False)
        return img.copy()

    def _read_first_frame(self, path, for_thumbnail):
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
                still = video_still(load_path)          # the video's own color matrix (BT.709 for HD)
                if still is not None:
                    return still.convert('RGBA')
                cap = cv2.VideoCapture(load_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
            elif ext in IMAGE_EXTS:
                return Image.open(path).convert('RGBA')
        except Exception:
            log.exception(f"Error loading frame from {load_path}")
            return None
        
        return None

    def _select_column(self, col):
        """Make `col` the live cue immediately (a cut)."""
        if self.transition_active or not self.re:
            return
        self._cut_to_column(col)

    def _transition_for(self, to_col):
        """(type, seconds) for going to `to_col`: the cue's own setting, else the show default."""
        transition_type = self.transition_var.get()
        try:
            duration = float(self.transition_timing_var.get())
        except Exception:
            duration = 0.0
        cs = self.cue_settings.get(str(to_col), {}) if to_col is not None else {}
        transition_type = cs.get('transition', transition_type)
        duration = cs.get('duration', duration)
        return transition_type, duration

    def _curve_for(self, transition_type, from_layers, to_layers):
        """The fade curve for a transition type. A dip to black from or to
        black is a plain fade (otherwise it would only use half the time)."""
        curve = project_schema.TRANSITION_CURVES.get(transition_type, 'linear')
        if curve == 'dip' and (not from_layers or not to_layers):
            return 'linear'
        return curve

    def _initiate_transition(self, from_col, to_col):
        """GO from `from_col` to `to_col` with that cue's transition (or the show default)."""
        transition_type, duration = self._transition_for(to_col)
        if transition_type == self.NONE or duration <= 0:
            self._select_column_immediate_cut(to_col)
            return

        to_act = self._claim_activation(to_col)
        # Fade out what the audience sees (unapplied edits to it are dropped from the output).
        self.transition_from_layers = (self._program_layers() if from_col == self.active_col
                                       else self._get_layers_for_col(from_col))
        self.transition_to_layers = self._get_layers_for_col(to_col, activation=to_act)
        self._transition_activation = (to_col, to_act)

        self.transition_active = True
        self.transition_duration = duration
        self.transition_target_col = to_col
        self.transition_start_time = time.perf_counter()
        self._schedule_follow(to_col)   # follow-on time counts from the start of the fade

        # Starts the incoming cue's players (or releases its preloaded ones).
        self._update_active_players()
        self._warm_durations(self._next_cue_with_media_after(to_col))

        if getattr(self, "transition_thread", None):
            output_mode = self.screen_var.get()
            routing = {
                'output_mode': output_mode,
                'target_monitor': self.selected_display if output_mode not in (self.MULTIPLE, self.SPAN) else None,
                'all_assigned_screens': self._assigned_screens(output_mode),
                'gui_monitor': self.gui_monitor_index,
            }
            self.transition_thread.start_transition(
                self.transition_from_layers,
                self.transition_to_layers,
                duration,
                to_col,
                routing,
                curve=self._curve_for(transition_type, self.transition_from_layers, self.transition_to_layers),
            )

    def _finalize_transition_from_thread(self, target_col):
        """A fade finished (called on the Tk thread). The incoming cue's players
        keep running: they become the live cue's players."""
        self.transition_active = False
        if target_col is None:              # fade to black finished
            self._transition_activation = None
            self.transition_from_layers = {}
            self.transition_to_layers = {}
            self._set_program(None, {})
            self.render_command_queue.put(dict(self._output_routing(), action='render', layers={},
                                               transition_state={'active': False}))
            self._update_active_players()
            self._highlight_live_column()
            self._refresh_live_edit_ui()
            self.set_preview_dirty()         # show the result (a still cue never asked again)
            return
        self.active_col = target_col
        ta = self._transition_activation
        if ta and ta[0] == target_col:
            self._live_activation = {target_col: ta[1]}
        self._transition_activation = None

        self._highlight_live_column()
        self._update_cue_dropdown()

        # The incoming layers as faded in become the program. Edits made to
        # this cue during the fade apply now in live mode and wait for Update
        # Output in blind mode.
        self._set_program(self.active_col, self.transition_to_layers or self._get_layers_for_col(self.active_col))
        self.transition_from_layers = {}
        self.transition_to_layers = {}
        self.set_preview_dirty()             # show the result (a still cue never asked again)
        if self.live_edit_mode == 'live' and self._live_edits_pending():
            self._send_program(self._get_layers_for_col(self.active_col))
            return
        self.render_command_queue.put(dict(self._output_routing(), action='render', layers=self._program_layers(),
                                           transition_state={'active': False}))
        # Queued after the render above: stops the outgoing cue, preloads the next.
        self._update_active_players()
        self._refresh_live_edit_ui()

    def _output_is_black(self):
        prog = self._program
        return prog is not None and prog['col'] is None and not prog['layers']

    def _highlight_live_column(self):
        """Live cue light blue; if nothing from the grid is live (blackout, deleted
        live cue) the playhead is shown dimmed."""
        prog = self._program
        live = prog is None or prog['col'] is not None
        try:
            for c, lbl in self.col_labels.items():
                lbl.config(bg=((T.ACCENT if live else T.PLAYHEAD) if c == self.active_col else T.RAISED),
                           text=self._cue_header_text(c))
        except Exception:
            pass  # never let a UI hiccup block a cut
        try:
            self._update_cue_notes_label()
            lbl = getattr(self, 'test_pattern_label', None)
            if lbl is not None:
                if self._test_pattern is not None and not lbl.winfo_ismapped():
                    lbl.pack(after=self._preview_top_bar, fill='x', padx=5)
                elif self._test_pattern is None and lbl.winfo_ismapped():
                    lbl.pack_forget()
            btn = getattr(self, 'blackout_button', None)
            if btn is not None:
                black = self._output_is_black()
                btn.config(text=('OUTPUT BLACK' if black else 'Blackout'),
                           bg=(T.STOP if black else T.RAISED), fg=('white' if black else '#ddd'))
        except Exception:
            log.exception("Could not refresh cue status widgets")

    def _cue_header_text(self, c):
        name = self.cue_names.get(str(c), f'Cue {c+1}')
        cs = self.cue_settings.get(str(c), {})
        if cs.get('number'):
            name = f"{cs['number']}  {name}"
        marks = ''
        if 'transition' in cs or 'duration' in cs:
            marks += ' \u2022'                  # own transition
        if cs.get('follow'):
            marks += ' \u2192'                  # continues by itself
        if cs.get('notes'):
            marks += ' *'
        return name + marks

    def _update_cue_notes_label(self):
        lbl = getattr(self, 'cue_notes_label', None)
        if lbl is None:
            return
        prog = self._program
        col = self.active_col if (prog is None or prog['col'] is not None) else None
        notes = self.cue_settings.get(str(col), {}).get('notes') if col is not None else None
        if notes:
            lbl.config(text=f"{self.cue_names.get(str(col), f'Cue {col+1}')}: {notes}")
            if not lbl.winfo_ismapped():
                lbl.pack(after=self._preview_top_bar, fill='x', padx=5)
        elif lbl.winfo_ismapped():
            lbl.pack_forget()

    # -- follow (auto-continue) -------------------------------------------------
    def _next_cue_with_media_after(self, col):
        return next((c for c in range(col + 1, self.COLS)
                     if any(self.grid_cells.get((r, c)) for r in range(self.ROWS))), None)

    # -- cue clock: how long the live cue has run, against its longest video ----------
    def _start_cue_clock(self, col, elapsed=0.0):
        length, loops = self._cue_video_length(col)
        self._cue_live_since = (col, time.monotonic() - max(0.0, elapsed), length, loops)

    def _clip_seconds(self, path, pos):
        """Seconds one pass of a video takes: between its in and out points, at
        its speed, twice for ping-pong."""
        dur = float(self.mm.get_media_info(path)[1] or 0)
        a, b = layer_trim(pos)
        end = min(b, dur) if b and dur else (b or dur)
        try:
            speed = max(0.1, float((pos or {}).get('speed', 1.0)))
        except (TypeError, ValueError):
            speed = 1.0
        return max(0.0, end - (a or 0.0)) / speed * self._direction_factor(pos)

    @staticmethod
    def _direction_factor(pos):
        """A ping-pong video takes twice its length to come back to the start."""
        return 2.0 if (pos or {}).get('direction') == 'pingpong' else 1.0

    def _cue_video_length(self, col):
        """(seconds, loops) of the cue's longest video at its speed; (0.0, False) without video."""
        best, loops = 0.0, False
        for (r, c), p in self.grid_cells.items():
            if c != col or not p or os.path.splitext(p)[1].lower() not in VIDEO_EXTS:
                continue
            if not os.path.exists(p):              # missing: not drawn, so it sets no length (its
                continue                           # remembered length made a 0.7 s clip "loop" forever)
            pos = self.cell_positions.get((r, c)) or {}
            try:
                length = self._clip_seconds(p, pos)
            except Exception:  # noqa: BLE001
                continue
            if length > best:
                best, loops = length, bool(pos.get('loop', True))
        return best, loops

    def _cue_clock(self, now=None):
        """(col, elapsed, length, loops) for the live cue, or None (black, test pattern, nothing live)."""
        since = getattr(self, '_cue_live_since', None)
        if since is None or self._output_is_black() or self._test_pattern is not None:
            return None
        col, t0, length, loops = since
        if col != self.active_col:
            return None
        now = time.monotonic() if now is None else now
        return col, max(0.0, now - t0), length, loops

    @staticmethod
    def _clock_text(clock):
        if clock is None:
            return ""
        def hms(t):
            t = int(t)
            return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"
        _col, elapsed, length, loops = clock
        if length <= 0:
            return f"Cue {hms(elapsed)}"
        if loops:
            return f"Video {hms(elapsed % length)} / {hms(length)}  loop {int(elapsed // length) + 1}"
        return f"Video {hms(min(elapsed, length))} / {hms(length)}  -{hms(math.ceil(max(0.0, length - elapsed)))}"

    def _refresh_cue_clock(self):
        lbl = getattr(self, 'cue_clock_label', None)
        if lbl is not None:
            text = self._clock_text(self._cue_clock())
            sched = self._schedule_text()
            notice = getattr(self, '_notice', None)
            if notice and time.monotonic() < notice[1]:
                sched = notice[0] + ("\n" + sched if sched else "")
            lbl.config(text=text + ("\n" if text and sched else "") + sched,
                       fg=T.WARN if notice and time.monotonic() < notice[1] else T.MUTED)

    # -- edit lock (Show > Lock Editing) -------------------------------------------------
    def _set_edit_lock(self, locked):
        """Lock or unlock changes to the cue grid and media list for the rest of
        this session (it is not saved with the show)."""
        self.edit_locked = bool(locked)
        var = getattr(self, '_lock_var', None)
        if var is not None and bool(var.get()) != self.edit_locked:
            var.set(self.edit_locked)
        self._update_title()
        self._notice = None
        log.info("Editing locked" if self.edit_locked else "Editing unlocked")

    def _toggle_edit_lock(self):
        self._set_edit_lock(not getattr(self, 'edit_locked', False))

    def _locked_notice(self, action=''):
        self._notice = ("Editing is locked \u2014 Show \u203a Lock Editing (Ctrl+L) to change the show",
                        time.monotonic() + LOCK_NOTICE_S)
        log.info(f"Refused while locked: {action.strip('_').replace('_', ' ')}")
        try:
            self.root.bell()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._refresh_cue_clock()
        except Exception:  # noqa: BLE001
            pass

    def _cue_video_duration(self, col):
        """Longest video in the cue, in seconds (0 if none or unknown)."""
        best = 0.0
        for (r, c), p in self.grid_cells.items():
            if c == col and p and os.path.splitext(p)[1].lower() in VIDEO_EXTS and os.path.exists(p):
                try:
                    best = max(best, self._clip_seconds(p, self.cell_positions.get((r, c)) or {}))
                except Exception as e:  # noqa: BLE001
                    log.warning(f"Could not read the length of {os.path.basename(p)}: {e}")
        return best

    def _warm_durations(self, col):
        """Read the next cue's video durations off the UI thread (cached), so a
        'continue when videos end' cue does not probe files on GO."""
        if col is None or not self.cue_settings.get(str(col), {}).get('follow'):
            return
        paths = [p for (r, c), p in self.grid_cells.items()
                 if c == col and p and os.path.splitext(p)[1].lower() in VIDEO_EXTS]
        for p in paths:
            try:
                self.proxy_executor.submit(self.mm.get_media_info, p)
            except RuntimeError:
                pass   # executor shut down (exiting)

    def _cancel_follow(self):
        self._follow_token += 1
        self._follow_due = None
        if self._follow_timer is not None:
            try:
                self.root.after_cancel(self._follow_timer)
            except Exception:  # noqa: BLE001
                pass
            self._follow_timer = None

    def _schedule_follow(self, col, elapsed=0.0):
        """Arm the cue's follow when it goes live (cut, or start of its fade);
        `elapsed`: the cue starts that many seconds in (Start From)."""
        self._start_cue_clock(col, elapsed)
        self._cancel_follow()
        if self.presentation_active:
            return
        f = self.cue_settings.get(str(col), {}).get('follow')
        if not f:
            return
        if f['mode'] == 'after':
            delay = float(f.get('seconds', 0.0))
        else:
            delay = self._cue_video_duration(col)
            if delay <= 0:
                log.warning(f"Cue {col + 1} is set to continue when its videos end, but has no video "
                            f"with a known length; it waits for GO")
                return
        nxt = self._next_cue_with_media_after(col)
        if nxt is None:
            log.info(f"Cue {col + 1} has a follow, but no later cue has media")
            return
        delay = max(0.0, delay - max(0.0, elapsed))
        token = self._follow_token
        self._follow_due = (time.monotonic() + delay, col, nxt)
        self._follow_watch = None
        wait = delay
        if f['mode'] != 'after':
            key = self._follow_watch_key(col)
            if key is not None:
                info = (getattr(self.re, 'active_players', None) or {}).get(key)
                self._follow_watch = (key, getattr(info['player'], 'loops', 0) if info else 0)
                wait = max(0.0, delay - self.FOLLOW_EARLY_S)   # then watch the video itself
        self._follow_timer = self.root.after(max(1, int(round(wait * 1000))), lambda: self._follow_fire(token))

    FOLLOW_EARLY_S = 1.0     # "when its videos end": watch the longest video from this long before
    FOLLOW_GRACE_S = 2.0     # ... and wait for it at most this long after its computed end
    FOLLOW_POLL_MS = 10

    def _follow_watch_key(self, col):
        """Player key of the longest video in the cue going live (its incoming
        layers during a fade, else the program's), or None."""
        if self.transition_active and getattr(self, 'transition_target_col', None) == col:
            layers = self.transition_to_layers or {}
        elif self._program is not None and self._program.get('col') == col:
            layers = self._program.get('layers') or {}
        else:
            return None
        best, key = 0.0, None
        for layer in layers.values():
            path = layer.get('path')
            if not path or os.path.splitext(path)[1].lower() not in VIDEO_EXTS or not layer.get('instance'):
                continue
            try:
                length = self._clip_seconds(path, layer.get('position') or {})
            except Exception:  # noqa: BLE001
                continue
            if length > best:
                best, key = length, self.re.source_key(layer) if self.re else None
        return key

    def _follow_video_ended(self):
        """True when the watched video has looped or ended since the cue went live
        (or cannot be watched); False while it is still on its first pass."""
        watch = getattr(self, '_follow_watch', None)
        if not watch or not self.re:
            return True
        key, loops0 = watch
        info = (getattr(self.re, 'active_players', None) or {}).get(key)
        if info is None:
            return True
        p = info['player']
        return getattr(p, 'loops', 0) > loops0 or getattr(p, 'state', None) in ('ended', 'stopped', 'error',
                                                                                'no video')

    def _follow_fire(self, token):
        if token != self._follow_token:
            return                                   # canceled or superseded
        self._follow_timer = None
        if self.transition_active:                   # e.g. a long fade still running: try again shortly
            self._follow_timer = self.root.after(50, lambda: self._follow_fire(token))
            return
        due = self._follow_due
        if due and getattr(self, '_follow_watch', None) and not self._follow_video_ended() \
                and time.monotonic() < due[0] + self.FOLLOW_GRACE_S:
            self._follow_timer = self.root.after(self.FOLLOW_POLL_MS, lambda: self._follow_fire(token))
            return                                   # at the video's own end, not the computed one
        self._follow_watch = None
        self._follow_due = None
        if not due or due[1] != self.active_col or not self.re:
            return
        log.info(f"Follow: cue {due[1] + 1} continues to cue {due[2] + 1}")
        self._initiate_transition(due[1], due[2])

    def _refresh_follow_label(self):
        lbl = getattr(self, 'follow_label', None)
        if lbl is None:
            return
        due = self._follow_due
        if not due and time.monotonic() < getattr(self, '_end_of_show_until', 0.0):
            lbl.config(text="End of show: GO stays on the last cue (Show \u203a GO After the Last Cue Returns "
                            "to the First)")
            if not lbl.winfo_ismapped():
                lbl.pack(after=self._preview_top_bar, fill='x', padx=5)
        elif due:
            left = max(0.0, due[0] - time.monotonic())
            lbl.config(text=f"\u2192 continues to {self.cue_names.get(str(due[2]), f'Cue {due[2]+1}')} in {left:.1f} s")
            if not lbl.winfo_ismapped():
                lbl.pack(after=self._preview_top_bar, fill='x', padx=5)
        elif lbl.winfo_ismapped():
            lbl.pack_forget()

    # -- OSC remote control --------------------------------------------------
    def _restart_remote(self):
        self._stop_remote()
        if not self.osc_enabled:
            return
        host = '0.0.0.0' if self.osc_network else '127.0.0.1'
        q = self._remote_queue
        self._remote = remote_control.OSCRemote(lambda cmd, arg, addr: q.put((cmd, arg, addr)), host, self.osc_port)
        self._remote.start()   # on failure .error is set; the Health window reports it

    def _stop_remote(self):
        if self._remote is not None:
            self._remote.stop()
            self._remote = None

    def _restart_web(self):
        self._stop_web()
        if not self.web_enabled:
            return
        host = '0.0.0.0' if self.web_network else '127.0.0.1'
        q = self._remote_queue
        self._web = web_remote.WebRemote(lambda cmd, arg, addr: q.put((cmd, arg, addr)),
                                         lambda: self._web_status, host, self.web_port)
        self._web.start()

    def _stop_web(self):
        if self._web is not None:
            self._web.stop()
            self._web = None

    def _restart_midi(self):
        self._stop_midi()
        if not self.midi_enabled:
            return
        q = self._remote_queue
        self._midi = midi_remote.MidiRemote(lambda cmd, arg: q.put((cmd, arg, ('MIDI', 0))), self.midi_port,
                                            self.midi_mapping, self.midi_program_change,
                                            timecode=self.midi_timecode, faders=self.midi_faders,
                                            msc=self.midi_msc, msc_device=self.midi_msc_device)
        self._tc_prev_s = None                # a new connection: the first time is a start, not a crossing
        self._midi.start()   # on failure .error is set; the Health window reports it

    def _restart_audio(self):
        """Open (or close) the audio output for video sound (Preferences > Audio)."""
        self._stop_audio()
        if not self.audio_device:
            return
        eng = audio_engine.AudioEngine(self.audio_device if self.audio_device != audio_engine.DEFAULT_DEVICE else "",
                                       delay_ms=self.audio_delay_ms)
        if eng.start():
            self._audio = eng
            rt = getattr(self, 'render_thread', None)
            if rt is not None:
                rt.audio = eng
        else:
            self._audio_error = eng.error

    def _stop_audio(self):
        rt = getattr(self, 'render_thread', None)
        if rt is not None:
            rt.audio = None
        eng, self._audio = getattr(self, '_audio', None), None
        if eng is not None:
            eng.stop()
        self._audio_error = None

    def _play_test_sound(self, device):
        """One second of 440 Hz on `device` (the running output if it is that
        one, else a temporary one). Returns (ok, message)."""
        if not device:
            return False, "Choose an output first."
        eng = getattr(self, '_audio', None)
        if eng is not None and eng.device_name == (device if device != audio_engine.DEFAULT_DEVICE else ""):
            eng.play_test_tone()
            return True, f"Playing a 440 Hz tone on '{getattr(eng, 'opened', device)}'\u2026"
        tmp = audio_engine.AudioEngine(device if device != audio_engine.DEFAULT_DEVICE else "")
        if not tmp.start():
            return False, tmp.error or "could not open the output"
        tmp.play_test_tone()
        self.root.after(1400, tmp.stop)
        return True, f"Playing a 440 Hz tone on '{getattr(tmp, 'opened', device)}'\u2026"

    AUDIO_STALL_S = 2.0

    def _audio_watchdog(self, now=None):
        """Reopen the sound output if its callbacks stopped (the device was
        unplugged, or something shut SDL's audio down). Returns True if reopened."""
        eng = getattr(self, '_audio', None)
        if eng is None or eng.backend != 'sdl':
            return False
        now = time.monotonic() if now is None else now
        last = getattr(self, '_audio_cb_seen', None)
        if last is None or eng.callbacks != last[0]:
            self._audio_cb_seen = (eng.callbacks, now)
            return False
        if now - last[1] < self.AUDIO_STALL_S:
            return False
        log.warning(f"Audio output stopped responding ({self.AUDIO_STALL_S:g} s without a callback); reopening it")
        self._audio_cb_seen = None
        self._restart_audio()
        self._audio_restarts = getattr(self, '_audio_restarts', 0) + 1
        return True

    def _audio_status(self):
        if not self.audio_device:
            return {'enabled': False}
        eng = getattr(self, '_audio', None)
        if eng is None:
            return {'enabled': True, 'running': False, 'error': getattr(self, '_audio_error', None) or 'not started'}
        return dict(eng.status(), enabled=True, restarts=getattr(self, '_audio_restarts', 0))

    @staticmethod
    def _audio_text(a):
        if not a.get('enabled'):
            return "Audio off (Preferences)"
        if not a.get('running'):
            return f"Audio not playing: {a.get('error', '?')}"
        streams = a.get('stream_states') or []
        quiet = [f"{s['file']}: {s['state']}" + (f" ({s['error']})" if s.get('error') else '')
                 for s in streams if s.get('playing') and s.get('state') != 'playing']
        return (f"Audio on '{a.get('device')}'   {a.get('playing', 0)} video(s) sounding   peak {a.get('peak', 0):.2f}"
                f"   underruns {a.get('underruns', 0)}" + (f"   reopened {a['restarts']}x" if a.get('restarts') else '')
                + ("   silent: " + "; ".join(quiet) if quiet else ""))

    def _stop_midi(self):
        if self._midi is not None:
            self._midi.stop()
            self._midi = None

    # -- SMPTE LTC from an audio input -----------------------------------------------------
    def _restart_ltc(self):
        self._stop_ltc()
        if not self.ltc_enabled:
            return
        self._ltc = ltc.LTCInput(self.ltc_device, self.ltc_channel)
        self._tc_prev_s = None                # a new input: the first time is a start, not a crossing
        self._ltc.start()                     # on failure .error is set; Health reports it

    def _stop_ltc(self):
        if self._ltc is not None:
            self._ltc.stop()
            self._ltc = None

    def _ltc_status(self):
        if not self.ltc_enabled:
            return {'enabled': False}
        if self._ltc is None:
            return {'enabled': True, 'listening': False, 'error': 'not started'}
        st = dict(self._ltc.status(), enabled=True)
        if st.get('age_s', 99) >= 2.0:                     # not running now: show no time
            st.pop('timecode', None)
        return st

    @staticmethod
    def _ltc_text(m):
        if not m.get('enabled'):
            return "LTC off (Preferences)"
        if not m.get('listening'):
            return f"LTC not listening: {m.get('error', '?')}"
        tc = m.get('timecode')
        return (f"LTC on '{m.get('device')}' ({m.get('channel')})   "
                + (f"{tc} ({m.get('rate')} fps)" if tc else "no timecode now")
                + f"   level {m.get('level', 0):.2f}")

    def _open_ltc_setup(self, parent=None, status_label=None):
        """Follow SMPTE LTC from an audio input: the input and its channel.
        Saved (and applied) with this window's Save button."""
        parent = parent or self.root
        win = tk.Toplevel(parent)
        win.title("LTC Setup")
        win.configure(bg=T.SURFACE)
        win.transient(parent)
        win.grab_set()
        body = tk.Frame(win, bg=T.SURFACE, padx=15, pady=15)
        body.pack(fill='both', expand=True)
        enabled_var = tk.BooleanVar(value=self.ltc_enabled)
        names, err = ltc.capture_device_names()
        choices = [ltc.DEFAULT_DEVICE] + names
        dev_var = tk.StringVar(value=self.ltc_device or ltc.DEFAULT_DEVICE)
        if dev_var.get() not in choices:
            choices.append(dev_var.get())              # saved but not connected now: keep it
        ch_var = tk.StringVar(value=self.ltc_channel)
        ttk.Checkbutton(body, text="Follow LTC: go to each cue at its Timecode (Cue Settings)",
                        variable=enabled_var).grid(row=0, column=0, columnspan=3, sticky='w')
        tk.Label(body, text="Audio input:", bg=T.SURFACE, fg=T.TEXT).grid(row=1, column=0, sticky='w', pady=(8, 2))
        ttk.Combobox(body, textvariable=dev_var, values=choices, width=34, state='readonly').grid(
            row=1, column=1, columnspan=2, sticky='w', pady=(8, 2))
        tk.Label(body, text="Channel:", bg=T.SURFACE, fg=T.TEXT).grid(row=2, column=0, sticky='w', pady=2)
        ttk.Combobox(body, textvariable=ch_var, values=list(ltc.CHANNELS), width=8, state='readonly').grid(
            row=2, column=1, sticky='w', pady=2)
        tk.Label(body, bg=T.SURFACE, fg=T.MUTED, justify='left', anchor='w', wraplength=460,
                 text="Connect the LTC (an audio track or a timecode generator) to an audio input. 24, 25, "
                      "29.97 drop and 30 fps are recognized. While this is on, MIDI timecode is not followed."
                      + (f"\nAudio inputs cannot be listed: {err}" if err else "")).grid(
            row=3, column=0, columnspan=3, sticky='w', pady=(10, 0))
        status = tk.Label(body, bg=T.SURFACE, fg=T.MUTED, anchor='w', text=self._ltc_text(self._ltc_status()))
        status.grid(row=4, column=0, columnspan=3, sticky='w', pady=(6, 0))

        def refresh_status():              # live: the incoming timecode and level
            try:
                if win.winfo_exists():
                    status.config(text=self._ltc_text(self._ltc_status()))
                    win.after(500, refresh_status)
            except tk.TclError:
                pass
        win.after(500, refresh_status)

        def close():
            win.destroy()

        def save():
            self.ltc_enabled = enabled_var.get()
            dev = dev_var.get()
            self.ltc_device = '' if dev == ltc.DEFAULT_DEVICE else dev
            self.ltc_channel = ch_var.get() if ch_var.get() in ltc.CHANNELS else 'Left'
            self._save_config()
            self._restart_ltc()
            log.info(f"LTC settings saved: enabled={self.ltc_enabled} device={self.ltc_device!r} "
                     f"channel={self.ltc_channel}")
            if status_label is not None:
                try:
                    status_label.config(text=self._ltc_text(self._ltc_status()).replace(" (Preferences)", ""))
                except tk.TclError:
                    pass
            close()

        btns = tk.Frame(body, bg=T.SURFACE)
        btns.grid(row=5, column=0, columnspan=3, sticky='e', pady=(14, 0))
        tk.Button(btns, text="Save", command=save, bg=T.GO, fg='black', width=9).pack(side='right', padx=(6, 0))
        tk.Button(btns, text="Cancel", command=close, width=9).pack(side='right')
        win.protocol("WM_DELETE_WINDOW", close)
        win.bind('<Escape>', lambda e: close())
        win._vd = {'save': save, 'close': close, 'enabled_var': enabled_var, 'dev_var': dev_var,
                   'ch_var': ch_var, 'status': status, 'choices': choices}     # for tests
        return win

    def _open_midi_setup(self, parent=None, status_label=None):
        """MIDI input, Program Change and learned triggers. Saved (and applied)
        with this window's Save button, independently of Preferences."""
        parent = parent or self.root
        win = tk.Toplevel(parent)
        win.title("MIDI Setup")
        win.configure(bg=T.SURFACE)
        win.transient(parent)
        win.grab_set()
        self._midi_win = win
        body = tk.Frame(win, bg=T.SURFACE, padx=15, pady=15)
        body.pack(fill='both', expand=True)
        enabled_var = tk.BooleanVar(value=self.midi_enabled)
        pc_var = tk.BooleanVar(value=self.midi_program_change)
        port_var = tk.StringVar(value=self.midi_port)
        mapping = dict(self.midi_mapping)
        ttk.Checkbutton(body, text="Listen for MIDI input", variable=enabled_var).grid(
            row=0, column=0, columnspan=4, sticky='w')
        tk.Label(body, text="Input port:", bg=T.SURFACE, fg=T.TEXT).grid(row=1, column=0, sticky='w', pady=(8, 2))
        port_box = ttk.Combobox(body, textvariable=port_var, width=32, state='readonly')
        port_box.grid(row=1, column=1, columnspan=2, sticky='w', pady=(8, 2))
        note = tk.Label(body, bg=T.SURFACE, fg=T.MUTED, justify='left', anchor='w', wraplength=420)

        def refresh_ports():
            ports, err = midi_remote.list_ports()
            port_box['values'] = ports
            if ports and port_var.get() not in ports:
                idx = midi_remote.find_port(ports, port_var.get())
                port_var.set(ports[idx if idx is not None else 0])
            note.config(text=err or ("" if ports else "No MIDI input ports. Connect a device, or on Windows "
                                     "create a virtual port (for example with loopMIDI), then Refresh."))
        ttk.Button(body, text="Refresh", command=refresh_ports).grid(row=1, column=3, sticky='w', padx=(6, 0),
                                                                    pady=(8, 2))
        opts = tk.Frame(body, bg=T.SURFACE)
        opts.grid(row=2, column=0, columnspan=4, sticky='w', pady=(6, 2))
        ttk.Checkbutton(opts, text="Program Change selects the cue (program 1 = cue 1)", variable=pc_var).pack(
            anchor='w')
        tc_var = tk.BooleanVar(value=self.midi_timecode)
        ttk.Checkbutton(opts, text="Follow MIDI timecode: go to each cue at its Timecode (Cue Settings)",
                        variable=tc_var).pack(anchor='w', pady=(2, 0))
        msc_row = tk.Frame(opts, bg=T.SURFACE)
        msc_row.pack(anchor='w', pady=(2, 0))
        msc_var = tk.BooleanVar(value=self.midi_msc)
        msc_dev_var = tk.StringVar(value=str(self.midi_msc_device))
        msc_cb = ttk.Checkbutton(msc_row, text="MIDI Show Control (MSC), device ID", variable=msc_var)
        msc_cb.pack(side='left')
        ttk.Spinbox(msc_row, from_=0, to=111, width=4, textvariable=msc_dev_var).pack(side='left', padx=(4, 0))
        CreateToolTip(msc_cb, "From a lighting or sound desk or QLab: GO with a cue number goes to that cue "
                              "(cue numbers first, then positions), GO without one is GO, STOP and ALL_OFF fade to "
                              "black. Answers its device ID and 127 (all), command formats Video, Projection and "
                              "All types. Help \u203a Remote control has the details.")
        self._help_link(msc_row, "Remote control", bg=T.SURFACE).pack(side='left', padx=(6, 0))
        tk.Label(body, text="Triggers (a key or button: Note On, or a Control Change going to 64 or more)",
                 bg=T.SURFACE, fg=T.TEXT, font=T.font(10, 'bold')).grid(row=3, column=0, columnspan=4, sticky='w',
                                                                        pady=(12, 4))
        labels, buttons = {}, {}
        learning = {'action': None, 'listener': None, 'deadline': 0.0}
        results = queue.Queue()

        def show(action):
            labels[action].config(text=midi_remote.trigger_text(mapping.get(action)))

        def end_learn(message=None):
            lst, learning['listener'] = learning['listener'], None
            if lst is not None:
                lst.stop()
            act, learning['action'] = learning['action'], None
            if act is not None and act in buttons:
                try:
                    buttons[act].config(text="Learn")
                except tk.TclError:
                    pass
            if message is not None:
                note.config(text=message)
            if self._midi is None and self.midi_enabled:
                self._restart_midi()          # the saved settings until Save is pressed

        def poll():
            if learning['action'] is None:
                return
            try:
                trig = results.get_nowait()
            except queue.Empty:
                if time.monotonic() > learning['deadline']:
                    end_learn("Nothing received in 15 s.")
                else:
                    win.after(50, poll)
                return
            act = learning['action']
            if isinstance(act, tuple) and trig[0] != 'cc':
                end_learn("A fader needs a Control Change (a fader or knob), not a key.")
                return
            moved = [a for a, t in mapping.items() if a != act and t == trig]
            for a in moved:
                del mapping[a]
                show(a)
            moved_faders = [j for j, f in enumerate(faders) if ('fader', j) != act and trig[0] == 'cc'
                            and f['cc'] == [trig[1], trig[2]]]
            for j in moved_faders:
                faders[j]['cc'] = None
            if isinstance(act, tuple):
                faders[act[1]]['cc'] = [trig[1], trig[2]]
            else:
                mapping[act] = trig
                show(act)
            draw_faders()
            was = [midi_remote.ACTION_LABELS[a] for a in moved] + [f"fader {j + 1}" for j in moved_faders]
            end_learn(f"{label_of(act)}: {midi_remote.trigger_text(trig)}" +
                      (f" (was {', '.join(was)})" if was else ""))

        def label_of(action):
            if isinstance(action, tuple):
                return f"Fader {action[1] + 1}"
            return midi_remote.ACTION_LABELS[action]

        def learn(action):
            if learning['action'] is not None:
                end_learn("")
                return
            self._stop_midi()                     # Windows opens a MIDI input for one program at a time
            lst = midi_remote.MidiRemote(lambda *a: None, port_var.get())
            if not lst.start():
                note.config(text=lst.error)
                end_learn()
                return
            lst.learn(results.put)
            learning.update(action=action, listener=lst, deadline=time.monotonic() + 15.0)
            if isinstance(action, tuple):
                buttons[action].config(text="Move it…")
                note.config(text=f"Move the fader or knob for {label_of(action)} up past the middle on "
                                 f"'{lst.opened_port}' (click again to stop).")
            else:
                buttons[action].config(text="Press a key…")
                note.config(text=f"Press the key or button for {midi_remote.ACTION_LABELS[action]} on "
                                 f"'{lst.opened_port}' (click again to stop).")
            win.after(50, poll)

        def clear(action):
            mapping.pop(action, None)
            show(action)

        for i, action in enumerate(midi_remote.ACTIONS):
            r = 4 + i
            tk.Label(body, text=midi_remote.ACTION_LABELS[action], bg=T.SURFACE, fg=T.TEXT, width=14,
                     anchor='w').grid(row=r, column=0, sticky='w', pady=2)
            labels[action] = tk.Label(body, bg=T.SURFACE, fg=T.TEXT, width=24, anchor='w')
            labels[action].grid(row=r, column=1, sticky='w')
            buttons[action] = ttk.Button(body, text="Learn", width=13, command=lambda a=action: learn(a))
            buttons[action].grid(row=r, column=2, sticky='w', padx=(6, 0))
            ttk.Button(body, text="Clear", width=7, command=lambda a=action: clear(a)).grid(
                row=r, column=3, sticky='w', padx=(6, 0))
            show(action)
        tk.Label(body, text="Faders for effect inputs (a Control Change: every value 0–127 sets the input)",
                 bg=T.SURFACE, fg=T.TEXT, font=T.font(10, 'bold')).grid(row=9, column=0, columnspan=4, sticky='w',
                                                                        pady=(12, 2))
        tk.Label(body, bg=T.SURFACE, fg=T.MUTED, anchor='w', justify='left', wraplength=520, text=(
            "An input of a shader effect on the output: on a layer of the live cue (1 = first) or on a display. "
            "The input by its name or label; with several effects, 2:amount is the second effect's. "
            "A fader moves a slider across its range, picks a list's values in turn, and turns an on/off "
            "on from the middle.")).grid(row=10, column=0, columnspan=4, sticky='w')
        faders = [dict(f) for f in self.midi_faders]
        fader_box = tk.Frame(body, bg=T.SURFACE)
        fader_box.grid(row=11, column=0, columnspan=4, sticky='we', pady=(4, 0))

        def draw_faders():
            for w in fader_box.winfo_children():
                w.destroy()
            for key in [k for k in buttons if isinstance(k, tuple)]:
                del buttons[key]
            for i, f in enumerate(faders):
                rowf = tk.Frame(fader_box, bg=T.SURFACE)
                rowf.pack(fill='x', pady=1)
                tk.Label(rowf, text=f"{i + 1}.", bg=T.SURFACE, fg=T.MUTED, width=3, anchor='w').pack(side='left')
                tvar = tk.StringVar(value='Layer' if f['target'] == 'layer' else 'Display')
                ivar = tk.StringVar(value=str(f['index']))
                nvar = tk.StringVar(value=f['input'])

                def changed(*_a, i=i, tvar=tvar, ivar=ivar, nvar=nvar):
                    fd = faders[i]
                    fd['target'] = 'layer' if tvar.get() == 'Layer' else 'display'
                    try:
                        fd['index'] = int(ivar.get())
                    except ValueError:
                        pass
                    fd['input'] = nvar.get().strip()
                tbox = ttk.Combobox(rowf, textvariable=tvar, state='readonly', width=8, values=['Layer', 'Display'])
                tbox.pack(side='left')
                tbox.bind('<<ComboboxSelected>>', changed)
                ttk.Spinbox(rowf, from_=0, to=99, width=4, textvariable=ivar, command=changed).pack(
                    side='left', padx=(4, 0))
                tk.Label(rowf, text="input", bg=T.SURFACE, fg=T.MUTED).pack(side='left', padx=(6, 2))
                ttk.Entry(rowf, textvariable=nvar, width=14).pack(side='left')
                for v in (ivar, nvar):
                    v.trace_add('write', changed)
                tk.Label(rowf, bg=T.SURFACE, fg=T.TEXT, width=14, anchor='w',
                         text=midi_remote.trigger_text(['cc'] + f['cc']) if f['cc'] else "not learned").pack(
                    side='left', padx=(6, 0))
                buttons[('fader', i)] = ttk.Button(rowf, text="Learn", width=10,
                                                   command=lambda i=i: learn(('fader', i)))
                buttons[('fader', i)].pack(side='left', padx=(4, 0))
                ttk.Button(rowf, text="✕", width=3, command=lambda i=i: remove_fader(i)).pack(side='left',
                                                                                             padx=(4, 0))
            add = ttk.Button(fader_box, text="Add fader", command=add_fader)
            add.pack(anchor='w', pady=(4, 0))
            if len(faders) >= midi_remote.MAX_FADERS:
                add.state(['disabled'])

        def add_fader():
            if learning['action'] is not None:
                end_learn("")
            faders.append({'cc': None, 'target': 'layer', 'index': 1, 'input': 'amount'})
            draw_faders()

        def remove_fader(i):
            if learning['action'] is not None:
                end_learn("")
            if 0 <= i < len(faders):
                del faders[i]
                draw_faders()

        draw_faders()
        note.grid(row=12, column=0, columnspan=4, sticky='w', pady=(10, 0))
        status = tk.Label(body, bg=T.SURFACE, fg=T.MUTED, anchor='w', text=self._midi_text(self._midi_status()))
        status.grid(row=13, column=0, columnspan=4, sticky='w', pady=(4, 0))

        def refresh_status():              # live: commands received and the incoming timecode
            try:
                if win.winfo_exists():
                    status.config(text=self._midi_text(self._midi_status()))
                    win.after(500, refresh_status)
            except tk.TclError:
                pass
        win.after(500, refresh_status)

        def close():
            end_learn()
            self._midi_win = None
            win.destroy()

        def save():
            end_learn()
            self.midi_enabled, self.midi_port = enabled_var.get(), port_var.get()
            self.midi_program_change, self.midi_mapping = pc_var.get(), midi_remote.clean_mapping(mapping)
            self.midi_timecode = tc_var.get()
            self.midi_msc = msc_var.get()
            try:
                dev = int(msc_dev_var.get())
            except ValueError:
                dev = self.midi_msc_device
            self.midi_msc_device = dev if 0 <= dev <= 111 else self.midi_msc_device
            self.midi_faders = midi_remote.clean_faders(faders)
            self._save_config()
            self._restart_midi()
            log.info(f"MIDI settings saved: enabled={self.midi_enabled} port={self.midi_port!r} "
                     f"program_change={self.midi_program_change} timecode={self.midi_timecode} "
                     f"msc={self.midi_msc} msc_device={self.midi_msc_device} "
                     f"mapping={self.midi_mapping} faders={self.midi_faders}")
            if status_label is not None:
                try:
                    status_label.config(text=self._midi_text(self._midi_status()).replace(" (Preferences)", ""))
                except tk.TclError:
                    pass
            close()

        btns = tk.Frame(body, bg=T.SURFACE)
        btns.grid(row=14, column=0, columnspan=4, sticky='e', pady=(14, 0))
        tk.Button(btns, text="Save", command=save, bg=T.GO, fg='black', width=9).pack(side='right', padx=(6, 0))
        tk.Button(btns, text="Cancel", command=close, width=9).pack(side='right')
        win.protocol("WM_DELETE_WINDOW", close)
        win.bind('<Escape>', lambda e: close())
        refresh_ports()
        win._vd = {'learn': learn, 'save': save, 'close': close, 'labels': labels, 'mapping': mapping,
                   'port_var': port_var, 'enabled_var': enabled_var, 'pc_var': pc_var, 'tc_var': tc_var, 'note': note,
                   'buttons': buttons, 'faders': faders, 'add_fader': add_fader, 'remove_fader': remove_fader,
                   'fader_box': fader_box}                  # for tests
        return win

    # -- show schedule ---------------------------------------------------------------------
    def _schedule_check(self, now=None):
        """Run events due now (local time). Returns the events run."""
        if not self.schedule.get('enabled') or not self.schedule.get('events'):
            return []
        now = datetime.datetime.now() if now is None else now
        ran = scheduler.due(self.schedule['events'], now, self._schedule_fired)
        for ev in ran:
            log.info(f"Schedule: {scheduler.event_text(ev)}")
            if ev['action'] == 'cue':
                arg = ev['cue']
                self._handle_remote_command('cue', int(arg) if arg.isdigit() else arg, ('Schedule', 0))
            else:
                self._handle_remote_command(ev['action'], None, ('Schedule', 0))
        return ran

    def _schedule_tick(self):
        try:
            self._schedule_check()
            self._refresh_ready()
        except Exception:  # noqa: BLE001 - the schedule must never stop the UI loop
            log.exception("Schedule check failed")
        finally:
            self.root.after(1000, self._schedule_tick)

    def _schedule_text(self, now=None):
        sc = self.schedule
        if not sc.get('enabled') or not sc.get('events'):
            return ""
        now = datetime.datetime.now() if now is None else now
        nxt = scheduler.next_event(sc['events'], now)
        if nxt is None:
            return "Schedule on: no upcoming event"
        when, ev = nxt
        mins = max(1, math.ceil((when - now).total_seconds() / 60))
        left = f"{mins // 60} h {mins % 60:02d} min" if mins >= 60 else f"{mins} min"
        what = f"cue {ev['cue']}" if ev['action'] == 'cue' else scheduler.ACTION_LABELS[ev['action']]
        day = "" if when.date() == now.date() else f"{scheduler.DAY_NAMES[when.weekday()]} "
        return f"Schedule: next {day}{ev['time']} {what} (in {left})"

    @_locked_out
    def _open_schedule(self):
        """Show > Schedule...: times of day that go to a cue, GO or black out."""
        win = tk.Toplevel(self.root)
        win.title("Schedule")
        win.configure(bg=T.SURFACE)
        win.transient(self.root)
        self._schedule_win = win
        body = tk.Frame(win, bg=T.SURFACE, padx=14, pady=14)
        body.pack(fill='both', expand=True)
        st = {'events': [dict(e) for e in self.schedule['events']]}
        enabled_var = tk.BooleanVar(value=self.schedule['enabled'])
        ttk.Checkbutton(body, text="Run this schedule while the show is open", variable=enabled_var).pack(anchor='w')
        tk.Label(body, bg=T.SURFACE, fg=T.MUTED, justify='left', anchor='w', wraplength=460, text=(
            "Each event runs in its minute, once a day, using this computer's clock. Events missed while "
            "VisualDeck was closed are not caught up.")).pack(fill='x', pady=(4, 8))
        lst = tk.Listbox(body, height=8, width=52, bg=T.INPUT, fg=T.TEXT, selectbackground=T.ACCENT,
                         highlightthickness=0, font=T.mono(10))
        lst.pack(fill='x')

        def refill():
            lst.delete(0, 'end')
            for ev in st['events']:
                lst.insert('end', scheduler.event_text(ev))

        form = tk.Frame(body, bg=T.SURFACE)
        form.pack(fill='x', pady=(10, 0))
        time_var = tk.StringVar(value="09:00")
        days_var = tk.StringVar(value="Every day")
        action_var = tk.StringVar(value=scheduler.ACTION_LABELS['cue'])
        cue_var = tk.StringVar(value="1")
        tk.Label(form, text="Time", bg=T.SURFACE, fg=T.TEXT).grid(row=0, column=0, sticky='w')
        tk.Entry(form, textvariable=time_var, width=6).grid(row=1, column=0, sticky='w', padx=(0, 8))
        tk.Label(form, text="Days", bg=T.SURFACE, fg=T.TEXT).grid(row=0, column=1, sticky='w')
        ttk.Combobox(form, textvariable=days_var, state='readonly', width=10,
                     values=list(scheduler.DAY_PRESETS)).grid(row=1, column=1, sticky='w', padx=(0, 8))
        tk.Label(form, text="Action", bg=T.SURFACE, fg=T.TEXT).grid(row=0, column=2, sticky='w')
        ttk.Combobox(form, textvariable=action_var, state='readonly', width=10,
                     values=[scheduler.ACTION_LABELS[a] for a in scheduler.ACTIONS]).grid(row=1, column=2, sticky='w',
                                                                                       padx=(0, 8))
        tk.Label(form, text="Cue", bg=T.SURFACE, fg=T.TEXT).grid(row=0, column=3, sticky='w')
        tk.Entry(form, textvariable=cue_var, width=8).grid(row=1, column=3, sticky='w')
        err = tk.Label(body, bg=T.SURFACE, fg='#ff8888', anchor='w')
        err.pack(fill='x', pady=(4, 0))

        def add():
            action = next(a for a, l in scheduler.ACTION_LABELS.items() if l == action_var.get())
            ev = scheduler.clean_event({'time': time_var.get(), 'days': scheduler.DAY_PRESETS[days_var.get()],
                                        'action': action, 'cue': cue_var.get()})
            if ev is None:
                err.config(text="Time as HH:MM (24 h), and a cue for “Go to cue”")
                return
            err.config(text="")
            st['events'] = scheduler.clean({'events': st['events'] + [ev]})['events']
            refill()

        def remove():
            for i in sorted(lst.curselection(), reverse=True):
                del st['events'][i]
            refill()

        row = tk.Frame(body, bg=T.SURFACE)
        row.pack(fill='x', pady=(6, 0))
        ttk.Button(row, text="Add", command=add).pack(side='left')
        ttk.Button(row, text="Remove Selected", command=remove).pack(side='left', padx=(6, 0))

        def save():
            new = scheduler.clean({'enabled': enabled_var.get(), 'events': st['events']})
            if new != self.schedule:
                self.schedule = new
                self._schedule_fired = set()
                # Events of the current minute do not run just because the schedule was saved now.
                scheduler.due(new['events'], datetime.datetime.now(), self._schedule_fired)
                self._set_dirty_flag()
                log.info(f"Schedule {'on' if new['enabled'] else 'off'}: "
                         + "; ".join(scheduler.event_text(e) for e in new['events']))
            win.destroy()

        btns = tk.Frame(body, bg=T.SURFACE)
        btns.pack(fill='x', pady=(14, 0))
        tk.Button(btns, text="Save", command=save, bg=T.GO, fg='black', width=9).pack(side='right', padx=(6, 0))
        tk.Button(btns, text="Cancel", command=win.destroy, width=9).pack(side='right')
        win.bind('<Escape>', lambda e: win.destroy())
        refill()
        win._vd = {'st': st, 'add': add, 'remove': remove, 'save': save, 'list': lst, 'time_var': time_var,
                   'days_var': days_var, 'action_var': action_var, 'cue_var': cue_var, 'enabled_var': enabled_var,
                   'err': err}                                    # for tests
        return win

    # -- live sources (live_source.py) --------------------------------------------------
    LIVE_DEFAULT_SIZE = (1920, 1080)

    def _live_size(self, path):
        """(width, height) of a live source: as last seen (Test, or the running
        player), else 1920x1080 until it has been seen."""
        sizes = self.__dict__.setdefault('live_sizes', {})
        info = (getattr(self.re, 'active_players', None) or {}).get(path) if self.re else None
        seen = getattr(info.get('player'), 'size', None) if info else None
        if seen:
            sizes[path] = tuple(seen)
        return tuple(sizes.get(path) or self.LIVE_DEFAULT_SIZE)

    def _live_card(self, path, size=None):
        """A stand-in picture for a live source (thumbnails, editor): its
        size, a LIVE badge and its name."""
        from PIL import ImageFont
        w, h = size or self._live_size(path)
        img = Image.new('RGBA', (int(w), int(h)), (28, 30, 36, 255))
        d = ImageDraw.Draw(img)
        fs = max(10, int(h) // 7)
        try:
            font = ImageFont.load_default(size=fs)
            small = ImageFont.load_default(size=max(8, fs // 2))
        except TypeError:                      # Pillow < 10.1
            font = small = ImageFont.load_default()
        bw, bh = int(fs * 2.6), int(fs * 1.4)
        x0, y0 = (int(w) - bw) // 2, int(h) // 2 - bh
        d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=max(2, bh // 5), fill=(210, 40, 40, 255))
        d.text((x0 + bw // 2, y0 + bh // 2), "LIVE", fill=(255, 255, 255, 255), font=font, anchor='mm')
        d.text((int(w) // 2, y0 + bh + fs // 2), live_source.label(path)[:48], fill=(220, 220, 220, 255),
               font=small, anchor='ma')
        return img

    @_locked_out
    def _open_live_source_dialog(self, path=None):
        """Insert > Add Live Source...: a network stream or capture device,
        tested before it is added to the media list. With `path`: test it."""
        win = tk.Toplevel(self.root)
        win.title("Live Source")
        win.configure(bg=T.SURFACE)
        win.transient(self.root)
        body = tk.Frame(win, bg=T.SURFACE, padx=14, pady=14)
        body.pack(fill='both', expand=True)
        devices = live_source.available()
        examples = [("udp://0.0.0.0:5000", "MPEG-TS sent to this computer, port 5000"),
                    ("rtsp://192.168.1.20/stream1", "IP camera"),
                    ("srt://host:9000", "SRT; also rtmp://, https://…/live.m3u8")]
        dev_line = {'dshow': ("dshow://video=Logitech BRIO", "camera on Windows"),
                    'v4l2': ("v4l2:///dev/video0", "camera on Linux"),
                    'avfoundation': ("avfoundation://0", "camera on macOS")}
        examples += [dev_line[k] for k, ok in devices.items() if ok]
        tk.Label(body, text="Address", bg=T.SURFACE, fg=T.TEXT).pack(anchor='w')
        url_var = tk.StringVar(value=path or "udp://0.0.0.0:5000")
        entry = tk.Entry(body, textvariable=url_var, width=56)
        entry.pack(fill='x', pady=(2, 6))
        ex = tk.Frame(body, bg=T.SURFACE)
        ex.pack(fill='x')
        for i, (addr, what) in enumerate(examples):
            tk.Label(ex, text=addr, bg=T.SURFACE, fg=T.TEXT, font=T.mono(9), anchor='w').grid(row=i, column=0,
                                                                                            sticky='w')
            tk.Label(ex, text=what, bg=T.SURFACE, fg=T.MUTED, anchor='w').grid(row=i, column=1, sticky='w',
                                                                              padx=(12, 0))
        if not any(devices.values()):
            tk.Label(body, bg=T.SURFACE, fg=T.MUTED, justify='left', anchor='w', wraplength=440,
                     text="Capture devices: the FFmpeg in this VisualDeck has no camera support "
                          "(network streams work).").pack(fill='x', pady=(4, 0))
        # Capture devices FFmpeg can see (Find Devices): click one to use it.
        dev_list = tk.Listbox(body, height=0, bg=T.INPUT, fg=T.TEXT, selectbackground=T.ACCENT,
                              highlightthickness=0, activestyle='none')
        found = {'urls': []}

        def pick(_event=None):
            sel = dev_list.curselection()
            if sel:
                url_var.set(found['urls'][sel[0]])
        dev_list.bind('<<ListboxSelect>>', pick)
        result = tk.Label(body, bg=T.SURFACE, fg=T.TEXT, anchor='w', justify='left', wraplength=440)
        result.pack(fill='x', pady=(10, 0))
        picture = tk.Label(body, bg=T.SURFACE)      # what the source sends (after Test)
        st = {'testing': False, 'last': None, 'finding': False}
        results = queue.Queue()
        dev_results = queue.Queue()

        def poll_devices():
            try:
                r = dev_results.get_nowait()
            except queue.Empty:
                if st['finding'] and win.winfo_exists():
                    win.after(100, poll_devices)
                return
            st['finding'] = False
            if not win.winfo_exists():
                return
            found['urls'] = [u for u, _ in r['devices']]
            dev_list.delete(0, 'end')
            for _, desc in r['devices']:
                dev_list.insert('end', desc)
            if r['devices']:
                dev_list.config(height=min(6, len(r['devices'])))
                dev_list.pack(fill='x', pady=(6, 0), before=result)
                result.config(fg=T.MUTED, text=f"{len(r['devices'])} device(s) found: click one, then Test."
                              + (f" ({r['note']})" if r.get('note') else ""))
            else:
                dev_list.pack_forget()
                result.config(fg=T.MUTED, text=r.get('note') or "No capture devices found.")

        def find_devices():
            if st['finding']:
                return
            st['finding'] = True
            result.config(fg=T.MUTED, text="Looking for capture devices…")
            threading.Thread(target=lambda: dev_results.put(live_source.find_devices()), daemon=True,
                             name="FindDevices").start()
            win.after(100, poll_devices)

        def poll():
            try:
                url, r = results.get_nowait()
            except queue.Empty:
                if st['testing'] and win.winfo_exists():
                    win.after(100, poll)
                return
            st['testing'], st['last'] = False, (url, r)
            if not win.winfo_exists():
                return
            if r['ok']:
                self.__dict__.setdefault('live_sizes', {})[url] = (r['width'], r['height'])
                fps = f", {r['fps']:g} fps" if r.get('fps') else ""
                sound = f", sound {r['audio']}" if r.get('audio') else ", no sound"
                result.config(fg=T.GO, text=f"Receiving: {r['width']}\u00d7{r['height']}{fps}, {r['codec']}{sound} "
                                            f"(first picture after {r['seconds']:g} s)")
                if r.get('preview') is not None:
                    st['photo'] = ImageTk.PhotoImage(r['preview'])
                    picture.config(image=st['photo'])
                    picture.pack(anchor='w', pady=(6, 0), after=result)
            else:
                result.config(fg='#ff8888', text=f"No picture: {r['error']}")
                picture.pack_forget()

        def test():
            url = live_source.clean(url_var.get())
            if url is None:
                result.config(fg='#ff8888', text="Type an address such as udp://0.0.0.0:5000 or rtsp://camera/stream.")
                return
            if st['testing']:
                return
            st['testing'] = True
            result.config(fg=T.MUTED, text=f"Connecting to {live_source.label(url)}… (up to "
                                           f"{2 * live_source.OPEN_TIMEOUT:g} s)")
            threading.Thread(target=lambda: results.put((url, live_source.probe(url))), daemon=True,
                             name="LiveProbe").start()
            win.after(100, poll)

        def add():
            url = live_source.clean(url_var.get())
            if url is None:
                result.config(fg='#ff8888', text="Type an address such as udp://0.0.0.0:5000 or rtsp://camera/stream.")
                return None
            if url in self.project_media_paths:
                result.config(fg=T.MUTED, text=f"{live_source.label(url)} is already in the media list.")
                return None
            self.project_media_paths.append(url)
            self._build_media_list()
            self._set_dirty_flag()
            log.info(f"Live source added: {live_source.label(url)}")
            win.destroy()
            return url

        btns = tk.Frame(body, bg=T.SURFACE)
        btns.pack(fill='x', pady=(14, 0))
        ttk.Button(btns, text="Test", command=test).pack(side='left')
        ttk.Button(btns, text="Find Devices", command=find_devices).pack(side='left', padx=(6, 0))
        tk.Button(btns, text="Close", command=win.destroy, width=9).pack(side='right', padx=(6, 0))
        if path is None:
            tk.Button(btns, text="Add", command=add, bg=T.GO, fg='black', width=9).pack(side='right')
        win.bind('<Escape>', lambda e: win.destroy())
        entry.focus_set()
        if path is not None:
            test()
        win._vd = {'url_var': url_var, 'test': test, 'add': add, 'result': result, 'st': st,
                   'poll': poll, 'find_devices': find_devices, 'poll_devices': poll_devices,
                   'dev_list': dev_list, 'picture': picture}      # for tests
        return win

    # -- output presets ----------------------------------------------------------------
    def _output_presets_path(self):
        return os.path.join(self.CONFIG_DIR, output_presets.FILE_NAME)

    def _current_output_preset(self):
        return output_presets.capture(self.screen_var.get(), self.layer_screens, self.output_warps,
                                      self.output_blends, self.re.monitors if self.re else [],
                                      meshes=self.output_meshes, span=self.output_span)

    def _apply_output_preset(self, preset):
        """Copy a preset's output settings into the show and the output (the
        caller has asked the operator). Returns the output it ended on."""
        self.layer_screens = {k: v for k, v in preset['layer_screens'].items()}
        if self.output_warps != preset['warps']:
            self.output_warps = {k: [list(p) for p in c] for k, c in preset['warps'].items()}
            self.render_command_queue.put({'action': 'set_warps', 'warps': dict(self.output_warps)})
        if self.output_blends != preset['blends']:
            self.output_blends = {k: dict(b) for k, b in preset['blends'].items()}
            self.render_command_queue.put({'action': 'set_blends', 'blends': dict(self.output_blends)})
        if self.output_meshes != preset['meshes']:
            self.output_meshes = {k: mesh_warp.as_dict(mesh_warp.clean(m)) for k, m in preset['meshes'].items()}
            self.render_command_queue.put({'action': 'set_meshes', 'meshes': dict(self.output_meshes)})
        if preset.get('span') is not None or self.output_span is not None:
            self.output_span = preset.get('span')
            self._refresh_output_options()
        if self.screen_var.get() == preset['mode']:
            self._on_output_change()          # same output: layer displays may still differ
        else:
            self.screen_var.set(preset['mode'])   # trace: _on_output_change (checks the display exists)
        self._set_dirty_flag()
        log.info(f"Output preset applied: {output_presets.summary(preset)} (output now {self.screen_var.get()})")
        return self.screen_var.get()

    def _open_output_presets(self):
        """Window > Output Presets...: save the output setup (output, layer
        displays, corner pins, edge blends) under a name on this computer and
        apply it to any show."""
        path = self._output_presets_path()
        presets, note, writable = output_presets.load(path)
        st = {'presets': presets, 'writable': writable}
        win = tk.Toplevel(self.root)
        win.title("Output Presets")
        win.configure(bg=T.SURFACE)
        win.transient(self.root)
        self._output_presets_win = win
        body = tk.Frame(win, bg=T.SURFACE, padx=14, pady=14)
        body.pack(fill='both', expand=True)
        tk.Label(body, bg=T.SURFACE, fg=T.MUTED, justify='left', anchor='w', wraplength=460, text=(
            "A preset is the output setup: the output, each layer's display, and each display's corner pin "
            "and edge blend. Presets are kept on this computer; applying one changes the open show and the "
            "output at once.")).pack(fill='x', pady=(0, 8))
        lst = tk.Listbox(body, height=8, width=60, bg=T.INPUT, fg=T.TEXT, selectbackground=T.ACCENT,
                         highlightthickness=0, font=T.mono(10), exportselection=False)
        lst.pack(fill='x')
        msg = tk.Label(body, bg=T.SURFACE, fg=T.MUTED, anchor='w', justify='left', wraplength=460)
        msg.pack(fill='x', pady=(4, 0))

        def say(text, error=False):
            msg.config(text=text, fg='#ff8888' if error else T.MUTED)

        def names():
            return sorted(st['presets'], key=str.lower)

        def refill(select=None):
            lst.delete(0, 'end')
            for n in names():
                lst.insert('end', f"{n:<20}  {output_presets.summary(st['presets'][n])}")
            if select in st['presets']:
                i = names().index(select)
                lst.selection_set(i)
                lst.see(i)

        def selected():
            sel = lst.curselection()
            return names()[sel[0]] if sel else None

        form = tk.Frame(body, bg=T.SURFACE)
        form.pack(fill='x', pady=(10, 0))
        tk.Label(form, text="Name", bg=T.SURFACE, fg=T.TEXT).pack(side='left')
        name_var = tk.StringVar()
        name_entry = tk.Entry(form, textvariable=name_var, width=24)
        name_entry.pack(side='left', padx=(6, 6))

        def write():
            if not st['writable']:
                say("Presets cannot be saved: the presets file could not be read or moved aside "
                    "(see the log).", error=True)
                return False
            try:
                output_presets.save(path, st['presets'])
            except Exception as e:  # noqa: BLE001
                log.exception("Output presets could not be saved")
                say(f"Could not save presets: {e}", error=True)
                return False
            return True

        def save_current():
            name = output_presets.clean_name(name_var.get())
            if name is None:
                say("Type a name for the preset.", error=True)
                return
            if name in st['presets'] and not messagebox.askyesno(
                    "Output Presets", f"Replace the preset “{name}” with the current output setup?", parent=win):
                return
            if name not in st['presets'] and len(st['presets']) >= output_presets.MAX_PRESETS:
                say(f"At most {output_presets.MAX_PRESETS} presets; delete one first.", error=True)
                return
            old = st['presets'].get(name)
            st['presets'][name] = self._current_output_preset()
            if not write():
                if old is None:
                    st['presets'].pop(name, None)
                else:
                    st['presets'][name] = old
                return
            refill(select=name)
            say(f"Saved “{name}”: {output_presets.summary(st['presets'][name])}")
            log.info(f"Output preset saved: {name}: {output_presets.summary(st['presets'][name])}")

        def apply():
            name = selected()
            if name is None:
                say("Select a preset to apply.", error=True)
                return
            preset = st['presets'][name]
            problems = output_presets.mismatches(preset, self.re.monitors if self.re else [])
            text = f"Apply “{name}” ({output_presets.summary(preset)})?\n\nThe output changes now."
            if problems:
                text += "\n\nThis preset may not fit this computer's displays:\n- " + "\n- ".join(problems)
            if not messagebox.askyesno("Output Presets", text, parent=win, icon='warning' if problems else 'question'):
                return
            ended = self._apply_output_preset(preset)
            say(f"Applied “{name}”." + (f" The output is {ended}." if ended != preset['mode'] else ""))

        def delete():
            name = selected()
            if name is None:
                say("Select a preset to delete.", error=True)
                return
            if not messagebox.askyesno("Output Presets", f"Delete the preset “{name}”?", parent=win):
                return
            old = st['presets'].pop(name)
            if not write():
                st['presets'][name] = old
                return
            refill()
            say(f"Deleted “{name}”.")

        def on_select(_e=None):
            name = selected()
            if name is not None:
                name_var.set(name)
                problems = output_presets.mismatches(st['presets'][name], self.re.monitors if self.re else [])
                say("Note: " + "; ".join(problems) if problems else "")

        lst.bind('<<ListboxSelect>>', on_select)
        lst.bind('<Double-Button-1>', lambda e: apply())
        ttk.Button(form, text="Save Current Setup", command=save_current).pack(side='left')
        btns = tk.Frame(body, bg=T.SURFACE)
        btns.pack(fill='x', pady=(14, 0))
        ttk.Button(btns, text="Delete", command=delete).pack(side='left')
        tk.Button(btns, text="Close", command=win.destroy, width=9).pack(side='right', padx=(6, 0))
        tk.Button(btns, text="Apply", command=apply, bg=T.GO, fg='black', width=9).pack(side='right')
        win.bind('<Escape>', lambda e: win.destroy())
        refill()
        if note:
            say(note, error=not writable)
        win._vd = {'st': st, 'list': lst, 'name_var': name_var, 'save': save_current, 'apply': apply,
                   'delete': delete, 'msg': msg, 'select': on_select}           # for tests
        return win

    # -- display changes (hot-plug) -----------------------------------------------------
    # Displays are listed when the render engine starts. If a projector is
    # unplugged or added later, the output may sit on the wrong screen. This
    # is detected and reported (Health turns WARN); nothing is moved by itself:
    # Output Health > Rescan Displays restarts the output with the new layout.
    DISPLAY_POLL_MS = 3000

    @staticmethod
    def _display_signature(monitors):
        return tuple((int(m.x), int(m.y), int(m.width), int(m.height)) for m in monitors or ())

    def _check_displays(self, current=None):
        """Compare the displays now with the engine's; returns the change or None."""
        if not self.re:
            return None
        if current is None:
            try:
                current = get_monitors()
            except Exception as e:  # noqa: BLE001
                log.debug(f"Display check failed: {e}")
                return getattr(self, '_display_change', None)
        then, now = self._display_signature(self.re.monitors), self._display_signature(current)
        # Only a different number of displays counts as a change: on Windows the
        # same displays can be reported in other units if the process's DPI
        # awareness changes, and a false WARN that never clears would be worse
        # than missing a resolution change (that is logged, at INFO).
        change = None if len(then) == len(now) else {'engine': list(then), 'now': list(now)}
        if change is None and then != now and now != getattr(self, '_display_geometry_seen', then):
            log.info(f"Display geometry differs from when the output started: {list(then)} -> {list(now)}")
        self._display_geometry_seen = now
        if change != getattr(self, '_display_change', None):
            if change:
                log.warning(f"Displays changed since the output started: {len(then)} -> {len(now)} "
                            f"({list(then)} -> {list(now)}). Output Health > Rescan Displays applies the new layout.")
            else:
                log.info("Displays are back to the layout the output started with")
        self._display_change = change
        return change

    def _display_poll_tick(self):
        try:
            self._check_displays()
        finally:
            self.root.after(self.DISPLAY_POLL_MS, self._display_poll_tick)

    def _rescan_displays(self, ask=True):
        """Restart the output with the displays as they are now (operator's choice)."""
        if ask and not messagebox.askyesno(
                "Rescan Displays", "Restart the output with the displays connected now? The output goes black "
                "briefly and videos start again.", parent=self.root):
            return False
        log.info("Rescanning displays (operator)")
        self._initialize_render_engine()
        self._display_change = None
        self._update_active_players()
        self._trigger_column()
        return True

    # -- corner pin ------------------------------------------------------------------
    def _set_output_warp(self, display, corners):
        """Pin display `display` to `corners` (None or identity: no pin). Returns
        False for a shape that would fold the picture (nothing changes)."""
        key = str(display)
        cleaned = output_warp.clean(corners) if corners is not None else None
        if corners is not None and cleaned is None and not output_warp.is_identity(
                tuple(tuple(map(float, p)) for p in corners)):
            return False
        if cleaned is None:
            changed = self.output_warps.pop(key, None) is not None
        else:
            new = [list(p) for p in cleaned]
            changed = self.output_warps.get(key) != new
            self.output_warps[key] = new
        if changed:
            self._set_dirty_flag()
            self.render_command_queue.put({'action': 'set_warps', 'warps': dict(self.output_warps)})
        return True

    def _apply_span(self, sp, blend=True):
        """Use span `sp` (span.clean) and switch the output to it; with `blend`,
        edge-blend the sides that meet. Returns False if it does not fit."""
        sp = span_mod.clean(sp, len(self.re.monitors) if self.re else 0)
        if sp is None or span_mod.layout(self.re.monitors, sp) is None:
            return False
        self.output_span = sp
        if blend:
            for key, b in span_mod.blends(sp, self.re.monitors, self.output_blends).items():
                self._set_output_blend(int(key), b)
        self._refresh_output_options()
        log.info(f"Span: displays {sp['displays']}, overlap {sp['overlap']} px")
        if self.screen_var.get() == self.SPAN:
            self._on_output_change()               # same mode, new span
        else:
            self.screen_var.set(self.SPAN)          # trace: _on_output_change
        self._set_dirty_flag()
        return True

    def _remove_span(self):
        """No span; an output on Span goes to the span's first display."""
        first = (self._span_displays() or [None])[0]
        self.output_span = None
        self._refresh_output_options()
        if self.screen_var.get() == self.SPAN:
            self.screen_var.set(f"Display {first}" if first is not None else self.NONE)
        self._send_regions()
        self._set_dirty_flag()

    def _open_span(self):
        """Window > Span Displays: one picture across displays side by side."""
        mons = self.re.monitors if self.re and getattr(self.re, 'monitors', None) else []
        if not mons:
            messagebox.showinfo("Span Displays", "No displays were found.", parent=self.root)
            return None
        cur = self._span_clean() or {'displays': [], 'overlap': 0}
        win = tk.Toplevel(self.root)
        win.title("Span Displays")
        win.configure(bg=T.SURFACE)
        win.transient(self.root)
        body = tk.Frame(win, bg=T.SURFACE, padx=15, pady=15)
        body.pack(fill='both', expand=True)
        tk.Label(body, bg=T.SURFACE, fg=T.TEXT, justify='left', anchor='w', wraplength=460,
                 text="One picture across displays side by side, e.g. two projectors with overlapping "
                      "edges. Layers are laid out on the whole picture; each display shows its part. "
                      "With parts set by hand, each display shows any rectangle of the picture "
                      "(also one display: a zoom or an LED processor's input).").pack(fill='x')
        order = sorted(range(len(mons)), key=lambda i: (mons[i].y, mons[i].x))     # row by row
        use = {}
        box = tk.Frame(body, bg=T.SURFACE)
        box.pack(fill='x', pady=(10, 0))
        for i in order:
            m = mons[i]
            use[i] = tk.BooleanVar(value=i in cur['displays'] or not cur['displays'])
            ttk.Checkbutton(box, text=f"Display {i}   {m.width} \u00d7 {m.height} at x {m.x}",
                            variable=use[i]).pack(anchor='w')
        row = tk.Frame(body, bg=T.SURFACE)
        row.pack(fill='x', pady=(10, 0))
        tk.Label(row, text="Overlap:", bg=T.SURFACE, fg=T.TEXT).pack(side='left')
        ov_var = tk.StringVar(value=str(cur['overlap']))
        ttk.Spinbox(row, from_=0, to=4000, increment=10, width=7, textvariable=ov_var).pack(side='left', padx=6)
        tk.Label(row, text="px where neighboring displays show the same strip", bg=T.SURFACE,
                 fg=T.MUTED).pack(side='left')
        grid_row = tk.Frame(body, bg=T.SURFACE)
        grid_row.pack(fill='x', pady=(6, 0))
        tk.Label(grid_row, text="Displays per row:", bg=T.SURFACE, fg=T.TEXT).pack(side='left')
        cols_var = tk.StringVar(value=str(cur.get('columns', '')) if cur.get('columns') else "all")
        ttk.Spinbox(grid_row, values=["all"] + [str(n) for n in range(1, len(mons) + 1)], width=5,
                    textvariable=cols_var, state='readonly').pack(side='left', padx=6)
        tk.Label(grid_row, text="Vertical overlap:", bg=T.SURFACE, fg=T.TEXT).pack(side='left', padx=(10, 0))
        ovy_var = tk.StringVar(value=str(cur.get('overlap_y', 0)))
        ttk.Spinbox(grid_row, from_=0, to=4000, increment=10, width=7, textvariable=ovy_var).pack(side='left', padx=6)
        tk.Label(grid_row, text="px (a grid: rows top to bottom)", bg=T.SURFACE, fg=T.MUTED).pack(side='left')
        blend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(body, text="Blend the overlaps (edge blend on the sides that meet)",
                        variable=blend_var).pack(anchor='w', pady=(8, 0))
        # Slices: the part of the picture each display shows, set by hand.
        slices_var = tk.BooleanVar(value=span_mod.has_slices(cur))
        slices_chk = ttk.Checkbutton(body, text="Set each display's part by hand (slices)", variable=slices_var)
        slices_chk.pack(anchor='w', pady=(8, 0))
        sl = tk.Frame(body, bg=T.SURFACE)
        tk.Label(sl, text="Picture", bg=T.SURFACE, fg=T.TEXT).grid(row=0, column=0, sticky='w')
        canvas_vars = [tk.StringVar(value=str(v)) for v in (cur.get('canvas') or ("", ""))]
        for k, (lbl, var) in enumerate((("width", canvas_vars[0]), ("height", canvas_vars[1]))):
            tk.Label(sl, text=lbl, bg=T.SURFACE, fg=T.MUTED).grid(row=0, column=1 + 2 * k, sticky='e', padx=(8, 2))
            ttk.Entry(sl, textvariable=var, width=7).grid(row=0, column=2 + 2 * k, sticky='w')
        part_vars = {}
        for r, i in enumerate(order, start=1):
            vals = (cur.get('slices') or {}).get(str(i)) or ("", "", "", "")
            part_vars[i] = [tk.StringVar(value=str(v)) for v in vals]
            tk.Label(sl, text=f"Display {i}", bg=T.SURFACE, fg=T.TEXT).grid(row=r, column=0, sticky='w', pady=(4, 0))
            for k, lbl in enumerate(("x", "y", "width", "height")):
                tk.Label(sl, text=lbl, bg=T.SURFACE, fg=T.MUTED).grid(row=r, column=1 + 2 * k, sticky='e',
                                                                     padx=(8, 2), pady=(4, 0))
                ttk.Entry(sl, textvariable=part_vars[i][k], width=7).grid(row=r, column=2 + 2 * k, sticky='w',
                                                                          pady=(4, 0))
        info = tk.Label(body, bg=T.SURFACE, fg=T.MUTED, justify='left', anchor='w', font=T.mono(9))
        info.pack(fill='x', pady=(10, 0))

        def row_layout():
            """The parts the row/grid settings give (to start the slices from)."""
            try:
                ov, ov_y = int(float(ov_var.get() or 0)), int(float(ovy_var.get() or 0))
            except ValueError:
                ov = ov_y = 0
            disp = [i for i in order if use[i].get()] or order
            cols = len(disp) if cols_var.get() in ("", "all") else int(cols_var.get())
            sp = span_mod.clean({'displays': disp, 'overlap': ov, 'columns': cols, 'overlap_y': ov_y}, len(mons))
            lay = span_mod.layout(mons, sp) if sp else None
            if lay is None:                          # side by side, no overlap
                xs = [sum(mons[j].width for j in order[:k]) for k in range(len(order))]
                return ([xs[-1] + mons[order[-1]].width, max(m.height for m in mons)],
                        {i: [x, 0, mons[i].width, mons[i].height] for i, x in zip(order, xs)})
            (cw, ch), parts = lay
            return [cw, ch], {i: [p['offset'], p['offset_y'], mons[i].width, mons[i].height]
                              for i, p in parts.items()}

        def show_slices(*_):
            if slices_var.get():
                if not canvas_vars[0].get().strip():
                    canvas, parts = row_layout()
                    for var, v in zip(canvas_vars, canvas):
                        var.set(str(v))
                    for i, vals in parts.items():
                        for var, v in zip(part_vars[i], vals):
                            var.set(str(v))
                sl.pack(fill='x', pady=(6, 0), after=slices_chk)
            else:
                sl.pack_forget()

        def chosen_slices(disp):
            if not disp:
                return None, "Choose at least one display."
            try:
                canvas = [int(v.get()) for v in canvas_vars]
                parts = {str(i): [int(v.get()) for v in part_vars[i]] for i in disp}
            except ValueError:
                return None, "Sizes and positions are whole numbers of pixels."
            sp = span_mod.clean({'displays': disp, 'overlap': 0, 'canvas': canvas, 'slices': parts}, len(mons))
            if sp is None:
                return None, (f"Each part must lie inside the picture (16 to {span_mod.MAX_CANVAS} px a side), "
                              "at least 1 px.")
            return sp, None

        def chosen():
            try:
                ov = int(float(ov_var.get() or 0))
                ov_y = int(float(ovy_var.get() or 0))
            except ValueError:
                return None, "The overlap is a number of pixels."
            disp = [i for i in order if use[i].get()]
            if slices_var.get():
                return chosen_slices(disp)
            cols = len(disp) if cols_var.get() in ("", "all") else int(cols_var.get())
            if len(disp) >= 2 and len(disp) % max(1, cols):
                return None, f"{len(disp)} displays do not fill rows of {cols}."
            sp = span_mod.clean({'displays': disp, 'overlap': ov, 'columns': cols, 'overlap_y': ov_y}, len(mons))
            if sp is None:
                return None, "Choose two or more displays (or set the parts by hand)."
            if span_mod.layout(mons, sp) is None:
                return None, "The overlap is more than half of a display."
            return sp, None

        def refresh(*_):
            sp, err = chosen()
            info.config(fg=T.WARN if err else T.MUTED, text=err or "\n".join(span_mod.describe(mons, sp)))

        for v in list(use.values()) + [ov_var, ovy_var, cols_var] + canvas_vars + \
                [v for vs in part_vars.values() for v in vs]:
            v.trace_add('write', refresh)
        slices_var.trace_add('write', lambda *_: (show_slices(), refresh()))
        show_slices()

        def apply():
            sp, err = chosen()
            if sp is None:
                refresh()
                return False
            ok = self._apply_span(sp, blend=blend_var.get())
            info.config(fg=T.MUTED, text="\n".join(span_mod.describe(mons, sp)) +
                        "\nOutput: Span. Corner pins, meshes and blends still apply per display.")
            return ok

        def remove():
            self._remove_span()
            info.config(fg=T.MUTED, text="No span. Edge blends set for it stay (Corner Pin, Mesh and Edge Blend).")

        btns = tk.Frame(body, bg=T.SURFACE)
        btns.pack(fill='x', pady=(14, 0))
        tk.Button(btns, text="Close", width=9, command=win.destroy).pack(side='right')
        tk.Button(btns, text="Apply", width=9, bg=T.GO, fg='black', command=apply).pack(side='right', padx=(0, 6))
        ttk.Button(btns, text="Remove Span", command=remove).pack(side='left')
        win.bind('<Escape>', lambda e: win.destroy())
        refresh()
        win._vd = {'use': use, 'ov_var': ov_var, 'blend_var': blend_var, 'apply': apply, 'remove': remove,
                   'info': info, 'cols_var': cols_var, 'ovy_var': ovy_var, 'slices_var': slices_var,
                   'canvas_vars': canvas_vars, 'part_vars': part_vars}   # for tests
        return win

    def _open_output_effects(self, display=None):
        """Window > Output Effects...: a shader effect (isf.py) on everything a
        display shows, after its layers and before its corner pin, mesh and
        edge blend. Changes reach the output at once and are saved with the
        show; only Performance Mode draws them."""
        n = len(self.re.monitors) if self.re and getattr(self.re, 'monitors', None) else 0
        if n == 0:
            messagebox.showinfo("Output Effects", "No displays were found.", parent=self.root)
            return None
        old = getattr(self, '_output_effects_win', None)
        if old is not None:
            try:
                if old.winfo_exists():
                    old.destroy()
            except tk.TclError:
                pass
        if display is None:
            display = self.selected_display if self.selected_display is not None else 0
        display = min(max(0, int(display)), n - 1)
        from effect_ui import EffectStack
        win = tk.Toplevel(self.root)
        win.title("Output Effects")
        win.configure(bg=T.SURFACE)
        win.transient(self.root)
        self._output_effects_win = win
        body = tk.Frame(win, bg=T.SURFACE, padx=14, pady=14)
        body.pack(fill='both', expand=True)
        tk.Label(body, bg=T.SURFACE, fg=T.MUTED, justify='left', anchor='w', wraplength=340, text=(
            "Shader effects on everything the display shows, in order: after its layers, before its corner pin, "
            "mesh and edge blend. Changes reach the output at once (also with blind editing) and are "
            "saved with the show. Performance Mode only.")).pack(fill='x', pady=(0, 10))
        top = tk.Frame(body, bg=T.SURFACE)
        top.pack(fill='x', pady=(0, 8))
        tk.Label(top, text="Display:", bg=T.SURFACE, fg=T.TEXT).pack(side='left')
        disp_var = tk.StringVar(value=f"Display {display}")
        disp_box = ttk.Combobox(top, textvariable=disp_var, state='readonly', width=12,
                                values=[f"Display {i}" for i in range(n)])
        disp_box.pack(side='left', padx=(6, 0))
        holder = tk.Frame(body, bg=T.SURFACE)
        holder.pack(fill='x')
        st = {'display': display, 'controls': None}

        def show(idx):
            for w in holder.winfo_children():
                w.destroy()
            st['display'] = idx
            st['controls'] = EffectStack(holder, self.output_effects.get(str(idx)),
                                         lambda v, i=idx: self._set_output_effect(i, v), self.re,
                                         bg=T.SURFACE, wraplength=340, preview_hint=False)
            st['controls'].frame.pack(fill='x')

        disp_box.bind('<<ComboboxSelected>>', lambda e: show(int(disp_var.get().split()[-1])))
        bar = tk.Frame(body, bg=T.SURFACE)
        bar.pack(fill='x', pady=(12, 0))
        ttk.Button(bar, text="Close", command=win.destroy).pack(side='right')
        win.bind('<Escape>', lambda e: win.destroy())
        show(display)
        win._vd = {'show': show, 'state': st, 'display_var': disp_var}      # for tests
        return win

    def _set_output_effect(self, display, value):
        """Shader effects for display `display`: a list of {'name', 'inputs'}
        (or one; None or [] for none)."""
        key = str(display)
        new = isf.stack(value) or None
        if self.output_effects.get(key) == new:
            return
        if new is None:
            self.output_effects.pop(key, None)
        else:
            self.output_effects[key] = new
        self._set_dirty_flag()
        self.render_command_queue.put({'action': 'set_output_effects', 'effects': dict(self.output_effects)})

    def _open_corner_pin(self, display=None):
        """Drag the four corners of a display's picture (keystone / corner pin).
        Changes reach the output at once and are saved with the show."""
        n = len(self.re.monitors) if self.re and getattr(self.re, 'monitors', None) else 0
        if n == 0:
            messagebox.showinfo("Corner Pin", "No displays were found.", parent=self.root)
            return None
        if display is None:
            display = self.selected_display if self.selected_display is not None else 0
        win = tk.Toplevel(self.root)
        win.title("Corner Pin, Mesh and Edge Blend")
        win.configure(bg=T.SURFACE)
        win.transient(self.root)
        self._corner_pin_win = win
        body = tk.Frame(win, bg=T.SURFACE, padx=12, pady=12)
        body.pack(fill='both', expand=True)
        top = tk.Frame(body, bg=T.SURFACE)
        top.pack(fill='x')
        tk.Label(top, text="Display:", bg=T.SURFACE, fg=T.TEXT).pack(side='left')
        disp_var = tk.StringVar(value=f"Display {display}")
        disp_box = ttk.Combobox(top, textvariable=disp_var, state='readonly', width=12,
                                values=[f"Display {i}" for i in range(n)])
        disp_box.pack(side='left', padx=(6, 0))
        size_lbl = tk.Label(top, bg=T.SURFACE, fg=T.MUTED)
        size_lbl.pack(side='left', padx=(10, 0))
        M, CW = 50, 480                                       # canvas margin and output width in canvas px
        canvas = tk.Canvas(body, bg=T.INPUT, highlightthickness=0)
        canvas.pack(pady=(10, 6))
        readout = tk.Label(body, bg=T.SURFACE, fg=T.TEXT, justify='left', anchor='w', font=T.mono(9))
        readout.pack(fill='x')
        # Edge blend: soft edges where this projector overlaps another one.
        blend_box = tk.Frame(body, bg=T.SURFACE)
        blend_box.pack(fill='x', pady=(10, 0))
        tk.Label(blend_box, text="Edge blend", bg=T.SURFACE, fg=T.TEXT, font=T.font(10, 'bold')).grid(
            row=0, column=0, columnspan=8, sticky='w')
        blend_vars = {}
        for i, side in enumerate(edge_blend.SIDES):
            tk.Label(blend_box, text=side.capitalize(), bg=T.SURFACE, fg=T.TEXT).grid(row=1, column=2 * i, sticky='w',
                                                                                     padx=(0 if i == 0 else 10, 4))
            blend_vars[side] = tk.StringVar(value="0")
            ttk.Spinbox(blend_box, from_=0, to=4000, increment=10, width=6, textvariable=blend_vars[side]).grid(
                row=1, column=2 * i + 1, sticky='w')
        for i, (key, label, lo, hi, step) in enumerate((("gamma", "Gamma", 1.0, 3.0, 0.1),
                                                        ("curve", "Curve", 1.0, 4.0, 0.5))):
            tk.Label(blend_box, text=label, bg=T.SURFACE, fg=T.TEXT).grid(row=2, column=2 * i, sticky='w',
                                                                          padx=(0 if i == 0 else 10, 4), pady=(4, 0))
            blend_vars[key] = tk.StringVar(value=f"{edge_blend.DEFAULT_GAMMA if key == 'gamma' else edge_blend.DEFAULT_CURVE:g}")
            ttk.Spinbox(blend_box, from_=lo, to=hi, increment=step, width=6, format="%.1f",
                        textvariable=blend_vars[key]).grid(row=2, column=2 * i + 1, sticky='w', pady=(4, 0))
        tk.Label(blend_box, text="px of overlap on each side; gamma of the projector", bg=T.SURFACE,
                 fg=T.MUTED).grid(row=2, column=4, columnspan=4, sticky='w', padx=(10, 0), pady=(4, 0))
        # Mesh warp: a grid of points that bends the picture inside the corner pin.
        mesh_box = tk.Frame(body, bg=T.SURFACE)
        mesh_box.pack(fill='x', pady=(10, 0))
        tk.Label(mesh_box, text="Mesh warp", bg=T.SURFACE, fg=T.TEXT, font=T.font(10, 'bold')).grid(
            row=0, column=0, columnspan=6, sticky='w')
        mesh_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(mesh_box, text="Edit mesh points", variable=mesh_on).grid(row=1, column=0, sticky='w')
        tk.Label(mesh_box, text="Columns", bg=T.SURFACE, fg=T.TEXT).grid(row=1, column=1, sticky='w', padx=(14, 4))
        cols_var = tk.StringVar(value="3")
        ttk.Spinbox(mesh_box, from_=mesh_warp.MIN_POINTS, to=mesh_warp.MAX_POINTS, width=4, state='readonly',
                    textvariable=cols_var).grid(row=1, column=2, sticky='w')
        tk.Label(mesh_box, text="Rows", bg=T.SURFACE, fg=T.TEXT).grid(row=1, column=3, sticky='w', padx=(10, 4))
        rows_var = tk.StringVar(value="3")
        ttk.Spinbox(mesh_box, from_=mesh_warp.MIN_POINTS, to=mesh_warp.MAX_POINTS, width=4, state='readonly',
                    textvariable=rows_var).grid(row=1, column=4, sticky='w')
        tk.Label(mesh_box, text="points; changing them keeps the shape", bg=T.SURFACE, fg=T.MUTED).grid(
            row=1, column=5, sticky='w', padx=(10, 0))
        note = tk.Label(body, bg=T.SURFACE, fg=T.MUTED, justify='left', anchor='w', wraplength=CW + 2 * M,
                        text="Drag a corner, or click one and use the arrow keys (Shift: 10 px). "
                             "Window \u203a Test Pattern \u203a Grid helps line up a projector.")
        note.pack(fill='x', pady=(6, 0))
        st = {'display': display, 'corners': None, 'sel': 0, 'drag': False,
              'mcols': 3, 'mrows': 3, 'mpts': mesh_warp.identity(3, 3), 'msel': 4}

        def mon():
            return self.re.monitors[st['display']]

        def ch():
            m = mon()
            return int(round(CW * m.height / m.width))

        def pin():
            """The corner pin being edited (None when there is none)."""
            return output_warp.clean(st['corners'])

        def to_out(x, y):
            """Mesh coordinates -> output coordinates (through the pin)."""
            c = pin()
            return output_warp.apply(output_warp.homography(c), x, y) if c is not None else (x, y)

        def from_out(x, y):
            c = pin()
            return output_warp.apply(output_warp.inverse(c), x, y) if c is not None else (x, y)

        def mesh_raw(pts=None):
            return {'cols': st['mcols'], 'rows': st['mrows'], 'points': [list(p) for p in (pts or st['mpts'])]}

        def load_mesh():
            m = mesh_warp.parse(self.output_meshes.get(str(st['display'])))
            if m is not None:
                st['mcols'], st['mrows'], st['mpts'] = m[0], m[1], [list(p) for p in m[2]]
            else:
                st['mpts'] = mesh_warp.identity(st['mcols'], st['mrows'])
            st['msel'] = min(st['msel'], len(st['mpts']) - 1)
            st['loading'] = True
            cols_var.set(str(st['mcols']))
            rows_var.set(str(st['mrows']))
            st['loading'] = False

        def load():
            c = self.output_warps.get(str(st['display']))
            st['corners'] = [list(p) for p in (c or output_warp.IDENTITY)]
            load_mesh()
            m = mon()
            size_lbl.config(text=f"{m.width} \u00d7 {m.height}")
            canvas.config(width=CW + 2 * M, height=ch() + 2 * M)
            b = self.output_blends.get(str(st['display'])) or {}
            st['loading'] = True
            for side in edge_blend.SIDES:
                span = m.width if side in ('left', 'right') else m.height
                blend_vars[side].set(str(int(round(float(b.get(side, 0.0)) * span))))
            blend_vars['gamma'].set(f"{float(b.get('gamma', edge_blend.DEFAULT_GAMMA)):g}")
            blend_vars['curve'].set(f"{float(b.get('curve', edge_blend.DEFAULT_CURVE)):g}")
            st['loading'] = False
            draw()

        def blend_changed(*_):
            if st.get('loading'):
                return
            m = mon()
            raw = {}
            try:
                for side in edge_blend.SIDES:
                    span = m.width if side in ('left', 'right') else m.height
                    raw[side] = max(0.0, float(blend_vars[side].get() or 0)) / span
                raw['gamma'] = float(blend_vars['gamma'].get())
                raw['curve'] = float(blend_vars['curve'].get())
            except ValueError:
                return                                    # half-typed number: wait for the rest
            self._set_output_blend(st['display'], raw)
            draw()

        for v in blend_vars.values():
            v.trace_add('write', blend_changed)

        def to_canvas(x, y):
            return M + x * CW, M + y * ch()

        def draw():
            canvas.delete('all')
            h = ch()
            canvas.create_rectangle(M, M, M + CW, M + h, outline=T.BORDER, dash=(4, 4))
            b = edge_blend.clean(self.output_blends.get(str(st['display'])))
            if b is not None:                 # blend zones, in output (projector) coordinates
                left, right, top, bottom = b[:4]
                for x0, y0, x1, y1 in ((0, 0, left, 1), (1 - right, 0, 1, 1), (0, 0, 1, top), (0, 1 - bottom, 1, 1)):
                    if x1 > x0 and y1 > y0:
                        canvas.create_rectangle(M + x0 * CW, M + y0 * h, M + x1 * CW, M + y1 * h, outline='',
                                                fill=T.WARN, stipple='gray25')
            c = st['corners']
            mesh = mesh_warp.parse(mesh_raw())
            editing_mesh = mesh_on.get()
            if mesh is not None and (editing_mesh or not mesh_warp.is_identity(*mesh)):
                ts = [k / 24 for k in range(25)]      # the mesh's rows and columns, through the pin
                cols, rows = mesh[0], mesh[1]
                lines = [(mesh_warp.surface(mesh, ts, (j / (rows - 1),))[0], j in (0, rows - 1)) for j in range(rows)]
                lines += [(mesh_warp.surface(mesh, (i / (cols - 1),), ts)[:, 0], i in (0, cols - 1))
                          for i in range(cols)]
            else:
                ts = [k / 8 for k in range(9)]        # the picture's grid, through the pin
                lines = [([(t, v) for v in ts], i in (0, 4)) for i, t in enumerate(k / 4 for k in range(5))]
                lines += [([(u, t) for u in ts], i in (0, 4)) for i, t in enumerate(k / 4 for k in range(5))]
            try:
                for pts, outer in lines:
                    xy = [to_canvas(*to_out(float(x), float(y))) for x, y in pts]
                    canvas.create_line(*[v for p in xy for v in p], fill=T.ACCENT if outer else T.MUTED)
            except Exception:  # noqa: BLE001 - a pin being dragged through a fold: no grid this time
                pass
            m = mon()
            if editing_mesh:
                for i, (x, y) in enumerate(st['mpts']):
                    cx, cy = to_canvas(*to_out(x, y))
                    r = 5
                    canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline='white', width=1,
                                       fill=T.WARN if i == st['msel'] else T.ACCENT, tags=(f"m{i}",))
                i = st['msel']
                x, y = to_out(*st['mpts'][i])
                readout.config(text=f"Mesh {st['mcols']} \u00d7 {st['mrows']} points\n"
                                    f"point column {i % st['mcols'] + 1}, row {i // st['mcols'] + 1}   "
                                    f"x {int(round(x * m.width)):>6} px   y {int(round(y * m.height)):>6} px")
                return
            for i, (x, y) in enumerate(c):
                cx, cy = to_canvas(x, y)
                r = 7
                canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline='white', width=2,
                                   fill=T.WARN if i == st['sel'] else T.ACCENT, tags=(f"c{i}",))
            readout.config(text="\n".join(
                f"{output_warp.CORNER_NAMES[i]:<13} x {int(round(x * m.width)):>6} px   y {int(round(y * m.height)):>6} px"
                for i, (x, y) in enumerate(c)))

        def commit(i, x, y):
            c = [list(p) for p in st['corners']]
            lim = output_warp.LIMIT
            c[i] = [min(1 + lim, max(-lim, x)), min(1 + lim, max(-lim, y))]
            if not output_warp.is_identity(tuple(map(tuple, c))) and output_warp.clean(c) is None:
                note.config(fg=T.WARN, text="That shape would fold the picture; the corner stays where it was.")
                return False
            mesh = mesh_warp.clean(self.output_meshes.get(str(st['display'])))
            if mesh is not None and not mesh_warp.through_pin_ok(mesh, output_warp.clean(c)):
                note.config(fg=T.WARN, text="That corner would fold the mesh; the corner stays where it was.")
                return False
            note.config(fg=T.MUTED, text="Changes are on the output now and are saved with the show.")
            st['corners'] = c
            self._set_output_warp(st['display'], c)
            draw()
            return True

        def commit_mesh(i, x, y):
            """Move mesh point i to (x, y) in mesh coordinates (inside the pin)."""
            pts = [list(p) for p in st['mpts']]
            lim = mesh_warp.LIMIT
            pts[i] = [min(1 + lim, max(-lim, x)), min(1 + lim, max(-lim, y))]
            raw = mesh_raw(pts)
            parsed = mesh_warp.parse(raw)
            if parsed is None or mesh_warp.folds(parsed) or not mesh_warp.through_pin_ok(parsed, pin()):
                note.config(fg=T.WARN, text="That would fold the picture over itself; the point stays where it was.")
                return False
            note.config(fg=T.MUTED, text="Changes are on the output now and are saved with the show.")
            st['mpts'] = pts
            self._set_output_mesh(st['display'], raw)
            draw()
            return True

        def grid_changed(*_):
            if st.get('loading'):
                return
            try:
                cols, rows = int(cols_var.get()), int(rows_var.get())
            except ValueError:
                return
            if (cols, rows) == (st['mcols'], st['mrows']) or not (
                    mesh_warp.MIN_POINTS <= min(cols, rows) and max(cols, rows) <= mesh_warp.MAX_POINTS):
                return
            current = mesh_warp.clean(mesh_raw())
            pts = mesh_warp.resize(current, cols, rows)
            st['mcols'], st['mrows'] = cols, rows
            if current is not None and self._set_output_mesh(st['display'], mesh_raw(pts)) is False:
                pts = mesh_warp.identity(cols, rows)        # the resampled shape folds: start flat
                self._set_output_mesh(st['display'], None)
                note.config(fg=T.WARN, text="The mesh could not keep its shape at that size; it is flat again.")
            st['mpts'] = pts
            st['msel'] = min(st['msel'], len(pts) - 1)
            draw()

        for v in (cols_var, rows_var):
            v.trace_add('write', grid_changed)

        def reset_mesh():
            st['mpts'] = mesh_warp.identity(st['mcols'], st['mrows'])
            self._set_output_mesh(st['display'], None)
            note.config(fg=T.MUTED, text="Mesh reset: the picture fills the corner pin again.")
            draw()

        def nearest(ev):
            if mesh_on.get():
                pts = [to_out(*p) for p in st['mpts']]
            else:
                pts = st['corners']
            d = [((to_canvas(*p)[0] - ev.x) ** 2 + (to_canvas(*p)[1] - ev.y) ** 2, i)
                 for i, p in enumerate(pts)]
            return min(d)

        def press(ev):
            dist, i = nearest(ev)
            if mesh_on.get():
                st['msel'] = i
            else:
                st['sel'] = i
            st['drag'] = dist <= 20 ** 2
            canvas.focus_set()
            draw()

        def motion(ev):
            if not st['drag']:
                return
            x, y = (ev.x - M) / CW, (ev.y - M) / ch()
            if mesh_on.get():
                commit_mesh(st['msel'], *from_out(x, y))
            else:
                commit(st['sel'], x, y)

        def release(_ev):
            st['drag'] = False

        def nudge(dx, dy, ev):
            step = 10 if ev.state & 0x1 else 1
            m = mon()
            if mesh_on.get():                     # moves the point by output pixels, through the pin
                x, y = to_out(*st['mpts'][st['msel']])
                commit_mesh(st['msel'], *from_out(x + dx * step / m.width, y + dy * step / m.height))
                return "break"
            x, y = st['corners'][st['sel']]
            commit(st['sel'], x + dx * step / m.width, y + dy * step / m.height)
            return "break"

        def reset():
            st['corners'] = [list(p) for p in output_warp.IDENTITY]
            self._set_output_warp(st['display'], None)
            note.config(fg=T.MUTED, text="Reset: the display shows the picture unwarped.")
            draw()

        canvas.bind('<ButtonPress-1>', press)
        canvas.bind('<B1-Motion>', motion)
        canvas.bind('<ButtonRelease-1>', release)
        for key, d in (('Left', (-1, 0)), ('Right', (1, 0)), ('Up', (0, -1)), ('Down', (0, 1))):
            canvas.bind(f'<{key}>', lambda e, d=d: nudge(d[0], d[1], e))
            canvas.bind(f'<Shift-{key}>', lambda e, d=d: nudge(d[0], d[1], e))
        for i in range(4):
            canvas.bind(f'{i + 1}', lambda e, i=i: (st.update(sel=i), draw()))
        disp_box.bind('<<ComboboxSelected>>', lambda e: (st.update(display=int(disp_var.get().split()[-1]), sel=0),
                                                         load()))
        btns = tk.Frame(body, bg=T.SURFACE)
        btns.pack(fill='x', pady=(10, 0))
        tk.Button(btns, text="Close", width=9, command=win.destroy).pack(side='right')
        ttk.Button(btns, text="Reset Corners", command=reset).pack(side="right", padx=(0, 6))
        ttk.Button(btns, text="Reset Mesh", command=reset_mesh).pack(side="right", padx=(0, 6))
        mesh_on.trace_add('write', lambda *_: (note.config(
            fg=T.MUTED, text="Drag a mesh point, or click one and use the arrow keys (Shift: 10 px)."
            if mesh_on.get() else "Drag a corner, or click one and use the arrow keys (Shift: 10 px)."), draw()))
        win.bind('<Escape>', lambda e: win.destroy())
        load()
        canvas.focus_set()
        win._vd = {'state': st, 'commit': commit, 'reset': reset, 'press': press, 'motion': motion,
                   'blend_vars': blend_vars, 'mesh_on': mesh_on, 'cols_var': cols_var, 'rows_var': rows_var,
                   'commit_mesh': commit_mesh, 'reset_mesh': reset_mesh, 'to_out': to_out,
                   'nudge': nudge, 'canvas': canvas, 'note': note, 'readout': readout, 'M': M, 'CW': CW,
                   'ch': ch}                              # for tests
        return win

    @staticmethod
    def _blend_dict(b):
        return dict(zip(edge_blend.SIDES + ('curve', 'gamma'), b))

    def _set_output_blend(self, display, raw):
        """Edge blend for display `display` from a dict (None or all widths 0: off)."""
        key = str(display)
        b = edge_blend.clean(raw) if raw else None
        new = self._blend_dict(b) if b is not None else None
        if self.output_blends.get(key) == new:
            return
        if new is None:
            self.output_blends.pop(key, None)
        else:
            self.output_blends[key] = new
        self._set_dirty_flag()
        self.render_command_queue.put({'action': 'set_blends', 'blends': dict(self.output_blends)})

    def _set_output_mesh(self, display, raw):
        """Mesh warp for display `display` from {'cols', 'rows', 'points'}
        (None or a regular grid: no mesh). Returns False for a shape that
        would fold the picture (nothing changes)."""
        key = str(display)
        parsed = mesh_warp.parse(raw) if raw is not None else None
        if raw is not None and parsed is None:
            return False
        m = mesh_warp.clean(raw) if parsed is not None else None
        if parsed is not None and m is None and not mesh_warp.is_identity(*parsed):
            return False                                  # folded
        new = mesh_warp.as_dict(m) if m is not None else None
        if self.output_meshes.get(key) == new:
            return True
        if new is None:
            self.output_meshes.pop(key, None)
        else:
            self.output_meshes[key] = new
        self._set_dirty_flag()
        self.render_command_queue.put({'action': 'set_meshes', 'meshes': dict(self.output_meshes)})
        return True

    # -- DMX (Art-Net / sACN) ------------------------------------------------------------
    def _restart_dmx(self):
        self._stop_dmx()
        if not self.dmx_enabled:
            return
        q = self._remote_queue
        self._dmx = dmx_remote.DMXRemote(lambda cmd, arg: q.put((cmd, arg, ('DMX', 0))), self.dmx_protocol,
                                         self.dmx_universe, self.dmx_channel, self.dmx_network)
        self._dmx.start()      # on failure .error is set; the Health window reports it

    def _stop_dmx(self):
        if self._dmx is not None:
            self._dmx.stop()
            self._dmx = None

    def _dmx_status(self):
        d = self._dmx
        if not self.dmx_enabled:
            return {'enabled': False}
        if d is None:
            return {'enabled': True, 'listening': False, 'error': 'not started'}
        st = {'enabled': True, 'listening': d.listening, 'packets': d.packets, 'commands': d.commands,
              'protocol': d.protocol, 'universe': d.universe, 'channel': d.decoder.channel,
              'values': list(d.decoder.values) if d.decoder.values else None}
        if d.listening:
            st['port'] = d.port
        if d.error:
            st['error'] = d.error
        return st

    @staticmethod
    def _dmx_text(d):
        if not d.get('enabled'):
            return "DMX off (Preferences)"
        if not d.get('listening'):
            return f"DMX not listening: {d.get('error', '?')}"
        name = 'Art-Net' if d.get('protocol') == 'artnet' else 'sACN'
        vals = d.get('values')
        return (f"DMX {name} universe {d.get('universe')} ch {d.get('channel')}-{d.get('channel', 1) + 2}: "
                + (f"{vals[0]} / {vals[1]} / {vals[2]}" if vals else "no data yet")
                + f"   {d.get('commands', 0)} commands")

    def _open_dmx_setup(self, parent=None, status_label=None):
        """Protocol, universe and start channel; live view of the three channels."""
        parent = parent or self.root
        win = tk.Toplevel(parent)
        win.title("DMX Setup")
        win.configure(bg=T.SURFACE)
        win.transient(parent)
        win.grab_set()
        self._dmx_win = win
        body = tk.Frame(win, bg=T.SURFACE, padx=15, pady=15)
        body.pack(fill='both', expand=True)
        enabled_var = tk.BooleanVar(value=self.dmx_enabled)
        proto_var = tk.StringVar(value='Art-Net' if self.dmx_protocol == 'artnet' else 'sACN (E1.31)')
        uni_var = tk.StringVar(value=str(self.dmx_universe))
        ch_var = tk.StringVar(value=str(self.dmx_channel))
        net_var = tk.BooleanVar(value=self.dmx_network)
        ttk.Checkbutton(body, text="Listen for DMX from a lighting console", variable=enabled_var).grid(
            row=0, column=0, columnspan=4, sticky='w')
        tk.Label(body, text="Protocol", bg=T.SURFACE, fg=T.TEXT).grid(row=1, column=0, sticky='w', pady=(8, 2))
        ttk.Combobox(body, textvariable=proto_var, state='readonly', width=14,
                     values=['Art-Net', 'sACN (E1.31)']).grid(row=1, column=1, sticky='w', pady=(8, 2))
        tk.Label(body, text="Universe", bg=T.SURFACE, fg=T.TEXT).grid(row=2, column=0, sticky='w', pady=2)
        ttk.Spinbox(body, from_=0, to=63999, width=8, textvariable=uni_var).grid(row=2, column=1, sticky='w')
        tk.Label(body, text="Art-Net: 0 = first; sACN: 1 = first", bg=T.SURFACE, fg=T.MUTED).grid(
            row=2, column=2, columnspan=2, sticky='w', padx=(8, 0))
        tk.Label(body, text="Start channel", bg=T.SURFACE, fg=T.TEXT).grid(row=3, column=0, sticky='w', pady=2)
        ttk.Spinbox(body, from_=1, to=510, width=8, textvariable=ch_var).grid(row=3, column=1, sticky='w')
        ttk.Checkbutton(body, text="Listen on the network (needed for a console on another computer)",
                        variable=net_var).grid(row=4, column=0, columnspan=4, sticky='w', pady=(6, 2))
        tk.Label(body, justify='left', anchor='w', bg=T.SURFACE, fg=T.MUTED, wraplength=440, text=(
            "Patch a 3-channel fixture on the console at the start channel:\n"
            "  1: cue select (value n goes to cue n; 0 does nothing)\n"
            "  2: GO (fires when it rises to 128 or more)\n"
            "  3: Blackout (fires when it rises to 128 or more)\n"
            "The values already being sent when VisualDeck starts listening do not trigger anything.")).grid(
            row=5, column=0, columnspan=4, sticky='w', pady=(8, 0))
        live = tk.Label(body, bg=T.SURFACE, fg=T.TEXT, anchor='w', font=T.mono(9))
        live.grid(row=6, column=0, columnspan=4, sticky='w', pady=(8, 0))
        err = tk.Label(body, bg=T.SURFACE, fg='#ff8888', anchor='w')
        err.grid(row=7, column=0, columnspan=4, sticky='w')
        # Sending: the live cue, for a lighting console to follow
        out_var = tk.BooleanVar(value=self.dmx_out_enabled)
        ttk.Checkbutton(body, text="Also send the live cue (channel: cue number, next channel: 255 while "
                                   "something is on the output)", variable=out_var).grid(
            row=8, column=0, columnspan=4, sticky='w', pady=(12, 2))
        out_row = tk.Frame(body, bg=T.SURFACE)
        out_row.grid(row=9, column=0, columnspan=4, sticky='w')
        out_uni = tk.StringVar(value=str(self.dmx_out_universe))
        out_ch = tk.StringVar(value=str(self.dmx_out_channel))
        out_to = tk.StringVar(value=self.dmx_out_target)
        for label, var, w in (("Universe", out_uni, 7), ("Channel", out_ch, 5), ("Art-Net to", out_to, 16)):
            tk.Label(out_row, text=label, bg=T.SURFACE, fg=T.TEXT).pack(side='left', padx=(0, 4))
            ttk.Entry(out_row, textvariable=var, width=w).pack(side='left', padx=(0, 12))
        tk.Label(body, bg=T.SURFACE, fg=T.MUTED, anchor='w', text="Sent with the protocol above; sACN goes to "
                 "its multicast group. 255.255.255.255 reaches every computer on this network.").grid(
            row=10, column=0, columnspan=4, sticky='w')

        def refresh():
            if not win.winfo_exists():
                return
            live.config(text=self._dmx_text(self._dmx_status()))
            win.after(250, refresh)

        def values():
            u, ch = int(uni_var.get()), int(ch_var.get())
            proto = 'artnet' if proto_var.get() == 'Art-Net' else 'sacn'
            if not (0 <= u <= 32767 if proto == 'artnet' else 1 <= u <= 63999):
                raise ValueError("Universe: 0-32767 for Art-Net, 1-63999 for sACN")
            if not 1 <= ch <= 512 - dmx_remote.FOOTPRINT + 1:
                raise ValueError("Start channel: 1-510")
            return proto, u, ch

        def save():
            try:
                proto, u, ch = values()
            except ValueError as e:
                err.config(text=str(e) if 'Universe' in str(e) or 'channel' in str(e) else "Enter whole numbers")
                return
            try:
                ou, oc = int(out_uni.get()), int(out_ch.get())
                if not (0 <= ou <= 63999 and 1 <= oc <= 511):
                    raise ValueError
            except ValueError:
                err.config(text="Sending: universe 0-63999, channel 1-511")
                return
            self.dmx_enabled, self.dmx_protocol, self.dmx_universe = enabled_var.get(), proto, u
            self.dmx_channel, self.dmx_network = ch, net_var.get()
            self.dmx_out_enabled, self.dmx_out_protocol = out_var.get(), proto
            self.dmx_out_universe, self.dmx_out_channel = ou, oc
            self.dmx_out_target = out_to.get().strip() or '255.255.255.255'
            self._save_config()
            self._restart_dmx()
            self._restart_dmx_out()
            log.info(f"DMX settings saved: enabled={self.dmx_enabled} {proto} universe {u} channel {ch} "
                     f"network={self.dmx_network}")
            if status_label is not None:
                try:
                    status_label.config(text=self._dmx_text(self._dmx_status()).replace(" (Preferences)", ""))
                except tk.TclError:
                    pass
            win.destroy()

        btns = tk.Frame(body, bg=T.SURFACE)
        btns.grid(row=11, column=0, columnspan=4, sticky='e', pady=(14, 0))
        tk.Button(btns, text="Save", command=save, bg=T.GO, fg='black', width=9).pack(side='right', padx=(6, 0))
        tk.Button(btns, text="Cancel", command=win.destroy, width=9).pack(side='right')
        win.bind('<Escape>', lambda e: win.destroy())
        refresh()
        win._vd = {'save': save, 'enabled_var': enabled_var, 'proto_var': proto_var, 'uni_var': uni_var,
                   'ch_var': ch_var, 'net_var': net_var, 'live': live, 'err': err, 'out_var': out_var,
                   'out_uni': out_uni, 'out_ch': out_ch, 'out_to': out_to}   # for tests
        return win

    def _midi_status(self):
        m = self._midi
        if not self.midi_enabled:
            return {'enabled': False}
        if m is None:
            return {'enabled': True, 'listening': False, 'error': 'not started'}
        st = {'enabled': True, 'listening': m.listening, 'received': m.received}
        if m.listening:
            st['port'] = m.opened_port
        tc = getattr(m, 'timecode', None)
        if tc is not None and time.monotonic() - tc[5] < 2.0:
            st['timecode'] = f"{timecode.fmt(tc)} ({timecode.RATE_LABELS.get(tc[4], '?')} fps)"
        if getattr(m, 'msc', False):
            last = getattr(m, 'msc_last', None)
            st['msc'] = f"device {m.msc_device}" + (
                f", last {last['command']}" + (f" {last['cue']}" if last.get('cue') else "") if last else "")
        if m.error:
            st['error'] = m.error
        return st

    @staticmethod
    def _displays_text(d):
        ch = d.get('changed')
        if not ch:
            return f"{d.get('count', 0)} as when the output started"
        return (f"CHANGED: {len(ch['engine'])} when the output started, {len(ch['now'])} now "
                f"\u2013 Rescan Displays to use them")

    @staticmethod
    def _midi_text(m):
        if not m.get('enabled'):
            return "MIDI off (Preferences)"
        if m.get('listening'):
            tc = m.get('timecode')
            return (f"MIDI listening on '{m.get('port')}'   {m.get('received', 0)} commands"
                    + (f"   MTC {tc}" if tc else "") + (f"   MSC {m['msc']}" if m.get('msc') else ""))
        return f"MIDI not listening: {m.get('error', '?')}"

    def _web_remote_status(self):
        w = self._web
        if not self.web_enabled:
            return {'enabled': False}
        if w is None:
            return {'enabled': True, 'listening': False, 'error': 'not started'}
        st = {'enabled': True, 'listening': w.listening, 'requests': w.requests}
        if w.listening:
            st['address'] = '%s:%d' % w.address()
        if w.error:
            st['error'] = w.error
        return st

    def _build_web_status(self):
        """Snapshot for the browser remote (UI thread; the server only reads it)."""
        prog = self._program
        black = self._output_is_black()
        live = None
        if not black and (prog is None or prog['col'] is not None):
            c = self.active_col
            live = {'number': c + 1, 'name': self.cue_names.get(str(c), f'Cue {c+1}'),
                    'cue_number': self._cue_number(c)}
        follow = None
        if self._follow_due:
            t, _frm, to = self._follow_due
            follow = {'to': self.cue_names.get(str(to), f'Cue {to+1}'), 'seconds': max(0.0, t - time.monotonic())}
        cues = []
        for c in range(self.COLS):
            cues.append({'number': c + 1, 'name': self.cue_names.get(str(c), f'Cue {c+1}'),
                         'cue_number': self._cue_number(c),
                         'has_media': any(self.grid_cells.get((r, c)) for r in range(self.ROWS)),
                         'notes': self.cue_settings.get(str(c), {}).get('notes')})
        return {'live': live, 'black': black, 'transition': bool(self.transition_active),
                'edits_pending': self._live_edits_pending(), 'test_pattern': self._test_pattern is not None,
                'follow': follow, 'cues': cues, 'clock': self._clock_text(self._cue_clock()),
                'effects': self._web_effects()}

    def _web_effects(self):
        """The effects on the output with their slider, list and on/off inputs,
        for the browser remote's controls: [{'kind': 'layer' | 'display', 'index'
        (layer from 1, display from 0), 'label', 'position' (in its stack, from
        1), 'title', 'inputs': [{'name', 'label', 'type', 'min', 'max',
        'values', 'labels', 'value'}]}]."""
        out = []
        prog = self._program
        groups = []
        if prog is not None and prog['col'] is not None:
            for r, layer in sorted(prog['layers'].items(), key=lambda kv: prog['rows'].get(kv[0], kv[0])):
                row = prog['rows'].get(r, r)
                if row is None:
                    continue
                name = self.layer_names.get(str(row), f'Layer {row + 1}')
                groups.append(('layer', row + 1, name, isf.layer_stack(layer)))
        for key in sorted(self.output_effects, key=lambda k: int(k) if str(k).isdigit() else 0):
            groups.append(('display', int(key), f'Display {key}', isf.stack(self.output_effects[key])))
        for kind, index, label, effects in groups:
            for pos, fx in enumerate(effects, start=1):
                inputs = []
                for inp in isf.inputs_of(fx['name']):
                    t = inp['TYPE']
                    if t not in ('float', 'long', 'bool'):
                        continue                       # a color or a point: the API takes them, the page does not
                    v = isf.clean_value(inp, fx['inputs'].get(inp['NAME'], inp['DEFAULT']))
                    d = {'name': inp['NAME'], 'label': str(inp.get('LABEL') or inp['NAME']), 'type': t, 'value': v}
                    if t == 'long' and inp.get('VALUES'):
                        d['values'] = list(inp['VALUES'])
                        labels = inp.get('LABELS') or inp['VALUES']
                        d['labels'] = [str(x) for x in labels] if len(labels) == len(inp['VALUES']) else \
                            [str(x) for x in inp['VALUES']]
                    elif t != 'bool':
                        lo, hi = isf._num(inp.get('MIN'), 0.0), isf._num(inp.get('MAX'), 1.0 if t == 'float' else 100)
                        d['min'], d['max'] = min(lo, hi), max(lo, hi)
                    inputs.append(d)
                out.append({'kind': kind, 'index': index, 'label': label, 'position': pos,
                            'title': isf.title(fx['name']), 'inputs': inputs})
        return out

    def _remote_status(self):
        r = self._remote
        if not self.osc_enabled:
            return {'enabled': False}
        if r is None:
            return {'enabled': True, 'listening': False, 'error': 'not started'}
        st = {'enabled': True, 'listening': r.listening, 'received': r.received}
        if r.listening:
            st['address'] = '%s:%d' % r.address()
        if r.error:
            st['error'] = r.error
        return st

    def _restart_dmx_out(self):
        if self._dmx_out is not None:
            self._dmx_out.stop()
            self._dmx_out = None
        if self.dmx_out_enabled:
            self._dmx_out = dmx_remote.DMXOutput(self.dmx_out_protocol, self.dmx_out_universe,
                                                 self.dmx_out_channel, self.dmx_out_target)
            log.info(f"DMX out: {self.dmx_out_protocol} universe {self.dmx_out_universe} channel "
                     f"{self.dmx_out_channel} to {self._dmx_out.address()[0]}")

    def _dmx_out_state(self):
        """(cue number for DMX, output level 0..1): the live cue's number (its
        Cue Settings number, else its position from 1), 0 when nothing is on
        the output."""
        col = self.active_col
        if col is None or self._output_is_black():
            return 0, 0.0
        num = self._cue_number(col)
        try:
            n = int(float(num)) if num not in (None, '') else col + 1
        except (TypeError, ValueError):
            n = col + 1
        return max(0, min(255, n)), 1.0

    def _dmx_out_tick(self, now=None):
        if self._dmx_out is None:
            return False
        cue, level = self._dmx_out_state()
        return self._dmx_out.update(cue, level, time.monotonic() if now is None else now)

    def _poll_remote(self):
        try:
            self._dmx_out_tick()
        except Exception:  # noqa: BLE001 - never stop remote control
            log.exception("DMX out failed")
        try:
            self._timecode_tick()
        except Exception:  # noqa: BLE001 - timecode must never stop remote control
            log.exception("Timecode check failed")
        try:
            self._audio_watchdog()
        except Exception:  # noqa: BLE001
            log.exception("Audio check failed")
        for _ in range(20):      # bounded per tick: a flood cannot stall the UI
            try:
                cmd, arg, addr = self._remote_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_remote_command(cmd, arg, addr)
        self._flush_remote_effects()

    def _cue_timecodes(self, fps):
        """[(seconds, col)] of cues with a Timecode (Cue Settings)."""
        out = []
        for key, cs in self.cue_settings.items():
            t = timecode.parse(cs.get('timecode'))
            if t is not None and 0 <= int(key) < self.COLS:
                out.append((timecode.seconds(t, fps), int(key)))
        return out

    def _timecode_source(self):
        """('ltc' or 'mtc', the newest time): LTC when Follow LTC is on, else MIDI."""
        if self.ltc_enabled:
            src = getattr(self, '_ltc', None)
            return 'ltc', (src.timecode if src is not None else None)
        m = getattr(self, '_midi', None)
        return 'mtc', (getattr(m, 'timecode', None) if m is not None else None)

    def _timecode_tick(self, tc=None):
        """Follow timecode (MIDI timecode or LTC): go to a cue when the running
        time reaches its Timecode (rules in timecode.py). `tc` for tests; else
        the newest time from the source being followed."""
        src = 'given'
        if tc is None:
            src, tc = self._timecode_source()
        if src != getattr(self, '_tc_src', None):        # another source: its first time is a start
            self._tc_src = src
            self._tc_prev_s = None
        pending = getattr(self, '_tc_pending', None)
        if pending is not None and not self.transition_active:
            self._tc_pending = None
            if 0 <= pending < self.COLS and pending != self.active_col:
                log.info(f"Timecode: cue {pending + 1} (after the running transition)")
                self._go_to_cue(pending)
        if tc is None or tc is getattr(self, '_tc_seen', None):
            return None
        self._tc_seen = tc
        fps = timecode.RATES.get(tc[4], 25)
        self._tc_fps = fps
        now_s = timecode.seconds(tc, fps)
        prev, self._tc_prev_s = getattr(self, '_tc_prev_s', None), now_s
        self._refresh_ready()
        following = {'ltc': self.ltc_enabled, 'mtc': self.midi_timecode}.get(
            src, self.midi_timecode or self.ltc_enabled)
        if not following or not self.re:
            return None
        live = None if self._output_is_black() else self.active_col
        col = timecode.cue_to_fire(self._cue_timecodes(fps), prev, now_s, live)
        if col is None:
            return None
        if self.transition_active:
            self._tc_pending = col              # fire when the running fade ends
            return col
        log.info(f"Timecode {timecode.fmt(tc)}: cue {col + 1}")
        if self.presentation_active:
            self._stop_presentation()
        self._go_to_cue(col)
        return col

    def _cue_number(self, col):
        return self.cue_settings.get(str(col), {}).get('number')

    def _cue_from_remote(self, arg):
        """Column for an OSC/MIDI cue argument: a cue number (Cue Settings)
        first, then a position (1 = first), then a cue name."""
        wanted = str(arg).strip().lower()
        for c in range(self.COLS):
            num = self._cue_number(c)
            if num is not None and num.lower() == wanted:
                return c
        if isinstance(arg, int):
            return arg - 1 if 1 <= arg <= self.COLS else None
        for c in range(self.COLS):
            if self.cue_names.get(str(c), f'Cue {c+1}').strip().lower() == wanted:
                return c
        return None

    def _handle_remote_command(self, cmd, arg, addr=None):
        """Run one OSC command on the UI thread (same code paths as the buttons)."""
        (log.debug if cmd in remote_control.EFFECT_COMMANDS + ('fader',) else log.info)(   # a fader sends many
            f"Remote {cmd}" + (f" {arg!r}" if arg is not None else "") + (f" from {addr[0]}" if addr else ""))
        if cmd == 'ping':
            if addr:
                try:
                    col = self.active_col
                    remote_control.send_reply(addr, '/visualdeck/pong',
                                              [col + 1, self.cue_names.get(str(col), f'Cue {col+1}')])
                except Exception as e:  # noqa: BLE001
                    log.warning(f"OSC: could not reply to {addr}: {e}")
            return
        if not self.re or not getattr(self, '_ui_ready', False):
            log.warning(f"OSC {cmd}: ignored, output not ready")
            return
        if cmd == 'go':
            self._trigger_go()
        elif cmd == 'back':
            self._go_back()
        elif cmd == 'blackout':
            self._blackout()
        elif cmd == 'update':
            self._apply_live_edits()
        elif cmd in ('cue', 'cue_at'):
            col = (arg - 1 if isinstance(arg, int) and 1 <= arg <= self.COLS else None) if cmd == 'cue_at' \
                else self._cue_from_remote(arg)
            if col is None:
                log.warning(f"Remote cue {arg!r}: no such cue")
                return
            self._go_to_cue(col)
        elif cmd in ('start_from', 'start_from_at'):
            cue, t = arg
            col = (cue - 1 if isinstance(cue, int) and 1 <= cue <= self.COLS else None) if cmd == 'start_from_at' \
                else self._cue_from_remote(cue)
            if col is None:
                log.warning(f"Remote start from cue {cue!r}: no such cue")
                return
            self._cut_to_column_at(col, t)
        elif cmd == 'effect':
            self._remote_layer_effect(arg[0] - 1, arg[1], arg[2])
        elif cmd == 'display_effect':
            self._remote_display_effect(arg[0], arg[1], arg[2])
        elif cmd == 'fader':
            self._midi_fader(*arg)

    def _midi_fader(self, i, v):
        """A MIDI fader (Control Change 0-127) set to an effect input (MIDI Setup)."""
        if not 0 <= i < len(self.midi_faders):
            return False
        f = self.midi_faders[i]
        if f['target'] == 'layer':
            prog = self._program
            row = f['index'] - 1
            layer = None if prog is None or prog['col'] is None else next(
                (l for r, l in prog['layers'].items() if prog['rows'].get(r, r) == row), None)
            effects = isf.layer_stack(layer)
        else:
            effects = isf.stack(self.output_effects.get(str(f['index'])))
        _i, _fx, inp = isf.pick(effects, f['input'])
        value = isf.fader_value(inp, v) if inp is not None else None
        if value is None:
            log.debug(f"MIDI fader {midi_remote.fader_text(f)}: no such input on the output")
            return False
        values = [value]
        if f['target'] == 'layer':
            return self._remote_layer_effect(f['index'] - 1, f['input'], values)
        return self._remote_display_effect(f['index'], f['input'], values)

    def _remote_layer_effect(self, row, input_name, values):
        """An input of an effect on layer `row` of the cue on the output, from a
        remote control (isf.pick: "amount", "2:amount", "Invert:amount"): the
        output changes at once (also with blind editing; it is playing the
        show, like a fader) and so does the cue's cell when its effects are the
        same. Sent once per poll however many arrive."""
        prog = self._program
        col = prog['col'] if prog is not None else None
        if col is None:
            log.warning("Remote effect: nothing is on the output")
            return False
        layer = next((l for r, l in prog['layers'].items() if prog['rows'].get(r, r) == row), None)
        effects = isf.layer_stack(layer)
        if not effects:
            log.warning(f"Remote effect: layer {row + 1} of the live cue has no effect")
            return False
        i, fx, inp = isf.pick(effects, input_name)
        v = isf.remote_value(inp, values) if inp is not None else None
        if v is None:
            log.warning(f"Remote effect: no effect on layer {row + 1} has an input {input_name!r} "
                        f"that takes {values!r}")
            return False
        effects[i] = {'name': fx['name'], 'inputs': dict(fx['inputs'], **{inp['NAME']: v})}
        for d in (layer, layer.get('position')):
            if isinstance(d, dict):
                d['effects'] = copy.deepcopy(effects)
                d.pop('effect', None)
        pos = self.cell_positions.get((row, col))
        cell = isf.layer_stack(pos)
        if [f['name'] for f in cell] == [f['name'] for f in effects]:
            cell[i] = {'name': fx['name'], 'inputs': dict(cell[i]['inputs'], **{inp['NAME']: v})}
            new_pos = {k: val for k, val in pos.items() if k != 'effect'}
            new_pos['effects'] = cell
            self.cell_positions[(row, col)] = new_pos
            self._set_dirty_flag()
        self._remote_fx_send = True
        return True

    def _remote_display_effect(self, display, input_name, values):
        """An input of an effect of display `display` from a remote control (saved with the show)."""
        effects = isf.stack(self.output_effects.get(str(display)))
        if not effects:
            log.warning(f"Remote effect: Display {display} has no effect")
            return False
        i, fx, inp = isf.pick(effects, input_name)
        v = isf.remote_value(inp, values) if inp is not None else None
        if v is None:
            log.warning(f"Remote effect: no effect of Display {display} has an input {input_name!r} "
                        f"that takes {values!r}")
            return False
        effects[i] = {'name': fx['name'], 'inputs': dict(fx['inputs'], **{inp['NAME']: v})}
        self.output_effects[str(display)] = effects
        self._set_dirty_flag()
        self._remote_out_fx_send = True
        return True

    def _flush_remote_effects(self):
        """Send what remote effect changes did, once per poll."""
        if getattr(self, '_remote_out_fx_send', False):
            self._remote_out_fx_send = False
            self.render_command_queue.put({'action': 'set_output_effects', 'effects': dict(self.output_effects)})
        if getattr(self, '_remote_fx_send', False):
            self._remote_fx_send = False
            prog = self._program
            if prog is None or prog['col'] is None or getattr(self, 'transition_active', False):
                return          # a fade owns the output; its end shows the cue as it now is
            self.render_command_queue.put(dict(self._output_routing(), action='render',
                                               layers=self._program_layers(), transition_state={'active': False}))
            self.set_preview_dirty()

    # -- test patterns ------------------------------------------------------------
    def _output_screens(self):
        if not self.re or not getattr(self.re, 'monitors', None):
            return []
        n = len(self.re.monitors)
        if self.screen_var.get() == self.MULTIPLE:
            used = sorted({s for s in self.layer_screens.values() if s is not None and 0 <= s < n})
            return used or list(range(n))
        if self.screen_var.get() == self.SPAN:
            return self._span_displays()
        if self.selected_display is not None and 0 <= self.selected_display < n:
            return [self.selected_display]
        return []

    def _show_test_pattern(self, kind):
        """Replace the output with a test pattern ('off' restores what was there)."""
        if kind == 'off':
            self._end_test_pattern()
            return
        if self.transition_active or not self.re:
            return
        screens = self._output_screens()
        if not screens:
            messagebox.showinfo("Test pattern", "Choose an output display first (Output: at the top of the preview).",
                                parent=self.root)
            self._test_pattern_menu_off()
            return
        if self._test_pattern is None and self._program_layers() and not messagebox.askyesno(
                "Test pattern", "This replaces what is on the output now. Window \u203a Test Pattern \u203a Off "
                "puts it back. Continue?", parent=self.root):
            self._test_pattern_menu_off()
            return
        out_dir = os.path.join(self.CONFIG_DIR, 'test_patterns')
        layers = {}
        groups = [(f"Displays {', '.join(map(str, screens))} (span)", screens, self._layout_size(screens[0]))] \
            if self.screen_var.get() == self.SPAN else \
            [(f"Display {sidx}", [sidx], (self.re.monitors[sidx].width, self.re.monitors[sidx].height))
             for sidx in screens]
        for i, (label, scr, (w, h)) in enumerate(groups):
            path = test_patterns.ensure_file(kind, w, h, label, out_dir)
            layers[i] = {'path': path, 'screens': list(scr), 'order': i, 'loop': True, 'opacity': 1.0,
                         'position': {'x': 0.5, 'y': 0.5, 'scale_x': 1.0, 'scale_y': 1.0, 'rotation': 0}}
        saved = self._test_pattern['saved'] if self._test_pattern is not None else copy.deepcopy(self._program)
        if self.presentation_active:
            self._stop_presentation()
        self._cancel_follow()
        log.info(f"Test pattern '{kind}' on display(s) {screens}")
        self._set_program(None, layers, fixed_screens=True)
        self._test_pattern = {'kind': kind, 'saved': saved}
        self.render_command_queue.put(dict(self._output_routing(), action='render', layers=self._program_layers(),
                                           transition_state={'active': False},
                                           all_assigned_screens=set(screens)))
        self._update_active_players()
        self._highlight_live_column()
        self._refresh_live_edit_ui()

    def _test_pattern_menu_off(self):
        if getattr(self, 'test_pattern_var', None) is not None:
            self.test_pattern_var.set('off')

    def _end_test_pattern(self):
        tp = self._test_pattern
        if tp is None:
            return
        self._test_pattern = None
        saved = tp['saved']
        log.info("Test pattern off; restoring the output")
        if saved is None:
            self._set_program(None, {})
        else:
            self._program = copy.deepcopy(saved)
        self.render_command_queue.put(dict(self._output_routing(), action='render', preroll=True,
                                           layers=self._program_layers(), transition_state={'active': False}))
        self._update_active_players()
        self._highlight_live_column()
        self._refresh_live_edit_ui()

    def _go_back(self):
        """BACK: go to the previous cue with media (not past the first cue)."""
        if self.transition_active or not self.re:
            return
        if self.presentation_active:
            self._stop_presentation()
        prev = self._prev_cue_col()
        if prev is None:
            log.info("BACK: already at the first cue")
            return
        self._initiate_transition(self.active_col, prev)

    def _blackout(self):
        """Fade (or cut) the output to black with the show's default transition.
        GO afterwards continues with the cue after the one that was live."""
        if self.transition_active or not self.re or self._output_is_black():
            return
        if self.presentation_active:
            self._stop_presentation()
        self._cancel_follow()
        log.info("Blackout")
        transition_type, duration = self._transition_for(None)
        from_layers = self._program_layers()
        if transition_type == self.NONE or duration <= 0 or not getattr(self, 'transition_thread', None):
            self._set_program(None, {})
            self.render_command_queue.put(dict(self._output_routing(), action='render', layers={},
                                               transition_state={'active': False}))
            self._update_active_players()
            self._highlight_live_column()
            self._refresh_live_edit_ui()
            return
        self.transition_from_layers = from_layers
        self.transition_to_layers = {}
        self._transition_activation = None
        self.transition_active = True
        self.transition_duration = duration
        self.transition_target_col = None
        self.transition_start_time = time.perf_counter()
        r = self._output_routing()
        self.transition_thread.start_transition(from_layers, {}, duration, None,
                                                {k: r[k] for k in ('output_mode', 'target_monitor',
                                                                   'all_assigned_screens', 'gui_monitor')},
                                                curve=self._curve_for(transition_type, from_layers, {}))

    # -- cue settings dialog --------------------------------------------------
    @_locked_out
    def _open_cue_settings(self, col):
        key = str(col)
        cs = dict(self.cue_settings.get(key, {}))
        win = tk.Toplevel(self.root)
        win.title(f"Cue Settings \u2013 {self.cue_names.get(key, f'Cue {col+1}')}")
        win.configure(bg=T.SURFACE, padx=12, pady=12)
        win.transient(self.root)
        win.resizable(False, False)
        lab = lambda t, r: Label(win, text=t, bg=T.SURFACE, fg='white', anchor='w').grid(row=r, column=0, sticky='w', pady=3)  # noqa: E731
        lab("Name", 0)
        name_var = tk.StringVar(value=self.cue_names.get(key, f'Cue {col+1}'))
        name_row = Frame(win, bg=T.SURFACE)
        name_row.grid(row=0, column=1, columnspan=2, sticky='w')
        tk.Entry(name_row, textvariable=name_var, width=22).pack(side='left')
        Label(name_row, text="Number", bg=T.SURFACE, fg='white').pack(side='left', padx=(10, 4))
        number_var = tk.StringVar(value=cs.get('number', ''))
        number_entry = tk.Entry(name_row, textvariable=number_var, width=7)
        number_entry.pack(side='left')
        CreateToolTip(number_entry, "Optional cue number such as 12 or 12.5, shown on the header and used by OSC "
                                    "and MIDI (/cue 12.5). Blank: none.")

        default_type, default_dur = self.transition_var.get(), self.transition_timing_var.get()
        lab("Transition", 1)
        tchoices = [f"Show default ({default_type}, {default_dur} s)", "None (cut)", "Fade", "Smooth Fade",
                    "Dip to Black"]
        tvar = tk.StringVar(value={None: tchoices[0], 'None': tchoices[1]}.get(cs.get('transition'),
                                                                               cs.get('transition') or tchoices[0]))
        ttk.Combobox(win, textvariable=tvar, values=tchoices, state='readonly', width=31).grid(
            row=1, column=1, columnspan=2, sticky='w')
        lab("Fade time (s)", 2)
        dvar = tk.StringVar(value='' if cs.get('duration') is None else f"{cs['duration']:g}")
        tk.Entry(win, textvariable=dvar, width=8).grid(row=2, column=1, sticky='w')
        Label(win, text="blank = show default", bg=T.SURFACE, fg='#aaa').grid(row=2, column=2, sticky='w')

        lab("Then", 3)
        fchoices = ["Wait for GO", "Continue to the next cue after\u2026", "Continue when its videos end"]
        f = cs.get('follow') or {}
        fvar = tk.StringVar(value=fchoices[0] if not f else (fchoices[1] if f['mode'] == 'after' else fchoices[2]))
        ttk.Combobox(win, textvariable=fvar, values=fchoices, state='readonly', width=31).grid(
            row=3, column=1, columnspan=2, sticky='w')
        lab("Seconds", 4)
        svar = tk.StringVar(value=f"{f.get('seconds', 5.0):g}")
        tk.Entry(win, textvariable=svar, width=8).grid(row=4, column=1, sticky='w')
        Label(win, text="for \u201cafter\u2026\u201d", bg=T.SURFACE, fg='#aaa').grid(row=4, column=2, sticky='w')

        lab("Timecode", 5)
        tc_var = tk.StringVar(value=cs.get('timecode', ''))
        tc_entry = tk.Entry(win, textvariable=tc_var, width=12)
        tc_entry.grid(row=5, column=1, sticky='w')
        Label(win, text="HH:MM:SS:FF, blank = none", bg=T.SURFACE, fg='#aaa').grid(row=5, column=2, sticky='w')
        CreateToolTip(tc_entry, "With MIDI Setup > Follow MIDI timecode or LTC Setup > Follow LTC on, VisualDeck "
                                "goes to this cue when the running timecode reaches this time.")

        ready_var = tk.BooleanVar(value=bool(cs.get('ready')))
        ready_chk = ttk.Checkbutton(win, text="Keep ready (videos loaded and paused, like the next cue)",
                                    variable=ready_var)
        ready_chk.grid(row=6, column=1, columnspan=2, sticky='w', pady=3)
        CreateToolTip(ready_chk, "For a cue you jump to out of order (a standby, an encore): going to it is "
                                 f"as quick as GO. Each such cue keeps a decoder busy; at most {self.MAX_READY} "
                                 "are kept ready.")
        lab("Notes", 7)
        notes = tk.Text(win, width=40, height=5, wrap='word')
        notes.insert('1.0', cs.get('notes', ''))
        notes.grid(row=7, column=1, columnspan=2, sticky='w', pady=3)
        err = Label(win, text='', bg=T.SURFACE, fg='#ff8888', anchor='w')
        err.grid(row=8, column=0, columnspan=3, sticky='w')

        def ok():
            try:
                values = self._cue_settings_from_form(tvar.get(), tchoices, dvar.get(), fvar.get(), fchoices,
                                                      svar.get(), notes.get('1.0', 'end'), number_var.get(),
                                                      tc_var.get(), ready_var.get())
                taken = self._cue_number_taken(values.get('number'), col)
                if taken is not None:
                    raise ValueError(f"Number {values['number']} is already used by "
                                     f"{self.cue_names.get(str(taken), f'Cue {taken + 1}')}")
            except ValueError as e:
                err.config(text=str(e))
                return
            self._apply_cue_settings(col, name_var.get().strip(), values)
            win.destroy()

        btns = Frame(win, bg=T.SURFACE)
        btns.grid(row=9, column=0, columnspan=3, sticky='e', pady=(8, 0))
        Button(btns, text="Cancel", command=win.destroy).pack(side='right')
        Button(btns, text="OK", width=8, command=ok).pack(side='right', padx=6)
        win.bind('<Escape>', lambda e: win.destroy())
        win._vd = {'ready_var': ready_var, 'ok': ok, 'err': err}           # for tests
        return win

    @staticmethod
    def _cue_settings_from_form(t, tchoices, d, fsel, fchoices, secs, notes, number='', tc_text='', ready=False):
        out = {}
        if ready:
            out['ready'] = True
        if tc_text.strip():
            tc = timecode.clean(tc_text)
            if tc is None:
                raise ValueError("Timecode as HH:MM:SS:FF, e.g. 01:00:12:00")
            out['timecode'] = tc
        if number.strip():
            num = project_schema.clean_cue_number(number)
            if num is None:
                raise ValueError("Cue number: up to 16 letters, digits, '.', '-' or '_'")
            out['number'] = num
        if t == tchoices[1]:
            out['transition'] = 'None'
        elif t in tchoices[2:]:
            out['transition'] = t                        # "Fade", "Smooth Fade" or "Dip to Black"
        if d.strip():
            try:
                dur = float(d)
            except ValueError:
                raise ValueError("Fade time must be a number of seconds") from None
            if not 0 <= dur <= 600:
                raise ValueError("Fade time must be between 0 and 600 seconds")
            out['duration'] = dur
        if fsel == fchoices[1]:
            try:
                sv = float(secs)
            except ValueError:
                raise ValueError("Seconds must be a number") from None
            if not 0 <= sv <= 86400:
                raise ValueError("Seconds must be between 0 and 86400")
            out['follow'] = {'mode': 'after', 'seconds': sv}
        elif fsel == fchoices[2]:
            out['follow'] = {'mode': 'end'}
        if notes.strip():
            out['notes'] = notes.strip()
        return out

    def _cue_number_taken(self, number, col):
        """Column of another cue that already has `number` (case-insensitive), or None."""
        if not number:
            return None
        for c in range(self.COLS):
            other = self._cue_number(c)
            if c != col and other is not None and other.lower() == number.lower():
                return c
        return None

    @_locked_out
    def _apply_cue_settings(self, col, name, values):
        key = str(col)
        if name:
            self.cue_names[key] = name
        if values:
            self.cue_settings[key] = values
        else:
            self.cue_settings.pop(key, None)
        self._highlight_live_column()
        self._update_cue_dropdown()
        self._set_dirty_flag()
        self._refresh_ready()                       # Keep Ready (or a timecode) may have changed

    def _select_column_immediate_cut(self, col):
        self._cut_to_column(col)

    def _cut_to_column(self, col):
        """Cut to `col`. The render thread keeps the current cue on screen until
        the new cue's videos have a frame (preroll), so a cut never shows black.
        Selecting the live cue again restarts its videos (new activation)."""
        col = max(0, min(col, self.COLS - 1))
        self.active_col = col
        self._live_activation = {col: self._claim_activation(col)}
        self._highlight_live_column()
        layers = self._get_layers_for_col(col)
        self._set_program(col, layers)
        self.render_command_queue.put(dict(self._output_routing(), action='render', preroll=True, layers=layers,
                                           transition_state={'active': False}))
        self._update_cue_dropdown()
        self.set_preview_dirty()
        self._refresh_live_edit_ui()
        self.root.after(100, self._update_active_players)
        self._schedule_follow(col)

    # -- Start From (rehearsal) -------------------------------------------------
    def _start_point(self, path, pos, t):
        """Where a video is `t` seconds after its cue went live, allowing for its
        speed, in and out points, direction and loop: {'start_at': seconds into
        the file, 'start_back': on the way back (ping-pong)}, or None (at its
        start, or its length is unknown). A video that does not loop and is
        shorter than `t` is at its end."""
        if t <= 0 or os.path.splitext(path)[1].lower() not in VIDEO_EXTS:
            return None
        try:
            dur = float(self.mm.get_media_info(path)[1] or 0)
            speed = max(0.1, float(pos.get('speed', 1.0)))
        except (TypeError, ValueError):
            return None
        a, b = layer_trim(pos)
        a = a or 0.0
        end = min(b, dur) if b and dur else (b or dur)
        if not end or end <= a:
            return None
        way = (end - a) / speed                     # cue seconds of one way through the clip
        direction = pos.get('direction', 'forward')
        cycle = way * (2 if direction == 'pingpong' else 1)
        u = t % cycle if pos.get('loop', True) else min(t, cycle - 1e-3)
        if direction == 'pingpong':
            if u < way:
                return {'start_at': a + u * speed, 'start_back': False}
            return {'start_at': end - (u - way) * speed, 'start_back': True}
        if direction == 'reverse':
            return {'start_at': end - u * speed}
        return {'start_at': a + u * speed}

    def _cut_to_column_at(self, col, t):
        """Start From: cut to `col` with its videos and their sound `t` seconds
        into the cue, as if it had been playing that long (rehearsals). The cue
        clock and its follow count from there. Only this start is affected:
        loops, later passes and the next GO are as usual."""
        col = max(0, min(col, self.COLS - 1))
        t = max(0.0, float(t))
        if self.transition_active or not self.re:
            log.info(f"Start From cue {col + 1}: ignored during a transition")
            return False
        if t <= 0:
            self._cut_to_column(col)
            return True
        self._reserved_activation.pop(col, None)    # its preloaded players wait at the start: not used
        self.active_col = col
        self._live_activation = {col: next(self._activation_seq)}
        self._highlight_live_column()
        layers = self._get_layers_for_col(col)
        self._set_program(col, layers)
        started = {}
        for k, layer in layers.items():
            layer = dict(layer)
            point = self._start_point(layer['path'], layer.get('position') or {}, t)
            if point:
                layer.update(point)
            started[k] = layer
        self.render_command_queue.put(dict(self._output_routing(), action='render', preroll=True, layers=started,
                                           transition_state={'active': False}))
        self._update_cue_dropdown()
        self.set_preview_dirty()
        self._refresh_live_edit_ui()
        self.root.after(100, self._update_active_players)
        self._schedule_follow(col, elapsed=t)
        log.info(f"Start From: cue {col + 1} at {t:.2f} s")
        return True

    def _open_start_from(self):
        """Show > Start Cue From...: a cue and a time into it."""
        cues = self._soak_cues()
        if not cues:
            messagebox.showinfo("Start Cue From", "The show has no cues with media.")
            return None
        win = tk.Toplevel(self.root)
        win.title("Start Cue From")
        win.configure(bg=T.BG)
        win.transient(self.root)
        names = [f"{c + 1}  {self.cue_names.get(str(c), f'Cue {c + 1}')}" for c in cues]
        start_col = self.active_col if self.active_col in cues else cues[0]
        cue_var = tk.StringVar(value=names[cues.index(start_col)])
        time_var = tk.StringVar(value="0:00")
        frm = tk.Frame(win, bg=T.BG)
        frm.pack(fill='both', expand=True, padx=12, pady=10)
        tk.Label(frm, text="Cue", bg=T.BG, fg=T.TEXT).grid(row=0, column=0, sticky='w', pady=3)
        box = ttk.Combobox(frm, textvariable=cue_var, values=names, state='readonly', width=32)
        box.grid(row=0, column=1, sticky='we', pady=3)
        tk.Label(frm, text="Time into the cue", bg=T.BG, fg=T.TEXT).grid(row=1, column=0, sticky='w', pady=3)
        entry = tk.Entry(frm, textvariable=time_var, width=12)
        entry.grid(row=1, column=1, sticky='w', pady=3)
        info = tk.Label(frm, text="", bg=T.BG, fg=T.MUTED, justify='left', wraplength=360)
        info.grid(row=2, column=0, columnspan=2, sticky='w', pady=(6, 0))
        status = tk.Label(frm, text="", bg=T.BG, fg=T.WARN, justify='left', wraplength=360)
        status.grid(row=3, column=0, columnspan=2, sticky='w')

        def chosen():
            return cues[names.index(cue_var.get())]

        def show_length(*_):
            length, loops = self._cue_video_length(chosen())
            if length > 0:
                m, sec = divmod(length, 60)
                info.config(text=f"Its longest video takes {int(m)}:{sec:05.2f}" + (" and loops." if loops else "."))
            else:
                info.config(text="No video in this cue: its stills show as usual.")
        box.bind('<<ComboboxSelected>>', show_length)
        show_length()

        def start(*_):
            try:
                t = MediaEditor._parse_s(time_var.get()) or 0.0
            except (ValueError, TypeError):
                status.config(text="The time is seconds (12.5) or minutes:seconds (1:02.5).")
                return False
            if not self._cut_to_column_at(chosen(), t):
                status.config(text="A transition is running: try again when it has finished.")
                return False
            win.destroy()
            return True

        btns = tk.Frame(win, bg=T.BG)
        btns.pack(fill='x', padx=12, pady=(0, 10))
        tk.Button(btns, text="Cancel", width=9, command=win.destroy).pack(side='right')
        tk.Button(btns, text="Start", width=9, command=start).pack(side='right', padx=(0, 6))
        tk.Label(btns, text="Cuts to the cue now.", bg=T.BG, fg=T.MUTED).pack(side='left')
        win.bind('<Return>', start)
        win.bind('<Escape>', lambda e: win.destroy())
        entry.focus_set()
        entry.select_range(0, 'end')
        win._vd = {'cue_var': cue_var, 'time_var': time_var, 'start': start, 'status': status, 'info': info,
                   'names': names}                                  # for tests
        return win

    def _trigger_go(self):
        if self.presentation_active:
            self._stop_presentation()
        elif self.presentation_mode_var.get():
            self._start_presentation()
        else:
            self._go_to_next_cue_manual()

    def _manual_next_col(self, from_col):
        """Where GO goes from `from_col`: the next column; after the last cue
        with media either back to the first column (go_at_end 'wrap') or
        nowhere (None, 'stop')."""
        if self.go_at_end == 'stop' and self._next_cue_with_media_after(from_col) is None:
            return None
        return (from_col + 1) % self.COLS

    def _on_go_wraps_toggle(self):
        self.go_at_end = 'wrap' if self._go_wraps_var.get() else 'stop'
        self._set_dirty_flag()
        if self.re and not self.transition_active:
            self._update_active_players()     # the preloaded next cue depends on it
        log.info(f"GO after the last cue: {'returns to the first cue' if self.go_at_end == 'wrap' else 'stops'}")

    def _go_to_next_cue_manual(self):
        if self.transition_active or not self.re: return
        from_col = self.active_col
        to_col = self._manual_next_col(from_col)
        if to_col is None:
            log.info(f"GO at the end of the show (cue {from_col + 1}); nothing to go to")
            self._end_of_show_until = time.monotonic() + 3.0
            self._refresh_follow_label()
            return
        self._initiate_transition(from_col, to_col)

    def _get_layers_for_col(self, col, activation=None, hold=False):
        """Render layers for cue column `col`.

        Video layers carry an 'instance' key "@<activation>:<row>:<col>|<path>"
        that the render engine uses for players, frames and textures. By
        default the column's live activation is used (re-rendering a live cue
        never restarts its videos); pass `activation` for a cue that is about
        to go live and hold=True for a paused preload.
        """
        if activation is None:
            activation = self._live_activation.get(col)
            if activation is None:
                activation = self._live_activation[col] = next(self._activation_seq)
        layers = {}
        output_mode = self.screen_var.get()
        for (r, c), p in self.grid_cells.items():
            if c != col or not p:
                continue
            # A copy: the program snapshot must not change when the show data does.
            pos_data = copy.deepcopy(self.cell_positions.get((r, c)) or {})  # null in some show files
            layer = {
                'path': p,
                'screens': self._screens_for_row(r, output_mode),
                'order': r,
                'position': pos_data,
                'loop': pos_data.get('loop', True),
                'opacity': pos_data.get('opacity', 1.0),
                'blend': pos_data.get('blend', 'normal'),
                'speed': pos_data.get('speed', 1.0),
                **{k: pos_data[k] for k in ('brightness', 'contrast', 'saturation', 'mask', 'volume', 'direction',
                                            'in_s', 'out_s', 'effect', 'effects', 'warp') if k in pos_data},
            }
            if os.path.splitext(p)[1].lower() in VIDEO_EXTS:
                layer['instance'] = self._instance_key(activation, r, c, p)
                if hold:
                    layer['hold'] = True
            layers[r] = layer
        return layers

    def _screens_for_row(self, r, output_mode=None):
        output_mode = self.screen_var.get() if output_mode is None else output_mode
        if output_mode == self.MULTIPLE:
            screen_idx = self.layer_screens.get(str(r))  # JSON keys are strings
            return [screen_idx] if screen_idx is not None else []
        if output_mode == self.SPAN:
            return self._span_displays()
        return [self.selected_display] if self.selected_display is not None else []

    def _instance_key(self, activation, r, c, p):
        """Player key for cell (r, c) of an activation. A cell moved by a layer
        or cue reorder keeps the key it had, so its video does not restart."""
        alias = self._instance_alias.get((activation, r, c, p))
        if alias is not None:
            return alias
        key = f"@{activation}:{r}:{c}|{p}"
        if key in self._instance_alias.values():  # taken by a cell that moved away
            n = 2
            while f"@{activation}:{r}:{c}.{n}|{p}" in self._instance_alias.values():
                n += 1
            key = f"@{activation}:{r}:{c}.{n}|{p}"
        return key

    # -- program (what the output shows) and blind editing --------------------
    def _output_routing(self):
        output_mode = self.screen_var.get()
        return {
            'target_monitor': self.selected_display if output_mode not in (self.MULTIPLE, self.SPAN) else None,
            'gui_monitor': self.gui_monitor_index,
            'output_mode': output_mode,
            'all_assigned_screens': self._assigned_screens(output_mode),
        }

    def _assigned_screens(self, output_mode):
        if output_mode == self.MULTIPLE:
            return {s for s in self.layer_screens.values() if s is not None}
        if output_mode == self.SPAN:
            return set(self._span_displays())
        return set()

    # -- span: one picture across several displays (span.py) ----------------------------
    def _span_clean(self):
        n = len(self.re.monitors) if self.re and getattr(self.re, 'monitors', None) else 0
        return span_mod.clean(self.output_span, n)

    def _span_displays(self):
        sp = self._span_clean()
        return list(sp['displays']) if sp and span_mod.layout(self.re.monitors, sp) else []

    def _span_regions(self):
        """Regions for the engine: the span's when Output is Span, else none."""
        if self.screen_var.get() != self.SPAN:
            return {}
        sp = self._span_clean()
        lay = span_mod.layout(self.re.monitors, sp) if sp else None
        if lay is None:
            return {}
        return {str(i): {'rect': p['rect'], 'canvas': p['canvas']} for i, p in lay[1].items()}

    def _send_regions(self):
        regions = self._span_regions()
        if regions != self._regions_sent:
            self._regions_sent = regions
            self.render_command_queue.put({'action': 'set_regions', 'regions': dict(regions)})

    def _layout_size(self, display):
        """Pixel size layers on `display` are laid out in: the span's canvas in
        Span mode, else the display's size (1920 x 1080 when unknown)."""
        mons = (getattr(self.re, 'monitors', None) or []) if self.re else []
        if self.screen_var.get() == self.SPAN and display in self._span_displays():
            lay = span_mod.layout(mons, self._span_clean())
            return tuple(lay[0])
        if display is not None and 0 <= display < len(mons):
            return mons[display].width, mons[display].height
        return 1920, 1080

    def _refresh_output_options(self):
        if not self.re or not hasattr(self, 'output_screen_combobox'):
            return
        opts = [self.NONE, self.MULTIPLE] + [f'Display {i}' for i in range(len(self.re.monitors))]
        if self._span_displays():
            opts.append(self.SPAN)
        self.output_screen_combobox['values'] = opts

    def _set_program(self, col, layers, fixed_screens=False):
        if col is not None and self._test_pattern is not None:
            self._test_pattern = None            # a cue replaced the test pattern
            if getattr(self, 'test_pattern_var', None) is not None:
                self.test_pattern_var.set('off')
        self._program = {'col': col, 'layers': copy.deepcopy(layers), 'rows': {r: r for r in layers},
                         'fixed_screens': fixed_screens}
        current = set(self._live_activation.values()) | set(self._reserved_activation.values())
        self._instance_alias = {k: v for k, v in self._instance_alias.items() if k[0] in current}
        self._highlight_live_column()   # header colors, notes, Blackout button follow the program

    def _program_layers(self):
        """The live cue as it is on the output, with current output routing."""
        prog = self._program
        if prog is None:
            return self._get_layers_for_col(self.active_col)
        layers = copy.deepcopy(prog['layers'])
        if prog.get('fixed_screens'):
            return layers                        # e.g. a test pattern: one layer per display
        output_mode = self.screen_var.get()
        for r, layer in layers.items():
            cur = prog['rows'].get(r, r)
            if cur is None or not 0 <= cur < self.ROWS:
                continue                         # its layer was deleted: it keeps the displays it is on
            layer['screens'] = self._screens_for_row(cur, output_mode)
        return layers

    @staticmethod
    def _layer_content(layers):
        """What the audience sees from a set of layers, ignoring routing and keys."""
        return sorted((l.get('order'), l.get('path'), json.dumps(l.get('position'), sort_keys=True, default=str),
                       bool(l.get('loop', True))) for l in layers.values())

    def _live_edits_pending(self):
        prog = self._program
        if prog is None or prog['col'] != self.active_col or getattr(self, 'transition_active', False):
            return False
        return self._layer_content(prog['layers']) != self._layer_content(self._get_layers_for_col(self.active_col))

    def _send_program(self, layers):
        """Put `layers` (the live cue) on the output. New videos are prerolled
        so the output never shows a gap; unchanged videos keep playing."""
        old = self._program['layers'] if self._program and self._program['col'] == self.active_col else {}
        keys = lambda ls: {l['instance'] for l in ls.values() if 'instance' in l}  # noqa: E731
        command = dict(self._output_routing(), action='render', layers=layers,
                       transition_state={'active': False})
        if keys(layers) - keys(old):
            command['preroll'] = True
        self._set_program(self.active_col, layers)
        self.render_command_queue.put(command)
        self._update_active_players()
        self.set_preview_dirty()
        self._refresh_live_edit_ui()

    def _apply_live_edits(self, event=None):
        """Update Output: put the edited live cue on the output."""
        if not self.re or getattr(self, 'transition_active', False):
            return 'break'
        if self._live_edits_pending():
            log.info(f"Update Output: applying edits to live cue {self.active_col + 1}")
            self._send_program(self._get_layers_for_col(self.active_col))
        return 'break'

    def _remap_program(self, row_map=None, col_map=None):
        """After a layer/cue reorder: keep the live and preloaded cues' player
        keys with their cells, and the program's rows pointing at their layers."""
        row_map = row_map or (lambda r: r)
        _cm = col_map or (lambda c: c)
        col_map = lambda c: None if c is None else _cm(c)  # noqa: E731
        alias = {}
        for (act, r, c, p), key in self._instance_alias.items():
            if col_map(c) is not None:
                alias[(act, row_map(r), col_map(c), p)] = key
        prog = self._program
        if prog is not None:
            act = self._live_activation.get(prog['col'])
            for r, layer in prog['layers'].items():
                if act is not None and 'instance' in layer:
                    cur_r = prog['rows'].get(r, r)
                    alias[(act, row_map(cur_r), col_map(prog['col']), layer['path'])] = layer['instance']
            prog['rows'] = {r: row_map(cur) for r, cur in prog['rows'].items()}
            prog['col'] = col_map(prog['col'])
        self._instance_alias = alias
        if self._follow_due is not None:     # a pending follow moves with its cues
            t, frm, to = self._follow_due
            if col_map(frm) is None or col_map(to) is None:
                self._cancel_follow()
            else:
                self._follow_due = (t, col_map(frm), col_map(to))
        self._live_activation = {col_map(c): a for c, a in self._live_activation.items() if col_map(c) is not None}
        self._reserved_activation = {col_map(c): a for c, a in self._reserved_activation.items()
                                     if col_map(c) is not None}

    # -- edited-cue preview (blind editing) ------------------------------------
    EDIT_PREVIEW_WIDTH = 480

    def _edit_preview_reference(self):
        """Pixel size of the output the live cue is laid out for."""
        monitors = getattr(self.re, 'monitors', None) or []
        idx = self.selected_display if self.screen_var.get() != self.MULTIPLE else 0
        if self.screen_var.get() == self.SPAN:
            return self._layout_size(idx)
        if idx is not None and 0 <= idx < len(monitors):
            w, h = getattr(monitors[idx], 'width', None), getattr(monitors[idx], 'height', None)
            if w and h:
                return w, h
        return 1920, 1080

    def _update_edit_preview(self, pending):
        """Show/hide the edited-cue preview; request a new composite when the edits change."""
        if not pending:
            self._edit_preview_sig = None
            self._edit_preview_seq += 1          # results still in flight are now stale
            self._hide_edit_preview()
            return
        layers = self._get_layers_for_col(self.active_col)
        ref = self._edit_preview_reference()
        dims = (self.EDIT_PREVIEW_WIDTH, max(1, round(self.EDIT_PREVIEW_WIDTH * ref[1] / ref[0])))
        sig = (tuple(self._layer_content(layers)), dims, ref)
        if sig == self._edit_preview_sig:
            return
        self._edit_preview_sig = sig
        self._edit_preview_seq += 1
        seq, source, results = self._edit_preview_seq, self._edit_preview_source, self._edit_preview_results

        def job():
            try:
                img = preview_compose.compose(layers, dims, ref, source)
            except Exception:  # noqa: BLE001
                log.exception("Edited-cue preview composite failed")
                img = None
            results.put((seq, img))
        self._edit_preview_executor.submit(job)

    def _poll_edit_preview(self):
        latest = None
        while True:
            try:
                latest = self._edit_preview_results.get_nowait()
            except queue.Empty:
                break
        if latest is None or latest[0] != self._edit_preview_seq:
            return                                # nothing new, or superseded by a newer edit
        self._show_edit_preview(latest[1])

    def _show_edit_preview(self, img):
        self._edit_preview_image = img
        frame = getattr(self, 'edit_preview_frame', None)
        if frame is None:
            return
        multiple = self.screen_var.get() == self.MULTIPLE
        self.edit_preview_caption.config(
            text="Update Output or Ctrl+U puts it on the output. Videos show a still"
                 + ("; all displays combined." if multiple else "."))
        if not frame.winfo_ismapped():
            frame.pack(after=self.preview_canvas, fill='x', padx=5, pady=(0, 5))
        self._redraw_edit_preview()

    def _redraw_edit_preview(self):
        canvas = getattr(self, 'edit_preview_canvas', None)
        if canvas is not None:
            self._draw_on_canvas(canvas, self._edit_preview_image, placeholder="Edited cue")

    def _hide_edit_preview(self):
        self._edit_preview_image = None
        frame = getattr(self, 'edit_preview_frame', None)
        if frame is not None and frame.winfo_ismapped():
            frame.pack_forget()

    def _refresh_live_edit_ui(self):
        btn = getattr(self, 'update_output_button', None)
        if btn is None:
            return
        try:
            pending = self._live_edits_pending()
            self._update_edit_preview(pending)
            if pending:
                btn.config(state='normal', text='Update Output\n(live cue edited)', bg=T.WARN, fg='black')
            else:
                btn.config(state='disabled', text='Update Output', bg=T.RAISED, fg='#888')
        except Exception:  # noqa: BLE001 - a UI refresh must never break editing
            log.exception("Could not refresh the Update Output button")

    def _device_layers(self):
        """Cameras and capture cards used anywhere in the show stay connected.
        Opening one takes longer than a cut waits for a first picture: on the
        owner's PC a cut to a cue with a webcam waited the full 350 ms and went
        on without it. Kept connected, a cut to any cue with a camera is
        immediate, and no other program can take the camera in between. One
        connection per device, shared by every cue (as for all live sources)."""
        out = {}
        for col in range(self.COLS):
            if not any(live_source.is_device(self.grid_cells.get((r, col))) for r in range(self.ROWS)):
                continue
            for layer in self._get_layers_for_col(col).values():
                path = layer.get('path')
                if live_source.is_device(path) and f"dev:{path}" not in out:
                    out[f"dev:{path}"] = layer
        return out

    # Besides the next and the previous cue, cues the cue list says will come
    # are kept ready (players paused on their first frame, so going to them is
    # as quick as GO and their sound starts with them): the cue the timecode
    # reaches next while following timecode, a scheduled cue due within a
    # minute, and cues marked Keep Ready in Cue Settings. Each is a decoder
    # with a few frames in memory, so at most MAX_READY of them.
    MAX_READY = 4
    SCHEDULE_READY_S = 60.0

    def _ready_cols(self):
        out = []
        for c in (self._upcoming_timecode_col(), self._upcoming_schedule_col()):
            if c is not None and c not in out:
                out.append(c)
        for key in sorted((k for k, cs in self.cue_settings.items() if cs.get('ready')), key=int):
            if int(key) not in out:
                out.append(int(key))
        has_media = lambda c: any(self.grid_cells.get((r, c)) for r in range(self.ROWS))  # noqa: E731
        out = [c for c in out if 0 <= c < self.COLS and c != self.active_col and has_media(c)]
        return out[:self.MAX_READY]

    def _upcoming_timecode_col(self):
        """The cue the running timecode reaches next (following timecode), or None."""
        if not (getattr(self, 'midi_timecode', False) or getattr(self, 'ltc_enabled', False)):
            return None
        times = sorted(self._cue_timecodes(getattr(self, '_tc_fps', 25)))
        prev = getattr(self, '_tc_prev_s', None)
        later = [(t, c) for t, c in times if prev is None or t > prev]
        return later[0][1] if later else None

    def _upcoming_schedule_col(self, now=None):
        """A scheduled cue due within SCHEDULE_READY_S, or None."""
        sc = getattr(self, 'schedule', None) or {}
        if not sc.get('enabled') or not sc.get('events'):
            return None
        now = datetime.datetime.now() if now is None else now
        nxt = scheduler.next_event(sc['events'], now)
        if nxt is None or nxt[1].get('action') != 'cue' or (nxt[0] - now).total_seconds() > self.SCHEDULE_READY_S:
            return None
        arg = nxt[1]['cue']
        return self._cue_from_remote(int(arg) if str(arg).isdigit() else arg)

    def _refresh_ready(self):
        """Preload again if the cues to keep ready changed (timecode moved on, a
        scheduled cue came within a minute)."""
        cols = self._ready_cols()
        if cols != getattr(self, '_ready_last', None):
            self._ready_last = cols
            if getattr(self, '_ui_ready', False) and not getattr(self, 'transition_active', False):
                self._update_active_players()

    def _claim_activation(self, col):
        """Activation id for `col` going live now: the preloaded one if `col`
        was preloaded (its players are paused on frame 0), else a new one."""
        act = self._reserved_activation.pop(col, None)
        return act if act is not None else next(self._activation_seq)

    def _prev_cue_col(self):
        """The cue BACK goes to: the previous cue with media, or None."""
        return next((c for c in range(self.active_col - 1, -1, -1)
                     if any(self.grid_cells.get((r, c)) for r in range(self.ROWS))), None)

    def _next_cue_col(self):
        """The cue GO will go to next (used for preloading)."""
        if getattr(self, 'presentation_active', False):
            return self._find_next_cue_with_media(self.active_col)
        return self._manual_next_col(self.active_col)

    def _on_cue_select(self, event):
        if self.transition_active: return
        selected_name = self.cue_name_var.get()
        for idx_str, name in self.cue_names.items():
            if name == selected_name:
                self._select_column(int(idx_str))
                break

    def _on_output_change(self, *args, mark_dirty=True):
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
        elif val == self.SPAN:
            disp = self._span_displays()
            if disp:
                self.selected_display = disp[0]        # layout size and the editor use the span's canvas
            else:
                self.selected_display = None
                self.screen_var.set(self.NONE)          # trace: comes back here with None
                return
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

        self._send_regions()

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
        if mark_dirty:
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

        if getattr(self, 'transition_active', False):
            # The fade owns the output; its end applies (live) or flags (blind) edits.
            self._refresh_live_edit_ui()
            return
        prog = self._program
        if prog is not None and prog['col'] is None:
            pass    # blackout, or the live cue was deleted: keep the output as it is
        elif self.live_edit_mode == 'live' or prog is None or prog['col'] != self.active_col:
            # Live editing (or nothing on the output yet): show the cue as edited.
            self._send_program(self._get_layers_for_col(self.active_col))
            return
        # Blind editing: re-send what is on the output (output routing may
        # have changed); edits wait for Update Output.
        command = dict(self._output_routing(), action='render', layers=self._program_layers(),
                       transition_state={'active': False})
        self.render_command_queue.put(command)
        self.set_preview_dirty()
        self._refresh_live_edit_ui()

    def _show_cell_menu(self, event, row, col):
        menu = Menu(self.root, tearoff=0)
        path = self.grid_cells.get((row, col))
        if path and (os.path.exists(path) or live_source.is_live(path)):
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

    @_locked_out
    def _set_layer_output(self, row, screen_index):
        """Callback to set a layer's target screen and update rendering."""
        row_str = str(row)
        if screen_index is None:
            if row_str in self.layer_screens:
                del self.layer_screens[row_str]
        else:
            self.layer_screens[row_str] = screen_index
        
        log.info(f"Set Layer {row+1} output to: {'Display ' + str(screen_index) if screen_index is not None else 'None'}")
        
        # Update cell previews for the entire row
        for c in range(self.COLS):
            self._update_cell_preview(row, c)
            
        self._trigger_column()
        self._set_dirty_flag()

    @_locked_out
    def _toggle_loop(self, row, col, var):
        if (row, col) in self.cell_positions:
            self.cell_positions[(row, col)]['loop'] = var.get()
            if col == self.active_col:
                self._trigger_column()   # applies now in live mode, waits in blind mode
            else:
                self._update_active_players()
            self._set_dirty_flag()

    @_locked_out
    def _clear_cell(self, row, col):
        had_media = self.grid_cells.get((row, col)) is not None
        self.grid_cells[(row, col)] = None
        if (row, col) in self.cell_positions: 
            del self.cell_positions[(row, col)]
        if had_media:
            self._set_dirty_flag()            # after the change: undo records the state as it is now
        self._update_cell_preview(row, col)
        
        # ADD THIS - Invalidate static cache for this column
        if self.re:
            self.re.invalidate_static_cache_for_column(col)
        
        if col == self.active_col:
            self._trigger_column()

    def _show_cue_menu(self, event, col):
        menu = Menu(self.root, tearoff=0)
        menu.add_command(label='Cue Settings…', command=lambda: self._open_cue_settings(col))
        menu.add_command(label='Rename', command=lambda: self._rename_cue(col))
        menu.add_separator()
        menu.add_command(label='Duplicate Cue', command=lambda: self._duplicate_cue(col))
        menu.add_command(label='Clear Cue', command=lambda: self._clear_cue(col))
        menu.add_command(label='Delete Cue', command=lambda: self._delete_cue(col))
        menu.tk_popup(event.x_root, event.y_root)

    @_locked_out
    @_undo_step_op
    def _duplicate_cue(self, col):
        """A copy of cue `col` right after it: its media, cell settings and cue
        settings (not its cue number, which must stay unique). The cue on the
        output keeps playing, even when it is the one copied or moves right."""
        if not 0 <= col < self.COLS:
            return None
        old_cols = self.COLS
        cmap = lambda c: c + 1 if c > col else c  # noqa: E731
        new_cells, new_positions = {}, {}
        for (r, c), media in self.grid_cells.items():
            new_cells[(r, cmap(c))] = media
        for (r, c), pos in self.cell_positions.items():
            new_positions[(r, cmap(c))] = pos
        for r in range(self.ROWS):
            new_cells[(r, col + 1)] = self.grid_cells.get((r, col))
            if (r, col) in self.cell_positions:
                new_positions[(r, col + 1)] = copy.deepcopy(self.cell_positions[(r, col)])
        new_names = {str(cmap(int(k))): v for k, v in self.cue_names.items()}
        new_names[str(col + 1)] = f"{self.cue_names.get(str(col), f'Cue {col + 1}')} copy"
        new_settings = project_schema.remap_cue_keys(self.cue_settings, cmap)
        copied = {k: copy.deepcopy(v) for k, v in self.cue_settings.get(str(col), {}).items() if k != 'number'}
        if copied:
            new_settings[str(col + 1)] = copied
        self.grid_cells, self.cell_positions, self.cue_names = new_cells, new_positions, new_names
        self.cue_settings = new_settings
        self.COLS += 1
        if self.active_col > col:
            self.active_col += 1
        self._remap_program(col_map=cmap)
        self._build_grid()
        self._highlight_live_column()
        self._update_cue_dropdown()
        self._trigger_column()
        self._update_active_players()
        self._undo_pending = ('Duplicate Cue', None, {c: cmap(c) for c in range(old_cols)})
        self._set_dirty_flag()
        log.info(f"Duplicated cue {col + 1} as cue {col + 2}")
        return col + 1

    @_locked_out
    def _clear_cue(self, col):
        for r in range(self.ROWS): self._clear_cell(r, col)

    def _show_layer_menu(self, event, row):
        menu = Menu(self.root, tearoff=0)
        menu.add_command(label='Rename', command=lambda: self._rename_layer(row))
        menu.add_separator()
        menu.add_command(label='Clear Layer', command=lambda: self._clear_layer(row))
        menu.add_command(label='Delete Layer', command=lambda: self._delete_layer(row))
        menu.tk_popup(event.x_root, event.y_root)

    @_locked_out
    def _clear_layer(self, row):
        for c in range(self.COLS): self._clear_cell(row, c)

    def _show_media_menu(self, event, path):
        menu = Menu(self.root, tearoff=0)
        if live_source.is_live(path):
            menu.add_command(label='Test Live Source…', command=lambda: self._open_live_source_dialog(path))
            menu.add_separator()
        elif os.path.exists(path):
            ext = os.path.splitext(path)[1].lower()
            # Only show Optimize for videos if explicitly allowed
            if ext in VIDEO_EXTS:
                menu.add_command(label='Make Playback Copy…', command=lambda: self._open_playback_copy(path))
            if ext in VIDEO_EXTS and VD_SHOW_VIDEO_OPTIMIZE_MENU:
                menu.add_command(label='Optimize', command=lambda: self._open_optimize_window(path))
            if ext in VIDEO_EXTS:
                menu.add_separator()
        else:
            menu.add_command(label='Locate Missing File…', command=lambda: self._locate_missing(path))
            menu.add_command(label='Relink Missing Media…', command=self._open_relink_dialog)
            menu.add_separator()
        menu.add_command(label='Remove from Project', command=lambda: self._remove_media(path, delete_from_disk=False))
        # Hide the destructive option unless explicitly allowed
        if VD_SHOW_DELETE_FROM_DISK_MENU:
            menu.add_command(label='Remove and DELETE FROM DISK',
                             command=lambda: self._remove_media(path, delete_from_disk=True))
        menu.tk_popup(event.x_root, event.y_root)


    @_locked_out
    def _rename_cue(self, col):
        col_str = str(col)
        current_name = self.cue_names.get(col_str, f'Cue {col+1}')
        new_name = simpledialog.askstring('Rename Cue', 'Enter new name:', initialvalue=current_name)
        if new_name:
            self.cue_names[col_str] = new_name
            self.col_labels[col].config(text=self._cue_header_text(col))
            self._update_cue_dropdown()
            self._set_dirty_flag()

    @_locked_out
    def _rename_layer(self, row):
        row_str = str(row)
        current_name = self.layer_names.get(row_str, f'Layer {row+1}')
        new_name = simpledialog.askstring('Rename Layer', 'Enter new name:', initialvalue=current_name)
        if new_name:
            self.layer_names[row_str] = new_name
            canvas = self.row_labels[row]
            canvas.itemconfig(canvas.text_id, text=new_name)
            self._set_dirty_flag()

    @_locked_out
    @_undo_step_op
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
        cmap = lambda c: None if c == col else (c - 1 if c > col else c)  # noqa: E731
        self.cue_settings = project_schema.remap_cue_keys(self.cue_settings, cmap)
        deleting_live = (col == self.active_col)
        if deleting_live:
            self._cancel_follow()
        if deleting_live and self._program is not None:
            self._program['col'] = None   # the output keeps showing it until the next GO/cut
        self._remap_program(col_map=cmap)
        self._undo_pending = ('Delete Cue', None, {c: cmap(c) for c in range(self.COLS)})
        self.COLS -= 1
        if deleting_live:
            # Playhead on the cue before the deleted one, so GO goes to the one after it.
            self.active_col = (col - 1) % self.COLS
        elif self.active_col > col:
            self.active_col -= 1
        self._build_grid()
        self._highlight_live_column()
        self._update_cue_dropdown()
        self._trigger_column()
        self._update_active_players()
        self._set_dirty_flag()

    @_locked_out
    @_undo_step_op
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
        old_rows = self.ROWS
        self.ROWS -= 1
        rmap = lambda r: None if r == row else (r - 1 if r > row else r)  # noqa: E731
        # The live cue keeps playing (no re-cut): its player keys and program
        # rows follow their layers. A live layer on the deleted row stays on the
        # output until the next cue or Update Output, like a deleted live cue.
        self._remap_program(row_map=rmap)
        if self.re:
            self.re.invalidate_static_cache_for_column(self.active_col)
        self._build_grid()
        self._highlight_live_column()
        self._trigger_column()
        self._update_active_players()
        self._undo_pending = ('Delete Layer', {r: rmap(r) for r in range(old_rows)}, None)
        self._set_dirty_flag()

    @_locked_out
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

    @_locked_out
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

    @_locked_out
    @_undo_step_op
    def _add_cue(self):
        self.COLS += 1
        for r in range(self.ROWS): self.grid_cells[(r, self.COLS-1)] = None
        self.cue_names[str(self.COLS-1)] = f'Cue {self.COLS}'
        self._build_grid()
        self._highlight_live_column()     # no re-cut: the live cue keeps playing
        self._update_cue_dropdown()
        self._update_active_players()
        self._undo_pending = ('Add Cue', None, {c: c for c in range(self.COLS - 1)})
        self._set_dirty_flag()

    @_locked_out
    @_undo_step_op
    def _add_layer(self):
        self.ROWS += 1
        for c in range(self.COLS): self.grid_cells[(self.ROWS-1, c)] = None
        self.layer_names[str(self.ROWS-1)] = f'Layer {self.ROWS}'
        self._build_grid()
        self._highlight_live_column()     # no re-cut: the live cue keeps playing and blind edits stay blind
        self._trigger_column()
        self._update_active_players()
        self._undo_pending = ('Add Layer', {r: r for r in range(self.ROWS - 1)}, None)
        self._set_dirty_flag()

    def _update_cue_dropdown(self):
        cue_list = sorted(self.cue_names.values(), key=lambda name: [int(s) if s.isdigit() else s for s in re.split('([0-9]+)', name)])
        self.cue_select_combobox['values'] = cue_list
        if self.COLS > 0 and self.active_col < self.COLS:
            self.cue_name_var.set(self.cue_names.get(str(self.active_col), ''))

    @_locked_out
    def _edit_media(self, row, col):
        if not self.re: return
        media_path = self.grid_cells.get((row, col))
        if not media_path or not (os.path.exists(media_path) or live_source.is_live(media_path)):
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

    @_locked_out
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
        
        self.go_button.config(text="GO", bg=T.GO)
        self._set_dirty_flag()


    def _start_presentation(self):
        if self.presentation_active: return
        log.info("Starting presentation...")
        self.presentation_active = True
        self._cancel_follow()            # the presentation timer governs while it runs
        self.go_button.config(text="STOP", bg=T.STOP)
        self._advance_presentation()

    def _stop_presentation(self):
        if not self.presentation_active: return
        log.info("Stopping presentation...")
        if self.presentation_timer:
            self.root.after_cancel(self.presentation_timer)
            self.presentation_timer = None
        self.presentation_active = False
        self.go_button.config(text="GO", bg=T.GO)

    def _advance_presentation(self):
        if not self.presentation_active: return

        from_col = self.active_col
        next_col = self._find_next_cue_with_media(from_col)

        if next_col is None:
            log.info("End of presentation.")
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
        self._undo_note()

    # -- undo / redo (undo.py) ----------------------------------------------------------
    # Covers the cue grid: media in cells and their settings, cue and layer names,
    # cue settings, cues and layers added, deleted or moved, each layer's display,
    # the show's media list, GO at the end and the schedule. Output settings
    # (display, corner pins, edge blends) are not undone: that would change the
    # output by itself. Undoing is an edit like any other: with blind editing
    # the live cue's changes wait for Update Output; nothing is re-cut.
    _UNDO_KEYS = ('grid_cells', 'cell_positions', 'cue_names', 'layer_names', 'cue_settings', 'layer_screens',
                  'project_media_paths', 'go_at_end', 'schedule')

    def _undo_snapshot(self):
        st = {k: copy.deepcopy(getattr(self, k, None)) for k in self._UNDO_KEYS}
        st['grid_cells'] = {k: v for k, v in (st['grid_cells'] or {}).items() if v}   # empty = absent
        st['cell_positions'] = {k: v for k, v in (st['cell_positions'] or {}).items() if v is not None}
        st['rows'], st['cols'] = self.ROWS, self.COLS
        return st

    def _undo_stack(self):
        stack = getattr(self, '_undo_history', None)
        if stack is None:
            stack = self._undo_history = undo.UndoStack()
        return stack

    def _undo_reset(self):
        """A new baseline (a show was opened, recovered or started)."""
        self._undo_pending = None
        self._undo_stack().reset(self._undo_snapshot())
        self._refresh_undo_menu()

    def _undo_note(self):
        """Called whenever the show changes (_set_dirty_flag)."""
        if getattr(self, '_undo_in_op', False):
            return                     # recorded when the cue/layer change is complete (_undo_step_op)
        pending, self._undo_pending = getattr(self, '_undo_pending', None), None
        if getattr(self, '_undo_busy', False):
            return
        label, rows, cols = pending or ('Edit', None, None)
        try:
            if self._undo_stack().record(self._undo_snapshot(), label, rows=rows, cols=cols):
                self._refresh_undo_menu()
        except Exception:  # noqa: BLE001 - undo bookkeeping must never break an edit
            log.exception("Undo: could not record a change")

    @_locked_out
    def _undo(self):
        self._undo_step(self._undo_stack().undo, "Undo")

    @_locked_out
    def _redo(self):
        self._undo_step(self._undo_stack().redo, "Redo")

    def _undo_step(self, step, what):
        if getattr(self, 'transition_active', False):
            log.info(f"{what}: ignored during a transition")
            return
        got = step()
        if got is None:
            return
        state, rows, cols = got
        self._undo_busy = True
        try:
            self._undo_restore(state, undo.map_fn(rows), undo.map_fn(cols))
            self.project_is_dirty = True
            self._update_title()
        finally:
            self._undo_busy = False
            self._undo_pending = None
        self._refresh_undo_menu()
        log.info(f"{what}: show restored ({state['rows']} layers x {state['cols']} cues)")

    def _undo_restore(self, state, row_map=None, col_map=None):
        st = copy.deepcopy(state)
        self.ROWS, self.COLS = st['rows'], st['cols']
        self.grid_cells = {(r, c): st['grid_cells'].get((r, c)) for r in range(self.ROWS) for c in range(self.COLS)}
        for k in self._UNDO_KEYS:
            if k != 'grid_cells':
                setattr(self, k, st[k])
        if row_map is not None or col_map is not None:
            # Cues or layers moved: the live cue's player keys and program rows
            # follow their cells, as they did when the change was made.
            self._remap_program(row_map=row_map, col_map=col_map)
            if col_map is not None:
                ac = col_map(self.active_col)
                self.active_col = ac if ac is not None else self.active_col
        self.active_col = max(0, min(self.active_col, self.COLS - 1))
        if self.re:
            self.re.invalidate_static_cache_for_column(self.active_col)
        self._build_grid()
        self._highlight_live_column()
        self._update_cue_dropdown()
        self._trigger_column()
        self._update_active_players()
        if getattr(self, '_go_wraps_var', None) is not None:
            self._go_wraps_var.set(self.go_at_end == 'wrap')
        self._build_media_list()

    def _refresh_undo_menu(self):
        menu = getattr(self, 'edit_menu', None)
        if menu is None:
            return
        stack = self._undo_stack()
        try:
            u, r = stack.undo_label(), stack.redo_label()
            menu.entryconfigure(0, label=f"Undo {u}" if u else "Undo", state='normal' if u else 'disabled')
            menu.entryconfigure(1, label=f"Redo {r}" if r else "Redo", state='normal' if r else 'disabled')
        except tk.TclError:
            pass

    def _title_text(self):
        name = os.path.basename(self.current_project_path) if self.current_project_path else "Untitled Project"
        dirty_marker = "*" if self.project_is_dirty else ""
        lock = "  [LOCKED]" if getattr(self, 'edit_locked', False) else ""
        return f"VisualDeck - {name}{dirty_marker}{lock}"

    def _update_title(self):
        self.root.title(self._title_text())

    def _check_before_closing_project(self):
        if not self.project_is_dirty:
            return True
        
        response = messagebox.askyesnocancel("Save Changes?", "You have unsaved changes. Would you like to save them?")
        
        if response is True:  # Yes
            return self._save_project()
        elif response is False:  # No: the operator discards the changes, so the autosave goes too
            self._discard_autosave(self.current_project_path)
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
        self._undo_reset()

    def _auto_load_last_project(self):
        # This is the very first action after startup, so we don't need to check for unsaved changes.
        self._startup_load()
        try:
            if self._recoverable_autosave(None) is not None and not self.project_is_dirty:
                self._offer_recovery(None)        # an untitled show that was never saved
        except Exception:  # noqa: BLE001 - recovery must never stop the app from starting
            log.exception("Checking for an untitled autosave failed")

    def _startup_load(self):
        if self.recent_files:
            last_project = self.recent_files[0]
            if os.path.exists(last_project):
                log.info(f"Auto-loading last project: {last_project}")
                # The `initial_load=True` flag will bypass the "save changes?" check
                self._open_project(path=last_project, initial_load=True)
            else:
                log.warning("Last project not found, starting new project.")
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
            log.info(f"Successfully opened project: {path}")
            self._warm_effects()
            if self._ui_ready:
                self.root.after(self.GC_FREEZE_DELAY_MS, lambda: self._freeze_gc("opening the show"))
            if self._recoverable_autosave(path) is not None:
                self._offer_recovery(path)

        except Exception:
            log.exception(f"Could not open or parse project file: {path}")
            messagebox.showerror("Error Opening Project", f"Could not open or parse project file.")
            return

    def _apply_project_data(self, data, path):
        """Applies loaded project data to the application state."""
        win = getattr(self, "_relink_win", None)
        if win is not None:  # belongs to the previous show
            try:
                win.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._relink_win = None
        self._reset_project_state(new_project=False)
        isf.set_show_folder(os.path.dirname(os.path.abspath(path)) if path else None)   # its own effects
        data, schema_info = project_schema.migrate(data)
        self._unknown_project_keys = project_schema.unknown_keys(data)
        if schema_info['newer']:
            log.warning(f"Show saved by a newer VisualDeck (schema {schema_info['from']} > "
                        f"{project_schema.CURRENT}); unknown settings are kept but not used")
            self.root.after(300, lambda: messagebox.showwarning(
                "Newer show file", "This show was saved by a newer version of VisualDeck. It has been opened as "
                "far as this version understands it; settings it does not know are kept when you save.",
                parent=self.root))
        self.cue_settings = data.get('cue_settings', {})
        self.go_at_end = data.get('go_at_end') if data.get('go_at_end') in project_schema.GO_AT_END else 'wrap'
        self.schedule = scheduler.clean(data.get('schedule'))
        self._schedule_fired = set()
        if self.schedule['enabled'] and self.schedule['events']:
            log.info(f"Show schedule is on: {len(self.schedule['events'])} event(s)")
        if getattr(self, '_go_wraps_var', None) is not None:
            self._go_wraps_var.set(self.go_at_end == 'wrap')
        
        # Load settings
        # The render backend and decoder are this computer's choice (Preferences),
        # not the show's: a show from a machine with a stronger GPU must not
        # switch this one to a mode it cannot run (F20). The show still records
        # them for older versions of VisualDeck.
        settings = data.get('settings') if isinstance(data.get('settings'), dict) else {}
        for key, mine in (('render_mode', self.render_mode), ('hw_accel', self.hw_accel)):
            theirs = settings.get(key)
            if theirs is not None and theirs != mine:
                log.info(f"Show was saved with {key}={theirs}; using this computer's {key}={mine} (Preferences)")

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
                pos = val.get('pos')
                self.cell_positions[(r, c)] = pos if isinstance(pos, dict) else {}   # null/garbage -> defaults
            except (ValueError, KeyError, AttributeError, TypeError):
                log.warning(f"Could not parse media cell data for key '{key}'")

        # Missing media: files that moved together with the show file are
        # found through their stored relative path; anything else is offered
        # to the operator in the relink dialog once the UI is up.
        info = data.get('media_info')
        self.media_info = info if isinstance(info, dict) else {}
        missing = media_relink.find_missing(self._all_media_paths())
        self._missing_cache = None
        auto_relinked = media_relink.resolve_relative(missing, self.media_info, path)
        if auto_relinked:
            media_relink.apply_mapping(auto_relinked, self.grid_cells, self.project_media_paths, self.media_info)
            for old, new in auto_relinked.items():
                log.info(f"Media relinked via show-relative path: {old} -> {new}")
        still_missing = len(missing) - len(auto_relinked)
        if still_missing:
            log.warning(f"{still_missing} media file(s) used by this show are missing")

        # Load output settings
        outputs = data.get('outputs', {})
        self.output_span = span_mod.clean(outputs.get('span'))     # before the mode: "Span" needs it
        self.screen_var.set(outputs.get('mode', self.NONE))
        self.layer_screens = outputs.get('layer_screens', {})
        warps = outputs.get('warps') if isinstance(outputs.get('warps'), dict) else {}
        self.output_warps = {str(k): [list(p) for p in c] for k, c in warps.items()
                             if output_warp.clean(c) is not None}
        blends = outputs.get('blends') if isinstance(outputs.get('blends'), dict) else {}
        self.output_blends = {str(k): self._blend_dict(edge_blend.clean(b)) for k, b in blends.items()
                              if edge_blend.clean(b) is not None}
        meshes = outputs.get('meshes') if isinstance(outputs.get('meshes'), dict) else {}
        self.output_meshes = {str(k): mesh_warp.as_dict(mesh_warp.clean(m)) for k, m in meshes.items()
                              if mesh_warp.clean(m) is not None}
        effects = outputs.get('effects') if isinstance(outputs.get('effects'), dict) else {}
        self.output_effects = {str(k): isf.stack(e) for k, e in effects.items() if isf.stack(e)}

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
        self._undo_reset()           # relinked paths are part of the opened show, not an undoable edit
        if auto_relinked:
            # The file on disk still has the old paths until the show is saved.
            self._set_dirty_flag()
        if still_missing:
            self.root.after(400, lambda: self._open_relink_dialog(on_open=True))

    # -- output health ------------------------------------------------------
    HEALTH_COLORS = {'ok': '#5fbf5f', 'warn': T.WARN, 'error': '#ff5c5c', 'none': '#888888'}

    def _missing_media(self, max_age=5.0):
        """The show's missing media files, re-checked at most every max_age seconds (stat calls)."""
        t = time.monotonic()
        cached = getattr(self, '_missing_cache', None)
        if cached is None or t - cached[0] > max_age:
            cached = (t, list(media_relink.find_missing(self._all_media_paths())))
            self._missing_cache = cached
        return cached[1]

    def _missing_media_count(self, max_age=5.0):
        return len(self._missing_media(max_age))

    # -- soak test (soak.py) --------------------------------------------------------------
    # Python's full garbage collection scans every object and stops every
    # thread while it runs, the render thread too: measured in the app here
    # (Performance Mode, a 20-minute soak test) two full collections, 25 ms
    # each, longer than a 60 fps frame; timed directly with a video playing,
    # 17-38 ms before freezing and 0.0-0.1 ms after. Once the
    # app has started, and after a show is opened, what is there is collected
    # once and then set aside (gc.freeze): it is not scanned again, so later
    # full collections only look at what came after. Objects set aside are
    # never collected by the garbage collector; ones that die later are still
    # freed when nothing refers to them (most are), only those in a reference
    # cycle stay. A few seconds after the event, so the window is up first.
    GC_FREEZE_DELAY_MS = 3000

    def _freeze_gc(self, why):
        try:
            t0 = time.perf_counter()
            gc.collect()
            gc.freeze()
            log.info(f"Garbage collector: {gc.get_freeze_count()} objects set aside after {why} "
                     f"(collection took {(time.perf_counter() - t0) * 1000:.0f} ms)")
        except Exception as e:  # noqa: BLE001
            log.debug(f"gc.freeze failed: {e}")

    def _soak_cues(self):
        return [c for c in range(self.COLS)
                if any(self.grid_cells.get((r, c)) for r in range(self.ROWS))]

    def _open_soak(self):
        """Help > Soak Test: GO through the show's cues for a long time and
        record whether memory, threads, decoders or the frame rate drift."""
        if getattr(self, '_soak', None) is not None and getattr(self, '_soak_win', None) is not None:
            try:
                self._soak_win.deiconify()
                self._soak_win.lift()
                return self._soak_win
            except tk.TclError:
                pass
        win = tk.Toplevel(self.root)
        self._soak_win = win
        win.title("Soak Test")
        win.configure(bg=T.SURFACE)
        win.transient(self.root)
        body = tk.Frame(win, bg=T.SURFACE, padx=15, pady=15)
        body.pack(fill='both', expand=True)
        tk.Label(body, bg=T.SURFACE, fg=T.TEXT, justify='left', anchor='w', wraplength=460,
                 text="Plays this show's cues on the output, GO every few seconds with each cue's own "
                      "transition, and records memory, threads, decoders, GPU textures, frame rate, late "
                      "frames, errors and sound underruns. Use it before a long show or an installation, not "
                      "during one. Nothing in the show is changed or saved. If Presentation Mode is "
                      "running, it plays the cues and the soak test only records.").pack(fill='x')
        row = tk.Frame(body, bg=T.SURFACE)
        row.pack(fill='x', pady=(10, 0))
        minutes_var, every_var = tk.StringVar(value="60"), tk.StringVar(value="20")
        for label, var, unit in (("Run for", minutes_var, "minutes"), ("GO every", every_var, "seconds")):
            tk.Label(row, text=label, bg=T.SURFACE, fg=T.TEXT).pack(side='left')
            ttk.Entry(row, textvariable=var, width=6).pack(side='left', padx=4)
            tk.Label(row, text=unit, bg=T.SURFACE, fg=T.MUTED).pack(side='left', padx=(0, 14))
        status = tk.Label(body, bg=T.SURFACE, fg=T.MUTED, justify='left', anchor='w', font=T.mono(9),
                          wraplength=460, text=f"{len(self._soak_cues())} cue(s) with media in this show.")
        status.pack(fill='x', pady=(10, 0))
        btns = tk.Frame(body, bg=T.SURFACE)
        btns.pack(fill='x', pady=(14, 0))
        start_btn = tk.Button(btns, text="Start", width=9, bg=T.GO, fg='black')
        stop_btn = tk.Button(btns, text="Stop", width=9, state='disabled')
        tk.Button(btns, text="Close", width=9, command=win.withdraw).pack(side='right')
        stop_btn.pack(side='right', padx=(0, 6))
        start_btn.pack(side='right', padx=(0, 6))

        def start():
            try:
                minutes = float(minutes_var.get())
                every = float(every_var.get())
            except ValueError:
                status.config(fg=T.WARN, text="Minutes and seconds are numbers.")
                return False
            if minutes <= 0 or every < 2:
                status.config(fg=T.WARN, text="Run for more than 0 minutes, GO every 2 seconds or more.")
                return False
            if not self._soak_cues():
                status.config(fg=T.WARN, text="The show has no cues with media.")
                return False
            self._soak_start(minutes * 60.0, every, status)
            start_btn.config(state='disabled')
            stop_btn.config(state='normal')
            return True

        def stop():
            self._soak_stop("stopped")
            start_btn.config(state='normal')
            stop_btn.config(state='disabled')

        start_btn.config(command=start)
        stop_btn.config(command=stop)
        win.protocol("WM_DELETE_WINDOW", win.withdraw)          # closing the window does not stop a run
        win._vd = {'minutes_var': minutes_var, 'every_var': every_var, 'start': start, 'stop': stop,
                   'status': status}                              # for tests
        return win

    def _soak_start(self, duration_s, every_s, status_label=None, sample_s=None):
        folder = os.path.join(self.CONFIG_DIR, 'soak')
        rec = soak.Recorder(folder)
        self._soak = {'rec': rec, 't0': time.monotonic(), 'duration': duration_s, 'every': every_s,
                      'sample': sample_s or soak.SAMPLE_S, 'gos': 0, 'label': status_label, 'token': object(),
                      'next_go': time.monotonic() + every_s, 'next_sample': time.monotonic()}
        log.info(f"Soak test started: {duration_s / 60:.0f} min, GO every {every_s:g} s, recording to {rec.csv_path}")
        cues = self._soak_cues()
        if self.presentation_active:
            log.info("Soak test: Presentation Mode is running the cues; the soak test records only")
        elif self.active_col not in cues:
            self._go_to_cue(cues[0])                 # start from the first cue with media
            self._soak['gos'] += 1
        self._soak_tick(self._soak['token'])

    def _soak_tick(self, token):
        st = getattr(self, '_soak', None)
        if st is None or st['token'] is not token:
            return
        now = time.monotonic()
        if self.presentation_active:
            # Presentation Mode runs the cues itself (checklist I2: 72 h with Loop): the
            # soak test only records, and counts each cue change as a GO. Its own GO
            # would stop the presentation (GO reads STOP while it runs).
            if st.get('last_col', self.active_col) != self.active_col:
                st['gos'] += 1
            st['next_go'] = now + st['every']
        elif now >= st['next_go'] and not self.transition_active:
            if (self.active_col is not None and not self.presentation_mode_var.get()
                    and self._manual_next_col(self.active_col) is None):
                self._go_to_cue(self._soak_cues()[0])   # the show stops after its last cue: start over
            else:
                self._trigger_go()
            st['gos'] += 1
            st['next_go'] = now + st['every']
        st['last_col'] = self.active_col
        if now >= st['next_sample']:
            row = soak.sample(self._current_health(), self.re, gos=st['gos'], active_cue=self.active_col,
                              t_s=now - st['t0'])
            st['rec'].add(row)
            st['next_sample'] = now + st['sample']
            lbl = st.get('label')
            if lbl is not None:
                try:
                    lbl.config(fg=T.MUTED, text=(
                        f"Running: {row['t_s'] / 60:.1f} of {st['duration'] / 60:.0f} min, {st['gos']} GO\n"
                        f"memory {row['rss_mb']:.0f} MB, threads {row['threads']}, decoders {row['decoders']}, "
                        f"{row['renders_per_s']} renders/s\nRecording to {st['rec'].csv_path}"))
                except tk.TclError:
                    pass
        if 'census' not in st and now - st['t0'] >= st['duration'] / 5:   # after warm-up (as soak.summarize)
            st['census'] = soak.object_census()
            if soak.trace_wanted():                  # diagnostics (soak.TRACE_ENV)
                try:
                    soak.trace_start()
                    st['tracing'] = True
                    log.info(f"Soak test: tracing Python allocations from now ({soak.TRACE_ENV}); "
                             f"the baseline follows at {0.4 * st['duration'] / 60:.0f} min")
                except Exception as e:  # noqa: BLE001
                    log.warning(f"Soak test: could not trace allocations: {e}")
        if st.get('tracing') and 'trace' not in st and now - st['t0'] >= 0.4 * st['duration']:
            try:                                     # the baseline, once buffers hold traced allocations
                st['tk_commands'] = soak.tk_command_count(self.root)
                st['trace'] = soak.trace_baseline()
                log.info("Soak test: allocation baseline taken")
            except Exception as e:  # noqa: BLE001
                st['tracing'] = False
                log.warning(f"Soak test: could not trace allocations: {e}")
        if now - st['t0'] >= st['duration']:
            self._soak_stop("finished")
            return
        st['timer'] = self.root.after(200, lambda: self._soak_tick(token))

    def _soak_stop(self, how="stopped"):
        st, self._soak = getattr(self, '_soak', None), None
        if st is None:
            return None
        if st.get('timer'):
            try:
                self.root.after_cancel(st['timer'])
            except Exception:  # noqa: BLE001
                pass
        now = time.monotonic()
        rows = st['rec'].rows
        if not rows or now - st['t0'] - rows[-1]['t_s'] >= 1.0:   # the tick that ended the run may have sampled
            st['rec'].add(soak.sample(self._current_health(), self.re, gos=st['gos'], active_cue=self.active_col,
                                      t_s=now - st['t0']))
        pacer = getattr(getattr(self, 'render_thread', None), 'frame_pacer', None)
        target = pacer.get_target_fps() if pacer is not None else None
        extra = {'how': how, 'every_s': st['every'], 'render_mode': getattr(self, 'render_mode', None)}
        if 'census' in st:                     # which kinds of Python objects grew (for a report)
            end = soak.object_census()
            grew = soak.census_growth(st['census'], end)
            extra.update(objects_after_warmup=sum(st['census'].values()), objects_end=sum(end.values()),
                         object_growth=grew)
        if 'trace' in st:
            try:
                path = st['rec'].summary_path[:-len('.json')] + '-tracemalloc.txt'
                tk_now = soak.tk_command_count(self.root)
                text = soak.trace_report(st['trace'], extra_lines=(
                    f"Tkinter callbacks registered: {st.get('tk_commands')} at the baseline, {tk_now} at the end.",))
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(text + "\n")
                extra['tracemalloc_report'] = path
                log.info(f"Soak test: allocation report written to {path}")
            except Exception as e:  # noqa: BLE001
                log.warning(f"Soak test: allocation report failed: {e}")
        elif st.get('tracing'):
            soak.trace_stop()                        # stopped before the baseline: no report
            log.info("Soak test: stopped before the allocation baseline; no allocation report")
        summary = st['rec'].finish(target_fps=target, extra=extra)
        if 'census' in st:
            note = soak.census_note(summary['object_growth'],
                                    summary['objects_end'] - summary['objects_after_warmup'])
            if note:
                summary.setdefault('notes', []).append(note)
                st['rec'].rewrite(summary)
        log.info(f"Soak test {how}: {summary['findings']} ({st['rec'].summary_path})")
        lbl = st.get('label')
        if lbl is not None:
            try:
                lbl.config(fg=T.MUTED if summary['ok'] else T.WARN,
                           text=f"Soak test {how} after {summary['duration_s'] / 60:.1f} min, {summary['gos']} GO.\n"
                                + "\n".join(summary['findings'] + summary.get('notes', []))
                                + f"\nSummary: {st['rec'].summary_path}"
                                + (f"\nAllocations: {summary['tracemalloc_report']}"
                                   if summary.get('tracemalloc_report') else ""))
            except tk.TclError:
                pass
        return summary

    def _process_status(self):
        """Memory and threads of the app, and how long it has run (for Output
        Health and Copy Report: growth over a long show shows here)."""
        t0 = getattr(self, '_process_t0', None)
        if t0 is None:
            t0 = self._process_t0 = time.monotonic()
        rss = soak.rss_mb()
        peak = max(getattr(self, '_process_peak_mb', 0.0), rss or 0.0)
        self._process_peak_mb = peak
        return {'rss_mb': round(rss, 1) if rss else None, 'peak_mb': round(peak, 1) if peak else None,
                'threads': threading.active_count(), 'up_s': round(time.monotonic() - t0),
                'gdi': soak.gdi_objects()}

    @staticmethod
    def _process_text(p):
        if not p:
            return "?"
        up = int(p.get('up_s') or 0)
        mem = f"memory {p['rss_mb']:.0f} MB (highest {p['peak_mb']:.0f})" if p.get('rss_mb') else "memory ?"
        gdi = f"   GDI objects {p['gdi']}" if p.get('gdi') is not None else ""      # Windows only
        return f"{mem}   threads {p.get('threads')}{gdi}   up {up // 3600}:{up % 3600 // 60:02d}:{up % 60:02d}"

    @staticmethod
    def _effects_text(e):
        n = e.get('count') or 0
        if not n:
            return "none on the output"
        text = f"{n} on the output" + ("" if e.get('drawn', True) else " (not drawn: this render mode has none)")
        for name, msg in sorted((e.get('errors') or {}).items()):
            text += f"; {isf.title(name)} not drawn: {msg}"
        return text

    @staticmethod
    def _gc_text(g):
        """Python's garbage collector: its pauses hold up the render thread too."""
        if not g:
            return "?"
        ms = lambda v: "?" if v is None else f"{v:.0f}" if v >= 10 else f"{v:.1f}"  # noqa: E731
        return (f"{g.get('blocks', 0) / 1000:.0f}k objects   longest pause {ms(g.get('pause_max_ms_window'))} ms "
                f"(last {int(health.WINDOW_S)} s)   full collections {g.get('full_total', 0)}, "
                f"last {ms(g.get('full_pause_last_ms'))} ms, longest {ms(g.get('full_pause_max_ms'))} ms")

    # -- Preshow Check (preshow.py) ----------------------------------------------------
    def _preshow_facts(self):
        """What preshow.checks needs to know about the show besides Output Health."""
        import shutil as _sh
        out = self.screen_var.get() if getattr(self, 'screen_var', None) is not None else self.NONE
        size = None
        idx = getattr(self, 'selected_display', None)
        mons = getattr(self.re, 'monitors', None) or []
        if out not in (self.NONE, self.MULTIPLE) and idx is not None and 0 <= idx < len(mons):
            w, h = getattr(mons[idx], 'width', None), getattr(mons[idx], 'height', None)
            size = (w, h) if w and h else None
        cells = [(r, c) for (r, c), p in self.grid_cells.items() if p]
        positions = [self.cell_positions.get(k) or {} for k in cells]
        n_fx = sum(len(isf.layer_stack(p)) for p in positions)
        n_fx += sum(len(isf.stack(e)) for e in (self.output_effects or {}).values())
        where = os.path.dirname(self.current_project_path) if self.current_project_path else self.CONFIG_DIR
        try:
            free = _sh.disk_usage(where if os.path.isdir(where) else os.path.expanduser('~')).free
        except OSError:
            free = None
        return {'output': out, 'output_size': size, 'render_mode': getattr(self, 'render_mode', None),
                'cues': len({c for _r, c in cells}), 'effects': n_fx,
                'blends': sum(1 for p in positions if str(p.get('blend', 'normal')) != 'normal'),
                'dirty': bool(self.project_is_dirty), 'path': self.current_project_path,
                'locked': bool(getattr(self, 'edit_locked', False)),
                'test_pattern': getattr(self, '_test_pattern', None) is not None, 'disk_free': free,
                'presentation': bool(self.presentation_mode_var.get()) if getattr(
                    self, 'presentation_mode_var', None) is not None else False,
                'schedule': (self._schedule_text() or '') if (self.schedule or {}).get('enabled') else ''}

    def _preshow_rows(self):
        import preshow
        snap = self._current_health()
        if snap is not None:                   # now, not the cached list
            gone = self._missing_media(max_age=0)
            snap = dict(snap, missing_media=len(gone), missing_media_names=[os.path.basename(p) for p in gone])
        return preshow.checks(snap, self._preshow_facts())

    def _open_preshow(self):
        """Show > Preshow Check…: problems first, then what is fine. Changes nothing."""
        import preshow
        win = getattr(self, '_preshow_win', None)
        if win is not None:
            try:
                if win.winfo_exists():
                    win.lift()
                    win._vd_refresh()
                    return win
            except tk.TclError:
                pass
        win = tk.Toplevel(self.root)
        self._preshow_win = win
        win.title("Preshow Check")
        win.configure(bg=T.SURFACE)
        win.transient(self.root)
        win.geometry("640x520")
        head = Label(win, bg=T.SURFACE, fg=T.TEXT, font=T.font(14, 'bold'), anchor='w')
        head.pack(fill='x', padx=16, pady=(14, 2))
        Label(win, text="Before the house opens: fix the problems, look at the rest. Nothing here changes the show.",
              bg=T.SURFACE, fg=T.MUTED, anchor='w', font=T.font(9)).pack(fill='x', padx=16)
        btns = tk.Frame(win, bg=T.SURFACE)                 # packed first: always visible
        btns.pack(side='bottom', fill='x', padx=12, pady=(0, 12))
        text = tk.Text(win, wrap='word', bg=T.BG, fg=T.TEXT, relief='flat', borderwidth=0, highlightthickness=0,
                       padx=14, pady=10, cursor='arrow', font=T.font(10), spacing1=3, spacing3=3)
        text.pack(fill='both', expand=True, padx=12, pady=10)
        colors = {preshow.FAIL: '#ff5c5c', preshow.WARN: T.WARN, preshow.OK: '#5fbf5f'}
        marks = {preshow.FAIL: '\u2715', preshow.WARN: '!', preshow.OK: '\u2713'}
        for level, col in colors.items():
            text.tag_configure(level, foreground=col, font=T.font(11, 'bold'))
        text.tag_configure('title', font=T.font(10, 'bold'))
        text.tag_configure('detail', foreground=T.MUTED, lmargin1=26, lmargin2=26)
        state = {'rows': []}

        def refresh():
            rows = self._preshow_rows()
            state['rows'] = rows
            level, words = preshow.summary(rows)
            head.config(text=words, fg=colors[level])
            text.configure(state='normal')
            text.delete('1.0', 'end')
            for lv, title, detail in rows:
                text.insert('end', f"{marks[lv]}  ", lv)
                text.insert('end', title + "\n", 'title')
                if detail:
                    text.insert('end', detail + "\n", 'detail')
            text.configure(state='disabled')
            log.info(f"Preshow Check: {words}")
        win._vd_refresh = refresh
        win._vd_state = state

        def copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(preshow.as_text(state['rows']))
        ttk.Button(btns, text="Close", command=win.destroy).pack(side='right')
        ttk.Button(btns, text="Copy", command=copy).pack(side='right', padx=(0, 6))
        ttk.Button(btns, text="Check Again", command=refresh).pack(side='right', padx=(0, 6))
        self._help_link(btns, "Running the show", bg=T.SURFACE).pack(side='left')
        win.bind('<Escape>', lambda e: win.destroy())
        refresh()
        return win

    def _current_health(self):
        """Latest render-thread snapshot plus what only the UI knows, or None."""
        rt = getattr(self, 'render_thread', None)
        snap = getattr(rt, 'health_snapshot', None) if rt is not None else None
        if snap is None:
            return None
        snap = dict(snap)
        snap['render_thread_alive'] = rt.is_alive()
        snap['stale_s'] = round(max(0.0, time.time() - snap.get('time', time.time())), 1)
        gone = self._missing_media()
        snap['missing_media'] = len(gone)
        snap['missing_media_names'] = [os.path.basename(p) for p in gone]
        snap['remote'] = self._remote_status()
        snap['web'] = self._web_remote_status()
        snap['midi'] = self._midi_status()
        snap['ltc'] = self._ltc_status()
        snap['process'] = self._process_status()
        snap['audio'] = self._audio_status()
        snap['dmx'] = self._dmx_status()
        snap['schedule'] = self._schedule_text() or "off"
        snap['displays'] = {'count': len(self.re.monitors) if self.re else 0,
                            'changed': getattr(self, '_display_change', None)}
        sv = getattr(self, 'screen_var', None)
        if sv is not None and sv.get() == self.SPAN and self._span_displays():
            snap['displays']['span'] = span_mod.describe(self.re.monitors, self._span_clean())
        return snap

    def _update_health_indicator(self):
        if not hasattr(self, 'fps_label'):
            return
        snap = self._current_health()
        if snap is None:
            self.fps_label.config(text="Output: starting…", fg=self.HEALTH_COLORS['none'])
            return
        level, reasons = health.assess(snap)
        out = snap.get('output') or {}
        errs = sum((snap.get('errors_recent') or {}).values())
        rps = out.get('renders_per_s', 0) or 0
        # A still picture is drawn once, not every frame: "0 fps" read like a dead
        # output on the owner's PC (2026-11-14), so it says what it is.
        rate = "still picture" if rps < 1 and not snap.get('active_content') else f"{rps:.0f} fps"
        text = (f"\u25CF {level.upper()}   {rate}   "
                f"{out.get('late_window', 0)} late   {errs} error{'s' if errs != 1 else ''}")
        if reasons:
            text += "   \u2013 " + reasons[0] + (f" (+{len(reasons) - 1} more)" if len(reasons) > 1 else "")
        self.fps_label.config(text=text, fg=self.HEALTH_COLORS[level])

    def _health_report(self):
        """Everything in the Health window as one JSON-serializable dict."""
        snap = self._current_health()
        level, reasons = health.assess(snap) if snap else ('none', ['no data yet'])
        try:
            import logging_setup
            env = logging_setup.environment_report()
            log_dir = logging_setup.log_dir()
        except Exception:  # noqa: BLE001
            env, log_dir = {}, None
        return {'visualdeck': SPLASH_VERSION_TEXT, 'status': level, 'reasons': reasons,
                'show': self.current_project_path, 'snapshot': snap, 'environment': env, 'log_dir': log_dir}

    @staticmethod
    def _remote_text(r):
        if not r.get('enabled'):
            return "OSC off (Preferences)"
        if r.get('listening'):
            return f"OSC listening on {r.get('address')}   {r.get('received', 0)} messages"
        return f"OSC not listening: {r.get('error', '?')}"

    @staticmethod
    def _web_text(w):
        if not w.get('enabled'):
            return "browser remote off (Preferences)"
        if w.get('listening'):
            return f"page at http://{w.get('address')}/   {w.get('requests', 0)} requests"
        return f"browser remote not listening: {w.get('error', '?')}"

    @staticmethod
    def _loop_text(lp):
        """Where the render loop's time goes (p99, last 5 s) and ticks that did not render."""
        ph = lp.get('phase_ms') or {}
        if not ph:
            return "Loop       n/a"
        parts = "   ".join(f"{k} {ph[k]['p99']}" for k in ('pump', 'commands', 'preview', 'wait') if k in ph)
        return (f"Loop p99   {parts} ms   skipped {lp.get('skipped_active_ticks', 0)} of "
                f"{lp.get('active_ticks', 0)} ticks with video")

    @staticmethod
    def _decoder_row(d):
        where = 'On output' if d.get('live') else ('Next cue' if d.get('state') == 'held' else 'Other')
        hw = d.get('hw_engaged')
        dec = ('HW ' + str(d.get('hw_requested'))) if hw else ('SW' if hw is False else '?')
        if hw is False and d.get('hw_requested') not in (None, 'auto', 'none'):
            dec = f"SW (asked {d['hw_requested']})"
        m = d.get('media') or {}
        if m.get('codec'):
            size = m.get('size') or []
            sc = d.get('scaled_to')
            dec += f"  {m['codec']}" + (f" {size[0]}\u00d7{size[1]}" if len(size) == 2 else "") \
                + (f" scaled to {sc[0]}\u00d7{sc[1]}" if sc else "") \
                + (f" {m['pix_fmt']}" if m.get('pix_fmt') else "") \
                + (f"  threads {m['threads']}" if m.get('threads') else "")
        fmt = lambda v: '' if v is None else str(v)  # noqa: E731
        return (d.get('file', ''), d.get('instance', ''), where, d.get('state', ''), dec,
                fmt(d.get('source_fps')) + (f" x{d['speed']:g}" if d.get('speed') not in (None, 1, 1.0) else '')
                + {'reverse': ' rev', 'pingpong': ' ping-pong'}.get(d.get('direction'), ''),
                fmt(d.get('decode_fps')), fmt(d.get('frames_decoded')),
                fmt(d.get('frames_shown')), fmt(d.get('frames_dropped')), d.get('last_error') or '')

    def _open_health_window(self):
        existing = getattr(self, '_health_win', None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    return
            except Exception:  # noqa: BLE001
                pass
        win = tk.Toplevel(self.root)
        self._health_win = win
        win.title("Output Health")
        win.configure(bg=T.BG)
        win.transient(self.root)
        status = Label(win, bg=T.BG, fg='white', font=T.font(11, 'bold'), anchor='w', justify='left')
        status.pack(fill='x', padx=10, pady=(10, 0))
        reasons = Label(win, bg=T.BG, fg='#dddddd', anchor='w', justify='left', wraplength=900)
        reasons.pack(fill='x', padx=10)
        body = Label(win, bg=T.INPUT, fg='#e0e0e0', anchor='nw', justify='left', font=T.mono(9))
        body.pack(fill='x', padx=10, pady=8)
        # The buttons and the note are packed at the bottom first and the window keeps
        # its size once filled (refresh below), so a warning line that comes and goes
        # (e.g. the render target going 60 -> 45 -> 60) only changes the clip table's
        # height: the Close button no longer jumps away under the mouse (owner's PC,
        # 2026-11-16).
        btns = Frame(win, bg=T.BG)
        btns.pack(side='bottom', fill='x', padx=10, pady=(0, 10))
        note = Label(win, bg=T.BG, fg='#999999', anchor='w', justify='left', wraplength=900, text=(
            "Late frame: a gap longer than 1.5 output frames while video or a fade is on screen. "
            "Dropped: decoded frames replaced by a newer one before the output used them (normal when a clip's "
            "frame rate is higher than the output's). Decode: HW = hardware decoding engaged, SW = software."))
        note.pack(side='bottom', fill='x', padx=10, pady=6)
        cols = ('Clip', 'Instance', 'Where', 'State', 'Decode', 'Clip fps', 'Decode fps', 'Decoded', 'Shown',
                'Dropped', 'Last error')
        widths = (170, 80, 85, 70, 250, 70, 90, 80, 65, 80, 160)   # Decode: "h264 3840×2160 scaled to …"
        tree = ttk.Treeview(win, columns=cols, show='headings', height=8)
        for c, w in zip(cols, widths):
            tree.heading(c, text=c)
            tree.column(c, width=w, anchor='w', stretch=(c == 'Last error'))
        tree.pack(fill='both', expand=True, padx=10)

        def copy_report():
            try:
                text = json.dumps(self._health_report(), indent=2, default=str)
                win.clipboard_clear()
                win.clipboard_append(text)
                status_tail.config(text="Report copied to the clipboard.")
            except Exception as e:  # noqa: BLE001
                status_tail.config(text=f"Could not copy the report: {e}")

        def open_logs():
            try:
                import logging_setup
                d = logging_setup.log_dir()
                if sys.platform == 'win32':
                    os.startfile(d)  # noqa: S606 - opens Explorer on the log folder
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', d])
                else:
                    subprocess.Popen(['xdg-open', d])
            except Exception as e:  # noqa: BLE001
                status_tail.config(text=f"Could not open the log folder: {e}")

        Button(btns, text="Copy Report", command=copy_report).pack(side='left')
        Button(btns, text="Open Log Folder", command=open_logs).pack(side='left', padx=6)
        Button(btns, text="Rescan Displays", command=lambda: self._rescan_displays()).pack(side='left')
        status_tail = Label(btns, bg=T.BG, fg='#cccccc')
        status_tail.pack(side='left', padx=6)

        def close():
            self._health_win = None
            win.destroy()
        Button(btns, text="Close", command=close).pack(side='right')
        win.protocol("WM_DELETE_WINDOW", close)
        win.bind('<Escape>', lambda e: close())         # like the other windows

        sized = []

        def refresh():
            if self._health_win is not win:
                return
            try:
                self._fill_health_window(status, reasons, body, tree)
            except Exception:
                log.exception("Health window refresh failed")
            if not sized:                 # keep this size from now on (room for 3 more warning lines)
                sized.append(True)
                try:
                    win.update_idletasks()
                    extra = 3 * max(12, int(reasons.winfo_reqheight() / max(1, reasons.cget('text').count('\n') + 1)))
                    win.geometry(f"{win.winfo_reqwidth()}x{win.winfo_reqheight() + extra}")
                except Exception:  # noqa: BLE001 - only the size
                    pass
            win.after(500, refresh)
        refresh()

    def _fill_health_window(self, status, reasons, body, tree):
        snap = self._current_health()
        if snap is None:
            status.config(text="No data yet (output engine starting)", fg=self.HEALTH_COLORS['none'])
            return
        level, why = health.assess(snap)
        status.config(text=f"\u25CF {level.upper()}", fg=self.HEALTH_COLORS[level])
        reasons.config(text="\n".join(why) if why else "No problems detected.")
        out = snap.get('output') or {}
        iv, rc = out.get('interval_ms') or {}, out.get('render_ms') or {}
        gl = snap.get('gl') or {}
        pre = snap.get('preroll') or {}
        errs_t, errs_r = snap.get('errors_total') or {}, snap.get('errors_recent') or {}
        last = snap.get('last_error')
        lines = [
            f"Output     {out.get('renders_per_s', 0)} renders/s   target {snap.get('target_fps')} fps   "
            f"{'video/fade on screen' if snap.get('active_content') else 'static'}",
            f"Intervals  p50 {iv.get('p50')} ms   p99 {iv.get('p99')} ms   max {iv.get('max')} ms   "
            f"(last {int(health.WINDOW_S)} s)",
            f"Render     p50 {rc.get('p50')} ms   p99 {rc.get('p99')} ms   max {rc.get('max')} ms",
            self._loop_text(snap.get('loop') or {}),
            f"Late       {out.get('late_window', 0)} of {out.get('frames_window', 0)} in the last "
            f"{int(health.WINDOW_S)} s   {out.get('late_total', 0)} of {out.get('frames_total', 0)} since start",
            f"Errors     {sum(errs_r.values())} in the last minute   {sum(errs_t.values())} since start"
            + (f"   by phase: {errs_t}" if errs_t else ""),
        ]
        if last:
            lines.append(f"Last error {last['phase']}: {last['message']}")
        lines += [
            f"Cuts       {pre.get('cuts_total', 0)}   last preroll wait {pre.get('last_wait_ms')} ms   "
            f"max {pre.get('max_wait_ms')} ms   timeouts (last minute) {pre.get('timeouts_recent', 0)}",
            f"Media      {snap.get('missing_media', 0)} missing" + health.names_text(snap.get('missing_media_names')),
            "Remote     " + self._remote_text(snap.get('remote') or {}),
            "Browser    " + self._web_text(snap.get('web') or {}),
            "MIDI       " + self._midi_text(snap.get('midi') or {}),
            "LTC        " + self._ltc_text(snap.get('ltc') or {}),
            "Process    " + self._process_text(snap.get('process') or {}),
            "Python     " + self._gc_text(snap.get('gc') or {}),
            "Effects    " + self._effects_text(snap.get('effects') or {}),
            "Audio      " + self._audio_text(snap.get('audio') or {}).replace("Audio ", "", 1),
            "DMX        " + self._dmx_text(snap.get('dmx') or {}),
            "Schedule   " + str(snap.get('schedule', 'off')),
            "Displays   " + self._displays_text(snap.get('displays') or {}),
            f"Engine     {snap.get('engine')}   GPU: {gl.get('renderer', 'n/a')}   ({gl.get('vendor', '')} "
            f"{gl.get('version', '')})".rstrip(),
        ]
        body.config(text="\n".join(lines))
        rows = [self._decoder_row(d) for d in snap.get('decoders') or []]
        existing = tree.get_children()
        for i, row in enumerate(rows):
            if i < len(existing):
                tree.item(existing[i], values=row)
            else:
                tree.insert('', 'end', values=row)
        for item in existing[len(rows):]:
            tree.delete(item)

    def _all_media_paths(self):
        """Every media path the show refers to (media list and grid), unique, in order."""
        seen, out = set(), []
        for p in list(self.project_media_paths) + [p for _, p in sorted(self.grid_cells.items()) if p]:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _media_info_for_save(self, save_path):
        out = {}
        for p in self._all_media_paths():
            if live_source.is_live(p):
                continue                     # not a file: nothing to relink
            if os.path.exists(p):
                out[p] = media_relink.media_info_for(p, save_path)
            elif p in self.media_info:
                # Keep what we knew about a missing file, re-expressing its
                # relative path if the show is being saved somewhere else.
                entry = dict(self.media_info[p])
                rel = entry.pop('rel', None)
                if rel and self.current_project_path:
                    rel = media_relink.rebase_rel(rel, self.current_project_path, save_path)
                elif rel and not self.current_project_path:
                    rel = None
                if rel:
                    entry['rel'] = rel
                out[p] = entry
        return out

    @staticmethod
    def _video_summary(path):
        """(width, height, fps, codec, seconds between key frames (from the first
        ones) or None) of a video file; None if it cannot be read."""
        try:
            import av
            with av.open(path) as c:
                v = c.streams.video[0]
                fps = float(v.average_rate or v.guessed_rate or 0) or None
                keys, n = [], 0
                for p in c.demux(v):
                    if p.size:
                        if p.is_keyframe:
                            keys.append(n)
                        n += 1
                        if n >= 900 or len(keys) >= 4:
                            break
                gap = (keys[1] - keys[0]) / fps if len(keys) >= 2 and fps else None
                return v.codec_context.width, v.codec_context.height, fps, v.codec_context.name, gap
        except Exception:  # noqa: BLE001
            return None

    def _use_playback_copy(self, src, dst):
        """Point every cue at the copy instead of `src`, with layer sizes kept:
        a layer's scale is in the media's own pixels, so a smaller copy gets a
        larger scale. Returns _apply_relink's result."""
        a, b = self._video_summary(src), self._video_summary(dst)
        if a and b and (a[0], a[1]) != (b[0], b[1]):
            fx, fy = a[0] / b[0], a[1] / b[1]
            for rc, p in self.grid_cells.items():
                if p == src and rc in self.cell_positions:
                    pos = self.cell_positions[rc]
                    pos['scale_x'] = pos.get('scale_x', 1.0) * fx
                    pos['scale_y'] = pos.get('scale_y', 1.0) * fy
        res = self._apply_relink({src: dst})
        self.media_info.pop(dst, None)             # its own size and length, not the original's
        return res

    @_locked_out
    def _open_playback_copy(self, path):
        """Media > Make Playback Copy...: a new file with a key frame every half
        second (or every frame), optionally smaller (playback_copy.py)."""
        win = tk.Toplevel(self.root)
        win.title(f"Playback Copy \u2013 {os.path.basename(path)}")
        win.configure(bg=T.SURFACE)
        win.transient(self.root)
        body = tk.Frame(win, bg=T.SURFACE, padx=14, pady=14)
        body.pack(fill='both', expand=True)
        summ = self._video_summary(path)
        if summ:
            w, h, fps, codec, gap = summ
            gap_t = (f"a key frame every {gap:.1f} s" if gap else "one key frame in its first 900 frames")
            src_t = f"This video: {w}\u00d7{h}, {fps or 0:g} fps, {codec}, {gap_t}."
        else:
            src_t = "This video could not be read."
        tk.Label(body, bg=T.SURFACE, fg=T.TEXT, justify='left', anchor='w', wraplength=440, text=(
            src_t + " A playback copy starts quickly anywhere (Start Cue From, in points, reverse) and can be "
            "smaller, so it is lighter to decode. It is a new file next to this one; this file is not changed."
        )).pack(fill='x')
        form = tk.Frame(body, bg=T.SURFACE)
        form.pack(fill='x', pady=(10, 0))
        size_var = tk.StringVar(value=playback_copy.SIZES[1][0] if summ and summ[0] > 1920 else
                                playback_copy.SIZES[0][0])
        key_var = tk.StringVar(value=playback_copy.KEYFRAMES[0][0])
        tk.Label(form, text="Size", bg=T.SURFACE, fg=T.TEXT).grid(row=0, column=0, sticky='w', pady=3)
        ttk.Combobox(form, textvariable=size_var, values=[s[0] for s in playback_copy.SIZES], state='readonly',
                     width=34).grid(row=0, column=1, sticky='w')
        tk.Label(form, text="Key frames", bg=T.SURFACE, fg=T.TEXT).grid(row=1, column=0, sticky='w', pady=3)
        ttk.Combobox(form, textvariable=key_var, values=[k[0] for k in playback_copy.KEYFRAMES], state='readonly',
                     width=34).grid(row=1, column=1, sticky='w')
        use_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(body, text="Use the copy in the cues that use this video (layer sizes are kept)",
                        variable=use_var).pack(anchor='w', pady=(8, 0))
        bar = ttk.Progressbar(body, maximum=1.0, length=440)
        bar.pack(fill='x', pady=(10, 0))
        status = tk.Label(body, bg=T.SURFACE, fg=T.MUTED, justify='left', anchor='w', wraplength=440)
        status.pack(fill='x', pady=(6, 0))
        st = {'running': False, 'cancel': False, 'result': None}
        q = queue.Queue()

        def work(dst, size, key_s):
            t0 = time.monotonic()
            try:
                r = playback_copy.make(path, dst, max_size=size, keyframe_s=key_s,
                                       progress=lambda f: q.put(('progress', f)), canceled=lambda: st['cancel'])
                q.put(('done', dict(r, seconds=time.monotonic() - t0, path=dst)))
            except playback_copy.Canceled:
                q.put(('canceled', None))
            except Exception as e:  # noqa: BLE001
                log.exception("Playback copy failed")
                q.put(('error', str(e)))

        def poll():
            done = None
            try:
                while True:
                    kind, val = q.get_nowait()
                    if kind == 'progress':
                        bar['value'] = val
                    else:
                        done = (kind, val)
            except queue.Empty:
                pass
            if done is None:
                if st['running'] and win.winfo_exists():
                    win.after(200, poll)
                return
            st['running'] = False
            kind, val = done
            if not win.winfo_exists():
                return
            make_btn.config(state='normal')
            if kind == 'done':
                st['result'] = val
                gop = "every frame" if val['gop'] == 1 else f"every {val['gop']} frames"
                msg = (f"Made {os.path.basename(val['path'])}: {val['size'][0]}\u00d7{val['size'][1]}, a key frame "
                       f"{gop}{', with sound' if val['audio'] else ''}, in {val['seconds']:.0f} s.")
                if use_var.get():
                    res = self._use_playback_copy(path, val['path'])
                    msg += f" Used in {res['cells']} cell(s)."
                    if res['live_affected']:
                        msg += " The live cue shows it the next time it goes live."
                else:
                    if val['path'] not in self.project_media_paths:
                        i = self.project_media_paths.index(path) + 1 if path in self.project_media_paths else len(
                            self.project_media_paths)
                        self.project_media_paths.insert(i, val['path'])
                        self._build_media_list()
                        self._set_dirty_flag()
                status.config(fg=T.GO, text=msg)
                log.info(f"Playback copy: {msg}")
            elif kind == 'canceled':
                bar['value'] = 0
                status.config(fg=T.MUTED, text="Stopped; nothing was written.")
            else:
                status.config(fg='#ff8888', text=f"Could not make the copy: {val}")

        def make():
            if st['running']:
                return None
            size = dict(playback_copy.SIZES)[size_var.get()]
            key_s = dict(playback_copy.KEYFRAMES)[key_var.get()]
            dst = playback_copy.target_path(path, size)
            st.update(running=True, cancel=False)
            make_btn.config(state='disabled')
            status.config(fg=T.MUTED, text=f"Making {os.path.basename(dst)}\u2026")
            threading.Thread(target=work, args=(dst, size, key_s), daemon=True, name="PlaybackCopy").start()
            win.after(200, poll)
            return dst

        def close():
            st['cancel'] = True
            win.destroy()

        btns = tk.Frame(body, bg=T.SURFACE)
        btns.pack(fill='x', pady=(12, 0))
        tk.Button(btns, text="Close", width=9, command=close).pack(side='right', padx=(6, 0))
        ttk.Button(btns, text="Stop", command=lambda: st.update(cancel=True)).pack(side='right')
        make_btn = tk.Button(btns, text="Make", width=9, bg=T.GO, fg='black', command=make)
        make_btn.pack(side='left')
        win.protocol("WM_DELETE_WINDOW", close)
        win.bind('<Escape>', lambda e: close())
        win._vd = {'size_var': size_var, 'key_var': key_var, 'use_var': use_var, 'make': make, 'st': st,
                   'status': status, 'poll': poll}                     # for tests
        return win

    def _apply_relink(self, mapping):
        """Point the show at relocated media. Does not change what is on the
        outputs: if the live cue uses a relinked file it appears the next time
        that cue is triggered."""
        if not mapping:
            return {'cells': 0, 'live_affected': False}
        affected = [rc for rc, p in self.grid_cells.items() if p in mapping]
        live_affected = self.transition_active or any(c == self.active_col for (_, c) in affected)
        n = media_relink.apply_mapping(mapping, self.grid_cells, self.project_media_paths, self.media_info)
        for old in mapping:
            self.thumbnail_cache.pop(old, None)
        for old, new in mapping.items():
            log.info(f"Media relinked: {old} -> {new}")
        self._build_media_list()
        for (r, c) in affected:
            self._update_cell_preview(r, c)
        if not live_affected:
            self._update_active_players()  # refresh the preloaded next cue
        self._set_dirty_flag()
        self._missing_cache = None
        return {'cells': n, 'live_affected': live_affected}

    def _locate_missing(self, old_path, parent=None, missing=None, hint_dir=None):
        """Let the operator pick the new location of one missing file; other
        missing files with the same names in that folder are relinked too."""
        ext = os.path.splitext(old_path)[1]
        new = filedialog.askopenfilename(
            parent=parent or self.root,
            title=f"Locate {media_relink.basename(old_path)}",
            initialdir=hint_dir or None,
            filetypes=[("Media", f"*{ext}"), ("All files", "*.*")] if ext else [("All files", "*.*")])
        if not new:
            return {}
        new = os.path.normpath(new)
        others = [m for m in (missing if missing is not None else media_relink.find_missing(self._all_media_paths()))
                  if m != old_path]
        mapping = media_relink.siblings_mapping(new, others, self.media_info)
        mapping[old_path] = new
        self._apply_relink(mapping)
        return mapping

    def _open_relink_dialog(self, on_open=False):
        """Non-modal window listing missing media with search/locate actions."""
        if not getattr(self, "_ui_ready", True):
            self.root.after(500, lambda: self._open_relink_dialog(on_open=on_open))
            return
        missing = media_relink.find_missing(self._all_media_paths())
        if not missing:
            if not on_open:
                messagebox.showinfo("Relink Media", "No missing media in this show.", parent=self.root)
            return
        existing = getattr(self, "_relink_win", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            return

        win = tk.Toplevel(self.root)
        self._relink_win = win
        win.title("Missing Media")
        win.configure(bg=T.BG)
        win.transient(self.root)
        state = {'missing': missing, 'ambiguous': {}, 'closed': False, 'busy': False}
        results_q = queue.Queue()

        header = Label(win, bg=T.BG, fg='white', justify='left', anchor='w')
        header.pack(fill='x', padx=10, pady=(10, 4))
        body = Frame(win, bg=T.BG)
        body.pack(fill='both', expand=True, padx=10)
        lb = tk.Listbox(body, width=100, height=min(14, max(4, len(missing))), selectmode='extended',
                        bg=T.INPUT, fg='#ff8888', selectbackground='#555', activestyle='none')
        sb = ttk.Scrollbar(body, orient='vertical', command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        status = Label(win, bg=T.BG, fg='#cccccc', justify='left', anchor='w', wraplength=700, text=(
            "Search a folder: files are matched by name (and size when known); when several files share a "
            "name, the one in the most similar folder is used, and ties are left for you to choose."))
        status.pack(fill='x', padx=10, pady=6)
        btns = Frame(win, bg=T.BG)
        btns.pack(fill='x', padx=10, pady=(0, 10))

        def refresh():
            state['missing'] = media_relink.find_missing(self._all_media_paths())
            n = len(state['missing'])
            header.config(text=f"{n} media file{'s' if n != 1 else ''} used by this show can't be found."
                          if n else "All media found.")
            lb.delete(0, 'end')
            for m in state['missing']:
                tag = "   (several matches)" if m in state['ambiguous'] else ""
                lb.insert('end', m + tag)

        def set_busy(busy):
            state['busy'] = busy
            for b in (b_search, b_locate):
                b.config(state='disabled' if busy else 'normal')

        def search():
            folder = filedialog.askdirectory(parent=win, title="Search for missing media in…", mustexist=True)
            if not folder:
                return
            set_busy(True)
            status.config(text=f"Searching {folder} …")
            todo = list(state['missing'])
            info = dict(self.media_info)

            def work():
                try:
                    res = media_relink.search_folder(todo, folder, info, cancel=lambda: state['closed'])
                except Exception as e:  # noqa: BLE001 - report, don't crash the UI
                    log.exception("Media search failed")
                    res = e
                results_q.put(res)
            threading.Thread(target=work, name="MediaRelinkSearch", daemon=True).start()
            poll()

        def poll():
            if state['closed']:
                return
            try:
                res = results_q.get_nowait()
            except queue.Empty:
                win.after(100, poll)
                return
            set_busy(False)
            if isinstance(res, Exception):
                status.config(text=f"Search failed: {res}")
                return
            outcome = self._apply_relink(res.found)
            state['ambiguous'].update(res.ambiguous)
            for old in res.found:
                state['ambiguous'].pop(old, None)
            refresh()
            text = media_relink.summarize(res, len(state['missing']))
            if outcome['live_affected']:
                text += " The live cue uses relinked media; it will appear when that cue is next triggered."
            status.config(text=text)

        def locate():
            sel = lb.curselection()
            if not sel:
                status.config(text="Select a file in the list first.")
                return
            old = state['missing'][sel[0]]
            cands = state['ambiguous'].get(old) or []
            hint = os.path.dirname(cands[0]) if cands else None
            mapping = self._locate_missing(old, parent=win, missing=state['missing'], hint_dir=hint)
            if mapping:
                for k in mapping:
                    state['ambiguous'].pop(k, None)
                refresh()
                extra = len(mapping) - 1
                status.config(text=f"Relinked {os.path.basename(mapping[old])}"
                              + (f" and {extra} more found in the same folder." if extra else "."))

        def close():
            state['closed'] = True
            self._relink_win = None
            win.destroy()

        b_search = Button(btns, text="Search Folder…", command=search)
        b_search.pack(side='left')
        b_locate = Button(btns, text="Locate Selected…", command=locate)
        b_locate.pack(side='left', padx=6)
        Button(btns, text="Close", command=close).pack(side='right')
        lb.bind('<Double-Button-1>', lambda e: locate())
        win.protocol("WM_DELETE_WINDOW", close)
        refresh()

    def _save_project(self):
        if not self.current_project_path:
            return self._save_project_as()
        return self._perform_save(self.current_project_path)

    def _save_project_as(self):
        path = filedialog.asksaveasfilename(
            title="Save Project As",
            initialfile="Untitled.vjdeck",
            defaultextension=".vjdeck",
            filetypes=[("VisualDeck Project", "*.vjdeck"), ("All files", "*.*")]
        )
        if path and self._perform_save(path):
            self._add_to_recent(path)
            self._save_config()
            return True
        return False

    def _perform_save(self, path):
        """Gathers all state and writes it to a JSON file."""
        data = self._build_save_data(path)
        try:
            atomic_write_json(path, data, keep_backup=True, indent=4)
        except Exception as e:
            log.exception(f"Could not save project file: {path}")
            messagebox.showerror("Save Error", f"Could not save project file:\n{e}\n\nThe previous version on disk was not modified.")
            return False
        previous = self.current_project_path
        self.current_project_path = path
        self.project_is_dirty = False
        self._update_title()
        log.info(f"Project saved to {path}")
        self._discard_autosave(path)
        if previous != path:
            self._discard_autosave(previous)      # e.g. an untitled show saved for the first time
        return True

    # -- autosave and recovery -------------------------------------------------
    # Unsaved changes are written every AUTOSAVE_PERIOD_MS to a separate file in
    # ~/.visualdeck/autosave (never to the show file). Saving, or answering "No"
    # to "Save changes?", removes it. If VisualDeck stops without either, the
    # next time the show is opened (or, for an untitled show, at start) the
    # operator is offered the newer autosaved version.
    AUTOSAVE_PERIOD_MS = 120_000

    def _autosave_path(self, show_path):
        if show_path:
            name = os.path.splitext(os.path.basename(show_path))[0]
            key = hashlib.sha1(os.path.abspath(show_path).encode('utf-8')).hexdigest()[:8]
        else:
            name, key = 'Untitled', 'untitled'
        safe = re.sub(r'[^\w\- ]', '_', name)[:60] or 'show'
        return os.path.join(self.CONFIG_DIR, 'autosave', f"{safe}-{key}.vjdeck")

    def _autosave_now(self):
        """Write unsaved changes to the autosave file. Returns its path, or None."""
        if not self.project_is_dirty:
            return None
        show = self.current_project_path
        path = self._autosave_path(show)
        try:
            data = self._build_save_data(show or path)
            data['autosave_of'] = show or None
            data['autosaved_at'] = time.time()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            atomic_write_json(path, data, keep_backup=False, indent=1)
        except Exception:  # noqa: BLE001 - an autosave must never disturb the show
            log.exception(f"Autosave to {path} failed")
            return None
        log.info(f"Autosaved unsaved changes to {path}")
        return path

    def _autosave_tick(self):
        try:
            self._autosave_now()
        finally:
            self.root.after(self.AUTOSAVE_PERIOD_MS, self._autosave_tick)

    def _discard_autosave(self, show_path):
        p = self._autosave_path(show_path)
        try:
            os.remove(p)
            log.info(f"Removed autosave {p}")
        except FileNotFoundError:
            pass
        except OSError as e:
            log.warning(f"Could not remove autosave {p}: {e}")

    def _recoverable_autosave(self, show_path):
        """(autosave path, data, time) if an autosave newer than the show exists, else None."""
        p = self._autosave_path(show_path)
        if not os.path.exists(p):
            return None
        try:
            with open(p, encoding='utf-8') as f:
                data = json.load(f)
            t = float(data.get('autosaved_at'))
        except Exception as e:  # noqa: BLE001
            log.warning(f"Ignoring unreadable autosave {p}: {e}")
            return None
        if show_path and os.path.exists(show_path) and t <= os.path.getmtime(show_path):
            log.info(f"Autosave {p} is older than the show; removed")
            self._discard_autosave(show_path)
            return None
        return p, data, t

    def _offer_recovery(self, show_path):
        """Ask whether to load the autosaved version; True if it was loaded."""
        found = self._recoverable_autosave(show_path)
        if found is None:
            return False
        p, data, t = found
        when = time.strftime('%Y-%m-%d %H:%M', time.localtime(t))
        what = f"“{os.path.basename(show_path)}”" if show_path else "an untitled show"
        if not messagebox.askyesno(
                "Recover unsaved changes?",
                f"VisualDeck found changes to {what} that were not saved (autosaved {when}). VisualDeck probably "
                f"closed unexpectedly.\n\nRecover them? They are not saved into the show until you save it.\n\n"
                f"No keeps the show as it was saved; the autosave is set aside as\n{p[:-7]}.declined.vjdeck",
                parent=self.root):
            try:
                os.replace(p, p[:-7] + '.declined.vjdeck')
            except OSError as e:
                log.warning(f"Could not set the autosave aside: {e}")
            log.info(f"Autosave for {what} declined")
            return False
        data = {k: v for k, v in data.items() if k not in ('autosave_of', 'autosaved_at')}
        self._apply_project_data(data, show_path or p)
        self.current_project_path = show_path or None
        self.project_is_dirty = True
        self._update_title()
        log.info(f"Recovered unsaved changes for {what} from {p}")
        return True

    def _build_save_data(self, path):
        """The show as saved to `path` (reads Tk variables: call on the UI thread)."""
        data = {}
        data.update(self._unknown_project_keys)   # keep what a newer version wrote
        data['version'] = '2.8.3'
        data['schema_version'] = project_schema.CURRENT
        data['cue_settings'] = project_schema.clean_cue_settings(self.cue_settings)
        data['go_at_end'] = self.go_at_end
        if self.schedule['events'] or self.schedule['enabled']:
            data['schedule'] = self.schedule
        data['settings'] = {'render_mode': self.render_mode, 'hw_accel': self.hw_accel}
        data['grid'] = {'rows': self.ROWS, 'cols': self.COLS, 'active_cue': self.active_col}
        data['cues'] = self.cue_names
        data['layers'] = self.layer_names
        data['project_media'] = self.project_media_paths
        # Portable hints for relinking on another machine; ignored by older versions.
        data['media_info'] = self._media_info_for_save(path)
        data['outputs'] = {'mode': self.screen_var.get(), 'layer_screens': self.layer_screens}
        if self.output_warps:
            data['outputs']['warps'] = self.output_warps
        if self.output_blends:
            data['outputs']['blends'] = self.output_blends
        if self.output_meshes:
            data['outputs']['meshes'] = self.output_meshes
        if self.output_effects:
            data['outputs']['effects'] = self.output_effects
        if self.output_span:
            data['outputs']['span'] = self.output_span
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
        return data

    def _effect_names(self):
        """File names of the shader effects the show uses (layers and displays)."""
        names = set()
        for pos in self.cell_positions.values():
            names.update(fx['name'] for fx in isf.layer_stack(pos))
        for raw in self.output_effects.values():
            names.update(fx['name'] for fx in isf.stack(raw))
        return names

    # --- Collect Show ---
    def _collect_show(self):
        """File > Collect Show: copy the show and its media into one new folder."""
        parent = filedialog.askdirectory(parent=self.root, title="Collect the show into which folder?", mustexist=True)
        if not parent:
            return
        name = (os.path.splitext(os.path.basename(self.current_project_path))[0]
                if self.current_project_path else "Untitled Show")
        folder = show_collect.unique_folder(parent, name)
        media = [p for p in self._all_media_paths() if not live_source.is_live(p)]   # live sources stay as they are
        mapping = show_collect.plan(media, os.path.join(folder, 'media'))
        fx_files = isf.custom_files(self._effect_names())
        effects = show_collect.effect_plan(fx_files.values(), os.path.join(folder, 'effects'))
        needed = sum(os.path.getsize(p) for p in list(media) + list(effects) if os.path.isfile(p))
        try:
            show_collect.check_destination(folder, needed)
        except show_collect.CollectError as e:
            messagebox.showerror("Collect Show", str(e), parent=self.root)
            return
        show_path = os.path.join(folder, name + '.vjdeck')
        data = self._build_save_data(show_path)       # UI thread; the worker only copies and writes
        log.info(f"Collect Show: {len(mapping)} media file(s), {show_collect.human_size(needed)} -> {folder}")

        win = tk.Toplevel(self.root)
        win.title("Collect Show")
        win.configure(bg=T.BG, padx=12, pady=12)
        win.transient(self.root)
        msg = Label(win, bg=T.BG, fg='white', anchor='w', justify='left', wraplength=460,
                    text=f"Copying {len(mapping)} file(s) ({show_collect.human_size(needed)}) to\n{folder}")
        msg.pack(fill='x')
        bar = ttk.Progressbar(win, length=460, maximum=max(1, needed))
        bar.pack(fill='x', pady=8)
        now = Label(win, bg=T.BG, fg='#bbbbbb', anchor='w')
        now.pack(fill='x')
        state = {'cancel': False, 'result': None, 'error': None, 'progress': (0, needed, '')}

        def work():
            try:
                state['result'] = show_collect.collect(
                    data, mapping, show_path,
                    progress=lambda d, t, n: state.__setitem__('progress', (d, t, n)),
                    cancel=lambda: state['cancel'], effects=effects)
            except Exception as e:  # noqa: BLE001 - report, never crash the UI
                log.exception("Collect Show failed")
                state['error'] = e
        threading.Thread(target=work, name="CollectShow", daemon=True).start()

        def cancel():
            state['cancel'] = True
            cbtn.config(state='disabled', text="Canceling…")
        cbtn = Button(win, text="Cancel", command=cancel)
        cbtn.pack(anchor='e')
        win.protocol("WM_DELETE_WINDOW", cancel)

        def poll():
            d, t, n = state['progress']
            bar.config(value=d)
            now.config(text=f"{d / 1e6:,.0f} of {t / 1e6:,.0f} MB   {n}")
            if state['result'] is None and state['error'] is None:
                win.after(100, poll)
                return
            win.destroy()
            self._collect_finished(folder, state['result'], state['error'])
        poll()

    def _collect_finished(self, folder, result, error):
        if error is not None:
            messagebox.showerror("Collect Show", f"Collecting stopped: {error}\n\nNothing was overwritten. {folder} "
                                 "may contain a partial copy.", parent=self.root)
            return
        if result['canceled']:
            messagebox.showinfo("Collect Show", f"Canceled. {folder} contains a partial copy and is not a "
                                "usable show; you can delete it.", parent=self.root)
            return
        fx = result.get('effects') or 0
        text = (f"Collected {result['copied']} file(s) ({show_collect.human_size(result['bytes'])}) "
                + (f"{fx} shader effect(s) " if fx else "") + f"and the show into\n{folder}")
        if result['missing']:
            names = ", ".join(media_relink.basename(p) for p in result['missing'][:5])
            more = f" and {len(result['missing']) - 5} more" if len(result['missing']) > 5 else ""
            text += f"\n\n{len(result['missing'])} missing file(s) could not be copied: {names}{more}."
        log.info(text.replace("\n", " "))
        if messagebox.askyesno("Collect Show", text + "\n\nOpen the folder?", parent=self.root):
            try:
                if sys.platform == 'win32':
                    os.startfile(folder)  # noqa: S606
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', folder])
                else:
                    subprocess.Popen(['xdg-open', folder])
            except Exception as e:  # noqa: BLE001
                log.warning(f"Could not open {folder}: {e}")

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
                    mode = config.get('live_edit_mode', 'blind')
                    self.live_edit_mode = mode if mode in ('blind', 'live') else 'blind'
                    self.osc_enabled = bool(config.get('osc_enabled', False))
                    try:
                        port = int(config.get('osc_port', 7700))
                        self.osc_port = port if 1 <= port <= 65535 else 7700
                    except (TypeError, ValueError):
                        self.osc_port = 7700
                    self.osc_network = bool(config.get('osc_network', False))
                    self.web_enabled = bool(config.get('web_enabled', False))
                    try:
                        wport = int(config.get('web_port', 7701))
                        self.web_port = wport if 1 <= wport <= 65535 else 7701
                    except (TypeError, ValueError):
                        self.web_port = 7701
                    self.web_network = bool(config.get('web_network', False))
                    self.midi_enabled = bool(config.get('midi_enabled', False))
                    self.midi_port = str(config.get('midi_port') or '')
                    self.midi_mapping = midi_remote.clean_mapping(config.get('midi_mapping'))
                    self.midi_faders = midi_remote.clean_faders(config.get('midi_faders'))
                    self.midi_program_change = bool(config.get('midi_program_change', True))
                    self.midi_timecode = bool(config.get('midi_timecode', False))
                    self.midi_msc = bool(config.get('midi_msc', False))
                    dev = config.get('midi_msc_device', 0)
                    self.midi_msc_device = dev if isinstance(dev, int) and not isinstance(dev, bool) \
                        and 0 <= dev <= 111 else 0
                    self.ltc_enabled = bool(config.get('ltc_enabled', False))
                    self.ltc_device = str(config.get('ltc_device') or '')
                    ch = config.get('ltc_channel', 'Left')
                    self.ltc_channel = ch if ch in ltc.CHANNELS else 'Left'
                    dev = config.get('audio_device', '')
                    self.audio_device = dev if isinstance(dev, str) else ''
                    try:
                        self.audio_delay_ms = max(0, min(1000, int(config.get('audio_delay_ms', 0))))
                    except (TypeError, ValueError):
                        self.audio_delay_ms = 0
                    self.dmx_enabled = bool(config.get('dmx_enabled', False))
                    proto = config.get('dmx_protocol', 'artnet')
                    self.dmx_protocol = proto if proto in dmx_remote.PROTOCOLS else 'artnet'
                    try:
                        u = int(config.get('dmx_universe', 0))
                        self.dmx_universe = u if 0 <= u <= 63999 else 0
                        ch = int(config.get('dmx_channel', 1))
                        self.dmx_channel = ch if 1 <= ch <= 512 - dmx_remote.FOOTPRINT + 1 else 1
                    except (TypeError, ValueError):
                        self.dmx_universe, self.dmx_channel = 0, 1
                    self.dmx_network = bool(config.get('dmx_network', True))
                    self.dmx_out_enabled = bool(config.get('dmx_out_enabled', False))
                    self.dmx_out_protocol = config.get('dmx_out_protocol', 'artnet') \
                        if config.get('dmx_out_protocol') in dmx_remote.PROTOCOLS else 'artnet'
                    try:
                        self.dmx_out_universe = max(0, min(63999, int(config.get('dmx_out_universe', 0))))
                        self.dmx_out_channel = max(1, min(511, int(config.get('dmx_out_channel', 1))))
                    except (TypeError, ValueError):
                        self.dmx_out_universe, self.dmx_out_channel = 0, 1
                    self.dmx_out_target = str(config.get('dmx_out_target') or '255.255.255.255')
        except (json.JSONDecodeError, IOError, AttributeError, TypeError) as e:
            log.warning(f"Ignoring unreadable config {self.config_path}: {e}")
            self.recent_files = []

    def _save_config(self):
        config_data = {
            'recent_files': self.recent_files,
            'render_mode': self.render_mode,
            'hw_accel': self.hw_accel,
            'use_pbo': self.use_pbo,
            'use_vsync': self.use_vsync,
            'force_multithread': self.force_multithread,
            'live_edit_mode': self.live_edit_mode,
            'osc_enabled': self.osc_enabled,
            'osc_port': self.osc_port,
            'osc_network': self.osc_network,
            'web_enabled': self.web_enabled,
            'web_port': self.web_port,
            'web_network': self.web_network,
            'midi_enabled': self.midi_enabled,
            'midi_port': self.midi_port,
            'midi_mapping': self.midi_mapping,
            'midi_faders': self.midi_faders,
            'midi_program_change': self.midi_program_change,
            'midi_timecode': self.midi_timecode,
            'midi_msc': self.midi_msc,
            'midi_msc_device': self.midi_msc_device,
            'ltc_enabled': self.ltc_enabled,
            'ltc_device': self.ltc_device,
            'ltc_channel': self.ltc_channel,
            'audio_device': self.audio_device,
            'audio_delay_ms': self.audio_delay_ms,
            'dmx_enabled': self.dmx_enabled,
            'dmx_protocol': self.dmx_protocol,
            'dmx_universe': self.dmx_universe,
            'dmx_channel': self.dmx_channel,
            'dmx_network': self.dmx_network,
            'dmx_out_enabled': self.dmx_out_enabled,
            'dmx_out_protocol': self.dmx_out_protocol,
            'dmx_out_universe': self.dmx_out_universe,
            'dmx_out_channel': self.dmx_out_channel,
            'dmx_out_target': self.dmx_out_target,
        }
        try:
            atomic_write_json(self.config_path, config_data, keep_backup=False, indent=None)
        except (OSError, TypeError, ValueError) as e:
            log.warning(f"Could not save config file: {e}")

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
