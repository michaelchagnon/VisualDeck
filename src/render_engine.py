# =====================================================================
# src/render_engine.py
# =====================================================================
import os
import sys
import ctypes
import pygame
import numpy as np
import cv2
import threading
import multiprocessing
import time
import av
import logging
from ring_buffer import FastRingBuffer, SimpleFrameQueue
from multiprocessing import shared_memory
from queue import Queue, Empty, Full
from threading import Lock
from pygame.locals import *
from screeninfo import get_monitors
from PIL import Image, ImageDraw
from collections import deque 
from collections import OrderedDict

# --- OpenGL specific imports ---
from OpenGL.GL import *
from OpenGL.GL.ARB.map_buffer_range import glMapBufferRange, GL_MAP_WRITE_BIT, GL_MAP_UNSYNCHRONIZED_BIT
from OpenGL.GL import shaders
from OpenGL.error import GLError
from OpenGL.GL.ARB.pixel_buffer_object import *

# --- Constants ---
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')
FRAME_BUFFER_SIZE = 30  # Increased for GPU players
CPU_FRAME_BUFFER_SIZE = 10 # Added for CPU players

# ── PreciseFramePacer (drop-in replacement) ──────────────────────────────────
class PreciseFramePacer:
    """
    High-precision frame pacer with dynamic retargeting and basic overrun stats.
    - target_fps can be changed at runtime via set_target_fps()
    - actual_fps is a smoothed EMA
    - reports % of frames that overran their budget in the last N samples
    """
    def __init__(self, target_fps=60.0, ema_alpha=0.1, window=120):
        self._target_fps = max(1.0, float(target_fps))
        self._frame_period = 1.0 / self._target_fps
        self._last_ts = time.perf_counter()
        self.actual_fps = 0.0
        self._ema_alpha = float(ema_alpha)
        self._overruns = deque(maxlen=window)
        self._window = window

    def set_target_fps(self, fps: float):
        fps = max(1.0, float(fps))
        if abs(fps - self._target_fps) > 0.1:
            self._target_fps = fps
            self._frame_period = 1.0 / self._target_fps

    def get_target_fps(self) -> float:
        return self._target_fps

    def wait_for_next_frame(self):
        now = time.perf_counter()
        elapsed = now - self._last_ts
        sleep_time = self._frame_period - elapsed

        # Hybrid sleep+spin
        if sleep_time > 0.002:
            time.sleep(sleep_time - 0.001)
        # Busy wait the final ~1ms
        while (time.perf_counter() - self._last_ts) < self._frame_period:
            pass

        end = time.perf_counter()
        frame_time = end - self._last_ts
        self._last_ts = end

        # Update stats
        if self.actual_fps == 0.0:
            self.actual_fps = 1.0 / frame_time if frame_time > 0 else self._target_fps
        else:
            inst = 1.0 / frame_time if frame_time > 0 else self._target_fps
            self.actual_fps = (self._ema_alpha * inst) + ((1 - self._ema_alpha) * self.actual_fps)

        overran = frame_time > (self._frame_period * 1.05)  # 5% tolerance
        self._overruns.append(1 if overran else 0)

        return frame_time  # caller can also inspect per-frame cost

    def overrun_ratio(self) -> float:
        if not self._overruns:
            return 0.0
        return sum(self._overruns) / len(self._overruns)
# ─────────────────────────────────────────────────────────────────────────────


class FrameMemoryPool:
    """Memory pool for reusing frame buffers to reduce allocation overhead."""
    def __init__(self, num_buffers=10, shape=(1080, 1920, 3)):
        self.shape = shape
        self.num_buffers = num_buffers
        self.buffers = [np.empty(shape, dtype=np.uint8) for _ in range(num_buffers)]
        self.available = deque(self.buffers)
        self.in_use = set()
        self.lock = threading.Lock()
        self.allocation_count = 0  # Track new allocations for debugging
    
    def get_buffer(self, shape=None):
        """Get a buffer from the pool, allocating a new one if needed."""
        with self.lock:
            # If shape doesn't match, allocate new
            if shape and shape != self.shape:
                self.allocation_count += 1
                return np.empty(shape, dtype=np.uint8)
            
            if self.available:
                buf = self.available.popleft()
                self.in_use.add(id(buf))
                return buf
            else:
                # Pool exhausted, allocate new
                self.allocation_count += 1
                logging.debug(f"Frame pool exhausted, allocating new buffer. Total allocations: {self.allocation_count}")
                return np.empty(self.shape, dtype=np.uint8)
    
    def return_buffer(self, buf):
        """Return a buffer to the pool."""
        if buf is None:
            return
            
        with self.lock:
            buf_id = id(buf)
            if buf_id in self.in_use:
                self.in_use.remove(buf_id)
                if len(self.available) < self.num_buffers:
                    self.available.append(buf)

# Global frame pool - shared across all video players
FRAME_POOL = FrameMemoryPool(num_buffers=20, shape=(1080, 1920, 3))

# --- PyOpenGL Debug ---
if os.environ.get('VISUALDECK_DEBUG') != '1':
    try:
        # These flags are from PyOpenGL 3.1.0 and later
        import OpenGL
        OpenGL.ERROR_CHECKING = False
        OpenGL.ERROR_LOGGING = False
        OpenGL.CONTEXT_CHECKING = False
        OpenGL.STORE_POINTERS = False
        logging.info("PyOpenGL error checking disabled for production.")
    except AttributeError:
        # Fallback for older PyOpenGL versions if necessary
        logging.warning("Could not disable all PyOpenGL checks, consider updating PyOpenGL.")
        pass


# --- Video Player Classes ---
class BaseVideoPlayer(threading.Thread):
    """Base class for video player threads."""
    def __init__(self, path, frame_queue, loop=True):
        super().__init__(daemon=True)
        self.path = path
        self.frame_queue = frame_queue
        self.loop = loop
        self.is_running = True
        self.lock = threading.Lock()
        
    def stop(self):
        with self.lock:
            self.is_running = False
            
    def _set_thread_priority(self):
        """Set thread priority on Windows for better performance."""
        if sys.platform == "win32":
            try:
                import win32api, win32process
                thread_handle = win32api.GetCurrentThread()
                # MODIFICATION: Changed to HIGHEST to keep up with the render thread
                win32process.SetThreadPriority(thread_handle, 
                    win32process.THREAD_PRIORITY_HIGHEST)
                logging.debug(f"Set video decoder thread priority to HIGHEST for {os.path.basename(self.path)}")
            except:
                pass  # Silent fail if pywin32 not available


# MODIFICATION: Re-instated the smoother, thread-based player for CPU mode from the old codebase.
class CPUVideoPlayerThread(BaseVideoPlayer):
    """A dedicated thread for reading frames from a video file for CPU mode."""
    def __init__(self, path, frame_queue):
        super().__init__(path, frame_queue)
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_delay = 1.0 / self.fps if self.fps > 0 else 1/30.0

    def run(self):
        logging.info(f"CPUVideoPlayerThread starting for {os.path.basename(self.path)}")
        self._set_thread_priority()
        frame_buffer = None  # Reusable buffer
        
        while self.is_running:
            start_time = time.perf_counter()

            with self.lock:
                if not self.is_running: break
                ret, frame = self.cap.read()

            if not ret:
                logging.debug(f"Failed to read frame, rewinding {os.path.basename(self.path)}")
                with self.lock:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Get buffer from pool if we don't have one or if shape changed
            if frame_buffer is None or frame_buffer.shape != frame.shape:
                if frame_buffer is not None:
                    FRAME_POOL.return_buffer(frame_buffer)
                frame_buffer = FRAME_POOL.get_buffer(frame.shape)
            
            # Copy frame data to our buffer
            np.copyto(frame_buffer, frame)

            # Log frame info occasionally
            if hasattr(self, '_frame_count'):
                self._frame_count += 1
            else:
                self._frame_count = 0
            
            if self._frame_count % 30 == 0:  # Log every 30 frames
                logging.debug(f"Read frame {self._frame_count} from {os.path.basename(self.path)}: shape={frame_buffer.shape}")

            try:
                # Send the buffer reference (not a copy)
                if hasattr(self.frame_queue, 'put_nowait_drop_old'):
                    self.frame_queue.put_nowait_drop_old(frame_buffer)
                else:
                    self.frame_queue.put_nowait(frame_buffer)
            except Exception as e:
                logging.debug(f"Failed to put frame in queue: {e}")
                pass
            
            # Use precise frame pacing for video playback
            if not hasattr(self, 'frame_pacer'):
                self.frame_pacer = PreciseFramePacer(target_fps=self.fps)

            self.frame_pacer.wait_for_next_frame()
    
        # Return buffer when done
        if frame_buffer is not None:
            FRAME_POOL.return_buffer(frame_buffer)
            
    def stop(self):
        super().stop()
        with self.lock:
            self.cap.release()

class GPUVideoPlayerThread(BaseVideoPlayer):
    def __init__(self, path, frame_queue, hw_accel=None, loop=True, output_format='numpy', force_multithread=True):
        super().__init__(path, frame_queue, loop)
        self.hw_accel = hw_accel
        self.output_format = output_format
        self.force_multithread = force_multithread # Store the setting


    def run(self):
        self._set_thread_priority()
        
        while self.is_running:
            container = self._open_container()
            if not container:
                time.sleep(0.5)
                continue
            
            stream = container.streams.video[0]
            frame_delay = 1.0 / float(stream.average_rate or 30)

            try:
                for frame in container.decode(video=0):
                    if not self.is_running:
                        break
                    
                    t0 = time.perf_counter()
                    
                    output_item = self._convert_frame(frame)
                    
                    try:
                        if hasattr(self.frame_queue, 'put_nowait_drop_old'):
                            self.frame_queue.put_nowait_drop_old(output_item)
                        else:
                            self.frame_queue.put_nowait(output_item)
                    except Exception:
                        pass
                    
                    # Use precise frame pacing for video playback
                    if not hasattr(self, 'frame_pacer'):
                        fps = float(stream.average_rate or 30)
                        self.frame_pacer = PreciseFramePacer(target_fps=fps)

                    self.frame_pacer.wait_for_next_frame()

            except Exception as e:
                logging.error(f"PyAV Error during playback of {self.path}: {e}")
            finally:
                container.close()
                
            if not self.loop:
                break
                
    def _open_container(self):
        """Open video container with optional hardware acceleration."""
        container = None
        options = {}

        # ADD THIS: Conditionally add the multi-threading option
        if self.force_multithread:
            options['threads'] = 'auto' # Let PyAV determine the optimal number of threads

        if self.hw_accel and self.hw_accel != 'auto':
            try:
                options['hwaccel'] = self.hw_accel
                container = av.open(self.path, options=options)
                logging.info(f"Successfully opened {os.path.basename(self.path)} with options: {options}")
            except Exception as e:
                logging.warning(f"PyAV open failed with options {options} for {os.path.basename(self.path)}: {e}. Falling back.")
                container = None
                options.pop('hwaccel', None) # Remove failed option

        if container is None:
            try:
                # Try again without hwaccel but with other options
                container = av.open(self.path, options=options)
            except Exception as e:
                logging.error(f"PyAV could not open {os.path.basename(self.path)} at all: {e}")
                
        return container
        
    def _process_video(self, container):
        """Process video frames from container."""
        stream = container.streams.video[0]
        frame_delay = 1.0 / (stream.average_rate or 30)
        
        # Pre-decode first few frames to fill the pipeline
        frame_buffer = deque(maxlen=3)
        frame_count = 0
        
        for frame in container.decode(video=0):
            if not self.is_running:
                break
            
            # For the first few frames, just fill the buffer
            if frame_count < 3:
                frame_buffer.append(frame)
                frame_count += 1
                continue
            
            # Process buffered frame while decoding next
            t0 = time.perf_counter()
            
            # Get oldest frame from buffer
            old_frame = frame_buffer.popleft()
            frame_buffer.append(frame)
            
            # Process the frame
            output_item = self._convert_frame(old_frame)
            
            try:
                if hasattr(self.frame_queue, 'put_nowait_drop_old'):
                    self.frame_queue.put_nowait_drop_old(output_item)
                else:
                    self.frame_queue.put_nowait(output_item)
            except Exception:
                pass
            
            elapsed = time.perf_counter() - t0
            if (delay := frame_delay - elapsed) > 0:
                time.sleep(delay)
        
        # Process remaining buffered frames
        while frame_buffer and self.is_running:
            t0 = time.perf_counter()
            old_frame = frame_buffer.popleft()
            
            output_item = self._convert_frame(old_frame)
            
            try:
                if hasattr(self.frame_queue, 'put_nowait_drop_old'):
                    self.frame_queue.put_nowait_drop_old(output_item)
                else:
                    self.frame_queue.put_nowait(output_item)
            except Exception:
                pass
            
            elapsed = time.perf_counter() - t0
            if (delay := frame_delay - elapsed) > 0:
                time.sleep(delay)
                
    def _convert_frame(self, frame):
        """Convert frame to the appropriate output format."""
        if self.output_format == 'pygame':
            # Pygame still needs BGR for compatibility
            arr = frame.to_ndarray(format='bgr24')
            frame_bgra = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
            h, w = frame_bgra.shape[:2]
            return (frame_bgra.tobytes(), (w, h))
        elif self.output_format == 'yuv':
            # New YUV path - keep native format
            try:
                # Method 1: Direct YUV420P array extraction
                # This is more compatible across PyAV versions
                yuv_array = frame.to_ndarray(format='yuv420p')
                
                width = frame.width
                height = frame.height
                
                # YUV420P layout in memory:
                # Y plane: width * height bytes
                # U plane: (width/2) * (height/2) bytes  
                # V plane: (width/2) * (height/2) bytes
                y_size = width * height
                uv_size = (width // 2) * (height // 2)
                
                # Extract planes from the flat array
                y_data = yuv_array[:y_size].tobytes()
                u_data = yuv_array[y_size:y_size + uv_size]
                v_data = yuv_array[y_size + uv_size:y_size + 2*uv_size]
                
                # Interleave U and V for more efficient GPU upload
                uv_interleaved = np.empty(uv_size * 2, dtype=np.uint8)
                uv_interleaved[0::2] = u_data
                uv_interleaved[1::2] = v_data
                
                return {
                    'format': 'yuv420p',
                    'width': width,
                    'height': height,
                    'y_data': y_data,
                    'uv_data': uv_interleaved.tobytes(),
                    'timestamp': float(frame.pts * frame.time_base) if frame.pts else 0
                }
            except Exception as e:
                # If YUV extraction fails, try alternative method
                try:
                    # Method 2: Reformat then extract
                    yuv_frame = frame.reformat(format='yuv420p')
                    yuv_array = yuv_frame.to_ndarray()
                    
                    width = yuv_frame.width
                    height = yuv_frame.height
                    
                    # Calculate plane sizes
                    y_size = width * height
                    uv_size = (width // 2) * (height // 2)
                    
                    # Flatten and extract
                    flat_data = yuv_array.flatten()
                    y_data = flat_data[:y_size].tobytes()
                    u_data = flat_data[y_size:y_size + uv_size]
                    v_data = flat_data[y_size + uv_size:y_size + 2*uv_size]
                    
                    # Interleave U and V
                    uv_interleaved = np.empty(uv_size * 2, dtype=np.uint8)
                    uv_interleaved[0::2] = u_data
                    uv_interleaved[1::2] = v_data
                    
                    return {
                        'format': 'yuv420p',
                        'width': width,
                        'height': height,
                        'y_data': y_data,
                        'uv_data': uv_interleaved.tobytes(),
                        'timestamp': float(frame.pts * frame.time_base) if frame.pts else 0
                    }
                except Exception as e2:
                    logging.warning(f"Failed to get YUV format (both methods), falling back to RGB: Method1: {e}, Method2: {e2}")
                    # Fallback to RGB
                    return frame.to_ndarray(format='bgr24')
        else:
            # Original numpy/BGR path for CPU mode
            return frame.to_ndarray(format='bgr24')

class BaseRenderEngine:
    def __init__(self, media_manager, mode='cpu', hw_accel=None):
        self.mm = media_manager
        self.mode = mode
        self.hw_accel = hw_accel
        self.monitors = get_monitors()
        self.window = None
        self.active_players = {}
        self.video_frame_cache = {}
        self.shared_memory_blocks = {}
        self.static_cue_cache = {}  # {cache_key: pre_rendered_data}
        self.cue_is_static = {}  # {col_index: bool}
        self.static_cue_cache_hits = 0
        self.static_cue_cache_misses = 0   
            
        # Initialize frame pool with monitor-appropriate size
        if self.monitors:
            max_width = max(m.width for m in self.monitors)
            max_height = max(m.height for m in self.monitors)
            # Reinitialize global pool with appropriate size
            global FRAME_POOL
            FRAME_POOL = FrameMemoryPool(
                num_buffers=20,
                shape=(max_height, max_width, 3)
            )
            logging.info(f"Initialized frame pool for {max_width}x{max_height} resolution")

    def is_cue_static(self, layers):
        """Check if a cue contains only static media (no videos)."""
        if not layers:
            return True  # Empty cue is considered static
        
        for layer in layers.values():
            path = layer.get('path')
            if path and os.path.exists(path):
                ext = os.path.splitext(path)[1].lower()
                if ext in VIDEO_EXTS:
                    return False  # Found a video, not static
        
        return True  # All media are static images

    def get_cue_cache_key(self, layers, target_monitor=None):
        """Generate a unique cache key for a cue configuration."""
        # Create a stable key based on the cue's content
        layer_info = []
        for layer_id, layer in sorted(layers.items()):
            if layer.get('path'):
                # Include all relevant layer properties in the key
                pos = layer.get('position', {})
                layer_info.append((
                    layer_id,
                    layer['path'],
                    pos.get('x', 0.5),
                    pos.get('y', 0.5),
                    pos.get('scale_x', 1.0),
                    pos.get('scale_y', 1.0),
                    pos.get('rotation', 0),
                    layer.get('order', 0),
                    tuple(layer.get('screens', []))
                ))
        
        # Include monitor configuration in the key
        monitor_info = None
        if target_monitor is not None and self.monitors:
            if 0 <= target_monitor < len(self.monitors):
                mon = self.monitors[target_monitor]
                monitor_info = (mon.width, mon.height)
        
        return (tuple(layer_info), monitor_info)

    def invalidate_static_cache_for_column(self, col_index):
        """Invalidate all cached data for a specific column."""
        # Remove any cache entries that might be related to this column
        keys_to_remove = []
        for key in self.static_cue_cache.keys():
            # Since we don't store column index in the key directly,
            # we'll need to invalidate based on a flag
            keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.static_cue_cache[key]
        
        # Reset the static flag for this column
        if col_index in self.cue_is_static:
            del self.cue_is_static[col_index]
        
        logging.debug(f"Invalidated static cache for column {col_index}")

    def clear_static_cache(self):
        """Clear all static cue caches."""
        self.static_cue_cache.clear()
        self.cue_is_static.clear()
        self.static_cue_cache_hits = 0
        self.static_cue_cache_misses = 0
        logging.debug("Cleared all static cue caches")

    def _init_pygame_and_gl(self):
        """Base initialization - can be overridden by subclasses"""
        pygame.init()
        pygame.display.init()

    def update_layers(self, layers, force_restart_paths=None):
        if force_restart_paths is None:
            force_restart_paths = set()

        for path in force_restart_paths:
            if path in self.active_players:
                logging.info(f"Forcing restart of player for: {path}")
                p_info = self.active_players.pop(path)
                p_info['player'].stop()
                if hasattr(p_info['player'], 'join'):
                    p_info['player'].join(0.5)
                self.video_frame_cache.pop(path, None)
                if hasattr(self, 'release_texture'):
                    self.release_texture(path)
                # No shared memory for CPU mode anymore, so no cleanup needed here

        active_paths = {l['path'] for l in layers.values() if l['path'] and os.path.exists(l['path']) and os.path.splitext(l['path'])[1].lower() in VIDEO_EXTS}

        for path in list(self.active_players):
            if path not in active_paths:
                p_info = self.active_players.pop(path)
                p_info['player'].stop()
                if hasattr(p_info['player'], 'join'):
                    p_info['player'].join(0.5)
                self.video_frame_cache.pop(path, None)
                if hasattr(self, 'release_texture'):
                    self.release_texture(path)

        for m in layers.values():
            path = m['path']
            if not path or not os.path.exists(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext not in VIDEO_EXTS or path in self.active_players:
                continue
            loop = m.get('loop', True)

            # MODIFICATION: Replaced the multiprocessing logic with the smoother threading logic for CPU mode.
            if self.mode == 'cpu':
                q = SimpleFrameQueue(maxsize=CPU_FRAME_BUFFER_SIZE)
                pl = CPUVideoPlayerThread(path, q)
                self.active_players[path] = {'player': pl, 'type': 'cpu_thread', 'queue': q}
            else:
                # GPU mode - use YUV for OpenGL, pygame format for pygame_gpu
                if self.mode == 'opengl':
                    output_fmt = 'yuv'  # NEW: Use YUV for OpenGL
                else:
                    output_fmt = 'pygame'  # Keep pygame format for pygame_gpu mode
                
                # Use FastRingBuffer for GPU mode with potential frame shape optimization
                q = FastRingBuffer(FRAME_BUFFER_SIZE)
                # Pass the multithread setting to the player thread
                pl = GPUVideoPlayerThread(
                    path, q, self.hw_accel, loop, 
                    output_format=output_fmt, 
                    force_multithread=getattr(self, 'force_multithread', True)
                )
                self.active_players[path] = {'player': pl, 'type': 'gpu_queue', 'queue': q}
            
            self.active_players[path]['player'].start()

    def _set_preview_mode(self, preview_mode):
        """Temporarily set preview mode to use BGR instead of YUV for CPU previews."""
        self._preview_mode = preview_mode
        
    def _get_latest_video_frame(self, path):
        if path in self.active_players:
            player_info = self.active_players[path]
            q = player_info['queue']
            frame = None

            is_first_frame_request = path not in self.video_frame_cache

            if is_first_frame_request:
                try:
                    frame = q.get(timeout=0.1)
                    if frame is not None:
                        logging.debug(f"Got first frame for {os.path.basename(path)}, type: {type(frame)}, shape: {getattr(frame, 'shape', 'N/A')}")
                except (Exception, TimeoutError) as e:
                    logging.debug(f"Failed to get first frame for {os.path.basename(path)}: {e}")
                    pass
            else:
                # Use the optimized get_latest method if available
                if hasattr(q, 'get_latest'):
                    frame = q.get_latest()
                else:
                    # Fallback for compatibility
                    frames_to_get = q.qsize()
                    if frames_to_get > 1:
                        for _ in range(frames_to_get - 1):
                            try: q.get_nowait()
                            except: break

                    if not q.empty():
                        try: frame = q.get_nowait()
                        except: pass

            # Convert YUV to BGR if in preview mode
            if frame is not None and isinstance(frame, dict) and hasattr(self, '_preview_mode') and self._preview_mode:
                # Don't cache YUV frames when in preview mode
                return frame  # Return the dict, let render_layers_to_image handle it
            
            if frame is not None:
                self.video_frame_cache[path] = frame
                logging.debug(f"Cached frame for {os.path.basename(path)}")

            return self.video_frame_cache.get(path)

        logging.debug(f"No active player for {os.path.basename(path)}")
        return None

    def render_layers_to_image(self, layers, dimensions):
        w, h = dimensions
        if not layers:
            return None
        
        # Calculate scale factor for preview (comparing to actual monitor size)
        preview_scale = 1.0
        if hasattr(self, 'monitors') and self.monitors:
            # Assume we're rendering for the first monitor or the typical case
            typical_width = 1920  # Default assumption
            typical_height = 1080
            
            # Try to get actual monitor dimensions if available
            if len(self.monitors) > 0:
                typical_width = self.monitors[0].width
                typical_height = self.monitors[0].height
            
            # Calculate how much we've scaled down the preview
            preview_scale = min(w / typical_width, h / typical_height, 1.0)
        
        target_image = Image.new('RGBA', (w, h), (0,0,0,0))
            
        for m in sorted(layers.values(), key=lambda l: l.get('order', 0)):
            path = m['path']
            if not path or not os.path.exists(path): continue
            ext = os.path.splitext(path)[1].lower()
            is_video = ext in VIDEO_EXTS
            
            frame_data = self._get_latest_video_frame(path) if is_video else cv2.imread(path, cv2.IMREAD_UNCHANGED)
            
            pil_img = None
            if frame_data is not None:
                try:
                    # Handle YUV data (dict format)
                    if isinstance(frame_data, dict) and 'format' in frame_data:
                        # Convert YUV back to RGB for CPU preview
                        # This is not optimal but necessary for CPU preview path
                        width = frame_data['width']
                        height = frame_data['height']
                        y_data = np.frombuffer(frame_data['y_data'], dtype=np.uint8).reshape(height, width)
                        uv_data = np.frombuffer(frame_data['uv_data'], dtype=np.uint8)
                        
                        # Deinterleave UV
                        uv_size = (width // 2) * (height // 2)
                        u_data = uv_data[0::2].reshape(height // 2, width // 2)
                        v_data = uv_data[1::2].reshape(height // 2, width // 2)
                        
                        # Upsample U and V to full resolution
                        u_upsampled = cv2.resize(u_data, (width, height), interpolation=cv2.INTER_LINEAR)
                        v_upsampled = cv2.resize(v_data, (width, height), interpolation=cv2.INTER_LINEAR)
                        
                        # YUV to RGB conversion (BT.709 coefficients)
                        yuv = np.stack([y_data, u_upsampled, v_upsampled], axis=-1).astype(np.float32)
                        yuv[:,:,1:] -= 128.0  # Center U and V
                        
                        # Conversion matrix
                        rgb = np.zeros_like(yuv)
                        rgb[:,:,0] = yuv[:,:,0] + 1.5748 * yuv[:,:,2]  # R
                        rgb[:,:,1] = yuv[:,:,0] - 0.1873 * yuv[:,:,1] - 0.4681 * yuv[:,:,2]  # G
                        rgb[:,:,2] = yuv[:,:,0] + 1.8556 * yuv[:,:,1]  # B
                        
                        # Clip and convert to uint8
                        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                        pil_img = Image.fromarray(rgb).convert('RGBA')
                        
                    elif isinstance(frame_data, tuple):
                        # Pygame format (bytes, dims) - skip for CPU preview
                        continue 
                    else:
                        # Regular numpy array path
                        if frame_data.ndim == 2:
                            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
                        elif frame_data.shape[2] == 4:
                            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGRA2RGBA)
                        elif frame_data.shape[2] == 3:
                            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                        
                        pil_img = Image.fromarray(frame_data).convert('RGBA')
                except Exception as e:
                    logging.warning(f"Could not convert frame for CPU preview: {e}")
                    continue

            if pil_img:
                pos = m.get('position', {})
                # Scale the scale values by the preview scale factor
                scale_x = pos.get('scale_x', 1.0) * preview_scale
                scale_y = pos.get('scale_y', 1.0) * preview_scale
                rotation = pos.get('rotation', 0)
                img_w, img_h = pil_img.size
                
                new_w, new_h = int(img_w * scale_x), int(img_h * scale_y)
                if new_w <= 0 or new_h <= 0: continue
                
                transformed_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                if rotation != 0:
                    transformed_img = transformed_img.rotate(rotation, expand=True, resample=Image.BICUBIC)
                
                x_norm, y_norm = pos.get('x', 0.5), pos.get('y', 0.5)
                paste_x = int(x_norm * w - transformed_img.width / 2)
                paste_y = int(y_norm * h - transformed_img.height / 2)
                
                target_image.alpha_composite(transformed_img, (paste_x, paste_y))
        return target_image

    def close_window(self):
        if self.window:
            pygame.display.quit()
            self.window = None

    def clear(self):
        self.close_window()
        for path, p_info in list(self.active_players.items()):
            p_info['player'].stop()
            if hasattr(p_info['player'], 'join'):
                p_info['player'].join(0.5)
        self.active_players.clear()
        self.video_frame_cache.clear()
        self.shared_memory_blocks.clear() # This was for the old CPU method, clearing it is safe.

VERTEX_SHADER = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aTexCoord;
out vec2 TexCoord;
uniform mat4 model;
uniform mat4 projection;
void main() {
    gl_Position = projection * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D ourTexture;
uniform sampler2D yTexture;
uniform sampler2D uvTexture;
uniform float alpha;
uniform bool isBGR;
uniform bool isYUV;

void main() {
    vec4 texColor;
    
    if (isYUV) {
        // YUV to RGB conversion (BT.709 coefficients)
        float y = texture(yTexture, TexCoord).r;
        vec2 uv = texture(uvTexture, TexCoord).rg - vec2(0.5, 0.5);
        
        // BT.709 YUV to RGB conversion
        float r = y + 1.5748 * uv.y;
        float g = y - 0.1873 * uv.x - 0.4681 * uv.y;
        float b = y + 1.8556 * uv.x;
        
        texColor = vec4(r, g, b, 1.0);
    } else {
        texColor = texture(ourTexture, TexCoord);
        if (isBGR) {
            texColor.rgb = texColor.bgr;  // Swizzle BGR to RGB
        }
    }
    
    if(texColor.a < 0.1) discard;
    FragColor = vec4(texColor.rgb, texColor.a * alpha);
}
"""

# Instanced rendering shaders - separate from main shaders
INSTANCED_VERTEX_SHADER = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aTexCoord;

// Instance attributes
layout(location=2) in mat4 instanceModel;
layout(location=6) in float instanceAlpha;
layout(location=7) in float instanceTextureIndex;

out vec2 TexCoord;
flat out float Alpha;
flat out int TextureIndex;

uniform mat4 projection;

void main() {
    gl_Position = projection * instanceModel * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
    Alpha = instanceAlpha;
    TextureIndex = int(instanceTextureIndex);
}
"""

INSTANCED_FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
flat in float Alpha;
flat in int TextureIndex;

out vec4 FragColor;

// Single texture array for all textures
uniform sampler2D textures[16];

void main() {
    vec4 texColor = texture(textures[TextureIndex], TexCoord);
    
    if(texColor.a < 0.1) discard;
    FragColor = vec4(texColor.rgb, texColor.a * Alpha);
}
"""

class OpenGLRenderEngine(BaseRenderEngine):
    def __init__(self, media_manager, mode='opengl', hw_accel=None, use_pbo=True, use_vsync=False, use_instancing=True, preview_bridge=None):
        super().__init__(media_manager, mode, hw_accel)
        logging.info(f"OpenGLRenderEngine init '{mode}', accel={hw_accel}, PBO: {use_pbo}, VSync: {use_vsync}, Instancing: {use_instancing}")
        self.shader = self.vao = self.vbo = self.ebo = None
        self.instanced_shader = None  # ADD THIS - Separate shader for instancing
        self.instanced_vao = None     # ADD THIS - Separate VAO for instancing
        self.textures = {}
        self.texture_formats = {}
        self.u_locs = {}
        self.instanced_u_locs = {}    # ADD THIS - Uniform locations for instanced shader
        self.spanning_window_details = None
        self.preview_bridge = preview_bridge

        # Store the advanced settings
        self.use_pbo = use_pbo
        self.use_vsync = use_vsync

        # PBO management
        self.pbo_pool = {}
        self.pbo_index = {}
        self.PBO_POOL_SIZE = 3
        
        # YUV texture management
        self.yuv_textures = {}
        self.yuv_pbo_pool = {}
        self.yuv_pbo_index = {}
        
        # Instance rendering support
        self.instance_vbo = None
        self.max_instances = 16
        self.use_instancing = use_instancing  # Use the passed parameter

    def _init_pygame_and_gl(self):
        import os, sys, pygame
        pygame.init()
        pygame.display.init()

        # Ensure the output window uses the same icon as the main GUI (.ico file)
        try:
            from PIL import Image
            base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ico_path = os.path.join(base_dir, 'src', 'visualdeck.ico')
            if os.path.exists(ico_path):
                im = Image.open(ico_path)
                # Pick largest frame in the .ico for best fidelity
                if getattr(im, "n_frames", 1) > 1:
                    best = None; best_area = -1
                    for i in range(im.n_frames):
                        im.seek(i)
                        area = im.size[0] * im.size[1]
                        if area > best_area:
                            best = im.copy()
                            best_area = area
                    im = best
                if im.mode != "RGBA":
                    im = im.convert("RGBA")
                surf = pygame.image.frombuffer(im.tobytes(), im.size, "RGBA")
                pygame.display.set_icon(surf)
        except Exception:
            # Best-effort; never fail initialization over an icon
            pass

        if sys.platform == "win32":
            pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 1)

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

        vsync_val = 1 if self.use_vsync else 0
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, vsync_val)

    def pre_render_static_cue(self, layers, target_monitor=None):
        """Pre-render a static cue to a framebuffer object (FBO)."""
        if not layers or not self.window:
            return None
        
        # Determine dimensions based on target monitor
        if target_monitor is not None and 0 <= target_monitor < len(self.monitors):
            mon = self.monitors[target_monitor]
            width, height = mon.width, mon.height
        else:
            width, height = 1920, 1080  # Default dimensions
        
        # Create FBO for off-screen rendering
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        
        # Create texture to render to
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Attach texture to FBO
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
        
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            logging.error("Failed to create framebuffer for static cue pre-rendering")
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [texture])
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return None
        
        # Set viewport for FBO rendering
        glViewport(0, 0, width, height)
        
        # Clear and render
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Set up projection for FBO
        projection_matrix = self._ortho(0, width, 0, height, -1, 1)
        glUniformMatrix4fv(self.u_locs['projection'], 1, GL_TRUE, projection_matrix)
        
        # Render all layers to the FBO
        for layer in sorted(layers.values(), key=lambda x: x.get('order', 0)):
            path = layer.get('path')
            if not path or not os.path.exists(path):
                continue
            
            # Update/get texture for this media
            texture_id = self._update_texture_pbo(path)
            if not texture_id:
                continue
            
            tex_info = self.textures[path]
            
            # Bind texture and set uniforms
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glUniform1i(self.u_locs['ourTexture'], 0)
            glUniform1i(self.u_locs['isYUV'], 0)
            
            is_bgr = self.texture_formats.get(path, False)
            glUniform1i(self.u_locs['isBGR'], int(is_bgr))
            
            # Set layer transform
            model_matrix = self._model_matrix(
                layer.get('position', {}),
                tex_info['width'],
                tex_info['height'],
                (width, height)
            )
            
            glUniformMatrix4fv(self.u_locs['model'], 1, GL_TRUE, model_matrix)
            glUniform1f(self.u_locs['alpha'], layer.get('alpha', 1.0))
            
            # Draw
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        # Unbind FBO
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Store the pre-rendered texture and FBO for later use
        pre_rendered_data = {
            'texture': texture,
            'fbo': fbo,
            'width': width,
            'height': height
        }
        
        return pre_rendered_data

    def render_pre_rendered_cue(self, pre_rendered_data, viewport_dims, alpha=1.0):
        """Render a pre-rendered static cue texture to the current viewport."""
        if not pre_rendered_data:
            return
        
        texture = pre_rendered_data['texture']
        width = pre_rendered_data['width']
        height = pre_rendered_data['height']
        
        # Bind the pre-rendered texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        glUniform1i(self.u_locs['ourTexture'], 0)
        glUniform1i(self.u_locs['isYUV'], 0)
        glUniform1i(self.u_locs['isBGR'], 0)
        
        # FIX: Create a flipped model matrix to account for FBO coordinate system
        # We need to flip the Y-axis to correct the upside-down rendering
        model_matrix = self._model_matrix(
            {'x': 0.5, 'y': 0.5, 'scale_x': 1.0, 'scale_y': -1.0, 'rotation': 0},  # Note: scale_y is negative
            width,
            height,
            viewport_dims
        )
        
        glUniformMatrix4fv(self.u_locs['model'], 1, GL_TRUE, model_matrix)
        glUniform1f(self.u_locs['alpha'], alpha)
        
        # Draw the quad
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    def cleanup_pre_rendered_cue(self, pre_rendered_data):
        """Clean up resources used by a pre-rendered cue."""
        if pre_rendered_data:
            if 'fbo' in pre_rendered_data:
                glDeleteFramebuffers(1, [pre_rendered_data['fbo']])
            if 'texture' in pre_rendered_data:
                glDeleteTextures(1, [pre_rendered_data['texture']])

    def _create_window(self, geom, is_spanning=False):
        """Creates a single window, potentially spanning multiple monitors."""
        x, y, w, h = geom
        
        # Check if a window with these exact dimensions already exists
        if self.window and self.window.get_size() == (w, h) and self.spanning_window_details == geom:
            return True
            
        self.close_window()
        # Reset per-context GL caches – the old IDs are invalid after we recreate the window
        self.yuv_textures.clear()
        self.yuv_pbo_pool.clear()
        self.yuv_pbo_index.clear()
        
        # Prevent implicit centering and request explicit position
        os.environ['SDL_VIDEO_CENTERED'] = '0'
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
        
        self.window = pygame.display.set_mode((w, h), NOFRAME | DOUBLEBUF | OPENGL)
        pygame.display.set_caption("VisualDeck Output (OpenGL)")
        self.spanning_window_details = geom if is_spanning else None

        # Force the position after creation (some drivers ignore WINDOW_POS on first create)
        try:
            import pygame._sdl2 as sdl2
            win = sdl2.Window.from_display_module()
            win.position = (int(x), int(y))
        except Exception:
            pass
        
        # Windows-specific OpenGL optimizations
        if sys.platform == "win32":
            try:
                # Try to disable vsync for lower latency
                # This works on some drivers
                import pygame._sdl2 as sdl2
                window = sdl2.Window.from_display_module()
                window.get_surface()  # Force context creation
                
                # Also try via OpenGL directly
                try:
                    from OpenGL import WGL
                    if hasattr(WGL, 'wglSwapIntervalEXT'):
                        WGL.wglSwapIntervalEXT(0)  # 0 = no vsync
                except:
                    pass
            except Exception as e:
                logging.debug(f"Could not set Windows-specific OpenGL options: {e}")
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Re-initialize GL resources only if they don't exist
        if not self.shader:
            # === REGULAR SHADER INITIALIZATION ===
            self.shader = shaders.compileProgram(
                shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            )
            glUseProgram(self.shader)
            
            # Get uniform locations for regular shader
            self.u_locs['projection'] = glGetUniformLocation(self.shader, "projection")
            self.u_locs['model'] = glGetUniformLocation(self.shader, "model")
            self.u_locs['alpha'] = glGetUniformLocation(self.shader, "alpha")
            self.u_locs['isBGR'] = glGetUniformLocation(self.shader, "isBGR")
            self.u_locs['isYUV'] = glGetUniformLocation(self.shader, "isYUV")
            self.u_locs['yTexture'] = glGetUniformLocation(self.shader, "yTexture")
            self.u_locs['uvTexture'] = glGetUniformLocation(self.shader, "uvTexture")
            self.u_locs['ourTexture'] = glGetUniformLocation(self.shader, "ourTexture")
            
            # Geometry data
            quad = np.array([
                 0.5,  0.5, 0.0, 1.0, 0.0,
                 0.5, -0.5, 0.0, 1.0, 1.0,
                -0.5, -0.5, 0.0, 0.0, 1.0,
                -0.5,  0.5, 0.0, 0.0, 0.0
            ], dtype=np.float32)
            indices = np.array([0, 1, 3, 1, 2, 3], dtype=np.uint32)
            
            # Create regular VAO
            self.vao = glGenVertexArrays(1)
            self.vbo = glGenBuffers(1)
            self.ebo = glGenBuffers(1)
            
            glBindVertexArray(self.vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
            
            # Regular vertex attributes
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
            glEnableVertexAttribArray(1)
            
            # Set default texture units for regular shader
            glUseProgram(self.shader)
            glUniform1i(self.u_locs['ourTexture'], 0)
            glUniform1i(self.u_locs['yTexture'], 0)
            glUniform1i(self.u_locs['uvTexture'], 1)
            
            # === INSTANCED SHADER INITIALIZATION ===
            if self.use_instancing:
                try:
                    self.instanced_shader = shaders.compileProgram(
                        shaders.compileShader(INSTANCED_VERTEX_SHADER, GL_VERTEX_SHADER),
                        shaders.compileShader(INSTANCED_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
                    )
                    
                    # Get uniform locations for instanced shader
                    glUseProgram(self.instanced_shader)
                    self.instanced_u_locs['projection'] = glGetUniformLocation(self.instanced_shader, "projection")
                    
                    # Set texture unit uniforms for instanced shader
                    for i in range(16):
                        loc = glGetUniformLocation(self.instanced_shader, f"textures[{i}]")
                        if loc != -1:
                            glUniform1i(loc, i)
                    
                    # Create instanced VAO with instance attributes
                    self.instanced_vao = glGenVertexArrays(1)
                    self.instance_vbo = glGenBuffers(1)
                    
                    glBindVertexArray(self.instanced_vao)
                    
                    # Share the same vertex and element buffers
                    glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
                    
                    # Set up vertex attributes (same as regular VAO)
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
                    glEnableVertexAttribArray(0)
                    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
                    glEnableVertexAttribArray(1)
                    
                    # Set up instance buffer
                    glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
                    # Each instance: 4x4 matrix (16 floats) + alpha (1) + texture index (1) = 18 floats
                    instance_stride = 18 * 4  # 18 floats * 4 bytes
                    glBufferData(GL_ARRAY_BUFFER, instance_stride * self.max_instances, None, GL_DYNAMIC_DRAW)
                    
                    # Instance attributes
                    # Model matrix (4x4) - locations 2-5
                    for i in range(4):
                        glVertexAttribPointer(2 + i, 4, GL_FLOAT, GL_FALSE, instance_stride, 
                                            ctypes.c_void_p(i * 16))  # Matrix row offset
                        glEnableVertexAttribArray(2 + i)
                        glVertexAttribDivisor(2 + i, 1)  # One per instance
                    
                    # Alpha - location 6
                    glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, instance_stride, 
                                         ctypes.c_void_p(64))  # After matrix (16 * 4 bytes)
                    glEnableVertexAttribArray(6)
                    glVertexAttribDivisor(6, 1)
                    
                    # Texture index - location 7
                    glVertexAttribPointer(7, 1, GL_FLOAT, GL_FALSE, instance_stride, 
                                         ctypes.c_void_p(68))  # After alpha
                    glEnableVertexAttribArray(7)
                    glVertexAttribDivisor(7, 1)
                    
                    logging.info("Instanced rendering initialized successfully")
                    
                except Exception as e:
                    logging.error(f"Failed to initialize instanced rendering: {e}")
                    self.use_instancing = False
                    self.instanced_shader = None
                    self.instanced_vao = None
            
            # Restore the regular shader and VAO as default
            glUseProgram(self.shader)
            glBindVertexArray(self.vao)
            
        return True

    def _ortho(self, l, r, b, t, n, f):
        return np.array([
            [2/(r-l), 0, 0, -(r+l)/(r-l)],
            [0, 2/(t-b), 0, -(t+b)/(t-b)],
            [0, 0, -2/(f-n), -(f+n)/(f-n)],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def _model_matrix(self, pos_data, width, height, viewport_dims):
        win_w, win_h = viewport_dims
        S = np.eye(4, dtype=np.float32)
        S[0,0] = width * pos_data.get('scale_x', 1.0)
        S[1,1] = height * pos_data.get('scale_y', 1.0)
        angle_rad = np.radians(pos_data.get('rotation', 0))
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s, 0, 0], [s,  c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[0,3] = pos_data.get('x', 0.5) * win_w
        T[1,3] = (1.0 - pos_data.get('y', 0.5)) * win_h
        return T @ R @ S

    def _render_instanced_batch(self, layers_list, viewport_dims):
        """Render multiple layers in a single instanced draw call (simplified for RGB only)."""
        if not layers_list or not self.instanced_shader or not self.instanced_vao:
            return []  # Return empty list instead of False
        
        # Filter to only RGB textures (no YUV for now)
        rgb_layers = []
        for layer in layers_list:
            path = layer.get('path')
            if path and path in self.textures:
                tex_info = self.textures[path]
                # Skip YUV textures in instanced mode for now
                if not tex_info.get('is_yuv', False):
                    rgb_layers.append(layer)
        
        if not rgb_layers or len(rgb_layers) < 2:
            return []  # Not worth instancing for single layer
        
        # Limit to max instances
        if len(rgb_layers) > self.max_instances:
            rgb_layers = rgb_layers[:self.max_instances]
        
        # Switch to instanced shader and VAO
        glUseProgram(self.instanced_shader)
        glBindVertexArray(self.instanced_vao)
        
        # Set projection matrix
        projection_matrix = self._ortho(0, viewport_dims[0], 0, viewport_dims[1], -1, 1)
        glUniformMatrix4fv(self.instanced_u_locs['projection'], 1, GL_TRUE, projection_matrix)
        
        # Prepare instance data and bind textures
        instance_data = []
        texture_unit_map = {}
        next_unit = 0
        rendered_layers = []  # Track what we actually rendered
        
        for layer in rgb_layers:
            path = layer.get('path')
            tex_info = self.textures[path]
            
            # Assign texture unit
            if path not in texture_unit_map:
                if next_unit >= 16:
                    break  # Out of texture units
                texture_unit_map[path] = next_unit
                
                # Bind texture to its unit
                glActiveTexture(GL_TEXTURE0 + next_unit)
                glBindTexture(GL_TEXTURE_2D, tex_info['id'])
                
                next_unit += 1
            
            # Create model matrix
            model_matrix = self._model_matrix(
                layer.get('position', {}),
                tex_info['width'],
                tex_info['height'],
                viewport_dims
            )
            
            # Pack instance data: matrix (16) + alpha (1) + tex_index (1)
            for col in range(4):
                for row in range(4):
                    instance_data.append(model_matrix[row, col])
            
            instance_data.append(layer.get('alpha', 1.0))
            instance_data.append(float(texture_unit_map[path]))
            
            rendered_layers.append(layer)  # Mark this layer as rendered
        
        if not instance_data:
            # Restore regular shader
            glUseProgram(self.shader)
            glBindVertexArray(self.vao)
            return []
        
        # Upload instance data
        instance_array = np.array(instance_data, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_array.nbytes, instance_array)
        
        # Draw all instances
        num_instances = len(instance_data) // 18  # 18 floats per instance
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, num_instances)
        
        # Restore regular shader and VAO
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)
        
        return rendered_layers  # Return list of rendered layers
    
    def _render_instanced(self, layers_list, viewport_dims):
        """Render multiple layers in a single instanced draw call."""
        if not layers_list:
            return
            
        # Limit to max instances
        if len(layers_list) > self.max_instances:
            # Split into batches if needed
            for i in range(0, len(layers_list), self.max_instances):
                batch = layers_list[i:i + self.max_instances]
                self._render_instanced(batch, viewport_dims)
            return
        
        # Prepare instance data
        instance_data = []
        texture_unit_assignments = {}
        next_texture_unit = 0
        
        for layer in layers_list:
            path = layer.get('path')
            if not path or path not in self.textures:
                continue
                
            tex_info = self.textures[path]
            
            # Assign texture unit if not already assigned
            if path not in texture_unit_assignments:
                if next_texture_unit >= 16:  # Max texture units
                    # Need to render this batch and start a new one
                    break
                texture_unit_assignments[path] = next_texture_unit
                next_texture_unit += 1
            
            # Create model matrix
            model_matrix = self._model_matrix(
                layer.get('position', {}),
                tex_info['width'],
                tex_info['height'],
                viewport_dims
            )
            
            # Pack instance data: matrix (16) + alpha (1) + tex_index (1) + isYUV (1) + isBGR (1)
            instance_entry = []
            # Flatten matrix (column-major for OpenGL)
            for col in range(4):
                for row in range(4):
                    instance_entry.append(model_matrix[row, col])
            
            # Add per-instance attributes
            instance_entry.append(layer.get('alpha', 1.0))
            instance_entry.append(float(texture_unit_assignments[path]))
            instance_entry.append(1.0 if tex_info.get('is_yuv', False) else 0.0)
            instance_entry.append(1.0 if self.texture_formats.get(path, False) else 0.0)
            
            instance_data.extend(instance_entry)
        
        if not instance_data:
            return
        
        # Bind all textures to their assigned units
        for path, unit in texture_unit_assignments.items():
            if path in self.yuv_textures:
                # Bind YUV textures
                yuv_info = self.yuv_textures[path]
                glActiveTexture(GL_TEXTURE0 + unit + 16)  # Y texture
                glBindTexture(GL_TEXTURE_2D, yuv_info['y_tex'])
                glActiveTexture(GL_TEXTURE0 + unit + 32)  # UV texture
                glBindTexture(GL_TEXTURE_2D, yuv_info['uv_tex'])
            else:
                # Bind regular texture
                glActiveTexture(GL_TEXTURE0 + unit)
                glBindTexture(GL_TEXTURE_2D, self.textures[path]['id'])
        
        # Upload instance data
        instance_array = np.array(instance_data, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_array.nbytes, instance_array)
        
        # Draw all instances in one call
        num_instances = len(instance_data) // 20  # 20 floats per instance
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, num_instances)

    def _update_texture_pbo(self, path):
        """Updates texture using PBO for asynchronous transfer with BGR support."""
        # Quick return if texture exists and is static (not a video)
        if path in self.textures and os.path.splitext(path)[1].lower() not in VIDEO_EXTS:
            return self.textures[path].get('id')
            
        is_video = os.path.splitext(path)[1].lower() in VIDEO_EXTS
        frame = self._get_latest_video_frame(path) if is_video else cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if frame is None:
            return self.textures.get(path, {}).get('id')

        # Determine format and prepare frame
        h, w = frame.shape[:2]
        use_bgr = is_video  # Use BGR upload for videos, RGBA for images
        
        if is_video:
            # Keep BGR format for videos
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            # If already 3 channels, assume it's BGR from the video decoder
            gl_format = GL_BGR
            gl_internal_format = GL_RGB  # Internal storage is still RGB
            channels = 3
        else:
            # Convert images to RGBA as before
            if frame.ndim == 2: 
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
            elif frame.shape[2] == 3: 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            else: 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            gl_format = GL_RGBA
            gl_internal_format = GL_RGBA
            channels = 4
        
        frame = np.ascontiguousarray(frame)
        frame_size = frame.nbytes
        
        # Store format information - we don't need shader swizzling when using GL_BGR
        self.texture_formats[path] = False  # Shader doesn't need to swizzle

        # Check if this is a restart (texture exists but PBO index was reset)
        is_restart = path in self.textures and path not in self.pbo_pool

        if path not in self.textures:
            # First time setup for this media
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            # Allocate texture memory with appropriate format
            glTexImage2D(GL_TEXTURE_2D, 0, gl_internal_format, w, h, 0, gl_format, GL_UNSIGNED_BYTE, frame)
            
            self.textures[path] = {'id': texture_id, 'width': w, 'height': h, 'channels': channels}
            
            # Create PBO pool for video files
            if is_video:
                self.pbo_index[path] = 0
                pbos = glGenBuffers(self.PBO_POOL_SIZE)
                # Convert to list if it's a numpy array
                if isinstance(pbos, np.ndarray):
                    self.pbo_pool[path] = pbos.tolist()
                elif isinstance(pbos, int):
                    self.pbo_pool[path] = [pbos]
                else:
                    self.pbo_pool[path] = list(pbos)
                
                # Initialize ALL PBOs with the first frame data
                for i, pbo in enumerate(self.pbo_pool[path]):
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, int(pbo))
                    glBufferData(GL_PIXEL_UNPACK_BUFFER, frame_size, frame, GL_STREAM_DRAW)
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        else:
            # Update existing texture
            tex_info = self.textures[path]
            
            # Handle size or format changes
            if tex_info['width'] != w or tex_info['height'] != h or tex_info.get('channels', 4) != channels:
                glBindTexture(GL_TEXTURE_2D, tex_info['id'])
                glTexImage2D(GL_TEXTURE_2D, 0, gl_internal_format, w, h, 0, gl_format, GL_UNSIGNED_BYTE, frame)
                tex_info['width'], tex_info['height'], tex_info['channels'] = w, h, channels
                
                # Recreate PBOs with new size if needed
                if path in self.pbo_pool:
                    glDeleteBuffers(len(self.pbo_pool[path]), self.pbo_pool[path])
                
                if is_video:
                    pbos = glGenBuffers(self.PBO_POOL_SIZE)
                    # Convert to list if it's a numpy array
                    if isinstance(pbos, np.ndarray):
                        self.pbo_pool[path] = pbos.tolist()
                    elif isinstance(pbos, int):
                        self.pbo_pool[path] = [pbos]
                    else:
                        self.pbo_pool[path] = list(pbos)
                    self.pbo_index[path] = 0
                    
                    # Initialize ALL PBOs with the first frame
                    for pbo in self.pbo_pool[path]:
                        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, int(pbo))
                        glBufferData(GL_PIXEL_UNPACK_BUFFER, frame_size, frame, GL_STREAM_DRAW)
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
                
                return tex_info['id']
            
            # Handle restart case - recreate PBOs
            if is_restart and is_video:
                self.pbo_index[path] = 0
                pbos = glGenBuffers(self.PBO_POOL_SIZE)
                # Convert to list if it's a numpy array
                if isinstance(pbos, np.ndarray):
                    self.pbo_pool[path] = pbos.tolist()
                elif isinstance(pbos, int):
                    self.pbo_pool[path] = [pbos]
                else:
                    self.pbo_pool[path] = list(pbos)
                
                # Initialize ALL PBOs with the first frame
                for pbo in self.pbo_pool[path]:
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, int(pbo))
                    glBufferData(GL_PIXEL_UNPACK_BUFFER, frame_size, frame, GL_STREAM_DRAW)
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        
        # Use PBO for video updates, direct update for images
        if is_video and path in self.pbo_pool and self.use_pbo:
            # For the first few frames after restart, update texture directly
            current_index = self.pbo_index[path]
            
            # Check if we need to prime the PBO pipeline
            if current_index == 0 and is_restart:
                # Direct update for the first frame after restart
                glBindTexture(GL_TEXTURE_2D, self.textures[path]['id'])
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, gl_format, GL_UNSIGNED_BYTE, frame)
                # Also fill the first PBO
                pbo = int(self.pbo_pool[path][0])
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
                glBufferData(GL_PIXEL_UNPACK_BUFFER, frame_size, frame, GL_STREAM_DRAW)
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
                self.pbo_index[path] = 1
            else:
                # Normal PBO update path
                next_index = (current_index + 1) % self.PBO_POOL_SIZE
                
                # Upload frame data to the NEXT PBO
                next_pbo = int(self.pbo_pool[path][next_index])
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, next_pbo)
                
                # Orphan the buffer to avoid sync issues
                glBufferData(GL_PIXEL_UNPACK_BUFFER, frame_size, None, GL_STREAM_DRAW)
                
                # Map buffer and copy data
                try:
                    ptr = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, frame_size, 
                                         GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT)
                    if ptr:
                        ctypes.memmove(ptr, frame.ctypes.data, frame_size)
                        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER)
                except GLError as e:
                    logging.warning(f"PBO mapping failed, falling back to glBufferSubData: {e}")
                    # Fallback to glBufferSubData if mapping fails
                    glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, frame_size, frame)
                
                # Update texture from CURRENT PBO (data from previous frame)
                current_pbo = int(self.pbo_pool[path][current_index])
                glBindTexture(GL_TEXTURE_2D, self.textures[path]['id'])
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, current_pbo)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, gl_format, GL_UNSIGNED_BYTE, None)
                
                # Advance index for next frame
                self.pbo_index[path] = next_index
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        else:
            # Direct update for images or if PBOs are disabled
            glBindTexture(GL_TEXTURE_2D, self.textures[path]['id'])
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, gl_format, GL_UNSIGNED_BYTE, frame)
                
        return self.textures[path]['id']

    def _update_texture_yuv(self, path):
        """Updates YUV texture for video playback with zero CPU color conversion."""
        frame_data = self._get_latest_video_frame(path)
        
        if frame_data is None:
            # Return existing texture if available
            if path in self.yuv_textures:
                return self.yuv_textures[path]['y_tex']
            return None
        
        # Check if this is YUV data
        if not isinstance(frame_data, dict) or 'format' not in frame_data:
            # Fallback to regular texture update for non-YUV data
            return self._update_texture_pbo(path)
        
        width = frame_data['width']
        height = frame_data['height']
        y_data = frame_data['y_data']
        uv_data = frame_data['uv_data']
        
        # Initialize YUV textures if needed
        if path not in self.yuv_textures:
            # Create Y texture (full resolution)
            y_tex = glGenTextures(1)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, y_tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, y_data)
            
            # Create UV texture (half resolution, 2 channels)
            uv_tex = glGenTextures(1)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, uv_tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, width//2, height//2, 0, GL_RG, GL_UNSIGNED_BYTE, uv_data)
            
            self.yuv_textures[path] = {
                'y_tex': y_tex,
                'uv_tex': uv_tex,
                'width': width,
                'height': height
            }
            
            # Store in main textures dict for compatibility
            self.textures[path] = {'id': y_tex, 'width': width, 'height': height, 'channels': 3, 'is_yuv': True}
            
            if self.use_pbo:
                # Initialize PBOs for YUV data
                self.yuv_pbo_index[path] = 0
                y_pbos = glGenBuffers(self.PBO_POOL_SIZE)
                uv_pbos = glGenBuffers(self.PBO_POOL_SIZE)
                
                if isinstance(y_pbos, np.ndarray):
                    y_pbos = y_pbos.tolist()
                    uv_pbos = uv_pbos.tolist()
                elif isinstance(y_pbos, int):
                    y_pbos = [y_pbos]
                    uv_pbos = [uv_pbos]
                
                self.yuv_pbo_pool[path] = {
                    'y_pbos': y_pbos,
                    'uv_pbos': uv_pbos
                }
                
                # Initialize PBOs with first frame
                for pbo in y_pbos:
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, int(pbo))
                    glBufferData(GL_PIXEL_UNPACK_BUFFER, len(y_data), y_data, GL_STREAM_DRAW)
                for pbo in uv_pbos:
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, int(pbo))
                    glBufferData(GL_PIXEL_UNPACK_BUFFER, len(uv_data), uv_data, GL_STREAM_DRAW)
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        else:
            # Update existing YUV textures
            yuv_info = self.yuv_textures[path]
            
            if self.use_pbo and path in self.yuv_pbo_pool:
                # Use PBO for async upload
                current_idx = self.yuv_pbo_index[path]
                next_idx = (current_idx + 1) % self.PBO_POOL_SIZE
                
                # Upload Y data to next PBO
                next_y_pbo = int(self.yuv_pbo_pool[path]['y_pbos'][next_idx])
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, next_y_pbo)
                glBufferData(GL_PIXEL_UNPACK_BUFFER, len(y_data), None, GL_STREAM_DRAW)
                glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, len(y_data), y_data)
                
                # Update Y texture from current PBO
                current_y_pbo = int(self.yuv_pbo_pool[path]['y_pbos'][current_idx])
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, yuv_info['y_tex'])
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, current_y_pbo)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_BYTE, None)
                
                # Upload UV data to next PBO
                next_uv_pbo = int(self.yuv_pbo_pool[path]['uv_pbos'][next_idx])
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, next_uv_pbo)
                glBufferData(GL_PIXEL_UNPACK_BUFFER, len(uv_data), None, GL_STREAM_DRAW)
                glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, len(uv_data), uv_data)
                
                # Update UV texture from current PBO
                current_uv_pbo = int(self.yuv_pbo_pool[path]['uv_pbos'][current_idx])
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, yuv_info['uv_tex'])
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, current_uv_pbo)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width//2, height//2, GL_RG, GL_UNSIGNED_BYTE, None)
                
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
                self.yuv_pbo_index[path] = next_idx
            else:
                # Direct update without PBO
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, yuv_info['y_tex'])
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_BYTE, y_data)
                
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, yuv_info['uv_tex'])
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width//2, height//2, GL_RG, GL_UNSIGNED_BYTE, uv_data)
        
        return self.yuv_textures[path]['y_tex']

    def render(self, layers, target_monitor=None, transition_state=None, all_assigned_screens=None):
        # Determine all layers to be drawn, including those in transition
        layers_to_draw = []
        if transition_state and transition_state.get('active'):
            progress = transition_state['progress']
            for layer in transition_state['from'].values():
                layer_copy = layer.copy()
                layer_copy['alpha'] = 1.0 - progress
                layer_copy['transition_group'] = 'from'
                layers_to_draw.append(layer_copy)
            for layer in transition_state['to'].values():
                layer_copy = layer.copy()
                layer_copy['alpha'] = progress
                layer_copy['transition_group'] = 'to'
                layers_to_draw.append(layer_copy)
        elif layers:
            for layer in layers.values():
                layer_copy = layer.copy()
                layer_copy['alpha'] = layer.get('alpha', 1.0)
                layers_to_draw.append(layer_copy)

        # Group layers by their target screen(s)
        layers_by_screen = {}
        all_screens = set(all_assigned_screens) if all_assigned_screens is not None else set()
        
        for layer in layers_to_draw:
            # In single-output mode, force all layers to the target monitor
            screens = [target_monitor] if target_monitor is not None else layer.get('screens', [])
            for screen_idx in screens:
                if screen_idx is None: continue
                if screen_idx not in layers_by_screen:
                    layers_by_screen[screen_idx] = []
                layers_by_screen[screen_idx].append(layer)
                all_screens.add(screen_idx)
        
        # BUG FIX: Ensure that in single output mode, the target monitor is always active
        # even if there are no layers, to keep the screen black.
        if target_monitor is not None:
            all_screens.add(target_monitor)

        if not all_screens:
            self.close_window()
            return

        # Calculate bounding box for a single window spanning all required monitors
        min_x = min(self.monitors[s].x for s in all_screens)
        min_y = min(self.monitors[s].y for s in all_screens)
        max_r = max(self.monitors[s].x + self.monitors[s].width for s in all_screens)
        max_b = max(self.monitors[s].y + self.monitors[s].height for s in all_screens)
        total_w, total_h = max_r - min_x, max_b - min_y
        
        if not self._create_window((min_x, min_y, total_w, total_h), is_spanning=True):
            return

        # Prepare for rendering
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

        # Pre-update all textures to minimize updates during rendering
        texture_update_cache = {}
        for screen_idx in all_screens:
            screen_layers = layers_by_screen.get(screen_idx, [])
            for layer in screen_layers:
                path = layer.get('path')
                if path and os.path.exists(path) and path not in texture_update_cache:
                    # Use YUV path for videos in OpenGL mode
                    ext = os.path.splitext(path)[1].lower()
                    if ext in VIDEO_EXTS and path in self.active_players:
                        # Check if the player is outputting YUV
                        player_info = self.active_players[path]
                        if player_info.get('type') == 'gpu_queue':
                            texture_update_cache[path] = self._update_texture_yuv(path)
                        else:
                            texture_update_cache[path] = self._update_texture_pbo(path)
                    else:
                        texture_update_cache[path] = self._update_texture_pbo(path)

        # Check if we're rendering a static cue (no transition)
        use_static_cache = False
        static_cache_key = None
        pre_rendered_data = None

        if not (transition_state and transition_state.get('active')):
            # Not in transition, check if this is a static cue
            if self.is_cue_static(layers):
                static_cache_key = self.get_cue_cache_key(layers, target_monitor)
                
                if static_cache_key in self.static_cue_cache:
                    # Use cached pre-rendered cue
                    pre_rendered_data = self.static_cue_cache[static_cache_key]
                    use_static_cache = True
                    self.static_cue_cache_hits += 1
                    logging.debug(f"Static cue cache hit (total hits: {self.static_cue_cache_hits})")
                else:
                    # Pre-render the static cue
                    pre_rendered_data = self.pre_render_static_cue(layers, target_monitor)
                    if pre_rendered_data:
                        self.static_cue_cache[static_cache_key] = pre_rendered_data
                        use_static_cache = True
                        self.static_cue_cache_misses += 1
                        logging.debug(f"Pre-rendered static cue (total misses: {self.static_cue_cache_misses})")
                        
                        # Limit cache size
                        if len(self.static_cue_cache) > 10:
                            # Remove oldest entries
                            oldest_key = next(iter(self.static_cue_cache))
                            old_data = self.static_cue_cache.pop(oldest_key)
                            self.cleanup_pre_rendered_cue(old_data)

        # Iterate through each screen and render its layers
        for screen_idx in all_screens:
            mon = self.monitors[screen_idx]
            screen_layers = layers_by_screen.get(screen_idx, [])
            
            # Calculate viewport position relative to the spanning window
            vp_x = mon.x - min_x
            vp_y = total_h - (mon.y - min_y) - mon.height # OpenGL Y is bottom-up
            glViewport(vp_x, vp_y, mon.width, mon.height)
            
            # Set projection matrix
            projection_matrix = self._ortho(0, mon.width, 0, mon.height, -1, 1)
            glUniformMatrix4fv(self.u_locs['projection'], 1, GL_TRUE, projection_matrix)
            
            # Check if we can use pre-rendered static cue
            if use_static_cache and pre_rendered_data and target_monitor == screen_idx:
                self.render_pre_rendered_cue(pre_rendered_data, (mon.width, mon.height))
                continue  # Skip regular rendering for this screen
            
            # Sort layers once for this screen
            sorted_layers = sorted(screen_layers, key=lambda x: x.get('order', 0))
            
            # Try instanced rendering first if enabled and worthwhile
            rendered_layers = []
            if self.use_instancing and len(sorted_layers) >= 2:
                rendered_layers = self._render_instanced_batch(sorted_layers, (mon.width, mon.height))
            
            # Create set of rendered paths for quick lookup
            rendered_paths = {layer.get('path') for layer in rendered_layers}
            
            # Render remaining layers (YUV videos, single layers, etc.) using immediate mode
            if len(rendered_layers) < len(sorted_layers):
                # Render layers that weren't handled by instancing
                for layer in sorted_layers:
                    path = layer.get('path')
                    
                    # Skip if already rendered by instancing
                    if path in rendered_paths:
                        continue
                        
                    if not path or not os.path.exists(path): 
                        continue
                    
                    texture_id = texture_update_cache.get(path)
                    if not texture_id: 
                        continue
                    
                    tex_info = self.textures[path]
                    
                    # Check if this is a YUV texture
                    is_yuv = tex_info.get('is_yuv', False)
                    
                    if is_yuv and path in self.yuv_textures:
                        # Bind YUV textures
                        yuv_info = self.yuv_textures[path]
                        glActiveTexture(GL_TEXTURE0)
                        glBindTexture(GL_TEXTURE_2D, yuv_info['y_tex'])
                        glUniform1i(self.u_locs['yTexture'], 0)
                        
                        glActiveTexture(GL_TEXTURE1)
                        glBindTexture(GL_TEXTURE_2D, yuv_info['uv_tex'])
                        glUniform1i(self.u_locs['uvTexture'], 1)
                        
                        glUniform1i(self.u_locs['isYUV'], 1)
                    else:
                        # Regular RGB/BGR texture
                        glActiveTexture(GL_TEXTURE0)
                        glBindTexture(GL_TEXTURE_2D, texture_id)
                        glUniform1i(self.u_locs['ourTexture'], 0)
                        glUniform1i(self.u_locs['isYUV'], 0)
                        
                        # Update texture-specific uniforms
                        is_bgr = self.texture_formats.get(path, False)
                        glUniform1i(self.u_locs['isBGR'], int(is_bgr))
                    
                    # Set layer-specific uniforms
                    model_matrix = self._model_matrix(
                        layer.get('position', {}), 
                        tex_info['width'], 
                        tex_info['height'], 
                        (mon.width, mon.height)
                    )
                    
                    glUniformMatrix4fv(self.u_locs['model'], 1, GL_TRUE, model_matrix)
                    glUniform1f(self.u_locs['alpha'], layer.get('alpha', 1.0))
                    
                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        pygame.display.flip()
        
        # ---- Async Preview Push ----
        try:
            if self.preview_bridge and self.use_pbo:
                frame = self.pbo_helper.try_map_latest()
                if frame is not None:
                    self.preview_bridge.push((frame.copy(), self.backbuffer_size, time.time()))
        except Exception:
            pass
        # ----------------------------

    def read_pixels_for_preview(self):
        if not self.window: return None
        glFinish()
        
        # For preview, we only read the primary monitor's viewport (usually index 0)
        preview_monitor_idx = 0
        if self.spanning_window_details and self.monitors:
            min_x, min_y, total_w, total_h = self.spanning_window_details
            mon = self.monitors[preview_monitor_idx]
            
            # Check if the primary monitor is part of the current spanning window
            if not (mon.x >= min_x and mon.x + mon.width <= min_x + total_w and
                    mon.y >= min_y and mon.y + mon.height <= min_y + total_h):
                return None # Primary monitor not being rendered

            vp_x = mon.x - min_x
            vp_y = total_h - (mon.y - min_y) - mon.height
            w, h = mon.width, mon.height
        else: # Single window mode
            w, h = self.window.get_size()
            vp_x, vp_y = 0, 0
            
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(vp_x, vp_y, w, h, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (w, h), pixels)
        return image.transpose(Image.FLIP_TOP_BOTTOM)

    def release_texture(self, path):
        """Cleanup texture and associated PBOs."""
        if path in self.textures:
            if self.window:
                tex_info = self.textures.pop(path)
                
                # Clean up YUV textures if they exist
                if path in self.yuv_textures:
                    yuv_info = self.yuv_textures.pop(path)
                    glDeleteTextures(1, [yuv_info['y_tex']])
                    glDeleteTextures(1, [yuv_info['uv_tex']])
                    
                    # Clean up YUV PBOs
                    if path in self.yuv_pbo_pool:
                        pbos_info = self.yuv_pbo_pool.pop(path)
                        glDeleteBuffers(len(pbos_info['y_pbos']), pbos_info['y_pbos'])
                        glDeleteBuffers(len(pbos_info['uv_pbos']), pbos_info['uv_pbos'])
                    self.yuv_pbo_index.pop(path, None)
                else:
                    # Regular texture cleanup
                    glDeleteTextures(1, [tex_info['id']])
                    
                    # Clean up regular PBOs
                    if path in self.pbo_pool:
                        pbos = self.pbo_pool.pop(path)
                        glDeleteBuffers(len(pbos), pbos)
                
                self.pbo_index.pop(path, None)

    def _release_gl_resources(self):
        """Clean up all OpenGL resources."""
        if self.shader: glDeleteProgram(self.shader)
        if self.vao: glDeleteVertexArrays(1, [self.vao])
        if self.vbo: glDeleteBuffers(1, [self.vbo])
        if self.ebo: glDeleteBuffers(1, [self.ebo])
        
        # Clean up textures
        if self.textures:
            try:
                all_texture_ids = [t['id'] for t in self.textures.values()]
                glDeleteTextures(len(all_texture_ids), all_texture_ids)
            except (GLError, KeyError): pass
        
        # Clean up all PBOs
        for path, pbos in self.pbo_pool.items():
            try:
                glDeleteBuffers(len(pbos), pbos)
            except GLError:
                pass
        
        self.shader = self.vao = self.vbo = self.ebo = None
        self.textures.clear()
        self.u_locs.clear()
        self.pbo_pool.clear()
        self.pbo_index.clear()
        self.texture_formats.clear()

    def close_window(self):
        if self.window:
            self._release_gl_resources()
            self.spanning_window_details = None
        super().close_window()

    def clear(self):
        super().clear()
        if self.window:
            self.close_window()


class PygameRenderEngine(BaseRenderEngine):
    def __init__(self, media_manager, mode='cpu', hw_accel=None):
        super().__init__(media_manager, mode, hw_accel)
        logging.info(f"PygameRenderEngine initialized in '{mode}' mode.")
        if self.mode == 'pygame_gpu':
            logging.info(f"-> GPU Accel: {hw_accel}")
        self.spanning_window_details = None
        
        # Add transition surface caching with separate keys for from/to
        self.transition_cache = {
            'from_surface': None,
            'to_surface': None,
            'from_cache_key': None,
            'to_cache_key': None
        }
        
        # Surface cache for transformed media
        self.surface_cache = OrderedDict()  # Cache for transformed surfaces
        self.surface_cache_size = 50  # Limit cache size
        self.cache_hits = 0  # For debugging
        self.cache_misses = 0  # For debugging

    def _init_pygame_and_gl(self):
        import os, sys, pygame
        pygame.init()
        pygame.display.init()

        # Ensure the output window uses the same icon as the main GUI (.ico file)
        try:
            from PIL import Image
            base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ico_path = os.path.join(base_dir, 'src', 'visualdeck.ico')
            if os.path.exists(ico_path):
                im = Image.open(ico_path)
                # Pick largest frame in the .ico for best fidelity
                if getattr(im, "n_frames", 1) > 1:
                    best = None; best_area = -1
                    for i in range(im.n_frames):
                        im.seek(i)
                        area = im.size[0] * im.size[1]
                        if area > best_area:
                            best = im.copy()
                            best_area = area
                    im = best
                if im.mode != "RGBA":
                    im = im.convert("RGBA")
                surf = pygame.image.frombuffer(im.tobytes(), im.size, "RGBA")
                pygame.display.set_icon(surf)
        except Exception:
            # Best-effort; never fail initialization over an icon
            pass


    def pre_render_static_cue(self, layers, dims):
        """Pre-render a static cue to a surface."""
        if not layers:
            return None
        
        # Use the existing _render_layers_to_surface method
        surface = self._render_layers_to_surface(layers, dims)
        
        if surface:
            # Convert the surface for optimal blitting
            surface = surface.convert_alpha()
        
        return surface
        
    def clear(self):
        """Clear all resources including transition cache and surface cache."""
        # Clear transition cache
        self.transition_cache = {
            'from_surface': None,
            'to_surface': None,
            'from_cache_key': None,
            'to_cache_key': None
        }
        # Clear surface cache
        self.surface_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        # Call parent class clear method
        super().clear()

    def _create_window(self, geom):
        x, y, w, h = geom
        if self.window and self.window.get_size() == (w, h) and self.spanning_window_details == geom:
            return True

        # Clear transition cache on window change
        self.transition_cache = {
            'from_surface': None,
            'to_surface': None,
            'from_cache_key': None,
            'to_cache_key': None
        }

        self.close_window()
        
        # Prevent implicit centering and request explicit position
        os.environ['SDL_VIDEO_CENTERED'] = '0'
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
        
        # Use only safe, standard flags
        flags = NOFRAME | DOUBLEBUF
        
        # Try to use a hardware surface, but fall back safely if not supported
        try:
            self.window = pygame.display.set_mode((w, h), flags | HWSURFACE)
        except pygame.error:
            self.window = pygame.display.set_mode((w, h), flags)

        pygame.display.set_caption("VisualDeck Output (Pygame)")
        self.spanning_window_details = geom

        # Force the position after creation (some drivers ignore WINDOW_POS on first create)
        try:
            import pygame._sdl2 as sdl2
            win = sdl2.Window.from_display_module()
            win.position = (int(x), int(y))
        except Exception:
            pass

        return True

    def _get_transition_cache_key(self, layers, dims):
        """Generate a cache key for transition surfaces based on layers and dimensions."""
        # Create a unique key based on layer paths and dimensions
        layer_info = []
        for layer_id, layer in sorted(layers.items()):
            if layer.get('path'):
                layer_info.append((layer_id, layer['path'], layer.get('order', 0)))
        return (tuple(layer_info), dims)

    def _render_layers_to_surface(self, layers, dims):
        w, h = dims
        
        if not layers:
            logging.debug(f"No layers to render")
            return None

        # Ensure layers is a dictionary before calling .values()
        if isinstance(layers, list):
            # This is a quick fix for the error.
            # A better solution would be to ensure
            # the data structure is consistent throughout the application.
            # Assuming the list contains layer dictionaries.
            layers_dict = {i: layer for i, layer in enumerate(layers)}
        else:
            layers_dict = layers

        surf = pygame.Surface((w, h), SRCALPHA)
        surf.fill((0, 0, 0, 0))
        
        drawn_anything = False
        layers_rendered = 0
        for m in sorted(layers_dict.values(), key=lambda x: x.get('order', 0)):
            sf = self._to_surface(m)
            if not sf: 
                logging.debug(f"No surface returned for layer with path: {m.get('path', 'Unknown')}")
                continue
            
            drawn_anything = True
            layers_rendered += 1
            pos = m.get('position', {})
            x = int(pos.get('x', 0.5) * w)
            y = int(pos.get('y', 0.5) * h)
            rect = sf.get_rect(center=(x, y))
            surf.blit(sf, rect)
            logging.debug(f"Blitted surface at ({x}, {y}), size: {sf.get_size()}")

        logging.debug(f"Rendered {layers_rendered} layers, drawn_anything: {drawn_anything}")
        return surf if drawn_anything else None

    def render_layers_to_image(self, layers, dimensions):
        # FIX: Ensure a display mode is set before performing surface operations.
        needs_display_init = not pygame.display.get_active()
        if needs_display_init:
            # If no display is active (e.g., generating a CPU preview in a thread),
            # create a temporary 1x1 pixel display to provide the necessary context.
            pygame.display.set_mode((1, 1), NOFRAME)

        pygame_surface = self._render_layers_to_surface(layers, dimensions)
        
        pil_image = None
        if pygame_surface:
            try:
                image_data = pygame.image.tostring(pygame_surface, "RGBA", False)
                pil_image = Image.frombytes("RGBA", dimensions, image_data)
            except pygame.error as e:
                logging.error(f"Failed to convert Pygame surface to Image for preview: {e}")
                pil_image = None

        # We don't quit the display here, as the main application thread will handle it.
        # The check for get_active() prevents this from running unnecessarily.

        return pil_image

    def _to_surface(self, media):
        path = media.get('path')
        if not path or not os.path.exists(path): 
            return None

        ext = os.path.splitext(path)[1].lower()

        # Generate cache key including all transformations
        pos = media.get('position', {})
        cache_key = (
            path,
            pos.get('scale_x', 1.0),
            pos.get('scale_y', 1.0),
            pos.get('rotation', 0),
            id(self.video_frame_cache.get(path)) if ext in VIDEO_EXTS else None
        )

        # Only cache non-video surfaces
        if ext not in VIDEO_EXTS and cache_key in self.surface_cache:
            self.surface_cache.move_to_end(cache_key)
            self.cache_hits += 1
            return self.surface_cache[cache_key]

        self.cache_misses += 1

        # Original surface creation logic
        frame_data = self._get_latest_video_frame(path) if ext in VIDEO_EXTS else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if frame_data is None: 
            return None

        sf = None
        if ext in VIDEO_EXTS and isinstance(frame_data, tuple):
            # This is pygame_gpu mode data (already processed)
            frame_bytes, dims = frame_data
            sf = pygame.image.frombuffer(frame_bytes, dims, "BGRA")
        else:
            # Handles both CPU mode video frames and static images
            frm = frame_data
            if frm.ndim == 2: 
                frm = cv2.cvtColor(frm, cv2.COLOR_GRAY2BGRA)
            elif frm.shape[2] == 3: 
                frm = cv2.cvtColor(frm, cv2.COLOR_BGR2BGRA)
            # If shape[2] == 4, already BGRA

            # Create surface from the numpy array
            sf = pygame.image.frombuffer(frm.tobytes(), frm.shape[1::-1], "BGRA")

            # Convert the surface for optimal blitting
            sf = sf.convert_alpha()
            logging.debug(f"Created surface: {sf.get_size()}")

        w0, h0 = sf.get_width(), sf.get_height()
        scale_x = pos.get('scale_x', 1.0)
        scale_y = pos.get('scale_y', 1.0)
        new_w, new_h = int(w0 * scale_x), int(h0 * scale_y)

        if new_w > 0 and new_h > 0 and (scale_x != 1.0 or scale_y != 1.0):
            sf = pygame.transform.smoothscale(sf, (new_w, new_h))

        if pos.get('rotation', 0):
            sf = pygame.transform.rotate(sf, pos.get('rotation', 0))

        # LRU cache for non-video surfaces
        if ext not in VIDEO_EXTS:
            self.surface_cache[cache_key] = sf
            self.surface_cache.move_to_end(cache_key)
            if len(self.surface_cache) > self.surface_cache_size:
                self.surface_cache.popitem(last=False)

        # Log cache performance occasionally
        if (self.cache_hits + self.cache_misses) % 100 == 0 and self.cache_hits > 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100
            logging.debug(f"Surface cache hit rate: {hit_rate:.1f}%")

        return sf


    def get_transition_frame(self, from_surf, to_surf, progress):
        """Blend two surfaces for transition effect."""
        if not from_surf and not to_surf:
            return None
            
        if from_surf:
            size = from_surf.get_size()
        elif to_surf:
            size = to_surf.get_size()
        else:
            return None 

        # Create base surface only if we have valid input
        base = pygame.Surface(size, SRCALPHA)
        base.fill((0, 0, 0, 0))

        # Apply alpha and blit
        if from_surf and progress < 1.0:
            from_alpha = int((1.0 - progress) * 255)
            if from_alpha > 0:
                from_surf_copy = from_surf.copy()
                from_surf_copy.set_alpha(from_alpha)
                base.blit(from_surf_copy, (0, 0))

        if to_surf and progress > 0.0:
            to_alpha = int(progress * 255)
            if to_alpha > 0:
                to_surf_copy = to_surf.copy()
                to_surf_copy.set_alpha(to_alpha)
                base.blit(to_surf_copy, (0, 0))
                
        return base
        
    def _layers_contain_video(self, layers):
        """Check if any layer contains a video file."""
        for layer in layers.values():
            path = layer.get('path')
            if path and os.path.exists(path):
                ext = os.path.splitext(path)[1].lower()
                if ext in VIDEO_EXTS:
                    return True
        return False

    def render(self, layers, target_monitor=None, transition_state=None, all_assigned_screens=None):
        layers_to_draw = []
        if transition_state and transition_state.get('active'):
            progress = transition_state['progress']
            for layer in transition_state['from'].values():
                layer['alpha'] = 1.0 - progress
                layers_to_draw.append(layer)
            for layer in transition_state['to'].values():
                layer['alpha'] = progress
                layers_to_draw.append(layer)
        elif layers:
            layers_to_draw.extend(layers.values())

        if target_monitor is not None:
            # FIX: Validate the target_monitor index
            if not (self.monitors and 0 <= target_monitor < len(self.monitors)):
                logging.warning(f"Invalid target_monitor index ({target_monitor}) provided. Closing output window.")
                self.close_window()
                return

            if not self._create_window((self.monitors[target_monitor].x, self.monitors[target_monitor].y, 
                                       self.monitors[target_monitor].width, self.monitors[target_monitor].height)): 
                logging.debug("Failed to create window")
                return
            
            final_surface = None
            dims = self.window.get_size()
            
            # Check for static cue caching opportunity (only when not in transition)
            use_static_cache = False
            static_cache_key = None
            
            if not (transition_state and transition_state.get('active')):
                # Not in transition, check if this is a static cue
                if self.is_cue_static(layers):
                    static_cache_key = self.get_cue_cache_key(layers, target_monitor)
                    
                    if static_cache_key in self.static_cue_cache:
                        # Use cached pre-rendered cue
                        final_surface = self.static_cue_cache[static_cache_key]
                        use_static_cache = True
                        self.static_cue_cache_hits += 1
                        logging.debug(f"Static cue cache hit (total hits: {self.static_cue_cache_hits})")
                    else:
                        # Pre-render the static cue
                        final_surface = self.pre_render_static_cue(layers, dims)
                        if final_surface:
                            # Make a copy for caching to avoid surface modification issues
                            cached_surface = final_surface.copy()
                            self.static_cue_cache[static_cache_key] = cached_surface
                            use_static_cache = True
                            self.static_cue_cache_misses += 1
                            logging.debug(f"Pre-rendered static cue (total misses: {self.static_cue_cache_misses})")
                            
                            # Limit cache size
                            if len(self.static_cue_cache) > 10:
                                # Remove oldest entry
                                oldest_key = next(iter(self.static_cue_cache))
                                self.static_cue_cache.pop(oldest_key)
                                logging.debug("Removed oldest static cue from cache")
            
            # If we didn't use static cache, proceed with normal rendering
            if not use_static_cache:
                if transition_state and transition_state.get('active'):
                    # Handle transition rendering (existing code)
                    from_has_video = self._layers_contain_video(transition_state['from'])
                    to_has_video = self._layers_contain_video(transition_state['to'])
                    
                    # Use transition cache for static content
                    if from_has_video:
                        from_surface = self._render_layers_to_surface(transition_state['from'], dims)
                    else:
                        from_cache_key = self._get_transition_cache_key(transition_state['from'], dims)
                        if self.transition_cache.get('from_cache_key') != from_cache_key:
                            self.transition_cache['from_surface'] = self._render_layers_to_surface(transition_state['from'], dims)
                            self.transition_cache['from_cache_key'] = from_cache_key
                        from_surface = self.transition_cache['from_surface']
                    
                    if to_has_video:
                        to_surface = self._render_layers_to_surface(transition_state['to'], dims)
                    else:
                        to_cache_key = self._get_transition_cache_key(transition_state['to'], dims)
                        if self.transition_cache.get('to_cache_key') != to_cache_key:
                            self.transition_cache['to_surface'] = self._render_layers_to_surface(transition_state['to'], dims)
                            self.transition_cache['to_cache_key'] = to_cache_key
                        to_surface = self.transition_cache['to_surface']
                    
                    final_surface = self.get_transition_frame(from_surface, to_surface, transition_state['progress'])
                else:
                    # Clear transition cache if not transitioning
                    if self.transition_cache.get('from_cache_key') is not None or self.transition_cache.get('to_cache_key') is not None:
                        self.transition_cache = {
                            'from_surface': None,
                            'to_surface': None,
                            'from_cache_key': None,
                            'to_cache_key': None
                        }
                    # Render current layers normally
                    final_surface = self._render_layers_to_surface(layers, dims)

            # Blit the final surface to the window
            self.window.fill((0, 0, 0))
            if final_surface:
                self.window.blit(final_surface, (0, 0))
            pygame.display.flip()
            return

        # --- MULTI-MONITOR MODE ---
        layers_by_screen = {}
        all_screens = set(all_assigned_screens) if all_assigned_screens is not None else set()
        
        for layer in layers_to_draw:
            for screen_idx in layer.get('screens', []):
                if screen_idx is None: continue
                if screen_idx not in layers_by_screen:
                    layers_by_screen[screen_idx] = []
                layers_by_screen[screen_idx].append(layer)
                all_screens.add(screen_idx)

        if not all_screens:
            self.close_window()
            return

        min_x = min(self.monitors[s].x for s in all_screens)
        min_y = min(self.monitors[s].y for s in all_screens)
        max_r = max(self.monitors[s].x + self.monitors[s].width for s in all_screens)
        max_b = max(self.monitors[s].y + self.monitors[s].height for s in all_screens)
        total_w, total_h = max_r - min_x, max_b - min_y
        
        if not self._create_window((min_x, min_y, total_w, total_h)): 
            return

        self.window.fill((0, 0, 0))

        # Multi-monitor mode with static cue caching per screen
        for screen_idx in all_screens:
            mon = self.monitors[screen_idx]
            dims = (mon.width, mon.height)
            blit_pos = (mon.x - min_x, mon.y - min_y)
            
            final_surface = None
            
            if transition_state and transition_state.get('active'):
                # Handle transitions in multi-monitor mode
                layers_from = {k: v for k, v in transition_state['from'].items() if screen_idx in v.get('screens', [])}
                layers_to = {k: v for k, v in transition_state['to'].items() if screen_idx in v.get('screens', [])}
                
                from_surf = self._render_layers_to_surface(layers_from, dims)
                to_surf = self._render_layers_to_surface(layers_to, dims)
                final_surface = self.get_transition_frame(from_surf, to_surf, transition_state['progress'])
            else:
                # Check for static cue caching per screen
                screen_layers = layers_by_screen.get(screen_idx, [])
                screen_layers_dict = {i: layer for i, layer in enumerate(screen_layers)}
                
                if self.is_cue_static(screen_layers_dict):
                    # Generate cache key for this screen's layers
                    static_cache_key = self.get_cue_cache_key(screen_layers_dict, screen_idx)
                    
                    if static_cache_key in self.static_cue_cache:
                        # Use cached surface for this screen
                        final_surface = self.static_cue_cache[static_cache_key]
                        self.static_cue_cache_hits += 1
                        logging.debug(f"Static cue cache hit for screen {screen_idx}")
                    else:
                        # Pre-render and cache this screen's static content
                        final_surface = self._render_layers_to_surface(screen_layers, dims)
                        if final_surface:
                            cached_surface = final_surface.copy()
                            self.static_cue_cache[static_cache_key] = cached_surface
                            self.static_cue_cache_misses += 1
                            logging.debug(f"Pre-rendered static cue for screen {screen_idx}")
                            
                            # Limit cache size
                            if len(self.static_cue_cache) > 20:  # Allow more entries for multi-monitor
                                oldest_key = next(iter(self.static_cue_cache))
                                self.static_cue_cache.pop(oldest_key)
                else:
                    # Render normally for dynamic content
                    final_surface = self._render_layers_to_surface(screen_layers, dims)
            
            if final_surface:
                self.window.blit(final_surface, blit_pos)

        pygame.display.flip()
