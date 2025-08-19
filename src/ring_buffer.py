# =====================================================================
# src/ring_buffer.py
# =====================================================================
import threading
import numpy as np
from collections import deque
import time

class FastRingBuffer:
    """
    A fast ring buffer implementation optimized for video frames.
    Uses a fixed-size buffer to avoid allocations during operation.
    """
    def __init__(self, maxsize, frame_shape=None):
        self.maxsize = maxsize
        self.frame_shape = frame_shape
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        
        if frame_shape:
            # Pre-allocate numpy arrays for zero-copy operation
            self.buffer = [np.empty(frame_shape, dtype=np.uint8) for _ in range(maxsize)]
            self.is_preallocated = True
        else:
            # Fallback to regular object storage
            self.buffer = [None] * maxsize
            self.is_preallocated = False
            
        self.read_idx = 0
        self.write_idx = 0
        self.count = 0
        
    def put(self, item, timeout=None):
        """Put an item into the buffer."""
        with self.not_full:
            if timeout is None:
                while self.count == self.maxsize:
                    self.not_full.wait()
            else:
                deadline = time.monotonic() + timeout
                while self.count == self.maxsize:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Ring buffer is full")
                    self.not_full.wait(remaining)
            
            if self.is_preallocated and isinstance(item, np.ndarray):
                # Copy data into pre-allocated buffer
                np.copyto(self.buffer[self.write_idx], item)
            else:
                # Store reference
                self.buffer[self.write_idx] = item
                
            self.write_idx = (self.write_idx + 1) % self.maxsize
            self.count += 1
            self.not_empty.notify()
    
    def put_nowait(self, item):
        """Put an item without waiting. Raises exception if full."""
        with self.lock:
            if self.count == self.maxsize:
                raise Exception("Ring buffer is full")
            
            if self.is_preallocated and isinstance(item, np.ndarray):
                np.copyto(self.buffer[self.write_idx], item)
            else:
                self.buffer[self.write_idx] = item
                
            self.write_idx = (self.write_idx + 1) % self.maxsize
            self.count += 1
            self.not_empty.notify()

    def put_nowait_drop_old(self, item):
        """Put item, dropping oldest frames if full - no exceptions"""
        with self.lock:
            if self.count == self.maxsize:
                # Skip read pointer ahead to drop oldest frame
                self.read_idx = (self.read_idx + 1) % self.maxsize
                self.count -= 1
            
            if self.is_preallocated and isinstance(item, np.ndarray):
                np.copyto(self.buffer[self.write_idx], item)
            else:
                self.buffer[self.write_idx] = item
                
            self.write_idx = (self.write_idx + 1) % self.maxsize
            self.count += 1
            self.not_empty.notify()
                
    def get(self, timeout=None):
        """Get an item from the buffer."""
        with self.not_empty:
            if timeout is None:
                while self.count == 0:
                    self.not_empty.wait()
            else:
                deadline = time.monotonic() + timeout
                while self.count == 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Ring buffer is empty")
                    self.not_empty.wait(remaining)
            
            item = self.buffer[self.read_idx]
            if self.is_preallocated:
                # Return a copy to avoid data races
                result = item.copy()
            else:
                result = item
                self.buffer[self.read_idx] = None  # Help GC
                
            self.read_idx = (self.read_idx + 1) % self.maxsize
            self.count -= 1
            self.not_full.notify()
            return result
    
    def get_nowait(self):
        """Get an item without waiting. Raises exception if empty."""
        with self.lock:
            if self.count == 0:
                raise Exception("Ring buffer is empty")
            
            item = self.buffer[self.read_idx]
            if self.is_preallocated:
                result = item.copy()
            else:
                result = item
                self.buffer[self.read_idx] = None
                
            self.read_idx = (self.read_idx + 1) % self.maxsize
            self.count -= 1
            self.not_full.notify()
            return result
    
    def qsize(self):
        """Return the current number of items in the buffer."""
        with self.lock:
            return self.count
    
    def empty(self):
        """Return True if the buffer is empty."""
        with self.lock:
            return self.count == 0
    
    def full(self):
        """Return True if the buffer is full."""
        with self.lock:
            return self.count == self.maxsize
    
    def get_latest(self):
        """Get the most recent item, dropping older ones. Returns None if empty."""
        with self.lock:
            if self.count == 0:
                return None
            
            # Skip to the most recent item
            if self.count > 1:
                skip_count = self.count - 1
                self.read_idx = (self.read_idx + skip_count) % self.maxsize
                self.count = 1
            
            item = self.buffer[self.read_idx]
            if self.is_preallocated:
                result = item.copy()
            else:
                result = item
                self.buffer[self.read_idx] = None
                
            self.read_idx = (self.read_idx + 1) % self.maxsize
            self.count = 0
            self.not_full.notify()
            return result


class SimpleFrameQueue:
    """
    A simpler, faster alternative to Queue for frames using deque.
    Less features but lower overhead.
    """
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = deque()
        self.lock = threading.Lock()
        
    def put_nowait(self, item):
        with self.lock:
            if len(self.queue) >= self.maxsize:
                # Drop oldest frame
                self.queue.popleft()
            self.queue.append(item)
    
    def get_nowait(self):
        with self.lock:
            if not self.queue:
                raise Exception("Queue is empty")
            return self.queue.popleft()
    
    def get(self, timeout=0.1):
        """Simple get with timeout - polls instead of using conditions for simplicity."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.queue:
                    return self.queue.popleft()
            time.sleep(0.001)  # Small sleep to avoid busy waiting
        raise TimeoutError("Queue is empty")
    
    def qsize(self):
        with self.lock:
            return len(self.queue)
    
    def empty(self):
        with self.lock:
            return len(self.queue) == 0
    
    def full(self):
        with self.lock:
            return len(self.queue) >= self.maxsize
    
    def get_latest(self):
        """Get the most recent item, dropping all others."""
        with self.lock:
            if not self.queue:
                return None
            # Keep only the last item
            while len(self.queue) > 1:
                self.queue.popleft()
            return self.queue.popleft()
