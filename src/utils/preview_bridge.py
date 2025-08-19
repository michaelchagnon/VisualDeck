# =====================================================================
# utils/preview_bridge.py
# =====================================================================

import threading

class PreviewBridge:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def push(self, frame_tuple):
        with self._lock:
            self._frame = frame_tuple

    def pop_latest(self):
        with self._lock:
            f = self._frame
            self._frame = None
            return f
