# =====================================================================
# src/media_manager.py
# =====================================================================
import os
import cv2
import time
import logging
import av
from functools import lru_cache

class MediaManager:
    """A lightweight class to get media information using a cache."""
    def __init__(self):
        # This class no longer manages directories or a list of media.
        # It's a utility for fetching media metadata.
        pass

    @lru_cache(maxsize=256) # Cache info for up to 256 files
    def get_media_info(self, path):
        """Returns size, duration, width, and height of the media file."""
        if not path or not os.path.exists(path):
            return 0, 0, 0, 0
            
        try:
            size = os.path.getsize(path)
            ext = os.path.splitext(path)[1].lower()
            duration, w, h = 0, 0, 0
            
            if ext in ['.mp4', '.avi', '.mov', '.mkv']:
                # --- THIS IS THE REWRITTEN LOGIC ---
                container = None
                try:
                    container = av.open(path)
                    stream = container.streams.video[0]
                    w = stream.width
                    h = stream.height
                    # Duration in PyAV is in stream.time_base units, convert to seconds
                    if stream.duration is not None and stream.time_base is not None:
                        duration = stream.duration * stream.time_base
                    else: # Fallback for streams with no duration info
                         duration = container.duration / av.time_base
                finally:
                    if container:
                        container.close()
                # --- END REWRITTEN LOGIC ---
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                # OpenCV is still fine for images
                img = cv2.imread(path)
                if img is not None:
                    h, w, _ = img.shape
            
            return size, duration, w, h

        except Exception as e:
            logging.warning(f"Could not get media info for {path}: {e}")
            return 0, 0, 0, 0 # Failed to open or read
