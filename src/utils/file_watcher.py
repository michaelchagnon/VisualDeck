# =====================================================================
# src/utils/file_watcher.py
# =====================================================================

import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileWatcherHandler(FileSystemEventHandler):
    def __init__(self, media_manager):
        self.media_manager = media_manager
        self.next_layer = 0
    def on_created(self, event):
        if not event.is_directory:
            # This is deprecated as we no longer watch a single media folder
            pass

class FileWatcher:
    "Watch media folder for new files, handling missing dirs gracefully."
    def __init__(self, media_manager):
        # This class is currently unused due to the new referenced media model
        # but is kept for potential future features.
        self.media_manager = media_manager
        self.observer = Observer()
    def start(self):
        pass
    def stop(self):
        self.observer.stop()
        self.observer.join()
