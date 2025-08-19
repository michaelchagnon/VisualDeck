# =====================================================================
# src/utils/ffmpeg_utils.py
# =====================================================================

import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import subprocess
import threading
import json
import hashlib
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FFMPEG_DIR = os.path.join(BASE_DIR, '..', 'ffmpeg', 'bin')
CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.visualdeck')

if sys.platform == "win32":
    FFPROBE_PATH = os.path.join(FFMPEG_DIR, 'ffprobe.exe')
    FFMPEG_PATH = os.path.join(FFMPEG_DIR, 'ffmpeg.exe')
else:
    FFPROBE_PATH = os.path.join(FFMPEG_DIR, 'ffprobe')
    FFMPEG_PATH = os.path.join(FFMPEG_DIR, 'ffmpeg')


class FFMpegProxyGenerator:
    """Generates a low-resolution proxy video in a background thread."""
    PROXY_SUFFIX = ".mov"
    PROXY_SUBDIR = "proxies"

    def __init__(self, file_path):
        self.file_path = file_path
        self.proxy_dir = os.path.join(CONFIG_DIR, self.PROXY_SUBDIR)
        self.output_path = self.get_proxy_path(self.file_path)

    @classmethod
    def get_proxy_path(cls, source_path):
        """Static method to consistently determine the path for a proxy file."""
        proxy_dir = os.path.join(CONFIG_DIR, cls.PROXY_SUBDIR)
        # Create a stable hash from the file path to use as a unique filename
        path_hash = hashlib.sha1(source_path.encode('utf-8')).hexdigest()
        return os.path.join(proxy_dir, f"{path_hash}{cls.PROXY_SUFFIX}")

    def _run_ffmpeg(self):
        """Executes the FFmpeg command."""
        if not os.path.exists(FFMPEG_PATH):
            logging.error(f"FFmpeg not found at {FFMPEG_PATH}. Cannot generate proxy.")
            return

        command = [
            FFMPEG_PATH, '-y', '-i', self.file_path,
            '-vf', 'scale=320:-2',
            '-vcodec', 'mjpeg',
            '-q:v', '5',
            '-an',
            self.output_path
        ]
        
        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            subprocess.run(command, check=True, capture_output=True, creationflags=creation_flags)
            logging.info(f"Successfully generated proxy for {os.path.basename(self.file_path)}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.error(f"Error generating proxy for {os.path.basename(self.file_path)}: {e}")
            if os.path.exists(self.output_path):
                os.remove(self.output_path)

    def generate(self):
        """Generates the proxy if it doesn't already exist."""
        if os.path.exists(self.output_path):
            return

        os.makedirs(self.proxy_dir, exist_ok=True)
        
        thread = threading.Thread(target=self._run_ffmpeg, daemon=True)
        thread.start()


class FFMpegOptimizer:
    def __init__(self, parent_root, file_path):
        self.root = parent_root
        self.file_path = file_path
        self.window = None
        self.proc = None

    def _get_video_duration(self):
        if not os.path.exists(FFPROBE_PATH): return None
        try:
            cmd = [FFPROBE_PATH, '-v', 'error', '-show_entries', 'format=duration',
                   '-of', 'default=noprint_wrappers=1:nokey=1', self.file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            logging.warning(f"Could not get video duration: {e}")
            return None

    def show(self):
        self.window = tk.Toplevel(self.root)
        self.window.title("Optimize Media")
        self.window.configure(bg='#3e3e3e')
        self.window.transient(self.root)
        self.window.grab_set()

        main_frame = tk.Frame(self.window, bg='#3e3e3e', padx=15, pady=15)
        main_frame.pack(fill='both', expand=True)

        tk.Label(main_frame, text=f"Optimizing: {os.path.basename(self.file_path)}", fg='white', bg='#3e3e3e').pack(anchor='w')

        codec_lf = ttk.LabelFrame(main_frame, text="Target Codec", padding=(10, 5))
        codec_lf.pack(fill='x', expand=True, pady=10)

        self.codec_var = tk.StringVar(value='hap')
        hap_rb = ttk.Radiobutton(codec_lf, text="HAP", variable=self.codec_var, value='hap')
        hap_rb.pack(anchor='w', padx=5, pady=2)

        duration = self._get_video_duration()
        est_time_str = "Unknown (ffprobe not found)"
        if duration:
            est_seconds = int(duration * 0.2)
            est_time_str = f"~{est_seconds // 60} min {est_seconds % 60} sec"
        
        tk.Label(main_frame, text=f"Estimated time: {est_time_str}", fg='white', bg='#3e3e3e').pack(anchor='w', pady=(5,10))

        self.progress = ttk.Progressbar(main_frame, orient='horizontal', length=300, mode='determinate')
        self.progress.pack(fill='x', expand=True, pady=5)
        
        self.status_label = tk.Label(main_frame, text="Ready to start.", fg='white', bg='#3e3e3e')
        self.status_label.pack(anchor='w')

        btn_frame = tk.Frame(main_frame, bg='#3e3e3e')
        btn_frame.pack(fill='x', pady=(15, 0))

        self.start_btn = tk.Button(btn_frame, text="Start", command=self._start_optimization, bg='#5a5', fg='black')
        self.start_btn.pack(side='right', padx=5)
        self.cancel_btn = tk.Button(btn_frame, text="Close", command=self._cancel, bg='#a55', fg='white')
        self.cancel_btn.pack(side='right')

        if not os.path.exists(FFMPEG_PATH):
            self.start_btn.config(state='disabled')
            self.status_label.config(text="Error: ffmpeg.exe not found in ffmpeg/bin/")

        self.window.protocol("WM_DELETE_WINDOW", self._cancel)

    def _start_optimization(self):
        self.start_btn.config(state='disabled')
        self.status_label.config(text="Starting FFmpeg...")

        # Save the optimized file in the same directory as the source
        output_dir = os.path.dirname(self.file_path)
        basename, _ = os.path.splitext(os.path.basename(self.file_path))
        output_path = os.path.join(output_dir, f"{basename}_HAP.mov")

        if os.path.exists(output_path):
            if not messagebox.askyesno("File Exists", "Optimized file already exists. Overwrite?"):
                self.start_btn.config(state='normal')
                return
        
        command = [
            FFMPEG_PATH, '-y', '-i', self.file_path,
            '-vcodec', 'hap', '-format', 'hap_q',
            '-progress', 'pipe:1',
            output_path
        ]

        try:
            self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            threading.Thread(target=self._monitor_progress, daemon=True).start()
        except FileNotFoundError:
            messagebox.showerror("Error", f"FFmpeg not found at {FFMPEG_PATH}. Please check the path.")
            self.start_btn.config(state='normal')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start FFmpeg: {e}")
            self.start_btn.config(state='normal')

    def _monitor_progress(self):
        duration = self._get_video_duration() or 1
        self.progress['maximum'] = duration

        while True:
            if self.proc.stdout is None: break
            line = self.proc.stdout.readline()
            if not line: break
            
            if 'out_time_ms' in line:
                parts = line.strip().split('=')
                if len(parts) == 2:
                    try:
                        time_ms = int(parts[1])
                        current_time = time_ms / 1_000_000
                        self.progress['value'] = current_time
                        self.status_label.config(text=f"Progress: {int(100 * current_time / duration)}%")
                    except ValueError:
                        continue
            elif 'progress=end' in line:
                break
        
        self.proc.wait()
        if self.proc.returncode == 0:
            self.status_label.config(text="Optimization complete!")
            self.progress['value'] = duration
        else:
            self.status_label.config(text="Error during optimization.")
        
        self.start_btn.config(state='normal')

    def _cancel(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.window.destroy()
