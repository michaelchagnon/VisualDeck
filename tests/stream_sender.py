# SPDX-License-Identifier: MPL-2.0
"""A local MPEG-TS-over-UDP sender for live-source tests (no camera or network
needed): solid-color 320x240 frames at 25 fps, color changeable while sending."""
from __future__ import annotations

import socket
import threading
import time

import av
import numpy as np


def free_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class UdpSender:
    def __init__(self, port, color=(255, 0, 0), fps=25, size=(320, 240), tone_hz=None):
        self.port, self.color, self.fps, self.size = port, tuple(color), fps, size
        self.tone_hz = tone_hz                   # also send a stereo tone (MP2, 48 kHz) when set
        self._stop = threading.Event()
        self._thread = None
        self.frames = 0

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="UdpSender")
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _run(self):
        out = av.open(f"udp://127.0.0.1:{self.port}?pkt_size=1316", mode="w", format="mpegts")
        st = out.add_stream("mpeg2video", rate=self.fps)
        st.width, st.height = self.size
        st.pix_fmt = "yuv420p"
        st.codec_context.gop_size = 5
        st.codec_context.bit_rate = 2_000_000
        ast = None
        if self.tone_hz:
            from fractions import Fraction
            ast = out.add_stream("mp2", rate=48000, layout="stereo")
            ast.codec_context.bit_rate = 192_000
            a_tb, a_buf, a_n = Fraction(1, 48000), np.zeros(0, np.float32), 0
        t0 = time.monotonic()
        i = 0
        try:
            while not self._stop.is_set():
                img = np.empty((self.size[1], self.size[0], 3), np.uint8)
                img[...] = self.color
                fr = av.VideoFrame.from_ndarray(img, format="rgb24")
                fr.pts = i
                for p in st.encode(fr):
                    out.mux(p)
                if ast is not None:                  # 1/fps s of tone, in the encoder's frame size
                    n = 48000 // self.fps
                    t = (a_n + len(a_buf) + np.arange(n)) / 48000
                    a_buf = np.concatenate([a_buf, (0.3 * np.sin(2 * np.pi * self.tone_hz * t)).astype(np.float32)])
                    fs = ast.codec_context.frame_size or 1152
                    while len(a_buf) >= fs:
                        blk, a_buf = a_buf[:fs], a_buf[fs:]
                        pcm = (np.stack([blk, blk], axis=1).reshape(1, -1) * 32767).astype(np.int16)
                        af = av.AudioFrame.from_ndarray(pcm, format="s16", layout="stereo")
                        af.rate, af.pts, af.time_base = 48000, a_n, a_tb
                        a_n += fs
                        for p in ast.encode(af):
                            out.mux(p)
                i += 1
                self.frames = i
                time.sleep(max(0.0, t0 + i / self.fps - time.monotonic()))
        finally:
            try:
                out.close()
            except Exception:  # noqa: BLE001
                pass
