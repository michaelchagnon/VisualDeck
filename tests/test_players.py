# SPDX-License-Identifier: MPL-2.0
"""Decoder thread behavior (no display required)."""
import time

import numpy as np

import render_engine as rem
from ring_buffer import FastRingBuffer, SimpleFrameQueue


def _drain(q):
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except Exception:
            return items


def test_cpu_player_queues_independent_frames(fixtures):
    """Each queued frame must be its own array: the consumer must never see a
    buffer the decoder thread is still overwriting."""
    q = SimpleFrameQueue(maxsize=10)
    pl = rem.CPUVideoPlayerThread(fixtures["counting"], q)
    pl.start()
    time.sleep(0.4)  # ~12 frames at 30 fps
    pl.stop()
    pl.join(2)
    frames = _drain(q)
    assert len(frames) >= 3
    assert len({id(f) for f in frames}) == len(frames), "same ndarray object queued repeatedly"
    lumas = [int(f[..., 0].mean()) for f in frames]
    assert len(set(lumas)) > 1, f"all queued frames show identical content: {lumas}"


def test_gpu_player_yuv_output_planes(fixtures):
    q = FastRingBuffer(30)
    pl = rem.GPUVideoPlayerThread(fixtures["split_854"], q, output_format="yuv")
    pl.start()
    deadline = time.time() + 5
    item = None
    while time.time() < deadline and item is None:
        item = q.get_latest()
        time.sleep(0.02)
    pl.stop()
    pl.join(2)
    assert isinstance(item, dict) and item["width"] == 854 and item["height"] == 480
    y = np.frombuffer(item["y_data"], np.uint8) if "y_data" in item else np.asarray(item["y"])
    assert y.size == 854 * 480
    y = y.reshape(480, 854)
    assert y[:, :400].mean() > 200 and y[:, 460:].mean() < 40


def test_gpu_player_survives_missing_file(tmp_path):
    q = FastRingBuffer(4)
    pl = rem.GPUVideoPlayerThread(str(tmp_path / "missing.mp4"), q, output_format="yuv")
    pl.start()
    time.sleep(0.3)
    alive = pl.is_alive()
    pl.stop()
    pl.join(3)
    assert alive, "player thread should keep retrying (with backoff), not crash"
    assert not pl.is_alive(), "player thread must stop promptly when asked"


def test_gpu_player_loops_by_seeking(fixtures):
    """An 8-frame clip must keep producing frames well past its end."""
    q = FastRingBuffer(4)
    pl = rem.GPUVideoPlayerThread(fixtures["short_loop"], q, output_format="yuv")
    pl.start()
    time.sleep(0.2)
    q.get_latest()
    time.sleep(0.6)  # > 2 loops of 0.27 s
    late = q.get_latest()
    alive = pl.is_alive()
    pl.stop()
    pl.join(2)
    assert alive and late is not None


def test_requested_hwaccel_falls_back_to_software(fixtures, caplog):
    """Requesting a hardware decoder that is not present must still play (software)."""
    import logging
    caplog.set_level(logging.INFO)
    q = FastRingBuffer(4)
    pl = rem.GPUVideoPlayerThread(fixtures["black_709"], q, hw_accel="cuda", output_format="yuv")
    pl.start()
    deadline = time.time() + 5
    item = None
    while time.time() < deadline and item is None:
        item = q.get_latest()
        time.sleep(0.02)
    pl.stop()
    pl.join(2)
    assert item is not None
    assert any("hwaccel" in r.getMessage() for r in caplog.records)
