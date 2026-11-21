# SPDX-License-Identifier: MPL-2.0
import threading

import numpy as np
import pytest

from ring_buffer import FastRingBuffer, SimpleFrameQueue


def test_fifo_order_and_capacity():
    rb = FastRingBuffer(3)
    for i in range(3):
        rb.put_nowait(i)
    assert rb.full()
    with pytest.raises(Exception):
        rb.put_nowait(99)
    assert [rb.get_nowait() for _ in range(3)] == [0, 1, 2]
    assert rb.empty()


def test_drop_old_keeps_newest():
    rb = FastRingBuffer(3)
    for i in range(10):
        rb.put_nowait_drop_old(i)
    assert rb.qsize() == 3
    assert [rb.get_nowait() for _ in range(3)] == [7, 8, 9]


def test_get_latest_drains_to_newest():
    rb = FastRingBuffer(5)
    for i in range(4):
        rb.put_nowait(i)
    assert rb.get_latest() == 3
    assert rb.empty()
    assert rb.get_latest() is None


def test_get_timeout_raises():
    rb = FastRingBuffer(2)
    with pytest.raises(TimeoutError):
        rb.get(timeout=0.01)


def test_preallocated_buffer_returns_copies():
    rb = FastRingBuffer(2, frame_shape=(2, 2))
    src = np.ones((2, 2), np.uint8)
    rb.put_nowait(src)
    src[:] = 7  # mutating the source after put must not affect the queued frame
    assert rb.get_nowait().sum() == 4


def test_producer_consumer_threads_no_loss_when_not_full():
    rb = FastRingBuffer(64)
    got = []

    def consume():
        for _ in range(500):
            got.append(rb.get(timeout=2))

    t = threading.Thread(target=consume)
    t.start()
    for i in range(500):
        rb.put(i, timeout=2)
    t.join(5)
    assert got == list(range(500))


def test_simple_frame_queue_drops_oldest():
    q = SimpleFrameQueue(2)
    for i in range(5):
        q.put_nowait(i)
    assert q.qsize() == 2
    assert q.get_latest() == 4
