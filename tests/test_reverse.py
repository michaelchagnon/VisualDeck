# SPDX-License-Identifier: MPL-2.0
"""Reverse and ping-pong playback: frame order, long GOPs, hold, the engine and the app."""
import os
import time
from fractions import Fraction

import av
import pytest

import media_fixtures as mf
import render_engine as rem

N = 40


def counting(path, opts):
    """N frames whose luma is 16 + 3 * index (H.264 with the given encoder options)."""
    c = av.open(path, mode="w")
    st = c.add_stream("libx264", rate=30)
    st.width, st.height, st.pix_fmt = 160, 90, "yuv420p"
    st.options = dict({"crf": "1", "preset": "ultrafast"}, **opts)
    for i in range(N):
        f = av.VideoFrame.from_ndarray(mf.solid_yuv(160, 90, 16 + 3 * i, 128, 128), format="yuv420p")
        f.pts, f.time_base = i, Fraction(1, 30)
        for pkt in st.encode(f):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("rev")
    return {"short_gop": counting(str(d / "short.mp4"), {"g": "8", "keyint_min": "8", "bf": "2"}),
            "one_gop": counting(str(d / "one.mp4"), {"g": "250", "bf": "3"})}


class Collect:
    def __init__(self):
        self.items = []

    def put_nowait_drop_old(self, x):
        self.items.append(x)


def index(item):
    y = item["y"] if isinstance(item, dict) else item[..., 0]       # yuv dict, or a BGR array
    return int(round((float(y.mean()) - 16) / 3)) if isinstance(item, dict) else None


def play(path, direction, frames, loop=True, fmt="yuv", hold=False):
    q = Collect()
    p = rem.GPUVideoPlayerThread(path, q, None, loop=loop, output_format=fmt, direction=direction, hold=hold)
    p.set_speed(8.0)
    p.start()
    deadline = time.monotonic() + 20
    while len(q.items) < frames and p.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    p.stop()
    p.join(5)
    return p, q.items[:frames]


def test_reverse_plays_every_frame_backwards_and_loops(clips):
    p, items = play(clips["short_gop"], "reverse", 2 * N + 5)
    assert [index(i) for i in items] == list(range(N - 1, -1, -1)) * 2 + list(range(N - 1, N - 6, -1))
    assert p.loops >= 2 and p.last_error is None


def test_ping_pong_turns_at_each_end_without_doubling_a_frame(clips):
    p, items = play(clips["short_gop"], "pingpong", 3 * N)
    want = list(range(N)) + list(range(N - 2, -1, -1)) + list(range(1, N)) + list(range(N - 2, -1, -1))
    assert [index(i) for i in items] == want[:3 * N]


def test_long_gop_with_a_small_memory_budget_still_plays_every_frame(clips, monkeypatch):
    monkeypatch.setattr(rem, "REVERSE_MAX_FRAMES", 5)            # chunks of 5 out of one 40-frame GOP
    p, items = play(clips["one_gop"], "reverse", N)
    assert [index(i) for i in items] == list(range(N - 1, -1, -1))


def test_not_looping_plays_once_and_ends(clips):
    p, items = play(clips["short_gop"], "reverse", 10 * N, loop=False)
    assert [index(i) for i in items] == list(range(N - 1, -1, -1)) and p.state == "ended"


def test_a_held_reverse_player_waits_on_the_last_frame(clips):
    q = Collect()
    p = rem.GPUVideoPlayerThread(clips["short_gop"], q, None, output_format="yuv", direction="reverse", hold=True)
    p.start()
    try:
        deadline = time.monotonic() + 10
        while not q.items and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.3)
        assert [index(i) for i in q.items] == [N - 1]                # pre-rolled, waiting for GO
        p.release()
        time.sleep(0.4)
        assert [index(i) for i in q.items[:4]] == [N - 1, N - 2, N - 3, N - 4]
    finally:
        p.stop()
        p.join(5)


def test_every_output_format_plays_backwards(clips):
    for fmt in ("pygame", "numpy"):
        p, items = play(clips["short_gop"], "reverse", 3, fmt=fmt)
        assert len(items) == 3 and p.last_error is None, fmt


def test_direction_of_a_layer():
    assert rem.layer_direction({}) == "forward" and rem.layer_direction(None) == "forward"
    assert rem.layer_direction({"direction": "pingpong"}) == "pingpong"
    assert rem.layer_direction({"direction": "sideways"}) == "forward"


def test_engine_starts_a_new_player_when_the_direction_changes(clips, monkeypatch):
    started = []

    class Fake:
        def __init__(self, path, q, *a, direction="forward", **k):
            self.path, self.direction, self.loop, self.held = path, direction, True, False
            self.stopped = False
            started.append(self)

        def start(self):
            pass

        def stop(self):
            self.stopped = True

        def set_speed(self, s):
            pass

    class FakeCPU(Fake):
        pass
    monkeypatch.setattr(rem, "GPUVideoPlayerThread", Fake)
    monkeypatch.setattr(rem, "CPUVideoPlayerThread", FakeCPU)
    for mode in ("pygame_gpu", "cpu"):
        started.clear()
        eng = rem.BaseRenderEngine(None, mode=mode)
        L = {0: {"path": clips["short_gop"], "instance": "@1:0:0", "screens": [0], "order": 0}}
        eng.update_layers(L)
        first = started[-1]
        assert first.direction == "forward" and not isinstance(first, FakeCPU)   # PyAV in every mode
        eng.video_frame_cache["@1:0:0|" + clips["short_gop"]] = "last frame"
        eng.update_layers({0: dict(L[0], direction="reverse")})
        second = started[-1]
        assert first.stopped and second is not first and second.direction == "reverse"
        assert not isinstance(second, FakeCPU)                     # CPU mode goes backwards through PyAV
        assert "last frame" in eng.video_frame_cache.values()       # shown until the new player has a frame
        eng.update_layers({0: dict(L[0], direction="reverse")})
        assert started[-1] is second and not second.stopped        # same direction: the same player


def test_backwards_videos_are_silent_and_ping_pong_counts_twice(monkeypatch):
    import control_interface as ci_mod
    assert ci_mod.ControlInterface._direction_factor({"direction": "pingpong"}) == 2.0
    assert ci_mod.ControlInterface._direction_factor({"direction": "reverse"}) == 1.0
    assert ci_mod.ControlInterface._direction_factor(None) == 1.0

    class Audio:
        def set_streams(self, wanted):
            self.wanted = wanted
    rt = ci_mod.RenderThread.__new__(ci_mod.RenderThread)
    rt.audio = Audio()

    class Eng:
        active_players = {}

        @staticmethod
        def source_key(layer):
            return layer["path"]
    rt.re = Eng()
    layers = {0: {"path": "/a.mp4"}, 1: {"path": "/b.mp4", "direction": "reverse"},
              2: {"path": "/c.mp4", "direction": "pingpong"}}
    rt._update_audio({"layers": layers, "transition_state": {"active": False}})
    gains = {k: v["gain"] for k, v in rt.audio.wanted.items()}
    assert gains == {"/a.mp4": 1.0, "/b.mp4": 0.0, "/c.mp4": 0.0}


def test_editor_direction_control(monkeypatch):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import editor as ed

    class Host:
        pass
    root = tk.Tk()
    root.withdraw()
    try:
        e = ed.MediaEditor.__new__(ed.MediaEditor)
        e.current_pos = {}
        calls = []
        e._trigger_realtime_callback = lambda: calls.append(dict(e.current_pos))
        e.direction_var = tk.StringVar(master=root, value="Ping-pong")
        e._on_direction_change()
        assert e.current_pos == {"direction": "pingpong"}
        e.direction_var.set("Forward")
        e._on_direction_change()
        assert e.current_pos == {} and len(calls) == 2
    finally:
        root.destroy()


@pytest.mark.parametrize("direction", ["reverse", "pingpong"])
def test_turns_are_decoded_ahead_so_playback_does_not_stall(clips, monkeypatch, direction):
    """At a loop (reverse) or a turn (ping-pong) the next frames were decoded while
    the previous ones were shown: finish() has nothing left to decode there."""
    late = []
    orig = rem._ReverseChunk.finish

    def finish(self):
        before = self.decoded
        orig(self)
        if self.end is None:                       # the chunk at the clip's end: a loop or a turn
            late.append(self.decoded - before)
    monkeypatch.setattr(rem._ReverseChunk, "finish", finish)
    play(clips["one_gop"], direction, 3 * N)
    assert len(late) >= 2, late
    if direction == "reverse":
        assert late[0] == N                          # the very first chunk, before anything is shown
        late = late[1:]
    assert all(n == 0 for n in late), late


# --- In and out points ------------------------------------------------------------------

@pytest.mark.parametrize("direction,want", [
    ("forward", list(range(10, 20)) * 3),
    ("reverse", list(range(19, 9, -1)) * 3),
    ("pingpong", list(range(10, 20)) + list(range(18, 9, -1)) + list(range(11, 20)) + list(range(18, 16, -1))),
])
def test_in_and_out_points_in_every_direction(clips, direction, want):
    q = Collect()
    p = rem.GPUVideoPlayerThread(clips["short_gop"], q, None, output_format="yuv", direction=direction,
                                 trim=(10 / 30, 20 / 30))
    p.set_speed(8.0)
    p.start()
    deadline = time.monotonic() + 20
    while len(q.items) < len(want) and time.monotonic() < deadline:
        time.sleep(0.02)
    p.stop()
    p.join(5)
    assert [index(i) for i in q.items[:len(want)]] == want


def test_in_point_inside_a_long_gop_and_no_out_point(clips):
    q = Collect()
    p = rem.GPUVideoPlayerThread(clips["one_gop"], q, None, output_format="yuv", trim=(25 / 30, None), loop=False)
    p.start()
    p.join(10)
    assert [index(i) for i in q.items] == list(range(25, N)) and p.state == "ended"


def test_layer_trim_values():
    assert rem.layer_trim({}) == (None, None)
    assert rem.layer_trim({"in_s": 1.5, "out_s": 4}) == (1.5, 4.0)
    assert rem.layer_trim({"in_s": 0, "out_s": None}) == (None, None)
    assert rem.layer_trim({"in_s": 2.0, "out_s": 2.01}) == (None, None)      # shorter than a frame
    assert rem.layer_trim({"in_s": "x"}) == (None, None)
    assert rem.layer_trim({"out_s": float("nan")}) == (None, None)


def test_trimmed_sound_starts_at_the_in_point_and_stops_at_the_out_point(tmp_path):
    import numpy as np
    import audio_engine as ae
    # 3 s of a tone whose frequency steps up every second: 300, 600, 900 Hz
    path = str(tmp_path / "tones.wav")
    rate = 48000
    t = np.arange(3 * rate) / rate
    sig = (0.5 * np.sin(2 * np.pi * (300 * (1 + (t // 1))) * t)).astype(np.float32)
    c = av.open(path, mode="w")
    st = c.add_stream("pcm_f32le", rate=rate, layout="mono")
    frame = av.AudioFrame.from_ndarray(sig.reshape(1, -1), format="flt", layout="mono")
    frame.rate = rate
    for pkt in st.encode(frame):
        c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()
    s = ae.AudioStream(path, rate, loop=False, trim=(1.0, 2.0))
    s.start()
    deadline = time.monotonic() + 5
    while s.state != "playing" and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.3)
    chunks = []
    for _ in range(15):                                                    # 1.5 s, read in real time
        chunks.append(s.read(4800))
        time.sleep(0.1)
    out = np.concatenate(chunks)[:, 0]
    s.stop()
    nz = np.nonzero(np.abs(out) > 1e-4)[0]
    played = out[nz[0]:nz[-1] + 1]
    assert abs(len(played) - rate) < rate * 0.005, len(played)              # one second of sound
    spec = np.abs(np.fft.rfft(played[: rate // 2]))
    assert abs(np.argmax(spec) * 2 - 600) < 10                             # the second second: 600 Hz


def test_clip_seconds_allows_for_trim_speed_and_direction():
    import control_interface as ci_mod

    class MM:
        @staticmethod
        def get_media_info(path):
            return (None, 10.0, 1920, 1080)
    ci = ci_mod.ControlInterface.__new__(ci_mod.ControlInterface)
    ci.mm = MM()
    assert ci._clip_seconds("/a.mp4", {}) == 10.0
    assert ci._clip_seconds("/a.mp4", {"in_s": 2, "out_s": 6}) == 4.0
    assert ci._clip_seconds("/a.mp4", {"in_s": 2, "speed": 2.0}) == 4.0
    assert ci._clip_seconds("/a.mp4", {"out_s": 30, "direction": "pingpong"}) == 20.0


def test_editor_in_and_out_fields():
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import editor as ed
    root = tk.Tk()
    root.withdraw()
    try:
        e = ed.MediaEditor.__new__(ed.MediaEditor)
        e.current_pos, e.clip_seconds = {}, 10.0
        calls = []
        e._trigger_realtime_callback = lambda: calls.append(dict(e.current_pos))
        e.in_var, e.out_var = tk.StringVar(master=root), tk.StringVar(master=root)
        e.trim_note = tk.Label(root)
        e.in_var.set("1:02.5")
        assert e._on_trim_change() is False and "before the clip's end" in e.trim_note.cget("text")
        e.in_var.set("2.5")
        e.out_var.set("7")
        assert e._on_trim_change() and e.current_pos == {"in_s": 2.5, "out_s": 7.0} and len(calls) == 1
        assert "plays 4.50 s" in e.trim_note.cget("text")
        e.out_var.set("abc")
        assert e._on_trim_change() is False and e.current_pos == {"in_s": 2.5, "out_s": 7.0}
        e.in_var.set("")
        e.out_var.set("12")                                    # past the end: the end
        assert e._on_trim_change() and e.current_pos == {} and e.out_var.get() == ""
    finally:
        root.destroy()
