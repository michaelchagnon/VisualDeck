# SPDX-License-Identifier: MPL-2.0
"""Start From (rehearsal): a cue's videos and sound from a time into the cue."""
import time

import numpy as np
import pytest

import render_engine as rem
from test_reverse import N, Collect, clips, index  # noqa: F401


def play(path, direction, frames, start_at, start_back=False, loop=True, trim=(None, None)):
    q = Collect()
    p = rem.GPUVideoPlayerThread(path, q, None, loop=loop, output_format="yuv", direction=direction,
                                 trim=trim, start_at=start_at, start_back=start_back)
    p.set_speed(8.0)
    p.start()
    deadline = time.monotonic() + 20
    while len(q.items) < frames and p.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    p.stop()
    p.join(5)
    return p, [index(i) for i in q.items[:frames]]


def test_forward_starts_at_the_frame_and_loops_from_the_start(clips):
    p, got = play(clips["short_gop"], "forward", N + 5, start_at=25 / 30)       # frame 25 (between key frames)
    assert got == list(range(25, N)) + list(range(0, 30)), got
    assert p.loops >= 1


def test_forward_with_an_in_point_loops_to_the_in_point(clips):
    p, got = play(clips["one_gop"], "forward", 30, start_at=30 / 30, trim=(10 / 30, None))
    assert got == list(range(30, N)) + list(range(10, 30)), got


def test_a_start_outside_the_in_and_out_points_is_kept_inside(clips):
    _, got = play(clips["short_gop"], "forward", 3, start_at=2 / 30, trim=(10 / 30, 20 / 30))
    assert got == [10, 11, 12]
    _, got = play(clips["short_gop"], "forward", 3, start_at=35 / 30, trim=(10 / 30, 20 / 30))
    assert got == [19, 10, 11]                    # the last frame before the out point, then the loop


def test_reverse_starts_at_the_frame_going_back(clips):
    p, got = play(clips["short_gop"], "reverse", 30, start_at=12 / 30)
    assert got == list(range(12, -1, -1)) + list(range(N - 1, N - 18, -1)), got


def test_ping_pong_starts_on_the_way_out_or_on_the_way_back(clips):
    _, got = play(clips["short_gop"], "pingpong", 30, start_at=30 / 30)
    assert got == list(range(30, N)) + list(range(N - 2, N - 22, -1)), got
    p, got = play(clips["short_gop"], "pingpong", 30, start_at=12 / 30, start_back=True)
    assert got == list(range(12, -1, -1)) + list(range(1, 18)), got
    assert p.loops >= 1


def test_without_a_start_nothing_changes(clips):
    _, got = play(clips["short_gop"], "forward", 5, start_at=None)
    assert got == [0, 1, 2, 3, 4]


def tones(tmp_path, rate=48000):
    """3 s of a tone that steps up every second: 300, 600, 900 Hz."""
    import av
    path = str(tmp_path / "tones.wav")
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
    return path


def read_all(s, rate, seconds):
    s.start()
    deadline = time.monotonic() + 5
    while s.state != "playing" and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.3)
    chunks = []
    for _ in range(int(seconds * 10)):
        chunks.append(s.read(rate // 10))
        time.sleep(0.1)
    s.stop()
    return np.concatenate(chunks)[:, 0]


def peak_hz(x, rate):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.argmax(spec) * rate / len(x)


def test_sound_starts_at_the_start_point(tmp_path):
    import audio_engine as ae
    rate = 48000
    s = ae.AudioStream(tones(tmp_path, rate), rate, loop=False, start_at=1.5)
    out = read_all(s, rate, 2.0)
    nz = np.nonzero(np.abs(out) > 1e-4)[0]
    played = out[nz[0]:nz[-1] + 1]
    assert abs(len(played) - 1.5 * rate) < rate * 0.01, len(played) / rate      # 1.5 s of the 3 s
    assert abs(peak_hz(played[: rate // 2], rate) - 600) < 10                   # starts in the 600 Hz second
    assert abs(peak_hz(played[-rate // 2:], rate) - 900) < 10
    s = ae.AudioStream(tones(tmp_path, rate), rate, loop=False, trim=(1.0, 2.0), start_at=0.2)
    out = read_all(s, rate, 1.5)
    nz = np.nonzero(np.abs(out) > 1e-4)[0]
    assert abs((nz[-1] + 1 - nz[0]) - rate) < rate * 0.01, "a start before the in point starts at the in point"


def test_sound_loops_from_the_in_point_after_starting_later(tmp_path):
    import audio_engine as ae
    rate = 48000
    s = ae.AudioStream(tones(tmp_path, rate), rate, loop=True, start_at=2.5)
    s.start()
    deadline = time.monotonic() + 5
    # 0.4 s decoded before reading it at once (faster than the device would): read
    # earlier, the missing part counts as late and is skipped (failed on the owner's PC)
    while s._buffered < 0.4 * rate and time.monotonic() < deadline:
        time.sleep(0.01)
    assert s.state == "playing"
    first = np.concatenate([s.read(4800) for _ in range(4)])[:, 0]     # 0.4 s of the last half second
    s.restart()                                                         # the video looped
    time.sleep(0.4)
    after = np.concatenate([s.read(4800) for _ in range(4)])[:, 0]
    s.stop()
    assert abs(peak_hz(first, rate) - 900) < 10
    assert abs(peak_hz(after, rate) - 300) < 10, "the next pass starts at the clip's start"


def test_the_mixer_makes_the_stream_with_its_start():
    import audio_engine as ae
    eng = ae.AudioEngine(backend="offline")
    eng.set_streams({"k": {"path": "/nonexistent.mp4", "gain": 1.0, "playing": True, "start_at": 12.5}})
    st = eng._streams["k"]["stream"]
    assert st.start_at == 12.5
    eng.set_streams({"k": {"path": "/nonexistent.mp4", "gain": 1.0, "playing": True}})   # later states: kept
    assert eng._streams["k"]["stream"] is st
    eng.set_streams({})


# --- the app: where each video is, the cut, the clock and the follow ----------------------------
from test_follow import ci, last_follow  # noqa: E402,F401
from test_blind_editing import ci as blind_ci  # noqa: E402,F401
from test_cue_activation import ci as _base_ci, commands  # noqa: E402,F401


class MM:
    def __init__(self, dur):
        self.dur = dur

    def get_media_info(self, path):
        return (None, self.dur, 1920, 1080)


@pytest.mark.parametrize("pos, t, want", [
    ({}, 3.0, {"start_at": 3.0}),
    ({}, 12.5, {"start_at": 2.5}),                                     # looped once (10 s clip)
    ({"loop": False}, 12.5, {"start_at": 10.0 - 1e-3}),                # ended: at its end
    ({"speed": 2.0}, 3.0, {"start_at": 6.0}),
    ({"in_s": 2.0, "out_s": 6.0}, 5.0, {"start_at": 3.0}),             # 4 s passes: 1 s into the second
    ({"direction": "reverse"}, 3.0, {"start_at": 7.0}),
    ({"direction": "pingpong"}, 3.0, {"start_at": 3.0, "start_back": False}),
    ({"direction": "pingpong"}, 13.0, {"start_at": 7.0, "start_back": True}),
    ({"direction": "pingpong", "in_s": 2.0, "speed": 0.5}, 20.0, {"start_at": 8.0, "start_back": True}),
    ({}, 0.0, None),
])
def test_where_a_video_is_after_t_seconds(pos, t, want):
    import control_interface as ci_mod
    c = ci_mod.ControlInterface.__new__(ci_mod.ControlInterface)
    c.mm = MM(10.0)
    got = c._start_point("/x/clip.mp4", pos, t)
    if want is None:
        assert got is None
    else:
        assert got.keys() == want.keys() and all(abs(got[k] - want[k]) < 1e-6 for k in want), got
    assert c._start_point("/x/still.png", {}, 3.0) is None
    c.mm = MM(0.0)
    assert c._start_point("/x/clip.mp4", {}, 3.0) is None, "length unknown: from the start"


def test_start_from_cuts_with_the_start_on_its_videos_only(ci):
    import time as _time
    ci.cue_settings = {"1": {"follow": {"mode": "after", "seconds": 10.0}}}
    ci._update_active_players()                                         # cue 1 preloaded
    commands(ci)
    ci._cut_to_column_at(1, 4.0)
    (cmd,) = commands(ci, "render")
    vids = [l for l in cmd["layers"].values() if "instance" in l]
    stills = [l for l in cmd["layers"].values() if "instance" not in l]
    assert vids and all(abs(l["start_at"] - 4.0 % ci._clip_seconds(l["path"], l["position"])) < 1e-6
                        for l in vids)
    assert stills and all("start_at" not in l for l in stills)
    assert cmd["preroll"] is True and ci.active_col == 1
    assert all("start_at" not in l for l in ci._program["layers"].values()), "later re-sends do not restart"
    assert 1 not in ci._reserved_activation, "the preload (at the start) is not used"
    clock = ci._cue_clock()
    assert clock[0] == 1 and abs(clock[1] - 4.0) < 0.05
    ms, _ = last_follow(ci)
    assert abs(ms - 6000) <= 40 and abs(ci._follow_due[0] - _time.monotonic() - 6.0) < 0.05
    ci._cut_to_column_at(0, 0)                                          # 0: a normal cut
    (cmd,) = commands(ci, "render")
    assert all("start_at" not in l for l in cmd["layers"].values())
    ci.transition_active = True
    assert ci._cut_to_column_at(1, 3.0) is False and commands(ci, "render") == []
    ci.transition_active = False


def test_start_from_window(ci, monkeypatch):
    went = []
    monkeypatch.setattr(type(ci), "_cut_to_column_at", lambda self, col, t: went.append((col, t)) or True)
    import tkinter as tk
    try:
        ci.root = tk.Tk()
    except tk.TclError:
        pytest.skip("needs a display")
    try:
        ci.root.withdraw()
        win = ci._open_start_from()
        v = win._vd
        assert "Its longest video takes" in v["info"].cget("text")
        v["cue_var"].set(v["names"][1])
        v["time_var"].set("x")
        assert v["start"]() is False and "minutes:seconds" in v["status"].cget("text") and went == []
        v["time_var"].set("1:02.5")
        assert v["start"]() is True and went == [(ci._soak_cues()[1], 62.5)]
    finally:
        ci.root.destroy()


def test_a_cold_cut_holds_its_decoders_until_the_cut(fixtures):
    """A cue that was not preloaded (and Start From) used to decode on while the
    render thread waited for its first frames, so the picture ran ahead of the
    sound, which starts at the cut. Now its decoders wait like a preload's."""
    import queue
    import threading
    import control_interface as ci_mod
    from test_player_instances import vlayer, wait_for
    eng = rem.BaseRenderEngine(media_manager=None, mode="opengl")
    rt = ci_mod.RenderThread(eng, queue.Queue(), queue.Queue(maxsize=1), threading.Event())
    try:
        playing = vlayer(fixtures["counting"], 1)
        eng.update_layers({0: playing})                       # the live cue, playing
        rt.current_render_state["layers"] = {0: playing}
        cold = dict(vlayer(fixtures["counting"], 2, row=1), start_at=0.5)
        rt._begin_preroll({"action": "render", "preroll": True, "layers": {1: cold, 0: playing},
                           "transition_state": {"active": False}})
        p = eng.active_players[eng.source_key(cold)]["player"]
        assert wait_for(lambda: p.frames_decoded >= 1)
        time.sleep(0.3)
        assert p.held and p.frames_decoded == 1, "waits on its first frame"
        live = eng.active_players[eng.source_key(playing)]["player"]
        assert not live.held, "a player already on the output is not held again"
        assert rt._preroll_ready()
        rt._commit_preroll()
        assert not p.held and wait_for(lambda: p.frames_decoded > 3)
        # a newer state that supersedes a prerolling cut lets its players run too
        cold2 = vlayer(fixtures["counting"], 3, row=2)
        rt._begin_preroll({"action": "render", "preroll": True, "layers": {2: cold2},
                           "transition_state": {"active": False}})
        p2 = eng.active_players[eng.source_key(cold2)]["player"]
        assert wait_for(lambda: p2.frames_decoded >= 1) and p2.held
        rt._apply_command({"action": "render", "layers": {2: cold2}, "transition_state": {"active": False}})
        assert not p2.held and wait_for(lambda: p2.frames_decoded > 3)
    finally:
        eng.stop_players(list(eng.active_players))


def test_a_held_start_from_player_gets_its_sound_from_its_start():
    import control_interface as ci_mod

    class Audio:
        def set_streams(self, w):
            self.w = w

    class Held:
        held, loop, loops, trim, start_at, frames_decoded = True, True, 0, (None, None), 12.5, 1

    class Eng:
        @staticmethod
        def source_key(layer):
            return layer["path"]
    rt = ci_mod.RenderThread.__new__(ci_mod.RenderThread)
    rt.audio, rt.re = Audio(), Eng()
    rt.re.active_players = {"@5:0:1|/x/a.mp4": {"player": Held(), "path": "/x/a.mp4", "type": "gpu_queue"}}
    rt._update_audio({"layers": {}, "transition_state": {"active": False}})
    assert rt.audio.w["@5:0:1|/x/a.mp4"]["start_at"] == 12.5 and rt.audio.w["@5:0:1|/x/a.mp4"]["playing"] is False


# --- from the remotes --------------------------------------------------------------------------
def test_osc_start_from_is_parsed():
    import remote_control as rc
    assert rc.parse("/visualdeck/start_from", [3, 12.5]) == ("start_from", (3, 12.5))
    assert rc.parse("/visualdeck/start_from", ["Finale", "1:02.5"]) == ("start_from", ("Finale", 62.5))
    assert rc.parse("/visualdeck/start_from", [12.5, 0]) == ("start_from", ("12.5", 0.0))   # a cue number
    for bad in ([3], [3, -1], [3, "x"], [0, 5], [True, 5], [3, float("nan")], [3, 5, 6]):
        assert rc.parse("/visualdeck/start_from", bad) is None, bad
    assert rc.parse("/visualdeck/cue", [2]) == ("cue", 2)                       # unchanged


def test_http_start_from_is_queued():
    import queue
    import web_remote as wr
    from test_web_remote import request
    q = queue.Queue()
    srv = wr.WebRemote(lambda c, a, addr: q.put((c, a)), lambda: {}, "127.0.0.1", 0)
    assert srv.start()
    try:
        H = {"X-VisualDeck": "1"}
        assert request(srv, "POST", "/api/cue/2/from/12.5", H)[0] == 202
        assert request(srv, "POST", "/api/cue/2/from/1%3A02", H)[0] == 202
        assert q.get_nowait() == ("start_from_at", (2, 12.5)) and q.get_nowait() == ("start_from_at", (2, 62.0))
        for bad in ("/api/cue/0/from/3", "/api/cue/2/from/x", "/api/cue/2/from/-1", "/api/cue/x/from/3"):
            assert request(srv, "POST", bad, H)[0] == 404, bad
        assert request(srv, "POST", "/api/cue/2/from/3")[0] == 403              # the header is still required
    finally:
        srv.stop()


def test_remote_start_from_uses_the_same_path_as_the_menu(ci, monkeypatch):
    went = []
    monkeypatch.setattr(type(ci), "_cut_to_column_at", lambda self, col, t: went.append((col, t)) or True)
    ci._ui_ready = True
    ci.cue_names["1"] = "Finale"
    ci._handle_remote_command("start_from", ("finale", 4.0))
    ci._handle_remote_command("start_from_at", (1, 2.5))
    ci._handle_remote_command("start_from", ("nobody", 1.0))
    ci._handle_remote_command("start_from_at", (99, 1.0))
    assert went == [(1, 4.0), (0, 2.5)]


# --- sound kept with the picture: catch up to the picture's clock --------------------------------
def test_sound_catches_up_to_the_pictures_start_and_loops(tmp_path):
    import audio_engine as ae
    rate = 48000
    s = ae.AudioStream(tones(tmp_path, rate), rate, loop=True)
    s.start()
    deadline = time.monotonic() + 5
    while s._buffered < rate // 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    s.skip(0.6)                                       # the picture started 0.6 s ago
    chunks = []
    for _ in range(80):                               # read in real time, as the sound card does
        chunks.append(s.read(480)[:, 0])
        time.sleep(0.01)
    out = np.concatenate(chunks)
    change = next(k for k in range(0, len(out) - 960, 240) if abs(peak_hz(out[k:k + 960], rate) - 600) < 60)
    assert abs(change / rate - 0.4) < 0.04, change / rate   # 300 -> 600 Hz at 1.0 s of the clip, 0.4 s in
    s.skip(5.0)                                       # more than MAX_CATCH_UP_S: something else is wrong
    assert s._owed < rate
    s.restart(since=time.perf_counter() - 0.6)        # the picture looped 0.6 s ago
    chunks = []
    for _ in range(80):
        chunks.append(s.read(480)[:, 0])
        time.sleep(0.01)
    s.stop()
    out = np.concatenate(chunks)
    hz = [(k, peak_hz(out[k:k + 960], rate)) for k in range(0, len(out) - 960, 240)]
    k300 = next(k for k, f in hz if abs(f - 300) < 60)               # the new pass (the old one was at 600)
    change = next(k for k, f in hz if k > k300 and abs(f - 600) < 60)
    assert k300 / rate < 0.05 and abs(change / rate - 0.4) < 0.06, (k300 / rate, change / rate)
    assert s._owed == 0


def test_the_mixer_skips_what_it_missed_once_when_a_stream_starts():
    import audio_engine as ae

    class Stream:
        skipped = []

        def skip(self, sec):
            self.skipped.append(round(sec, 2))

        def read(self, n):
            return np.zeros((n, 2), np.float32)
    eng = ae.AudioEngine(backend="offline")
    st = Stream()
    eng._streams = {"k": {"stream": st, "gain": 0.0, "target": 1.0, "playing": True, "path": "x", "loops": 0,
                          "clock_start": time.perf_counter() - 0.25}}
    eng.mix(480)
    eng.mix(480)
    assert len(st.skipped) == 1 and abs(st.skipped[0] - 0.25) < 0.05


def test_players_record_when_their_picture_starts_and_loops(fixtures):
    for cls, kw in ((rem.GPUVideoPlayerThread, {"output_format": "yuv"}), (rem.CPUVideoPlayerThread, {})):
        q = Collect()
        p = cls(fixtures["red_709"], q, loop=True, hold=True, **kw)        # 0.5 s clip
        p.set_speed(4.0)
        p.start()
        try:
            deadline = time.monotonic() + 5
            while p.frames_decoded < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            time.sleep(0.1)
            assert p.clock_start is None, "held: not started"
            t = time.perf_counter()
            p.release()
            while p.loops < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert p.clock_start is not None and abs(p.clock_start - t) < 0.06, cls.__name__
            assert p.loops >= 1 and p.loop_at is not None and p.loop_at > p.clock_start, cls.__name__
        finally:
            p.stop()
            p.join(5)
