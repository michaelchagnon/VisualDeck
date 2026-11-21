# SPDX-License-Identifier: MPL-2.0
"""Per-cue follow: 'continue after N s' and 'continue when its videos end'."""
import pytest

from test_blind_editing import ci as blind_ci, renders  # noqa: F401
from test_cue_activation import ci as _base_ci, commands  # noqa: F401


class TimerRoot:
    def __init__(self):
        self.timers = []

    def after(self, ms, fn=None, *a):
        self.timers.append((ms, fn))
        return len(self.timers)

    def after_cancel(self, ident):
        pass


@pytest.fixture
def ci(blind_ci):
    blind_ci.root = TimerRoot()
    return blind_ci


def follow_timers(ci):
    return [(ms, fn) for ms, fn in ci.root.timers if fn and getattr(fn, "__name__", "") == "<lambda>"
            and "follow" in fn.__code__.co_names[0:1] + ("_follow_fire",) and ms != 100]


def last_follow(ci):
    # the follow timer is the last after() whose callback calls _follow_fire
    for ms, fn in reversed(ci.root.timers):
        if fn is not None and "_follow_fire" in fn.__code__.co_names:
            return ms, fn
    return None


def test_continue_after_seconds_goes_to_the_next_cue_with_media(ci, fixtures):
    ci.grid_cells = {k: v for k, v in ci.grid_cells.items() if k[1] != 1}   # cue 2 empty
    ci.cue_settings = {"0": {"follow": {"mode": "after", "seconds": 2.5}}}
    ci._cut_to_column(0)
    ms, fire = last_follow(ci)
    assert ms == 2500 and ci._follow_due[1:] == (0, 2)
    fire()
    assert ci.transition_thread.started[-1][3] == 2


def test_continue_when_videos_end_uses_the_longest_video(ci, fixtures, tmp_path):
    from media_fixtures import solid_clip
    ci.grid_cells[(1, 0)] = solid_clip(str(tmp_path / "3s.mp4"), 64, 36, 63, 102, 240, n_frames=90)  # 3 s
    ci.grid_cells[(2, 0)] = fixtures["red_709"]          # 0.5 s; counting (row 0) is 2 s
    ci.cue_settings = {"0": {"follow": {"mode": "end"}}}
    ci._cut_to_column(0)
    ms, _ = last_follow(ci)
    assert abs(ci._follow_due[0] - __import__("time").monotonic() - 3.0) <= 0.05
    assert abs(ms - (3000 - 1000 * ci.FOLLOW_EARLY_S)) <= 40      # watched from a second before
    assert ci._follow_watch[0].endswith("3s.mp4")                  # the longest video's player


class Player:
    def __init__(self):
        self.loops, self.state = 0, "playing"


def _watched(ci, monkeypatch, t0=1000.0):
    clock = {"t": t0}
    monkeypatch.setattr("control_interface.time.monotonic", lambda: clock["t"])
    ci.cue_settings = {"0": {"follow": {"mode": "end"}}}
    p = Player()
    ci._cut_to_column(0)
    key = ci._follow_watch[0]
    ci.re.active_players = {key: {"player": p}}
    return clock, p


def test_end_follow_waits_for_the_video_itself(ci, monkeypatch):
    clock, p = _watched(ci, monkeypatch)
    due = ci._follow_due[0]
    n = len(ci.transition_thread.started)
    clock["t"] = due + 0.3                         # past the computed end, the video still on its first pass
    last_follow(ci)[1]()
    assert len(ci.transition_thread.started) == n and last_follow(ci)[0] == ci.FOLLOW_POLL_MS
    p.loops = 1                                    # it wraps: continue now
    last_follow(ci)[1]()
    assert len(ci.transition_thread.started) == n + 1 and ci.transition_thread.started[-1][3] == 1


def test_end_follow_fires_when_the_video_ends_early(ci, monkeypatch):
    clock, p = _watched(ci, monkeypatch)
    due = ci._follow_due[0]
    n = len(ci.transition_thread.started)
    clock["t"] = due - 0.9
    last_follow(ci)[1]()
    assert len(ci.transition_thread.started) == n          # still playing: poll
    p.state = "ended"                                      # a non-looping clip finished 0.8 s early
    clock["t"] = due - 0.8
    last_follow(ci)[1]()
    assert len(ci.transition_thread.started) == n + 1


def test_end_follow_gives_up_waiting_after_the_grace_time(ci, monkeypatch):
    clock, p = _watched(ci, monkeypatch)
    n = len(ci.transition_thread.started)
    clock["t"] = ci._follow_due[0] + ci.FOLLOW_GRACE_S + 0.01   # a stuck decoder must not stop the show
    last_follow(ci)[1]()
    assert len(ci.transition_thread.started) == n + 1


def test_end_follow_without_video_waits_for_go(ci, fixtures):
    ci.grid_cells = {(0, 0): fixtures["red_png"], (0, 1): fixtures["counting"]}
    ci.cue_settings = {"0": {"follow": {"mode": "end"}}}
    ci._cut_to_column(0)
    assert last_follow(ci) is None and ci._follow_due is None


def test_manual_go_cancels_a_pending_follow(ci):
    ci.cue_settings = {"0": {"follow": {"mode": "after", "seconds": 5}}}
    ci._cut_to_column(0)
    _, stale = last_follow(ci)
    ci._go_to_next_cue_manual()                          # operator GOes early
    n = len(ci.transition_thread.started)
    stale()
    assert len(ci.transition_thread.started) == n


def test_follow_waits_for_a_running_fade(ci):
    ci.cue_settings = {"1": {"follow": {"mode": "after", "seconds": 0.2}}}
    ci._initiate_transition(0, 1)                        # 1 s fade; follow armed at its start
    _, fire = last_follow(ci)
    fire()                                               # fires during the fade: re-arms
    ms, retry = last_follow(ci)
    assert ms == 50 and len(ci.transition_thread.started) == 1
    ci._finalize_transition_from_thread(1)
    retry()
    assert ci.transition_thread.started[-1][3] == 2


@pytest.mark.parametrize("what", ["blackout", "presentation", "last_cue"])
def test_no_follow_when_it_would_be_wrong(ci, what):
    ci.cue_settings = {"0": {"follow": {"mode": "after", "seconds": 1}}, "2": {"follow": {"mode": "after", "seconds": 1}}}
    if what == "last_cue":
        ci._cut_to_column(2)
        assert ci._follow_due is None
        return
    ci._cut_to_column(0)
    _, fire = last_follow(ci)
    if what == "blackout":
        ci.transition_var.set("None")
        ci._blackout()
    else:
        ci.go_button = type("B", (), {"config": lambda self, **k: None})()
        ci._start_presentation = ci_start_presentation_without_timer(ci)
        ci._start_presentation()
    n = len(ci.transition_thread.started)
    fire()
    assert len(ci.transition_thread.started) == n and ci._follow_due is None


def ci_start_presentation_without_timer(ci):
    def start():
        ci.presentation_active = True
        ci._cancel_follow()
    return start


def test_countdown_label(ci):
    class L:
        mapped = False
        def config(self, **k): self.text = k.get("text")
        def winfo_ismapped(self): return self.mapped
        def pack(self, **k): self.mapped = True
        def pack_forget(self): self.mapped = False
    ci.follow_label, ci._preview_top_bar = L(), None
    ci.cue_settings = {"0": {"follow": {"mode": "after", "seconds": 4}}}
    ci._cut_to_column(0)
    ci._refresh_follow_label()
    assert ci.follow_label.mapped and ci.follow_label.text.startswith("→ continues to Cue 2 in 4.0")
    ci._cancel_follow()
    ci._refresh_follow_label()
    assert not ci.follow_label.mapped


def test_a_pending_follow_moves_with_reordered_cues(ci):
    ci.cue_settings = {"0": {"follow": {"mode": "after", "seconds": 3}}}
    ci._cut_to_column(0)
    ci._build_grid = lambda: None
    ci._reorder_cues(0, 2)                               # live cue moves to position 2
    assert ci.active_col == 2 and ci._follow_due[1] == 2
