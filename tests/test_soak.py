# SPDX-License-Identifier: MPL-2.0
"""Soak test: samples, findings, the recorder and the window's run."""
import collections
import csv
import json
import os

import pytest

import soak
from test_cue_activation import ci, commands  # noqa: F401


def rows(n, step=60.0, **ramp):
    out = []
    for i in range(n):
        r = {"t_s": i * step, "rss_mb": 300.0, "threads": 20, "decoders": 2, "gl_textures": 6,
             "renders_per_s": 60.0, "late_window": 0, "frames_window": 600, "errors_total": 0,
             "audio_underruns": 0, "gos": i, "active_cue": i % 3, "render_thread_alive": True}
        for k, per in ramp.items():
            r[k] = r[k] + per * i
        out.append(r)
    return out


def test_a_level_run_has_no_findings():
    s = soak.summarize(rows(30), target_fps=60)
    assert s["ok"] and s["findings"][0].startswith("Nothing drifted") and s["rss_growth_mb_per_h"] == 0.0


def test_growth_errors_and_drops_are_found():
    r = rows(60, rss_mb=4.0, threads=1, decoders=1)           # +240 MB/h, a thread and a decoder a minute
    r[40]["renders_per_s"] = 41.0
    r[-1]["errors_total"] = 3
    r[-1]["audio_underruns"] = 2
    r[30]["late_window"] = 60
    s = soak.summarize(r, target_fps=60)
    text = " ".join(s["findings"])
    assert not s["ok"]
    assert "Memory keeps growing: 240 MB per hour" in text
    assert "threads grew" in text and "decoders grew" in text
    assert "3 render error(s)" in text and "dropped to 41.0" in text and "2 sound underrun(s)" in text
    assert "GPU textures" not in text


def test_warm_up_growth_is_not_a_leak():
    r = rows(30)
    for i in range(6):                                       # the first fifth fills caches
        r[i]["rss_mb"] = 100.0 + 40 * i
    for x in r[6:]:
        x["rss_mb"] = 320.0
    s = soak.summarize(r, target_fps=60)
    assert s["ok"], s["findings"]


def test_memory_is_compared_at_the_same_cue():
    """As on the owner's PC: a still cue uses less memory than a video cue,
    and a run that ends on the video looked like growth from first to last."""
    r = rows(60, step=10.0)
    for x in r:
        x["rss_mb"] = 500.0 if x["active_cue"] == 0 else 850.0     # level at each cue
    r[-1]["active_cue"], r[-1]["rss_mb"] = 1, 860.0
    s = soak.summarize(r, target_fps=60)
    assert s["ok"], s["findings"]
    assert abs(s["rss_same_cue_growth_mb_per_h"]) < 30
    r = rows(240, step=20.0)                                       # 80 minutes (64 after warm-up), +90 MB/h
    for x in r:
        x["rss_mb"] = (500.0 if x["active_cue"] == 0 else 850.0) + 90.0 * x["t_s"] / 3600.0
    s = soak.summarize(r, target_fps=60)
    text = " ".join(s["findings"])
    assert "Memory keeps growing: 90 MB per hour after warm-up, comparing samples at the same cue" in text
    assert "(cue 1: 524 to " in text, text                     # columns count from 0, cues from 1
    assert s["rss_same_cue_growth_mb_per_h"] == 90.0


def test_without_cue_numbers_memory_is_compared_first_to_last():
    r = rows(60, rss_mb=4.0)
    for x in r:
        x["active_cue"] = None
    text = " ".join(soak.summarize(r)["findings"])
    assert "Memory keeps growing: 240 MB per hour after warm-up (" in text


def test_python_objects_and_collector_pauses_are_found():
    r = rows(240, step=20.0)                                       # 80 minutes
    for i, x in enumerate(r):
        x["py_blocks"] = 400_000 + int(80_000 * x["t_s"] / 3600)   # +80k per hour at every cue
        x["gc_full_total"], x["gc_full_pause_total_ms"] = i // 10, 12.0 * (i // 10)
        x["gc_pause_max_ms"] = 3.0
    r[200]["gc_pause_max_ms"] = 41.5
    s = soak.summarize(r, target_fps=60)
    text = " ".join(s["findings"])
    assert "Python objects keep growing: about 80 thousand per hour at the same cue (cue 1: 421 to " in text, text
    assert "garbage collector held everything up for up to 42 ms (one frame is 16.7 ms)" in text
    assert s["gc_full_pause_avg_ms"] == 12.0 and s["gc_pause_max_ms"] == 41.5 and s["gc_full_collections"] > 10
    for x in r:                                                    # level objects, short pauses: nothing
        x["py_blocks"] = 400_000 + (x["t_s"] % 7) * 100
        x["gc_pause_max_ms"] = 3.0
    s = soak.summarize(r, target_fps=60)
    assert s["ok"], s["findings"]


def test_sample_takes_the_collector_from_the_snapshot():
    snap = {"gc": {"blocks": 512_000, "full_total": 7, "full_pause_total_ms": 80.5, "pause_max_ms_window": 9.1}}
    row = soak.sample(snap)
    assert (row["py_blocks"], row["gc_full_total"], row["gc_full_pause_total_ms"], row["gc_pause_max_ms"]) == \
        (512_000, 7, 80.5, 9.1)
    assert set(row) == set(soak.FIELDS)


def test_a_stopped_render_thread_and_no_cues_are_findings():
    r = rows(5)
    r[-1]["render_thread_alive"] = False
    for x in r:
        x["gos"] = 0
    text = " ".join(soak.summarize(r)["findings"])
    assert "render thread stopped" in text and "No cue was played" in text
    assert soak.summarize([])["ok"] is False


def test_rss_and_sample_from_a_snapshot():
    assert soak.rss_mb() > 10
    snap = {"output": {"renders_per_s": 59.9, "late_window": 1, "frames_window": 300},
            "decoders": [{}, {}, {}], "errors_total": {"KeyError": 2}, "audio": {"underruns": 4},
            "render_thread_alive": True}
    row = soak.sample(snap, gos=7, active_cue=2, t_s=12.34)
    assert row["decoders"] == 3 and row["errors_total"] == 2 and row["audio_underruns"] == 4
    assert row["t_s"] == 12.3 and row["gos"] == 7 and row["threads"] >= 1 and row["gl_textures"] is None


def test_recorder_writes_as_it_goes(tmp_path):
    rec = soak.Recorder(str(tmp_path), stamp="x")
    for r in rows(3):
        rec.add(r)
    lines = list(csv.reader(open(rec.csv_path)))
    assert lines[0] == list(soak.FIELDS) and len(lines) == 4    # on disk before the end
    s = rec.finish(target_fps=60, extra={"how": "finished"})
    assert json.load(open(rec.summary_path))["how"] == "finished" and s["samples"] == 3


def test_app_runs_gos_samples_and_writes_a_summary(ci, tmp_path, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr("control_interface.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr(type(ci), "CONFIG_DIR", str(tmp_path))
    ci._current_health = lambda: {"output": {"renders_per_s": 60.0}, "decoders": [],
                                  "errors_total": {}, "render_thread_alive": True}
    gos = []
    ci._trigger_go = lambda: gos.append(clock["t"])
    ci.active_col = None                                     # nothing on the output yet
    went = []
    ci._go_to_cue = lambda col: went.append(col)
    ci._soak_start(60.0, 5.0, sample_s=10.0)
    assert ci._soak["gos"] == 1 and went == [ci._soak_cues()[0]]  # went to the first cue with media
    for _ in range(70):                                      # 70 s of ticks every second
        clock["t"] += 1.0
        ci._soak_tick(ci._soak["token"]) if ci._soak else None
    assert ci._soak is None                                  # finished after 60 s
    assert 10 <= len(gos) <= 12                              # a GO every 5 s
    files = sorted(os.listdir(tmp_path / "soak"))
    summary = json.load(open(tmp_path / "soak" / [f for f in files if f.endswith(".json")][0]))
    assert summary["how"] == "finished" and summary["gos"] == len(gos) + 1 and summary["samples"] >= 6


def test_presentation_mode_runs_the_cues_and_the_soak_only_records(ci, tmp_path, monkeypatch):
    """Checklist I2 (72 h with Presentation Mode and Loop): the soak test's own GO would
    stop a running presentation (GO reads STOP then), so it only records and counts
    the presentation's cue changes as GOs."""
    clock = {"t": 1000.0}
    monkeypatch.setattr("control_interface.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr(type(ci), "CONFIG_DIR", str(tmp_path))
    ci._current_health = lambda: {"output": {"renders_per_s": 60.0}, "decoders": [],
                                  "errors_total": {}, "render_thread_alive": True}
    cues = ci._soak_cues()
    gos, went = [], []
    ci._trigger_go = lambda: gos.append(1)
    ci._go_to_cue = lambda col: went.append(col)
    ci.presentation_active = True
    ci.active_col = cues[0]
    ci._soak_start(60.0, 5.0, sample_s=10.0)
    for i in range(70):
        clock["t"] += 1.0
        if i % 20 == 19:                                     # the presentation goes on every 20 s
            ci.active_col = cues[(cues.index(ci.active_col) + 1) % len(cues)] if len(cues) > 1 else ci.active_col + 1
        ci._soak_tick(ci._soak["token"]) if ci._soak else None
    assert gos == [] and went == []                          # never GO, never jumped
    files = sorted(os.listdir(tmp_path / "soak"))
    summary = json.load(open(tmp_path / "soak" / [f for f in files if f.endswith(".json")][0]))
    assert summary["gos"] == 3                               # the changes at 20, 40 and 60 s
    assert not any("No cue was played" in f for f in summary["findings"])
    ci.presentation_active = False


def test_a_show_that_stops_at_its_end_starts_over(ci, tmp_path, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr("control_interface.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr(type(ci), "CONFIG_DIR", str(tmp_path))
    ci._current_health = lambda: {"output": {}, "decoders": [], "errors_total": {}, "render_thread_alive": True}
    cues = ci._soak_cues()
    ci.go_at_end = "stop"
    ci.active_col = cues[-1]                                 # on the last cue with media
    gos, went = [], []
    ci._trigger_go = lambda: gos.append(1)
    ci._go_to_cue = lambda col: went.append(col)
    ci._soak_start(30.0, 5.0, sample_s=10.0)
    clock["t"] += 5.0
    ci._soak_tick(ci._soak["token"])
    assert went == [cues[0]] and gos == [] and ci._soak["gos"] == 1
    ci._soak_stop()


def test_soak_window(ci, tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import ui_theme
    monkeypatch.setattr(type(ci), "CONFIG_DIR", str(tmp_path))
    root = tk.Tk()
    root.withdraw()
    ui_theme.apply(root)
    ci.root = root
    ci._current_health = lambda: None
    ci._trigger_go = lambda: None
    try:
        win = ci._open_soak()
        vd = win._vd
        vd["every_var"].set("1")
        assert vd["start"]() is False and "2 seconds or more" in vd["status"].cget("text")
        vd["every_var"].set("x")
        assert vd["start"]() is False and "numbers" in vd["status"].cget("text")
        vd["every_var"].set("20")
        assert vd["start"]() is True and ci._soak is not None
        vd["stop"]()
        assert ci._soak is None and "Soak test stopped" in vd["status"].cget("text")
        win.destroy()
    finally:
        root.destroy()


def test_stills_on_the_output_are_not_a_frame_rate_drop():
    r = rows(20)
    for x in r[10:15]:
        x["renders_per_s"], x["active_content"] = 0.5, False     # a still cue: drawn once
    assert soak.summarize(r, target_fps=60)["ok"]
    r[12]["active_content"] = True
    assert "dropped to 0.5" in " ".join(soak.summarize(r, target_fps=60)["findings"])


def test_a_window_with_only_a_transitions_frames_is_not_a_frame_rate_drop():
    """The owner's 72-hour soak (2026-11-20): Presentation Mode's GO and the soak's
    sample are timed apart, so a sample can fall on a GO onto a still picture, when
    the window holds the transition's first frame only (0.0 renders/s) or a part of it."""
    r = rows(30, frames_window=0)
    for x in r:
        x["frames_window"], x["active_content"] = 300, True
    r[20].update(renders_per_s=0.0, frames_window=1)          # the GO's first frame
    r[21].update(renders_per_s=0.0, frames_window=0, active_content=False)    # then a still
    r[24].update(renders_per_s=36.0, frames_window=8)          # 0.2 s of a transition
    s = soak.summarize(r, target_fps=60)
    assert s["ok"] and s["renders_per_s_min"] == 60.0
    r[26].update(renders_per_s=47.8, frames_window=239)       # a full window running slow
    s = soak.summarize(r, target_fps=60)
    assert "dropped to 47.8" in " ".join(s["findings"])


def test_moving_content_without_frames_twice_is_a_stall():
    r = rows(30)
    for x in r:
        x["frames_window"], x["active_content"] = 300, True
    r[20].update(renders_per_s=0.0, frames_window=0)
    assert soak.summarize(r, target_fps=60)["ok"]              # once: a GO at the sample
    r[21].update(renders_per_s=0.0, frames_window=1)
    s = soak.summarize(r, target_fps=60)
    assert not s["ok"] and "stopped drawing moving content 1 time(s), first at 20 min" in " ".join(s["findings"])


def test_the_last_row_is_not_recorded_twice(ci, tmp_path, monkeypatch):
    """The tick that ends the run samples, then the stop took its own sample at the same
    moment (the 72-hour soak's CSV ended with two equal rows)."""
    clock = {"t": 1000.0}
    monkeypatch.setattr("control_interface.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr(type(ci), "CONFIG_DIR", str(tmp_path))
    ci._current_health = lambda: {"output": {"renders_per_s": 60.0}, "decoders": [],
                                  "errors_total": {}, "render_thread_alive": True}
    ci._trigger_go = lambda: None
    ci._go_to_cue = lambda col: None
    ci.active_col = ci._soak_cues()[0]
    ci._soak_start(60.0, 5.0, sample_s=10.0)
    for _ in range(70):
        clock["t"] += 1.0
        ci._soak_tick(ci._soak["token"]) if ci._soak else None
    path = [p for p in (tmp_path / "soak").iterdir() if p.suffix == ".csv"][0]
    times = [r["t_s"] for r in csv.DictReader(open(path, encoding="utf-8"))]
    assert len(times) == len(set(times)) and times[-1] == "60.0"


def test_process_line_in_output_health(ci):
    p = ci._process_status()
    assert p["rss_mb"] > 10 and p["threads"] >= 1 and p["up_s"] >= 0 and p["peak_mb"] >= p["rss_mb"]
    text = ci._process_text({"rss_mb": 812.4, "peak_mb": 1210.0, "threads": 31, "up_s": 3725})
    assert text == "memory 812 MB (highest 1210)   threads 31   up 1:02:05"
    assert ci._process_text({}) == "?"


def test_gdi_objects_on_windows_only_and_their_growth_is_a_finding(ci):
    import sys
    assert (soak.gdi_objects() is None) == (sys.platform != "win32")
    assert ci._process_status()["gdi"] == soak.gdi_objects()
    text = ci._process_text({"rss_mb": 812.4, "peak_mb": 1210.0, "threads": 31, "up_s": 3725, "gdi": 412})
    assert text == "memory 812 MB (highest 1210)   threads 31   GDI objects 412   up 1:02:05"
    r = rows(360, step=10.0)                                       # an hour, one more every 3 s (screeninfo)
    for x in r:
        x["gdi_objects"] = 400 + int(x["t_s"] / 3)
    s = soak.summarize(r, target_fps=60)
    text = " ".join(s["findings"])
    assert "GDI objects keep growing: 1200 per hour (" in text, text
    assert "limit of 10,000 is reached in about 7 hours" in text
    for x in r:
        x["gdi_objects"] = 400 + (x["t_s"] % 3 == 0)
    assert soak.summarize(r, target_fps=60)["ok"]


def test_a_short_run_does_not_judge_memory():
    r = rows(24, step=10.0, rss_mb=2.0)                            # 4 minutes, +720 MB/h
    s = soak.summarize(r, target_fps=60)
    assert s["ok"] and not any("Memory" in f for f in s["findings"])
    assert s["findings"][0].startswith("Nothing drifted: threads,") and "too short" in s["notes"][0]
    assert soak.summarize(rows(30), target_fps=60)["notes"] == []


class _Fn:
    """A stand-in Windows function that takes argtypes/restype like ctypes' do."""
    def __init__(self, fn):
        self.fn, self.argtypes, self.restype = fn, None, None

    def __call__(self, *a):
        return self.fn(*a)


def test_windows_memory_reading_makes_its_structure_once(monkeypatch):
    """On the owner's PC the status light's memory reading (twice a second)
    made a new ctypes structure each time, and ctypes kept them all: Python's
    allocations grew 408 thousand blocks an hour."""
    import ctypes
    import gc
    import sys

    def info(handle, ref, size):
        ref._obj.WorkingSetSize = 512 * 2 ** 20
        return 1
    dlls = {"kernel32": type("K", (), {"GetCurrentProcess": _Fn(lambda: 1)})(),
            "psapi": type("P", (), {"GetProcessMemoryInfo": _Fn(info)})()}
    monkeypatch.setattr(ctypes, "WinDLL", lambda name: dlls[name], raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(soak, "_mem", None)
    assert soak.rss_mb() == 512.0
    api = soak._mem
    for _ in range(50):
        soak.rss_mb()
    gc.collect()
    before = sys.getallocatedblocks()
    for _ in range(2000):
        assert soak.rss_mb() == 512.0
    gc.collect()
    assert soak._mem is api, "made once"
    assert (sys.getallocatedblocks() - before) / 2000 < 1, "nothing kept per call"


class _Leaky:
    pass


def test_object_census_names_the_kinds_that_grew():
    a = soak.object_census()
    kept = [_Leaky() for _ in range(6000)]
    b = soak.object_census()
    grew = dict(soak.census_growth(a, b))
    assert grew["test_soak._Leaky"] >= 6000
    assert soak.census_growth(a, b)[0][0] == "test_soak._Leaky", "largest first"
    note = soak.census_note(soak.census_growth(a, b), sum(b.values()) - sum(a.values()))
    assert note.startswith("Python objects that grew most after warm-up: test_soak._Leaky +6,0")
    assert soak.census_note([("list", 10)], 10) is None, "small growth: no note"
    del kept


def test_the_soak_window_records_the_census(ci, tmp_path, monkeypatch):
    import json
    clock = {"t": 1000.0}
    monkeypatch.setattr("control_interface.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr(type(ci), "CONFIG_DIR", str(tmp_path))
    ci._current_health = lambda: {"output": {}, "decoders": [], "errors_total": {}, "render_thread_alive": True}
    ci._trigger_go = lambda: None
    ci._soak_start(100.0, 50.0, sample_s=10.0)
    clock["t"] += 25.0                                     # past warm-up (a fifth): census
    ci._soak_tick(ci._soak["token"])
    assert "census" in ci._soak
    kept = [_Leaky() for _ in range(10000)]
    s = ci._soak_stop()
    assert s["objects_end"] - s["objects_after_warmup"] >= 9000
    assert s["object_growth"][0][0] == "test_soak._Leaky"
    assert any(n.startswith("Python objects that grew most") for n in s["notes"])
    saved = json.load(open(next(p for p in tmp_path.joinpath("soak").iterdir() if p.suffix == ".json")))
    assert saved["object_growth"][0][0] == "test_soak._Leaky" and saved["notes"] == s["notes"]
    del kept


_kept_strings = []


def _grow_untracked(n):
    """Strings: Python allocations the garbage collector does not track (like the
    owner's PC's steady growth, which the object census did not see)."""
    for i in range(n):
        _kept_strings.append(f"untracked {i} " * 2)


_ring = collections.deque(maxlen=8000)


def _refill_ring():
    """A bounded buffer refilled with new items (like Health's samples): not growth."""
    for i in range(8000):
        _ring.append(f"sample {i} " * 2)


def test_the_soak_can_trace_allocations_and_names_the_site(ci, tmp_path, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr("control_interface.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr(type(ci), "CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv(soak.TRACE_ENV, "1")
    ci._current_health = lambda: {"output": {}, "decoders": [], "errors_total": {}, "render_thread_alive": True}
    ci._trigger_go = lambda: None
    ci._soak_start(100.0, 50.0, sample_s=10.0)
    seen = []
    for dt in (25.0, 20.0, 0.0):        # past warm-up: tracing starts; at 45 s of 100: the baseline
        _refill_ring()                  # the same code refilling a bounded buffer each time
        clock["t"] += dt
        ci._soak_tick(ci._soak["token"])
        seen.append((bool(ci._soak.get("tracing")), "trace" in ci._soak))
    assert seen == [(True, False), (True, True), (True, True)]
    _grow_untracked(3000)
    s = ci._soak_stop()
    report = open(s["tracemalloc_report"], encoding="utf-8").read()
    first = report.split("\n\n")[1]                        # the largest growth
    assert "_grow_untracked" in first and "test_soak.py" in first, first
    assert "_refill_ring" not in report.split("\n\n", 2)[2].split("\n\n")[0], "turnover reported as growth"
    assert report.startswith("Python allocations from the baseline to the end")
    assert "Tkinter callbacks registered" in report
    import tracemalloc
    assert not tracemalloc.is_tracing()
    _kept_strings.clear()


def test_a_soak_stopped_before_the_baseline_stops_tracing(ci, tmp_path, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr("control_interface.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr(type(ci), "CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv(soak.TRACE_ENV, "1")
    ci._current_health = lambda: {"output": {}, "decoders": [], "errors_total": {}, "render_thread_alive": True}
    ci._trigger_go = lambda: None
    ci._soak_start(100.0, 50.0, sample_s=10.0)
    clock["t"] += 25.0
    ci._soak_tick(ci._soak["token"])
    import tracemalloc
    assert tracemalloc.is_tracing()
    assert "tracemalloc_report" not in ci._soak_stop()
    assert not tracemalloc.is_tracing()


def test_tracing_is_off_unless_asked(ci, tmp_path, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr("control_interface.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr(type(ci), "CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv(soak.TRACE_ENV, raising=False)
    ci._current_health = lambda: {"output": {}, "decoders": [], "errors_total": {}, "render_thread_alive": True}
    ci._trigger_go = lambda: None
    ci._soak_start(100.0, 50.0, sample_s=10.0)
    clock["t"] += 25.0
    ci._soak_tick(ci._soak["token"])
    assert "trace" not in ci._soak
    assert "tracemalloc_report" not in ci._soak_stop()


def test_growth_at_the_long_show_bar_is_noted_not_a_finding():
    """The owner's 8-hour soak: 10.2 MB an hour read as "Nothing drifted"."""
    rows = [{"t_s": t * 60.0, "rss_mb": 400 + t * 10.5 / 60, "threads": 7, "decoders": 2, "gl_textures": 5,
             "renders_per_s": 60.0, "late_window": 0, "frames_window": 300, "errors_total": 0, "gos": t,
             "render_thread_alive": True, "active_content": True} for t in range(0, 240)]
    s = soak.summarize(rows)
    assert s["ok"] and s["findings"][0].startswith("Nothing drifted")
    assert any("about 10 MB an hour" in n and "a long show should stay under" in n for n in s["notes"]), s["notes"]
    flat = [dict(r, rss_mb=400.0) for r in rows]
    assert not any("MB an hour after warm-up" in n for n in soak.summarize(flat)["notes"])
    short = rows[:40]                                       # 40 minutes: no judgement against the bar
    assert not any("a long show should stay under" in n for n in soak.summarize(short)["notes"])


def test_memory_that_rises_once_and_stays_level_is_told_apart():
    """The owner's 8-hour soak of 2026-11-15: about 90 MB more at 3 h 9 min, then level
    to the end (15.5 MB an hour by a straight line over the run)."""
    r = rows(8 * 60 // 2, step=120.0)                    # 8 hours, a sample every 2 minutes
    for x in r:
        x["rss_mb"] += 90.0 if x["t_s"] >= 189 * 60 else 0.0
    s = soak.summarize(r, target_fps=60)
    assert s["rss_same_cue_growth_mb_per_h"] > soak.BAR_MB_PER_H          # the straight line: over the bar
    assert abs(s["rss_same_cue_growth_second_half_mb_per_h"]) < 1.0         # then level
    assert any("stayed level" in n and "second half" in n for n in s["notes"]), s["notes"]
    steady = rows(8 * 60 // 2, step=120.0, rss_mb=0.75)                     # 22.5 MB an hour, all the way
    s2 = soak.summarize(steady, target_fps=60)
    assert s2["rss_same_cue_growth_second_half_mb_per_h"] > soak.BAR_MB_PER_H
    assert any("a long show should stay under" in n for n in s2["notes"]), s2["notes"]
