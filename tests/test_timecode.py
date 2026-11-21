# SPDX-License-Identifier: MPL-2.0
"""MIDI timecode: parsing, the MTC decoder (quarter frames, full frames), the
firing rules, the MIDI listener, Cue Settings, and the app following it. No
MTC source was available: messages are built to the MIDI 1.0 MTC layout."""
import json

import pytest

import midi_remote
import project_schema
import timecode as tcm
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401
from test_midi_remote import FakeMidiIn, fake  # noqa: F401


def test_parse_clean_and_frame_arithmetic():
    assert tcm.clean(" 1:02:03:04 ") == "01:02:03:04" and tcm.clean("01:02:03;29") == "01:02:03:29"
    for bad in ("", "1:2:3", "24:00:00:00", "00:60:00:00", "00:00:00:30", "aa:bb:cc:dd", None):
        assert tcm.clean(bad) is None
    assert tcm.add_frames((0, 0, 59, 24), 2, 25) == (0, 1, 0, 1)
    assert tcm.add_frames((23, 59, 59, 29), 1, 30) == (0, 0, 0, 0)
    assert tcm.seconds((1, 0, 0, 12), 24) == 3600.5


@pytest.mark.parametrize("rate", [0, 1, 2, 3])
def test_quarter_frames_decode_two_frames_later(rate):
    d = tcm.MTCDecoder()
    fps = tcm.RATES[rate]
    out = [d.feed(m) for m in tcm.quarter_frames((1, 23, 45, 10), rate)]
    assert out[:7] == [None] * 7
    assert out[7] == tcm.add_frames((1, 23, 45, 10), 2, fps) + (rate,)


def test_out_of_order_pieces_restart_and_full_frames_locate():
    d = tcm.MTCDecoder()
    qf = tcm.quarter_frames((0, 0, 10, 0))
    for m in qf[3:]:                                  # joined in the middle: nothing until a whole set
        assert d.feed(m) is None
    assert [d.feed(m) for m in qf][-1][:4] == (0, 0, 10, 2)
    for m in reversed(qf):                            # running backwards: ignored
        assert d.feed(m) is None
    assert d.feed(tcm.full_frame((10, 0, 0, 5), rate=3)) == (10, 0, 0, 5, 3)
    assert d.feed([0xF0, 0x7E, 0x7F, 0x06, 0x01, 0xF7]) is None and d.feed([0xF8]) is None


def test_firing_rules():
    cues = [(10.0, 1), (12.0, 2), (12.5, 3)]
    assert tcm.cue_to_fire(cues, 9.9, 10.0, 0) == 1, "reaching the time fires"
    assert tcm.cue_to_fire(cues, 10.0, 10.1, 1) is None
    assert tcm.cue_to_fire(cues, 11.9, 12.6, 0) == 3, "several passed in one step: the last"
    assert tcm.cue_to_fire(cues, 5.0, 10.0, 0) is None, "a jump (locate) fires nothing"
    assert tcm.cue_to_fire(cues, 12.1, 11.95, 0) is None, "backwards fires nothing"
    assert tcm.cue_to_fire(cues, None, 10.0, 0) is None, "the first time seen is a start"
    assert tcm.cue_to_fire(cues, 9.9, 10.0, 1) is None, "the live cue does not fire again"


def test_listener_reads_mtc_only_when_asked(fake):
    calls = []
    FakeMidiIn.ignore_types = lambda self, **kw: calls.append(kw)
    try:
        got = []
        m = midi_remote.MidiRemote(lambda c, a: got.append((c, a)), "loopMIDI Port 2", backend=FakeMidiIn, timecode=True)
        assert m.start() and calls == [{"sysex": False, "timing": False, "active_sense": True}]
        cb = FakeMidiIn.instances[-1].cb
        for msg in tcm.quarter_frames((0, 0, 5, 0)):
            cb((msg, 0.001))
        assert m.timecode[:5] == (0, 0, 5, 2, 1) and got == [], "MTC is not a command"
        cb(([0xF8], 0.0))
        cb(([0xC0, 2], 0.0))
        assert got == [("cue", 3)], "Program Change still works"
        calls.clear()
        off = midi_remote.MidiRemote(lambda c, a: None, "loopMIDI Port 2", backend=FakeMidiIn)
        assert off.start() and calls == [], "without timecode the library keeps filtering"
        for msg in tcm.quarter_frames((0, 0, 5, 0)):
            FakeMidiIn.instances[-1].cb((msg, 0.0))
        assert off.timecode is None
    finally:
        del FakeMidiIn.ignore_types


def run(ci, start, frames, fps=25, rate=1):
    """Feed the app a running timecode, frame by frame (two per MTC update)."""
    t = start
    fired = []
    for _ in range(frames // 2):
        tc = t + (rate, 0.0)
        r = ci._timecode_tick(tc)
        if r is not None:
            fired.append((tcm.fmt(t), r))
        t = tcm.add_frames(t, 2, fps)
    return fired


def test_app_follows_running_timecode(blind_ci):
    ci = blind_ci
    ci.midi_timecode = True
    ci.cue_settings["1"] = {"timecode": "01:00:00:00"}
    ci.cue_settings["2"] = {"timecode": "01:00:02:00"}
    fired = run(ci, (0, 59, 59, 0), 40)
    assert [c for _, c in fired] == [1], fired
    assert ci.transition_thread.started[-1][3] == 1, "with the cue's transition (a fade)"
    # While that fade runs, the next time is reached: it fires when the fade ends.
    ci.transition_active = True
    assert run(ci, (1, 0, 1, 20), 20) == [("01:00:02:01", 2)] and ci._tc_pending == 2
    n = len(ci.transition_thread.started)
    ci.transition_active = False
    ci._timecode_tick()
    assert len(ci.transition_thread.started) == n + 1 and ci._tc_pending is None
    # A locate past a cue time changes nothing; switched off, nothing happens.
    ci.active_col = 0
    assert run(ci, (1, 0, 5, 0), 4) == [] and ci._timecode_tick((1, 0, 1, 0, 1, 0.0)) is None
    ci.midi_timecode = False
    assert run(ci, (0, 59, 59, 20), 20) == []


def test_cue_settings_timecode_field_and_show_file():
    tch = ["Show default", "None (cut)", "Fade", "Smooth Fade", "Dip to Black"]
    fch = ["Wait", "After", "End"]
    import control_interface as ci_mod
    form = ci_mod.ControlInterface._cue_settings_from_form
    assert form(tch[0], tch, "", fch[0], fch, "5", "", "", "01:00:12:05") == {"timecode": "01:00:12:05"}
    with pytest.raises(ValueError, match="HH:MM:SS:FF"):
        form(tch[0], tch, "", fch[0], fch, "5", "", "", "12 seconds")
    assert project_schema.clean_cue_settings({"0": {"timecode": "1:00:00:00"}, "1": {"timecode": "x"}}) == \
        {"0": {"timecode": "01:00:00:00"}}


def test_setting_persists(tmp_path, monkeypatch):
    import control_interface as ci_mod
    from media_manager import MediaManager
    monkeypatch.setattr(ci_mod.ControlInterface, "CONFIG_DIR", str(tmp_path / ".vd"))
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    a = ci_mod.ControlInterface(MediaManager())
    assert a.midi_timecode is False
    a.midi_timecode = True
    a._save_config()
    assert json.load(open(a.config_path))["midi_timecode"] is True
    b = ci_mod.ControlInterface(MediaManager())
    b._load_config()
    assert b.midi_timecode is True
    for o in (a, b):
        o.proxy_executor.shutdown(wait=False)
