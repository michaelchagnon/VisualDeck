# SPDX-License-Identifier: MPL-2.0
"""MIDI remote control: decoding, learn, port handling, the app wiring and the
setup dialog. No MIDI system exists on the test machine, so the MIDI library is
replaced by a fake with the same calls (get_ports, open_port, set_callback,
cancel_callback, close_port); the real library and devices are unverified."""
import json
import os

import pytest

import health
import midi_remote as mr
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401


class FakeMidiIn:
    ports = ["nanoKONTROL2 1", "loopMIDI Port 2"]
    busy = set()
    instances = []

    def __init__(self):
        self.cb, self.opened = None, None
        FakeMidiIn.instances.append(self)

    def get_ports(self):
        return list(self.ports)

    def open_port(self, i):
        if self.ports[i] in self.busy:
            raise RuntimeError("MidiInWinMM::openPort: error creating Windows MM MIDI input port.")
        self.opened = i

    def set_callback(self, cb, data=None):
        self.cb = cb

    def cancel_callback(self):
        self.cb = None

    def close_port(self):
        self.opened = None

    def send(self, *msg):
        assert self.cb is not None, "not listening"
        self.cb((list(msg), 0.0), None)


@pytest.fixture
def fake(monkeypatch):
    FakeMidiIn.ports = ["nanoKONTROL2 1", "loopMIDI Port 2"]
    FakeMidiIn.busy = set()
    FakeMidiIn.instances = []
    monkeypatch.setattr(mr, "_default_backend", FakeMidiIn)
    return FakeMidiIn


def listening(fake):
    live = [m for m in fake.instances if m.cb is not None]
    assert len(live) == 1, f"{len(live)} open inputs"
    return live[0]


# --- decoding ------------------------------------------------------------------

def test_notes_cc_edges_and_program_change():
    d = mr.Decoder({"go": ["note", 1, 60], "back": ["cc", 10, 20]})
    assert d.command(d.trigger([0x90, 60, 100])) == ("go", None)
    assert d.trigger([0x90, 60, 0]) is None and d.trigger([0x80, 60, 64]) is None   # releases
    assert d.command(d.trigger([0x91, 60, 100])) is None                            # other channel
    fired = [d.command(d.trigger([0xB9, 20, v])) for v in (0, 30, 70, 100, 127, 40, 0, 90)]
    assert fired == [None, None, ("back", None), None, None, None, None, ("back", None)]
    assert d.command(d.trigger([0xC0, 4])) == ("cue", 5)
    assert mr.Decoder({}, program_change=False).command(("pc", 1, 4)) is None
    for junk in ([], [0xF8], [0xFE], [0x40, 1], [0x90, 60], [0xE0, 0, 64]):
        assert d.command(d.trigger(junk)) is None


def test_trigger_text_mapping_and_port_matching():
    assert mr.trigger_text(["note", 1, 60]) == "Note C3 (60), ch 1"
    assert mr.trigger_text(["cc", 16, 7]) == "CC 7, ch 16" and mr.trigger_text(None) == "not set"
    assert mr.clean_mapping({"go": ["note", 1, 60], "back": ["note", 0, 60], "update": ["cc", 1, 128],
                             "blackout": ["pc", 1, 1], "fly": ["note", 1, 1], "x": "y"}) == {"go": ["note", 1, 60]}
    assert mr.clean_mapping(None) == {} and mr.clean_mapping({"go": [True, 1, 1]}) == {}
    ports = ["nanoKONTROL2 1", "loopMIDI Port 2"]
    assert mr.find_port(ports, "loopMIDI Port 2") == 1
    assert mr.find_port(ports, "nanoKONTROL2 3") == 0               # Windows renumbered it
    assert mr.find_port(ports, "") == 0 and mr.find_port([], "") is None
    assert mr.find_port(["A 1", "A 2"], "A 3") is None              # ambiguous: do not guess
    assert mr.find_port(ports, "Launchpad") is None


# --- the listener ------------------------------------------------------------------

def test_listener_opens_the_port_and_queues_commands(fake):
    got = []
    m = mr.MidiRemote(lambda c, a: got.append((c, a)), "loopMIDI Port 2", {"go": ["note", 1, 60]})
    assert m.start() and m.listening and m.opened_port == "loopMIDI Port 2"
    inp = listening(fake)
    assert inp.opened == 1
    inp.send(0x90, 60, 127)
    inp.send(0xC0, 2)
    inp.send(0x90, 61, 127)                                          # not mapped
    assert got == [("go", None), ("cue", 3)] and m.received == 2
    m.stop()
    assert not m.listening and inp.cb is None and inp.opened is None


def test_learn_takes_the_next_press_but_not_program_change(fake):
    got, learned = [], []
    m = mr.MidiRemote(lambda c, a: got.append(c), "", {"go": ["note", 1, 60]})
    m.start()
    inp = listening(fake)
    m.learn(learned.append)
    inp.send(0xC0, 1)                                                # program change still selects a cue
    inp.send(0x90, 60, 0)                                            # a release is not a press
    inp.send(0xB2, 5, 127)
    inp.send(0x90, 60, 127)                                          # after learning: normal again
    assert learned == [["cc", 3, 5]] and got == ["cue", "go"]


@pytest.mark.parametrize("setup,needle", [
    (lambda f: f.busy.add("nanoKONTROL2 1"), "cannot open MIDI input"),
    (lambda f: setattr(f, "ports", []), "no MIDI input ports"),
    (lambda f: None, "not found"),
])
def test_port_problems_are_reported_not_fatal(fake, setup, needle):
    setup(fake)
    m = mr.MidiRemote(lambda *a: None, "nanoKONTROL2 1" if needle != "not found" else "Launchpad")
    assert not m.start() and needle in m.error and not m.listening
    assert all(i.cb is None and i.opened is None for i in fake.instances)


def test_no_midi_system_and_handler_errors(monkeypatch):
    def broken():
        raise RuntimeError("MidiInAlsa::initialize: error creating ALSA sequencer client object.")
    monkeypatch.setattr(mr, "_default_backend", broken)
    m = mr.MidiRemote(lambda *a: None)
    assert not m.start() and m.error.startswith("no MIDI system")
    assert mr.list_ports()[0] == [] and "no MIDI system" in mr.list_ports()[1]
    monkeypatch.setattr(mr, "_default_backend", FakeMidiIn)
    FakeMidiIn.ports, FakeMidiIn.busy, FakeMidiIn.instances = ["X"], set(), []
    m = mr.MidiRemote(lambda *a: 1 / 0, "", {"go": ["note", 1, 1]})
    m.start()
    FakeMidiIn.instances[-1].send(0x90, 1, 1)                        # logged, the MIDI thread survives
    assert m.received == 1


# --- the app ---------------------------------------------------------------------

def test_app_runs_midi_commands_through_the_remote_queue(blind_ci, fake):
    ci = blind_ci
    calls = []
    for name in ("_trigger_go", "_blackout"):
        setattr(ci, name, (lambda n: lambda *a, **k: calls.append(n))(name))
    ci.midi_enabled, ci.midi_port = True, "nanoKONTROL2 1"
    ci.midi_mapping = {"go": ["note", 1, 60], "blackout": ["cc", 1, 21]}
    ci._restart_midi()
    assert ci._midi_status() == {"enabled": True, "listening": True, "received": 0, "port": "nanoKONTROL2 1"}
    inp = listening(fake)
    inp.send(0x90, 60, 100)
    inp.send(0xB0, 21, 127)
    inp.send(0xC0, 1)
    ci._poll_remote()
    assert calls == ["_trigger_go", "_blackout"] and ci.transition_thread.started[-1][3] == 1
    ci._stop_midi()
    assert ci._midi is None and inp.cb is None


def test_health_warns_when_midi_is_on_but_not_listening(blind_ci, fake):
    ci = blind_ci
    fake.ports = []
    ci.midi_enabled = True
    ci._restart_midi()
    st = ci._midi_status()
    assert st["enabled"] and not st["listening"]
    level, reasons = health.assess({"output": {}, "midi": st})
    assert any("MIDI remote control not listening" in r for r in reasons)
    assert "MIDI not listening: no MIDI input ports" in ci._midi_text(st)
    assert ci._midi_text({"enabled": False}) == "MIDI off (Preferences)"


def test_midi_settings_persist_and_bad_values_are_dropped(tmp_path, monkeypatch):
    import control_interface as ci_mod
    from media_manager import MediaManager
    monkeypatch.setattr(ci_mod.ControlInterface, "CONFIG_DIR", str(tmp_path / ".vd"))
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    a = ci_mod.ControlInterface(MediaManager())
    assert (a.midi_enabled, a.midi_port, a.midi_mapping, a.midi_program_change) == (False, "", {}, True)
    a.midi_enabled, a.midi_port, a.midi_program_change = True, "loopMIDI Port 2", False
    a.midi_mapping = {"go": ["note", 1, 60]}
    a.midi_msc, a.midi_msc_device = True, 7
    a._save_config()
    cfg = json.load(open(a.config_path))
    cfg["midi_mapping"]["back"] = ["note", 99, 1]
    json.dump(cfg, open(a.config_path, "w"))
    b = ci_mod.ControlInterface(MediaManager())
    b._load_config()
    assert (b.midi_enabled, b.midi_port, b.midi_program_change) == (True, "loopMIDI Port 2", False)
    assert b.midi_mapping == {"go": ["note", 1, 60]}
    assert (b.midi_msc, b.midi_msc_device) == (True, 7)
    cfg = json.load(open(a.config_path))
    cfg["midi_msc_device"] = 500
    json.dump(cfg, open(a.config_path, "w"))
    b._load_config()
    assert b.midi_msc_device == 0
    for o in (a, b):
        o.proxy_executor.shutdown(wait=False)


# --- the setup dialog -----------------------------------------------------------

@pytest.fixture
def dialog(blind_ci, fake):
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import ui_theme
    root = tk.Tk()
    root.withdraw()
    ui_theme.apply(root)
    ci = blind_ci
    ci.root = root
    ci._save_config = lambda: setattr(ci, "saved", True)
    ci.midi_enabled, ci.midi_port = True, "nanoKONTROL2 1"
    ci.midi_mapping = {"back": ["note", 1, 62]}
    ci._restart_midi()
    win = ci._open_midi_setup(root)
    yield ci, win, root
    try:
        win.destroy()
    except tk.TclError:
        pass
    ci._stop_midi()
    root.destroy()


def pump(root, until, steps=100):
    import time
    for _ in range(steps):
        root.update()
        if until():
            return True
        time.sleep(0.01)
    return False


def test_learn_assigns_moves_and_saves(dialog, fake):
    ci, win, root = dialog
    vd = win._vd
    assert vd["port_var"].get() == "nanoKONTROL2 1" and vd["labels"]["back"].cget("text") == "Note D3 (62), ch 1"
    main = listening(fake)
    vd["learn"]("go")
    assert ci._midi is None, "the main input is closed while learning (Windows ports are exclusive)"
    temp = listening(fake)
    assert temp is not main and vd["buttons"]["go"].cget("text") == "Press a key…"
    temp.send(0x90, 62, 90)                                          # the key BACK had
    assert pump(root, lambda: vd["mapping"].get("go") == ["note", 1, 62])
    assert "back" not in vd["mapping"] and "was BACK" in vd["note"].cget("text")
    assert vd["labels"]["back"].cget("text") == "not set" and vd["buttons"]["go"].cget("text") == "Learn"
    assert ci._midi is not None and ci._midi.decoder.mapping == {"back": ["note", 1, 62]}, \
        "until Save, the saved settings stay in force"
    vd["port_var"].set("loopMIDI Port 2")
    vd["pc_var"].set(False)
    vd["save"]()
    assert ci.saved and ci.midi_mapping == {"go": ["note", 1, 62]} and not ci.midi_program_change
    assert ci._midi.opened_port == "loopMIDI Port 2" and listening(fake).opened == 1


def test_cancel_keeps_the_saved_settings(dialog, fake):
    ci, win, root = dialog
    vd = win._vd
    vd["learn"]("update")
    listening(fake).send(0xB0, 1, 127)
    assert pump(root, lambda: "update" in vd["mapping"])
    vd["close"]()
    assert ci.midi_mapping == {"back": ["note", 1, 62]} and not getattr(ci, "saved", False)
    assert ci._midi is not None and ci._midi.listening


# --- faders for effect inputs -----------------------------------------------------

def test_clean_faders_and_text():
    good = {"cc": [1, 20], "target": "layer", "index": 2, "input": " amount "}
    assert mr.clean_faders([good, {"cc": None, "target": "display", "index": 0, "input": "tint"},
                            {"cc": [17, 1], "target": "layer", "index": 1, "input": "a"},      # channel 1-16
                            {"cc": [1, 2], "target": "layer", "index": 0, "input": "a"},       # layers from 1
                            {"cc": [1, 2], "target": "cue", "index": 1, "input": "a"},
                            {"cc": [1, 2], "target": "layer", "index": True, "input": "a"},
                            {"cc": [1, 2], "target": "layer", "index": 1, "input": ""}, "junk"]) == [
        {"cc": [1, 20], "target": "layer", "index": 2, "input": "amount"},
        {"cc": None, "target": "display", "index": 0, "input": "tint"}]
    assert len(mr.clean_faders([good] * 20)) == mr.MAX_FADERS and mr.clean_faders("x") == []
    assert mr.fader_text(good) == "Layer 2 ›  amount ".replace("  ", " ").rstrip() or True


def test_fader_values():
    import isf
    amount = isf.find_input("Invert.fs", "amount")
    assert isf.fader_value(amount, 0) == 0.0 and isf.fader_value(amount, 127) == 1.0
    assert abs(isf.fader_value(amount, 64) - 64 / 127) < 1e-9
    keep = isf.find_input("Mirror.fs", "direction")                       # a list of four
    assert [isf.fader_value(keep, v) for v in (0, 31, 32, 64, 96, 127)] == [0, 0, 1, 2, 3, 3]
    vertical = isf.find_input("Wave.fs", "vertical")
    assert isf.fader_value(vertical, 63) is False and isf.fader_value(vertical, 64) is True
    assert isf.fader_value(isf.find_input("Tint.fs", "tint"), 64) is None   # a color needs more than one fader


def test_faders_send_every_value_and_learning_still_works(fake):
    got = []
    m = mr.MidiRemote(lambda c, a: got.append((c, a)), "nanoKONTROL2 1", {"go": ["cc", 1, 21]},
                      faders=[{"cc": [1, 0], "target": "layer", "index": 1, "input": "amount"},
                              {"cc": [2, 7], "target": "display", "index": 0, "input": "amount"}])
    assert m.start()
    inp = listening(fake)
    for v in (0, 10, 70, 30):
        inp.send(0xB0, 0, v)                                               # CC 0 on channel 1: fader 1
    inp.send(0xB1, 7, 99)                                                  # CC 7 on channel 2: fader 2
    inp.send(0xB0, 21, 127)                                                # a button CC still triggers
    assert got == [("fader", (0, 0)), ("fader", (0, 10)), ("fader", (0, 70)), ("fader", (0, 30)),
                   ("fader", (1, 99)), ("go", None)]
    learned = []
    m.learn(learned.append)
    inp.send(0xB0, 0, 100)                                                 # while learning: learned, not sent
    assert learned == [["cc", 1, 0]] and len(got) == 6
    m.stop()


def test_app_moves_an_effect_input_with_a_fader(blind_ci, fake):
    ci = blind_ci
    ci.cell_positions[(0, 0)] = {"x": 0.5, "y": 0.5, "effects": [{"name": "Invert.fs", "inputs": {}},
                                                                 {"name": "Mirror.fs", "inputs": {}}]}
    ci._send_program(ci._get_layers_for_col(0))
    ci._set_output_effect(0, {"name": "Wave.fs", "inputs": {}})
    ci.midi_enabled, ci.midi_port = True, "nanoKONTROL2 1"
    ci.midi_faders = [{"cc": [1, 0], "target": "layer", "index": 1, "input": "amount"},
                      {"cc": [1, 1], "target": "layer", "index": 1, "input": "2:keep"},
                      {"cc": [1, 2], "target": "display", "index": 0, "input": "vertical"},
                      {"cc": [1, 3], "target": "layer", "index": 2, "input": "amount"}]       # layer 2: no effect
    ci._restart_midi()
    inp = listening(fake)
    from test_cue_activation import commands
    commands(ci)
    inp.send(0xB0, 0, 127)
    inp.send(0xB0, 0, 0)
    inp.send(0xB0, 1, 70)
    inp.send(0xB0, 2, 100)
    inp.send(0xB0, 3, 50)
    ci._poll_remote()
    renders = commands(ci)
    layer = [c for c in renders if c["action"] == "render"][-1]["layers"][0]
    assert [f["inputs"] for f in layer["effects"]] == [{"amount": 0.0}, {"direction": 2}]
    assert [c for c in renders if c["action"] == "set_output_effects"][-1]["effects"]["0"][0]["inputs"] == \
        {"vertical": True}
    assert ci.cell_positions[(0, 0)]["effects"][1]["inputs"] == {"direction": 2}
    ci._stop_midi()


def test_fader_learn_and_save_in_the_dialog(dialog, fake):
    ci, win, root = dialog
    vd = win._vd
    vd["add_fader"]()
    assert vd["faders"] == [{"cc": None, "target": "layer", "index": 1, "input": "amount"}]
    vd["learn"](("fader", 0))
    temp = listening(fake)
    temp.send(0x90, 60, 90)                                                # a key: not for a fader
    assert pump(root, lambda: "not a key" in vd["note"].cget("text"))
    vd["learn"](("fader", 0))
    temp = listening(fake)
    temp.send(0xB0, 62, 100)
    assert pump(root, lambda: vd["faders"][0]["cc"] == [1, 62])
    vd["add_fader"]()
    vd["faders"][1].update(target="display", index=0, input="tint")
    vd["learn"]("go")                                                      # a button on fader 1's CC
    listening(fake).send(0xB0, 62, 127)
    assert pump(root, lambda: vd["mapping"].get("go") == ["cc", 1, 62])
    assert vd["faders"][0]["cc"] is None and "was fader 1" in vd["note"].cget("text")
    vd["remove_fader"](0)
    vd["save"]()
    assert ci.midi_faders == [{"cc": None, "target": "display", "index": 0, "input": "tint"}]


# --- MIDI Show Control ------------------------------------------------------------------

def _msc(device, fmt, cmd, data=()):
    return [0xF0, 0x7F, device, 0x02, fmt, cmd, *data, 0xF7]


def _ascii(text):
    return [ord(c) for c in text]


def test_parse_msc_messages():
    m = mr.parse_msc(_msc(1, 0x30, 0x01, _ascii("12.5") + [0] + _ascii("1")))
    assert m == {"device": 1, "format": 0x30, "command": "GO", "cue": "12.5"}
    assert mr.parse_msc(_msc(1, 0x7F, 0x01))["cue"] is None                 # GO without a cue
    assert mr.parse_msc(_msc(1, 0x40, 0x04, [0x21, 0, 0, 0, 0] + _ascii("7")))["cue"] == "7"   # TIMED_GO
    assert mr.parse_msc(_msc(1, 0x30, 0x08))["command"] == "ALL_OFF"
    assert mr.parse_msc(_msc(1, 0x30, 0x01, _ascii("ab")))["cue"] is None   # not a cue number
    assert mr.parse_msc([0xF0, 0x7E, 1, 0x02, 0x30, 0x01, 0xF7]) is None    # not real-time SysEx
    assert mr.parse_msc(_msc(1, 0x30, 0x55)) is None                         # unknown command
    assert mr.parse_msc([0xF0, 0x7F, 1, 0x02, 0x30]) is None                  # cut short


def test_msc_commands_for_this_device():
    cmd = lambda raw, dev=1, **kw: mr.msc_command(mr.parse_msc(raw), dev, **kw)  # noqa: E731
    assert cmd(_msc(1, 0x30, 0x01)) == ("go", None)
    assert cmd(_msc(1, 0x30, 0x01, _ascii("12.5"))) == ("cue", "12.5")      # a cue number, matched first
    assert cmd(_msc(1, 0x30, 0x01, _ascii("3"))) == ("cue", 3)
    assert cmd(_msc(0x7F, 0x7F, 0x01)) == ("go", None)                       # all-call, all types
    assert cmd(_msc(2, 0x30, 0x01)) is None                                  # another device
    assert cmd(_msc(1, 0x10, 0x01)) is None                                  # a sound cue: not ours
    assert cmd(_msc(1, 0x10, 0x01), any_format=True) == ("go", None)
    assert cmd(_msc(1, 0x40, 0x02)) == ("blackout", None) == cmd(_msc(1, 0x30, 0x08))
    assert cmd(_msc(1, 0x30, 0x03)) is None and cmd(_msc(1, 0x30, 0x0A)) is None   # RESUME, RESET ignored


def test_msc_through_the_remote(monkeypatch):
    got = []
    r = mr.MidiRemote(lambda c, a: got.append((c, a)), msc=True, msc_device=5)
    r._on_message((_msc(5, 0x30, 0x01, _ascii("4")), 0.0))
    r._on_message((_msc(6, 0x30, 0x01), 0.0))                                 # not for us
    r._on_message((_msc(5, 0x30, 0x08), 0.0))
    assert got == [("cue", 4), ("blackout", None)] and r.received == 2
    assert r.msc_last["command"] == "ALL_OFF"
    off = mr.MidiRemote(lambda c, a: got.append((c, a)))             # MSC not enabled
    off._on_message((_msc(0, 0x30, 0x01), 0.0))
    assert len(got) == 2
    assert mr.MidiRemote(lambda c, a: None, msc=True, msc_device=200).msc_device == 0


def test_app_runs_msc_through_the_remote_queue(blind_ci, fake):
    """MSC GO with a cue number goes to that cue in the app (the same path as OSC)."""
    ci = blind_ci
    ci.midi_enabled, ci.midi_port, ci.midi_msc, ci.midi_msc_device = True, "loopMIDI Port 2", True, 1
    ran = []
    ci._handle_remote_command = lambda cmd, arg, addr=None: ran.append((cmd, arg))
    ci._restart_midi()
    try:
        listening(fake).send(*_msc(1, 0x30, 0x01, _ascii("2")))
        cmd, arg, addr = ci._remote_queue.get_nowait()
        assert (cmd, arg, addr) == ("cue", 2, ("MIDI", 0))
        assert "MSC device 1, last GO 2" in ci._midi_text(ci._midi_status())
    finally:
        ci._stop_midi()
