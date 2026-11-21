# SPDX-License-Identifier: MPL-2.0
"""MIDI remote control.

Off by default (Preferences > Remote control > MIDI). VisualDeck listens on one
MIDI input port (a controller, a show-control computer through a MIDI
interface, or a virtual port such as loopMIDI on Windows).

What it understands:
- Learned triggers for GO, BACK, Blackout and Update Output: a Note On
  (velocity above 0) or a Control Change rising to 64 or more (a button press;
  a fader sweeping up fires once, not on every step).
- Program Change, if enabled: program 1 (sent as 0) goes to cue 1, and so on.
  Any channel.
- Faders for effect inputs: a Control Change set as a fader sends its every
  value (0-127) to an input of a shader effect on the output (a layer of the
  live cue, or a display): command "fader" with (fader index, value).
- MIDI Show Control (MSC), if enabled: the System Exclusive show-control
  messages lighting and sound desks and QLab send (see parse_msc).

A trigger is stored as [kind, channel, number] with kind "note" or "cc",
channel 1-16 and number 0-127, in ~/.visualdeck/config.json.

The MIDI library calls back on its own thread. Nothing here touches the UI:
commands go to `on_command`, which must only enqueue (see
ControlInterface._poll_remote).
"""
from __future__ import annotations

import logging
import re
import threading

log = logging.getLogger("visualdeck.midi")

ACTIONS = ("go", "back", "blackout", "update")
ACTION_LABELS = {"go": "GO", "back": "BACK", "blackout": "Blackout", "update": "Update Output"}
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def trigger_text(trig):
    """'Note C3 (60), ch 1' / 'CC 20, ch 10' / 'not set'. C3 = 60 (the Yamaha/Ableton naming)."""
    if not valid_trigger(trig):
        return "not set"
    kind, ch, num = trig
    if kind == "note":
        return f"Note {NOTE_NAMES[num % 12]}{num // 12 - 2} ({num}), ch {ch}"
    return f"CC {num}, ch {ch}"


def valid_trigger(trig):
    try:
        kind, ch, num = trig
    except (TypeError, ValueError):
        return False
    return (kind in ("note", "cc") and isinstance(ch, int) and isinstance(num, int)
            and not isinstance(ch, bool) and not isinstance(num, bool) and 1 <= ch <= 16 and 0 <= num <= 127)


MAX_FADERS = 8
FADER_TARGETS = ("layer", "display")


def clean_faders(raw):
    """[{'cc': [channel 1-16, number 0-127] or None (not learned yet), 'target':
    'layer' | 'display', 'index': layer 1.. | display 0.., 'input': name}], the
    valid ones (config files can be edited by hand), at most MAX_FADERS."""
    out = []
    for f in raw if isinstance(raw, list) else []:
        if not isinstance(f, dict):
            continue
        cc, target, idx, inp = f.get("cc"), f.get("target"), f.get("index"), f.get("input")
        if cc is not None and not valid_trigger(["cc"] + list(cc if isinstance(cc, (list, tuple)) else [None])):
            continue
        if target not in FADER_TARGETS or isinstance(idx, bool) or not isinstance(idx, int):
            continue
        if idx < (1 if target == "layer" else 0) or not isinstance(inp, str) or not inp.strip():
            continue
        out.append({"cc": [int(cc[0]), int(cc[1])] if cc is not None else None, "target": target,
                    "index": idx, "input": inp.strip()})
    return out[:MAX_FADERS]


def fader_text(f):
    """'Layer 2 › amount' / 'Display 0 › tint'."""
    return f"{'Layer' if f['target'] == 'layer' else 'Display'} {f['index']} \u203a {f['input']}"


def clean_mapping(raw):
    """Keep only known actions with valid triggers (config files can be edited by hand)."""
    out = {}
    if isinstance(raw, dict):
        for action, trig in raw.items():
            if action in ACTIONS and valid_trigger(trig):
                out[action] = [trig[0], int(trig[1]), int(trig[2])]
    return out


class Decoder:
    """Turns raw MIDI bytes into triggers and commands. Keeps the Control
    Change state needed for rising-edge detection; not thread-safe (one
    callback thread uses it)."""

    def __init__(self, mapping=None, program_change=True):
        self.mapping = clean_mapping(mapping or {})
        self.program_change = bool(program_change)
        self._cc = {}

    def trigger(self, msg):
        """('note'|'cc', channel 1-16, number) for a press, else None. Program
        Change returns ('pc', channel, program)."""
        if not msg:
            return None
        status = msg[0]
        if status < 0x80 or status >= 0xF0:
            return None                                   # running status / system messages
        kind, ch = status & 0xF0, (status & 0x0F) + 1
        if kind == 0x90 and len(msg) >= 3:
            return ("note", ch, msg[1] & 0x7F) if msg[2] > 0 else None
        if kind == 0xB0 and len(msg) >= 3:
            key = (ch, msg[1] & 0x7F)
            was, now = self._cc.get(key, 0), msg[2]
            self._cc[key] = now
            return ("cc", ch, msg[1] & 0x7F) if now >= 64 > was else None
        if kind == 0xC0 and len(msg) >= 2:
            return ("pc", ch, msg[1] & 0x7F)
        return None

    def command(self, trig):
        """(command, argument) for a trigger, or None."""
        if trig is None:
            return None
        if trig[0] == "pc":
            return ("cue", trig[2] + 1) if self.program_change else None
        for action, t in self.mapping.items():
            if (t[0], t[1], t[2]) == tuple(trig):
                return (action, None)
        return None


# --- MIDI Show Control ----------------------------------------------------------------
# A MSC message is F0 7F <device ID> 02 <command format> <command> <data> F7
# (MIDI 1.0 Recommended Practice RP-002). The device ID is 0-111 for one device,
# 112-126 for groups (not used here) and 127 for everyone. Data for GO is the
# cue number in ASCII ("12.5"), then optionally 00 and the cue list, 00 and the
# cue path; only the cue number is used.
MSC_ALL_CALL = 0x7F
MSC_ALL_TYPES = 0x7F
# Command formats VisualDeck answers to by default: all types, video and projection.
MSC_FORMATS = {MSC_ALL_TYPES: "All types", **{f: "Video" for f in range(0x30, 0x40)},
               **{f: "Projection" for f in range(0x40, 0x50)}}
MSC_COMMAND_NAMES = {0x01: "GO", 0x02: "STOP", 0x03: "RESUME", 0x04: "TIMED_GO", 0x05: "LOAD", 0x06: "SET",
                     0x07: "FIRE", 0x08: "ALL_OFF", 0x09: "RESTORE", 0x0A: "RESET", 0x0B: "GO_OFF"}


def parse_msc(msg):
    """{'device', 'format', 'command' (its name), 'cue' (str or None)} for a MIDI
    Show Control message (a list of bytes), or None if it is not one."""
    if len(msg) < 7 or msg[0] != 0xF0 or msg[1] != 0x7F or msg[3] != 0x02 or msg[-1] != 0xF7:
        return None
    device, fmt, cmd = msg[2], msg[4], msg[5]
    if cmd not in MSC_COMMAND_NAMES:
        return None
    data = list(msg[6:-1])
    if cmd == 0x04 and len(data) >= 5:          # TIMED_GO: the time (5 bytes) comes before the cue
        data = data[5:]
    cue = None
    if cmd in (0x01, 0x04, 0x05, 0x0B) and data:
        field = data[:data.index(0)] if 0 in data else data
        text = "".join(chr(b) for b in field if 0x20 <= b < 0x7F)
        if text and all(c.isdigit() or c == "." for c in text) and any(c.isdigit() for c in text):
            cue = text
    return {"device": device, "format": fmt, "command": MSC_COMMAND_NAMES[cmd], "cue": cue}


def msc_command(m, device_id, any_format=False):
    """(command, argument) for a parsed MSC message addressed to this device, or None.
    GO (and TIMED_GO) with a cue number goes to that cue, without one does GO;
    STOP and ALL_OFF fade to black (VisualDeck has no pause); the rest are ignored."""
    if m is None or m["device"] not in (device_id, MSC_ALL_CALL):
        return None
    if not any_format and m["format"] not in MSC_FORMATS:
        return None
    if m["command"] in ("GO", "TIMED_GO"):
        return ("cue", _cue_arg(m["cue"])) if m["cue"] else ("go", None)
    if m["command"] in ("STOP", "ALL_OFF"):
        return ("blackout", None)
    return None


def _cue_arg(text):
    import remote_control
    return remote_control.cue_arg(text)


def _default_backend():
    import rtmidi
    return rtmidi.MidiIn()


def list_ports(backend=None):
    """(port names, error or None)."""
    try:
        m = (backend or _default_backend)()
    except Exception as e:  # noqa: BLE001 - no MIDI system, missing driver, ...
        return [], f"no MIDI system: {e}"
    try:
        return list(m.get_ports()), None
    except Exception as e:  # noqa: BLE001
        return [], f"cannot list MIDI ports: {e}"
    finally:
        _close(m)


def _close(m):
    for name in ("cancel_callback", "close_port", "delete"):
        fn = getattr(m, name, None)
        if fn is not None:
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass


def _base(name):
    return re.sub(r"\s+\d+$", "", name or "").strip().lower()


def find_port(ports, wanted):
    """Index of `wanted` in `ports`: exact name, else the same name with a
    different trailing number (Windows numbers ports by plug-in order), else
    None. An empty `wanted` means the first port."""
    if not ports:
        return None
    if not wanted:
        return 0
    if wanted in ports:
        return ports.index(wanted)
    same = [i for i, p in enumerate(ports) if _base(p) == _base(wanted)]
    return same[0] if len(same) == 1 else None


class MidiRemote:
    def __init__(self, on_command, port_name="", mapping=None, program_change=True, backend=None,
                 timecode=False, faders=None, msc=False, msc_device=0):
        """on_command(command, argument) is called on the MIDI thread; it must only enqueue.
        timecode=True also reads MIDI timecode: the newest time is in
        self.timecode as (h, m, s, f, rate_code, monotonic seconds) for the UI to poll."""
        import timecode as _tc
        self.on_command = on_command
        self.read_timecode = bool(timecode)
        self.msc = bool(msc)                        # MIDI Show Control (parse_msc)
        self.msc_device = msc_device if isinstance(msc_device, int) and 0 <= msc_device <= 0x6F else 0
        self.msc_last = None                        # the last MSC message received (for the setup window)
        self._mtc = _tc.MTCDecoder()
        self.timecode = None
        self.port_name = port_name or ""
        self.decoder = Decoder(mapping, program_change)
        # (channel, number) of a Control Change -> fader index (clean_faders)
        self.faders = {tuple(f["cc"]): i for i, f in enumerate(clean_faders(faders)) if f["cc"] is not None}
        self._backend = backend or _default_backend
        self._midi = None
        self._lock = threading.Lock()
        self._learn = None
        self.opened_port = None
        self.error = None
        self.received = 0

    @property
    def listening(self):
        return self._midi is not None

    def start(self):
        """Open the port. Returns True, or False with self.error set."""
        try:
            m = self._backend()
        except Exception as e:  # noqa: BLE001
            self.error = f"no MIDI system: {e}"
            log.error(f"MIDI remote control: {self.error}")
            return False
        try:
            ports = list(m.get_ports())
            idx = find_port(ports, self.port_name)
            if idx is None:
                self.error = (f"MIDI input '{self.port_name}' not found" if ports else "no MIDI input ports") + \
                             (f" (available: {', '.join(ports)})" if ports else "")
                log.error(f"MIDI remote control: {self.error}")
                _close(m)
                return False
            m.open_port(idx)
            if (self.read_timecode or self.msc) and hasattr(m, 'ignore_types'):
                # MTC is "timing" + SysEx; MSC is SysEx
                m.ignore_types(sysex=False, timing=not self.read_timecode, active_sense=True)
            m.set_callback(self._on_message)
        except Exception as e:  # noqa: BLE001 - e.g. the port is in use by another program (Windows)
            self.error = f"cannot open MIDI input: {e}"
            log.error(f"MIDI remote control: {self.error}")
            _close(m)
            return False
        self._midi, self.opened_port, self.error = m, ports[idx], None
        log.info(f"MIDI remote control listening on '{ports[idx]}'")
        return True

    def stop(self):
        m, self._midi = self._midi, None
        if m is not None:
            _close(m)
        self.opened_port = None

    def learn(self, callback):
        """The next press (note or CC; not Program Change) goes to callback(trigger)
        instead of running a command. callback runs on the MIDI thread."""
        with self._lock:
            self._learn = callback

    def cancel_learn(self):
        with self._lock:
            self._learn = None

    def _on_message(self, event, data=None):
        try:
            msg = event[0] if isinstance(event, tuple) else event
            if msg and msg[0] == 0xF0 and self.msc and len(msg) > 4 and msg[1] == 0x7F and msg[3] == 0x02:
                m = parse_msc(list(msg))
                if m is not None:
                    self.msc_last = m
                    cmd = msc_command(m, self.msc_device)
                    log.info(f"MSC {m['command']}" + (f" cue {m['cue']}" if m['cue'] else "")
                             + f" (device {m['device']}, format {m['format']:#04x})"
                             + ("" if cmd else ": not for VisualDeck, ignored"))
                    if cmd is not None:
                        self.received += 1
                        self.on_command(cmd[0], cmd[1])
                return
            if msg and msg[0] >= 0xF0:                       # system messages: MTC (and MSC above)
                if self.read_timecode:
                    tc = self._mtc.feed(list(msg))
                    if tc is not None:
                        import time as _time
                        self.timecode = tuple(tc) + (_time.monotonic(),)
                return
            if self.faders and len(msg) >= 3 and msg[0] & 0xF0 == 0xB0 and self._learn is None:
                i = self.faders.get(((msg[0] & 0x0F) + 1, msg[1] & 0x7F))
                if i is not None:                         # a fader: every value, not a press
                    self.received += 1
                    self.on_command("fader", (i, msg[2] & 0x7F))
                    return
            trig = self.decoder.trigger(list(msg))
            if trig is None:
                return
            with self._lock:
                learn = self._learn
                if learn is not None and trig[0] != "pc":
                    self._learn = None
            if learn is not None and trig[0] != "pc":
                learn([trig[0], trig[1], trig[2]])
                return
            cmd = self.decoder.command(trig)
            if cmd is None:
                return
            self.received += 1
            self.on_command(cmd[0], cmd[1])
        except Exception:  # noqa: BLE001 - never let a handler kill the MIDI thread
            log.exception("MIDI message could not be handled")
