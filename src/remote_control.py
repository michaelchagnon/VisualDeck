# SPDX-License-Identifier: MPL-2.0
"""OSC remote control.

Off by default (Preferences > Remote control). When on, VisualDeck listens for
OSC over UDP, by default only from this computer (127.0.0.1); "accept from the
network" listens on all interfaces, e.g. for a tablet or a show-control
computer.

Addresses (arguments in brackets):
  /visualdeck/go                 GO (same as the GO button / space bar)
  /visualdeck/back               BACK
  /visualdeck/blackout           fade to black
  /visualdeck/update             Update Output (apply edits to the live cue)
  /visualdeck/cue  [n | "name"]  go to cue number n (1 = first) or the cue with that name
  /visualdeck/start_from [cue, seconds]
                                 Start Cue From: cut to that cue (as for /cue) with its videos
                                 that many seconds in (a number, or "m:ss" as a string)
  /visualdeck/effect [layer, input, value...]
                                 set an input of the shader effect on that layer (1 = first)
                                 of the cue on the output: one number (float, whole number,
                                 on/off), three or four for a color (0-1), two for a point.
                                 The input by its name in the effect file or its label.
  /visualdeck/display_effect [display, input, value...]
                                 the same for a display's effect (display as in the app: 0 = first)
  /visualdeck/ping               replies /visualdeck/pong [live cue number, live cue name]
                                 to the sender's address and port

The server thread never touches the UI: each command is put on a queue that
the UI thread drains (see ControlInterface._poll_remote).
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("visualdeck.remote")

PREFIX = "/visualdeck"
COMMANDS = ("go", "back", "blackout", "update", "cue", "start_from", "ping", "effect", "display_effect")
EFFECT_COMMANDS = ("effect", "display_effect")

# Every OSC command, for Help (help_content) and checked against COMMANDS and
# docs/REMOTE_CONTROL.md by the tests: (address, arguments, what it does, example).
OSC_REFERENCE = [
    ("/visualdeck/go", "none",
     "GO, as the GO button or the space bar: the next cue with its transition. With Presentation Mode "
     "enabled it starts the automatic run, or stops it while it runs.",
     "/visualdeck/go"),
    ("/visualdeck/back", "none", "BACK: the previous cue that has media.", "/visualdeck/back"),
    ("/visualdeck/blackout", "none", "Fade the output to black with the show's transition.",
     "/visualdeck/blackout"),
    ("/visualdeck/update", "none",
     "Update Output: put edits waiting on the live cue on the output (blind editing).", "/visualdeck/update"),
    ("/visualdeck/cue", "cue number, position or name",
     "Go to a cue with its transition. A cue number set in Cue Settings (12 or 12.5, as a number or a "
     "string) is looked up first, then the position (1 = first cue), then the name. Sending the live cue "
     "triggers it again. Ignored during a transition.",
     "/visualdeck/cue 3   or   /visualdeck/cue \"Act 2\""),
    ("/visualdeck/start_from", "cue, seconds",
     "Start Cue From (rehearsals): cut to the cue (as for /cue) with its videos and their sound that many "
     "seconds in, as Show \u203a Start Cue From\u2026 does. Seconds as a number or \"m:ss\".",
     "/visualdeck/start_from 4 \"1:30\""),
    ("/visualdeck/effect", "layer, input, value(s)",
     "Set an input of the shader effect on a layer of the cue on the output (layer 1 = first). The input by "
     "its name in the effect file or its label in the editor (any case); with several effects on the layer, "
     "2:amount is the second effect's and Invert:amount the first Invert's. One number for a slider, a "
     "whole number or an on/off (not 0 = on); three or four numbers 0\u20131 for a color; two for a point. "
     "Kept within the input's range. Changes the output at once, also with blind editing, and the cue in "
     "the show.",
     "/visualdeck/effect 1 amount 0.5"),
    ("/visualdeck/display_effect", "display, input, value(s)",
     "The same for a display's effect (Window \u203a Output Effects; display 0 = first). Saved with the show.",
     "/visualdeck/display_effect 0 tint 1 0.5 0"),
    ("/visualdeck/ping", "none",
     "Replies /visualdeck/pong with the live cue's number and name to the sender's address and port: "
     "a show-control system can check that VisualDeck is running.",
     "/visualdeck/ping"),
]


def parse(address, args):
    """(command, argument) for a message, or None if it is not ours / malformed."""
    if not address.lower().startswith(PREFIX + "/"):
        return None
    cmd = address[len(PREFIX) + 1:].strip("/").lower()
    if cmd not in COMMANDS:
        return None
    if cmd in EFFECT_COMMANDS:
        return effect_args(cmd, args)
    if cmd == "start_from":
        if len(args) != 2:
            return None
        cue, t = cue_arg(args[0]), seconds_arg(args[1])
        return None if cue is None or t is None else ("start_from", (cue, t))
    if cmd != "cue":
        return (cmd, None)
    if len(args) != 1:
        return None
    a = cue_arg(args[0])
    return None if a is None else ("cue", a)


def cue_arg(a):
    """A cue as sent: position n (int, 1 = first), a cue number ("12.5") or a name; None if malformed."""
    if isinstance(a, bool):
        return None
    if isinstance(a, (int, float)):
        if a != a or a < 0:
            return None
        if a != int(a):
            return repr(float(a))                 # 12.5 -> "12.5", matched against cue numbers
        if a < 1:
            return None
        return int(a)
    if isinstance(a, str) and a.strip():
        s = a.strip()
        return int(s) if s.isdigit() and int(s) >= 1 else s
    return None


def effect_args(cmd, args):
    """(cmd, (layer 1.. or display 0.., input name, [numbers])) or None if malformed."""
    if len(args) < 3 or len(args) > 6:
        return None
    n, name, values = args[0], args[1], list(args[2:])
    if isinstance(n, str) and n.strip().isdigit():
        n = int(n.strip())
    if isinstance(n, bool) or not isinstance(n, (int, float)) or n != n or n != int(n):
        return None
    n = int(n)
    if n < (1 if cmd == "effect" else 0):
        return None
    if not isinstance(name, str) or not name.strip():
        return None
    out = []
    for v in values:
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                return None
        if not isinstance(v, (int, float)) or (isinstance(v, float) and v != v):
            return None
        out.append(v if isinstance(v, bool) else float(v))
    return (cmd, (n, name.strip(), out))


def seconds_arg(a):
    """Seconds >= 0 from a number, "12.5" or "1:02.5"; None if malformed."""
    if isinstance(a, bool):
        return None
    try:
        if isinstance(a, str):
            t = a.strip()
            if ":" in t:
                m, sec = t.split(":", 1)
                v = int(m) * 60 + float(sec)
            else:
                v = float(t)
        else:
            v = float(a)
    except (TypeError, ValueError):
        return None
    return v if v == v and 0 <= v < 360000 else None


class OSCRemote:
    def __init__(self, on_command, host="127.0.0.1", port=7700):
        """on_command(command, argument, reply_address) is called on the server thread;
        it must only enqueue."""
        self.on_command = on_command
        self.host, self.port = host, int(port)
        self._server = None
        self._thread = None
        self.error = None
        self.received = 0

    @property
    def listening(self):
        return self._server is not None

    def address(self):
        if self._server is None:
            return None
        h, p = self._server.server_address[:2]
        return h, p

    def start(self):
        """Start listening. Returns True, or False with self.error set (e.g. port in use)."""
        from pythonosc.dispatcher import Dispatcher
        from pythonosc.osc_server import BlockingOSCUDPServer

        disp = Dispatcher()
        disp.set_default_handler(self._handle, needs_reply_address=True)
        try:
            self._server = BlockingOSCUDPServer((self.host, self.port), disp)
        except OSError as e:
            self.error = f"cannot listen on {self.host}:{self.port}: {e}"
            log.error(f"OSC remote control: {self.error}")
            self._server = None
            return False
        self.error = None
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.2},
                                         name="OSCRemote", daemon=True)
        self._thread.start()
        h, p = self.address()
        log.info(f"OSC remote control listening on {h}:{p}")
        return True

    def stop(self):
        srv, self._server = self._server, None
        if srv is not None:
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:  # noqa: BLE001
                log.exception("Stopping the OSC server failed")
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _handle(self, client_address, address, *args):
        parsed = parse(address, args)
        if parsed is None:
            log.warning(f"OSC: ignored {address} {list(args)} from {client_address[0]}")
            return
        self.received += 1
        try:
            self.on_command(parsed[0], parsed[1], client_address)
        except Exception:  # noqa: BLE001 - never let a handler kill the server thread
            log.exception("OSC command could not be queued")


def send_reply(address, osc_address, values):
    """Send one OSC message to (host, port)."""
    from pythonosc.udp_client import SimpleUDPClient
    SimpleUDPClient(address[0], address[1]).send_message(osc_address, values)
