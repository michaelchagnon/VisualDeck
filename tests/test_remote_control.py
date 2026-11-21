# SPDX-License-Identifier: MPL-2.0
"""OSC remote control over real UDP on the loopback interface."""
import queue
import socket
import time

import pytest
from pythonosc.osc_message import OscMessage
from pythonosc.udp_client import SimpleUDPClient

import health
import remote_control as rc
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci, commands  # noqa: F401


def test_parse_accepts_only_well_formed_commands():
    p = rc.parse
    assert p("/visualdeck/go", ()) == ("go", None)
    assert p("/VisualDeck/Blackout/", ()) == ("blackout", None)
    assert p("/visualdeck/cue", (3,)) == ("cue", 3)
    assert p("/visualdeck/cue", (3.0,)) == ("cue", 3)
    assert p("/visualdeck/cue", ("12",)) == ("cue", 12)
    assert p("/visualdeck/cue", (" Intro ",)) == ("cue", "Intro")
    for bad in (("/visualdeck/cue", ()), ("/visualdeck/cue", (0,)), ("/visualdeck/cue", (-2.5,)),
                ("/visualdeck/cue", (True,)), ("/visualdeck/cue", (1, 2)), ("/visualdeck/delete", ()),
                ("/other/go", ()), ("/visualdeckgo", ())):
        assert p(*bad) is None, bad


@pytest.fixture
def server():
    q = queue.Queue()
    srv = rc.OSCRemote(lambda c, a, addr: q.put((c, a, addr)), "127.0.0.1", 0)   # 0: any free port
    assert srv.start()
    yield srv, q
    srv.stop()


def recv(q, n, timeout=2.0):
    out, end = [], time.monotonic() + timeout
    while len(out) < n and time.monotonic() < end:
        try:
            out.append(q.get(timeout=0.05))
        except queue.Empty:
            pass
    return out


def test_messages_arrive_as_queued_commands(server):
    srv, q = server
    host, port = srv.address()
    client = SimpleUDPClient(host, port)
    client.send_message("/visualdeck/go", [])
    client.send_message("/visualdeck/nonsense", [])          # ignored, counted as nothing
    client.send_message("/visualdeck/cue", ["Intro"])
    client.send_message("/visualdeck/cue", [2])
    got = recv(q, 3)
    assert [(c, a) for c, a, _ in got] == [("go", None), ("cue", "Intro"), ("cue", 2)]
    assert srv.received == 3 and got[0][2][0] == "127.0.0.1"


def test_reply_reaches_the_sender():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2)
    rc.send_reply(sock.getsockname(), "/visualdeck/pong", [3, "Finale"])
    msg = OscMessage(sock.recv(1024))
    sock.close()
    assert msg.address == "/visualdeck/pong" and msg.params == [3, "Finale"]


def test_a_busy_port_is_reported_not_fatal(server):
    srv, _ = server
    second = rc.OSCRemote(lambda *a: None, "127.0.0.1", srv.address()[1])
    assert second.start() is False and "cannot listen" in second.error and not second.listening
    second.stop()
    snap = {"remote": {"enabled": True, "listening": False, "error": second.error}}
    assert "OSC remote control not listening" in health.assess(snap)[1][0]


def test_stop_releases_the_port():
    srv = rc.OSCRemote(lambda *a: None, "127.0.0.1", 0)
    assert srv.start()
    port = srv.address()[1]
    srv.stop()
    again = rc.OSCRemote(lambda *a: None, "127.0.0.1", port)
    assert again.start()
    again.stop()


@pytest.fixture
def ci(blind_ci):
    calls = []
    for name in ("_trigger_go", "_go_back", "_blackout", "_apply_live_edits"):
        setattr(blind_ci, name, (lambda n: lambda *a, **k: calls.append(n))(name))
    blind_ci.calls = calls
    return blind_ci


def test_commands_use_the_same_paths_as_the_buttons(ci):
    for cmd in ("go", "back", "blackout", "update"):
        ci._handle_remote_command(cmd, None)
    assert ci.calls == ["_trigger_go", "_go_back", "_blackout", "_apply_live_edits"]


def test_cue_by_number_and_by_name(ci):
    ci.cue_names["2"] = "Finale"
    ci._handle_remote_command("cue", "finale")
    assert ci.transition_thread.started[-1][3] == 2
    ci._finalize_transition_from_thread(2)
    ci._handle_remote_command("cue", 2)
    assert ci.transition_thread.started[-1][3] == 1
    n = len(ci.transition_thread.started)
    ci._handle_remote_command("cue", 99)            # no such cue
    ci._handle_remote_command("cue", "nobody")
    assert len(ci.transition_thread.started) == n


def test_commands_are_ignored_until_the_output_is_ready(ci):
    ci._ui_ready = False
    ci._handle_remote_command("go", None)
    assert ci.calls == []


def test_ping_replies_with_the_live_cue(ci):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2)
    ci.cue_names["0"] = "Opening"
    ci._handle_remote_command("ping", None, sock.getsockname())
    msg = OscMessage(sock.recv(1024))
    sock.close()
    assert msg.params == [1, "Opening"]


def test_a_flood_is_handled_a_bounded_amount_per_tick(ci):
    for _ in range(50):
        ci._remote_queue.put(("update", None, None))
    ci._poll_remote()
    assert len(ci.calls) == 20 and ci._remote_queue.qsize() == 30


def test_osc_settings_persist_and_are_validated(tmp_path, monkeypatch):
    import json
    import control_interface as ci_mod
    from media_manager import MediaManager
    monkeypatch.setattr(ci_mod.ControlInterface, "CONFIG_DIR", str(tmp_path / ".vd"))
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    a = ci_mod.ControlInterface(MediaManager())
    assert (a.osc_enabled, a.osc_port, a.osc_network) == (False, 7700, False)
    a.osc_enabled, a.osc_port, a.osc_network = True, 9001, True
    a._save_config()
    b = ci_mod.ControlInterface(MediaManager())
    b._load_config()
    assert (b.osc_enabled, b.osc_port, b.osc_network) == (True, 9001, True)
    cfg = json.load(open(b.config_path))
    cfg["osc_port"] = 70000
    json.dump(cfg, open(b.config_path, "w"))
    c = ci_mod.ControlInterface(MediaManager())
    c._load_config()
    assert c.osc_port == 7700
    for o in (a, b, c):
        o.proxy_executor.shutdown(wait=False)


def test_parse_effect_inputs():
    p = rc.parse
    assert p("/visualdeck/effect", (2, "amount", 0.5)) == ("effect", (2, "amount", [0.5]))
    assert p("/visualdeck/effect", (1.0, " Amount ", 1)) == ("effect", (1, "Amount", [1.0]))
    assert p("/visualdeck/effect", (1, "vertical", True)) == ("effect", (1, "vertical", [True]))
    assert p("/visualdeck/display_effect", (0, "tint", 1, 0.5, 0)) == ("display_effect", (0, "tint", [1.0, 0.5, 0.0]))
    for bad in (("/visualdeck/effect", (0, "amount", 0.5)),          # layers count from 1
                ("/visualdeck/effect", (1.5, "amount", 0.5)), ("/visualdeck/effect", (True, "amount", 1)),
                ("/visualdeck/effect", (1, "", 0.5)), ("/visualdeck/effect", (1, "amount")),
                ("/visualdeck/effect", (1, "amount", "x")), ("/visualdeck/effect", (1, "amount", float("nan"))),
                ("/visualdeck/effect", (1, "c", 1, 2, 3, 4, 5)), ("/visualdeck/display_effect", (-1, "a", 1))):
        assert p(*bad) is None, bad


def effect_cell(ci, effect):
    ci.cell_positions[(0, 0)] = {"x": 0.5, "y": 0.5, "scale_x": 1, "scale_y": 1, "rotation": 0}
    if effect is not None:
        ci.cell_positions[(0, 0)]["effect"] = effect           # as older shows saved one effect


def test_osc_sets_an_effect_input_on_the_output_once_per_tick(ci):
    """Blind editing: an effect input from a remote control changes the output
    at once (it is playing the show), other pending edits stay pending, and a
    burst of messages is one render command."""
    effect_cell(ci, {"name": "Invert.fs", "inputs": {"amount": 1.0}})
    ci._send_program(ci._get_layers_for_col(0))               # Update Output: the effect is on it
    commands(ci)
    ci.cell_positions[(0, 0)]["x"] = 0.25                    # a blind edit, not applied
    for v in (0.1, 0.2, 0.3):
        ci._remote_queue.put(("effect", (1, "amount", [v]), None))
    ci._poll_remote()
    sent = commands(ci, "render")
    assert len(sent) == 1
    layer = sent[0]["layers"][0]
    assert layer["effects"] == [{"name": "Invert.fs", "inputs": {"amount": 0.3}}]
    assert layer["position"]["x"] == 0.5                      # the blind edit is still waiting
    assert ci.cell_positions[(0, 0)]["effects"][0]["inputs"]["amount"] == 0.3     # and the show has it
    assert "effect" not in ci.cell_positions[(0, 0)]          # an older single effect becomes the list
    assert ci._live_edits_pending()
    # By label, and a list input only takes its values
    effect_cell(ci, {"name": "Mirror.fs", "inputs": {}})
    ci._send_program(ci._get_layers_for_col(0))
    commands(ci)
    ci._handle_remote_command("effect", (1, "Keep", [2.0]))
    ci._handle_remote_command("effect", (1, "keep", [9.0]))      # not one of its values: ignored
    ci._flush_remote_effects()
    assert commands(ci, "render")[-1]["layers"][0]["effects"][0]["inputs"] == {"direction": 2}
    # A stack: "2:amount" is the second effect's input
    effect_cell(ci, None)
    ci.cell_positions[(0, 0)]["effects"] = [{"name": "Invert.fs", "inputs": {}}, {"name": "Invert.fs", "inputs": {}}]
    ci._send_program(ci._get_layers_for_col(0))
    commands(ci)
    ci._handle_remote_command("effect", (1, "2:amount", [0.25]))
    ci._flush_remote_effects()
    assert [f["inputs"] for f in commands(ci, "render")[-1]["layers"][0]["effects"]] == [{}, {"amount": 0.25}]


def test_osc_effect_input_errors_change_nothing(ci):
    effect_cell(ci, {"name": "Invert.fs", "inputs": {}})
    ci._send_program(ci._get_layers_for_col(0))
    commands(ci)
    assert ci._remote_layer_effect(1, "amount", [0.5]) is False          # layer 2 has no effect
    assert ci._remote_layer_effect(0, "nope", [0.5]) is False            # no such input
    assert ci._remote_layer_effect(0, "amount", [0.5, 1.0]) is False     # one number for a float
    ci._flush_remote_effects()
    assert commands(ci, "render") == []


def test_osc_sets_a_display_effect_input(ci):
    ci._set_output_effect(0, {"name": "Tint.fs", "inputs": {}})
    commands(ci)
    ci._handle_remote_command("display_effect", (0, "tint", [0.0, 1.0, 0.0]))
    ci._handle_remote_command("display_effect", (0, "amount", [0.5]))
    ci._handle_remote_command("display_effect", (3, "amount", [0.5]))          # no effect there: ignored
    ci._flush_remote_effects()
    sent = commands(ci, "set_output_effects")
    assert len(sent) == 1 and sent[0]["effects"]["0"][0]["inputs"] == {"tint": [0.0, 1.0, 0.0, 1.0], "amount": 0.5}
    assert ci.output_effects["0"][0]["inputs"]["amount"] == 0.5


def test_effect_messages_over_udp(server):
    srv, q = server
    host, port = srv.address()
    c = SimpleUDPClient(host, port)
    c.send_message("/visualdeck/effect", [1, "amount", 0.25])
    c.send_message("/visualdeck/display_effect", [0, "tint", 1.0, 0.0, 0.0])
    got = recv(q, 2)
    assert [(g[0], g[1]) for g in got] == [("effect", (1, "amount", [0.25])),
                                           ("display_effect", (0, "tint", [1.0, 0.0, 0.0]))]
