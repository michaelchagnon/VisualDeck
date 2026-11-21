# SPDX-License-Identifier: MPL-2.0
"""Browser remote over real HTTP on loopback, including its safety rules."""
import http.client
import json
import queue

import pytest

import web_remote as wr
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401


@pytest.fixture
def server():
    q = queue.Queue()
    status = {"live": {"number": 1, "name": "A"}, "cues": []}
    srv = wr.WebRemote(lambda c, a, addr: q.put((c, a)), lambda: status, "127.0.0.1", 0)
    assert srv.start()
    yield srv, q, status
    srv.stop()


def request(srv, method, path, headers=None, host=None):
    h, p = srv.address()
    conn = http.client.HTTPConnection(h, p, timeout=3)
    hdrs = dict(headers or {})
    conn.putrequest(method, path, skip_host=True)
    conn.putheader("Host", host if host is not None else f"{h}:{p}")
    for k, v in hdrs.items():
        conn.putheader(k, v)
    conn.endheaders()
    r = conn.getresponse()
    body = r.read()
    conn.close()
    return r.status, body


def test_page_and_status(server):
    srv, _, status = server
    code, body = request(srv, "GET", "/")
    assert code == 200 and b"<button id=\"go\">GO</button>" in body and b"X-VisualDeck" in body
    code, body = request(srv, "GET", "/api/status")
    assert code == 200 and json.loads(body) == status
    assert request(srv, "GET", "/nope")[0] == 404


def test_commands_need_post_and_the_header(server):
    srv, q, _ = server
    assert request(srv, "POST", "/api/go")[0] == 403                       # e.g. a form on another website
    assert request(srv, "GET", "/api/go")[0] == 404
    assert request(srv, "OPTIONS", "/api/go")[0] == 403                    # cross-origin pre-flight refused
    assert q.empty()
    H = {"X-VisualDeck": "1"}
    for path in ("/api/go", "/api/back", "/api/blackout", "/api/update", "/api/cue/3"):
        assert request(srv, "POST", path, H)[0] == 202
    assert [q.get_nowait() for _ in range(5)] == [("go", None), ("back", None), ("blackout", None),
                                                 ("update", None), ("cue_at", 3)]
    for bad in ("/api/cue/0", "/api/cue/x", "/api/cue/", "/api/delete"):
        assert request(srv, "POST", bad, H)[0] == 404


def test_dns_rebinding_hosts_are_refused(server):
    srv, q, _ = server
    for host in ("evil.example.com", "evil.example.com:7701", ""):
        assert request(srv, "GET", "/api/status", host=host)[0] == 403, host
        assert request(srv, "POST", "/api/go", {"X-VisualDeck": "1"}, host=host)[0] == 403
    assert q.empty()
    assert request(srv, "GET", "/api/status", host="localhost:7701")[0] == 200


@pytest.mark.parametrize("host,ok", [("127.0.0.1:80", True), ("[::1]:7701", True), ("192.168.1.20", True),
                                     ("LOCALHOST", True), ("example.com", False), ("127.0.0.1.evil.com", False)])
def test_host_check(host, ok):
    assert wr.host_allowed(host) is ok


def test_busy_port_is_reported(server):
    srv, _, _ = server
    other = wr.WebRemote(lambda *a: None, dict, "127.0.0.1", srv.address()[1])
    assert other.start() is False and "cannot listen" in other.error


def test_status_snapshot_from_the_controller(blind_ci):
    ci = blind_ci
    ci.cue_names["1"] = "Song"
    ci.cue_settings = {"1": {"notes": "slow"}}
    st = ci._build_web_status()
    assert st["live"] == {"number": 1, "name": "Cue 1", "cue_number": None} and not st["black"] and not st["edits_pending"]
    assert st["cues"][1] == {"number": 2, "name": "Song", "has_media": True, "notes": "slow", "cue_number": None}
    assert [c["has_media"] for c in st["cues"][:4]] == [True, True, True, False]
    json.dumps(st)
    ci.transition_var.set("None")
    ci._blackout()
    st = ci._build_web_status()
    assert st["black"] and st["live"] is None


def test_web_settings_persist_and_are_validated(tmp_path, monkeypatch):
    import control_interface as ci_mod
    from media_manager import MediaManager
    monkeypatch.setattr(ci_mod.ControlInterface, "CONFIG_DIR", str(tmp_path / ".vd"))
    monkeypatch.setattr(ci_mod.ControlInterface, "_start_gui", lambda self: None)
    a = ci_mod.ControlInterface(MediaManager())
    assert (a.web_enabled, a.web_port, a.web_network) == (False, 7701, False)
    a.web_enabled, a.web_port, a.web_network = True, 8088, True
    a._save_config()
    b = ci_mod.ControlInterface(MediaManager())
    b._load_config()
    assert (b.web_enabled, b.web_port, b.web_network) == (True, 8088, True)
    for o in (a, b):
        o.proxy_executor.shutdown(wait=False)


def test_the_page_starts_a_cue_from_a_time_once():
    """In a real browser (Chromium through Playwright, when installed): with a
    time typed in "Start a cue from", tapping a cue sends /from/<time>; the
    field is cleared, so the next tap is a plain cue again."""
    pw = pytest.importorskip("playwright.sync_api")
    import queue
    q = queue.Queue()
    status = {"live": {"number": 1, "name": "A"}, "black": False, "edits_pending": False,
              "cues": [{"number": 1, "name": "A", "has_media": True}, {"number": 2, "name": "B", "has_media": True}]}
    srv = wr.WebRemote(lambda c, a, addr: q.put((c, a)), lambda: status, "127.0.0.1", 0)
    assert srv.start()
    try:
        with pw.sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as e:  # noqa: BLE001
                pytest.skip(f"no browser: {e}")
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{srv.address()[1]}/")
            page.wait_for_selector("li")
            page.fill("#from", "1:30")
            page.click("li:nth-child(2)")
            assert q.get(timeout=5) == ("start_from_at", (2, 90.0))
            assert page.input_value("#from") == ""
            page.click("li:nth-child(2)")
            assert q.get(timeout=5) == ("cue_at", 2)
            browser.close()
    finally:
        srv.stop()


def test_effect_inputs_over_http(server):
    srv, q, _ = server
    H = {"X-VisualDeck": "1"}
    assert request(srv, "POST", "/api/effect/2/amount/0.5", H)[0] == 202
    assert request(srv, "POST", "/api/display_effect/0/tint/1,0.5,0", H)[0] == 202
    assert request(srv, "POST", "/api/effect/1/RGB%20shift/0.01", H)[0] == 202
    assert [q.get_nowait() for _ in range(3)] == [("effect", (2, "amount", [0.5])),
                                                 ("display_effect", (0, "tint", [1.0, 0.5, 0.0])),
                                                 ("effect", (1, "RGB shift", [0.01]))]
    for bad in ("/api/effect/0/amount/1", "/api/effect/1/amount", "/api/effect/1/amount/x",
                "/api/effect/x/amount/1", "/api/display_effect/-1/a/1", "/api/effect/1/amount/1/2"):
        assert request(srv, "POST", bad, H)[0] == 404, bad
    assert request(srv, "POST", "/api/effect/1/amount/0.5")[0] == 403        # the header rule holds
    assert q.empty()


def test_effect_controls_on_the_page():
    """In a real browser (Chromium through Playwright, when installed): the
    effects on the output get controls; moving one sends its input (by its
    effect's position, so a stack works), and values VisualDeck reports
    later show up."""
    pw = pytest.importorskip("playwright.sync_api")
    import queue
    import time
    q = queue.Queue()
    fx = [{"kind": "layer", "index": 2, "label": "Band", "position": 1, "title": "Invert",
           "inputs": [{"name": "amount", "label": "Amount", "type": "float", "min": 0.0, "max": 1.0, "value": 1.0}]},
          {"kind": "layer", "index": 2, "label": "Band", "position": 2, "title": "Mirror",
           "inputs": [{"name": "direction", "label": "Keep", "type": "long", "values": [0, 1, 2, 3],
                       "labels": ["Left half", "Right half", "Top half", "Bottom half"], "value": 0}]},
          {"kind": "display", "index": 0, "label": "Display 0", "position": 1, "title": "Wave",
           "inputs": [{"name": "vertical", "label": "Up and down", "type": "bool", "value": False}]}]
    status = {"live": {"number": 1, "name": "A"}, "black": False, "edits_pending": False,
              "cues": [{"number": 1, "name": "A", "has_media": True}], "effects": fx}
    srv = wr.WebRemote(lambda c, a, addr: q.put((c, a)), lambda: status, "127.0.0.1", 0)
    assert srv.start()
    try:
        with pw.sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as e:  # noqa: BLE001
                pytest.skip(f"no browser: {e}")
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{srv.address()[1]}/")
            page.wait_for_selector("#fxbox:not([hidden])")
            page.click("summary")
            assert page.inner_text("#fx").count("Band ›") == 2 and "Display 0 › Wave" in page.inner_text("#fx")
            page.eval_on_selector('input[type=range]', 'el => { el.value = 0.5; el.dispatchEvent(new Event("input")); }')
            assert q.get(timeout=5) == ("effect", (2, "1:amount", [0.5]))
            page.select_option("select", "2")
            assert q.get(timeout=5) == ("effect", (2, "2:direction", [2.0]))
            page.check("input[type=checkbox]")
            assert q.get(timeout=5) == ("display_effect", (0, "1:vertical", [1.0]))
            fx[0]["inputs"][0]["value"] = 0.25                   # changed elsewhere (MIDI, another remote)
            time.sleep(1.6)
            assert abs(float(page.input_value('input[type=range]')) - 0.25) < 0.01
            status["effects"] = []                               # nothing with an effect on the output
            page.wait_for_selector("#fxbox[hidden]", state="attached", timeout=3000)
            browser.close()
    finally:
        srv.stop()


def test_status_lists_the_effects_on_the_output(blind_ci):
    ci = blind_ci
    ci.cell_positions[(0, 0)] = {"x": 0.5, "y": 0.5, "effects": [
        {"name": "Invert.fs", "inputs": {"amount": 0.25}}, {"name": "Mirror.fs", "inputs": {}},
        {"name": "Tint.fs", "inputs": {}}]}
    ci.layer_names["0"] = "Band"
    ci._send_program(ci._get_layers_for_col(0))
    ci._set_output_effect(0, [{"name": "Wave.fs", "inputs": {"vertical": True}}])
    fx = ci._build_web_status()["effects"]
    assert [(g["kind"], g["index"], g["label"], g["position"], g["title"]) for g in fx] == [
        ("layer", 1, "Band", 1, "Invert"), ("layer", 1, "Band", 2, "Mirror"), ("layer", 1, "Band", 3, "Tint"),
        ("display", 0, "Display 0", 1, "Wave")]
    assert fx[0]["inputs"] == [{"name": "amount", "label": "Amount", "type": "float", "value": 0.25,
                                "min": 0.0, "max": 1.0}]
    keep = fx[1]["inputs"][0]
    assert keep["values"] == [0, 1, 2, 3] and keep["labels"][2] == "Top half" and keep["value"] == 0
    assert [i["name"] for i in fx[2]["inputs"]] == ["amount"], "a color is not a page control"
    assert {"name": "vertical", "label": "Up and down", "type": "bool", "value": True} in fx[3]["inputs"]
    import json
    json.dumps(fx)                                       # the server sends it as JSON
    ci._blackout()
