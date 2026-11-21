# SPDX-License-Identifier: MPL-2.0
"""Browser remote: a control page for phones, tablets and other computers.

Off by default (Preferences > Remote control). When on, VisualDeck serves:

  GET  /              the control page (GO, BACK, Blackout, Update Output, cue list)
  GET  /api/status    JSON: live cue, cue list, blackout, pending edits, follow, cue clock
  POST /api/go | /api/back | /api/blackout | /api/update
  POST /api/cue/<n>   go to cue n (1 = first)
  POST /api/cue/<n>/from/<seconds>   Start Cue From: cue n (1 = first), that many seconds in
  POST /api/effect/<layer>/<input>/<value>          an input of the effect on that layer (1 = first)
  POST /api/display_effect/<display>/<input>/<value> the same for a display (0 = first); a color or a
                                    point as numbers separated by commas ("1,0.5,0")

Safety:
- Listens on 127.0.0.1 unless "accept from the network" is on.
- Commands must be POSTs carrying the header "X-VisualDeck: 1". A web page
  from another site cannot add that header without a CORS pre-flight, which
  this server never approves, so other sites open in the same browser cannot
  trigger cues.
- The Host header must be an IP address, "localhost" or this computer's name,
  which blocks DNS-rebinding pages that pretend to be served from here.
- There is no password: anyone who can reach the port can use the page.

The server thread never touches the UI: commands go on the same queue as OSC,
and status is a snapshot the UI thread publishes twice a second.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from remote_control import EFFECT_COMMANDS, effect_args, seconds_arg

log = logging.getLogger("visualdeck.web")


class _Server(ThreadingHTTPServer):
    """ThreadingHTTPServer without SO_REUSEADDR on Windows. There it lets a second
    socket bind a port that is already in use (both "listen", and requests go to
    either), so a busy port was not reported: found by the tests on the owner's
    PC. Elsewhere it only allows binding while old connections wait out
    TIME_WAIT, which a quick restart needs. (Python's socket.create_server makes
    the same choice.)"""
    allow_reuse_address = os.name != "nt"
    daemon_threads = True

# The HTTP API, for Help (help_content): (method and path, what it does).
HTTP_REFERENCE = [
    ("GET /", "The control page: GO, BACK, Blackout, Update Output, the cue list and Effects on the output."),
    ("GET /api/status", "JSON: the live cue, the cue list, Blackout, edits waiting, a running follow, the cue "
                        "clock and the effects on the output."),
    ("POST /api/go", "GO."),
    ("POST /api/back", "BACK."),
    ("POST /api/blackout", "Blackout."),
    ("POST /api/update", "Update Output."),
    ("POST /api/cue/<n>", "Go to cue n by position (1 = first), with its transition."),
    ("POST /api/cue/<n>/from/<seconds>", "Start Cue From: cue n (position) that many seconds in (12.5 or 1:02.5)."),
    ("POST /api/effect/<layer>/<input>/<value>", "As /visualdeck/effect over OSC; a color or a point as numbers "
                                                 "separated by commas (1,0.5,0)."),
    ("POST /api/display_effect/<display>/<input>/<value>", "As /visualdeck/display_effect over OSC."),
]

SIMPLE = {"/api/go": "go", "/api/back": "back", "/api/blackout": "blackout", "/api/update": "update"}

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisualDeck remote</title>
<style>
 :root { color-scheme: dark; }
 body { margin: 0; font: 16px system-ui, sans-serif; background: #1c1c1c; color: #eee; }
 header { padding: 12px 16px; background: #2b2b2b; }
 #live { font-size: 20px; font-weight: 600; }
 #sub { color: #bbb; font-size: 14px; min-height: 1.2em; }
 .row { display: flex; gap: 10px; padding: 12px 16px; }
 button { flex: 1; font-size: 18px; padding: 18px 8px; border: 0; border-radius: 8px; color: #111; }
 #go { background: #5a5; font-size: 28px; } #back { background: #999; }
 #black { background: #a33; color: #fff; } #upd { background: #e6b422; }
 button:disabled { opacity: .35; }
 ul { list-style: none; margin: 0; padding: 0 16px 24px; }
 li { padding: 14px 12px; margin: 6px 0; background: #2b2b2b; border-radius: 6px; }
 li.live { background: #3b6b8f; } li.empty { color: #777; }
 .notes { color: #9fd3ff; font-size: 13px; font-style: italic; }
 #err { color: #ff8a8a; padding: 0 16px; }
 #from { flex: 1; font-size: 16px; padding: 10px; border-radius: 6px; border: 1px solid #555;
         background: #2b2b2b; color: #eee; } #clr { flex: 0 0 48px; background: #555; color: #eee; padding: 10px; }
 details { margin: 0 16px 8px; background: #242424; border-radius: 6px; }
 summary { padding: 12px; cursor: pointer; color: #ccc; }
 .fxg { padding: 4px 12px 10px; } .fxg h3 { font-size: 14px; color: #9fd3ff; margin: 8px 0 4px; font-weight: 600; }
 .fxi { display: flex; align-items: center; gap: 8px; margin: 8px 0; }
 .fxi label { flex: 0 0 34%; font-size: 14px; } .fxi input[type=range] { flex: 1; min-width: 0; }
 .fxi output { flex: 0 0 44px; text-align: right; font-size: 13px; color: #bbb; }
 .fxi select { flex: 1; font-size: 15px; padding: 6px; background: #2b2b2b; color: #eee; border: 1px solid #555; }
 .fxi input[type=checkbox] { width: 24px; height: 24px; }
</style></head><body>
<header><div id="live">Connecting…</div><div id="sub"></div></header>
<div class="row"><button id="go">GO</button></div>
<div class="row"><button id="back">◀ BACK</button><button id="black">Blackout</button>
<button id="upd" disabled>Update Output</button></div>
<div class="row"><input id="from" placeholder="Start a cue from… (1:30)" title="Type a time, then tap a cue"
 inputmode="decimal" autocomplete="off"><button id="clr" title="Clear">×</button></div>
<div id="err"></div>
<details id="fxbox" hidden><summary>Effects on the output</summary><div id="fx"></div></details>
<ul id="cues"></ul>
<script>
const H = {"X-VisualDeck": "1"};
const $ = id => document.getElementById(id);
async function cmd(path) {
  try { const r = await fetch(path, {method: "POST", headers: H}); if (!r.ok) throw new Error(r.status);
        $("err").textContent = ""; } catch (e) { $("err").textContent = "Command failed: " + e.message; }
  refresh();
}
$("go").onclick = () => cmd("/api/go"); $("back").onclick = () => cmd("/api/back");
$("black").onclick = () => cmd("/api/blackout"); $("upd").onclick = () => cmd("/api/update");
$("clr").onclick = () => { $("from").value = ""; };
function goCue(n) {
  // With a time in "Start a cue from", the tapped cue starts that far in (once).
  const t = $("from").value.trim();
  if (t) { $("from").value = ""; cmd("/api/cue/" + n + "/from/" + encodeURIComponent(t)); }
  else cmd("/api/cue/" + n);
}
async function refresh() {
  try {
    const s = await (await fetch("/api/status", {cache: "no-store"})).json();
    $("live").textContent = s.black ? "OUTPUT BLACK" : (s.live ? (s.live.cue_number || s.live.number) + ". " + s.live.name : "—");
    let sub = s.transition ? "Transition running" : (s.clock || "");
    if (s.follow) sub = "→ continues to " + s.follow.to + " in " + s.follow.seconds.toFixed(1) + " s";
    if (s.test_pattern) sub = "Test pattern on the output";
    $("sub").textContent = sub;
    $("upd").disabled = !s.edits_pending;
    const ul = $("cues"); ul.innerHTML = "";
    for (const c of s.cues) {
      const li = document.createElement("li");
      li.className = (s.live && c.number === s.live.number && !s.black ? "live" : "") + (c.has_media ? "" : " empty");
      li.textContent = (c.cue_number || c.number) + ". " + c.name;
      if (c.notes) { const n = document.createElement("div"); n.className = "notes"; n.textContent = c.notes; li.appendChild(n); }
      li.onclick = () => goCue(c.number);
      ul.appendChild(li);
    }
    showFx(s.effects || []);
    $("err").textContent = "";
  } catch (e) { $("err").textContent = "VisualDeck is not reachable"; }
}
// Effect inputs on the output: a control per slider, list and on/off input.
let fxSig = "", fxBusy = 0;
const fxPending = {};
const fmt = v => Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2);
function fxKey(g, inp) { return g.kind + g.index + ":" + g.position + ":" + inp.name; }
function fxSend(g, inp, value) {
  // The newest value of a control, at most every 60 ms (a slider sends many).
  const key = fxKey(g, inp), kind = g.kind === "layer" ? "effect" : "display_effect";
  fxPending[key] = "/api/" + kind + "/" + g.index + "/" + encodeURIComponent(g.position + ":" + inp.name) + "/" +
                   encodeURIComponent(value);
  if (fxPending[key + " t"]) return;
  fxPending[key + " t"] = setTimeout(async () => {
    const path = fxPending[key]; delete fxPending[key + " t"];
    try { const r = await fetch(path, {method: "POST", headers: H}); if (!r.ok) throw new Error(r.status); }
    catch (e) { $("err").textContent = "Effect change failed: " + e.message; }
  }, 60);
}
function buildFx(list) {
  const box = $("fx"); box.innerHTML = "";
  for (const g of list) {
    const d = document.createElement("div"); d.className = "fxg";
    const h = document.createElement("h3"); h.textContent = g.label + " \u203a " + g.title; d.appendChild(h);
    for (const inp of g.inputs) {
      const row = document.createElement("div"); row.className = "fxi";
      const lab = document.createElement("label"); lab.textContent = inp.label; row.appendChild(lab);
      let el, out = null;
      if (inp.type === "bool") {
        el = document.createElement("input"); el.type = "checkbox"; el.checked = !!inp.value;
        el.onchange = () => fxSend(g, inp, el.checked ? 1 : 0);
      } else if (inp.values) {
        el = document.createElement("select");
        inp.values.forEach((v, i) => { const o = document.createElement("option"); o.value = v;
                                       o.textContent = inp.labels[i]; el.appendChild(o); });
        el.value = inp.value; el.onchange = () => fxSend(g, inp, el.value);
      } else {
        el = document.createElement("input"); el.type = "range"; el.min = inp.min; el.max = inp.max;
        el.step = inp.type === "long" ? 1 : (inp.max - inp.min) / 200; el.value = inp.value;
        out = document.createElement("output"); out.textContent = fmt(inp.value);
        el.oninput = () => { out.textContent = fmt(+el.value); fxSend(g, inp, el.value); };
        el.onpointerdown = () => { fxBusy++; };
        el.onpointerup = el.onpointercancel = () => { fxBusy = Math.max(0, fxBusy - 1); };
      }
      el.dataset.key = fxKey(g, inp); row.appendChild(el); if (out) row.appendChild(out);
      d.appendChild(row);
    }
    box.appendChild(d);
  }
}
function updateFx(list) {
  // Values from VisualDeck (another remote, MIDI, the editor), except a control being moved here.
  for (const g of list) for (const inp of g.inputs) {
    const el = document.querySelector('[data-key="' + CSS.escape(fxKey(g, inp)) + '"]');
    if (!el || el === document.activeElement || fxBusy || fxPending[fxKey(g, inp) + " t"]) continue;
    if (inp.type === "bool") el.checked = !!inp.value; else el.value = inp.value;
    const out = el.nextElementSibling;
    if (out && out.tagName === "OUTPUT") out.textContent = fmt(+inp.value);
  }
}
function showFx(list) {
  $("fxbox").hidden = !list.length;
  const sig = JSON.stringify(list.map(g => [g.kind, g.index, g.position, g.title, g.inputs.map(i => i.name)]));
  if (sig !== fxSig) { fxSig = sig; buildFx(list); } else updateFx(list);
}
refresh(); setInterval(refresh, 700);
</script></body></html>
"""


def host_allowed(host_header, extra_names=()):
    """True if the Host header names this machine (IP literal, localhost, hostname)."""
    if not host_header:
        return False
    host = host_header.strip()
    if host.startswith("["):                      # [IPv6]:port
        host = host[1:host.find("]")] if "]" in host else host
    elif host.count(":") == 1:
        host = host.split(":", 1)[0]
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    names = {"localhost"} | {n.lower() for n in extra_names if n}
    try:
        hn = socket.gethostname().lower()
        names |= {hn, hn + ".local", hn.split(".")[0]}
    except OSError:
        pass
    return host.lower().rstrip(".") in names


class WebRemote:
    def __init__(self, on_command, get_status, host="127.0.0.1", port=7701):
        self.on_command, self.get_status = on_command, get_status
        self.host, self.port = host, int(port)
        self._server = None
        self._thread = None
        self.error = None
        self.requests = 0

    @property
    def listening(self):
        return self._server is not None

    def address(self):
        return None if self._server is None else self._server.server_address[:2]

    def start(self):
        remote = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "VisualDeck"
            sys_version = ""

            def log_message(self, fmt, *args):       # route to our logger, quietly
                log.debug("%s %s", self.client_address[0], fmt % args)

            def _send(self, code, body=b"", ctype="text/plain; charset=utf-8"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _host_ok(self):
                if host_allowed(self.headers.get("Host")):
                    return True
                log.warning(f"Web remote: refused request for Host {self.headers.get('Host')!r} "
                            f"from {self.client_address[0]}")
                self._send(403, b"Forbidden host")
                return False

            def do_GET(self):  # noqa: N802
                remote.requests += 1
                if not self._host_ok():
                    return
                if self.path in ("/", "/index.html"):
                    self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                elif self.path == "/api/status":
                    try:
                        body = json.dumps(remote.get_status() or {}).encode("utf-8")
                    except Exception:  # noqa: BLE001
                        log.exception("Web remote status failed")
                        self._send(500, b"status unavailable")
                        return
                    self._send(200, body, "application/json")
                else:
                    self._send(404, b"Not found")

            def do_POST(self):  # noqa: N802
                remote.requests += 1
                if not self._host_ok():
                    return
                if self.headers.get("X-VisualDeck") != "1":
                    log.warning(f"Web remote: refused {self.path} from {self.client_address[0]} (missing header)")
                    self._send(403, b"Missing X-VisualDeck header")
                    return
                cmd, arg = SIMPLE.get(self.path), None
                if cmd is None and self.path.startswith("/api/cue/"):
                    tail = self.path[len("/api/cue/"):]
                    if tail.isdigit() and int(tail) >= 1:
                        cmd, arg = "cue_at", int(tail)      # by position in the list, never by cue number
                    elif "/from/" in tail:
                        n, t = tail.split("/from/", 1)
                        secs = seconds_arg(unquote(t))
                        if n.isdigit() and int(n) >= 1 and secs is not None:
                            cmd, arg = "start_from_at", (int(n), secs)
                for kind in EFFECT_COMMANDS:
                    if cmd is None and self.path.startswith(f"/api/{kind}/"):
                        parts = [unquote(x) for x in self.path[len(f"/api/{kind}/"):].split("/")]
                        if len(parts) == 3:
                            parsed = effect_args(kind, [parts[0], parts[1]] + parts[2].split(","))
                            if parsed is not None:
                                cmd, arg = parsed
                if cmd is None:
                    self._send(404, b"Not found")
                    return
                remote.on_command(cmd, arg, self.client_address)
                self._send(202, b"queued")

            def do_OPTIONS(self):  # noqa: N802  - never approve cross-origin pre-flights
                self._send(403, b"Cross-origin requests are not allowed")

        try:
            self._server = _Server((self.host, self.port), Handler)
        except OSError as e:
            self.error = f"cannot listen on {self.host}:{self.port}: {e}"
            log.error(f"Web remote: {self.error}")
            self._server = None
            return False
        self.error = None
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.2},
                                         name="WebRemote", daemon=True)
        self._thread.start()
        log.info("Web remote at http://%s:%d/" % self.address())
        return True

    def stop(self):
        srv, self._server = self._server, None
        if srv is not None:
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:  # noqa: BLE001
                log.exception("Stopping the web remote failed")
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
