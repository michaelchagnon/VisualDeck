# SPDX-License-Identifier: MPL-2.0
"""Shared pytest fixtures for VisualDeck.

- Puts src/ on sys.path so tests import the application modules directly.
- Generates synthetic media once per session (see media_fixtures.py).
- Provides an X display for OpenGL tests: uses $DISPLAY if set, otherwise
  starts a private Xvfb server when one is installed, otherwise GL tests skip.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
TESTS = os.path.join(ROOT, "tests")
for p in (SRC, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory):
    from media_fixtures import build_fixture_set
    d = tmp_path_factory.mktemp("media")
    paths = build_fixture_set(str(d))
    manifest = d / "fixtures.json"
    manifest.write_text(json.dumps(paths))
    paths["_manifest"] = str(manifest)
    return paths


@pytest.fixture(scope="session")
def x_display():
    """Return a DISPLAY string usable by SDL/OpenGL, or skip.

    Windows and macOS need no X server: GL tests open windows on the desktop
    (yields None). On Linux, $DISPLAY is used if set, else a private Xvfb.
    """
    if os.environ.get("VD_SKIP_GL") == "1":
        # e.g. CI machines without OpenGL 3.3 (GitHub's Windows runners)
        pytest.skip("OpenGL tests disabled (VD_SKIP_GL=1)")
    if sys.platform in ("win32", "darwin"):
        yield None
        return
    if os.environ.get("DISPLAY"):
        yield os.environ["DISPLAY"]
        return
    xvfb = shutil.which("Xvfb")
    if not xvfb:
        pytest.skip("No DISPLAY and Xvfb not installed: OpenGL tests skipped")
    for num in range(91, 120):
        if not os.path.exists(f"/tmp/.X11-unix/X{num}") and not os.path.exists(f"/tmp/.X{num}-lock"):
            break
    disp = f":{num}"
    proc = subprocess.Popen([xvfb, disp, "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 10
    while time.time() < deadline and not os.path.exists(f"/tmp/.X11-unix/X{num}"):
        time.sleep(0.05)
    try:
        yield disp
    finally:
        proc.terminate()
        proc.wait(timeout=5)


GL_SEEN = {}  # renderer -> GL version, from scenarios that report it


def pytest_terminal_summary(terminalreporter):
    """Say which GPU/driver the OpenGL tests ran on (hybrid-graphics laptops
    may use the integrated GPU), so a saved test log records it."""
    for renderer, version in GL_SEEN.items():
        terminalreporter.write_line(f"OpenGL tests ran on: {renderer} | {version}")


def run_gl_scenario(name, fixtures, display, extra_env=None, timeout=180):
    """Run tests/gl_harness.py <name> in a subprocess and return its JSON result."""
    env = dict(os.environ, SDL_AUDIODRIVER="dummy")
    if display:
        env["DISPLAY"] = display
    env.update(extra_env or {})
    proc = subprocess.run([sys.executable, os.path.join(TESTS, "gl_harness.py"), name, fixtures["_manifest"]],
                          capture_output=True, text=True, env=env, timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result = json.loads(line[len("RESULT_JSON:"):])
            if isinstance(result, dict) and result.get("renderer"):
                GL_SEEN[result["renderer"]] = result.get("gl", "?")
            return result
    raise AssertionError(f"GL scenario {name} produced no result.\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}")


@pytest.fixture(autouse=True)
def _collect_garbage_on_the_main_thread():
    """Tests that open Tk windows leave reference cycles (callbacks holding
    widgets and variables). If Python's cycle collector frees them later on a
    decoder thread, Tcl aborts the process ("deleted by the wrong thread").
    Collect after every test, here on the main thread."""
    yield
    import gc
    gc.collect()
