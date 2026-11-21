# SPDX-License-Identifier: MPL-2.0
"""The Windows build's extra entry points (app.py --self-test / --benchmark)."""
import json
import os
import subprocess
import sys

import pytest

import benchmark

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _needs_display():
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")


def test_self_test_reports_every_check(tmp_path):
    _needs_display()
    report = tmp_path / "selftest.json"
    env = dict(os.environ, PYGAME_HIDE_SUPPORT_PROMPT="1")
    if os.name != "nt":
        env["SDL_AUDIODRIVER"] = "dummy"           # no sound card in a Linux container
    r = subprocess.run([sys.executable, os.path.join(SRC, "app.py"), "--self-test", str(report)], cwd=SRC,
                       env=env, capture_output=True, text=True, timeout=120)
    data = json.loads(report.read_text(encoding="utf-8"))
    checks = data["checks"]
    assert set(checks) == {"imports", "decode", "opencv", "effects", "assets", "tk", "sound", "midi_osc", "opengl",
                           "awake", "ffmpeg"}
    assert "distribution is" in checks["ffmpeg"]["detail"]          # LGPL or GPL: release notes need it
    # MIDI needs the system's MIDI service (none in a Linux container), OpenGL a GPU (none on CI's
    # Windows runners, which set VD_SKIP_GL); everything else must pass here.
    allowed = {"midi_osc"} | ({"opengl"} if os.environ.get("VD_SKIP_GL") == "1" else set())
    failed = {k: v.get("error") for k, v in checks.items() if not v["ok"] and k not in allowed}
    assert not failed, failed
    assert r.returncode == (0 if all(v["ok"] for v in checks.values()) else 1)
    assert data["frozen"] is False and "9 effects" in checks["effects"]["detail"]


def test_benchmark_command_runs_the_script_or_the_frozen_program(monkeypatch):
    assert benchmark.command("opengl")[1].endswith("benchmark.py")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert benchmark.command("opengl") == [sys.executable, "--benchmark", "opengl"]


def test_app_benchmark_entry_prints_a_time():
    _needs_display()
    r = subprocess.run([sys.executable, os.path.join(SRC, "app.py"), "--benchmark", "pygame_gpu"], cwd=SRC,
                       env=dict(os.environ, PYGAME_HIDE_SUPPORT_PROMPT="1"), capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0
    float(r.stdout.strip().splitlines()[-1])           # what Find Best Setting For Me reads
