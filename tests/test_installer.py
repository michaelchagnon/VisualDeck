# SPDX-License-Identifier: MPL-2.0
"""The Windows installer script (tools/windows/installer.nsi) builds with NSIS.

makensis runs on Linux too; the installer itself was run on the owner's PC."""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSI = os.path.join(ROOT, "tools", "windows", "installer.nsi")


@pytest.mark.skipif(shutil.which("makensis") is None, reason="needs NSIS (makensis)")
def test_the_installer_builds_from_a_program_folder(tmp_path):
    dist = tmp_path / "dist" / "VisualDeck"
    (dist / "_internal" / "src").mkdir(parents=True)
    shutil.copy(os.path.join(ROOT, "src", "visualdeck.ico"), dist / "_internal" / "src" / "visualdeck.ico")
    (dist / "VisualDeck.exe").write_bytes(b"MZ not really")
    (dist / "THIRD_PARTY_NOTICES.md").write_text("notices")
    out = tmp_path / "VisualDeck-test-setup.exe"
    r = subprocess.run(["makensis", "-V2", f"-DVERSION=test", f"-DDIST={dist}", f"-DOUT={out}", NSI],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.read_bytes()[:2] == b"MZ" and out.stat().st_size > 50_000


def test_the_installer_needs_no_administrator_and_keeps_user_data():
    text = open(NSI, encoding="utf-8").read()
    assert "RequestExecutionLevel user" in text and "$LOCALAPPDATA\\Programs\\VisualDeck" in text
    assert "MUI_FINISHPAGE_RUN" in text                         # offers to start VisualDeck at the end
    assert ".visualdeck" not in text.split('Section "Uninstall"')[1].split("RMDir")[0]   # settings not removed
