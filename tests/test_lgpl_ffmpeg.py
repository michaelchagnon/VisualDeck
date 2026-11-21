# SPDX-License-Identifier: MPL-2.0
"""Putting an LGPL FFmpeg into a built program (tools/windows/lgpl_ffmpeg.py).

The import-table rewrite itself needs Windows modules; it was run on a copy of the
owner's build on 2026-11-16 (49 PyAV modules relinked). These tests cover the names
and the files."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "windows"))
import lgpl_ffmpeg  # noqa: E402

LGPL = ["avcodec-61.dll", "avdevice-61.dll", "avfilter-10.dll", "avformat-61.dll", "avutil-59.dll",
        "swresample-5.dll", "swscale-8.dll"]


def test_the_wheel_names_map_to_the_plain_ones():
    assert lgpl_ffmpeg.plain_name("avcodec-61-7f5ba29974161bb2dc8f58d741e9b107.dll") == "avcodec-61.dll"
    assert lgpl_ffmpeg.plain_name("swresample-5-bdff1e4ddf8a951323b3fa44d9ed1067.dll") == "swresample-5.dll"
    assert lgpl_ffmpeg.plain_name("libx264-165-de8de7d5ffe9d75d839a96621ec53cd9.dll") is None   # not FFmpeg's own
    assert lgpl_ffmpeg.plain_name("python312.dll") is None and lgpl_ffmpeg.plain_name("avcodec-61.dll") is None


def _tree(tmp_path):
    dist = tmp_path / "VisualDeck"
    (dist / "_internal" / "av").mkdir(parents=True)
    libs = dist / "_internal" / "av.libs"
    libs.mkdir()
    for n in ("avcodec-61-7f5ba29974161bb2dc8f58d741e9b107.dll", "libx264-165-de8de7d5ffe9d75d839a96621ec53cd9.dll",
              "libx265-9adc56be88b1827574686b8a839bca58.dll", "libcrypto-1_1-x64-0d95c8bbfefbc533ef8513ecd31047ae.dll"):
        (libs / n).write_bytes(b"gpl wheel")
    bin_dir = tmp_path / "ffmpeg" / "bin"
    bin_dir.mkdir(parents=True)
    for n in LGPL + ["ffmpeg.exe", "ffprobe.exe"]:
        (bin_dir / n).write_bytes(b"lgpl " + n.encode())
    lic = tmp_path / "ffmpeg" / "LICENSE.txt"
    lic.write_text("GNU LESSER GENERAL PUBLIC LICENSE Version 3")
    return dist, libs, bin_dir, lic


def test_the_wheel_libraries_are_replaced_and_the_license_written(tmp_path):
    dist, libs, bin_dir, lic = _tree(tmp_path)
    patched, removed, copied = lgpl_ffmpeg.apply(str(dist), str(bin_dir), str(lic), "the test build", log=lambda m: None)
    assert sorted(os.listdir(libs)) == LGPL                         # no x264, x265, OpenSSL; no programs
    assert any("x264" in n for n in removed) and len(removed) == 4 and patched == 0
    licenses = dist / "licenses"
    assert (licenses / "FFmpeg-LICENSE.txt").read_text().startswith("GNU LESSER")
    assert "GNU GENERAL PUBLIC LICENSE" in (licenses / "GPL-3.0.txt").read_text()       # LGPLv3 needs both
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in (licenses / "LGPL-3.0.txt").read_text()
    src = (licenses / "FFmpeg-SOURCE.txt").read_text()
    assert "Lesser General Public License" in src and "the test build" in src and "can be replaced" in src


def test_a_folder_that_is_not_an_ffmpeg_7_1_build_is_refused(tmp_path):
    dist, libs, bin_dir, lic = _tree(tmp_path)
    os.remove(bin_dir / "avcodec-61.dll")
    with pytest.raises(SystemExit):
        lgpl_ffmpeg.apply(str(dist), str(bin_dir), log=lambda m: None)
    assert len(os.listdir(libs)) == 4                                # nothing removed
