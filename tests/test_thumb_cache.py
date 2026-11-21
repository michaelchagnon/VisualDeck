# SPDX-License-Identifier: MPL-2.0
"""F21: thumbnail sources are cached (reduced, per file version), so grid
rebuilds do not reopen every video and full-size image on the UI thread."""
import os

from PIL import Image

import control_interface as ci_mod
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401


def test_second_load_comes_from_the_cache(blind_ci, fixtures, monkeypatch):
    ci = blind_ci
    opened = []
    real = ci_mod.video_still                                   # PyAV, the video's own colors
    monkeypatch.setattr(ci_mod, "video_still", lambda p: opened.append(p) or real(p))
    a = ci._load_frame(fixtures["counting"], for_thumbnail=True)
    b = ci._load_frame(fixtures["counting"], for_thumbnail=True)
    assert len(opened) == 1 and a.tobytes() == b.tobytes()
    a.paste((255, 0, 0, 255), (0, 0, 10, 10))                   # callers may change what they get
    assert ci._load_frame(fixtures["counting"], for_thumbnail=True).tobytes() == b.tobytes()


def test_large_media_is_reduced_and_changes_are_seen(blind_ci, tmp_path):
    ci = blind_ci
    p = str(tmp_path / "big.png")
    Image.new("RGBA", (3840, 2160), (0, 200, 0, 255)).save(p)
    img = ci._load_frame(p, for_thumbnail=True)
    assert img.size == (480, 270) and img.getpixel((5, 5)) == (0, 200, 0, 255)
    Image.new("RGBA", (3840, 2160), (200, 0, 0, 255)).save(p)
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # make sure the time differs
    assert ci._load_frame(p, for_thumbnail=True).getpixel((5, 5)) == (200, 0, 0, 255)
    assert sum(1 for k in ci._thumb_sources if k[0] == p) == 1, "the old version is dropped"
    assert ci._load_frame(str(tmp_path / "missing.png"), for_thumbnail=True) is None


def test_cache_is_bounded(blind_ci, tmp_path, monkeypatch):
    ci = blind_ci
    monkeypatch.setattr(ci, "THUMB_SOURCE_ENTRIES", 3)
    for i in range(5):
        p = str(tmp_path / f"{i}.png")
        Image.new("RGB", (20, 10), (i, i, i)).save(p)
        ci._load_frame(p, for_thumbnail=True)
    assert [os.path.basename(k[0]) for k in ci._thumb_sources] == ["2.png", "3.png", "4.png"]
