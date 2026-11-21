# SPDX-License-Identifier: MPL-2.0
"""F24: a missing bundled FFmpeg is reported once at INFO, not as an ERROR per video."""
import logging
import threading

from utils import ffmpeg_utils as fu


def test_missing_ffmpeg_is_reported_once_and_starts_no_threads(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(fu, "FFMPEG_PATH", str(tmp_path / "nope" / "ffmpeg"))
    monkeypatch.setattr(fu, "_missing_reported", False)
    monkeypatch.setattr(fu, "CONFIG_DIR", str(tmp_path / "cfg"))
    before = threading.active_count()
    with caplog.at_level(logging.INFO, logger="visualdeck.ffmpeg"):
        for name in ("a.mp4", "b.mp4", "c.mp4"):
            fu.FFMpegProxyGenerator(str(tmp_path / name)).generate()
    recs = [r for r in caplog.records if r.name == "visualdeck.ffmpeg"]
    assert len(recs) == 1 and recs[0].levelno == logging.INFO and "proxies are off" in recs[0].getMessage()
    assert threading.active_count() == before
