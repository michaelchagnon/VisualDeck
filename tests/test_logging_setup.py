# SPDX-License-Identifier: MPL-2.0
import json
import os
import logging
import re

import pytest

import logging_setup


@pytest.fixture
def clean_root():
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield root
    for h in list(root.handlers):
        if h not in saved:
            root.removeHandler(h)
            h.close()
    root.setLevel(level)


def test_text_log_has_offset_timestamp_logger_and_thread(tmp_path, clean_root):
    path = logging_setup.configure(str(tmp_path), env={})
    logging.getLogger("visualdeck.render").warning("hello %s", "world")
    for h in clean_root.handlers:
        h.flush()
    line = open(path, encoding="utf-8").read().strip().splitlines()[-1]
    assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d WARNING visualdeck\.render \[MainThread\] hello world$",
                    line), line


def test_configure_is_idempotent(tmp_path, clean_root):
    n0 = len(clean_root.handlers)
    logging_setup.configure(str(tmp_path), env={})
    logging_setup.configure(str(tmp_path), env={})
    assert len(clean_root.handlers) == n0 + 1


def test_json_lines_carry_structured_data(tmp_path, clean_root):
    logging_setup.configure(str(tmp_path), env={"VISUALDECK_LOG_JSON": "1", "VISUALDECK_LOG_LEVEL": "debug"})
    assert clean_root.level == logging.DEBUG
    logging.getLogger("visualdeck.control").info("health", extra={"data": {"fps": 59.9}})
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("visualdeck.control").exception("failed")
    for h in clean_root.handlers:
        h.flush()
    recs = [json.loads(l) for l in open(tmp_path / "app.jsonl", encoding="utf-8")]
    assert recs[-2]["data"] == {"fps": 59.9} and recs[-2]["logger"] == "visualdeck.control"
    assert "ValueError: boom" in recs[-1]["exc"]


def test_environment_report_includes_time_zone():
    rep = logging_setup.environment_report(env={"TZ": "UTC"})
    assert rep["time_zone"]["TZ_env"] == "UTC"
    assert re.match(r"^[+-]\d\d:\d\d$", rep["time_zone"]["utc_offset"])


def test_windows_iana_tz_is_removed_and_reread():
    calls = []
    env = {"TZ": "America/New_York", "OTHER": "x"}
    note = logging_setup.fix_windows_tz(env, "win32", lambda: calls.append(1))
    assert "TZ" not in env and calls == [1] and "America/New_York" in note
    for env, plat in (({"TZ": "EST5EDT"}, "win32"), ({"TZ": "America/New_York"}, "linux"), ({}, "win32")):
        before = dict(env)
        assert logging_setup.fix_windows_tz(env, plat, lambda: 1 / 0) is None and env == before


def test_a_native_crash_is_reported_at_the_next_start(tmp_path):
    """faulthandler writes the threads' stacks to faults.log when the process dies in
    native code; the next start returns them, and a clean exit is not a crash."""
    import subprocess
    import sys
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    crash = ("import sys; sys.path.insert(0, %r); import logging_setup, faulthandler; "
             "logging_setup.enable_crash_report(%r); faulthandler._sigsegv()" % (src, str(tmp_path)))
    r = subprocess.run([sys.executable, "-c", crash], capture_output=True, timeout=60)
    assert r.returncode != 0
    import logging_setup
    logging_setup._fault_file = None
    path, report = logging_setup.enable_crash_report(str(tmp_path))
    assert report and ("Segmentation fault" in report or "access violation" in report), report
    assert "<module>" in report or "line 1" in report            # where it was
    logging_setup.mark_clean_exit()                              # this session ends cleanly
    logging_setup._fault_file = None
    path, report = logging_setup.enable_crash_report(str(tmp_path))
    assert report is None
    logging_setup.mark_clean_exit()
