# SPDX-License-Identifier: MPL-2.0
"""Logging configuration for VisualDeck.

- Human-readable log at ~/.visualdeck/app.log (rotating, 5 x 5 MB), one line per
  record: ISO timestamp *with UTC offset*, level, logger, thread, message.
  The offset makes logs from different machines and time zones comparable.
- Optional JSON-lines log (~/.visualdeck/app.jsonl) for tools: set the
  environment variable VISUALDECK_LOG_JSON=1. Records logged with
  extra={'data': {...}} (e.g. the periodic health summary) carry that data.
- Level from VISUALDECK_LOG_LEVEL (DEBUG, INFO, WARNING, ...; default INFO).

Modules log through named loggers under "visualdeck." (visualdeck.render,
visualdeck.control, ...), so a line says which subsystem wrote it.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import platform
import sys
import time
from logging.handlers import RotatingFileHandler

import version

TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(threadName)s] %(message)s"


def _iso(created: float) -> str:
    return _dt.datetime.fromtimestamp(created).astimezone().isoformat(timespec="milliseconds")


class TextFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(TEXT_FORMAT)

    def formatTime(self, record, datefmt=None):  # noqa: N802 (logging API)
        return _iso(record.created)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        out = {"ts": _iso(record.created), "level": record.levelname, "logger": record.name,
               "thread": record.threadName, "msg": record.getMessage()}
        data = getattr(record, "data", None)
        if data is not None:
            out["data"] = data
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str, ensure_ascii=True)


def log_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".visualdeck")


def configure(directory: str | None = None, env=None) -> str:
    """Install the handlers on the root logger (idempotent). Returns the text log path."""
    env = os.environ if env is None else env
    directory = directory or log_dir()
    os.makedirs(directory, exist_ok=True)
    text_path = os.path.join(directory, "app.log")
    root = logging.getLogger()
    level = getattr(logging, str(env.get("VISUALDECK_LOG_LEVEL", "INFO")).upper(), logging.INFO)
    root.setLevel(level)

    def have(path):
        return any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(path)
                   for h in root.handlers)

    if not have(text_path):
        h = RotatingFileHandler(text_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        h.setFormatter(TextFormatter())
        root.addHandler(h)
    if str(env.get("VISUALDECK_LOG_JSON", "")).strip().lower() in ("1", "true", "yes", "on"):
        json_path = os.path.join(directory, "app.jsonl")
        if not have(json_path):
            h = RotatingFileHandler(json_path, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8")
            h.setFormatter(JsonFormatter())
            root.addHandler(h)
    for noisy in ("PIL", "OpenGL", "watchdog", "asyncio"):
        logging.getLogger(noisy).setLevel(max(level, logging.INFO))
    return text_path


def fix_windows_tz(env=None, platform=None, tzset=None):
    """On Windows, a TZ variable in IANA form ("America/New_York") makes the C
    runtime, and so Python's local time, behave as UTC. Remove it from this
    process and re-read the time zone. Returns what was done (for the log)."""
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    tz = env.get("TZ")
    if platform != "win32" or not tz or "/" not in tz:
        return None
    del env["TZ"]
    try:
        if tzset is None:
            import ctypes
            tzset = ctypes.cdll.ucrtbase._tzset
        tzset()
        return f"ignored TZ={tz} (not understood by Windows); using the Windows time zone"
    except Exception as e:  # noqa: BLE001
        return f"removed TZ={tz} but could not re-read the time zone: {e}"


def environment_report(env=None) -> dict:
    """Facts worth having at the top of every session log."""
    env = os.environ if env is None else env
    rep = {"visualdeck": version.VERSION, "python": sys.version.split()[0], "platform": platform.platform(),
           "executable": sys.executable}
    off = _dt.datetime.now().astimezone().strftime("%z")
    rep["time_zone"] = {"tzname": list(time.tzname), "utc_offset": f"{off[:3]}:{off[3:]}" if off else "?",
                        "TZ_env": env.get("TZ")}
    for mod in ("av", "numpy", "pygame", "OpenGL", "cv2", "PIL"):
        m = sys.modules.get(mod)
        if m is not None:
            rep.setdefault("packages", {})[mod] = getattr(m, "__version__", "?")
    av = sys.modules.get("av")
    if av is not None:
        try:
            v = av.library_versions.get("libavcodec")
            rep["libavcodec"] = ".".join(map(str, v)) if v else "?"
        except Exception:  # noqa: BLE001
            pass
    return rep


# -- crashes outside Python ---------------------------------------------------------
# A crash in native code (FFmpeg, a camera or graphics driver) ends the process
# without a Python traceback or a log line. faulthandler writes every thread's
# Python stack to faults.log at that moment; the next start reports it in app.log.
# (Owner's PC, 2026-11-15: VisualDeck closed with an access violation when the
# USB camera was unplugged during its cue, and nothing said where.)
FAULTS_FILE = "faults.log"
SESSION_MARK = "=== VisualDeck session"
CLEAN_EXIT_MARK = "=== clean exit"
_fault_file = None


def _previous_session(text):
    """(fault text, ended cleanly) of the last session recorded in faults.log."""
    last = text.rsplit(SESSION_MARK, 1)[-1] if SESSION_MARK in text else ""
    body = last.split("\n", 1)[1] if "\n" in last else ""
    clean = CLEAN_EXIT_MARK in body
    body = body.replace(CLEAN_EXIT_MARK, "").strip()
    return body, clean


def enable_crash_report(directory: str | None = None, max_bytes: int = 1_000_000):
    """Start faulthandler on faults.log; returns (path, report) where report is the
    previous session's fault text if that session did not end cleanly, else None."""
    import faulthandler
    global _fault_file
    directory = directory or log_dir()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, FAULTS_FILE)
    report = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            body, clean = _previous_session(f.read())
        if body and not clean:
            report = body
    except OSError:
        pass
    try:
        mode = "w" if os.path.getsize(path) > max_bytes else "a"
    except OSError:
        mode = "a"
    if _fault_file is not None:
        return path, report
    f = open(path, mode, encoding="utf-8", errors="replace", buffering=1)
    f.write(f"{SESSION_MARK} {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}\n")
    f.flush()
    faulthandler.enable(file=f, all_threads=True)
    _fault_file = f
    return path, report


def mark_clean_exit():
    """At a normal exit: the next start does not report this session's faults.log
    entries (if any: handled exceptions in drivers can be written too) as a crash."""
    global _fault_file
    f = _fault_file
    if f is None:
        return
    try:
        import faulthandler
        faulthandler.disable()
        f.write(CLEAN_EXIT_MARK + "\n")
        f.close()
    except Exception:  # noqa: BLE001 - never stop the exit
        pass
    _fault_file = None
