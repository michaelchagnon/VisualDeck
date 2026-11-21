# SPDX-License-Identifier: MPL-2.0
# =====================================================================
# src/app.py
# =====================================================================
import os
import sys

# Modes for the Windows build, which has no Python to run scripts with, before
# the app's heavy imports: Find Best Setting For Me starts the program with
# --benchmark <mode> (benchmark.py), and the build is checked with
# --self-test [report.json] (selftest.py).
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] in ('--benchmark', '--self-test'):
    if sys.argv[1] == '--benchmark':
        import benchmark
        print(benchmark.run(sys.argv[2] if len(sys.argv) > 2 else None), flush=True)
        sys.exit(0)
    import selftest
    sys.exit(selftest.run(sys.argv[2] if len(sys.argv) > 2 else None))

import logging
import platform
import json
import logging_setup

log = logging.getLogger("visualdeck.app")
from media_manager import MediaManager
from control_interface import ControlInterface

def setup_logging():
    """Configures the logging for the entire application."""
    import threading

    log_file = logging_setup.configure()

    # --- Crash UI helper (best-effort; fully guarded) ---
    def _show_crash_dialog(exc_text: str):
        try:
            import tkinter as tk
            from tkinter import messagebox
            r = tk.Tk()
            r.withdraw()
            try:
                # Build a message that points to the actual resolved log path
                message = (
                    "A fatal error occurred.\n\n"
                    f"Open the log at:\n{log_file}\n\n"
                    "Include it in your issue report.\n\n"
                    "System info and traceback have been copied to your clipboard."
                )
                messagebox.showerror("VisualDeck crashed", message, parent=r)
            except Exception:
                pass
            try:
                r.clipboard_clear()
                r.clipboard_append(exc_text)
                # Ensure clipboard contents persist after the window is destroyed
                r.update()
            except Exception:
                pass
            try:
                r.destroy()
            except Exception:
                pass
        except Exception:
            # If Tk is unavailable (e.g., headless), silently skip the dialog
            pass

    # --- This is the key part for catching crashes ---
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Preserve default behavior for Ctrl+C
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        import traceback as _tb
        import platform as _pf

        # Try to obtain a user-facing version string without importing heavy UI code
        try:
            # control_interface already imports in app; this is a light import of a constant
            from control_interface import SPLASH_VERSION_TEXT as _VD_VER
        except Exception:
            _VD_VER = ""

        ver_line = f"VisualDeck {_VD_VER}".strip() or "VisualDeck"
        sys_line = f"{_pf.platform()} Py{_pf.python_version()}"
        trace_text = "".join(_tb.format_exception(exc_type, exc_value, exc_traceback))
        info = f"{ver_line}\n{sys_line}\n\n{trace_text}"

        # Log full traceback to file
        log.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

        # Best-effort crash dialog + clipboard
        _show_crash_dialog(info)

    sys.excepthook = handle_exception

    # Also capture uncaught exceptions in threads (Python 3.8+)
    try:
        def _threading_excepthook(args):
            handle_exception(args.exc_type, args.exc_value, args.exc_traceback)
        threading.excepthook = _threading_excepthook
    except Exception:
        # If not supported or reassignment fails, ignore silently
        pass

def main():
    tz_note = logging_setup.fix_windows_tz()   # before the first log line, so its time is right
    setup_logging()
    log.info("=====================================================")
    log.info("           VisualDeck Application Starting Up        ")
    log.info("=====================================================")
    try:
        from control_interface import SPLASH_VERSION_TEXT as _ver
    except Exception:  # noqa: BLE001
        _ver = "?"
    try:
        env = logging_setup.environment_report()
        log.info("Session: VisualDeck %s | %s", _ver, json.dumps(env, ensure_ascii=True),
                     extra={'data': {'session': env}})
    except Exception:  # noqa: BLE001 - diagnostics must never stop start-up
        log.exception("Could not write the session header")
    if tz_note:
        log.info(f"Time zone: {tz_note}")
    try:
        fault_path, crash = logging_setup.enable_crash_report()
        if crash:
            log.error("The previous session ended with a crash outside Python (a native crash). "
                      f"Where each thread was at that moment ({fault_path}):\n{crash[:6000]}")
    except Exception:  # noqa: BLE001 - diagnostics must never stop start-up
        log.exception("Could not start the crash report")
    # Experiment (performance budget): how long a thread may keep
    # Python's lock before another waiting thread (the render thread) gets it.
    # Python's default is 5 ms; set VISUALDECK_SWITCH_INTERVAL_MS to try others.
    _si = os.environ.get("VISUALDECK_SWITCH_INTERVAL_MS")
    if _si:
        try:
            sys.setswitchinterval(max(0.0005, float(_si) / 1000.0))
            log.info(f"Python switch interval set to {sys.getswitchinterval() * 1000:.1f} ms (VISUALDECK_SWITCH_INTERVAL_MS)")
        except (ValueError, TypeError):
            log.warning(f"Ignoring VISUALDECK_SWITCH_INTERVAL_MS={_si!r}")
    
    # A show machine must not sleep or blank its screens while VisualDeck is open
    # (Windows; no setting is changed, the request ends with the program).
    import keep_awake
    keep_awake.hold()

    # ADD THIS - Track if we need Windows cleanup
    windows_cleanup_needed = False
    
    try:
        # ADD THIS - Windows detection
        if sys.platform == "win32":
            windows_cleanup_needed = True
            log.info(f"Running on Windows {platform.version()}")
        
        # MediaManager is now a lightweight class that doesn't manage a central folder.
        # It's primarily used for getting media info.
        media_manager = MediaManager()
        
        # The RenderEngine and all project/media state are handled
        # within the ControlInterface.
        ControlInterface(media_manager)
        
        log.info("Application shutting down normally.")
        logging_setup.mark_clean_exit()
    except Exception:
        log.critical("A fatal error occurred in the main application loop.", exc_info=True)
    finally:
        keep_awake.release()
        # ADD THIS - Windows cleanup
        if windows_cleanup_needed and sys.platform == "win32":
            try:
                import ctypes
                winmm = ctypes.WinDLL('winmm')
                winmm.timeEndPeriod(1)  # Reset timer resolution
                log.info("Reset Windows timer resolution")
            except:
                pass


if __name__ == '__main__':
    main()
