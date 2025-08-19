# =====================================================================
# src/app.py
# =====================================================================
import os
import sys
import logging
import platform
from logging.handlers import RotatingFileHandler
from media_manager import MediaManager
from control_interface import ControlInterface

def setup_logging():
    """Configures the logging for the entire application."""
    import threading

    log_dir = os.path.join(os.path.expanduser('~'), '.visualdeck')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'app.log')

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the lowest level to capture

    # Create a rotating file handler (5 files, 5MB each)
    handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Avoid duplicate handlers if setup_logging is called twice
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', None) == handler.baseFilename
               for h in logger.handlers):
        logger.addHandler(handler)

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
        logging.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

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
    setup_logging()
    logging.info("=====================================================")
    logging.info("           VisualDeck Application Starting Up        ")
    logging.info("=====================================================")
    
    # ADD THIS - Track if we need Windows cleanup
    windows_cleanup_needed = False
    
    try:
        # ADD THIS - Windows detection
        if sys.platform == "win32":
            windows_cleanup_needed = True
            logging.info(f"Running on Windows {platform.version()}")
        
        # MediaManager is now a lightweight class that doesn't manage a central folder.
        # It's primarily used for getting media info.
        media_manager = MediaManager()
        
        # The RenderEngine and all project/media state are handled
        # within the ControlInterface.
        ControlInterface(media_manager)
        
        logging.info("Application shutting down normally.")
    except Exception:
        logging.critical("A fatal error occurred in the main application loop.", exc_info=True)
    finally:
        # ADD THIS - Windows cleanup
        if windows_cleanup_needed and sys.platform == "win32":
            try:
                import ctypes
                winmm = ctypes.WinDLL('winmm')
                winmm.timeEndPeriod(1)  # Reset timer resolution
                logging.info("Reset Windows timer resolution")
            except:
                pass


if __name__ == '__main__':
    main()
