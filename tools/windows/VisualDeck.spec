# PyInstaller spec for the Windows build: one folder, VisualDeck.exe without a
# console window. Built by build_windows.ps1 (next to this file); see docs/BUILD.md.
# -*- mode: python -*-
import glob
import os

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))      # noqa: F821 - SPECPATH is set by PyInstaller
SRC = os.path.join(ROOT, "src")

# Every VisualDeck module, also the ones only imported inside functions.
ours = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(SRC, "*.py")))
ours += ["utils." + os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(SRC, "utils", "*.py"))
         if not p.endswith("__init__.py")]

hidden = ours + collect_submodules("OpenGL.platform") + collect_submodules("OpenGL.arrays")
try:
    hidden += collect_submodules("OpenGL_accelerate")
except Exception:  # noqa: BLE001 - the accelerator is optional
    pass
hidden += ["pygame._sdl2.audio", "pygame._sdl2.sdl2", "rtmidi", "pythonosc.dispatcher", "pythonosc.osc_server",
           "pythonosc.udp_client", "win32api", "win32process", "win32con"]

datas = [
    (os.path.join(SRC, "effects", "*.fs"), "effects"),     # isf.BUILTIN_DIR: next to the modules
    (os.path.join(SRC, "visualdeck.ico"), "src"),          # the window icon (control_interface, render_engine)
    (os.path.join(SRC, "splash.png"), "src"),
]

# PyAV and NumPy wheels keep their DLLs (FFmpeg, OpenBLAS) in a "<package>.libs" folder next to the
# package (delvewheel); make sure the build has them where the package looks.
binaries = []
try:
    from PyInstaller.utils.hooks import collect_delvewheel_libs_directory
except ImportError:  # older PyInstaller: its own hooks decide
    collect_delvewheel_libs_directory = None
for pkg in ("av", "numpy") if collect_delvewheel_libs_directory else ():
    try:
        datas, binaries = collect_delvewheel_libs_directory(pkg, datas=datas, binaries=binaries)
    except Exception as e:  # noqa: BLE001 - no .libs folder: nothing to add
        print(f"VisualDeck.spec: no delvewheel libs for {pkg}: {e}")

a = Analysis(  # noqa: F821
    [os.path.join(SRC, "app.py")],
    pathex=[SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=["pytest", "_pytest", "playwright", "IPython", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VisualDeck",
    icon=os.path.join(SRC, "visualdeck.ico"),
    console=False,
    disable_windowed_traceback=False,
    upx=False,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    upx=False,
    name="VisualDeck",
)
