# Building, running and testing VisualDeck

## Requirements

- Python 3.12 (3.10+ should work; NumPy 2.1 requires ≥ 3.10). Tkinter must be available (`python3-tk` on Debian/Ubuntu; included in the python.org Windows installer).
- A GPU with OpenGL 3.3 core for Performance mode. Software OpenGL (Mesa llvmpipe) works for development and tests.
- Windows 11 is the primary target. Linux runs and is used for automated testing. macOS is untested.

## Windows quick start (development launcher)

`tools/windows/` holds a launcher for running the current development version on a Windows PC with one double-click. Copy `Start-VisualDeck.cmd` and `README.txt` into a folder (e.g. `%USERPROFILE%\dev\VisualDeck-dev`), `launcher.ps1` into its `launcher\` subfolder, and a full code bundle (`git bundle create visualdeck.bundle main --tags`) into its `updates\` subfolder. On each start the launcher:

1. checks for Git and Python 3.12, offering to install them with winget if missing (it asks first);
2. creates `app\` from the bundle on first run, otherwise fast-forwards it to the newest bundle (it also picks up a newer `VisualDeck-update*.bundle` saved in Downloads) and lists the new commits;
3. reinstalls Python packages into `app\.venv` only when the requirements files changed;
4. starts VisualDeck (`test` runs the test suite instead, `setup` stops after installing) and copies the app log into `logs\`.

It never deletes user files, never overwrites local edits in `app\` (a non-fast-forward update is skipped with a message), and keeps CRLF line endings via `.gitattributes`.

## Set up

```bash
git clone https://github.com/michaelchagnon/VisualDeck.git
cd VisualDeck
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Run

```bash
python src/app.py
```

Logs: `~/.visualdeck/app.log` (timestamps with UTC offset, subsystem, thread; a health summary every 60 s). `VISUALDECK_LOG_JSON=1` adds `app.jsonl`; `VISUALDECK_LOG_LEVEL=DEBUG` for more detail. In the app: Window → Output Health… → Open Log Folder. Set `VISUALDECK_DEBUG=1` to enable PyOpenGL error checking (slower; turns GL errors into logged exceptions).

## Test

```bash
python -m pytest
```

- On Linux, OpenGL tests need an X display: `$DISPLAY` is used if set, otherwise the suite starts a private Xvfb, otherwise those tests are skipped. On Windows and macOS they open short-lived windows on the desktop (run them in a normal desktop session). They passed on the owner's Windows 11 PC on 2026-11-11; macOS is untested.
- `VD_SKIP_GL=1` skips the OpenGL tests (used by CI on Windows, whose hosted runners have no OpenGL 3.3).
- CI: `.github/workflows/tests.yml` runs the suite on Linux (with OpenGL via Mesa) and Windows (without OpenGL) on every push and pull request.
- Test media is generated on the fly with PyAV (`tests/media_fixtures.py`); no media files are stored in the repository.
- `tests/gl_harness.py` runs each OpenGL scenario in its own process and reads back the frame the engine was about to present.

## Benchmark

```bash
xvfb-run -a -s "-screen 0 3840x1080x24" python bench/bench_render_thread.py --seconds 8 --label mymachine
```

Simulates GUI on display 0 and output on display 1 with one 1080p clip, with and without preview requests. On Windows with two real displays, run it in a desktop session (the script patches monitor enumeration to two 1920×1080 displays side by side; adjust `MONITORS` to match). Results go to `bench/results/<label>.json`. Compare only runs from the same machine.

## Packaging (Windows build)

`tools/windows/build_windows.ps1` builds VisualDeck as one folder that runs without the development launcher, Git or an installed Python (PyInstaller 6.11.1, spec `tools/windows/VisualDeck.spec`):

- On the Windows PC, in the development folder (`%USERPROFILE%\dev\VisualDeck-dev`): copy `app\tools\windows\Build-VisualDeck.cmd` next to `Start-VisualDeck.cmd` and double-click it (or run `powershell -ExecutionPolicy Bypass -File app\tools\windows\build_windows.ps1`). It needs the folder made by `Start-VisualDeck.cmd` (its Python 3.12) and the network for pip the first time.
- It makes its own environment in `build\venv` (the app's `.venv` is not touched), runs PyInstaller, copies LICENSE, README, CHANGELOG and THIRD_PARTY_NOTICES next to the program, writes `VERSION.txt` (the code's `git describe`), replaces the PyAV wheel's FFmpeg (GPL: x264, x265) with the LGPL Windows build of FFmpeg 7.1.5 from FFmpeg-Builds (downloaded once into `build\ffmpeg-lgpl`, checked against its SHA-256; `tools/windows/lgpl_ffmpeg.py`), runs **`VisualDeck.exe --self-test`** and stops if any check fails or the FFmpeg is not LGPL, then zips the folder.
- Output: `dist\VisualDeck\VisualDeck.exe`, `dist\VisualDeck-<version>-windows.zip`, and with NSIS 3 installed `dist\VisualDeck-<version>-setup.exe` (`tools/windows/installer.nsi`: installs for the current user without administrator rights, Start menu and desktop shortcuts, an uninstaller, offers to start VisualDeck; `makensis` on Linux builds it too); logs in `logs\build-<time>.log` and the self-test report in `logs\selftest-<time>.json`.
- `VisualDeck.exe --self-test [report.json]` (also `python src/app.py --self-test`) checks, without opening a show or an output: every module imports, PyAV writes and decodes a clip, OpenCV, the nine built-in effects parse, the icon and splash are there, Tk starts, SDL's sound outputs are listed (and pygame's mixer is closed), MIDI and OSC load, and a hidden OpenGL 3.3 window compiles the output's shaders. Exit code 0 only if all pass.
- In the build, Find Best Setting For Me starts `VisualDeck.exe --benchmark <mode>` instead of running `benchmark.py` with Python.
- Settings, effects, presets, autosaves and logs stay in `%USERPROFILE%\.visualdeck`, shared with the development version.

**Release builds.** The installer is made from `dist\VisualDeck` with `makensis /DVERSION=<version> /DDIST=<folder> /DOUT=<setup.exe> tools/windows/installer.nsi` (NSIS 3, on Windows or Linux). A release must ship the LGPL FFmpeg, which the build's self-test checks (`ffmpeg` line: "distribution is LGPL"); the PyAV wheel's own FFmpeg includes the GPL x264 and x265 encoders. The zip also runs without installing: unpack it anywhere and start `VisualDeck.exe`.
