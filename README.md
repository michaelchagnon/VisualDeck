# VisualDeck: free video playback, show control and projection mapping software for Windows

<p align="center">
  <a href="https://github.com/michaelchagnon/VisualDeck/releases/latest"><img src="https://img.shields.io/github/v/release/michaelchagnon/VisualDeck?label=download&color=2ea44f" alt="Download the latest release"></a>
  <img src="https://img.shields.io/badge/license-MPL--2.0-blue.svg" alt="License: MPL-2.0">
  <img src="https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-0078d6.svg" alt="Platform: Windows 10 and 11">
</p>

<p align="center">
  <img src="https://visualdeck.org/visualdeck.png" alt="VisualDeck, cue-based video playback software">
</p>

**VisualDeck** is free, open-source software for playing video, images, cameras and live streams on projectors and screens, cue by cue. Build your show as a grid of cues and layers, then press **GO**. It's made for theater, live events, installations and VJ sets. Projection mapping, multi-screen output and remote control by OSC, MIDI, timecode and DMX are all built in.

### [⬇ Download VisualDeck 1.0 for Windows](https://github.com/michaelchagnon/VisualDeck/releases/latest)

One installer with everything included: no Python, codecs or FFmpeg to install separately.

## Who it's for

- **Theater and dance:** video cues called by the stage manager, with GO, BACK, follows and a locked show file.
- **Concerts, conferences and live events:** backgrounds, walk-in loops, sponsor reels and camera feeds on several screens.
- **Installations, museums and exhibitions:** unattended looping playback that runs for days.
- **VJs and live visuals:** layers, blend modes and real-time shader effects you can control from MIDI or OSC.

## Features

### Cue-based show control
- A **cue grid**: each column is a cue and each row is a layer. Drag media into it.
- **GO, BACK and Blackout.** Click any cue to cut straight to it. Cues can have numbers, names and notes.
- **Transitions** per cue: cut, fade, smooth fade or dip to black.
- **Follows:** continue automatically after a set time or when a cue's videos end.
- **Presentation Mode** for slideshows and loops, plus a **schedule** for cues at set times of day.
- **Seamless cuts:** the next cue is preloaded, and **Keep Ready** holds any cue ready to play at once.
- **Blind editing:** change the live cue without the audience seeing it, preview the change, then **Update Output**.
- **Rehearsal tools:** start a cue from any point, and undo or redo any change.
- **Lock Editing** for the show, and a **Preshow Check** that lists problems before the doors open.

### Video, images and sound
- Plays **H.264, H.265 (HEVC), ProRes (including 4444 with transparency), HAP, HAP Alpha and DXV** in MP4, MOV, AVI and MKV files.
- Shows **JPG, PNG (with transparency), BMP and GIF** images.
- Playback **speed**, **reverse**, **ping-pong** and **in and out points** for each clip.
- **Sound** from the video files, with a choice of audio output, volume for each layer, and a delay to line sound up with the picture.
- **Make Playback Copy:** converts a hard-to-play file into one that starts instantly.
- Three render modes: **Performance** (GPU, OpenGL), **Balanced** and **Compatibility**, so it runs on most PCs.

### Layers and effects
- Set each layer's **position, size and rotation**, with **opacity**, **blend modes**, **color correction** and **masks** (rectangle or ellipse, soft edges).
- **Real-time shader effects:** up to four on each layer and on each display. The nine built-in effects are Blur, Invert, Kaleidoscope, Mirror, Pixelate, RGB Split, Tint, Trails and Wave. You can add your own **ISF** effects.
- **Warp points** bend a single picture to fit a piece of scenery.

### Projection mapping and multiple screens
- Use **any number of displays**, and send each layer to the display you choose.
- **Corner pin** and **mesh warp** for each display, with 1-pixel nudges from the arrow keys.
- **Edge blending** and **Span Displays** join projectors into one wide picture, side by side or in grids. **Slices** let you place each display's part of the picture by hand.
- **Test patterns** (grid, color bars, white) and **output presets** for each venue.

### Live cameras and streams
- **USB cameras and capture cards** as live layers, with a test view before you use them.
- **Network streams** that FFmpeg can receive (such as UDP or RTSP), with sound, and automatic reconnection.

### Show control and remote control
- **OSC** commands, so any OSC sender (QLab, Bitfocus Companion, TouchOSC and others) can trigger cues.
- A **browser remote** for phones and tablets, plus an **HTTP API**.
- **MIDI**: notes, Program Change, faders and MIDI learn. Also **MIDI Show Control (MSC)** and **MIDI timecode (MTC)**.
- **SMPTE LTC** timecode from an audio input.
- **DMX** over Art-Net or sACN, in from a lighting console and out to it.

### Built for reliable shows
- **Output Health** shows frame rate, late frames, decoders and sound, with plain-language warnings, and watches for frozen video.
- **Autosave and crash recovery**, **missing-media relinking**, and **Collect Show** to package a show with its media.
- A built-in **Soak Test**. VisualDeck 1.0 passed a **72-hour unattended test** with no errors.

## Install

1. Download **`VisualDeck-1.0.0-setup.exe`** from the [latest release](https://github.com/michaelchagnon/VisualDeck/releases/latest).
2. Run it. It installs for your Windows user only, with **no administrator rights needed**, and adds a Start menu entry, a desktop shortcut and an uninstaller.
3. If Windows SmartScreen warns about a new app, choose **More info › Run anyway**.

To uninstall, use Settings › Apps, or Start › VisualDeck › Uninstall VisualDeck. Your shows and settings are kept.

**System requirements:** Windows 10 or 11, 64-bit (tested on Windows 11). Performance Mode needs a graphics card or integrated GPU with OpenGL 3.3; Balanced and Compatibility Mode work without it. 8 GB of memory is recommended.

## Quick start

1. **Insert › Add Media to Project…** (Ctrl+I) adds videos and images.
2. Drag files onto the grid. Each column is a cue and each row is a layer.
3. Under **Output** above the preview, choose the display your audience sees.
4. Press **GO** (Space) to play the next cue, or click a cue's header to jump to it.
5. **File › Save** (Ctrl+S). Before a show, run **Show › Preshow Check…**.

Press **F1** in the app for the full help, which is also in the [user guide](docs/USER_GUIDE.md).

## FAQ

**Is VisualDeck free?** Yes. It's free and open source under the Mozilla Public License 2.0, for personal and commercial use.

**Is it a QLab alternative for Windows?** For video, yes. VisualDeck gives Windows users a cue list for video playback with GO, BACK, follows, timecode, OSC and MIDI Show Control. It doesn't run audio-only or lighting cues.

**Do I need to install anything else?** No. The installer includes Python, FFmpeg (an LGPL build) and every library VisualDeck uses.

**Does it do projection mapping?** Yes: corner pin, mesh warp, warp points, masks, edge blending and multi-projector spans. It doesn't do 3D projection mapping.

**Can it run unattended?** Yes. Presentation Mode with Loop, or the schedule, runs a show by itself, and the PC is kept awake while VisualDeck is open.

**Mac or Linux?** The release is for Windows. VisualDeck also runs from source on Linux, which is used for its automated tests. macOS is untested.

## Documentation

- [User guide](docs/USER_GUIDE.md): every window and menu, with a quick start for show operators
- [Remote control reference](docs/REMOTE_CONTROL.md): OSC, HTTP, MIDI and DMX commands
- [Known issues](docs/KNOWN_ISSUES.md) and [changelog](CHANGELOG.md)
- [Building and testing from source](docs/BUILD.md)

## Run from source

```bash
git clone https://github.com/michaelchagnon/VisualDeck.git
cd VisualDeck
python -m venv .venv
.venv\Scripts\activate        # on Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/app.py
```

Use Python 3.12 (64-bit). See [docs/BUILD.md](docs/BUILD.md) for the tests and the Windows build.

## License

VisualDeck is licensed under the [Mozilla Public License 2.0](LICENSE). The libraries it includes keep their own licenses, listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The installer uses an LGPL build of FFmpeg; its license and where to get its source are in the `licenses` folder of the installation.

**Older version:** the 1.0b source is kept on the [`archive/1.0b`](https://github.com/michaelchagnon/VisualDeck/tree/archive/1.0b) branch and the [`v1.0b`](https://github.com/michaelchagnon/VisualDeck/releases/tag/v1.0b) tag.

## Support the project

Website: [visualdeck.org](https://visualdeck.org/)

<p align="center">
  <img src="https://visualdeck.org/bitcoin.png" alt="Donate Bitcoin: bc1qv6aw43faucxyvne7z4hr7yystxu8vu5sf5fc3m"><br>
  Donate Bitcoin: bc1qv6aw43faucxyvne7z4hr7yystxu8vu5sf5fc3m
</p>
