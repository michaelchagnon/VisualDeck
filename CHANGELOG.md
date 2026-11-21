# Changelog

## 1.0.0 (2026-11-21)

VisualDeck 1.0 is the first stable release. It is free and open source under the **Mozilla Public License 2.0**, which replaces 1.0b's Source-Available License and EULA.

### Install
- A Windows installer with everything included: Python, the libraries and an LGPL build of FFmpeg. It installs per user without administrator rights and has a Start menu entry and an uninstaller.

### Running a show
- Cue numbers, names and notes; GO, BACK and Blackout; click a cue to cut to it.
- Transitions for each cue (cut, fade, smooth fade, dip to black), and follows after a time or at a video's end.
- Seamless cuts with preloading, and Keep Ready for any cue.
- Blind editing with a preview of the edited cue and Update Output; undo and redo.
- Start Cue From for rehearsals, Lock Editing, Preshow Check, a schedule, and Presentation Mode with Loop.
- Autosave with crash recovery, missing-media relinking, and Collect Show.

### Video, sound and effects
- Sound from video files, with an output device choice, volume and delay.
- Speed, reverse, ping-pong, and in and out points for each clip; Make Playback Copy.
- Correct colors in every render mode, transparency for ProRes 4444, HAP Alpha and PNG, and crossfades that keep their brightness.
- Opacity, blend modes, color correction and masks for each layer.
- Real-time ISF shader effects (nine built in, up to four on each layer and on each display), controllable by OSC, the browser remote or MIDI faders.

### Outputs and projection mapping
- Corner pin, mesh warp, edge blending, Span Displays (rows and grids), slices, and warp points for each layer.
- Test patterns, output presets, and detection of displays that are plugged in or removed.

### Live sources and show control
- USB cameras, capture cards and network streams, with sound and automatic reconnection.
- OSC, a browser remote with an HTTP API, MIDI (notes, Program Change, faders, learn), MIDI Show Control, MIDI timecode, SMPTE LTC, and Art-Net/sACN in and out.

### Reliability
- Output Health with frame timing, decoder and sound status, a frozen-video watchdog, and logs.
- A built-in Soak Test. The 1.0 release candidates passed an 8-hour run and a 72-hour unattended run with no render errors.
- Fixed: leaks of memory and Windows GDI objects in long shows, a crash when a USB camera was unplugged during its cue, a hang when the sound output stopped, and black frames on cuts.

### Help
- Built-in help (F1) that covers every window and menu, with a quick start for show operators.

## 1.0b (2025-08-19)

The original beta: a cue and layer grid, multi-display output, fades, the position editor, Presentation Mode, and OpenGL and CPU rendering. Its source is kept on the `archive/1.0b` branch and the `v1.0b` tag.
