# VisualDeck user guide

This guide is the same text as Help › VisualDeck Help (F1) in the app.

## Quick start: running a show

For the operator of a show someone else programmed (a theater performance, an event). The stage manager calls the cues; you press GO.

- Before the house opens (30 minutes or more): open the show (File › Open Project… or File › Recent Projects). Check that Output (above the preview) is the projector or screen the audience sees, and that the projector shows the first cue.
- Run Show › Preshow Check…: fix everything marked ✕ (missing media, no output, a camera that cannot be opened, a remote control that is not listening), look at what is marked !, then press Check Again until it says Ready for the show.
- Turn on Show › Lock Editing (Ctrl+L): nothing in the cue grid can be changed by accident; GO, BACK, Blackout and remote controls still work. The title bar shows [LOCKED].
- During the show: press GO (Space) on the stage manager's call. The preview on the right is what the audience sees; the highlighted cue header is the cue on the output. A countdown above the preview means the cue will continue by itself (Cue Settings); GO, BACK or Blackout cancel it.
- A wrong GO: BACK (Shift+Space) goes to the previous cue; clicking a cue header cuts straight to it.
- Something goes wrong on the output: Blackout (Ctrl+B) fades to black at once; GO continues with the next cue. A frozen or black picture: click its cue header to start the cue again. The status line under the preview turns WARN or ERROR and says why; click it for Output Health.
- After the show: Blackout, then File › Exit. Write down anything odd for the stage manager; Help › Open Log Folder has the log of the evening if someone asks for it.

## Getting started

VisualDeck plays video and still images on one or more displays, cue by cue. The grid in the middle of the window is the show: each column is a cue, each row is a layer. Layers lower in the grid are drawn on top of the ones above them.

- Add media: Insert › Add Media to Project… (Ctrl+I). Files appear in the media list on the left.
- Build cues: drag a file from the media list onto a cell of the grid.
- Choose the output: Output (above the preview) › the display the audience sees.
- Run the show: GO (Space) goes to the next cue; click a cue header to cut straight to that cue.
- Save: File › Save (Ctrl+S). Saves are atomic and the previous version is kept as .bak. Unsaved changes are also autosaved every 2 minutes to a separate file (never into the show); if VisualDeck closes unexpectedly, opening the show again offers to recover them.

Supported media: video .mp4, .mov, .avi, .mkv; images .jpg, .jpeg, .png, .bmp, .gif (PNG transparency is kept).

## Shows and files

- File › New Project (Ctrl+N): an empty show. If the open show has unsaved changes, VisualDeck asks first.
- File › Open Project… (Ctrl+O) opens a show file (.vjdeck); File › Recent Projects lists the last ones opened. The last show opened is opened again when VisualDeck starts. Opening a show puts its cue on the output as it was saved.
- File › Save (Ctrl+S) and File › Save As… (Ctrl+Shift+S): the show file holds the cue grid, every cell's settings, cue settings, the output setup (display, corner pins, meshes, edge blends, span, display effects) and the schedule. Media files are not inside it; they are found by their paths (also relative to the show, so a show folder can move). Saving is atomic and keeps the previous version as .bak. Unsaved changes are autosaved every 2 minutes to a separate file.
- File › Exit closes VisualDeck (asking to save unsaved changes). Closing the main window does the same.
- File › Relink Missing Media… and File › Collect Show…: see Missing media and Collect Show. File › Preferences… (Ctrl+comma): see Preferences.

## Cues and layers

- Click a cue header to cut straight to that cue (no transition). Ctrl+1 … Ctrl+9 go to cues 1–9 with their transition.
- Drag a cue header left or right to reorder cues; drag a layer header to reorder layers. An amber bar shows where it will land and the dragged label says “after” or “before” which cue or layer. Drop on a header. The cue on the output keeps playing.
- Right-click a cue header: Cue Settings…, Rename, Duplicate Cue (a copy right after it, with its media, cell settings and cue settings but no cue number), Clear Cue, Delete Cue.
- Right-click a layer header: Rename, Clear Layer, Delete Layer. With Output set to Multiple, the ⚙ button under a layer header chooses the display that layer goes to.
- Drag a cell onto another cell to swap them; hold Ctrl while dropping to copy instead.
- Double-click a cell, or right-click › Edit, to open the position editor. Right-click › Loop turns looping on or off for a video; Clear empties the cell.
- Insert › Cue (Ctrl+Shift+C) and Insert › Layer (Ctrl+Shift+L) add a column or row. Adding or deleting a layer does not interrupt the cue on the output.
- Edit › Undo (Ctrl+Z) and Redo (Ctrl+Shift+Z or Ctrl+Y) step back and forward through changes to the grid: media in cells and their position-editor settings, cue and layer names, cue settings, cues and layers added, deleted or moved, layer displays, the media list, and the schedule. Up to 100 steps; changes less than 0.6 s apart count as one. Output settings (the display, corner pins, edge blends) are not undone. An undo is an edit like any other: with blind editing, a change to the live cue waits for Update Output, and the cue on the output never restarts.

Header marks: • the cue has its own transition, → it continues by itself, * it has notes.

## Sound

Preferences › Audio › “Play the sound of videos on”: Off (the default), the system default output, or a specific device. Then the videos on the output play their first audio track.

- Test Sound (next to the delay) plays a one-second 440 Hz tone on the chosen output, even before you save, so you can check the device and the speakers.
- Each video's Volume is in the position editor (100 % = as recorded, up to 200 %).
- Delay the sound by … ms (Preferences › Audio, 0–1000): projectors and TVs show the picture a little late, often 30–100 ms; delay the sound by the same amount so they match. Film a clap, or play a clip with a flash and a beep, to find the value.
- Fades fade the sound with the picture; a crossfade crossfades it. Blackout fades it out.
- The sound starts with the picture and starts again when the video loops. The next cue's sound is prepared before GO.
- A video at a speed other than 1× plays silent. A live source with sound (a stream or camera that sends audio) plays it as it arrives, at the layer's Volume, about a tenth of a second behind; the sound delay setting does not apply to it.
- Output Health shows the output, how many videos are sounding, the level and any underruns (gaps when sound could not be decoded in time).

## Live sources

Insert › Add Live Source… adds a network stream or a camera to the media list (right-click it there › Test Live Source… to check it again later); drag it onto cells like any media. Type its address and press Test: VisualDeck connects and shows the picture size, frame rate, codec, whether it has sound and a still of what it sends (half a second in, after a camera's exposure settles), or why it got no picture.

- udp://0.0.0.0:5000 receives MPEG-TS sent to this computer on port 5000 (a multicast address joins the group). rtsp://, srt://, rtmp://, http(s):// (HLS) and tcp:// addresses are opened the same way. RTSP uses TCP.
- Cameras: dshow://video=<name> on Windows, v4l2:///dev/video0 on Linux, avfoundation://<index> on macOS, if the FFmpeg inside VisualDeck has camera support (the Add Live Source window says when it has none). Find Devices lists the cameras and capture cards FFmpeg can see; click one to put its address in, then Test. A camera used in any cue stays connected while the show uses it (opening one takes longer than a cut waits), so cutting to it is immediate.
- A source is opened once and shared by every cue that uses it, so a fade between two cues with the same camera works. The next cue's sources are connected before GO, so the picture is there at once.
- If the signal stops, the last picture stays on the output and VisualDeck reconnects by itself; Output Health shows the source as retrying (an error while it is on the output).
- Position, size, rotation, opacity, color and masks work as for videos. A live source has no length: “Continue when its videos end” ignores it.

## Running the show

- GO (Space): next cue (an empty cue shows black). After the last cue GO returns to the first. To make GO stay on the last cue that has media instead, untick Show › GO After the Last Cue Returns to the First (saved with the show).
- Show › Preshow Check…: before the house opens, one list of what to fix (✕) and what to look at (!), then what is fine (✓): the output display, missing media, output errors and late frames, cameras and streams that cannot be opened, effects that cannot be drawn or need Performance Mode, remote controls that are on but not listening, the sound output, free disk space, a test pattern left on, unsaved changes and whether editing is locked. It changes nothing; fix what it lists and press Check Again. Copy puts the list on the clipboard.
- Show › Lock Editing (Ctrl+L): during a show, nothing in the cue grid or the media list can be changed by accident (drops, deletes, edits, renames, cue settings, undo); an attempt shows a notice above the preview. GO, BACK, selecting cues, Blackout, remotes and output settings still work. The title bar shows [LOCKED]. The lock lasts until you turn it off or close VisualDeck.
- BACK (Shift+Space): previous cue that has media. The next and the previous cue are prepared in the background, so GO and BACK start their videos at once.
- Blackout (Ctrl+B): fades the output to black with the show's transition; the button turns red (OUTPUT BLACK). GO then continues with the next cue.
- Go to cue 1–9 with its transition: Ctrl+1 … Ctrl+9. Clicking a cue header cuts to it.
- Show › Start Cue From… (rehearsals): choose a cue and a time into it (12.5 or 1:02.5); Start cuts to the cue with its videos and their sound at that point, as if it had been playing that long, allowing for each video's speed, in and out points, direction and loop. The cue clock and the cue's follow count from there. On the browser remote: type the time in “Start a cue from…”, then tap the cue (the field clears after one use).
- Transition and Timing (under the preview) are the show default; a cue can override them in Cue Settings. Fade is a straight crossfade, Smooth Fade eases in and out, Dip to Black fades the old cue to black and then the new one up (each half takes half the time). Transition None cuts; the next cue's videos are prepared before the cut, so a cut never shows black.
- Presentation Mode (under the preview; Show › Presentation Mode On/Off, Ctrl+P) makes the show run by itself: see the Presentation Mode topic.
- Show › Cue Settings (live cue)… (Ctrl+E) opens the settings of the cue on the output; Show › Go to Cue 1 … 9 is a reminder of Ctrl+1 … Ctrl+9.

The cue after the live one is preloaded, so it starts at once on GO.

While VisualDeck is open, Windows does not go to sleep or turn its displays off. No power setting is changed: when VisualDeck closes, the computer's own settings apply again.

Show › Schedule… runs cues at set times of day (for installations and opening hours): add events with a time (24 h), days, and Go to cue / GO / Blackout, then tick Run this schedule. Each event runs in its minute, once a day, on this computer's clock; events missed while VisualDeck was closed are not caught up. The next event is shown under the cue clock. The schedule is saved with the show.

Under Output and Cue, the cue clock shows how far the live cue's longest video has got (at its speed): “Video 00:12 / 01:30  -01:18”, or “loop 2” while it loops; “Cue 00:12” for cues without video. It counts from the cut or the start of the fade, so it can run a fraction of a second ahead of the picture, and it is also shown on the browser remote.

## Presentation Mode

Presentation Mode plays the cue list by itself, like a slideshow: one GO starts it, and VisualDeck then goes to the next cue every few seconds without anyone pressing GO. Use it for a looping foyer or exhibition display, a rehearsal run-through, or any show that should advance on a fixed beat. It is off unless you tick Enable.

- Turn it on: under the preview, Presentation Mode › tick Enable (Ctrl+P switches it on and off). Set Seconds (how long each cue stays) and tick Loop to start again from the first cue after the last one. These are saved with the show.
- Start: press GO (Space, the GO button, OSC /visualdeck/go or the browser remote's GO). It goes to the next cue at once, with that cue's transition, and then on to the next one every Seconds. The GO button turns red and reads STOP while it runs.
- Stop: press STOP (the same button, Space or /visualdeck/go). The cue on the output stays up; nothing is cut. Unticking Enable also stops it.
- Empty cues are skipped. At the end of the cue list it stops on the last cue with media, or with Loop goes on with the first one.
- Seconds counts from the start of each cue's transition, so it includes the fade: with a 2 s fade and 5 s, each cue is fully up for about 3 s. Keep Seconds longer than the longest fade.
- While it runs, each cue's own follow (Cue Settings › Then: Continue …) is not used; the Seconds timer decides. Videos keep playing (and looping) for as long as their cue is up.
- BACK and a click on a cue header work while it runs; the next step still comes when the timer runs out, and goes on from the cue then on the output. To hold on a cue, press STOP.

To have some cues continue by themselves and others wait for GO, leave Presentation Mode off and use Cue Settings › Then: Continue to the next cue after a number of seconds, or when its videos end, on the cues that should continue.

## Cue settings and follow

Right-click a cue header › Cue Settings… (Ctrl+E for the live cue):

- Name.
- Transition: Show default, None (cut), Fade, Smooth Fade or Dip to Black, and its fade time (blank uses the show default).
- Then: Wait for GO, Continue to the next cue after a number of seconds, or Continue when its videos end (the longest video). The time counts from the cut or from the start of the cue's fade. A countdown appears above the preview; GO, BACK or Blackout cancel it. “When its videos end” waits for the video itself to reach its end (or its out point) rather than a computed time, so a slow start does not cut off its last frames; if the video is stuck it continues 2 s late.
- Notes: shown above the preview while the cue is live (and on the browser remote).
- Number (next to Name): an optional cue number such as 12 or 12.5, shown on the header. OSC /visualdeck/cue 12.5 and MIDI Program Change use it before the cue's position, so inserting a cue does not change what a remote calls up. Ctrl+1 … Ctrl+9 and the browser remote's list stay positional.
- Timecode: HH:MM:SS:FF. With MIDI Setup › Follow MIDI timecode or LTC Setup › Follow LTC on, VisualDeck goes to the cue (with its transition) when the running timecode reaches that time. Jumping the timecode (locate, rewind) changes nothing; the next time reached while running does. A time reached during a fade runs when the fade ends.
- Keep ready: the cue's videos stay loaded and paused on their first frame, like the next cue's, so jumping to it out of order (a standby, an encore) is as quick as GO. At most four cues are kept ready this way; each keeps a decoder busy. While following timecode, the cue the timecode reaches next is kept ready by itself, and so is a scheduled cue in the minute before it runs.

While Presentation Mode runs, its timer is used instead of per-cue follows.

## Blind editing

By default, editing the cue that is on the output does not change the output. Your changes are saved in the show, the button next to GO turns amber (Update Output – live cue edited) and a second preview, EDITED CUE, shows the cue as edited (each video as a still: its first frame, or a later one when it starts black, as in a fade-in).

- Update Output (Ctrl+U) puts the edits on the output. Unchanged videos keep playing; new videos are prepared first, so there is no black frame.
- Triggering the cue again, or going to another cue, also uses the edited version.
- Changing the output display or a layer's display applies at once and never reveals pending edits.
- Preferences › Editing the live cue › Apply immediately switches to live editing.

## Position editor

Double-click a cell (or right-click › Edit). The canvas on the left shows the layer on its output. The settings are on the right, in sections: Position and size, Opacity and blend, Color, Playback (videos only), Mask and Effects. Click a section's name to open or close it; a closed section shows what is set in it (for example “50 %, Add” or the effects' names), and the sections you leave open are open again for the next layer. Drag the picture to move it, drag a corner to resize it (hold Ctrl to keep its shape), then Save (Enter) or Cancel (Esc).

- Opacity (0–100 %) multiplies with fades.
- Blend: Normal, Add, Multiply or Screen. Blend modes need Performance Mode; the other render modes draw them as Normal.
- Color: Brightness, Contrast and Saturation (Reset color puts them back). They work in every render mode; in Balanced and Compatibility Mode they cost some CPU time per video frame.
- Speed (videos only): 0.25× to 4×, applied at once to a playing video. “Continue when its videos end” allows for the speed. Fast speeds need the decoder to keep up; Output Health shows if it does not.
- Direction (videos only): Forward, Reverse (from the last frame to the first) or Ping-pong (forward, then back, and so on). Changing it starts the video again from its first frame in that direction; the old picture stays up until then. Backwards videos play silent. Long-GOP files (most H.264 and HEVC) cost more to play backwards; files with only key frames (MJPEG, ProRes) cost the same as forward. A ping-pong video counts twice its length for “Continue when its videos end”.
- In and out (videos only): play only part of a clip, from In to Out, in seconds from the clip's start (12.5, or 1:02.5 for minutes). Empty In is the start, empty Out the end. Loops, Reverse, Ping-pong, the sound, the cue clock and “Continue when its videos end” all use the part between them. Changing them starts the video again at In.
- Volume (videos only): 0–200 % of the clip's sound, when Preferences › Audio plays the sound of videos (off until you choose an output).
- Warp points: bend the layer's picture onto a surface — a set piece, a column, a screen at an angle. Open Warp points and dots appear on the picture: drag a corner dot for a keystone, or set Points to more than 2 × 2 (up to 9 × 9) to bend it along a curve; the picture follows a smooth surface through every dot. Click a dot and use the arrow keys for 1 px steps (Shift: 10 px). Changing the number of points keeps the shape; a position that would fold the picture over itself is refused; Reset warp makes it flat again. The warp moves, turns and scales with the layer, works in every render mode, and the mask, color, opacity and effects follow the bent picture. A display's corner pin and mesh (Window › Corner Pin, Mesh and Edge Blend) bend everything on that display; warp points bend one layer.
- Mask: show only part of the layer. Rectangle (a crop) or Ellipse (fits the rectangle); Left, Right, Top and Bottom move each side in (as a percentage of the media); Soft edge fades the edge over a width given as a percentage of the media's shorter side; Invert shows only what is outside. The mask moves, scales and turns with the layer, and the dashed outline on the canvas shows it. None removes it. Reset (under Presets) resets position, size and rotation only.
- Effects (Performance mode only): shader effects on the layer's picture — Blur, Invert, Kaleidoscope, Mirror, Pixelate, RGB Split, Tint, Trails (fading trails behind what moves) or Wave — each with its own controls. Add effect… adds one (up to four, applied in order: each works on the result of the one above it); ▲ ▼ change the order and ✕ removes one. Effects are drawn before position, opacity, color, mask and blend. The editor's canvas shows them as the output draws them (moving ones move) while the output window is open in Performance Mode; the line under the canvas says when it cannot. Wave moves by itself, so a still image with Wave is redrawn every frame. Balanced and Compatibility mode draw the layer without its effect. Each effect is described in the Shader effects topic.
- Your own effects: put ISF filter files (.fs, the Interactive Shader Format: a GLSL shader with a JSON header, with an image input named inputImage) in the folder .visualdeck\effects in your user folder. They are listed, and changes to them used, the next time you open a layer's editor; a file with the name of a built-in effect replaces it. File › Collect Show copies the show's own effects into an effects folder next to the collected show, and a show finds the effects in the effects folder next to its file first. Filters with float, bool, long, color and point2D inputs work, also with several passes and buffers (PASSES with TARGET, PERSISTENT, FLOAT, WIDTH and HEIGHT); extra images, audio inputs and imported images do not (the editor says why). If the output cannot compile an effect, the layer is drawn without it and the editor shows the compiler's message with the file's line number.
- With live editing, changes reach the output while you drag; with blind editing they wait for Update Output.

## Shader effects

Effects change a layer's picture (Position editor › Effects) or everything a display shows (Window › Output Effects). They need Performance Mode (Preferences › Rendering); Balanced and Compatibility Mode draw the picture without them. Up to four on one layer or display, applied in order: each works on the result of the one above it.

- **Blur**: Softens the picture. Radius in pixels (0–50).
- **Invert**: Turns the picture into its negative. Amount blends from the picture (0) to the negative (1).
- **Kaleidoscope**: Repeats a wedge of the picture around a center, like a kaleidoscope. Sides (2–16) is how many times; Turn rotates the wedge (animate it from a fader or OSC for a turning kaleidoscope); Center is the point it turns around (0.5, 0.5 is the middle). On a picture with large plain areas the result can look plain too: move Center onto the detail.
- **Mirror**: Mirrors one half onto the other. Keep chooses the half that stays: left, right, top or bottom.
- **Pixelate**: Large square pixels. Size is their width as a share of the picture's width.
- **RGB Split**: Shifts the red and blue channels apart. Distance and Angle (direction).
- **Tint**: Colors the picture with one color, keeping its light and dark. Amount blends it in.
- **Trails**: Leaves fading trails behind what moves: each frame keeps part of the frame before (Keep, 0–0.98; higher = longer trails). Only something that moves leaves a trail: on a still image, or a layer that does not move, Trails shows no change. Use it on video, on a live camera, or on a layer that a fade or another effect (Wave, Kaleidoscope turning) is moving.
- **Wave**: A moving ripple across the picture. Strength is how far it bends (0 to 0.08, a share of the picture; 0.01 is subtle, 0.05 strong), Waves how many ripples across it, Speed how fast they travel, Up and down turns them vertical. It moves by itself, so a still image with Wave is redrawn every frame.

- Seeing the result: the position editor's canvas shows the layer with its effects in Performance Mode; the output shows them when the cue is on it (with blind editing, after Save and Update Output).
- Effects run at the size the layer is drawn on the output (a 4K clip shown at 1920 wide runs its effects at 1920), so they cost no more than the picture the audience sees.
- Remote control: every input can be set by OSC (/visualdeck/effect), the browser remote (Effects on the output) and MIDI faders (MIDI Setup); see the OSC commands topic.
- Your own effects: ISF filter files (.fs) in the folder .visualdeck\effects in your user folder (see Position editor).
- If the graphics card cannot compile an effect, the layer is shown without it, the editor shows the compiler's message, and Output Health turns to WARN.

## Outputs and test patterns

- Output: None (no output window), a single display, or Multiple: then the ⚙ button under each layer header chooses that layer's display.
- Window › Corner Pin, Mesh and Edge Blend…: drag the four corners of a display's picture so it lines up with a screen a projector hits at an angle (keystone). Click a corner and use the arrow keys for 1 px steps (Shift: 10 px); Reset Corners puts them back.
- Mesh warp (same window, Edit mesh points): for a curved screen or a surface four corners cannot fit. Choose how many points (Columns and Rows, 2 to 9; changing them keeps the shape), then drag a point or click it and use the arrow keys. The picture bends smoothly through every point, inside the corner pin (moving a corner moves the whole mesh). A shape that would fold the picture over itself is refused. Reset Mesh makes it flat again. In Balanced and Compatibility Mode a change takes a moment (about a tenth of a second at 1080p) to reach the output.
- Edge blend (same window): where two projectors overlap, give each the overlap in pixels on that side (Left, Right, Top, Bottom). The picture fades out across it so the overlap is not brighter. Gamma matches the projector (2.2 is typical); Curve sets how quickly the fade rises. Both apply at once, in every render mode, and are saved with the show. The output preview shows the result.
- Window › Span Displays…: one picture across displays side by side, e.g. two projectors whose edges overlap. Tick the displays (left to right), give the overlap in pixels, keep Blend the overlaps on, and Apply: Output becomes Span, every layer goes to every display in the span, and the position editor lays layers out on the whole picture (its size is shown in the window). Each display shows its part at full resolution; corner pins, meshes and edge blends still work per display for fine alignment. The preview shows the whole picture. Remove Span goes back to the first display; the edge blends Apply set stay on the displays, so remove them in Corner Pin, Mesh and Edge Blend… if a display is used alone.
- A span can also be a grid (a video wall, four projectors): set Displays per row and a Vertical overlap; the displays fill the rows top to bottom, left to right, in their screen order.
- Slices: tick Set each display's part by hand to give the picture's size and, for each display, the rectangle of the picture it shows (x, y, width, height in pixels; it starts from the layout above). A part of another size than its display is scaled to fill it, so one display can show a zoomed part of the picture, or an LED processor can take its input from any part. Parts may overlap or leave gaps; blends are not set for them (set them in Corner Pin, Mesh and Edge Blend).
- Window › Output Effects… (Performance mode only): shader effects on everything a display shows — the same effects as a layer's (Invert, Tint, Mirror, Kaleidoscope, Wave, …), up to four in order — after its layers and before its corner pin, mesh and edge blend. It changes the output at once, also with blind editing, and is saved with the show (not in output presets). Parts of the display no layer covers stay black whatever the effect does to black, so a fade to black or Blackout still ends in black.
- Window › Output Presets…: save the output setup (the output, each layer's display, corner pins, meshes, edge blends and a span) under a name, and apply it to any show later, e.g. one preset per venue or rig. Presets are kept on this computer. Apply asks first, changes the output at once, and warns when the preset was made for other displays.
- Window › Test Pattern (Ctrl+T toggles the grid): Grid (edges, aspect circle, center), Color bars with a gray ramp, or White, drawn at each display's resolution. It asks before replacing a live output; Off puts the previous output back (its videos restart).

## Output Health

The line under the preview shows OK, WARN or ERROR with the output frame rate, late frames and errors. Click it, or Window › Output Health… (Ctrl+H), for details:

- Late frame: a gap longer than 1.5 output frames while video or a fade is on screen.
- Render cost, errors by type with the last message, how long cuts waited for videos, missing media, and which GPU draws the output.
- One row per video: on the output or preloaded, playing or held, hardware (HW) or software (SW) decoding, clip and decode frame rate, frames decoded, shown and dropped. In every render mode a video larger than 1920 × 1080 that is drawn at half its size or less (a 4K clip at Fit on a 1080p display) is scaled to that size as it is decoded, which costs the output much less; its row then says, for example, “scaled to 1920×1080”. Warp points on the layer keep its full size.
- Copy Report puts everything on the clipboard (useful for support); Open Log Folder opens the logs. The Process line shows the app's memory (now and the highest so far), threads and how long it has been running; on Windows also the GDI objects it holds (Windows stops drawing for a program at 10,000, so a number that keeps rising is a fault to report). The Python line shows how many objects the app holds and its garbage collector's pauses: while it collects, everything waits, so a pause longer than a frame (16.7 ms at 60 fps) is a late frame. The Effects line counts the shader effects on the output; one the graphics card could not compile turns the status to WARN with the compiler's message (the layer or display is shown without it).
- Displays: if a display or projector is plugged in or out while VisualDeck runs, the status turns WARN. Nothing moves by itself; Rescan Displays restarts the output with the displays connected now (a short black frame, videos start again).

## Missing media and Collect Show

- Missing files are marked [MISSING]. Files that moved together with the show are found again automatically when it opens.
- File › Relink Missing Media… (Ctrl+Shift+R): Search Folder… matches by file name and size and prefers the most similar folder; duplicates are left for you to choose with Locate Selected… (other missing files in the same folder are found too).
- File › Collect Show… copies the show and all its media into one new folder that can be moved to another computer, with the shader effects it uses that are not built in (in an effects folder next to the show). It never overwrites anything and can be canceled.
- Media list right-click: Remove from Project (the file stays on disk). For a missing file: Locate Missing File… and Relink Missing Media….
- Make Playback Copy… (right-click a video): a new file next to it with a key frame every half second (or every frame), and smaller if you choose (1920 × 1080 or 1280 × 720 at most). Camera and download files often have a key frame only every few seconds, so starting them anywhere else than at the start (Start Cue From, in points, reverse) means decoding everything since the last one; a copy starts at once and a smaller one is lighter to decode. With Use the copy in the cues, every cue that used the video uses the copy, at the same size on the output. The original file is not changed.

## Remote control

Preferences › Remote control. All are off by default; OSC and the browser remote only accept commands from this computer unless you allow the network. There is no password: only allow the network on one you control.

- OSC (UDP, default port 7700): /visualdeck/go, /back, /blackout, /update, /cue <cue number, position or name>, /ping (replies with the live cue), /effect <layer, input, value> and /display_effect <display, input, value> (an input of a shader effect on the output, also with blind editing; see docs/REMOTE_CONTROL.md).
- Browser remote (default port 7701): open http://<this computer>:7701/ on a phone or tablet for GO, BACK, Blackout, Update Output and a tappable cue list; Effects on the output (fold-out) has a slider, list or on/off for the inputs of the effects on the output.
- MIDI: Preferences › Remote control › MIDI Setup… – choose the input port, let Program Change select cues (program 1 = cue 1), and Learn a key or button for GO, BACK, Blackout and Update Output. Save applies it at once. Follow MIDI timecode (same window) runs cues at their Timecode (Cue Settings); the window shows the timecode coming in. Faders for effect inputs (same window): Add fader, choose a layer of the live cue or a display and the effect input (its name, or 2:amount for the second effect's), Learn, and the fader moves that input on the output.
- MIDI Show Control (MSC), for a lighting or sound desk or QLab: in MIDI Setup tick MIDI Show Control and set the device ID the desk sends to (0–111; messages to 127, everyone, are also taken). GO with a cue number goes to that cue (a cue number set in Cue Settings first, then the position), GO without one is GO, STOP and ALL_OFF fade to black (VisualDeck has no pause); other commands are ignored. It answers the Video, Projection and All types command formats (a Sound or Lighting cue is not for it). MIDI Setup and Output Health show the last MSC message.
- SMPTE LTC (timecode as audio, e.g. a track from the show playback or a timecode generator): Preferences › Remote control › LTC Setup… – tick Follow LTC, choose the audio input and the channel that carries it (Left, Right, or Mix). 24, 25, 29.97 drop and 30 fps are recognized. Cues run at their Timecode as with MIDI timecode, which is not followed while LTC is on. The window and Output Health show the timecode and the input level. On Windows, the Microphone privacy setting may have to allow desktop apps to use audio inputs.
- DMX from a lighting console: Preferences › Remote control › DMX Setup… – Art-Net or sACN, universe and start channel of a 3-channel fixture: cue select (value n goes to cue n), GO and Blackout (each fires when it rises to 128 or more). Values already being sent when listening starts do nothing.
- DMX to a lighting console: in the same window, Also send the live cue – the chosen channel carries the live cue's number (0 when nothing is on the output) and the next one 255 while something is on the output, so the console can follow the video.

Details: docs/REMOTE_CONTROL.md.

## OSC commands

OSC (Open Sound Control) is how show-control software (QLab, Companion and a Stream Deck, TouchOSC, a lighting console, Chataigne, a script) tells VisualDeck what to do over the network. Turn it on in Preferences › Remote control › Listen for OSC on port (UDP, default 7700). It accepts messages from this computer only, unless you tick Accept commands from other computers on the network. There is no password: anyone who can reach the port can run the show.

- **/visualdeck/go  (none)**: GO, as the GO button or the space bar: the next cue with its transition. With Presentation Mode enabled it starts the automatic run, or stops it while it runs. Example: /visualdeck/go
- **/visualdeck/back  (none)**: BACK: the previous cue that has media. Example: /visualdeck/back
- **/visualdeck/blackout  (none)**: Fade the output to black with the show's transition. Example: /visualdeck/blackout
- **/visualdeck/update  (none)**: Update Output: put edits waiting on the live cue on the output (blind editing). Example: /visualdeck/update
- **/visualdeck/cue  (cue number, position or name)**: Go to a cue with its transition. A cue number set in Cue Settings (12 or 12.5, as a number or a string) is looked up first, then the position (1 = first cue), then the name. Sending the live cue triggers it again. Ignored during a transition. Example: /visualdeck/cue 3   or   /visualdeck/cue "Act 2"
- **/visualdeck/start_from  (cue, seconds)**: Start Cue From (rehearsals): cut to the cue (as for /cue) with its videos and their sound that many seconds in, as Show › Start Cue From… does. Seconds as a number or "m:ss". Example: /visualdeck/start_from 4 "1:30"
- **/visualdeck/effect  (layer, input, value(s))**: Set an input of the shader effect on a layer of the cue on the output (layer 1 = first). The input by its name in the effect file or its label in the editor (any case); with several effects on the layer, 2:amount is the second effect's and Invert:amount the first Invert's. One number for a slider, a whole number or an on/off (not 0 = on); three or four numbers 0–1 for a color; two for a point. Kept within the input's range. Changes the output at once, also with blind editing, and the cue in the show. Example: /visualdeck/effect 1 amount 0.5
- **/visualdeck/display_effect  (display, input, value(s))**: The same for a display's effect (Window › Output Effects; display 0 = first). Saved with the show. Example: /visualdeck/display_effect 0 tint 1 0.5 0
- **/visualdeck/ping  (none)**: Replies /visualdeck/pong with the live cue's number and name to the sender's address and port: a show-control system can check that VisualDeck is running. Example: /visualdeck/ping

- Addresses are not case sensitive. A message VisualDeck does not understand (a wrong address, a cue that does not exist, the wrong arguments) is ignored and written to the log; the live show is not affected.
- Output Health shows whether VisualDeck is listening, on which port, and how many messages it has received; a port another program already uses shows as a warning there.
- Test from QLab: a Network cue with destination this computer's address, port 7700, and the message /visualdeck/go. From Companion: the generic OSC module with the same address and port.

## Browser remote and HTTP API

Preferences › Remote control › Serve a browser remote page on port (default 7701). Open http://localhost:7701/ on this computer, or tick Allow phones and other computers on the network and open http://<this computer's address>:7701/ on a phone or tablet (Output Health shows the address). The page has GO, BACK, Blackout, Update Output, the cue list, the live cue, a running follow countdown and, in a fold-out, the inputs of the effects on the output.

Scripts and show-control systems can use the same commands over HTTP. Every command is a POST with the header X-VisualDeck: 1 (other web pages cannot add it, so they cannot trigger cues):

- **GET /**: The control page: GO, BACK, Blackout, Update Output, the cue list and Effects on the output.
- **GET /api/status**: JSON: the live cue, the cue list, Blackout, edits waiting, a running follow, the cue clock and the effects on the output.
- **POST /api/go**: GO.
- **POST /api/back**: BACK.
- **POST /api/blackout**: Blackout.
- **POST /api/update**: Update Output.
- **POST /api/cue/<n>**: Go to cue n by position (1 = first), with its transition.
- **POST /api/cue/<n>/from/<seconds>**: Start Cue From: cue n (position) that many seconds in (12.5 or 1:02.5).
- **POST /api/effect/<layer>/<input>/<value>**: As /visualdeck/effect over OSC; a color or a point as numbers separated by commas (1,0.5,0).
- **POST /api/display_effect/<display>/<input>/<value>**: As /visualdeck/display_effect over OSC.

Example: curl -X POST -H "X-VisualDeck: 1" http://localhost:7701/api/go

## Preferences

File › Preferences… (Ctrl+comma) lists its pages on the left: click one, or use the arrow keys, to see its settings on the right. Save and Apply saves the settings of every page; Cancel or Esc discards them.

- Editing the live cue: wait for Update Output (recommended) or apply immediately.
- Remote control: OSC, the browser remote, MIDI, LTC and DMX (see Remote control).
- Audio: where the sound of videos plays, the sound delay and Test Sound (see Sound).
- Rendering: Performance Mode (OpenGL; best quality, blend modes, GPU preview), Balanced Mode (labeled Recommended) or Compatibility Mode (safe mode, CPU only). Advanced Performance Options (Performance Mode only): asynchronous PBO uploads, VSync (smoother, more latency), multi-threaded decoding.
- Find Best Setting For Me (Rendering) runs a short drawing test (Performance Test) in Performance and in Balanced Mode, each in its own process, and offers the faster one (Compatibility Mode if neither runs). It measures drawing speed only: effects, blend modes and warp points on the GPU need Performance Mode whatever it suggests.
- Video decoding: GPU Hardware Acceleration, Software decoding (default) or a hardware decoder; the Health window shows whether hardware decoding really engaged. Compatibility Mode always decodes in software.
- Video decoding › Test with a Video…: decodes a video you choose in software and with each hardware decoder this computer offers (for example D3D11VA on Windows) and shows the frames per second of each in the Decoding Test window, marking decoders that were not really used. Use it picks the fastest; then Save and Apply. A video that decodes below its own frame rate stutters.

Save and Apply restarts the output only when a setting on the Rendering or Video decoding page changed: the output redraws and the live cue starts again, so change those before the show. The other pages never touch the output or mark the show changed.

Preferences are saved for this computer (~/.visualdeck/config.json), not in the show.

## Keyboard shortcuts

Shortcuts are ignored while you type in a text field (except F1).

**Show**

| Key | Does |
|---|---|
| Space | GO: next cue (in presentation mode: start/stop) |
| Shift+Space | BACK: previous cue with media |
| Ctrl+B | Blackout: fade the output to black |
| Ctrl+U | Update Output: apply edits to the live cue |
| Ctrl+1 … Ctrl+9 | Go to cue 1–9 |
| Ctrl+E | Cue Settings of the live cue |
| Ctrl+P | Presentation mode on/off |
| Ctrl+L | Lock or unlock editing (GO, BACK and Blackout still work) |

**Output**

| Key | Does |
|---|---|
| Ctrl+T | Test pattern (grid) on/off |
| Ctrl+H | Output Health window |

**File**

| Key | Does |
|---|---|
| Ctrl+N | New show |
| Ctrl+O | Open show |
| Ctrl+S | Save show |
| Ctrl+Shift+S | Save show as… |
| Ctrl+Shift+R | Relink missing media |
| Ctrl+, | Preferences |

**Edit**

| Key | Does |
|---|---|
| Ctrl+Z | Undo the last change to the show |
| Ctrl+Shift+Z | Redo (also Ctrl+Y) |

**Insert**

| Key | Does |
|---|---|
| Ctrl+I | Add media to the show |
| Ctrl+Shift+C | Add a cue (column) |
| Ctrl+Shift+L | Add a layer (row) |

**Help**

| Key | Does |
|---|---|
| F1 | Help |

## Troubleshooting

- Black output: check Output (above the preview), that the cue has media, and the Health window (missing media, errors).
- Stutter: look at late frames and the decoder rows in Output Health; try Performance Mode, a hardware decoder, or Make Playback Copy… (right-click the video in the media list; a 1080p copy of a 4K clip). Very heavy files (for example 4K at a high bit rate) may need converting to a lighter format beforehand with a tool such as FFmpeg or HandBrake.
- On laptops with two GPUs, the Health window shows which one draws the output.
- Logs: Help › Open Log Folder (~/.visualdeck/app.log). Times include the UTC offset.
- Help › Soak Test…: before a long show or an installation, let VisualDeck play the open show by itself (GO every few seconds, each cue with its own transition) for as long as you choose, and record memory, threads, decoders, GPU textures, frame rate, late frames, errors, sound underruns, Python objects and garbage collector pauses (and which kinds of objects grew). At the end it says in plain words whether anything kept growing or went wrong (memory is compared at the same cue, since cues differ, and only in runs of 11 minutes or more: before that it is still warming up), and leaves a CSV and a summary in ~/.visualdeck/soak/. It plays on the output; nothing in the show is changed or saved. If the show stops after its last cue, the run starts again from the first. Closing the window does not stop a run; Stop does.

## Help menu

- Help › VisualDeck Help (F1): this window. Type in the box above the topics to find a word in every topic; the topic list shows only the topics that contain it, with the word highlighted. A ? next to a setting opens the topic about it.
- Help › Keyboard Shortcuts: the Keyboard shortcuts topic.
- Help › Open Log Folder: the folder with app.log (what VisualDeck did and any errors, with times).
- Help › Soak Test…: see Troubleshooting. Help › About VisualDeck: the version and the license.

## About VisualDeck

VisualDeck 1.0 (version 1.0.0), Community Edition, is a live-visuals and media-server application, free software under the Mozilla Public License 2.0 (MPL-2.0): see LICENSE in the installation folder, and THIRD_PARTY_NOTICES.md for the libraries it includes and their licenses.

Built with Python, Tk, pygame/SDL, PyOpenGL, PyAV (FFmpeg), NumPy, OpenCV, Pillow and python-osc.
