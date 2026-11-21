# Known issues and limitations (VisualDeck 1.0)

## Not yet tested with real equipment

- **Show-control hardware.** OSC, MIDI (notes, Program Change, MIDI Show Control, MIDI timecode), SMPTE LTC and Art-Net/sACN were tested with simulated input only: messages built to each standard and sent on the same computer. Please report anything that a real console, desk or MIDI device does differently.
- **Projectors.** Corner pin, mesh warp, edge blending and spans were checked on monitors and a second display output, not yet on real projectors. Check edge blends on your own screen and adjust the blend gamma if the overlap looks brighter.
- **Hardware decoding** (Preferences › GPU Hardware Acceleration) is off by default. On the integrated GPU it was tried on, D3D11VA was slower than software decoding.

## Render modes and performance

- **Shader effects and blend modes** (Add, Multiply, Screen) need **Performance Mode**. Balanced and Compatibility Mode draw those layers without them, and Output Health says so.
- **4K video with an effect** on an integrated GPU runs at about 1 % late frames. For heavy clips, use **Make Playback Copy** at 1920 × 1080; the copy with an effect stayed under 1 %.
- **Balanced Mode** with a 4K video at full size, or with a corner pin, may not hold 60 frames a second. The output then steps down to 45 or 30 automatically. Performance Mode is the fastest.
- Color correction and masks in Balanced and Compatibility Mode run on the CPU for every video frame (a few milliseconds per 1080p frame).
- **Memory** can rise once by about 90 MB a few hours into a long run, then stays level. In the 72-hour test the overall growth was 0.1 MB an hour.

## Sound

- Sound is lined up with the picture when a cue starts and at every loop. There is no continuous shared clock.
- At playback speeds other than 1×, the video plays without sound.
- Sound from live sources plays as it arrives, about 60–250 ms behind the picture. The sound delay setting doesn't apply to it.
- Plugging in an HDMI display can change Windows' default sound device. Check **Preferences › Audio** after connecting displays.

## Cues and editing

- Only the next and previous cues are preloaded, plus cues marked **Keep Ready**. Jumping to any other cue starts its videos from cold, which takes up to about 350 ms without a black frame.
- A file dropped on a cell is shown at its own pixel size (100 %), so a 4K clip on a 1080p display shows only its middle. Use the position editor's Fit or Fill.
- **Relink** searches by file name. A renamed file has to be located by hand.
- Undo doesn't cover output settings (displays, corner pins, meshes, edge blends) or the show-wide transition and Presentation Mode settings.
- Turning a test pattern off restarts the videos of the cue it covered.
- Grid thumbnails don't show warp points.
- The control window needs about 700 pixels of height; on shorter windows some settings below GO are cut off.

## Outputs

- **VSync is off by default**, so the output can tear on projectors and LED walls. Turn it on in Preferences.
- Corner-pinned edges aren't anti-aliased, so a strongly angled edge shows small steps.
- The schedule uses the computer's clock while VisualDeck is open. It doesn't wake a sleeping computer or start VisualDeck.

## Security

- OSC and the browser remote have **no password**. They listen only on this computer unless you allow network access. Allow it only on a show network you trust.
- A show file connects to its live-source addresses (cameras, streams) when their cue is live or next. Check shows you receive from others before opening them on a networked show computer.

## Not included in 1.0

- NDI, Spout and Syphon; MIDI feedback to controller lights; rotated or masked slices; 3D projection mapping; a timeline mode.
- macOS and Linux releases (the source runs on Linux for development).
