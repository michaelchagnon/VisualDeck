# Remote control (OSC, browser, MIDI and DMX)

## OSC

Off by default. Turn it on in **Preferences › Remote control**: VisualDeck then listens for OSC messages over UDP on the chosen port (default 7700). By default only programs on the same computer can send commands; tick **Accept commands from other computers on the network** to allow a tablet, phone app or show-control computer. Anyone who can reach that port can then trigger cues: there is no password, so only enable it on a show network you control.

| Address | Arguments | Does |
|---|---|---|
| `/visualdeck/go` | – | GO (same as the GO button or the space bar; in presentation mode, starts/stops it) |
| `/visualdeck/back` | – | BACK: previous cue with media |
| `/visualdeck/blackout` | – | Fade the output to black |
| `/visualdeck/update` | – | Update Output (apply edits to the live cue) |
| `/visualdeck/cue` | cue number, position or name | Go to the cue with that **cue number** (Cue Settings › Number, e.g. `12.5`, sent as a string or a float), else to position *n* (1 = first), else to the cue with that name, with that cue's transition. Sending the live cue re-triggers it |
| `/visualdeck/start_from` | cue (as for `/cue`), seconds | Start Cue From: cut to that cue with its videos and their sound that many seconds in (a number, or `"m:ss"` as a string), as Show › Start Cue From… does. Ignored during a transition |
| `/visualdeck/effect` | layer, input, value(s) | Set an input of the shader effect on that layer (1 = first) of the cue on the output: one number for a slider, a whole number or an on/off (not 0 = on), three or four numbers 0–1 for a color, two for a point. The input by its name in the effect file (`amount`, `sides`, …) or its label as the editor shows it (any case): with several effects on the layer, the first effect that has that input; `2:amount` is the second effect's, `Invert:amount` the first Invert's. Values are kept within the input's range; a list input (Mirror's Keep) only takes its own values. The output changes at once, also with blind editing (it is playing the show, like a fader), and so does the cue in the show when its layer has the same effect; other edits waiting for Update Output keep waiting. Many messages in a row are sent to the output once per 33 ms |
| `/visualdeck/display_effect` | display, input, value(s) | The same for a display's effect (Window › Output Effects; display as the app numbers them, 0 = first); saved with the show |
| `/visualdeck/ping` | – | Replies `/visualdeck/pong [live cue number, live cue name]` to the sender's address and port |

Commands arriving before the output is ready, for cues that do not exist, or during a transition (`/cue`) are ignored and logged; so are effect messages for a layer or display without an effect, an input it does not have, or the wrong count of numbers. Every command is logged (effect messages only at debug level, as a fader sends many) with the sender's address (`~/.visualdeck/app.log`, logger `visualdeck.control`). **Window › Output Health** shows whether VisualDeck is listening and how many messages it has received; a port that is already in use shows as a warning there.

Example from a terminal with python-osc installed:

```python
from pythonosc.udp_client import SimpleUDPClient
c = SimpleUDPClient("127.0.0.1", 7700)
c.send_message("/visualdeck/cue", 3)
c.send_message("/visualdeck/go", [])
```

## Browser remote

Off by default. In **Preferences › Remote control** tick **Serve a browser remote page on port** (default 7701). On this computer open `http://localhost:7701/`; to use a phone or tablet, also tick **Allow phones and other computers on the network to open it** and open `http://<this computer's IP address>:7701/` there (the Health window shows the address it listens on).

The page has GO, BACK, Blackout, Update Output (enabled only while the live cue has unapplied edits) and the cue list: tap a cue to go to it. It shows the live cue, OUTPUT BLACK, a running follow countdown and a test-pattern notice, refreshed about once a second.

Safety: commands must be sent by the page itself (a POST with an `X-VisualDeck` header), so other websites open in the same browser cannot trigger cues; requests addressed to other host names are refused (DNS-rebinding protection). There is **no password**: anyone who can open the page can run the show, so keep network access off unless the network is yours.

**Effects on the output** (a fold-out section, shown when the live cue's layers or the displays have effects): a slider, list or on/off for each such input of every effect, as the editor names them; moving one changes the output at once (as the OSC and API effect commands do), and values changed elsewhere (MIDI, the editor, another phone) show up within a second. Colors and points are not on the page (the API takes them).

For scripts: `POST /api/go`, `/api/back`, `/api/blackout`, `/api/update`, `/api/cue/<n>`, `/api/cue/<n>/from/<seconds>` (Start Cue From; *n* is the position, 1 = first; seconds as `12.5` or `1:02.5`), `/api/effect/<layer>/<input>/<value>` and `/api/display_effect/<display>/<input>/<value>` (as the OSC messages; a color or a point as numbers separated by commas, e.g. `/api/display_effect/0/tint/1,0.5,0`) with the header `X-VisualDeck: 1`; `GET /api/status` returns JSON.

## MIDI

Off by default. In **Preferences › Remote control** press **MIDI Setup…**:

- **Listen for MIDI input** and the **Input port** (a USB controller, a MIDI interface, or a virtual port such as loopMIDI on Windows; **Refresh** lists them again).
- **Program Change selects the cue**: program 1 goes to the cue *numbered* 1 (Cue Settings › Number) or, if no cue has that number, to the first cue; program 2 likewise, and so on, on any channel, with that cue's transition. Note that devices show programs as 1–128 while the message carries 0–127; VisualDeck follows the 1–128 numbering.
- **Triggers** for GO, BACK, Blackout and Update Output: press **Learn**, then the key or button on the controller (within 15 s). A trigger is a Note On (any velocity above 0) or a Control Change that rises to 64 or more, so a button fires once per press and a fader pushed up fires once. Learning a key that another action had moves it. **Clear** removes a trigger.
- **Follow MIDI timecode**: VisualDeck reads MTC (quarter frames and full-frame messages) from the same input and goes to a cue, with its transition, when the running timecode reaches the cue's **Timecode** (Cue Settings, `HH:MM:SS:FF`). Only running through a time fires a cue: a locate, rewind or start in the middle changes nothing, and if several cue times are passed in one step (more than one in 1 s) only the last fires. The live cue does not fire again. A time reached during a fade runs when the fade ends. Rates 24, 25, 29.97 drop-frame (counted as 30 frames per timecode second, like the labels) and 30. The setup window and Output Health show the incoming timecode.
- **Faders for effect inputs** (up to 8): **Add fader**, choose **Layer** (of the live cue, 1 = first) or **Display** (0 = first) and the effect **input** (its name or label; with several effects on that layer or display, `2:amount` is the second effect's, `Invert:amount` the first Invert's), then **Learn** and move the fader or knob up past the middle. From then on every value the Control Change sends (0–127) sets that input on the output at once, also with blind editing, and in the show (as `/visualdeck/effect` over OSC does): a slider goes across its range, a list (such as Mirror's Keep) steps through its values, an on/off is on from 64. A color or a point cannot be set by one fader. A Control Change that is a fader no longer fires a trigger; learning a trigger on a fader's Control Change (or the reverse) moves it, and the window says so.
- **MIDI Show Control (MSC)** with a **device ID** (0–111): System Exclusive show-control messages (`F0 7F <device> 02 <format> <command> <data> F7`, MIDI 1.0 Recommended Practice RP-002) from a lighting or sound desk or QLab. Messages to the device ID or to 127 (all-call) with the command format Video (0x30–0x3F), Projection (0x40–0x4F) or All types (0x7F) are taken; group IDs (112–126) are not. **GO** (and TIMED_GO) with a cue number in its data goes to that cue (a Cue Settings number first, then the position, as `/visualdeck/cue`); without one it is GO. **STOP** and **ALL_OFF** fade to black (VisualDeck has no pause). LOAD, RESUME, SET, FIRE, RESTORE, RESET and GO_OFF are logged and ignored. The last MSC message is shown in MIDI Setup's status line and Output Health. Tested with messages built to that layout through the stand-in MIDI library; no desk or real MIDI interface has sent MSC to VisualDeck yet.
- **Save** applies the settings at once (independently of the Preferences window's Save). **Cancel** keeps the saved ones.

Settings are stored for this computer in `~/.visualdeck/config.json` (`midi_enabled`, `midi_port`, `midi_mapping`, `midi_program_change`, `midi_timecode`, `midi_faders`, `midi_msc`, `midi_msc_device`). The port is found again by name; if Windows renumbers it (e.g. "nanoKONTROL2 1" becomes "nanoKONTROL2 2" after replugging) the same device is used as long as only one matches.

On Windows a MIDI input can be open in one program at a time: if another program has the port, VisualDeck cannot open it, and **Output Health** shows "MIDI not listening: cannot open MIDI input". While **Learn** waits, VisualDeck's own MIDI input is closed and reopened afterwards. Unplugging the device while VisualDeck runs is not detected; open MIDI Setup and press Save to reconnect.

Every command is logged as `Remote go from MIDI` and so on. MIDI was tested against a stand-in for the MIDI library on a machine without MIDI hardware; real devices have not been tested yet.

## SMPTE LTC (timecode as audio)

Off by default. In **Preferences › Remote control** press **LTC Setup…**: tick **Follow LTC**, choose the **Audio input** (*System default* is the first input SDL lists) and the **Channel** that carries the timecode (Left, Right, or Mix for a mono source on both). **Save** opens the input at once.

Cues then run at their **Timecode** (Cue Settings) by the same rules as MIDI timecode above: only running through a time fires, a locate changes nothing, a time reached during a fade runs when it ends. While Follow LTC is on, MIDI timecode is not followed (MIDI triggers still work); switching between them never fires a cue by itself.

The decoder (`src/ltc.py`) reads the biphase-mark signal at 48 kHz from any level (it measures the signal's range over the last 20 ms), with either polarity and a DC offset, and follows speed changes (tested ±8 %). A frame's time is known when its last bit has arrived, so the time used is that frame plus one. The frame rate is counted at the first second boundary (frame 24 → 0 means 25 fps); until then it is estimated from the bit rate, so in the first second a slowed 25 fps source can read as 24 fps. 29.97 drop-frame is recognized by its flag and, like MTC, counted as 30 frames per timecode second. Timecode running backwards is recognized and ignored. A gap in the signal (silence) drops the frame it cut off; the next whole frame locks again.

The setup window, the Preferences line and **Output Health** show the input, the incoming timecode (when it is running) and the input level; Output Health warns when the input cannot be opened. Settings are stored in `~/.visualdeck/config.json` (`ltc_enabled`, `ltc_device`, `ltc_channel`).

Tested here with LTC generated by `ltc.encode()` (all four rates, 44.1 to 96 kHz, noise, DC offset, inverted, ±8 % speed, dropouts, backwards) and in the real app through SDL's audio capture (its file-reading driver): with LTC from 09:59:58:00, cues at 10:00:00:00 and 10:00:03:00 fired on those frames, 1.96–1.98 s and 4.88–4.95 s after the input opened (that driver delivers audio slightly faster than real time), in Performance and Balanced Mode. **No real LTC source, sound card input or Windows audio input has been used.** On Windows, *Settings › Privacy & security › Microphone* may have to allow desktop apps to use audio inputs (not checked here).

## DMX (Art-Net and sACN)

Off by default. In **Preferences › Remote control** press **DMX Setup…**: tick **Listen for DMX from a lighting console**, choose **Art-Net** (UDP 6454; universe = port address, 0 = the first) or **sACN (E1.31)** (UDP 5568; universes 1–63999; VisualDeck joins the universe's multicast group and also accepts unicast), the **Universe** and the **Start channel**. **Listen on the network** is on by default because a console is normally another computer; untick it to accept only software on this computer.

Patch a 3-channel fixture on the console at the start channel:

| Channel | Does |
|---|---|
| start | **Cue select**: when the value changes to *n* (1–255), go to the cue *numbered* *n* (Cue Settings › Number) or, if none has that number, the *n*-th cue, with its transition. 0 does nothing. |
| start + 1 | **GO** when the value rises to 128 or more |
| start + 2 | **Blackout** when the value rises to 128 or more |

The values a console is already sending when VisualDeck starts listening only set the starting point; nothing happens until they change. A console that repeats the same frame triggers once. Preview packets (sACN) and non-DMX start codes are ignored. If two consoles send the same universe, the last packet wins (no HTP/LTP merging). The setup window shows the three values live, and **Output Health** shows them with the number of commands; a port already used by another program is shown there as a warning. On Windows, other Art-Net software on the same computer can share the port.

DMX was tested with packets built to the Art-Net 4 ArtDmx and E1.31 layouts and sent over UDP on this computer; no lighting console has been used yet.

### Sending the live cue (DMX out)

Also in **DMX Setup**: tick **Also send the live cue** to let a lighting console follow the video. VisualDeck sends one universe (Art-Net or sACN, the protocol chosen above) where the chosen **Channel** carries the live cue's number (its Cue Settings number, else its position from 1; 0 when nothing is on the output or after Blackout) and the next channel is 255 while something is on the output, 0 otherwise. It is sent when that changes and once a second. Art-Net goes to the address you give (a console's address, or 255.255.255.255 for every computer on this network); sACN goes to the universe's multicast group (239.255.x.y). Settings are stored in `~/.visualdeck/config.json` (`dmx_out_*`).

Tested with packets read back by VisualDeck's own Art-Net and sACN parsers, and in the real app with an Art-Net receiver on this computer (cue numbers, Blackout, about one packet a second); **no console or DMX node has received them**.

