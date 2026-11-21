# SPDX-License-Identifier: MPL-2.0
"""MIDI timecode (MTC): decoding, cue times, and when to fire a cue.

A cue can have a timecode, "HH:MM:SS:FF" (Cue Settings > Timecode). With
Preferences > MIDI Setup > "Follow MIDI timecode" on, VisualDeck reads MTC
from the MIDI input and goes to a cue (with its transition) when the running
timecode reaches the cue's time.

Rules, chosen so following timecode never surprises the operator:
- A cue fires when the timecode *runs through* its time: from below it to at
  or past it, in a step of at most JUMP_S. Jumping (locate, rewind, starting
  in the middle) changes nothing; the next cue time reached while running
  fires.
- If several cue times are passed in one step, only the last one fires.
- The cue that is already live does not fire again.
- Frames are counted at the rate the MTC says (24, 25, 29.97 drop-frame
  is treated as 30 frames per timecode second, or 30).

MTC: eight quarter-frame messages (0xF1) carry one time over two frames; a
full-frame SysEx (F0 7F 7F 01 01 hh mm ss ff F7) is sent on locate. No UI here.
"""
from __future__ import annotations

import re

RATES = {0: 24, 1: 25, 2: 30, 3: 30}            # rate code -> frames per timecode second
RATE_LABELS = {0: "24", 1: "25", 2: "29.97 drop", 3: "30"}
JUMP_S = 1.0
_TC_RE = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2})[:;.](\d{2})\s*$")


def parse(text):
    """(h, m, s, f) from "HH:MM:SS:FF" (";" or "." also accepted before the frames), else None."""
    m = _TC_RE.match(str(text or ""))
    if not m:
        return None
    h, mi, s, f = (int(x) for x in m.groups())
    return (h, mi, s, f) if h < 24 and mi < 60 and s < 60 and f < 30 else None


def clean(text):
    """Normalized "HH:MM:SS:FF", or None."""
    t = parse(text)
    return None if t is None else "%02d:%02d:%02d:%02d" % t


def fmt(tc):
    return "%02d:%02d:%02d:%02d" % tuple(tc[:4])


def seconds(tc, fps):
    h, m, s, f = tc[:4]
    return h * 3600 + m * 60 + s + f / float(fps)


def add_frames(tc, n, fps):
    h, m, s, f = tc[:4]
    total = ((h * 60 + m) * 60 + s) * fps + f + n
    total %= 24 * 3600 * fps
    f = total % fps
    s = total // fps
    return (s // 3600, s // 60 % 60, s % 60, f)


class MTCDecoder:
    """Feed raw MIDI messages; get (h, m, s, f, rate_code) when a time is known."""

    def __init__(self):
        self._pieces = [None] * 8
        self._expect = 0

    def feed(self, msg):
        if not msg:
            return None
        if msg[0] == 0xF1 and len(msg) >= 2:
            return self._quarter(msg[1])
        if (msg[0] == 0xF0 and len(msg) >= 10 and list(msg[1:5]) == [0x7F, 0x7F, 0x01, 0x01]):
            hh, mm, ss, ff = msg[5], msg[6], msg[7], msg[8]
            rate = (hh >> 5) & 0x03
            self._pieces = [None] * 8          # a locate: quarter frames start over
            self._expect = 0
            tc = (hh & 0x1F, mm & 0x3F, ss & 0x3F, ff & 0x1F)
            return tc + (rate,) if parse(fmt(tc)) is not None else None
        return None

    def _quarter(self, data):
        piece, value = (data >> 4) & 0x07, data & 0x0F
        if piece != self._expect:              # out of order (or running backwards): start again at piece 0
            self._pieces = [None] * 8
            self._expect = 0
            if piece != 0:
                return None
        self._pieces[piece] = value
        self._expect = (piece + 1) % 8
        if piece != 7 or None in self._pieces:
            return None
        p = self._pieces
        f = p[0] | ((p[1] & 0x01) << 4)
        s = p[2] | ((p[3] & 0x03) << 4)
        m = p[4] | ((p[5] & 0x03) << 4)
        h = p[6] | ((p[7] & 0x01) << 4)
        rate = (p[7] >> 1) & 0x03
        self._pieces = [None] * 8
        if parse(fmt((h, m, s, f))) is None:
            return None
        # The eight pieces took two frames to send: the time now is two frames later.
        return add_frames((h, m, s, f), 2, RATES[rate]) + (rate,)


def quarter_frames(tc, rate=1):
    """The eight 0xF1 messages that encode `tc` (for tests and loopback senders)."""
    h, m, s, f = tc[:4]
    vals = [f & 0x0F, f >> 4, s & 0x0F, s >> 4, m & 0x0F, m >> 4, h & 0x0F, (h >> 4) | (rate << 1)]
    return [[0xF1, (i << 4) | v] for i, v in enumerate(vals)]


def full_frame(tc, rate=1):
    h, m, s, f = tc[:4]
    return [0xF0, 0x7F, 0x7F, 0x01, 0x01, h | (rate << 5), m, s, f, 0xF7]


def cue_to_fire(cue_times, prev_s, now_s, live_col):
    """The column to go to when the timecode moved from prev_s to now_s
    (seconds), or None. cue_times: [(seconds, col)]."""
    if prev_s is None or now_s <= prev_s or now_s - prev_s > JUMP_S:
        return None
    passed = [(t, c) for t, c in cue_times if prev_s < t <= now_s]
    if not passed:
        return None
    col = max(passed)[1]
    return None if col == live_col else col
