# SPDX-License-Identifier: MPL-2.0
"""DMX cue control from a lighting console, over Art-Net or sACN (E1.31).

Off by default (Preferences > Remote control > DMX Setup...). VisualDeck
takes three consecutive channels of one universe, starting at the chosen
channel (patch it on the console as a 3-channel fixture):

  start     cue select: when the value changes to n (1-255), go to cue n
            (the cue *numbered* n if there is one, else the n-th cue); 0 does
            nothing
  start+1   GO: fires when the value rises to 128 or more
  start+2   Blackout: fires when the value rises to 128 or more

The first packet only sets the starting values: whatever the console is
already sending when VisualDeck starts listening does not trigger anything.
After that only changes count, so a console that repeats the same frame 40
times a second triggers once.

Art-Net: ArtDmx (OpCode 0x5000) on UDP 6454, universe = 15-bit port address
(Net/Sub-Net/Universe), 0 = the first. sACN: E1.31 data packets on UDP 5568,
universes 1-63999; VisualDeck joins the universe's multicast group
(239.255.x.y) and also accepts unicast. Preview packets and other start
codes are ignored. When several consoles send the same universe the last
packet wins (no merging).

The listener thread never touches the UI: commands go to `on_command`, which
must only enqueue (see ControlInterface._poll_remote).
"""
from __future__ import annotations

import logging
import socket
import struct
import threading
import time

log = logging.getLogger("visualdeck.dmx")

ARTNET_PORT = 6454
SACN_PORT = 5568
PROTOCOLS = ("artnet", "sacn")
FOOTPRINT = 3
_ARTNET_ID = b"Art-Net\x00"
_ACN_ID = b"ASC-E1.17\x00\x00\x00"


def parse_artnet(packet):
    """(universe, dmx bytes) for an ArtDmx packet, else None."""
    if len(packet) < 18 or packet[:8] != _ARTNET_ID:
        return None
    opcode = struct.unpack_from("<H", packet, 8)[0]
    if opcode != 0x5000:
        return None
    universe = packet[14] | ((packet[15] & 0x7F) << 8)
    length = struct.unpack_from(">H", packet, 16)[0]
    if not 1 <= length <= 512 or len(packet) < 18 + length:
        return None
    return universe, bytes(packet[18:18 + length])


def parse_sacn(packet):
    """(universe, dmx bytes) for an E1.31 data packet (start code 0, not
    preview), else None."""
    if len(packet) < 126 or packet[4:16] != _ACN_ID:
        return None
    if struct.unpack_from(">I", packet, 18)[0] != 0x00000004:        # root: E1.31 data
        return None
    if struct.unpack_from(">I", packet, 40)[0] != 0x00000002:        # framing: DMP
        return None
    options = packet[112]
    if options & 0x40:                                                 # preview data
        return None
    universe = struct.unpack_from(">H", packet, 113)[0]
    if packet[117] != 0x02 or packet[118] != 0xA1:
        return None
    count = struct.unpack_from(">H", packet, 123)[0]
    if count < 1 or len(packet) < 125 + count:
        return None
    if packet[125] != 0x00:                                            # not DMX512 null start code
        return None
    return universe, bytes(packet[126:125 + count])


def build_artnet(universe, data, sequence=0):
    """An ArtDmx packet (for tests and the setup dialog's self-test)."""
    data = bytes(data)
    if len(data) % 2:
        data += b"\x00"
    return (_ARTNET_ID + struct.pack("<H", 0x5000) + bytes([0, 14, sequence & 0xFF, 0, universe & 0xFF,
                                                           (universe >> 8) & 0x7F]) + struct.pack(">H", len(data)) + data)


def build_sacn(universe, data, sequence=0, preview=False, source="VisualDeck test"):
    data = bytes(data)
    n = len(data) + 1
    dmp = struct.pack(">HBBHHH", 0x7000 | (10 + n), 0x02, 0xA1, 0, 1, n) + b"\x00" + data
    name = source.encode("utf-8")[:63].ljust(64, b"\x00")
    framing = (struct.pack(">HI", 0x7000 | (77 + len(dmp)), 0x00000002) + name + bytes([100]) + b"\x00\x00"
               + bytes([sequence & 0xFF, 0x40 if preview else 0]) + struct.pack(">H", universe) + dmp)
    root = (struct.pack(">HH", 0x0010, 0) + _ACN_ID + struct.pack(">HI", 0x7000 | (22 + len(framing)), 4)
            + b"\x01" * 16 + framing)
    return root


class Decoder:
    """Turns the three channels into commands (changes only; first frame is the baseline)."""

    def __init__(self, channel=1):
        self.channel = max(1, min(512 - FOOTPRINT + 1, int(channel)))
        self.values = None

    def feed(self, data):
        i = self.channel - 1
        new = tuple(data[i + k] if i + k < len(data) else 0 for k in range(FOOTPRINT))
        old, self.values = self.values, new
        if old is None:
            return []
        out = []
        if new[0] != old[0] and new[0] >= 1:
            out.append(("cue", new[0]))
        if new[1] >= 128 > old[1]:
            out.append(("go", None))
        if new[2] >= 128 > old[2]:
            out.append(("blackout", None))
        return out


class DMXRemote:
    def __init__(self, on_command, protocol="artnet", universe=0, channel=1, network=True):
        """on_command(command, argument) is called on the listener thread; it must only enqueue."""
        self.on_command = on_command
        self.protocol = protocol if protocol in PROTOCOLS else "artnet"
        self.universe = int(universe)
        self.decoder = Decoder(channel)
        self.network = bool(network)
        self.port = ARTNET_PORT if self.protocol == "artnet" else SACN_PORT
        self._sock = None
        self._thread = None
        self._stop = threading.Event()
        self.error = None
        self.packets = 0           # packets for our universe
        self.commands = 0
        self.last_packet = None    # monotonic time

    @property
    def listening(self):
        return self._sock is not None

    def start(self, port=None):
        """Open the socket. Returns True, or False with self.error set."""
        port = self.port if port is None else port
        host = "0.0.0.0" if self.network else "127.0.0.1"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.settimeout(0.2)
            if self.protocol == "sacn" and self.network and 1 <= self.universe <= 63999:
                group = f"239.255.{self.universe >> 8}.{self.universe & 0xFF}"
                try:
                    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                                 socket.inet_aton(group) + socket.inet_aton("0.0.0.0"))
                except OSError as e:          # no multicast route: unicast still works
                    log.warning(f"sACN: could not join {group} ({e}); only unicast packets will arrive")
        except OSError as e:
            self.error = f"cannot listen on {host}:{port}: {e}"
            log.error(f"DMX remote control: {self.error}")
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass
            return False
        self._sock, self.error = s, None
        self.port = s.getsockname()[1]
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="DMXRemote", daemon=True)
        self._thread.start()
        log.info(f"DMX remote control: {self.protocol} universe {self.universe}, channels "
                 f"{self.decoder.channel}-{self.decoder.channel + FOOTPRINT - 1}, on {host}:{self.port}")
        return True

    def stop(self):
        self._stop.set()
        s, self._sock = self._sock, None
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        if s is not None:
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass

    def handle_packet(self, packet):
        parsed = parse_artnet(packet) if self.protocol == "artnet" else parse_sacn(packet)
        if parsed is None or parsed[0] != self.universe:
            return
        self.packets += 1
        self.last_packet = time.monotonic()
        for cmd, arg in self.decoder.feed(parsed[1]):
            self.commands += 1
            try:
                self.on_command(cmd, arg)
            except Exception:  # noqa: BLE001 - never let a handler kill the listener
                log.exception("DMX command could not be queued")

    def _run(self):
        s = self._sock
        while not self._stop.is_set() and s is not None:
            try:
                packet, _addr = s.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self.handle_packet(packet)
            except Exception:  # noqa: BLE001
                log.exception("DMX packet could not be handled")


class DMXOutput:
    """Sends the live cue on DMX so a lighting console can follow the video:
    channel `channel` = the live cue's number (1-255; 0 when the output is
    black or off), channel + 1 = the output level (255 at full, lower during
    a fade to black). Sent when it changes and every KEEPALIVE_S (Art-Net
    and sACN receivers expect a refresh). Art-Net goes to `target` (a
    console's address or a broadcast address); sACN to its multicast group."""

    KEEPALIVE_S = 1.0

    def __init__(self, protocol="artnet", universe=0, channel=1, target="255.255.255.255", sock=None):
        self.protocol = protocol if protocol in PROTOCOLS else "artnet"
        self.universe = max(0, int(universe))
        self.channel = min(511, max(1, int(channel)))
        self.target = target or "255.255.255.255"
        self._sock = sock
        self._seq = 0
        self._last = None
        self._last_t = 0.0
        self.sent = 0
        self.error = None

    def address(self):
        if self.protocol == "sacn":
            return (f"239.255.{(self.universe >> 8) & 0xFF}.{self.universe & 0xFF}", SACN_PORT)
        return (self.target, ARTNET_PORT)

    def _socket(self):
        if self._sock is None:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            if self.protocol == "sacn":
                s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
            self._sock = s
        return self._sock

    def frame(self, cue, level):
        data = bytearray(512)
        data[self.channel - 1] = max(0, min(255, int(cue or 0)))
        data[self.channel] = max(0, min(255, int(round(level * 255))))
        return bytes(data)

    def update(self, cue, level, now):
        """Send if (cue, level) changed or the keepalive is due. True when sent."""
        state = (int(cue or 0), round(float(level), 3))
        if state == self._last and now - self._last_t < self.KEEPALIVE_S:
            return False
        data = self.frame(*state)
        self._seq = (self._seq + 1) % 256 or 1
        pkt = build_sacn(self.universe, data, self._seq, source="VisualDeck") if self.protocol == "sacn" \
            else build_artnet(self.universe, data, self._seq)
        try:
            self._socket().sendto(pkt, self.address())
            self.sent += 1
            self.error = None
        except OSError as e:
            self.error = f"{type(e).__name__}: {e}"[:200]
        self._last, self._last_t = state, now
        return True

    def stop(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
