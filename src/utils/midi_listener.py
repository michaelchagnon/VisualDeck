# =====================================================================
# src/utils/midi_listener.py
# =====================================================================

import rtmidi
import sys

class MidiListener:
    """Listens to MIDI input and calls a callback."""
    def __init__(self, callback):
        self.callback = callback
        try:
            self.midi_in = rtmidi.MidiIn()
        except Exception as e:
            print(f"Error initializing MIDI input: {e}", file=sys.stderr)
            self.midi_in = None
            return

        ports = self.midi_in.get_ports()
        if ports:
            try:
                self.midi_in.open_port(0)
                print(f"Opened MIDI port: {ports[0]}")
            except Exception as e:
                print(f"Error opening MIDI port '{ports[0]}': {e}", file=sys.stderr)
                self.midi_in = None
        else:
            try:
                self.midi_in.open_virtual_port("VisualDeck Virtual MIDI")
                print("Opened virtual MIDI port.")
            except NotImplementedError:
                print("Virtual MIDI ports not supported on this platform. MIDI disabled.")
                self.midi_in = None
            except Exception as e:
                print(f"Error opening virtual MIDI port: {e}", file=sys.stderr)
                self.midi_in = None

    def start(self):
        """Begin listening for MIDI messages if available."""
        if not self.midi_in:
            return
        try:
            while True:
                msg = self.midi_in.get_message()
                if msg:
                    message, _ = msg
                    self.callback(message)
        except Exception as e:
            print(f"MIDI listener encountered an error: {e}", file=sys.stderr)
