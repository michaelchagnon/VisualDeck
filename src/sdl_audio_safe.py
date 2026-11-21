# SPDX-License-Identifier: MPL-2.0
"""Pause and close SDL audio devices without deadlocking.

pygame._sdl2.audio.AudioDevice.pause() and close() call SDL while holding
Python's global interpreter lock. SDL takes the device's lock for them, and
SDL's audio thread holds that same lock while it runs our Python callback,
for which it needs the interpreter lock. If the callback is starting at that
moment, each waits for the other forever. Reproduced here: with three busy
Python threads, the first pause(1) of an open/pause/close loop never
returned; in the app (Balanced Mode, a live source with sound) the exit hung
in AudioEngine.stop().

ctypes releases the interpreter lock during a foreign call, so SDL's own
SDL_PauseAudioDevice / SDL_CloseAudioDevice called through ctypes let the
callback finish. The library must be the copy of SDL that pygame uses (a
second copy has its own devices): the one bundled with pygame, or the
system library when pygame has none, accepted only if its audio subsystem
is the one pygame started (SDL_WasInit). Otherwise pygame's own methods are
used, as before.

init_pygame() is pygame.init() without pygame.mixer: pygame.init() also starts
the mixer, which opens the default sound output and keeps it open, playing
silence, next to the output VisualDeck opens itself (audio_engine). VisualDeck
does not use pygame.mixer. Seen in the tests on the owner's Windows PC: with
SDL's file output the second open failed ("Couldn't open ..."), because the
mixer already had the file.

No UI here.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import glob
import logging
import os

log = logging.getLogger("visualdeck.audio")

SDL_INIT_AUDIO = 0x00000010
_lib = None
_searched = False


def _candidates():
    import pygame
    base = os.path.dirname(os.path.abspath(pygame.__file__))
    pats = [os.path.join(base, "SDL2.dll"),                         # Windows wheels
            os.path.join(base + ".libs", "libSDL2-2*.so*"),         # Linux wheels (auditwheel)
            os.path.join(base, ".dylibs", "libSDL2-2*.dylib"),      # macOS wheels
            os.path.join(base, "libSDL2-2*.so*")]
    out = []
    for p in pats:
        out += sorted(glob.glob(p))
    sys_lib = ctypes.util.find_library("SDL2") or ctypes.util.find_library("SDL2-2.0")
    if sys_lib:
        out.append(sys_lib)
    return out


def sdl():
    """pygame's SDL library through ctypes, or None."""
    global _lib, _searched
    if _searched:
        return _lib
    _searched = True
    try:
        for path in _candidates():
            try:
                lib = ctypes.CDLL(path)
                lib.SDL_WasInit.restype = ctypes.c_uint32
                lib.SDL_WasInit.argtypes = [ctypes.c_uint32]
                if not lib.SDL_WasInit(SDL_INIT_AUDIO):
                    continue                           # not the copy pygame's audio runs in
                lib.SDL_PauseAudioDevice.argtypes = [ctypes.c_uint32, ctypes.c_int]
                lib.SDL_PauseAudioDevice.restype = None
                lib.SDL_CloseAudioDevice.argtypes = [ctypes.c_uint32]
                lib.SDL_CloseAudioDevice.restype = None
                lib.SDL_InitSubSystem.argtypes = [ctypes.c_uint32]
                lib.SDL_InitSubSystem.restype = ctypes.c_int
                _lib = lib
                log.debug(f"SDL audio calls through ctypes: {path}")
                return _lib
            except (OSError, AttributeError):
                continue
    except Exception:  # noqa: BLE001
        pass
    log.info("SDL library for audio not found through ctypes; using pygame's pause/close")
    return None


def init_pygame():
    """pygame.init(), then pygame's mixer closed again with SDL's audio
    subsystem kept started (device lists, the sound output and LTC need it).
    If pygame's SDL cannot be reached through ctypes, the mixer stays open (as
    before). Returns pygame.init()'s result."""
    import pygame
    res = pygame.init()
    mixer = getattr(pygame, "mixer", None)
    try:
        if mixer is None or not mixer.get_init():
            return res
        lib = sdl()                      # found while the mixer has SDL's audio started
        if lib is None:
            return res
        mixer.quit()                     # closes its device and stops SDL's audio subsystem
        if not lib.SDL_WasInit(SDL_INIT_AUDIO) and lib.SDL_InitSubSystem(SDL_INIT_AUDIO) != 0:
            log.warning("SDL audio could not be started again after closing pygame's mixer")
    except Exception as e:  # noqa: BLE001
        log.warning(f"Closing pygame's mixer failed: {e}")
    return res


def pause_and_close(dev):
    """Stop and close a pygame._sdl2.audio.AudioDevice."""
    lib = sdl()
    devid = getattr(dev, "deviceid", None)
    if lib is not None and devid:
        lib.SDL_PauseAudioDevice(int(devid), 1)    # the interpreter lock is released meanwhile
        lib.SDL_CloseAudioDevice(int(devid))
        return
    dev.pause(1)
    dev.close()
