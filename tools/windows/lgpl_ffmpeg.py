# SPDX-License-Identifier: MPL-2.0
"""Put an LGPL FFmpeg into a built VisualDeck folder in place of PyAV's own.

The PyAV 15.0.0 wheel bundles an FFmpeg 7.1.1 with libx264 and libx265, which are
GPL, so a VisualDeck built with it would be a GPL work.
This replaces those libraries, in the folder PyInstaller made, with an FFmpeg of
the same major versions built without GPL parts (FFmpeg-Builds' "lgpl-shared"
Windows build: avcodec-61, avformat-61, avutil-59, swscale-8, swresample-5,
avfilter-10, avdevice-61, with their other libraries built in):

  * every file in _internal\\av.libs (the wheel's FFmpeg and the libraries it links,
    x264 and x265 among them) is removed, and the LGPL build's DLLs are copied there;
  * PyAV's modules (_internal\\av\\**\\*.pyd) import the wheel's DLLs under names that
    delvewheel made unique ("avcodec-61-7f5b....dll"); each such name in their import
    tables is rewritten to the plain name ("avcodec-61.dll"), which is shorter, so
    the file only changes there;
  * FFmpeg's license and where the build came from are written to licenses\\.

FFmpeg keeps its ABI within a major version, and PyAV calls nothing else, so PyAV
compiled for 7.1.1 runs on 7.1.5. The build's self-test then decodes video with it,
and src/ffmpeg_license.py must call the result LGPL.

    python lgpl_ffmpeg.py <dist\\VisualDeck> <ffmpeg\\bin> [--license LICENSE.txt] [--source URL]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

FFMPEG_LIBS = ("avcodec", "avdevice", "avfilter", "avformat", "avutil", "swresample", "swscale", "postproc")
_UNIQUE = re.compile(r"^(?P<base>(?:%s)-\d+)-[0-9a-f]{32}\.dll$" % "|".join(FFMPEG_LIBS), re.I)
_PLAIN = re.compile(r"^(?:%s)-\d+\.dll$" % "|".join(FFMPEG_LIBS), re.I)


def plain_name(name):
    """'avcodec-61-7f5b...dll' -> 'avcodec-61.dll'; None for any other DLL."""
    m = _UNIQUE.match(name)
    return m.group("base") + ".dll" if m else None


def lgpl_dlls(bin_dir):
    """The FFmpeg DLLs of the LGPL build, by lower-case name."""
    return {n.lower(): os.path.join(bin_dir, n) for n in os.listdir(bin_dir) if _PLAIN.match(n)}


def patch_imports(path, available):
    """Rewrite the unique FFmpeg names in one module's import table; returns the changes."""
    import pefile
    with open(path, "rb") as f:
        pe = pefile.PE(data=f.read(), fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                                           pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"]])
    if getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", None):
        for e in pe.DIRECTORY_ENTRY_DELAY_IMPORT:
            if plain_name(e.dll.decode()):
                raise SystemExit(f"{path}: delay-loads {e.dll.decode()} (not handled)")
    changes = []
    for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
        old = entry.dll.decode()
        new = plain_name(old)
        if new is None:
            continue
        if new.lower() not in available:
            raise SystemExit(f"{path} needs {new}, which the LGPL build does not have")
        off = pe.get_offset_from_rva(entry.struct.Name)
        pe.set_bytes_at_offset(off, new.encode("ascii") + b"\0" * (len(old) - len(new) + 1))
        changes.append((old, new))
    if changes:
        tmp = path + ".lgpl-tmp"
        pe.write(filename=tmp)
        os.replace(tmp, path)
    return changes


def apply(dist, bin_dir, license_file=None, source=None, log=print):
    internal = os.path.join(dist, "_internal")
    av_dir = os.path.join(internal, "av")
    libs = os.path.join(internal, "av.libs")
    if not os.path.isdir(av_dir):
        raise SystemExit(f"no PyAV in {internal}")
    available = lgpl_dlls(bin_dir)
    if "avcodec-61.dll" not in available:
        raise SystemExit(f"{bin_dir} has no avcodec-61.dll: not an FFmpeg 7.1 shared build")
    modules = [os.path.join(r, n) for r, _, files in os.walk(av_dir) for n in files if n.lower().endswith(".pyd")]
    patched = 0
    for m in modules:
        if patch_imports(m, available):
            patched += 1
    removed = sorted(os.listdir(libs)) if os.path.isdir(libs) else []
    for n in removed:
        os.remove(os.path.join(libs, n))
    os.makedirs(libs, exist_ok=True)
    for n, src in sorted(available.items()):
        shutil.copy2(src, os.path.join(libs, os.path.basename(src)))
    lic_dir = os.path.join(dist, "licenses")
    os.makedirs(lic_dir, exist_ok=True)
    if license_file and os.path.exists(license_file):
        shutil.copy2(license_file, os.path.join(lic_dir, "FFmpeg-LICENSE.txt"))
    # LGPL version 3 adds permissions to the GPL version 3, and asks for both texts.
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "licenses")
    for n in ("GPL-3.0.txt", "LGPL-3.0.txt"):
        if os.path.exists(os.path.join(here, n)):
            shutil.copy2(os.path.join(here, n), os.path.join(lic_dir, n))
    with open(os.path.join(lic_dir, "FFmpeg-SOURCE.txt"), "w", encoding="utf-8") as f:
        f.write("VisualDeck uses FFmpeg (https://ffmpeg.org), licensed under the GNU Lesser General Public "
                "License, version 3 or later (see FFmpeg-LICENSE.txt or LGPL-3.0.txt, and GPL-3.0.txt, whose terms it builds on). The FFmpeg libraries are the separate "
                "files in _internal\\av.libs and can be replaced by another build of the same versions.\n\n"
                f"This copy is the build from {source or bin_dir}.\n"
                "Its source code: https://github.com/FFmpeg/FFmpeg (tag n7.1.5 and the commits named in the "
                "build's version) and the build scripts at https://github.com/FFmpeg-Builds/FFmpeg-Builds.\n")
    log(f"LGPL FFmpeg: {patched} of {len(modules)} PyAV modules relinked, {len(removed)} files removed from "
        f"av.libs ({', '.join(n for n in removed if 'x26' in n.lower()) or 'no x264/x265'}), "
        f"{len(available)} copied: {', '.join(sorted(os.path.basename(p) for p in available.values()))}")
    return patched, removed, sorted(available)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dist")
    ap.add_argument("ffmpeg_bin")
    ap.add_argument("--license")
    ap.add_argument("--source")
    a = ap.parse_args(argv)
    apply(a.dist, a.ffmpeg_bin, a.license, a.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
