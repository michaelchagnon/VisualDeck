# SPDX-License-Identifier: MPL-2.0
"""What license the FFmpeg inside this installation carries (for release builds).

VisualDeck decodes with PyAV, whose binary wheels bundle their own FFmpeg. The
wheels used up to now (PyAV 15.0.0, Linux and Windows) contain libx264 and
libx265, which are GPL, although FFmpeg itself reports "LGPL version 3 or
later" (checked 2026-11-15: the configure line has --enable-libx264 and
--enable-libx265 but not --enable-gpl, and av.libs holds libx264 and libx265 on
both systems). Shipping those makes the binary distribution a GPL work, which
MPL-2.0 allows (MPL section 3.3) but which the Windows release avoids: it ships
an LGPL build of FFmpeg instead (tools/windows/lgpl_ffmpeg.py).

    python src/ffmpeg_license.py            print the report
    python src/ffmpeg_license.py --lgpl     exit 2 if any GPL or nonfree part is found
                                            (for an LGPL release build)

The self-test of a built program includes this report.
"""
from __future__ import annotations

import os
import re
import sys

# Libraries that FFmpeg can link and that are GPL (or not redistributable at all,
# "nonfree"), by the names of their files and their configure switches.
# (FFmpeg's configure refuses these without --enable-gpl; libzvbi, which it once listed,
# is LGPL and is in the LGPL Windows build VisualDeck uses.)
GPL_LIBS = ("x264", "x265", "xvidcore", "xvid", "vidstab", "rubberband", "frei0r", "postproc", "cdio",
            "avisynth", "davs2", "xavs", "xavs2")
NONFREE_LIBS = ("fdk-aac", "fdk_aac", "libnpp", "decklink")


def _lib_files(av_dir):
    """Library files bundled next to PyAV (av.libs, or the package folder itself)."""
    names = []
    for d in (os.path.join(os.path.dirname(av_dir), "av.libs"), os.path.join(os.path.dirname(av_dir), ".dylibs"),
              av_dir):
        try:
            names += [n for n in os.listdir(d) if re.search(r"\.(dll|so(\.\d+)*|dylib)$", n, re.I)
                      and "cpython" not in n]                       # not PyAV's own modules
        except OSError:
            pass
    return sorted(set(names))


def analyze(configuration, license_text, lib_files):
    """Report from FFmpeg's configure line, its own license string and the bundled files."""
    flags = configuration.split()
    enabled = {f[len("--enable-"):] for f in flags if f.startswith("--enable-")}

    def present(name):
        base = name[3:] if name.startswith("lib") else name
        in_files = any(re.match(rf"(lib)?{re.escape(base)}([-_.\d]|$)", f, re.I) for f in lib_files)
        return "lib" + base in enabled or base in enabled or in_files

    gpl = [n for n in GPL_LIBS if present(n)]
    nonfree = [n for n in NONFREE_LIBS if present(n)] + (["--enable-nonfree"] if "nonfree" in enabled else [])
    if "gpl" in enabled and not gpl:
        gpl = ["--enable-gpl"]
    if nonfree:
        verdict = "nonfree: not redistributable"
    elif gpl:
        verdict = "GPL (because of " + ", ".join(gpl) + ")"
    else:
        verdict = "LGPL"
    return {"ffmpeg_license": license_text, "gpl_parts": gpl, "nonfree_parts": nonfree, "verdict": verdict,
            "lgpl_only": not gpl and not nonfree, "bundled_files": lib_files}


def report():
    import av
    from av import _core
    meta = _core.library_meta.get("libavcodec", {})
    out = analyze(meta.get("configuration", ""), meta.get("license", "?"), _lib_files(os.path.dirname(av.__file__)))
    out["pyav"] = av.__version__
    out["ffmpeg"] = getattr(av, "ffmpeg_version_info", "?")
    return out


def summary(r):
    return (f"PyAV {r['pyav']}, FFmpeg {r['ffmpeg']} ({r['ffmpeg_license']}): distribution is {r['verdict']}, "
            f"{len(r['bundled_files'])} bundled libraries")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    r = report()
    print(summary(r))
    for f in r["bundled_files"]:
        print("  " + f)
    if "--lgpl" in argv and not r["lgpl_only"]:
        print("Not an LGPL build: " + ", ".join(r["gpl_parts"] + r["nonfree_parts"]), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
