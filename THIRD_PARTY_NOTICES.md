# Third-party software in VisualDeck builds

This lists what the Windows build (`tools/windows/build_windows.ps1`, PyInstaller) and its installer bundle besides VisualDeck's own code, with each component's license. It is not legal advice.

**Source of each license.** *Metadata* means the package's own metadata in the Windows build environment (read 2026-11-15). *File* means a license file or library file found in that environment. *Recalled* means not checked against the component's own license file.

## Python packages

| Component | Version | License | Source |
|---|---|---|---|
| Python (runtime, `python312.dll` and standard library) | 3.12.10 | PSF License | Recalled |
| PyAV (`av`) | 15.0.0 | BSD-3-Clause | Metadata |
| FFmpeg libraries (`av.libs`), an LGPL build in place of the wheel's own | 7.1.5 | LGPL-3.0-or-later; see *FFmpeg* below | File |
| NumPy | 2.1.2 | BSD-3-Clause (bundles OpenBLAS, BSD-3-Clause) | Metadata (BSD); OpenBLAS recalled |
| opencv-python (`cv2`) | 4.12.0.88 | Apache-2.0; its bundled FFmpeg (`opencv_videoio_ffmpeg4120_64.dll`) is LGPL-2.1 | Metadata; `cv2/LICENSE-3RD-PARTY.txt` |
| pygame | 2.6.1 | LGPL-2.1 | Metadata ("LGPL") |
| SDL2, SDL2_image, SDL2_mixer, SDL2_ttf (in pygame) | | zlib | Recalled |
| FreeType, libjpeg, libpng, libtiff, libwebp, libmodplug, libogg, Opus, opusfile, PortMidi, zlib (in pygame) | | FreeType License, IJG, libpng, libtiff, BSD-3-Clause, public domain, BSD-3-Clause, BSD-3-Clause, BSD-3-Clause, MIT-style, zlib | Files present; licenses recalled |
| Pillow | 11.3.0 | MIT-CMU | Metadata |
| PyOpenGL, PyOpenGL-accelerate | 3.1.9 | BSD | Metadata |
| python-osc | 1.9.3 | Unlicense (public domain) | Metadata |
| python-rtmidi (includes RtMidi) | 1.5.8 | MIT | Metadata; RtMidi recalled |
| screeninfo | 0.8.1 | MIT | Metadata |
| watchdog | 6.0.0 | Apache-2.0 | Metadata |
| pywin32 | 306 | PSF | Metadata |
| PyInstaller bootloader | | GPL-2.0 with the bootloader exception (allows any program to ship it) | Recalled |
| OpenSSL 3 (`libcrypto-3.dll`, `libssl-3.dll`, part of Python) | 3.x | Apache-2.0 | Recalled |
| libffi (`libffi-8.dll`, part of Python) | | MIT | Recalled |
| Tcl/Tk (`tcl86t.dll`, `tk86t.dll`, part of Python) | 8.6 | Tcl/Tk License (BSD-style) | Recalled |
| Microsoft Visual C++ runtime (`MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`) | | Microsoft's redistribution terms for the Visual C++ runtime | Recalled |

Development tools (pytest, pluggy, iniconfig, colorama, packaging, Pygments, pip) are in the environment but are not imported by VisualDeck, so the build does not include them.

## FFmpeg

`python src/ffmpeg_license.py` reports what the installed FFmpeg is; the build's self-test prints it too (check `ffmpeg`), and the build stops unless it is LGPL.

**What the build ships (since 2026-11-16):** the LGPL Windows build of FFmpeg 7.1.5 by the FFmpeg-Builds project (`ffmpeg-n7.1.5-12-g1fdbca85aa-win64-lgpl-shared-7.1.zip`, release `autobuild-2026-07-31-14-10`, SHA-256 `0f376f96…c5d0`), as seven DLLs in `_internal\av.libs`. It is configured without `--enable-gpl` and with `--disable-libx264 --disable-libx265` (configure line read from its `avcodec-61.dll`), and reports itself as "LGPL version 3 or later". `tools/windows/lgpl_ffmpeg.py` puts it in place of the FFmpeg that the PyAV wheel bundles. The installation folder's `licenses\` holds its license (`FFmpeg-LICENSE.txt`, the LGPL version 3), the GPL version 3 that the LGPL version 3 builds on (`GPL-3.0.txt`), and where its source is (`FFmpeg-SOURCE.txt`). The libraries are separate DLLs that a user can replace with another build of the same versions, as the LGPL requires.

Libraries built into those DLLs, from the configure line (licenses recalled, to confirm): dav1d, libaom, OpenH264, openjpeg, libvpx, libwebp, libjxl, rav1e, kvazaar, uavs3d, Theora, Vorbis and Opus (BSD-style); SVT-AV1 and VVenC (BSD-3-Clause-Clear); LAME, TwoLAME, libsoxr, libbluray, libssh, libgme, OpenAL Soft, libplacebo, FriBidi, aribb24, zvbi and GMP (LGPL); libass, LV2 (ISC); HarfBuzz, AMF headers, nv-codec-headers, oneVPL, libaribcaption (MIT); FreeType (FreeType License); SDL2 (zlib); SRT and ZeroMQ (MPL-2.0); opencore-amr (Apache-2.0); libvmaf (BSD-2-Clause-Patent); libxml2 (MIT); snappy (BSD-3-Clause); zimg (WTFPL); xz (public domain/0BSD); zlib (zlib). TLS goes through Windows (Schannel), not OpenSSL.

**What the PyAV wheel bundles otherwise** (not shipped by the build any more): FFmpeg 7.1.1 with libx264 and libx265 (GPL-2.0-or-later), OpenSSL 1.1.1's `libcrypto`, and the MinGW runtime (checked in the Windows and Linux wheels). The development environment still uses it: the automated tests make their clips with x264. That copy is not distributed.

H.264, HEVC and some other formats are patent-licensed whichever FFmpeg is used; whether the official installer needs a patent license is a question for a lawyer.

## Media and assets

The built-in effects (`src/effects/*.fs`), the icon and the test patterns are VisualDeck's own. The test media used by the automated tests is generated by the tests themselves and is not shipped.
