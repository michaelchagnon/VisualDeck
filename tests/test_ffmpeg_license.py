# SPDX-License-Identifier: MPL-2.0
"""The FFmpeg license report (the Windows release must ship an LGPL FFmpeg)."""
import ffmpeg_license as fl

# The configure line of PyAV 15.0.0's own FFmpeg (Linux wheel, checked 2026-11-15), shortened.
PYAV15 = ("--disable-static --enable-shared --enable-version3 --enable-gmp --enable-libaom --enable-libdav1d "
          "--enable-libopenh264 --enable-zlib --enable-libx264 --enable-libx265 --enable-nvenc --enable-nvdec")
WINDOWS_FILES = ["avcodec-61-7f5ba29974161bb2dc8f58d741e9b107.dll", "libopenh264-7-0ede3c4a12caec2801f13fef5615b2f5.dll",
                 "libx264-165-de8de7d5ffe9d75d839a96621ec53cd9.dll", "libx265-9adc56be88b1827574686b8a839bca58.dll"]
LGPL_FILES = ["avcodec-61.dll", "avformat-61.dll", "libopenh264-7.dll", "libdav1d.dll", "swscale-8.dll"]


def test_the_pyav_wheel_ffmpeg_is_gpl_because_of_x264_and_x265():
    r = fl.analyze(PYAV15, "LGPL version 3 or later", WINDOWS_FILES)
    assert r["gpl_parts"] == ["x264", "x265"] and not r["lgpl_only"]
    assert r["verdict"] == "GPL (because of x264, x265)"


def test_the_bundled_files_alone_give_it_away():
    """A build whose configure line is not available is judged by its files."""
    r = fl.analyze("", "?", WINDOWS_FILES)
    assert r["gpl_parts"] == ["x264", "x265"]


def test_an_lgpl_build_is_reported_as_such():
    r = fl.analyze("--enable-shared --enable-libopenh264 --enable-libdav1d --enable-zlib",
                   "LGPL version 2.1 or later", LGPL_FILES)
    assert r["lgpl_only"] and r["verdict"] == "LGPL" and r["gpl_parts"] == [] and r["nonfree_parts"] == []


def test_enable_gpl_and_nonfree_are_caught():
    assert fl.analyze("--enable-gpl --enable-libvidstab", "GPL version 2 or later", [])["gpl_parts"] == ["vidstab"]
    assert fl.analyze("--enable-gpl", "GPL version 2 or later", [])["gpl_parts"] == ["--enable-gpl"]
    r = fl.analyze("--enable-nonfree --enable-libfdk-aac", "nonfree and unredistributable", [])
    assert not r["lgpl_only"] and r["verdict"].startswith("nonfree")


def test_this_installation_is_reported(capsys):
    """Documents the FFmpeg in this environment (the PyAV 15.0.0 wheel: GPL parts)."""
    r = fl.report()
    print(fl.summary(r))
    assert r["pyav"] and r["ffmpeg_license"] and r["bundled_files"]
    assert fl.main(["--lgpl"]) == (0 if r["lgpl_only"] else 2)


def test_the_self_test_reports_it():
    import selftest
    results = {}
    selftest._check(results, "ffmpeg", selftest._ffmpeg_license)
    assert results["ffmpeg"]["ok"] and "distribution is" in results["ffmpeg"]["detail"]


# The LGPL Windows build put into VisualDeck builds (FFmpeg-Builds, ffmpeg-n7.1.5-12-g1fdbca85aa-win64-lgpl-shared-7.1),
# part of its configure line as read from its avcodec-61.dll on 2026-11-16.
BTBN_LGPL = ("--enable-version3 --disable-debug --enable-shared --disable-static --enable-libdav1d --enable-amf "
             "--enable-libopenh264 --enable-libsnappy --enable-libsrt --enable-libvpx --disable-libx264 "
             "--disable-libx265 --disable-libxvid --disable-libvidstab --disable-librubberband --disable-frei0r "
             "--disable-avisynth --enable-libzvbi --enable-libzimg")
BTBN_FILES = ["avcodec-61.dll", "avdevice-61.dll", "avfilter-10.dll", "avformat-61.dll", "avutil-59.dll",
              "swresample-5.dll", "swscale-8.dll"]


def test_the_lgpl_windows_build_is_lgpl():
    r = fl.analyze(BTBN_LGPL, "LGPL version 3 or later", BTBN_FILES)
    assert r["lgpl_only"] and r["verdict"] == "LGPL", r
