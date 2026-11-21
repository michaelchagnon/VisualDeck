"""SPDX license headers (MPL-2.0) on VisualDeck's own source files.

    python tools/license_headers.py --check    list files without the header (exit 1 if any)
    python tools/license_headers.py --apply    add it (run once, in its own commit, when the
                                               owner switches the license)

The header is one line, `SPDX-License-Identifier: MPL-2.0`, in each file's
comment syntax: Python and PowerShell `#`, shader (ISF) files `//` after their
JSON header (ISF hosts read the first comment as the header), Windows command
scripts `REM`, the NSIS installer script `;`. A Python file's first line stays first if it is a shebang or an
encoding line. Files listed by git only; nothing else is touched.
"""
from __future__ import annotations

import os
import subprocess
import sys

SPDX = "SPDX-License-Identifier: MPL-2.0"
COMMENT = {".py": "# ", ".ps1": "# ", ".fs": "// ", ".cmd": "REM ", ".nsi": "; "}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def files(root=ROOT):
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout
    return [os.path.join(root, *f.split("/")) for f in out.splitlines() if os.path.splitext(f)[1] in COMMENT]


def has_header(text):
    return SPDX in "\n".join(text.splitlines()[:40])


def with_header(path, text):
    ext = os.path.splitext(path)[1]
    line = COMMENT[ext] + SPDX
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    if ext == ".fs":
        if lines and lines[0].lstrip().startswith("/*"):
            for i, ln in enumerate(lines):
                if "*/" in ln:
                    return nl.join(lines[:i + 1] + [line] + lines[i + 1:])
        return nl.join([line] + lines)
    at = 0
    if ext == ".py":
        while at < len(lines) and at < 2 and (lines[at].startswith("#!") or "coding" in lines[at][:30]):
            at += 1
    if ext == ".cmd" and lines and lines[0].strip().lower().startswith("@echo off"):
        at = 1
    return nl.join(lines[:at] + [line] + lines[at:])


def main(argv):
    if len(argv) != 2 or argv[1] not in ("--check", "--apply"):
        print(__doc__)
        return 2
    missing = []
    for p in files():
        with open(p, encoding="utf-8", newline="") as f:
            text = f.read()
        if has_header(text):
            continue
        missing.append(os.path.relpath(p, ROOT))
        if argv[1] == "--apply":
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(with_header(p, text))
    verb = "added to" if argv[1] == "--apply" else "missing in"
    print(f"SPDX header {verb} {len(missing)} file(s)")
    for m in missing:
        print("  " + m)
    return 0 if argv[1] == "--apply" or not missing else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
