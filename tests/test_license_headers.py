"""tools/license_headers.py: the SPDX line for the switch to MPL-2.0, in each file's comment syntax."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import license_headers as lh  # noqa: E402


@pytest.mark.parametrize("name,text,first_lines", [
    ("a.py", "import os\n", ["# SPDX-License-Identifier: MPL-2.0", "import os"]),
    ("b.py", "#!/usr/bin/env python\nx = 1\n", ["#!/usr/bin/env python", "# SPDX-License-Identifier: MPL-2.0"]),
    ("c.fs", '/*{\n "INPUTS": []\n}*/\nvoid main() {}\n',
     ["/*{", ' "INPUTS": []', "}*/", "// SPDX-License-Identifier: MPL-2.0", "void main() {}"]),
    ("d.cmd", "@echo off\r\necho hi\r\n", ["@echo off", "REM SPDX-License-Identifier: MPL-2.0", "echo hi"]),
    ("e.ps1", "param()\n", ["# SPDX-License-Identifier: MPL-2.0", "param()"]),
    ("f.nsi", "Unicode true\n", ["; SPDX-License-Identifier: MPL-2.0", "Unicode true"]),
])
def test_header_goes_where_the_file_type_allows(name, text, first_lines):
    out = lh.with_header(name, text)
    nl = "\r\n" if "\r\n" in text else "\n"
    assert out.split(nl)[:len(first_lines)] == first_lines
    assert lh.has_header(out) and not lh.has_header(text)


def test_shader_with_header_still_parses(tmp_path):
    """An ISF effect keeps working with the line after its JSON header."""
    import isf
    src = os.path.join(os.path.dirname(isf.__file__), "effects", "Invert.fs")
    text = lh.with_header(src, open(src, encoding="utf-8").read())
    eff = isf.parse(text, "Invert.fs")
    assert [i["NAME"] for i in eff["inputs"]] == ["amount"]
    assert "SPDX" in isf.fragment_source(eff)


def test_check_lists_files_in_the_repository():
    found = lh.files()
    assert any(f.endswith(os.path.join("src", "app.py")) for f in found)
    assert any(f.endswith(".fs") for f in found) and not any(f.endswith(".md") for f in found)


def test_every_file_carries_the_mpl_header():
    """After the switch to MPL-2.0 (release branch): no file without it."""
    missing = [f for f in lh.files() if not lh.has_header(open(f, encoding="utf-8", errors="replace").read())]
    assert not missing, missing
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, "LICENSE"), encoding="utf-8").read()
    assert text.startswith("Mozilla Public License Version 2.0") and "Exhibit B" in text
    assert not os.path.exists(os.path.join(root, "EULA"))
