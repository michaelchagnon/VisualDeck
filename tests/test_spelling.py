# SPDX-License-Identifier: MPL-2.0
"""The owner asked for American spelling. Fail if British forms come back in
the app's text, the help or the operator docs."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
BRITISH = re.compile(r"colour|centre|\bgrey|behaviour|licence|theatre|serialis|normalis|cancell|labell|authoris|"
                     r"synchronis|prioritis|optimis|maximis|minimis|initialis|neighbour|\bmaths\b|organis|recognis|"
                     r"favourite|analys(e|ed|es|ing)\b|catalogue|travell|modell|whilst|programme\b", re.IGNORECASE)


def test_no_british_spellings():
    files = list((ROOT / "src").rglob("*.py")) + [ROOT / "docs" / n for n in (
        "USER_GUIDE.md", "REMOTE_CONTROL.md", "KNOWN_ISSUES.md", "ROADMAP.md", "BUILD.md",
        "LICENSING_AND_EDITIONS.md")] + [ROOT / "CHANGELOG.md", ROOT / "README.md", ROOT / "THIRD_PARTY_NOTICES.md"] \
        + list((ROOT / "public").glob("*.md"))
    hits = []
    for f in files:
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if "http" in line:
                continue                      # quoted titles and URLs keep their own spelling
            m = BRITISH.search(line)
            if m:
                hits.append(f"{f.relative_to(ROOT)}:{i}: {m.group(0)}")
    assert not hits, "\n".join(hits)
