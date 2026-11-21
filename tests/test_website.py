# SPDX-License-Identifier: MPL-2.0
"""The website builds, every Help topic has a page, and its links and structured data are sound."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("jinja2")
pytest.importorskip("markdown")

ROOT = Path(__file__).resolve().parents[1]


def load_build():
    spec = importlib.util.spec_from_file_location("website_build", ROOT / "website" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_site_builds_with_a_page_per_help_topic(tmp_path):
    build = load_build()
    pages = build.build(tmp_path)
    import help_content
    for title, _ in help_content.TOPICS:
        slug = build.TOPIC_META[title][0]
        assert f"/docs/{slug}/" in pages
        assert (tmp_path / "docs" / slug / "index.html").is_file()
    for name in ("index.html", "download/index.html", "docs/index.html", "404.html", "sitemap.xml",
                 "robots.txt", "llms.txt", "llms-full.txt", "favicon.ico", "assets/og.png", "assets/site.css"):
        assert (tmp_path / name).is_file(), name


def test_internal_links_images_and_json_ld(tmp_path):
    build = load_build()
    build.build(tmp_path)
    kept_on_server = {"bitcoin.png", "download/VisualDeck%20v1.0b.zip"}   # files the old site keeps
    for page in tmp_path.rglob("*.html"):
        text = page.read_text(encoding="utf-8")
        for block in re.findall(r'<script type="application/ld\+json">(.*?)</script>', text, re.S):
            json.loads(block)
        assert text.count("<h1") == 1, page
        for url in re.findall(r'(?:href|src)="(/[^"#?]*)', text):
            rel = url.lstrip("/")
            if rel in kept_on_server:
                continue
            target = tmp_path / rel
            if url.endswith("/"):
                target = target / "index.html"
            assert target.exists(), f"{page.relative_to(tmp_path)} links to missing {url}"
