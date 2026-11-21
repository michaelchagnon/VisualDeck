# SPDX-License-Identifier: MPL-2.0
"""Build visualdeck.org as plain static files.

    python website/build.py [--out website/_site]

The documentation pages come from the in-app Help (src/help_content.py, one page per
topic), docs/REMOTE_CONTROL.md, public/KNOWN_ISSUES.md and public/CHANGELOG.md, so the
site says what the app says. Screenshots come from docs/images/ (the README uses the
same files). The output folder is the site's root: upload its contents to public_html.

Needs Jinja2, Markdown and Pillow (website/requirements.txt).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import re
import shutil
import sys
from pathlib import Path

import jinja2
import markdown
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
import help_content  # noqa: E402
import version  # noqa: E402

SITE = "https://visualdeck.org"
GITHUB = "https://github.com/michaelchagnon/VisualDeck"
BITCOIN = "bc1qv6aw43faucxyvne7z4hr7yystxu8vu5sf5fc3m"
RELEASE = {
    "label": version.LABEL,
    "version": version.VERSION,
    "date": "2026-11-21",
    "date_text": "November 21, 2026",
    "filename": f"VisualDeck-{version.VERSION}-setup.exe",
    "size_bytes": 93278786,
    "sha256": "f255aae7802d048a2139c6556e8eb9e41fd08a35ab5e88f15a9db37fcdaba9e5",
}
RELEASE["download_url"] = f"{GITHUB}/releases/download/v{RELEASE['version']}/{RELEASE['filename']}"
RELEASE["size_text"] = f"{RELEASE['size_bytes'] / 1048576:.0f} MB"
RELEASE["size_bytes_text"] = f"{RELEASE['size_bytes']:,}"

DOWNLOAD_ICON = ('<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><path d="M8 1.5v9m0 0L4.2 6.7M8 10.5l3.8-3.8'
                 'M2 14.5h12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="square"/></svg>')

# --- Documentation: groups, page names and descriptions ------------------------------------------

GROUPS = [
    ("Start here", ["Quick start: running a show", "Getting started", "Shows and files"]),
    ("Building a show", ["Cues and layers", "Position editor", "Shader effects", "Sound", "Live sources"]),
    ("Running a show", ["Running the show", "Cue settings and follow", "Blind editing", "Presentation Mode"]),
    ("Outputs", ["Outputs and test patterns", "Output Health", "Missing media and Collect Show"]),
    ("Remote control", ["Remote control", "OSC commands", "Browser remote and HTTP API", "@remote-control-reference"]),
    ("Reference", ["Preferences", "Keyboard shortcuts", "Troubleshooting", "Help menu", "About VisualDeck",
                   "@known-issues", "@changelog"]),
]

# slug, short name for the breadcrumb, meta description (at most about 160 characters)
TOPIC_META = {
    "Quick start: running a show": ("quick-start", "Quick start",
        "A checklist for the operator: open the show, run Preshow Check, lock editing, press GO on the call, and "
        "recover from a wrong GO or a frozen picture."),
    "Getting started": ("getting-started", None,
        "How VisualDeck works: the cue grid, adding media, building cues, choosing the output, running the show, "
        "saving, and the video and image formats it plays."),
    "Shows and files": ("shows-and-files", None,
        "New, open, recent and saved shows: what a .vjdeck show file holds, how media is found by its path, "
        "atomic saves, the .bak copy and autosave."),
    "Cues and layers": ("cues-and-layers", None,
        "Cut to a cue, go to cues 1 to 9, reorder cues and layers by dragging, duplicate, clear and delete cues, "
        "and choose each layer's display."),
    "Position editor": ("position-editor", None,
        "Position, size, rotation, opacity, blend mode, color, speed, sound volume, masks, warp points and effects "
        "for one cell of the cue grid."),
    "Shader effects": ("shader-effects", None,
        "VisualDeck's nine built-in real-time ISF shader effects and their settings, up to four on each layer and "
        "display, and how to add your own."),
    "Sound": ("sound", None,
        "Play the sound of videos: choose the output device, test it, set each video's volume, and delay the sound "
        "to match the projector."),
    "Live sources": ("live-sources", None,
        "Use cameras, capture cards and network streams (UDP, RTSP, SRT, RTMP, HLS) as live layers, with a test, "
        "sound and automatic reconnection."),
    "Running the show": ("running-the-show", None,
        "GO, BACK and Blackout, Preshow Check, cutting to a cue, Start Cue From, Lock Editing, the schedule, and "
        "what happens after the last cue."),
    "Cue settings and follow": ("cue-settings-and-follow", None,
        "Cue Settings: name, number, transition and fade time, follow after a time or when the videos end, "
        "timecode, Keep Ready and notes."),
    "Blind editing": ("blind-editing", None,
        "Change the cue that is on the output without the audience seeing it, check the edit in a second preview, "
        "then apply it with Update Output."),
    "Presentation Mode": ("presentation-mode", None,
        "Let the cue list run by itself every few seconds, like a slideshow, for foyer loops, exhibitions and "
        "rehearsal run-throughs, with Loop."),
    "Outputs and test patterns": ("outputs-and-test-patterns", None,
        "Choose displays, corner pin, mesh warp, edge blending, Span Displays, slices, test patterns and output "
        "presets for projectors and screens."),
    "Output Health": ("output-health", None,
        "Output Health shows the frame rate, late frames, render cost, errors, missing media, every video's "
        "decoder and which GPU draws the output."),
    "Missing media and Collect Show": ("missing-media-and-collect-show", None,
        "Relink media that moved, and use Collect Show to copy a show with all of its media into one folder for "
        "another computer."),
    "Remote control": ("remote-control", None,
        "An overview of controlling VisualDeck from other devices: OSC, the browser remote and HTTP API, MIDI, "
        "MIDI Show Control, timecode and DMX."),
    "OSC commands": ("osc-commands", None,
        "Every OSC address VisualDeck accepts, with its arguments and an example: GO, BACK, Blackout, go to a cue, "
        "Start Cue From, effects and ping."),
    "Browser remote and HTTP API": ("browser-remote-and-http-api", None,
        "Run VisualDeck from a phone or tablet browser, and from scripts over HTTP: the remote page, its safety "
        "rules and every API endpoint."),
    "Preferences": ("preferences", None,
        "VisualDeck's Preferences: editing the live cue, remote control, audio, the three render modes with their "
        "advanced options, and video decoding."),
    "Keyboard shortcuts": ("keyboard-shortcuts", None,
        "Every VisualDeck keyboard shortcut: GO (Space), BACK (Shift+Space), Blackout (Ctrl+B), Update Output, "
        "cues 1 to 9, files and windows."),
    "Troubleshooting": ("troubleshooting", None,
        "What to check for black output and stutter, heavy video files and laptops with two GPUs, where the logs "
        "are, and the built-in Soak Test."),
    "Help menu": ("help-menu", None,
        "What is in VisualDeck's Help menu: searchable help (F1), keyboard shortcuts, the log folder, the Soak "
        "Test and About."),
    "About VisualDeck": ("about", "About",
        f"VisualDeck {version.LABEL} Community Edition: its license, the Mozilla Public License 2.0, and the "
        "libraries it is built with."),
}

# Longer page titles where the topic name alone says little to a search engine (at most 42 characters)
SEO_TITLES = {
    "Sound": "Sound: audio device, volume and delay",
    "Live sources": "Live cameras, capture cards and streams",
    "Preferences": "Preferences and render modes",
    "Shader effects": "Shader effects (ISF)",
    "Outputs and test patterns": "Corner pin, mesh warp and edge blending",
    "Output Health": "Output Health: frame rate and decoders",
    "Help menu": "Help menu and logs",
    "About VisualDeck": "About VisualDeck and its license",
    "Position editor": "Position editor: layers, blend and masks",
}

MD_PAGES = {
    "@remote-control-reference": {
        "slug": "remote-control-reference", "title": "Remote control reference", "file": "docs/REMOTE_CONTROL.md",
        "h1": "Remote control reference: OSC, browser, MIDI, timecode and DMX",
        "description": "The full reference for controlling VisualDeck: OSC, the browser remote and HTTP API, MIDI, "
                       "MIDI Show Control, MIDI timecode, SMPTE LTC, Art-Net and sACN."},
    "@known-issues": {
        "slug": "known-issues", "title": "Known issues", "file": "public/KNOWN_ISSUES.md",
        "h1": f"Known issues and limitations in VisualDeck {version.LABEL}",
        "description": f"What VisualDeck {version.LABEL} has and hasn't been tested with, and its known limits in "
                       "render modes, sound, cues, outputs and security."},
    "@changelog": {
        "slug": "changelog", "title": "Changelog", "file": "public/CHANGELOG.md", "h1": "VisualDeck changelog",
        "description": "What changed in each VisualDeck release, from the 1.0 stable release back to the original "
                       "1.0b beta."},
}

# Screenshots shown on documentation pages: topic -> (image, alt text, caption)
TOPIC_IMAGES = {
    "Quick start: running a show": ("preshow-check",
        "The Preshow Check window reading Ready for the show, with green checks for the output display, media "
        "files, sound output, saved show and locked editing.",
        "Show › Preshow Check… when everything is ready."),
    "Getting started": ("main-window",
        "The VisualDeck main window: the media list on the left, the cue grid in the middle and the preview "
        "with the GO button on the right.",
        "The main window: media list, cue grid, and the preview of what the audience sees."),
    "Cue settings and follow": ("cue-settings",
        "The Cue Settings window for a cue named Sunset, number 3, with Smooth Fade, a fade time of 3 seconds, "
        "Wait for GO and a note.",
        "Cue Settings for the live cue (Ctrl+E)."),
    "Position editor": ("position-editor",
        "The position editor for an image with a Kaleidoscope effect: the canvas on the left and the Position "
        "and size and Effects sections on the right.",
        "The position editor, with the Effects section open."),
    "Outputs and test patterns": ("corner-pin-output",
        "The Corner Pin, Mesh and Edge Blend window in front of a keystoned grid test pattern on the output.",
        "Window › Corner Pin, Mesh and Edge Blend… with the grid test pattern on the output behind it."),
    "Output Health": ("output-health",
        "The Output Health window reading OK, with frame timing, remote-control and audio status and a table "
        "of the videos and their decoders.",
        "Window › Output Health… (Ctrl+H)."),
    "Browser remote and HTTP API": ("browser-remote",
        "The browser remote on a phone with GO, BACK, Blackout and Update Output, effect sliders and the cue "
        "list.",
        "The browser remote on a phone."),
    "Preferences": ("preferences-remote-control",
        "The Preferences window on its Remote control page, with settings for OSC, the browser remote, and "
        "buttons for MIDI, LTC and DMX setup.",
        "Preferences › Remote control. Everything is off until you turn it on."),
    "Help menu": ("help",
        "The VisualDeck Help window with a search box and the list of topics on the left, and the quick start "
        "on the right.",
        "Help › VisualDeck Help (F1)."),
}

FAQ = [
    ("Is VisualDeck free?",
     "Yes. VisualDeck is free and open source under the Mozilla Public License 2.0, for personal and commercial "
     "use. There's no account, trial period or watermark."),
    ("Is VisualDeck a QLab alternative for Windows?",
     "For video, yes. QLab runs only on macOS; VisualDeck gives Windows a cue list for video with GO, BACK, "
     "follows, timecode, OSC and MIDI Show Control. It doesn't run audio-only or lighting cues, so it works "
     "alongside your sound and lighting control rather than replacing it."),
    ("What video formats does VisualDeck play?",
     "H.264, H.265 (HEVC), ProRes including 4444 with transparency, HAP, HAP Alpha and DXV, in MP4, MOV, AVI and "
     "MKV files. It also shows JPG, PNG, BMP and GIF images, USB cameras, capture cards and network streams."),
    ("Does VisualDeck do projection mapping?",
     "Yes, in 2D: corner pin and mesh warp for each display, warp points and masks for each layer, edge blending, "
     "and spans across several projectors. It doesn't map onto 3D models."),
    ("How many screens or projectors can it use?",
     "As many displays as the computer has. Send the whole show to one display, or choose a display for each "
     "layer; Span Displays joins several projectors into one picture."),
    ("Can a lighting or sound desk trigger VisualDeck?",
     "Yes, over OSC, MIDI, MIDI Show Control, MIDI timecode, SMPTE LTC, Art-Net or sACN. These connections were "
     "tested with messages built to each standard, not yet with physical consoles, so try yours in a rehearsal "
     "before the show."),
    ("Can it run an installation unattended?",
     "Yes. Presentation Mode loops cues on a timer, the schedule runs cues at set times of day, and Windows "
     "doesn't go to sleep while VisualDeck is open. A 1.0 release candidate ran for 72 hours unattended in the "
     "built-in Soak Test with no render errors. The schedule works while VisualDeck is open; it doesn't wake a "
     "sleeping computer or start VisualDeck."),
    ("What computer do I need?",
     "Windows 10 or 11, 64-bit, with 8 GB of memory recommended. Performance Mode, which effects and blend modes "
     "need, uses a graphics card or integrated GPU with OpenGL 3.3; Balanced and Compatibility Mode work without "
     "one. For 4K video on integrated graphics, Make Playback Copy makes a lighter 1080p copy."),
    ("Does VisualDeck run on Mac or Linux?",
     "The release is for Windows. VisualDeck also runs from source on Linux, where its automated tests run. "
     "macOS hasn't been tested."),
    ("What happens to my show if VisualDeck crashes?",
     "Unsaved changes are autosaved every two minutes to a separate file, and VisualDeck offers to recover them "
     "when you open the show again. Saving replaces the show file in one step and keeps the previous version as "
     "a .bak file."),
]

SHOTS_FOR_SCHEMA = ["main-window", "position-editor", "corner-pin-output", "output-health"]

# --- Text helpers ---------------------------------------------------------------------------------

KEY_RE = re.compile(r"(?<![\w+])((?:Ctrl|Shift|Alt)(?:\+(?:Ctrl|Shift|Alt))*\+(?:Space|comma|Enter|Esc|Tab|"
                    r"F\d{1,2}|[A-Z0-9])|F(?:1[0-2]|[1-9]))(?![\w+])")
PAREN_KEY_RE = re.compile(r"\((Space|Enter|Esc)([,)])")


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def topic_url(title):
    if title.startswith("@"):
        return f"/docs/{MD_PAGES[title]['slug']}/"
    return f"/docs/{TOPIC_META[title][0]}/"


def topic_name(title):
    return MD_PAGES[title]["title"] if title.startswith("@") else title


def inline(text, here=None):
    """Plain help text to HTML: escaped, with keys as <kbd> and topic mentions as links."""
    s = html.escape(text, quote=False)
    s = KEY_RE.sub(lambda m: f"<kbd>{m.group(1)}</kbd>", s)
    s = PAREN_KEY_RE.sub(lambda m: f"(<kbd>{m.group(1)}</kbd>{m.group(2)}", s)
    s = s.replace("docs/REMOTE_CONTROL.md", '<a href="/docs/remote-control-reference/">the remote control reference</a>')
    for title in sorted(TOPIC_META, key=len, reverse=True):
        if title == here:
            continue
        url = topic_url(title)
        if f"see the {title} topic" in s:
            s = s.replace(f"see the {title} topic", f'see the <a href="{url}">{title}</a> topic')
        elif f"see {title}" in s:
            s = s.replace(f"see {title}", f'see <a href="{url}">{title}</a>')
    return s


def key_cell(label):
    """A shortcut's label for the keys table: each key combination as <kbd>."""
    return " ".join(t if t in ("…", "or", "/") else f"<kbd>{html.escape(t)}</kbd>" for t in label.split())


def list_item(text, here):
    """A list item; a short lead before the first colon ("File › Save (Ctrl+S): …") is set in bold."""
    i = text.find(": ")
    if 0 < i <= 80 and "." not in text[:i] and text[:i].count("(") == text[:i].count(")"):
        return f"<strong>{inline(text[:i], here)}</strong>{inline(text[i:], here)}"
    return inline(text, here)


def render_blocks(title, blocks, figure_html=""):
    out, placed = [], False
    for kind, val in blocks:
        if kind == "p":
            out.append(f"<p>{inline(val, title)}</p>")
        elif kind == "list":
            items = "\n".join(f"<li>{list_item(it, title)}</li>" for it in val)
            out.append(f"<ul>\n{items}\n</ul>")
        elif kind == "keys":
            rows = "\n".join(f"<tr><td class=\"nowrap\">{key_cell(k)}</td><td>{inline(d)}</td></tr>"
                             for k, d in help_content._rows_for(val))
            out.append(f'<h2 class="keys-title" id="{slugify(val)}">{html.escape(val)}</h2>\n<div class="table-scroll">'
                       f'<table class="keys"><thead><tr><th scope="col">Key</th><th scope="col">Does</th></tr></thead>'
                       f"<tbody>\n{rows}\n</tbody></table></div>")
        elif kind == "defs":
            items = "\n".join(f"<dt><code>{html.escape(t)}</code></dt><dd>{inline(d, title)}</dd>"
                              for t, d in help_content.defs_rows(val))
            out.append(f'<dl class="defs">\n{items}\n</dl>')
        else:
            raise ValueError(f"unknown help block {kind!r} in {title!r}")
        if figure_html and not placed:
            out.append(figure_html)
            placed = True
    return "\n".join(out)


# In the public repository these two are at their public paths (tools/public_release.py puts them there).
PUBLIC_PATHS = {"public/CHANGELOG.md": "CHANGELOG.md", "public/KNOWN_ISSUES.md": "docs/KNOWN_ISSUES.md"}


def source(path):
    p = ROOT / path
    return p if p.exists() or path not in PUBLIC_PATHS else ROOT / PUBLIC_PATHS[path]


def render_markdown(path):
    text = source(path).read_text(encoding="utf-8")
    text = re.sub(r"\A# .*\n+", "", text)                      # the page template has the H1
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc"],
                           extension_configs={"toc": {"permalink": False}})
    body = md.convert(text)
    body = body.replace("<table>", '<div class="table-scroll"><table>').replace("</table>", "</table></div>")
    body = body.replace('href="docs/', f'href="{GITHUB}/blob/main/docs/')
    return body


def plain(html_text):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", html_text))).strip()


# --- Images ---------------------------------------------------------------------------------------

class Images:
    def __init__(self, out):
        self.src = ROOT / "docs" / "images"
        self.out = out / "assets" / "img"
        self.out.mkdir(parents=True, exist_ok=True)
        self.info = {}

    def get(self, name):
        if name in self.info:
            return self.info[name]
        im = Image.open(self.src / f"{name}.png").convert("RGB")
        widths = [im.width]
        for w in (800, 420):
            if im.width > w * 1.4:
                widths.append(w)
        files = []
        for w in sorted(set(widths)):
            h = round(im.height * w / im.width)
            fn = f"{name}.webp" if w == im.width else f"{name}-{w}.webp"
            (im if w == im.width else im.resize((w, h), Image.LANCZOS)).save(self.out / fn, "WEBP", quality=84, method=6)
            files.append((fn, w))
        self.info[name] = {"w": im.width, "h": im.height, "files": files}
        return self.info[name]

    def img(self, name, alt, eager=False, sizes=None):
        i = self.get(name)
        srcset = ", ".join(f"/assets/img/{fn} {w}w" for fn, w in i["files"])
        attrs = [f'src="/assets/img/{name}.webp"', f'width="{i["w"]}"', f'height="{i["h"]}"',
                 f'alt="{html.escape(alt)}"']
        if len(i["files"]) > 1:
            attrs.append(f'srcset="{srcset}"')
            attrs.append(f'sizes="{sizes or "(min-width: 60rem) 50vw, 100vw"}"')
        attrs.append('fetchpriority="high"' if eager else 'loading="lazy" decoding="async"')
        return f"<img {' '.join(attrs)}>"

    def figure(self, name, alt, caption, cls="", sizes=None):
        c = f"shot {cls}".strip()
        return (f'<figure class="{c}">{self.img(name, alt, sizes=sizes)}'
                f"<figcaption>{html.escape(caption)}</figcaption></figure>")


# --- Pages ----------------------------------------------------------------------------------------

def ld(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def breadcrumb(items):
    return {"@type": "BreadcrumbList", "itemListElement": [
        {"@type": "ListItem", "position": i + 1, "name": n, "item": SITE + u} for i, (n, u) in enumerate(items)]}


def software_ld():
    return {
        "@type": "SoftwareApplication", "@id": f"{SITE}/#software", "name": "VisualDeck",
        "description": "Free, open-source video playback, projection mapping and show control software for Windows. "
                       "It plays video, images and live cameras on projectors and screens, cue by cue.",
        "url": f"{SITE}/", "applicationCategory": "MultimediaApplication",
        "applicationSubCategory": "Video playback and show control",
        "operatingSystem": "Windows 10, Windows 11", "softwareVersion": RELEASE["version"],
        "datePublished": RELEASE["date"], "downloadUrl": RELEASE["download_url"],
        "fileSize": RELEASE["size_text"], "memoryRequirements": "8 GB recommended",
        "softwareRequirements": "Windows 10 or 11, 64-bit; OpenGL 3.3 for Performance Mode",
        "license": "https://www.mozilla.org/en-US/MPL/2.0/", "isAccessibleForFree": True,
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
        "screenshot": [f"{SITE}/assets/img/{n}.webp" for n in SHOTS_FOR_SCHEMA],
        "featureList": [
            "Cue grid of cues and layers with GO, BACK and Blackout",
            "Per-cue transitions and follows", "Blind editing with Update Output", "Preshow Check and Lock Editing",
            "H.264, HEVC, ProRes 4444, HAP, HAP Alpha and DXV playback", "USB cameras, capture cards and network streams",
            "Corner pin, mesh warp, edge blending, multi-projector spans and slices",
            "Real-time ISF shader effects", "OSC, browser remote and HTTP API",
            "MIDI, MIDI Show Control, MIDI timecode and SMPTE LTC", "Art-Net and sACN in and out",
            "Output Health, autosave and crash recovery"],
        "sameAs": [GITHUB],
    }


def build(out):
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)
    shutil.copytree(HERE / "static" / "fonts", out / "assets" / "fonts",
                    ignore=shutil.ignore_patterns("README.md"))
    css = (HERE / "static" / "site.css").read_bytes()
    (out / "assets" / "site.css").write_bytes(css)
    css_version = hashlib.sha256(css).hexdigest()[:10]
    for f in ("og.png",):
        shutil.copy2(HERE / "static" / f, out / "assets" / f)
    make_icons(out)

    images = Images(out)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(HERE / "templates"), autoescape=True,
                             trim_blocks=True, lstrip_blocks=True)
    common = {"site": SITE, "github": GITHUB, "release": RELEASE, "css_version": css_version,
              "download_icon": DOWNLOAD_ICON, "bitcoin": BITCOIN, "img": images.img, "figure": images.figure}
    pages = []   # (path, lastmod)

    def write(path, template, **ctx):
        dest = out / path.lstrip("/") / "index.html" if path.endswith("/") else out / path.lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        page = env.get_template(template).render(path=path, **common, **ctx)
        dest.write_text(page, encoding="utf-8")
        if not path.endswith("404.html"):
            pages.append(path)

    # Navigation shared by the documentation pages
    seen = set()
    nav, order = [], []
    for group, titles in GROUPS:
        items = []
        for t in titles:
            seen.add(t)
            desc = MD_PAGES[t]["description"] if t.startswith("@") else TOPIC_META[t][2]
            items.append({"title": topic_name(t), "url": topic_url(t), "description": desc, "key": t})
        nav.append((group, items))
        order += items
    missing = [t for t, _ in help_content.TOPICS if t not in seen]
    if missing:
        raise SystemExit(f"help topics missing from GROUPS: {missing}")
    for t, (slug, _s, desc) in TOPIC_META.items():
        assert len(desc) <= 165, (t, len(desc))
    for t, seo in SEO_TITLES.items():
        assert t in TOPIC_META and len(seo) <= 42, (t, len(seo))

    today = datetime.date.today().isoformat()
    website_ref = {"@type": "WebSite", "@id": f"{SITE}/#website", "url": f"{SITE}/", "name": "VisualDeck",
                   "inLanguage": "en"}

    # Home
    faq_ld = {"@type": "FAQPage", "@id": f"{SITE}/#faq", "mainEntity": [
        {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": plain(a)}} for q, a in FAQ]}
    home_ld = ld({"@context": "https://schema.org", "@graph": [website_ref, software_ld(), faq_ld]})
    write("/", "home.html", title="VisualDeck: free video playback and show control for Windows",
          description="VisualDeck plays video, images and live cameras on projectors and screens, cue by cue. Free "
                      "and open source for Windows, with projection mapping, OSC, MIDI and DMX.",
          jsonld=[home_ld], faq=[(q, html.escape(a)) for q, a in FAQ], section="home")

    # Download
    dl_ld = ld({"@context": "https://schema.org", "@graph": [
        {**software_ld()}, breadcrumb([("Home", "/"), ("Download", "/download/")])]})
    write("/download/", "download.html", title=f"Download VisualDeck {RELEASE['label']} for Windows (free)",
          description=f"Download VisualDeck {RELEASE['label']}, free video cue and projection mapping software for "
                      f"Windows 10 and 11. One {RELEASE['size_text']} installer, no admin rights, SHA-256 included.",
          jsonld=[dl_ld], section="download")

    # Documentation index
    idx_ld = ld({"@context": "https://schema.org", "@graph": [
        {"@type": "CollectionPage", "@id": f"{SITE}/docs/", "url": f"{SITE}/docs/", "name": "VisualDeck documentation",
         "isPartOf": {"@id": f"{SITE}/#website"}, "about": {"@id": f"{SITE}/#software"}},
        breadcrumb([("Home", "/"), ("Documentation", "/docs/")])]})
    write("/docs/", "docs_index.html", title="VisualDeck documentation: user guide and reference",
          description="The VisualDeck user guide: a quick start for show operators, building and running shows, "
                      "outputs and projection mapping, remote control and every shortcut.",
          jsonld=[idx_ld], nav=nav, section="docs")

    # Documentation pages
    topics = dict(help_content.TOPICS)
    for i, it in enumerate(order):
        key = it["key"]
        prev = order[i - 1] if i > 0 else None
        nxt = order[i + 1] if i + 1 < len(order) else None
        if key.startswith("@"):
            meta = MD_PAGES[key]
            body = render_markdown(meta["file"])
            page = {"slug": meta["slug"], "title": meta["title"], "h1": meta["h1"], "short": meta["title"],
                    "description": meta["description"]}
            src = f'{GITHUB}/blob/main/{PUBLIC_PATHS.get(meta["file"], meta["file"])}'
            page["source_note"] = f'This page is <a href="{src}">{meta["file"].split("/")[-1]}</a> from the source code.'
            head_title = f"{meta['title']} | VisualDeck docs"
        else:
            slug, short, desc = TOPIC_META[key]
            fig = ""
            if key in TOPIC_IMAGES:
                name, alt, cap = TOPIC_IMAGES[key]
                fig = images.figure(name, alt, cap, sizes="(min-width: 60rem) 52rem, 100vw")
            body = render_blocks(key, topics[key], fig)
            page = {"slug": slug, "title": key, "short": short or key, "description": desc}
            page["source_note"] = ("This is the same text as the app's built-in help (F1), for VisualDeck "
                                   f"{RELEASE['label']}.")
            head_title = f"{SEO_TITLES.get(key, key)} | VisualDeck docs"
        page["body"] = body
        page["prev"] = {"title": prev["title"], "url": prev["url"]} if prev else None
        page["next"] = {"title": nxt["title"], "url": nxt["url"]} if nxt else None
        url = it["url"]
        doc_ld = ld({"@context": "https://schema.org", "@graph": [
            {"@type": "TechArticle", "@id": SITE + url, "url": SITE + url, "headline": page.get("h1") or page["title"],
             "description": page["description"], "inLanguage": "en", "dateModified": today,
             "isPartOf": {"@id": f"{SITE}/#website"}, "about": {"@id": f"{SITE}/#software"},
             "proficiencyLevel": "Beginner" if key in ("Quick start: running a show", "Getting started") else "Expert"},
            breadcrumb([("Home", "/"), ("Documentation", "/docs/"), (page["short"], url)])]})
        write(url, "doc.html", title=head_title, description=page["description"], jsonld=[doc_ld], nav=nav,
              page=page, section="docs", og_type="article")

    # 404
    write("/404.html", "404.html", title="Page not found | VisualDeck",
          description="This page isn't on visualdeck.org.", jsonld=[], section="", noindex=True)

    # sitemap, robots, llms.txt
    urls = "\n".join(f"  <url><loc>{SITE}{p}</loc><lastmod>{today}</lastmod></url>" for p in pages)
    (out / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n'
                                     '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                                     f"{urls}\n</urlset>\n", encoding="utf-8")
    (out / "robots.txt").write_text(
        "# visualdeck.org is public: search engines and AI assistants are welcome to read and cite it.\n"
        "User-agent: *\nAllow: /\n\n"
        f"Sitemap: {SITE}/sitemap.xml\n", encoding="utf-8")
    write_llms(out, nav)
    return pages


def write_llms(out, nav):
    lines = [
        "# VisualDeck", "",
        f"> VisualDeck is free, open-source video playback, projection mapping and show control software for "
        f"Windows 10 and 11. It plays video, images and live cameras on projectors and screens, cue by cue, from "
        f"a grid of cues and layers. Version {RELEASE['label']} was released on {RELEASE['date_text']} under the "
        f"Mozilla Public License 2.0.", "",
        f"- Download: {RELEASE['download_url']} ({RELEASE['size_text']}, SHA-256 {RELEASE['sha256']})",
        f"- Source code: {GITHUB}",
        "- Formats: H.264, H.265 (HEVC), ProRes including 4444 with transparency, HAP, HAP Alpha, DXV; JPG, PNG, "
        "BMP, GIF; USB cameras, capture cards and network streams.",
        "- Show control: OSC, browser remote and HTTP API, MIDI, MIDI Show Control, MIDI timecode, SMPTE LTC, "
        "Art-Net and sACN.",
        "- Not included: audio-only or lighting cues, NDI, Spout, Syphon, 3D projection mapping, a timeline, macOS.",
        "",
    ]
    for group, items in nav:
        lines.append(f"## {group}")
        lines.append("")
        for it in items:
            lines.append(f"- [{it['title']}]({SITE}{it['url']}): {it['description']}")
        lines.append("")
    lines += ["## Optional", "", f"- [The whole user guide and reference as one Markdown file]({SITE}/llms-full.txt)", ""]
    (out / "llms.txt").write_text("\n".join(lines), encoding="utf-8")
    full = [help_content.to_markdown().rstrip(), ""]
    for key in ("@remote-control-reference", "@known-issues", "@changelog"):
        full += [source(MD_PAGES[key]["file"]).read_text(encoding="utf-8").rstrip(), ""]
    (out / "llms-full.txt").write_text("\n".join(full), encoding="utf-8")


def make_icons(out):
    ico = Image.open(ROOT / "src" / "visualdeck.ico").convert("RGBA")
    ico.save(out / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    for size, name in ((64, "icon-64.png"), (192, "icon-192.png"), (180, "apple-touch-icon.png")):
        im = ico.resize((size, size), Image.LANCZOS)
        if name == "apple-touch-icon.png":                    # iOS draws its own rounded corners
            bg = Image.new("RGBA", (size, size), (255, 255, 255, 255))
            bg.alpha_composite(im)
            im = bg.convert("RGB")
        im.save(out / "assets" / name, optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(HERE / "_site"))
    a = ap.parse_args()
    pages = build(Path(a.out))
    print(f"built {len(pages)} pages in {a.out}")


if __name__ == "__main__":
    main()
