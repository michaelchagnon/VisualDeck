# visualdeck.org

The website is plain static HTML built by `build.py`:

    pip install -r website/requirements.txt
    python website/build.py --out website/_site

Upload the contents of `website/_site` to the site's root (`public_html`). Nothing runs on the
server: no PHP, no database, no cookies, no analytics.

## Where the content comes from

- **Home and download pages:** `templates/home.html` and `templates/download.html`. The release's
  version, date, file size and SHA-256 are in `RELEASE` at the top of `build.py`; update them for a
  new release (the version label comes from `src/version.py`).
- **Documentation:** one page per topic of the in-app Help (`src/help_content.py`), plus
  `docs/REMOTE_CONTROL.md`, `public/KNOWN_ISSUES.md` and `public/CHANGELOG.md`. The build stops if a
  Help topic is missing from `GROUPS`, so a new topic needs a group, a slug and a description in
  `TOPIC_META`.
- **Screenshots:** `docs/images/*.png` (shared with the README), converted to WebP at build time.
  They were taken on Windows 11 from the demo show *The Keeper*, whose media was generated for it,
  so no outside footage or artwork is used.
- **Social image:** `static/og.png`, made by `make_og.py` from `docs/images/main-window.png`.
- **Fonts:** see `static/fonts/README.md`.

## Search and AI assistants

Every page has a title, a description, a canonical URL, Open Graph tags and schema.org data
(SoftwareApplication and FAQPage on the home page, TechArticle and BreadcrumbList on the docs).
The build also writes `sitemap.xml`, `robots.txt` (everything allowed), `llms.txt` (a short index)
and `llms-full.txt` (the whole guide as Markdown).

## Design notes

The site is laid out like a prompt book: each section begins on a cue line with its number boxed
in the margin, and the margin holds the details an operator would pencil in (shortcuts, ports).
One family (Archivo) for text and headings, a monospace for cue numbers and keys, black ink on
white, and the app's GO green only for the download button. The one dark band is the "stage"
behind the main screenshot.
